param(
    [string]$Region = "us-east-1",
    [string]$InstanceType = "c7i.2xlarge",
    [string]$SubnetId = "subnet-1d8c3c7b",
    [string]$AmiId = "ami-0f303bae6b670e0ed",
    [int]$HardStopMinutes = 55
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$experimentRoot = Split-Path -Parent $PSScriptRoot
$runnerPath = Join-Path $experimentRoot "run_experiment.py"
$presignPath = Join-Path $PSScriptRoot "presign_put.py"
$runId = "aws-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$runRoot = Join-Path (Split-Path -Parent $PSScriptRoot) "runs\$runId"
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$account = aws sts get-caller-identity --query Account --output text
if ($LASTEXITCODE -ne 0) { throw "AWS identity lookup failed" }
$bucket = "closed-source-pilot-$account-" + $runId.Replace("aws-", "").ToLowerInvariant()
$resultKey = "results/$runId.tar.gz"
$securityGroupId = $null
$instanceId = $null
$bucketCreated = $false
$objectDownloaded = $false
$launchUtc = $null

try {
    Write-Output "Creating temporary S3 bucket $bucket"
    aws s3api create-bucket --bucket $bucket --region $Region | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "S3 bucket creation failed" }
    $bucketCreated = $true
    aws s3api put-public-access-block --bucket $bucket --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true | Out-Null

    $putUrl = python $presignPath --bucket $bucket --key $resultKey --region $Region --expires 7200
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($putUrl)) {
        throw "Could not generate result upload URL"
    }

    $sourceBytes = [IO.File]::ReadAllBytes($runnerPath)
    $memory = New-Object IO.MemoryStream
    $gzip = New-Object IO.Compression.GzipStream($memory, [IO.Compression.CompressionMode]::Compress, $true)
    $gzip.Write($sourceBytes, 0, $sourceBytes.Length)
    $gzip.Close()
    $encodedRunner = [Convert]::ToBase64String($memory.ToArray())
    $memory.Dispose()

    $userDataTemplate = @'
#!/bin/bash
set -uo pipefail
mkdir -p /opt/closed-source-pilot/results
exec > >(tee -a /var/log/closed-source-pilot.log) 2>&1
echo "pilot_boot_utc=$(date -u +%FT%TZ)"
shutdown -h +__HARD_STOP_MINUTES__

echo '__ENCODED_RUNNER__' | base64 -d | gzip -d > /opt/closed-source-pilot/run_experiment.py
chmod 0555 /opt/closed-source-pilot/run_experiment.py

PYTHON_BIN=""
for candidate in /opt/pytorch/bin/python3 /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/local/bin/python3 /usr/bin/python3; do
  if [ -x "$candidate" ] && "$candidate" -c 'import torch, numpy, matplotlib' >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "Installing isolated CPU PyTorch environment"
  dnf install -y python3.11 python3.11-pip
  python3.11 -m venv /opt/closed-source-pilot/venv
  /opt/closed-source-pilot/venv/bin/pip install --upgrade pip
  /opt/closed-source-pilot/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
  /opt/closed-source-pilot/venv/bin/pip install numpy matplotlib
  PYTHON_BIN=/opt/closed-source-pilot/venv/bin/python
fi

if [ ! -x "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c 'import torch, numpy, matplotlib' >/dev/null 2>&1; then
  echo "No working Python experiment environment was created"
  RUN_EXIT=91
else
  TOKEN=$(curl -fsS -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' http://169.254.169.254/latest/api/token || true)
  INSTANCE_ID=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id || true)
  export AWS_INSTANCE_ID="$INSTANCE_ID"
  export AWS_REGION="__REGION__"
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  echo "python_bin=$PYTHON_BIN"
  "$PYTHON_BIN" -c 'import torch; print("torch=" + torch.__version__); print("threads=" + str(torch.get_num_threads())); print("cuda=" + str(torch.cuda.is_available()))'
  timeout --signal=INT 2700 "$PYTHON_BIN" /opt/closed-source-pilot/run_experiment.py \
    --output-dir /opt/closed-source-pilot/results \
    --max-runtime-seconds 2400
  RUN_EXIT=$?
fi

echo "run_exit_code=$RUN_EXIT" | tee /opt/closed-source-pilot/results/cloud_exit_status.txt
cp /var/log/closed-source-pilot.log /opt/closed-source-pilot/results/instance_console.log
tar -czf /tmp/closed-source-results.tar.gz -C /opt/closed-source-pilot results
curl --fail --retry 5 --retry-delay 3 \
  -H 'Content-Type: application/gzip' \
  --upload-file /tmp/closed-source-results.tar.gz \
  '__PUT_URL__'
UPLOAD_EXIT=$?
echo "upload_exit_code=$UPLOAD_EXIT"
sync
shutdown -h now
exit $RUN_EXIT
'@
    $userData = $userDataTemplate.Replace("__HARD_STOP_MINUTES__", [string]$HardStopMinutes)
    $userData = $userData.Replace("__ENCODED_RUNNER__", $encodedRunner)
    $userData = $userData.Replace("__REGION__", $Region)
    $userData = $userData.Replace("__PUT_URL__", $putUrl.Trim())
    $userDataPath = Join-Path $runRoot "user_data.sh"
    [IO.File]::WriteAllText($userDataPath, $userData, [Text.UTF8Encoding]::new($false))
    $userDataBytes = [Text.Encoding]::UTF8.GetByteCount($userData)
    if ($userDataBytes -gt 16384) {
        throw "EC2 user data is $userDataBytes bytes, above the 16 KiB limit"
    }

    $vpcId = aws ec2 describe-subnets --subnet-ids $SubnetId --query 'Subnets[0].VpcId' --output text --region $Region
    $securityGroupId = aws ec2 create-security-group `
        --group-name "closed-source-$runId" `
        --description "No-ingress group for capped closed-source pilot" `
        --vpc-id $vpcId `
        --region $Region `
        --query GroupId `
        --output text
    if ($LASTEXITCODE -ne 0) { throw "Security group creation failed" }
    Write-Output "Created no-ingress security group $securityGroupId"

    $tagSpec = "ResourceType=instance,Tags=[{Key=Name,Value=closed-source-pilot},{Key=Project,Value=model-trajectory-dynamics},{Key=RunId,Value=$runId},{Key=AutoTerminateMinutes,Value=$HardStopMinutes}]"
    $instanceId = aws ec2 run-instances `
        --image-id $AmiId `
        --instance-type $InstanceType `
        --subnet-id $SubnetId `
        --security-group-ids $securityGroupId `
        --associate-public-ip-address `
        --user-data "file://$userDataPath" `
        --instance-initiated-shutdown-behavior terminate `
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=25,VolumeType=gp3,DeleteOnTermination=true}' `
        --tag-specifications $tagSpec `
        --region $Region `
        --query 'Instances[0].InstanceId' `
        --output text
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($instanceId)) {
        throw "EC2 launch failed"
    }
    $launchUtc = (Get-Date).ToUniversalTime()
    Write-Output "Launched $instanceId at $($launchUtc.ToString('o'))"

    $state = [ordered]@{
        run_id = $runId
        account = $account
        region = $Region
        instance_id = $instanceId
        instance_type = $InstanceType
        ami_id = $AmiId
        subnet_id = $SubnetId
        security_group_id = $securityGroupId
        temporary_bucket = $bucket
        result_key = $resultKey
        launch_utc = $launchUtc.ToString("o")
        hard_stop_minutes = $HardStopMinutes
        user_data_bytes = $userDataBytes
    }
    [IO.File]::WriteAllText(
        (Join-Path $runRoot "launch.json"),
        ($state | ConvertTo-Json -Depth 4),
        [Text.UTF8Encoding]::new($false)
    )

    $deadline = $launchUtc.AddMinutes($HardStopMinutes + 3)
    while ((Get-Date).ToUniversalTime() -lt $deadline) {
        $objectListing = aws s3 ls "s3://$bucket/$resultKey" --region $Region
        if (-not [string]::IsNullOrWhiteSpace($objectListing)) {
            Write-Output "Result archive is available"
            break
        }
        $instanceState = aws ec2 describe-instances --instance-ids $instanceId --region $Region --query 'Reservations[0].Instances[0].State.Name' --output text
        Write-Output "$(Get-Date -Format o) instance=$instanceState waiting-for-results"
        Start-Sleep -Seconds 20
    }

    $objectListing = aws s3 ls "s3://$bucket/$resultKey" --region $Region
    if (-not [string]::IsNullOrWhiteSpace($objectListing)) {
        $archivePath = Join-Path $runRoot "results.tar.gz"
        aws s3api get-object --bucket $bucket --key $resultKey $archivePath --region $Region | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Result download failed" }
        tar -xzf $archivePath -C $runRoot
        $objectDownloaded = $true
    } else {
        throw "The instance did not upload results before the hard deadline"
    }

    aws ec2 get-console-output --instance-id $instanceId --latest --region $Region --query Output --output text 2>$null | Out-File -FilePath (Join-Path $runRoot "ec2_console_output.txt") -Encoding utf8
    aws ec2 describe-instances --instance-ids $instanceId --region $Region --output json | Out-File -FilePath (Join-Path $runRoot "ec2_instance.json") -Encoding utf8
}
finally {
    if ($instanceId) {
        $currentState = aws ec2 describe-instances --instance-ids $instanceId --region $Region --query 'Reservations[0].Instances[0].State.Name' --output text 2>$null
        if ($currentState -notin @("terminated", "shutting-down")) {
            Write-Output "Terminating $instanceId during cleanup"
            aws ec2 terminate-instances --instance-ids $instanceId --region $Region | Out-Null
        }
        aws ec2 wait instance-terminated --instance-ids $instanceId --region $Region 2>$null
    }
    if ($securityGroupId) {
        for ($attempt = 0; $attempt -lt 6; $attempt++) {
            aws ec2 delete-security-group --group-id $securityGroupId --region $Region 2>$null
            if ($LASTEXITCODE -eq 0) { break }
            Start-Sleep -Seconds 10
        }
    }
    if ($bucketCreated) {
        aws s3 rm "s3://$bucket" --recursive --region $Region 2>$null | Out-Null
        aws s3api delete-bucket --bucket $bucket --region $Region 2>$null | Out-Null
    }
}

if (-not $objectDownloaded) { throw "No local result archive was collected" }
Write-Output "Collected results in $runRoot"
