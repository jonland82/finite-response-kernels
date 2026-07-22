param(
    [ValidateSet("baseline", "constraint")]
    [string]$Experiment = "baseline",
    [string]$Region = "us-east-1",
    [string]$InstanceType = "c7i.8xlarge",
    [string]$SubnetId = "subnet-1d8c3c7b",
    [string]$AmiId = "ami-0f303bae6b670e0ed",
    [int]$HardStopMinutes = 42,
    [double]$VerifiedHourlyRate = 1.428,
    [double]$BudgetCeiling = 5.0,
    [int]$Workers = 32,
    [int]$WorkerSeconds = 1200
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$experimentRoot = Split-Path -Parent $PSScriptRoot
$presignPath = Join-Path $PSScriptRoot "presign_put.py"
$experimentScript = if ($Experiment -eq "constraint") { "run_constraint_comparison.py" } else { "run_experiment.py" }
$runId = "inverse-$Experiment-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$runRoot = Join-Path $experimentRoot "runs\$runId"
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$maximumInstanceCost = $VerifiedHourlyRate * $HardStopMinutes / 60.0
if ($maximumInstanceCost -ge ($BudgetCeiling - 0.50)) {
    throw "Configured instance ceiling $maximumInstanceCost leaves insufficient budget margin"
}

$account = aws sts get-caller-identity --query Account --output text
if ($LASTEXITCODE -ne 0) { throw "AWS identity lookup failed" }
$bucket = "inverse-takeoff-$account-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddHHmmss")
$sourceKey = "source/$runId.tar.gz"
$resultKey = "results/$runId.tar.gz"
$sourceArchive = Join-Path $runRoot "source.tar.gz"
$userDataPath = Join-Path $runRoot "user_data.sh"
$securityGroupId = $null
$instanceId = $null
$bucketCreated = $false
$objectDownloaded = $false
$launchUtc = $null

try {
    $quota = aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --region $Region --query 'Quota.Value' --output text
    if ($LASTEXITCODE -ne 0 -or [double]$quota -lt $Workers) {
        throw "Standard-instance vCPU quota ($quota) is below requested workers ($Workers)"
    }
    $subnetAz = aws ec2 describe-subnets --subnet-ids $SubnetId --region $Region --query 'Subnets[0].AvailabilityZone' --output text
    $offered = aws ec2 describe-instance-type-offerings --location-type availability-zone --filters "Name=instance-type,Values=$InstanceType" "Name=location,Values=$subnetAz" --region $Region --query 'length(InstanceTypeOfferings)' --output text
    if ($LASTEXITCODE -ne 0 -or [int]$offered -lt 1) {
        throw "$InstanceType is not offered in subnet AZ $subnetAz"
    }

    Write-Output "Creating temporary private S3 bucket $bucket"
    aws s3api create-bucket --bucket $bucket --region $Region | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "S3 bucket creation failed" }
    $bucketCreated = $true
    aws s3api put-public-access-block --bucket $bucket --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true | Out-Null

    tar -czf $sourceArchive -C $experimentRoot run_experiment.py run_constraint_comparison.py requirements.txt README.md
    if ($LASTEXITCODE -ne 0) { throw "Source packaging failed" }
    aws s3api put-object --bucket $bucket --key $sourceKey --body $sourceArchive --content-type application/gzip --region $Region | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Source upload failed" }
    $getUrl = aws s3 presign "s3://$bucket/$sourceKey" --region $Region --expires-in 7200
    $putUrl = python $presignPath --bucket $bucket --key $resultKey --region $Region --expires 7200
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($putUrl)) { throw "Presign failed" }

    $userDataTemplate = @'
#!/bin/bash
set -uo pipefail
mkdir -p /opt/inverse-takeoff/results
exec > >(tee -a /var/log/inverse-takeoff.log) 2>&1
echo "boot_utc=$(date -u +%FT%TZ)"
shutdown -h +__HARD_STOP_MINUTES__

curl --fail --retry 5 --retry-delay 2 '__GET_URL__' -o /tmp/source.tar.gz
tar -xzf /tmp/source.tar.gz -C /opt/inverse-takeoff
dnf install -y python3.11 python3.11-pip
python3.11 -m venv /opt/inverse-takeoff/venv
/opt/inverse-takeoff/venv/bin/pip install --upgrade pip
/opt/inverse-takeoff/venv/bin/pip install -r /opt/inverse-takeoff/requirements.txt
PYTHON_BIN=/opt/inverse-takeoff/venv/bin/python

if [ ! -x "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c 'import numpy, scipy' >/dev/null 2>&1; then
  echo "Python environment validation failed"
  RUN_EXIT=91
else
  TOKEN=$(curl -fsS -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' http://169.254.169.254/latest/api/token || true)
  INSTANCE_ID=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id || true)
  export AWS_INSTANCE_ID="$INSTANCE_ID"
  export AWS_REGION="__REGION__"
  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  timeout --signal=INT 1800 "$PYTHON_BIN" /opt/inverse-takeoff/__EXPERIMENT_SCRIPT__ \
    --output-dir /opt/inverse-takeoff/results \
    --workers __WORKERS__ \
    --worker-seconds __WORKER_SECONDS__ \
    --max-schemas-per-worker 100000
  RUN_EXIT=$?
fi

echo "run_exit_code=$RUN_EXIT" | tee /opt/inverse-takeoff/results/cloud_exit_status.txt
cp /var/log/inverse-takeoff.log /opt/inverse-takeoff/results/instance_console.log
tar -czf /tmp/inverse-takeoff-results.tar.gz -C /opt/inverse-takeoff results
curl --fail --retry 5 --retry-delay 3 \
  -H 'Content-Type: application/gzip' \
  --upload-file /tmp/inverse-takeoff-results.tar.gz \
  '__PUT_URL__'
UPLOAD_EXIT=$?
echo "upload_exit_code=$UPLOAD_EXIT"
sync
shutdown -h now
exit $RUN_EXIT
'@
    $userData = $userDataTemplate.Replace("__HARD_STOP_MINUTES__", [string]$HardStopMinutes)
    $userData = $userData.Replace("__GET_URL__", $getUrl.Trim())
    $userData = $userData.Replace("__PUT_URL__", $putUrl.Trim())
    $userData = $userData.Replace("__REGION__", $Region)
    $userData = $userData.Replace("__WORKERS__", [string]$Workers)
    $userData = $userData.Replace("__WORKER_SECONDS__", [string]$WorkerSeconds)
    $userData = $userData.Replace("__EXPERIMENT_SCRIPT__", $experimentScript)
    [IO.File]::WriteAllText($userDataPath, $userData, [Text.UTF8Encoding]::new($false))
    if ([Text.Encoding]::UTF8.GetByteCount($userData) -gt 16384) { throw "EC2 user data exceeds 16 KiB" }

    $vpcId = aws ec2 describe-subnets --subnet-ids $SubnetId --query 'Subnets[0].VpcId' --output text --region $Region
    $securityGroupId = aws ec2 create-security-group --group-name $runId --description "No-ingress group for capped inverse takeoff experiment" --vpc-id $vpcId --region $Region --query GroupId --output text
    if ($LASTEXITCODE -ne 0) { throw "Security group creation failed" }
    $tagSpec = "ResourceType=instance,Tags=[{Key=Name,Value=inverse-takeoff-experiment},{Key=Project,Value=inverse-takeoff},{Key=RunId,Value=$runId},{Key=AutoTerminateMinutes,Value=$HardStopMinutes}]"
    $instanceId = aws ec2 run-instances `
        --image-id $AmiId `
        --instance-type $InstanceType `
        --subnet-id $SubnetId `
        --security-group-ids $securityGroupId `
        --associate-public-ip-address `
        --user-data "file://$userDataPath" `
        --instance-initiated-shutdown-behavior terminate `
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=15,VolumeType=gp3,DeleteOnTermination=true}' `
        --tag-specifications $tagSpec `
        --region $Region `
        --query 'Instances[0].InstanceId' `
        --output text
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($instanceId)) { throw "EC2 launch failed" }
    $launchUtc = (Get-Date).ToUniversalTime()
    Write-Output "Launched $instanceId at $($launchUtc.ToString('o')); instance ceiling USD $($maximumInstanceCost.ToString('0.0000'))"

    $state = [ordered]@{
        run_id = $runId; experiment = $Experiment; experiment_script = $experimentScript
        account = $account; region = $Region; instance_id = $instanceId
        instance_type = $InstanceType; ami_id = $AmiId; subnet_id = $SubnetId
        security_group_id = $securityGroupId; temporary_bucket = $bucket
        source_key = $sourceKey; result_key = $resultKey; launch_utc = $launchUtc.ToString("o")
        hard_stop_minutes = $HardStopMinutes; verified_hourly_rate_usd = $VerifiedHourlyRate
        maximum_instance_cost_usd = $maximumInstanceCost; budget_ceiling_usd = $BudgetCeiling
        workers = $Workers; worker_seconds = $WorkerSeconds
    }
    [IO.File]::WriteAllText((Join-Path $runRoot "launch.json"), ($state | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false))
    Remove-Item -LiteralPath $userDataPath -Force
    Remove-Item -LiteralPath $sourceArchive -Force

    $deadline = $launchUtc.AddMinutes($HardStopMinutes + 2)
    while ((Get-Date).ToUniversalTime() -lt $deadline) {
        $listing = aws s3 ls "s3://$bucket/$resultKey" --region $Region
        if (-not [string]::IsNullOrWhiteSpace($listing)) { break }
        $instanceState = aws ec2 describe-instances --instance-ids $instanceId --region $Region --query 'Reservations[0].Instances[0].State.Name' --output text
        Write-Output "$(Get-Date -Format o) instance=$instanceState waiting-for-results"
        Start-Sleep -Seconds 20
    }
    $listing = aws s3 ls "s3://$bucket/$resultKey" --region $Region
    if ([string]::IsNullOrWhiteSpace($listing)) { throw "No result archive before hard deadline" }
    $archivePath = Join-Path $runRoot "results.tar.gz"
    aws s3api get-object --bucket $bucket --key $resultKey $archivePath --region $Region | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Result download failed" }
    tar -xzf $archivePath -C $runRoot
    $objectDownloaded = $true
    aws ec2 get-console-output --instance-id $instanceId --latest --region $Region --query Output --output text | Out-File -FilePath (Join-Path $runRoot "ec2_console_output.txt") -Encoding utf8
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
    if (Test-Path -LiteralPath $userDataPath) { Remove-Item -LiteralPath $userDataPath -Force }
    if (Test-Path -LiteralPath $sourceArchive) { Remove-Item -LiteralPath $sourceArchive -Force }
}

if (-not $objectDownloaded) { throw "No local result archive was collected" }
Write-Output "Collected results in $runRoot"
