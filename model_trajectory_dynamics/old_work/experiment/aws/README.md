# AWS execution

`launch_and_collect.ps1` launches exactly one EC2 instance with no inbound
security-group rules. The runner is gzip-compressed into EC2 user data, so no
AWS credentials are placed on the instance. Results return through a two-hour
presigned S3 `PutObject` URL.

Cost and time controls:

- instance-initiated shutdown behavior is `terminate`;
- the instance schedules its own shutdown at 55 minutes;
- the Python experiment has a 40-minute internal deadline;
- GNU `timeout` supplies an independent 45-minute process limit;
- the launcher terminates the instance if the result deadline is exceeded;
- the 25 GB root disk has `DeleteOnTermination=true`;
- the temporary bucket and no-ingress security group are removed after result
  collection.

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File `
  model_trajectory_dynamics/experiment/aws/launch_and_collect.ps1
```

The account used for the July 2026 pilot had zero quota for both on-demand and
Spot G/VT instances. The launcher therefore uses `c7i.2xlarge` (8 vCPUs) with
the AWS-published standard Amazon Linux 2023 AMI and installs an isolated
CPU-only PyTorch environment. This does not change the model or protocol; only
the PyTorch execution device changes.

`launch_composition_and_collect.ps1` runs the longer response/composition
falsification pilot. It packages both Python sources into the temporary private
bucket, uses short-lived presigned GET and PUT URLs, applies a 20-minute Python
deadline and 25-minute process timeout, and terminates the instance at 30
minutes at the latest.
