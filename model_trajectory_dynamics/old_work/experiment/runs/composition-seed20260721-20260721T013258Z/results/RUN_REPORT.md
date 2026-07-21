# Long-horizon causal composition pilot

## Response finiteness diagnostics

| injection | horizon | effective lags | last-50 mass | endpoint/peak | late log slope | signed/absolute |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 300 | 236.9 | 0.399 | 0.548 | 0.00912 | 0.406 |
| 150 | 300 | 116.8 | 0.704 | 0.878 | 0.02735 | 0.442 |
| 200 | 300 | 197.0 | 0.296 | 0.418 | 0.01140 | 0.402 |
| 300 | 300 | 226.0 | 0.175 | 0.464 | -0.00417 | 0.189 |
| 400 | 300 | 283.7 | 0.185 | 0.750 | -0.00164 | 0.253 |
| 500 | 300 | 200.1 | 0.439 | 0.867 | 0.01531 | 0.627 |

## Operational two-block composition

| start | block length | JS (bits) | TV | Wasserstein/horizon |
|---:|---:|---:|---:|---:|
| 50 | 100 | 0.223 [0.147, 0.299] | 0.474 [0.363, 0.567] | 0.163 [0.126, 0.243] |
| 200 | 100 | 0.167 [0.084, 0.658] | 0.375 [0.225, 0.838] | 0.102 [0.049, 0.256] |
| 400 | 100 | 0.137 [0.081, 0.360] | 0.315 [0.225, 0.605] | 0.101 [0.069, 0.241] |

## Local-linearity check

At injection step 200, derivatives estimated with epsilon=0.050 and epsilon=0.100 had relative L2 discrepancy 2.2876 and cosine similarity -0.0844.

## Interpretation boundary

The composition comparison normalizes the first block of a response and a separate response beginning at the second block, convolves those curves, and compares the result with the independently retained two-block response from the first injection. Failure rejects this operational scalar semigroup construction. Success would not prove that the scalar observable is a sufficient interface for model training.

```json
{
  "aws_instance_id": "i-0868caaa156f5f37d",
  "aws_region": "us-east-1",
  "base_runner_sha256": "b94e64d3b80850f2a72fcc4ac00251bd16cd45a1e9817182f1fb68d7caa60777",
  "block_length": 100,
  "bootstrap_samples": 400,
  "device": "cpu",
  "elapsed_seconds": 148.970976396,
  "epsilon": 0.1,
  "horizon": 300,
  "injection_times": [
    50,
    150,
    200,
    300,
    400,
    500
  ],
  "platform": "Linux-6.1.176-221.367.amzn2023.x86_64-x86_64-with-glibc2.34",
  "python": "3.11.15 (main, Jul  6 2026, 00:00:00) [GCC 11.5.0 20240719 (Red Hat 11.5.0-5)]",
  "replicas": 64,
  "script_sha256": "9e3adf94f7ac155ec23c9bf535f916759df613d56b4bd6285e4cd7e1af201e62",
  "seed": 20260721,
  "starts": [
    50,
    200,
    400
  ],
  "status": "complete",
  "timestamp_utc": "2026-07-21T01:36:37Z",
  "torch": "2.10.0+cpu"
}
```
