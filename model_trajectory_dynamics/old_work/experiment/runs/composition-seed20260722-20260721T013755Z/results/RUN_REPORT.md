# Long-horizon causal composition pilot

## Response finiteness diagnostics

| injection | horizon | effective lags | last-50 mass | endpoint/peak | late log slope | signed/absolute |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 300 | 278.6 | 0.254 | 0.610 | 0.00490 | 0.331 |
| 150 | 300 | 174.5 | 0.394 | 0.531 | 0.01384 | 0.774 |
| 200 | 300 | 228.0 | 0.327 | 0.372 | 0.01335 | 0.509 |
| 300 | 300 | 146.4 | 0.521 | 0.857 | 0.01380 | 0.513 |
| 400 | 300 | 172.7 | 0.340 | 0.853 | 0.00234 | 0.282 |
| 500 | 300 | 274.0 | 0.149 | 0.503 | -0.00173 | 0.213 |

## Operational two-block composition

| start | block length | JS (bits) | TV | Wasserstein/horizon |
|---:|---:|---:|---:|---:|
| 50 | 100 | 0.236 [0.072, 0.304] | 0.467 [0.246, 0.549] | 0.167 [0.079, 0.213] |
| 200 | 100 | 0.319 [0.153, 0.518] | 0.585 [0.368, 0.749] | 0.187 [0.103, 0.221] |
| 400 | 100 | 0.298 [0.223, 0.479] | 0.517 [0.446, 0.674] | 0.200 [0.155, 0.278] |

## Local-linearity check

At injection step 200, derivatives estimated with epsilon=0.050 and epsilon=0.100 had relative L2 discrepancy 3.1241 and cosine similarity 0.1514.

## Interpretation boundary

The composition comparison normalizes the first block of a response and a separate response beginning at the second block, convolves those curves, and compares the result with the independently retained two-block response from the first injection. Failure rejects this operational scalar semigroup construction. Success would not prove that the scalar observable is a sufficient interface for model training.

```json
{
  "aws_instance_id": "i-001ec27d6e867185a",
  "aws_region": "us-east-1",
  "base_runner_sha256": "b94e64d3b80850f2a72fcc4ac00251bd16cd45a1e9817182f1fb68d7caa60777",
  "block_length": 100,
  "bootstrap_samples": 400,
  "device": "cpu",
  "elapsed_seconds": 140.458659891,
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
  "seed": 20260722,
  "starts": [
    50,
    200,
    400
  ],
  "status": "complete",
  "timestamp_utc": "2026-07-21T01:41:23Z",
  "torch": "2.10.0+cpu"
}
```
