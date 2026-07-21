# Long-horizon causal composition pilot

## Response finiteness diagnostics

| injection | horizon | effective lags | last-50 mass | endpoint/peak | late log slope | signed/absolute |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 300 | 265.3 | 0.110 | 0.267 | -0.00063 | 0.176 |
| 150 | 300 | 156.4 | 0.525 | 0.130 | 0.01706 | 0.522 |
| 200 | 300 | 110.6 | 0.451 | 0.427 | 0.03436 | 0.495 |
| 300 | 300 | 194.7 | 0.249 | 0.526 | 0.00017 | 0.635 |
| 400 | 300 | 227.3 | 0.113 | 0.212 | -0.00455 | 0.828 |
| 500 | 300 | 245.7 | 0.134 | 0.129 | 0.00177 | 0.760 |

## Operational two-block composition

| start | block length | JS (bits) | TV | Wasserstein/horizon |
|---:|---:|---:|---:|---:|
| 50 | 100 | 0.119 [0.090, 0.145] | 0.323 [0.262, 0.376] | 0.104 [0.083, 0.127] |
| 200 | 100 | 0.336 [0.093, 0.558] | 0.584 [0.285, 0.780] | 0.153 [0.092, 0.310] |
| 400 | 100 | 0.053 [0.036, 0.204] | 0.215 [0.159, 0.473] | 0.060 [0.019, 0.177] |

## Local-linearity check

At injection step 200, derivatives estimated with epsilon=0.050 and epsilon=0.100 had relative L2 discrepancy 1.2712 and cosine similarity 0.0098.

## Interpretation boundary

The composition comparison normalizes the first block of a response and a separate response beginning at the second block, convolves those curves, and compares the result with the independently retained two-block response from the first injection. Failure rejects this operational scalar semigroup construction. Success would not prove that the scalar observable is a sufficient interface for model training.

```json
{
  "aws_instance_id": "i-0ba5eebced1d2922e",
  "aws_region": "us-east-1",
  "base_runner_sha256": "b94e64d3b80850f2a72fcc4ac00251bd16cd45a1e9817182f1fb68d7caa60777",
  "block_length": 100,
  "bootstrap_samples": 400,
  "device": "cpu",
  "elapsed_seconds": 124.795877718,
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
  "script_sha256": "5e20eeb39994aff19d135d08e8a2b0babae929d70f0d4c4c2438e4cab14db2e1",
  "seed": 20260720,
  "starts": [
    50,
    200,
    400
  ],
  "status": "complete",
  "timestamp_utc": "2026-07-21T01:07:54Z",
  "torch": "2.10.0+cpu"
}
```
