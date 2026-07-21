# Long-horizon causal composition pilot

## Response finiteness diagnostics

| injection | horizon | effective lags | last-50 mass | endpoint/peak | late log slope | signed/absolute |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10 | 10.5 | 1.000 | 0.713 | -0.14165 | 0.201 |
| 10 | 10 | 10.8 | 1.000 | 0.850 | -0.05129 | 0.022 |

## Operational two-block composition

| start | block length | JS (bits) | TV | Wasserstein/horizon |
|---:|---:|---:|---:|---:|
| 5 | 5 | 0.077 [0.072, 0.121] | 0.266 [0.253, 0.328] | 0.123 [0.112, 0.184] |

## Local-linearity check

At injection step 5, derivatives estimated with epsilon=0.050 and epsilon=0.100 had relative L2 discrepancy 0.0003 and cosine similarity 1.0000.

## Interpretation boundary

The composition comparison normalizes the first block of a response and a separate response beginning at the second block, convolves those curves, and compares the result with the independently retained two-block response from the first injection. Failure rejects this operational scalar semigroup construction. Success would not prove that the scalar observable is a sufficient interface for model training.

```json
{
  "aws_instance_id": null,
  "aws_region": null,
  "base_runner_sha256": "b94e64d3b80850f2a72fcc4ac00251bd16cd45a1e9817182f1fb68d7caa60777",
  "block_length": 5,
  "bootstrap_samples": 20,
  "device": "cpu",
  "elapsed_seconds": 1.766396300008637,
  "epsilon": 0.1,
  "horizon": 10,
  "injection_times": [
    5,
    10
  ],
  "platform": "Windows-11-10.0.22631-SP0",
  "python": "3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:37:03) [MSC v.1929 64 bit (AMD64)]",
  "replicas": 4,
  "script_sha256": "5e20eeb39994aff19d135d08e8a2b0babae929d70f0d4c4c2438e4cab14db2e1",
  "seed": 20260720,
  "starts": [
    5
  ],
  "status": "complete",
  "timestamp_utc": "2026-07-21T01:02:49Z",
  "torch": "2.10.0+cpu"
}
```
