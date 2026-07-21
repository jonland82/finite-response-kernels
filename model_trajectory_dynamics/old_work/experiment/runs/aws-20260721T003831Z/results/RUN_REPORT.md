# AWS closed-source language-model pilot

## Outcome

The closed condition's observable information lower bound moved from -0.032 to 12.176 bits out of a 16-bit source. Its final checkpoint increment proxy was -0.066 bits.

The open control ended at 31.999 recoverable bits after the available source expanded to 32 bits.

These quantities are variational lower bounds derived from the model's own red/blue probabilities. They are not estimates of the exact mutual information in the complete real-valued parameter trajectory, and finite-horizon behavior does not prove an asymptotic theorem.

## Final checkpoint

| condition | available bits | lower bound (bits) | code accuracy | code loss (nats) |
|---|---:|---:|---:|---:|
| closed | 16 | 12.176 | 1.000 | 0.0020 |
| open | 32 | 31.999 | 1.000 | 0.0041 |

## Influence summaries

| injection step | effective lag count | peak mass | last-ten-lag mass | signed/absolute ratio |
|---:|---:|---:|---:|---:|
| 50 | 35.86 | 0.052 | 0.335 | 0.531 |
| 150 | 30.31 | 0.079 | 0.146 | 0.966 |
| 300 | 49.84 | 0.030 | 0.182 | 0.246 |

A large last-ten-lag mass means the response did not decay within the measurement window, so treating the truncated curve as a finite normalized kernel would be questionable.

## Reproduction metadata

```json
{
  "aws_instance_id": "i-0e56a7ec222026546",
  "aws_region": "us-east-1",
  "config": {
    "batch_size": 4,
    "bootstrap_samples": 500,
    "d_model": 64,
    "ff_width": 256,
    "heads": 4,
    "influence_epsilon": 0.1,
    "influence_horizon": 50,
    "influence_replicas": 16,
    "influence_times": [
      50,
      150,
      300
    ],
    "initial_facts": 16,
    "layers": 2,
    "learning_rate": 0.003,
    "max_runtime_seconds": 2400,
    "maximum_facts": 32,
    "output_dir": "/opt/closed-source-pilot/results",
    "replicas": 128,
    "reveal_interval": 20,
    "seed": 20260720,
    "skip_influence": false,
    "smoke": false,
    "steps": 400,
    "weight_decay": 0.001
  },
  "cuda_available": false,
  "cuda_device": null,
  "elapsed_seconds": 36.089902120999994,
  "platform": "Linux-6.1.176-221.367.amzn2023.x86_64-x86_64-with-glibc2.34",
  "python": "3.11.15 (main, Jul  6 2026, 00:00:00) [GCC 11.5.0 20240719 (Red Hat 11.5.0-5)]",
  "script_sha256": "b94e64d3b80850f2a72fcc4ac00251bd16cd45a1e9817182f1fb68d7caa60777",
  "status": "complete",
  "timestamp_utc": "2026-07-21T00:40:12Z",
  "torch": "2.10.0+cpu"
}
```
