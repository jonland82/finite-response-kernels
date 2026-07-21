# AWS closed-source language-model pilot

## Outcome

The closed condition's observable information lower bound moved from -0.047 to -0.411 bits out of a 16-bit source. Its final checkpoint increment proxy was 0.068 bits.

The open control ended at -0.085 recoverable bits after the available source expanded to 7 bits.

These quantities are variational lower bounds derived from the model's own red/blue probabilities. They are not estimates of the exact mutual information in the complete real-valued parameter trajectory, and finite-horizon behavior does not prove an asymptotic theorem.

## Final checkpoint

| condition | available bits | lower bound (bits) | code accuracy | code loss (nats) |
|---|---:|---:|---:|---:|
| closed | 4 | -0.411 | 0.469 | 1.6332 |
| open | 7 | -0.085 | 0.531 | 1.7044 |

## Influence summaries

| injection step | effective lag count | peak mass | last-ten-lag mass | signed/absolute ratio |
|---:|---:|---:|---:|---:|
| 4 | 3.93 | 0.289 | 1.000 | 0.158 |

A large last-ten-lag mass means the response did not decay within the measurement window, so treating the truncated curve as a finite normalized kernel would be questionable.

## Reproduction metadata

```json
{
  "aws_instance_id": null,
  "aws_region": null,
  "config": {
    "batch_size": 4,
    "bootstrap_samples": 20,
    "d_model": 32,
    "ff_width": 64,
    "heads": 4,
    "influence_epsilon": 0.1,
    "influence_horizon": 3,
    "influence_replicas": 2,
    "influence_times": [
      4
    ],
    "initial_facts": 4,
    "layers": 1,
    "learning_rate": 0.003,
    "max_runtime_seconds": 300,
    "maximum_facts": 8,
    "output_dir": "model_trajectory_dynamics/experiment/smoke_results",
    "replicas": 4,
    "reveal_interval": 3,
    "seed": 20260720,
    "skip_influence": false,
    "smoke": true,
    "steps": 12,
    "weight_decay": 0.001
  },
  "cuda_available": false,
  "cuda_device": null,
  "elapsed_seconds": 2.1682507999939844,
  "platform": "Windows-11-10.0.22631-SP0",
  "python": "3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:37:03) [MSC v.1929 64 bit (AMD64)]",
  "script_sha256": "df7b1b559ffd6cd5fb234ecca263240ee6ca22921a75c61d1451a53807af5323",
  "status": "complete",
  "timestamp_utc": "2026-07-21T00:24:39Z",
  "torch": "2.10.0+cpu"
}
```
