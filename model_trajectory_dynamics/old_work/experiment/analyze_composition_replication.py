#!/usr/bin/env python3
"""Aggregate seeded causal-composition runs and calculate robust controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_causal_composition import composition_metrics, distribution_metrics


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def robust_split_half_noise_floor(
    first: np.ndarray,
    second: np.ndarray,
    aggregation: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    replicas = first.shape[1]
    half = replicas // 2
    rng = np.random.default_rng(seed)
    keys = (
        "js_divergence_bits",
        "total_variation",
        "wasserstein_fraction_of_horizon",
    )
    values = {key: [] for key in keys}
    for _ in range(samples):
        permutation = rng.permutation(replicas)
        group_a = permutation[:half]
        group_b = permutation[half : 2 * half]
        _, measured_a, predicted_a = composition_metrics(
            first[:, group_a], second[:, group_a], 100, aggregation=aggregation
        )
        _, measured_b, predicted_b = composition_metrics(
            first[:, group_b], second[:, group_b], 100, aggregation=aggregation
        )
        measured_noise = distribution_metrics(measured_a, measured_b)
        predicted_noise = distribution_metrics(predicted_a, predicted_b)
        for key in keys:
            values[key].append(max(measured_noise[key], predicted_noise[key]))
    result: dict[str, float] = {}
    for key, sampled in values.items():
        result[f"{key}_control_median"] = float(np.median(sampled))
        result[f"{key}_control_95th"] = float(np.quantile(sampled, 0.95))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--control-samples", type=int, default=400)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    composition_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    linearity_rows: list[dict[str, object]] = []
    response_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for run_dir_argument in arguments.run_dirs:
        run_dir = run_dir_argument.resolve()
        results = run_dir / "results"
        metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
        seed = int(metadata["seed"])
        run_rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "instance_id": metadata["aws_instance_id"],
                "status": metadata["status"],
                "elapsed_seconds": metadata["elapsed_seconds"],
                "script_sha256": metadata["script_sha256"],
            }
        )
        for name, target in (
            ("composition_robustness.csv", composition_rows),
            ("linearity_summary.csv", linearity_rows),
            ("response_summary.csv", response_rows),
        ):
            for row in read_csv(results / name):
                target.append({"seed": seed, **row})

        responses = np.load(results / "response_derivatives.npz")
        for start in (50, 200, 400):
            first = responses[f"step_{start:04d}"].astype(np.float64)
            second = responses[f"step_{start + 100:04d}"].astype(np.float64)
            for aggregation in ("mean", "median", "trimmed_mean"):
                observed, _, _ = composition_metrics(
                    first, second, 100, aggregation=aggregation
                )
                control = robust_split_half_noise_floor(
                    first,
                    second,
                    aggregation,
                    arguments.control_samples,
                    seed + start + {"mean": 0, "median": 1000, "trimmed_mean": 2000}[aggregation],
                )
                control_rows.append(
                    {
                        "seed": seed,
                        "start_step": start,
                        "aggregation": aggregation,
                        **observed,
                        **control,
                        "js_exceeds_control_95th": observed["js_divergence_bits"]
                        > control["js_divergence_bits_control_95th"],
                        "tv_exceeds_control_95th": observed["total_variation"]
                        > control["total_variation_control_95th"],
                        "wasserstein_exceeds_control_95th": observed[
                            "wasserstein_fraction_of_horizon"
                        ]
                        > control["wasserstein_fraction_of_horizon_control_95th"],
                    }
                )

    write_csv(output_dir / "runs.csv", run_rows)
    write_csv(output_dir / "composition_all_seeds.csv", composition_rows)
    write_csv(output_dir / "robust_noise_floor_all_seeds.csv", control_rows)
    write_csv(output_dir / "linearity_all_seeds.csv", linearity_rows)
    write_csv(output_dir / "responses_all_seeds.csv", response_rows)

    cross_seed: dict[str, object] = {}
    for aggregation in ("mean", "median", "trimmed_mean"):
        cross_seed[aggregation] = {}
        for start in (50, 200, 400):
            selected = [
                row
                for row in control_rows
                if row["aggregation"] == aggregation and row["start_step"] == start
            ]
            cross_seed[aggregation][str(start)] = {
                "tv_by_seed": [float(row["total_variation"]) for row in selected],
                "tv_cross_seed_median": float(
                    np.median([float(row["total_variation"]) for row in selected])
                ),
                "js_cross_seed_median": float(
                    np.median([float(row["js_divergence_bits"]) for row in selected])
                ),
                "wasserstein_cross_seed_median": float(
                    np.median(
                        [float(row["wasserstein_fraction_of_horizon"]) for row in selected]
                    )
                ),
                "tv_exceeds_control_95th_count": int(
                    sum(row["tv_exceeds_control_95th"] for row in selected)
                ),
            }

    linearity_curve_cosines = [
        float(row["median_absolute_curve_cosine"]) for row in linearity_rows
    ]
    linearity_replica_cosines = [
        float(row["median_per_replica_cosine"]) for row in linearity_rows
    ]
    response_tail_mass = [float(row["last_fifty_lag_mass"]) for row in response_rows]
    response_endpoint = [float(row["endpoint_to_peak_ratio"]) for row in response_rows]
    response_slopes = [
        float(row["late_log_magnitude_slope_per_update"]) for row in response_rows
    ]
    summary = {
        "run_count": len(run_rows),
        "all_runs_complete": all(row["status"] == "complete" for row in run_rows),
        "scientific_elapsed_seconds_total": float(
            sum(float(row["elapsed_seconds"]) for row in run_rows)
        ),
        "composition": cross_seed,
        "linearity": {
            "minimum_median_absolute_curve_cosine": min(linearity_curve_cosines),
            "minimum_median_per_replica_cosine": min(linearity_replica_cosines),
            "replica_pair_evaluations_below_0_9": int(
                sum(int(row["replicas_below_0_9_cosine"]) for row in linearity_rows)
            ),
            "replica_pair_evaluations_total": 64 * len(linearity_rows),
        },
        "response_finiteness": {
            "last_fifty_mass_minimum": min(response_tail_mass),
            "last_fifty_mass_median": float(np.median(response_tail_mass)),
            "last_fifty_mass_maximum": max(response_tail_mass),
            "endpoint_to_peak_minimum": min(response_endpoint),
            "endpoint_to_peak_median": float(np.median(response_endpoint)),
            "endpoint_to_peak_maximum": max(response_endpoint),
            "positive_late_slope_count": int(sum(value > 0 for value in response_slopes)),
            "response_count": len(response_rows),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
