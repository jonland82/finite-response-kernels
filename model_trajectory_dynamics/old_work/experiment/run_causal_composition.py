#!/usr/bin/env python3
"""Long-horizon response and convolution-semigroup falsification pilot.

The script reuses the miniature decoder-only language model from
``run_experiment.py``. For paired +/- example-weight interventions it records
per-replica signed derivatives over long horizons. It then asks whether a
measured two-block response resembles the convolution of normalized response
curves from the two component blocks.

This is an operational semigroup test of the manuscript's proposed kernel
construction. It is not a proof that the chosen scalar observable is a
sufficient interface between training blocks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from run_experiment import (
    ID_BASE,
    BatchedTinyLM,
    Config,
    RuntimeGuard,
    build_corpus,
    build_labels,
    clone_training_state,
    injection_step,
    make_schedules,
    regular_step,
    selected_margin,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values)
    return values / total


def distribution_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    measured = normalize(measured)
    predicted = normalize(predicted)
    midpoint = 0.5 * (measured + predicted)
    measured_mask = measured > 0
    predicted_mask = predicted > 0
    js_nats = 0.5 * float(
        np.sum(measured[measured_mask] * np.log(measured[measured_mask] / midpoint[measured_mask]))
        + np.sum(predicted[predicted_mask] * np.log(predicted[predicted_mask] / midpoint[predicted_mask]))
    )
    total_variation = 0.5 * float(np.sum(np.abs(measured - predicted)))
    wasserstein_lags = float(np.sum(np.abs(np.cumsum(measured) - np.cumsum(predicted))))
    return {
        "js_divergence_bits": js_nats / math.log(2.0),
        "total_variation": total_variation,
        "wasserstein_lags": wasserstein_lags,
        "wasserstein_fraction_of_horizon": wasserstein_lags / max(1, len(measured) - 1),
    }


def response_summary(injection_step: int, derivative: np.ndarray) -> dict[str, object]:
    magnitude = np.mean(np.abs(derivative), axis=1)
    mass = normalize(magnitude)
    nonzero = mass[mass > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) else 0.0
    tail_width = min(50, len(mass))
    peak = float(np.max(magnitude)) if len(magnitude) else 0.0
    endpoint = float(np.mean(magnitude[-min(20, len(magnitude)) :]))
    positive = np.maximum(magnitude, max(peak * 1e-12, 1e-30))
    fit_start = len(positive) // 2
    slope = float(
        np.polyfit(np.arange(fit_start, len(positive)), np.log(positive[fit_start:]), 1)[0]
    )
    absolute_total = float(np.mean(np.abs(derivative), axis=1).sum())
    signed_total = float(np.abs(np.mean(derivative, axis=1)).sum())
    return {
        "injection_step": injection_step,
        "observed_horizon": len(derivative) - 1,
        "effective_lag_count": math.exp(entropy),
        "peak_mass": float(mass.max()) if len(mass) else 0.0,
        "last_fifty_lag_mass": float(mass[-tail_width:].sum()),
        "endpoint_to_peak_ratio": endpoint / peak if peak > 0 else 0.0,
        "late_log_magnitude_slope_per_update": slope,
        "signed_to_absolute_ratio": signed_total / absolute_total if absolute_total > 0 else 0.0,
    }


def run_response(
    config: Config,
    replicas: int,
    injection_time: int,
    horizon: int,
    epsilon: float,
    schedule: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    guard: RuntimeGuard,
    smoke: bool,
) -> np.ndarray:
    corpus = build_corpus(labels, device)
    selected = torch.arange(replicas, device=device) % config.initial_facts
    model = BatchedTinyLM(config, replicas, ID_BASE + config.maximum_facts).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    reserve = 0 if smoke else 120
    for step in range(injection_time):
        guard.check(reserve_seconds=reserve)
        regular_step(model, optimizer, corpus, schedule[step])
    plus_model, plus_optimizer = clone_training_state(model, optimizer, config, replicas, device)
    minus_model, minus_optimizer = clone_training_state(model, optimizer, config, replicas, device)
    del model, optimizer
    injection_step(
        plus_model,
        plus_optimizer,
        corpus,
        schedule[injection_time],
        selected,
        1.0 + epsilon,
    )
    injection_step(
        minus_model,
        minus_optimizer,
        corpus,
        schedule[injection_time],
        selected,
        1.0 - epsilon,
    )
    plus_values = [selected_margin(plus_model, corpus, selected)]
    minus_values = [selected_margin(minus_model, corpus, selected)]
    for lag in range(1, horizon + 1):
        guard.check(reserve_seconds=reserve)
        step = injection_time + lag
        regular_step(plus_model, plus_optimizer, corpus, schedule[step])
        regular_step(minus_model, minus_optimizer, corpus, schedule[step])
        plus_values.append(selected_margin(plus_model, corpus, selected))
        minus_values.append(selected_margin(minus_model, corpus, selected))
    derivative = (np.stack(plus_values) - np.stack(minus_values)) / (2.0 * epsilon)
    del plus_model, minus_model, plus_optimizer, minus_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return derivative.astype(np.float32)


def composition_metrics(
    start_derivative: np.ndarray,
    second_derivative: np.ndarray,
    block_length: int,
    aggregation: str = "mean",
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    def aggregate(values: np.ndarray) -> np.ndarray:
        absolute = np.abs(values)
        if aggregation == "mean":
            return np.mean(absolute, axis=1)
        if aggregation == "median":
            return np.median(absolute, axis=1)
        if aggregation == "trimmed_mean":
            trim = max(1, int(absolute.shape[1] * 0.10))
            if absolute.shape[1] <= 2 * trim:
                return np.mean(absolute, axis=1)
            return np.sort(absolute, axis=1)[:, trim:-trim].mean(axis=1)
        raise ValueError(f"unknown aggregation: {aggregation}")

    first_magnitude = aggregate(start_derivative[: block_length + 1])
    second_magnitude = aggregate(second_derivative[: block_length + 1])
    measured = normalize(aggregate(start_derivative[: 2 * block_length + 1]))
    predicted = normalize(np.convolve(normalize(first_magnitude), normalize(second_magnitude)))
    return distribution_metrics(measured, predicted), measured, predicted


def split_half_noise_floor(
    start_derivative: np.ndarray,
    second_derivative: np.ndarray,
    block_length: int,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Estimate finite-replica curve disagreement with random equal-size splits."""
    replicas = start_derivative.shape[1]
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
        first_indices = permutation[:half]
        second_indices = permutation[half : 2 * half]
        _, measured_a, predicted_a = composition_metrics(
            start_derivative[:, first_indices],
            second_derivative[:, first_indices],
            block_length,
        )
        _, measured_b, predicted_b = composition_metrics(
            start_derivative[:, second_indices],
            second_derivative[:, second_indices],
            block_length,
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


def bootstrap_composition(
    start_derivative: np.ndarray,
    second_derivative: np.ndarray,
    block_length: int,
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    replicas = start_derivative.shape[1]
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "js_divergence_bits": [],
        "total_variation": [],
        "wasserstein_fraction_of_horizon": [],
    }
    for _ in range(samples):
        indices = rng.integers(0, replicas, size=replicas)
        metrics, _, _ = composition_metrics(
            start_derivative[:, indices], second_derivative[:, indices], block_length
        )
        for key in values:
            values[key].append(metrics[key])
    return {
        key: (float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)))
        for key, sampled in values.items()
    }


def create_figure(
    output_dir: Path,
    curve_rows: list[tuple[int, np.ndarray, np.ndarray]],
    response_arrays: dict[int, np.ndarray],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"plotting unavailable: {error}", file=sys.stderr)
        return
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.2))
    for axis, (start, measured, predicted) in zip(axes[0], curve_rows, strict=True):
        axis.plot(measured, label="measured 2-block", linewidth=2)
        axis.plot(predicted, label="convolved blocks", linewidth=1.5)
        axis.set_title(f"start={start}")
        axis.set_xlabel("lag")
        axis.set_ylabel("normalized mass")
        axis.legend(frameon=False, fontsize=8)
    for axis, injection_time in zip(axes[1], sorted(response_arrays)[:3], strict=True):
        derivative = response_arrays[injection_time]
        magnitude = np.mean(np.abs(derivative), axis=1)
        axis.plot(magnitude)
        axis.set_yscale("log")
        axis.set_title(f"long response, t={injection_time}")
        axis.set_xlabel("lag")
        axis.set_ylabel("mean absolute derivative")
    figure.tight_layout()
    figure.savefig(output_dir / "causal_composition.png", dpi=180)
    plt.close(figure)


def write_report(
    output_dir: Path,
    response_rows: list[dict[str, object]],
    composition_rows: list[dict[str, object]],
    linearity: dict[str, float],
    metadata: dict[str, object],
) -> None:
    lines = [
        "# Long-horizon causal composition pilot",
        "",
        "## Response finiteness diagnostics",
        "",
        "| injection | horizon | effective lags | last-50 mass | endpoint/peak | late log slope | signed/absolute |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in response_rows:
        lines.append(
            f"| {int(row['injection_step'])} | {int(row['observed_horizon'])} | "
            f"{float(row['effective_lag_count']):.1f} | {float(row['last_fifty_lag_mass']):.3f} | "
            f"{float(row['endpoint_to_peak_ratio']):.3f} | "
            f"{float(row['late_log_magnitude_slope_per_update']):.5f} | "
            f"{float(row['signed_to_absolute_ratio']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Operational two-block composition",
            "",
            "| start | block length | JS (bits) | TV | Wasserstein/horizon |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in composition_rows:
        lines.append(
            f"| {int(row['start_step'])} | {int(row['block_length'])} | "
            f"{float(row['js_divergence_bits']):.3f} "
            f"[{float(row['js_ci_low']):.3f}, {float(row['js_ci_high']):.3f}] | "
            f"{float(row['total_variation']):.3f} "
            f"[{float(row['tv_ci_low']):.3f}, {float(row['tv_ci_high']):.3f}] | "
            f"{float(row['wasserstein_fraction_of_horizon']):.3f} "
            f"[{float(row['wasserstein_ci_low']):.3f}, {float(row['wasserstein_ci_high']):.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Local-linearity check",
            "",
            f"At injection step {int(linearity['injection_step'])}, derivatives estimated with "
            f"epsilon={linearity['epsilon_small']:.3f} and epsilon={linearity['epsilon_large']:.3f} "
            f"had relative L2 discrepancy {linearity['relative_l2_discrepancy']:.4f} and "
            f"cosine similarity {linearity['cosine_similarity']:.4f}.",
            "",
            "## Interpretation boundary",
            "",
            "The composition comparison normalizes the first block of a response and a separate "
            "response beginning at the second block, convolves those curves, and compares the "
            "result with the independently retained two-block response from the first injection. "
            "Failure rejects this operational scalar semigroup construction. Success would not "
            "prove that the scalar observable is a sufficient interface for model training.",
            "",
            "```json",
            json.dumps(metadata, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    (output_dir / "RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="composition_results")
    parser.add_argument("--replicas", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--block-length", type=int, default=100)
    parser.add_argument("--max-runtime-seconds", type=int, default=1200)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.smoke:
        arguments.replicas = min(arguments.replicas, 4)
        arguments.horizon = 10
        arguments.block_length = 5
        arguments.bootstrap_samples = min(arguments.bootstrap_samples, 20)
        starts = (5,)
    else:
        starts = (50, 200, 400)
    injection_times = sorted(set(starts) | {start + arguments.block_length for start in starts})
    steps = max(injection_times) + arguments.horizon + 2
    output_dir = Path(arguments.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Config(
        output_dir=str(output_dir),
        replicas=arguments.replicas,
        steps=steps,
        influence_replicas=arguments.replicas,
        influence_horizon=arguments.horizon,
        max_runtime_seconds=arguments.max_runtime_seconds,
        bootstrap_samples=arguments.bootstrap_samples,
        seed=arguments.seed,
    )
    if arguments.smoke:
        config.d_model = 32
        config.layers = 1
        config.ff_width = 64
        config.initial_facts = 4
        config.maximum_facts = 8
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guard = RuntimeGuard(arguments.max_runtime_seconds)
    closed_schedule, _, _ = make_schedules(config)
    labels = build_labels(config, arguments.replicas)
    responses: dict[int, np.ndarray] = {}
    response_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []
    robust_composition_rows: list[dict[str, object]] = []
    noise_floor_rows: list[dict[str, object]] = []
    linearity_rows: list[dict[str, object]] = []
    curve_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    status = "complete"
    linearity: dict[str, float] = {}
    try:
        for injection_time in injection_times:
            derivative = run_response(
                config,
                arguments.replicas,
                injection_time,
                arguments.horizon,
                config.influence_epsilon,
                closed_schedule,
                labels,
                device,
                guard,
                arguments.smoke,
            )
            responses[injection_time] = derivative
            summary = response_summary(injection_time, derivative)
            response_rows.append(summary)
            np.savez_compressed(
                output_dir / "response_derivatives.npz",
                **{f"step_{step:04d}": values for step, values in responses.items()},
            )
            print(
                f"response t={injection_time} tail50={summary['last_fifty_lag_mass']:.3f} "
                f"endpoint/peak={summary['endpoint_to_peak_ratio']:.3f}",
                flush=True,
            )
        for start in starts:
            second = start + arguments.block_length
            metrics, measured, predicted = composition_metrics(
                responses[start], responses[second], arguments.block_length
            )
            intervals = bootstrap_composition(
                responses[start],
                responses[second],
                arguments.block_length,
                arguments.bootstrap_samples,
                config.seed + start,
            )
            row: dict[str, object] = {
                "start_step": start,
                "second_step": second,
                "block_length": arguments.block_length,
                **metrics,
                "js_ci_low": intervals["js_divergence_bits"][0],
                "js_ci_high": intervals["js_divergence_bits"][1],
                "tv_ci_low": intervals["total_variation"][0],
                "tv_ci_high": intervals["total_variation"][1],
                "wasserstein_ci_low": intervals["wasserstein_fraction_of_horizon"][0],
                "wasserstein_ci_high": intervals["wasserstein_fraction_of_horizon"][1],
            }
            composition_rows.append(row)
            for aggregation in ("mean", "median", "trimmed_mean"):
                robust_metrics, _, _ = composition_metrics(
                    responses[start],
                    responses[second],
                    arguments.block_length,
                    aggregation=aggregation,
                )
                robust_composition_rows.append(
                    {
                        "start_step": start,
                        "aggregation": aggregation,
                        **robust_metrics,
                    }
                )
            control = split_half_noise_floor(
                responses[start],
                responses[second],
                arguments.block_length,
                arguments.bootstrap_samples,
                config.seed + 50000 + start,
            )
            noise_floor_rows.append(
                {
                    "start_step": start,
                    **metrics,
                    **control,
                    "js_exceeds_control_95th": metrics["js_divergence_bits"]
                    > control["js_divergence_bits_control_95th"],
                    "tv_exceeds_control_95th": metrics["total_variation"]
                    > control["total_variation_control_95th"],
                    "wasserstein_exceeds_control_95th": metrics[
                        "wasserstein_fraction_of_horizon"
                    ]
                    > control["wasserstein_fraction_of_horizon_control_95th"],
                }
            )
            curve_rows.append((start, measured, predicted))
            print(
                f"composition start={start} JS={metrics['js_divergence_bits']:.3f} "
                f"TV={metrics['total_variation']:.3f} "
                f"W/H={metrics['wasserstein_fraction_of_horizon']:.3f}",
                flush=True,
            )
        linearity_start = starts[len(starts) // 2]
        linearity_derivatives: dict[float, np.ndarray] = {
            0.1: responses[linearity_start][: min(arguments.horizon, 200) + 1]
        }
        for epsilon in (0.025, 0.05):
            linearity_derivatives[epsilon] = run_response(
                config,
                arguments.replicas,
                linearity_start,
                min(arguments.horizon, 200),
                epsilon,
                closed_schedule,
                labels,
                device,
                guard,
                arguments.smoke,
            )
        for epsilon_small, epsilon_large in ((0.025, 0.05), (0.05, 0.1)):
            small_values = linearity_derivatives[epsilon_small].astype(np.float64)
            large_values = linearity_derivatives[epsilon_large].astype(np.float64)
            small_flat_pair = small_values.ravel()
            large_flat_pair = large_values.ravel()
            per_replica_cosines = []
            for replica in range(arguments.replicas):
                first = small_values[:, replica]
                second_values = large_values[:, replica]
                pair_denominator = float(np.linalg.norm(first) * np.linalg.norm(second_values))
                per_replica_cosines.append(
                    float(np.dot(first, second_values) / pair_denominator)
                    if pair_denominator
                    else 1.0
                )
            linearity_rows.append(
                {
                    "epsilon_small": epsilon_small,
                    "epsilon_large": epsilon_large,
                    "relative_l2_discrepancy": float(
                        np.linalg.norm(small_flat_pair - large_flat_pair)
                        / np.linalg.norm(large_flat_pair)
                    ),
                    "aggregate_cosine": float(
                        np.dot(small_flat_pair, large_flat_pair)
                        / (np.linalg.norm(small_flat_pair) * np.linalg.norm(large_flat_pair))
                    ),
                    "median_absolute_curve_cosine": float(
                        np.dot(
                            np.median(np.abs(small_values), axis=1),
                            np.median(np.abs(large_values), axis=1),
                        )
                        / (
                            np.linalg.norm(np.median(np.abs(small_values), axis=1))
                            * np.linalg.norm(np.median(np.abs(large_values), axis=1))
                        )
                    ),
                    "median_per_replica_cosine": float(np.median(per_replica_cosines)),
                    "replicas_below_0_9_cosine": int(
                        np.sum(np.asarray(per_replica_cosines) < 0.9)
                    ),
                }
            )
        epsilon_small = 0.05
        small_derivative = linearity_derivatives[epsilon_small]
        large_derivative = linearity_derivatives[0.1]
        small_flat = small_derivative.astype(np.float64).ravel()
        large_flat = large_derivative.astype(np.float64).ravel()
        denominator = float(np.linalg.norm(large_flat))
        linearity = {
            "injection_step": linearity_start,
            "epsilon_small": epsilon_small,
            "epsilon_large": config.influence_epsilon,
            "relative_l2_discrepancy": float(np.linalg.norm(small_flat - large_flat) / denominator),
            "cosine_similarity": float(
                np.dot(small_flat, large_flat)
                / (np.linalg.norm(small_flat) * np.linalg.norm(large_flat))
            ),
        }
        np.savez_compressed(
            output_dir / "linearity_derivatives.npz",
            epsilon_0_025=linearity_derivatives[0.025],
            epsilon_0_050=linearity_derivatives[0.05],
            epsilon_0_100=linearity_derivatives[0.1],
        )
    except TimeoutError as error:
        status = f"partial_timeout: {error}"
        print(status, file=sys.stderr, flush=True)
    finally:
        write_csv(output_dir / "response_summary.csv", response_rows)
        write_csv(output_dir / "composition_summary.csv", composition_rows)
        write_csv(output_dir / "composition_robustness.csv", robust_composition_rows)
        write_csv(output_dir / "noise_floor_summary.csv", noise_floor_rows)
        write_csv(output_dir / "linearity_summary.csv", linearity_rows)
        lag_rows: list[dict[str, object]] = []
        for injection_time, derivative in sorted(responses.items()):
            magnitude = np.mean(np.abs(derivative), axis=1)
            mass = normalize(magnitude)
            signed = np.mean(derivative, axis=1)
            for lag in range(len(derivative)):
                lag_rows.append(
                    {
                        "injection_step": injection_time,
                        "lag": lag,
                        "mean_absolute_derivative": float(magnitude[lag]),
                        "normalized_mass": float(mass[lag]),
                        "mean_signed_derivative": float(signed[lag]),
                    }
                )
        write_csv(output_dir / "response_lags.csv", lag_rows)
        if curve_rows and len(curve_rows) == 3:
            create_figure(output_dir, curve_rows, responses)
        script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        base_hash = hashlib.sha256((Path(__file__).parent / "run_experiment.py").read_bytes()).hexdigest()
        metadata: dict[str, object] = {
            "status": status,
            "replicas": arguments.replicas,
            "horizon": arguments.horizon,
            "block_length": arguments.block_length,
            "starts": list(starts),
            "injection_times": injection_times,
            "epsilon": config.influence_epsilon,
            "bootstrap_samples": arguments.bootstrap_samples,
            "elapsed_seconds": guard.elapsed,
            "seed": config.seed,
            "device": str(device),
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "aws_instance_id": os.environ.get("AWS_INSTANCE_ID"),
            "aws_region": os.environ.get("AWS_REGION"),
            "script_sha256": script_hash,
            "base_runner_sha256": base_hash,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        if response_rows and composition_rows and linearity:
            write_report(output_dir, response_rows, composition_rows, linearity, metadata)
    print(f"status={status} elapsed_seconds={guard.elapsed:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
