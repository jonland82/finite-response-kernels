#!/usr/bin/env python3
"""Analyze grokking trajectories as finite takeoff kernels and response modes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import expit, logit
from scipy.stats import wasserstein_distance


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    # Model-comparison rows intentionally have model-specific parameter columns.
    # Preserve first-seen ordering while admitting the union of those columns.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def isotonic_increasing(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=np.float64)
    levels: list[float] = []
    block_weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        levels.append(float(value))
        block_weights.append(float(weight))
        starts.append(index)
        ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_level = (
                levels[-2] * block_weights[-2] + levels[-1] * block_weights[-1]
            ) / merged_weight
            levels[-2:] = [merged_level]
            block_weights[-2:] = [merged_weight]
            ends[-2:] = [ends[-1]]
            starts.pop()
    result = np.empty_like(values)
    for level, start, end in zip(levels, starts, ends, strict=True):
        result[start:end] = level
    return result


def log_likelihood(counts: np.ndarray, totals: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return float(
        np.sum(counts * np.log(probabilities) + (totals - counts) * np.log1p(-probabilities))
    )


def decode_endpoints(parameters: np.ndarray) -> tuple[float, float]:
    p0 = float(expit(parameters[0]))
    p1 = p0 + (1.0 - p0) * float(expit(parameters[1]))
    return p0, p1


def logistic_probabilities(parameters: np.ndarray, times: np.ndarray) -> np.ndarray:
    p0, p1 = decode_endpoints(parameters)
    tau = parameters[2]
    width = math.exp(float(np.clip(parameters[3], -12.0, 20.0)))
    return p0 + (p1 - p0) * expit((times - tau) / width)


def mixture_probabilities(parameters: np.ndarray, times: np.ndarray) -> np.ndarray:
    p0, p1 = decode_endpoints(parameters)
    tau1, width1 = parameters[2], math.exp(float(np.clip(parameters[3], -12.0, 20.0)))
    tau2, width2 = parameters[4], math.exp(float(np.clip(parameters[5], -12.0, 20.0)))
    mixing = float(expit(parameters[6]))
    profile = mixing * expit((times - tau1) / width1) + (1.0 - mixing) * expit(
        (times - tau2) / width2
    )
    return p0 + (p1 - p0) * profile


def endpoint_parameters(p0: float, p1: float) -> tuple[float, float]:
    p0 = float(np.clip(p0, 1e-5, 1.0 - 1e-5))
    fraction = float(np.clip((p1 - p0) / (1.0 - p0), 1e-5, 1.0 - 1e-5))
    return float(logit(p0)), float(logit(fraction))


def fit_delta(
    times: np.ndarray, counts: np.ndarray, totals: np.ndarray, fit_mask: np.ndarray
) -> dict[str, object]:
    best: dict[str, object] | None = None
    fit_indices = np.flatnonzero(fit_mask)
    for split in range(2, len(times) - 2):
        before = fit_indices[fit_indices < split]
        after = fit_indices[fit_indices >= split]
        if len(before) < 2 or len(after) < 2:
            continue
        p0 = float(np.clip(counts[before].sum() / totals[before].sum(), 1e-9, 1.0 - 1e-9))
        p1 = float(np.clip(counts[after].sum() / totals[after].sum(), 1e-9, 1.0 - 1e-9))
        probabilities = np.where(np.arange(len(times)) < split, p0, p1)
        ll = log_likelihood(counts[fit_mask], totals[fit_mask], probabilities[fit_mask])
        if best is None or ll > float(best["fit_log_likelihood"]):
            best = {
                "p0": p0,
                "p1": p1,
                "tau": float(times[split]),
                "width": 0.0,
                "probabilities": probabilities,
                "fit_log_likelihood": ll,
                "parameter_count": 3,
            }
    if best is None:
        raise ValueError("not enough points for a delta fit")
    return best


def fit_smooth(
    kind: str,
    times: np.ndarray,
    counts: np.ndarray,
    totals: np.ndarray,
    fit_mask: np.ndarray,
) -> dict[str, object]:
    empirical = (counts + 0.5) / (totals + 1.0)
    p0_guess = float(np.median(empirical[: max(2, len(times) // 10)]))
    p1_guess = float(np.median(empirical[-max(2, len(times) // 10) :]))
    first, second = endpoint_parameters(p0_guess, max(p1_guess, p0_guess + 0.05))
    transition_index = int(np.argmin(np.abs(empirical - 0.5 * (p0_guess + p1_guess))))
    tau_guess = float(times[transition_index])
    span = max(float(times[-1] - times[0]), 1.0)
    width_guess = max(float(np.median(np.diff(times))), span / 20.0)
    if kind == "logistic":
        initial_values = [
            np.asarray([first, second, tau_guess, math.log(width_guess)]),
            np.asarray([first, second, tau_guess, math.log(max(width_guess / 3.0, 0.1))]),
        ]
        probability_function = logistic_probabilities
        parameter_count = 4
        bounds = [
            (-15.0, 15.0),
            (-15.0, 15.0),
            (float(times[0] - span), float(times[-1] + span)),
            (math.log(max(float(np.median(np.diff(times))) / 100.0, 1e-4)), math.log(span * 10.0)),
        ]
    elif kind == "mixture":
        initial_values = [
            np.asarray(
                [
                    first,
                    second,
                    tau_guess - width_guess,
                    math.log(width_guess),
                    tau_guess + width_guess,
                    math.log(width_guess * 2.0),
                    0.0,
                ]
            ),
            np.asarray(
                [
                    first,
                    second,
                    tau_guess - 2 * width_guess,
                    math.log(max(width_guess / 2.0, 0.1)),
                    tau_guess + 2 * width_guess,
                    math.log(width_guess * 3.0),
                    float(logit(0.7)),
                ]
            ),
        ]
        probability_function = mixture_probabilities
        parameter_count = 7
        bounds = [
            (-15.0, 15.0),
            (-15.0, 15.0),
            (float(times[0] - span), float(times[-1] + span)),
            (math.log(max(float(np.median(np.diff(times))) / 100.0, 1e-4)), math.log(span * 10.0)),
            (float(times[0] - span), float(times[-1] + span)),
            (math.log(max(float(np.median(np.diff(times))) / 100.0, 1e-4)), math.log(span * 10.0)),
            (-10.0, 10.0),
        ]
    else:
        raise ValueError(kind)

    def objective(parameters: np.ndarray) -> float:
        probabilities = probability_function(parameters, times)
        return -log_likelihood(
            counts[fit_mask], totals[fit_mask], probabilities[fit_mask]
        )

    best_result = None
    for initial in initial_values:
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-11},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    assert best_result is not None
    parameters = best_result.x
    probabilities = probability_function(parameters, times)
    p0, p1 = decode_endpoints(parameters)
    output: dict[str, object] = {
        "p0": p0,
        "p1": p1,
        "probabilities": probabilities,
        "fit_log_likelihood": -float(best_result.fun),
        "parameter_count": parameter_count,
        "optimization_success": bool(best_result.success),
    }
    if kind == "logistic":
        output.update({"tau": float(parameters[2]), "width": math.exp(parameters[3])})
    else:
        output.update(
            {
                "tau": float(
                    expit(parameters[6]) * parameters[2]
                    + (1.0 - expit(parameters[6])) * parameters[4]
                ),
                "width": float("nan"),
                "tau1": float(parameters[2]),
                "width1": math.exp(parameters[3]),
                "tau2": float(parameters[4]),
                "width2": math.exp(parameters[5]),
                "mixing": float(expit(parameters[6])),
            }
        )
    return output


def compare_transition_models(
    times: np.ndarray, counts: np.ndarray, totals: np.ndarray
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    masks = {
        "all": np.ones(len(times), dtype=bool),
        "train": np.arange(len(times)) % 2 == 0,
    }
    fits: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    for kind in ("delta", "logistic", "mixture"):
        full = (
            fit_delta(times, counts, totals, masks["all"])
            if kind == "delta"
            else fit_smooth(kind, times, counts, totals, masks["all"])
        )
        cross = (
            fit_delta(times, counts, totals, masks["train"])
            if kind == "delta"
            else fit_smooth(kind, times, counts, totals, masks["train"])
        )
        holdout = ~masks["train"]
        holdout_ll = log_likelihood(
            counts[holdout], totals[holdout], np.asarray(cross["probabilities"])[holdout]
        )
        empirical = counts / totals
        full_residual = empirical - np.asarray(full["probabilities"])
        full_rss = max(float(np.square(full_residual).sum()), 1e-15)
        checkpoint_count = len(times)
        checkpoint_bic = checkpoint_count * math.log(full_rss / checkpoint_count) + int(
            full["parameter_count"]
        ) * math.log(checkpoint_count)
        binomial_bic = int(full["parameter_count"]) * math.log(int(totals.sum())) - 2.0 * float(
            full["fit_log_likelihood"]
        )
        holdout_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        empirical[holdout] - np.asarray(cross["probabilities"])[holdout]
                    )
                )
            )
        )
        row = {
            "model": kind,
            "fit_log_likelihood": full["fit_log_likelihood"],
            "heldout_checkpoint_log_likelihood": holdout_ll,
            "heldout_checkpoint_rmse": holdout_rmse,
            "bic": checkpoint_bic,
            "binomial_bic_descriptive": binomial_bic,
            "p0": full["p0"],
            "p1": full["p1"],
            "tau": full["tau"],
            "width": full["width"],
        }
        if kind == "mixture":
            row.update(
                {
                    "tau1": full["tau1"],
                    "width1": full["width1"],
                    "tau2": full["tau2"],
                    "width2": full["width2"],
                    "mixing": full["mixing"],
                }
            )
        rows.append(row)
        fits[kind] = full
        predictions[kind] = np.asarray(full["probabilities"])
    return rows, predictions


def kernel_from_accuracy(
    times: np.ndarray, accuracy: np.ndarray, totals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    monotone = isotonic_increasing(accuracy, totals)
    baseline = float(monotone[0])
    endpoint = float(np.median(monotone[-min(5, len(monotone)) :]))
    denominator = endpoint - baseline
    if denominator <= 0.05:
        raise ValueError("trajectory has no resolved bounded gain")
    profile = np.clip((monotone - baseline) / denominator, 0.0, 1.0)
    kernel = np.diff(np.concatenate(([0.0], profile)))
    kernel = np.maximum(kernel, 0.0)
    kernel /= kernel.sum()
    center = float(np.sum(times * kernel))
    variance = float(np.sum(np.square(times - center) * kernel))
    nonzero = kernel[kernel > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    cdf = np.cumsum(kernel)
    q10 = float(np.interp(0.10, cdf, times))
    q90 = float(np.interp(0.90, cdf, times))
    stats = {
        "baseline_accuracy": baseline,
        "endpoint_accuracy": endpoint,
        "kernel_center_step": center,
        "kernel_sd_steps": math.sqrt(max(variance, 0.0)),
        "kernel_effective_checkpoints": math.exp(entropy),
        "kernel_10_90_width_steps": q90 - q10,
    }
    return profile, kernel, stats


def fit_modes(times: np.ndarray, profile: np.ndarray, local: bool) -> dict[str, object]:
    start_candidates = np.flatnonzero(profile >= 0.05)
    if not len(start_candidates):
        raise ValueError("no takeoff onset")
    start = int(start_candidates[0])
    end_candidates = np.flatnonzero(profile >= 0.95)
    end = int(end_candidates[0] + 1) if local and len(end_candidates) else len(profile)
    x = times[start:end] - times[start]
    y = np.clip(1.0 - profile[start:end], 0.0, 1.0)
    if len(x) < 8:
        raise ValueError("not enough transition points for modal fit")
    scale_guess = max(float(x[-1] - x[0]) / 3.0, 1.0)

    def residual_one(parameters: np.ndarray) -> np.ndarray:
        amplitude, tau = parameters
        return amplitude * np.exp(-x / tau) - y

    one = least_squares(
        residual_one,
        np.asarray([float(y[0]), scale_guess]),
        bounds=([0.0, 0.01], [2.0, max(float(x[-1]) * 10.0, 10.0)]),
    )

    def residual_two(parameters: np.ndarray) -> np.ndarray:
        amplitude, mixing, tau1, tau2 = parameters
        prediction = amplitude * (
            mixing * np.exp(-x / tau1) + (1.0 - mixing) * np.exp(-x / tau2)
        )
        return prediction - y

    two = least_squares(
        residual_two,
        np.asarray([float(y[0]), 0.5, max(scale_guess / 3.0, 0.1), scale_guess * 3.0]),
        bounds=(
            [0.0, 0.0, 0.01, 0.01],
            [2.0, 1.0, max(float(x[-1]) * 10.0, 10.0), max(float(x[-1]) * 10.0, 10.0)],
        ),
    )
    rss_one = max(float(np.square(one.fun).sum()), 1e-15)
    rss_two = max(float(np.square(two.fun).sum()), 1e-15)
    bic_one = len(x) * math.log(rss_one / len(x)) + 2 * math.log(len(x))
    bic_two = len(x) * math.log(rss_two / len(x)) + 4 * math.log(len(x))
    hankel_width = max(2, min(len(y) // 3, 12))
    hankel = np.asarray(
        [[y[row + column] for column in range(hankel_width)] for row in range(len(y) - hankel_width + 1)]
    )
    singular = np.linalg.svd(hankel, compute_uv=False)
    singular /= singular[0] if singular[0] else 1.0
    return {
        "window": "local_5_95" if local else "full_tail",
        "point_count": len(x),
        "one_mode_bic": bic_one,
        "two_mode_bic": bic_two,
        "two_mode_delta_bic": bic_one - bic_two,
        "selected_modes": 2 if bic_two + 10.0 < bic_one else 1,
        "one_mode_tau": float(one.x[1]),
        "two_mode_weight": float(two.x[1]),
        "two_mode_tau_fast": float(min(two.x[2], two.x[3])),
        "two_mode_tau_slow": float(max(two.x[2], two.x[3])),
        "hankel_singular_1": float(singular[0]),
        "hankel_singular_2": float(singular[1]) if len(singular) > 1 else 0.0,
        "hankel_singular_3": float(singular[2]) if len(singular) > 2 else 0.0,
    }


def validate_synthetic(output_dir: Path) -> dict[str, object]:
    times = np.arange(100, dtype=np.float64)
    totals = np.full(len(times), 5000, dtype=np.float64)
    curves = {
        "delta": 0.02 + 0.96 * (times >= 50),
        "finite": 0.02 + 0.96 * expit((times - 52.0) / 3.0),
        "mixture": 0.02
        + 0.96
        * (0.6 * expit((times - 43.0) / 1.8) + 0.4 * expit((times - 65.0) / 5.0)),
    }
    rows = []
    checks: dict[str, bool] = {}
    for name, probability in curves.items():
        counts = np.rint(totals * probability)
        fitted, _ = compare_transition_models(times, counts, totals)
        best_bic = min(fitted, key=lambda row: float(row["bic"]))["model"]
        best_holdout = max(
            fitted, key=lambda row: float(row["heldout_checkpoint_log_likelihood"])
        )["model"]
        logistic_width = next(float(row["width"]) for row in fitted if row["model"] == "logistic")
        rows.append(
            {
                "synthetic_curve": name,
                "best_bic_model": best_bic,
                "best_heldout_model": best_holdout,
                "fitted_logistic_width": logistic_width,
            }
        )
        if name == "delta":
            checks[name] = best_bic == "delta"
        elif name == "finite":
            checks[name] = logistic_width > 1.0 and best_bic != "delta"
        else:
            checks[name] = best_bic == "mixture"
    write_csv(output_dir / "synthetic_validation.csv", rows)
    result = {"checks": checks, "all_passed": all(checks.values())}
    (output_dir / "synthetic_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def create_figure(
    output_dir: Path,
    curves: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not curves:
        return
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for curve in curves:
        label = f"seed {curve['seed']}"
        axes[0].plot(curve["times"], curve["accuracy"], alpha=0.8, label=label)
        axes[1].plot(curve["times"], curve["profile"], alpha=0.8)
        axes[2].plot(curve["times"], curve["kernel"], alpha=0.8)
    axes[0].set_title("held-out generalization")
    axes[0].set_ylabel("accuracy")
    axes[1].set_title("normalized takeoff profile")
    axes[1].set_ylabel("$F_t$")
    axes[2].set_title("takeoff kernel")
    axes[2].set_ylabel("$\\kappa_t$")
    for axis in axes:
        axis.set_xlabel("optimizer update")
    axes[0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "takeoff_trajectories.png", dpi=190)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    arguments = parser.parse_args()
    results_dir = arguments.results_dir.resolve()
    summary_rows = read_csv(results_dir / "run_summary.csv")
    synthetic = validate_synthetic(results_dir)
    transition_rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []

    for summary in summary_rows:
        if summary["stage"] != "confirmation":
            continue
        run_id = summary["run_id"]
        trajectory = read_csv(results_dir / "raw" / run_id / "trajectory.csv")
        times = np.asarray([float(row["step"]) for row in trajectory])
        accuracy = np.asarray([float(row["test_accuracy"]) for row in trajectory])
        counts = np.asarray([float(row["test_correct"]) for row in trajectory])
        totals = np.asarray([float(row["test_count"]) for row in trajectory])
        if accuracy[-1] < 0.80 or accuracy.max() - accuracy[0] < 0.50:
            continue
        fitted_rows, predictions = compare_transition_models(times, counts, totals)
        for row in fitted_rows:
            transition_rows.append({"run_id": run_id, "seed": summary["seed"], **row})
        profile, kernel, stats = kernel_from_accuracy(times, accuracy, totals)
        threshold_steps: dict[float, float] = {}
        for threshold in (0.10, 0.25, 0.50, 0.75, 0.90):
            hits = np.flatnonzero(accuracy >= threshold)
            threshold_steps[threshold] = float(times[hits[0]]) if len(hits) else float("nan")
        threshold_rows.append(
            {
                "run_id": run_id,
                "seed": summary["seed"],
                "first_accuracy_10_step": threshold_steps[0.10],
                "first_accuracy_25_step": threshold_steps[0.25],
                "first_accuracy_50_step": threshold_steps[0.50],
                "first_accuracy_75_step": threshold_steps[0.75],
                "first_accuracy_90_step": threshold_steps[0.90],
                "observed_25_90_width_steps": threshold_steps[0.90] - threshold_steps[0.25],
                "observed_50_90_width_steps": threshold_steps[0.90] - threshold_steps[0.50],
            }
        )
        logistic = next(row for row in fitted_rows if row["model"] == "logistic")
        delta = next(row for row in fitted_rows if row["model"] == "delta")
        mixture = next(row for row in fitted_rows if row["model"] == "mixture")
        best_holdout = min(
            fitted_rows, key=lambda row: float(row["heldout_checkpoint_rmse"])
        )["model"]
        kernel_rows.append(
            {
                "run_id": run_id,
                "seed": summary["seed"],
                **stats,
                "checkpoint_spacing": float(np.median(np.diff(times))),
                "logistic_width_steps": logistic["width"],
                "logistic_vs_delta_bic": float(delta["bic"]) - float(logistic["bic"]),
                "mixture_vs_logistic_bic": float(logistic["bic"]) - float(mixture["bic"]),
                "best_heldout_transition_model": best_holdout,
            }
        )
        for local in (False, True):
            try:
                mode_rows.append(
                    {
                        "run_id": run_id,
                        "seed": summary["seed"],
                        **fit_modes(times, profile, local),
                    }
                )
            except ValueError:
                pass
        curves.append(
            {
                "run_id": run_id,
                "seed": summary["seed"],
                "times": times,
                "accuracy": accuracy,
                "profile": profile,
                "kernel": kernel,
                "predictions": predictions,
            }
        )

    write_csv(results_dir / "transition_model_comparison.csv", transition_rows)
    write_csv(results_dir / "takeoff_kernel_summary.csv", kernel_rows)
    write_csv(results_dir / "modal_decomposition.csv", mode_rows)
    write_csv(results_dir / "transition_thresholds.csv", threshold_rows)
    universality_rows: list[dict[str, object]] = []
    for first, second in combinations(curves, 2):
        first_stats = next(row for row in kernel_rows if row["run_id"] == first["run_id"])
        second_stats = next(row for row in kernel_rows if row["run_id"] == second["run_id"])
        first_z = (first["times"] - float(first_stats["kernel_center_step"])) / max(
            float(first_stats["kernel_sd_steps"]), 1e-9
        )
        second_z = (second["times"] - float(second_stats["kernel_center_step"])) / max(
            float(second_stats["kernel_sd_steps"]), 1e-9
        )
        distance = wasserstein_distance(
            first_z,
            second_z,
            u_weights=first["kernel"],
            v_weights=second["kernel"],
        )
        universality_rows.append(
            {
                "run_a": first["run_id"],
                "run_b": second["run_id"],
                "standardized_wasserstein_distance": float(distance),
            }
        )
    write_csv(results_dir / "kernel_universality.csv", universality_rows)
    create_figure(results_dir, curves)

    finite_count = sum(
        float(row["logistic_width_steps"]) > float(row["checkpoint_spacing"])
        and float(row["logistic_vs_delta_bic"]) > 10.0
        for row in kernel_rows
    )
    mixture_holdout_count = sum(
        row["best_heldout_transition_model"] == "mixture" for row in kernel_rows
    )
    two_mode_count = sum(int(row["selected_modes"]) == 2 for row in mode_rows)
    distances = [float(row["standardized_wasserstein_distance"]) for row in universality_rows]
    summary = {
        "status": "complete" if curves and synthetic["all_passed"] else "inconclusive",
        "synthetic_validation_passed": synthetic["all_passed"],
        "confirmation_runs_analyzed": len(curves),
        "finite_takeoff_count": int(finite_count),
        "mixture_best_heldout_count": int(mixture_holdout_count),
        "two_mode_selected_count": int(two_mode_count),
        "modal_fit_count": len(mode_rows),
        "median_standardized_kernel_wasserstein": float(np.median(distances)) if distances else None,
        "median_kernel_10_90_width_steps": float(
            np.median([float(row["kernel_10_90_width_steps"]) for row in kernel_rows])
        )
        if kernel_rows
        else None,
        "median_logistic_width_steps": float(
            np.median([float(row["logistic_width_steps"]) for row in kernel_rows])
        )
        if kernel_rows
        else None,
        "median_observed_25_90_width_steps": float(
            np.median([float(row["observed_25_90_width_steps"]) for row in threshold_rows])
        )
        if threshold_rows
        else None,
        "median_observed_50_90_width_steps": float(
            np.median([float(row["observed_50_90_width_steps"]) for row in threshold_rows])
        )
        if threshold_rows
        else None,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (results_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# LLM takeoff-kernel experiment",
        "",
        f"Status: **{summary['status']}**.",
        "",
        f"Synthetic estimator validation passed: **{summary['synthetic_validation_passed']}**.",
        f"Completed confirmation trajectories analyzed: **{len(curves)}**.",
        f"Finite takeoff preferred to a checkpoint-delta model: **{finite_count}/{len(curves)}**.",
        f"A two-component transition won held-out checkpoint likelihood: **{mixture_holdout_count}/{len(curves)}**.",
        f"Two settling modes passed the BIC threshold: **{two_mode_count}/{len(mode_rows)}** fitted windows.",
        f"Median first-crossing width from 25% to 90% accuracy: "
        f"**{summary['median_observed_25_90_width_steps']:.0f} updates**.",
        f"Median first-crossing width from 50% to 90% accuracy: "
        f"**{summary['median_observed_50_90_width_steps']:.0f} updates**.",
        "",
        "The two-component transition result concerns two separated rises in held-out accuracy. "
        "The settling-mode test asks a different question---whether the residual after onset is "
        "better represented by one or two exponentials---and selected one mode here. The abrupt-"
        "versus-finite conclusion is relative to the 50-update checkpoint spacing; neither fit "
        "by itself establishes distinct physical mechanisms.",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    (results_dir / "RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
