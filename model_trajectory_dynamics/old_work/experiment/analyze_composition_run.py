#!/usr/bin/env python3
"""Robustness analysis for a completed causal-composition run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    return values / total if total > 0 else np.zeros_like(values)


def metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    measured = normalize(first[:201])
    predicted = normalize(
        np.convolve(normalize(first[:101]), normalize(second[:101]))
    )
    midpoint = 0.5 * (measured + predicted)
    measured_mask = measured > 0
    predicted_mask = predicted > 0
    js_nats = 0.5 * (
        np.sum(measured[measured_mask] * np.log(measured[measured_mask] / midpoint[measured_mask]))
        + np.sum(predicted[predicted_mask] * np.log(predicted[predicted_mask] / midpoint[predicted_mask]))
    )
    return {
        "js_divergence_bits": float(js_nats / math.log(2.0)),
        "total_variation": float(0.5 * np.abs(measured - predicted).sum()),
        "wasserstein_fraction_of_horizon": float(
            np.abs(np.cumsum(measured) - np.cumsum(predicted)).sum() / 200.0
        ),
    }


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_results", type=Path)
    arguments = parser.parse_args()
    run_results = arguments.run_results.resolve()

    responses = np.load(run_results / "response_derivatives.npz")
    response_concentration: dict[str, dict[str, float | int]] = {}
    for name in responses.files:
        values = responses[name].astype(np.float64)
        per_replica_l1 = np.abs(values).sum(axis=0)
        largest = int(np.argmax(per_replica_l1))
        response_concentration[name] = {
            "largest_replica": largest,
            "largest_replica_l1_share": float(per_replica_l1[largest] / per_replica_l1.sum()),
            "largest_replica_l2_energy_share": float(
                np.square(values[:, largest]).sum() / np.square(values).sum()
            ),
        }

    composition: dict[str, dict[str, dict[str, float]]] = {}
    for start in (50, 200, 400):
        first = responses[f"step_{start:04d}"].astype(np.float64)
        second = responses[f"step_{start + 100:04d}"].astype(np.float64)
        trim = max(1, int(first.shape[1] * 0.10))
        first_sorted = np.sort(np.abs(first), axis=1)
        second_sorted = np.sort(np.abs(second), axis=1)
        composition[str(start)] = {
            "mean_absolute": metrics(np.mean(np.abs(first), axis=1), np.mean(np.abs(second), axis=1)),
            "median_absolute": metrics(np.median(np.abs(first), axis=1), np.median(np.abs(second), axis=1)),
            "ten_percent_trimmed_mean_absolute": metrics(
                first_sorted[:, trim:-trim].mean(axis=1),
                second_sorted[:, trim:-trim].mean(axis=1),
            ),
        }

    linearity = np.load(run_results / "linearity_derivatives.npz")
    small = linearity["epsilon_small"].astype(np.float64)
    large = linearity["epsilon_large"].astype(np.float64)
    per_replica_cosines = np.asarray(
        [cosine(small[:, index], large[:, index]) for index in range(small.shape[1])]
    )
    small_energy = np.square(small).sum(axis=0)
    large_energy = np.square(large).sum(axis=0)
    linearity_summary = {
        "raw_flat_cosine": cosine(small.ravel(), large.ravel()),
        "median_absolute_curve_cosine": cosine(
            np.median(np.abs(small), axis=1), np.median(np.abs(large), axis=1)
        ),
        "per_replica_cosine_median": float(np.median(per_replica_cosines)),
        "per_replica_cosine_minimum": float(np.min(per_replica_cosines)),
        "replicas_below_0_9_cosine": int(np.sum(per_replica_cosines < 0.9)),
        "epsilon_small_top_replica": int(np.argmax(small_energy)),
        "epsilon_small_top_replica_energy_share": float(np.max(small_energy) / small_energy.sum()),
        "epsilon_large_top_replica": int(np.argmax(large_energy)),
        "epsilon_large_top_replica_energy_share": float(np.max(large_energy) / large_energy.sum()),
    }

    output = {
        "response_concentration": response_concentration,
        "composition_robustness": composition,
        "linearity_robustness": linearity_summary,
    }
    output_path = run_results / "robustness_analysis.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
