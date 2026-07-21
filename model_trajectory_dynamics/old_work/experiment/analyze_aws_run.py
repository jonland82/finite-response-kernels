#!/usr/bin/env python3
"""Derive fixed-decoder information statistics from a collected AWS run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values: np.ndarray, seed: int, samples: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_results", type=Path)
    arguments = parser.parse_args()
    root = arguments.run_results.resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    trajectory = read_csv(root / "trajectory.csv")
    observables = np.load(root / "observables.npz")
    rng = np.random.default_rng(int(config["seed"]))
    labels = rng.integers(
        0,
        2,
        size=(int(config["replicas"]), int(config["maximum_facts"])),
        dtype=np.int64,
    )
    derived: list[dict[str, object]] = []
    for condition in ("closed", "open"):
        condition_rows = [row for row in trajectory if row["condition"] == condition]
        margins = observables[f"{condition}_margins"]
        for index, row in enumerate(condition_rows):
            source_bits = int(row["available_source_bits"])
            selected_margins = np.clip(margins[index, :, :source_bits], -30.0, 30.0)
            selected_labels = labels[:, :source_bits]
            cross_entropy = np.logaddexp(0.0, selected_margins) - selected_labels * selected_margins
            per_world = (1.0 - cross_entropy / math.log(2.0)).sum(axis=1)
            low, high = bootstrap(
                per_world,
                int(config["seed"]) + int(row["step"]) + (0 if condition == "closed" else 10000),
            )
            derived.append(
                {
                    "condition": condition,
                    "step": int(row["step"]),
                    "epoch_equivalent": float(row["epoch_equivalent"]),
                    "available_source_bits": source_bits,
                    "direct_information_lower_bound_bits": float(per_world.mean()),
                    "direct_information_ci_low": low,
                    "direct_information_ci_high": high,
                    "direct_decoder_accuracy": float(
                        np.mean((selected_margins >= 0.0) == selected_labels)
                    ),
                    "crossfit_information_lower_bound_bits": float(
                        row["information_lower_bound_bits"]
                    ),
                    "code_loss_nats": float(row["code_loss_nats"]),
                }
            )
    write_csv(root / "direct_information.csv", derived)

    closed = [row for row in derived if row["condition"] == "closed"]
    opened = [row for row in derived if row["condition"] == "open"]
    plateau = [row for row in closed if int(row["step"]) >= 192]
    slope = float(
        np.polyfit(
            [float(row["step"]) for row in plateau],
            [float(row["direct_information_lower_bound_bits"]) for row in plateau],
            1,
        )[0]
    )
    summary = {
        "closed_final_direct_bits": closed[-1]["direct_information_lower_bound_bits"],
        "closed_final_source_bits": closed[-1]["available_source_bits"],
        "closed_final_direct_fraction": float(
            closed[-1]["direct_information_lower_bound_bits"]
            / closed[-1]["available_source_bits"]
        ),
        "closed_direct_gain_step_192_to_400": float(
            closed[-1]["direct_information_lower_bound_bits"]
            - plateau[0]["direct_information_lower_bound_bits"]
        ),
        "closed_plateau_slope_bits_per_update": slope,
        "closed_code_loss_reduction_step_192_to_400": float(
            1.0 - closed[-1]["code_loss_nats"] / plateau[0]["code_loss_nats"]
        ),
        "open_final_direct_bits": opened[-1]["direct_information_lower_bound_bits"],
        "open_final_source_bits": opened[-1]["available_source_bits"],
        "open_final_direct_fraction": float(
            opened[-1]["direct_information_lower_bound_bits"]
            / opened[-1]["available_source_bits"]
        ),
    }
    (root / "ANALYSIS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for condition, color in (("closed", "#1f77b4"), ("open", "#d62728")):
        rows = [row for row in derived if row["condition"] == condition]
        x = np.array([float(row["epoch_equivalent"]) for row in rows])
        y = np.array([float(row["direct_information_lower_bound_bits"]) for row in rows])
        low = np.array([float(row["direct_information_ci_low"]) for row in rows])
        high = np.array([float(row["direct_information_ci_high"]) for row in rows])
        axis.plot(x, y, marker="o", label=condition, color=color)
        axis.fill_between(x, low, high, color=color, alpha=0.15)
    axis.axhline(16, color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.7)
    axis.set_xlabel("closed-corpus epoch equivalents")
    axis.set_ylabel("fixed-decoder information lower bound (bits)")
    axis.set_title("Closed-source saturation and growing-source control")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(root / "direct_information.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
