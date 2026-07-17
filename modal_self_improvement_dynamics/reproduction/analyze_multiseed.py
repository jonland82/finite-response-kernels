"""Aggregate held-out modal audits across fresh-seed reproduction runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_reproduction import (
    SERIES,
    TRIPLETS,
    fit_one,
    fit_shared_rank_one,
    read_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", nargs="+", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--output-directory", type=Path, default=Path("runs"))
    return parser.parse_args()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pseudolabel_stats(path: Path) -> dict[str, float | int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        "count": len(rows),
        "accuracy": float(np.mean([row["selected"]["correct"] for row in rows])),
        "fallback_fraction": float(np.mean([row["fallback"] for row in rows])),
    }


def main() -> None:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    loaded: list[tuple[int, dict[str, np.ndarray]]] = []

    for path in args.trajectories:
        data = read_trajectory(path)
        rows = csv_rows(path)
        seed = int(rows[0]["seed"])
        checkpoint_count = data["epoch"].size
        train_count = max(
            4,
            min(checkpoint_count - 2, int(args.train_fraction * checkpoint_count)),
        )
        series = {
            name: fit_one(data[name], data["epoch"], train_count) for name in SERIES
        }
        shared: dict[str, object] = {}
        for label, names in TRIPLETS.items():
            values = np.vstack([data[name] for name in names])
            shared_fit = fit_shared_rank_one(values, data["epoch"], train_count)
            train_ranges = np.ptp(values[:, :train_count], axis=1)
            errors = []
            for row, name in enumerate(names):
                forecast = np.asarray(series[name]["rank_1"]["forecast"])
                errors.append(
                    (forecast - values[row, train_count:]) / train_ranges[row]
                )
            separate_mse = float(np.mean(np.concatenate(errors) ** 2))
            shared[label] = {
                "mode_per_epoch": shared_fit["mode_per_epoch"],
                "shared_rank1_heldout_mse": shared_fit["heldout_mse"],
                "separate_rank1_heldout_mse": separate_mse,
                "factor_shared_over_separate": float(
                    shared_fit["heldout_mse"] / (separate_mse + 1e-15)
                ),
            }
        runs.append(
            {
                "seed": seed,
                "trajectory": str(path),
                "checkpoint_count": int(checkpoint_count),
                "train_count": train_count,
                "pseudolabels": pseudolabel_stats(path.with_name("pseudolabels.jsonl")),
                "baseline": {
                    name: float(rows[0][name])
                    for name in (
                        "solver_accuracy",
                        "verifier_accuracy",
                        "solver_normalized_uncertainty",
                        "verifier_normalized_uncertainty",
                        "normalized_uncertainty_gap",
                    )
                },
                "final": {
                    name: float(rows[-1][name])
                    for name in (
                        "solver_accuracy",
                        "verifier_accuracy",
                        "solver_normalized_uncertainty",
                        "verifier_normalized_uncertainty",
                        "normalized_uncertainty_gap",
                    )
                },
                "series": series,
                "shared_mode": shared,
            }
        )
        loaded.append((seed, data))

    summary: dict[str, object] = {
        "seed_count": len(runs),
        "seeds": [run["seed"] for run in runs],
        "pseudolabel_accuracy": [run["pseudolabels"]["accuracy"] for run in runs],
        "solver_accuracy_baseline": [
            run["baseline"]["solver_accuracy"] for run in runs
        ],
        "solver_accuracy_final": [run["final"]["solver_accuracy"] for run in runs],
        "verifier_accuracy_baseline": [
            run["baseline"]["verifier_accuracy"] for run in runs
        ],
        "verifier_accuracy_final": [
            run["final"]["verifier_accuracy"] for run in runs
        ],
        "series": {},
        "shared_mode": {},
    }
    for name in SERIES:
        factors = [
            run["series"][name]["rank_1"]["heldout_mse"]
            / (run["series"][name]["rank_2"]["heldout_mse"] + 1e-15)
            for run in runs
        ]
        modal_over_persistence = [
            run["series"][name]["mse_factor_best_modal_over_persistence"]
            for run in runs
        ]
        summary["series"][name] = {
            "rank2_wins": int(sum(factor > 1.0 for factor in factors)),
            "mse_factors_rank1_over_rank2": factors,
            "median_mse_factor_rank1_over_rank2": float(np.median(factors)),
            "best_modal_beats_persistence": int(
                sum(factor < 1.0 for factor in modal_over_persistence)
            ),
            "mse_factors_best_modal_over_persistence": modal_over_persistence,
            "median_mse_factor_best_modal_over_persistence": float(
                np.median(modal_over_persistence)
            ),
        }
    for label in TRIPLETS:
        factors = [run["shared_mode"][label]["factor_shared_over_separate"] for run in runs]
        summary["shared_mode"][label] = {
            "separate_modes_win": int(sum(factor > 1.0 for factor in factors)),
            "mse_factors_shared_over_separate": factors,
            "median_mse_factor_shared_over_separate": float(np.median(factors)),
            "qualification": (
                "descriptive: each gap is algebraically determined by its solver "
                "and verifier series"
            ),
        }

    output = {"summary": summary, "runs": runs}
    json_path = args.output_directory / "multiseed_modal_audit.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    names = TRIPLETS["per_token_nll"]
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharex=True)
    for seed, data in loaded:
        for axis, name in zip(axes, names):
            axis.plot(data["epoch"], data[name], "o-", markersize=2.8, label=str(seed))
    for axis, name in zip(axes, names):
        axis.set_title(name.replace("_normalized_uncertainty", "").replace("normalized_uncertainty_", ""))
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("NLL per token")
    axes[-1].legend(title="seed", frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output_directory / "multiseed_trajectories.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
