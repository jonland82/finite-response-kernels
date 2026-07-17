"""Audit the dense-start reproduction without fitting a smooth response law.

The dense run repeats seed 20260719 with evaluation after every optimizer update
through epoch 4 and twice per epoch thereafter.  This script verifies that the
shared half-epoch checkpoints exactly reproduce the original trajectory, writes
all adjacent-checkpoint changes, and plots uncertainty, length, and accuracy on
the same training clock.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
TRAJECTORY_METRICS = (
    "solver_uncertainty",
    "verifier_uncertainty",
    "uncertainty_gap",
    "solver_normalized_uncertainty",
    "verifier_normalized_uncertainty",
    "normalized_uncertainty_gap",
    "solver_mean_token_count",
    "verifier_mean_token_count",
    "solver_accuracy",
    "verifier_accuracy",
    "candidate_accuracy",
    "verifier_judgment_accuracy",
)
CHANGE_METRICS = (
    "solver_uncertainty",
    "verifier_uncertainty",
    "solver_normalized_uncertainty",
    "verifier_normalized_uncertainty",
    "solver_mean_token_count",
    "verifier_mean_token_count",
    "solver_accuracy",
    "verifier_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIRECTORY.parent)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric_row(row: dict[str, str]) -> dict[str, float]:
    return {key: float(row[key]) for key in ("epoch", *TRAJECTORY_METRICS)}


def compare_common_checkpoints(
    sparse: list[dict[str, str]], dense: list[dict[str, str]]
) -> dict[str, object]:
    dense_by_epoch = {float(row["epoch"]): row for row in dense}
    maximum_difference = 0.0
    missing_epochs: list[float] = []
    for sparse_row in sparse:
        epoch = float(sparse_row["epoch"])
        dense_row = dense_by_epoch.get(epoch)
        if dense_row is None:
            missing_epochs.append(epoch)
            continue
        for metric in TRAJECTORY_METRICS:
            difference = abs(float(sparse_row[metric]) - float(dense_row[metric]))
            maximum_difference = max(maximum_difference, difference)
    return {
        "common_checkpoint_count": len(sparse) - len(missing_epochs),
        "missing_sparse_epochs": missing_epochs,
        "maximum_absolute_metric_difference": maximum_difference,
        "exact_metric_match": not missing_epochs and maximum_difference == 0.0,
    }


def step_changes(dense: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for before_raw, after_raw in zip(dense[:-1], dense[1:]):
        before = numeric_row(before_raw)
        after = numeric_row(after_raw)
        row: dict[str, object] = {
            "from_epoch": before["epoch"],
            "to_epoch": after["epoch"],
        }
        for metric in CHANGE_METRICS:
            row[f"{metric}_before"] = before[metric]
            row[f"{metric}_after"] = after[metric]
            row[f"{metric}_change"] = after[metric] - before[metric]
        rows.append(row)
    return rows


def transition_record(
    dense_by_epoch: dict[float, dict[str, str]], from_epoch: float, to_epoch: float
) -> dict[str, object]:
    before = numeric_row(dense_by_epoch[from_epoch])
    after = numeric_row(dense_by_epoch[to_epoch])
    return {
        "from_epoch": from_epoch,
        "to_epoch": to_epoch,
        "before": {metric: before[metric] for metric in CHANGE_METRICS},
        "after": {metric: after[metric] for metric in CHANGE_METRICS},
        "change": {
            metric: after[metric] - before[metric] for metric in CHANGE_METRICS
        },
    }


def make_figure(dense: list[dict[str, str]], output_path: Path) -> None:
    epochs = np.asarray([float(row["epoch"]) for row in dense])

    def values(column: str) -> np.ndarray:
        return np.asarray([float(row[column]) for row in dense])

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True)
    panels = (
        (
            axes[0, 0],
            "Total response NLL",
            "total NLL",
            "solver_uncertainty",
            "verifier_uncertainty",
        ),
        (
            axes[0, 1],
            "Length-normalized NLL",
            "NLL per token",
            "solver_normalized_uncertainty",
            "verifier_normalized_uncertainty",
        ),
        (
            axes[1, 0],
            "Mean response length",
            "tokens",
            "solver_mean_token_count",
            "verifier_mean_token_count",
        ),
        (
            axes[1, 1],
            "Held-out accuracy",
            "accuracy",
            "solver_accuracy",
            "verifier_accuracy",
        ),
    )
    for axis, title, ylabel, solver_column, verifier_column in panels:
        axis.axvspan(0.0, 4.0, color="0.92", zorder=0, label="dense window")
        axis.plot(
            epochs,
            values(solver_column),
            marker="o",
            markersize=3.2,
            linewidth=1.25,
            label="solver",
        )
        axis.plot(
            epochs,
            values(verifier_column),
            marker="s",
            markersize=3.0,
            linewidth=1.25,
            label="verifier",
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("training epoch")
    axes[1, 1].set_xlabel("training epoch")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    figure.suptitle(
        "Dense checkpoints resolve one-update behavioral transitions, not a smooth landing",
        fontsize=11,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_root = project_root / "reproduction" / "runs"
    sparse_path = run_root / "arithmetic_seed_20260719" / "trajectory.csv"
    dense_path = run_root / "arithmetic_dense_seed_20260719" / "trajectory.csv"
    sparse = read_rows(sparse_path)
    dense = read_rows(dense_path)
    dense_by_epoch = {float(row["epoch"]): row for row in dense}

    comparison = compare_common_checkpoints(sparse, dense)
    changes = step_changes(dense)
    results_directory = project_root / "results"
    write_rows(results_directory / "dense_start_step_changes.csv", changes)

    metadata = {
        "seed": int(dense[0]["seed"]),
        "sparse_checkpoint_count": len(sparse),
        "dense_checkpoint_count": len(dense),
        "dense_interval_epochs": 0.25,
        "dense_until_epoch": 4.0,
        "optimizer_updates_per_epoch": 4,
        "common_checkpoint_validation": comparison,
        "selected_transitions": [
            transition_record(dense_by_epoch, 1.25, 1.5),
            transition_record(dense_by_epoch, 3.0, 3.25),
            transition_record(dense_by_epoch, 9.5, 10.0),
        ],
        "qualification": (
            "The held-out panel has eight prompts. Exact agreement at common "
            "checkpoints validates measurement isolation for this seed, but the "
            "observed jumps need not generalize to larger models or tasks."
        ),
    }
    (results_directory / "dense_start_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(dense, project_root / "figures" / "fig7_dense_start_audit.png")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
