"""Fit modal forecasts to a reproduction trajectory and produce a compact audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from fit_modal_models import DEFAULT_MODE_GRID, fit_modal_batch  # noqa: E402


SERIES = (
    "solver_uncertainty",
    "verifier_uncertainty",
    "uncertainty_gap",
    "solver_normalized_uncertainty",
    "verifier_normalized_uncertainty",
    "normalized_uncertainty_gap",
)

TRIPLETS = {
    "total_nll": (
        "solver_uncertainty",
        "verifier_uncertainty",
        "uncertainty_gap",
    ),
    "per_token_nll": (
        "solver_normalized_uncertainty",
        "verifier_normalized_uncertainty",
        "normalized_uncertainty_gap",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_trajectory(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 6:
        raise ValueError("modal audit requires at least six checkpoints")
    output = {"epoch": np.asarray([float(row["epoch"]) for row in rows])}
    for name in SERIES:
        output[name] = np.asarray([float(row[name]) for row in rows])
    return output


def fit_one(values: np.ndarray, epochs: np.ndarray, train_count: int) -> dict[str, object]:
    train_values = values[:train_count]
    value_range = float(np.ptp(train_values))
    padding = (
        value_range if value_range > 0.0 else max(1.0, abs(float(train_values[0])))
    )
    bounds = (
        float(np.min(train_values) - padding),
        float(np.max(train_values) + padding),
    )
    heldout_values = values[train_count:]
    persistence_forecast = np.full(heldout_values.size, train_values[-1])
    output: dict[str, object] = {
        "persistence": {
            "last_training_value": float(train_values[-1]),
            "heldout_mse": float(
                np.mean((persistence_forecast - heldout_values) ** 2)
            ),
            "forecast": persistence_forecast.tolist(),
        }
    }
    for order in (1, 2):
        fit = fit_modal_batch(
            values[:train_count],
            epochs[:train_count],
            epochs[train_count:],
            order,
            mode_grid=DEFAULT_MODE_GRID,
            endpoint_bounds=bounds,
            direction="signed",
        )
        error = fit.forecast[0] - values[train_count:]
        train_tss = float(np.sum((train_values - np.mean(train_values)) ** 2))
        output[f"rank_{order}"] = {
            "endpoint": float(fit.endpoints[0]),
            "modes_per_epoch": fit.modes[0].tolist(),
            "amplitudes": fit.amplitudes[0].tolist(),
            "train_sse": float(fit.train_sse[0]),
            "train_r_squared": float(
                1.0 - fit.train_sse[0] / train_tss
                if train_tss > 0.0
                else float("nan")
            ),
            "heldout_mse": float(np.mean(error**2)),
            "forecast": fit.forecast[0].tolist(),
        }
    mse_1 = output["rank_1"]["heldout_mse"]
    mse_2 = output["rank_2"]["heldout_mse"]
    output["log_mse_ratio_rank1_over_rank2"] = float(
        np.log((mse_1 + 1e-15) / (mse_2 + 1e-15))
    )
    output["best_modal_order"] = 1 if mse_1 <= mse_2 else 2
    output["mse_factor_best_modal_over_persistence"] = float(
        min(mse_1, mse_2) / (output["persistence"]["heldout_mse"] + 1e-15)
    )
    return output


def fit_shared_rank_one(
    values: np.ndarray, epochs: np.ndarray, train_count: int
) -> dict[str, object]:
    """Fit one signed decay mode shared by three independently scaled series."""

    train_minima = np.min(values[:, :train_count], axis=1, keepdims=True)
    train_ranges = np.ptp(values[:, :train_count], axis=1, keepdims=True)
    if np.any(train_ranges <= 0.0):
        raise ValueError("shared-mode audit requires nonconstant training series")
    normalized = (values - train_minima) / train_ranges
    best: dict[str, object] | None = None
    for mode in DEFAULT_MODE_GRID:
        train_design = np.column_stack(
            (np.ones(train_count), mode ** epochs[:train_count])
        )
        coefficients = np.linalg.lstsq(
            train_design, normalized[:, :train_count].T, rcond=None
        )[0].T
        residual = normalized[:, :train_count].T - train_design @ coefficients.T
        train_sse = float(np.sum(residual**2))
        if best is None or train_sse < best["train_sse"]:
            forecast_design = np.column_stack(
                (
                    np.ones(epochs.size - train_count),
                    mode ** epochs[train_count:],
                )
            )
            forecast = (forecast_design @ coefficients.T).T
            best = {
                "mode_per_epoch": float(mode),
                "endpoints": coefficients[:, 0].tolist(),
                "amplitudes": coefficients[:, 1].tolist(),
                "train_sse": train_sse,
                "heldout_mse": float(
                    np.mean((forecast - normalized[:, train_count:]) ** 2)
                ),
                "forecast": forecast.tolist(),
                "scale": "each series affinely normalized by its training range",
            }
    if best is None:
        raise RuntimeError("no shared rank-one candidate")
    return best


def main() -> None:
    args = parse_args()
    data = read_trajectory(args.trajectory)
    checkpoint_count = data["epoch"].size
    train_count = max(4, min(checkpoint_count - 2, int(args.train_fraction * checkpoint_count)))
    output_path = args.output or args.trajectory.with_name("modal_audit.json")
    audit = {
        "trajectory": str(args.trajectory),
        "checkpoint_count": int(checkpoint_count),
        "train_count": train_count,
        "heldout_count": checkpoint_count - train_count,
        "series": {},
    }
    for name in SERIES:
        audit["series"][name] = fit_one(data[name], data["epoch"], train_count)

    audit["shared_mode_audit"] = {}
    for label, names in TRIPLETS.items():
        values = np.vstack([data[name] for name in names])
        shared = fit_shared_rank_one(values, data["epoch"], train_count)
        train_ranges = np.ptp(values[:, :train_count], axis=1)
        separate_errors = []
        for row, name in enumerate(names):
            forecast = np.asarray(audit["series"][name]["rank_1"]["forecast"])
            separate_errors.append(
                (forecast - values[row, train_count:]) / train_ranges[row]
            )
        separate_mse = float(np.mean(np.concatenate(separate_errors) ** 2))
        audit["shared_mode_audit"][label] = {
            "series": list(names),
            "shared_rank1": shared,
            "separate_rank1_heldout_mse": separate_mse,
            "heldout_mse_factor_shared_over_separate": float(
                shared["heldout_mse"] / (separate_mse + 1e-15)
            ),
            "qualification": (
                "descriptive only: the gap is algebraically determined by solver "
                "and verifier uncertainty, and this run contains one seed"
            ),
        }
    output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.1), sharex=True)
    for axis, name in zip(axes.ravel(), SERIES):
        axis.plot(data["epoch"], data[name], "o-", label="observed")
        for order, style in ((1, "--"), (2, ":")):
            forecast = audit["series"][name][f"rank_{order}"]["forecast"]
            axis.plot(data["epoch"][train_count:], forecast, style, label=f"rank {order}")
        axis.plot(
            data["epoch"][train_count:],
            audit["series"][name]["persistence"]["forecast"],
            "-.",
            color="0.35",
            label="persistence",
        )
        axis.axvline(data["epoch"][train_count], color="0.5", linewidth=0.8)
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("total NLL")
    axes[1, 0].set_ylabel("NLL per token")
    axes[-1, -1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path.with_name("modal_audit.png"), dpi=180)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
