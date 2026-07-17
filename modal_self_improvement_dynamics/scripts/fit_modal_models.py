"""Constrained finite-modal fits with a jointly estimated asymptote.

The model is

    y_n = y_infinity + sum_j amplitude_j * mode_j**n.

Modes are selected from a fixed grid and the endpoint/amplitudes are solved by
least squares for every candidate.  This variable-projection formulation is
deterministic, supports batched Monte Carlo studies, and avoids local nonlinear
optimizer failures.  The default monotone-decreasing constraint requires
nonnegative amplitudes.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np


Direction = Literal["decreasing", "increasing", "signed"]

DEFAULT_MODE_GRID = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.985,
    0.99,
    0.995,
)


@dataclass(frozen=True)
class ModalBatchFit:
    order: int
    modes: np.ndarray
    endpoints: np.ndarray
    amplitudes: np.ndarray
    train_sse: np.ndarray
    forecast: np.ndarray
    valid_candidate_counts: np.ndarray


def candidate_mode_tuples(
    order: int,
    mode_grid: tuple[float, ...] = DEFAULT_MODE_GRID,
    minimum_separation: float = 0.01,
) -> np.ndarray:
    grid = tuple(sorted(set(float(value) for value in mode_grid)))
    if any(not 0.0 < value < 1.0 for value in grid):
        raise ValueError("all candidate modes must lie in (0, 1)")
    if order == 1:
        candidates = [(value,) for value in grid]
    elif order == 2:
        candidates = [
            pair
            for pair in combinations(grid, 2)
            if pair[1] - pair[0] >= minimum_separation
        ]
    else:
        raise ValueError("only rank-one and rank-two fits are implemented")
    if not candidates:
        raise ValueError("mode grid produced no valid candidates")
    return np.asarray(candidates, dtype=float)


def design_matrices(
    checkpoints: np.ndarray,
    candidate_modes: np.ndarray,
) -> np.ndarray:
    checkpoints = np.asarray(checkpoints, dtype=float)
    candidate_modes = np.asarray(candidate_modes, dtype=float)
    modal_columns = candidate_modes[:, None, :] ** checkpoints[None, :, None]
    intercept = np.ones(
        (candidate_modes.shape[0], checkpoints.size, 1), dtype=float
    )
    return np.concatenate((intercept, modal_columns), axis=2)


def coefficient_validity(
    coefficients: np.ndarray,
    endpoint_bounds: tuple[float, float],
    direction: Direction,
    tolerance: float = 1e-10,
) -> np.ndarray:
    endpoint = coefficients[:, :, 0]
    amplitudes = coefficients[:, :, 1:]
    valid = (endpoint >= endpoint_bounds[0]) & (endpoint <= endpoint_bounds[1])
    if direction == "decreasing":
        valid &= np.all(amplitudes >= -tolerance, axis=2)
    elif direction == "increasing":
        valid &= np.all(amplitudes <= tolerance, axis=2)
    elif direction != "signed":
        raise ValueError(f"unknown direction: {direction}")
    return valid


def fit_modal_batch(
    observed_train: np.ndarray,
    train_checkpoints: np.ndarray,
    forecast_checkpoints: np.ndarray,
    order: int,
    *,
    mode_grid: tuple[float, ...] = DEFAULT_MODE_GRID,
    minimum_separation: float = 0.01,
    endpoint_bounds: tuple[float, float] = (-0.25, 0.25),
    direction: Direction = "decreasing",
) -> ModalBatchFit:
    observed_train = np.asarray(observed_train, dtype=float)
    if observed_train.ndim == 1:
        observed_train = observed_train[None, :]
    if observed_train.ndim != 2:
        raise ValueError("observed_train must be one- or two-dimensional")
    train_checkpoints = np.asarray(train_checkpoints, dtype=float)
    forecast_checkpoints = np.asarray(forecast_checkpoints, dtype=float)
    if observed_train.shape[1] != train_checkpoints.size:
        raise ValueError("training observations and checkpoints do not align")
    if endpoint_bounds[0] >= endpoint_bounds[1]:
        raise ValueError("endpoint bounds must be ordered")

    candidates = candidate_mode_tuples(order, mode_grid, minimum_separation)
    train_design = design_matrices(train_checkpoints, candidates)
    forecast_design = design_matrices(forecast_checkpoints, candidates)
    pseudoinverse = np.linalg.pinv(train_design)

    coefficients = np.einsum(
        "cpn,rn->rcp", pseudoinverse, observed_train, optimize=True
    )
    xty = np.einsum(
        "cnp,rn->rcp", train_design, observed_train, optimize=True
    )
    yty = np.sum(observed_train**2, axis=1)
    sse = yty[:, None] - np.sum(coefficients * xty, axis=2)
    sse = np.maximum(sse, 0.0)

    valid = coefficient_validity(
        coefficients, endpoint_bounds=endpoint_bounds, direction=direction
    )
    valid_counts = np.sum(valid, axis=1)
    if np.any(valid_counts == 0):
        raise RuntimeError(
            "at least one series has no valid modal candidate; widen bounds or grid"
        )
    constrained_sse = np.where(valid, sse, np.inf)
    best_indices = np.argmin(constrained_sse, axis=1)
    row_indices = np.arange(observed_train.shape[0])
    best_coefficients = coefficients[row_indices, best_indices]
    selected_design = forecast_design[best_indices]
    forecast = np.einsum(
        "rnp,rp->rn", selected_design, best_coefficients, optimize=True
    )

    return ModalBatchFit(
        order=order,
        modes=candidates[best_indices],
        endpoints=best_coefficients[:, 0],
        amplitudes=best_coefficients[:, 1:],
        train_sse=constrained_sse[row_indices, best_indices],
        forecast=forecast,
        valid_candidate_counts=valid_counts,
    )


def read_series(
    input_path: Path,
    checkpoint_column: str,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoints: list[float] = []
    values: list[float] = []
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if checkpoint_column not in (reader.fieldnames or []):
            raise ValueError(f"missing checkpoint column: {checkpoint_column}")
        if value_column not in (reader.fieldnames or []):
            raise ValueError(f"missing value column: {value_column}")
        for row in reader:
            checkpoints.append(float(row[checkpoint_column]))
            values.append(float(row[value_column]))
    return np.asarray(checkpoints), np.asarray(values)


def parse_mode_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("mode grid cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint-column", default="checkpoint")
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--train-count", type=int)
    parser.add_argument(
        "--mode-grid",
        type=parse_mode_grid,
        default=DEFAULT_MODE_GRID,
    )
    parser.add_argument(
        "--direction",
        choices=("decreasing", "increasing", "signed"),
        default="decreasing",
    )
    parser.add_argument("--endpoint-lower", type=float)
    parser.add_argument("--endpoint-upper", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints, values = read_series(
        args.input, args.checkpoint_column, args.value_column
    )
    train_count = args.train_count or max(4, int(np.floor(0.75 * values.size)))
    if not 4 <= train_count < values.size:
        raise ValueError("train-count must leave at least one held-out observation")

    observed_range = float(np.ptp(values))
    padding = observed_range if observed_range > 0.0 else 1.0
    endpoint_bounds = (
        args.endpoint_lower
        if args.endpoint_lower is not None
        else float(np.min(values) - padding),
        args.endpoint_upper
        if args.endpoint_upper is not None
        else float(np.max(values) + padding),
    )
    train_values = values[:train_count]
    heldout_values = values[train_count:]
    train_checkpoints = checkpoints[:train_count]
    heldout_checkpoints = checkpoints[train_count:]

    output: dict[str, object] = {
        "input": str(args.input),
        "value_column": args.value_column,
        "train_count": train_count,
        "heldout_count": int(heldout_values.size),
        "endpoint_bounds": endpoint_bounds,
        "models": {},
    }
    for order in (1, 2):
        fit = fit_modal_batch(
            train_values,
            train_checkpoints,
            heldout_checkpoints,
            order,
            mode_grid=args.mode_grid,
            endpoint_bounds=endpoint_bounds,
            direction=args.direction,
        )
        forecast_error = fit.forecast[0] - heldout_values
        output["models"][f"rank_{order}"] = {
            "endpoint": float(fit.endpoints[0]),
            "modes": fit.modes[0].tolist(),
            "amplitudes": fit.amplitudes[0].tolist(),
            "train_sse": float(fit.train_sse[0]),
            "heldout_mse": float(np.mean(forecast_error**2)),
            "forecast": fit.forecast[0].tolist(),
        }

    serialized = json.dumps(output, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
