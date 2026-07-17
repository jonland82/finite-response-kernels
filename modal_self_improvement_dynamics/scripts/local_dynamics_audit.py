"""Test global shared modes against a time-local two-phase response.

The audit fits solver, verifier, and gap trajectories jointly.  It compares
one global shared mode, two global shared modes, and a continuous two-phase
exponential with a shared change point.  Separate rank-one, power-law, local
linear, and persistence forecasts provide less constrained baselines.

All model comparisons use rolling training prefixes and untouched held-out
tails.  Matched synthetic controls quantify confusion among true rank-one,
rank-two, and two-phase dynamics at the available 20--21 checkpoint horizon.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from fit_modal_models import (  # noqa: E402
    DEFAULT_MODE_GRID,
    candidate_mode_tuples,
    design_matrices,
)
from forecast_baseline_audit import (  # noqa: E402
    Forecast,
    endpoint_bounds,
    local_linear_forecast,
    modal_forecast,
    power_law_forecast,
    read_csv,
    split_counts,
    write_csv,
)


MODEL_NAMES = (
    "persistence",
    "separate_local",
    "separate_power",
    "separate_rank_1",
    "shared_rank_1",
    "shared_rank_2",
    "shared_two_phase",
)
STRUCTURAL_MODELS = ("shared_rank_1", "shared_rank_2", "shared_two_phase")
SERIES_NAMES = ("solver", "verifier", "gap")


@dataclass(frozen=True)
class GroupTrajectory:
    source: str
    name: str
    checkpoints: np.ndarray
    values: np.ndarray
    truth: str = "unknown"


@dataclass(frozen=True)
class GroupForecast:
    values: np.ndarray
    normalized_train_sse: float
    parameters: dict[str, object]


_MODAL_CACHE: dict[tuple[object, ...], tuple[np.ndarray, ...]] = {}
_PHASE_CACHE: dict[tuple[object, ...], tuple[np.ndarray, ...]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIRECTORY.parent)
    parser.add_argument("--synthetic-replicates", type=int, default=100)
    parser.add_argument("--synthetic-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--local-window", type=int, default=7)
    return parser.parse_args()


def group_scales(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.ptp(values, axis=1), 1e-8)


def load_published_groups(project_root: Path) -> list[GroupTrajectory]:
    rows = read_csv(project_root / "results" / "sun_phi4_figure4_digitized.csv")
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        groups[row["panel"]][row["series"]].append(row)
    output: list[GroupTrajectory] = []
    for panel, series_groups in sorted(groups.items()):
        ordered_values = []
        checkpoints: np.ndarray | None = None
        for series in SERIES_NAMES:
            series_rows = sorted(
                series_groups[series], key=lambda row: float(row["checkpoint"])
            )
            current_checkpoints = np.asarray(
                [float(row["epoch"]) for row in series_rows]
            )
            if checkpoints is None:
                checkpoints = current_checkpoints
            elif not np.allclose(checkpoints, current_checkpoints):
                raise ValueError(f"published checkpoints do not align in {panel}")
            ordered_values.append(
                [float(row["affine_uncertainty"]) for row in series_rows]
            )
        if checkpoints is None:
            continue
        output.append(
            GroupTrajectory(
                source="published",
                name=panel,
                checkpoints=checkpoints,
                values=np.asarray(ordered_values, dtype=float),
            )
        )
    return output


def load_reproduction_groups(project_root: Path) -> list[GroupTrajectory]:
    output: list[GroupTrajectory] = []
    run_root = project_root / "reproduction" / "runs"
    columns = (
        "solver_normalized_uncertainty",
        "verifier_normalized_uncertainty",
        "normalized_uncertainty_gap",
    )
    for path in sorted(run_root.glob("arithmetic_seed_*/trajectory.csv")):
        rows = read_csv(path)
        output.append(
            GroupTrajectory(
                source="reproduction",
                name=rows[0]["seed"],
                checkpoints=np.asarray([float(row["epoch"]) for row in rows]),
                values=np.asarray(
                    [[float(row[column]) for row in rows] for column in columns]
                ),
            )
        )
    if not output:
        raise FileNotFoundError("no arithmetic_seed reproduction trajectories found")
    return output


def phase_basis(
    checkpoints: np.ndarray,
    early_mode: float,
    late_mode: float,
    change_index: int,
) -> np.ndarray:
    relative = np.asarray(checkpoints, dtype=float)
    early_time = np.minimum(relative, float(change_index))
    late_time = np.maximum(relative - float(change_index), 0.0)
    return early_mode**early_time * late_mode**late_time


def generate_synthetic_groups(
    replicates: int, noise_sd: float, seed: int
) -> list[GroupTrajectory]:
    if replicates < 1:
        raise ValueError("synthetic replicates must be positive")
    if noise_sd < 0.0:
        raise ValueError("synthetic noise must be nonnegative")
    rng = np.random.default_rng(seed)
    checkpoints = np.arange(21, dtype=float)
    mode_grid = np.asarray(DEFAULT_MODE_GRID)
    output: list[GroupTrajectory] = []
    for truth in STRUCTURAL_MODELS:
        for replicate in range(replicates):
            endpoints = rng.uniform(-0.15, 0.15, size=2)
            amplitudes = np.asarray(
                [rng.uniform(0.85, 1.15), rng.uniform(0.35, 0.65)]
            )
            parameters: dict[str, object]
            if truth == "shared_rank_1":
                mode = float(rng.choice(mode_grid[5:17]))
                solver_basis = mode**checkpoints
                verifier_basis = solver_basis
                parameters = {"mode": mode}
            elif truth == "shared_rank_2":
                fast = float(rng.choice(mode_grid[2:9]))
                slow_choices = mode_grid[(mode_grid >= max(0.80, fast + 0.15))]
                slow = float(rng.choice(slow_choices))
                solver_weight = rng.uniform(0.45, 0.80)
                verifier_weight = rng.uniform(0.15, 0.55)
                solver_basis = (
                    solver_weight * fast**checkpoints
                    + (1.0 - solver_weight) * slow**checkpoints
                )
                verifier_basis = (
                    verifier_weight * fast**checkpoints
                    + (1.0 - verifier_weight) * slow**checkpoints
                )
                parameters = {
                    "fast_mode": fast,
                    "slow_mode": slow,
                    "solver_weight": solver_weight,
                    "verifier_weight": verifier_weight,
                }
            else:
                early = float(rng.choice(mode_grid[3:11]))
                late_choices = mode_grid[np.abs(mode_grid - early) >= 0.15]
                late = float(rng.choice(late_choices))
                change_index = int(rng.integers(4, 11))
                solver_basis = phase_basis(checkpoints, early, late, change_index)
                verifier_basis = solver_basis
                parameters = {
                    "early_mode": early,
                    "late_mode": late,
                    "change_index": change_index,
                }
            solver = endpoints[0] + amplitudes[0] * solver_basis
            verifier = endpoints[1] + amplitudes[1] * verifier_basis
            noisy_solver = solver + rng.normal(0.0, noise_sd, checkpoints.size)
            noisy_verifier = verifier + rng.normal(0.0, noise_sd, checkpoints.size)
            values = np.vstack(
                (noisy_solver, noisy_verifier, noisy_solver - noisy_verifier)
            )
            output.append(
                GroupTrajectory(
                    source="synthetic_local",
                    name=f"{truth}_{replicate:03d}",
                    checkpoints=checkpoints,
                    values=values,
                    truth=truth,
                )
            )
            _ = parameters
    return output


def relative_checkpoints(checkpoints: np.ndarray) -> tuple[np.ndarray, float]:
    spacing = float(np.median(np.diff(checkpoints)))
    if spacing <= 0.0:
        raise ValueError("checkpoints must be strictly increasing")
    return (checkpoints - checkpoints[0]) / spacing, spacing


def modal_cache(
    train_x: np.ndarray, forecast_x: np.ndarray, order: int
) -> tuple[np.ndarray, ...]:
    relative_train, spacing = relative_checkpoints(train_x)
    relative_forecast = (forecast_x - train_x[0]) / spacing
    key = (
        "modal",
        order,
        tuple(np.round(relative_train, 10)),
        tuple(np.round(relative_forecast, 10)),
    )
    cached = _MODAL_CACHE.get(key)
    if cached is not None:
        return cached
    candidates = candidate_mode_tuples(order)
    train_design = design_matrices(relative_train, candidates)
    forecast_design = design_matrices(relative_forecast, candidates)
    pseudoinverse = np.linalg.pinv(train_design)
    cached = (candidates, train_design, forecast_design, pseudoinverse)
    _MODAL_CACHE[key] = cached
    return cached


def fit_shared_modal(
    train_x: np.ndarray,
    train_y: np.ndarray,
    forecast_x: np.ndarray,
    order: int,
) -> GroupForecast:
    candidates, train_design, forecast_design, pseudoinverse = modal_cache(
        train_x, forecast_x, order
    )
    coefficients = np.einsum(
        "cpn,sn->csp", pseudoinverse, train_y, optimize=True
    )
    fitted = np.einsum(
        "cnp,csp->csn", train_design, coefficients, optimize=True
    )
    sse = np.sum((fitted - train_y[None, :, :]) ** 2, axis=2)
    scales = group_scales(train_y)
    objective = np.sum(sse / scales[None, :] ** 2, axis=1)
    valid = np.ones(candidates.shape[0], dtype=bool)
    for series_index in range(train_y.shape[0]):
        lower, upper = endpoint_bounds(train_y[series_index])
        endpoints = coefficients[:, series_index, 0]
        valid &= (endpoints >= lower) & (endpoints <= upper)
    objective = np.where(valid, objective, np.inf)
    if not np.any(np.isfinite(objective)):
        raise RuntimeError("no valid shared modal candidate")
    best = int(np.argmin(objective))
    forecast = np.einsum(
        "np,sp->sn", forecast_design[best], coefficients[best], optimize=True
    )
    return GroupForecast(
        values=forecast,
        normalized_train_sse=float(objective[best]),
        parameters={
            "modes": candidates[best].tolist(),
            "endpoints": coefficients[best, :, 0].tolist(),
            "amplitudes": coefficients[best, :, 1:].tolist(),
        },
    )


def phase_cache(
    train_x: np.ndarray, forecast_x: np.ndarray
) -> tuple[np.ndarray, ...]:
    relative_train, spacing = relative_checkpoints(train_x)
    relative_forecast = (forecast_x - train_x[0]) / spacing
    key = (
        "phase",
        tuple(np.round(relative_train, 10)),
        tuple(np.round(relative_forecast, 10)),
    )
    cached = _PHASE_CACHE.get(key)
    if cached is not None:
        return cached
    maximum_index = int(np.floor(relative_train[-1]))
    change_indices = range(2, max(3, maximum_index - 1))
    metadata = []
    train_basis = []
    forecast_basis = []
    for change_index in change_indices:
        for early_mode in DEFAULT_MODE_GRID:
            for late_mode in DEFAULT_MODE_GRID:
                if abs(early_mode - late_mode) < 0.08:
                    continue
                metadata.append((early_mode, late_mode, change_index))
                train_basis.append(
                    phase_basis(
                        relative_train, early_mode, late_mode, change_index
                    )
                )
                forecast_basis.append(
                    phase_basis(
                        relative_forecast, early_mode, late_mode, change_index
                    )
                )
    cached = (
        np.asarray(metadata, dtype=float),
        np.asarray(train_basis, dtype=float),
        np.asarray(forecast_basis, dtype=float),
    )
    _PHASE_CACHE[key] = cached
    return cached


def fit_shared_two_phase(
    train_x: np.ndarray, train_y: np.ndarray, forecast_x: np.ndarray
) -> GroupForecast:
    metadata, train_basis, forecast_basis = phase_cache(train_x, forecast_x)
    basis_mean = np.mean(train_basis, axis=1)
    centered_basis = train_basis - basis_mean[:, None]
    denominator = np.sum(centered_basis**2, axis=1)
    centered_y = train_y - np.mean(train_y, axis=1)[:, None]
    amplitudes = np.divide(
        centered_basis @ centered_y.T,
        denominator[:, None],
        out=np.zeros((train_basis.shape[0], train_y.shape[0])),
        where=denominator[:, None] > 1e-20,
    )
    endpoints = np.mean(train_y, axis=1)[None, :] - amplitudes * basis_mean[:, None]
    fitted = endpoints[:, :, None] + amplitudes[:, :, None] * train_basis[:, None, :]
    sse = np.sum((fitted - train_y[None, :, :]) ** 2, axis=2)
    scales = group_scales(train_y)
    objective = np.sum(sse / scales[None, :] ** 2, axis=1)
    valid = np.ones(train_basis.shape[0], dtype=bool)
    for series_index in range(train_y.shape[0]):
        lower, upper = endpoint_bounds(train_y[series_index])
        valid &= (endpoints[:, series_index] >= lower) & (
            endpoints[:, series_index] <= upper
        )
    objective = np.where(valid, objective, np.inf)
    if not np.any(np.isfinite(objective)):
        raise RuntimeError("no valid two-phase candidate")
    best = int(np.argmin(objective))
    forecast = endpoints[best, :, None] + amplitudes[best, :, None] * forecast_basis[
        best, None, :
    ]
    early_mode, late_mode, change_index = metadata[best]
    return GroupForecast(
        values=forecast,
        normalized_train_sse=float(objective[best]),
        parameters={
            "early_mode": float(early_mode),
            "late_mode": float(late_mode),
            "change_index": int(change_index),
            "change_checkpoint": float(
                train_x[0]
                + change_index * float(np.median(np.diff(train_x)))
            ),
            "endpoints": endpoints[best].tolist(),
            "amplitudes": amplitudes[best].tolist(),
        },
    )


def fit_separate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    forecast_x: np.ndarray,
    model: str,
) -> GroupForecast:
    forecasts: list[Forecast] = []
    for values in train_y:
        if model == "rank_1":
            forecast = modal_forecast(train_x, values, forecast_x, 1)
        elif model == "power":
            forecast = power_law_forecast(train_x, values, forecast_x)
        elif model == "local":
            forecast = local_linear_forecast(train_x, values, forecast_x)
        else:
            raise ValueError(f"unknown separate model: {model}")
        forecasts.append(forecast)
    scales = group_scales(train_y)
    normalized_sse = sum(
        forecast.train_sse / scales[index] ** 2
        for index, forecast in enumerate(forecasts)
    )
    return GroupForecast(
        values=np.vstack([forecast.values for forecast in forecasts]),
        normalized_train_sse=float(normalized_sse),
        parameters={
            SERIES_NAMES[index]: forecast.parameters
            for index, forecast in enumerate(forecasts)
        },
    )


def persistence_group(train_y: np.ndarray, forecast_count: int) -> GroupForecast:
    forecast = np.repeat(train_y[:, -1:], forecast_count, axis=1)
    scales = group_scales(train_y)
    sse = np.sum((train_y - train_y[:, -1:]) ** 2, axis=1)
    return GroupForecast(
        values=forecast,
        normalized_train_sse=float(np.sum(sse / scales**2)),
        parameters={"last_values": train_y[:, -1].tolist()},
    )


def fit_all_models(
    train_x: np.ndarray, train_y: np.ndarray, forecast_x: np.ndarray
) -> dict[str, GroupForecast]:
    return {
        "persistence": persistence_group(train_y, forecast_x.size),
        "separate_local": fit_separate(
            train_x, train_y, forecast_x, "local"
        ),
        "separate_power": fit_separate(
            train_x, train_y, forecast_x, "power"
        ),
        "separate_rank_1": fit_separate(
            train_x, train_y, forecast_x, "rank_1"
        ),
        "shared_rank_1": fit_shared_modal(train_x, train_y, forecast_x, 1),
        "shared_rank_2": fit_shared_modal(train_x, train_y, forecast_x, 2),
        "shared_two_phase": fit_shared_two_phase(train_x, train_y, forecast_x),
    }


def choose_winner(errors: dict[str, float], candidates: tuple[str, ...]) -> str:
    minimum = min(errors[model] for model in candidates)
    tolerance = max(1e-14, 1e-8 * max(errors[model] for model in candidates))
    for model in candidates:
        if errors[model] <= minimum + tolerance:
            return model
    raise RuntimeError("no forecast winner")


def audit_group(group: GroupTrajectory) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_count in split_counts(group.checkpoints.size):
        train_x = group.checkpoints[:train_count]
        forecast_x = group.checkpoints[train_count:]
        train_y = group.values[:, :train_count]
        heldout_y = group.values[:, train_count:]
        scales = group_scales(train_y)
        forecasts = fit_all_models(train_x, train_y, forecast_x)
        errors = {
            model: float(
                np.mean(
                    ((forecast.values - heldout_y) / scales[:, None]) ** 2
                )
            )
            for model, forecast in forecasts.items()
        }
        winner = choose_winner(errors, MODEL_NAMES)
        structural_winner = choose_winner(errors, STRUCTURAL_MODELS)
        persistence_error = errors["persistence"]
        for model, forecast in forecasts.items():
            rows.append(
                {
                    "source": group.source,
                    "group": group.name,
                    "truth": group.truth,
                    "checkpoint_count": group.checkpoints.size,
                    "train_count": train_count,
                    "heldout_count": heldout_y.shape[1],
                    "model": model,
                    "winner": model == winner,
                    "structural_winner": model == structural_winner,
                    "normalized_train_sse": forecast.normalized_train_sse,
                    "heldout_normalized_mse": errors[model],
                    "mse_ratio_to_persistence": errors[model]
                    / (persistence_error + 1e-15),
                    "parameters": json.dumps(forecast.parameters, sort_keys=True),
                }
            )
    return rows


def local_difference_rates(
    groups: list[GroupTrajectory], window: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in groups:
        if not 5 <= window <= group.checkpoints.size:
            raise ValueError("local window must be between 5 and trajectory length")
        for start in range(group.checkpoints.size - window + 1):
            stop = start + window
            values = group.values[:, start:stop]
            differences = np.diff(values, axis=1)
            predictors = differences[:, :-1]
            responses = differences[:, 1:]
            denominator = float(np.sum(predictors**2))
            shared_mode = (
                float(np.sum(predictors * responses) / denominator)
                if denominator > 1e-20
                else float("nan")
            )
            residual = responses - shared_mode * predictors
            response_energy = max(float(np.sum(responses**2)), 1e-20)
            series_modes = []
            for series_index in range(values.shape[0]):
                series_denominator = float(np.sum(predictors[series_index] ** 2))
                series_modes.append(
                    float(
                        np.sum(
                            predictors[series_index] * responses[series_index]
                        )
                        / series_denominator
                    )
                    if series_denominator > 1e-20
                    else float("nan")
                )
            rows.append(
                {
                    "source": group.source,
                    "group": group.name,
                    "window_start": float(group.checkpoints[start]),
                    "window_end": float(group.checkpoints[stop - 1]),
                    "window_midpoint": float(
                        0.5 * (group.checkpoints[start] + group.checkpoints[stop - 1])
                    ),
                    "relative_midpoint": float(
                        (0.5 * (start + stop - 1)) / (group.checkpoints.size - 1)
                    ),
                    "shared_difference_mode": shared_mode,
                    "solver_difference_mode": series_modes[0],
                    "verifier_difference_mode": series_modes[1],
                    "gap_difference_mode": series_modes[2],
                    "shared_relative_residual_energy": float(
                        np.sum(residual**2) / response_energy
                    ),
                }
            )
    return rows


def case_errors(
    rows: list[dict[str, object]], source: str
) -> dict[tuple[str, int], dict[str, float]]:
    output: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["source"] == source:
            output[(str(row["group"]), int(row["train_count"]))][
                str(row["model"])
            ] = float(row["heldout_normalized_mse"])
    return output


def summarize_models(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["model"]))].append(row)
    output = []
    for (source, model), group in sorted(grouped.items()):
        ratios = np.asarray(
            [float(row["mse_ratio_to_persistence"]) for row in group]
        )
        output.append(
            {
                "source": source,
                "model": model,
                "cases": len(group),
                "wins": sum(bool(row["winner"]) for row in group),
                "structural_wins": sum(
                    bool(row["structural_winner"]) for row in group
                ),
                "beats_persistence_fraction": float(np.mean(ratios < 1.0)),
                "median_mse_ratio_to_persistence": float(np.median(ratios)),
            }
        )
    return output


def summarize_local_rates(
    rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["group"]))].append(row)
    output = []
    for (source, name), group in sorted(grouped.items()):
        group.sort(key=lambda row: float(row["window_midpoint"]))
        modes = np.asarray(
            [float(row["shared_difference_mode"]) for row in group], dtype=float
        )
        finite = modes[np.isfinite(modes)]
        third = max(1, len(group) // 3)
        output.append(
            {
                "source": source,
                "group": name,
                "window_count": len(group),
                "median_shared_mode": float(np.median(finite)),
                "shared_mode_iqr": float(
                    np.quantile(finite, 0.75) - np.quantile(finite, 0.25)
                ),
                "early_median_shared_mode": float(np.nanmedian(modes[:third])),
                "late_median_shared_mode": float(np.nanmedian(modes[-third:])),
                "early_late_mode_shift": float(
                    np.nanmedian(modes[-third:]) - np.nanmedian(modes[:third])
                ),
                "stable_decay_window_fraction": float(
                    np.mean((finite > 0.0) & (finite < 1.0))
                ),
            }
        )
    return output


def aggregate_metadata(
    groups: list[GroupTrajectory], rows: list[dict[str, object]]
) -> dict[str, object]:
    sources = sorted({group.source for group in groups})
    comparisons: dict[str, object] = {}
    for source in sources:
        cases = case_errors(rows, source)
        source_rows = [row for row in rows if row["source"] == source]
        comparisons[source] = {
            "case_count": len(cases),
            "winner_counts": {
                model: sum(
                    row["model"] == model and bool(row["winner"])
                    for row in source_rows
                )
                for model in MODEL_NAMES
            },
            "structural_winner_counts": {
                model: sum(
                    row["model"] == model and bool(row["structural_winner"])
                    for row in source_rows
                )
                for model in STRUCTURAL_MODELS
            },
            "two_phase_beats_shared_rank_1": sum(
                values["shared_two_phase"] < values["shared_rank_1"]
                for values in cases.values()
            ),
            "two_phase_beats_shared_rank_2": sum(
                values["shared_two_phase"] < values["shared_rank_2"]
                for values in cases.values()
            ),
            "best_structural_beats_persistence": sum(
                min(values[model] for model in STRUCTURAL_MODELS)
                < values["persistence"]
                for values in cases.values()
            ),
        }

    synthetic_cases = case_errors(rows, "synthetic_local")
    truth_by_group = {
        group.name: group.truth
        for group in groups
        if group.source == "synthetic_local"
    }
    confusion = {
        truth: {model: 0 for model in STRUCTURAL_MODELS}
        for truth in STRUCTURAL_MODELS
    }
    confusion_by_train_count: dict[str, dict[str, dict[str, int]]] = {}
    for (name, _train_count), values in synthetic_cases.items():
        selected = choose_winner(values, STRUCTURAL_MODELS)
        truth = truth_by_group[name]
        confusion[truth][selected] += 1
        split_confusion = confusion_by_train_count.setdefault(
            str(_train_count),
            {
                current_truth: {model: 0 for model in STRUCTURAL_MODELS}
                for current_truth in STRUCTURAL_MODELS
            },
        )
        split_confusion[truth][selected] += 1
    row_normalized_confusion = {}
    for truth, counts in confusion.items():
        total = sum(counts.values())
        row_normalized_confusion[truth] = {
            model: counts[model] / total for model in STRUCTURAL_MODELS
        }
    accuracy_by_train_count = {}
    for train_count, split_confusion in sorted(
        confusion_by_train_count.items(), key=lambda item: int(item[0])
    ):
        accuracy_by_train_count[train_count] = {}
        for truth, counts in split_confusion.items():
            total = sum(counts.values())
            accuracy_by_train_count[train_count][truth] = counts[truth] / total

    two_phase_parameter_stability: dict[str, object] = {}
    real_two_phase_rows = [
        row
        for row in rows
        if row["source"] in ("published", "reproduction")
        and row["model"] == "shared_two_phase"
    ]
    parameter_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in real_two_phase_rows:
        parameter_groups[(str(row["source"]), str(row["group"]))].append(row)
    for (source, name), group_rows in sorted(parameter_groups.items()):
        group_rows.sort(key=lambda row: int(row["train_count"]))
        parameters = [json.loads(str(row["parameters"])) for row in group_rows]
        key = f"{source}::{name}"
        two_phase_parameter_stability[key] = {
            "train_counts": [int(row["train_count"]) for row in group_rows],
            "change_checkpoints": [
                float(parameter["change_checkpoint"]) for parameter in parameters
            ],
            "early_modes": [float(parameter["early_mode"]) for parameter in parameters],
            "late_modes": [float(parameter["late_mode"]) for parameter in parameters],
            "structural_win_count": sum(
                bool(row["structural_winner"]) for row in group_rows
            ),
            "overall_win_count": sum(bool(row["winner"]) for row in group_rows),
        }
    return {
        "trajectory_group_counts": {
            source: sum(group.source == source for group in groups)
            for source in sources
        },
        "models": MODEL_NAMES,
        "structural_models": STRUCTURAL_MODELS,
        "comparisons": comparisons,
        "synthetic_structural_confusion_counts": confusion,
        "synthetic_structural_confusion_fractions": row_normalized_confusion,
        "synthetic_structural_accuracy_by_train_count": accuracy_by_train_count,
        "two_phase_parameter_stability": two_phase_parameter_stability,
        "qualification": (
            "All forecasts use training prefixes only. Errors are normalized "
            "by each series' training range and averaged across solver, verifier, "
            "and the derived gap. The published values remain figure-derived."
        ),
    }


def make_figure(
    rows: list[dict[str, object]],
    local_rows: list[dict[str, object]],
    metadata: dict[str, object],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    real_sources = ("published", "reproduction")
    x = np.arange(len(STRUCTURAL_MODELS))
    width = 0.36
    for source_index, source in enumerate(real_sources):
        counts = metadata["comparisons"][source]["structural_winner_counts"]
        axes[0].bar(
            x + (source_index - 0.5) * width,
            [counts[model] for model in STRUCTURAL_MODELS],
            width,
            label=source,
        )
    axes[0].set_xticks(x, ("global R1", "global R2", "two phase"), rotation=20)
    axes[0].set_ylabel("rolling structural wins")
    axes[0].legend(frameon=False)
    axes[0].set_title("Held-out structural comparison")

    colors = {"published": "tab:blue", "reproduction": "tab:orange"}
    for source in real_sources:
        source_rows = [row for row in local_rows if row["source"] == source]
        axes[1].scatter(
            [float(row["relative_midpoint"]) for row in source_rows],
            [float(row["shared_difference_mode"]) for row in source_rows],
            s=18,
            alpha=0.55,
            color=colors[source],
            label=source,
        )
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].axhline(1.0, color="0.5", linewidth=0.8, linestyle="--")
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_xlabel("relative window midpoint")
    axes[1].set_ylabel("endpoint-free local rate")
    axes[1].set_title("Rolling local-rate instability")

    confusion = metadata["synthetic_structural_confusion_fractions"]
    matrix = np.asarray(
        [
            [confusion[truth][model] for model in STRUCTURAL_MODELS]
            for truth in STRUCTURAL_MODELS
        ]
    )
    image = axes[2].imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axes[2].text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > 0.55 else "black",
            )
    labels = ("R1", "R2", "two phase")
    axes[2].set_xticks(range(3), labels)
    axes[2].set_yticks(range(3), labels)
    axes[2].set_xlabel("selected model")
    axes[2].set_ylabel("true model")
    axes[2].set_title("Synthetic identifiability")
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    real_groups = load_published_groups(project_root)
    real_groups.extend(load_reproduction_groups(project_root))
    synthetic_groups = generate_synthetic_groups(
        args.synthetic_replicates, args.synthetic_noise, args.seed
    )
    groups = synthetic_groups + real_groups

    detail_rows: list[dict[str, object]] = []
    for index, group in enumerate(groups, start=1):
        detail_rows.extend(audit_group(group))
        if index % 25 == 0:
            print(f"audited {index}/{len(groups)} trajectory groups")

    local_rows = local_difference_rates(real_groups, args.local_window)
    summary_rows = summarize_models(detail_rows)
    local_summary_rows = summarize_local_rates(local_rows)
    metadata = aggregate_metadata(groups, detail_rows)
    metadata.update(
        {
            "seed": args.seed,
            "synthetic_replicates_per_truth": args.synthetic_replicates,
            "synthetic_noise": args.synthetic_noise,
            "local_window": args.local_window,
        }
    )

    results_directory = project_root / "results"
    figure_directory = project_root / "figures"
    write_csv(results_directory / "local_dynamics_group_details.csv", detail_rows)
    write_csv(results_directory / "local_dynamics_summary.csv", summary_rows)
    write_csv(results_directory / "local_rate_windows.csv", local_rows)
    write_csv(results_directory / "local_rate_summary.csv", local_summary_rows)
    (results_directory / "local_dynamics_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(
        detail_rows,
        local_rows,
        metadata,
        figure_directory / "fig6_local_dynamics_audit.png",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
