"""Calibrated Hankel-rank and matrix-pencil audit of response-mode count.

For a response with unknown endpoint,

    y_n = y_infinity + sum_j a_j theta_j**n,

the first-difference sequence removes the endpoint and has Hankel rank equal to
the number of distinct active response modes.  Exact rank is unusable under
generic noise, so this script calibrates tail singular-energy tests on matched
synthetic rank-R nulls.  Matrix-pencil recovery is assessed separately: model
order detection can work even when the individual poles are not stable.

The audit uses only NumPy and Matplotlib.  It reuses the trajectory loaders from
``forecast_baseline_audit.py`` so that the published and reproduction panels
match the rolling forecast audit exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from forecast_baseline_audit import (  # noqa: E402
    Trajectory,
    load_published,
    load_reproduction,
    split_counts,
)


PRIMARY_NOISE = 0.01
NOISE_LEVELS = (0.005, 0.01, 0.02, 0.03)
NULL_ORDERS = (1, 2, 3)
TRUTH_ORDERS = (1, 2, 3, 4)
ORDER_LABELS = ("1", "2", "3", "4+")


@dataclass(frozen=True)
class SyntheticCurve:
    name: str
    order: int
    values: np.ndarray
    modes: np.ndarray


@dataclass(frozen=True)
class PencilResult:
    requested_order: int
    poles: np.ndarray
    all_real_stable: bool
    reconstruction_nrmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIRECTORY.parent)
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def response_modes(order: int, rng: np.random.Generator) -> np.ndarray:
    bands = {
        1: ((0.18, 0.98),),
        2: ((0.12, 0.58), (0.70, 0.98)),
        3: ((0.10, 0.36), (0.44, 0.70), (0.78, 0.98)),
        4: ((0.07, 0.25), (0.31, 0.50), (0.57, 0.78), (0.84, 0.98)),
    }
    return np.asarray([rng.uniform(low, high) for low, high in bands[order]])


def generate_synthetic_curves(
    replicates: int,
    noise_fraction: float,
    seed: int,
    checkpoint_count: int,
) -> list[SyntheticCurve]:
    """Generate matched positive finite-modal curves with response-scale noise."""
    parameter_rng = np.random.default_rng(seed)
    noise_seed = seed + int(round(noise_fraction * 1_000_000)) + 7919
    noise_rng = np.random.default_rng(noise_seed)
    checkpoints = np.arange(checkpoint_count, dtype=float)
    curves: list[SyntheticCurve] = []
    for order in TRUTH_ORDERS:
        for replicate in range(replicates):
            modes = response_modes(order, parameter_rng)
            raw_weights = parameter_rng.uniform(0.15, 1.0, size=order)
            weights = raw_weights / np.sum(raw_weights)
            endpoint = parameter_rng.uniform(-0.2, 0.2)
            signal = endpoint + np.sum(
                weights[:, None] * modes[:, None] ** checkpoints[None, :], axis=0
            )
            values = signal + noise_rng.normal(
                0.0, noise_fraction, size=checkpoint_count
            )
            curves.append(
                SyntheticCurve(
                    name=f"rank_{order}_{replicate:04d}",
                    order=order,
                    values=values,
                    modes=modes,
                )
            )
    return curves


def full_hankel(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=float)
    rows = (sequence.size + 1) // 2
    columns = sequence.size - rows + 1
    return np.asarray(
        [[sequence[row + column] for column in range(columns)] for row in range(rows)]
    )


def pencil_hankels(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sequence = np.asarray(sequence, dtype=float)
    rows = sequence.size // 2
    columns = sequence.size - rows
    h0 = np.asarray(
        [[sequence[row + column] for column in range(columns)] for row in range(rows)]
    )
    h1 = np.asarray(
        [
            [sequence[row + column + 1] for column in range(columns)]
            for row in range(rows)
        ]
    )
    return h0, h1


def tail_energy_ratios(values: np.ndarray, count: int) -> dict[int, float]:
    differences = np.diff(np.asarray(values[:count], dtype=float))
    singular_values = np.linalg.svd(full_hankel(differences), compute_uv=False)
    energy = float(np.sum(singular_values**2))
    if energy <= 1e-30:
        return {order: 0.0 for order in NULL_ORDERS}
    return {
        order: float(np.sum(singular_values[order:] ** 2) / energy)
        for order in NULL_ORDERS
    }


def matrix_pencil_poles(sequence: np.ndarray, order: int) -> np.ndarray:
    h0, h1 = pencil_hankels(sequence)
    if min(h0.shape) < order:
        return np.asarray([], dtype=complex)
    u, singular_values, vh = np.linalg.svd(h0, full_matrices=False)
    if singular_values[order - 1] <= max(singular_values[0], 1.0) * 1e-12:
        return np.asarray([], dtype=complex)
    ur = u[:, :order]
    vr = vh[:order, :].T
    inverse_singular = np.diag(1.0 / singular_values[:order])
    reduced_shift = ur.T @ h1 @ vr @ inverse_singular
    return np.linalg.eigvals(reduced_shift)


def fit_matrix_pencil(values: np.ndarray, count: int, order: int) -> PencilResult:
    train = np.asarray(values[:count], dtype=float)
    poles = matrix_pencil_poles(np.diff(train), order)
    real_stable = (
        poles.size == order
        and np.all(np.abs(np.imag(poles)) <= 1e-6)
        and np.all(np.real(poles) > 0.0)
        and np.all(np.real(poles) < 1.0)
    )
    if not real_stable:
        return PencilResult(order, poles, False, float("nan"))
    real_poles = np.sort(np.real(poles))
    checkpoints = np.arange(count, dtype=float)
    design = np.column_stack(
        (np.ones(count), real_poles[None, :] ** checkpoints[:, None])
    )
    coefficients = np.linalg.lstsq(design, train, rcond=None)[0]
    fitted = design @ coefficients
    scale = max(float(np.ptp(train)), 1e-12)
    nrmse = float(np.sqrt(np.mean((fitted - train) ** 2)) / scale)
    return PencilResult(order, real_poles, True, nrmse)


def calibrate_thresholds(
    curves_by_noise: dict[float, list[SyntheticCurve]],
    counts: list[int],
) -> tuple[dict[tuple[float, int, int], float], list[dict[str, object]]]:
    thresholds: dict[tuple[float, int, int], float] = {}
    rows: list[dict[str, object]] = []
    for noise_fraction, curves in curves_by_noise.items():
        by_order = defaultdict(list)
        for curve in curves:
            by_order[curve.order].append(curve)
        for count in counts:
            for null_order in NULL_ORDERS:
                scores = np.asarray(
                    [
                        tail_energy_ratios(curve.values, count)[null_order]
                        for curve in by_order[null_order]
                    ]
                )
                threshold = float(np.quantile(scores, 0.95))
                thresholds[(noise_fraction, count, null_order)] = threshold
                rows.append(
                    {
                        "noise_fraction": noise_fraction,
                        "train_count": count,
                        "null_order": null_order,
                        "replicates": scores.size,
                        "tail_energy_threshold_95": threshold,
                        "empirical_false_positive_rate": float(
                            np.mean(scores > threshold)
                        ),
                        "median_null_tail_energy": float(np.median(scores)),
                    }
                )
    return thresholds, rows


def select_order(
    ratios: dict[int, float],
    thresholds: dict[tuple[float, int, int], float],
    noise_fraction: float,
    count: int,
) -> str:
    for order in NULL_ORDERS:
        if ratios[order] <= thresholds[(noise_fraction, count, order)]:
            return str(order)
    return "4+"


def calibration_confusion(
    curves_by_noise: dict[float, list[SyntheticCurve]],
    rolling_counts: list[int],
    thresholds: dict[tuple[float, int, int], float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for noise_fraction, curves in curves_by_noise.items():
        counter: Counter[tuple[int, str]] = Counter()
        totals: Counter[int] = Counter()
        for curve in curves:
            for count in rolling_counts:
                selected = select_order(
                    tail_energy_ratios(curve.values, count),
                    thresholds,
                    noise_fraction,
                    count,
                )
                counter[(curve.order, selected)] += 1
                totals[curve.order] += 1
        for truth_order in TRUTH_ORDERS:
            for selected_order in ORDER_LABELS:
                cases = counter[(truth_order, selected_order)]
                rows.append(
                    {
                        "noise_fraction": noise_fraction,
                        "truth_order": truth_order,
                        "selected_order": selected_order,
                        "cases": cases,
                        "fraction": cases / totals[truth_order],
                    }
                )
    return rows


def pencil_recovery_summary(
    curves: list[SyntheticCurve], counts: list[int]
) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[tuple[bool, bool, float]]] = defaultdict(list)
    for curve in curves:
        for count in counts:
            result = fit_matrix_pencil(curve.values, count, curve.order)
            within_tolerance = False
            maximum_error = float("nan")
            if result.all_real_stable:
                maximum_error = float(
                    np.max(np.abs(np.sort(result.poles) - np.sort(curve.modes)))
                )
                within_tolerance = maximum_error <= 0.05
            groups[(curve.order, count)].append(
                (result.all_real_stable, within_tolerance, maximum_error)
            )
    rows: list[dict[str, object]] = []
    for (order, count), values in sorted(groups.items()):
        valid_errors = np.asarray([value[2] for value in values if value[0]])
        rows.append(
            {
                "truth_order": order,
                "train_count": count,
                "replicates": len(values),
                "real_stable_pole_fraction": float(
                    np.mean([value[0] for value in values])
                ),
                "all_poles_within_0_05_fraction": float(
                    np.mean([value[1] for value in values])
                ),
                "median_maximum_pole_error_when_valid": (
                    float(np.median(valid_errors)) if valid_errors.size else float("nan")
                ),
            }
        )
    return rows


def pole_text(poles: np.ndarray) -> str:
    if poles.size == 0:
        return ""
    return "|".join(
        f"{value.real:.6g}{value.imag:+.6g}j" if abs(value.imag) > 1e-8 else f"{value.real:.6g}"
        for value in np.sort_complex(poles)
    )


def audit_real_trajectories(
    trajectories: list[Trajectory],
    thresholds: dict[tuple[float, int, int], float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trajectory in trajectories:
        for count in split_counts(trajectory.values.size):
            ratios = tail_energy_ratios(trajectory.values, count)
            for noise_fraction in NOISE_LEVELS:
                selected = select_order(
                    ratios, thresholds, noise_fraction, count
                )
                pencil_order = 4 if selected == "4+" else int(selected)
                pencil = fit_matrix_pencil(
                    trajectory.values, count, pencil_order
                )
                rows.append(
                    {
                        "source": trajectory.source,
                        "series": trajectory.name,
                        "checkpoint_count": trajectory.values.size,
                        "train_count": count,
                        "assumed_noise_fraction": noise_fraction,
                        "tail_energy_rank_1": ratios[1],
                        "tail_energy_rank_2": ratios[2],
                        "tail_energy_rank_3": ratios[3],
                        "selected_order": selected,
                        "pencil_order": pencil_order,
                        "pencil_all_real_stable": pencil.all_real_stable,
                        "pencil_poles": pole_text(pencil.poles),
                        "pencil_train_nrmse": pencil.reconstruction_nrmse,
                    }
                )
    return rows


def bootstrap_pole_stability(
    trajectory: Trajectory,
    selected_order: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    order = 4 if selected_order == "4+" else int(selected_order)
    values = np.asarray(trajectory.values, dtype=float)
    scale = max(float(np.ptp(values)), 1e-12)
    rng = np.random.default_rng(seed)
    recovered: list[np.ndarray] = []
    for _ in range(replicates):
        perturbed = values + rng.normal(0.0, PRIMARY_NOISE * scale, size=values.size)
        result = fit_matrix_pencil(perturbed, values.size, order)
        if result.all_real_stable:
            recovered.append(np.sort(result.poles))
    if recovered:
        pole_matrix = np.vstack(recovered)
        medians = np.median(pole_matrix, axis=0)
        lower = np.quantile(pole_matrix, 0.10, axis=0)
        upper = np.quantile(pole_matrix, 0.90, axis=0)
        maximum_width = float(np.max(upper - lower))
    else:
        medians = np.asarray([])
        lower = np.asarray([])
        upper = np.asarray([])
        maximum_width = float("nan")
    valid_fraction = len(recovered) / replicates
    return {
        "bootstrap_replicates": replicates,
        "bootstrap_valid_pole_fraction": valid_fraction,
        "bootstrap_pole_medians": pole_text(medians),
        "bootstrap_pole_10pct": pole_text(lower),
        "bootstrap_pole_90pct": pole_text(upper),
        "bootstrap_maximum_80pct_width": maximum_width,
        "bootstrap_poles_stable": bool(
            valid_fraction >= 0.80
            and np.isfinite(maximum_width)
            and maximum_width <= 0.10
        ),
    }


def summarize_real_series(
    trajectories: list[Trajectory],
    detail_rows: list[dict[str, object]],
    thresholds: dict[tuple[float, int, int], float],
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    primary = [
        row
        for row in detail_rows
        if float(row["assumed_noise_fraction"]) == PRIMARY_NOISE
    ]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in primary:
        grouped[(str(row["source"]), str(row["series"]))].append(row)
    trajectory_map = {(item.source, item.name): item for item in trajectories}
    rows: list[dict[str, object]] = []
    for index, key in enumerate(sorted(grouped)):
        group = sorted(grouped[key], key=lambda row: int(row["train_count"]))
        selections = [str(row["selected_order"]) for row in group]
        consensus = max(
            ORDER_LABELS,
            key=lambda label: (selections.count(label), -ORDER_LABELS.index(label)),
        )
        trajectory = trajectory_map[key]
        full_count = trajectory.values.size
        full_ratios = tail_energy_ratios(trajectory.values, full_count)
        full_selection = select_order(
            full_ratios, thresholds, PRIMARY_NOISE, full_count
        )
        full_pencil_order = 4 if full_selection == "4+" else int(full_selection)
        full_pencil = fit_matrix_pencil(
            trajectory.values, full_count, full_pencil_order
        )
        bootstrap = bootstrap_pole_stability(
            trajectory,
            full_selection,
            bootstrap_replicates,
            seed + 1009 * (index + 1),
        )
        row: dict[str, object] = {
            "source": key[0],
            "series": key[1],
            "split_count": len(group),
            "train_count_sequence": "|".join(str(row["train_count"]) for row in group),
            "selected_order_sequence": "|".join(selections),
            "consensus_order": consensus,
            "consensus_fraction": selections.count(consensus) / len(selections),
            "fully_stable_order": len(set(selections)) == 1,
            "full_series_selected_order": full_selection,
            "full_series_pencil_all_real_stable": full_pencil.all_real_stable,
            "full_series_pencil_poles": pole_text(full_pencil.poles),
            "full_series_pencil_train_nrmse": full_pencil.reconstruction_nrmse,
        }
        row.update(bootstrap)
        row["stable_order_and_poles"] = bool(
            row["fully_stable_order"] and row["bootstrap_poles_stable"]
        )
        rows.append(row)
    return rows


def read_rolling_forecast_context(project_root: Path) -> dict[str, object]:
    path = project_root / "results" / "rolling_forecast_metadata.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    comparisons = data["comparisons"]
    return {
        source: {
            "case_count": comparisons[source]["case_count"],
            "winner_counts": comparisons[source]["winner_counts"],
            "power_law_beats_both_modal": comparisons[source][
                "power_law_beats_both_modal"
            ],
        }
        for source in ("published", "reproduction")
    }


def aggregate_metadata(
    trajectories: list[Trajectory],
    calibration_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
    recovery_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    series_rows: list[dict[str, object]],
    forecast_context: dict[str, object],
    replicates: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    noiseless_curves = generate_synthetic_curves(20, 0.0, seed, 21)
    noiseless_validation: dict[str, object] = {}
    for order in TRUTH_ORDERS:
        errors = []
        for curve in noiseless_curves:
            if curve.order != order:
                continue
            result = fit_matrix_pencil(curve.values, 21, order)
            if result.all_real_stable:
                errors.append(
                    float(
                        np.max(
                            np.abs(np.sort(result.poles) - np.sort(curve.modes))
                        )
                    )
                )
        noiseless_validation[str(order)] = {
            "recovered_fraction": len(errors) / 20,
            "maximum_pole_error": max(errors) if errors else float("nan"),
        }
    primary_details = [
        row
        for row in detail_rows
        if float(row["assumed_noise_fraction"]) == PRIMARY_NOISE
    ]
    real_summary: dict[str, object] = {}
    for source in ("published", "reproduction"):
        source_details = [row for row in primary_details if row["source"] == source]
        source_series = [row for row in series_rows if row["source"] == source]
        real_summary[source] = {
            "trajectory_count": sum(item.source == source for item in trajectories),
            "rolling_case_count": len(source_details),
            "selected_order_counts_at_1pct": {
                label: sum(row["selected_order"] == label for row in source_details)
                for label in ORDER_LABELS
            },
            "rank_one_rejected_cases_at_1pct": sum(
                row["selected_order"] != "1" for row in source_details
            ),
            "rank_two_rejected_cases_at_1pct": sum(
                row["selected_order"] in ("3", "4+") for row in source_details
            ),
            "fully_stable_order_trajectories": sum(
                bool(row["fully_stable_order"]) for row in source_series
            ),
            "bootstrap_stable_pole_trajectories": sum(
                bool(row["bootstrap_poles_stable"]) for row in source_series
            ),
            "stable_order_and_pole_trajectories": sum(
                bool(row["stable_order_and_poles"]) for row in source_series
            ),
            "full_series_selected_order_counts": {
                label: sum(
                    row["full_series_selected_order"] == label
                    for row in source_series
                )
                for label in ORDER_LABELS
            },
        }

    sensitivity: dict[str, object] = {}
    for source in ("published", "reproduction"):
        sensitivity[source] = {}
        for noise_fraction in NOISE_LEVELS:
            group = [
                row
                for row in detail_rows
                if row["source"] == source
                and float(row["assumed_noise_fraction"]) == noise_fraction
            ]
            sensitivity[source][str(noise_fraction)] = {
                "rank_one_rejected_fraction": float(
                    np.mean([row["selected_order"] != "1" for row in group])
                ),
                "rank_two_rejected_fraction": float(
                    np.mean(
                        [row["selected_order"] in ("3", "4+") for row in group]
                    )
                ),
                "selected_order_counts": {
                    label: sum(row["selected_order"] == label for row in group)
                    for label in ORDER_LABELS
                },
            }

    primary_confusion = [
        row
        for row in confusion_rows
        if float(row["noise_fraction"]) == PRIMARY_NOISE
    ]
    exact_selection = {
        str(order): next(
            float(row["fraction"])
            for row in primary_confusion
            if int(row["truth_order"]) == order
            and str(row["selected_order"]) == (str(order) if order < 4 else "4+")
        )
        for order in TRUTH_ORDERS
    }
    rank_one_rejection = {
        str(order): float(
            sum(
                float(row["fraction"])
                for row in primary_confusion
                if int(row["truth_order"]) == order
                and str(row["selected_order"]) != "1"
            )
        )
        for order in TRUTH_ORDERS
    }
    rank_two_rejection = {
        str(order): float(
            sum(
                float(row["fraction"])
                for row in primary_confusion
                if int(row["truth_order"]) == order
                and str(row["selected_order"]) in ("3", "4+")
            )
        )
        for order in TRUTH_ORDERS
    }
    recovery_at_full = {
        str(order): {
            "real_stable_pole_fraction": float(row["real_stable_pole_fraction"]),
            "all_poles_within_0_05_fraction": float(
                row["all_poles_within_0_05_fraction"]
            ),
        }
        for order in TRUTH_ORDERS
        for row in recovery_rows
        if int(row["truth_order"]) == order and int(row["train_count"]) == 21
    }
    published_series = [row for row in series_rows if row["source"] == "published"]
    role_poles: dict[str, list[float]] = defaultdict(list)
    panels: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in published_series:
        panel, role = str(row["series"]).split("__", 1)
        median = float(row["bootstrap_pole_medians"])
        lower = float(row["bootstrap_pole_10pct"])
        upper = float(row["bootstrap_pole_90pct"])
        role_poles[role].append(median)
        panels[panel][role] = {"median": median, "lower": lower, "upper": upper}
    nonoverlapping_panels = 0
    ordered_panels = 0
    for roles in panels.values():
        pairs = (("gap", "solver"), ("gap", "verifier"), ("solver", "verifier"))
        if all(
            roles[left]["upper"] < roles[right]["lower"]
            or roles[right]["upper"] < roles[left]["lower"]
            for left, right in pairs
        ):
            nonoverlapping_panels += 1
        if roles["gap"]["median"] < roles["solver"]["median"] < roles["verifier"]["median"]:
            ordered_panels += 1
    return {
        "method": {
            "observable": "first differences, which remove the unknown endpoint",
            "differencing_cost": (
                "a response amplitude a_j becomes a_j*(theta_j-1), so slow modes "
                "are attenuated while observation noise is amplified"
            ),
            "rank_statistic": "fraction of squared Hankel singular values beyond candidate order R",
            "selection_rule": "smallest R in 1..3 not rejected at a calibrated 5% level; otherwise 4+",
            "primary_assumed_noise_fraction": PRIMARY_NOISE,
            "noise_sensitivity_levels": NOISE_LEVELS,
            "matrix_pencil_validity": "all requested poles real and strictly inside (0,1)",
            "bootstrap_stability": "at least 80% valid pencils and maximum 10th--90th pole width at most 0.10",
        },
        "calibration": {
            "replicates_per_order": replicates,
            "noiseless_21_checkpoint_validation": noiseless_validation,
            "rolling_exact_order_selection_fraction_at_1pct": exact_selection,
            "rolling_rank_one_rejection_fraction_at_1pct_by_truth": rank_one_rejection,
            "rolling_rank_two_rejection_fraction_at_1pct_by_truth": rank_two_rejection,
            "full_21_checkpoint_matrix_pencil_recovery": recovery_at_full,
            "false_positive_rates": sorted(
                {
                    round(float(row["empirical_false_positive_rate"]), 6)
                    for row in calibration_rows
                }
            ),
        },
        "real_data": real_summary,
        "published_conditional_dominant_poles": {
            "median_by_role": {
                role: float(np.median(values))
                for role, values in sorted(role_poles.items())
            },
            "panels_with_gap_below_solver_below_verifier": ordered_panels,
            "panels_with_pairwise_nonoverlapping_bootstrap_80pct_intervals": nonoverlapping_panels,
            "panel_count": len(panels),
            "qualification": (
                "These are perturbation-stable one-pole summaries, not evidence that "
                "each curve truly has only one pole. Their separation rejects the exact "
                "shared-single-mode restriction more directly than it identifies a "
                "larger common mode count."
            ),
        },
        "noise_sensitivity": sensitivity,
        "forecast_context": forecast_context,
        "bootstrap_replicates_per_real_trajectory": bootstrap_replicates,
        "seed": seed,
        "qualification": (
            "Mode counts are effective orders conditional on a finite-exponential model "
            "and the assumed response-scale noise. In this calibration the endpoint-free "
            "rank test has almost no power against heterogeneous ranks above one, so "
            "accepting rank one on the published curves is not evidence for rank one. "
            "Failure to recover stable poles is nonidentifiability, not evidence that "
            "the corresponding modes do not exist."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    confusion_rows: list[dict[str, object]],
    recovery_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    series_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.1))

    primary_confusion = [
        row
        for row in confusion_rows
        if float(row["noise_fraction"]) == PRIMARY_NOISE
    ]
    matrix = np.zeros((4, 4))
    for row in primary_confusion:
        matrix[int(row["truth_order"]) - 1, ORDER_LABELS.index(str(row["selected_order"]))] = float(
            row["fraction"]
        )
    image = axes[0, 0].imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    for row in range(4):
        for column in range(4):
            axes[0, 0].text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > 0.55 else "black",
                fontsize=8,
            )
    axes[0, 0].set_xticks(range(4), ORDER_LABELS)
    axes[0, 0].set_yticks(range(4), ("1", "2", "3", "4"))
    axes[0, 0].set_xlabel("selected effective order")
    axes[0, 0].set_ylabel("true order")
    axes[0, 0].set_title("A. Calibrated selection, 1% noise")
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)

    primary_details = [
        row
        for row in detail_rows
        if float(row["assumed_noise_fraction"]) == PRIMARY_NOISE
    ]
    x = np.arange(4)
    width = 0.34
    for offset, source in ((-width / 2, "published"), (width / 2, "reproduction")):
        group = [row for row in primary_details if row["source"] == source]
        fractions = [
            np.mean([row["selected_order"] == label for row in group])
            for label in ORDER_LABELS
        ]
        axes[0, 1].bar(x + offset, fractions, width, label=source)
    axes[0, 1].set_xticks(x, ORDER_LABELS)
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_xlabel("selected effective order")
    axes[0, 1].set_ylabel("rolling-case fraction")
    axes[0, 1].set_title("B. Real trajectories, 1% calibration")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.25)

    for order in TRUTH_ORDERS:
        group = sorted(
            (row for row in recovery_rows if int(row["truth_order"]) == order),
            key=lambda row: int(row["train_count"]),
        )
        axes[1, 0].plot(
            [int(row["train_count"]) for row in group],
            [float(row["all_poles_within_0_05_fraction"]) for row in group],
            marker="o",
            label=f"rank {order}",
        )
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_xlabel("checkpoints")
    axes[1, 0].set_ylabel("all poles recovered within 0.05")
    axes[1, 0].set_title("C. Matrix-pencil recovery, 1% noise")
    axes[1, 0].legend(frameon=False, ncol=2)
    axes[1, 0].grid(alpha=0.25)

    published_series = [row for row in series_rows if row["source"] == "published"]
    panel_order = ("gsm8k_qe", "gsm8k_tf", "math_qe", "math_tf")
    role_style = {
        "gap": ("C0", "o"),
        "solver": ("C2", "s"),
        "verifier": ("C3", "^"),
    }
    for offset, role in zip((-0.12, 0.0, 0.12), ("gap", "solver", "verifier")):
        role_rows = {
            str(row["series"]).split("__", 1)[0]: row
            for row in published_series
            if str(row["series"]).endswith(f"__{role}")
        }
        medians = np.asarray(
            [float(role_rows[panel]["bootstrap_pole_medians"]) for panel in panel_order]
        )
        lowers = np.asarray(
            [float(role_rows[panel]["bootstrap_pole_10pct"]) for panel in panel_order]
        )
        uppers = np.asarray(
            [float(role_rows[panel]["bootstrap_pole_90pct"]) for panel in panel_order]
        )
        color, marker = role_style[role]
        axes[1, 1].errorbar(
            np.arange(len(panel_order)) + offset,
            medians,
            yerr=np.vstack((medians - lowers, uppers - medians)),
            color=color,
            marker=marker,
            capsize=3,
            linestyle="none",
            label=role,
        )
    axes[1, 1].set_ylim(0.65, 0.94)
    axes[1, 1].set_xticks(
        np.arange(len(panel_order)),
        ("GSM8K QE", "GSM8K TF", "MATH QE", "MATH TF"),
        rotation=18,
    )
    axes[1, 1].set_ylabel("conditional dominant pole")
    axes[1, 1].set_title("D. Published one-pole summaries")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=3)
    axes[1, 1].grid(alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.replicates < 20:
        raise ValueError("at least 20 calibration replicates are required")
    if args.bootstrap_replicates < 20:
        raise ValueError("at least 20 bootstrap replicates are required")
    project_root = args.project_root.resolve()
    trajectories = load_published(project_root) + load_reproduction(project_root)
    rolling_counts = sorted(
        {count for item in trajectories for count in split_counts(item.values.size)}
    )
    all_counts = sorted(
        set(rolling_counts) | {int(item.values.size) for item in trajectories}
    )
    maximum_count = max(all_counts)

    curves_by_noise = {
        noise_fraction: generate_synthetic_curves(
            args.replicates,
            noise_fraction,
            args.seed,
            maximum_count,
        )
        for noise_fraction in NOISE_LEVELS
    }
    thresholds, calibration_rows = calibrate_thresholds(
        curves_by_noise, all_counts
    )
    confusion_rows = calibration_confusion(
        curves_by_noise, rolling_counts, thresholds
    )
    recovery_rows = pencil_recovery_summary(
        curves_by_noise[PRIMARY_NOISE], all_counts
    )
    detail_rows = audit_real_trajectories(trajectories, thresholds)
    series_rows = summarize_real_series(
        trajectories,
        detail_rows,
        thresholds,
        args.bootstrap_replicates,
        args.seed,
    )
    forecast_context = read_rolling_forecast_context(project_root)
    metadata = aggregate_metadata(
        trajectories,
        calibration_rows,
        confusion_rows,
        recovery_rows,
        detail_rows,
        series_rows,
        forecast_context,
        args.replicates,
        args.bootstrap_replicates,
        args.seed,
    )

    results_directory = project_root / "results"
    write_csv(results_directory / "spectral_calibration_summary.csv", calibration_rows)
    write_csv(results_directory / "spectral_selection_confusion.csv", confusion_rows)
    write_csv(results_directory / "matrix_pencil_recovery.csv", recovery_rows)
    write_csv(results_directory / "spectral_mode_details.csv", detail_rows)
    write_csv(results_directory / "spectral_mode_series_summary.csv", series_rows)
    (results_directory / "spectral_mode_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(
        confusion_rows,
        recovery_rows,
        detail_rows,
        series_rows,
        project_root / "figures" / "fig5_spectral_mode_audit.png",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
