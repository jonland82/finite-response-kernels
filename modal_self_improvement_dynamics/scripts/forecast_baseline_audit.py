"""Rolling held-out forecast audit with modal and non-modal baselines.

The audit intentionally fits every model using only a training prefix.  It
then forecasts the untouched tail at several split points for four trajectory
sources: exact synthetic examples, a fixed-seed noisy synthetic benchmark,
the digitized Sun et al. curves, and the local three-seed reproduction.
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
from fit_modal_models import DEFAULT_MODE_GRID, fit_modal_batch  # noqa: E402


MODEL_NAMES = ("persistence", "local_linear", "power_law", "rank_1", "rank_2")
SPLIT_FRACTIONS = (0.50, 0.60, 0.70, 0.75, 0.80)
REPRODUCTION_SERIES = (
    "solver_uncertainty",
    "verifier_uncertainty",
    "uncertainty_gap",
    "solver_normalized_uncertainty",
    "verifier_normalized_uncertainty",
    "normalized_uncertainty_gap",
)


@dataclass(frozen=True)
class Trajectory:
    source: str
    name: str
    checkpoints: np.ndarray
    values: np.ndarray
    truth: str = "unknown"


@dataclass(frozen=True)
class Forecast:
    values: np.ndarray
    train_sse: float
    parameters: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=SCRIPT_DIRECTORY.parent)
    parser.add_argument("--synthetic-replicates", type=int, default=200)
    parser.add_argument("--synthetic-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_exact_synthetic(project_root: Path) -> list[Trajectory]:
    rows = read_csv(project_root / "results" / "endpoint_separation_trajectories.csv")
    checkpoints = np.asarray([float(row["checkpoint"]) for row in rows])
    return [
        Trajectory(
            source="synthetic_exact",
            name="rank_one_residual",
            checkpoints=checkpoints,
            values=np.asarray([float(row["single_mode_residual"]) for row in rows]),
            truth="rank_1",
        ),
        Trajectory(
            source="synthetic_exact",
            name="rank_two_residual",
            checkpoints=checkpoints,
            values=np.asarray([float(row["two_mode_residual"]) for row in rows]),
            truth="rank_2",
        ),
    ]


def generate_noisy_synthetic(
    replicates: int, noise_sd: float, seed: int
) -> list[Trajectory]:
    if replicates < 1:
        raise ValueError("synthetic replicates must be positive")
    if noise_sd < 0.0:
        raise ValueError("synthetic noise must be nonnegative")
    rng = np.random.default_rng(seed)
    checkpoints = np.arange(21, dtype=float)
    output: list[Trajectory] = []
    for replicate in range(replicates):
        endpoint = rng.uniform(-0.20, 0.20)
        mode = rng.uniform(0.30, 0.95)
        truth = endpoint + mode**checkpoints
        values = truth + rng.normal(0.0, noise_sd, size=checkpoints.size)
        output.append(
            Trajectory(
                source="synthetic_1pct",
                name=f"rank_one_{replicate:03d}",
                checkpoints=checkpoints,
                values=values,
                truth="rank_1",
            )
        )
    for replicate in range(replicates):
        endpoint = rng.uniform(-0.20, 0.20)
        fast_mode = rng.uniform(0.15, 0.60)
        slow_mode = rng.uniform(max(0.72, fast_mode + 0.12), 0.98)
        fast_weight = rng.uniform(0.20, 0.80)
        truth = endpoint + fast_weight * fast_mode**checkpoints
        truth += (1.0 - fast_weight) * slow_mode**checkpoints
        values = truth + rng.normal(0.0, noise_sd, size=checkpoints.size)
        output.append(
            Trajectory(
                source="synthetic_1pct",
                name=f"rank_two_{replicate:03d}",
                checkpoints=checkpoints,
                values=values,
                truth="rank_2",
            )
        )
    return output


def load_published(project_root: Path) -> list[Trajectory]:
    rows = read_csv(project_root / "results" / "sun_phi4_figure4_digitized.csv")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["panel"], row["series"])].append(row)
    output = []
    for (panel, series), group in sorted(groups.items()):
        group.sort(key=lambda row: float(row["checkpoint"]))
        output.append(
            Trajectory(
                source="published",
                name=f"{panel}__{series}",
                checkpoints=np.asarray([float(row["epoch"]) for row in group]),
                values=np.asarray(
                    [float(row["affine_uncertainty"]) for row in group]
                ),
            )
        )
    return output


def load_reproduction(project_root: Path) -> list[Trajectory]:
    run_root = project_root / "reproduction" / "runs"
    output = []
    for path in sorted(run_root.glob("arithmetic_seed_*/trajectory.csv")):
        rows = read_csv(path)
        seed = rows[0]["seed"]
        checkpoints = np.asarray([float(row["epoch"]) for row in rows])
        for series in REPRODUCTION_SERIES:
            output.append(
                Trajectory(
                    source="reproduction",
                    name=f"{seed}__{series}",
                    checkpoints=checkpoints,
                    values=np.asarray([float(row[series]) for row in rows]),
                )
            )
    if not output:
        raise FileNotFoundError("no arithmetic_seed reproduction trajectories found")
    return output


def split_counts(checkpoint_count: int) -> list[int]:
    counts = {
        max(6, min(checkpoint_count - 3, int(checkpoint_count * fraction)))
        for fraction in SPLIT_FRACTIONS
    }
    return sorted(counts)


def endpoint_bounds(values: np.ndarray) -> tuple[float, float]:
    value_range = float(np.ptp(values))
    padding = value_range if value_range > 1e-12 else max(1.0, abs(float(values[0])))
    return float(np.min(values) - padding), float(np.max(values) + padding)


def persistence_forecast(train_values: np.ndarray, forecast_count: int) -> Forecast:
    values = np.full(forecast_count, train_values[-1])
    train_sse = float(np.sum((train_values - train_values[-1]) ** 2))
    return Forecast(values, train_sse, {"last_value": float(train_values[-1])})


def weighted_local_linear(
    train_x: np.ndarray, train_y: np.ndarray, forecast_x: np.ndarray, bandwidth: float
) -> tuple[np.ndarray, np.ndarray]:
    origin = float(train_x[-1])
    centered = train_x - origin
    weights = np.exp(-0.5 * (centered / bandwidth) ** 2)
    design = np.column_stack((np.ones(train_x.size), centered))
    root_weights = np.sqrt(np.maximum(weights, 1e-12))
    coefficients = np.linalg.lstsq(
        design * root_weights[:, None], train_y * root_weights, rcond=None
    )[0]
    forecast_design = np.column_stack(
        (np.ones(forecast_x.size), forecast_x - origin)
    )
    return forecast_design @ coefficients, coefficients


def local_linear_forecast(
    train_x: np.ndarray, train_y: np.ndarray, forecast_x: np.ndarray
) -> Forecast:
    spacing = float(np.median(np.diff(train_x)))
    bandwidths = spacing * np.asarray((1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0))
    first_validation = max(5, train_x.size // 2)
    validation_errors = []
    for bandwidth in bandwidths:
        errors = []
        for cutoff in range(first_validation, train_x.size):
            prediction, _ = weighted_local_linear(
                train_x[:cutoff],
                train_y[:cutoff],
                train_x[cutoff : cutoff + 1],
                float(bandwidth),
            )
            errors.append(float(prediction[0] - train_y[cutoff]) ** 2)
        validation_errors.append(float(np.mean(errors)))
    best_index = int(np.argmin(validation_errors))
    bandwidth = float(bandwidths[best_index])
    forecast, coefficients = weighted_local_linear(
        train_x, train_y, forecast_x, bandwidth
    )
    fitted, _ = weighted_local_linear(train_x, train_y, train_x, bandwidth)
    return Forecast(
        forecast,
        float(np.sum((fitted - train_y) ** 2)),
        {
            "bandwidth": bandwidth,
            "validation_mse": validation_errors[best_index],
            "intercept_at_last_checkpoint": float(coefficients[0]),
            "slope": float(coefficients[1]),
        },
    )


def power_law_forecast(
    train_x: np.ndarray, train_y: np.ndarray, forecast_x: np.ndarray
) -> Forecast:
    relative_train = train_x - train_x[0]
    relative_forecast = forecast_x - train_x[0]
    spacing = float(np.median(np.diff(train_x)))
    taus = spacing * np.geomspace(0.10, 20.0, 36)
    powers = np.geomspace(0.08, 6.0, 56)
    lower, upper = endpoint_bounds(train_y)
    best: tuple[float, float, float, float, float] | None = None
    unconstrained: tuple[float, float, float, float, float] | None = None
    for tau in taus:
        basis = (relative_train[None, :] + tau) ** (-powers[:, None])
        basis_mean = np.mean(basis, axis=1)
        centered_basis = basis - basis_mean[:, None]
        denominator = np.sum(centered_basis**2, axis=1)
        centered_y = train_y - np.mean(train_y)
        amplitudes = np.divide(
            centered_basis @ centered_y,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-20,
        )
        endpoints = np.mean(train_y) - amplitudes * basis_mean
        residuals = endpoints[:, None] + amplitudes[:, None] * basis - train_y
        sses = np.sum(residuals**2, axis=1)
        for index in range(powers.size):
            candidate = (
                float(sses[index]),
                float(endpoints[index]),
                float(amplitudes[index]),
                float(powers[index]),
                float(tau),
            )
            if unconstrained is None or candidate[0] < unconstrained[0]:
                unconstrained = candidate
            if lower <= candidate[1] <= upper and (
                best is None or candidate[0] < best[0]
            ):
                best = candidate
    selected = best or unconstrained
    if selected is None:
        raise RuntimeError("power-law grid produced no candidate")
    train_sse, endpoint, amplitude, power, tau = selected
    forecast = endpoint + amplitude * (relative_forecast + tau) ** (-power)
    return Forecast(
        forecast,
        train_sse,
        {
            "endpoint": endpoint,
            "amplitude": amplitude,
            "power": power,
            "time_shift": tau,
            "endpoint_constrained": best is not None,
        },
    )


def modal_forecast(
    train_x: np.ndarray, train_y: np.ndarray, forecast_x: np.ndarray, order: int
) -> Forecast:
    fit = fit_modal_batch(
        train_y,
        train_x - train_x[0],
        forecast_x - train_x[0],
        order,
        mode_grid=DEFAULT_MODE_GRID,
        endpoint_bounds=endpoint_bounds(train_y),
        direction="signed",
    )
    return Forecast(
        fit.forecast[0],
        float(fit.train_sse[0]),
        {
            "endpoint": float(fit.endpoints[0]),
            "modes": fit.modes[0].tolist(),
            "amplitudes": fit.amplitudes[0].tolist(),
        },
    )


def fit_all_models(
    train_x: np.ndarray, train_y: np.ndarray, forecast_x: np.ndarray
) -> dict[str, Forecast]:
    return {
        "persistence": persistence_forecast(train_y, forecast_x.size),
        "local_linear": local_linear_forecast(train_x, train_y, forecast_x),
        "power_law": power_law_forecast(train_x, train_y, forecast_x),
        "rank_1": modal_forecast(train_x, train_y, forecast_x, 1),
        "rank_2": modal_forecast(train_x, train_y, forecast_x, 2),
    }


def choose_winner(errors: dict[str, float]) -> str:
    minimum = min(errors.values())
    tolerance = max(1e-14, 1e-8 * max(errors.values()))
    for model in MODEL_NAMES:
        if errors[model] <= minimum + tolerance:
            return model
    raise RuntimeError("no forecast winner")


def audit_trajectory(trajectory: Trajectory) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_count in split_counts(trajectory.values.size):
        train_x = trajectory.checkpoints[:train_count]
        train_y = trajectory.values[:train_count]
        forecast_x = trajectory.checkpoints[train_count:]
        heldout_y = trajectory.values[train_count:]
        scale = max(float(np.ptp(train_y)), 1e-8)
        forecasts = fit_all_models(train_x, train_y, forecast_x)
        errors = {
            model: float(np.mean((forecast.values - heldout_y) ** 2))
            for model, forecast in forecasts.items()
        }
        winner = choose_winner(errors)
        persistence_mse = errors["persistence"]
        for model, forecast in forecasts.items():
            rows.append(
                {
                    "source": trajectory.source,
                    "series": trajectory.name,
                    "truth": trajectory.truth,
                    "checkpoint_count": trajectory.values.size,
                    "train_count": train_count,
                    "heldout_count": heldout_y.size,
                    "train_fraction": train_count / trajectory.values.size,
                    "model": model,
                    "winner": model == winner,
                    "train_sse": forecast.train_sse,
                    "heldout_mse": errors[model],
                    "heldout_nrmse": np.sqrt(errors[model]) / scale,
                    "mse_ratio_to_persistence": (
                        1.0
                        if model == "persistence"
                        else errors[model] / (persistence_mse + 1e-15)
                    ),
                    "parameters": json.dumps(forecast.parameters, sort_keys=True),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source"]), str(row["model"]))].append(row)
    summary = []
    for (source, model), group in sorted(groups.items()):
        ratios = np.asarray([float(row["mse_ratio_to_persistence"]) for row in group])
        nrmses = np.asarray([float(row["heldout_nrmse"]) for row in group])
        summary.append(
            {
                "source": source,
                "model": model,
                "cases": len(group),
                "wins": sum(bool(row["winner"]) for row in group),
                "win_fraction": np.mean([bool(row["winner"]) for row in group]),
                "beats_persistence_fraction": np.mean(ratios < 1.0),
                "median_nrmse": float(np.median(nrmses)),
                "median_mse_ratio_to_persistence": float(np.median(ratios)),
            }
        )
    return summary


def summarize_series(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source"]), str(row["series"]))].append(row)
    output = []
    for (source, series), group in sorted(groups.items()):
        winner_rows = sorted(
            (row for row in group if bool(row["winner"])),
            key=lambda row: int(row["train_count"]),
        )
        winners = [str(row["model"]) for row in winner_rows]
        consensus_model = max(
            MODEL_NAMES,
            key=lambda model: (winners.count(model), -MODEL_NAMES.index(model)),
        )
        cases = case_model_map(group, source)
        output.append(
            {
                "source": source,
                "series": series,
                "truth": str(group[0]["truth"]),
                "split_count": len(winners),
                "train_count_sequence": "|".join(
                    str(row["train_count"]) for row in winner_rows
                ),
                "winner_sequence": "|".join(winners),
                "consensus_model": consensus_model,
                "consensus_fraction": winners.count(consensus_model) / len(winners),
                "rank2_beats_rank1_splits": sum(
                    values["rank_2"] < values["rank_1"]
                    for values in cases.values()
                ),
                "power_law_beats_both_modal_splits": sum(
                    values["power_law"]
                    < min(values["rank_1"], values["rank_2"])
                    for values in cases.values()
                ),
            }
        )
    return output


def case_model_map(
    rows: list[dict[str, object]], source: str
) -> dict[tuple[str, int], dict[str, float]]:
    output: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["source"] == source:
            output[(str(row["series"]), int(row["train_count"]))][
                str(row["model"])
            ] = float(row["heldout_mse"])
    return output


def aggregate_metadata(
    trajectories: list[Trajectory], rows: list[dict[str, object]]
) -> dict[str, object]:
    sources = sorted({trajectory.source for trajectory in trajectories})
    comparisons: dict[str, object] = {}
    for source in sources:
        cases = case_model_map(rows, source)
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
            "rank2_beats_rank1": sum(
                values["rank_2"] < values["rank_1"] for values in cases.values()
            ),
            "rank1_beats_persistence": sum(
                values["rank_1"] < values["persistence"]
                for values in cases.values()
            ),
            "rank2_beats_persistence": sum(
                values["rank_2"] < values["persistence"]
                for values in cases.values()
            ),
            "power_law_beats_rank1": sum(
                values["power_law"] < values["rank_1"] for values in cases.values()
            ),
            "local_linear_beats_rank1": sum(
                values["local_linear"] < values["rank_1"]
                for values in cases.values()
            ),
            "power_law_beats_both_modal": sum(
                values["power_law"]
                < min(values["rank_1"], values["rank_2"])
                for values in cases.values()
            ),
            "best_modal_beats_persistence": sum(
                min(values["rank_1"], values["rank_2"])
                < values["persistence"]
                for values in cases.values()
            ),
        }

    synthetic_rows = [row for row in rows if row["source"] == "synthetic_1pct"]
    truth_by_case: dict[tuple[str, int], str] = {}
    for row in synthetic_rows:
        truth_by_case[(str(row["series"]), int(row["train_count"]))] = str(
            row["truth"]
        )
    synthetic_cases = case_model_map(rows, "synthetic_1pct")
    correct = 0
    truth_counts: dict[str, list[bool]] = defaultdict(list)
    train_count_results: dict[int, list[bool]] = defaultdict(list)
    for case, values in synthetic_cases.items():
        tolerance = max(
            1e-14,
            1e-8 * max(values["rank_1"], values["rank_2"]),
        )
        selected = (
            "rank_1"
            if values["rank_1"] <= values["rank_2"] + tolerance
            else "rank_2"
        )
        is_correct = selected == truth_by_case[case]
        correct += is_correct
        truth_counts[truth_by_case[case]].append(is_correct)
        train_count_results[case[1]].append(is_correct)

    calibrated_detection: dict[str, object] = {}
    for train_count in sorted(train_count_results):
        null_scores = []
        alternative_scores = []
        for case, values in synthetic_cases.items():
            if case[1] != train_count:
                continue
            score = float(
                np.log(
                    (values["rank_1"] + 1e-15)
                    / (values["rank_2"] + 1e-15)
                )
            )
            if truth_by_case[case] == "rank_1":
                null_scores.append(score)
            else:
                alternative_scores.append(score)
        threshold = float(np.quantile(null_scores, 0.95))
        calibrated_detection[str(train_count)] = {
            "log_mse_ratio_threshold": threshold,
            "false_positive_rate": float(np.mean(np.asarray(null_scores) > threshold)),
            "rank2_detection_power": float(
                np.mean(np.asarray(alternative_scores) > threshold)
            ),
        }

    winner_stability: dict[str, object] = {}
    for source in ("published", "reproduction"):
        series_winners: dict[str, list[str]] = defaultdict(list)
        for (series, _train_count), values in case_model_map(rows, source).items():
            series_winners[series].append(min(values, key=values.get))
        consensus = []
        for winners in series_winners.values():
            consensus.append(max(winners.count(model) for model in MODEL_NAMES) / len(winners))
        winner_stability[source] = {
            "series_count": len(series_winners),
            "mean_consensus_fraction": float(np.mean(consensus)),
            "fully_stable_winner_count": sum(value == 1.0 for value in consensus),
        }

    return {
        "trajectory_counts": {
            source: sum(trajectory.source == source for trajectory in trajectories)
            for source in sources
        },
        "split_fractions": SPLIT_FRACTIONS,
        "models": MODEL_NAMES,
        "comparisons": comparisons,
        "synthetic_modal_order_accuracy": {
            "method": "naive lower held-out MSE, with numerical ties favoring rank one",
            "overall": correct / len(synthetic_cases),
            "by_truth": {
                truth: float(np.mean(values))
                for truth, values in sorted(truth_counts.items())
            },
            "by_train_count": {
                str(train_count): float(np.mean(values))
                for train_count, values in sorted(train_count_results.items())
            },
        },
        "synthetic_calibrated_rank2_detection": {
            "method": (
                "split-specific 95th percentile of log(MSE_rank1/MSE_rank2) "
                "under the heterogeneous rank-one synthetic null"
            ),
            "by_train_count": calibrated_detection,
        },
        "winner_stability": winner_stability,
        "qualification": (
            "All fits use training prefixes only. Error normalization uses the "
            "training range; model wins use unnormalized MSE within each case."
        ),
    }


def make_figure(rows: list[dict[str, object]], output_path: Path) -> None:
    sources = ("synthetic_1pct", "published", "reproduction")
    compared_models = ("local_linear", "power_law", "rank_1", "rank_2")
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=True)
    for axis, source in zip(axes, sources):
        data = []
        for model in compared_models:
            ratios = [
                max(float(row["mse_ratio_to_persistence"]), 1e-8)
                for row in rows
                if row["source"] == source and row["model"] == model
            ]
            data.append(np.log10(ratios))
        axis.boxplot(data, tick_labels=("local", "power", "rank 1", "rank 2"), showfliers=False)
        axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
        axis.set_title(source.replace("_", " "))
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(r"$\log_{10}$(held-out MSE / persistence MSE)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    trajectories = load_exact_synthetic(project_root)
    trajectories.extend(
        generate_noisy_synthetic(
            args.synthetic_replicates, args.synthetic_noise, args.seed
        )
    )
    trajectories.extend(load_published(project_root))
    trajectories.extend(load_reproduction(project_root))

    detail_rows: list[dict[str, object]] = []
    for index, trajectory in enumerate(trajectories, start=1):
        detail_rows.extend(audit_trajectory(trajectory))
        if index % 25 == 0:
            print(f"audited {index}/{len(trajectories)} trajectories")

    summary_rows = summarize(detail_rows)
    series_rows = summarize_series(detail_rows)
    metadata = aggregate_metadata(trajectories, detail_rows)
    results_directory = project_root / "results"
    figure_directory = project_root / "figures"
    write_csv(results_directory / "rolling_forecast_details.csv", detail_rows)
    write_csv(results_directory / "rolling_forecast_summary.csv", summary_rows)
    write_csv(results_directory / "rolling_forecast_series_summary.csv", series_rows)
    (results_directory / "rolling_forecast_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(detail_rows, figure_directory / "fig4_rolling_forecast.png")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
