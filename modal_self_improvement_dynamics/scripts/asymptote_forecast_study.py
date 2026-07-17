"""Power study for modal order when the limiting endpoint is unknown.

Rank-one and rank-two models jointly estimate the asymptote from an initial
training prefix.  Model selection uses only held-out tail prediction.  A
rank-one Monte Carlo calibration sets a 5% false-positive threshold for the
log forecast-MSE improvement of rank two over rank one.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_modal_models import DEFAULT_MODE_GRID, fit_modal_batch


@dataclass(frozen=True)
class ForecastStudyConfig:
    checkpoint_counts: tuple[int, ...]
    noise_levels: tuple[float, ...]
    replicates: int
    holdout_fraction: float
    minimum_holdout: int
    false_positive_level: float
    target_power: float
    seed: int
    fast_mode: float
    slow_mode: float
    fast_weight: float
    endpoint_lower: float
    endpoint_upper: float


def parse_number_list(text: str, caster: type[int] | type[float]) -> tuple:
    values = tuple(caster(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def modal_residual(
    checkpoint_count: int,
    modes: tuple[float, ...],
    weights: tuple[float, ...],
) -> np.ndarray:
    checkpoints = np.arange(checkpoint_count, dtype=float)
    mode_array = np.asarray(modes, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    return np.sum(
        weight_array[:, None] * mode_array[:, None] ** checkpoints[None, :],
        axis=0,
    )


def simulate_observations(
    truth: np.ndarray,
    noise_sd: float,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(0.0, noise_sd, size=(replicates, truth.size))
    noise[:, 0] = 0.0
    return truth[None, :] + noise


def forecast_score(
    rank_one_forecast: np.ndarray,
    rank_two_forecast: np.ndarray,
    observed_holdout: np.ndarray,
    noise_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank_one_mse = np.mean((rank_one_forecast - observed_holdout) ** 2, axis=1)
    rank_two_mse = np.mean((rank_two_forecast - observed_holdout) ** 2, axis=1)
    numerical_floor = max(1e-14, 1e-8 * noise_sd**2)
    score = np.log(
        (rank_one_mse + numerical_floor) / (rank_two_mse + numerical_floor)
    )
    return score, rank_one_mse, rank_two_mse


def run_study(
    config: ForecastStudyConfig,
) -> list[dict[str, str | int | float]]:
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, str | int | float]] = []
    endpoint_bounds = (config.endpoint_lower, config.endpoint_upper)

    for checkpoint_count in config.checkpoint_counts:
        holdout_count = max(
            config.minimum_holdout,
            int(round(config.holdout_fraction * checkpoint_count)),
        )
        train_count = checkpoint_count - holdout_count
        if train_count < 6:
            raise ValueError("each condition needs at least six training checkpoints")

        rank_one_truth = modal_residual(
            checkpoint_count, (config.fast_mode,), (1.0,)
        )
        rank_two_truth = modal_residual(
            checkpoint_count,
            (config.fast_mode, config.slow_mode),
            (config.fast_weight, 1.0 - config.fast_weight),
        )
        train_checkpoints = np.arange(train_count, dtype=float)
        holdout_checkpoints = np.arange(
            train_count, checkpoint_count, dtype=float
        )

        for noise_sd in config.noise_levels:
            calibration_null = simulate_observations(
                rank_one_truth, noise_sd, config.replicates, rng
            )
            test_null = simulate_observations(
                rank_one_truth, noise_sd, config.replicates, rng
            )
            alternative = simulate_observations(
                rank_two_truth, noise_sd, config.replicates, rng
            )
            combined = np.concatenate(
                (calibration_null, test_null, alternative), axis=0
            )
            combined_train = combined[:, :train_count]
            combined_holdout = combined[:, train_count:]

            rank_one_fit = fit_modal_batch(
                combined_train,
                train_checkpoints,
                holdout_checkpoints,
                1,
                mode_grid=DEFAULT_MODE_GRID,
                endpoint_bounds=endpoint_bounds,
                direction="decreasing",
            )
            rank_two_fit = fit_modal_batch(
                combined_train,
                train_checkpoints,
                holdout_checkpoints,
                2,
                mode_grid=DEFAULT_MODE_GRID,
                endpoint_bounds=endpoint_bounds,
                direction="decreasing",
            )
            scores, rank_one_mse, rank_two_mse = forecast_score(
                rank_one_fit.forecast,
                rank_two_fit.forecast,
                combined_holdout,
                noise_sd,
            )
            calibration_slice = slice(0, config.replicates)
            null_slice = slice(config.replicates, 2 * config.replicates)
            alternative_slice = slice(2 * config.replicates, 3 * config.replicates)
            threshold = float(
                np.quantile(
                    scores[calibration_slice],
                    1.0 - config.false_positive_level,
                    method="higher",
                )
            )
            false_positive_rate = float(np.mean(scores[null_slice] > threshold))
            detection_power = float(
                np.mean(scores[alternative_slice] > threshold)
            )

            alternative_rank_two_modes = rank_two_fit.modes[alternative_slice]
            fast_error = np.abs(
                alternative_rank_two_modes[:, 0] - config.fast_mode
            )
            slow_error = np.abs(
                alternative_rank_two_modes[:, 1] - config.slow_mode
            )
            mode_recovery_rate = float(
                np.mean((fast_error <= 0.02) & (slow_error <= 0.02))
            )
            alternative_endpoint_error = np.abs(
                rank_two_fit.endpoints[alternative_slice]
            )
            null_endpoint_error = np.abs(rank_one_fit.endpoints[null_slice])

            rows.append(
                {
                    "checkpoint_count": checkpoint_count,
                    "train_count": train_count,
                    "holdout_count": holdout_count,
                    "noise_sd": noise_sd,
                    "noise_percent_of_total_response": 100.0 * noise_sd,
                    "replicates": config.replicates,
                    "score_threshold": threshold,
                    "false_positive_rate": false_positive_rate,
                    "detection_power": detection_power,
                    "rank_two_mode_recovery_rate": mode_recovery_rate,
                    "median_alt_endpoint_abs_error": float(
                        np.median(alternative_endpoint_error)
                    ),
                    "median_null_endpoint_abs_error": float(
                        np.median(null_endpoint_error)
                    ),
                    "median_alt_rank1_forecast_mse": float(
                        np.median(rank_one_mse[alternative_slice])
                    ),
                    "median_alt_rank2_forecast_mse": float(
                        np.median(rank_two_mse[alternative_slice])
                    ),
                }
            )
    return rows


def minimum_checkpoint_rows(
    rows: list[dict[str, str | int | float]],
    config: ForecastStudyConfig,
) -> list[dict[str, str | int | float]]:
    output: list[dict[str, str | int | float]] = []
    for noise_sd in config.noise_levels:
        candidates = sorted(
            (
                row
                for row in rows
                if np.isclose(float(row["noise_sd"]), noise_sd)
            ),
            key=lambda row: int(row["checkpoint_count"]),
        )
        qualifying = [
            row
            for row in candidates
            if float(row["detection_power"]) >= config.target_power
        ]
        output.append(
            {
                "noise_sd": noise_sd,
                "noise_percent_of_total_response": 100.0 * noise_sd,
                "target_power": config.target_power,
                "minimum_checkpoint_count": (
                    int(qualifying[0]["checkpoint_count"]) if qualifying else ""
                ),
                "maximum_tested_checkpoint_count": max(
                    int(row["checkpoint_count"]) for row in candidates
                ),
            }
        )
    return output


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def result_matrix(
    rows: list[dict[str, str | int | float]],
    config: ForecastStudyConfig,
    field: str,
) -> np.ndarray:
    lookup = {
        (int(row["checkpoint_count"]), float(row["noise_sd"])): float(row[field])
        for row in rows
    }
    return np.asarray(
        [
            [
                lookup[(checkpoint_count, noise_sd)]
                for checkpoint_count in config.checkpoint_counts
            ]
            for noise_sd in config.noise_levels
        ],
        dtype=float,
    )


def annotate_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    formatter: str,
    dark_threshold: float,
) -> None:
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                format(value, formatter),
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if value < dark_threshold else "black",
            )


def make_figure(
    output_stem: Path,
    rows: list[dict[str, str | int | float]],
    config: ForecastStudyConfig,
) -> None:
    power = result_matrix(rows, config, "detection_power")
    endpoint_error_percent = 100.0 * result_matrix(
        rows, config, "median_alt_endpoint_abs_error"
    )
    endpoint_scale_max = max(1.0, float(np.quantile(endpoint_error_percent, 0.95)))

    figure = plt.figure(figsize=(8.45, 3.7))
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 0.045, 1.0, 0.045),
        left=0.085,
        right=0.955,
        bottom=0.19,
        top=0.82,
        wspace=0.22,
    )
    power_axis = figure.add_subplot(grid[0, 0])
    power_colorbar_axis = figure.add_subplot(grid[0, 1])
    endpoint_axis = figure.add_subplot(grid[0, 2])
    endpoint_colorbar_axis = figure.add_subplot(grid[0, 3])

    power_image = power_axis.imshow(
        power, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis"
    )
    annotate_heatmap(
        power_axis, power, formatter=".2f", dark_threshold=0.48
    )
    endpoint_image = endpoint_axis.imshow(
        endpoint_error_percent,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=endpoint_scale_max,
        cmap="magma_r",
    )
    annotate_heatmap(
        endpoint_axis,
        endpoint_error_percent,
        formatter=".2f",
        dark_threshold=0.45 * endpoint_scale_max,
    )

    for axis in (power_axis, endpoint_axis):
        axis.set_xticks(
            np.arange(len(config.checkpoint_counts)),
            labels=[str(value) for value in config.checkpoint_counts],
        )
        axis.set_yticks(
            np.arange(len(config.noise_levels)),
            labels=[f"{100.0 * value:g}%" for value in config.noise_levels],
        )
        axis.set_xlabel("observed checkpoints")
    endpoint_axis.tick_params(axis="y", labelleft=False)
    power_axis.set_ylabel("noise SD as fraction of total response")
    power_axis.set_title("Held-out rank-two detection")
    endpoint_axis.set_title("Median endpoint error")

    power_colorbar = figure.colorbar(power_image, cax=power_colorbar_axis)
    power_colorbar.set_label("power")
    endpoint_colorbar = figure.colorbar(
        endpoint_image, cax=endpoint_colorbar_axis
    )
    endpoint_colorbar.set_label("absolute error (% of response)")

    figure.suptitle(
        "Joint endpoint estimation and held-out modal forecasting",
        fontsize=11,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Rank-two alternative: modes {config.fast_mode:g}, "
            f"{config.slow_mode:g}; slow weight {1.0 - config.fast_weight:g}; "
            f"{config.replicates} replicates; "
            f"{100.0 * config.false_positive_level:g}% calibrated false-positive rate."
        ),
        ha="center",
        fontsize=7.5,
    )
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-counts",
        default="10,12,15,20,30,40,60",
    )
    parser.add_argument(
        "--noise-levels",
        default="0.001,0.0025,0.005,0.01,0.02,0.05",
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-holdout", type=int, default=3)
    parser.add_argument("--false-positive-level", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--fast-mode", type=float, default=0.45)
    parser.add_argument("--slow-mode", type=float, default=0.94)
    parser.add_argument("--fast-weight", type=float, default=0.75)
    parser.add_argument("--endpoint-lower", type=float, default=-0.25)
    parser.add_argument("--endpoint-upper", type=float, default=0.25)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ForecastStudyConfig(
        checkpoint_counts=parse_number_list(args.checkpoint_counts, int),
        noise_levels=parse_number_list(args.noise_levels, float),
        replicates=args.replicates,
        holdout_fraction=args.holdout_fraction,
        minimum_holdout=args.minimum_holdout,
        false_positive_level=args.false_positive_level,
        target_power=args.target_power,
        seed=args.seed,
        fast_mode=args.fast_mode,
        slow_mode=args.slow_mode,
        fast_weight=args.fast_weight,
        endpoint_lower=args.endpoint_lower,
        endpoint_upper=args.endpoint_upper,
    )
    if config.replicates < 100:
        raise ValueError("use at least 100 replicates")
    if any(count < 9 for count in config.checkpoint_counts):
        raise ValueError("checkpoint counts must be at least nine")
    if any(noise <= 0.0 for noise in config.noise_levels):
        raise ValueError("noise levels must be positive")
    if not 0.0 < config.holdout_fraction < 0.5:
        raise ValueError("holdout fraction must lie in (0, 0.5)")
    if not 0.0 < config.fast_mode < config.slow_mode < 1.0:
        raise ValueError("modes must satisfy 0 < fast < slow < 1")
    if not 0.0 < config.fast_weight < 1.0:
        raise ValueError("fast weight must lie in (0, 1)")

    result_dir = args.project_root / "results"
    figure_dir = args.project_root / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = run_study(config)
    minimum_rows = minimum_checkpoint_rows(rows, config)
    write_csv(result_dir / "asymptote_forecast_grid.csv", rows)
    write_csv(
        result_dir / "asymptote_forecast_min_checkpoints.csv", minimum_rows
    )
    with (result_dir / "asymptote_forecast_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(config), handle, indent=2)
        handle.write("\n")
    make_figure(figure_dir / "fig3_asymptote_forecast", rows, config)

    print("Minimum checkpoints for held-out target power:")
    for row in minimum_rows:
        minimum = row["minimum_checkpoint_count"]
        display = minimum if minimum != "" else f">{row['maximum_tested_checkpoint_count']}"
        print(
            f"noise={float(row['noise_percent_of_total_response']):5.2f}% "
            f"checkpoints={display}"
        )


if __name__ == "__main__":
    main()
