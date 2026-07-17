"""Monte Carlo power study for rank-one versus rank-two modal takeoff.

For each checkpoint count and observation-noise level, this script calibrates a
Hankel singular-value-ratio test under a rank-one trajectory and estimates its
power against the endpoint-separation rank-two trajectory.  Residual and kernel
Hankels are evaluated separately because differencing a trajectory to form its
kernel changes the noise geometry.

The baseline experiment assumes the limiting endpoint is known, so the
normalized residual is observed up to additive Gaussian noise.  This is an
optimistic identification benchmark; asymptote uncertainty is a later
sensitivity study.
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


@dataclass(frozen=True)
class StudyConfig:
    checkpoint_counts: tuple[int, ...]
    noise_levels: tuple[float, ...]
    replicates: int
    false_positive_level: float
    target_power: float
    seed: int
    fast_mode: float
    slow_mode: float
    fast_weight: float


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


def batch_hankel_statistic(
    observed_residuals: np.ndarray,
    hankel_size: int,
    representation: str,
) -> np.ndarray:
    if representation == "residual":
        series = observed_residuals
    elif representation == "kernel":
        series = observed_residuals[:, :-1] - observed_residuals[:, 1:]
    else:
        raise ValueError(f"unknown representation: {representation}")

    indices = np.add.outer(
        np.arange(hankel_size, dtype=int), np.arange(hankel_size, dtype=int)
    )
    if int(indices.max()) >= series.shape[1]:
        raise ValueError("series is too short for the requested Hankel matrix")
    hankel_batch = series[:, indices]
    singular_values = np.linalg.svd(hankel_batch, compute_uv=False)
    denominator = np.maximum(singular_values[:, 0], np.finfo(float).tiny)
    return singular_values[:, 1] / denominator


def calibrated_test(
    calibration_null: np.ndarray,
    test_null: np.ndarray,
    alternative: np.ndarray,
    false_positive_level: float,
) -> tuple[float, float, float]:
    threshold = float(
        np.quantile(
            calibration_null,
            1.0 - false_positive_level,
            method="higher",
        )
    )
    false_positive_rate = float(np.mean(test_null > threshold))
    detection_power = float(np.mean(alternative > threshold))
    return threshold, false_positive_rate, detection_power


def run_study(config: StudyConfig) -> list[dict[str, str | int | float]]:
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, str | int | float]] = []

    for checkpoint_count in config.checkpoint_counts:
        hankel_size = checkpoint_count // 2
        if hankel_size < 2:
            raise ValueError("each condition needs at least four checkpoints")

        rank_one = modal_residual(
            checkpoint_count, (config.fast_mode,), (1.0,)
        )
        rank_two = modal_residual(
            checkpoint_count,
            (config.fast_mode, config.slow_mode),
            (config.fast_weight, 1.0 - config.fast_weight),
        )

        for noise_sd in config.noise_levels:
            calibration_data = simulate_observations(
                rank_one, noise_sd, config.replicates, rng
            )
            test_null_data = simulate_observations(
                rank_one, noise_sd, config.replicates, rng
            )
            alternative_data = simulate_observations(
                rank_two, noise_sd, config.replicates, rng
            )

            for representation in ("residual", "kernel"):
                calibration_statistic = batch_hankel_statistic(
                    calibration_data, hankel_size, representation
                )
                test_null_statistic = batch_hankel_statistic(
                    test_null_data, hankel_size, representation
                )
                alternative_statistic = batch_hankel_statistic(
                    alternative_data, hankel_size, representation
                )
                threshold, false_positive_rate, detection_power = calibrated_test(
                    calibration_statistic,
                    test_null_statistic,
                    alternative_statistic,
                    config.false_positive_level,
                )
                rows.append(
                    {
                        "representation": representation,
                        "checkpoint_count": checkpoint_count,
                        "hankel_size": hankel_size,
                        "noise_sd": noise_sd,
                        "noise_percent_of_total_response": 100.0 * noise_sd,
                        "replicates": config.replicates,
                        "null_threshold_s2_over_s1": threshold,
                        "false_positive_rate": false_positive_rate,
                        "detection_power": detection_power,
                    }
                )
    return rows


def minimum_checkpoint_rows(
    study_rows: list[dict[str, str | int | float]],
    config: StudyConfig,
) -> list[dict[str, str | int | float]]:
    output: list[dict[str, str | int | float]] = []
    for representation in ("residual", "kernel"):
        for noise_sd in config.noise_levels:
            candidates = sorted(
                (
                    row
                    for row in study_rows
                    if row["representation"] == representation
                    and np.isclose(float(row["noise_sd"]), noise_sd)
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
                    "representation": representation,
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


def power_matrix(
    rows: list[dict[str, str | int | float]],
    representation: str,
    config: StudyConfig,
) -> np.ndarray:
    lookup = {
        (int(row["checkpoint_count"]), float(row["noise_sd"])): float(
            row["detection_power"]
        )
        for row in rows
        if row["representation"] == representation
    }
    return np.asarray(
        [
            [lookup[(checkpoint_count, noise_sd)] for checkpoint_count in config.checkpoint_counts]
            for noise_sd in config.noise_levels
        ],
        dtype=float,
    )


def make_figure(
    output_stem: Path,
    rows: list[dict[str, str | int | float]],
    config: StudyConfig,
) -> None:
    figure = plt.figure(figsize=(8.25, 3.65))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        left=0.09,
        right=0.94,
        bottom=0.19,
        top=0.82,
        wspace=0.18,
    )
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    )
    colorbar_axis = figure.add_subplot(grid[0, 2])
    image = None
    for axis, representation, title in zip(
        axes,
        ("residual", "kernel"),
        ("Residual Hankel", "Kernel Hankel"),
        strict=True,
    ):
        matrix = power_matrix(rows, representation, config)
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                text_color = "white" if value < 0.48 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=text_color,
                )
        axis.set_xticks(
            np.arange(len(config.checkpoint_counts)),
            labels=[str(value) for value in config.checkpoint_counts],
        )
        axis.set_yticks(
            np.arange(len(config.noise_levels)),
            labels=[f"{100.0 * value:g}%" for value in config.noise_levels],
        )
        axis.set_xlabel("observed checkpoints")
        axis.set_title(title)

    axes[0].set_ylabel("noise SD as fraction of total response")
    axes[1].tick_params(axis="y", labelleft=False)
    if image is None:
        raise RuntimeError("power figure was not populated")
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("power to reject rank one")
    figure.suptitle(
        "Rank-two detection at a calibrated 5% false-positive rate",
        fontsize=11,
    )
    figure.text(
        0.5,
        0.005,
        (
            f"Alternative: modes {config.fast_mode:g} and {config.slow_mode:g}; "
            f"slow-mode weight {1.0 - config.fast_weight:g}; "
            f"{config.replicates} Monte Carlo replicates per calibration/test sample."
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
        default="8,10,12,15,20,30,40,60",
        help="comma-separated counts including the baseline checkpoint",
    )
    parser.add_argument(
        "--noise-levels",
        default="0,0.001,0.0025,0.005,0.01,0.02,0.05",
        help="comma-separated Gaussian SDs on the normalized residual scale",
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--false-positive-level", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--fast-mode", type=float, default=0.45)
    parser.add_argument("--slow-mode", type=float, default=0.94)
    parser.add_argument("--fast-weight", type=float, default=0.75)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StudyConfig(
        checkpoint_counts=parse_number_list(args.checkpoint_counts, int),
        noise_levels=parse_number_list(args.noise_levels, float),
        replicates=args.replicates,
        false_positive_level=args.false_positive_level,
        target_power=args.target_power,
        seed=args.seed,
        fast_mode=args.fast_mode,
        slow_mode=args.slow_mode,
        fast_weight=args.fast_weight,
    )
    if any(count < 4 for count in config.checkpoint_counts):
        raise ValueError("checkpoint counts must be at least four")
    if any(noise < 0.0 for noise in config.noise_levels):
        raise ValueError("noise levels must be nonnegative")
    if config.replicates < 100:
        raise ValueError("use at least 100 replicates for calibration")
    if not 0.0 < config.false_positive_level < 1.0:
        raise ValueError("false-positive level must lie in (0, 1)")
    if not 0.0 < config.target_power < 1.0:
        raise ValueError("target power must lie in (0, 1)")
    if not 0.0 < config.fast_mode < config.slow_mode < 1.0:
        raise ValueError("modes must satisfy 0 < fast < slow < 1")
    if not 0.0 < config.fast_weight < 1.0:
        raise ValueError("fast weight must lie in (0, 1)")

    result_dir = args.project_root / "results"
    figure_dir = args.project_root / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    study_rows = run_study(config)
    minimum_rows = minimum_checkpoint_rows(study_rows, config)
    write_csv(result_dir / "modal_power_grid.csv", study_rows)
    write_csv(result_dir / "modal_power_min_checkpoints.csv", minimum_rows)
    with (result_dir / "modal_power_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(config), handle, indent=2)
        handle.write("\n")
    make_figure(figure_dir / "fig2_modal_power", study_rows, config)

    print("Minimum checkpoints for target power:")
    for row in minimum_rows:
        minimum = row["minimum_checkpoint_count"]
        display = minimum if minimum != "" else f">{row['maximum_tested_checkpoint_count']}"
        print(
            f"{row['representation']:8s} "
            f"noise={float(row['noise_percent_of_total_response']):5.2f}% "
            f"checkpoints={display}"
        )


if __name__ == "__main__":
    main()
