"""Generate the first synthetic modal-takeoff experiment.

The experiment instantiates Corollary 4 of the working manuscript: two
trajectories have the same initial value and limit, but one is rank one while
the other contains a lightly weighted slow mode.  The script writes a figure,
the plotted trajectories, and a compact summary table.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ModalTrajectory:
    name: str
    modes: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.modes) != len(self.weights):
            raise ValueError("modes and weights must have the same length")
        if not self.modes:
            raise ValueError("at least one mode is required")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("weights must sum to one")
        if any(weight <= 0.0 for weight in self.weights):
            raise ValueError("this monotone experiment requires positive weights")
        if any(not 0.0 < mode < 1.0 for mode in self.modes):
            raise ValueError("this monotone experiment requires modes in (0, 1)")

    def residual(self, checkpoints: np.ndarray) -> np.ndarray:
        modes = np.asarray(self.modes, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        return np.sum(
            weights[:, None] * modes[:, None] ** checkpoints[None, :], axis=0
        )

    def profile(self, checkpoints: np.ndarray) -> np.ndarray:
        return 1.0 - self.residual(checkpoints)

    def kernel(self, checkpoints: np.ndarray) -> np.ndarray:
        profile = self.profile(checkpoints)
        return np.diff(profile)

    @property
    def modal_radius(self) -> float:
        return max(self.modes)


def settling_time(residual: np.ndarray, epsilon: float) -> int:
    """Return the first index after which the residual stays below epsilon."""
    above = np.flatnonzero(np.abs(residual) > epsilon)
    return 0 if above.size == 0 else int(above[-1] + 1)


def hankel_matrix(kernel: np.ndarray, size: int) -> np.ndarray:
    """Build H_ij = kappa_(i+j+1) from a zero-based kernel array."""
    if kernel.size < 2 * size - 1:
        raise ValueError("not enough kernel values for requested Hankel size")
    return np.fromfunction(
        lambda i, j: kernel[(i + j).astype(int)], (size, size), dtype=int
    )


def numerical_rank(matrix: np.ndarray, relative_tolerance: float = 1e-10) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = relative_tolerance * singular_values[0]
    return int(np.count_nonzero(singular_values > threshold))


def write_trajectories(
    output_path: Path,
    checkpoints: np.ndarray,
    trajectories: tuple[ModalTrajectory, ...],
) -> None:
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for trajectory in trajectories:
        residual = trajectory.residual(checkpoints)
        profile = 1.0 - residual
        kernel = np.concatenate(([np.nan], np.diff(profile)))
        series[trajectory.name] = (residual, profile, kernel)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["checkpoint"]
        for trajectory in trajectories:
            header.extend(
                [
                    f"{trajectory.name}_residual",
                    f"{trajectory.name}_profile",
                    f"{trajectory.name}_kernel",
                ]
            )
        writer.writerow(header)
        for index, checkpoint in enumerate(checkpoints):
            row: list[float | int | str] = [int(checkpoint)]
            for trajectory in trajectories:
                residual, profile, kernel = series[trajectory.name]
                kernel_value: float | str = (
                    "" if np.isnan(kernel[index]) else float(kernel[index])
                )
                row.extend(
                    [float(residual[index]), float(profile[index]), kernel_value]
                )
            writer.writerow(row)


def write_summary(
    output_path: Path,
    checkpoints: np.ndarray,
    trajectories: tuple[ModalTrajectory, ...],
    epsilon: float,
    hankel_size: int,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for trajectory in trajectories:
        residual = trajectory.residual(checkpoints)
        kernel = trajectory.kernel(checkpoints)
        hankel = hankel_matrix(kernel, hankel_size)
        rows.append(
            {
                "case": trajectory.name,
                "modal_order": len(trajectory.modes),
                "modes": ";".join(f"{mode:.6g}" for mode in trajectory.modes),
                "weights": ";".join(
                    f"{weight:.6g}" for weight in trajectory.weights
                ),
                "modal_radius": trajectory.modal_radius,
                f"T_{epsilon:g}": settling_time(residual, epsilon),
                "hankel_rank": numerical_rank(hankel),
                "initial_profile": float(1.0 - residual[0]),
                "terminal_profile_at_horizon": float(1.0 - residual[-1]),
                "kernel_mass_at_horizon": float(np.sum(kernel)),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_figure(
    output_stem: Path,
    checkpoints: np.ndarray,
    trajectories: tuple[ModalTrajectory, ...],
    epsilon: float,
) -> None:
    colors = ("#1f5a94", "#c04b3c")
    linestyles = ("-", "--")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))

    for trajectory, color, linestyle in zip(
        trajectories, colors, linestyles, strict=True
    ):
        residual = trajectory.residual(checkpoints)
        profile = 1.0 - residual
        kernel = trajectory.kernel(checkpoints)
        settle = settling_time(residual, epsilon)
        label = (
            f"{trajectory.name.replace('_', ' ')} "
            f"(rank {len(trajectory.modes)}, $T_{{{epsilon:.2f}}}={settle}$)"
        )
        axes[0].plot(
            checkpoints,
            profile,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        axes[1].semilogy(
            checkpoints[1:],
            kernel,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=trajectory.name.replace("_", " "),
        )

    axes[0].axhline(
        1.0 - epsilon, color="#777777", linewidth=0.9, linestyle=":", zorder=0
    )
    axes[0].set(
        xlabel="checkpoint $n$",
        ylabel="realized response $F_n$",
        xlim=(0, int(checkpoints[-1])),
        ylim=(-0.015, 1.015),
        title="Same endpoints",
    )
    axes[0].legend(frameon=False, fontsize=7.4, loc="lower right")

    axes[1].set(
        xlabel="checkpoint $n$",
        ylabel="kernel mass $\\kappa_n$",
        xlim=(1, int(checkpoints[-1])),
        title="Different response kernels",
    )
    axes[1].legend(frameon=False, fontsize=7.4, loc="upper right")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(alpha=0.18, linewidth=0.6)

    figure.suptitle(
        "A small slow mode controls late self-improvement",
        fontsize=11,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--fast-mode", type=float, default=0.45)
    parser.add_argument("--slow-mode", type=float, default=0.94)
    parser.add_argument("--fast-weight", type=float, default=0.75)
    parser.add_argument("--hankel-size", type=int, default=8)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon < 2 * args.hankel_size:
        raise ValueError("horizon must be at least twice the Hankel size")
    if not 0.0 < args.epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1)")
    if not 0.0 < args.fast_weight < 1.0:
        raise ValueError("fast-weight must lie in (0, 1)")

    checkpoints = np.arange(args.horizon + 1, dtype=int)
    trajectories = (
        ModalTrajectory("single_mode", (args.fast_mode,), (1.0,)),
        ModalTrajectory(
            "two_mode",
            (args.fast_mode, args.slow_mode),
            (args.fast_weight, 1.0 - args.fast_weight),
        ),
    )

    figure_dir = args.project_root / "figures"
    result_dir = args.project_root / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    write_trajectories(
        result_dir / "endpoint_separation_trajectories.csv",
        checkpoints,
        trajectories,
    )
    rows = write_summary(
        result_dir / "endpoint_separation_summary.csv",
        checkpoints,
        trajectories,
        args.epsilon,
        args.hankel_size,
    )
    make_figure(
        figure_dir / "fig1_endpoint_separation",
        checkpoints,
        trajectories,
        args.epsilon,
    )

    for row in rows:
        print(
            f"{row['case']}: rank={row['hankel_rank']}, "
            f"modal_radius={row['modal_radius']:.3f}, "
            f"T={row[f'T_{args.epsilon:g}']}"
        )


if __name__ == "__main__":
    main()
