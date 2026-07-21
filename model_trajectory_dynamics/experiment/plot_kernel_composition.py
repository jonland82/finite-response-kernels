#!/usr/bin/env python3
"""Plot the fitted two-component LLM response and its component kernels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


INK = "#111827"
GRID = "#d6dce5"
MUTE = "#536070"
BLUE = "#2563ad"
BLUE_HI = "#4f8edb"
TEAL = "#0f8a83"
AMBER = "#c26a00"
WHITE = "#ffffff"

mpl.rcParams.update(
    {
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "grid.color": GRID,
        "font.family": "DejaVu Sans",
        "font.size": 14.0,
        "axes.titlesize": 15.0,
        "axes.labelsize": 13.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 10.5,
        "axes.linewidth": 1.0,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def logistic_cdf(times: np.ndarray, center: float, width: float) -> np.ndarray:
    argument = np.clip((times - center) / width, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-argument))


def logistic_pdf(times: np.ndarray, center: float, width: float) -> np.ndarray:
    cdf = logistic_cdf(times, center, width)
    return cdf * (1.0 - cdf) / width


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)


def mixture_for_run(
    transition_rows: list[dict[str, str]], run_id: str
) -> tuple[dict[str, str], dict[str, str], list[dict[str, float]]]:
    rows = [row for row in transition_rows if row["run_id"] == run_id]
    mixture = next(row for row in rows if row["model"] == "mixture")
    single = next(row for row in rows if row["model"] == "logistic")
    mixing = float(mixture["mixing"])
    components = [
        {
            "center": float(mixture["tau1"]),
            "width": float(mixture["width1"]),
            "mass": mixing,
        },
        {
            "center": float(mixture["tau2"]),
            "width": float(mixture["width2"]),
            "mass": 1.0 - mixing,
        },
    ]
    components.sort(key=lambda component: component["center"])
    return mixture, single, components


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"{stem}.{extension}", dpi=240, bbox_inches="tight"
        )


def representative_figure(
    results_dir: Path,
    transition_rows: list[dict[str, str]],
    run_id: str,
) -> None:
    trajectory = read_csv(results_dir / "raw" / run_id / "trajectory.csv")
    observed_times = np.asarray([float(row["step"]) for row in trajectory])
    observed_accuracy = np.asarray([float(row["test_accuracy"]) for row in trajectory])
    mixture, single, components = mixture_for_run(transition_rows, run_id)
    early, late = components
    dense_times = np.linspace(observed_times[0], observed_times[-1], 2400)

    early_profile = early["mass"] * logistic_cdf(
        dense_times, early["center"], early["width"]
    )
    late_profile = late["mass"] * logistic_cdf(
        dense_times, late["center"], late["width"]
    )
    total_profile = early_profile + late_profile
    early_kernel = early["mass"] * logistic_pdf(
        dense_times, early["center"], early["width"]
    )
    late_kernel = late["mass"] * logistic_pdf(
        dense_times, late["center"], late["width"]
    )
    total_kernel = early_kernel + late_kernel
    p0, p1 = float(mixture["p0"]), float(mixture["p1"])
    fitted_accuracy = p0 + (p1 - p0) * total_profile
    single_accuracy = float(single["p0"]) + (
        float(single["p1"]) - float(single["p0"])
    ) * logistic_cdf(dense_times, float(single["tau"]), float(single["width"]))

    # These panels are stacked deliberately.  In the manuscript a two-column
    # grid made every axis less than three inches wide and reduced otherwise
    # reasonable source fonts to roughly six-point type on the printed page.
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.6, 8.0),
        constrained_layout=True,
    )
    for axis in axes:
        style_axis(axis)

    axis = axes[0]
    axis.scatter(
        observed_times,
        observed_accuracy,
        s=9,
        color=MUTE,
        alpha=0.48,
        linewidths=0,
        label="held-out accuracy",
        zorder=3,
    )
    axis.plot(dense_times, single_accuracy, color=AMBER, linewidth=1.8, linestyle=(0, (5, 2)), label="one response")
    axis.plot(dense_times, fitted_accuracy, color=INK, linewidth=2.5, label="two-response sum")
    axis.set_title(
        "(a) Observed trajectory and fitted responses",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("optimizer update")
    axis.set_ylabel("held-out accuracy")
    axis.set_ylim(-0.02, 1.04)
    axis.legend(loc="upper left", frameon=True, facecolor=WHITE, edgecolor=GRID)

    axis = axes[1]
    axis.fill_between(dense_times, 0, early_profile, color=TEAL, alpha=0.27)
    axis.fill_between(dense_times, early_profile, total_profile, color=BLUE_HI, alpha=0.30)
    axis.plot(dense_times, early_profile, color=TEAL, linewidth=2.0, label=fr"early contribution ($\pi={early['mass']:.2f}$)")
    axis.plot(dense_times, late_profile, color=BLUE_HI, linewidth=2.0, label=fr"late contribution ($1-\pi={late['mass']:.2f}$)")
    axis.plot(dense_times, total_profile, color=INK, linewidth=2.4, label=r"$F=\pi F_1+(1-\pi)F_2$")
    axis.set_title(
        "(b) Cumulative components add to the trajectory",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("optimizer update")
    axis.set_ylabel("normalized delivered gain")
    axis.set_ylim(-0.02, 1.04)
    axis.legend(loc="upper left", frameon=True, facecolor=WHITE, edgecolor=GRID)

    axis = axes[2]
    axis.fill_between(dense_times, 0, early_kernel, color=TEAL, alpha=0.22)
    axis.fill_between(dense_times, 0, late_kernel, color=BLUE_HI, alpha=0.23)
    axis.plot(dense_times, early_kernel, color=TEAL, linewidth=2.0, label=r"$\pi\kappa_1$")
    axis.plot(dense_times, late_kernel, color=BLUE_HI, linewidth=2.0, label=r"$(1-\pi)\kappa_2$")
    axis.plot(dense_times, total_kernel, color=INK, linewidth=2.5, label=r"$\kappa=\pi\kappa_1+(1-\pi)\kappa_2$")
    axis.set_title(
        "(c) Component kernels add to the takeoff kernel",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("optimizer update")
    axis.set_ylabel("response mass per update")
    axis.legend(loc="upper left", frameon=True, facecolor=WHITE, edgecolor=GRID)

    save_figure(figure, results_dir, "kernel_composition_representative")
    plt.close(figure)


def all_seed_figure(results_dir: Path, transition_rows: list[dict[str, str]]) -> None:
    run_ids = sorted(
        {row["run_id"] for row in transition_rows if row["run_id"].startswith("confirm-")}
    )
    figure, axes = plt.subplots(3, 2, figsize=(9.0, 9.5), sharey=True)
    figure.suptitle(
        "Two-component finite-response fits across six seeds",
        x=0.075,
        y=0.995,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    for axis, run_id in zip(axes.flat, run_ids, strict=True):
        style_axis(axis)
        trajectory = read_csv(results_dir / "raw" / run_id / "trajectory.csv")
        times = np.asarray([float(row["step"]) for row in trajectory])
        accuracy = np.asarray([float(row["test_accuracy"]) for row in trajectory])
        mixture, _, components = mixture_for_run(transition_rows, run_id)
        early, late = components
        dense = np.linspace(times[0], times[-1], 1800)
        gain = float(mixture["p1"]) - float(mixture["p0"])
        baseline = float(mixture["p0"])
        early_gain = gain * early["mass"] * logistic_cdf(dense, early["center"], early["width"])
        late_gain = gain * late["mass"] * logistic_cdf(dense, late["center"], late["width"])
        total = baseline + early_gain + late_gain
        axis.scatter(times, accuracy, s=5, color=MUTE, alpha=0.35, linewidths=0)
        axis.plot(dense, baseline + early_gain, color=TEAL, linewidth=1.7, label="baseline + early")
        axis.plot(dense, baseline + late_gain, color=BLUE_HI, linewidth=1.7, label="baseline + late")
        axis.plot(dense, total, color=INK, linewidth=2.2, label="sum")
        axis.set_title(f"seed {run_id.replace('confirm-seed', '')}", loc="left", fontweight="bold")
        axis.set_xlabel("optimizer update")
        axis.set_ylabel("held-out accuracy")
        axis.set_ylim(-0.02, 1.04)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=True, facecolor=WHITE, edgecolor=GRID)
    figure.tight_layout(rect=(0, 0.055, 1, 0.96), pad=1.1)
    save_figure(figure, results_dir, "kernel_composition_all_seeds")
    plt.close(figure)


def overview_figure(results_dir: Path, transition_rows: list[dict[str, str]]) -> None:
    """Summarize the raw trajectories, resolved timing, and predictive comparison."""
    run_summary = {
        row["run_id"]: row
        for row in read_csv(results_dir / "run_summary.csv")
        if row["stage"] == "confirmation"
    }
    thresholds = {
        row["run_id"]: row for row in read_csv(results_dir / "transition_thresholds.csv")
    }
    run_ids = sorted(run_summary)
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 0.88, len(run_ids)))

    # A vertical layout keeps every panel at the full manuscript width.  It is
    # taller on screen but substantially clearer at journal-page scale.
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.6, 7.35),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.15, 1.1, 1.0)},
    )
    for axis in axes:
        style_axis(axis)

    axis = axes[0]
    for color, run_id in zip(colors, run_ids, strict=True):
        trajectory = read_csv(results_dir / "raw" / run_id / "trajectory.csv")
        times = np.asarray([float(row["step"]) for row in trajectory])
        accuracy = np.asarray([float(row["test_accuracy"]) for row in trajectory])
        axis.plot(times, accuracy, color=color, linewidth=1.45, alpha=0.9, label=run_id[-8:])
    axis.set_title("(a) Six held-out trajectories", loc="left", fontweight="bold")
    axis.set_xlabel("optimizer update")
    axis.set_ylabel("held-out accuracy")
    axis.set_ylim(-0.02, 1.04)

    axis = axes[1]
    for row_index, (color, run_id) in enumerate(zip(colors, run_ids, strict=True)):
        train_step = float(run_summary[run_id]["train_fit_step"])
        t25 = float(thresholds[run_id]["first_accuracy_25_step"])
        t50 = float(thresholds[run_id]["first_accuracy_50_step"])
        t90 = float(thresholds[run_id]["first_accuracy_90_step"])
        axis.plot([train_step, t90], [row_index, row_index], color=color, alpha=0.42, linewidth=2.2)
        axis.scatter(train_step, row_index, marker="s", s=35, color=INK, zorder=4)
        axis.scatter(t25, row_index, marker="o", s=38, color=TEAL, edgecolor=WHITE, linewidth=0.5, zorder=4)
        axis.scatter(t50, row_index, marker="D", s=38, color=BLUE, edgecolor=WHITE, linewidth=0.5, zorder=4)
        axis.scatter(t90, row_index, marker="*", s=75, color=AMBER, edgecolor=WHITE, linewidth=0.5, zorder=4)
    axis.set_yticks(range(len(run_ids)), [run_id[-8:] for run_id in run_ids])
    axis.invert_yaxis()
    axis.set_xlabel("optimizer update")
    axis.set_ylabel("seed")
    axis.set_title("(b) Memorization comes first", loc="left", fontweight="bold")
    axis.scatter([], [], marker="s", s=35, color=INK, label="train 99%")
    axis.scatter([], [], marker="o", s=38, color=TEAL, label="test 25%")
    axis.scatter([], [], marker="D", s=38, color=BLUE, label="test 50%")
    axis.scatter([], [], marker="*", s=75, color=AMBER, label="test 90%")
    axis.legend(
        loc="upper center",
        ncol=4,
        frameon=True,
        facecolor=WHITE,
        edgecolor=GRID,
    )

    axis = axes[2]
    model_names = ["delta", "logistic", "mixture"]
    model_labels = ["step", "one\nresponse", "two\nresponses"]
    by_run = {
        run_id: {
            row["model"]: float(row["heldout_checkpoint_rmse"])
            for row in transition_rows
            if row["run_id"] == run_id
        }
        for run_id in run_ids
    }
    values = np.asarray([[by_run[run_id][model] for model in model_names] for run_id in run_ids])
    for color, row in zip(colors, values, strict=True):
        axis.plot(range(3), row, color=color, linewidth=1.35, alpha=0.7)
        axis.scatter(range(3), row, color=color, s=24, edgecolor=WHITE, linewidth=0.4, zorder=3)
    medians = np.median(values, axis=0)
    axis.plot(range(3), medians, color=INK, linewidth=2.7, marker="o", markersize=5.5, label="median")
    axis.set_xticks(range(3), model_labels)
    axis.set_ylabel("held-out checkpoint RMSE")
    axis.set_title("(c) Two responses predict best", loc="left", fontweight="bold")
    axis.set_ylim(0, 0.145)
    axis.legend(loc="upper right", frameon=True, facecolor=WHITE, edgecolor=GRID)

    save_figure(figure, results_dir, "trajectory_results_overview")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--representative", default="confirm-seed20260803")
    arguments = parser.parse_args()
    results_dir = arguments.results_dir.resolve()
    transition_rows = read_csv(results_dir / "transition_model_comparison.csv")
    overview_figure(results_dir, transition_rows)
    representative_figure(results_dir, transition_rows, arguments.representative)
    all_seed_figure(results_dir, transition_rows)
    print(results_dir / "kernel_composition_representative.pdf")
    print(results_dir / "kernel_composition_all_seeds.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
