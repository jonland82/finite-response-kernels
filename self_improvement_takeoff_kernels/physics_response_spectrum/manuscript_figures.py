"""Generate compact publication figures for the physics response-spectrum paper."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "runs" / "primary-eed-tool-20260719"
RHOS = (-6.0, -2.0, 0.0, 2.0, 6.0)
COLORS = {
    -6.0: "#3558b8",
    -2.0: "#79a8ef",
    0.0: "#777777",
    2.0: "#ef8a6c",
    6.0: "#b40426",
}
MARKERS = {-6.0: "o", -2.0: "s", 0.0: "D", 2.0: "^", 6.0: "P"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_se(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def primary_figure() -> None:
    rows = read_csv(PRIMARY / "aggregate.csv")
    terminal = {float(row["rho"]): row for row in read_csv(PRIMARY / "terminal_summary.csv")}

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.0, 3.0),
        gridspec_kw={"width_ratios": [1.45, 1.20, 1.0]},
        constrained_layout=True,
    )

    for rho in RHOS:
        series = sorted(
            [row for row in rows if float(row["rho"]) == rho],
            key=lambda row: int(row["round"]),
        )
        rounds = [int(row["round"]) for row in series]
        for axis, metric in ((axes[0], "mean_fitness"), (axes[1], "mean_accuracy")):
            means = [float(row[f"{metric}_mean"]) for row in series]
            ses = [float(row[f"{metric}_se"]) for row in series]
            axis.plot(
                rounds,
                means,
                marker="o",
                markersize=3.2,
                linewidth=1.45,
                color=COLORS[rho],
                label=fr"$\rho={rho:g}$",
            )
            axis.fill_between(
                rounds,
                [mean - se for mean, se in zip(means, ses)],
                [mean + se for mean, se in zip(means, ses)],
                color=COLORS[rho],
                alpha=0.12,
                linewidth=0,
            )

    axes[0].set_title("A. Continuous solution fitness")
    axes[0].set_ylabel("mean EED fitness")
    axes[0].set_ylim(-0.005, 0.245)
    axes[0].legend(frameon=False, ncol=1, loc="upper left")

    axes[1].set_title("B. Exact answers")
    axes[1].set_ylabel("population accuracy")
    axes[1].set_ylim(-0.005, 0.125)

    for axis in axes[:2]:
        axis.set_xlabel("recursive round")
        axis.set_xticks(range(7))
        axis.grid(alpha=0.22, linewidth=0.6)

    for rho in RHOS:
        series = sorted(
            [row for row in rows if float(row["rho"]) == rho],
            key=lambda row: int(row["round"]),
        )
        rounds = [int(row["round"]) for row in series]
        means = [float(row["effective_answers_mean"]) for row in series]
        ses = [float(row["effective_answers_se"]) for row in series]
        axes[2].plot(
            rounds,
            means,
            marker="o",
            markersize=3.2,
            linewidth=1.45,
            color=COLORS[rho],
        )
        axes[2].fill_between(
            rounds,
            [mean - se for mean, se in zip(means, ses)],
            [mean + se for mean, se in zip(means, ses)],
            color=COLORS[rho],
            alpha=0.12,
            linewidth=0,
        )
    axes[2].set_title("C. Population concentration")
    axes[2].set_xlabel("recursive round")
    axes[2].set_ylabel("effective answers (of four)")
    axes[2].set_xticks(range(7))
    axes[2].set_ylim(1.25, 3.9)
    axes[2].grid(alpha=0.22, linewidth=0.6)

    fig.savefig(ROOT / "fig1_physics_response_spectrum.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "fig1_physics_response_spectrum.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def numeric_trajectory(run_name: str) -> list[dict[str, float | int | str]]:
    rows = read_csv(ROOT / "runs" / run_name / "trajectory.csv")
    output = []
    for row in rows:
        output.append(
            {
                **row,
                "problem_id": int(row["problem_id"]),
                "replicate": int(row["replicate"]),
                "rho": float(row["rho"]),
                "round": int(row["round"]),
                "mean_fitness": float(row["mean_fitness"]),
                "mean_accuracy": float(row["mean_accuracy"]),
            }
        )
    return output


def context_figure() -> None:
    restart_name = "control-restart-20260719"
    restart_config = json.loads(
        (ROOT / "runs" / restart_name / "config.json").read_text(encoding="utf-8")
    )
    problem_ids = set(restart_config["problem_ids"])
    primary = numeric_trajectory("primary-eed-tool-20260719")
    recursive = [
        row
        for row in primary
        if row["problem_id"] in problem_ids
        and row["replicate"] == 0
        and row["rho"] in {-6.0, 0.0, 2.0}
    ]
    modes = {
        "recursive": recursive,
        "restart": numeric_trajectory(restart_name),
        "frozen": numeric_trajectory("control-frozen-20260719"),
    }
    colors = {"recursive": "#b40426", "restart": "#2166ac", "frozen": "#666666"}
    markers = {"recursive": "o", "restart": "s", "frozen": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.55), constrained_layout=True)
    for mode, rows in modes.items():
        x_values = []
        fitness_means = []
        fitness_se = []
        accuracy_means = []
        accuracy_se = []
        for rho in (-6.0, 0.0, 2.0):
            terminal_round = max(row["round"] for row in rows if row["rho"] == rho)
            selected = [
                row for row in rows if row["rho"] == rho and row["round"] == terminal_round
            ]
            x_values.append(rho)
            mean, se = mean_se([float(row["mean_fitness"]) for row in selected])
            fitness_means.append(mean)
            fitness_se.append(se)
            mean, se = mean_se([float(row["mean_accuracy"]) for row in selected])
            accuracy_means.append(mean)
            accuracy_se.append(se)

        for axis, means, errors in (
            (axes[0], fitness_means, fitness_se),
            (axes[1], accuracy_means, accuracy_se),
        ):
            axis.errorbar(
                x_values,
                means,
                yerr=errors,
                marker=markers[mode],
                markersize=4,
                linewidth=1.45,
                capsize=2.2,
                color=colors[mode],
                label=mode,
            )

    axes[0].set_title("A. Terminal EED fitness")
    axes[0].set_ylabel("mean over eight problems")
    axes[0].set_ylim(-0.02, 0.51)
    axes[1].set_title("B. Terminal exact accuracy")
    axes[1].set_ylabel("mean over eight problems")
    axes[1].set_ylim(-0.02, 0.34)
    for axis in axes:
        axis.set_xlabel(r"selection pressure $\rho$")
        axis.set_xticks([-6, 0, 2])
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper left")

    fig.savefig(ROOT / "fig2_context_controls.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "fig2_context_controls.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def takeoff_kernel_figure() -> None:
    rows = read_csv(PRIMARY / "aggregate.csv")
    terminal = {float(row["rho"]): row for row in read_csv(PRIMARY / "terminal_summary.csv")}
    selected_rhos = (-6.0, 0.0, 2.0)
    regime_names = {-6.0: "collapse", 0.0: "neutral churn", 2.0: "takeoff"}

    fig, axes_grid = plt.subplots(2, 2, figsize=(7.5, 4.0), constrained_layout=True)
    axes = axes_grid.ravel()
    for index, (axis, rho) in enumerate(zip(axes[:3], selected_rhos)):
        series = sorted(
            [row for row in rows if float(row["rho"]) == rho],
            key=lambda row: int(row["round"]),
        )
        values = [float(row["mean_fitness_mean"]) for row in series]
        deltas = [right - left for left, right in zip(values, values[1:])]
        gain = float(terminal[rho]["terminal_gain"])
        kernel = [delta / abs(gain) for delta in deltas]
        color = COLORS[rho]
        axis.bar(range(1, 7), kernel, color=color, alpha=0.88, width=0.72)
        axis.axhline(0, color="#333333", linewidth=0.7)
        panel = "ABC"[index]
        axis.set_title(
            fr"{panel}. {regime_names[rho]}: $\rho={rho:g}$, $g={gain:+.3f}$",
            loc="left",
        )
        axis.set_xlabel("round $r$")
        axis.set_xticks(range(1, 7))
        axis.set_ylim(-0.72, 1.22)
        axis.grid(axis="y", alpha=0.20, linewidth=0.6)
    axes[0].set_ylabel(r"signed kernel $\kappa_r$")
    axes[2].set_ylabel(r"signed kernel $\kappa_r$")

    geometry = axes[3]
    limit = 0.160
    x_values = [-limit + 2 * limit * index / 300 for index in range(301)]
    walls = [abs(value) for value in x_values]
    geometry.fill_between(
        x_values,
        walls,
        [limit] * len(x_values),
        color="#e8edf4",
        alpha=0.72,
        zorder=0,
    )
    geometry.plot([-limit, 0], [limit, 0], color="#3558b8", linewidth=2.0)
    geometry.plot([0, limit], [0, limit], color="#b40426", linewidth=2.0)
    geometry.axvline(0, color="#666666", linestyle=":", linewidth=1.2)
    for rho in RHOS:
        row = terminal[rho]
        gain = float(row["terminal_gain"])
        activity = float(row["total_response"])
        geometry.scatter(
            gain,
            activity,
            s=60,
            marker=MARKERS[rho],
            color=COLORS[rho],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[rho],
            linestyle="none",
            markerfacecolor=COLORS[rho],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=6.8,
            label=fr"${rho:g}$",
        )
        for rho in RHOS
    ]
    geometry.legend(
        handles=legend_handles,
        title=r"selection pressure $\rho$",
        loc="upper left",
        bbox_to_anchor=(0.025, 0.975),
        ncol=3,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#c9d2dc",
        fontsize=9.2,
        title_fontsize=9.2,
        columnspacing=0.7,
        handletextpad=0.35,
        labelspacing=0.3,
        borderpad=0.35,
    )
    geometry.set_title("D. Full response cone", loc="left")
    geometry.set_xlabel(r"signed displacement $g$")
    geometry.set_ylabel(r"total response $A$")
    geometry.set_xlim(-limit, limit)
    geometry.set_ylim(-0.005, limit)
    geometry.set_xticks([-0.10, 0, 0.10])
    geometry.set_yticks([0, 0.05, 0.10, 0.15])
    geometry.grid(alpha=0.16, linewidth=0.6)

    fig.savefig(ROOT / "fig3_takeoff_kernels.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "fig3_takeoff_kernels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure()
    primary_figure()
    context_figure()
    takeoff_kernel_figure()


if __name__ == "__main__":
    main()
