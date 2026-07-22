"""Generate publication figures for the blinded inverse takeoff note.

The shape figures use representative schemas from the experiment generator.
The performance figure is computed directly from the completed AWS summaries.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_experiment import tree_profile


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
BROAD_SUMMARY = (
    ROOT
    / "runs"
    / "inverse-takeoff-20260722T203012Z"
    / "results"
    / "summary.csv"
)
HORIZON_SUMMARY = (
    ROOT
    / "runs"
    / "inverse-takeoff-20260722T205043Z"
    / "results"
    / "summary.csv"
)

INK = "#16212B"
NAVY = "#17324D"
TEAL = "#0B7A75"
GOLD = "#D28F2C"
CORAL = "#C65D43"
PURPLE = "#6554C0"
MUTED = "#667085"
GRID = "#D8E2E3"
WASH = "#F1F6F6"
COLORS = [TEAL, GOLD, CORAL, PURPLE]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.titleweight": "bold",
        "axes.titlecolor": NAVY,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def kernel_atlas() -> None:
    representatives = [
        ("Monotone", "immediate", {1: 2}),
        ("Alternating", "finite horizon", {4: 10, 5: 2, 6: 20}),
        ("Damped ringing", "random split", {3: 5, 4: 1, 5: 7, 6: 2, 7: 5, 8: 5, 9: 2}),
        ("Delayed echo", "slow / near-periodic", {9: 511, 10: 2}),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 3.55), sharex="col")
    fig.subplots_adjust(wspace=0.31, hspace=0.22)

    for column, ((shape, family, schema), color) in enumerate(zip(representatives, COLORS)):
        f_values = tree_profile(schema, 34)
        kernel = np.diff(np.r_[0.0, f_values])
        indices = np.arange(len(f_values))
        top = axes[0, column]
        bottom = axes[1, column]

        top.plot(indices, f_values, color=color, linewidth=1.55)
        top.scatter(indices, f_values, color=color, s=5, zorder=3)
        top.axhline(1.0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.75)
        top.set_title(shape, loc="left", pad=10)
        top.text(
            0.0,
            1.015,
            f"{family}  |  max lag {max(schema)}",
            transform=top.transAxes,
            color=MUTED,
            fontsize=6.5,
            va="bottom",
        )
        top.grid(axis="y")
        top.set_xlim(-0.5, 34.5)
        top.margins(y=0.12)

        bottom.vlines(indices, 0.0, kernel, color=color, linewidth=1.0)
        bottom.scatter(indices, kernel, color=color, s=6, zorder=3)
        bottom.axhline(0.0, color=MUTED, linewidth=0.65)
        bottom.grid(axis="y")
        bottom.margins(y=0.13)
        bottom.set_xlabel("input step $n$")
        if column == 0:
            top.set_ylabel("takeoff $F_n$")
            bottom.set_ylabel("kernel $\\kappa_n$")

    axes[0, 0].annotate(
        "$F_\\infty=1$",
        xy=(25, 1),
        xytext=(19, 1.34),
        color=MUTED,
        fontsize=6.5,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6},
    )
    save(fig, "kernel_atlas")


def hidden_lag_collision() -> None:
    lag = 12
    parameters = [1, 500, 2000]
    labels = ["$t=1$", "$t=500$", "$t=2000$"]
    schemas = [
        {
            lag: t,
            lag + 1: (1 << (lag + 1)) - 3 * t,
            lag + 2: 2 * t,
        }
        for t in parameters
    ]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.15, 2.62))
    fig.subplots_adjust(wspace=0.34)

    x = np.arange(3)
    width = 0.23
    for offset, (schema, label, color) in enumerate(zip(schemas, labels, COLORS[:3])):
        dyadic_mass = [schema[j] * 2.0 ** (-j) for j in range(lag, lag + 3)]
        left.bar(x + (offset - 1) * width, dyadic_mass, width, color=color, label=label)
    left.set_xticks(x, [f"$a_{{{j}}}2^{{-{j}}}$" for j in range(lag, lag + 3)])
    left.set_ylim(0.0, 1.02)
    left.set_ylabel("share of fixed-speed identity")
    left.set_title("Different hidden branch stencils", loc="left")
    left.grid(axis="y")
    left.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.7)
    left.text(
        0.02,
        0.95,
        "$\\sum_j a_j2^{-j}=1$ in every case",
        transform=left.transAxes,
        color=MUTED,
        fontsize=6.6,
        va="top",
    )

    indices = np.arange(39)
    right.axvspan(-0.5, lag - 0.5, color=WASH, zorder=0)
    for schema, label, color in zip(schemas, labels, COLORS[:3]):
        values = tree_profile(schema, 38)
        right.plot(indices, values, color=color, linewidth=1.35, marker="o", markersize=2.2, label=label)
    right.axvline(lag - 0.5, color=NAVY, linestyle=(0, (3, 2)), linewidth=0.8)
    right.text(1.0, 12.15, "exactly identical prefix", color=MUTED, fontsize=6.7)
    right.annotate(
        "first reveal",
        xy=(lag - 0.5, 7.0),
        xytext=(lag + 2.0, 10.6),
        fontsize=6.7,
        color=NAVY,
        arrowprops={"arrowstyle": "->", "color": NAVY, "linewidth": 0.7},
    )
    right.set_xlim(-0.5, 38.5)
    right.set_ylim(-0.25, 13.6)
    right.set_xlabel("input step $n$")
    right.set_ylabel("takeoff $F_n$")
    right.set_title("Same observation, then different echoes", loc="left")
    right.grid(axis="y")
    save(fig, "hidden_lag_collision")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aws_recovery_results() -> None:
    horizon_rows = read_rows(HORIZON_SUMMARY)
    broad_rows = read_rows(BROAD_SUMMARY)

    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "correct": 0.0, "forecast": 0.0}
    )
    for row in horizon_rows:
        if row["family"] != "finite_horizon" or row["regime"] != "origin" or float(row["noise"]) != 0.0:
            continue
        count = float(row["count"])
        target = grouped[row["horizon_bin"]]
        target["count"] += count
        target["correct"] += count * float(row["class_accuracy"])
        target["forecast"] += count * float(row["forecast_success_rate"])

    horizon_order = ["lt_0.5", "0.5_to_1", "1_to_2", "ge_2"]
    horizon_labels = ["$<0.5$", "$0.5$–$1$", "$1$–$2$", "$\\geq2$"]
    class_accuracy = [100.0 * grouped[key]["correct"] / grouped[key]["count"] for key in horizon_order]
    forecast_success = [100.0 * grouped[key]["forecast"] / grouped[key]["count"] for key in horizon_order]

    noise_rows = [
        row
        for row in broad_rows
        if row["family"] == "random_split"
        and row["regime"] == "post_lag"
        and row["window"] == "96"
    ]
    noise_rows.sort(key=lambda row: float(row["noise"]))
    noise_labels = ["0" if float(row["noise"]) == 0.0 else f"$10^{{{int(np.log10(float(row['noise'])))}}}$" for row in noise_rows]
    noise_accuracy = [100.0 * float(row["class_accuracy"]) for row in noise_rows]
    noise_forecast = [100.0 * float(row["forecast_success_rate"]) for row in noise_rows]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.15, 2.55))
    fig.subplots_adjust(wspace=0.3)

    positions = np.arange(len(horizon_order))
    width = 0.34
    left.bar(positions - width / 2, class_accuracy, width, color=TEAL, label="modal class")
    left.bar(positions + width / 2, forecast_success, width, color=GOLD, label="forecast")
    left.axvline(1.5, color=NAVY, linestyle=(0, (3, 2)), linewidth=0.75)
    left.set_xticks(positions, horizon_labels)
    left.set_xlabel("observation ratio $T/L$")
    left.set_ylabel("success rate (%)")
    left.set_ylim(0, 18)
    left.set_title("Hidden-lag family stays hard", loc="left")
    left.grid(axis="y")
    left.legend(loc="upper left", fontsize=6.8)
    left.text(0.02, 0.02, "noiseless origin windows", transform=left.transAxes, color=MUTED, fontsize=6.5)

    noise_positions = np.arange(len(noise_rows))
    right.plot(noise_positions, noise_accuracy, color=TEAL, marker="o", linewidth=1.7, label="modal class")
    right.plot(noise_positions, noise_forecast, color=GOLD, marker="o", linewidth=1.7, label="forecast")
    right.fill_between(noise_positions, noise_accuracy, noise_forecast, color=GOLD, alpha=0.09)
    right.set_xticks(noise_positions, noise_labels)
    right.set_xlabel("relative noise $\\sigma$")
    right.set_ylabel("success rate (%)")
    right.set_ylim(0, 106)
    right.set_title("Prediction survives a false explanation", loc="left")
    right.grid(axis="y")
    right.legend(loc="lower left", fontsize=6.8)
    right.annotate(
        "99.8% forecast",
        xy=(3, noise_forecast[-1]),
        xytext=(2.0, 87),
        color=GOLD,
        fontsize=6.6,
        arrowprops={"arrowstyle": "->", "color": GOLD, "linewidth": 0.7},
    )
    right.annotate(
        "32.1% class",
        xy=(3, noise_accuracy[-1]),
        xytext=(1.85, 20),
        color=TEAL,
        fontsize=6.6,
        arrowprops={"arrowstyle": "->", "color": TEAL, "linewidth": 0.7},
    )
    save(fig, "aws_recovery_results")


def main() -> None:
    kernel_atlas()
    hidden_lag_collision()
    aws_recovery_results()
    print(f"Wrote six figure files to {FIGURES}")


if __name__ == "__main__":
    main()
