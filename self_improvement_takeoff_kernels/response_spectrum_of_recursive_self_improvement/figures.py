"""Build the two figures for The Entropy of Recursive Response."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

ARMS = ["raw_replace", "anchored", "verified", "guided"]
LABELS = {
    "raw_replace": "Raw replacement",
    "anchored": "Fixed anchors",
    "verified": "Diversity verification",
    "guided": "Diversity-guided prompt",
}
COLORS = {
    "raw_replace": "#4C78A8",
    "anchored": "#E45756",
    "verified": "#54A24B",
    "guided": "#B279A2",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def response_curves() -> None:
    aggregate = pd.read_csv(DATA / "aggregate.csv")
    panels = [
        ("quality", r"Composite response $Q_r$"),
        ("entropy", r"Marginal entropy $\bar H_r$"),
        ("unique_fraction", r"Distinct combinations $D_r$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.38), sharex=True)

    for ax, (metric, title) in zip(axes, panels):
        for arm in ARMS:
            frame = aggregate[aggregate["arm"] == arm].sort_values("round")
            x = frame["round"].to_numpy()
            y = frame[f"{metric}_mean"].to_numpy()
            se = frame[f"{metric}_se"].to_numpy()
            ax.plot(x, y, color=COLORS[arm], lw=1.65, label=LABELS[arm])
            ax.fill_between(x, y - se, y + se, color=COLORS[arm], alpha=0.14, linewidth=0)
        ax.set_title(title, pad=5)
        ax.set_xlabel("Recursive round")
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.8)

    axes[0].set_ylim(0.68, 1.015)
    axes[1].set_ylim(0.80, 1.01)
    axes[2].set_ylim(0.50, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(wspace=0.28, bottom=0.25)
    save(fig, "fig1_recursive_population")


def response_geometry() -> None:
    trajectory = pd.read_csv(DATA / "trajectory.csv").sort_values(["arm", "replicate", "round"])
    records = []
    for (arm, replicate), frame in trajectory.groupby(["arm", "replicate"], sort=False):
        q = frame["quality"].to_numpy()
        delta = np.diff(q)
        records.append(
            {
                "arm": arm,
                "replicate": replicate,
                "gain": q[-1] - q[0],
                "variation": np.abs(delta).sum(),
            }
        )
    geometry = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(6.60, 2.55))
    limit = max(0.38, float(geometry["variation"].max()) * 1.08)
    x = np.linspace(-limit, limit, 500)
    ax.fill_between(x, np.abs(x), limit, color="#F2F2F2", zorder=0)
    ax.plot(x, np.abs(x), color="#555555", lw=1.1, ls="--")

    rng = np.random.default_rng(20260717)
    for arm in ARMS:
        frame = geometry[geometry["arm"] == arm]
        jitter = rng.normal(0, 0.0022, len(frame))
        ax.scatter(
            frame["gain"] + jitter,
            frame["variation"],
            s=19,
            facecolor=COLORS[arm],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.66,
            zorder=2,
        )
        mean = frame[["gain", "variation"]].mean()
        ax.scatter(
            [mean["gain"]],
            [mean["variation"]],
            s=78,
            marker="D",
            facecolor=COLORS[arm],
            edgecolor="black",
            linewidth=0.65,
            label=LABELS[arm],
            zorder=4,
        )

    ax.axvline(0, color="#888888", lw=0.65)
    ax.text(-0.19, 0.215, r"monotone wall $A=|g|$", rotation=-31, ha="center", va="bottom", color="#555555", fontsize=8)
    ax.text(-0.205, 0.365, "collapse", ha="center", va="top", color="#666666", fontsize=8)
    ax.text(0.135, 0.365, "improvement", ha="center", va="top", color="#666666", fontsize=8)
    ax.text(0.005, 0.305, "reversals / overshoot", ha="center", va="center", color="#777777", fontsize=8)
    ax.set_xlabel(r"Terminal gain $g=Q_{10}-Q_0$", fontsize=10)
    ax.set_ylabel(r"Total response $A=\sum_r |Q_r-Q_{r-1}|$", fontsize=10)
    ax.set_xlim(-0.26, 0.18)
    ax.set_ylim(0, 0.39)
    ax.grid(color="#D9D9D9", lw=0.45, alpha=0.7)
    ax.tick_params(labelsize=9)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 1.08), frameon=False, ncol=4, handletextpad=0.45, columnspacing=1.1, fontsize=8.5)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.97)
    save(fig, "fig2_response_geometry")


if __name__ == "__main__":
    response_curves()
    response_geometry()
