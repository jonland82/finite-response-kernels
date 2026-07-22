"""Render the sharp-boundary comparison from a completed constraint run."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
INK = "#16212B"
NAVY = "#17324D"
TEAL = "#0B7A75"
GOLD = "#D28F2C"
CORAL = "#C65D43"
MUTED = "#667085"
GRID = "#D8E2E3"
METHOD_LABELS = {
    "constraint_integer": "integer + dyadic constraints",
    "continuous_round": "continuous fit + rounding",
    "matrix_pencil_6": "six-mode matrix pencil",
}
COLORS = {"constraint_integer": TEAL, "continuous_round": GOLD, "matrix_pencil_6": CORAL}
METHODS_FOR_METRIC = {
    "exact": ("constraint_integer", "continuous_round"),
    "forecast": ("constraint_integer", "continuous_round", "matrix_pencil_6"),
}


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 8.4,
    "axes.labelsize": 7.5, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.55,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def latest_constraint_results() -> Path:
    candidates = sorted((ROOT / "runs").glob("inverse-constraint-*/results"))
    if not candidates:
        raise FileNotFoundError("no completed constraint run found")
    return candidates[-1]


def aggregate(results: Path) -> tuple[dict[tuple[str, str, str], list[int]], list[int]]:
    totals: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    version_counts: list[int] = []
    for path in sorted((results / "raw").glob("worker_*.csv.gz")):
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                offset = int(row["horizon_offset"])
                horizon_label = str(offset) if offset in (-2, -1, 0, 2) else "long"
                method = row["method"]
                noise = row["noise"]
                if method == "constraint_integer" and noise == "0" and offset == -2:
                    version_counts.append(int(row["theoretical_version_count"]))
                if method in METHODS_FOR_METRIC["exact"]:
                    key = ("exact", method, f"{horizon_label}|{noise}")
                    totals[key][0] += int(row["exact_source"] or 0)
                    totals[key][1] += 1
                key = ("forecast", method, f"{horizon_label}|{noise}")
                totals[key][0] += int(row["forecast_success"])
                totals[key][1] += 1
    return totals, version_counts


def rate(totals: dict[tuple[str, str, str], list[int]], metric: str,
         method: str, horizon: str, noise: str) -> float:
    successes, count = totals[(metric, method, f"{horizon}|{noise}")]
    return successes / count if count else float("nan")


def render(results: Path) -> tuple[dict[tuple[str, str, str], list[int]], list[int]]:
    totals, version_counts = aggregate(results)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.18))
    fig.subplots_adjust(wspace=0.38, bottom=0.27)

    horizon_keys = ["-2", "-1", "0", "2", "long"]
    horizon_labels = ["$D-2$", "$D-1$", "$D$", "$D+2$", "$2D+2$"]
    for method in ("constraint_integer", "continuous_round"):
        values = [rate(totals, "exact", method, key, "0") for key in horizon_keys]
        axes[0].plot(range(len(values)), values, marker="o", markersize=3.5,
                     linewidth=1.5, color=COLORS[method], label=METHOD_LABELS[method])
    axes[0].axvline(0.5, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.75)
    axes[0].set_xticks(range(5), horizon_labels)
    axes[0].set_title("Noiseless source recovery", loc="left", pad=8)
    axes[0].set_xlabel("observed samples $T$")
    axes[0].set_ylabel("exact recovery rate")

    noise_keys = ["0", "1e-08", "1e-05", "0.001"]
    noise_labels = ["0", "$10^{-8}$", "$10^{-5}$", "$10^{-3}$"]
    for method in ("constraint_integer", "continuous_round"):
        values = [rate(totals, "exact", method, "-1", noise) for noise in noise_keys]
        axes[1].plot(range(4), values, marker="o", markersize=3.5,
                     linewidth=1.5, color=COLORS[method])
    axes[1].set_xticks(range(4), noise_labels)
    axes[1].set_title("Recovery at $T=D-1$", loc="left", pad=8)
    axes[1].set_xlabel("relative noise")

    for method in ("constraint_integer", "continuous_round", "matrix_pencil_6"):
        values = [rate(totals, "forecast", method, "-1", noise) for noise in noise_keys]
        axes[2].plot(range(4), values, marker="o", markersize=3.5,
                     linewidth=1.5, color=COLORS[method], label=METHOD_LABELS[method])
    axes[2].set_xticks(range(4), noise_labels)
    axes[2].set_title("Forecast at $T=D-1$", loc="left", pad=8)
    axes[2].set_xlabel("relative noise")
    axes[2].set_ylabel("NRMSE $<0.1$")

    for axis in axes:
        axis.set_ylim(-0.03, 1.03)
        axis.set_yticks([0, 0.5, 1.0])
        axis.grid(axis="y")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=6.6,
               bbox_to_anchor=(0.5, 0.01))
    FIGURES.mkdir(exist_ok=True)
    for extension, kwargs in (("pdf", {}), ("png", {"dpi": 240})):
        fig.savefig(FIGURES / f"constraint_boundary_results.{extension}",
                    bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    return totals, version_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    results = args.results.resolve() if args.results else latest_constraint_results()
    totals, version_counts = render(results)
    snapshot: dict[str, object] = {"results": str(results), "rates": {}}
    multiple = sum(value > 1 for value in version_counts)
    snapshot["version_space_at_d_minus_2"] = {
        "schemas": len(version_counts), "multiple_source_schemas": multiple,
        "multiple_source_rate": multiple / len(version_counts),
        "median_version_count": statistics.median(version_counts),
        "maximum_version_count": max(version_counts),
    }
    for metric in ("exact", "forecast"):
        for method in METHODS_FOR_METRIC[metric]:
            for horizon in ("-2", "-1", "0", "2", "long"):
                for noise in ("0", "1e-08", "1e-05", "0.001"):
                    successes, count = totals[(metric, method, f"{horizon}|{noise}")]
                    snapshot["rates"][f"{metric}|{method}|{horizon}|{noise}"] = {
                        "successes": successes, "count": count,
                        "rate": successes / count if count else None,
                    }
    (results / "figure_summary.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Rendered constraint comparison from {results}")


if __name__ == "__main__":
    main()
