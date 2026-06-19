"""Generate figures for "The Shape of Recursive Redundancy".

Run from any directory:

    python experiments.py

The script writes publication-ready PDF and PNG files next to this script.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from takeoff import (
    centered_moving_average,
    dag_velocity,
    distinct_labels,
    modal_structure,
    terminal_alpha,
    tree_sizes,
    velocity_profile,
)

HERE = Path(__file__).resolve().parent
PHI = (1.0 + 5.0**0.5) / 2.0

NAVY = "#0B1F3A"
BLUE_GRAY = "#34495E"
SLATE = "#6B7280"
LIGHT_GRAY = "#D7DCE2"
GRID = "#E5E7EB"
BLACK = "#111111"
WHITE = "#FFFFFF"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.4,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.5,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.75,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.facecolor": WHITE,
            "legend.edgecolor": LIGHT_GRAY,
            "legend.fontsize": 7.3,
            "lines.linewidth": 1.45,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "savefig.dpi": 260,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png")


def leading_kind(mode: complex) -> str:
    if abs(mode.imag) < 1e-9 and mode.real > 0:
        return "monotone"
    if abs(mode.imag) < 1e-9:
        return "alternating"
    period = 2.0 * math.pi / abs(math.atan2(mode.imag, mode.real))
    return f"ring p~{period:.1f}"


def schema_profile(schema: dict[int, int], nmax: int):
    sizes = tree_sizes(schema, nmax)
    alpha, rho = terminal_alpha(schema)
    _, modal_radius, leading = modal_structure(schema)
    velocity, profile, kernel = velocity_profile(sizes, alpha)
    return sizes, alpha, rho, modal_radius, leading, velocity, profile, kernel


def fibonacci_leaf_profile(nmax: int):
    sizes = [0] * (nmax + 1)
    sizes[0] = sizes[1] = 1
    for n in range(2, nmax + 1):
        sizes[n] = sizes[n - 1] + sizes[n - 2]
    alpha = math.log(PHI)
    velocity, profile, kernel = velocity_profile(sizes, alpha)
    return sizes, alpha, velocity, profile, kernel


def central_binomial_profile(nmax: int):
    counts = [math.comb(2 * n, n) for n in range(nmax + 1)]
    alpha = 2.0 * math.log(2.0)
    velocity, profile, kernel = velocity_profile(counts, alpha)
    return counts, alpha, velocity, profile, kernel


def print_validation(rows: list[tuple[str, dict[int, int], int, int]]) -> None:
    print("=" * 96)
    print(
        f"{'schema':<24}{'alpha_thy':>10}{'alpha_emp':>11}"
        f"{'|F-1| tail':>13}{'lambda*':>10}{'leading mode':>17}"
    )
    print("-" * 96)
    for name, schema, nmax, period in rows:
        _, alpha, _, modal_radius, leading, velocity, profile, _ = schema_profile(schema, nmax)
        print(
            f"{name:<24}{alpha:>10.5f}{velocity[-period:].mean():>11.5f}"
            f"{abs(profile[-1] - 1):>13.2e}{modal_radius:>10.4f}"
            f"{leading_kind(leading[0][1]):>17}"
        )
    print("=" * 96)


def make_separation_figure(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.85), constrained_layout=True)
    fig.suptitle(
        "Same terminal overlap velocity, different finite takeoff",
        fontsize=10.2,
        fontweight="bold",
        color=BLACK,
    )

    immediate = data["immediate"]
    slow4 = data["slow4"]
    slow8 = data["slow8"]

    ax = axes[0, 0]
    ax.axhline(1.0, color=BLACK, linewidth=0.75, linestyle=":")
    ax.plot(immediate["profile"], color=NAVY, label="immediate binary")
    ax.plot(slow4["profile"], color=BLUE_GRAY, linestyle="--", label="slow family L=4")
    ax.set_xlim(0, 88)
    ax.set_ylim(0, 4.35)
    ax.set_title("(a) normalized tree velocity")
    ax.set_xlabel(r"input size $n$")
    ax.set_ylabel(r"$F_n$")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.axhline(1.0, color=BLACK, linewidth=0.75, linestyle=":")
    ax.plot(slow8["profile"], color=NAVY, label="slow family L=8")
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 8.25)
    ax.set_title("(b) near-periodic slow takeoff")
    ax.set_xlabel(r"input size $n$")
    ax.set_ylabel(r"$F_n$")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    for item, color, linestyle, label, period in [
        (immediate, NAVY, "-", "immediate", 2),
        (slow4, BLUE_GRAY, "--", "L=4", 4),
        (slow8, SLATE, "-.", "L=8", 8),
    ]:
        profile = item["profile"]
        schema = item["schema"]
        counts = distinct_labels(schema, item["nmax"])
        overlap_velocity = math.log(2.0) * profile[: len(counts) - 1] - dag_velocity(counts)
        x, average = centered_moving_average(overlap_velocity, period)
        ax.plot(overlap_velocity, color=color, alpha=0.13, linewidth=0.7)
        ax.plot(x, average, color=color, linestyle=linestyle, label=f"{label} period mean")
    ax.axhline(math.log(2.0), color=BLACK, linewidth=0.75, linestyle=":")
    ax.set_xlim(0, 120)
    ax.set_ylim(0.30, 1.08)
    ax.set_title("(c) same limiting overlap velocity")
    ax.set_xlabel(r"input size $n$")
    ax.set_ylabel(r"$v_n$")
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    for item, color, linestyle, label in [
        (immediate, NAVY, "-", "immediate"),
        (slow4, BLUE_GRAY, "--", "L=4"),
        (slow8, SLATE, "-.", "L=8"),
    ]:
        error = np.abs(item["profile"] - 1.0) + 1e-16
        ax.semilogy(
            error,
            color=color,
            linestyle=linestyle,
            label=fr"{label}, $\lambda_\star={item['modal_radius']:.3f}$",
        )
    ax.set_xlim(0, 120)
    ax.set_ylim(1e-7, 12)
    ax.set_title("(d) modal radius controls settling")
    ax.set_xlabel(r"input size $n$")
    ax.set_ylabel(r"$|F_n-1|$")
    ax.legend(loc="lower right")

    save_figure(fig, "fig1_separation")
    plt.close(fig)


def make_fibonacci_figure(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25), constrained_layout=True)
    fig.suptitle(
        "Fibonacci: boundary convention changes the visible kernel prefix",
        fontsize=9.9,
        fontweight="bold",
        color=BLACK,
    )

    nodes = data["fib_nodes"]
    leaves = data["fib_leaves"]
    n = np.arange(len(nodes["profile"]))

    axes[0].axhline(0.0, color=BLACK, linewidth=0.7)
    axes[0].plot(nodes["kernel"], color=NAVY, marker="o", markersize=2.3)
    axes[0].set_xlim(0, 24)
    axes[0].set_title("(a) node-count kernel")
    axes[0].set_xlabel(r"$n$")
    axes[0].set_ylabel(r"$\kappa_n$")

    axes[1].axhline(0.0, color=BLACK, linewidth=0.7)
    axes[1].plot(leaves["kernel"], color=BLUE_GRAY, marker="s", markersize=2.2)
    axes[1].set_xlim(0, 24)
    axes[1].set_title("(b) leaf-count kernel")
    axes[1].set_xlabel(r"$n$")

    fit_window = slice(6, 22)
    coeff = np.polyfit(n[fit_window], np.log(np.abs(nodes["profile"] - 1.0)[fit_window]), 1)
    monotone_fit = np.sign((nodes["profile"] - 1.0)[fit_window]).mean() * np.exp(coeff[1]) * np.exp(coeff[0] * n)
    residual = (nodes["profile"] - 1.0) - monotone_fit
    axes[2].axhline(0.0, color=BLACK, linewidth=0.7)
    axes[2].plot(n, nodes["profile"] - 1.0, color=SLATE, linewidth=1.1, label=r"$F_n-1$")
    axes[2].plot(n, monotone_fit, color=BLACK, linestyle="--", linewidth=0.95, label=r"$\rho^n$ fit")
    axes[2].plot(n, residual, color=NAVY, marker="^", markersize=2.2, label="ripple")
    axes[2].set_xlim(2, 24)
    axes[2].set_ylim(-0.06, 0.18)
    axes[2].set_title("(c) monotone tail plus ripple")
    axes[2].set_xlabel(r"$n$")
    axes[2].legend(loc="upper right")

    save_figure(fig, "fig2_fibonacci")
    plt.close(fig)


def make_decay_figure(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 4.25), constrained_layout=True)
    fig.suptitle(
        "Settling class: finite-lag vs grid DP",
        fontsize=9.6,
        fontweight="bold",
        color=BLACK,
    )

    immediate = data["immediate"]
    fib_nodes = data["fib_nodes"]
    binomial = data["binomial"]

    ax = axes[0]
    ax.semilogy(np.abs(immediate["profile"] - 1.0) + 1e-16, color=NAVY, label="immediate binary")
    ax.semilogy(np.abs(fib_nodes["profile"] - 1.0) + 1e-16, color=BLUE_GRAY, linestyle="--", label="Fibonacci nodes")
    ax.semilogy(np.abs(binomial["profile"] - 1.0) + 1e-16, color=BLACK, linestyle="-.", label="central binomial")
    ax.set_xlim(0, 60)
    ax.set_ylim(1e-7, 3)
    ax.set_title("(a) semilog error")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$|F_n-1|$")
    ax.legend(loc="upper right")

    ax = axes[1]
    n = np.arange(1, len(binomial["profile"]))
    ax.loglog(n, np.abs(binomial["profile"] - 1.0)[1:], color=BLACK, label="central binomial")
    ax.loglog(n, 0.25 / n, color=SLATE, linestyle="--", label="0.25/n")
    ax.set_title("(b) algebraic grid-DP tail")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$|F_n-1|$")
    ax.legend(loc="upper right")

    save_figure(fig, "fig3_decay_class")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()

    immediate = {1: 2}
    slow4 = {4: 15, 5: 2}
    slow8 = {8: 255, 9: 2}
    fibonacci = {1: 1, 2: 1}

    print_validation(
        [
            ("immediate binary", immediate, 46, 1),
            ("fibonacci nodes", fibonacci, 70, 1),
            ("slow family L=4", slow4, 120, 4),
            ("slow family L=8", slow8, 160, 8),
        ]
    )

    data: dict[str, object] = {}
    for key, schema, nmax in [
        ("immediate", immediate, 46),
        ("slow4", slow4, 120),
        ("slow8", slow8, 160),
        ("fib_nodes", fibonacci, 70),
    ]:
        sizes, alpha, rho, modal_radius, leading, velocity, profile, kernel = schema_profile(schema, nmax)
        data[key] = {
            "schema": schema,
            "nmax": nmax,
            "sizes": sizes,
            "alpha": alpha,
            "rho": rho,
            "modal_radius": modal_radius,
            "leading": leading,
            "velocity": velocity,
            "profile": profile,
            "kernel": kernel,
        }

    leaf_sizes, leaf_alpha, leaf_velocity, leaf_profile, leaf_kernel = fibonacci_leaf_profile(70)
    data["fib_leaves"] = {
        "sizes": leaf_sizes,
        "alpha": leaf_alpha,
        "velocity": leaf_velocity,
        "profile": leaf_profile,
        "kernel": leaf_kernel,
    }

    binomial_sizes, binomial_alpha, binomial_velocity, binomial_profile, binomial_kernel = central_binomial_profile(220)
    data["binomial"] = {
        "sizes": binomial_sizes,
        "alpha": binomial_alpha,
        "velocity": binomial_velocity,
        "profile": binomial_profile,
        "kernel": binomial_kernel,
    }

    make_separation_figure(data)
    make_fibonacci_figure(data)
    make_decay_figure(data)

    tail_constant = (1.0 - binomial_profile[-1]) * (len(binomial_profile) - 1)
    print(f"central binomial tail constant estimate: {(tail_constant):.3f}")
    print(f"figures written to {HERE}")


if __name__ == "__main__":
    main()
