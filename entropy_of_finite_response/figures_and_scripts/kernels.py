"""
Figures for "The Entropy of Finite Response" (J. Landers).

Illustrative family: Gamma kernels K(t; k, theta) = t^(k-1) e^{-t/theta} / (Gamma(k) theta^k), t >= 0.
Why gamma:
  - passive causal: support [0, inf), nonnegative, normalized
  - mean   mu  = k theta
  - var    s2  = k theta^2
  - closed under convolution: Gamma(k1,th)*Gamma(k2,th) = Gamma(k1+k2,th)  <-- the sequential law
  - k -> inf gives the Gaussian limit, but never reaches it (support stays on [0,inf))
Entropy (nats):  h = k + ln theta + ln Gamma(k) + (1-k) psi(k)
Entropy power :  N = (1/(2 pi e)) exp(2 h)   [so N = s2 exactly for a Gaussian]
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import FancyArrowPatch
from scipy.special import gammaln, digamma
from scipy.stats import gamma as gammadist, norm

# ----------------------------------------------------------------------------- style
OUTDIR   = Path(__file__).resolve().parent

INK      = "#111827"
PANEL    = "#ffffff"
GRID     = "#d6dce5"
TEXT     = "#111827"
MUTE     = "#536070"
BLUE     = "#2563ad"
BLUE_HI  = "#4f8edb"
TEAL     = "#0f8a83"
AMBER    = "#c26a00"      # the forbidden Gaussian / efficient wall
ROSE     = "#c8324b"
WHITE    = "#ffffff"
FIGSIZE  = (6.35, 4.15)
FIGSIZE_TALL = (6.35, 4.35)

mpl.rcParams.update({
    "figure.facecolor": WHITE, "axes.facecolor": PANEL, "savefig.facecolor": WHITE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": MUTE, "ytick.color": MUTE, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12.5,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.2, "ytick.labelsize": 9.2,
    "legend.fontsize": 8.8,
    "axes.linewidth": 1.0, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42,
})

def style_ax(ax, top_right_off=True):
    ax.grid(True, lw=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    if top_right_off:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

def save_figure(fig, stem):
    fig.tight_layout(pad=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=240, bbox_inches="tight")

# ----------------------------------------------------------------------------- gamma helpers
def gamma_pdf(t, k, theta):
    return gammadist.pdf(t, a=k, scale=theta)

def gamma_h(k, theta):           # differential entropy, nats
    return k + np.log(theta) + gammaln(k) + (1.0 - k) * digamma(k)

def gamma_N(k, theta):           # entropy power
    return np.exp(2.0 * gamma_h(k, theta)) / (2.0 * np.pi * np.e)

def gamma_mu(k, theta):  return k * theta
def gamma_s2(k, theta):  return k * theta**2


# ============================================================================= FIG 1
# Shape gallery: the realizable interior between two walls (delta, Gaussian).
def fig1_gallery():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    style_ax(ax)

    mu = 1.0                                  # fix the mean; shape runs k=1 -> large
    ks = [1, 2, 3, 5, 9]
    shades = np.linspace(0.0, 1.0, len(ks))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("c", [TEAL, BLUE, BLUE_HI])

    t = np.linspace(-0.6, 3.4, 1400)
    tp = t[t >= 0]

    for k, sh in zip(ks, shades):
        theta = mu / k
        y = np.zeros_like(t)
        y[t >= 0] = gamma_pdf(tp, k, theta)
        c = cmap(sh)
        ax.plot(t, y, color=c, lw=2.2, zorder=4,
                label=fr"$k={k}$")

    # forbidden Gaussian: efficiency wall for the broad (k=1) case, leaks past t=0
    s = np.sqrt(gamma_s2(1, mu))              # variance-matched Gaussian, mean=mu
    g = norm.pdf(t, loc=mu, scale=s)
    ax.plot(t, g, color=AMBER, lw=2.0, ls=(0, (5, 2)), zorder=5,
            label="Gaussian")
    neg = t < 0
    ax.fill_between(t[neg], 0, g[neg], color=AMBER, alpha=0.16, zorder=2)

    # delta wall at t = mu (the k -> inf, sigma^2 -> 0 limit)
    ax.add_patch(FancyArrowPatch((mu, 0), (mu, 1.95), arrowstyle="-|>",
                 mutation_scale=16, color=INK, lw=2.3, zorder=6))
    ax.text(mu + 0.08, 1.78, r"$\delta(t-\mu)$" + "\nsharp wall", color=INK,
            fontsize=9.6, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc=WHITE, ec="none", alpha=0.9))

    ax.axvline(0, color=MUTE, lw=1.0, alpha=0.7)
    ax.text(-0.34, 0.10, "$t<0$\nforbidden", color=ROSE, fontsize=8.8,
            ha="center", va="bottom")

    ax.set_xlim(-0.6, 3.4); ax.set_ylim(0, 2.15)
    ax.set_xlabel("delay  $t$"); ax.set_ylabel("$K(t)$")
    ax.set_title("Kernel shapes between two walls", color=TEXT, pad=10, loc="left",
                 fontweight="bold")
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8.8, ncol=2,
                    columnspacing=0.9, handlelength=1.8,
                    facecolor=WHITE, edgecolor=GRID, labelcolor=TEXT)
    leg.get_frame().set_alpha(0.96)
    save_figure(fig, "fig1_kernel_gallery")
    plt.close(fig)


# ============================================================================= FIG 2
# Sequential stages: self-convolution of an exponential = Gamma(n). mu adds, s2 adds.
def fig2_cascade():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    style_ax(ax)

    theta = 1.0
    ns = [1, 2, 4, 8, 16]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("c", [TEAL, BLUE, BLUE_HI])
    shades = np.linspace(0.0, 1.0, len(ns))

    t = np.linspace(0, 30, 2000)
    for n, sh in zip(ns, shades):
        y = gamma_pdf(t, n, theta)
        c = cmap(sh)
        ax.plot(t, y, color=c, lw=2.2, zorder=4,
                label=fr"$n={n}$")
        # mean marker
        ax.plot([n], [gamma_pdf(n, n, theta)], "o", ms=4, color=c, zorder=5)

    # the matched Gaussian for the n=16 stage, to show the kernel sits inside it
    n = 16
    g = norm.pdf(t, loc=n, scale=np.sqrt(n))
    ax.plot(t, g, color=AMBER, lw=1.8, ls=(0, (5, 2)), zorder=3,
            label=r"Gaussian ($n=16$)")

    ax.set_xlim(0, 30); ax.set_ylim(0, 1.02)
    ax.set_xlabel("delay  $t$"); ax.set_ylabel("$K^{*n}(t)$")
    ax.set_title("Sequential stages broaden and drift right", color=TEXT, pad=10,
                 loc="left", fontweight="bold")
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8.8, ncol=2,
                    columnspacing=0.9, handlelength=1.8,
                    facecolor=WHITE, edgecolor=GRID, labelcolor=TEXT)
    leg.get_frame().set_alpha(0.96)
    save_figure(fig, "fig2_convolution_cascade")
    plt.close(fig)


# ============================================================================= FIG 3
# Parallel channels: convex mixture, the extra-variance term made visible.
def fig3_mixture():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    style_ax(ax)

    # two fairly sharp channels at different arrival times
    k = 12
    mu1, mu2 = 1.0, 4.0
    th1, th2 = mu1 / k, mu2 / k
    t = np.linspace(0, 7, 2000)
    K1 = gamma_pdf(t, k, th1)
    K2 = gamma_pdf(t, k, th2)

    ax.plot(t, K1, color=MUTE, lw=1.6, ls=(0, (4, 2)), zorder=3,
            label=fr"$K_1$ ($\mu_1={mu1:.0f}$)")
    ax.plot(t, K2, color=MUTE, lw=1.6, ls=(0, (1, 2)), zorder=3,
            label=fr"$K_2$ ($\mu_2={mu2:.0f}$)")

    lams = [0.25, 0.5, 0.75]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("c", [TEAL, BLUE, BLUE_HI])
    for lam, sh in zip(lams, np.linspace(0.15, 1.0, len(lams))):
        Kl = (1 - lam) * K1 + lam * K2
        # exact variance of the mixture (each channel var = k*theta^2)
        s2 = ((1-lam)*(k*th1**2) + lam*(k*th2**2)
              + lam*(1-lam)*(mu2-mu1)**2)
        ax.plot(t, Kl, color=cmap(sh), lw=2.4, zorder=4,
                label=fr"$\lambda={lam:.2f}$")

    # annotate the cross-term: variance of a single sharp channel for comparison
    ax.text(0.36, 0.84,
            "extra variance:\n"
            r"$\lambda(1-\lambda)(\mu_2-\mu_1)^2$",
            transform=ax.transAxes, color=TEXT, fontsize=8.8, va="top",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.45", fc=WHITE, ec=GRID, alpha=0.96))

    ax.set_xlim(0, 7); ax.set_ylim(0, None)
    ax.set_xlabel("delay  $t$"); ax.set_ylabel("$K_\\lambda(t)$")
    ax.set_title("Parallel channels broaden without sharpening", color=TEXT, pad=10,
                 loc="left", fontweight="bold")
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8.8,
                    facecolor=WHITE, edgecolor=GRID, labelcolor=TEXT)
    leg.get_frame().set_alpha(0.96)
    save_figure(fig, "fig3_parallel_mixture")
    plt.close(fig)


# ============================================================================= FIG 4
# The boundary picture: entropy power N vs variance sigma^2, two walls + cascade path.
def fig4_plane():
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    style_ax(ax)

    s2max = 18.0
    xs = np.linspace(0, s2max, 200)

    # upper wall: Gaussian locus N = sigma^2 (unreachable)
    ax.plot(xs, xs, color=AMBER, lw=2.2, ls=(0, (5, 2)), zorder=5,
            label=r"Gaussian wall  $N=\sigma^2$  (efficient, non-causal)")
    ax.fill_between(xs, xs, s2max*1.05, color=AMBER, alpha=0.06, zorder=1)
    ax.text(11.3, 12.8, "forbidden:\n $N>\\sigma^2$ impossible", color=AMBER,
            fontsize=9.5, rotation=33, ha="center", va="center")

    # lower wall: delta locus N = 0
    ax.axhline(0, color=INK, lw=1.8, zorder=4)
    ax.text(s2max*0.5, 0.18, r"delta wall  $N=0$  (infinitely sharp, reached only in the limit)",
            color=INK, fontsize=9.0, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec="none", alpha=0.9))

    # generic causal kernels (gammas over a grid of k, theta) -> scatter the interior
    rng = np.random.default_rng(3)
    pts_s2, pts_N = [], []
    for _ in range(450):
        k = 10**rng.uniform(-0.05, 1.6)        # k in ~[0.9, 40]
        theta = 10**rng.uniform(-0.6, 0.45)
        s2 = gamma_s2(k, theta)
        if s2 <= s2max:
            pts_s2.append(s2); pts_N.append(gamma_N(k, theta))
    ax.scatter(pts_s2, pts_N, s=10, c=BLUE, alpha=0.24, edgecolors="none", zorder=3,
               label="passive causal kernels (gamma)")

    # the convolution cascade trajectory: Gamma(n, 1), n = 1..20
    theta = 1.0
    ns = np.arange(1, 21)
    cs2 = [gamma_s2(n, theta) for n in ns]
    cN  = [gamma_N(n, theta) for n in ns]
    ax.plot(cs2, cN, color=TEAL, lw=2.1, zorder=6)
    ax.scatter(cs2, cN, s=28, c=TEAL, edgecolors=INK, lw=0.45, zorder=7,
               label=r"cascade $\Gamma(n,\theta)$, $n=1\ldots20$")
    for n, offset in ((1, (6, 8)), (4, (6, 8)), (16, (6, -10))):
        ax.annotate(fr"$n={n}$", (cs2[n-1], cN[n-1]), color=TEAL, fontsize=9,
                    xytext=offset, textcoords="offset points", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.08", fc=WHITE, ec="none", alpha=0.8))

    ax.set_xlim(0, s2max); ax.set_ylim(-0.6, s2max)
    ax.set_xlabel(r"variance  $\sigma^2$  (adds exactly under composition)")
    ax.set_ylabel(r"entropy power  $N$")
    ax.set_title("The causal interior, pinned between two walls", color=TEXT, pad=10,
                 loc="left", fontweight="bold")
    leg = ax.legend(loc="upper left", frameon=True, fontsize=9.0,
                    facecolor=WHITE, edgecolor=GRID, labelcolor=TEXT)
    leg.get_frame().set_alpha(0.96)
    save_figure(fig, "fig4_entropy_power_plane")
    plt.close(fig)


if __name__ == "__main__":
    fig1_gallery()
    fig2_cascade()
    fig3_mixture()
    fig4_plane()

    # quick sanity print: superadditivity of N along the cascade
    theta = 1.0
    N1 = gamma_N(1, theta)
    print("Cascade check  (Gamma(n,1)):")
    print(f"{'n':>3} {'sigma^2':>9} {'N':>9} {'N/sigma^2':>10} {'N(n)/(nN1)':>11}")
    for n in (1, 2, 4, 8, 16):
        s2 = gamma_s2(n, theta); N = gamma_N(n, theta)
        print(f"{n:>3} {s2:>9.3f} {N:>9.3f} {N/s2:>10.3f} {N/(n*N1):>11.3f}")
    print("done")
