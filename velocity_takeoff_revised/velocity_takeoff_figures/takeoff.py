"""Measurement helpers for velocity takeoff experiments.

The finite-lag schema is

    N_n = 1 + sum_j a_j N_{n-j},     N_m = 1 for m <= 0.

The functions here compute exact tree sizes, exact reachable label counts,
terminal speed, modal radius, and the normalized velocity takeoff profile.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

Schema = Mapping[int, int]


def tree_sizes(schema: Schema, nmax: int) -> list[int]:
    """Return exact naive tree sizes N_0,...,N_nmax."""
    sizes = [0] * (nmax + 1)
    for n in range(nmax + 1):
        if n <= 0:
            sizes[n] = 1
            continue

        total = 1
        for lag, multiplicity in sorted(schema.items()):
            previous = n - lag
            total += multiplicity * (sizes[previous] if previous >= 0 else 1)
        sizes[n] = total
    return sizes


def distinct_labels(schema: Schema, nmax: int) -> list[int]:
    """Return M_n, the number of distinct labels reachable from input n."""
    lags = tuple(sorted(schema))
    counts = [0] * (nmax + 1)

    for n in range(nmax + 1):
        reachable: set[int] = set()
        stack = [n]
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            if current > 0:
                stack.extend(current - lag for lag in lags)
        counts[n] = len(reachable)

    return counts


def terminal_alpha(schema: Schema) -> tuple[float, float]:
    """Return (alpha, rho), where sum_j a_j rho^j = 1 and alpha = -log rho."""

    def equation(rho: float) -> float:
        return sum(multiplicity * rho**lag for lag, multiplicity in schema.items()) - 1.0

    lo, hi = 1e-15, 1.0 - 1e-15
    for _ in range(240):
        mid = 0.5 * (lo + hi)
        if equation(mid) < 0:
            lo = mid
        else:
            hi = mid

    rho = 0.5 * (lo + hi)
    return -math.log(rho), rho


def recurrence_roots(schema: Schema) -> list[complex]:
    """Return roots of Q(z)=1-sum_j a_j z^j."""
    degree = max(schema)
    coeffs = [0.0] * (degree + 1)
    coeffs[0] = 1.0
    for lag, multiplicity in schema.items():
        coeffs[lag] = -float(multiplicity)
    return list(np.roots(coeffs[::-1]))


def modal_structure(schema: Schema) -> tuple[float, float, list[tuple[float, complex]]]:
    """Return (rho, lambda_star, leading_modes).

    The affine +1 in the tree recurrence contributes the singularity z=1.
    The modal radius is max |rho/zeta| over non-dominant singularities.
    """
    _, rho = terminal_alpha(schema)
    roots = recurrence_roots(schema)
    singularities = roots + [1.0 + 0j]

    dominant_index = min(range(len(singularities)), key=lambda i: abs(singularities[i] - rho))
    candidates = [
        z
        for i, z in enumerate(singularities)
        if i != dominant_index and abs(z - rho) > 1e-9
    ]
    ratios = sorted(((abs(rho / z), z) for z in candidates), key=lambda item: -item[0])
    return rho, ratios[0][0], ratios[:4]


def velocity_profile(sizes: list[int], alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tree velocity u_n, normalized profile F_n, and kernel kappa_n."""
    velocity = np.array(
        [math.log(sizes[n + 1]) - math.log(sizes[n]) for n in range(len(sizes) - 1)]
    )
    profile = velocity / alpha
    kernel = np.empty_like(profile)
    kernel[0] = profile[0]
    kernel[1:] = profile[1:] - profile[:-1]
    return velocity, profile, kernel


def dag_velocity(counts: list[int]) -> np.ndarray:
    """Return w_n = log M_{n+1} - log M_n."""
    return np.array([math.log(counts[n + 1]) - math.log(counts[n]) for n in range(len(counts) - 1)])


def centered_moving_average(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x positions and a simple moving average for plotted period means."""
    if width <= 1:
        return np.arange(len(values)), values
    average = np.convolve(values, np.ones(width) / width, mode="valid")
    x = np.arange(width - 1, width - 1 + len(average))
    return x, average
