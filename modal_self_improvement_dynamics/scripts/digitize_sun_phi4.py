"""Extract and fit the Phi-4 training trajectories from Sun et al. Figure 4.

The arXiv source contains the plotted scatter points as vector paths but no
numerical data files.  Convert the target PDF to SVG (for example with
``pdftocairo -svg``), then pass that SVG to this script.  The extraction is
exact at the vector-graphic level; it is still figure-derived rather than raw
seed-level data and should only be used for an exploratory audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from fit_modal_models import DEFAULT_MODE_GRID, fit_modal_batch


COLORS = {
    "solver": "rgb(12.156677%,46.665955%,70.587158%)",
    "verifier": "rgb(83.920288%,15.293884%,15.686035%)",
    "gap": "rgb(100%,79.998779%,0%)",
}
PANELS = ("math_qe", "gsm8k_qe", "math_tf", "gsm8k_tf")
PANEL_EDGES = (50.0, 400.0, 750.0, 1100.0, 1450.0)
PLOT_TOP = 26.855469
PLOT_BOTTOM = 303.460938

MARKER_PATTERN = re.compile(
    r'<path style="(?P<style>[^"]*fill-opacity:0\.8[^"]*)" '
    r'd="[^"]*" transform="matrix\([^,]+,[^,]+,[^,]+,[^,]+,'
    r'(?P<x>[-+0-9.eE]+),(?P<y>[-+0-9.eE]+)\)"/>'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/sun_phi4_figure4_digitized.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/sun_phi4_figure4_modal_audit.json"),
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=Path("results/sun_phi4_figure4_modal_summary.csv"),
    )
    parser.add_argument("--train-count", type=int, default=15)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def panel_for_x(x: float) -> str | None:
    for index, (left, right) in enumerate(zip(PANEL_EDGES[:-1], PANEL_EDGES[1:])):
        if left <= x < right:
            return PANELS[index]
    return None


def extract_markers(svg_text: str) -> dict[tuple[str, str], list[tuple[float, float]]]:
    points: dict[tuple[str, str], list[tuple[float, float]]] = {
        (panel, series): [] for panel in PANELS for series in COLORS
    }
    for match in MARKER_PATTERN.finditer(svg_text):
        style = match.group("style")
        series = next((name for name, color in COLORS.items() if color in style), None)
        if series is None:
            continue
        x = float(match.group("x"))
        y = float(match.group("y"))
        panel = panel_for_x(x)
        if panel is not None:
            points[(panel, series)].append((x, y))

    for key, values in points.items():
        values.sort()
        if len(values) != 20:
            raise ValueError(f"expected 20 vector markers for {key}, found {len(values)}")
    return points


def normalized_uncertainty(svg_y: float) -> float:
    """Affine map from SVG coordinates to a common decreasing response scale."""

    return (PLOT_BOTTOM - svg_y) / (PLOT_BOTTOM - PLOT_TOP)


def fit_shared_rank_one(
    panel_values: np.ndarray, train_count: int
) -> dict[str, object]:
    """Fit one common mode after equal-range normalization of each series."""

    checkpoints = np.arange(panel_values.shape[1], dtype=float)
    series_minima = np.min(panel_values, axis=1, keepdims=True)
    series_ranges = np.ptp(panel_values, axis=1, keepdims=True)
    if np.any(series_ranges <= 0.0):
        raise ValueError("shared-mode audit requires nonconstant series")
    normalized = (panel_values - series_minima) / series_ranges
    best: dict[str, object] | None = None
    for mode in DEFAULT_MODE_GRID:
        train_design = np.column_stack(
            (np.ones(train_count), mode ** checkpoints[:train_count])
        )
        coefficients = np.linalg.lstsq(
            train_design, normalized[:, :train_count].T, rcond=None
        )[0].T
        endpoints = coefficients[:, 0]
        amplitudes = coefficients[:, 1]
        lower = -np.ones(panel_values.shape[0])
        upper = 2.0 * np.ones(panel_values.shape[0])
        valid = np.all(amplitudes >= -1e-10) and np.all(
            (endpoints >= lower) & (endpoints <= upper)
        )
        if not valid:
            continue
        residual = normalized[:, :train_count].T - train_design @ coefficients.T
        train_sse = float(np.sum(residual**2))
        if best is None or train_sse < best["train_sse"]:
            forecast_design = np.column_stack(
                (
                    np.ones(panel_values.shape[1] - train_count),
                    mode ** checkpoints[train_count:],
                )
            )
            forecast = (forecast_design @ coefficients.T).T
            best = {
                "mode_per_half_epoch": mode,
                "mode_per_epoch": mode**2,
                "endpoints": endpoints.tolist(),
                "amplitudes": amplitudes.tolist(),
                "train_sse": train_sse,
                "forecast": forecast.tolist(),
                "heldout_mse": float(
                    np.mean((forecast - normalized[:, train_count:]) ** 2)
                ),
                "scale": "each series affinely normalized to its observed range",
            }
    if best is None:
        raise RuntimeError("no valid shared rank-one candidate")
    return best


def fit_series(
    values: np.ndarray,
    train_count: int,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    if not 4 <= train_count < values.size:
        raise ValueError("train-count must leave held-out checkpoints")
    checkpoints = np.arange(values.size, dtype=float)
    observed_range = float(np.ptp(values))
    padding = observed_range if observed_range > 0.0 else 1.0
    endpoint_bounds = (
        float(np.min(values) - padding),
        float(np.max(values) + padding),
    )
    train_values = values[:train_count]
    heldout_values = values[train_count:]
    models: dict[str, object] = {}
    for order in (1, 2):
        fit = fit_modal_batch(
            train_values,
            checkpoints[:train_count],
            checkpoints[train_count:],
            order,
            mode_grid=DEFAULT_MODE_GRID,
            endpoint_bounds=endpoint_bounds,
            direction="decreasing",
        )
        error = fit.forecast[0] - heldout_values
        models[f"rank_{order}"] = {
            "endpoint": float(fit.endpoints[0]),
            "modes_per_half_epoch": fit.modes[0].tolist(),
            "modes_per_epoch": (fit.modes[0] ** 2).tolist(),
            "amplitudes": fit.amplitudes[0].tolist(),
            "train_sse": float(fit.train_sse[0]),
            "heldout_mse": float(np.mean(error**2)),
            "forecast": fit.forecast[0].tolist(),
        }
    mse_1 = float(models["rank_1"]["heldout_mse"])
    mse_2 = float(models["rank_2"]["heldout_mse"])
    observed_score = float(np.log((mse_1 + 1e-15) / (mse_2 + 1e-15)))

    full_null = fit_modal_batch(
        values,
        checkpoints,
        checkpoints,
        1,
        mode_grid=DEFAULT_MODE_GRID,
        endpoint_bounds=endpoint_bounds,
        direction="decreasing",
    )
    null_mean = full_null.forecast[0]
    null_residual = values - null_mean
    null_noise_sd = float(
        np.sqrt(np.sum(null_residual**2) / max(1, values.size - 3))
    )
    simulated = null_mean[None, :] + rng.normal(
        0.0, null_noise_sd, size=(bootstrap_replicates, values.size)
    )
    bootstrap_mse: dict[int, np.ndarray] = {}
    for order in (1, 2):
        fit = fit_modal_batch(
            simulated[:, :train_count],
            checkpoints[:train_count],
            checkpoints[train_count:],
            order,
            mode_grid=DEFAULT_MODE_GRID,
            endpoint_bounds=endpoint_bounds,
            direction="decreasing",
        )
        bootstrap_mse[order] = np.mean(
            (fit.forecast - simulated[:, train_count:]) ** 2, axis=1
        )
    bootstrap_scores = np.log(
        (bootstrap_mse[1] + 1e-15) / (bootstrap_mse[2] + 1e-15)
    )
    critical_value = float(np.quantile(bootstrap_scores, 0.95))
    p_value = float(
        (1 + np.count_nonzero(bootstrap_scores >= observed_score))
        / (bootstrap_replicates + 1)
    )
    return {
        "models": models,
        "log_heldout_mse_ratio_rank1_over_rank2": observed_score,
        "parametric_bootstrap": {
            "replicates": bootstrap_replicates,
            "rank1_null_noise_sd": null_noise_sd,
            "critical_value_5_percent": critical_value,
            "one_sided_p_value": p_value,
            "prefer_rank2_at_5_percent": observed_score > critical_value,
            "qualification": (
                "conditional on a fitted rank-one null, iid Gaussian residuals, "
                "the fixed modal grid, and vector-figure values"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    svg_bytes = args.svg.read_bytes()
    svg_text = svg_bytes.decode("utf-8")
    points = extract_markers(svg_text)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, object]] = []
    panel_values: dict[str, dict[str, np.ndarray]] = {panel: {} for panel in PANELS}
    audit: dict[str, object] = {
        "source": {
            "paper": "Sun et al., arXiv:2507.00075v4",
            "figure_file": "fig/fit/dataset_exponential_fit_train_Phi_4.pdf",
            "input_svg": str(args.svg),
            "input_svg_sha256": hashlib.sha256(svg_bytes).hexdigest(),
            "status": "exploratory vector-figure extraction; not raw seed-level data",
        },
        "checkpoint_spacing_epochs": 0.5,
        "train_count": args.train_count,
        "heldout_count": 20 - args.train_count,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "fits": {},
    }
    for panel in PANELS:
        for series in COLORS:
            marker_values = points[(panel, series)]
            values = np.asarray(
                [normalized_uncertainty(y) for _, y in marker_values], dtype=float
            )
            panel_values[panel][series] = values
            for index, ((svg_x, svg_y), value) in enumerate(
                zip(marker_values, values), start=1
            ):
                rows.append(
                    {
                        "panel": panel,
                        "series": series,
                        "checkpoint": index,
                        "epoch": 0.5 * index,
                        "svg_x": svg_x,
                        "svg_y": svg_y,
                        "affine_uncertainty": value,
                    }
                )
            audit["fits"][f"{panel}__{series}"] = fit_series(
                values, args.train_count, args.bootstrap_replicates, rng
            )

    p_values = {
        name: result["parametric_bootstrap"]["one_sided_p_value"]
        for name, result in audit["fits"].items()
    }
    ordered = sorted(p_values, key=p_values.get)
    holm_rejections: set[str] = set()
    family_size = len(ordered)
    for rank, name in enumerate(ordered):
        if p_values[name] <= 0.05 / (family_size - rank):
            holm_rejections.add(name)
        else:
            break
    for name, result in audit["fits"].items():
        result["parametric_bootstrap"]["holm_bonferroni_5_percent"] = (
            name in holm_rejections
        )
    audit["multiple_testing"] = {
        "method": "Holm-Bonferroni",
        "family_size": family_size,
        "familywise_level": 0.05,
        "rejected_series": sorted(holm_rejections),
    }

    audit["shared_mode_audit"] = {}
    for panel in PANELS:
        values = np.vstack([panel_values[panel][series] for series in COLORS])
        shared = fit_shared_rank_one(values, args.train_count)
        separate_mse: dict[int, float] = {}
        for order in (1, 2):
            errors = []
            for row, series in enumerate(COLORS):
                forecast = np.asarray(
                    audit["fits"][f"{panel}__{series}"]["models"][
                        f"rank_{order}"
                    ]["forecast"]
                )
                errors.append(
                    (forecast - values[row, args.train_count :])
                    / np.ptp(values[row])
                )
            separate_mse[order] = float(np.mean(np.concatenate(errors) ** 2))
        audit["shared_mode_audit"][panel] = {
            "shared_rank1": shared,
            "separate_rank1_heldout_mse": separate_mse[1],
            "separate_rank2_heldout_mse": separate_mse[2],
            "log_mse_ratio_shared_rank1_over_separate_rank1": float(
                np.log(
                    (shared["heldout_mse"] + 1e-15)
                    / (separate_mse[1] + 1e-15)
                )
            ),
            "qualification": (
                "descriptive only; a joint test requires seed-level covariance, "
                "and the plotted gap is algebraically related to solver and verifier"
            ),
        }

    summary_rows: list[dict[str, object]] = []
    for name, result in audit["fits"].items():
        rank_1 = result["models"]["rank_1"]
        rank_2 = result["models"]["rank_2"]
        bootstrap = result["parametric_bootstrap"]
        summary_rows.append(
            {
                "series": name,
                "log_heldout_mse_ratio_rank1_over_rank2": result[
                    "log_heldout_mse_ratio_rank1_over_rank2"
                ],
                "rank1_heldout_mse": rank_1["heldout_mse"],
                "rank2_heldout_mse": rank_2["heldout_mse"],
                "rank1_mode_per_epoch": rank_1["modes_per_epoch"][0],
                "rank2_modes_per_epoch": ";".join(
                    str(value) for value in rank_2["modes_per_epoch"]
                ),
                "bootstrap_p_value": bootstrap["one_sided_p_value"],
                "uncorrected_rank2_at_5_percent": bootstrap[
                    "prefer_rank2_at_5_percent"
                ],
                "holm_bonferroni_rank2_at_5_percent": bootstrap[
                    "holm_bonferroni_5_percent"
                ],
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    ratios = {
        name: result["log_heldout_mse_ratio_rank1_over_rank2"]
        for name, result in audit["fits"].items()
    }
    print(json.dumps({"rows": len(rows), "log_mse_ratios": ratios}, indent=2))


if __name__ == "__main__":
    main()
