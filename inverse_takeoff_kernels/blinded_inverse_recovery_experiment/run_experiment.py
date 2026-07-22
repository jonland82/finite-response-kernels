"""Blinded inverse-recovery benchmark for velocity takeoff kernels."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import os
import platform
import random
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy


ALPHA = math.log(2.0)
RHO = 0.5
FIELDNAMES = [
    "worker",
    "schema_index",
    "family",
    "schema",
    "max_lag",
    "branch_count",
    "window",
    "window_start",
    "regime",
    "horizon_ratio",
    "horizon_bin",
    "noise",
    "true_radius",
    "true_class",
    "estimated_order",
    "recovered_radius",
    "recovered_class",
    "radius_error",
    "class_correct",
    "forecast_nrmse",
    "forecast_success",
    "signal_rms",
]


@dataclass(frozen=True)
class Config:
    output_dir: str
    workers: int
    max_schemas_per_worker: int
    worker_seconds: float
    seed: int
    windows: tuple[int, ...]
    noise_levels: tuple[float, ...]
    holdout: int
    max_order: int


def schema_text(schema: dict[int, int]) -> str:
    return ";".join(f"{lag}:{schema[lag]}" for lag in sorted(schema))


def fixed_speed_check(schema: dict[int, int]) -> bool:
    degree = max(schema)
    return sum(count * (1 << (degree - lag)) for lag, count in schema.items()) == (1 << degree)


def generate_schema(rng: random.Random) -> tuple[str, dict[int, int]]:
    draw = rng.random()
    if draw < 0.03:
        return "immediate", {1: 2}
    if draw < 0.28:
        lag = rng.randint(4, 24)
        return "slow", {lag: (1 << lag) - 1, lag + 1: 2}
    if draw < 0.53:
        lag = rng.randint(4, 32)
        upper = ((1 << (lag + 1)) - 1) // 3
        t = rng.randint(1, max(1, upper))
        return "finite_horizon", {
            lag: t,
            lag + 1: (1 << (lag + 1)) - 3 * t,
            lag + 2: 2 * t,
        }
    if draw < 0.73:
        lag = rng.randint(4, 24)
        return "two_scale", {1: 1, lag: 1 << (lag - 1)}

    max_lag = rng.randint(4, 20)
    schema = {1: 2}
    steps = rng.randint(max_lag, 8 * max_lag)
    for _ in range(steps):
        choices = [lag for lag, count in schema.items() if count > 0 and lag < max_lag]
        if not choices:
            break
        lag = rng.choice(choices)
        schema[lag] -= 1
        if schema[lag] == 0:
            del schema[lag]
        schema[lag + 1] = schema.get(lag + 1, 0) + 2
    if math.gcd(*schema.keys()) != 1:
        return "two_scale", {1: 1, max_lag: 1 << (max_lag - 1)}
    return "random_split", schema


def tree_profile(schema: dict[int, int], nmax: int) -> np.ndarray:
    sizes: list[int] = []
    for n in range(nmax + 2):
        if n == 0:
            sizes.append(1)
            continue
        total = 1
        for lag, count in schema.items():
            previous = n - lag
            total += count * (sizes[previous] if previous >= 0 else 1)
        sizes.append(total)
    velocity = np.asarray(
        [math.log(sizes[n + 1]) - math.log(sizes[n]) for n in range(nmax + 1)],
        dtype=float,
    )
    return velocity / ALPHA


def true_modal_structure(schema: dict[int, int]) -> tuple[float, str]:
    degree = max(schema)
    coefficients = np.zeros(degree + 1, dtype=float)
    coefficients[0] = 1.0
    for lag, count in schema.items():
        coefficients[lag] = -float(count)
    roots = list(np.roots(coefficients[::-1]))
    dominant = min(range(len(roots)), key=lambda index: abs(roots[index] - RHO))
    singularities = [root for index, root in enumerate(roots) if index != dominant]
    singularities.append(1.0 + 0.0j)
    poles = [RHO / root for root in singularities if abs(root) > 1e-14]
    leading = max(poles, key=abs)
    return float(abs(leading)), classify_pole(leading)


def classify_pole(pole: complex) -> str:
    tolerance = 1e-5 * (1.0 + abs(pole.real))
    if abs(pole.imag) <= tolerance:
        return "monotone" if pole.real >= 0.0 else "alternating"
    return "ringing"


def _fit_poles(values: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray] | None:
    sample_count = len(values)
    rows = sample_count // 2
    columns = sample_count - rows
    if order < 1 or rows < order or columns < order or sample_count < 2 * order + 2:
        return None
    h0 = np.empty((rows, columns), dtype=float)
    h1 = np.empty((rows, columns), dtype=float)
    for row in range(rows):
        h0[row] = values[row : row + columns]
        h1[row] = values[row + 1 : row + columns + 1]
    try:
        u, singular, vh = np.linalg.svd(h0, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if len(singular) < order or singular[order - 1] <= max(singular[0], 1.0) * 1e-14:
        return None
    ur = u[:, :order]
    vr = vh.conj().T[:, :order]
    reduced = (ur.conj().T @ h1 @ vr) @ np.diag(1.0 / singular[:order])
    try:
        poles = np.linalg.eigvals(reduced)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(poles)) or np.max(np.abs(poles)) > 1.25:
        return None
    indices = np.arange(sample_count, dtype=float)[:, None]
    vandermonde = poles[None, :] ** indices
    try:
        amplitudes = np.linalg.lstsq(vandermonde, values, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return poles, amplitudes


def _predict(poles: np.ndarray, amplitudes: np.ndarray, indices: np.ndarray) -> np.ndarray:
    prediction = (poles[None, :] ** indices[:, None]) @ amplitudes
    return np.real(prediction)


def recover_modes(values: np.ndarray, max_order: int, forecast_count: int) -> tuple[int, np.ndarray, np.ndarray]:
    if len(values) < 6 or float(np.sqrt(np.mean(values * values))) < 1e-14:
        return 0, np.asarray([], dtype=complex), np.zeros(forecast_count)
    split = min(len(values) - 2, max(4, int(0.75 * len(values))))
    validation = values[split:]
    best: tuple[float, int] | None = None
    order_limit = min(max_order, max(1, split // 3))
    for order in range(1, order_limit + 1):
        fitted = _fit_poles(values[:split], order)
        if fitted is None:
            continue
        poles, amplitudes = fitted
        prediction = _predict(poles, amplitudes, np.arange(split, len(values), dtype=float))
        mse = float(np.mean((prediction - validation) ** 2))
        score = len(validation) * math.log(mse + 1e-24) + 4.0 * order * math.log(len(values))
        if best is None or score < best[0]:
            best = (score, order)
    if best is None:
        return 0, np.asarray([], dtype=complex), np.zeros(forecast_count)
    order = best[1]
    fitted = _fit_poles(values, order)
    if fitted is None:
        return 0, np.asarray([], dtype=complex), np.zeros(forecast_count)
    poles, amplitudes = fitted
    forecast = _predict(
        poles,
        amplitudes,
        np.arange(len(values), len(values) + forecast_count, dtype=float),
    )
    return order, poles, forecast


def verify_finite_horizon_family() -> dict[str, object]:
    lag = 9
    parameters = (3, 37, 101)
    schemas = [
        {lag: t, lag + 1: (1 << (lag + 1)) - 3 * t, lag + 2: 2 * t}
        for t in parameters
    ]
    profiles = [tree_profile(schema, lag + 3) for schema in schemas]
    prefix_equal = all(np.array_equal(profiles[0][:lag], profile[:lag]) for profile in profiles[1:])
    later_separates = any(not np.array_equal(profiles[0][lag:], profile[lag:]) for profile in profiles[1:])
    checks = [fixed_speed_check(schema) for schema in schemas]
    if not prefix_equal or not later_separates or not all(checks):
        raise AssertionError("finite-horizon construction failed its exact validation")
    return {
        "lag": lag,
        "parameters": list(parameters),
        "fixed_speed_checks": checks,
        "prefix_through_F": lag - 1,
        "prefix_equal": prefix_equal,
        "later_separates": later_separates,
    }


def run_worker(worker: int, config: Config) -> dict[str, object]:
    rng = random.Random(config.seed + 1_000_003 * worker)
    np_rng = np.random.default_rng(config.seed + 2_000_003 * worker)
    output = Path(config.output_dir) / "raw" / f"worker_{worker:03d}.csv.gz"
    started = time.monotonic()
    deadline = started + config.worker_seconds
    schema_count = 0
    trial_count = 0
    families: Counter[str] = Counter()
    with gzip.open(output, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        while schema_count < config.max_schemas_per_worker and time.monotonic() < deadline:
            family, schema = generate_schema(rng)
            if not fixed_speed_check(schema):
                raise AssertionError(f"schema lost fixed speed: {schema}")
            max_lag = max(schema)
            branch_count = sum(schema.values())
            true_radius, true_class = true_modal_structure(schema)
            max_start = max_lag
            nmax = max_start + max(config.windows) + config.holdout + 2
            profile = tree_profile(schema, nmax)
            y_full = profile - 1.0
            for regime, start in (("origin", 0), ("post_lag", max_lag)):
                for window in config.windows:
                    horizon_ratio = window / max_lag
                    if horizon_ratio < 0.5:
                        horizon_bin = "lt_0.5"
                    elif horizon_ratio < 1.0:
                        horizon_bin = "0.5_to_1"
                    elif horizon_ratio < 2.0:
                        horizon_bin = "1_to_2"
                    else:
                        horizon_bin = "ge_2"
                    clean = y_full[start : start + window]
                    truth = y_full[start + window : start + window + config.holdout]
                    signal_rms = float(np.sqrt(np.mean(clean * clean)))
                    for noise in config.noise_levels:
                        noise_scale = noise * max(signal_rms, 1e-12)
                        observed = clean + np_rng.normal(0.0, noise_scale, size=len(clean))
                        order, poles, forecast = recover_modes(observed, config.max_order, config.holdout)
                        if len(poles):
                            leading = poles[int(np.argmax(np.abs(poles)))]
                            recovered_radius = float(abs(leading))
                            recovered_class = classify_pole(leading)
                        else:
                            recovered_radius = 0.0
                            recovered_class = "unresolved"
                        denominator = max(
                            float(np.sqrt(np.mean(truth * truth))), signal_rms, 1e-12
                        )
                        forecast_nrmse = float(
                            np.sqrt(np.mean((forecast - truth) ** 2)) / denominator
                        )
                        row = {
                            "worker": worker,
                            "schema_index": schema_count,
                            "family": family,
                            "schema": schema_text(schema),
                            "max_lag": max_lag,
                            "branch_count": branch_count,
                            "window": window,
                            "window_start": start,
                            "regime": regime,
                            "horizon_ratio": f"{horizon_ratio:.12g}",
                            "horizon_bin": horizon_bin,
                            "noise": f"{noise:.12g}",
                            "true_radius": f"{true_radius:.12g}",
                            "true_class": true_class,
                            "estimated_order": order,
                            "recovered_radius": f"{recovered_radius:.12g}",
                            "recovered_class": recovered_class,
                            "radius_error": f"{abs(recovered_radius - true_radius):.12g}",
                            "class_correct": int(recovered_class == true_class),
                            "forecast_nrmse": f"{forecast_nrmse:.12g}",
                            "forecast_success": int(forecast_nrmse < 0.1),
                            "signal_rms": f"{signal_rms:.12g}",
                        }
                        writer.writerow(row)
                        trial_count += 1
            schema_count += 1
            families[family] += 1
    return {
        "worker": worker,
        "schemas": schema_count,
        "trials": trial_count,
        "elapsed_seconds": time.monotonic() - started,
        "families": dict(families),
        "path": str(output),
    }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    values.sort()
    index = min(len(values) - 1, max(0, round(probability * (len(values) - 1))))
    return values[index]


def analyze(config: Config, worker_results: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str, int, str, str], dict[str, object]] = {}
    global_radius: list[float] = []
    global_forecast: list[float] = []
    total = 0
    class_correct = 0
    forecast_success = 0
    hard: list[tuple[float, int, dict[str, str]]] = []
    serial = 0
    for result in worker_results:
        with gzip.open(str(result["path"]), "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total += 1
                correct = int(row["class_correct"])
                success = int(row["forecast_success"])
                radius_error = float(row["radius_error"])
                forecast_error = float(row["forecast_nrmse"])
                class_correct += correct
                forecast_success += success
                if len(global_radius) < 100_000:
                    global_radius.append(radius_error)
                    global_forecast.append(forecast_error)
                key = (
                    row["family"],
                    row["regime"],
                    int(row["window"]),
                    row["noise"],
                    row["horizon_bin"],
                )
                if key not in groups:
                    groups[key] = {
                        "count": 0,
                        "class_correct": 0,
                        "forecast_success": 0,
                        "radius": [],
                        "forecast": [],
                    }
                group = groups[key]
                group["count"] = int(group["count"]) + 1
                group["class_correct"] = int(group["class_correct"]) + correct
                group["forecast_success"] = int(group["forecast_success"]) + success
                if len(group["radius"]) < 5_000:
                    group["radius"].append(radius_error)
                    group["forecast"].append(forecast_error)
                serial += 1
                item = (forecast_error, serial, row)
                if len(hard) < 200:
                    heapq.heappush(hard, item)
                elif forecast_error > hard[0][0]:
                    heapq.heapreplace(hard, item)

    summary_rows: list[dict[str, object]] = []
    for (family, regime, window, noise, horizon_bin), group in sorted(groups.items()):
        count = int(group["count"])
        summary_rows.append(
            {
                "family": family,
                "regime": regime,
                "window": window,
                "noise": noise,
                "horizon_bin": horizon_bin,
                "count": count,
                "class_accuracy": int(group["class_correct"]) / count,
                "forecast_success_rate": int(group["forecast_success"]) / count,
                "median_radius_error": percentile(group["radius"], 0.5),
                "median_forecast_nrmse": percentile(group["forecast"], 0.5),
                "p90_forecast_nrmse": percentile(group["forecast"], 0.9),
            }
        )
    output_dir = Path(config.output_dir)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    with (output_dir / "hard_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for _, _, row in sorted(hard, reverse=True):
            writer.writerow(row)
    summary = {
        "total_trials": total,
        "total_schemas": sum(int(result["schemas"]) for result in worker_results),
        "class_accuracy": class_correct / total if total else float("nan"),
        "forecast_success_rate": forecast_success / total if total else float("nan"),
        "median_radius_error": percentile(global_radius, 0.5),
        "median_forecast_nrmse": percentile(global_forecast, 0.5),
        "workers": worker_results,
        "group_count": len(summary_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Blinded inverse-recovery run",
        "",
        f"- Schemas: {summary['total_schemas']:,}",
        f"- Recovery trials: {summary['total_trials']:,}",
        f"- Takeoff-class accuracy: {summary['class_accuracy']:.3f}",
        f"- Forecast success rate (NRMSE < 0.1): {summary['forecast_success_rate']:.3f}",
        f"- Median leading-radius error: {summary['median_radius_error']:.4g}",
        f"- Median holdout NRMSE: {summary['median_forecast_nrmse']:.4g}",
        "",
        "See `summary.csv` for the family/window/noise phase diagram and",
        "`hard_cases.csv` for the largest forecast failures.",
    ]
    (output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--max-schemas-per-worker", type=int, default=100_000)
    parser.add_argument("--worker-seconds", type=float, default=1_200.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--windows", type=parse_int_tuple, default=(8, 16, 32, 64))
    parser.add_argument(
        "--noise-levels", type=parse_float_tuple, default=(0.0, 1e-8, 1e-5, 1e-3)
    )
    parser.add_argument("--holdout", type=int, default=24)
    parser.add_argument("--max-order", type=int, default=6)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    config = Config(
        output_dir=str(output_dir),
        workers=args.workers,
        max_schemas_per_worker=args.max_schemas_per_worker,
        worker_seconds=args.worker_seconds,
        seed=args.seed,
        windows=tuple(args.windows),
        noise_levels=tuple(args.noise_levels),
        holdout=args.holdout,
        max_order=args.max_order,
    )
    construction_check = verify_finite_horizon_family()
    metadata = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "aws_instance_id": os.environ.get("AWS_INSTANCE_ID"),
        "aws_region": os.environ.get("AWS_REGION"),
        "finite_horizon_construction_check": construction_check,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(run_worker, worker, config) for worker in range(config.workers)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"worker={result['worker']} schemas={result['schemas']} "
                f"trials={result['trials']} elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    results.sort(key=lambda item: int(item["worker"]))
    summary = analyze(config, results)
    metadata["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["summary"] = {key: value for key, value in summary.items() if key != "workers"}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
