"""Sharp-boundary comparison for blinded inverse takeoff recovery.

Compares a generic matrix-pencil forecast, a nonnegative continuous source fit,
and a constraint-aware integer source decoder.  The experiment is centered on
the sharp finite-horizon boundary for schemas of known maximum lag D:

    F_0,...,F_{D-3} can be ambiguous, while F_0,...,F_{D-2} identify the schema.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, milp

from run_experiment import ALPHA, fixed_speed_check, recover_modes, schema_text, tree_profile


METHODS = ("constraint_integer", "continuous_round", "matrix_pencil_6")
FIELDNAMES = [
    "worker", "schema_index", "family", "schema", "max_lag", "branch_count",
    "horizon", "horizon_offset", "horizon_ratio", "identifiability", "noise",
    "method", "status", "exact_source", "support_f1", "coefficient_rel_l1",
    "dyadic_valid", "forecast_nrmse", "forecast_success", "estimated_order",
    "theoretical_version_count", "runtime_ms",
]


@dataclass(frozen=True)
class Config:
    output_dir: str
    workers: int
    max_schemas_per_worker: int
    worker_seconds: float
    seed: int
    max_lags: tuple[int, ...]
    offsets: tuple[int, ...]
    noise_levels: tuple[float, ...]
    holdout: int
    milp_seconds: float


def exact_counts(schema: dict[int, int], nmax: int) -> list[int]:
    """Return exact call-tree sizes N_0,...,N_nmax."""
    counts = [1]
    for n in range(1, nmax + 1):
        counts.append(1 + sum(a * counts[n - lag] if n >= lag else a
                              for lag, a in schema.items()))
    return counts


def fixed_lag_schema(rng: random.Random, family: str, degree: int) -> dict[int, int]:
    if family == "finite_horizon":
        lag = degree - 2
        upper = ((1 << (lag + 1)) - 1) // 3
        t = rng.randint(1, upper)
        return {lag: t, lag + 1: (1 << (lag + 1)) - 3 * t, degree: 2 * t}
    if family == "slow":
        return {degree - 1: (1 << (degree - 1)) - 1, degree: 2}
    if family == "two_scale":
        return {1: 1, degree: 1 << (degree - 1)}
    if family != "random_split":
        raise ValueError(f"unknown family: {family}")

    # Splitting one lag-j branch into two lag-(j+1) branches preserves
    # sum_j a_j 2^{-j}=1.  First force a path to D, then randomize the rest.
    schema = {1: 2}
    for lag in range(1, degree):
        schema[lag] -= 1
        if schema[lag] == 0:
            del schema[lag]
        schema[lag + 1] = schema.get(lag + 1, 0) + 2
    for _ in range(rng.randint(degree, 5 * degree)):
        choices = [lag for lag, count in schema.items() if count and lag < degree]
        if not choices:
            break
        lag = rng.choice(choices)
        schema[lag] -= 1
        if schema[lag] == 0:
            del schema[lag]
        schema[lag + 1] = schema.get(lag + 1, 0) + 2
    return schema


def schema_vector(schema: dict[int, int], degree: int) -> np.ndarray:
    return np.asarray([schema.get(lag, 0) for lag in range(1, degree + 1)], dtype=np.int64)


def response_increments(observed_f: np.ndarray) -> np.ndarray:
    log_counts = np.concatenate(([0.0], ALPHA * np.cumsum(observed_f)))
    counts = np.exp(np.clip(log_counts, -700.0, 700.0))
    return np.diff(counts)


def inverse_equations(observed_f: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear equations in a_j after reconstructing approximate increments."""
    increments = response_increments(observed_f)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    first = np.ones(degree, dtype=float)
    rows.append(first)
    targets.append(float(increments[0]))
    for n in range(2, len(increments) + 1):
        row = np.zeros(degree, dtype=float)
        for lag in range(1, min(degree, n - 1) + 1):
            row[lag - 1] = increments[n - lag - 1]
        rows.append(row)
        targets.append(float(increments[n - 1]))
    matrix = np.vstack(rows)
    target = np.asarray(targets)
    scales = np.maximum.reduce((np.max(np.abs(matrix), axis=1), np.abs(target), np.ones(len(target))))
    return matrix / scales[:, None], target / scales


def continuous_decode(observed_f: np.ndarray, degree: int) -> tuple[np.ndarray | None, str]:
    matrix, target = inverse_equations(observed_f, degree)
    upper = np.asarray([float(1 << lag) for lag in range(1, degree + 1)])
    try:
        fitted = lsq_linear(matrix, target, bounds=(np.zeros(degree), upper),
                            lsmr_tol="auto", max_iter=200)
    except (ValueError, np.linalg.LinAlgError):
        return None, "failed"
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return None, "failed"
    return np.rint(fitted.x).astype(np.int64), "solved"


def constrained_decode(
    observed_f: np.ndarray, degree: int, time_limit: float
) -> tuple[np.ndarray | None, str]:
    """Integer L1 fit with the exact dyadic branching invariant."""
    matrix, target = inverse_equations(observed_f, degree)
    equation_count = len(target)
    variable_count = degree + equation_count
    objective = np.concatenate((np.zeros(degree), np.full(equation_count, 1.0 / equation_count)))
    integrality = np.concatenate((np.ones(degree), np.zeros(equation_count)))
    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.concatenate((
        np.asarray([float(1 << lag) for lag in range(1, degree + 1)]),
        np.full(equation_count, np.inf),
    ))

    constraint_rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for index, (row, value) in enumerate(zip(matrix, target)):
        positive = np.zeros(variable_count)
        positive[:degree] = row
        positive[degree + index] = -1.0
        constraint_rows.append(positive)
        lower.append(-np.inf)
        upper.append(float(value))
        constraint_rows.append(-positive)
        # -row*a-residual <= -target, not the negation of the full first row.
        constraint_rows[-1][degree + index] = -1.0
        lower.append(-np.inf)
        upper.append(float(-value))

    dyadic = np.zeros(variable_count)
    dyadic[:degree] = np.asarray([float(1 << (degree - lag)) for lag in range(1, degree + 1)])
    constraint_rows.append(dyadic)
    lower.append(float(1 << degree))
    upper.append(float(1 << degree))
    try:
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=LinearConstraint(np.vstack(constraint_rows), np.asarray(lower), np.asarray(upper)),
            options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True},
        )
    except (ValueError, RuntimeError):
        return None, "failed"
    if result.x is None or not np.all(np.isfinite(result.x[:degree])):
        return None, "timeout" if result.status == 1 else "failed"
    status = "solved" if result.success else "feasible_timeout"
    return np.rint(result.x[:degree]).astype(np.int64), status


def exact_boundary_decode(schema: dict[int, int], degree: int) -> dict[int, int]:
    """Constructive proof decoder from the exact D-1-sample count prefix."""
    counts = exact_counts(schema, degree - 1)
    increments = [counts[n] - counts[n - 1] for n in range(1, degree)]
    total_branches = increments[0]
    decoded: dict[int, int] = {}
    for lag in range(1, degree - 1):
        numerator = increments[lag] - sum(
            decoded[j] * increments[lag - j] for j in range(1, lag)
        )
        if numerator % increments[0]:
            raise AssertionError("nonintegral triangular recovery")
        decoded[lag] = numerator // increments[0]
    tail_sum = total_branches - sum(decoded.values())
    used_mass = sum(a * (1 << (degree - lag)) for lag, a in decoded.items())
    tail_mass = (1 << degree) - used_mass
    decoded[degree - 1] = tail_mass - tail_sum
    decoded[degree] = 2 * tail_sum - tail_mass
    return {lag: a for lag, a in decoded.items() if a}


def ambiguity_count_at_d_minus_2(schema: dict[int, int], degree: int) -> int:
    """Number of nonnegative integer tails at lags D-2,D-1,D."""
    known = {lag: a for lag, a in schema.items() if lag <= degree - 3}
    tail_sum = sum(schema.values()) - sum(known.values())
    tail_mass = (1 << degree) - sum(
        a * (1 << (degree - lag)) for lag, a in known.items()
    )
    # 4x+2y+z=M and x+y+z=S imply 3x+y=M-S.
    lower = max(0, math.ceil((tail_mass - 2 * tail_sum) / 2))
    upper = (tail_mass - tail_sum) // 3
    return max(0, upper - lower + 1)


def source_metrics(estimate: np.ndarray | None, truth: np.ndarray) -> tuple[object, object, object, object]:
    if estimate is None:
        return 0, 0.0, 1.0, 0
    exact = int(np.array_equal(estimate, truth))
    true_support = truth > 0
    estimated_support = estimate > 0
    intersection = int(np.sum(true_support & estimated_support))
    denominator = int(np.sum(true_support) + np.sum(estimated_support))
    support_f1 = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    relative_l1 = float(np.sum(np.abs(estimate.astype(float) - truth))) / max(float(np.sum(truth)), 1.0)
    degree = len(truth)
    dyadic = int(sum(int(estimate[lag - 1]) * (1 << (degree - lag))
                      for lag in range(1, degree + 1)) == (1 << degree))
    return exact, support_f1, relative_l1, dyadic


def source_forecast(estimate: np.ndarray | None, horizon: int, holdout: int) -> np.ndarray:
    if estimate is None:
        return np.zeros(holdout)
    schema = {lag: int(a) for lag, a in enumerate(estimate, start=1) if a > 0}
    if not schema:
        return np.zeros(holdout)
    return tree_profile(schema, horizon + holdout)[horizon:horizon + holdout] - 1.0


def forecast_score(forecast: np.ndarray, truth: np.ndarray, training: np.ndarray) -> tuple[float, int]:
    denominator = max(float(np.sqrt(np.mean(truth * truth))),
                      float(np.sqrt(np.mean(training * training))), 1e-12)
    score = float(np.sqrt(np.mean((forecast - truth) ** 2)) / denominator)
    return score, int(score < 0.1)


def identifiability_label(horizon: int, degree: int) -> str:
    if horizon <= degree - 2:
        return "nonidentifiable"
    if horizon == degree - 1:
        return "sharp_boundary"
    return "overdetermined"


def run_worker(worker: int, config: Config) -> dict[str, object]:
    rng = random.Random(config.seed + 1_000_003 * worker)
    np_rng = np.random.default_rng(config.seed + 2_000_003 * worker)
    output = Path(config.output_dir) / "raw" / f"worker_{worker:03d}.csv.gz"
    started = time.monotonic()
    deadline = started + config.worker_seconds
    schema_count = 0
    row_count = 0
    families: Counter[str] = Counter()
    with gzip.open(output, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        while schema_count < config.max_schemas_per_worker and time.monotonic() < deadline:
            degree = rng.choice(config.max_lags)
            family = rng.choice(("finite_horizon", "slow", "two_scale", "random_split"))
            schema = fixed_lag_schema(rng, family, degree)
            if max(schema) != degree or not fixed_speed_check(schema):
                raise AssertionError(f"invalid generated schema: {schema}")
            if exact_boundary_decode(schema, degree) != schema:
                raise AssertionError(f"sharp-boundary decoder failed: {schema}")
            truth_vector = schema_vector(schema, degree)
            horizons = sorted(set(max(4, degree + offset) for offset in config.offsets) | {2 * degree + 2})
            profile = tree_profile(schema, max(horizons) + config.holdout + 1)
            for horizon in horizons:
                clean_f = profile[:horizon]
                clean_y = clean_f - 1.0
                holdout_truth = profile[horizon:horizon + config.holdout] - 1.0
                signal_rms = max(float(np.sqrt(np.mean(clean_y * clean_y))), 1e-12)
                versions = ambiguity_count_at_d_minus_2(schema, degree) if horizon == degree - 2 else 1
                for noise in config.noise_levels:
                    observed_f = clean_f + np_rng.normal(0.0, noise * signal_rms, size=horizon)
                    observed_y = observed_f - 1.0
                    method_results: list[tuple[str, np.ndarray | None, str, np.ndarray, int, float]] = []

                    began = time.perf_counter()
                    constrained, constrained_status = constrained_decode(observed_f, degree, config.milp_seconds)
                    runtime = 1e3 * (time.perf_counter() - began)
                    method_results.append(("constraint_integer", constrained, constrained_status,
                                           source_forecast(constrained, horizon, config.holdout), 0, runtime))

                    began = time.perf_counter()
                    continuous, continuous_status = continuous_decode(observed_f, degree)
                    runtime = 1e3 * (time.perf_counter() - began)
                    method_results.append(("continuous_round", continuous, continuous_status,
                                           source_forecast(continuous, horizon, config.holdout), 0, runtime))

                    began = time.perf_counter()
                    order, _, pencil_forecast = recover_modes(observed_y, 6, config.holdout)
                    runtime = 1e3 * (time.perf_counter() - began)
                    method_results.append(("matrix_pencil_6", None,
                                           "solved" if order else "unresolved",
                                           pencil_forecast, order, runtime))

                    for method, estimate, status, forecast, order, runtime_ms in method_results:
                        exact, support_f1, coefficient_error, dyadic = source_metrics(estimate, truth_vector)
                        forecast_nrmse, forecast_success = forecast_score(forecast, holdout_truth, clean_y)
                        writer.writerow({
                            "worker": worker, "schema_index": schema_count, "family": family,
                            "schema": schema_text(schema), "max_lag": degree,
                            "branch_count": sum(schema.values()), "horizon": horizon,
                            "horizon_offset": horizon - degree,
                            "horizon_ratio": f"{horizon / degree:.12g}",
                            "identifiability": identifiability_label(horizon, degree),
                            "noise": f"{noise:.12g}", "method": method, "status": status,
                            "exact_source": exact, "support_f1": support_f1,
                            "coefficient_rel_l1": coefficient_error, "dyadic_valid": dyadic,
                            "forecast_nrmse": f"{forecast_nrmse:.12g}",
                            "forecast_success": forecast_success, "estimated_order": order,
                            "theoretical_version_count": versions, "runtime_ms": f"{runtime_ms:.12g}",
                        })
                        row_count += 1
            schema_count += 1
            families[family] += 1
    return {"worker": worker, "schemas": schema_count, "rows": row_count,
            "elapsed_seconds": time.monotonic() - started, "families": dict(families),
            "path": str(output)}


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    values.sort()
    return values[min(len(values) - 1, max(0, round(probability * (len(values) - 1))))]


def analyze(config: Config, worker_results: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str, str], dict[str, object]] = defaultdict(
        lambda: {"count": 0, "source_count": 0, "exact": 0, "support": [],
                 "coefficient": [], "forecast": [], "forecast_success": 0,
                 "runtime": [], "valid": 0}
    )
    total_rows = 0
    for result in worker_results:
        with gzip.open(str(result["path"]), "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total_rows += 1
                key = (row["method"], row["identifiability"], row["noise"])
                group = groups[key]
                group["count"] += 1
                group["forecast"].append(float(row["forecast_nrmse"]))
                group["forecast_success"] += int(row["forecast_success"])
                group["runtime"].append(float(row["runtime_ms"]))
                if row["method"] in ("constraint_integer", "continuous_round"):
                    group["source_count"] += 1
                    group["exact"] += int(row["exact_source"] or 0)
                    group["support"].append(float(row["support_f1"] or 0.0))
                    group["coefficient"].append(float(row["coefficient_rel_l1"] or 1.0))
                    group["valid"] += int(row["dyadic_valid"] or 0)

    summary_rows: list[dict[str, object]] = []
    for (method, identifiability, noise), group in sorted(groups.items()):
        count = int(group["count"])
        source_count = int(group["source_count"])
        summary_rows.append({
            "method": method, "identifiability": identifiability, "noise": noise,
            "count": count,
            "exact_source_rate": (int(group["exact"]) / source_count if source_count else ""),
            "median_support_f1": (percentile(group["support"], 0.5) if source_count else ""),
            "median_coefficient_rel_l1": (percentile(group["coefficient"], 0.5) if source_count else ""),
            "dyadic_valid_rate": (int(group["valid"]) / source_count if source_count else ""),
            "forecast_success_rate": int(group["forecast_success"]) / count,
            "median_forecast_nrmse": percentile(group["forecast"], 0.5),
            "p90_forecast_nrmse": percentile(group["forecast"], 0.9),
            "median_runtime_ms": percentile(group["runtime"], 0.5),
        })
    output = Path(config.output_dir)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    summary = {"schemas": sum(int(item["schemas"]) for item in worker_results),
               "rows": total_rows, "groups": len(summary_rows), "workers": worker_results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    boundary = [row for row in summary_rows if row["identifiability"] in
                ("nonidentifiable", "sharp_boundary") and row["noise"] == "0"]
    report = [
        "# Constraint-aware sharp-boundary comparison", "",
        f"- Schemas: {summary['schemas']:,}", f"- Method trials: {summary['rows']:,}",
        "- Boundary: T = D - 1 observed transitions F_0,...,F_{D-2}.",
        "- Nonidentifiable control: T = D - 2 observed transitions F_0,...,F_{D-3}.", "",
        "## Noiseless boundary snapshot", "",
        "| Method | Regime | Exact source | Forecast success | Median NRMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in boundary:
        exact = row["exact_source_rate"] if row["exact_source_rate"] != "" else "n/a"
        exact_text = f"{exact:.3f}" if isinstance(exact, float) else exact
        report.append(f"| {row['method']} | {row['identifiability']} | {exact_text} | "
                      f"{row['forecast_success_rate']:.3f} | {row['median_forecast_nrmse']:.3g} |")
    report.extend(["", "See `summary.csv` for the full noise-by-regime phase diagram."])
    (output / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def validate_sharp_boundary() -> dict[str, object]:
    degree = 11
    lag = degree - 2
    parameters = (3, 37, 101)
    schemas = [{lag: t, lag + 1: (1 << (lag + 1)) - 3 * t, degree: 2 * t}
               for t in parameters]
    prefixes = [tree_profile(schema, degree - 2)[:degree - 2] for schema in schemas]
    return {
        "degree": degree, "ambiguous_sample_count": degree - 2,
        "identifying_sample_count": degree - 1,
        "ambiguous_prefix_equal": all(np.array_equal(prefixes[0], item) for item in prefixes[1:]),
        "version_counts": [ambiguity_count_at_d_minus_2(schema, degree) for schema in schemas],
        "boundary_decodes_exactly": all(exact_boundary_decode(schema, degree) == schema for schema in schemas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--max-schemas-per-worker", type=int, default=100_000)
    parser.add_argument("--worker-seconds", type=float, default=720.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-lags", type=parse_int_tuple, default=(8, 12, 16, 20))
    parser.add_argument("--offsets", type=parse_int_tuple, default=(-2, -1, 0, 2))
    parser.add_argument("--noise-levels", type=parse_float_tuple, default=(0.0, 1e-8, 1e-5, 1e-3))
    parser.add_argument("--holdout", type=int, default=16)
    parser.add_argument("--milp-seconds", type=float, default=0.2)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    config = Config(str(output), args.workers, args.max_schemas_per_worker,
                    args.worker_seconds, args.seed, tuple(args.max_lags), tuple(args.offsets),
                    tuple(args.noise_levels), args.holdout, args.milp_seconds)
    validation = validate_sharp_boundary()
    if not validation["ambiguous_prefix_equal"] or not validation["boundary_decodes_exactly"]:
        raise AssertionError(f"sharp-boundary validation failed: {validation}")
    metadata = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": asdict(config), "python": platform.python_version(),
        "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
        "aws_instance_id": os.environ.get("AWS_INSTANCE_ID"),
        "aws_region": os.environ.get("AWS_REGION"), "sharp_boundary_validation": validation,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(run_worker, worker, config) for worker in range(config.workers)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"worker={result['worker']} schemas={result['schemas']} rows={result['rows']} "
                  f"elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    results.sort(key=lambda item: int(item["worker"]))
    summary = analyze(config, results)
    metadata["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["summary"] = {key: value for key, value in summary.items() if key != "workers"}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
