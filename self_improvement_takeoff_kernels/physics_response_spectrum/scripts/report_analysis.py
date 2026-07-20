#!/usr/bin/env python3
"""Create reproducible statistical summaries for the completed AWS runs."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
PRIMARY = "primary-eed-tool-20260719"
RESTART = "control-restart-20260719"
FROZEN = "control-frozen-20260719"
RHOS = (-6.0, -2.0, 0.0, 2.0, 6.0)
SEED = 20260719


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], seed_offset: int, draws: int = 10_000) -> tuple[float, float, float]:
    rng = random.Random(SEED + seed_offset)
    n = len(values)
    estimates = [statistics.fmean(rng.choices(values, k=n)) for _ in range(draws)]
    return statistics.fmean(values), percentile(estimates, 0.025), percentile(estimates, 0.975)


def numeric_trajectory(run_id: str) -> list[dict[str, Any]]:
    rows = read_csv(RUNS / run_id / "trajectory.csv")
    numeric = {
        "problem_id": int,
        "replicate": int,
        "rho": float,
        "round": int,
        "mean_accuracy": float,
        "mean_fitness": float,
        "pass_at_4": float,
        "distinct_fraction": float,
        "effective_answers": float,
        "invalid_fraction": float,
    }
    for row in rows:
        for key, converter in numeric.items():
            row[key] = converter(row[key])
    return rows


def problem_values(
    rows: list[dict[str, Any]],
    rho: float,
    metric: str,
    transform: Callable[[list[dict[str, Any]]], float],
) -> list[float]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["rho"] == rho:
            grouped[row["problem_id"]].append(row)
    return [transform(group) for _, group in sorted(grouped.items())]


def endpoint_gain(group: list[dict[str, Any]], metric: str) -> float:
    initial = [row[metric] for row in group if row["round"] == 0]
    terminal_round = max(row["round"] for row in group)
    terminal = [row[metric] for row in group if row["round"] == terminal_round]
    return statistics.fmean(terminal) - statistics.fmean(initial)


def terminal_value(group: list[dict[str, Any]], metric: str) -> float:
    terminal_round = max(row["round"] for row in group)
    return statistics.fmean(row[metric] for row in group if row["round"] == terminal_round)


def paired_effects(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rho_index, rho in enumerate(RHOS):
        for metric_index, metric in enumerate(("mean_fitness", "mean_accuracy")):
            gains = problem_values(
                primary,
                rho,
                metric,
                lambda group, metric=metric: endpoint_gain(group, metric),
            )
            mean, low, high = bootstrap_mean_ci(gains, 100 * rho_index + metric_index)

            terminal = problem_values(
                primary,
                rho,
                metric,
                lambda group, metric=metric: terminal_value(group, metric),
            )
            neutral = problem_values(
                primary,
                0.0,
                metric,
                lambda group, metric=metric: terminal_value(group, metric),
            )
            contrast = [left - right for left, right in zip(terminal, neutral)]
            c_mean, c_low, c_high = bootstrap_mean_ci(
                contrast, 1000 + 100 * rho_index + metric_index
            )
            rows.append(
                {
                    "rho": rho,
                    "metric": metric,
                    "problems": len(gains),
                    "endpoint_gain_mean": mean,
                    "endpoint_gain_ci_low": low,
                    "endpoint_gain_ci_high": high,
                    "terminal_minus_neutral_mean": c_mean,
                    "terminal_minus_neutral_ci_low": c_low,
                    "terminal_minus_neutral_ci_high": c_high,
                }
            )
    return rows


def domain_summary(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        if row["round"] in {0, 6}:
            grouped[(row["tag"], row["rho"])].append(row)
    output = []
    for (tag, rho), group in sorted(grouped.items()):
        initial = [row for row in group if row["round"] == 0]
        terminal = [row for row in group if row["round"] == 6]
        output.append(
            {
                "tag": tag,
                "rho": rho,
                "problems": len({row["problem_id"] for row in group}),
                "initial_fitness": statistics.fmean(row["mean_fitness"] for row in initial),
                "terminal_fitness": statistics.fmean(row["mean_fitness"] for row in terminal),
                "fitness_gain": statistics.fmean(row["mean_fitness"] for row in terminal)
                - statistics.fmean(row["mean_fitness"] for row in initial),
                "terminal_accuracy": statistics.fmean(row["mean_accuracy"] for row in terminal),
            }
        )
    return output


def context_summary(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    restart_config = json.loads((RUNS / RESTART / "config.json").read_text(encoding="utf-8"))
    problem_ids = set(restart_config["problem_ids"])
    recursive = [
        row for row in primary
        if row["problem_id"] in problem_ids and row["replicate"] == 0 and row["rho"] in {-6.0, 0.0, 2.0}
    ]
    modes = {
        "recursive": recursive,
        "restart": numeric_trajectory(RESTART),
        "frozen": numeric_trajectory(FROZEN),
    }
    output = []
    terminal_by_mode: dict[tuple[str, float], float] = {}
    terminal_problem_by_mode: dict[tuple[str, float], dict[int, float]] = {}
    for mode, rows in modes.items():
        for rho in (-6.0, 0.0, 2.0):
            selected = [row for row in rows if row["rho"] == rho]
            terminal_round = max(row["round"] for row in selected)
            initial = [row for row in selected if row["round"] == 0]
            terminal = [row for row in selected if row["round"] == terminal_round]
            initial_fitness = statistics.fmean(row["mean_fitness"] for row in initial)
            terminal_fitness = statistics.fmean(row["mean_fitness"] for row in terminal)
            terminal_by_mode[(mode, rho)] = terminal_fitness
            by_problem: dict[int, list[float]] = defaultdict(list)
            for row in terminal:
                by_problem[row["problem_id"]].append(row["mean_fitness"])
            terminal_problem_by_mode[(mode, rho)] = {
                problem_id: statistics.fmean(values)
                for problem_id, values in by_problem.items()
            }
            output.append(
                {
                    "mode": mode,
                    "rho": rho,
                    "problems": len(problem_ids),
                    "initial_fitness": initial_fitness,
                    "terminal_fitness": terminal_fitness,
                    "fitness_gain": terminal_fitness - initial_fitness,
                    "terminal_accuracy": statistics.fmean(row["mean_accuracy"] for row in terminal),
                    "terminal_distinct_fraction": statistics.fmean(
                        row["distinct_fraction"] for row in terminal
                    ),
                    "recursive_minus_restart": "",
                    "recursive_minus_restart_ci_low": "",
                    "recursive_minus_restart_ci_high": "",
                    "recursive_minus_frozen": "",
                    "recursive_minus_frozen_ci_low": "",
                    "recursive_minus_frozen_ci_high": "",
                }
            )
    for row in output:
        if row["mode"] == "recursive":
            rho = row["rho"]
            recursive_values = terminal_problem_by_mode[("recursive", rho)]
            for contrast_index, comparison in enumerate(("restart", "frozen")):
                comparison_values = terminal_problem_by_mode[(comparison, rho)]
                differences = [
                    recursive_values[problem_id] - comparison_values[problem_id]
                    for problem_id in sorted(problem_ids)
                ]
                mean, low, high = bootstrap_mean_ci(
                    differences,
                    5000 + int(rho + 6) * 10 + contrast_index,
                )
                row[f"recursive_minus_{comparison}"] = mean
                row[f"recursive_minus_{comparison}_ci_low"] = low
                row[f"recursive_minus_{comparison}_ci_high"] = high
    return output


def plot_context_controls(contexts: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    colors = {"recursive": "#b2182b", "restart": "#2166ac", "frozen": "#666666"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for mode in ("recursive", "restart", "frozen"):
        rows = sorted(
            [row for row in contexts if row["mode"] == mode],
            key=lambda row: row["rho"],
        )
        x = [row["rho"] for row in rows]
        axes[0].plot(
            x,
            [row["terminal_fitness"] for row in rows],
            marker="o",
            label=mode,
            color=colors[mode],
        )
        axes[1].plot(
            x,
            [row["terminal_accuracy"] for row in rows],
            marker="o",
            label=mode,
            color=colors[mode],
        )
    axes[0].set_title("Terminal EED fitness")
    axes[1].set_title("Terminal exact accuracy")
    for axis in axes:
        axis.set_xlabel("selection pressure rho")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Matched eight-problem context controls")
    fig.savefig(RESULTS / "context_controls.png", dpi=180)
    plt.close(fig)


def candidate_quality(run_id: str) -> dict[str, Any]:
    path = RUNS / run_id / "candidates.jsonl"
    candidates = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        "candidates": len(candidates),
        "api_errors": sum(bool(row.get("api_error")) for row in candidates),
        "format_errors": sum(bool(row.get("format_error")) for row in candidates),
        "parse_failures": sum(not row.get("candidate_parse_ok", False) for row in candidates),
        "exact_candidates": sum(int(row.get("correct", 0)) for row in candidates),
        "positive_eed_candidates": sum(float(row.get("eed_fitness", 0)) > 0 for row in candidates),
        "retried_candidates": sum(int(row.get("retry_count", 0)) > 0 for row in candidates),
        "retry_attempts": sum(int(row.get("retry_count", 0)) for row in candidates),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in candidates),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in candidates),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    primary = numeric_trajectory(PRIMARY)
    effects = paired_effects(primary)
    domains = domain_summary(primary)
    contexts = context_summary(primary)
    write_csv(RESULTS / "paired_bootstrap.csv", effects)
    write_csv(RESULTS / "domain_terminal.csv", domains)
    write_csv(RESULTS / "context_controls.csv", contexts)
    plot_context_controls(contexts)

    ledger = read_csv(ROOT / "usage_ledger.csv")
    manifests = {}
    for run_dir in sorted(path for path in RUNS.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifests[run_dir.name] = {
                "calls": manifest["calls"],
                "stage_spend_usd": manifest["stage_spend_usd"],
            }
    summary = {
        "primary_run": PRIMARY,
        "paired_effects": effects,
        "context_controls": contexts,
        "candidate_quality": {
            run_id: candidate_quality(run_id) for run_id in (PRIMARY, RESTART, FROZEN)
        },
        "all_runs": {
            "calls": len(ledger),
            "input_tokens": sum(int(row["input_tokens"]) for row in ledger),
            "output_tokens": sum(int(row["output_tokens"]) for row in ledger),
            "estimated_usd": sum(float(row["estimated_usd"]) for row in ledger),
            "terminal_errors": sum(bool(row["error"]) for row in ledger),
        },
        "manifests": manifests,
    }
    (RESULTS / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
