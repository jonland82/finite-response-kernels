#!/usr/bin/env python3
"""Recursive output-population experiment using Amazon Bedrock.

Each round asks a fixed model to reproduce a structured population from the
previous round. Interventions alter prompting or selection, producing an
inexpensive in-context analogue of recursive synthetic-data dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config


FIELDS = {
    "color": ("amber", "blue", "crimson", "ivory"),
    "shape": ("orb", "spire", "ring", "cube"),
    "habitat": ("forest", "desert", "ocean", "tundra"),
    "signal": ("hum", "click", "whistle", "pulse"),
}
ARMS = ("raw_replace", "anchored", "verified", "guided")
COLORS = {
    "raw_replace": "#d95f02",
    "anchored": "#7570b3",
    "verified": "#1b9e77",
    "guided": "#1f78b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--model-id", default="us.amazon.nova-lite-v1:0")
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--candidate-size", type=int, default=36)
    parser.add_argument("--anchor-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--max-estimated-usd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "population_results",
    )
    return parser.parse_args()


def initial_population(size: int, seed: int) -> list[dict[str, str]]:
    if size != 24:
        raise ValueError("the calibrated initial population currently requires size 24")
    weighted = [0] * 12 + [1] * 6 + [2] * 4 + [3] * 2
    rng = random.Random(seed)
    columns = {}
    for field in FIELDS:
        indices = weighted.copy()
        rng.shuffle(indices)
        columns[field] = [FIELDS[field][index] for index in indices]
    return [
        {field: columns[field][row] for field in FIELDS}
        for row in range(size)
    ]


def prompt_for(population: list[dict[str, str]], arm: str, count: int) -> str:
    allowed = "; ".join(f"{field}: {', '.join(values)}" for field, values in FIELDS.items())
    if arm == "guided":
        instruction = (
            "Generate a successor population with the most even possible marginal "
            "coverage of every allowed value and with as many distinct combinations "
            "as possible."
        )
    else:
        instruction = (
            "Generate a successor population that imitates the patterns and frequencies "
            "you observe in the current population."
        )
    return f"""You are reproducing a population of artificial artifacts.

Allowed categorical values are exactly:
{allowed}

{instruction}

Current population:
{json.dumps(population, separators=(',', ':'))}

Return exactly {count} successor records. Return JSON only: one object whose
single key is "population" and whose value is an array of records. Every record
must contain exactly the four keys color, shape, habitat, and signal. Include no
prose outside the JSON."""


def parse_population(text: str, limit: int) -> list[dict[str, str]]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    valid = []
    for item in payload.get("population", []):
        if not isinstance(item, dict):
            continue
        record = {field: str(item.get(field, "")).lower() for field in FIELDS}
        if all(record[field] in FIELDS[field] for field in FIELDS):
            valid.append(record)
        if len(valid) >= limit:
            break
    return valid


def record_tuple(record: dict[str, str]) -> tuple[str, ...]:
    return tuple(record[field] for field in FIELDS)


def select_diverse(candidates: list[dict[str, str]], size: int) -> list[dict[str, str]]:
    remaining = list(candidates)
    selected: list[dict[str, str]] = []
    counts = {field: Counter() for field in FIELDS}
    seen: set[tuple[str, ...]] = set()
    while remaining and len(selected) < size:
        def score(record: dict[str, str]) -> float:
            marginal = sum(1.0 / (1 + counts[field][record[field]]) for field in FIELDS)
            novelty = 1.5 if record_tuple(record) not in seen else 0.0
            return marginal + novelty

        best_index = max(range(len(remaining)), key=lambda index: score(remaining[index]))
        record = remaining.pop(best_index)
        selected.append(record)
        seen.add(record_tuple(record))
        for field in FIELDS:
            counts[field][record[field]] += 1
    return selected


def metrics(population: list[dict[str, str]], target_size: int) -> dict[str, float]:
    if not population:
        return {
            "entropy": 0.0,
            "coverage": 0.0,
            "unique_fraction": 0.0,
            "quality": 0.0,
            "population_fraction": 0.0,
        }
    entropies = []
    covered = 0
    for field, values in FIELDS.items():
        counts = Counter(record[field] for record in population)
        probabilities = [counts[value] / len(population) for value in values if counts[value]]
        entropy = -sum(p * math.log(p) for p in probabilities) / math.log(len(values))
        entropies.append(entropy)
        covered += sum(counts[value] > 0 for value in values)
    marginal_entropy = statistics.fmean(entropies)
    coverage = covered / sum(len(values) for values in FIELDS.values())
    unique_fraction = len({record_tuple(record) for record in population}) / len(population)
    quality = 0.5 * marginal_entropy + 0.5 * unique_fraction
    return {
        "entropy": marginal_entropy,
        "coverage": coverage,
        "unique_fraction": unique_fraction,
        "quality": quality,
        "population_fraction": len(population) / target_size,
    }


class Caller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=args.region,
            config=Config(
                read_timeout=90,
                connect_timeout=10,
                retries={"max_attempts": 7, "mode": "adaptive"},
                max_pool_connections=max(32, args.workers + 4),
            ),
        )

    def __call__(self, replicate: int, round_id: int, arm: str, population: list[dict]) -> dict:
        requested = self.args.candidate_size if arm == "verified" else self.args.population_size
        prompt = prompt_for(population, arm, requested)
        try:
            response = self.client.converse(
                modelId=self.args.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": self.args.max_tokens,
                    "temperature": self.args.temperature,
                    "topP": 0.9,
                },
            )
            text = "".join(
                block.get("text", "") for block in response["output"]["message"]["content"]
            )
            usage = response.get("usage", {})
            return {
                "replicate": replicate,
                "round": round_id,
                "arm": arm,
                "population": parse_population(text, requested),
                "input_tokens": int(usage.get("inputTokens", 0)),
                "output_tokens": int(usage.get("outputTokens", 0)),
                "error": "",
            }
        except Exception as exc:
            return {
                "replicate": replicate,
                "round": round_id,
                "arm": arm,
                "population": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "error": repr(exc),
            }


def mean_se(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_population = initial_population(args.population_size, args.seed)
    planned_calls = args.replicates * args.rounds * len(ARMS)
    max_prompt_characters = max(
        len(
            prompt_for(
                seed_population,
                arm,
                args.candidate_size if arm == "verified" else args.population_size,
            )
        )
        for arm in ARMS
    )
    # One token per ASCII character is deliberately conservative for input.
    preflight_worst_case_usd = planned_calls * (
        max_prompt_characters * 0.06 / 1_000_000
        + args.max_tokens * 0.24 / 1_000_000
    )
    if preflight_worst_case_usd > args.max_estimated_usd:
        raise SystemExit(
            f"preflight worst-case estimate ${preflight_worst_case_usd:.3f} "
            f"exceeds --max-estimated-usd ${args.max_estimated_usd:.3f}"
        )
    anchors = select_diverse(seed_population, args.anchor_size)
    states = {
        (replicate, arm): list(seed_population)
        for replicate in range(args.replicates)
        for arm in ARMS
    }
    caller = Caller(args)
    trajectory: list[dict] = []
    usages: list[dict] = []
    started = time.time()

    for round_id in range(args.rounds + 1):
        for replicate in range(args.replicates):
            for arm in ARMS:
                row = {"replicate": replicate, "arm": arm, "round": round_id}
                row.update(metrics(states[(replicate, arm)], args.population_size))
                trajectory.append(row)
        if round_id == args.rounds:
            break

        jobs = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for replicate in range(args.replicates):
                for arm in ARMS:
                    jobs.append(
                        executor.submit(
                            caller,
                            replicate,
                            round_id + 1,
                            arm,
                            states[(replicate, arm)],
                        )
                    )
            for future in as_completed(jobs):
                result = future.result()
                usages.append({key: value for key, value in result.items() if key != "population"})
                key = (result["replicate"], result["arm"])
                generated = result["population"]
                if result["arm"] == "verified":
                    next_state = select_diverse(generated, args.population_size)
                elif result["arm"] == "anchored":
                    room = args.population_size - len(anchors)
                    next_state = anchors + generated[:room]
                else:
                    next_state = generated[: args.population_size]
                # A malformed response is a failed reproduction, not silently repaired.
                states[key] = next_state

    aggregate = []
    for arm in ARMS:
        for round_id in range(args.rounds + 1):
            group = [
                row for row in trajectory
                if row["arm"] == arm and row["round"] == round_id
            ]
            row = {"arm": arm, "round": round_id, "n": len(group)}
            for metric in ("quality", "entropy", "coverage", "unique_fraction", "population_fraction"):
                mean, se = mean_se([item[metric] for item in group])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_se"] = se
            aggregate.append(row)

    terminal = []
    kernel = []
    for arm in ARMS:
        for replicate in range(args.replicates):
            rows = sorted(
                [row for row in trajectory if row["arm"] == arm and row["replicate"] == replicate],
                key=lambda row: row["round"],
            )
            values = [row["quality"] for row in rows]
            increments = [b - a for a, b in zip(values, values[1:])]
            net_gain = values[-1] - values[0]
            for round_id, increment in enumerate(increments, start=1):
                kernel.append(
                    {
                        "replicate": replicate,
                        "arm": arm,
                        "round": round_id,
                        "delta_quality": increment,
                        "signed_normalized_kernel": (
                            increment / abs(net_gain) if abs(net_gain) > 1e-12 else ""
                        ),
                    }
                )
            terminal.append(
                {
                    "replicate": replicate,
                    "arm": arm,
                    "initial_quality": values[0],
                    "terminal_quality": values[-1],
                    "terminal_gain": net_gain,
                    "peak_quality": max(values),
                    "peak_round": values.index(max(values)),
                    "negative_steps": sum(value < 0 for value in increments),
                }
            )

    write_csv(args.output_dir / "trajectory.csv", trajectory)
    write_csv(args.output_dir / "aggregate.csv", aggregate)
    write_csv(args.output_dir / "terminal_summary.csv", terminal)
    write_csv(args.output_dir / "kernel.csv", kernel)
    write_csv(args.output_dir / "usage.csv", usages)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    for arm in ARMS:
        rows = [row for row in aggregate if row["arm"] == arm]
        x = [row["round"] for row in rows]
        for axis, metric, title in (
            (axes[0], "quality", "Composite response"),
            (axes[1], "entropy", "Marginal entropy"),
            (axes[2], "unique_fraction", "Distinct combinations"),
        ):
            y = [row[f"{metric}_mean"] for row in rows]
            se = [row[f"{metric}_se"] for row in rows]
            axis.plot(x, y, marker="o", label=arm, color=COLORS[arm])
            axis.fill_between(x, [a-b for a,b in zip(y,se)], [a+b for a,b in zip(y,se)], color=COLORS[arm], alpha=0.13)
            axis.set_title(title)
            axis.set_xlabel("outer recursive round")
            axis.set_ylim(0, 1.03)
            axis.grid(alpha=0.25)
    axes[0].set_ylabel("normalized score")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Recursive prompting: collapse-to-improvement response spectrum")
    fig.savefig(args.output_dir / "recursive_population.png", dpi=180)
    plt.close(fig)

    input_tokens = sum(row["input_tokens"] for row in usages)
    output_tokens = sum(row["output_tokens"] for row in usages)
    estimated_cost = input_tokens * 0.06 / 1_000_000 + output_tokens * 0.24 / 1_000_000
    metadata = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "calls": len(usages),
        "errors": sum(bool(row["error"]) for row in usages),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_usd_at_public_nova_lite_rates": estimated_cost,
        "preflight_worst_case_usd": preflight_worst_case_usd,
        "elapsed_seconds": time.time() - started,
        "initial_metrics": metrics(seed_population, args.population_size),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    for arm in ARMS:
        rows = [row for row in aggregate if row["arm"] == arm]
        print(f"{arm:12s} quality {rows[0]['quality_mean']:.3f} -> {rows[-1]['quality_mean']:.3f}")


if __name__ == "__main__":
    main()
