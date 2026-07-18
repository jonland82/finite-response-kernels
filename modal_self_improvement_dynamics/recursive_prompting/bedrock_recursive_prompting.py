#!/usr/bin/env python3
"""Recursive in-context self-improvement pilot on Amazon Bedrock.

The model must infer a hidden Boolean rule from a small candidate family. Its
predictions become the demonstrations supplied in the next outer round. Four
selection regimes compare raw recursive replacement, verifier-filtered
replacement, anchored replacement, and an exogenous-gold control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import boto3
from botocore.config import Config


HYPOTHESES = ("H1", "H2", "H3", "H4")
X_MAX = 31
ARMS = ("raw_replace", "verified", "anchored", "gold")
COLORS = {
    "raw_replace": "#d95f02",
    "verified": "#1b9e77",
    "anchored": "#7570b3",
    "gold": "#333333",
}


@dataclass(frozen=True)
class World:
    world: int
    hypothesis: str
    seed_x: tuple[int, ...]
    eval_x: tuple[int, ...]
    train_x: tuple[tuple[int, ...], ...]

    def target(self, x: int) -> int:
        return hypothesis_value(self.hypothesis, x)


def hypothesis_value(hypothesis: str, x: int) -> int:
    bits = [int(bit) for bit in f"{x:05b}"]
    if hypothesis == "H1":
        return bits[0]
    if hypothesis == "H2":
        return bits[1]
    if hypothesis == "H3":
        return sum(bits) % 2
    if hypothesis == "H4":
        return int(sum(bits) >= 3)
    raise ValueError(f"unknown hypothesis: {hypothesis}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--model-id", default="us.amazon.nova-lite-v1:0"
    )
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed-examples", type=int, default=2)
    parser.add_argument("--eval-size", type=int, default=14)
    parser.add_argument("--train-size", type=int, default=12)
    parser.add_argument("--state-size", type=int, default=12)
    parser.add_argument("--anchor-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--train-temperature", type=float, default=0.8)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def make_worlds(args: argparse.Namespace) -> list[World]:
    rng = random.Random(args.seed)
    worlds: list[World] = []
    universe = list(range(X_MAX + 1))
    for world_id in range(args.worlds):
        hypothesis = HYPOTHESES[world_id % len(HYPOTHESES)]
        # 00000 and 11111 are deliberately compatible with every hypothesis.
        seed_x = tuple((0, X_MAX)[: args.seed_examples])
        shuffled = [x for x in universe if x not in seed_x]
        rng.shuffle(shuffled)
        eval_x = tuple(shuffled[: args.eval_size])
        eligible = [x for x in universe if x not in seed_x and x not in eval_x]
        train_rounds = []
        for _ in range(args.rounds):
            train_rounds.append(tuple(rng.sample(eligible, args.train_size)))
        worlds.append(World(world_id, hypothesis, seed_x, eval_x, tuple(train_rounds)))
    return worlds


def format_examples(examples: Iterable[tuple[int, int]]) -> str:
    return ", ".join(f"label({x:05b})={y}" for x, y in examples)


def make_prompt(examples: list[tuple[int, int]], query_x: tuple[int, ...]) -> str:
    return f"""Exactly one hidden rule labels five-bit strings:
H1: the first bit.
H2: the second bit.
H3: parity (1 when the number of 1 bits is odd, otherwise 0).
H4: majority (1 when at least three bits are 1, otherwise 0).

Infer which rule generated the demonstrations below. "First" and "second"
mean the leftmost and next-to-leftmost bits. Some demonstrations may be noisy.
Before responding, privately count how many demonstrations agree with each of
H1, H2, H3, and H4, then choose the rule with the largest count. Base the answer
only on those agreement counts.

Demonstrations: {format_examples(examples)}

Return JSON only. Include an integer agreement count for every rule and then
the selected rule. Use keys "counts" and "hypothesis"; "counts" must contain
keys "H1", "H2", "H3", and "H4", and "hypothesis" must be one of those four
names. Include no prose outside the JSON."""


def extract_predictions(text: str, expected_x: tuple[int, ...]) -> dict[int, int]:
    predictions: dict[int, int] = {}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            hypothesis = str(payload.get("hypothesis", "")).upper()
            if hypothesis in HYPOTHESES:
                return {x: hypothesis_value(hypothesis, x) for x in expected_x}
            for item in payload.get("answers", []):
                x = int(item["x"])
                y = int(item["y"])
                if x in expected_x:
                    predictions[x] = y
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass
    if len(predictions) < len(expected_x):
        match = re.search(r"\bH([1-4])\b", text.upper())
        if match:
            hypothesis = f"H{match.group(1)}"
            return {x: hypothesis_value(hypothesis, x) for x in expected_x}
        patterns = (
            r'"x"\s*:\s*(-?\d+)\s*,\s*"y"\s*:\s*(-?\d+)',
            r"f\s*\(\s*(-?\d+)\s*\)\s*=\s*(-?\d+)",
        )
        for pattern in patterns:
            for x_text, y_text in re.findall(pattern, text):
                x = int(x_text)
                if x in expected_x:
                    predictions[x] = int(y_text)
    return predictions


def best_hypothesis_fit(predictions: dict[int, int]) -> tuple[str | None, float]:
    if not predictions:
        return None, 0.0
    best_hypothesis, best_count = None, -1
    items = list(predictions.items())
    for hypothesis in HYPOTHESES:
        count = sum(y == hypothesis_value(hypothesis, x) for x, y in items)
        if count > best_count:
            best_hypothesis, best_count = hypothesis, count
    return best_hypothesis, best_count / len(items)


class BedrockCaller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=args.region,
            config=Config(
                read_timeout=90,
                connect_timeout=10,
                retries={"max_attempts": 8, "mode": "adaptive"},
                max_pool_connections=max(32, args.workers + 4),
            ),
        )

    def __call__(
        self,
        world: World,
        arm: str,
        round_id: int,
        kind: str,
        examples: list[tuple[int, int]],
        query_x: tuple[int, ...],
    ) -> dict:
        prompt = make_prompt(examples, query_x)
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.client.converse(
                    modelId=self.args.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={
                        "maxTokens": self.args.max_tokens,
                        "temperature": (
                            self.args.eval_temperature
                            if kind == "eval"
                            else self.args.train_temperature
                        ),
                        "topP": 0.9,
                    },
                )
                text = "".join(
                    block.get("text", "")
                    for block in response["output"]["message"]["content"]
                )
                usage = response.get("usage", {})
                return {
                    "world": world.world,
                    "arm": arm,
                    "round": round_id,
                    "kind": kind,
                    "predictions": extract_predictions(text, query_x),
                    "raw_text": text,
                    "input_tokens": int(usage.get("inputTokens", 0)),
                    "output_tokens": int(usage.get("outputTokens", 0)),
                    "latency_ms": int(response.get("metrics", {}).get("latencyMs", 0)),
                    "error": "",
                }
            except Exception as exc:  # SDK exceptions vary by service response.
                last_error = exc
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        return {
            "world": world.world,
            "arm": arm,
            "round": round_id,
            "kind": kind,
            "predictions": {},
            "raw_text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "error": repr(last_error),
        }


def accuracy(world: World, predictions: dict[int, int], expected_x: tuple[int, ...]) -> float:
    return sum(predictions.get(x) == world.target(x) for x in expected_x) / len(expected_x)


def update_state(
    world: World,
    arm: str,
    predictions: dict[int, int],
    query_x: tuple[int, ...],
    initial: list[tuple[int, int]],
    verified_history: list[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    proposed = [(x, predictions[x]) for x in query_x if x in predictions]
    if arm == "gold":
        return [(x, world.target(x)) for x in query_x[: args.state_size]], verified_history
    if arm == "raw_replace":
        return proposed[: args.state_size], verified_history
    if arm == "anchored":
        anchors = initial[: args.anchor_size]
        room = max(0, args.state_size - len(anchors))
        return anchors + proposed[:room], verified_history
    correct = [(x, y) for x, y in proposed if y == world.target(x)]
    combined = correct + verified_history
    deduped: list[tuple[int, int]] = []
    seen: set[int] = set()
    for pair in combined:
        if pair[0] not in seen:
            deduped.append(pair)
            seen.add(pair[0])
    history = deduped[: args.state_size]
    return history, history


def evaluate_record(
    world: World,
    response: dict,
    examples: list[tuple[int, int]],
) -> dict:
    predictions = response["predictions"]
    fit_hypothesis, coherence = best_hypothesis_fit(predictions)
    return {
        "world": world.world,
        "arm": response["arm"],
        "round": response["round"],
        "accuracy": accuracy(world, predictions, world.eval_x),
        "coverage": len(predictions) / len(world.eval_x),
        "coherence": coherence,
        "fit_is_true": int(fit_hypothesis == world.hypothesis),
        "state_size": len(examples),
        "state_accuracy": (
            sum(y == world.target(x) for x, y in examples) / len(examples)
            if examples
            else 0.0
        ),
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "latency_ms": response["latency_ms"],
        "error": response["error"],
    }


def parallel_calls(
    caller: BedrockCaller,
    jobs: list[tuple],
    workers: int,
) -> list[dict]:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(caller, *job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(records: list[dict], rounds: int) -> tuple[list[dict], list[dict]]:
    aggregate: list[dict] = []
    terminal: list[dict] = []
    for arm in ARMS:
        for round_id in range(rounds + 1):
            group = [r for r in records if r["arm"] == arm and r["round"] == round_id]
            row = {"arm": arm, "round": round_id, "n": len(group)}
            for metric in ("accuracy", "coverage", "coherence", "fit_is_true", "state_accuracy"):
                mean, se = mean_se([float(r[metric]) for r in group])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_se"] = se
            aggregate.append(row)
        for world in sorted({r["world"] for r in records}):
            trajectory = sorted(
                (r for r in records if r["arm"] == arm and r["world"] == world),
                key=lambda row: row["round"],
            )
            values = [r["accuracy"] for r in trajectory]
            increments = [b - a for a, b in zip(values, values[1:])]
            net = values[-1] - values[0]
            total_variation = sum(abs(value) for value in increments)
            terminal.append(
                {
                    "world": world,
                    "arm": arm,
                    "initial_accuracy": values[0],
                    "terminal_accuracy": values[-1],
                    "terminal_gain": net,
                    "peak_accuracy": max(values),
                    "peak_round": values.index(max(values)),
                    "negative_steps": sum(value < 0 for value in increments),
                    "variation_ratio": total_variation / abs(net) if abs(net) > 1e-12 else math.inf,
                }
            )
    return aggregate, terminal


def plot_results(path: Path, aggregate: list[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for arm in ARMS:
        rows = [row for row in aggregate if row["arm"] == arm]
        x = [row["round"] for row in rows]
        for axis, metric, title in (
            (axes[0], "accuracy", "External capability"),
            (axes[1], "state_accuracy", "Quality of recursive context"),
        ):
            y = [row[f"{metric}_mean"] for row in rows]
            se = [row[f"{metric}_se"] for row in rows]
            axis.plot(x, y, marker="o", linewidth=1.8, label=arm, color=COLORS[arm])
            axis.fill_between(
                x,
                [max(0.0, value - spread) for value, spread in zip(y, se)],
                [min(1.0, value + spread) for value, spread in zip(y, se)],
                color=COLORS[arm],
                alpha=0.14,
            )
            axis.set_title(title)
            axis.set_xlabel("outer recursive round")
            axis.set_ylim(-0.03, 1.03)
            axis.grid(alpha=0.25)
    axes[0].set_ylabel("fraction correct")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Recursive prompting response regimes (mean ± SE)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worlds = make_worlds(args)
    caller = BedrockCaller(args)
    initial_states = {
        (world.world, arm): [(x, world.target(x)) for x in world.seed_x]
        for world in worlds
        for arm in ARMS
    }
    states = {key: list(value) for key, value in initial_states.items()}
    histories = {key: list(value) for key, value in initial_states.items()}
    records: list[dict] = []
    raw_responses: list[dict] = []
    started = time.time()

    for round_id in range(args.rounds + 1):
        eval_jobs = []
        for world in worlds:
            for arm in ARMS:
                eval_jobs.append(
                    (
                        world,
                        arm,
                        round_id,
                        "eval",
                        states[(world.world, arm)],
                        world.eval_x,
                    )
                )
        eval_responses = parallel_calls(caller, eval_jobs, args.workers)
        for response in eval_responses:
            world = worlds[response["world"]]
            key = (world.world, response["arm"])
            records.append(evaluate_record(world, response, states[key]))
            raw_responses.append({**response, "predictions": response["predictions"]})

        if round_id == args.rounds:
            break

        train_jobs = []
        for world in worlds:
            query_x = world.train_x[round_id]
            gold_key = (world.world, "gold")
            states[gold_key] = [
                (x, world.target(x)) for x in query_x[: args.state_size]
            ]
            histories[gold_key] = list(states[gold_key])
            for arm in ARMS:
                if arm == "gold":
                    continue
                train_jobs.append(
                    (
                        world,
                        arm,
                        round_id + 1,
                        "train",
                        states[(world.world, arm)],
                        query_x,
                    )
                )
        train_responses = parallel_calls(caller, train_jobs, args.workers)
        for response in train_responses:
            world = worlds[response["world"]]
            key = (world.world, response["arm"])
            states[key], histories[key] = update_state(
                world,
                response["arm"],
                response["predictions"],
                world.train_x[round_id],
                initial_states[key],
                histories[key],
                args,
            )
            raw_responses.append({**response, "predictions": response["predictions"]})

    aggregate, terminal = summarize(records, args.rounds)
    write_csv(args.output_dir / "trajectory.csv", records)
    write_csv(args.output_dir / "aggregate.csv", aggregate)
    write_csv(args.output_dir / "terminal_summary.csv", terminal)
    plot_results(args.output_dir / "recursive_prompting.png", aggregate)

    input_tokens = sum(item["input_tokens"] for item in raw_responses)
    output_tokens = sum(item["output_tokens"] for item in raw_responses)
    # Approximate public standard-tier rates per million tokens.
    if "claude-3-haiku" in args.model_id:
        input_rate, output_rate = 0.25, 1.25
    elif "nova-lite" in args.model_id:
        input_rate, output_rate = 0.06, 0.24
    else:
        input_rate, output_rate = 0.035, 0.14
    estimated_cost = (
        input_tokens * input_rate / 1_000_000
        + output_tokens * output_rate / 1_000_000
    )
    metadata = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "rule_family": list(HYPOTHESES),
        "arms": list(ARMS),
        "calls": len(raw_responses),
        "errors": sum(bool(item["error"]) for item in raw_responses),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "assumed_input_usd_per_million_tokens": input_rate,
        "assumed_output_usd_per_million_tokens": output_rate,
        "estimated_usd_at_public_rates": estimated_cost,
        "elapsed_seconds": time.time() - started,
        "worlds": [
            {"world": world.world, "hypothesis": world.hypothesis}
            for world in worlds
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "raw_responses.jsonl").open("w", encoding="utf-8") as handle:
        for response in raw_responses:
            handle.write(json.dumps(response) + "\n")

    print(json.dumps(metadata, indent=2))
    for arm in ARMS:
        rows = [row for row in aggregate if row["arm"] == arm]
        print(
            f"{arm:12s} accuracy {rows[0]['accuracy_mean']:.3f} -> "
            f"{rows[-1]['accuracy_mean']:.3f}; "
            f"state quality {rows[-1]['state_accuracy_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
