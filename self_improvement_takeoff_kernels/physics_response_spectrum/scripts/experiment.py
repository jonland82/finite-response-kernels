#!/usr/bin/env python3
"""Run the fixed-weight PHYBench response-spectrum experiment on Bedrock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import boto3
from botocore.config import Config

from scoring import canonical_answer, score_answer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data" / "source" / "PHYBench-fullques_v1.json"
SPLITS_DATA = ROOT / "data" / "splits" / "phybench_splits.json"
RUNS_DIR = ROOT / "runs"
GLOBAL_LEDGER = ROOT / "usage_ledger.csv"
DEFAULT_MODEL = "us.amazon.nova-lite-v1:0"
DEFAULT_RHOS = (-6.0, -2.0, 0.0, 2.0, 6.0)
INPUT_USD_PER_MILLION = 0.06
OUTPUT_USD_PER_MILLION = 0.24
GLOBAL_STOP_USD = 9.50
SOLUTION_TOOL = {
    "tools": [
        {
            "toolSpec": {
                "name": "submit_physics_solution",
                "description": "Submit the final structured solution to the physics problem.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "derivation": {
                                "type": "string",
                                "description": "Concise derivation, at most six short steps.",
                            },
                            "final_answer": {
                                "type": "string",
                                "description": "One symbolic or numerical LaTeX expression. Escape every LaTeX backslash as two backslashes in the tool argument.",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Required unit, or the word null if none.",
                            },
                        },
                        "required": ["derivation", "final_answer", "unit"],
                    }
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": "submit_physics_solution"}},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source() -> list[dict[str, Any]]:
    return json.loads(SOURCE_DATA.read_text(encoding="utf-8"))


def prepare_splits(seed: int = 20260719) -> dict[str, Any]:
    records = load_source()
    eligible = []
    excluded = []
    for record in records:
        identity = score_answer(record["answer"], record["answer"])
        enriched = {
            "id": record["id"],
            "tag": record["tag"],
            "content": record["content"],
            "solution": record["solution"],
            "answer": record["answer"],
            "gold_canonical": identity.canonical_gold,
        }
        if identity.correct and identity.gold_parse_ok:
            eligible.append(enriched)
        else:
            excluded.append({"id": record["id"], "tag": record["tag"], "reason": identity.reason})

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        groups[record["tag"]].append(record)

    split_rows: dict[str, list[dict[str, Any]]] = {
        "calibration": [],
        "spectrum": [],
        "locked": [],
    }
    for tag in sorted(groups):
        items = groups[tag]
        rng.shuffle(items)
        n = len(items)
        n_calibration = round(0.20 * n)
        n_spectrum = round(0.40 * n)
        split_rows["calibration"].extend(items[:n_calibration])
        split_rows["spectrum"].extend(items[n_calibration : n_calibration + n_spectrum])
        split_rows["locked"].extend(items[n_calibration + n_spectrum :])

    for name, items in split_rows.items():
        rng.shuffle(items)
        for item in items:
            item["split"] = name

    payload = {
        "created_at": utc_now(),
        "seed": seed,
        "source_path": str(SOURCE_DATA.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE_DATA),
        "source_records": len(records),
        "eligible_records": len(eligible),
        "excluded_records": excluded,
        "splits": split_rows,
    }
    json_dump(SPLITS_DATA, payload)
    return payload


def load_splits() -> dict[str, Any]:
    if not SPLITS_DATA.exists():
        return prepare_splits()
    return json.loads(SPLITS_DATA.read_text(encoding="utf-8"))


def format_instructions(output_mode: str) -> str:
    if output_mode == "tool":
        return """Call the required submit_physics_solution tool exactly once.
Keep the derivation to at most six short steps. Put only one expression in
final_answer and use the word null for unit when no unit is required. In every
tool string, escape each LaTeX backslash as two backslashes."""
    return """Return JSON only, with exactly these keys:
"derivation": a concise derivation of at most six short steps;
"final_answer": one symbolic or numerical final expression;
"unit": the required unit as a string, or null when no unit is required.

Do not include Markdown fences or prose outside the JSON object."""


def initial_prompt(problem: str, output_mode: str) -> str:
    return f"""Solve the physics problem below from first principles.

{format_instructions(output_mode)}
Check the physical setup, algebra, limiting behavior, and dimensions before
answering. Prefer the right-hand side rather than an equation as final_answer.

Problem:
{problem}"""


def revision_prompt(problem: str, parent: dict[str, Any], output_mode: str) -> str:
    inherited = {
        "derivation": parent.get("derivation", ""),
        "final_answer": parent.get("final_answer", ""),
        "unit": parent.get("unit"),
    }
    return f"""Solve the physics problem below. You inherit one earlier attempt,
but it may contain a conceptual or algebraic error. Re-derive the result,
identify problems privately, and return a corrected successor. Do not assume
the inherited final answer is correct.

{format_instructions(output_mode)}
Check the physical setup, algebra, limiting behavior, and dimensions before
answering. Prefer the right-hand side rather than an equation as final_answer.

Problem:
{problem}

Inherited attempt:
{json.dumps(inherited, ensure_ascii=False, separators=(",", ":"))}"""


def parse_response(text: str) -> tuple[dict[str, Any], str]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}, "no_json_object"
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return {}, f"json_error: {exc}"
    return validate_payload(payload)


def validate_payload(payload: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(payload, dict):
        return {}, "json_not_object"
    required = ("derivation", "final_answer", "unit")
    if any(key not in payload for key in required):
        return payload, "missing_required_key"
    if not isinstance(payload["derivation"], str) or not isinstance(payload["final_answer"], str):
        return payload, "wrong_field_type"
    return payload, ""


class UsageLedger:
    fields = (
        "timestamp", "run_id", "stage", "call_id", "model_id",
        "input_tokens", "output_tokens", "estimated_usd", "error",
    )

    def __init__(self, path: Path = GLOBAL_LEDGER):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total = 0.0
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                self.total = sum(float(row["estimated_usd"]) for row in csv.DictReader(handle))

    def append(self, row: dict[str, Any]) -> None:
        exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in self.fields})
        self.total += float(row["estimated_usd"])


class BedrockCaller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=args.region,
            config=Config(
                read_timeout=120,
                connect_timeout=10,
                retries={"max_attempts": 7, "mode": "adaptive"},
                max_pool_connections=max(32, args.workers + 4),
            ),
        )

    def __call__(self, job: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        request: dict[str, Any] = {
            "modelId": self.args.model_id,
            "messages": [{"role": "user", "content": [{"text": job["prompt"]}]}],
            "inferenceConfig": {
                "maxTokens": self.args.max_tokens,
                "temperature": self.args.temperature,
                "topP": self.args.top_p,
            },
        }
        if self.args.output_mode == "tool":
            request["toolConfig"] = SOLUTION_TOOL

        transient_errors = []
        for attempt in range(3):
            try:
                response = self.client.converse(**request)
                content = response["output"]["message"]["content"]
                raw_text = "".join(block.get("text", "") for block in content)
                tool_payload = next(
                    (
                        block["toolUse"].get("input")
                        for block in content
                        if "toolUse" in block
                    ),
                    None,
                )
                usage = response.get("usage", {})
                stop_reason = response.get("stopReason", "")
                error = ""
                break
            except Exception as exc:
                error = repr(exc)
                transient_errors.append(error)
                retryable = "invalid sequence as part of ToolUse" in str(exc)
                if not retryable or attempt == 2:
                    raw_text = ""
                    tool_payload = None
                    usage = {}
                    stop_reason = ""
                    break
        return {
            **job,
            "raw_text": raw_text,
            "tool_payload": tool_payload,
            "stop_reason": stop_reason,
            "input_tokens": int(usage.get("inputTokens", 0)),
            "output_tokens": int(usage.get("outputTokens", 0)),
            "api_error": error,
            "retry_count": max(0, len(transient_errors) - int(bool(error))),
            "transient_errors": transient_errors,
            "elapsed_seconds": time.time() - started,
        }


def estimated_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_USD_PER_MILLION / 1_000_000
        + output_tokens * OUTPUT_USD_PER_MILLION / 1_000_000
    )


def conservative_job_cost(prompt: str, max_tokens: int) -> float:
    input_upper = max(1, math.ceil(len(prompt) / 2))
    return estimated_cost(input_upper, max_tokens)


def execute_jobs(
    jobs: list[dict[str, Any]],
    caller: BedrockCaller,
    ledger: UsageLedger,
    run_dir: Path,
    args: argparse.Namespace,
    stage_spend: float,
) -> tuple[list[dict[str, Any]], float]:
    worst_case = sum(conservative_job_cost(job["prompt"], args.max_tokens) for job in jobs)
    if stage_spend + worst_case > args.stage_cap_usd + 1e-12:
        raise RuntimeError(
            f"batch preflight USD {worst_case:.4f} would exceed stage cap "
            f"USD {args.stage_cap_usd:.2f} from current USD {stage_spend:.4f}"
        )
    if ledger.total + worst_case > args.global_stop_usd + 1e-12:
        raise RuntimeError(
            f"batch preflight USD {worst_case:.4f} would exceed global stop "
            f"USD {args.global_stop_usd:.2f} from current USD {ledger.total:.4f}"
        )

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(caller, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            cost = estimated_cost(result["input_tokens"], result["output_tokens"])
            usage_row = {
                "timestamp": utc_now(),
                "run_id": args.run_id,
                "stage": args.stage,
                "call_id": result["call_id"],
                "model_id": args.model_id,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "estimated_usd": f"{cost:.10f}",
                "error": result["api_error"],
            }
            ledger.append(usage_row)
            stage_spend += cost
            append_jsonl(run_dir / "usage.jsonl", usage_row)
            completed.append(result)
    return completed, stage_spend


def make_candidate(
    result: dict[str, Any],
    gold_answer: str,
    fitness_metric: str,
) -> dict[str, Any]:
    if result.get("tool_payload") is not None:
        payload, format_error = validate_payload(result["tool_payload"])
    else:
        payload, format_error = parse_response(result["raw_text"])
    final_answer = payload.get("final_answer", "") if isinstance(payload, dict) else ""
    score = score_answer(final_answer, gold_answer)
    valid = not format_error and not result["api_error"]
    correct = score.correct if valid else 0
    eed_fitness = score.eed_fitness if valid else 0.0
    fitness = float(correct) if fitness_metric == "exact" else eed_fitness
    return {
        "candidate_id": result["call_id"],
        "problem_id": result["problem_id"],
        "tag": result["tag"],
        "replicate": result["replicate"],
        "rho": result["rho"],
        "round": result["round"],
        "slot": result["slot"],
        "parent_id": result.get("parent_id", ""),
        "parent_correct": result.get("parent_correct", ""),
        "parent_fitness": result.get("parent_fitness", ""),
        "derivation": payload.get("derivation", "") if isinstance(payload, dict) else "",
        "final_answer": final_answer,
        "unit": payload.get("unit") if isinstance(payload, dict) else None,
        "format_error": format_error,
        "api_error": result["api_error"],
        "retry_count": result.get("retry_count", 0),
        "transient_errors": result.get("transient_errors", []),
        "correct": correct,
        "eed_fitness": eed_fitness,
        "fitness": fitness,
        "fitness_metric": fitness_metric,
        "score_reason": result["api_error"] or format_error or score.reason,
        "candidate_parse_ok": score.candidate_parse_ok,
        "gold_parse_ok": score.gold_parse_ok,
        "canonical_answer": score.canonical_candidate or canonical_answer(final_answer),
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "elapsed_seconds": result["elapsed_seconds"],
        "raw_text": result["raw_text"],
        "tool_payload": result.get("tool_payload"),
        "stop_reason": result.get("stop_reason", ""),
    }


def selection_probabilities(children: list[dict[str, Any]], rho: float) -> list[float]:
    logits = [rho * (float(child["fitness"]) - 0.5) for child in children]
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def selection_rng(seed: int, problem_id: int, replicate: int, rho: float, round_id: int) -> random.Random:
    material = f"{seed}|{problem_id}|{replicate}|{rho:.6f}|{round_id}"
    derived = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(derived)


def population_row(
    problem: dict[str, Any],
    replicate: int,
    rho: float,
    round_id: int,
    population: list[dict[str, Any]],
) -> dict[str, Any]:
    correctness = [int(item["correct"]) for item in population]
    fitness = [float(item["fitness"]) for item in population]
    answers = [item["canonical_answer"] or f"invalid:{item['candidate_id']}" for item in population]
    frequencies = Counter(answers)
    m = len(population)
    neff = 1.0 / sum((count / m) ** 2 for count in frequencies.values())
    generated_with_parents = [item for item in population if item.get("parent_correct") != ""]
    regressions = sum(
        int(item.get("parent_correct", 0)) == 1 and int(item["correct"]) == 0
        for item in generated_with_parents
    )
    recoveries = sum(
        int(item.get("parent_correct", 0)) == 0 and int(item["correct"]) == 1
        for item in generated_with_parents
    )
    fitness_children = [item for item in population if item.get("parent_fitness") != ""]
    fitness_regressions = sum(
        float(item["fitness"]) < float(item["parent_fitness"])
        for item in fitness_children
    )
    fitness_improvements = sum(
        float(item["fitness"]) > float(item["parent_fitness"])
        for item in fitness_children
    )
    return {
        "problem_id": problem["id"],
        "tag": problem["tag"],
        "replicate": replicate,
        "rho": rho,
        "round": round_id,
        "population_size": m,
        "mean_accuracy": statistics.fmean(correctness),
        "mean_fitness": statistics.fmean(fitness),
        "pass_at_4": max(correctness),
        "distinct_fraction": len(frequencies) / m,
        "effective_answers": neff,
        "invalid_fraction": statistics.fmean(
            int(bool(item["format_error"] or item["api_error"])) for item in population
        ),
        "regressions": regressions,
        "recoveries": recoveries,
        "fitness_regressions": fitness_regressions,
        "fitness_improvements": fitness_improvements,
        "candidate_ids": "|".join(item["candidate_id"] for item in population),
    }


def choose_problems(stage: str, problem_count: int) -> list[dict[str, Any]]:
    splits = load_splits()["splits"]
    split_name = "calibration" if stage in {"smoke", "pilot"} else "spectrum"
    available = splits[split_name]
    if problem_count <= 0:
        return available
    return available[: min(problem_count, len(available))]


def load_initial_population(
    source_run_id: str,
    problems: list[dict[str, Any]],
    replicates: int,
    population_size: int,
    fitness_metric: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    source_path = RUNS_DIR / source_run_id / "candidates.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"initial population source not found: {source_path}")
    problem_ids = {problem["id"] for problem in problems}
    initial: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            if (
                int(candidate["round"]) == 0
                and candidate["problem_id"] in problem_ids
                and int(candidate["replicate"]) < replicates
            ):
                if candidate.get("fitness_metric", "exact") != fitness_metric:
                    raise ValueError("source population uses a different fitness metric")
                initial[(candidate["problem_id"], int(candidate["replicate"]))].append(candidate)
    for problem in problems:
        for replicate in range(replicates):
            key = (problem["id"], replicate)
            initial[key].sort(key=lambda item: int(item["slot"]))
            if len(initial[key]) != population_size:
                raise ValueError(
                    f"source population {key} has {len(initial[key])} candidates; "
                    f"expected {population_size}"
                )
    return initial


def run_experiment(args: argparse.Namespace) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    problems = choose_problems(args.stage, args.problem_count)
    rhos = tuple(args.rhos)
    ledger = UsageLedger()
    caller = BedrockCaller(args)
    stage_spend = 0.0
    candidate_path = run_dir / "candidates.jsonl"
    selection_path = run_dir / "selections.jsonl"
    trajectory: list[dict[str, Any]] = []

    config = {
        "created_at": utc_now(),
        "stage": args.stage,
        "run_id": args.run_id,
        "region": args.region,
        "model_id": args.model_id,
        "problem_ids": [problem["id"] for problem in problems],
        "problem_count": len(problems),
        "population_size": args.population_size,
        "rounds": args.rounds,
        "replicates": args.replicates,
        "rhos": rhos,
        "fitness_metric": args.fitness_metric,
        "output_mode": args.output_mode,
        "context_mode": args.context_mode,
        "initial_population_run": args.initial_population_run,
        "workers": args.workers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "stage_cap_usd": args.stage_cap_usd,
        "global_stop_usd": args.global_stop_usd,
        "pricing": {
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
        },
        "source_sha256": sha256(SOURCE_DATA),
        "split_sha256": sha256(SPLITS_DATA),
        "global_spend_before": ledger.total,
    }
    json_dump(run_dir / "config.json", config)

    problem_lookup = {problem["id"]: problem for problem in problems}
    if args.initial_population_run:
        initial = load_initial_population(
            args.initial_population_run,
            problems,
            args.replicates,
            args.population_size,
            args.fitness_metric,
        )
        for population in initial.values():
            for source_candidate in population:
                candidate = {**source_candidate, "initial_population_source": args.initial_population_run}
                append_jsonl(candidate_path, candidate)
    else:
        initial_jobs = []
        for problem in problems:
            for replicate in range(args.replicates):
                for slot in range(args.population_size):
                    initial_jobs.append(
                        {
                            "call_id": f"{args.run_id}-{uuid.uuid4().hex[:16]}",
                            "problem_id": problem["id"],
                            "tag": problem["tag"],
                            "replicate": replicate,
                            "rho": "shared",
                            "round": 0,
                            "slot": slot,
                            "prompt": initial_prompt(problem["content"], args.output_mode),
                        }
                    )
        raw_initial, stage_spend = execute_jobs(
            initial_jobs, caller, ledger, run_dir, args, stage_spend
        )
        initial = defaultdict(list)
        for result in raw_initial:
            problem = problem_lookup[result["problem_id"]]
            candidate = make_candidate(result, problem["answer"], args.fitness_metric)
            initial[(problem["id"], result["replicate"])].append(candidate)
            append_jsonl(candidate_path, candidate)
        for key in initial:
            initial[key].sort(key=lambda item: item["slot"])

    states: dict[tuple[int, int, float], list[dict[str, Any]]] = {}
    for problem in problems:
        for replicate in range(args.replicates):
            for rho in rhos:
                population = list(initial[(problem["id"], replicate)])
                states[(problem["id"], replicate, rho)] = population
                trajectory.append(population_row(problem, replicate, rho, 0, population))

    for round_id in range(1, args.rounds + 1):
        jobs = []
        for problem in problems:
            for replicate in range(args.replicates):
                for rho in rhos:
                    population = states[(problem["id"], replicate, rho)]
                    for slot, parent in enumerate(population):
                        if args.context_mode == "restart":
                            prompt = initial_prompt(problem["content"], args.output_mode)
                            prompt_parent: dict[str, Any] | None = None
                        elif args.context_mode == "frozen":
                            prompt_parent = initial[(problem["id"], replicate)][slot]
                            prompt = revision_prompt(problem["content"], prompt_parent, args.output_mode)
                        else:
                            prompt_parent = parent
                            prompt = revision_prompt(problem["content"], prompt_parent, args.output_mode)
                        jobs.append(
                            {
                                "call_id": f"{args.run_id}-{uuid.uuid4().hex[:16]}",
                                "problem_id": problem["id"],
                                "tag": problem["tag"],
                                "replicate": replicate,
                                "rho": rho,
                                "round": round_id,
                                "slot": slot,
                                "parent_id": prompt_parent["candidate_id"] if prompt_parent else "",
                                "parent_correct": prompt_parent["correct"] if prompt_parent else "",
                                "parent_fitness": prompt_parent["fitness"] if prompt_parent else "",
                                "prompt": prompt,
                            }
                        )
        raw_children, stage_spend = execute_jobs(
            jobs, caller, ledger, run_dir, args, stage_spend
        )
        grouped: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(list)
        for result in raw_children:
            problem = problem_lookup[result["problem_id"]]
            candidate = make_candidate(result, problem["answer"], args.fitness_metric)
            key = (problem["id"], result["replicate"], float(result["rho"]))
            grouped[key].append(candidate)
            append_jsonl(candidate_path, candidate)

        for problem in problems:
            for replicate in range(args.replicates):
                for rho in rhos:
                    key = (problem["id"], replicate, rho)
                    children = sorted(grouped[key], key=lambda item: item["slot"])
                    probabilities = selection_probabilities(children, rho)
                    rng = selection_rng(args.seed, problem["id"], replicate, rho, round_id)
                    selected_indices = rng.choices(
                        range(len(children)), weights=probabilities, k=args.population_size
                    )
                    population = [children[index] for index in selected_indices]
                    states[key] = population
                    append_jsonl(
                        selection_path,
                        {
                            "problem_id": problem["id"],
                            "replicate": replicate,
                            "rho": rho,
                            "round": round_id,
                            "child_ids": [child["candidate_id"] for child in children],
                            "child_correct": [child["correct"] for child in children],
                            "child_fitness": [child["fitness"] for child in children],
                            "probabilities": probabilities,
                            "selected_indices": selected_indices,
                            "selected_ids": [child["candidate_id"] for child in population],
                        },
                    )
                    trajectory.append(population_row(problem, replicate, rho, round_id, population))
        write_csv(run_dir / "trajectory.csv", trajectory)
        print(
            f"round {round_id}/{args.rounds}: "
            f"stage USD {stage_spend:.4f}, global USD {ledger.total:.4f}",
            flush=True,
        )

    write_csv(run_dir / "trajectory.csv", trajectory)
    manifest = {
        **config,
        "completed_at": utc_now(),
        "calls": sum(1 for _ in (run_dir / "usage.jsonl").open(encoding="utf-8")),
        "stage_spend_usd": stage_spend,
        "global_spend_after_usd": ledger.total,
    }
    json_dump(run_dir / "manifest.json", manifest)
    analyze_run(run_dir)
    return run_dir


def mean_se(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def analyze_run(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    fitness_metric = config.get("fitness_metric", "exact")
    response_metric = "mean_accuracy" if fitness_metric == "exact" else "mean_fitness"
    with (run_dir / "trajectory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "rho", "round", "mean_accuracy", "mean_fitness", "pass_at_4", "distinct_fraction",
            "effective_answers", "invalid_fraction", "regressions", "recoveries",
            "fitness_regressions", "fitness_improvements",
        ):
            if key in row:
                row[key] = float(row[key])
        if "mean_fitness" not in row:
            row["mean_fitness"] = row["mean_accuracy"]
        if "fitness_regressions" not in row:
            row["fitness_regressions"] = row["regressions"]
            row["fitness_improvements"] = row["recoveries"]

    aggregate = []
    rhos = sorted({float(row["rho"]) for row in rows})
    rounds = sorted({int(row["round"]) for row in rows})
    for rho in rhos:
        for round_id in rounds:
            group = [
                row for row in rows
                if float(row["rho"]) == rho and int(row["round"]) == round_id
            ]
            item: dict[str, Any] = {"rho": rho, "round": round_id, "n": len(group)}
            for metric in (
                "mean_accuracy", "mean_fitness", "pass_at_4", "distinct_fraction",
                "effective_answers", "invalid_fraction",
            ):
                mean, se = mean_se(float(row[metric]) for row in group)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_se"] = se
            item["regressions"] = sum(float(row["regressions"]) for row in group)
            item["recoveries"] = sum(float(row["recoveries"]) for row in group)
            item["fitness_regressions"] = sum(float(row["fitness_regressions"]) for row in group)
            item["fitness_improvements"] = sum(float(row["fitness_improvements"]) for row in group)
            aggregate.append(item)

    terminal = []
    for rho in rhos:
        series = sorted(
            [row for row in aggregate if row["rho"] == rho],
            key=lambda row: row["round"],
        )
        values = [row[f"{response_metric}_mean"] for row in series]
        deltas = [b - a for a, b in zip(values, values[1:])]
        gain = values[-1] - values[0]
        variation = sum(abs(value) for value in deltas)
        terminal.append(
            {
                "rho": rho,
                "fitness_metric": fitness_metric,
                "initial_fitness": values[0],
                "terminal_fitness": values[-1],
                "initial_accuracy": series[0]["mean_accuracy_mean"],
                "terminal_accuracy": series[-1]["mean_accuracy_mean"],
                "terminal_gain": gain,
                "total_response": variation,
                "erased_response": variation - abs(gain),
                "terminal_pass_at_4": series[-1]["pass_at_4_mean"],
                "terminal_distinct_fraction": series[-1]["distinct_fraction_mean"],
                "terminal_effective_answers": series[-1]["effective_answers_mean"],
                "negative_steps": sum(value < 0 for value in deltas),
            }
        )

    write_csv(run_dir / "aggregate.csv", aggregate)
    write_csv(run_dir / "terminal_summary.csv", terminal)
    summary = {
        "fitness_metric": fitness_metric,
        "response_metric": response_metric,
        "aggregate": aggregate,
        "terminal": terminal,
    }
    json_dump(run_dir / "analysis.json", summary)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    colors = plt.cm.coolwarm
    for index, rho in enumerate(rhos):
        series = sorted(
            [row for row in aggregate if row["rho"] == rho],
            key=lambda row: row["round"],
        )
        color = colors(index / max(1, len(rhos) - 1))
        x = [row["round"] for row in series]
        fitness_title = "Population accuracy" if fitness_metric == "exact" else "Population EED fitness"
        for axis, metric, title in (
            (axes[0, 0], response_metric, fitness_title),
            (axes[0, 1], "pass_at_4", "Pass@4"),
            (axes[1, 0], "effective_answers", "Effective answer count"),
        ):
            y = [row[f"{metric}_mean"] for row in series]
            se = [row[f"{metric}_se"] for row in series]
            axis.plot(x, y, marker="o", color=color, label=f"rho={rho:g}")
            axis.fill_between(
                x,
                [a - b for a, b in zip(y, se)],
                [a + b for a, b in zip(y, se)],
                color=color,
                alpha=0.12,
            )
            axis.set_title(title)
            axis.set_xlabel("recursive round")
            axis.grid(alpha=0.25)
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 1].set_ylim(-0.03, 1.03)
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].axline((0, 0), slope=1, color="#999999", linestyle="--", linewidth=1)
    for row in terminal:
        axes[1, 1].scatter(
            abs(row["terminal_gain"]),
            row["total_response"],
            label=f"rho={row['rho']:g}",
        )
    axes[1, 1].set_title("Response geometry")
    axes[1, 1].set_xlabel("absolute terminal gain")
    axes[1, 1].set_ylabel("total response")
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle(f"PHYBench recursive response spectrum ({fitness_metric} selection)")
    fig.savefig(run_dir / "response_spectrum.png", dpi=180)
    plt.close(fig)
    print(json.dumps(terminal, indent=2))
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--seed", type=int, default=20260719)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("run_dir", type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--stage", choices=("smoke", "pilot", "primary", "control"), required=True)
    run.add_argument("--run-id")
    run.add_argument("--region", default="us-east-1")
    run.add_argument("--model-id", default=DEFAULT_MODEL)
    run.add_argument("--problem-count", type=int, default=0)
    run.add_argument("--population-size", type=int, default=4)
    run.add_argument("--rounds", type=int, default=6)
    run.add_argument("--replicates", type=int, default=2)
    run.add_argument("--rhos", type=float, nargs="+", default=DEFAULT_RHOS)
    run.add_argument("--fitness-metric", choices=("exact", "eed"), default="exact")
    run.add_argument("--output-mode", choices=("json", "tool"), default="tool")
    run.add_argument("--context-mode", choices=("recursive", "restart", "frozen"), default="recursive")
    run.add_argument("--initial-population-run", default="")
    run.add_argument("--workers", type=int, default=24)
    run.add_argument("--temperature", type=float, default=0.8)
    run.add_argument("--top-p", type=float, default=0.9)
    run.add_argument("--max-tokens", type=int, default=900)
    run.add_argument("--seed", type=int, default=20260719)
    run.add_argument("--stage-cap-usd", type=float, required=True)
    run.add_argument("--global-stop-usd", type=float, default=GLOBAL_STOP_USD)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare_splits(args.seed)
        print(json.dumps({
            "eligible_records": payload["eligible_records"],
            "excluded_records": len(payload["excluded_records"]),
            "split_sizes": {key: len(value) for key, value in payload["splits"].items()},
        }, indent=2))
        return
    if args.command == "analyze":
        analyze_run(args.run_dir)
        return
    if not args.run_id:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_id = f"{args.stage}-{stamp}"
    run_dir = run_experiment(args)
    print(f"completed {run_dir}")


if __name__ == "__main__":
    main()
