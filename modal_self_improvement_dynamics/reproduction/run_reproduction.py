"""Resource-scaled reproduction of solver-verifier self-improvement dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as functional
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


FINAL_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[0-9]+)?")


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    dataset_id: str
    dataset_config: str
    train_size: int
    eval_size: int
    candidate_count: int
    verification_threshold: float
    solver_temperature: float
    verifier_temperature: float
    max_new_tokens: int
    max_sequence_length: int
    epochs: int
    checkpoints_per_epoch: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    seed: int


@dataclass
class Candidate:
    text: str
    total_nll: float
    token_count: int
    verifier_score: float
    correct: bool

    @property
    def normalized_nll(self) -> float:
        return self.total_nll / max(1, self.token_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def load_config(path: Path, seed_override: int | None) -> ExperimentConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    if seed_override is not None:
        values["seed"] = seed_override
    config = ExperimentConfig(**values)
    if config.train_size < 1 or config.eval_size < 1:
        raise ValueError("train and evaluation sets must be nonempty")
    if config.candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    if config.checkpoints_per_epoch < 1:
        raise ValueError("checkpoints_per_epoch must be positive")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shuffled_subset(dataset, size: int, seed: int) -> list[dict[str, str]]:
    if size > len(dataset):
        raise ValueError(f"requested {size} records from a split of {len(dataset)}")
    return [dict(row) for row in dataset.shuffle(seed=seed).select(range(size))]


def synthetic_arithmetic_records(size: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    records: list[dict[str, str]] = []
    templates = ("add", "subtract", "multiply", "two_step", "groups_plus")
    for index in range(size):
        kind = templates[index % len(templates)]
        if kind == "add":
            first = rng.randint(5, 80)
            second = rng.randint(5, 80)
            question = (
                f"Nora has {first} stickers and buys {second} more. "
                "How many stickers does she have now?"
            )
            value = first + second
        elif kind == "subtract":
            first = rng.randint(30, 120)
            second = rng.randint(5, first - 5)
            question = (
                f"A shelf has {first} books. {second} books are removed. "
                "How many books remain?"
            )
            value = first - second
        elif kind == "multiply":
            groups = rng.randint(2, 12)
            per_group = rng.randint(2, 12)
            question = (
                f"There are {groups} bags with {per_group} apples in each bag. "
                "How many apples are there altogether?"
            )
            value = groups * per_group
        elif kind == "two_step":
            start = rng.randint(20, 90)
            gained = rng.randint(5, 40)
            given = rng.randint(3, min(35, start + gained - 1))
            question = (
                f"Eli starts with {start} cards, receives {gained} more, and "
                f"then gives away {given}. How many cards does Eli have left?"
            )
            value = start + gained - given
        else:
            groups = rng.randint(2, 10)
            per_group = rng.randint(2, 10)
            extra = rng.randint(1, 20)
            question = (
                f"A teacher fills {groups} boxes with {per_group} pencils in "
                f"each box and has {extra} extra pencils. How many pencils are "
                "there in total?"
            )
            value = groups * per_group + extra
        records.append({"question": question, "answer": f"#### {value}"})
    return records


def solver_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Solve the math problem carefully. End with a line of the form "
                "FINAL: <number>."
            ),
        },
        {"role": "user", "content": question},
    ]


def verifier_messages(question: str, response: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Judge whether the candidate solution's final numerical answer "
                "correctly solves the problem. Reply only TRUE or FALSE."
            ),
        },
        {
            "role": "user",
            "content": f"Problem:\n{question}\n\nCandidate solution:\n{response}",
        },
    ]


def chat_ids(tokenizer, messages: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    return encoded.to(device)


def trim_generated(tokens: torch.Tensor, eos_token_id: int | None, pad_token_id: int) -> torch.Tensor:
    values = tokens.tolist()
    end = len(values)
    for index, token in enumerate(values):
        if token == pad_token_id or (eos_token_id is not None and token == eos_token_id):
            end = index + (1 if token == eos_token_id else 0)
            break
    return tokens[:end]


@torch.inference_mode()
def response_nll(
    model,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> tuple[float, int]:
    full_ids = torch.cat((prompt_ids, response_ids.unsqueeze(0)), dim=1)
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    output = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), labels=labels)
    count = int(response_ids.numel())
    return float(output.loss.detach().cpu()) * count, count


@torch.inference_mode()
def conditional_label_logprob(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    label: str,
    temperature: float,
) -> float:
    label_ids = tokenizer(label, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(prompt_ids.device)
    full_ids = torch.cat((prompt_ids, label_ids), dim=1)
    logits = model(input_ids=full_ids).logits / temperature
    start = prompt_ids.shape[1] - 1
    label_logits = logits[:, start : start + label_ids.shape[1], :]
    log_probabilities = functional.log_softmax(label_logits, dim=-1)
    selected = log_probabilities.gather(2, label_ids.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().detach().cpu())


@torch.inference_mode()
def verifier_score(
    model,
    tokenizer,
    question: str,
    response: str,
    temperature: float,
    device: torch.device,
) -> float:
    prompt_ids = chat_ids(tokenizer, verifier_messages(question, response), device)
    true_logprob = conditional_label_logprob(
        model, tokenizer, prompt_ids, " TRUE", temperature
    )
    false_logprob = conditional_label_logprob(
        model, tokenizer, prompt_ids, " FALSE", temperature
    )
    maximum = max(true_logprob, false_logprob)
    true_weight = math.exp(true_logprob - maximum)
    false_weight = math.exp(false_logprob - maximum)
    return true_weight / (true_weight + false_weight)


def parse_number(text: str) -> Decimal | None:
    preferred = text.split("FINAL:")[-1] if "FINAL:" in text else text
    matches = FINAL_NUMBER.findall(preferred)
    if not matches:
        return None
    value = matches[-1].replace("$", "").replace(",", "")
    try:
        if "/" in value:
            numerator, denominator = value.split("/", maxsplit=1)
            return Decimal(numerator) / Decimal(denominator)
        return Decimal(value)
    except (InvalidOperation, ZeroDivisionError):
        return None


def gold_number(answer: str) -> Decimal | None:
    return parse_number(answer.split("####")[-1])


def is_correct(response: str, answer: str) -> bool:
    predicted = parse_number(response)
    target = gold_number(answer)
    if predicted is None or target is None:
        return False
    return abs(predicted - target) <= Decimal("1e-6")


@torch.inference_mode()
def generate_candidates(
    model,
    tokenizer,
    record: dict[str, str],
    config: ExperimentConfig,
    device: torch.device,
) -> list[Candidate]:
    prompt_ids = chat_ids(tokenizer, solver_messages(record["question"]), device)
    attention_mask = torch.ones_like(prompt_ids)
    generated = model.generate(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=config.solver_temperature,
        top_p=1.0,
        num_return_sequences=config.candidate_count,
        max_new_tokens=config.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    candidates: list[Candidate] = []
    for sequence in generated:
        response_ids = trim_generated(
            sequence[prompt_ids.shape[1] :],
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        total_nll, token_count = response_nll(model, prompt_ids, response_ids)
        score = verifier_score(
            model,
            tokenizer,
            record["question"],
            text,
            config.verifier_temperature,
            device,
        )
        candidates.append(
            Candidate(
                text=text,
                total_nll=total_nll,
                token_count=token_count,
                verifier_score=score,
                correct=is_correct(text, record["answer"]),
            )
        )
    return candidates


def select_candidate(
    candidates: list[Candidate], threshold: float
) -> tuple[Candidate, bool]:
    qualified = [candidate for candidate in candidates if candidate.verifier_score >= threshold]
    fallback = not qualified
    if fallback:
        highest_score = max(candidate.verifier_score for candidate in candidates)
        qualified = [
            candidate
            for candidate in candidates
            if candidate.verifier_score == highest_score
        ]
    return min(qualified, key=lambda candidate: candidate.normalized_nll), fallback


def evaluate(
    model,
    tokenizer,
    records: list[dict[str, str]],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    solver_uncertainty: list[float] = []
    verifier_uncertainty: list[float] = []
    solver_normalized_uncertainty: list[float] = []
    verifier_normalized_uncertainty: list[float] = []
    solver_token_counts: list[float] = []
    verifier_token_counts: list[float] = []
    solver_accuracy: list[float] = []
    verifier_accuracy: list[float] = []
    verifier_scores: list[float] = []
    fallbacks: list[float] = []
    candidate_correctness: list[float] = []
    judgment_correctness: list[float] = []
    verifier_brier: list[float] = []
    correct_candidate_scores: list[float] = []
    incorrect_candidate_scores: list[float] = []
    detail_rows: list[dict[str, object]] = []
    for record_index, record in enumerate(records):
        candidates = generate_candidates(model, tokenizer, record, config, device)
        selected, fallback = select_candidate(candidates, config.verification_threshold)
        solver = candidates[0]
        solver_uncertainty.append(solver.total_nll)
        verifier_uncertainty.append(selected.total_nll)
        solver_normalized_uncertainty.append(solver.normalized_nll)
        verifier_normalized_uncertainty.append(selected.normalized_nll)
        solver_token_counts.append(float(solver.token_count))
        verifier_token_counts.append(float(selected.token_count))
        solver_accuracy.append(float(solver.correct))
        verifier_accuracy.append(float(selected.correct))
        verifier_scores.append(selected.verifier_score)
        fallbacks.append(float(fallback))
        for candidate in candidates:
            target = float(candidate.correct)
            candidate_correctness.append(target)
            judgment_correctness.append(
                float((candidate.verifier_score >= 0.5) == candidate.correct)
            )
            verifier_brier.append((candidate.verifier_score - target) ** 2)
            if candidate.correct:
                correct_candidate_scores.append(candidate.verifier_score)
            else:
                incorrect_candidate_scores.append(candidate.verifier_score)
        detail_rows.append(
            {
                "record_index": record_index,
                "question": record["question"],
                "gold_answer": str(gold_number(record["answer"])),
                "solver": asdict(solver),
                "selected": asdict(selected),
                "candidates": [asdict(candidate) for candidate in candidates],
                "fallback": fallback,
            }
        )
    us = float(np.mean(solver_uncertainty))
    uv = float(np.mean(verifier_uncertainty))
    normalized_us = float(np.mean(solver_normalized_uncertainty))
    normalized_uv = float(np.mean(verifier_normalized_uncertainty))
    metrics = {
        "solver_uncertainty": us,
        "verifier_uncertainty": uv,
        "uncertainty_gap": us - uv,
        "solver_normalized_uncertainty": normalized_us,
        "verifier_normalized_uncertainty": normalized_uv,
        "normalized_uncertainty_gap": normalized_us - normalized_uv,
        "solver_mean_token_count": float(np.mean(solver_token_counts)),
        "verifier_mean_token_count": float(np.mean(verifier_token_counts)),
        "solver_accuracy": float(np.mean(solver_accuracy)),
        "verifier_accuracy": float(np.mean(verifier_accuracy)),
        "accuracy_gap": float(np.mean(verifier_accuracy) - np.mean(solver_accuracy)),
        "mean_selected_verifier_score": float(np.mean(verifier_scores)),
        "fallback_fraction": float(np.mean(fallbacks)),
        "candidate_accuracy": float(np.mean(candidate_correctness)),
        "verifier_judgment_accuracy": float(np.mean(judgment_correctness)),
        "verifier_brier_score": float(np.mean(verifier_brier)),
        "mean_score_correct_candidates": (
            float(np.mean(correct_candidate_scores))
            if correct_candidate_scores
            else float("nan")
        ),
        "mean_score_incorrect_candidates": (
            float(np.mean(incorrect_candidate_scores))
            if incorrect_candidate_scores
            else float("nan")
        ),
    }
    return metrics, detail_rows


class PseudoLabelDataset(Dataset):
    def __init__(self, examples: list[dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def build_training_examples(
    tokenizer,
    records: list[dict[str, str]],
    selected_responses: list[str],
    max_length: int,
) -> list[dict[str, torch.Tensor]]:
    examples: list[dict[str, torch.Tensor]] = []
    for record, response in zip(records, selected_responses):
        prompt = tokenizer.apply_chat_template(
            solver_messages(record["question"]),
            add_generation_prompt=True,
            tokenize=True,
        )
        response_ids = tokenizer(
            response + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
        available = max(1, max_length - len(prompt))
        response_ids = response_ids[:available]
        input_ids = (prompt + response_ids)[:max_length]
        prompt_length = min(len(prompt), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        examples.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )
    return examples


def collate_batch(
    examples: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    maximum = max(example["input_ids"].numel() for example in examples)
    input_ids = torch.full((len(examples), maximum), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), maximum), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), maximum), dtype=torch.long)
    for row, example in enumerate(examples):
        length = example["input_ids"].numel()
        input_ids[row, :length] = example["input_ids"]
        labels[row, :length] = example["labels"]
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def append_csv(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_pseudolabels(
    model,
    tokenizer,
    records: list[dict[str, str]],
    config: ExperimentConfig,
    device: torch.device,
    output_path: Path,
) -> list[str]:
    rows: list[dict[str, object]] = []
    selected_responses: list[str] = []
    model.eval()
    for index, record in enumerate(records):
        candidates = generate_candidates(model, tokenizer, record, config, device)
        selected, fallback = select_candidate(candidates, config.verification_threshold)
        selected_responses.append(selected.text)
        rows.append(
            {
                "record_index": index,
                "question": record["question"],
                "gold_answer": str(gold_number(record["answer"])),
                "fallback": fallback,
                "selected": asdict(selected),
                "candidates": [asdict(candidate) for candidate in candidates],
            }
        )
    write_jsonl(output_path, rows)
    return selected_responses


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = select_device(args.device)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
    ).to(device)
    base_model.config.use_cache = True

    if config.dataset_id == "synthetic_arithmetic":
        train_records = synthetic_arithmetic_records(config.train_size, config.seed)
        eval_records = synthetic_arithmetic_records(config.eval_size, config.seed + 1)
    else:
        raw_train = load_dataset(
            config.dataset_id, config.dataset_config, split="train"
        )
        raw_eval = load_dataset(
            config.dataset_id, config.dataset_config, split="test"
        )
        train_records = shuffled_subset(raw_train, config.train_size, config.seed)
        eval_records = shuffled_subset(raw_eval, config.eval_size, config.seed + 1)

    manifest = {
        "config": asdict(config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    selected_responses = prepare_pseudolabels(
        base_model,
        tokenizer,
        train_records,
        config,
        device,
        args.output / "pseudolabels.jsonl",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    examples = build_training_examples(
        tokenizer,
        train_records,
        selected_responses,
        config.max_sequence_length,
    )
    loader = DataLoader(
        PseudoLabelDataset(examples),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = len(loader)
    evaluation_interval = max(
        1, int(math.ceil(steps_per_epoch / config.checkpoints_per_epoch))
    )
    total_steps = config.epochs * steps_per_epoch
    trajectory_path = args.output / "trajectory.csv"
    detail_directory = args.output / "checkpoint_details"
    detail_directory.mkdir(exist_ok=True)

    def checkpoint(global_step: int, training_loss: float | None) -> None:
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        seed_everything(config.seed + 100_000)
        metrics, details = evaluate(model, tokenizer, eval_records, config, device)
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        row: dict[str, object] = {
            "seed": config.seed,
            "global_step": global_step,
            "epoch": global_step / steps_per_epoch,
            "training_loss": training_loss if training_loss is not None else "",
            "elapsed_seconds": time.time() - started,
            **metrics,
        }
        append_csv(trajectory_path, row)
        write_jsonl(detail_directory / f"step_{global_step:05d}.jsonl", details)

    checkpoint(0, None)
    model.config.use_cache = False
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    accumulated_loss = 0.0
    for _epoch in range(config.epochs):
        for batch_index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += float(output.loss.detach().cpu())
            should_update = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == steps_per_epoch
            )
            if should_update:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % evaluation_interval == 0 or global_step == total_steps:
                mean_loss = accumulated_loss / evaluation_interval
                model.config.use_cache = True
                checkpoint(global_step, mean_loss)
                model.config.use_cache = False
                model.train()
                accumulated_loss = 0.0

    model.save_pretrained(args.output / "final_adapter")
    tokenizer.save_pretrained(args.output / "final_adapter")
    elapsed = time.time() - started
    manifest["elapsed_seconds"] = elapsed
    manifest["peak_gpu_memory_mb"] = (
        float(torch.cuda.max_memory_allocated() / 2**20)
        if device.type == "cuda"
        else None
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"completed {global_step} training steps in {elapsed:.1f}s; "
        f"peak GPU memory {manifest['peak_gpu_memory_mb']} MB"
    )


if __name__ == "__main__":
    main()
