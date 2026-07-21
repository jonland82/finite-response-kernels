#!/usr/bin/env python3
"""Finite-horizon closed-source exhaustion pilot for a tiny language model.

The experiment trains independent replicas of a small decoder-only Transformer.
Each replica sees a synthetic corpus of records whose red/blue code tokens are
independent source bits.  In the closed condition the corpus is replayed; in the
open control, new record/code facts are revealed during training.

No claim is made that the observable information lower bound is the exact
mutual information of the full real-valued parameter trajectory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# PyTorch and the Windows matplotlib wheels can load separate Intel OpenMP
# runtimes.  This compatibility switch is restricted to local Windows runs;
# the AWS Linux execution does not use it.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BOS = 0
RECORD = 1
CODE = 2
RED = 3
BLUE = 4
EOS = 5
ID_BASE = 6


@dataclass
class Config:
    output_dir: str
    replicas: int = 128
    steps: int = 400
    batch_size: int = 4
    initial_facts: int = 16
    maximum_facts: int = 32
    reveal_interval: int = 20
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    ff_width: int = 256
    learning_rate: float = 0.003
    weight_decay: float = 0.001
    seed: int = 20260720
    influence_replicas: int = 16
    influence_horizon: int = 50
    influence_epsilon: float = 0.1
    influence_times: tuple[int, ...] = (50, 150, 300)
    bootstrap_samples: int = 500
    max_runtime_seconds: int = 2700
    skip_influence: bool = False
    smoke: bool = False


class ReplicaLayerNorm(nn.Module):
    def __init__(self, replicas: int, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(replicas, width))
        self.bias = nn.Parameter(torch.zeros(replicas, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) * torch.rsqrt(variance + 1e-5)
        return normalized * self.weight[:, None, None, :] + self.bias[:, None, None, :]


class ReplicaLinear(nn.Module):
    def __init__(
        self,
        replicas: int,
        input_width: int,
        output_width: int,
        generator: torch.Generator,
        scale: float,
    ) -> None:
        super().__init__()
        base = torch.randn(input_width, output_width, generator=generator) * scale
        self.weight = nn.Parameter(base.unsqueeze(0).repeat(replicas, 1, 1))
        self.bias = nn.Parameter(torch.zeros(replicas, output_width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("rbti, rio -> rbto", x, self.weight) + self.bias[:, None, None, :]


class ReplicaBlock(nn.Module):
    def __init__(
        self,
        replicas: int,
        width: int,
        heads: int,
        ff_width: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.ln1 = ReplicaLayerNorm(replicas, width)
        self.qkv = ReplicaLinear(replicas, width, 3 * width, generator, 0.02)
        self.attention_out = ReplicaLinear(replicas, width, width, generator, 0.02)
        self.ln2 = ReplicaLayerNorm(replicas, width)
        self.ff1 = ReplicaLinear(replicas, width, ff_width, generator, 0.02)
        self.ff2 = ReplicaLinear(replicas, ff_width, width, generator, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        replicas, batch, length, width = x.shape
        qkv = self.qkv(self.ln1(x))
        qkv = qkv.reshape(replicas, batch, length, 3, self.heads, self.head_width)
        q, k, v = (qkv[:, :, :, index].permute(0, 1, 3, 2, 4) for index in range(3))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v).permute(0, 1, 3, 2, 4)
        context = context.reshape(replicas, batch, length, width)
        x = x + self.attention_out(context)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class BatchedTinyLM(nn.Module):
    def __init__(self, config: Config, replicas: int, vocabulary_size: int) -> None:
        super().__init__()
        self.replicas = replicas
        self.vocabulary_size = vocabulary_size
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 101)
        base_tokens = torch.randn(vocabulary_size, config.d_model, generator=generator) * 0.02
        base_positions = torch.randn(5, config.d_model, generator=generator) * 0.02
        self.token_embedding = nn.Parameter(base_tokens.unsqueeze(0).repeat(replicas, 1, 1))
        self.position_embedding = nn.Parameter(
            base_positions.unsqueeze(0).repeat(replicas, 1, 1)
        )
        self.blocks = nn.ModuleList(
            [
                ReplicaBlock(
                    replicas,
                    config.d_model,
                    config.heads,
                    config.ff_width,
                    generator,
                )
                for _ in range(config.layers)
            ]
        )
        self.final_norm = ReplicaLayerNorm(replicas, config.d_model)
        self.lm_head = ReplicaLinear(
            replicas,
            config.d_model,
            vocabulary_size,
            generator,
            0.02,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[0] != self.replicas:
            raise ValueError("tokens must have shape [replicas, batch, length]")
        replica_index = torch.arange(self.replicas, device=tokens.device)[:, None, None]
        x = self.token_embedding[replica_index, tokens]
        x = x + self.position_embedding[:, None, : tokens.shape[-1], :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def build_labels(config: Config, replicas: int) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    return rng.integers(0, 2, size=(replicas, config.maximum_facts), dtype=np.int64)


def build_corpus(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    replicas, fact_count = labels.shape
    corpus = np.empty((replicas, fact_count, 6), dtype=np.int64)
    corpus[:, :, 0] = BOS
    corpus[:, :, 1] = RECORD
    corpus[:, :, 2] = ID_BASE + np.arange(fact_count)[None, :]
    corpus[:, :, 3] = CODE
    corpus[:, :, 4] = np.where(labels == 1, RED, BLUE)
    corpus[:, :, 5] = EOS
    return torch.as_tensor(corpus, dtype=torch.long, device=device)


def make_schedules(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed + 1)
    closed = rng.integers(
        0,
        config.initial_facts,
        size=(config.steps, config.batch_size),
        dtype=np.int64,
    )
    opened = np.empty_like(closed)
    revealed_counts = np.empty(config.steps, dtype=np.int64)
    revealed = config.initial_facts
    for step in range(config.steps):
        if (
            step > 0
            and step % config.reveal_interval == 0
            and revealed < config.maximum_facts
        ):
            new_fact = revealed
            revealed += 1
            opened[step, 0] = new_fact
            opened[step, 1:] = rng.integers(
                0, revealed, size=config.batch_size - 1, dtype=np.int64
            )
        else:
            opened[step] = rng.integers(
                0, revealed, size=config.batch_size, dtype=np.int64
            )
        revealed_counts[step] = revealed
    return closed, opened, revealed_counts


def select_sequences(corpus: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    replicas = corpus.shape[0]
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(replicas, -1)
    replica_index = torch.arange(replicas, device=corpus.device)[:, None]
    return corpus[replica_index, indices]


def sequence_loss(
    model: BatchedTinyLM,
    sequences: torch.Tensor,
    sequence_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = sequences[:, :, :-1]
    targets = sequences[:, :, 1:]
    logits = model(inputs)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(sequences.shape[0], sequences.shape[1], -1)
    per_sequence = token_losses.mean(dim=-1)
    if sequence_weights is not None:
        per_sequence = per_sequence * sequence_weights
    per_replica = per_sequence.mean(dim=-1)
    return per_replica.sum(), per_replica.detach()


def fit_logistic_calibrator(margins: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit a two-parameter logistic decoder with damped Newton updates."""
    center = float(np.mean(margins))
    scale = float(np.std(margins))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    standardized = (margins - center) / scale
    design = np.column_stack((standardized, np.ones_like(standardized)))
    coefficients = np.zeros(2, dtype=np.float64)
    ridge = np.diag([1e-4, 1e-8])
    for _ in range(40):
        logits = np.clip(design @ coefficients, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - labels) + ridge @ coefficients
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
        hessian = design.T @ (design * weights[:, None]) + ridge
        update = np.linalg.solve(hessian, gradient)
        coefficients -= update
        if float(np.linalg.norm(update)) < 1e-8:
            break
    return coefficients, center, scale


def crossfit_information(
    margins: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decode source bits with folds separated by complete corpus realization."""
    replicas = margins.shape[0]
    fold_count = 4 if replicas >= 8 else 2
    decoded = np.empty_like(margins, dtype=np.float64)
    world_indices = np.arange(replicas)
    for fold in range(fold_count):
        test_worlds = world_indices[world_indices % fold_count == fold]
        train_worlds = world_indices[world_indices % fold_count != fold]
        coefficients, center, scale = fit_logistic_calibrator(
            margins[train_worlds].reshape(-1).astype(np.float64),
            labels[train_worlds].reshape(-1).astype(np.float64),
        )
        standardized = (margins[test_worlds] - center) / scale
        logits = np.clip(
            coefficients[0] * standardized + coefficients[1], -30.0, 30.0
        )
        decoded[test_worlds] = 1.0 / (1.0 + np.exp(-logits))
    probabilities = np.clip(decoded, 1e-9, 1.0 - 1e-9)
    cross_entropy_nats = -(
        labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)
    )
    information_bits = 1.0 - cross_entropy_nats / math.log(2.0)
    return information_bits, probabilities


@torch.no_grad()
def evaluate(
    model: BatchedTinyLM,
    corpus: torch.Tensor,
    labels: np.ndarray,
    facts_for_loss: int,
) -> dict[str, np.ndarray | float]:
    sequences = corpus[:, :, :]
    logits = model(sequences[:, :, :-1])
    code_logits = logits[:, :, 3, :]
    margins = code_logits[:, :, RED] - code_logits[:, :, BLUE]
    label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=margins.device)
    margin_values = margins.cpu().numpy()
    information_values, decoded_probabilities = crossfit_information(
        margin_values, labels
    )
    decoded_predictions = decoded_probabilities >= 0.5
    accuracy_values = decoded_predictions == labels
    target_tokens = torch.where(
        label_tensor[:, :facts_for_loss] == 1,
        torch.full_like(label_tensor[:, :facts_for_loss], RED, dtype=torch.long),
        torch.full_like(label_tensor[:, :facts_for_loss], BLUE, dtype=torch.long),
    )
    code_loss = F.cross_entropy(
        code_logits[:, :facts_for_loss, :].reshape(-1, code_logits.shape[-1]),
        target_tokens.reshape(-1),
        reduction="none",
    ).reshape(model.replicas, facts_for_loss).mean(dim=1)
    source_slice = slice(0, facts_for_loss)
    return {
        "margins": margin_values,
        "information_per_bit": information_values,
        "information_per_world": information_values[:, source_slice].sum(axis=1),
        "accuracy_per_world": accuracy_values[:, source_slice].mean(axis=1),
        "code_loss_per_world": code_loss.cpu().numpy(),
    }


def checkpoint_steps(steps: int) -> list[int]:
    candidates = [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, steps]
    return sorted({value for value in candidates if 0 <= value <= steps})


def bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if len(values) < 2 or samples <= 0:
        value = float(np.mean(values))
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


class RuntimeGuard:
    def __init__(self, seconds: int) -> None:
        self.start = time.monotonic()
        self.deadline = self.start + seconds

    def check(self, reserve_seconds: int = 0) -> None:
        if time.monotonic() + reserve_seconds >= self.deadline:
            raise TimeoutError("experiment runtime guard reached")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    @property
    def remaining(self) -> float:
        return self.deadline - time.monotonic()


def train_condition(
    config: Config,
    condition: str,
    schedule: np.ndarray,
    revealed_counts: np.ndarray,
    device: torch.device,
    guard: RuntimeGuard,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    labels = build_labels(config, config.replicas)
    corpus = build_corpus(labels, device)
    model = BatchedTinyLM(config, config.replicas, ID_BASE + config.maximum_facts).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoints = checkpoint_steps(config.steps)
    rows: list[dict[str, object]] = []
    margin_history: list[np.ndarray] = []
    bit_information_history: list[np.ndarray] = []
    previous_information: float | None = None

    for completed_steps in range(config.steps + 1):
        if completed_steps in checkpoints:
            facts_for_loss = (
                config.initial_facts
                if condition == "closed"
                else (
                    config.initial_facts
                    if completed_steps == 0
                    else int(revealed_counts[completed_steps - 1])
                )
            )
            metrics = evaluate(model, corpus, labels, facts_for_loss)
            information_values = np.asarray(metrics["information_per_world"])
            mean_information = float(information_values.mean())
            low, high = bootstrap_interval(
                information_values,
                config.bootstrap_samples,
                config.seed + completed_steps + (0 if condition == "closed" else 10000),
            )
            delta = (
                0.0
                if previous_information is None
                else mean_information - previous_information
            )
            rows.append(
                {
                    "condition": condition,
                    "step": completed_steps,
                    "epoch_equivalent": completed_steps * config.batch_size / config.initial_facts,
                    "available_source_bits": facts_for_loss,
                    "information_lower_bound_bits": mean_information,
                    "information_ci_low": low,
                    "information_ci_high": high,
                    "information_increment_proxy_bits": delta,
                    "code_accuracy": float(np.mean(metrics["accuracy_per_world"])),
                    "code_loss_nats": float(np.mean(metrics["code_loss_per_world"])),
                }
            )
            previous_information = mean_information
            margin_history.append(np.asarray(metrics["margins"], dtype=np.float32))
            bit_information_history.append(
                np.asarray(metrics["information_per_bit"], dtype=np.float32)
            )
            print(
                f"{condition:6s} step={completed_steps:4d} "
                f"J_lower={mean_information:8.3f} bits "
                f"accuracy={rows[-1]['code_accuracy']:.3f}",
                flush=True,
            )
        if completed_steps == config.steps:
            break
        guard.check(reserve_seconds=0 if config.smoke else 180)
        indices = torch.as_tensor(schedule[completed_steps], dtype=torch.long, device=device)
        sequences = select_sequences(corpus, indices)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = sequence_loss(model, sequences)
        loss.backward()
        optimizer.step()

    return rows, np.stack(margin_history), np.stack(bit_information_history)


def clone_training_state(
    model: BatchedTinyLM,
    optimizer: torch.optim.Optimizer,
    config: Config,
    replicas: int,
    device: torch.device,
) -> tuple[BatchedTinyLM, torch.optim.Optimizer]:
    cloned_model = BatchedTinyLM(config, replicas, ID_BASE + config.maximum_facts).to(device)
    cloned_model.load_state_dict(copy.deepcopy(model.state_dict()))
    cloned_optimizer = torch.optim.AdamW(
        cloned_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    cloned_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return cloned_model, cloned_optimizer


@torch.no_grad()
def selected_margin(
    model: BatchedTinyLM,
    corpus: torch.Tensor,
    selected_facts: torch.Tensor,
) -> np.ndarray:
    sequences = select_sequences(corpus, selected_facts[:, None])
    logits = model(sequences[:, :, :-1])[:, 0, 3, :]
    return (logits[:, RED] - logits[:, BLUE]).cpu().numpy()


def injection_step(
    model: BatchedTinyLM,
    optimizer: torch.optim.Optimizer,
    corpus: torch.Tensor,
    ordinary_indices: np.ndarray,
    selected_facts: torch.Tensor,
    weight: float,
) -> None:
    replicas = corpus.shape[0]
    batch = torch.as_tensor(ordinary_indices, dtype=torch.long, device=corpus.device)
    batch = batch.unsqueeze(0).expand(replicas, -1).clone()
    batch[:, 0] = selected_facts
    sequences = select_sequences(corpus, batch)
    weights = torch.ones(replicas, batch.shape[1], device=corpus.device)
    weights[:, 0] = weight
    optimizer.zero_grad(set_to_none=True)
    loss, _ = sequence_loss(model, sequences, weights)
    loss.backward()
    optimizer.step()


def regular_step(
    model: BatchedTinyLM,
    optimizer: torch.optim.Optimizer,
    corpus: torch.Tensor,
    indices: np.ndarray,
) -> None:
    batch = torch.as_tensor(indices, dtype=torch.long, device=corpus.device)
    sequences = select_sequences(corpus, batch)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = sequence_loss(model, sequences)
    loss.backward()
    optimizer.step()


def run_influence(
    config: Config,
    closed_schedule: np.ndarray,
    device: torch.device,
    guard: RuntimeGuard,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    replicas = min(config.influence_replicas, config.replicas)
    labels = build_labels(config, replicas)
    corpus = build_corpus(labels, device)
    selected = torch.arange(replicas, device=device) % config.initial_facts
    lag_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for injection_time in config.influence_times:
        if injection_time >= config.steps:
            continue
        guard.check(reserve_seconds=0 if config.smoke else 300)
        model = BatchedTinyLM(config, replicas, ID_BASE + config.maximum_facts).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        for step in range(injection_time):
            regular_step(model, optimizer, corpus, closed_schedule[step])
        plus_model, plus_optimizer = clone_training_state(
            model, optimizer, config, replicas, device
        )
        minus_model, minus_optimizer = clone_training_state(
            model, optimizer, config, replicas, device
        )
        del model, optimizer
        injection_step(
            plus_model,
            plus_optimizer,
            corpus,
            closed_schedule[injection_time],
            selected,
            1.0 + config.influence_epsilon,
        )
        injection_step(
            minus_model,
            minus_optimizer,
            corpus,
            closed_schedule[injection_time],
            selected,
            1.0 - config.influence_epsilon,
        )
        plus_trajectory: list[np.ndarray] = []
        minus_trajectory: list[np.ndarray] = []
        plus_trajectory.append(selected_margin(plus_model, corpus, selected))
        minus_trajectory.append(selected_margin(minus_model, corpus, selected))
        horizon = min(config.influence_horizon, config.steps - injection_time - 1)
        for lag in range(1, horizon + 1):
            guard.check(reserve_seconds=0 if config.smoke else 180)
            step = injection_time + lag
            regular_step(plus_model, plus_optimizer, corpus, closed_schedule[step])
            regular_step(minus_model, minus_optimizer, corpus, closed_schedule[step])
            plus_trajectory.append(selected_margin(plus_model, corpus, selected))
            minus_trajectory.append(selected_margin(minus_model, corpus, selected))
        derivative = (
            np.stack(plus_trajectory) - np.stack(minus_trajectory)
        ) / (2.0 * config.influence_epsilon)
        magnitude = np.mean(np.abs(derivative), axis=1)
        total = float(magnitude.sum())
        q = magnitude / total if total > 0 else np.zeros_like(magnitude)
        nonzero = q[q > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) else 0.0
        tail_start = max(0, len(magnitude) - 10)
        tail_fraction = float(magnitude[tail_start:].sum() / total) if total > 0 else 0.0
        for lag, (mag, mass, signed) in enumerate(
            zip(magnitude, q, derivative.mean(axis=1), strict=True)
        ):
            lag_rows.append(
                {
                    "injection_step": injection_time,
                    "lag": lag,
                    "mean_absolute_derivative": float(mag),
                    "normalized_mass": float(mass),
                    "mean_signed_derivative": float(signed),
                }
            )
        summary_rows.append(
            {
                "injection_step": injection_time,
                "observed_horizon": horizon,
                "discrete_entropy_nats": entropy,
                "effective_lag_count": float(math.exp(entropy)),
                "peak_mass": float(q.max()) if len(q) else 0.0,
                "last_ten_lag_mass": tail_fraction,
                "signed_to_absolute_ratio": float(
                    np.abs(derivative.mean(axis=1)).sum() / total
                )
                if total > 0
                else 0.0,
            }
        )
        print(
            f"influence t={injection_time:4d} effective_lags={math.exp(entropy):.2f} "
            f"peak={summary_rows[-1]['peak_mass']:.3f} tail={tail_fraction:.3f}",
            flush=True,
        )
        del plus_model, minus_model, plus_optimizer, minus_optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return lag_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(
    output_dir: Path,
    trajectory_rows: list[dict[str, object]],
    influence_rows: list[dict[str, object]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - artifact generation is optional
        print(f"plotting unavailable: {error}", file=sys.stderr)
        return

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for condition, color in (("closed", "#1f77b4"), ("open", "#d62728")):
        selected = [row for row in trajectory_rows if row["condition"] == condition]
        x = np.array([float(row["epoch_equivalent"]) for row in selected])
        y = np.array([float(row["information_lower_bound_bits"]) for row in selected])
        low = np.array([float(row["information_ci_low"]) for row in selected])
        high = np.array([float(row["information_ci_high"]) for row in selected])
        axes[0].plot(x, y, marker="o", label=condition, color=color)
        axes[0].fill_between(x, low, high, color=color, alpha=0.15)
        axes[1].plot(
            x,
            [float(row["information_increment_proxy_bits"]) for row in selected],
            marker="o",
            label=condition,
            color=color,
        )
    axes[0].set_title("Recoverable source information")
    axes[0].set_xlabel("closed-corpus epoch equivalents")
    axes[0].set_ylabel("variational lower bound (bits)")
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Checkpoint increment proxy")
    axes[1].set_xlabel("closed-corpus epoch equivalents")
    axes[1].set_ylabel("change in lower bound (bits)")

    injection_times = sorted({int(row["injection_step"]) for row in influence_rows})
    for injection_time in injection_times:
        selected = [
            row for row in influence_rows if int(row["injection_step"]) == injection_time
        ]
        axes[2].plot(
            [int(row["lag"]) for row in selected],
            [float(row["normalized_mass"]) for row in selected],
            label=f"t={injection_time}",
        )
    axes[2].set_title("Empirical influence magnitude")
    axes[2].set_xlabel("lag (updates)")
    axes[2].set_ylabel("normalized mass")
    if injection_times:
        axes[2].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "closed_source_pilot.png", dpi=180)
    plt.close(figure)


def write_report(
    output_dir: Path,
    config: Config,
    trajectory_rows: list[dict[str, object]],
    influence_summary: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    closed = [row for row in trajectory_rows if row["condition"] == "closed"]
    opened = [row for row in trajectory_rows if row["condition"] == "open"]
    closed_first = closed[0]
    closed_final = closed[-1]
    open_final = opened[-1]
    last_increment = float(closed_final["information_increment_proxy_bits"])
    lines = [
        "# AWS closed-source language-model pilot",
        "",
        "## Outcome",
        "",
        f"The closed condition's observable information lower bound moved from "
        f"{float(closed_first['information_lower_bound_bits']):.3f} to "
        f"{float(closed_final['information_lower_bound_bits']):.3f} bits out of a "
        f"16-bit source. Its final checkpoint increment proxy was {last_increment:.3f} bits.",
        "",
        f"The open control ended at {float(open_final['information_lower_bound_bits']):.3f} "
        f"recoverable bits after the available source expanded to "
        f"{int(open_final['available_source_bits'])} bits.",
        "",
        "These quantities are variational lower bounds derived from the model's own "
        "red/blue probabilities. They are not estimates of the exact mutual information "
        "in the complete real-valued parameter trajectory, and finite-horizon behavior "
        "does not prove an asymptotic theorem.",
        "",
        "## Final checkpoint",
        "",
        "| condition | available bits | lower bound (bits) | code accuracy | code loss (nats) |",
        "|---|---:|---:|---:|---:|",
        f"| closed | {int(closed_final['available_source_bits'])} | "
        f"{float(closed_final['information_lower_bound_bits']):.3f} | "
        f"{float(closed_final['code_accuracy']):.3f} | "
        f"{float(closed_final['code_loss_nats']):.4f} |",
        f"| open | {int(open_final['available_source_bits'])} | "
        f"{float(open_final['information_lower_bound_bits']):.3f} | "
        f"{float(open_final['code_accuracy']):.3f} | "
        f"{float(open_final['code_loss_nats']):.4f} |",
        "",
        "## Influence summaries",
        "",
        "| injection step | effective lag count | peak mass | last-ten-lag mass | signed/absolute ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in influence_summary:
        lines.append(
            f"| {int(row['injection_step'])} | {float(row['effective_lag_count']):.2f} | "
            f"{float(row['peak_mass']):.3f} | {float(row['last_ten_lag_mass']):.3f} | "
            f"{float(row['signed_to_absolute_ratio']):.3f} |"
        )
    if not influence_summary:
        lines.append("| _not completed_ | | | | |")
    lines.extend(
        [
            "",
            "A large last-ten-lag mass means the response did not decay within the "
            "measurement window, so treating the truncated curve as a finite normalized "
            "kernel would be questionable.",
            "",
            "## Reproduction metadata",
            "",
            "```json",
            json.dumps(metadata, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    (output_dir / "RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--replicas", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--max-runtime-seconds", type=int, default=2700)
    parser.add_argument("--skip-influence", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    config = Config(
        output_dir=arguments.output_dir,
        replicas=arguments.replicas,
        steps=arguments.steps,
        bootstrap_samples=arguments.bootstrap_samples,
        max_runtime_seconds=arguments.max_runtime_seconds,
        skip_influence=arguments.skip_influence,
        smoke=arguments.smoke,
    )
    if config.smoke:
        config.replicas = min(config.replicas, 4)
        config.steps = min(config.steps, 12)
        config.initial_facts = 4
        config.maximum_facts = 8
        config.reveal_interval = 3
        config.d_model = 32
        config.heads = 4
        config.layers = 1
        config.ff_width = 64
        config.influence_replicas = 2
        config.influence_horizon = 3
        config.influence_times = (4,)
        config.bootstrap_samples = min(config.bootstrap_samples, 20)
    return config


def main() -> int:
    config = parse_args()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    guard = RuntimeGuard(config.max_runtime_seconds)
    closed_schedule, open_schedule, revealed_counts = make_schedules(config)
    status = "complete"
    trajectory_rows: list[dict[str, object]] = []
    influence_rows: list[dict[str, object]] = []
    influence_summary: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}

    try:
        for condition, schedule in (("closed", closed_schedule), ("open", open_schedule)):
            rows, margins, bit_information = train_condition(
                config,
                condition,
                schedule,
                revealed_counts,
                device,
                guard,
            )
            trajectory_rows.extend(rows)
            arrays[f"{condition}_margins"] = margins
            arrays[f"{condition}_bit_information"] = bit_information
            write_csv(output_dir / "trajectory.csv", trajectory_rows)
            np.savez_compressed(output_dir / "observables.npz", **arrays)
        if not config.skip_influence and (config.smoke or guard.remaining > 360):
            influence_rows, influence_summary = run_influence(
                config, closed_schedule, device, guard
            )
        elif not config.skip_influence:
            status = "source_complete_influence_skipped_for_deadline"
    except TimeoutError as error:
        status = f"partial_timeout: {error}"
        print(status, file=sys.stderr, flush=True)
    except Exception as error:
        status = f"failed: {type(error).__name__}: {error}"
        print(status, file=sys.stderr, flush=True)
        raise
    finally:
        write_csv(output_dir / "trajectory.csv", trajectory_rows)
        write_csv(output_dir / "influence_lags.csv", influence_rows)
        write_csv(output_dir / "influence_summary.csv", influence_summary)
        if arrays:
            np.savez_compressed(output_dir / "observables.npz", **arrays)
        make_figures(output_dir, trajectory_rows, influence_rows)
        script_bytes = Path(__file__).read_bytes()
        metadata: dict[str, object] = {
            "status": status,
            "config": asdict(config),
            "elapsed_seconds": guard.elapsed,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
            "aws_instance_id": os.environ.get("AWS_INSTANCE_ID"),
            "aws_region": os.environ.get("AWS_REGION"),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        if trajectory_rows:
            write_report(
                output_dir,
                config,
                trajectory_rows,
                influence_summary,
                metadata,
            )
    print(f"status={status} elapsed_seconds={guard.elapsed:.1f}", flush=True)
    return 0 if not status.startswith("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
