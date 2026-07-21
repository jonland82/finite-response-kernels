#!/usr/bin/env python3
"""Generate dense LLM grokking trajectories for takeoff-kernel analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    seed: int
    modulus: int
    train_fraction: float
    weight_decay: float
    learning_rate: float
    max_steps: int
    eval_every: int
    batch_size: int
    d_model: int
    heads: int
    layers: int
    ff_width: int
    threads: int
    stage: str


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, ff_width)
        self.ff2 = nn.Linear(ff_width, width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.ln1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        x = x + attended
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        modulus: int,
        d_model: int,
        heads: int,
        layers: int,
        ff_width: int,
    ) -> None:
        super().__init__()
        self.modulus = modulus
        self.token_embedding = nn.Embedding(modulus + 3, d_model)
        self.position_embedding = nn.Parameter(torch.empty(4, d_model))
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, heads, ff_width) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, modulus, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.apply(self._initialize)
        nn.init.normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens) + self.position_embedding[None, :, :]
        for block in self.blocks:
            x = block(x, self.causal_mask)
        return self.output(self.final_norm(x[:, -1]))


def build_dataset(spec: RunSpec) -> tuple[torch.Tensor, ...]:
    pairs = np.asarray(
        [(left, right) for left in range(spec.modulus) for right in range(spec.modulus)],
        dtype=np.int64,
    )
    labels = (pairs[:, 0] + pairs[:, 1]) % spec.modulus
    tokens = np.empty((len(pairs), 4), dtype=np.int64)
    tokens[:, 0] = pairs[:, 0]
    tokens[:, 1] = spec.modulus
    tokens[:, 2] = pairs[:, 1]
    tokens[:, 3] = spec.modulus + 1
    rng = np.random.default_rng(spec.seed + 17)
    order = rng.permutation(len(pairs))
    train_count = int(round(len(pairs) * spec.train_fraction))
    train_indices = order[:train_count]
    test_indices = order[train_count:]
    return (
        torch.from_numpy(tokens[train_indices]),
        torch.from_numpy(labels[train_indices]),
        torch.from_numpy(tokens[test_indices]),
        torch.from_numpy(labels[test_indices]),
    )


@torch.inference_mode()
def evaluate(
    model: TinyDecoderLM,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 2048,
) -> tuple[float, float, int]:
    model.eval()
    losses = 0.0
    correct = 0
    for start in range(0, len(tokens), batch_size):
        batch_tokens = tokens[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]
        logits = model(batch_tokens)
        losses += float(F.cross_entropy(logits, batch_labels, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == batch_labels).sum())
    return losses / len(tokens), correct / len(tokens), correct


def parameter_norm(model: nn.Module) -> float:
    return math.sqrt(
        sum(float(torch.square(value.detach()).sum()) for value in model.parameters())
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(spec_dict: dict[str, object], output_root: str) -> dict[str, object]:
    spec = RunSpec(**spec_dict)
    torch.set_num_threads(spec.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A forked parent may already have initialized PyTorch's interop pool.
        # Spawned AWS workers do not normally take this path, but retaining the
        # fallback makes local embedding safe.
        pass
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    output_dir = Path(output_root) / spec.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y = build_dataset(spec)
    model = TinyDecoderLM(
        spec.modulus,
        spec.d_model,
        spec.heads,
        spec.layers,
        spec.ff_width,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=spec.weight_decay,
    )
    generator = torch.Generator().manual_seed(spec.seed + 1001)
    rows: list[dict[str, object]] = []
    start_time = time.monotonic()
    consecutive_complete = 0
    train_fit_step: int | None = None
    test_half_step: int | None = None
    test_ninety_step: int | None = None

    for step in range(spec.max_steps + 1):
        if step % spec.eval_every == 0:
            train_loss, train_accuracy, train_correct = evaluate(model, train_x, train_y)
            test_loss, test_accuracy, test_correct = evaluate(model, test_x, test_y)
            if train_fit_step is None and train_accuracy >= 0.99:
                train_fit_step = step
            if test_half_step is None and test_accuracy >= 0.50:
                test_half_step = step
            if test_ninety_step is None and test_accuracy >= 0.90:
                test_ninety_step = step
            rows.append(
                {
                    "run_id": spec.run_id,
                    "stage": spec.stage,
                    "seed": spec.seed,
                    "step": step,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_correct": train_correct,
                    "train_count": len(train_y),
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "test_correct": test_correct,
                    "test_count": len(test_y),
                    "parameter_l2_norm": parameter_norm(model),
                    "elapsed_seconds": time.monotonic() - start_time,
                }
            )
            write_csv(output_dir / "trajectory.csv", rows)
            if train_accuracy >= 0.999 and test_accuracy >= 0.995:
                consecutive_complete += 1
            else:
                consecutive_complete = 0
            if consecutive_complete >= 5:
                break
        if step == spec.max_steps:
            break
        model.train()
        indices = torch.randint(
            len(train_y), (min(spec.batch_size, len(train_y)),), generator=generator
        )
        logits = model(train_x[indices])
        loss = F.cross_entropy(logits, train_y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    summary: dict[str, object] = {
        **asdict(spec),
        "train_count": len(train_y),
        "test_count": len(test_y),
        "completed_steps": int(rows[-1]["step"]),
        "elapsed_seconds": time.monotonic() - start_time,
        "train_fit_step": train_fit_step,
        "test_half_step": test_half_step,
        "test_ninety_step": test_ninety_step,
        "final_train_accuracy": float(rows[-1]["train_accuracy"]),
        "final_test_accuracy": float(rows[-1]["test_accuracy"]),
        "final_test_loss": float(rows[-1]["test_loss"]),
        "delayed_generalization": bool(
            train_fit_step is not None
            and test_half_step is not None
            and test_half_step - train_fit_step >= 5 * spec.eval_every
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def launch_specs(
    specs: list[RunSpec], output_root: Path, workers: int
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(run_one, asdict(spec), str(output_root)): spec for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            summary = future.result()
            summaries.append(summary)
            print(
                f"{spec.run_id}: train={summary['final_train_accuracy']:.3f} "
                f"test={summary['final_test_accuracy']:.3f} "
                f"train99={summary['train_fit_step']} test50={summary['test_half_step']} "
                f"seconds={summary['elapsed_seconds']:.1f}",
                flush=True,
            )
    return sorted(summaries, key=lambda row: str(row["run_id"]))


def choose_condition(summaries: list[dict[str, object]]) -> dict[str, object]:
    completed = [
        row
        for row in summaries
        if row["test_half_step"] is not None and row["final_test_accuracy"] >= 0.90
    ]
    candidates = completed or sorted(
        summaries, key=lambda row: float(row["final_test_accuracy"]), reverse=True
    )[:1]

    def score(row: dict[str, object]) -> tuple[float, float]:
        delay = (
            float(row["test_half_step"]) - float(row["train_fit_step"])
            if row["test_half_step"] is not None and row["train_fit_step"] is not None
            else -1.0
        )
        return delay, float(row["final_test_accuracy"])

    return max(candidates, key=score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confirmation-seeds", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output_dir = Path(arguments.output_dir).resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if arguments.smoke:
        modulus = 17
        max_steps = min(arguments.max_steps, 200)
        eval_every = min(arguments.eval_every, 20)
        d_model, layers, ff_width = 32, 1, 64
        pilot_grid = ((0.4, 0.1), (0.5, 1.0))
        confirmation_seeds = min(arguments.confirmation_seeds, 2)
    else:
        modulus = 67
        max_steps = arguments.max_steps
        eval_every = arguments.eval_every
        d_model, layers, ff_width = 64, 1, 256
        pilot_grid = ((0.30, 1.0), (0.40, 1.0), (0.50, 1.0), (0.40, 0.1))
        confirmation_seeds = arguments.confirmation_seeds
    common = {
        "modulus": modulus,
        "learning_rate": 1e-3,
        "max_steps": max_steps,
        "eval_every": eval_every,
        "batch_size": 512,
        "d_model": d_model,
        "heads": 4,
        "layers": layers,
        "ff_width": ff_width,
        "threads": max(1, 8 // max(1, arguments.workers)),
    }
    pilot_specs = [
        RunSpec(
            run_id=f"pilot-f{fraction:.2f}-wd{decay:g}",
            seed=20260720 + index,
            train_fraction=fraction,
            weight_decay=decay,
            stage="pilot",
            **common,
        )
        for index, (fraction, decay) in enumerate(pilot_grid)
    ]
    start = time.monotonic()
    pilot_summaries = launch_specs(pilot_specs, raw_dir, min(arguments.workers, len(pilot_specs)))
    chosen = choose_condition(pilot_summaries)
    print(
        f"chosen train_fraction={chosen['train_fraction']} "
        f"weight_decay={chosen['weight_decay']}",
        flush=True,
    )
    confirmation_specs = [
        RunSpec(
            run_id=f"confirm-seed{20260800 + index}",
            seed=20260800 + index,
            train_fraction=float(chosen["train_fraction"]),
            weight_decay=float(chosen["weight_decay"]),
            stage="confirmation",
            **common,
        )
        for index in range(confirmation_seeds)
    ]
    confirmation_summaries = launch_specs(
        confirmation_specs,
        raw_dir,
        min(arguments.workers, len(confirmation_specs)),
    )
    all_summaries = pilot_summaries + confirmation_summaries
    write_csv(output_dir / "run_summary.csv", all_summaries)
    metadata = {
        "status": "complete",
        "elapsed_seconds": time.monotonic() - start,
        "smoke": arguments.smoke,
        "pilot_count": len(pilot_summaries),
        "confirmation_count": len(confirmation_summaries),
        "chosen_condition": {
            "train_fraction": chosen["train_fraction"],
            "weight_decay": chosen["weight_decay"],
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "aws_instance_id": os.environ.get("AWS_INSTANCE_ID"),
        "aws_region": os.environ.get("AWS_REGION"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"status=complete elapsed_seconds={metadata['elapsed_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
