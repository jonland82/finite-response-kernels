# Local solver-verifier reproduction

This is a fresh-seed, resource-scaled reproduction of the inner-clock experiment in Sun et al. (ICLR 2026). It preserves the experimental structure while replacing the original 3--8B models and 80 GB A800 GPU with a 0.5B open model that can run on this machine.

## Preserved structure

- The same causal language model acts as solver and verifier.
- For each prompt, the solver samples $N$ candidates.
- The verifier estimates a True/False score for every candidate.
- Among candidates above threshold $\sigma$, Best-of-$N$ minimizes length-normalized negative log likelihood.
- The selected responses become fixed pseudo-labels for LoRA supervised fine-tuning.
- Evaluation occurs before training and twice per epoch.
- Every checkpoint logs solver uncertainty $U_s$, verifier uncertainty $U_v$, gap $G=U_s-U_v$, solver accuracy, verifier accuracy, and selection diagnostics.
- Total response NLL matches the paper's stated uncertainty. Per-token NLL and response lengths are also logged because the paper selects Best-of-$N$ using length-normalized NLL, and response-length changes can reverse the total-NLL gap.

The original paper used Phi-3/4 and Llama-3 models, $N=16$, 512 generated tokens, ten epochs, and 80 GB A800 GPUs. The controlled local trajectory uses Qwen2.5-0.5B-Instruct, $N=3$, 48 generated tokens, 16 training prompts, eight held-out prompts, and ten epochs. It is a test of the dynamical formalization, not a numerical replication of the authors' curves.

## Setup

From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The setup creates a local Python 3.10 environment and installs the CUDA 12.4 PyTorch build. Use `-Cpu` only if CUDA initialization fails.

## Run sequence

```powershell
.\.venv\Scripts\python.exe .\run_reproduction.py --config .\config_smoke.json --output .\runs\smoke
.\.venv\Scripts\python.exe .\run_reproduction.py --config .\config_arithmetic_calibration.json --output .\runs\calibration_arithmetic
.\.venv\Scripts\python.exe .\run_reproduction.py --config .\config_arithmetic_trajectory.json --seed 20260718 --output .\runs\arithmetic_seed_20260718
.\.venv\Scripts\python.exe .\run_reproduction.py --config .\config_arithmetic_trajectory.json --seed 20260719 --output .\runs\arithmetic_seed_20260719
.\.venv\Scripts\python.exe .\run_reproduction.py --config .\config_arithmetic_trajectory.json --seed 20260720 --output .\runs\arithmetic_seed_20260720
.\.venv\Scripts\python.exe .\analyze_multiseed.py .\runs\arithmetic_seed_20260718\trajectory.csv .\runs\arithmetic_seed_20260719\trajectory.csv .\runs\arithmetic_seed_20260720\trajectory.csv
```

The primary empirical test fits the first 15 of 21 checkpoints and forecasts the final six. It compares

$$
Y_n=Y_\infty+c\lambda^n
$$

against a signed rank-two alternative, and separately tests whether solver uncertainty, verifier uncertainty, and their gap can share one $\lambda$. The shared-mode comparison is descriptive because the gap is algebraically determined by the other two series.

`config_calibration.json` records the harder GSM8K boundary condition without training. `config_arithmetic_calibration.json` and `config_arithmetic_trajectory.json` use a deterministic local arithmetic benchmark to establish that the small model has enough correct candidates and verifier discrimination for a controlled dynamical experiment.

## Design qualifications

- The pseudo-label set is generated once from the baseline model, then held fixed during SFT. This matches the paper's inner-clock training-trajectory interpretation.
- If no candidate exceeds the self-verification threshold, the highest-scoring candidate is used and the fallback is logged.
- Ground-truth GSM8K answers are used only for accuracy measurement, never for Best-of-$N$ selection or training.
- Evaluation uses fixed prompts and a common random-number seed at every checkpoint to reduce trajectory noise without making candidate selection deterministic.
- The paper does not specify every prompt template, LoRA target module, empty-candidate rule, or random seed. These choices are explicit here.
