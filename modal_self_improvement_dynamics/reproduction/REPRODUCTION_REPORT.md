# Fresh-seed controlled reproduction

## Question

Can we recreate the solver--verifier training trajectory without the authors' exact seeds, then use the takeoff-kernel formalism to test a stronger claim than an in-sample exponential fit?

Yes, at structural rather than numerical-replication scale. The experiment below follows the inner-clock design of [Sun et al.](https://arxiv.org/abs/2507.00075): the same-model solver/verifier loop, thresholded Best-of-$N$ selection, fixed pseudo-label SFT, LoRA training, and solver/verifier uncertainty measurements. It uses new declared seeds and a task small enough for the local GPU.

## Protocol

The same [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model is used as solver and self-verifier. For each question it samples $N=3$ answers at temperature $1$. The verifier scores each answer by its conditional probability of `TRUE` rather than `FALSE` at temperature $0.1$. Candidates with score at least $0.5$ qualify; the qualified candidate with the smallest length-normalized negative log likelihood becomes the pseudo-label. If none qualifies, the highest-scoring candidate is selected and the fallback is logged.

Sixteen pseudo-labels are generated once at the baseline checkpoint. A rank-16 LoRA adapter is then trained for ten epochs with AdamW, learning rate $10^{-5}$, and weight decay $0.01$. Eight fixed held-out prompts are evaluated before training and twice per epoch, giving 21 checkpoints. Evaluation uses common random numbers across checkpoints. Ground truth is used only for diagnostics, never for selection or training.

The local task contains addition, subtraction, multiplication, and two-step arithmetic word problems. A preceding zero-shot calibration on GSM8K produced no correct candidates in 32 samples with the 0.5B model, so GSM8K would not have supplied a meaningful solver--verifier learning signal on this hardware.

The main deviations from Sun et al. are therefore explicit: 0.5B rather than 3--8B parameters, synthetic arithmetic rather than GSM8K/MATH/ProntoQA/MBPP, $N=3$ rather than $16$, 48 rather than 512 generated tokens, 16 rather than hundreds or thousands of training examples, eight held-out examples, and a 4 GB GTX 1650 rather than an 80 GB A800. This is a reproduction of the experimental structure and a test of the trajectory law, not a numerical replication of the published curves.

## Kernel test

For each observed trajectory, the rank-one null is

$$
Y_n=Y_\infty+c\lambda^n,
$$

and the minimal signed two-mode alternative is

$$
Y_n=Y_\infty+c_1\lambda_1^n+c_2\lambda_2^n.
$$

Both models estimate their endpoints on the first 15 checkpoints and forecast the last six. We also compare them with a last-value persistence forecast. This matters: rank two beating rank one does not establish useful modal dynamics if both lose to simply holding the final training value constant.

The coupled law makes the stronger prediction

$$
\begin{aligned}
U_s(n)&=U_{s,\infty}+c_s\lambda^n,\\
U_v(n)&=U_{v,\infty}+c_v\lambda^n,\\
G(n)&=G_\infty+c_g\lambda^n,
\end{aligned}
$$

with one shared $\lambda$. We compare this shared rank-one forecast with three separately fitted rank-one modes after scaling each series by its training range. This comparison is descriptive because $G=U_s-U_v$ is algebraically dependent on the other two series.

## Results

| Seed | Pseudo-label accuracy | Solver accuracy | Verifier accuracy | Per-token gap | Rank two wins | Best modal beats persistence |
|---:|---:|---:|---:|---:|---:|---:|
| 20260718 | 0.625 | $0.50\to0.875$ | $0.875\to0.75$ | $0.241\to-0.015$ | 4/6 | 1/6 |
| 20260719 | 0.625 | $0.50\to0.50$ | $0.625\to0.625$ | $0.054\to0.041$ | 4/6 | 2/6 |
| 20260720 | 0.625 | $0.75\to0.75$ | $0.875\to0.875$ | $0.058\to0.029$ | 1/6 | 2/6 |

The per-token gap narrowed in all three seeds, and in the first seed it crossed slightly below zero. This reproduces the qualitative gap-closure phenomenon. Accuracy improvement did not replicate: solver accuracy improved only in the first seed, while verifier accuracy was flat or lower.

Rank two had lower held-out MSE than rank one in 9 of 18 series-by-seed comparisons, exactly half. The advantage was not stable by observable: rank two won 2/3 solver-total fits, 0/3 verifier-total fits, 2/3 total-gap fits, 1/3 solver-per-token fits, 2/3 verifier-per-token fits, and 2/3 per-token-gap fits. More importantly, the better of rank one and rank two beat persistence in only 5 of 18 comparisons. The per-token gap was the only family for which a modal forecast beat persistence in two seeds, but the preferred order differed. The current data therefore do not verify a reproducible second mode.

A five-boundary rolling audit makes this conclusion stronger. Across 90 trajectory--split cases, persistence wins 52, power law 13, local linear 10, rank one eight, and rank two seven. Persistence is the consensus winner for 15 of 18 trajectories, and only two trajectories keep the same winner at every split. The result is therefore not an artifact of the original 15/6 boundary.

The endpoint-free spectral audit gives a superficially high-order but substantively negative result. At the primary 1% calibration, 74 of 90 rolling cases are labeled rank $4+$ and 11 rank three; on the full trajectories, 16 of 18 are labeled rank $4+$. Yet none of the 18 full trajectories yields perturbation-stable matrix-pencil poles. Combined with persistence dominance and the visible plateaus and jumps, the high numerical ranks indicate failure of a stable finite-exponential law rather than evidence for many physical modes. Full calibration and artifacts are in [../SPECTRAL_MODE_AUDIT.md](../SPECTRAL_MODE_AUDIT.md).

The shared-rate null is likewise not consistently rejected. On total NLL, shared-rank-one/separate-rank-one held-out MSE factors were $0.94$, $0.96$, and $0.83$, so sharing a mode helped in every seed. On per-token NLL the factors were $1.00$, $0.73$, and $1.44$, a mixed result. The fitted shared per-token modes were $0.70$, $0.60$, and $0.995$ per epoch, too unstable to interpret as one replicated system constant.

Total response NLL and NLL per token behave differently. Best-of-$N$ selection minimizes per-token NLL, but total NLL also grows with response length. Correct selected answers were often longer than sampled solver answers, so the total-NLL gap could have the opposite sign from the per-token gap. Any claimed solver--verifier advantage should therefore report response lengths and both quantities.

## What this establishes

We successfully recreated a fresh-seed, locally runnable version of the experiment. It verifies that fixed pseudo-label SFT can narrow the solver--verifier per-token uncertainty gap, but it does not verify that the trajectory follows one stable exponential, nor that a second mode replicates. The kernel approach improves the result type by replacing visual or in-sample curve agreement with rolling held-out forecasting, the shared-mode restriction, and comparison with persistence and power-law alternatives. Here those checks turn a tempting rank-two story into a mixed negative result.

The result does not identify physical modes or contradict the larger-model experiments. The held-out set is small, the generated outputs change discretely as LoRA weights cross decoding boundaries, and each seed also draws a new synthetic task sample. A more decisive reproduction would keep a larger evaluation panel fixed, use at least a 3B model on a published task, and add seed-level uncertainty to the joint shared-mode test.

## Reproduce

The environment and commands are in [README.md](README.md). Each run records its configuration and hardware in `manifest.json`, pseudo-labels in `pseudolabels.jsonl`, checkpoint-level metrics in `trajectory.csv`, and prompt-level samples in `checkpoint_details/`. `analyze_reproduction.py` performs the individual audit; `analyze_multiseed.py` aggregates the fresh seeds and produces the combined figure and JSON audit.
