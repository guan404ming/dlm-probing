# Probing Functional Correctness in Diffusion Language Models

First probing study of diffusion language model (DLM) hidden states. Linear classifiers on intermediate denoising steps predict whether outputs will be functionally correct.

## Key Findings

1. **Step-0 signal exists** (AUC 0.61-0.80). Prompt encoding alone predicts correctness before any denoising.
2. **Task-dependent emergence.** Structural tasks (JSON) show flat profiles from step 0, reasoning tasks (GSM8K, MBPP, ARC) show gradual buildup.
3. **Divergent layer dynamics.** LLaDA concentrates signal in upper layers (L22-28). Dream migrates from upper to lower layers on simple tasks.
4. **Selective generation.** Per-step probe confidence identifies likely failures, avoiding 36-98% of wasted compute.
5. **Seed reranking fails.** Probe captures instance difficulty, not seed quality.

## Models

| Key | Model | Layers |
|---|---|---|
| `llada` | GSAI-ML/LLaDA-8B-Instruct | 33 |
| `dream` | Dream-org/Dream-v0-Instruct-7B | 29 |

## Datasets

| Key | Source | N | Gen length | Correctness check |
|---|---|---|---|---|
| `jsonschema` | eth-sri/json-mode-eval-extended | 272 | 256 | JSON parse + reference match |
| `gsm8k` | openai/gsm8k (test) | 1,319 | 512 | Numeric answer match |
| `mbpp` | google-research-datasets/mbpp (sanitized test) | 257 | 256 | Code execution + test assertions |
| `arc` | allenai/ai2_arc (ARC-Challenge test) | 1,172 | 256 | Answer letter match |

## Results

### AUC heatmaps (layer x step)

Stars mark the best layer per step. JSON schema shows strong signal from step 0 (flat emergence), while GSM8K shows gradual buildup. Dream's best layer migrates from upper to lower layers on JSON schema.

![Step x Layer AUC heatmap](assets/fig1_heatmap.png)

### AUC vs. denoising step

Best AUC across layers at each step. JSON schema is flat (~0.80 from step 0), while GSM8K, MBPP, and ARC rise gradually.

![AUC vs diffusion step](assets/fig2_auc_curve.png)

## Method

- **Probe:** PCA(64) + StandardScaler + LogisticRegression, 5-fold stratified CV
- **Steps:** 7 checkpoints (0, 1, 4, 16, 32, 64, 127) during 128-step denoising
- **Regions:** Generation region split into 4 equal-length position regions, mean-pooled
- **Metric:** AUC (control probes on shuffled labels yield ~0.50)

## Usage

```bash
# Mid-step probe (8x A100)
.venv/bin/modal run src/modal_midstep_probe.py --dataset jsonschema --model llada --chunks 8

# Selective generation simulation (CPU)
.venv/bin/modal run src/modal_early_exit_sim.py --dataset gsm8k --model dream --chunks 8

# Adaptive compute simulation (CPU)
.venv/bin/modal run src/modal_adaptive_compute_sim.py --dataset gsm8k --model llada --chunks 8
```
