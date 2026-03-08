# dllm-probing

Probing functional correctness in diffusion language models.

## Project Overview
- Probe diffusion LLM hidden states at intermediate denoising steps to predict functional correctness
- 2 models: LLaDA-8B-Instruct, Dream-v0-Instruct-7B
- 2 datasets: jsonschema (272 instances), gsm8k (1319 instances)
- All experiments run on Modal (8x A100)

## Scripts
- `modal_midstep_probe.py` - Main: generation + hidden state extraction + probe training (GPU)
- `modal_early_exit_sim.py` - Early exit simulation (CPU)
- `modal_adaptive_compute_sim.py` - Adaptive compute simulation (CPU)
- `modal_seed_rerank.py` - Seed reranking experiment (GPU, negative result)

## Data (Modal volume `probe-results`)
Organized as `{dataset}_{model}/`:
- `chunk_off*.npz` - Per-chunk hidden state features + labels
- `midstep_probe_results.json` - Probe AUC across steps/layers
- `early_exit_results.json` - Early exit simulation results
- `adaptive_compute_results.json` - Adaptive compute simulation results
- Seed rerank data in `jsonschema_{llada,dream}/seed_rerank/`

## Experiment Results

### Baseline functional rates (seed=0, 128 steps)
| | jsonschema | gsm8k |
|---|---|---|
| LLaDA | 48.5% (132/272) | 66.3% (875/1319) |
| Dream | 46.0% (125/272) | 61.5% (811/1319) |

### Mid-step probe AUC (best layer, final step)
| | jsonschema | gsm8k |
|---|---|---|
| LLaDA | 0.809 (layer 24) | 0.786 (layer 24) |
| Dream | 0.828 (layer 5) | 0.818 (layer 19) |

### Probing details (jsonschema only, detailed analysis)
- Probe: PCA(64) + StandardScaler + LogisticRegression, 5-fold stratified CV
- 4 position regions (gen_length / 4 each, mean-pooled), 7 checkpoint steps (0,1,4,16,32,64,127)
- Step 0 already has signal (AUC ~0.78), meaning prompt encoding predicts correctness
- Signal peaks around step 64 (AUC ~0.84-0.85) for both models
- LLaDA best layers stay in upper layers (L22-26), Dream migrates from upper to lower layers during denoising
- LLaDA concentrates signal at JSON opening (region 0), Dream distributes more uniformly

### Early exit (threshold=0.80)
| | jsonschema | gsm8k |
|---|---|---|
| LLaDA | 98.4% saved, 73.5% acc | 61.3% saved, 73.5% acc |
| Dream | 96.7% saved, 72.4% acc | 36.0% saved, 73.5% acc |

### Adaptive compute (conf=0.75, easy=32 steps)
| | jsonschema | gsm8k |
|---|---|---|
| LLaDA | 30.1% saved, 72.5% easy precision | 33.5% saved, 80.0% easy precision |
| Dream | 23.2% saved, 65.5% easy precision | 22.3% saved, 76.3% easy precision |

Step-0 probe classifies instances as easy (P(func) >= threshold) or hard.
Easy instances get fewer denoising steps.

### Seed reranking (negative result, jsonschema only)
| | Baseline | Rerank | Probe train acc |
|---|---|---|---|
| LLaDA | 48.5% (132/272) | 48.5% (+0.0%) | 85.7% |
| Dream | 46.0% (125/272) | 46.0% (+0.0%) | 84.6% |

Probe learns instance difficulty, not seed quality.

## Key Findings
1. Both models encode correctness signal from step 0 (AUC ~0.78). Prompt encoding alone predicts output quality.
2. Signal peaks around step 64 (AUC ~0.84-0.85), not at the final step.
3. Best layer dynamics differ across models (LLaDA: upper layers, Dream: migrates to lower layers).
4. Seed reranking fails because the probe captures instance-level difficulty, not seed-dependent variation.
5. Adaptive compute can save 22-34% steps with 65-80% precision on "easy" classification.

## Tools & Environment
- Python 3.12, datasets==2.21.0, transformers==4.52.2
- Modal for cloud GPU (A100), run via `.venv/bin/modal`
- Use `--chunks 8` for 8x A100 parallel
- Use `uv` for Python package management (not pip)
