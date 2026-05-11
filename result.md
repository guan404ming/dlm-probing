# EMNLP Tracing Paper — Experimental Results

All numbers come from `emnlp_paper/latex/acl_latex.tex` and cached JSON under the Modal `probe-results` volume. Local mirrors live in `/tmp/paper_figs/` (figure-generation cache) and `/tmp/traj_{model}/` (trajectory-analysis cache). Producers: `src/applications/sae/modal_sae_diagnose.py`, `modal_sae_diagnose_dream.py`, `modal_sae_diagnose_qwen.py`, `modal_sae_steer.py`, `modal_sae_auc_compare.py`.

## 1. LLaDA-8B / MBPP per-step diagnose (single-cell trajectory)

Silhouette on the top-20 fail-enriched features peaks at step 64 (the only step with $p<0.05$) and `f15601` is the only feature persistent at all three peak-or-later steps. Source: Modal volume `probe-results`, paths `mbpp_llada/sae_diagnose_stage2_s{0,1,4,16,32,64,127}.json` (produced by `modal_sae_diagnose.py`); local mirror under `/tmp/traj_llada/mbpp_s{step}.json`.

| step | best K | silhouette | permutation $p$ | top fail feature | enrichment |
|---:|---:|---:|---:|:---|---:|
| 0   | 2 | 0.271 | 0.418 | f6421  | +0.295 |
| 1   | 2 | 0.220 | 0.793 | f5561  | +0.313 |
| 4   | 2 | 0.231 | 0.727 | f5561  | +0.309 |
| 16  | 2 | 0.440 | 0.541 | f3892  | +0.326 |
| 32  | 2 | 0.702 | 0.118 | f15601 | +0.391 |
| 64  | 2 | 0.662 | **0.033** | f15601 | +0.385 |
| 127 | 2 | 0.601 | 0.144 | f11265 | +0.329 |

## 2. Twelve-cell silhouette / $p$ at step 64

Across the 4 dataset × 3 model grid at step 64, only LLaDA-MBPP and Qwen-JSON reach $p<0.05$. Source: Modal volume `probe-results`, paths `{dataset}_{model}/sae_diagnose_stage2_s64.json` for LLaDA/Dream and `{dataset}_qwen/sae_diagnose_qwen.json` for Qwen (produced by `modal_sae_diagnose.py`, `modal_sae_diagnose_dream.py`, `modal_sae_diagnose_qwen.py`).

| dataset    | LLaDA-8B          | Dream-7B    | Qwen-2.5-7B |
|:-----------|:------------------|:------------|:-------------|
| mbpp       | 0.66 / **0.033** | 0.50 / 0.128 | 0.16 / 0.106 |
| jsonschema | 0.58 / 0.195      | 0.77 / 0.135 | 0.22 / **0.007** |
| gsm8k      | 0.38 / 0.432      | 0.29 / 0.449 | 0.15 / 0.315 |
| arc        | 0.43 / 0.312      | 0.18 / 0.882 | 0.13 / 0.343 |

## 3. Full cross-grid signal-to-null gap (silhouette $-$ permutation null mean)

Across the 2 DLM × 4 task grid the peak step is task-determined and replicates across LLaDA-8B and Dream-7B on 3/4 tasks (MBPP/JSON both s64, GSM8K both s127; ARC disagrees). Source: Modal volume `probe-results`, 40 files at `{dataset}_{model}/sae_diagnose_stage2_s{4,16,32,64,127}.json` for both LLaDA-8B and Dream-7B; aggregated by `emnlp_paper/scripts/analyze_trajectory.py`.

| model    | task       | s=4    | s=16   | s=32   | s=64           | s=127        | peak |
|:---------|:-----------|-------:|-------:|-------:|---------------:|-------------:|:----:|
| LLaDA-8B | mbpp       | −0.06 | −0.07 | +0.18 | **+0.31**⋆    | +0.18       | s64  |
| LLaDA-8B | jsonschema | +0.08 | +0.06 | +0.01 | **+0.14**     | +0.12       | s64  |
| LLaDA-8B | gsm8k      | −0.01 | −0.05 | −0.08 | −0.00         | **+0.22**   | s127 |
| LLaDA-8B | arc        | **+0.20** | +0.13 | +0.14 | +0.08      | −0.14       | s4   |
| Dream-7B | mbpp       | +0.15 | +0.03 | +0.02 | **+0.15**     | −0.03       | s64  |
| Dream-7B | jsonschema | +0.04 | +0.06 | +0.16 | **+0.23**     | +0.17       | s64  |
| Dream-7B | gsm8k      | +0.05 | +0.04 | +0.04 | +0.01         | **+0.16**   | s127 |
| Dream-7B | arc        | −0.14 | −0.05 | −0.02 | −0.15         | **+0.26**   | s127 |

## 4. Dream-7B / MBPP per-step diagnose

Peak at step 64 with f6326 as the dominant fail feature; signal collapses below null at step 127. Source: Modal volume `probe-results`, paths `mbpp_dream/sae_diagnose_stage2_s{4,16,32,64,127}.json` (produced by `modal_sae_diagnose_dream.py`).

| step | K | silhouette | $p$   | top fail feature | enrichment |
|---:|---:|---:|---:|:---|---:|
| 4   | 2 | 0.420 | 0.107 | f15414 | +0.272 |
| 16  | 2 | 0.413 | 0.231 | f6326  | +0.342 |
| 32  | 2 | 0.451 | 0.407 | f10891 | +0.408 |
| 64  | 3 | 0.497 | 0.128 | f6326  | +0.421 |
| 127 | 2 | 0.333 | 0.460 | f15414 | +0.270 |

## 5. Dream-7B / JSON schema per-step diagnose

Strongest cross-cell peak (silhouette 0.773, +0.23 signal-to-null gap) at step 64, again with f6326 as the dominant marker. Source: Modal volume `probe-results`, paths `jsonschema_dream/sae_diagnose_stage2_s{4,16,32,64,127}.json` (produced by `modal_sae_diagnose_dream.py`).

| step | K | silhouette | $p$   | top fail feature | enrichment |
|---:|---:|---:|---:|:---|---:|
| 4   | 2 | 0.344 | 0.256 | f6326  | +0.427 |
| 16  | 2 | 0.401 | 0.213 | f10823 | +0.433 |
| 32  | 2 | 0.572 | 0.121 | f6326  | +0.419 |
| 64  | 2 | 0.773 | 0.135 | f6326  | +0.420 |
| 127 | 3 | 0.475 | 0.109 | f233   | +0.349 |

## 6. LLaDA-8B / GSM8K per-step diagnose

Signal is below null until step 127, where silhouette spikes to 0.701 (+0.22 gap); no single feature is the top-1 marker beyond step 16. Source: Modal volume `probe-results`, paths `gsm8k_llada/sae_diagnose_stage2_s{4,16,32,64,127}.json` (produced by `modal_sae_diagnose.py`).

| step | K | silhouette | $p$   | top fail feature | enrichment |
|---:|---:|---:|---:|:---|---:|
| 4   | 3 | 0.190 | 0.394 | f2519  | +0.197 |
| 16  | 3 | 0.175 | 0.629 | f13085 | +0.192 |
| 32  | 2 | 0.255 | 0.661 | —      | —      |
| 64  | 2 | 0.379 | 0.432 | —      | —      |
| 127 | 2 | 0.701 | 0.197 | —      | (+0.224 gap) |

## 7. LLaDA-8B / ARC-Challenge per-step diagnose

Anomalous early peak at step 4 with f4643 (+0.20 gap); the only DLM cell with this shape. Source: Modal volume `probe-results`, paths `arc_llada/sae_diagnose_stage2_s{4,16,32,64,127}.json` (produced by `modal_sae_diagnose.py`).

| step | K | silhouette | $p$   | top fail feature | enrichment |
|---:|---:|---:|---:|:---|---:|
| 4   | 2 | 0.627 | 0.079 | f4643  | +0.186 |
| 16  | 2 | 0.532 | 0.227 | f3892  | +0.243 |
| 32  | 2 | 0.480 | 0.155 | —      | (+0.142 gap) |
| 64  | 2 | 0.427 | 0.312 | —      | (+0.075 gap) |
| 127 | 2 | 0.316 | 0.784 | —      | (−0.142 gap) |

## 8. Single-feature vs. top-N SAE vs. raw-probe AUC

Raw-probe AUC dominates in 11 of 12 cells; top-20 SAE features recover roughly 85--95% of the raw-probe AUC, indicating sparse atoms carry most but not all of the residual-stream signal. Source: Modal volume `probe-results`, paths `{dataset}_{model}/sae_auc_compare.json` across the 12 (dataset, model) cells (produced by `modal_sae_auc_compare.py`).

| model      | dataset    | top-1 | top-5 | top-20 | raw       |
|:-----------|:-----------|------:|------:|-------:|----------:|
| LLaDA-8B   | mbpp       | 0.59  | 0.73  | 0.73   | **0.82** |
| LLaDA-8B   | jsonschema | 0.67  | 0.66  | 0.69   | **0.81** |
| LLaDA-8B   | gsm8k      | 0.69  | 0.71  | 0.72   | **0.77** |
| LLaDA-8B   | arc        | 0.61  | 0.65  | 0.66   | **0.71** |
| Dream-7B   | mbpp       | 0.70  | 0.69  | 0.73   | **0.82** |
| Dream-7B   | jsonschema | 0.68  | 0.74  | 0.78   | **0.80** |
| Dream-7B   | gsm8k      | 0.57  | 0.61  | 0.66   | **0.70** |
| Dream-7B   | arc        | 0.49  | 0.54  | 0.56   | **0.57** |
| Qwen-2.5-7B| mbpp       | 0.57  | **0.66** | 0.59 | 0.61   |
| Qwen-2.5-7B| jsonschema | 0.59  | 0.66  | 0.71   | **0.75** |
| Qwen-2.5-7B| gsm8k      | 0.59  | 0.66  | 0.71   | **0.75** |
| Qwen-2.5-7B| arc        | 0.56  | 0.59  | 0.65   | **0.74** |

## 9. LLaDA-8B / MBPP steering sweep (intervention null result)

Across five conditions on 8 cluster-1 fail cases + 3 pass regression controls, no condition converts any fail→pass or pass→fail. Source: Modal volume `probe-results`, paths `mbpp_llada/sae_steer_stage4_{tag}_a{alpha}_s{step}.json` (e.g. `f15601_a5.0_s64`, `f15601_a5.0_s16`, `f15601_8825_2087_11404_9657_a5.0_s64`, `f15601_a-5.0_s64`; produced by `modal_sae_steer.py`).

| condition                                  | $\alpha$ | step (from) | fail→pass | pass→fail |
|:-------------------------------------------|---------:|------------:|----------:|----------:|
| baseline (no intervention)                 | —        | —           | 0 / 8     | 0 / 3     |
| suppress f15601 @ s64 (peak window)        | +5       | 64          | 0 / 8     | 0 / 3     |
| suppress f15601 @ s16 (pre-peak)           | +5       | 16          | 0 / 8     | 0 / 3     |
| suppress top-5 fail features @ s64         | +5       | 64          | 0 / 8     | 0 / 3     |
| reverse (amplify f15601) @ s64             | −5       | 64          | 0 / 8     | 0 / 3     |

## 10. LLaDA-8B / MBPP top-20 fail-feature Jaccard between consecutive steps

Jaccard is high among early steps (≥0.48 at s0--4 before committed-token information appears), moderate around the peak (0.29 between s32 and s64), and falls to 0.14 between s64 and s127. Source: derived from `mbpp_llada/sae_diagnose_stage2_s{0,1,4,16,32,64,127}.json` on Modal `probe-results`; computation lives in `emnlp_paper/scripts/analyze_trajectory.py` with summary saved to `/tmp/traj_llada/summary.json`.

| step pair  | Jaccard |
|:-----------|--------:|
| s0 → s4    | ≥ 0.48 |
| s32 → s64  | 0.29    |
| s64 → s127 | 0.14    |

## 11. f15601 fire-rate at LLaDA-8B / MBPP step 64

Per-feature fire-rate gap of ~6.6σ at step 64, yet ablating it flips no labels. Source: `top_fail_features` block inside `mbpp_llada/sae_diagnose_stage2_s64.json` on Modal `probe-results` (the `fail_fire_rate`, `pass_fire_rate`, `enrichment` fields for feature 15601).

| population              | fire rate |
|:------------------------|----------:|
| fails ($n=152$)         | 65%       |
| passes                  | 27%       |
| enrichment (fail − pass)| +0.385    |
| approx. effect size     | ~6.6σ     |

## 12. Fail set sizes (from Limitations section)

Cluster sample size is limited by the number of fail cases per cell. Source: `n_fail` field inside each `{dataset}_{model}/sae_diagnose_stage2_s64.json` on Modal `probe-results`; upstream correctness labels are cached at `/results/{dataset}_{model}/labels.json` (produced by the probe paper's `src/core/modal_midstep_probe.py`).

| model    | dataset | fails available |
|:---------|:--------|----------------:|
| LLaDA-8B | mbpp    | 152             |
| Dream-7B | mbpp    | 143             |
| LLaDA-8B | gsm8k   | < 250           |
| Dream-7B | gsm8k   | < 250           |

## 13. Setup constants

Single set of fixed hyperparameters used across every cell. Source: constants declared at the top of `src/applications/sae/modal_sae_diagnose.py`, `modal_sae_diagnose_dream.py`, `modal_sae_diagnose_qwen.py`, and `modal_sae_steer.py`; SAE weights pulled from HuggingFace `AwesomeInterpretability/llada-mask-topk-sae` and the matching Dream-7B / Qwen-2.5-7B DLM-Scope repos.

| item                         | value |
|:-----------------------------|:------|
| DLMs analysed                | LLaDA-8B-Instruct (33 layers), Dream-7B-Instruct (29 layers) |
| AR baseline                  | Qwen-2.5-7B |
| Denoising steps $T$          | 128 |
| Checkpoint steps             | {0, 1, 4, 16, 32, 64, 127} |
| Datasets (count)             | MBPP-sanitized (257), JSON schema (272), GSM8K (1,319), ARC-Challenge (1,172) |
| SAE source                   | DLM-Scope Mask-SAE, trainer index 2 ($K=160$) |
| SAE layer                    | LLaDA L26, Dream L23, Qwen L23 |
| SAE width $d_{sae}$          | 16,384 |
| Cluster $K$ sweep            | {2, 3, 4, 5}, best per cell |
| Top-N fail features          | top-20 |
| Permutation shuffles         | 1,000 (with feature re-selection per shuffle) |
| Generation seed              | 0 |
| Steering hook layer          | L26 (LLaDA), L23 (Dream) |
| Steering $\alpha$            | $\{+5, -5\}$ |
| Steering windows tested      | $t_0 \in \{0, 16, 32, 64\}$ |
| Compute                      | NVIDIA A100 80GB on Modal, bfloat16, ~40 A100-hours total |
