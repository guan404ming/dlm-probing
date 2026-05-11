# EMNLP paper scripts

Helpers for generating figures and trajectory analyses for the EMNLP paper
*Tracing Functional Correctness in Diffusion Language Models with Sparse Autoencoders*.

Both scripts read cached JSON results from the Modal volume `probe-results`
(or `/tmp/...` mirrors).

## Scripts

### `gen_paper_figures.py`
Produces the four PDF figures in `../figures/`:

- `fig1_trajectory.pdf` — LLaDA-8B / MBPP silhouette and null mean per step
- `fig2_cross.pdf` — 3-panel cross-condition trajectory (LLaDA mbpp, Dream mbpp, LLaDA jsonschema)
- `fig3_steering.pdf` — 5-condition steering null-effect bar chart
- `fig4_feature_drift.pdf` — Jaccard heatmap + per-feature enrichment trajectory (appendix)

Run from the repo root:

```
.venv/bin/python emnlp_paper/scripts/gen_paper_figures.py
```

### `analyze_trajectory.py`
Trajectory aggregation across cached diagnose JSONs for a single (model, dataset):
silhouette + null + permutation $p$ per step, top-N Jaccard between steps,
persistent fail features (top-20 at $\ge 3$ steps).

```
.venv/bin/python emnlp_paper/scripts/analyze_trajectory.py --model llada --dataset mbpp
.venv/bin/python emnlp_paper/scripts/analyze_trajectory.py --model dream --dataset mbpp
```

## Source data dependencies

These scripts assume the following Modal volume files exist:

- `mbpp_llada/sae_diagnose_stage2_s{0,1,4,16,32,64,127}.json`
- `mbpp_dream/sae_diagnose_stage2_s{32,64,127}.json`
- `jsonschema_llada/sae_diagnose_stage2_s{4,16,32,64,127}.json`
- `mbpp_llada/sae_steer_stage4_*.json` (5 conditions for fig 3)

To regenerate them, see the corresponding Modal scripts in
`../../src/applications/sae/`:

- `modal_sae_diagnose.py` (LLaDA)
- `modal_sae_diagnose_dream.py` (Dream)
- `modal_sae_diagnose_qwen.py` (Qwen AR baseline, appendix)
- `modal_sae_steer.py` (steering window / multi-feature / reverse)
