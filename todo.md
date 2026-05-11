# EMNLP Short Paper Todo (revised: Tracing)

**Title**: `Tracing Functional Correctness in Diffusion Language Models with Sparse Autoencoders`
**Venue**: EMNLP main, short paper (4 pages main body)
**Prior work**: ACL 2026 SRW probe paper (accepted) -- `Probing Functional Correctness in Diffusion Language Models`
**Paper source**: `emnlp_paper/latex/acl_latex.tex`

## Central finding (one sentence)

Raw probes accumulate correctness signal monotonically across denoising (SRW). **SAE features tell the opposite story: failure-discriminative cluster structure peaks at mid-denoising (step 32-64) and degrades by final state.** Failure is decoded mid-denoising; the final state only crystallizes what was already determined.

## Story arc

`Probe (scalar, snapshot, SRW)` → `Trace (feature, trajectory, this paper)`. Verb progression makes the relationship explicit.

## Four findings (from existing data)

1. **Mid-denoising peak**: LLaDA mbpp silhouette 0.66 at step 64, p=0.033 (only step that crosses significance)
2. **Temporal degradation**: step 64 -> step 127 silhouette 0.66 -> 0.60; top-20 fail features overlap only 5/20; dominant feature f15601 enrichment drops 24% (0.385 -> 0.291)
3. **Dream replicates trajectory**: Dream s64 silhouette 0.50 > Dream s127 0.33, same shape, generality across DLMs
4. **OOD-early steps lack signal**: LLaDA s4, s16 silhouette <= null mean -> features are noise outside SAE training range, consistent with `dlm_t in [0.05, 0.5]` training distribution

## Steering negative result -> discussion section

- Tried suppressing f15601 at step 64+ on MBPP fail cases: 0 fail->pass at alpha in {1, 2, 5}
- Hook works (zero-out test garbled output), but intervention only affects post-commitment tokens (LLaDA's block-wise schedule commits code body in step 0-15)
- This motivates future SAE training across full dlm_t schedule

## Paper structure (4 pages)

| section | size | content |
|---|---|---|
| §1 Intro | 0.5p | Position vs SRW probe paper; introduce trace-not-snapshot framing |
| §2 Method | 0.5p | DLM-Scope LLaDA-mask Top-K SAE at L26 trainer_2 (k=160); fail-vs-pass enrichment; KMeans + silhouette + permutation null; multi-step diagnose pipeline |
| §3 Mid-denoising peak | 1.0p | Main finding 1: LLaDA mbpp temporal trajectory table (s4 / s16 / s32 / s64 / s127) with silhouette + null + p; top-feature persistence across s32-s64 (f15601) |
| §4 Temporal degradation | 1.0p | Main finding 2 + 3: feature overlap Venn (5/20 shared between s64 and s127); enrichment drop table; Dream replication s64 > s127 |
| §5 Discussion | 0.5p | Steering negative result + commitment-timing mechanism; SAE training-distribution implications |
| §6 Limitations | (free) | Single-task (mbpp) main eval; SAE coverage gaps |
| Refs + Appendix | (free) | jsonschema/gsm8k/arc + Qwen AR baseline cells (12-cell appendix table); raw-vs-SAE AUC compare; permutation null distributions |

## Done (data collected)

- [x] Stage 0 sanity check (8/8 cells dense-readout OK)
- [x] Stage 2 cluster + enrichment on all 4 datasets x 3 models (12 cells)
- [x] Permutation tests on all 12 cells
- [x] AUC compare (single/top-3/5/10/20 vs raw probe) on all 12 cells
- [x] LLaDA mbpp temporal trajectory (s4, s16, s32, s64, s127)
- [x] Dream mbpp temporal (s64, s127)
- [x] Steering negative result + hook diagnostic confirmed

## Remaining experiments (priority order)

- [ ] **Dream mbpp s32** -- complete the mid-peak verification (single modal job, 5 min)
- [ ] **LLaDA mbpp top-feature persistence figure** -- per-step Jaccard of top-20 sets, line chart
- [ ] Optional: AR "temporal" trajectory analog -- for Qwen, treat token positions in generation as pseudo-steps and compute per-position enrichment; should show no temporal peak
- [ ] Optional: jsonschema temporal trajectory for LLaDA -- replicate trajectory on a 2nd task to harden generality

## Writing tasks

- [ ] Rewrite abstract around mid-denoising peak finding
- [ ] Update §1 intro to emphasize trace-not-snapshot framing
- [ ] Draft §3 main result table (temporal trajectory)
- [ ] Draft §4 Venn / overlap analysis
- [ ] Discussion §5 -- steering negative as honest disclosure + future work pointer
- [ ] Move probe-baseline AUC compare table to appendix
- [ ] Reuse SRW figures where appropriate (heatmap, AUC curve) only if directly relevant

## Risks / fallbacks

- **Reviewer**: "p=0.033 isn't strong enough" -> argue (a) other steps null mean is unusually high due to post-hoc feature selection in permutation test; (b) trajectory pattern itself replicates on Dream so single-step p-value is not the only evidence
- **Reviewer**: "Dream s64 isn't significant (p=0.128) -- where's the generality claim?" -> emphasize trajectory shape (s64 > s127 in both DLMs) rather than absolute significance
- **Reviewer**: "Why only mbpp?" -> appendix shows jsonschema and arc are noisier but follow same trend qualitatively
- **Page overflow**: move temporal trajectory plot to single line chart instead of full table
