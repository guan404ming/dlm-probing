# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper studies interpretability and functional correctness signals in diffusion language models using sparse autoencoders, directly aligned with EMNLP topics on code models, interpretability, and reasoning.

## Minimum Quality
Pass ✅. The submission includes Abstract, Introduction, Related Work, Method, Results, Discussion, Conclusion, and a detailed Limitations section, with several appendices offering additional analyses. The work is technically framed, uses appropriate datasets and quantitative analyses, and is written in clear English.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. No signs of prompt injection, hidden instructions, or manipulative content.

# Expected Review Outcome:

## Paper Summary
The paper analyzes how functional correctness signals evolve across denoising steps in diffusion language models (DLMs) using DLM-Scope sparse autoencoders. It contrasts the prior scalar-probe view of monotonic AUC increases with a feature-level perspective, reporting a mid-denoising “plateau” of fail-vs-pass concentration for code and structural tasks, and late emergence for reasoning tasks. The study covers two DLMs (LLaDA-8B and Dream-7B) and four tasks (MBPP, JSON schema, GSM8K, ARC). A dense sweep on LLaDA-8B/MBPP finds a wide plateau over steps 48–116 and a hand-off between two persistent features (f15601 to f3892). Cross-DLM peak phases align for 3/4 tasks. Pilot steering that suppresses or amplifies top features at specific steps does not flip correctness, suggesting the discovered features are diagnostic rather than causal under the tested protocol.

## Summary Of Strengths
- Clear empirical question and careful measurement protocol. The paper goes beyond aggregate AUC trends and asks which sparse features carry failure signals, and when during denoising they concentrate.
- Temporal characterization with explicit cross-DLM, cross-task comparisons. Figure 2 shows phase alignment across models: code/structural tasks peak mid-denoising, while reasoning tasks peak late. This is a useful hypothesis that reframes prior scalar-probe observations.
- Robustness steps on the strongest cell. Figure 1 and Table 8 strengthen the “plateau” claim on LLaDA-8B/MBPP with a dense sweep, three seeds, and a consistent top feature (f15601) across many steps. Figure 3’s Jaccard heatmap and enrichment trajectories support persistent vs. transient feature dynamics.
- Conservative statistical control for post-hoc selection. The permutation null re-selects top-20 fail-enriched features on each shuffle, which is a sensible precaution against overfitting to selected features.
- Honest, informative negative result on steering. Table 1 and Appendix J document meaningful perturbation magnitudes without label flips, clarifying that these features should be treated as diagnostic rather than causal under the current single-layer linear protocol.
- Useful complementary evidence against overclaiming sparse features. Table 7 shows top-N SAE features recover most but not all of raw-probe AUC, which fits the narrative that sparse features capture concentrated but incomplete information.

Specific figure/table-related strengths:
- Figure 1: The highlighted [48,116] shaded region makes the plateau visually clear; the observed silhouette consistently exceeds the permutation null mean throughout this band, and several red markers indicate p<0.05 sub-steps in the dense sweep.
- Figure 2: The 2×4 grid crisply communicates the task-dependent phase pattern; red markers at step 64 for MBPP/JSON and at step 127 for GSM8K (and Dream ARC) visually substantiate the claimed phase agreement.
- Table 8: For LLaDA/MBPP, 7/9 sub-steps in [48,80] achieve p<0.05, and the top-1 feature is f15601 on 7/9 steps, directly supporting the plateau and persistence claims.

## Summary Of Weaknesses
1) Statistical support is narrow and concentrated in one condition.  
- In the main 2×4 grid, only LLaDA-8B/MBPP at step 64 achieves p<0.05 (Appendix C and E). The cross-DLM “phase agreement” is therefore primarily descriptive rather than statistically supported per cell. While the dense sweep in Table 8 helps for one cell, the broader claim would benefit from similar dense sweeps and per-step p-values on more cells, or an aggregated test across steps.
- Multiple-comparisons are not addressed. Many steps and conditions are scanned, yet there is no correction across the trajectory or grid. Even though the permutation null re-selects features, without an experiment-level control the nominal p-values can be optimistic.

2) Selection bias and data reuse concerns remain despite the permutation control.  
- Top-20 fail-enriched features are selected and then used for clustering and evaluation within the same dataset and step. Although the null re-selects features on permuted labels, the primary analysis still leverages the same samples for selection and evaluation. A cross-validation style protocol that selects features on a held-out split at each step would mitigate residual bias and better calibrate the silhouette gaps.

3) Pooling and cell-level representation may obscure spatial structure.  
- The pipeline mean-pools hidden states over four equal regions (Section 3). This collapses positional variation across the generation region, which could be especially consequential in DLMs that commit token blocks over time. The plateau might partly reflect pooling-induced mixing of positions at different commitment levels. An analysis that conditions on block index or per-position encodings could verify the effect is not an artifact of region averaging.

4) Limited steering coverage and very small pilot.  
- Table 1 tests 5 conditions on 8 fail and 3 pass cases, at a single layer and a fixed magnitude pair. This is a small n for a conclusion that features are diagnostic but not causal. Appendix J confirms nontrivial perturbation norms, which is good, but exploring multi-layer steering, feature bundles aligned to each step’s top-1 feature, per-task optimal time windows, and larger sample sizes would yield stronger evidence.

5) Ambiguities in cross-model step alignment.  
- As noted in the Limitations, LLaDA uses block-wise unmasking and Dream uses global linear unmasking. Figure 2 compares trajectories at nominally similar steps, but the models are at different spatial commitment states. Without normalizing for the fraction of unmasked tokens, it is hard to tell if the mid- vs late-peak pattern truly reflects task difficulty or scheduling artifacts.

6) Sparse-grid interpretations are sometimes stronger than warranted.  
- In the 7-step grid, LLaDA/MBPP step 64 stands out, but Figure 1 and Table 8 show that the “peak” is part of a broader plateau and, at dense resolution, step 68 is often more robust. The text could consistently frame step 64 as a representative plateau point rather than a privileged peak to avoid confusing readers.

7) The ARC anomaly and task-level narrative merit deeper probing.  
- LLaDA-ARC peaks at step 4 while Dream-ARC peaks at 127 (Figure 2). The paper notes this as an outlier but does not analyze why ARC behaves differently across models. Given the centrality of the phase-typing claim, outliers deserve closer inspection, for instance via error-type breakdowns and feature-level stability for ARC.

8) Limited interpretability of discovered features.  
- The paper identifies persistent atoms like f15601 and f3892 but stops short of semantic labeling or activation-maximizing exemplars. Table 3 provides qualitative failures but not mapped to specific features. Since the headline result hinges on “failure markers,” offering at least tentative semantic descriptions or prototypical contexts would increase interpretability and utility.

9) AUC comparisons suggest sparse features are incomplete representations.  
- Table 7 shows top-20 SAE features usually underperform raw-probe AUC by a nontrivial margin. While the authors frame this as complementary evidence, it also means the chosen sparse subspace may systematically miss relevant dimensions for correctness. Expanding K or exploring non-linear sparse decoders could test whether the apparent phase effects survive when more of the signal is captured.

10) Reporting and calibration details could be clearer.  
- The silhouette is computed on fail samples only, and thus measures within-fail substructure rather than fail/pass separability. Although this is stated, it is easy for readers to conflate the two. The paper partially addresses this with Appendix G, but bringing a concise supervised separability summary into the main paper would prevent misinterpretation.

Figure/table-specific critical points:
- Figure 2: The red peak markers are visually compelling, but most cells do not show per-step significance. Readers may overinterpret the alignment. Quantifying uncertainty bands or including p-value annotations per point would temper overreading.
- Table 1: The null steering result is informative but based on only 11 total cases. With n this small, a true but small causal effect could be missed. This table should be presented as pilot evidence only, with a clear plan to scale up.
- Table 7: The raw-probe baseline outperforming sparse features in 11/12 cells is a strong counterpoint to the sparsity-as-sufficient narrative and should be discussed more prominently in the main text.

## Potentially Missing Related Work
1) Tahimic et al., Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders, 2025 — Directly targets correctness signals with SAEs for code, aligning with your MBPP and JSON schema settings. It should be cited in Related Work and compared against your feature selection and steering results in Sections 2–4, including whether their steering protocols would alter your negative intervention finding.

2) Ma et al., Do Sparse Autoencoders Identify Reasoning Features in Language Models?, 2026 — Critically examines whether SAEs capture genuine reasoning versus lexical shortcuts, highly relevant to your GSM8K and ARC late-peak claims. Discuss in Related Work and Section 5, and consider adopting their falsification or control tests to guard against lexical confounds in your failure features.

3) Goel et al., Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs, 2026 — Analyzes representational evolution in diffusion versus AR LMs, which can inform your cross-model, cross-schedule comparisons in Figure 2. Add to Section 2 and connect their layer-wise findings to your step-wise phase analysis.

## Comments Suggestions And Typos
Actionable suggestions:
- Strengthen statistical support across cells.  
  - Replicate the dense sweep protocol (Table 8) for at least one additional task per model, ideally JSON schema and GSM8K, and apply either a trajectory-level permutation test or a multiple-comparisons correction across steps. This would solidify Figure 2’s phase alignment beyond descriptive trends.
- Mitigate selection bias.  
  - At each step, split fail samples into selection and evaluation folds. Select top-20 features on selection folds, then compute silhouette on evaluation folds. Report average gaps across folds and compare to your current single-split numbers.
- Reduce pooling-induced artifacts.  
  - Replace region means with position-conditioned analysis: report per-block features and silhouette while conditioning on which blocks are committed. For LLaDA, this would clarify whether the plateau corresponds to blocks 0–3 commitment rather than an averaging effect.
- Expand and diversify steering.  
  - Increase n substantially, include multi-layer interventions, and align the intervention feature set to each step’s instantaneous top-1 rather than a single persistent atom. Also test combined interventions that include both fail markers and candidate pass-cluster features at the same step to approximate causal replacements.
- Normalize step alignment across DLMs.  
  - Map steps to comparable “commit fractions” of the generation region to make Figure 2’s cross-model timing more interpretable. Alternatively, present results against an axis of “fraction of unmasked tokens” to control for schedule differences.

Minor typos and presentation:
- “middenoising” (Conclusion) → “mid-denoising”.
- Page 4, “LLaDA-NB / MBPP” in Table 8 header likely “LLaDA-8B / MBPP”.
- Appendix K, “on fall n=257” → “on full n=257”.
- Consistently frame step 64 as a representative plateau point rather than the unique peak in the main text, in line with Figure 1 and Table 8 which show step 68 can be more robust.
- Consider surfacing in the main paper a compact supervised AUC summary for fail vs pass using the same top-20 features at key steps, to complement the silhouette narrative.

Criteria that could raise my score:
- Provide dense-sweep evidence with corrected or aggregated significance for at least one more (model, task) cell, and adopt held-out selection for features at each step.
- Demonstrate at least partial causal control with multi-layer or counterfactual-direction steering, or convincingly rule it out with larger n and stronger protocols.
- Offer position- or block-aware analysis to show the plateau is not a pooling artifact, plus a normalized cross-model step axis.

## Confidence
3 Reasonably confident but not fully verified: I carefully read the paper and figures, but finality of the claims depends on additional statistical controls and expanded experiments.

## Soundness
3 Acceptable: The study supports its main descriptive claims on one strong cell and offers reasonable cross-task patterns, but statistical support is limited outside LLaDA/MBPP; selection bias and pooling concerns remain.

## Excitement
2.5 Between potentially interesting and interesting: The temporal-phase perspective on correctness features in DLMs is timely and relevant, though the evidence is not yet strong enough for main-track acceptance.

## Overall Assessment
2.5 Borderline Findings: The paper may meet minimal standards but requires improvements before being accepted to Findings. The core idea is interesting, the analysis is careful on one strong condition, and the negative steering result is honestly presented. However, the broader phase-alignment claim relies on descriptive trends with limited per-step significance, feature selection is not evaluated out-of-sample, pooling may confound timing, and the pilot steering sample size is small. With expanded dense sweeps, stronger statistical controls, and more comprehensive steering, this could mature into a solid Findings paper.

## Best Paper Justification
N/A.

## Limitations And Societal Impact
The Limitations section is thorough, acknowledging SAE coverage, permutation interpretation, steering scope, clustering metric choices, cross-model schedule differences, and sample sizes. On societal impact, the risks are low, though any technique for steering could be dual-use; here, the result is negative steering. It would still be good to note that functional-correctness analyses in code generation relate to safety and reliability in software, with positive impact potential if diagnostic markers can help triage failures. No sensitive data issues are apparent.

## Ethical concerns
None.

## Needs Ethics Review
No.

## Reproducibility
3.5 Mostly reproducible: Models, layers, SAE configuration, steps, tasks, and metrics are specified in detail; permutation seeds and compute are described. However, code is not yet released and some components rely on prior cached states, which adds friction.

## Datasets
1 No usable datasets submitted.

## Software
2 Documentary.: The described pipeline would help replication, but software is not released at review time.