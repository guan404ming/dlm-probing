# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper squarely targets interpretability and probing of diffusion language models for NLP tasks such as MBPP, JSON schema, GSM8K, and ARC. This aligns with EMNLP topics including Interpretability, Code Models, Natural Language Generation, and Reasoning.

## Minimum Quality
Pass ✅. The paper includes Abstract, Introduction, Related Work, Method (with models, data, per-step SAE encoding, enrichment/clustering/steering procedures), Results, Discussion, Conclusion, and Limitations. Experiments span multiple models and tasks with quantitative results, figures, and tables. While I have concerns about some methodological choices and the strength of evidence for several claims, the paper meets minimum standards for scholarly completeness and clarity.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. I found no attempts to manipulate automated reviewing. The content is standard scholarly prose with explicit sections and references; figures and tables are conventional.

# Expected Review Outcome:

## Paper Summary
The paper investigates what “correctness probes” are reading in diffusion language models (DLMs). Rather than treating correctness predictability as a single scalar that grows monotonically through denoising (as in prior work), the authors use sparse autoencoders (SAEs) to trace feature-level signals over steps and layers. The central claim is that the prominent signal largely encodes prompt difficulty rather than generation-specific correctness. Evidence includes: cross-generator transport where a probe trained on Dream-7B’s hidden states predicts Qwen-2.5-7B-Instruct correctness with only small AUC loss; a three-way per-layer scan showing similar peak AUC across Qwen-Base, Qwen-Instruct, and Dream-Base but at different layers; temporal “plateau” structure where fail-enriched sparse features are concentrated mid-denoising for code/JSON tasks and late for reasoning; and a linear activation-steering null where suppressing or amplifying these fail features does not flip outcomes. The authors interpret the features as diagnostic correlates of prompt difficulty, not controllable causes of correctness.

## Summary Of Strengths
- Clear, falsifiable reinterpretation of correctness probes. The cross-generator transport result directly tests the “difficulty vs correctness” confound, advancing beyond typical probing correlations. Table 3 and Figure 6 show that different models reach similar peak AUC at different layers, while a Dream L26 probe transports to Qwen with only small loss, supporting a shared prompt-difficulty substrate.
- Careful temporal analysis across steps. Figure 1 and Figure 3 convincingly reveal a mid-denoising plateau for LLaDA on MBPP, with feature persistence and a within-plateau hand-off (f15601→f3892). Figure 2 extends this to a cross-model × cross-task grid, where code/JSON peak mid-denoising and reasoning peaks at the final state. Figure 4 aggregates significance, appropriately acknowledging sparse per-step significance.
- Strong negative result on linear steering. Table 1 and Appendix J systematically vary targets, magnitudes, steps, and even raw-residual directions, yet achieve 0/345 fail→pass flips in plateau windows, which meaningfully informs the community about limits of linear control for correctness in DLMs.
- Thoughtful controls and diagnostics. The permutation null that re-selects features controls for post-hoc selection. Held-out feature selection (Appendix K) strengthens the plateau claim. The gradient attribution (Appendix S) triangulates that dominant fail-enriched features are not on salient gradient paths.
- Communicative clarity and breadth. The paper is clearly written, with transparent methodology (Section 3), appropriate caveats (Limitations), and explicit references to figures and tables that support claims. It tests several datasets and two DLMs, plus AR siblings for layer scans.

Specific figure/table-based strengths:
- Figure 6 (per-layer AUC and cosine) is pivotal for the difficulty interpretation; it shows peak AUC parity but layer shifts, and a pronounced divergence at the last layer, supporting that posttraining relocates, rather than expands, the probed capacity.
- Table 1 comprehensively reports steering flips across four plateau and pre-plateau conditions, directly supporting the “diagnostic not causal” reading.
- Figure 1 and Figure 3 visualize the plateau structure and are integral to the paper’s core temporal claim; the inclusion of permutation nulls and seed overlays increases credibility.

## Summary Of Weaknesses
1) Strength of the causal language vs. evidence
- The paper often reads as if difficulty “dominates” correctness signals broadly, yet the per-step and per-task statistical support is mixed. For instance, in Figure 2 many individual steps are not significant after Holm-Bonferroni and only two cells reach aggregated significance (Figure 4). The paper does note this, but several claims in Abstract and Discussion still feel categorical. It would help to better quantify the fraction of AUC attributable to transportable difficulty versus generation-specific correctness per task, and to temper the generality of the conclusion accordingly.

2) Potential circularity and double-dipping in feature selection
- Top-20 features are selected by fail-enrichment using labels within the same cell, then used to study within-fail clustering and even to compare supervised AUC in that subspace (Appendix E). The permutation null re-selects features for shuffles to mitigate inflation, and Appendix K adds held-out feature selection. This is good practice, but the main text conclusions still lean heavily on in-sample selected features. The paper would be stronger if the main figures emphasized held-out feature selection or if more analyses in the main text used fully held-out selection to avoid any perception of circularity.

3) Limitations of the steering scope relative to the causal claim
- The negative steering result is thorough for linear, single-layer, SAE-anchored and raw-residual directions, yet the main claim is that these features are “diagnostic correlates” rather than causes. Given DLMs’ non-linear, distributed computation, claiming non-causality from null linear interventions may be overreaching. The authors recognize this (Limitations: Steering coverage) but the Abstract still frames the result strongly. Including at least a small-scale non-linear or multi-layer coordinated intervention in the main text (beyond the brief joint suppression across four layers that was also null) would solidify the inference, or the claim should be softened.

4) Difficulty proxy and disentanglement depth
- The cross-generator transport is persuasive, but the length-stratification proxy in Appendix P shows only small AUC reductions (0.026–0.036). This is framed as evidence that difficulty goes beyond length, which is reasonable, but the paper could more robustly operationalize difficulty (e.g., expert-annotated difficulty strata, or task-specific structural difficulty features) to demonstrate that the “difficulty floor” is indeed the majority of probe signal. Right now, the primary quantitative disentanglement lever is transport. Relying mainly on transport makes the strength of the conclusion hinge on the assumption that generators share a difficulty ordering; this is discussed in Limitations but is central enough to deserve stronger anchoring.

5) Statistical framing and multiple comparisons
- The paper is careful to acknowledge that most single steps do not survive Holm-Bonferroni across cells. However, several narrative elements (e.g., “phase tendencies,” “agreement on 3/4 tasks”) could be misread as confirmatory without sufficiently conservative corrections. I appreciate the Fisher-combination approach in Figure 4 and the dense sweep statistics on LLaDA-MBPP (Appendix F), but for the cross-grid analysis in Figure 2, conclusions should be framed as exploratory tendencies unless backed by family-wise or FDR-controlled tests across the full grid.

Additional, more minor concerns and clarifications:
- Per-step SAE encoding relies on region-mean pooling over the generation region (Page 3). This choice could blur localized signals that matter for correctness; a sensitivity check to pooling granularity would strengthen the claim that plateau structure is not an artifact of averaging.
- The per-layer AR–DLM scan uses last-prompt-token residuals only (Page 3). While a practical analog to DLM’s pre-generation state, it is a limited snapshot compared to DLM’s multi-token, step-wise states. The strong conclusions about “capacity shifts” versus “capacity expansion” might be nuanced if broader prompt states were used.
- In Table 3 and Figure 6, the peaks are close in AUC. Confidence intervals or per-fold AUC variance for those peaks would help assess whether the small differences are meaningful.
- The cross-model step-axis alignment is argued via matched unmask fractions (Appendix R). This is reasonable, but because commit orders differ, step comparisons remain approximate; wording should always reflect this approximate nature when drawing cross-model temporal conclusions.

Figure- and table-specific critiques:
- Figure 2: The peak-step markers are visually persuasive. Given the non-significance at many points post correction, it would help to overlay adjusted significance or include a panel that aggregates per-task evidence rather than relying on single-step peaks in the main text.
- Table 1: The steering summary is valuable, but it would benefit from confidence intervals on the flip rates (even if zero) and a short power analysis addressing what effect sizes the study is powered to detect under the tested magnitudes and sample sizes.
- Figure 6 and Table 3: Excellent for illustrating the locus-shift argument. However, please report the standard error of AUC per layer or a shading band, and clarify whether C=0.01 was tuned anywhere else. Appendix O’s robustness is helpful; moving a subset of that to the main text would raise confidence.

## Potentially Missing Related Work
1) Anonymous, “Emergence and Evolution of Interpretable Concepts in Diffusion Models Through the Lens of Sparse Autoencoders,” 2025 — Applies SAEs to diffusion models to study concept evolution across denoising steps. Highly relevant to the paper’s temporal SAE analysis. Should be cited in Related Work and contrasted with the current DLM-specific, correctness/difficulty framing; also mention alongside Figure 2’s phase analysis.

2) Korznikov et al., “Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?,” 2026 — Directly relevant to validating whether discovered SAE features encode meaningful structure versus random baselines. Add to Section 2 and discuss around the feature-selection pipeline and Appendix K, possibly adding a control that compares against random dictionaries or shuffled encoders.

3) Olmo et al., “Features that Make a Difference: Leveraging Gradients for Improved Dictionary Learning,” 2025 — Proposes gradient-aware SAEs to prioritize features with stronger downstream impact. Given the paper’s diagnostic-not-causal conclusion, it is important to discuss whether a gradient-aware dictionary could surface more causal commitment-level features. Add to Related Work and consider in Discussion §5 under “Future directions.”

4) Gourevitch et al., “Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation,” 2026 — Provides theoretical insights into discrete diffusion training and denoising dynamics, relevant to interpreting what information can be represented at each timestep. Cite in Related Work and briefly relate to the plateau and late-emergence observations in §4.

5) Anonymous, “Likelihood-Based Diffusion Language Models,” 2024 — An analysis of DLMs and their training/sampling behavior that would contextualize the representational capacities the paper probes. Add to Related Work.

6) Anonymous, “SparseD: Sparse Attention for Diffusion Language Models,” 2025 — Proposes sparse attention mechanisms for DLMs, with analyses of step-wise attention sparsity. Relevant as an architectural angle on where difficulty signals may concentrate; mention in Related Work.

7) Anonymous, “DINGO: Constrained Inference for Diffusion LLMs,” 2024 — Develops constraint decoding for DLMs, intersecting with correctness under structural tasks like JSON. Should be referenced in Related Work and discussed briefly around the structured-task cells in Figure 2.

## Comments Suggestions And Typos
Actionable questions and suggestions:
- Quantify difficulty vs. correctness components: Move part of Appendix N and P into the main text to make a per-task decomposition. For example, report “transport-retained AUC” as a floor for difficulty and the residual as generation-specific. If possible, introduce a richer difficulty proxy beyond length, or provide human-annotated difficulty bins for a subset of items to directly validate the interpretation.
- Tighten statistical framing in Figure 2: Add FDR- or Holm-adjusted significance overlays or downplay single-step peaks. Consider reporting trajectory-level tests in the figure caption to prevent misinterpretation.
- Emphasize held-out feature selection in the main text: Given the relevance to double-dipping concerns, elevate results currently in Appendix K (held-out top-20 feature selection showing all 9 dense sub-steps significant) into the main narrative around Figure 1/Figure 3.
- Pooling sensitivity: Add a short analysis showing that the plateau and hand-off persist with finer-grained pooling over generation regions, or with position-weighted pooling that focuses on early-committed blocks for LLaDA.
- Steering breadth vs. claim scope: Either soften the causal wording (“diagnostic correlate” → “diagnostic correlate under linear SAE- and raw-residual interventions at specific layers/steps”), or introduce at least one small non-linear, multi-layer coordinated intervention in the main paper to support stronger causal language. For example, a low-rank non-linear projector or a gated multi-layer edit informed by the per-layer AUC peaks in Figure 6.

Typos and minor edits:
- Page 3, “We report CV-AUC” — consider including per-fold standard deviations for key layers (e.g., Figure 6).
- Page 5, “ing pass with 105/152 pass/fail” — likely “yielding pass/fail counts of …”; clarify phrasing.
- Table 2 header rows appear slightly misaligned; ensure the dominant-context and split cells are clearly formatted.
- Consistent naming: “Dream-7B-Base” sometimes shortened to “Dream-Base”; verify consistency across figures and text.
- Ensure consistent treatment of steps: sometimes denoted “s64,” elsewhere “step 64.”

Criteria for score change after rebuttal:
- Provide main-text analyses with held-out feature selection and corrected multiple-comparison framing for Figure 2’s grid, plus confidence intervals for AUC in Figure 6/Table 3. 
- Add at least one non-linear or multi-layer coordinated steering attempt in the main text, or temper causal claims accordingly.
- Strengthen the difficulty disentanglement with an independent difficulty proxy or annotations on a subset, quantifying how much of AUC is attributable to difficulty vs. correctness.

## Confidence
4 Quite sure after careful checking: I tried to check the important points carefully. It’s unlikely, though conceivable, that I missed something that should affect my ratings.

## Soundness
3.5 Between Acceptable and Strong: The empirical protocols are generally solid and multi-pronged, with thoughtful controls. Some claims slightly overreach the statistical backing and could be presented more conservatively.

## Excitement
3.5 Between interesting and exciting: The reinterpretation of correctness probes as difficulty-sensitive and the comprehensive steering null are both insightful and will likely stimulate discussion in the community.

## Overall Assessment
3.5 Borderline Conference: The paper is close to being acceptable for the main conference but has notable weaknesses.
Justification: The work presents a concrete, testable reinterpretation with careful cross-model evidence, strong negative results for linear steering, and clear temporal analyses. However, the statistical support across the full grid is mixed, several conclusions would benefit from stronger disentanglement of difficulty, and the causal language is somewhat ahead of the intervention scope. With clarifications and modest extensions, this could become a solid main-track paper; as is, it sits on the borderline.

## Best Paper Justification
N/A.

## Limitations And Societal Impact
The Limitations section is candid and thorough, covering SAE coverage, permutation test interpretation, steering scope, clustering choices, cross-model comparison, sample size, and disentanglement caveats. Potential societal impacts are modest and mostly positive in the context of interpretability and control; one risk is overinterpreting diagnostic features as levers for behavior change, which this paper explicitly cautions against. Consider a brief paragraph explicitly noting risks of misusing steering protocols or mischaracterizing “difficulty” features as proxies for protected attributes when applying similar analyses to user-generated content.

## Ethical concerns
None.

## Needs Ethics Review
No.

## Reproducibility
4 Mostly reproducible: The paper provides datasets, seeds, model checkpoints, SAE hyperparameters, and denoising schedules; intervention magnitudes and hooks are described in detail. Releasing code, as promised, would push this to 5.

## Datasets
2 Documentary: The paper reuses known datasets with clear rubrics; no new datasets are created. The detailed functional-correctness definitions (Appendix M) are useful for replication.

## Software
3 Potentially useful: The authors commit to releasing code and figures. Given the careful pipeline (SAE encoding, permutation null, steering hooks), a clean release would be useful to others.