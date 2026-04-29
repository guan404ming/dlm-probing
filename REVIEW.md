Official Review of Submission45 by Reviewer CnfD
Official Reviewby Reviewer CnfD14 Apr 2026, 01:15 (modified: 25 Apr 2026, 08:59)Program Chairs, Area Chairs, Reviewer CnfD, AuthorsRevisions
Paper Summary:
This paper demostrates that i) correctness signal emergence is task dependent and ii) that correctness signals emerge even at Step-0 of denoising in Diffusion Language Models. While Diffusion is traditionally used in computer vision tasks, the Diffusion Models used have been adapted for natural language, which makes this work of interest to ACL. The authors explore signal in diffusion model layers across JSON formulation, math reasoning (GSM8K), coding tasks (MBPP), and general QA multiple choice (ARC-Challenge).

Summary Of Strengths:
Novelty: While this is not the first paper to probe DLM hidden states, this is one of the first papers to probe hidden states for Discrete Masked Diffusion LM correctness.

Reproducibility: The authors thoroughly describe their experimentation; they describe the construction of the linear probe in detail, detail the computational resources required (NVIDIA A100s), and provide datasets (ARC-Challenge, GSM8K, MBPP) and model specs.

Summary Of Weaknesses:
-There is no detail of the dataset used to evaluate Llada and Dream performances on JSON evaluation.

-The authors demonstrate probe signal across layers for various tasks, but should also include a baseline signal in addition to baseline correctness to contrast the signal with baseline noise.

-The claim that the authors ”reveal divergent layer dynamics across architectures” is only supported for Discrete Masked Diffusion LMs. Continuous Diffuion LMs are not evaluated, and there is no explanation as to why they are excluded.

-The claim in Section 5.3 that ”per step probe confidence can identify instances whose outputs will be correct” is not justified sufficiently through the given experiments.

Comments Suggestions And Typos:
-Figure 2 is very small for readers (at 100% zoom, the axis titles and units/increments aren’t readable). Making it bigger can improve clarity.

-The authors have a brief note in the appendix discussion τ and p, but describing what τ and p in the main body can help with clarity.

-I suggest providing a short explanation of StandardScaler used in Appendix A for clarity.

-I also suggests providing an example prompt that used for Dream and Llada.

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Rating: 6: Marginally above acceptance threshold
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits

Official Review of Submission45 by Reviewer CenP
Official Reviewby Reviewer CenP13 Apr 2026, 22:24 (modified: 25 Apr 2026, 08:59)Program Chairs, Area Chairs, Reviewer CenP, AuthorsRevisions
Paper Summary:
This paper studies whether hidden states in DLM encode whether the final output will be functionally correct, and how that signal changes over denoising steps. The authors probe intermediate representations from two DLMs, LLaDA-8B and Dream-7B, on four tasks: JSON schema validation, GSM8K, MBPP, and ARC-Challenge.

Summary Of Strengths:
The paper is timely and well-motivated. DLM are still much less understood than autoregressive models, and the paper asks a clear question that matters for both interpretability and inference behavior. The use of functional correctness as the target is also a good choice, since it makes the probing study more meaningful than probing for surface features alone. The empirical story is easy to follow, especially the contrast between structural tasks and reasoning-heavy tasks.

Summary Of Weaknesses:
The step-0 result is interesting, but the paper sometimes frames it too strongly as a DLM specific finding. The AR baseline achieves comparable step-0 AUC on most tasks, which suggests that much of this signal may reflect prompt difficulty rather than a uniquely diffusion-based correctness signal.
The interpretation of cross-model layer dynamics is somewhat limited by the setup. LLaDA and Dream use different unmasking schedules, so the same denoising step is not directly comparable across models.
Comments Suggestions And Typos:
NA

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 4 = Strong: This study provides sufficient support for all of its claims. Some extra experiments could be nice, but not essential.
Rating: 8: Top 50% of accepted papers, clear accept
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits

Official Review of Submission45 by Reviewer iqxQ
Official Reviewby Reviewer iqxQ09 Apr 2026, 06:35 (modified: 25 Apr 2026, 08:59)Program Chairs, Area Chairs, Reviewer iqxQ, AuthorsRevisions
Paper Summary:
This paper studies when diffusion language models start to encode signals that are predictive of whether their final output will be correct. The authors probe hidden states across different denoising steps and layers using simple linear classifiers, and evaluate this on a mix of structural and reasoning tasks. The paper is clean and the experiments are reasonably thorough, but the main takeaway is not as strong as it initially appears. In particular, the step-0 result does not seem specific to diffusion models, and the rest of the findings are not fully disentangled from more general effects of iterative computation.

Summary Of Strengths:
The paper is cleanly executed and asks a relevant question that hasn’t been directly studied for diffusion LMs. The experimental setup is systematic rather than selective probing across steps and layers gives a reasonably complete picture instead of cherry-picked observations. The distinction between structural and reasoning tasks is one of the more convincing parts, and the trends are consistent across datasets. The layer-wise differences between the two models are also interesting and not entirely obvious, suggesting there is something real happening in how these models organize information during denoising. Overall, it reads as a careful empirical study rather than a rushed one.

Summary Of Weaknesses:
The main issue is that the central claim doesn’t really hold up under closer inspection. The step-0 signal, which is presented as a key result, is largely explained by prompt difficulty and not specific to diffusion, especially given the AR baseline. The improvement over denoising is also not clearly tied to diffusion, it could just be a byproduct of iterative computation, and there’s no baseline to rule that out. The probe itself likely relies on shallow cues (length, formatting, answer patterns), and the controls don’t fully eliminate that concern. The layer dynamics are interesting but underexplained and mostly speculative. Finally, the selective generation results feel overstated since they are based on offline simulation and don’t reflect a realistic deployment setup.

Comments Suggestions And Typos:
Questions for the Authors Since autoregressive models achieve similar step-0 performance, what part of the signal can be attributed specifically to diffusion rather than prompt difficulty? How does the probe compare to simpler uncertainty measures (e.g., entropy, log-probability) for early stopping? Can you better isolate whether the probe is learning semantic correctness versus dataset artifacts such as length, formatting, or answer patterns?

Confidence: 5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work.
Soundness: 5 = Excellent: This study is one of the most thorough I have seen, given its type.
Rating: 6: Marginally above acceptance threshold
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review