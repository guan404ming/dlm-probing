# dllm-probing

See README.md for scripts, data format, results, and usage.

## Context
- Spun off from `dgrammar` repo's `probe/` directory
- Paper target: ACL SRW 2025 (4 pages)
- Framing: first empirical probing study on diffusion LLM hidden states

## Probing Details (jsonschema, detailed analysis)
- Probe: PCA(64) + StandardScaler + LogisticRegression, 5-fold stratified CV
- 4 position regions (gen_length / 4 each, mean-pooled), 7 checkpoint steps (0,1,4,16,32,64,127)
- Step 0 already has signal (AUC ~0.78), prompt encoding alone predicts correctness
- Signal peaks around step 64 (AUC ~0.84-0.85), not at final step
- LLaDA best layers stay in upper layers (L22-26), Dream migrates to lower layers during denoising
- LLaDA concentrates signal at JSON opening (region 0), Dream distributes more uniformly

## Key Findings
1. Correctness signal exists from step 0 (AUC ~0.78). Diffusion-specific: AR models can't do this.
2. Signal peaks at step 64, not the final step.
3. Layer dynamics differ: LLaDA upper layers, Dream migrates upper to lower.
4. Seed reranking fails: probe captures instance difficulty, not seed quality.
5. Adaptive compute saves 22-34% steps (simulation only, not validated with real reduced-step generation).

## Tools & Environment
- Python 3.12, datasets==2.21.0, transformers==4.52.2
- Modal for cloud GPU (A100), run via `.venv/bin/modal`
- Use `--chunks 8` for 8x A100 parallel
- Use `uv` for Python package management (not pip)
