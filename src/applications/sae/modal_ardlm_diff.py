"""Angle 3 feasibility: Qwen2.5-7B-Base vs Dream-7B-Base hidden-state diff at L23.

For 30 fail + 30 pass MBPP samples (labels from Dream-Instruct generation):
  1. Qwen2.5-7B-Base prompt forward pass -> h_Q[i] = hidden state at L23, last prompt token
  2. Dream-7B-Base denoise to plateau step 64 -> h_D[i] = L23 last prompt token
  3. d[i] = h_D[i] - h_Q[i]
  4. Metrics:
     - AUC(||d[i]||_2, correctness)
     - PCA-1 of {d[i]} variance explained
     - cos(PCA-1, mean_diff_pass - mean_diff_fail)

Output: /results/mbpp_dream/ardlm_diff_L23.json

Usage:
  .venv/bin/modal run src/applications/sae/modal_ardlm_diff.py
"""

import modal

app = modal.App("ardlm-diff")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0",
        "transformers==4.52.2",
        "accelerate>=0.30",
        "numpy",
        "scikit-learn",
        "datasets==2.21.0",
        "huggingface_hub",
    )
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

DREAM_BASE = "Dream-org/Dream-v0-Base-7B"
DREAM_INSTRUCT = "Dream-org/Dream-v0-Instruct-7B"
QWEN_BASE = "Qwen/Qwen2.5-7B"
TARGET_LAYER = 23
MASK_ID = 151666  # Dream mask token id
TEMPERATURE = 0.2
GEN_LENGTH = 256
STEPS = 128
BLOCK_LENGTH = 32
TARGET_STEP = 64  # plateau


@app.function(
    image=image, gpu="A100-80GB", timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run(n_fail: int = 0, n_pass: int = 0):
    import json
    import os
    import re
    import signal as pysig
    import time
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    # Reuse Dream-Instruct labels from prior diagnose (cluster fail samples + sampled pass)
    diag_path = "/results/mbpp_dream/sae_diagnose_stage2.json"
    if not os.path.exists(diag_path):
        raise FileNotFoundError(f"Need {diag_path} for fail/pass labels")
    with open(diag_path) as f:
        diag = json.load(f)
    fail_idxs = []
    for c in diag.get("clusters", []):
        fail_idxs.extend(c.get("fail_sample_indices", []))
    fail_idxs = list(dict.fromkeys(fail_idxs))
    if n_fail > 0:
        fail_idxs = fail_idxs[:n_fail]
    print(f"Loaded {len(fail_idxs)} fail indices from diagnose")

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])
    fail_set = set(fail_idxs)
    pass_pool = [i for i in range(len(instances)) if i not in fail_set]
    if n_pass > 0:
        rng = np.random.RandomState(42)
        pass_idxs = list(rng.choice(pass_pool, size=n_pass, replace=False))
    else:
        pass_idxs = pass_pool
    all_idxs = fail_idxs + pass_idxs
    labels = np.array([0] * len(fail_idxs) + [1] * len(pass_idxs))
    print(f"Total: {len(all_idxs)} samples (fail={len(fail_idxs)}, pass={len(pass_idxs)})")

    # Build prompts using Dream-Instruct chat template (matches our existing data)
    instruct_tok = AutoTokenizer.from_pretrained(DREAM_INSTRUCT, trust_remote_code=True)
    prompts_text = []
    for idx in all_idxs:
        inst = instances[idx]
        msgs = [{"role": "user",
                 "content": f"Write a Python function. Only output code in a Python block.\n\nProblem: {inst['prompt']}"}]
        text = instruct_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompts_text.append(text)

    # ---- Pass 1: Qwen2.5-7B-Base forward pass ----
    print(f"\n=== Loading Qwen2.5-7B-Base ===")
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True,
        output_hidden_states=True,
    ).cuda().eval()
    H_DIM = qwen.config.hidden_size
    h_Q = np.zeros((len(all_idxs), H_DIM), dtype=np.float32)
    for k, prompt in enumerate(prompts_text):
        ids = qwen_tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = qwen(ids, output_hidden_states=True)
        h = out.hidden_states[TARGET_LAYER + 1][0, -1, :].float().cpu().numpy()
        h_Q[k] = h
        if (k + 1) % 10 == 0:
            print(f"  Qwen {k+1}/{len(all_idxs)} h_norm={np.linalg.norm(h):.2f}")
    del qwen
    torch.cuda.empty_cache()
    print(f"Qwen done, h_Q shape={h_Q.shape}")

    # ---- Pass 2: Dream-7B-Base denoise to plateau step 64 ----
    print(f"\n=== Loading Dream-7B-Base ===")
    dream_tok = AutoTokenizer.from_pretrained(DREAM_BASE, trust_remote_code=True)
    dream = AutoModel.from_pretrained(
        DREAM_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    layers = None
    for path in [("model", "layers"), ("model", "transformer", "h"), ("transformer", "h")]:
        cur = dream
        ok = True
        for a in path:
            if hasattr(cur, a):
                cur = getattr(cur, a)
            else:
                ok = False; break
        if ok:
            layers = cur
            print(f"Dream layers at model.{'.'.join(path)}, n={len(layers)}")
            break
    if layers is None:
        raise RuntimeError("Cannot find Dream layers")

    captured = {"h": None}
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().clone()
    hook_handle = layers[TARGET_LAYER].register_forward_hook(hook)

    h_D = np.zeros((len(all_idxs), H_DIM), dtype=np.float32)
    for k, prompt in enumerate(prompts_text):
        ids = dream_tok(prompt, return_tensors="pt").input_ids.cuda()
        pl = ids.shape[1]
        full = torch.full((1, pl + GEN_LENGTH), MASK_ID, dtype=ids.dtype, device=ids.device)
        full[0, :pl] = ids[0]
        captured["h"] = None
        with torch.no_grad():
            _ = dream(full)
        if captured["h"] is None:
            raise RuntimeError(f"Hook did not fire on sample {k}")
        h_D[k] = captured["h"][0, pl - 1, :].float().cpu().numpy()
        if (k + 1) % 10 == 0 or k < 2:
            print(f"  Dream {k+1}/{len(all_idxs)} h_norm={np.linalg.norm(h_D[k]):.2f}  cap_shape={tuple(captured['h'].shape)}")
    hook_handle.remove()
    del dream
    torch.cuda.empty_cache()
    print(f"Dream done, h_D shape={h_D.shape}")

    # ---- Analysis ----
    d = h_D - h_Q
    d_norms = np.linalg.norm(d, axis=1)
    print(f"\n=== Diff stats ===")
    print(f"d_norm fail={d_norms[labels==0].mean():.2f}±{d_norms[labels==0].std():.2f}")
    print(f"d_norm pass={d_norms[labels==1].mean():.2f}±{d_norms[labels==1].std():.2f}")
    try:
        auc_norm = roc_auc_score(labels, d_norms)
    except Exception:
        auc_norm = float("nan")
    print(f"AUC(||d||, correctness)={auc_norm:.3f}")

    pca = PCA(n_components=10)
    pca.fit(d)
    var_exp = pca.explained_variance_ratio_
    print(f"PCA var_exp top-10: {[f'{v:.3f}' for v in var_exp]}")

    pc1_proj = d @ pca.components_[0]
    try:
        auc_pc1 = roc_auc_score(labels, pc1_proj)
        auc_pc1 = max(auc_pc1, 1 - auc_pc1)
    except Exception:
        auc_pc1 = float("nan")
    print(f"AUC(PC1 projection, correctness)={auc_pc1:.3f}")

    mean_diff_dir = d[labels == 1].mean(0) - d[labels == 0].mean(0)
    mean_diff_dir /= np.linalg.norm(mean_diff_dir) + 1e-8
    cos_pc1_mean = float(np.abs(pca.components_[0] @ mean_diff_dir))
    print(f"|cos(PC1, mean_diff_dir)|={cos_pc1_mean:.3f}")

    md_proj = d @ mean_diff_dir
    try:
        auc_md = roc_auc_score(labels, md_proj)
        auc_md = max(auc_md, 1 - auc_md)
    except Exception:
        auc_md = float("nan")
    print(f"AUC(mean-diff direction, correctness)={auc_md:.3f}")

    out = {
        "config": {"n_fail": n_fail, "n_pass": n_pass, "layer": TARGET_LAYER,
                    "target_step": TARGET_STEP, "qwen": QWEN_BASE, "dream": DREAM_BASE},
        "fail_idxs": [int(i) for i in fail_idxs],
        "pass_idxs": [int(i) for i in pass_idxs],
        "d_norms": d_norms.tolist(),
        "labels": labels.tolist(),
        "auc_norm": float(auc_norm),
        "pca_var_exp": var_exp.tolist(),
        "auc_pc1": float(auc_pc1),
        "cos_pc1_mean_diff_dir": float(cos_pc1_mean),
        "auc_mean_diff_dir": float(auc_md),
    }
    out_path = f"/results/mbpp_dream/ardlm_diff_L{TARGET_LAYER}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")

    np.savez(f"/results/mbpp_dream/ardlm_diff_vectors_L{TARGET_LAYER}.npz",
             h_Q=h_Q, h_D=h_D, labels=labels)
    RESULTS_VOL.commit()
    return json.dumps({"auc_norm": auc_norm, "auc_pc1": auc_pc1,
                       "auc_mean_diff": auc_md, "pca1_var": float(var_exp[0])})


@app.local_entrypoint()
def main(n_fail: int = 30, n_pass: int = 30):
    print(run.remote(n_fail, n_pass))
