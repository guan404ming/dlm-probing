"""All-layer AR-DLM diff: collect hidden states at every layer in one forward pass.

For all MBPP-sanitized test samples (n=257), capture last-prompt-token hidden
state at each layer for both Qwen2.5-7B-Base and Dream-7B-Base, then compute
5-fold CV AUC of correctness probe per layer.

If AUC(h_D, layer) curve matches AUC(h_Q, layer) curve at every layer, the
"correctness signal is inherited from AR base" hypothesis is strongly supported.

Usage:
  .venv/bin/modal run --detach src/applications/sae/modal_ardlm_diff_alllayers.py
"""

import modal

app = modal.App("ardlm-diff-alllayers")

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
N_LAYERS = 28
GEN_LENGTH = 256
MASK_ID = 151666


@app.function(
    image=image, gpu="A100-80GB", timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run():
    import json
    import os
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.decomposition import PCA

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    diag_path = "/results/mbpp_dream/sae_diagnose_stage2.json"
    with open(diag_path) as f:
        diag = json.load(f)
    fail_idxs = []
    for c in diag.get("clusters", []):
        fail_idxs.extend(c.get("fail_sample_indices", []))
    fail_idxs = list(dict.fromkeys(fail_idxs))

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])
    fail_set = set(fail_idxs)
    pass_idxs = [i for i in range(len(instances)) if i not in fail_set]
    all_idxs = fail_idxs + pass_idxs
    labels = np.array([0] * len(fail_idxs) + [1] * len(pass_idxs))
    print(f"Total {len(all_idxs)} samples: fail={len(fail_idxs)}, pass={len(pass_idxs)}")

    instruct_tok = AutoTokenizer.from_pretrained(DREAM_INSTRUCT, trust_remote_code=True)
    prompts_text = []
    for idx in all_idxs:
        inst = instances[idx]
        msgs = [{"role": "user",
                 "content": f"Write a Python function. Only output code in a Python block.\n\nProblem: {inst['prompt']}"}]
        text = instruct_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompts_text.append(text)

    def find_layers(model):
        for path in [("model", "layers"), ("model", "transformer", "h"), ("transformer", "h")]:
            cur = model
            ok = True
            for a in path:
                if hasattr(cur, a):
                    cur = getattr(cur, a)
                else:
                    ok = False; break
            if ok:
                return cur, path
        raise RuntimeError("Cannot find layers")

    def collect_all_layers(model, tok, prompts, mask_token=None, gen_mask_len=0, label="model"):
        layers, _ = find_layers(model)
        n_l = len(layers)
        d_h = model.config.hidden_size
        captured = [None] * n_l
        def make_hook(li):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[li] = h.detach().clone()
            return hook
        handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_l)]
        H = np.zeros((len(prompts), n_l, d_h), dtype=np.float32)
        for k, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            pl = ids.shape[1]
            if mask_token is not None and gen_mask_len > 0:
                full = torch.full((1, pl + gen_mask_len), mask_token, dtype=ids.dtype, device=ids.device)
                full[0, :pl] = ids[0]
                inp = full
            else:
                inp = ids
            for i in range(n_l):
                captured[i] = None
            with torch.no_grad():
                _ = model(inp)
            for i in range(n_l):
                if captured[i] is None:
                    raise RuntimeError(f"layer {i} hook did not fire")
                H[k, i] = captured[i][0, pl - 1, :].float().cpu().numpy()
            if (k + 1) % 20 == 0 or k < 2:
                print(f"  {label} {k+1}/{len(prompts)}")
        for h in handles:
            h.remove()
        return H

    print("\n=== Loading Qwen2.5-7B-Base ===")
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    H_Q = collect_all_layers(qwen, qwen_tok, prompts_text, label="Qwen")
    print(f"H_Q shape={H_Q.shape}")
    del qwen
    torch.cuda.empty_cache()

    print("\n=== Loading Dream-7B-Base ===")
    dream_tok = AutoTokenizer.from_pretrained(DREAM_BASE, trust_remote_code=True)
    dream = AutoModel.from_pretrained(
        DREAM_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    H_D = collect_all_layers(dream, dream_tok, prompts_text,
                             mask_token=MASK_ID, gen_mask_len=GEN_LENGTH, label="Dream")
    print(f"H_D shape={H_D.shape}")
    del dream
    torch.cuda.empty_cache()

    np.savez("/results/mbpp_dream/ardlm_diff_alllayers.npz",
             H_Q=H_Q, H_D=H_D, labels=labels)
    RESULTS_VOL.commit()

    def cv_auc(X, y, C=1.0):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=C).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        return float(np.mean(aucs)), float(np.std(aucs))

    n_l = H_Q.shape[1]
    rows = []
    print(f"\n{'layer':>5s}  {'AUC h_Q':>10s}  {'AUC h_D':>10s}  {'AUC d':>10s}  {'cos(Q,D)':>10s}")
    for li in range(n_l):
        hq = H_Q[:, li, :]
        hd = H_D[:, li, :]
        dd = hd - hq
        m_q, s_q = cv_auc(hq, labels, C=0.01)
        m_d, s_d = cv_auc(hd, labels, C=0.01)
        m_dd, s_dd = cv_auc(dd, labels, C=0.01)
        cos = float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                             for a, b in zip(hq, hd)]))
        rows.append({"layer": li, "auc_qwen": m_q, "auc_qwen_std": s_q,
                     "auc_dream": m_d, "auc_dream_std": s_d,
                     "auc_diff": m_dd, "auc_diff_std": s_dd,
                     "cos_q_d_mean": cos,
                     "norm_q": float(np.linalg.norm(hq, axis=1).mean()),
                     "norm_d": float(np.linalg.norm(hd, axis=1).mean())})
        print(f"  {li:3d}  {m_q:.3f}±{s_q:.3f}  {m_d:.3f}±{s_d:.3f}  {m_dd:.3f}±{s_dd:.3f}  {cos:>10.3f}")

    out = {
        "config": {"n_samples": len(labels), "n_fail": int((labels == 0).sum()),
                   "n_pass": int((labels == 1).sum()), "qwen": QWEN_BASE, "dream": DREAM_BASE},
        "per_layer": rows,
    }
    out_path = "/results/mbpp_dream/ardlm_diff_alllayers.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")
    return json.dumps({"ok": True, "n_layers": n_l, "n_samples": len(labels)})


@app.local_entrypoint()
def main():
    print(run.remote())
