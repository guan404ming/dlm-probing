"""Counterfactual feature replacement on LLaDA-MBPP (EMNLP expansion).

The plain suppression result (modal_sae_steer.py) is null: ablating
f15601 at step 64 flips no labels. This script tests a stronger
counterfactual: at every step >= s_from, simultaneously
(a) SUBTRACT the top-K fail-enriched direction(s) and
(b) ADD the top-K pass-enriched direction(s).
This replaces the fail-cluster signature with a pass-cluster signature
at the SAE-layer residual, rather than just removing the fail direction.

If a directional replacement at L26 can flip correctness on a non-trivial
fraction of the n=202 sample set, it falsifies the "diagnostic only"
interpretation. If it remains null, it strengthens it.

Usage:
  .venv/bin/modal run src/applications/sae/modal_counterfactual.py \\
    --n-fail-c1 126 --n-fail-c0 26 --n-pass 50 \\
    --top-k 5 --alpha 5.0 --steer-from-step 64
"""

import modal

app = modal.App("sae-counterfactual")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0",
        "transformers==4.52.2",
        "accelerate>=0.30",
        "numpy",
        "datasets==2.21.0",
        "huggingface_hub",
    )
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

LLADA_NAME = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336
TEMPERATURE = 0.2
GEN_LENGTH = 256
STEPS = 128
BLOCK_LENGTH = 32

SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYER = 26
SAE_TRAINER = 2


@app.function(
    image=image, gpu="A100", timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_counterfactual(
    n_fail_c1: int,
    n_fail_c0: int,
    n_pass: int,
    top_k: int,
    alpha: float,
    steer_from_step: int,
):
    import json
    import os
    import re
    import signal
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModel

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    # Load Stage 2 diag for cluster sample indices
    with open("/results/mbpp_llada/sae_diagnose_stage2.json") as f:
        diag = json.load(f)
    clusters = {c["cluster"]: c["fail_sample_indices"] for c in diag["clusters"]}
    fail_c1_idxs = clusters[1][:n_fail_c1]
    fail_c0_idxs = clusters[0][:n_fail_c0]

    # Load full diagnose for step 64 to pick fail vs pass enriched features
    with open("/results/mbpp_llada/sae_diagnose_stage2_s64.json") as f:
        d64 = json.load(f)
    top_fail = [f["feature_id"] for f in d64["top_fail_features"][:top_k]]
    # Top-K pass-enriched = features with most NEGATIVE enrichment
    # The diag file may not include this; we recompute below from cached chunks.

    # Recompute top pass-enriched features from cached chunks
    in_dir = "/results/mbpp_llada_dense"
    chunk_files = sorted([f for f in os.listdir(in_dir) if f.startswith("chunk_off") and f.endswith(".npz")])
    all_labels = []
    region_feats_s64 = {r: [] for r in range(4)}
    for cf in chunk_files:
        data = np.load(f"{in_dir}/{cf}")
        all_labels.append(data["labels"])
        if "feat_s64_r0" not in data.files:
            continue
        for r in range(4):
            region_feats_s64[r].append(data[f"feat_s64_r{r}"])
    labels = np.concatenate(all_labels).astype(int)
    print(f"Loaded {len(labels)} samples; pass={int(labels.sum())} fail={int((labels==0).sum())}")

    # Load SAE
    sae_path_dir = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
    ae_local = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path_dir}/ae.pt", cache_dir="/hf-cache")
    cfg_local = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path_dir}/config.json", cache_dir="/hf-cache")
    with open(cfg_local) as f:
        sae_cfg = json.load(f)
    sae_k = sae_cfg["trainer"]["k"]
    state = torch.load(ae_local, map_location="cpu", weights_only=True)
    W_enc = state["encoder.weight"].cuda().to(torch.bfloat16)
    b_enc = state["encoder.bias"].cuda().to(torch.bfloat16)
    W_dec = state["decoder.weight"].cuda().to(torch.bfloat16)
    b_dec_raw = state.get("b_dec")
    if b_dec_raw is None:
        b_dec_raw = state.get("decoder.bias")
    b_dec = b_dec_raw.cuda().to(torch.bfloat16)

    # Encode cached residuals at s64 to find top pass-enriched features
    sae_acts = []
    for r in range(4):
        feats = np.concatenate(region_feats_s64[r])
        x = feats[:, SAE_LAYER, :].astype(np.float32)
        with torch.no_grad():
            x_t = torch.from_numpy(x).cuda()
            x_norm = x_t - b_dec.float()
            pre = x_norm @ W_enc.T.float() + b_enc.float()
            tv, ti = pre.topk(sae_k, dim=-1)
            tv = tv.relu()
            z = torch.zeros_like(pre)
            z.scatter_(-1, ti, tv)
            sae_acts.append(z.cpu().numpy())
    sae_mean = np.mean(sae_acts, axis=0)
    active = (sae_mean > 0).astype(np.float32)
    p_fail = active[labels == 0].mean(axis=0)
    p_pass = active[labels == 1].mean(axis=0)
    enr = p_fail - p_pass
    top_pass = np.argsort(enr)[:top_k].tolist()  # most negative = pass-enriched
    print(f"top-{top_k} fail features: {top_fail}")
    print(f"top-{top_k} pass features: {top_pass} (enrichment: {[round(enr[i], 4) for i in top_pass]})")

    fail_dec_cols = W_dec[:, top_fail].clone()  # (d_in, k)
    pass_dec_cols = W_dec[:, top_pass].clone()  # (d_in, k)
    fail_feats_tensor = torch.tensor(top_fail, device="cuda")
    pass_feats_tensor = torch.tensor(top_pass, device="cuda")

    # Load LLaDA + MBPP
    tokenizer = AutoTokenizer.from_pretrained(LLADA_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLADA_NAME, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])

    # Pass sample indices (disjoint from fails)
    fail_set = set([i for ids in clusters.values() for i in ids])
    pass_pool = [i for i in range(len(instances)) if i not in fail_set]
    rng = np.random.RandomState(42)
    pass_idxs = list(rng.choice(pass_pool, size=n_pass, replace=False))

    # Steering hook: subtract fail_dir, add pass_dir
    state_box = {"step": -1, "steer_enabled": False, "max_delta_norm": 0.0, "mod_count": 0}

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if not state_box["steer_enabled"]:
            return output
        if state_box["step"] < steer_from_step:
            return output
        x = h - b_dec
        pre = torch.nn.functional.linear(x, W_enc, b_enc)
        topk_vals, topk_idx = pre.topk(sae_k, dim=-1)
        topk_vals = topk_vals.relu()
        delta = torch.zeros_like(h)
        for ti, fid in enumerate(top_fail):
            mask = (topk_idx == fid)
            a_t = (topk_vals * mask).sum(dim=-1, keepdim=True)
            delta = delta + alpha * a_t * fail_dec_cols[:, ti].view(1, 1, -1)
        for ti, pid in enumerate(top_pass):
            mask = (topk_idx == pid)
            a_t = (topk_vals * mask).sum(dim=-1, keepdim=True)
            # ADD pass direction with similar scale to the typical fail activation
            # Use mean fail activation magnitude as proxy (alpha already scales it)
            delta = delta - alpha * a_t.mean() * pass_dec_cols[:, ti].view(1, 1, -1)
        h.sub_(delta)
        dn = float(delta.norm().item())
        if dn > state_box["max_delta_norm"]:
            state_box["max_delta_norm"] = dn
        state_box["mod_count"] += 1
        return output

    # Locate layer
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        layers = model.model.transformer.blocks
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        layers = model.transformer.blocks
    else:
        layers = model.transformer.h
    hook_handle = layers[SAE_LAYER].register_forward_hook(hook)

    # MBPP check
    def check_mbpp(instance, output_text):
        try:
            m = re.search(r"```python\s*(.*?)```", output_text, re.DOTALL)
            code = m.group(1) if m else output_text
            full_code = code + "\n" + "\n".join(instance["test_imports"]) + "\n"
            full_code += "\n".join(instance["test_list"])
            old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(10)
            try:
                ns = {}
                exec(full_code, ns)
                return True
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            return False

    def generate(prompt_ids, prompt_len, steer_on):
        full = torch.full(
            (1, prompt_len + GEN_LENGTH), MASK_ID, dtype=prompt_ids.dtype, device=prompt_ids.device,
        )
        full[0, :prompt_len] = prompt_ids[0]
        state_box["steer_enabled"] = steer_on
        state_box["step"] = -1
        state_box["mod_count"] = 0
        state_box["max_delta_norm"] = 0.0
        x = full.clone()
        n_blocks = GEN_LENGTH // BLOCK_LENGTH
        steps_per_block = STEPS // n_blocks
        global_step = 0
        for b in range(n_blocks):
            blk_start = prompt_len + b * BLOCK_LENGTH
            blk_end = blk_start + BLOCK_LENGTH
            for inner in range(steps_per_block):
                state_box["step"] = global_step
                with torch.no_grad():
                    out = model(x)
                logits = out.logits
                block_mask = (x[0, blk_start:blk_end] == MASK_ID)
                if not block_mask.any():
                    global_step += 1
                    continue
                block_logits = logits[0, blk_start:blk_end]
                if TEMPERATURE == 0:
                    probs = torch.softmax(block_logits, dim=-1)
                    conf, pred = probs.max(dim=-1)
                else:
                    probs = torch.softmax(block_logits / TEMPERATURE, dim=-1)
                    conf, _ = probs.max(dim=-1)
                    pred = torch.multinomial(probs, 1).squeeze(-1)
                conf = conf.masked_fill(~block_mask, -1.0)
                n_to_unmask = max(1, int(block_mask.sum().item() / max(1, (steps_per_block - inner))))
                top_idx = conf.topk(n_to_unmask).indices
                x[0, blk_start + top_idx] = pred[top_idx]
                global_step += 1
        gen_ids = x[0, prompt_len:].tolist()
        gen_ids = [t for t in gen_ids if t != MASK_ID]
        return tokenizer.decode(gen_ids)

    def make_prompt(inst):
        msgs = [
            {"role": "user", "content": f"Write a Python function. Only output code in a Python block.\n\nProblem: {inst['prompt']}"}
        ]
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        return ids

    def run_label(label, idxs):
        rows = []
        for k, idx in enumerate(idxs):
            inst = instances[idx]
            prompt_ids = make_prompt(inst)
            prompt_len = prompt_ids.shape[1]
            t0 = time.time()
            base_txt = generate(prompt_ids, prompt_len, steer_on=False)
            steer_txt = generate(prompt_ids, prompt_len, steer_on=True)
            base_pass = check_mbpp(inst, base_txt)
            steer_pass = check_mbpp(inst, steer_txt)
            rows.append({"idx": idx, "task_id": inst["task_id"],
                         "base_pass": int(base_pass), "steer_pass": int(steer_pass)})
            if (k + 1) % 20 == 0 or k < 2:
                print(f"  [{label} {k+1}/{len(idxs)}] base={base_pass} steer={steer_pass} ({time.time()-t0:.1f}s)")
        return rows

    print(f"\n=== Counterfactual: fail-{top_k} OUT + pass-{top_k} IN, alpha={alpha}, from s{steer_from_step} ===")
    print(f"\n--- Cluster 1 fails (n={len(fail_c1_idxs)}) ---")
    rows_c1 = run_label("c1", fail_c1_idxs)
    print(f"\n--- Cluster 0 fails (n={len(fail_c0_idxs)}) ---")
    rows_c0 = run_label("c0", fail_c0_idxs)
    print(f"\n--- Pass regression (n={len(pass_idxs)}) ---")
    rows_pass = run_label("pass", pass_idxs)

    def summarize(rows, label):
        n = len(rows)
        if n == 0: return None
        bp = sum(r["base_pass"] for r in rows)
        sp = sum(r["steer_pass"] for r in rows)
        f2p = sum(1 for r in rows if not r["base_pass"] and r["steer_pass"])
        p2f = sum(1 for r in rows if r["base_pass"] and not r["steer_pass"])
        return {"label": label, "n": n, "base_pass": bp, "steer_pass": sp,
                "fail_to_pass": f2p, "pass_to_fail": p2f}

    summary = [
        summarize(rows_c1, "fail_c1"),
        summarize(rows_c0, "fail_c0"),
        summarize(rows_pass, "pass"),
    ]
    print("\n" + "="*70)
    print(f"Counterfactual summary (fail-{top_k} OUT, pass-{top_k} IN, alpha={alpha}, s{steer_from_step}):")
    for s in summary:
        if s:
            print(f"  {s['label']}: n={s['n']} base_pass={s['base_pass']} steer_pass={s['steer_pass']} "
                  f"f→p={s['fail_to_pass']} p→f={s['pass_to_fail']}")

    out = {
        "config": {"top_fail": top_fail, "top_pass": top_pass, "alpha": alpha,
                   "steer_from_step": steer_from_step, "sae_layer": SAE_LAYER},
        "summary": summary,
        "rows": rows_c1 + rows_c0 + rows_pass,
    }
    out_path = f"/results/mbpp_llada/sae_counterfactual_k{top_k}_a{alpha}_s{steer_from_step}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")

    hook_handle.remove()
    return json.dumps({"summary": summary})


@app.local_entrypoint()
def main(
    n_fail_c1: int = 126,
    n_fail_c0: int = 26,
    n_pass: int = 50,
    top_k: int = 5,
    alpha: float = 5.0,
    steer_from_step: int = 64,
):
    print(run_counterfactual.remote(n_fail_c1, n_fail_c0, n_pass, top_k, alpha, steer_from_step))
