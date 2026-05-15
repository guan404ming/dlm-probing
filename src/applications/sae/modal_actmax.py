"""Activation-maximizing token contexts for target SAE features (review W8).

For each target feature (f15601, f3892 on LLaDA L26 Mask-SAE), re-run LLaDA-8B
forward at step 64 on all 257 MBPP-sanitized samples, encode every generation
position through the SAE, and record the top-K (sample_id, token_position,
activation_value) entries. Token strings and a +/- 32-token context window are
saved so the features can be characterised semantically.

Output: /results/mbpp_llada/actmax_features.json with per-feature top-K examples.
"""

import modal

app = modal.App("sae-actmax")

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
GEN_LENGTH = 256
STEPS = 128
BLOCK_LENGTH = 32

SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYER = 26
SAE_TRAINER = 2
TARGET_FEATURES = [15601, 3892, 5561, 11265, 2144]  # main + secondary
TARGET_STEP = 64
TOP_K_EXAMPLES = 30
CONTEXT_TOKENS = 32


@app.function(
    image=image,
    gpu="A100:2",
    timeout=10800,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_actmax():
    import json
    import os

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModel

    os.environ["HF_HOME"] = "/hf-cache"
    RESULTS_VOL.reload()

    # SAE
    sae_path_dir = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
    ae_local = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path_dir}/ae.pt", cache_dir="/hf-cache")
    cfg_local = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path_dir}/config.json", cache_dir="/hf-cache")
    with open(cfg_local) as f:
        sae_cfg = json.load(f)
    sae_k = sae_cfg["trainer"]["k"]
    state = torch.load(ae_local, map_location="cpu", weights_only=True)
    W_enc = state["encoder.weight"].cuda().to(torch.bfloat16)
    b_enc = state["encoder.bias"].cuda().to(torch.bfloat16)
    b_dec_raw = state.get("b_dec")
    if b_dec_raw is None:
        b_dec_raw = state.get("decoder.bias")
    b_dec = b_dec_raw.cuda().to(torch.bfloat16)
    enc_op = "in_at_wt" if W_enc.shape[0] != b_dec.shape[0] else "in_at_w"
    print(f"SAE k={sae_k}, W_enc={tuple(W_enc.shape)}")

    # LLaDA
    tokenizer = AutoTokenizer.from_pretrained(LLADA_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLADA_NAME, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()

    # Diagnose stage 2: get fail/pass labels and ordering
    with open("/results/mbpp_llada/sae_diagnose_stage2.json") as f:
        diag = json.load(f)
    n_total = diag.get("n_samples", 257)
    # Reconstruct labels by examining diag clusters
    labels = np.zeros(n_total, dtype=int)
    for c in diag.get("clusters", []):
        # cluster 0/1 are fail subclusters
        for idx in c["fail_sample_indices"]:
            if idx < n_total:
                labels[idx] = 0
    pass_set = set(diag.get("pass_indices", []))
    for idx in pass_set:
        if idx < n_total:
            labels[idx] = 1
    # Fallback: all not-in-fail are pass
    fail_set = set()
    for c in diag.get("clusters", []):
        fail_set.update(c["fail_sample_indices"])
    for i in range(n_total):
        if i not in fail_set:
            labels[i] = 1
    print(f"labels: pass={int((labels==1).sum())} fail={int((labels==0).sum())}")

    # MBPP
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])

    # Hook to capture L26 residual at step 64
    captured = {"hidden": None}

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = h.detach()
        return output

    # Locate layer 26
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        layers = model.model.transformer.blocks
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        layers = model.transformer.blocks
    else:
        layers = model.transformer.h
    print(f"Found {len(layers)} layers, hooking layer {SAE_LAYER}")
    layers[SAE_LAYER].register_forward_hook(hook)

    # Setup generation prompt template
    def make_prompt(prompt_str, task_id):
        msgs = [
            {"role": "user", "content": f"Write a Python function. Only output code in a Python block.\n\nProblem: {prompt_str}"}
        ]
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        return ids

    # For each target feature: track top-K activations
    # entry: (activation, sample_id, token_pos_in_gen, label, token_str, context_str)
    top_per_feat = {fid: [] for fid in TARGET_FEATURES}

    @torch.no_grad()
    def run_one_sample(sample_id):
        inst = instances[sample_id]
        prompt_ids = make_prompt(inst["prompt"], inst["task_id"])
        prompt_len = prompt_ids.shape[1]
        # Build masked input: prompt + GEN_LENGTH masks
        full = torch.full(
            (1, prompt_len + GEN_LENGTH), MASK_ID, dtype=prompt_ids.dtype, device=prompt_ids.device,
        )
        full[0, :prompt_len] = prompt_ids[0]

        # Run block-wise unmasking up to step 64 (a single forward+commit loop)
        n_blocks = GEN_LENGTH // BLOCK_LENGTH
        steps_per_block = STEPS // n_blocks
        # Block 64 = first two blocks complete (step 64 = end of block 1)
        target_blocks = 2  # blocks 0 and 1 fully unmasked at step 64
        x = full.clone()
        for b in range(target_blocks):
            block_start = prompt_len + b * BLOCK_LENGTH
            block_end = block_start + BLOCK_LENGTH
            for inner_step in range(steps_per_block):
                # forward
                out = model(x)
                logits = out.logits
                # confidence of mask positions in this block
                block_mask = (x[0, block_start:block_end] == MASK_ID)
                if not block_mask.any():
                    break
                block_logits = logits[0, block_start:block_end]
                probs = F.softmax(block_logits, dim=-1)
                conf, pred = probs.max(dim=-1)
                conf = conf.masked_fill(~block_mask, -1.0)
                # Number to unmask this step
                n_to_unmask = max(1, int(block_mask.sum().item() / max(1, (steps_per_block - inner_step))))
                top_idx = conf.topk(n_to_unmask).indices
                x[0, block_start + top_idx] = pred[top_idx]
        # captured.hidden has shape (1, seq_len, d). We want positions [prompt_len : prompt_len+128]
        hidden = captured["hidden"]
        gen_h = hidden[0, prompt_len:prompt_len + GEN_LENGTH].to(torch.bfloat16)  # (256, d)
        # Move to SAE device (auto-device may place residual on a different GPU)
        gen_h = gen_h.to(W_enc.device)
        # SAE encode for each token position
        h_norm = gen_h - b_dec
        pre = (h_norm @ W_enc.T + b_enc) if enc_op == "in_at_wt" else (h_norm @ W_enc + b_enc)
        # Only top-K activations matter; we keep all activations for our target features only
        # Build active mask first
        topk_vals, topk_idx = pre.topk(sae_k, dim=-1)
        topk_vals = topk_vals.relu()
        # For each target feature, find positions where it is in top-K and get value
        result_acts = {}
        for fid in TARGET_FEATURES:
            mask = (topk_idx == fid)  # (256, k)
            vals = (topk_vals * mask).sum(dim=-1)  # (256,) zero if not in topk
            result_acts[fid] = vals.float().cpu().numpy()
        # Final committed tokens (some may still be mask)
        committed = x[0, prompt_len:prompt_len + GEN_LENGTH].cpu().tolist()
        return result_acts, committed

    # Iterate all samples
    n_iter = min(n_total, len(instances))
    for sid in range(n_iter):
        try:
            acts, tokens = run_one_sample(sid)
        except Exception as e:
            print(f"  [{sid}] error: {e}")
            continue
        # For each target feature, push to top-K heap
        for fid in TARGET_FEATURES:
            arr = acts[fid]
            # Find top-3 positions per sample to seed
            pos_sorted = np.argsort(arr)[::-1][:3]
            for p in pos_sorted:
                if arr[p] <= 0:
                    continue
                tok_id = tokens[p] if p < len(tokens) else MASK_ID
                tok_str = tokenizer.decode([tok_id]) if tok_id != MASK_ID else "[MASK]"
                lo = max(0, p - CONTEXT_TOKENS)
                hi = min(len(tokens), p + CONTEXT_TOKENS + 1)
                ctx_ids = [t for t in tokens[lo:hi] if t != MASK_ID]
                ctx = tokenizer.decode(ctx_ids).replace("\n", "\\n")
                entry = {
                    "act": float(arr[p]),
                    "sample_id": sid,
                    "task_id": instances[sid]["task_id"],
                    "pos": int(p),
                    "label": int(labels[sid]),
                    "token": tok_str,
                    "context": ctx[:280],
                }
                top_per_feat[fid].append(entry)
        if (sid + 1) % 20 == 0:
            print(f"  done {sid+1}/{n_iter}")
            # Keep top-K so far to limit memory
            for fid in TARGET_FEATURES:
                top_per_feat[fid] = sorted(top_per_feat[fid], key=lambda e: -e["act"])[:TOP_K_EXAMPLES * 3]

    # Final top-K per feature
    for fid in TARGET_FEATURES:
        top_per_feat[fid] = sorted(top_per_feat[fid], key=lambda e: -e["act"])[:TOP_K_EXAMPLES]
        print(f"\nFeature f{fid}: top examples")
        for i, e in enumerate(top_per_feat[fid][:10]):
            print(f"  [{i}] act={e['act']:.2f} sid={e['sample_id']} tok='{e['token']}' label={e['label']}")
            print(f"      ctx: {e['context'][:140]}")

    results = {
        "sae_layer": SAE_LAYER,
        "target_step": TARGET_STEP,
        "target_features": TARGET_FEATURES,
        "top_per_feature": {str(k): v for k, v in top_per_feat.items()},
    }
    out_path = "/results/mbpp_llada/actmax_features.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved to {out_path}")
    return json.dumps({"n_samples": n_iter, "features": list(TARGET_FEATURES)})


@app.local_entrypoint()
def main():
    print(run_actmax.remote())
