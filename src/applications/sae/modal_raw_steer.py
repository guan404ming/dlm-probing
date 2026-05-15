"""Raw-residual directional steering on LLaDA-MBPP (EMNLP expansion).

Tests Theory B: if the 5-15% correctness signal outside the sparse SAE
subspace is on the causal path, then steering along a *raw* residual
direction (mean-difference of fail vs pass residuals, or PCA-1 of the
residual covariance) should affect correctness, even if SAE steering
along sparse atoms does not.

The discriminating direction is computed at L26, step 64 from cached
plateau residuals (chunk_off*.npz). Two flavors:
  - meandiff: pass_mean - fail_mean direction, unit-normalized
  - pca1: top principal component of the centered residuals

The hook subtracts alpha * proj along the chosen direction at every step
>= steer_from_step, mirroring the SAE suppression protocol.

Usage:
  .venv/bin/modal run src/applications/sae/modal_raw_steer.py \\
    --direction-type meandiff --alpha 5.0 --steer-from-step 64 \\
    --n-fail-c1 15 --n-pass 5
"""

import modal

app = modal.App("sae-raw-steer")

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
SAE_LAYER = 26
N_REGIONS = 4


@app.function(
    image=image, gpu="A100", timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_raw_steer(
    direction_type: str,
    alpha: float,
    steer_from_step: int,
    n_fail_c1: int,
    n_pass: int,
    tag: str,
):
    import json
    import os
    import re
    import signal
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    with open("/results/mbpp_llada/sae_diagnose_stage2.json") as f:
        diag = json.load(f)
    clusters = {c["cluster"]: c["fail_sample_indices"] for c in diag["clusters"]}
    fail_c1_idxs = clusters[1][:n_fail_c1]

    # Load cached plateau residuals at step 64 to compute discriminating direction
    in_dir = "/results/mbpp_llada_dense"
    all_labels, region_feats = [], {r: [] for r in range(N_REGIONS)}
    for fn in sorted(os.listdir(in_dir)):
        if not fn.startswith("chunk_off") or not fn.endswith(".npz"):
            continue
        data = np.load(f"{in_dir}/{fn}")
        if "feat_s64_r0" not in data.files:
            continue
        all_labels.append(data["labels"])
        for r in range(N_REGIONS):
            region_feats[r].append(data[f"feat_s64_r{r}"])
    labels = np.concatenate(all_labels).astype(int)
    n_fail = int((labels == 0).sum())
    n_pass_dat = int(labels.sum())
    print(f"Loaded {len(labels)} samples; pass={n_pass_dat} fail={n_fail}")

    # Build per-sample L26 residual (mean over 4 regions)
    layer_feats = []
    for r in range(N_REGIONS):
        feats = np.concatenate(region_feats[r])  # (N, 33, d_in)
        layer_feats.append(feats[:, SAE_LAYER, :].astype(np.float32))
    residuals = np.mean(layer_feats, axis=0)  # (N, d_in)
    d_in = residuals.shape[1]
    print(f"residuals: {residuals.shape}")

    fail_res = residuals[labels == 0]
    pass_res = residuals[labels == 1]
    if direction_type == "meandiff":
        d = pass_res.mean(axis=0) - fail_res.mean(axis=0)  # pass minus fail
        d = d / (np.linalg.norm(d) + 1e-9)
        print(f"meandiff direction: norm pre-norm={np.linalg.norm(pass_res.mean(0) - fail_res.mean(0)):.2f}")
    elif direction_type == "pca1":
        # Top PC of fail-vs-pass discriminating projection
        # Center each class then take PC of (fail - pass) covariance
        centered = residuals - residuals.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        d = vt[0]  # top-1 PC
        # Orient toward fail (negative correlation with labels)
        proj = centered @ d
        if np.mean(proj[labels == 0]) < np.mean(proj[labels == 1]):
            d = -d  # ensure d points pass-ward
        print(f"pca1 direction: singular value s[0]={s[0]:.2f}")
    elif direction_type == "fisher":
        # Fisher LDA: (mu_p - mu_f)^T S_w^{-1}, simplified to (mu_p - mu_f)
        # If covariance is roughly isotropic, reduces to meandiff
        mu_diff = pass_res.mean(axis=0) - fail_res.mean(axis=0)
        Sw = np.cov(fail_res.T) + np.cov(pass_res.T)
        # Regularize and solve
        Sw_reg = Sw + 1e-2 * np.eye(d_in) * np.trace(Sw) / d_in
        d = np.linalg.solve(Sw_reg, mu_diff)
        d = d / (np.linalg.norm(d) + 1e-9)
    else:
        raise ValueError(f"Unknown direction_type: {direction_type}")

    d_torch = torch.from_numpy(d).cuda().to(torch.bfloat16)
    print(f"Direction {direction_type}: shape={d.shape}, mean={d.mean():.6f}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(LLADA_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLADA_NAME, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])
    fail_set = set([i for ids in clusters.values() for i in ids])
    pass_pool = [i for i in range(len(instances)) if i not in fail_set]
    rng = np.random.RandomState(42)
    pass_idxs = list(rng.choice(pass_pool, size=n_pass, replace=False))

    state_box = {"step": -1, "enabled": False, "max_dh": 0.0, "n_apply": 0}

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if not state_box["enabled"]:
            return output
        if state_box["step"] < steer_from_step:
            return output
        # h: (1, seq, d). Project each token onto d, scale by alpha, subtract.
        proj = (h.to(torch.bfloat16) * d_torch.view(1, 1, -1)).sum(dim=-1, keepdim=True)
        delta = alpha * proj * d_torch.view(1, 1, -1)
        h.sub_(delta.to(h.dtype))
        dn = float(delta.norm().item())
        if dn > state_box["max_dh"]:
            state_box["max_dh"] = dn
        state_box["n_apply"] += 1
        return output

    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        layers = model.model.transformer.blocks
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        layers = model.transformer.blocks
    else:
        layers = model.transformer.h
    hh = layers[SAE_LAYER].register_forward_hook(hook)

    def check_mbpp(inst, txt):
        try:
            m = re.search(r"```python\s*(.*?)```", txt, re.DOTALL)
            code = m.group(1) if m else txt
            full = code + "\n" + "\n".join(inst["test_imports"]) + "\n"
            full += "\n".join(inst["test_list"])
            old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(10)
            try:
                ns = {}; exec(full, ns); return True
            finally:
                signal.alarm(0); signal.signal(signal.SIGALRM, old)
        except Exception:
            return False

    def make_prompt(inst):
        msgs = [{"role": "user", "content": f"Write a Python function. Only output code in a Python block.\n\nProblem: {inst['prompt']}"}]
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        return ids

    def generate(prompt_ids, prompt_len, steer_on):
        full = torch.full(
            (1, prompt_len + GEN_LENGTH), MASK_ID, dtype=prompt_ids.dtype, device=prompt_ids.device,
        )
        full[0, :prompt_len] = prompt_ids[0]
        state_box["enabled"] = steer_on
        state_box["step"] = -1
        x = full.clone()
        n_blocks = GEN_LENGTH // BLOCK_LENGTH
        steps_per_block = STEPS // n_blocks
        gs = 0
        for b in range(n_blocks):
            bs = prompt_len + b * BLOCK_LENGTH
            be = bs + BLOCK_LENGTH
            for inner in range(steps_per_block):
                state_box["step"] = gs
                with torch.no_grad():
                    out = model(x)
                logits = out.logits
                bm = (x[0, bs:be] == MASK_ID)
                if not bm.any():
                    gs += 1; continue
                bl = logits[0, bs:be]
                if TEMPERATURE == 0:
                    probs = torch.softmax(bl, dim=-1)
                    conf, pred = probs.max(dim=-1)
                else:
                    probs = torch.softmax(bl / TEMPERATURE, dim=-1)
                    conf, _ = probs.max(dim=-1)
                    pred = torch.multinomial(probs, 1).squeeze(-1)
                conf = conf.masked_fill(~bm, -1.0)
                ntu = max(1, int(bm.sum().item() / max(1, (steps_per_block - inner))))
                ti = conf.topk(ntu).indices
                x[0, bs + ti] = pred[ti]
                gs += 1
        gen = x[0, prompt_len:].tolist()
        gen = [t for t in gen if t != MASK_ID]
        return tokenizer.decode(gen)

    def run_label(label, idxs):
        rows = []
        for k, idx in enumerate(idxs):
            inst = instances[idx]
            pids = make_prompt(inst); pl = pids.shape[1]
            t0 = time.time()
            b = generate(pids, pl, steer_on=False)
            s = generate(pids, pl, steer_on=True)
            bp = check_mbpp(inst, b); sp = check_mbpp(inst, s)
            rows.append({"idx": idx, "task_id": inst["task_id"],
                         "base_pass": int(bp), "steer_pass": int(sp)})
            if (k+1) % 5 == 0 or k < 2:
                print(f"  [{label} {k+1}/{len(idxs)}] base={bp} steer={sp} ({time.time()-t0:.1f}s)")
        return rows

    print(f"\n=== {tag}: direction={direction_type}, alpha={alpha}, from s{steer_from_step} ===")
    rows_c1 = run_label("c1", fail_c1_idxs)
    rows_pass = run_label("pass", pass_idxs)

    f2p = sum(1 for r in rows_c1 if not r["base_pass"] and r["steer_pass"])
    p2f = sum(1 for r in rows_pass if r["base_pass"] and not r["steer_pass"])
    print(f"\nSummary {tag}: c1 f→p={f2p}/{len(rows_c1)}  pass p→f={p2f}/{len(rows_pass)}  max_dh={state_box['max_dh']:.1f}")

    out = {
        "config": {"direction_type": direction_type, "alpha": alpha,
                   "steer_from_step": steer_from_step, "tag": tag,
                   "sae_layer": SAE_LAYER, "n_c1": len(rows_c1), "n_pass": len(rows_pass)},
        "summary": {"f2p_c1": f2p, "n_c1": len(rows_c1), "p2f": p2f, "n_pass": len(rows_pass),
                    "max_delta_h": float(state_box["max_dh"])},
        "rows": rows_c1 + rows_pass,
    }
    out_path = f"/results/mbpp_llada/sae_rawsteer_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")

    hh.remove()
    return json.dumps(out["summary"])


@app.local_entrypoint()
def main(
    direction_type: str = "meandiff",
    alpha: float = 5.0,
    steer_from_step: int = 64,
    n_fail_c1: int = 15,
    n_pass: int = 5,
    tag: str = "default",
):
    print(run_raw_steer.remote(
        direction_type, alpha, steer_from_step, n_fail_c1, n_pass, tag,
    ))
