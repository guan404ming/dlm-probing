"""Multi-layer SAE steering on LLaDA-MBPP (EMNLP expansion).

Re-runs the same suppression protocol as modal_sae_steer.py at SAE layers
{11, 16, 30}, using each layer's top-1 fail-enriched feature identified by
modal_crosslayer.py:

  L11 top-1: f13087 (step 64)
  L16 top-1: f2741  (step 64)
  L30 top-1: f8020  (step 64)

If steering at any non-L26 layer flips correctness while L26 does not, the
"single-layer linear steering is the wrong test" critique gains weight. If
all layers null, the cross-layer diagnose result generalises to causal
weakness across the layer span.

Usage:
  .venv/bin/modal run src/applications/sae/modal_multilayer_steer.py \\
    --sae-layer 11 --target-feature 13087 --n-fail-c1 30 --n-fail-c0 10 \\
    --n-pass 20 --alpha 5.0 --steer-from-step 64
"""

import modal

app = modal.App("sae-multilayer-steer")

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
SAE_TRAINER = 2


@app.function(
    image=image, gpu="A100", timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_steer(
    sae_layer: int,
    target_feature: int,
    n_fail_c1: int,
    n_fail_c0: int,
    n_pass: int,
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
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModel

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    with open("/results/mbpp_llada/sae_diagnose_stage2.json") as f:
        diag = json.load(f)
    clusters = {c["cluster"]: c["fail_sample_indices"] for c in diag["clusters"]}
    fail_c1_idxs = clusters[1][:n_fail_c1]
    fail_c0_idxs = clusters[0][:n_fail_c0]

    sae_path = f"resid_post_layer_{sae_layer}/trainer_{SAE_TRAINER}"
    ae_local = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path}/ae.pt", cache_dir="/hf-cache")
    cfg_local = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path}/config.json", cache_dir="/hf-cache")
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
    target_dec = W_dec[:, target_feature].clone()
    print(f"L{sae_layer} SAE: f{target_feature}, dec_norm={target_dec.float().norm().item():.3f}")

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

    state_box = {"step": -1, "steer_enabled": False, "mod_count": 0, "max_delta_norm": 0.0}

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
        mask = (topk_idx == target_feature)
        a = (topk_vals * mask).sum(dim=-1, keepdim=True)
        delta = alpha * a * target_dec.view(1, 1, -1)
        h.sub_(delta)
        dn = float(delta.norm().item())
        if dn > state_box["max_delta_norm"]:
            state_box["max_delta_norm"] = dn
        state_box["mod_count"] += 1
        return output

    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        layers = model.model.transformer.blocks
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        layers = model.transformer.blocks
    else:
        layers = model.transformer.h
    hook_handle = layers[sae_layer].register_forward_hook(hook)

    def check_mbpp(inst, txt):
        try:
            m = re.search(r"```python\s*(.*?)```", txt, re.DOTALL)
            code = m.group(1) if m else txt
            full = code + "\n" + "\n".join(inst["test_imports"]) + "\n"
            full += "\n".join(inst["test_list"])
            old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(10)
            try:
                ns = {}
                exec(full, ns)
                return True
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
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
        x = full.clone()
        n_blocks = GEN_LENGTH // BLOCK_LENGTH
        steps_per_block = STEPS // n_blocks
        global_step = 0
        for b in range(n_blocks):
            bs = prompt_len + b * BLOCK_LENGTH
            be = bs + BLOCK_LENGTH
            for inner in range(steps_per_block):
                state_box["step"] = global_step
                with torch.no_grad():
                    out = model(x)
                logits = out.logits
                bm = (x[0, bs:be] == MASK_ID)
                if not bm.any():
                    global_step += 1
                    continue
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
                global_step += 1
        gen = x[0, prompt_len:].tolist()
        gen = [t for t in gen if t != MASK_ID]
        return tokenizer.decode(gen)

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
            pids = make_prompt(inst)
            pl = pids.shape[1]
            t0 = time.time()
            b = generate(pids, pl, steer_on=False)
            s = generate(pids, pl, steer_on=True)
            bp = check_mbpp(inst, b)
            sp = check_mbpp(inst, s)
            rows.append({"idx": idx, "task_id": inst["task_id"],
                         "base_pass": int(bp), "steer_pass": int(sp)})
            if (k + 1) % 10 == 0 or k < 2:
                print(f"  [{label} {k+1}/{len(idxs)}] base={bp} steer={sp} ({time.time()-t0:.1f}s)")
        return rows

    print(f"\n=== L{sae_layer}/f{target_feature} suppress alpha={alpha} from s{steer_from_step} ===")
    rows_c1 = run_label("c1", fail_c1_idxs)
    rows_c0 = run_label("c0", fail_c0_idxs)
    rows_pass = run_label("pass", pass_idxs)

    def summ(rows, label):
        if not rows: return None
        f2p = sum(1 for r in rows if not r["base_pass"] and r["steer_pass"])
        p2f = sum(1 for r in rows if r["base_pass"] and not r["steer_pass"])
        return {"label": label, "n": len(rows), "fail_to_pass": f2p, "pass_to_fail": p2f}

    summary = [summ(rows_c1, "fail_c1"), summ(rows_c0, "fail_c0"), summ(rows_pass, "pass")]
    print(f"\nSummary L{sae_layer}/f{target_feature}:")
    for s in summary:
        if s:
            print(f"  {s['label']}: n={s['n']} f→p={s['fail_to_pass']} p→f={s['pass_to_fail']}")

    out = {
        "config": {"sae_layer": sae_layer, "target_feature": target_feature,
                   "alpha": alpha, "steer_from_step": steer_from_step,
                   "n_fail_c1": n_fail_c1, "n_fail_c0": n_fail_c0, "n_pass": n_pass},
        "summary": summary,
        "rows": rows_c1 + rows_c0 + rows_pass,
    }
    out_path = f"/results/mbpp_llada/sae_steer_L{sae_layer}_f{target_feature}_a{alpha}_s{steer_from_step}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")

    hook_handle.remove()
    return json.dumps({"summary": summary})


@app.local_entrypoint()
def main(
    sae_layer: int = 11,
    target_feature: int = 13087,
    n_fail_c1: int = 30,
    n_fail_c0: int = 10,
    n_pass: int = 20,
    alpha: float = 5.0,
    steer_from_step: int = 64,
):
    print(run_steer.remote(
        sae_layer, target_feature, n_fail_c1, n_fail_c0, n_pass, alpha, steer_from_step,
    ))
