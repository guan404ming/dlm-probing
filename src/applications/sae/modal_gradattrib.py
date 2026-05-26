"""Gradient-based attribution of SAE features to the output (reviewer #5 alt).

Complements the steering null: even if suppressing f15601 does not flip outcomes,
is it on a salient causal path from the L26 residual to the model's output? At the
plateau (step 64) partially-denoised state we backprop the model's self-confidence
(sum log p of its argmax over masked generation positions) to the L26 residual,
then score each SAE feature by gradient*activation in feature space
(attr_i = (grad_h . W_dec[:,i]) * z_i). We report where the diagnostic features
f15601 / f3892 / f11265 rank among all active features.

Output: /results/mbpp_llada/gradattrib_results.json

Usage:
  .venv/bin/modal run src/applications/sae/modal_gradattrib.py --n-fail 20
"""

import modal

app = modal.App("sae-gradattrib")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "curl", "build-essential")
         .pip_install("torch>=2.0", "transformers==4.52.2", "accelerate>=0.30",
                      "numpy", "datasets==2.21.0", "huggingface_hub"))
RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

LLADA = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336
TEMP = 0.2
GEN_LENGTH = 256
STEPS = 128
BLOCK = 32
SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYER = 26
SAE_TRAINER = 2
STEER_FROM = 64
TARGETS = [15601, 3892, 11265]


@app.function(image=image, gpu="A100", timeout=14400,
              volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL})
def run(n_fail: int = 20):
    import json, os, numpy as np, torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModel
    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    with open("/results/mbpp_llada/sae_diagnose_stage2.json") as f:
        diag = json.load(f)
    clusters = {c["cluster"]: c["fail_sample_indices"] for c in diag["clusters"]}
    fail_idxs = (clusters.get(1, []) + clusters.get(0, []))[:n_fail]

    sp = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
    ae = hf_hub_download(repo_id=SAE_REPO, filename=f"{sp}/ae.pt", cache_dir="/hf-cache")
    cfg = hf_hub_download(repo_id=SAE_REPO, filename=f"{sp}/config.json", cache_dir="/hf-cache")
    k = json.load(open(cfg))["trainer"]["k"]
    st = torch.load(ae, map_location="cpu", weights_only=True)
    W_enc = st["encoder.weight"].cuda().float()
    b_enc = st["encoder.bias"].cuda().float()
    W_dec = st["decoder.weight"].cuda().float()  # (d_in, d_sae)
    b_dec = st.get("b_dec", st.get("decoder.bias")).cuda().float()

    tok = AutoTokenizer.from_pretrained(LLADA, trust_remote_code=True)
    model = AutoModel.from_pretrained(LLADA, device_map="auto", torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).eval()
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        layers = model.model.transformer.blocks
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        layers = model.transformer.blocks
    else:
        layers = model.transformer.h
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])

    captured = {"h": None, "leaf": False}
    def fhook(m, a, o):
        if not captured["leaf"]:
            return o
        h = o[0] if isinstance(o, tuple) else o
        h2 = h.detach().requires_grad_(True)  # cut graph, insert leaf at L26
        captured["h"] = h2
        return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
    handle = layers[SAE_LAYER].register_forward_hook(fhook)

    def denoise_to(prompt_ids, pl, stop_step):
        full = torch.full((1, pl + GEN_LENGTH), MASK_ID, dtype=prompt_ids.dtype, device=prompt_ids.device)
        full[0, :pl] = prompt_ids[0]
        x = full.clone()
        nb = GEN_LENGTH // BLOCK; spb = STEPS // nb; g = 0
        for b in range(nb):
            bs = pl + b * BLOCK; be = bs + BLOCK
            for inner in range(spb):
                if g >= stop_step:
                    return x
                with torch.no_grad():
                    out = model(x)
                bm = (x[0, bs:be] == MASK_ID)
                if bm.any():
                    bl = out.logits[0, bs:be]
                    probs = torch.softmax(bl / max(TEMP, 1e-6), dim=-1)
                    conf, _ = probs.max(dim=-1)
                    pred = torch.multinomial(probs, 1).squeeze(-1)
                    conf = conf.masked_fill(~bm, -1.0)
                    ntu = max(1, int(bm.sum().item() / max(1, (spb - inner))))
                    ti = conf.topk(ntu).indices
                    x[0, bs + ti] = pred[ti]
                g += 1
        return x

    d_sae = W_enc.shape[0]
    attr_sum = np.zeros(d_sae, np.float64)
    z_sum = np.zeros(d_sae, np.float64)
    n_used = 0
    for idx in fail_idxs:
        inst = instances[idx]
        text = tok.apply_chat_template(
            [{"role": "user", "content": f"Write a Python function. Only output code in a Python block.\n\nProblem: {inst['prompt']}"}],
            add_generation_prompt=True, tokenize=False)
        pids = tok(text, return_tensors="pt").input_ids.cuda(); pl = pids.shape[1]
        x = denoise_to(pids, pl, STEER_FROM)
        gen_lo = pl
        model.zero_grad(set_to_none=True)
        captured["leaf"] = True
        with torch.enable_grad():
            out = model(x)  # fhook inserts a leaf at L26
            logits = out.logits[0, gen_lo:]  # (GEN, V)
            logp = torch.log_softmax(logits.float(), dim=-1)
            obj = logp.max(dim=-1).values.sum()  # self-confidence
            obj.backward()
        captured["leaf"] = False
        h = captured["h"][0]            # (seq, d_in)
        gh = captured["h"].grad[0]      # (seq, d_in)
        hg = h[gen_lo:].mean(0).float() # region-mean residual
        grad = gh[gen_lo:].mean(0).float()
        # feature activations on region-mean
        z = torch.relu((hg - b_dec) @ W_enc.T + b_enc)
        tv, ti = z.topk(k)
        zf = torch.zeros(d_sae, device="cuda"); zf[ti] = tv
        # attribution: (grad . W_dec[:,i]) * z_i
        gd = grad @ W_dec  # (d_sae,)
        attr = (gd * zf).detach().cpu().numpy()
        attr_sum += np.abs(attr); z_sum += zf.detach().cpu().numpy()
        n_used += 1
        print(f"  idx {idx}: top attr feature f{int(np.abs(attr).argmax())}")
    handle.remove()

    rank = np.argsort(attr_sum)[::-1]
    rank_of = {int(t): int(np.where(rank == t)[0][0]) for t in TARGETS}
    out = {"n_fail": n_used, "d_sae": int(d_sae), "step": STEER_FROM,
           "top20_features": [int(r) for r in rank[:20]],
           "target_ranks": rank_of,
           "target_attr": {int(t): float(attr_sum[t]) for t in TARGETS},
           "max_attr": float(attr_sum.max()),
           "active_feature_count": int((z_sum > 0).sum())}
    with open("/results/mbpp_llada/gradattrib_results.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\ntarget ranks (of {int((z_sum>0).sum())} active): {rank_of}")
    print(f"top-20 attribution features: {out['top20_features']}")
    return json.dumps(out)


@app.local_entrypoint()
def main(n_fail: int = 20):
    print(run.remote(n_fail))
