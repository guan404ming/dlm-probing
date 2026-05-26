"""Joint multi-layer SAE steering on LLaDA-MBPP (EMNLP rebuttal).

Reviewer ask: a single-layer linear intervention may miss a multi-layer write.
Here we suppress each layer's top-1 fail-enriched feature SIMULTANEOUSLY at
L11/L16/L26/L30 (hooks registered jointly), so the perturbation spans the
layer range over which correctness information is distributed (Figure 6).

  L11 f13087, L16 f2741, L26 f15601, L30 f8020  (top-1 fail @ step 64)

If joint suppression still flips 0 outcomes, the single-layer null is not an
artifact of intervening at one layer.

Usage:
  .venv/bin/modal run src/applications/sae/modal_joint_steer.py \\
    --n-fail-c1 30 --n-fail-c0 10 --n-pass 20 --alpha 5.0 --steer-from-step 64
"""

import modal

app = modal.App("sae-joint-steer")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install("torch>=2.0", "transformers==4.52.2", "accelerate>=0.30",
                 "numpy", "datasets==2.21.0", "huggingface_hub")
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
# (layer, top-1 fail feature at step 64)
LAYER_FEATS = [(11, 13087), (16, 2741), (26, 15601), (30, 8020)]


@app.function(image=image, gpu="A100", timeout=14400,
              volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL})
def run_joint(n_fail_c1: int, n_fail_c0: int, n_pass: int,
              alpha: float, steer_from_step: int):
    import json, os, re, signal, time
    import numpy as np, torch
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

    # load all SAEs
    sae_by_layer = {}
    for layer, feat in LAYER_FEATS:
        sp = f"resid_post_layer_{layer}/trainer_{SAE_TRAINER}"
        ae = hf_hub_download(repo_id=SAE_REPO, filename=f"{sp}/ae.pt", cache_dir="/hf-cache")
        cfg = hf_hub_download(repo_id=SAE_REPO, filename=f"{sp}/config.json", cache_dir="/hf-cache")
        k = json.load(open(cfg))["trainer"]["k"]
        st = torch.load(ae, map_location="cpu", weights_only=True)
        bdec = st.get("b_dec", st.get("decoder.bias"))
        sae_by_layer[layer] = {
            "W_enc": st["encoder.weight"].cuda().to(torch.bfloat16),
            "b_enc": st["encoder.bias"].cuda().to(torch.bfloat16),
            "W_dec": st["decoder.weight"].cuda().to(torch.bfloat16),
            "b_dec": bdec.cuda().to(torch.bfloat16),
            "k": k, "feat": feat,
        }
        print(f"L{layer} SAE f{feat} loaded (k={k})")

    tokenizer = AutoTokenizer.from_pretrained(LLADA_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(LLADA_NAME, device_map="auto",
                                      torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])
    fail_set = set(i for ids in clusters.values() for i in ids)
    pass_pool = [i for i in range(len(instances)) if i not in fail_set]
    rng = np.random.RandomState(42)
    pass_idxs = [int(i) for i in rng.choice(pass_pool, size=n_pass, replace=False)]

    state_box = {"step": -1, "on": False, "mods": 0}

    def make_hook(layer):
        p = sae_by_layer[layer]
        tdec = p["W_dec"][:, p["feat"]].clone()
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if not state_box["on"] or state_box["step"] < steer_from_step:
                return output
            x = h - p["b_dec"]
            pre = torch.nn.functional.linear(x, p["W_enc"], p["b_enc"])
            tv, ti = pre.topk(p["k"], dim=-1)
            tv = tv.relu()
            a = (tv * (ti == p["feat"])).sum(dim=-1, keepdim=True)
            h.sub_(alpha * a * tdec.view(1, 1, -1))
            state_box["mods"] += 1
            return output
        return hook

    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        layers = model.model.transformer.blocks
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        layers = model.transformer.blocks
    else:
        layers = model.transformer.h
    handles = [layers[L].register_forward_hook(make_hook(L)) for L, _ in LAYER_FEATS]

    def check_mbpp(inst, txt):
        try:
            m = re.search(r"```python\s*(.*?)```", txt, re.DOTALL)
            code = m.group(1) if m else txt
            full = code + "\n" + "\n".join(inst["test_imports"]) + "\n" + "\n".join(inst["test_list"])
            old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(10)
            try:
                exec(full, {})
                return True
            finally:
                signal.alarm(0); signal.signal(signal.SIGALRM, old)
        except Exception:
            return False

    def generate(prompt_ids, prompt_len, steer_on):
        full = torch.full((1, prompt_len + GEN_LENGTH), MASK_ID,
                          dtype=prompt_ids.dtype, device=prompt_ids.device)
        full[0, :prompt_len] = prompt_ids[0]
        state_box["on"] = steer_on; state_box["step"] = -1
        x = full.clone()
        n_blocks = GEN_LENGTH // BLOCK_LENGTH
        spb = STEPS // n_blocks
        gstep = 0
        for b in range(n_blocks):
            bs = prompt_len + b * BLOCK_LENGTH; be = bs + BLOCK_LENGTH
            for inner in range(spb):
                state_box["step"] = gstep
                with torch.no_grad():
                    out = model(x)
                bm = (x[0, bs:be] == MASK_ID)
                if not bm.any():
                    gstep += 1; continue
                bl = out.logits[0, bs:be]
                probs = torch.softmax(bl / max(TEMPERATURE, 1e-6), dim=-1)
                if TEMPERATURE == 0:
                    conf, pred = probs.max(dim=-1)
                else:
                    conf, _ = probs.max(dim=-1)
                    pred = torch.multinomial(probs, 1).squeeze(-1)
                conf = conf.masked_fill(~bm, -1.0)
                ntu = max(1, int(bm.sum().item() / max(1, (spb - inner))))
                ti = conf.topk(ntu).indices
                x[0, bs + ti] = pred[ti]
                gstep += 1
        gen = [t for t in x[0, prompt_len:].tolist() if t != MASK_ID]
        return tokenizer.decode(gen)

    def make_prompt(inst):
        msgs = [{"role": "user",
                 "content": f"Write a Python function. Only output code in a Python block.\n\nProblem: {inst['prompt']}"}]
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tokenizer(text, return_tensors="pt").input_ids.cuda()

    def run_label(label, idxs):
        rows = []
        for k, idx in enumerate(idxs):
            inst = instances[idx]
            pids = make_prompt(inst); pl = pids.shape[1]
            t0 = time.time()
            bp = check_mbpp(inst, generate(pids, pl, False))
            sp = check_mbpp(inst, generate(pids, pl, True))
            rows.append({"idx": idx, "task_id": inst["task_id"],
                         "base_pass": int(bp), "steer_pass": int(sp)})
            if (k + 1) % 10 == 0 or k < 2:
                print(f"  [{label} {k+1}/{len(idxs)}] base={bp} steer={sp} ({time.time()-t0:.1f}s)")
        return rows

    print(f"\n=== JOINT suppress {LAYER_FEATS} alpha={alpha} from s{steer_from_step} ===")
    rows_c1 = run_label("c1", fail_c1_idxs)
    rows_c0 = run_label("c0", fail_c0_idxs)
    rows_pass = run_label("pass", pass_idxs)

    def summ(rows, label):
        if not rows: return None
        return {"label": label, "n": len(rows),
                "fail_to_pass": sum(1 for r in rows if not r["base_pass"] and r["steer_pass"]),
                "pass_to_fail": sum(1 for r in rows if r["base_pass"] and not r["steer_pass"])}
    summary = [summ(rows_c1, "fail_c1"), summ(rows_c0, "fail_c0"), summ(rows_pass, "pass")]
    print("\nJOINT summary:")
    for s in summary:
        if s: print(f"  {s['label']}: n={s['n']} f->p={s['fail_to_pass']} p->f={s['pass_to_fail']}")

    out = {"config": {"layer_feats": LAYER_FEATS, "alpha": alpha,
                      "steer_from_step": steer_from_step, "n_fail_c1": n_fail_c1,
                      "n_fail_c0": n_fail_c0, "n_pass": n_pass},
           "summary": summary, "rows": rows_c1 + rows_c0 + rows_pass}
    with open(f"/results/mbpp_llada/sae_joint_steer_a{alpha}_s{steer_from_step}.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    for h in handles:
        h.remove()
    return json.dumps({"summary": summary})


@app.local_entrypoint()
def main(n_fail_c1: int = 30, n_fail_c0: int = 10, n_pass: int = 20,
         alpha: float = 5.0, steer_from_step: int = 64):
    print(run_joint.remote(n_fail_c1, n_fail_c0, n_pass, alpha, steer_from_step))
