"""Small-n steering protocol sweep for LLaDA-MBPP (EMNLP expansion).

Probe many intervention configurations on a small fixed n to see whether
any protocol flips correctness; addresses the open question raised by
DLM-Scope's positive steering demos that our previous null might be an
artifact of one too-weak protocol.

Each config keeps the same fail-cluster sample and pass control sample
set so flips are directly comparable. Per config we record fail_c1
fail->pass and pass->fail counts and a few example generations.

Configurations spanned (all on LLaDA-8B / MBPP, SAE layer 26):
  - alpha sweep on f15601 at s64: alpha in {1, 5, 10, 20, 50}
  - start-step sweep on f15601 at alpha=5: s in {0, 4, 32, 48, 64}
  - reverse alpha sweep at s64: alpha in {-5, -10, -20}
  - top-N sweep at s64, alpha=5: N in {1, 5, 10, 20}
  - replacement: subtract top-K fail + add top-K pass at s64, K in {1, 5, 20}

Output: /results/mbpp_llada/steer_sweep_summary.json
"""

import modal

app = modal.App("sae-steer-sweep")

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

# Small n per condition: enough to spot any nonzero flip rate
N_FAIL_C1 = 15
N_PASS = 5

# Configs to run. Each tuple: (name, target_features, pass_features, alpha, steer_from_step)
# pass_features != None enables replacement mode (subtract fail + add pass).
CONFIGS = [
    # alpha sweep on f15601 at s64
    ("alpha_1_s64",  [15601], None, 1.0,  64),
    ("alpha_5_s64",  [15601], None, 5.0,  64),
    ("alpha_10_s64", [15601], None, 10.0, 64),
    ("alpha_20_s64", [15601], None, 20.0, 64),
    ("alpha_50_s64", [15601], None, 50.0, 64),
    # start-step sweep at alpha=5
    ("alpha_5_s0",   [15601], None, 5.0,  0),
    ("alpha_5_s4",   [15601], None, 5.0,  4),
    ("alpha_5_s32",  [15601], None, 5.0,  32),
    ("alpha_5_s48",  [15601], None, 5.0,  48),
    # reverse (amplify f15601)
    ("alpha_-5_s64",  [15601], None, -5.0,  64),
    ("alpha_-10_s64", [15601], None, -10.0, 64),
    ("alpha_-20_s64", [15601], None, -20.0, 64),
    # top-N sweep at alpha=5 s64
    ("top5_a5_s64",  [15601, 8825, 2087, 11404, 9657], None, 5.0, 64),
    ("top10_a5_s64", [15601, 8825, 2087, 11404, 9657, 7444, 8200, 12305, 4456, 14502], None, 5.0, 64),
    # replacement at s64: top-K fail OUT, top-K pass IN
    ("replace_k1",  [15601], [7961], 5.0, 64),
    ("replace_k5",  [15601, 8825, 2087, 11404, 9657], [7961, 9732, 3011, 4420, 11861], 5.0, 64),
    ("replace_k1_strong", [15601], [7961], 20.0, 64),
]


@app.function(
    image=image, gpu="A100", timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_sweep():
    import json, os, re, signal, time
    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModel

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    # Load fail cluster sample IDs
    with open("/results/mbpp_llada/sae_diagnose_stage2.json") as f:
        diag = json.load(f)
    clusters = {c["cluster"]: c["fail_sample_indices"] for c in diag["clusters"]}
    fail_c1_idxs = clusters[1][:N_FAIL_C1]
    fail_set = set([i for ids in clusters.values() for i in ids])

    # SAE
    sae_path = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
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

    tokenizer = AutoTokenizer.from_pretrained(LLADA_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLADA_NAME, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])
    pass_pool = [i for i in range(len(instances)) if i not in fail_set]
    rng = np.random.RandomState(42)
    pass_idxs = list(rng.choice(pass_pool, size=N_PASS, replace=False))

    # Hook holder
    state_box = {
        "step": -1, "enabled": False,
        "fail_feats": [], "pass_feats": [],
        "alpha": 0.0, "steer_from": 64,
        "fail_dec": None, "pass_dec": None,
        "max_dh": 0.0,
    }

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if not state_box["enabled"]:
            return output
        if state_box["step"] < state_box["steer_from"]:
            return output
        x = h - b_dec
        pre = torch.nn.functional.linear(x, W_enc, b_enc)
        topk_vals, topk_idx = pre.topk(sae_k, dim=-1)
        topk_vals = topk_vals.relu()
        delta = torch.zeros_like(h)
        a = state_box["alpha"]
        for ti, fid in enumerate(state_box["fail_feats"]):
            mask = (topk_idx == fid)
            at = (topk_vals * mask).sum(dim=-1, keepdim=True)
            delta = delta + a * at * state_box["fail_dec"][:, ti].view(1, 1, -1)
        for ti, pid in enumerate(state_box["pass_feats"]):
            mask = (topk_idx == pid)
            at = (topk_vals * mask).sum(dim=-1, keepdim=True)
            delta = delta - a * at * state_box["pass_dec"][:, ti].view(1, 1, -1)
        h.sub_(delta)
        dn = float(delta.norm().item())
        if dn > state_box["max_dh"]:
            state_box["max_dh"] = dn
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

    # Baseline generations first (no steering), shared across configs
    baseline = {}
    print(f"\n=== Computing baseline (n_fail={N_FAIL_C1}, n_pass={N_PASS}) ===")
    state_box["enabled"] = False
    for idx in fail_c1_idxs + pass_idxs:
        inst = instances[idx]
        pids = make_prompt(inst); pl = pids.shape[1]
        txt = generate(pids, pl, False)
        baseline[idx] = (check_mbpp(inst, txt), txt[:200])

    results = []
    for cname, ff, pf, alpha, sfrom in CONFIGS:
        print(f"\n=== Config: {cname}  ff={ff} pf={pf} a={alpha} s{sfrom} ===")
        state_box["fail_feats"] = ff
        state_box["pass_feats"] = pf if pf else []
        state_box["alpha"] = alpha
        state_box["steer_from"] = sfrom
        state_box["fail_dec"] = W_dec[:, ff].clone() if ff else None
        state_box["pass_dec"] = W_dec[:, pf].clone() if pf else None
        state_box["max_dh"] = 0.0

        rows = []
        for label, idxs in [("c1", fail_c1_idxs), ("pass", pass_idxs)]:
            for idx in idxs:
                inst = instances[idx]
                pids = make_prompt(inst); pl = pids.shape[1]
                t0 = time.time()
                txt = generate(pids, pl, True)
                sp = check_mbpp(inst, txt)
                bp = baseline[idx][0]
                rows.append({"label": label, "idx": idx, "task_id": inst["task_id"],
                             "base_pass": int(bp), "steer_pass": int(sp),
                             "dt": round(time.time() - t0, 1)})
        c1_rows = [r for r in rows if r["label"] == "c1"]
        pa_rows = [r for r in rows if r["label"] == "pass"]
        f2p_c1 = sum(1 for r in c1_rows if not r["base_pass"] and r["steer_pass"])
        p2f = sum(1 for r in pa_rows if r["base_pass"] and not r["steer_pass"])
        summary = {
            "config": cname,
            "fail_feats": ff, "pass_feats": pf, "alpha": alpha, "steer_from": sfrom,
            "n_c1": len(c1_rows), "f2p_c1": f2p_c1,
            "n_pass": len(pa_rows), "p2f": p2f,
            "max_delta_h": round(state_box["max_dh"], 1),
        }
        print(f"  {cname}: c1 f→p={f2p_c1}/{len(c1_rows)}  pass p→f={p2f}/{len(pa_rows)}  max_dh={state_box['max_dh']:.1f}")
        results.append({"summary": summary, "rows": rows})

    out_path = "/results/mbpp_llada/steer_sweep_summary.json"
    with open(out_path, "w") as f:
        json.dump({"baseline_pass": {str(k): bool(v[0]) for k, v in baseline.items()},
                   "configs": results}, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")

    print("\n" + "=" * 80)
    print(f"{'config':28} {'f2p_c1':>8} {'p2f_pass':>9} {'max_dh':>9}")
    print("-" * 80)
    for r in results:
        s = r["summary"]
        print(f"{s['config']:28} {s['f2p_c1']:>3}/{s['n_c1']:<4} {s['p2f']:>3}/{s['n_pass']:<5} {s['max_delta_h']:>9}")

    hh.remove()
    return json.dumps([r["summary"] for r in results], indent=2)


@app.local_entrypoint()
def main():
    print(run_sweep.remote())
