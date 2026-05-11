"""Effect-size diagnostics for SAE steering interventions on LLaDA-MBPP.

Lightweight runner that re-uses the modal_sae_steer hook but records, per
instance: mean and max a_target activation across steps, mean and max
||Delta h||, and mean ||Delta h|| / ||h|| ratio. Saves per-instance rows so
reviewers can confirm that the intended manipulation truly perturbed the
SAE-layer residual rather than firing trivially.

Five conditions matching Table 1 (Steering intervention summary):
  - suppress f15601 at step 64 (alpha=+5)
  - suppress f15601 at step 16 (alpha=+5)
  - suppress top-5 fail features at step 64 (alpha=+5)
  - reverse f15601 at step 64 (alpha=-5; amplify)
  - baseline (no hook); recorded for reference only

Output: /results/mbpp_llada/steer_effect_size.json
"""

import modal

app = modal.App("sae-steer-effect")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0", "transformers==4.52.2", "accelerate>=0.30",
        "numpy", "datasets==2.21.0", "huggingface_hub",
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

# Top-5 fail-enriched features at LLaDA-MBPP step 64 (per Stage 2 diagnose).
TOP5_FEATURES = [15601, 3892, 13085, 11265, 2144]
TARGET_F = 15601

CONDITIONS = [
    {"name": "suppress_f15601_s64", "features": [TARGET_F], "alpha": 5.0, "step": 64},
    {"name": "suppress_f15601_s16", "features": [TARGET_F], "alpha": 5.0, "step": 16},
    {"name": "suppress_top5_s64", "features": TOP5_FEATURES, "alpha": 5.0, "step": 64},
    {"name": "reverse_f15601_s64", "features": [TARGET_F], "alpha": -5.0, "step": 64},
]


@app.function(
    image=image, gpu="A100", timeout=7200,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_effect_size(n_fail: int = 8):
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

    # ---- Load Stage 2 clusters ----
    with open("/results/mbpp_llada/sae_diagnose_stage2.json") as f:
        diag = json.load(f)
    clusters = {c["cluster"]: c["fail_sample_indices"] for c in diag["clusters"]}
    print(f"Cluster 0 size={len(clusters[0])}, cluster 1 size={len(clusters[1])}")
    fail_idxs = clusters[1][:n_fail]
    print(f"Using {len(fail_idxs)} cluster-1 fails: {fail_idxs}")

    # ---- Load SAE ----
    sae_path = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
    ae_local = hf_hub_download(
        repo_id=SAE_REPO, filename=f"{sae_path}/ae.pt", cache_dir="/hf-cache",
    )
    cfg_local = hf_hub_download(
        repo_id=SAE_REPO, filename=f"{sae_path}/config.json", cache_dir="/hf-cache",
    )
    with open(cfg_local) as f:
        sae_cfg = json.load(f)
    sae_k = sae_cfg["trainer"]["k"]
    state = torch.load(ae_local, map_location="cpu", weights_only=True)
    W_enc = state["encoder.weight"].cuda().to(torch.bfloat16)
    b_enc = state["encoder.bias"].cuda().to(torch.bfloat16)
    W_dec = state["decoder.weight"].cuda().to(torch.bfloat16)
    b_dec = state["b_dec"].cuda().to(torch.bfloat16)

    # ---- Load model + dataset ----
    tokenizer = AutoTokenizer.from_pretrained(LLADA_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLADA_NAME, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])

    # ---- Hook with per-step accumulators ----
    diag_box = {
        "step": -1, "enabled": False, "from_step": 64,
        "alpha": 5.0, "targets": [TARGET_F],
        "a_target_vals": [],  # one per hook fire (within enabled window)
        "delta_norms": [],
        "h_norms": [],
        "fire_count": 0,
        "modify_count": 0,
    }

    def hook(module, args, output):
        diag_box["fire_count"] += 1
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if not diag_box["enabled"]:
            return output
        if diag_box["step"] < diag_box["from_step"]:
            return output
        x = h - b_dec
        pre = torch.nn.functional.linear(x, W_enc, b_enc)
        topk_vals, topk_idx = pre.topk(sae_k, dim=-1)
        topk_vals = topk_vals.relu()
        delta_total = torch.zeros_like(h)
        a_target_max = 0.0
        for ti, fid in enumerate(diag_box["targets"]):
            mask = (topk_idx == fid)
            a_t = (topk_vals * mask).sum(dim=-1, keepdim=True)
            delta_total = delta_total + diag_box["alpha"] * a_t * W_dec[:, fid].view(1, 1, -1)
            a_target_max = max(a_target_max, float(a_t.abs().max().item()))
        h_norm_before = float(h.float().norm().item())
        h.sub_(delta_total)
        delta_norm = float(delta_total.float().norm().item())
        diag_box["a_target_vals"].append(a_target_max)
        diag_box["delta_norms"].append(delta_norm)
        diag_box["h_norms"].append(h_norm_before)
        diag_box["modify_count"] += 1
        return output

    layers = (
        model.model.transformer.blocks if hasattr(model, "model") and hasattr(model.model, "transformer")
        else model.model.layers
    )
    handle = layers[SAE_LAYER].register_forward_hook(hook)

    def add_gumbel_noise(logits, temperature=1.0):
        if temperature == 0:
            return logits
        noise = torch.rand_like(logits, dtype=torch.float64)
        noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        return logits.to(torch.float64) + noise * temperature

    def get_num_transfer_tokens(mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        return torch.where(
            torch.arange(steps, device=mask_index.device).unsqueeze(0) < remainder,
            base + 1, base,
        )

    def build_prompt(inst):
        tests = "\n".join(inst["test_list"])
        user = f"{inst['prompt']}\n\nYour code should pass these tests:\n{tests}"
        messages = [
            {"role": "system", "content": (
                "You are an expert Python programmer. "
                "Write a Python function that solves the given task. "
                "Output only the function definition, no explanations."
            )},
            {"role": "user", "content": user},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return torch.tensor(
            tokenizer(text)["input_ids"], device=model.device,
        ).unsqueeze(0)

    def generate_one(inst, condition):
        prompt_ids = build_prompt(inst)
        gen_start = prompt_ids.shape[1]
        torch.manual_seed(0)
        x = torch.full(
            (1, gen_start + GEN_LENGTH), MASK_ID,
            dtype=torch.long, device=model.device,
        )
        x[:, :gen_start] = prompt_ids.clone()
        # Set hook config
        diag_box["enabled"] = condition is not None
        diag_box["from_step"] = condition["step"] if condition else 64
        diag_box["alpha"] = condition["alpha"] if condition else 0.0
        diag_box["targets"] = condition["features"] if condition else []
        # Reset accumulators
        diag_box["a_target_vals"] = []
        diag_box["delta_norms"] = []
        diag_box["h_norms"] = []
        diag_box["fire_count"] = 0
        diag_box["modify_count"] = 0

        num_blocks = GEN_LENGTH // BLOCK_LENGTH
        steps_per_block = STEPS // num_blocks
        global_step = 0
        for nb in range(num_blocks):
            bs = gen_start + nb * BLOCK_LENGTH
            be = gen_start + (nb + 1) * BLOCK_LENGTH
            block_mask = (x[:, bs:be] == MASK_ID)
            n_tr = get_num_transfer_tokens(block_mask, steps_per_block)
            for si in range(steps_per_block):
                diag_box["step"] = global_step
                out = model(x)
                logits = out.logits
                logits_n = add_gumbel_noise(logits, temperature=TEMPERATURE)
                n_transfer = n_tr[0, si].item()
                if n_transfer == 0:
                    global_step += 1
                    continue
                mask_index = x == MASK_ID
                x0 = torch.argmax(logits_n, dim=-1)
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                x0_p[:, :bs] = -np.inf
                x0_p[:, be:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p, -np.inf)
                n_unmask = min(n_transfer, mask_index[0, bs:be].sum().item())
                if n_unmask > 0:
                    _, idx = torch.topk(conf[0], k=n_unmask)
                    x[0, idx] = x0[0, idx]
                global_step += 1
        return x[:, gen_start:].cpu()

    def check_pass(inst, x_out):
        import re as _re
        text = tokenizer.batch_decode(x_out, skip_special_tokens=True)[0]
        m = _re.search(r"```(?:python)?\s*\n(.*?)```", text, _re.DOTALL)
        code = m.group(1).strip() if m else text.strip()
        ti = inst.get("test_imports", "") or ""
        if isinstance(ti, list):
            ti = "\n".join(ti)
        full = ti + "\n" + code + "\n" + "\n".join(inst["test_list"])

        def _to(s, f):
            raise TimeoutError()
        signal.signal(signal.SIGALRM, _to)
        signal.alarm(10)
        try:
            exec(full, {})
            return True
        except Exception:
            return False
        finally:
            signal.alarm(0)

    results = {"conditions": []}

    # Baseline first to capture passive a_target distribution
    baseline_rows = []
    print("\n=== Baseline (no intervention, hook records only) ===")
    # For baseline diagnostics, still record a_target via hook but with alpha=0
    diag_baseline = {
        "name": "baseline_passive",
        "alpha": 0.0,
        "step": 64,
        "features": [TARGET_F],
    }
    # Use a tiny modification: alpha=0 so delta=0 but a_target gets recorded
    for idx in fail_idxs:
        inst = instances[idx]
        t0 = time.monotonic()
        x_out = generate_one(inst, diag_baseline)
        ok = check_pass(inst, x_out)
        a_vals = np.array(diag_box["a_target_vals"])
        d_vals = np.array(diag_box["delta_norms"])
        h_vals = np.array(diag_box["h_norms"])
        baseline_rows.append({
            "task_id": int(inst["task_id"]),
            "pass": bool(ok),
            "a_target_mean": float(a_vals.mean()) if len(a_vals) else 0.0,
            "a_target_max": float(a_vals.max()) if len(a_vals) else 0.0,
            "delta_norm_mean": float(d_vals.mean()) if len(d_vals) else 0.0,
            "delta_norm_max": float(d_vals.max()) if len(d_vals) else 0.0,
            "h_norm_mean": float(h_vals.mean()) if len(h_vals) else 0.0,
            "n_hook_fires_active": len(a_vals),
        })
        print(
            f"  task {inst['task_id']}: pass={ok} "
            f"a_target_mean={a_vals.mean() if len(a_vals) else 0:.3f} "
            f"({time.monotonic()-t0:.1f}s)"
        )
    results["conditions"].append({"condition": "baseline_passive", "rows": baseline_rows})

    for cond in CONDITIONS:
        print(f"\n=== {cond['name']} (alpha={cond['alpha']}, step>={cond['step']}) ===")
        rows = []
        for idx in fail_idxs:
            inst = instances[idx]
            t0 = time.monotonic()
            x_out = generate_one(inst, cond)
            ok = check_pass(inst, x_out)
            a_vals = np.array(diag_box["a_target_vals"])
            d_vals = np.array(diag_box["delta_norms"])
            h_vals = np.array(diag_box["h_norms"])
            ratio = d_vals / np.maximum(h_vals, 1e-6)
            rows.append({
                "task_id": int(inst["task_id"]),
                "pass": bool(ok),
                "a_target_mean": float(a_vals.mean()) if len(a_vals) else 0.0,
                "a_target_max": float(a_vals.max()) if len(a_vals) else 0.0,
                "delta_norm_mean": float(d_vals.mean()) if len(d_vals) else 0.0,
                "delta_norm_max": float(d_vals.max()) if len(d_vals) else 0.0,
                "h_norm_mean": float(h_vals.mean()) if len(h_vals) else 0.0,
                "ratio_mean": float(ratio.mean()) if len(ratio) else 0.0,
                "ratio_max": float(ratio.max()) if len(ratio) else 0.0,
                "n_hook_fires_active": len(a_vals),
            })
            print(
                f"  task {inst['task_id']}: pass={ok} "
                f"a_target_mean={a_vals.mean() if len(a_vals) else 0:.3f} "
                f"||Dh||_mean={d_vals.mean() if len(d_vals) else 0:.2f} "
                f"ratio_mean={ratio.mean() if len(ratio) else 0:.4f} "
                f"({time.monotonic()-t0:.1f}s)"
            )
        results["conditions"].append({
            "condition": cond["name"],
            "alpha": cond["alpha"],
            "step_from": cond["step"],
            "features": cond["features"],
            "rows": rows,
        })

    # Aggregate per condition
    summary = []
    for c in results["conditions"]:
        rows = c["rows"]
        if not rows:
            continue
        a_means = [r["a_target_mean"] for r in rows]
        d_means = [r["delta_norm_mean"] for r in rows]
        r_means = [r.get("ratio_mean", 0.0) for r in rows]
        n_flipped = sum(1 for r in rows if r["pass"])  # all should be False for fails
        summary.append({
            "condition": c["condition"],
            "n_samples": len(rows),
            "a_target_mean": float(np.mean(a_means)),
            "delta_norm_mean": float(np.mean(d_means)),
            "ratio_mean": float(np.mean(r_means)),
            "flips_to_pass": n_flipped,
        })
        print(
            f"\nSUMMARY {c['condition']}: a_target_mean={np.mean(a_means):.3f} "
            f"||Dh||_mean={np.mean(d_means):.2f} "
            f"ratio={np.mean(r_means):.4f} flips={n_flipped}/{len(rows)}"
        )
    results["summary"] = summary

    out_path = "/results/mbpp_llada/steer_effect_size.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()
    handle.remove()
    print(f"\nSaved {out_path}")
    return json.dumps(summary, indent=2)


@app.local_entrypoint()
def main(n_fail: int = 8):
    print(f"Effect-size diagnostics on LLaDA-MBPP, n_fail={n_fail}")
    print(run_effect_size.remote(n_fail))
