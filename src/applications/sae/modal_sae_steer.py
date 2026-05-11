"""Stage 4 SAE steering: suppress f15601 at L26 from step >= 64 on MBPP fails.

For each selected MBPP fail case (drawn from Stage 2 clusters):
  1. Generate baseline output (vanilla LLaDA).
  2. Generate steered output where, from denoising step 64 onward, the layer-26
     residual stream is patched as: h <- h - alpha * a_{15601} * W_dec[:, 15601]
     where a_{15601} is the SAE encoder activation of feature 15601 on h.
  3. Re-execute MBPP tests on both outputs to measure fail -> pass rate.

We also run a smaller set of pass cases as a regression control
(pass -> fail rate under the same steering).

Usage:
  .venv/bin/modal run src/applications/sae/modal_sae_steer.py \\
      --n-fail-c1 20 --n-fail-c0 10 --n-pass 10 --alpha 1.0
"""

import modal

app = modal.App("sae-steer-stage4")

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
GEN_LENGTH = 256  # MBPP
STEPS = 128
BLOCK_LENGTH = 32

SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYER = 26  # 0-indexed transformer block, residual_post
SAE_TRAINER = 2
TARGET_FEATURE = 15601  # top fail-leaning, +0.385 enrichment on MBPP
STEER_FROM_STEP = 64  # SAE training range begins around here


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_steering(
    n_fail_c1: int,
    n_fail_c0: int,
    n_pass: int,
    alpha: float,
    steer_from_step: int = STEER_FROM_STEP,
    target_features: list = None,
):
    if target_features is None:
        target_features = [TARGET_FEATURE]
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
    cluster_sizes = {c["cluster"]: c["size"] for c in diag["clusters"]}
    print(f"Cluster sizes: {cluster_sizes}")

    fail_c1_idxs = clusters[1][:n_fail_c1]
    fail_c0_idxs = clusters[0][:n_fail_c0]

    # ---- Load SAE ----
    sae_path_dir = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
    ae_local = hf_hub_download(
        repo_id=SAE_REPO,
        filename=f"{sae_path_dir}/ae.pt",
        cache_dir="/hf-cache",
    )
    cfg_local = hf_hub_download(
        repo_id=SAE_REPO,
        filename=f"{sae_path_dir}/config.json",
        cache_dir="/hf-cache",
    )
    with open(cfg_local) as f:
        sae_cfg = json.load(f)
    sae_k = sae_cfg["trainer"]["k"]
    sae_d_in = sae_cfg["trainer"]["activation_dim"]
    sae_d_sae = sae_cfg["trainer"]["dict_size"]
    print(f"SAE: d_in={sae_d_in}, d_sae={sae_d_sae}, k={sae_k}, "
          f"target_feature={TARGET_FEATURE}, alpha={alpha}")

    state = torch.load(ae_local, map_location="cpu", weights_only=True)
    W_enc = state["encoder.weight"].cuda().to(torch.bfloat16)   # (d_sae, d_in)
    b_enc = state["encoder.bias"].cuda().to(torch.bfloat16)     # (d_sae,)
    W_dec = state["decoder.weight"].cuda().to(torch.bfloat16)   # (d_in, d_sae)
    b_dec = state["b_dec"].cuda().to(torch.bfloat16)            # (d_in,)
    # target_features is a list (1+ feature IDs). Default keeps backward compat.
    target_dec_cols = W_dec[:, target_features].clone()  # (d_in, n_targets)
    target_feats_tensor = torch.tensor(target_features, device="cuda")
    print(
        f"W_enc {tuple(W_enc.shape)}, W_dec {tuple(W_dec.shape)}, "
        f"targets={target_features}, "
        f"dec_col_norms={[round(target_dec_cols[:, i].float().norm().item(), 3) for i in range(len(target_features))]}"
    )

    # ---- Load LLaDA + MBPP ----
    tokenizer = AutoTokenizer.from_pretrained(LLADA_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLADA_NAME, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])

    # Sample pass instances disjoint from fail clusters
    fail_set = set([i for ids in clusters.values() for i in ids])
    pass_pool = [i for i in range(len(instances)) if i not in fail_set]
    rng = np.random.RandomState(42)
    pass_idxs = list(rng.choice(pass_pool, size=n_pass, replace=False))

    # ---- Steering hook ----
    state_box = {
        "step": -1,
        "steer_enabled": False,
        "fire_count": 0,
        "mod_count": 0,
        "max_a_target": 0.0,
        "max_delta_norm": 0.0,
    }

    def steering_hook(module, args, output):
        state_box["fire_count"] += 1
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
        # For each target feature, compute its activation (0 if not in top-k)
        # and accumulate the delta = alpha * a_target * decoder_column.
        delta_total = torch.zeros_like(h)
        a_max_seen = 0.0
        for ti, fid in enumerate(target_features):
            mask = (topk_idx == fid)
            a_target = (topk_vals * mask).sum(dim=-1, keepdim=True)  # (1, seq, 1)
            delta_total = delta_total + alpha * a_target * target_dec_cols[:, ti].view(1, 1, -1)
            m = float(a_target.abs().max().item())
            if m > a_max_seen:
                a_max_seen = m
        # IN-PLACE: subtract the accumulated steering vector
        h.sub_(delta_total)
        a_max = a_max_seen
        delta_norm = float(delta_total.norm().item())
        if a_max > state_box["max_a_target"]:
            state_box["max_a_target"] = a_max
        if delta_norm > state_box["max_delta_norm"]:
            state_box["max_delta_norm"] = delta_norm
        state_box["mod_count"] += 1
        if state_box["mod_count"] <= 3:
            print(
                f"    HOOK step={state_box['step']} "
                f"h.shape={tuple(h.shape)} a_target_max={a_max:.4f} "
                f"delta_norm={delta_norm:.4f} h.norm_after={h.norm().item():.2f}"
            )
        return output

    # Locate transformer layer 26
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        layers = model.model.transformer.blocks
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        layers = model.transformer.blocks
    else:
        raise RuntimeError(
            f"Could not locate transformer layers. Model attrs: "
            f"{[a for a in dir(model) if not a.startswith('_')][:20]}"
        )
    print(f"Found {len(layers)} layers, hooking layer {SAE_LAYER}")
    hook_handle = layers[SAE_LAYER].register_forward_hook(steering_hook)

    # ---- LLaDA generation (matches modal_midstep_probe.py) ----

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

    def generate_llada(x, gen_start, steer: bool):
        state_box["steer_enabled"] = steer
        num_blocks = GEN_LENGTH // BLOCK_LENGTH
        steps_per_block = STEPS // num_blocks
        global_step = 0
        for num_block in range(num_blocks):
            block_start = gen_start + num_block * BLOCK_LENGTH
            block_end = gen_start + (num_block + 1) * BLOCK_LENGTH
            block_mask_index = (x[:, block_start:block_end] == MASK_ID)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            for si in range(steps_per_block):
                state_box["step"] = global_step
                out = model(x)
                logits = out.logits
                logits_with_noise = add_gumbel_noise(logits, temperature=TEMPERATURE)
                n_transfer = num_transfer_tokens[0, si].item()
                if n_transfer == 0:
                    global_step += 1
                    continue
                mask_index = x == MASK_ID
                x0 = torch.argmax(logits_with_noise, dim=-1)
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                x0_p[:, :block_start] = -np.inf
                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                n_unmask = min(
                    n_transfer, mask_index[0, block_start:block_end].sum().item(),
                )
                if n_unmask > 0:
                    _, indices = torch.topk(confidence[0], k=n_unmask)
                    x[0, indices] = x0[0, indices]
                global_step += 1
        state_box["steer_enabled"] = False
        return x

    # ---- Prompt + correctness helpers (MBPP only) ----

    def build_prompt(inst):
        sys_prompt = (
            "You are an expert Python programmer. "
            "Write a Python function that solves the given task. "
            "Output only the function definition, no explanations."
        )
        tests_str = "\n".join(inst["test_list"])
        user_prompt = (
            f"{inst['prompt']}\n\nYour code should pass these tests:\n{tests_str}"
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(text)["input_ids"]
        return torch.tensor(ids, device=model.device).unsqueeze(0)

    def extract_code(output_text):
        m = re.search(r"```python\n(.*?)```", output_text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\n(.*?)```", output_text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return output_text.strip()

    def check_mbpp(output_text, inst):
        code = extract_code(output_text)
        test_imports = inst.get("test_imports", "") or ""
        if isinstance(test_imports, list):
            test_imports = "\n".join(test_imports)
        tests = inst["test_list"]
        exec_code = ""
        if test_imports:
            exec_code += test_imports + "\n"
        exec_code += code + "\n"
        for test in tests:
            exec_code += test + "\n"

        def _to(signum, frame):
            raise TimeoutError()

        old_h = signal.signal(signal.SIGALRM, _to)
        signal.alarm(10)
        try:
            exec(exec_code, {"__builtins__": __builtins__}, {})
            return True
        except Exception:
            return False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_h)

    # ---- Generate baseline + steered per instance ----

    def generate_one(inst, steer: bool):
        prompt_ids = build_prompt(inst)
        gen_start = prompt_ids.shape[1]
        torch.manual_seed(0)
        x = torch.full(
            (1, gen_start + GEN_LENGTH), MASK_ID,
            dtype=torch.long, device=model.device,
        )
        x[:, :gen_start] = prompt_ids.clone()
        # Reset diagnostic counters
        state_box["fire_count"] = 0
        state_box["mod_count"] = 0
        state_box["max_a_target"] = 0.0
        state_box["max_delta_norm"] = 0.0
        x = generate_llada(x, gen_start, steer)
        if steer:
            print(
                f"    [diag] fire_count={state_box['fire_count']} "
                f"mod_count={state_box['mod_count']} "
                f"max_a_target={state_box['max_a_target']:.4f} "
                f"max_delta_norm={state_box['max_delta_norm']:.4f}"
            )
        text = tokenizer.batch_decode(
            x[:, gen_start:], skip_special_tokens=True,
        )[0]
        return text

    def run_group(group_label, idx_list, expected_label: int):
        rows = []
        t0 = time.monotonic()
        for j, idx in enumerate(idx_list):
            inst = instances[idx]
            base_text = generate_one(inst, steer=False)
            base_ok = check_mbpp(base_text, inst)
            steer_text = generate_one(inst, steer=True)
            steer_ok = check_mbpp(steer_text, inst)
            text_identical = (base_text == steer_text)
            row = {
                "group": group_label,
                "sample_idx": int(idx),
                "task_id": int(inst["task_id"]),
                "expected_pass": expected_label,
                "baseline_pass": bool(base_ok),
                "steer_pass": bool(steer_ok),
                "flipped": bool(base_ok) ^ bool(steer_ok),
                "text_identical": text_identical,
            }
            rows.append(row)
            elapsed = time.monotonic() - t0
            print(
                f"  [{group_label} {j+1}/{len(idx_list)}] task={inst['task_id']} "
                f"base={'P' if base_ok else 'F'} "
                f"steer={'P' if steer_ok else 'F'} "
                f"flip={'Y' if row['flipped'] else 'N'} "
                f"text_id={'Y' if text_identical else 'N'} "
                f"({elapsed:.0f}s)"
            )
            if j == 0 and group_label == "fail_c1":
                print("    base_text[:200]:", repr(base_text[:200]))
                print("    steer_text[:200]:", repr(steer_text[:200]))
        return rows

    print("\n=== Steering on Cluster 1 fails (n={}) ===".format(len(fail_c1_idxs)))
    rows_c1 = run_group("fail_c1", fail_c1_idxs, expected_label=0)
    print("\n=== Steering on Cluster 0 fails (n={}) ===".format(len(fail_c0_idxs)))
    rows_c0 = run_group("fail_c0", fail_c0_idxs, expected_label=0)
    print("\n=== Steering on pass cases (n={}) ===".format(len(pass_idxs)))
    rows_pass = run_group("pass", pass_idxs, expected_label=1)

    def summarize(rows, label):
        n = len(rows)
        if n == 0:
            return None
        base_p = sum(r["baseline_pass"] for r in rows)
        steer_p = sum(r["steer_pass"] for r in rows)
        fail_to_pass = sum(
            1 for r in rows if not r["baseline_pass"] and r["steer_pass"]
        )
        pass_to_fail = sum(
            1 for r in rows if r["baseline_pass"] and not r["steer_pass"]
        )
        return {
            "label": label,
            "n": n,
            "baseline_pass_rate": round(base_p / n, 4),
            "steer_pass_rate": round(steer_p / n, 4),
            "fail_to_pass": fail_to_pass,
            "pass_to_fail": pass_to_fail,
        }

    summaries = [
        summarize(rows_c1, "fail_c1 (steering target)"),
        summarize(rows_c0, "fail_c0 (control fails)"),
        summarize(rows_pass, "pass (regression control)"),
    ]
    print("\n" + "=" * 70)
    print("Summary (target features={}, alpha={}, steer from step {}):".format(
        target_features, alpha, steer_from_step))
    print("=" * 70)
    for s in summaries:
        if s:
            print(
                f"  {s['label']:>32}: n={s['n']:>3} "
                f"base_pass={s['baseline_pass_rate']:.3f} "
                f"steer_pass={s['steer_pass_rate']:.3f} "
                f"fail->pass={s['fail_to_pass']} "
                f"pass->fail={s['pass_to_fail']}"
            )

    results = {
        "config": {
            "sae_repo": SAE_REPO,
            "sae_layer": SAE_LAYER,
            "sae_k": sae_k,
            "target_features": target_features,
            "alpha": alpha,
            "steer_from_step": steer_from_step,
            "n_fail_c1": n_fail_c1,
            "n_fail_c0": n_fail_c0,
            "n_pass": n_pass,
        },
        "summaries": summaries,
        "rows": rows_c1 + rows_c0 + rows_pass,
    }
    tag = "f" + "_".join(str(f) for f in target_features)
    out_path = (
        f"/results/mbpp_llada/sae_steer_stage4_{tag}_a{alpha}_s{steer_from_step}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved to {out_path}")

    hook_handle.remove()
    return json.dumps({"summaries": summaries}, indent=2)


@app.local_entrypoint()
def main(
    n_fail_c1: int = 20,
    n_fail_c0: int = 10,
    n_pass: int = 10,
    alpha: float = 1.0,
    steer_from_step: int = STEER_FROM_STEP,
    features: str = str(TARGET_FEATURE),
):
    target_features = [int(x) for x in features.split(",") if x.strip()]
    print(
        f"Stage 4 SAE steering: targets={target_features}, alpha={alpha}, "
        f"steer_from={steer_from_step}, "
        f"n_fail_c1={n_fail_c1}, n_fail_c0={n_fail_c0}, n_pass={n_pass}"
    )
    result = run_steering.remote(
        n_fail_c1, n_fail_c0, n_pass, alpha, steer_from_step, target_features,
    )
    print("\n" + result)
