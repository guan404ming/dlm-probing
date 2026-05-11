"""Generate qualitative MBPP failure examples for SAE fail clusters.

This is a lightweight follow-up to the cached Stage-2 SAE diagnose output:
it loads the LLaDA-MBPP fail clusters, re-generates only a few selected MBPP
outputs, executes the tests, and saves prompt/output/error snippets for
paper-level qualitative inspection.

Usage:
  uv run modal run --detach src/applications/sae/modal_cluster_failure_examples.py --n-per-cluster 10
  uv run modal run src/applications/sae/modal_cluster_failure_examples.py --n-per-cluster 10 --wait
"""

import modal

app = modal.App("sae-cluster-failure-examples")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0",
        "transformers==4.52.2",
        "accelerate>=0.30",
        "numpy",
        "datasets==2.21.0",
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


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def collect_examples(n_per_cluster: int = 4):
    import json
    import os
    import re
    import signal
    import time
    import traceback

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    with open("/results/mbpp_llada/sae_diagnose_stage2.json") as f:
        diag = json.load(f)
    clusters = diag["clusters"]
    cluster_features = {
        c["cluster"]: [r["feature_id"] for r in c["characteristic_features"][:5]]
        for c in clusters
    }
    selected = []
    for c in clusters:
        for idx in c["fail_sample_indices"][:n_per_cluster]:
            selected.append((c["cluster"], idx))
    print(f"Selected {len(selected)} examples: {selected}")

    tokenizer = AutoTokenizer.from_pretrained(LLADA_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLADA_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])

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
            base + 1,
            base,
        )

    def generate_one(inst):
        prompt_ids = build_prompt(inst)
        gen_start = prompt_ids.shape[1]
        torch.manual_seed(0)
        x = torch.full(
            (1, gen_start + GEN_LENGTH),
            MASK_ID,
            dtype=torch.long,
            device=model.device,
        )
        x[:, :gen_start] = prompt_ids.clone()
        num_blocks = GEN_LENGTH // BLOCK_LENGTH
        steps_per_block = STEPS // num_blocks
        for num_block in range(num_blocks):
            block_start = gen_start + num_block * BLOCK_LENGTH
            block_end = gen_start + (num_block + 1) * BLOCK_LENGTH
            block_mask_index = x[:, block_start:block_end] == MASK_ID
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block,
            )
            for si in range(steps_per_block):
                out = model(x)
                logits = out.logits
                logits_with_noise = add_gumbel_noise(logits, TEMPERATURE)
                n_transfer = int(num_transfer_tokens[0, si].item())
                if n_transfer == 0:
                    continue
                mask_index = x == MASK_ID
                x0 = torch.argmax(logits_with_noise, dim=-1)
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1,
                )
                x0_p[:, :block_start] = -np.inf
                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                n_unmask = min(
                    n_transfer,
                    int(mask_index[0, block_start:block_end].sum().item()),
                )
                if n_unmask:
                    _, indices = torch.topk(confidence[0], k=n_unmask)
                    x[0, indices] = x0[0, indices]
        return tokenizer.batch_decode(x[:, gen_start:], skip_special_tokens=True)[0]

    def extract_code(output_text):
        m = re.search(r"```(?:python)?\s*\n(.*?)```", output_text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return output_text.strip()

    def check_mbpp_with_error(output_text, inst):
        code = extract_code(output_text)
        test_imports = inst.get("test_imports", "") or ""
        if isinstance(test_imports, list):
            test_imports = "\n".join(test_imports)
        exec_code = ""
        if test_imports:
            exec_code += test_imports + "\n"
        exec_code += code + "\n"
        for test in inst["test_list"]:
            exec_code += test + "\n"

        def _timeout(signum, frame):
            raise TimeoutError("MBPP execution timed out")

        old_h = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(10)
        try:
            exec(exec_code, {"__builtins__": __builtins__}, {})
            return True, None, None
        except Exception as exc:
            return False, type(exc).__name__, str(exc)[:500]
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_h)

    rows = []
    t0 = time.monotonic()
    for j, (cluster_id, sample_idx) in enumerate(selected, start=1):
        inst = instances[sample_idx]
        try:
            output_text = generate_one(inst)
            ok, error_type, error_msg = check_mbpp_with_error(output_text, inst)
        except Exception as exc:
            output_text = ""
            ok = False
            error_type = type(exc).__name__
            error_msg = traceback.format_exc(limit=3)[-500:]
        code = extract_code(output_text)
        row = {
            "cluster": int(cluster_id),
            "cluster_features": cluster_features[int(cluster_id)],
            "sample_idx": int(sample_idx),
            "task_id": int(inst["task_id"]),
            "prompt": inst["prompt"],
            "tests": inst["test_list"],
            "pass": bool(ok),
            "error_type": error_type,
            "error_msg": error_msg,
            "output_text": output_text[:2000],
            "extracted_code": code[:1600],
        }
        rows.append(row)
        print(
            f"[{j}/{len(selected)}] cluster={cluster_id} "
            f"sample={sample_idx} task={inst['task_id']} "
            f"ok={ok} error={error_type} elapsed={time.monotonic() - t0:.0f}s"
        )

    summary = {}
    for row in rows:
        key = str(row["cluster"])
        summary.setdefault(key, {"n": 0, "error_types": {}})
        summary[key]["n"] += 1
        et = row["error_type"] or "pass"
        summary[key]["error_types"][et] = summary[key]["error_types"].get(et, 0) + 1

    result = {
        "source": "/results/mbpp_llada/sae_diagnose_stage2.json",
        "n_per_cluster": n_per_cluster,
        "summary": summary,
        "rows": rows,
    }
    out_path = "/results/mbpp_llada/cluster_failure_examples.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    RESULTS_VOL.commit()
    print(f"Saved {len(rows)} rows to {out_path}")
    return json.dumps(summary, indent=2)


@app.local_entrypoint()
def main(n_per_cluster: int = 4, wait: bool = False):
    if wait:
        print(collect_examples.remote(n_per_cluster))
        return
    call = collect_examples.spawn(n_per_cluster)
    print(f"Spawned collect_examples: {call}")
