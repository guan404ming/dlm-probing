"""Control B: Qwen2.5-7B-Instruct all-layer hidden states (n=257 MBPP).

Companion to modal_ardlm_diff_alllayers.py (Qwen2.5-7B-Base + Dream-Base).
Together they distinguish:
  Diffusion-training effect: Dream-Base vs Qwen-Base
  Instruction-tuning effect: Qwen-Instruct vs Qwen-Base

If both rewrite L27 similarly, the late-layer divergence is not diffusion-specific.
If only Dream-Base diverges, the late-layer rewrite is diffusion-specific.

Usage:
  .venv/bin/modal run --detach src/applications/sae/modal_qwen_instruct_alllayers.py
"""

import modal

app = modal.App("qwen-instruct-alllayers")

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

QWEN_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
DREAM_INSTRUCT = "Dream-org/Dream-v0-Instruct-7B"


@app.function(
    image=image, gpu="A100-80GB", timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run():
    import json
    import os
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    diag_path = "/results/mbpp_dream/sae_diagnose_stage2.json"
    with open(diag_path) as f:
        diag = json.load(f)
    fail_idxs = []
    for c in diag.get("clusters", []):
        fail_idxs.extend(c.get("fail_sample_indices", []))
    fail_idxs = list(dict.fromkeys(fail_idxs))

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])
    fail_set = set(fail_idxs)
    pass_idxs = [i for i in range(len(instances)) if i not in fail_set]
    all_idxs = fail_idxs + pass_idxs
    labels = np.array([0] * len(fail_idxs) + [1] * len(pass_idxs))
    print(f"Total {len(all_idxs)} samples")

    instruct_tok = AutoTokenizer.from_pretrained(DREAM_INSTRUCT, trust_remote_code=True)
    prompts_text = []
    for idx in all_idxs:
        inst = instances[idx]
        msgs = [{"role": "user",
                 "content": f"Write a Python function. Only output code in a Python block.\n\nProblem: {inst['prompt']}"}]
        text = instruct_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompts_text.append(text)

    print(f"\n=== Loading {QWEN_INSTRUCT} ===")
    tok = AutoTokenizer.from_pretrained(QWEN_INSTRUCT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_INSTRUCT, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    layers = model.model.layers
    n_l = len(layers)
    d_h = model.config.hidden_size
    print(f"layers={n_l}, hidden={d_h}")

    captured = [None] * n_l
    def make_hook(li):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[li] = h.detach().clone()
        return hook
    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_l)]

    H_QI = np.zeros((len(prompts_text), n_l, d_h), dtype=np.float32)
    for k, prompt in enumerate(prompts_text):
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        pl = ids.shape[1]
        for i in range(n_l):
            captured[i] = None
        with torch.no_grad():
            _ = model(ids)
        for i in range(n_l):
            if captured[i] is None:
                raise RuntimeError(f"layer {i} hook did not fire")
            H_QI[k, i] = captured[i][0, pl - 1, :].float().cpu().numpy()
        if (k + 1) % 20 == 0 or k < 2:
            print(f"  Qwen-Instruct {k+1}/{len(prompts_text)}")
    for h in handles:
        h.remove()

    print(f"H_QI shape={H_QI.shape}")
    np.savez("/results/mbpp_dream/qwen_instruct_alllayers.npz",
             H_QI=H_QI, labels=labels)
    RESULTS_VOL.commit()
    print("Saved /results/mbpp_dream/qwen_instruct_alllayers.npz")
    return json.dumps({"ok": True, "n_layers": n_l, "n_samples": len(labels)})


@app.local_entrypoint()
def main():
    print(run.remote())
