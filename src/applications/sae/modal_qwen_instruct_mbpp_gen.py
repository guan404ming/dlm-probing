"""Generate Qwen-2.5-7B-Instruct MBPP solutions and label pass/fail.

Used to test difficulty-encoding hypothesis: if Dream L26 features encode
problem difficulty (not Dream-specific generation correctness), then they
should predict Qwen-Instruct's correctness on the same problems too.

Decision: compare AUC(Qwen-Instruct labels) and AUC(Dream labels) using
the same Dream L26 hidden states. If similar -> difficulty hypothesis confirmed.

Usage:
  .venv/bin/modal run --detach src/applications/sae/modal_qwen_instruct_mbpp_gen.py
"""

import modal

app = modal.App("qwen-instruct-mbpp-gen")

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

QWEN = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW = 256


@app.function(
    image=image, gpu="A100-80GB", timeout=14400,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run():
    import json
    import os
    import re
    import signal as pysig
    import time
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    instances = sorted(list(ds), key=lambda x: x["task_id"])

    tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    print(f"Loaded {QWEN}")

    def check_mbpp(inst, txt):
        try:
            m = re.search(r"```python\s*(.*?)```", txt, re.DOTALL)
            code = m.group(1) if m else txt
            full = code + "\n" + "\n".join(inst["test_imports"]) + "\n"
            full += "\n".join(inst["test_list"])
            old = pysig.signal(pysig.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            pysig.alarm(10)
            try:
                ns = {}
                exec(full, ns)
                return True
            finally:
                pysig.alarm(0)
                pysig.signal(pysig.SIGALRM, old)
        except Exception:
            return False

    rows = []
    t0 = time.time()
    for k, inst in enumerate(instances):
        example_test = inst.get("test_list", [""])[0] if inst.get("test_list") else ""
        content = (
            f"Write a Python function. Only output code in a Python block.\n\n"
            f"Problem: {inst['prompt']}\n"
            f"Example test: {example_test}"
        )
        msgs = [{"role": "user", "content": content}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=True,
                                 temperature=0.2, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        gen = out[0][ids.shape[1]:]
        gen_text = tok.decode(gen, skip_special_tokens=True)
        passed = check_mbpp(inst, gen_text)
        rows.append({"idx": k, "task_id": inst["task_id"], "passed": int(passed),
                     "output": gen_text[:1000]})
        if (k + 1) % 20 == 0:
            n_pass = sum(r["passed"] for r in rows)
            print(f"  {k+1}/{len(instances)} pass_rate={n_pass}/{len(rows)}={n_pass/len(rows):.2%}  ({time.time()-t0:.0f}s)")

    n_pass = sum(r["passed"] for r in rows)
    print(f"\nFinal: {n_pass}/{len(rows)} pass = {n_pass/len(rows):.2%}")

    out_path = "/results/mbpp_dream/qwen_instruct_mbpp_labels.json"
    with open(out_path, "w") as f:
        json.dump({"model": QWEN, "n": len(rows), "n_pass": n_pass,
                   "rows": rows}, f, indent=2)
    RESULTS_VOL.commit()
    print(f"Saved {out_path}")
    return json.dumps({"n": len(rows), "n_pass": n_pass})


@app.local_entrypoint()
def main():
    print(run.remote())
