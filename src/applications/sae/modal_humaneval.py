"""HumanEval difficulty-transport on a new code dataset (reviewer #10).

Tests whether the difficulty-transport finding generalises to a code dataset not
used anywhere in the pipeline. We extract Dream-Base all-layer hidden states on
HumanEval prompts (forward pass, the DLM's pre-generation difficulty estimate)
and ask how well they predict the pass/fail of two independent AR generators,
Qwen-2.5-7B-Instruct and Llama-3.1-8B-Instruct. If Dream features predict both
at similar AUC (and the two generators agree), the transported signal is
generator-agnostic problem difficulty on a held-out code task.

Output: /results/humaneval/transport_results.json

Usage:
  .venv/bin/modal run src/applications/sae/modal_humaneval.py
"""

import modal

app = modal.App("humaneval-transport")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "curl", "build-essential")
         .pip_install("torch>=2.0", "transformers==4.52.2", "accelerate>=0.30",
                      "numpy", "datasets==2.21.0", "huggingface_hub", "scikit-learn"))
RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

DREAM_BASE = "Dream-org/Dream-v0-Base-7B"
DREAM_INSTRUCT = "Dream-org/Dream-v0-Instruct-7B"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
MASK_ID = 151666
GEN_LEN = 320


def grade_humaneval(inst, txt):
    import re, signal
    m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL)
    code = m.group(1) if m else txt
    if f"def {inst['entry_point']}" not in code:
        code = inst["prompt"] + code
    full = code + "\n" + inst["test"] + f"\ncheck({inst['entry_point']})\n"
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(10)
    try:
        exec(full, {})
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)


@app.function(image=image, gpu="A100-80GB", timeout=14400,
              volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
              secrets=[modal.Secret.from_name("huggingface")])
def run():
    import json, os, numpy as np, torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"
    os.makedirs("/results/humaneval", exist_ok=True)

    insts = list(load_dataset("openai/openai_humaneval", split="test"))
    N = len(insts)
    print(f"HumanEval N={N}")

    def chat(tok, sys, usr):
        return tok.apply_chat_template(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            add_generation_prompt=True, tokenize=False)
    SYS = ("You are an expert Python programmer. Complete the function. "
           "Output only the full function in a Python code block.")

    # ---- Dream-Base all-layer hidden states (difficulty features) ----
    itok = AutoTokenizer.from_pretrained(DREAM_INSTRUCT, trust_remote_code=True)
    prompts = [chat(itok, SYS, inst["prompt"]) for inst in insts]
    dtok = AutoTokenizer.from_pretrained(DREAM_BASE, trust_remote_code=True)
    dream = AutoModel.from_pretrained(DREAM_BASE, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).cuda().eval()
    layers = dream.model.layers if hasattr(dream, "model") else dream.transformer.h
    nL = len(layers); dh = dream.config.hidden_size
    cap = [None] * nL
    hs = [layers[i].register_forward_hook(
        (lambda i: (lambda m, a, o: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach())))(i))
        for i in range(nL)]
    H = np.zeros((N, nL, dh), np.float32)
    for k, p in enumerate(prompts):
        ids = dtok(p, return_tensors="pt").input_ids.cuda(); pl = ids.shape[1]
        full = torch.full((1, pl + GEN_LEN), MASK_ID, dtype=ids.dtype, device=ids.device)
        full[0, :pl] = ids[0]
        for i in range(nL):
            cap[i] = None
        with torch.no_grad():
            dream(full)
        for i in range(nL):
            H[k, i] = cap[i][0, pl - 1, :].float().cpu().numpy()
    for h in hs:
        h.remove()
    del dream; torch.cuda.empty_cache()
    print(f"H_D {H.shape}")

    # ---- AR generators: labels ----
    def gen_labels(model_name):
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda().eval()
        texts = [chat(tok, SYS, inst["prompt"]) for inst in insts]
        passed = [0] * N
        BS = 16
        for s in range(0, N, BS):
            enc = tok(texts[s:s+BS], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=GEN_LEN, do_sample=True,
                                 temperature=0.2, top_p=0.95, pad_token_id=tok.pad_token_id)
            gen = out[:, enc.input_ids.shape[1]:]
            for j in range(gen.shape[0]):
                passed[s+j] = int(grade_humaneval(insts[s+j], tok.decode(gen[j], skip_special_tokens=True)))
        del m; torch.cuda.empty_cache()
        return np.array(passed)

    qwen_lab = gen_labels(QWEN); print(f"Qwen pass {qwen_lab.sum()}/{N}")
    llama_lab = gen_labels(LLAMA); print(f"Llama pass {llama_lab.sum()}/{N}")

    def cv_auc(X, y, C=0.01):
        if (y == 0).sum() < 5 or (y == 1).sum() < 5:
            return float("nan")
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        return float(np.mean([roc_auc_score(y[te], LogisticRegression(max_iter=2000, C=C)
                     .fit(X[tr], y[tr]).decision_function(X[te])) for tr, te in skf.split(X, y)]))
    rows = [{"layer": li, "auc_qwen": cv_auc(H[:, li, :], qwen_lab),
             "auc_llama": cv_auc(H[:, li, :], llama_lab)} for li in range(nL)]
    valid = [r for r in rows if r["auc_qwen"] == r["auc_qwen"]]
    bq = max(valid, key=lambda r: r["auc_qwen"])
    out = {"dataset": "humaneval", "N": N, "qwen_pass": int(qwen_lab.sum()),
           "llama_pass": int(llama_lab.sum()),
           "qwen_llama_agreement": float((qwen_lab == llama_lab).mean()),
           "best_layer_qwen": bq, "per_layer": rows}
    with open("/results/humaneval/transport_results.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nbest L{bq['layer']}: Dream feats -> Qwen {bq['auc_qwen']:.3f}, -> Llama {bq['auc_llama']:.3f}; "
          f"Qwen/Llama agree {out['qwen_llama_agreement']:.3f}")
    return json.dumps(out["best_layer_qwen"])


@app.local_entrypoint()
def main():
    print(run.remote())
