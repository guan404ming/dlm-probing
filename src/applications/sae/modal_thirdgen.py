"""Third-generator transport triangulation (reviewer positive-shift #1).

The MBPP/JSON/GSM8K transport tests use Dream features to predict Qwen-Instruct.
A third, architecturally unrelated generator (Llama-3.1-8B-Instruct) triangulates
the shared difficulty ordering: if Dream features also predict Llama-Instruct's
pass/fail at similar AUC, the transported signal is generator-agnostic difficulty.

For JSON/GSM8K we reuse the cached Dream H_D (transport_alllayers.npz); for MBPP
we re-extract Dream-Base all-layer hidden states. We generate + grade Llama
solutions, then report per-layer AUC(Dream feats -> Dream labels) vs
AUC(Dream feats -> Llama labels).

Usage:
  .venv/bin/modal run src/applications/sae/modal_thirdgen.py
"""

import modal

app = modal.App("transport-thirdgen")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install("torch>=2.0", "transformers==4.52.2", "accelerate>=0.30",
                 "numpy", "datasets==2.21.0", "huggingface_hub", "scikit-learn")
)
RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

DREAM_BASE = "Dream-org/Dream-v0-Base-7B"
DREAM_INSTRUCT = "Dream-org/Dream-v0-Instruct-7B"
GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # distinct family from Qwen/Dream
MASK_ID = 151666
TASKS = {"mbpp": 257, "jsonschema": 272, "gsm8k": 512}  # gsm8k subset for speed
GEN_LEN = {"mbpp": 256, "jsonschema": 256, "gsm8k": 512}


def load_instances(task):
    from datasets import load_dataset
    if task == "jsonschema":
        return sorted(list(load_dataset("eth-sri/json-mode-eval-extended", split="test")),
                      key=lambda x: x["instance_id"])
    if task == "gsm8k":
        return list(load_dataset("openai/gsm8k", "main", split="test"))
    return sorted(list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test")),
                  key=lambda x: x["task_id"])


def sys_prompt(task, inst):
    if task == "jsonschema":
        return ("You are a helpful assistant that answers in JSON. Here's the JSON schema "
                f"you must adhere to:\n<schema>\n{inst['schema']}\n</schema>\n")
    if task == "gsm8k":
        return ("Solve the math problem step by step. End your answer with #### "
                "followed by the final numeric answer.")
    return ("You are an expert Python programmer. Write a Python function that solves the "
            "given task. Output only the function definition, no explanations.")


def user_prompt(task, inst):
    if task == "jsonschema":
        return inst["input"]
    if task == "gsm8k":
        return inst["question"]
    tests = "\n".join(inst["test_list"])
    return f"{inst['prompt']}\n\nYour code should pass these tests:\n{tests}"


def grade(task, inst, txt):
    import json as J, re, signal
    if task == "jsonschema":
        ext = txt.split("```json\n", 1)[-1] if "```json\n" in txt else txt
        e = ext.find("```")
        ext = (ext[:e] if e != -1 else ext).strip().strip("`") + "\n"
        try:
            return J.dumps(J.loads(inst["output"]), indent=4) == J.dumps(J.loads(ext), indent=4)
        except (J.JSONDecodeError, ValueError):
            return False
    if task == "gsm8k":
        m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", txt)
        pred = m.group(1).replace(",", "") if m else (
            re.findall(r"[+-]?[\d,]+\.?\d*", txt)[-1].replace(",", "") if re.findall(r"[+-]?[\d,]+\.?\d*", txt) else None)
        g = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", inst["answer"])
        gold = g.group(1).replace(",", "") if g else None
        if pred is None or gold is None:
            return False
        try:
            return float(pred) == float(gold)
        except ValueError:
            return False
    # mbpp
    m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL)
    code = m.group(1) if m else txt
    ti = inst.get("test_imports", "") or ""
    if isinstance(ti, list):
        ti = "\n".join(ti)
    full = (ti + "\n" if ti else "") + code + "\n" + "\n".join(inst["test_list"])
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
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    # ---- gather Dream H_D + dream labels per task ----
    HD, DLAB, INST = {}, {}, {}
    for task, total in TASKS.items():
        INST[task] = load_instances(task)[:total]
        if task in ("jsonschema", "gsm8k"):
            z = np.load(f"/results/{task}_dream/transport_alllayers.npz")
            HD[task] = z["H_D"]; DLAB[task] = z["dream_labels"]
            print(f"{task}: loaded cached H_D {HD[task].shape}")

    # MBPP needs Dream-Base extraction
    if "mbpp" in TASKS:
        with open("/results/mbpp_dream/sae_diagnose_stage2.json") as f:
            diag = json.load(f)
        fail = set(i for c in diag.get("clusters", []) for i in c.get("fail_sample_indices", []))
        DLAB["mbpp"] = np.array([0 if i in fail else 1 for i in range(TASKS["mbpp"])])
        itok = AutoTokenizer.from_pretrained(DREAM_INSTRUCT, trust_remote_code=True)
        prompts = []
        for inst in INST["mbpp"]:
            msgs = [{"role": "system", "content": sys_prompt("mbpp", inst)},
                    {"role": "user", "content": user_prompt("mbpp", inst)}]
            prompts.append(itok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
        dtok = AutoTokenizer.from_pretrained(DREAM_BASE, trust_remote_code=True)
        dream = AutoModel.from_pretrained(DREAM_BASE, torch_dtype=torch.bfloat16,
                                          trust_remote_code=True).cuda().eval()
        layers = dream.model.layers if hasattr(dream, "model") else dream.transformer.h
        nL = len(layers); dh = dream.config.hidden_size
        cap = [None] * nL
        hs = [layers[i].register_forward_hook(
            (lambda i: (lambda m, a, o: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach())))(i))
            for i in range(nL)]
        H = np.zeros((len(prompts), nL, dh), np.float32)
        for k, p in enumerate(prompts):
            ids = dtok(p, return_tensors="pt").input_ids.cuda(); pl = ids.shape[1]
            full = torch.full((1, pl + GEN_LEN["mbpp"]), MASK_ID, dtype=ids.dtype, device=ids.device)
            full[0, :pl] = ids[0]
            for i in range(nL):
                cap[i] = None
            with torch.no_grad():
                dream(full)
            for i in range(nL):
                H[k, i] = cap[i][0, pl - 1, :].float().cpu().numpy()
        for h in hs:
            h.remove()
        HD["mbpp"] = H; del dream; torch.cuda.empty_cache()
        print(f"mbpp: extracted H_D {H.shape}")

    # ---- Llama generation + grading per task ----
    ltok = AutoTokenizer.from_pretrained(GEN_MODEL)
    ltok.padding_side = "left"
    if ltok.pad_token is None:
        ltok.pad_token = ltok.eos_token
    llama = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    third = {}
    for task, total in TASKS.items():
        texts = []
        for inst in INST[task]:
            msgs = [{"role": "system", "content": sys_prompt(task, inst)},
                    {"role": "user", "content": user_prompt(task, inst)}]
            t = ltok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            if task == "jsonschema":
                t += "```json\n"
            texts.append(t)
        passed = [0] * total
        BS = 16
        for s in range(0, total, BS):
            enc = ltok(texts[s:s+BS], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = llama.generate(**enc, max_new_tokens=GEN_LEN[task], do_sample=True,
                                     temperature=0.2, top_p=0.95, pad_token_id=ltok.pad_token_id)
            gen = out[:, enc.input_ids.shape[1]:]
            for j in range(gen.shape[0]):
                txt = ltok.decode(gen[j], skip_special_tokens=True)
                if task == "jsonschema":
                    txt = "```json\n" + txt
                passed[s+j] = int(grade(task, INST[task][s+j], txt))
        third[task] = np.array(passed)
        print(f"{task}: Llama pass {third[task].sum()}/{total}={third[task].mean():.2%}")

    # ---- transport ----
    def cv_auc(X, y, C=0.01):
        if (y == 0).sum() < 5 or (y == 1).sum() < 5:
            return float("nan")
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        return float(np.mean([roc_auc_score(y[te], LogisticRegression(max_iter=2000, C=C)
                     .fit(X[tr], y[tr]).decision_function(X[te])) for tr, te in skf.split(X, y)]))
    out = {}
    for task in TASKS:
        H = HD[task]; dl = DLAB[task]; tl = third[task]
        rows = [{"layer": li, "auc_dream": cv_auc(H[:, li, :], dl),
                 "auc_third": cv_auc(H[:, li, :], tl)} for li in range(H.shape[1])]
        valid = [r for r in rows if r["auc_dream"] == r["auc_dream"]]
        best = max(valid, key=lambda r: r["auc_dream"])
        out[task] = {"gen_model": GEN_MODEL, "third_pass": int(tl.sum()), "n": len(tl),
                     "agreement_with_dream": float((dl == tl).mean()),
                     "best_dream_layer": best, "per_layer": rows}
        print(f"\n[{task}] third pass {tl.sum()}/{len(tl)} agree(dream)={out[task]['agreement_with_dream']:.3f}"
              f"  best L{best['layer']}: dream {best['auc_dream']:.3f} third {best['auc_third']:.3f}")
    with open("/results/mbpp_dream/thirdgen_transport.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    return json.dumps({t: out[t]["best_dream_layer"] for t in out})


@app.local_entrypoint()
def main():
    print(run.remote())
