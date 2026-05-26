"""Cross-generator transport replication on JSON-schema and GSM8K.

Reviewer ask: replicate the MBPP cross-generator difficulty test on another task.
If Dream's hidden states encode problem difficulty (not Dream-specific generation
correctness), a probe trained on Dream features should predict Qwen-Instruct's
pass/fail about as well as it predicts Dream's own pass/fail.

For a given task we:
  1. load Dream-Instruct fail/pass labels from the cached diagnose,
  2. extract Dream-Base all-layer last-prompt-token hidden states (mask-appended),
  3. generate + grade Qwen-2.5-7B-Instruct solutions (its own labels),
  4. report per-layer AUC(Dream feats -> Dream labels) vs AUC(Dream feats -> Qwen labels).

Usage:
  .venv/bin/modal run --detach src/applications/sae/modal_transport.py --dataset jsonschema
  .venv/bin/modal run --detach src/applications/sae/modal_transport.py --dataset gsm8k
"""

import modal

app = modal.App("transport-crossgen")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0", "transformers==4.52.2", "accelerate>=0.30",
        "numpy", "datasets==2.21.0", "huggingface_hub", "scikit-learn",
    )
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

DREAM_BASE = "Dream-org/Dream-v0-Base-7B"
DREAM_INSTRUCT = "Dream-org/Dream-v0-Instruct-7B"
QWEN_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
MASK_ID = 151666
DATASET_TOTALS = {"jsonschema": 272, "gsm8k": 1319}
GEN_LEN = {"jsonschema": 256, "gsm8k": 512}


# ---- prompts / graders (mirror src/core/modal_entropy_baseline.py) ----
def load_instances(dataset_key):
    from datasets import load_dataset
    if dataset_key == "jsonschema":
        ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
        return sorted(list(ds), key=lambda x: x["instance_id"])
    if dataset_key == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return list(ds)
    raise ValueError(dataset_key)


def build_system_prompt(dataset_key, instance):
    if dataset_key == "jsonschema":
        return ("You are a helpful assistant that answers in JSON. "
                "Here's the JSON schema you must adhere to:\n"
                f"<schema>\n{instance['schema']}\n</schema>\n")
    return ("Solve the math problem step by step. "
            "End your answer with #### followed by the final numeric answer.")


def build_user_prompt(dataset_key, instance):
    if dataset_key == "jsonschema":
        return instance["input"]
    return instance["question"]


def check_functional(dataset_key, instance, output_text):
    import json as J
    import re
    if dataset_key == "jsonschema":
        extracted = output_text
        if "```json\n" in output_text:
            extracted = output_text.split("```json\n", 1)[-1]
        end = extracted.find("```")
        if end != -1:
            extracted = extracted[:end]
        extracted = extracted.strip().strip("`") + "\n"
        try:
            ref = J.dumps(J.loads(instance["output"]), indent=4)
            gen = J.dumps(J.loads(extracted), indent=4)
            return ref == gen
        except (J.JSONDecodeError, ValueError):
            return False
    # gsm8k
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", output_text)
    pred = m.group(1).replace(",", "") if m else None
    if pred is None:
        nums = re.findall(r"[+-]?[\d,]+\.?\d*", output_text)
        pred = nums[-1].replace(",", "") if nums else None
    g = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", instance["answer"])
    gold = g.group(1).replace(",", "") if g else None
    if pred is None or gold is None:
        return False
    try:
        return float(pred) == float(gold)
    except ValueError:
        return pred.strip() == gold.strip()


@app.function(image=image, gpu="A100-80GB", timeout=14400,
              volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL})
def run(dataset: str = "jsonschema"):
    import json, os, numpy as np, torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"
    total = DATASET_TOTALS[dataset]
    gen_len = GEN_LEN[dataset]
    outdir = f"/results/{dataset}_dream"

    # 1. Dream-Instruct fail/pass labels from cached diagnose
    with open(f"{outdir}/sae_diagnose_stage2.json") as f:
        diag = json.load(f)
    fail_idxs = []
    for c in diag.get("clusters", []):
        fail_idxs.extend(c.get("fail_sample_indices", []))
    fail_set = set(dict.fromkeys(fail_idxs))
    dream_labels = np.array([0 if i in fail_set else 1 for i in range(total)])
    print(f"[{dataset}] total={total} dream_fail={len(fail_set)} dream_pass={total-len(fail_set)}")

    instances = load_instances(dataset)[:total]
    instruct_tok = AutoTokenizer.from_pretrained(DREAM_INSTRUCT, trust_remote_code=True)
    prompts_text = []
    for inst in instances:
        msgs = [{"role": "system", "content": build_system_prompt(dataset, inst)},
                {"role": "user", "content": build_user_prompt(dataset, inst)}]
        prompts_text.append(instruct_tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False))

    def find_layers(model):
        for path in [("model", "layers"), ("model", "transformer", "h"), ("transformer", "h")]:
            cur = model; ok = True
            for a in path:
                if hasattr(cur, a):
                    cur = getattr(cur, a)
                else:
                    ok = False; break
            if ok:
                return cur
        raise RuntimeError("no layers")

    def collect_all_layers(model, tok, prompts, mask_token=None, gen_mask_len=0, label="m"):
        layers = find_layers(model); n_l = len(layers); d_h = model.config.hidden_size
        captured = [None] * n_l
        def mk(li):
            def hook(mod, args, out):
                captured[li] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            return hook
        handles = [layers[i].register_forward_hook(mk(i)) for i in range(n_l)]
        H = np.zeros((len(prompts), n_l, d_h), dtype=np.float32)
        for k, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            pl = ids.shape[1]
            if mask_token is not None and gen_mask_len > 0:
                full = torch.full((1, pl + gen_mask_len), mask_token, dtype=ids.dtype, device=ids.device)
                full[0, :pl] = ids[0]; inp = full
            else:
                inp = ids
            for i in range(n_l):
                captured[i] = None
            with torch.no_grad():
                _ = model(inp)
            for i in range(n_l):
                H[k, i] = captured[i][0, pl - 1, :].float().cpu().numpy()
            if (k + 1) % 100 == 0:
                print(f"  {label} {k+1}/{len(prompts)}")
        for h in handles:
            h.remove()
        return H

    # 2. Dream-Base all-layer hidden states
    print("=== Dream-Base hidden states ===")
    dream_tok = AutoTokenizer.from_pretrained(DREAM_BASE, trust_remote_code=True)
    dream = AutoModel.from_pretrained(DREAM_BASE, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).cuda().eval()
    H_D = collect_all_layers(dream, dream_tok, prompts_text,
                             mask_token=MASK_ID, gen_mask_len=gen_len, label="Dream")
    del dream; torch.cuda.empty_cache()
    print(f"H_D {H_D.shape}")

    # 3. Qwen-Instruct generation + grading (batched) + hidden states
    print("=== Qwen-Instruct generate + grade ===")
    qtok = AutoTokenizer.from_pretrained(QWEN_INSTRUCT, trust_remote_code=True)
    qtok.padding_side = "left"
    if qtok.pad_token is None:
        qtok.pad_token = qtok.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_INSTRUCT, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    qwen_texts = []
    for inst in instances:
        msgs = [{"role": "system", "content": build_system_prompt(dataset, inst)},
                {"role": "user", "content": build_user_prompt(dataset, inst)}]
        t = qtok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if dataset == "jsonschema":
            t += "```json\n"
        qwen_texts.append(t)
    qwen_passed = [0] * total
    BS = 16
    for s in range(0, total, BS):
        batch = qwen_texts[s:s + BS]
        enc = qtok(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = qwen.generate(**enc, max_new_tokens=gen_len, do_sample=True,
                                temperature=0.2, top_p=0.95, pad_token_id=qtok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        for j in range(len(batch)):
            txt = qtok.decode(gen[j], skip_special_tokens=True)
            if dataset == "jsonschema":
                txt = "```json\n" + txt
            qwen_passed[s + j] = int(check_functional(dataset, instances[s + j], txt))
        if (s + BS) % 160 == 0 or s + BS >= total:
            print(f"  gen {min(s+BS,total)}/{total} pass={sum(qwen_passed)}")
    qwen_labels = np.array(qwen_passed)
    print(f"Qwen-Instruct pass {qwen_labels.sum()}/{total}={qwen_labels.mean():.2%}")
    del qwen; torch.cuda.empty_cache()

    np.savez(f"{outdir}/transport_alllayers.npz",
             H_D=H_D, dream_labels=dream_labels, qwen_labels=qwen_labels)
    RESULTS_VOL.commit()

    # 4. per-layer transport AUC
    def cv_auc(X, y, C=0.01):
        if (y == 0).sum() < 5 or (y == 1).sum() < 5:
            return float("nan")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        a = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=C).fit(X[tr], y[tr])
            a.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        return float(np.mean(a))
    agree = float((dream_labels == qwen_labels).mean())
    rows = []
    print(f"\nagreement={agree:.3f}\n L  AUC|Dream  AUC|Qwen   diff")
    for li in range(H_D.shape[1]):
        f = H_D[:, li, :]
        md = cv_auc(f, dream_labels); mq = cv_auc(f, qwen_labels)
        rows.append({"layer": li, "auc_dream": md, "auc_qwen": mq, "diff": mq - md})
        print(f" {li:2d}  {md:.3f}     {mq:.3f}    {mq-md:+.3f}")
    best = max((r for r in rows if r["auc_dream"] == r["auc_dream"]),
               key=lambda r: r["auc_dream"])
    out = {"dataset": dataset, "total": total, "agreement": agree,
           "dream_pass": int(dream_labels.sum()), "qwen_pass": int(qwen_labels.sum()),
           "best_dream_layer": best, "per_layer": rows}
    with open(f"{outdir}/transport_results.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nbest Dream layer L{best['layer']}: dream {best['auc_dream']:.3f} "
          f"qwen {best['auc_qwen']:.3f} diff {best['diff']:+.3f}")
    return json.dumps({"dataset": dataset, "agree": agree, "best": best})


@app.local_entrypoint()
def main(dataset: str = "jsonschema"):
    print(run.remote(dataset))
