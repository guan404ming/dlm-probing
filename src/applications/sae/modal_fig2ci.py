"""Per-seed signal-to-null gap for Figure 2 confidence intervals (reviewer #6).

Regenerates a (model, task) cell under extra seeds and recomputes the sparse-grid
silhouette gap per checkpoint step, so Figure 2 can show seed variability rather
than a single-seed point. Reuses the generation/capture of modal_dense_sweep and
the prompts/graders of modal_entropy_baseline.

Usage:
  .venv/bin/modal run src/applications/sae/modal_fig2ci.py --model llada --dataset mbpp --seeds 1,2
"""

import modal

app = modal.App("sae-fig2ci")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "curl", "build-essential")
         .pip_install("torch>=2.0", "transformers==4.52.2", "accelerate>=0.30",
                      "numpy", "datasets==2.21.0", "huggingface_hub", "scikit-learn"))
RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL_CFGS = {
    "llada": {"name": "GSAI-ML/LLaDA-8B-Instruct", "mask_id": 126336, "temperature": 0.2,
              "sae_repo": "AwesomeInterpretability/llada-mask-topk-sae",
              "sae_path": "resid_post_layer_26/trainer_2", "sae_layer": 26},
    "dream": {"name": "Dream-org/Dream-v0-Instruct-7B", "mask_id": 151666, "temperature": 0.0,
              "sae_repo": "AwesomeInterpretability/dlm-mask-topk-sae",
              "sae_path": "saes_mask_Dream-org_Dream-v0-Base-7B_top_k/resid_post_layer_23/trainer_2",
              "sae_layer": 23},
}
DATASET_CFGS = {"mbpp": 257, "jsonschema": 272, "gsm8k": 1319, "arc": 1172}
GEN_LENGTH = 256
STEPS = 128
BLOCK_LENGTH = 32
N_REGIONS = 4
SPARSE_STEPS = [4, 16, 32, 64, 127]   # drop 0,1 (mask-geometry dominated, as in paper)
TOP_N = 20
N_PERM = 500


@app.function(image=image, gpu="A100", timeout=14400,
              volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL})
def run_cell(model_key: str, dataset: str, seeds: str = "1,2", limit: int = 0):
    import json, os, time
    import numpy as np, torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModel
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    os.environ["HF_HOME"] = "/hf-cache"
    RESULTS_VOL.reload()
    cfg = MODEL_CFGS[model_key]
    MASK_ID = cfg["mask_id"]; TEMP = cfg["temperature"]; SAE_LAYER = cfg["sae_layer"]
    region_size = GEN_LENGTH // N_REGIONS
    target = set(SPARSE_STEPS)
    total = limit or DATASET_CFGS[dataset]
    seed_list = [int(s) for s in seeds.split(",")]

    # ---- data / prompts / graders ----
    def load_insts():
        if dataset == "jsonschema":
            return sorted(list(load_dataset("eth-sri/json-mode-eval-extended", split="test")), key=lambda x: x["instance_id"])[:total]
        if dataset == "gsm8k":
            return list(load_dataset("openai/gsm8k", "main", split="test"))[:total]
        if dataset == "arc":
            return list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))[:total]
        return sorted(list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test")), key=lambda x: x["task_id"])[:total]

    def sysp(inst):
        if dataset == "jsonschema":
            return ("You are a helpful assistant that answers in JSON. Here's the JSON schema "
                    f"you must adhere to:\n<schema>\n{inst['schema']}\n</schema>\n")
        if dataset == "gsm8k":
            return "Solve the math problem step by step. End your answer with #### followed by the final numeric answer."
        if dataset == "arc":
            return ("Answer the multiple choice question. Think step by step, then give your final "
                    "answer as #### followed by a single letter (A, B, C, or D).")
        return ("You are an expert Python programmer. Write a Python function that solves the given "
                "task. Output only the function definition, no explanations.")

    def userp(inst):
        if dataset == "jsonschema":
            return inst["input"]
        if dataset == "gsm8k":
            return inst["question"]
        if dataset == "arc":
            ch = inst["choices"]
            return inst["question"] + "\n\n" + "\n".join(f"{l}. {t}" for l, t in zip(ch["label"], ch["text"]))
        return f"{inst['prompt']}\n\nYour code should pass these tests:\n" + "\n".join(inst["test_list"])

    def grade(inst, txt):
        import json as J, re, signal
        if dataset == "jsonschema":
            ext = txt.split("```json\n", 1)[-1] if "```json\n" in txt else txt
            e = ext.find("```"); ext = (ext[:e] if e != -1 else ext).strip().strip("`") + "\n"
            try:
                return J.dumps(J.loads(inst["output"]), indent=4) == J.dumps(J.loads(ext), indent=4)
            except (J.JSONDecodeError, ValueError):
                return False
        if dataset == "gsm8k":
            m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", txt)
            pred = m.group(1).replace(",", "") if m else (re.findall(r"[+-]?[\d,]+\.?\d*", txt)[-1].replace(",", "") if re.findall(r"[+-]?[\d,]+\.?\d*", txt) else None)
            g = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", inst["answer"]); gold = g.group(1).replace(",", "") if g else None
            try:
                return pred is not None and gold is not None and float(pred) == float(gold)
            except ValueError:
                return False
        if dataset == "arc":
            gold = inst["answerKey"].strip().upper()
            m = re.search(r"####\s*([A-Da-d])", txt)
            if m:
                return m.group(1).upper() == gold
            mm = re.findall(r"\b([A-Da-d])\b", txt)
            return bool(mm) and mm[-1].upper() == gold
        m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL)
        code = m.group(1) if m else txt
        ti = inst.get("test_imports", "") or ""
        ti = "\n".join(ti) if isinstance(ti, list) else ti
        full = (ti + "\n" if ti else "") + code + "\n" + "\n".join(inst["test_list"])
        old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError())); signal.alarm(10)
        try:
            exec(full, {}); return True
        except Exception:
            return False
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old)

    tok = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=True)
    model = AutoModel.from_pretrained(cfg["name"], device_map="auto", torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).eval()
    instances = load_insts()

    def build_prompt(inst):
        msgs = [{"role": "system", "content": sysp(inst)}, {"role": "user", "content": userp(inst)}]
        if model_key == "llada":
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if dataset == "jsonschema":
                text += "```json\n"
            return torch.tensor(tok(text)["input_ids"], device=model.device).unsqueeze(0), None
        if dataset == "jsonschema":
            msgs.append({"role": "assistant", "content": "```json\n"})
            inp = tok.apply_chat_template(msgs, return_tensors="pt", return_dict=True, continue_final_message=True)
        else:
            inp = tok.apply_chat_template(msgs, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        return inp.input_ids.to(model.device), inp.attention_mask.to(model.device)

    def capture(hs, gen_start):
        feats = {}
        for r in range(N_REGIONS):
            rs = gen_start + r * region_size; re_ = rs + region_size
            feats[r] = np.stack([hs[l][0, rs:re_].detach().float().mean(0).cpu().numpy() for l in range(len(hs))])
        return feats

    def gumbel(logits, t):
        if t == 0:
            return logits
        n = torch.rand_like(logits, dtype=torch.float64)
        return logits.to(torch.float64) + (-torch.log(-torch.log(n + 1e-20) + 1e-20)) * t

    def n_transfer(mask_index, steps):
        mn = mask_index.sum(1, keepdim=True); base = mn // steps; rem = mn % steps
        return torch.where(torch.arange(steps, device=mask_index.device).unsqueeze(0) < rem, base + 1, base)

    def gen_llada(x, gs):
        nb = GEN_LENGTH // BLOCK_LENGTH; spb = STEPS // nb; g = 0; sf = {}
        for b in range(nb):
            bs = gs + b * BLOCK_LENGTH; be = gs + (b + 1) * BLOCK_LENGTH
            bm = (x[:, bs:be] == MASK_ID); ntr = n_transfer(bm, spb)
            for si in range(spb):
                need = g in target
                out = model(x, output_hidden_states=need)
                if need and getattr(out, "hidden_states", None):
                    sf[g] = capture(out.hidden_states, gs)
                ln = gumbel(out.logits, TEMP); nt = ntr[0, si].item()
                if nt:
                    mi = x == MASK_ID; x0 = torch.argmax(ln, -1)
                    p = F.softmax(out.logits.to(torch.float64), -1)
                    x0p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                    x0p[:, :bs] = -np.inf; x0p[:, be:] = -np.inf
                    x0 = torch.where(mi, x0, x); conf = torch.where(mi, x0p, -np.inf)
                    nu = min(nt, mi[0, bs:be].sum().item())
                    if nu > 0:
                        _, idx = torch.topk(conf[0], nu); x[0, idx] = x0[0, idx]
                g += 1
        return x, sf

    def gen_dream(x, gs, am):
        EPS = 1e-3
        if am is not None and torch.any(am == 0.0):
            am = F.pad(am, (0, x.shape[1] - am.shape[1]), value=1.0)
            ti = am.long().cumsum(-1) - 1; ti.masked_fill_(am == 0, 1)
            am = torch.logical_and(am.unsqueeze(1).unsqueeze(-2), am.unsqueeze(1).unsqueeze(-1))
        else:
            ti = None; am = "full"
        ts = torch.linspace(1, EPS, STEPS + 1, device=x.device); sf = {}
        with torch.no_grad():
            for i in range(STEPS):
                mi = x == MASK_ID; need = i in target
                out = model(x, am, ti, output_hidden_states=need)
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], 1)
                if need and getattr(out, "hidden_states", None):
                    sf[i] = capture(out.hidden_states, gs)
                ml = logits[mi]; t = ts[i]; s = ts[i + 1]
                conf, x0 = ml.max(-1); nm = mi.sum() / mi.shape[0]
                ntt = int(nm * (1 - s / t)) if i < STEPS - 1 else int(nm)
                fc = torch.full_like(x, -torch.inf, dtype=logits.dtype); fc[mi] = conf
                if ntt > 0:
                    _, tr = torch.topk(fc, ntt)
                    xt = torch.zeros_like(x) + MASK_ID; xt[mi] = x0.clone()
                    rows = torch.arange(x.size(0), device=model.device).unsqueeze(1).expand_as(tr)
                    x[rows, tr] = xt[rows, tr]
        return x, sf

    # ---- SAE ----
    ae = hf_hub_download(repo_id=cfg["sae_repo"], filename=f"{cfg['sae_path']}/ae.pt", cache_dir="/hf-cache")
    sc = hf_hub_download(repo_id=cfg["sae_repo"], filename=f"{cfg['sae_path']}/config.json", cache_dir="/hf-cache")
    k = json.load(open(sc))["trainer"]["k"]
    st = torch.load(ae, map_location="cpu", weights_only=True)
    We = st["encoder.weight"].cuda().float(); be = st["encoder.bias"].cuda().float()
    bd = st.get("b_dec", st.get("decoder.bias")).cuda().float()

    def encode(x):
        xt = torch.from_numpy(x).cuda() - bd
        pre = (xt @ We.T + be) if We.shape[1] == x.shape[1] else (xt @ We + be)
        tv, ti = pre.topk(k, -1); tv = tv.relu()
        sp = torch.zeros_like(pre); sp.scatter_(-1, ti, tv); return sp.cpu().numpy()

    def diagnose(sae_mean, labels):
        active = (sae_mean > 0).astype(np.float32); fi = np.where(labels == 0)[0]
        if len(fi) < 6:
            return None
        enr = active[labels == 0].mean(0) - active[labels == 1].mean(0)
        top = np.argsort(enr)[::-1][:TOP_N]; sig = sae_mean[fi][:, top]; best = None
        for K in [2, 3, 4, 5]:
            try:
                sl = float(silhouette_score(sig, KMeans(K, random_state=42, n_init=10).fit_predict(sig)))
            except ValueError:
                sl = -1.0
            if best is None or sl > best[1]:
                best = (K, sl)
        rng = np.random.RandomState(42); null = []
        for _ in range(N_PERM):
            perm = rng.permutation(labels); fip = np.where(perm == 0)[0]
            if len(fip) < best[0] + 1:
                null.append(-1.0); continue
            tp = np.argsort(active[perm == 0].mean(0) - active[perm == 1].mean(0))[::-1][:TOP_N]
            try:
                null.append(float(silhouette_score(sae_mean[fip][:, tp], KMeans(best[0], random_state=42, n_init=5).fit_predict(sae_mean[fip][:, tp]))))
            except ValueError:
                null.append(-1.0)
        return best[1], float(np.mean(null)), best[0]

    out = {"model": model_key, "dataset": dataset, "sae_layer": SAE_LAYER, "seeds": {}}
    for seed in seed_list:
        t0 = time.time()
        feats = {s: {r: [] for r in range(N_REGIONS)} for s in SPARSE_STEPS}
        labels = []
        for inst in instances:
            pids, am = build_prompt(inst); gs = pids.shape[1]
            torch.manual_seed(seed)
            x = torch.full((1, gs + GEN_LENGTH), MASK_ID, dtype=torch.long, device=model.device)
            x[:, :gs] = pids.clone()
            with torch.no_grad():
                x, sf = (gen_llada(x, gs) if model_key == "llada" else gen_dream(x, gs, am))
            for s in SPARSE_STEPS:
                if s in sf:
                    for r in range(N_REGIONS):
                        feats[s][r].append(sf[s][r])
            gen = [t for t in x[0, gs:].tolist() if t != MASK_ID]
            labels.append(0 if not grade(inst, tok.decode(gen)) else 1)
        labels = np.array(labels)
        per_step = {}
        for s in SPARSE_STEPS:
            if not feats[s][0]:
                continue
            acts = [encode(np.stack(feats[s][r])[:, SAE_LAYER, :].astype(np.float32)) for r in range(N_REGIONS)]
            d = diagnose(np.mean(acts, 0), labels)
            if d:
                per_step[s] = {"silhouette": round(d[0], 4), "null_mean": round(d[1], 4),
                               "gap": round(d[0] - d[1], 4), "K": d[2]}
        out["seeds"][seed] = {"n_pass": int(labels.sum()), "n_fail": int((labels == 0).sum()),
                              "per_step": per_step}
        print(f"seed {seed} ({time.time()-t0:.0f}s) pass={labels.sum()}/{len(labels)}: "
              + " ".join(f"s{s}:gap{per_step[s]['gap']:+.3f}" for s in per_step))
    od = f"/results/{dataset}_{model_key}"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/fig2ci_seeds.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    return json.dumps(out)


@app.local_entrypoint()
def main(model: str = "llada", dataset: str = "mbpp", seeds: str = "1,2", limit: int = 0):
    print(run_cell.remote(model, dataset, seeds, limit))
