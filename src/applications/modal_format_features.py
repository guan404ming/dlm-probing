"""Format-features pass for shallow-cue control (review response).

Re-runs LLaDA / Dream generation (same seed=0 as midstep probe) without
extracting hidden states or entropy. For each instance, it records simple
surface-level features of the generated output (presence of format markers,
predicted answer letter for ARC, output length). These features power a
"shallow probe" baseline that tests whether the hidden-state probe is merely
picking up surface cues like formatting or answer patterns.

Saves:
  /results/{dataset}_{model}/format_chunk_off{offset}.npz
    labels: (n,)
    feat_names: array of str (feature names, same for all chunks of a dataset)
    feats: (n, n_features) float32

Usage:
  ../.venv/bin/modal run modal_format_features.py --dataset jsonschema --model llada --chunks 8
"""

import modal
import re

app = modal.App("probe-format-features")

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
        "scikit-learn",
    )
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)

MODEL_CFGS = {
    "llada": {"name": "GSAI-ML/LLaDA-8B-Instruct", "mask_id": 126336, "temperature": 0.2},
    "dream": {"name": "Dream-org/Dream-v0-Instruct-7B", "mask_id": 151666, "temperature": 0.0},
}

DATASET_CFGS = {
    "jsonschema": {"gen_length": 256, "total": 272},
    "gsm8k": {"gen_length": 512, "total": 1319},
    "mbpp": {"gen_length": 256, "total": 257},
    "arc": {"gen_length": 256, "total": 1172},
}

STEPS = 128
BLOCK_LENGTH = 32

FEATURE_NAMES = {
    "jsonschema": ["parses_as_json", "has_code_fence_open", "has_code_fence_close", "out_chars"],
    "gsm8k": ["has_####_marker", "has_any_number", "answer_chars", "out_chars"],
    "mbpp": ["has_python_block", "has_def_keyword", "has_code_fence", "n_lines", "out_chars"],
    "arc": ["has_####_letter", "letter_A", "letter_B", "letter_C", "letter_D", "out_chars"],
}


# ---- Dataset / prompt helpers (mirror modal_midstep_probe.py) ----

def load_instances(dataset_key, offset, limit):
    from datasets import load_dataset
    if dataset_key == "jsonschema":
        ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
        all_insts = sorted(list(ds), key=lambda x: x["instance_id"])
    elif dataset_key == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        all_insts = list(ds)
    elif dataset_key == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        all_insts = sorted(list(ds), key=lambda x: x["task_id"])
    elif dataset_key == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        all_insts = list(ds)
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    return all_insts[offset:offset + limit]


def build_system_prompt(dataset_key, instance):
    if dataset_key == "jsonschema":
        schema = instance["schema"]
        return (
            "You are a helpful assistant that answers in JSON. "
            "Here's the JSON schema you must adhere to:\n"
            f"<schema>\n{schema}\n</schema>\n"
        )
    elif dataset_key == "gsm8k":
        return (
            "Solve the math problem step by step. "
            "End your answer with #### followed by the final numeric answer."
        )
    elif dataset_key == "mbpp":
        return (
            "You are an expert Python programmer. "
            "Write a Python function that solves the given task. "
            "Output only the function definition, no explanations."
        )
    else:
        return (
            "Answer the multiple choice question. "
            "Think step by step, then give your final answer as "
            "#### followed by a single letter (A, B, C, or D)."
        )


def build_user_prompt(dataset_key, instance):
    if dataset_key == "jsonschema":
        return instance["input"]
    elif dataset_key == "gsm8k":
        return instance["question"]
    elif dataset_key == "mbpp":
        tests_str = "\n".join(instance["test_list"])
        return f"{instance['prompt']}\n\nYour code should pass these tests:\n{tests_str}"
    else:
        choices = instance["choices"]
        choices_str = "\n".join(
            f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])
        )
        return f"{instance['question']}\n\n{choices_str}"


def build_prompt_llada(tokenizer, model, dataset_key, instance):
    import torch
    sys_prompt = build_system_prompt(dataset_key, instance)
    user_prompt = build_user_prompt(dataset_key, instance)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if dataset_key == "jsonschema":
        text += "```json\n"
    ids = tokenizer(text)["input_ids"]
    return torch.tensor(ids, device=model.device).unsqueeze(0), None


def build_prompt_dream(tokenizer, model, dataset_key, instance):
    sys_prompt = build_system_prompt(dataset_key, instance)
    user_prompt = build_user_prompt(dataset_key, instance)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if dataset_key == "jsonschema":
        messages.append({"role": "assistant", "content": "```json\n"})
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            continue_final_message=True,
        )
    else:
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        )
    return (inputs.input_ids.to(device=model.device),
            inputs.attention_mask.to(device=model.device))


# ---- Functional check (mirror modal_midstep_probe.py) ----

def check_functional(dataset_key, instance, output_text):
    import json as json_mod
    if dataset_key == "jsonschema":
        prefix = "```json\n"
        extracted = output_text
        if prefix in output_text:
            extracted = output_text.split(prefix, 1)[-1]
        end = extracted.find("```")
        if end != -1:
            extracted = extracted[:end]
        extracted = extracted.strip().strip("`") + "\n"
        try:
            ref = json_mod.loads(instance["output"])
            ref_str = json_mod.dumps(ref, indent=4)
            gen = json_mod.loads(extracted)
            gen_str = json_mod.dumps(gen, indent=4)
            return ref_str == gen_str
        except (json_mod.JSONDecodeError, ValueError):
            return False
    elif dataset_key == "gsm8k":
        pred = _extract_gsm8k_answer(output_text)
        gold = _extract_gsm8k_gold(instance["answer"])
        if pred is None:
            return False
        try:
            return float(pred) == float(gold)
        except ValueError:
            return pred.strip() == gold.strip()
    elif dataset_key == "mbpp":
        return _check_mbpp(output_text, instance)
    else:
        return _check_arc(output_text, instance)


def _extract_gsm8k_answer(text):
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _extract_gsm8k_gold(answer_text):
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    raise ValueError(f"No answer found in: {answer_text}")


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _check_mbpp(output_text, instance):
    import signal
    code = _extract_code(output_text)
    test_imports = instance.get("test_imports", "") or ""
    if isinstance(test_imports, list):
        test_imports = "\n".join(test_imports)
    tests = instance["test_list"]
    exec_code = ""
    if test_imports:
        exec_code += test_imports + "\n"
    exec_code += code + "\n"
    for test in tests:
        exec_code += test + "\n"

    def _timeout_handler(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(10)
    try:
        exec(exec_code, {"__builtins__": __builtins__}, {})
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _check_arc(output_text, instance):
    gold = instance["answerKey"].strip().upper()
    m = re.search(r"####\s*([A-Da-d])", output_text)
    if m:
        return m.group(1).upper() == gold
    matches = re.findall(r"\b([A-Da-d])\b", output_text)
    if matches:
        return matches[-1].upper() == gold
    return False


# ---- Format feature extraction ----

def extract_format_features(dataset_key, output_text):
    """Return a list of float features matching FEATURE_NAMES[dataset_key]."""
    if dataset_key == "jsonschema":
        import json as json_mod
        # parses_as_json: take the part inside the first code fence (or whole text)
        prefix = "```json\n"
        extracted = output_text
        if prefix in output_text:
            extracted = output_text.split(prefix, 1)[-1]
        end = extracted.find("```")
        if end != -1:
            extracted = extracted[:end]
        extracted = extracted.strip().strip("`") + "\n"
        try:
            json_mod.loads(extracted)
            parses = 1.0
        except (json_mod.JSONDecodeError, ValueError):
            parses = 0.0
        has_open = 1.0 if "```json" in output_text else 0.0
        has_close = 1.0 if output_text.count("```") >= 2 else 0.0
        return [parses, has_open, has_close, float(len(output_text))]

    if dataset_key == "gsm8k":
        has_marker = 1.0 if re.search(r"####\s*[+-]?[\d,]+", output_text) else 0.0
        has_num = 1.0 if re.search(r"[+-]?[\d,]+\.?\d*", output_text) else 0.0
        # answer_chars: chars in fragment after ####, capped at 32
        m = re.search(r"####\s*(.+)", output_text)
        ac = float(min(len(m.group(1)) if m else 0, 32))
        return [has_marker, has_num, ac, float(len(output_text))]

    if dataset_key == "mbpp":
        has_block = 1.0 if re.search(r"```(?:python)?\s*\n.*?```", output_text, re.DOTALL) else 0.0
        has_def = 1.0 if re.search(r"\bdef\s+\w+\s*\(", output_text) else 0.0
        has_fence = 1.0 if "```" in output_text else 0.0
        return [has_block, has_def, has_fence, float(output_text.count("\n")), float(len(output_text))]

    # arc
    m = re.search(r"####\s*([A-Da-d])", output_text)
    has_marker = 1.0 if m else 0.0
    if m:
        letter = m.group(1).upper()
    else:
        m2 = re.findall(r"\b([A-Da-d])\b", output_text)
        letter = m2[-1].upper() if m2 else ""
    one_hot = [1.0 if letter == c else 0.0 for c in ("A", "B", "C", "D")]
    return [has_marker] + one_hot + [float(len(output_text))]


# ---- Generation ----

@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk_format(dataset_key: str, model_key: str, offset: int, limit: int):
    import os
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    cfg = MODEL_CFGS[model_key]
    dcfg = DATASET_CFGS[dataset_key]
    MODEL_NAME = cfg["name"]
    MASK_ID = cfg["mask_id"]
    TEMPERATURE = cfg["temperature"]
    GEN_LENGTH = dcfg["gen_length"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    instances = load_instances(dataset_key, offset, limit)

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

    def generate_llada(x, gen_start):
        num_blocks = GEN_LENGTH // BLOCK_LENGTH
        steps_per_block = STEPS // num_blocks
        for num_block in range(num_blocks):
            block_start = gen_start + num_block * BLOCK_LENGTH
            block_end = gen_start + (num_block + 1) * BLOCK_LENGTH
            block_mask_index = (x[:, block_start:block_end] == MASK_ID)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            for si in range(steps_per_block):
                out = model(x, output_hidden_states=False)
                logits = out.logits
                logits_with_noise = add_gumbel_noise(logits, temperature=TEMPERATURE)
                n_transfer = num_transfer_tokens[0, si].item()
                if n_transfer == 0:
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
                n_unmask = min(n_transfer, mask_index[0, block_start:block_end].sum().item())
                if n_unmask > 0:
                    _, indices = torch.topk(confidence[0], k=n_unmask)
                    x[0, indices] = x0[0, indices]
        return x

    def generate_dream(x, gen_start, attention_mask):
        EPS = 1e-3
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(
                attention_mask, (0, x.shape[1] - attention_mask.shape[1]), value=1.0
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, EPS, STEPS + 1, device=x.device)
        with torch.no_grad():
            for i in range(STEPS):
                mask_index = x == MASK_ID
                out = model(x, attention_mask, tok_idx, output_hidden_states=False)
                logits = out.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                mask_logits = logits[mask_index]
                t = timesteps[i]
                s = timesteps[i + 1]
                confidence, x0 = mask_logits.max(dim=-1)
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = (
                    int(num_mask_token * (1 - s / t))
                    if i < STEPS - 1
                    else int(num_mask_token)
                )
                full_confidence = torch.full_like(
                    x, -torch.inf, device=model.device, dtype=logits.dtype,
                )
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    _, transfer_index = torch.topk(
                        full_confidence, number_transfer_tokens,
                    )
                    x_ = torch.zeros_like(x) + MASK_ID
                    x_[mask_index] = x0.clone()
                    row_indices = (
                        torch.arange(x.size(0), device=model.device)
                        .unsqueeze(1).expand_as(transfer_index)
                    )
                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]
        return x

    feats = []
    labels = []

    print(f"Chunk offset={offset}, limit={limit}, "
          f"dataset={dataset_key}, model={model_key}")
    t_start = time.monotonic()

    for i, inst in enumerate(instances):
        if model_key == "llada":
            prompt_ids, _ = build_prompt_llada(tokenizer, model, dataset_key, inst)
        else:
            prompt_ids, attn_mask = build_prompt_dream(tokenizer, model, dataset_key, inst)
        gen_start = prompt_ids.shape[1]

        torch.manual_seed(0)
        t0 = time.monotonic()
        x = torch.full((1, gen_start + GEN_LENGTH), MASK_ID,
                       dtype=torch.long, device=model.device)
        x[:, :gen_start] = prompt_ids.clone()

        if model_key == "llada":
            x = generate_llada(x, gen_start)
        else:
            x = generate_dream(x, gen_start, attn_mask)

        output_text = tokenizer.batch_decode(
            x[:, gen_start:], skip_special_tokens=True,
        )[0]
        functional = check_functional(dataset_key, inst, output_text)
        labels.append(int(functional))
        feats.append(extract_format_features(dataset_key, output_text))

        elapsed = time.monotonic() - t0
        if (i + 1) % 10 == 0 or i == len(instances) - 1:
            n_func = sum(labels)
            print(f"  [{i+1}/{len(instances)}] functional={n_func}/{len(labels)} "
                  f"({100*n_func/len(labels):.1f}%), time={elapsed:.1f}s")

    total_time = time.monotonic() - t_start
    if not labels:
        return f"Chunk off={offset}: 0 samples"

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        f"{out_dir}/format_chunk_off{offset}.npz",
        labels=np.array(labels),
        feats=np.array(feats, dtype=np.float32),
        feat_names=np.array(FEATURE_NAMES[dataset_key]),
    )
    n_func = sum(labels)
    summary = (f"Chunk off={offset}: {len(labels)} samples, "
               f"{n_func} functional ({100*n_func/len(labels):.1f}%), "
               f"{total_time:.0f}s")
    print(summary)
    RESULTS_VOL.commit()
    return summary


# ---- Analysis ----

@app.function(
    image=image,
    timeout=1800,
    volumes={"/results": RESULTS_VOL},
)
def run_format_analysis(dataset_key: str, model_key: str, n_chunks: int, total: int):
    import json
    import os
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    RESULTS_VOL.reload()

    chunk_size = (total + n_chunks - 1) // n_chunks
    in_dir = f"/results/{dataset_key}_{model_key}"

    all_labels = []
    all_feats = []
    feat_names = None
    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/format_chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_labels.append(data["labels"])
        all_feats.append(data["feats"])
        if feat_names is None:
            feat_names = list(data["feat_names"])

    labels = np.concatenate(all_labels)
    feats = np.concatenate(all_feats, axis=0)
    n_samples = len(labels)
    n_func = int(labels.sum())
    print(f"Loaded {n_samples} samples, {n_func} functional "
          f"({100*n_func/n_samples:.1f}%), feature dim={feats.shape[1]}")
    print(f"Features: {feat_names}")

    # Univariate AUC for each format feature
    per_feat_auc = {}
    for j, name in enumerate(feat_names):
        v = feats[:, j]
        if v.std() < 1e-9:
            per_feat_auc[name] = 0.5
            continue
        try:
            a = roc_auc_score(labels, v)
            per_feat_auc[name] = max(a, 1 - a)
        except ValueError:
            per_feat_auc[name] = 0.5
    print("\nPer-feature univariate AUC (max of A, 1-A):")
    for k, v in per_feat_auc.items():
        print(f"  {k:30s}: {v:.3f}")

    # Multivariate shallow probe (LR on all format features)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(feats, labels):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        )
        clf.fit(feats[train_idx], labels[train_idx])
        prob = clf.predict_proba(feats[test_idx])[:, 1]
        try:
            aucs.append(roc_auc_score(labels[test_idx], prob))
        except ValueError:
            aucs.append(0.5)
    shallow_auc = float(np.mean(aucs))
    shallow_auc_std = float(np.std(aucs))
    print(f"\nShallow probe (LR on all format features): "
          f"AUC = {shallow_auc:.3f} +/- {shallow_auc_std:.3f}")

    # ARC-specific: predicted-letter-only probe
    letter_only_auc = None
    if dataset_key == "arc":
        letter_idx = [feat_names.index(c) for c in ["letter_A", "letter_B", "letter_C", "letter_D"]]
        X = feats[:, letter_idx]
        aucs = []
        for train_idx, test_idx in skf.split(X, labels):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
            )
            clf.fit(X[train_idx], labels[train_idx])
            prob = clf.predict_proba(X[test_idx])[:, 1]
            try:
                aucs.append(roc_auc_score(labels[test_idx], prob))
            except ValueError:
                aucs.append(0.5)
        letter_only_auc = float(np.mean(aucs))
        print(f"ARC predicted-letter-only probe: AUC = {letter_only_auc:.3f}")

    results = {
        "dataset": dataset_key,
        "model": model_key,
        "n_samples": n_samples,
        "n_functional": n_func,
        "feat_names": feat_names,
        "per_feat_auc": {k: round(v, 4) for k, v in per_feat_auc.items()},
        "shallow_probe_auc": round(shallow_auc, 4),
        "shallow_probe_auc_std": round(shallow_auc_std, 4),
    }
    if letter_only_auc is not None:
        results["letter_only_auc"] = round(letter_only_auc, 4)
    out_path = f"{in_dir}/format_features_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()

    print(f"\nResults saved to {out_path}")
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(
    dataset: str = "jsonschema",
    model: str = "llada",
    chunks: int = 8,
    total: int = 0,
    skip_gen: bool = False,
):
    if model not in MODEL_CFGS:
        raise ValueError(f"Unknown model: {model}")
    if dataset not in DATASET_CFGS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if total <= 0:
        total = DATASET_CFGS[dataset]["total"]

    chunk_size = (total + chunks - 1) // chunks
    print(f"Format-features pass: dataset={dataset}, model={model}, "
          f"{chunks}x A100, total={total}, chunk_size={chunk_size}")

    if not skip_gen:
        handles = []
        for i in range(chunks):
            offset = i * chunk_size
            limit = min(chunk_size, total - offset)
            if limit <= 0:
                break
            print(f"  Chunk {i}: offset={offset}, limit={limit}")
            handles.append(run_chunk_format.spawn(dataset, model, offset, limit))
        for handle in handles:
            print(f"  Done: {handle.get()}")

    print("\nRunning format analysis...")
    print(run_format_analysis.remote(dataset, model, chunks, total))
