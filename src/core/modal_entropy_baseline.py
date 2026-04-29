"""Entropy / max-prob baseline for selective generation (review response).

Re-runs LLaDA / Dream generation (same seed=0 as midstep probe) and captures
per-checkpoint, per-region statistics on the model's own token confidence
(entropy and max softmax probability) over masked positions in the generation
region. Then evaluates these as alternative confidence signals for selective
generation, in direct comparison to the probe-based approach.

Saves:
  /results/{dataset}_{model}/entropy_chunk_off{offset}.npz
    labels: (n,)
    entropy_s{step}_r{region}: (n,)   mean entropy over masked positions
    maxprob_s{step}_r{region}: (n,)   mean max-prob over masked positions
    n_mask_s{step}_r{region}:  (n,)   count of masked positions
  /results/{dataset}_{model}/entropy_baseline_results.json

Usage:
  ../.venv/bin/modal run modal_entropy_baseline.py --dataset jsonschema --model llada --chunks 8
  ../.venv/bin/modal run modal_entropy_baseline.py --dataset gsm8k --model dream --chunks 8
"""

import modal
import re

app = modal.App("probe-entropy-baseline")

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
CHECKPOINT_STEPS = sorted([0, 1, 4, 16, 32, 64, STEPS - 1])
N_REGIONS = 4


# ---- Dataset / prompt / correctness helpers (mirrors modal_midstep_probe.py) ----

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


# ---- Entropy capture ----

def capture_entropy_features(logits, x, gen_start, gen_length, mask_id, n_regions, region_size):
    """Compute per-region (mean entropy, mean max_prob, n_mask) over masked positions.

    Args:
        logits: (1, seq_len, V) float tensor
        x: (1, seq_len) long tensor (current token state)
        gen_start: int, start index of generation region
        gen_length: int, length of generation region
        mask_id: int, mask token id
        n_regions: int (4)
        region_size: int (gen_length / n_regions)

    Returns:
        dict with keys "entropy", "maxprob", "n_mask", each shape (n_regions,)
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    gen_logits = logits[0, gen_start:gen_start + gen_length].float()  # (gen_length, V)
    gen_x = x[0, gen_start:gen_start + gen_length]
    mask_pos = (gen_x == mask_id)  # (gen_length,)

    # Compute entropy and max prob per position
    log_probs = F.log_softmax(gen_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # (gen_length,)
    max_prob = probs.max(dim=-1).values  # (gen_length,)

    out = {
        "entropy": np.zeros(n_regions, dtype=np.float32),
        "maxprob": np.zeros(n_regions, dtype=np.float32),
        "n_mask": np.zeros(n_regions, dtype=np.int32),
    }
    for r in range(n_regions):
        rs = r * region_size
        re_ = rs + region_size
        region_mask = mask_pos[rs:re_]
        n = int(region_mask.sum().item())
        out["n_mask"][r] = n
        if n > 0:
            out["entropy"][r] = entropy[rs:re_][region_mask].mean().item()
            out["maxprob"][r] = max_prob[rs:re_][region_mask].mean().item()
        else:
            # No masked positions left in region — region is fully committed.
            # Convention: zero entropy, max_prob = 1 (perfectly confident).
            out["entropy"][r] = 0.0
            out["maxprob"][r] = 1.0
    return out


# ---- Generation (mirrors midstep_probe but captures entropy instead of HS) ----

@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk_entropy(dataset_key: str, model_key: str, offset: int, limit: int):
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
    region_size = GEN_LENGTH // N_REGIONS

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
        global_step = 0
        ent_features = {}

        for num_block in range(num_blocks):
            block_start = gen_start + num_block * BLOCK_LENGTH
            block_end = gen_start + (num_block + 1) * BLOCK_LENGTH
            block_mask_index = (x[:, block_start:block_end] == MASK_ID)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for si in range(steps_per_block):
                out = model(x, output_hidden_states=False)
                logits = out.logits

                if global_step in CHECKPOINT_STEPS:
                    ent_features[global_step] = capture_entropy_features(
                        logits, x, gen_start, GEN_LENGTH, MASK_ID, N_REGIONS, region_size,
                    )

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
                n_unmask = min(n_transfer, mask_index[0, block_start:block_end].sum().item())
                if n_unmask > 0:
                    _, indices = torch.topk(confidence[0], k=n_unmask)
                    x[0, indices] = x0[0, indices]
                global_step += 1

        return x, ent_features

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
        ent_features = {}

        with torch.no_grad():
            for i in range(STEPS):
                mask_index = x == MASK_ID
                out = model(x, attention_mask, tok_idx, output_hidden_states=False)
                logits = out.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                if i in CHECKPOINT_STEPS:
                    ent_features[i] = capture_entropy_features(
                        logits, x, gen_start, GEN_LENGTH, MASK_ID, N_REGIONS, region_size,
                    )

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

        return x, ent_features

    # Per-checkpoint per-region scalar arrays
    ent_arr = {(s, r): [] for s in CHECKPOINT_STEPS for r in range(N_REGIONS)}
    mp_arr = {(s, r): [] for s in CHECKPOINT_STEPS for r in range(N_REGIONS)}
    nm_arr = {(s, r): [] for s in CHECKPOINT_STEPS for r in range(N_REGIONS)}
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
            x, ent_features = generate_llada(x, gen_start)
        else:
            x, ent_features = generate_dream(x, gen_start, attn_mask)

        # Fill missing checkpoints from final state (no masks remain → defaults).
        for s in CHECKPOINT_STEPS:
            if s not in ent_features:
                ent_features[s] = {
                    "entropy": [0.0] * N_REGIONS,
                    "maxprob": [1.0] * N_REGIONS,
                    "n_mask": [0] * N_REGIONS,
                }

        output_text = tokenizer.batch_decode(
            x[:, gen_start:], skip_special_tokens=True,
        )[0]
        functional = check_functional(dataset_key, inst, output_text)
        labels.append(int(functional))

        for s in CHECKPOINT_STEPS:
            for r in range(N_REGIONS):
                ent_arr[(s, r)].append(float(ent_features[s]["entropy"][r]))
                mp_arr[(s, r)].append(float(ent_features[s]["maxprob"][r]))
                nm_arr[(s, r)].append(int(ent_features[s]["n_mask"][r]))

        elapsed = time.monotonic() - t0
        if (i + 1) % 10 == 0 or i == len(instances) - 1:
            n_func = sum(labels)
            print(f"  [{i+1}/{len(instances)}] functional={n_func}/{len(labels)} "
                  f"({100*n_func/len(labels):.1f}%), time={elapsed:.1f}s")

    total_time = time.monotonic() - t_start

    if not labels:
        return f"Chunk off={offset}: 0 samples (skipped)"

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    save_dict = {"labels": np.array(labels)}
    for s in CHECKPOINT_STEPS:
        for r in range(N_REGIONS):
            save_dict[f"entropy_s{s}_r{r}"] = np.array(ent_arr[(s, r)], dtype=np.float32)
            save_dict[f"maxprob_s{s}_r{r}"] = np.array(mp_arr[(s, r)], dtype=np.float32)
            save_dict[f"n_mask_s{s}_r{r}"] = np.array(nm_arr[(s, r)], dtype=np.int32)
    np.savez_compressed(f"{out_dir}/entropy_chunk_off{offset}.npz", **save_dict)

    n_func = sum(labels)
    summary = (f"Chunk off={offset}: {len(labels)} samples, "
               f"{n_func} functional ({100*n_func/len(labels):.1f}%), "
               f"{total_time:.0f}s")
    print(summary)
    RESULTS_VOL.commit()
    return summary


# ---- Selective generation analysis ----

@app.function(
    image=image,
    timeout=1800,
    volumes={"/results": RESULTS_VOL},
)
def run_entropy_baseline_analysis(dataset_key: str, model_key: str, n_chunks: int, total: int):
    import json
    import os
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    RESULTS_VOL.reload()

    chunk_size = (total + n_chunks - 1) // n_chunks
    in_dir = f"/results/{dataset_key}_{model_key}"

    all_labels = []
    ent = {(s, r): [] for s in CHECKPOINT_STEPS for r in range(N_REGIONS)}
    mp = {(s, r): [] for s in CHECKPOINT_STEPS for r in range(N_REGIONS)}

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/entropy_chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_labels.append(data["labels"])
        for s in CHECKPOINT_STEPS:
            for r in range(N_REGIONS):
                ent[(s, r)].append(data[f"entropy_s{s}_r{r}"])
                mp[(s, r)].append(data[f"maxprob_s{s}_r{r}"])

    labels = np.concatenate(all_labels)
    n_samples = len(labels)
    n_func = int(labels.sum())
    print(f"Loaded {n_samples} samples, {n_func} functional ({100*n_func/n_samples:.1f}%)")

    # Stack per (step, region)
    ent_step_region = {(s, r): np.concatenate(ent[(s, r)]) for s in CHECKPOINT_STEPS for r in range(N_REGIONS)}
    mp_step_region = {(s, r): np.concatenate(mp[(s, r)]) for s in CHECKPOINT_STEPS for r in range(N_REGIONS)}

    # Pool across regions: mean over regions
    ent_step = {s: np.mean([ent_step_region[(s, r)] for r in range(N_REGIONS)], axis=0) for s in CHECKPOINT_STEPS}
    mp_step = {s: np.mean([mp_step_region[(s, r)] for r in range(N_REGIONS)], axis=0) for s in CHECKPOINT_STEPS}

    # ---- Per-step AUC of raw signals (no learning) ----
    raw_auc = {"entropy": {}, "maxprob": {}}
    for s in CHECKPOINT_STEPS:
        # Higher max-prob → more functional? Sign assumed positive but check both.
        try:
            auc_mp = roc_auc_score(labels, mp_step[s])
            # Higher entropy → less functional, so use -entropy.
            auc_ent = roc_auc_score(labels, -ent_step[s])
        except ValueError:
            auc_mp = 0.5
            auc_ent = 0.5
        raw_auc["maxprob"][s] = auc_mp
        raw_auc["entropy"][s] = auc_ent
        print(f"  Step {s:>3}: raw entropy AUC={auc_ent:.3f}, raw maxprob AUC={auc_mp:.3f}")

    # ---- LR on entropy features (mirrors probe pipeline structure) ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_step_aucs = {}
    lr_step_probs = {}
    for s in CHECKPOINT_STEPS:
        # Features: 2 * N_REGIONS = 8 dims (entropy, maxprob per region)
        X = np.stack(
            [ent_step_region[(s, r)] for r in range(N_REGIONS)]
            + [mp_step_region[(s, r)] for r in range(N_REGIONS)],
            axis=1,
        )
        probs = np.zeros(n_samples)
        aucs = []
        for train_idx, test_idx in skf.split(X, labels):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
            )
            clf.fit(X[train_idx], labels[train_idx])
            p = clf.predict_proba(X[test_idx])[:, 1]
            probs[test_idx] = p
            try:
                aucs.append(roc_auc_score(labels[test_idx], p))
            except ValueError:
                aucs.append(0.5)
        lr_step_aucs[s] = float(np.mean(aucs))
        lr_step_probs[s] = probs
        print(f"  Step {s:>3}: LR-on-entropy AUC={lr_step_aucs[s]:.3f}")

    # ---- Selective generation simulation (matches modal_early_exit_sim.py) ----
    # For each instance, walk checkpoint steps in order; at the first step where
    # confidence >= threshold, use the prediction; else fall back to final step.
    final_step = CHECKPOINT_STEPS[-1]
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    def simulate(step_probs):
        rows = []
        for thresh in thresholds:
            exit_step = np.full(n_samples, -1, dtype=int)
            predictions = np.full(n_samples, -1, dtype=int)
            for idx in range(n_samples):
                for s in CHECKPOINT_STEPS:
                    p = step_probs[s][idx]
                    conf = max(p, 1 - p)
                    if conf >= thresh:
                        exit_step[idx] = s
                        predictions[idx] = 1 if p >= 0.5 else 0
                        break
                if exit_step[idx] == -1:
                    exit_step[idx] = final_step
                    predictions[idx] = 1 if step_probs[final_step][idx] >= 0.5 else 0
            avg_cost = np.mean([(s + 1) / STEPS for s in exit_step])
            compute_saved = 1 - avg_cost
            accuracy = (predictions == labels).mean()
            func_mask = labels == 1
            func_kept = (predictions[func_mask] == 1).mean() if func_mask.sum() > 0 else 0
            rows.append({
                "threshold": thresh,
                "compute_saved_pct": round(100 * compute_saved, 1),
                "accuracy": round(100 * accuracy, 1),
                "func_recall": round(100 * func_kept, 1),
            })
        return rows

    # Build "step_probs"-style dicts for raw baselines (calibrate to [0,1] via rank).
    # For raw maxprob: use the value directly as P(functional). For entropy: use
    # exp(-z) where z is the per-step normalized entropy. Both are heuristics
    # since the raw values are not calibrated probabilities; report alongside.
    def normalize_to_pseudoprob(step_arr_dict, higher_is_correct=True):
        out = {}
        for s in CHECKPOINT_STEPS:
            v = step_arr_dict[s]
            # Min-max normalize to [0,1] across the dataset at this step.
            lo, hi = v.min(), v.max()
            if hi - lo < 1e-9:
                norm = np.full_like(v, 0.5)
            else:
                norm = (v - lo) / (hi - lo)
            out[s] = norm if higher_is_correct else 1.0 - norm
        return out

    raw_mp_pp = normalize_to_pseudoprob(mp_step, higher_is_correct=True)
    raw_ent_pp = normalize_to_pseudoprob(ent_step, higher_is_correct=False)

    print(f"\n{'='*80}")
    print(f"Selective Generation: Raw maxprob (min-max normalized per step)")
    print(f"{'='*80}")
    raw_mp_results = simulate(raw_mp_pp)
    for r in raw_mp_results:
        print(f"  τ={r['threshold']:.2f}: saved={r['compute_saved_pct']:>5.1f}%, "
              f"acc={r['accuracy']:>5.1f}%, func_recall={r['func_recall']:>5.1f}%")

    print(f"\n{'='*80}")
    print(f"Selective Generation: Raw -entropy (min-max normalized per step)")
    print(f"{'='*80}")
    raw_ent_results = simulate(raw_ent_pp)
    for r in raw_ent_results:
        print(f"  τ={r['threshold']:.2f}: saved={r['compute_saved_pct']:>5.1f}%, "
              f"acc={r['accuracy']:>5.1f}%, func_recall={r['func_recall']:>5.1f}%")

    print(f"\n{'='*80}")
    print(f"Selective Generation: LR on entropy features")
    print(f"{'='*80}")
    lr_results = simulate(lr_step_probs)
    for r in lr_results:
        print(f"  τ={r['threshold']:.2f}: saved={r['compute_saved_pct']:>5.1f}%, "
              f"acc={r['accuracy']:>5.1f}%, func_recall={r['func_recall']:>5.1f}%")

    results = {
        "dataset": dataset_key,
        "model": model_key,
        "n_samples": n_samples,
        "n_functional": n_func,
        "raw_auc_maxprob": {str(s): round(a, 4) for s, a in raw_auc["maxprob"].items()},
        "raw_auc_neg_entropy": {str(s): round(a, 4) for s, a in raw_auc["entropy"].items()},
        "lr_step_aucs": {str(s): round(a, 4) for s, a in lr_step_aucs.items()},
        "selective_gen_raw_maxprob": raw_mp_results,
        "selective_gen_raw_neg_entropy": raw_ent_results,
        "selective_gen_lr_entropy": lr_results,
    }
    out_path = f"/results/{dataset_key}_{model_key}/entropy_baseline_results.json"
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
    print(f"Entropy baseline: dataset={dataset}, model={model}, "
          f"{chunks}x A100, total={total}, chunk_size={chunk_size}")

    if not skip_gen:
        handles = []
        for i in range(chunks):
            offset = i * chunk_size
            limit = min(chunk_size, total - offset)
            if limit <= 0:
                break
            print(f"  Chunk {i}: offset={offset}, limit={limit}")
            handles.append(run_chunk_entropy.spawn(dataset, model, offset, limit))
        for handle in handles:
            print(f"  Done: {handle.get()}")

    print("\nRunning entropy baseline analysis...")
    print(run_entropy_baseline_analysis.remote(dataset, model, chunks, total))
