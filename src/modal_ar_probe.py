"""Probe Qwen-2.5-7B (AR model) for comparison with dLLM step-0 probing.

Sub-experiment A: predict dLLM functional labels from AR prompt hidden states.
Sub-experiment B: generate with AR model, check correctness, probe own labels.

Usage:
  .venv/bin/modal run src/modal_ar_probe.py --dataset jsonschema --chunks 4
  .venv/bin/modal run src/modal_ar_probe.py --dataset gsm8k --chunks 8
"""

import re

import modal

app = modal.App("probe-ar-qwen")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers==4.52.2",
        "accelerate",
        "numpy",
        "datasets==2.21.0",
        "huggingface_hub",
        "scikit-learn",
    )
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

DATASET_CFGS = {
    "jsonschema": {"gen_length": 256, "total": 272},
    "gsm8k": {"gen_length": 512, "total": 1319},
    "mbpp": {"gen_length": 256, "total": 257},
    "arc": {"gen_length": 256, "total": 1172},
}


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
        all_insts = list(ds)
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
            "Write a Python function to solve the given task. "
            "Return only the function definition."
        )
    else:  # arc
        return (
            "Answer the multiple choice question. Think step by step, "
            "then give your final answer as #### followed by a single letter (A, B, C, or D)."
        )


def build_user_prompt(dataset_key, instance):
    if dataset_key == "jsonschema":
        return instance["input"]
    elif dataset_key == "gsm8k":
        return instance["question"]
    elif dataset_key == "mbpp":
        tests = "\n".join(instance["test_list"])
        return f"{instance['prompt']}\n\nTest cases:\n{tests}"
    else:  # arc
        q = instance["question"]
        choices = instance["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        return f"{q}\n\n{choice_str}"


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

    else:  # arc
        return _check_arc(output_text, instance)


def _check_mbpp(output_text, instance):
    """Check MBPP correctness by executing code + test assertions."""
    import subprocess
    import tempfile
    import os

    code = output_text.strip()
    # Strip markdown fences
    if "```python" in code:
        code = code.split("```python", 1)[-1]
    if "```" in code:
        code = code.split("```")[0]
    code = code.strip()

    tests = "\n".join(instance["test_list"])
    full_code = f"{code}\n\n{tests}\n"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(full_code)
            tmp = f.name
        result = subprocess.run(
            ["python", tmp],
            capture_output=True,
            timeout=10,
        )
        os.unlink(tmp)
        return result.returncode == 0
    except Exception:
        return False


def _check_arc(output_text, instance):
    """Check ARC correctness by extracting answer letter."""
    gold = instance["answerKey"].strip().upper()
    m = re.search(r"####\s*([A-Da-d])", output_text)
    if m:
        return m.group(1).upper() == gold
    matches = re.findall(r"\b([A-Da-d])\b", output_text)
    if matches:
        return matches[-1].upper() == gold
    return False


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


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(dataset_key: str, offset: int, limit: int):
    """Extract AR hidden states and generate outputs for a chunk."""
    import os
    import time

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    gen_length = DATASET_CFGS[dataset_key]["gen_length"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()

    instances = load_instances(dataset_key, offset, limit)
    n = len(instances)
    print(f"Processing {n} instances (offset={offset})")

    # Storage
    all_feats = []  # (n, n_layers, hidden_dim)
    ar_labels = []  # correctness of AR generation
    dllm_labels = None  # loaded from existing results

    for idx, inst in enumerate(instances):
        t0 = time.time()

        # Build prompt using Qwen chat template
        sys_prompt = build_system_prompt(dataset_key, inst)
        user_prompt = build_user_prompt(dataset_key, inst)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if dataset_key == "jsonschema":
            text += "```json\n"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        # Extract last prompt token hidden state from each layer
        # outputs.hidden_states is a tuple of (n_layers+1,) tensors
        # Index 0 is embedding layer, 1..n_layers are transformer layers
        hs = outputs.hidden_states
        n_layers = len(hs) - 1  # exclude embedding layer
        hidden_dim = hs[1].shape[-1]

        feat = np.zeros((n_layers, hidden_dim), dtype=np.float32)
        for li in range(n_layers):
            # li+1 to skip embedding layer, -1 for last prompt token
            feat[li] = hs[li + 1][0, -1, :].float().cpu().numpy()
        all_feats.append(feat)

        # Generate output for sub-experiment B
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=gen_length,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        gen_text = tokenizer.decode(
            gen_ids[0, prompt_len:], skip_special_tokens=True,
        )
        correct = check_functional(dataset_key, inst, gen_text)
        ar_labels.append(int(correct))

        elapsed = time.time() - t0
        if idx < 3 or idx % 20 == 0:
            print(f"  [{idx}/{n}] correct={correct} len={len(gen_text)} "
                  f"({elapsed:.1f}s)")

    # Load dLLM labels from existing results (use LLaDA labels)
    RESULTS_VOL.reload()
    dllm_dir = f"/results/{dataset_key}_llada"
    total = DATASET_CFGS[dataset_key]["total"]
    n_chunks_dllm = 8
    chunk_size_dllm = (total + n_chunks_dllm - 1) // n_chunks_dllm

    all_dllm_labels = []
    for i in range(n_chunks_dllm):
        off = i * chunk_size_dllm
        path = f"{dllm_dir}/chunk_off{off}.npz"
        if os.path.exists(path):
            data = np.load(path)
            all_dllm_labels.append(data["labels"])
    dllm_labels_full = np.concatenate(all_dllm_labels)
    dllm_labels = dllm_labels_full[offset:offset + n]

    # Save chunk results
    feats_array = np.stack(all_feats)  # (n, n_layers, hidden_dim)
    ar_labels_array = np.array(ar_labels, dtype=np.int32)

    out_dir = f"/results/{dataset_key}_qwen"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/chunk_off{offset}.npz"
    np.savez_compressed(
        out_path,
        feats=feats_array,
        ar_labels=ar_labels_array,
        dllm_labels=dllm_labels,
    )
    RESULTS_VOL.commit()

    ar_func = int(ar_labels_array.sum())
    dllm_func = int(dllm_labels.sum())
    print(f"\nSaved {out_path}: {n} samples, "
          f"AR functional={ar_func}/{n}, dLLM functional={dllm_func}/{n}")
    return {
        "offset": offset,
        "n": n,
        "ar_functional": ar_func,
        "dllm_functional": dllm_func,
    }


@app.function(
    image=image,
    timeout=1800,
    volumes={"/results": RESULTS_VOL},
)
def run_probe(dataset_key: str, n_chunks: int, total: int):
    """Load all chunks and train probes."""
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
    in_dir = f"/results/{dataset_key}_qwen"

    all_feats = []
    all_ar_labels = []
    all_dllm_labels = []

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_feats.append(data["feats"])
        all_ar_labels.append(data["ar_labels"])
        all_dllm_labels.append(data["dllm_labels"])
        print(f"  Loaded {path}: {len(data['ar_labels'])} samples")

    feats = np.concatenate(all_feats)        # (N, n_layers, hidden_dim)
    ar_labels = np.concatenate(all_ar_labels)
    dllm_labels = np.concatenate(all_dllm_labels)

    n_samples, n_layers, hidden_dim = feats.shape
    print(f"\nTotal: {n_samples} samples, {n_layers} layers, dim={hidden_dim}")
    print(f"AR functional: {int(ar_labels.sum())}/{n_samples} "
          f"({100*ar_labels.mean():.1f}%)")
    print(f"dLLM functional: {int(dllm_labels.sum())}/{n_samples} "
          f"({100*dllm_labels.mean():.1f}%)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        "model": "Qwen2.5-7B-Instruct",
        "dataset": dataset_key,
        "n_samples": n_samples,
    }

    for label_name, labels in [("sub_a", dllm_labels), ("sub_b", ar_labels)]:
        desc = ("AR probe with dLLM functional labels" if label_name == "sub_a"
                else "AR probe with AR own functional labels")
        print(f"\n=== {desc} ===")

        layer_aucs = []
        best_auc = -1
        best_layer = 0

        for layer_idx in range(n_layers):
            X = feats[:, layer_idx, :]
            aucs = []
            for train_idx, test_idx in skf.split(X, labels):
                clf = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=min(64, X.shape[1])),
                    LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
                )
                clf.fit(X[train_idx], labels[train_idx])
                prob = clf.predict_proba(X[test_idx])[:, 1]
                try:
                    aucs.append(roc_auc_score(labels[test_idx], prob))
                except ValueError:
                    aucs.append(0.5)
            mean_auc = np.mean(aucs)
            layer_aucs.append(round(mean_auc, 4))

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_layer = layer_idx

            if layer_idx % 5 == 0 or layer_idx == n_layers - 1:
                print(f"  Layer {layer_idx:>2}: AUC={mean_auc:.4f}")

        print(f"  Best: layer={best_layer}, AUC={best_auc:.4f}")

        sub_result = {
            "description": desc,
            "layer_auc": layer_aucs,
            "best_layer": best_layer,
            "best_auc": round(best_auc, 4),
        }
        if label_name == "sub_b":
            sub_result["n_functional"] = int(ar_labels.sum())
            sub_result["functional_rate"] = round(float(ar_labels.mean()), 4)

        results[label_name] = sub_result

    out_dir = f"/results/{dataset_key}_qwen"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/ar_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()

    print(f"\nResults saved to {out_path}")
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(
    dataset: str = "jsonschema",
    chunks: int = 4,
    total: int = 0,
    probe_only: bool = False,
):
    if total <= 0:
        total = DATASET_CFGS[dataset]["total"]
    chunk_size = (total + chunks - 1) // chunks

    print(f"AR probe: dataset={dataset}, total={total}, chunks={chunks}")

    if not probe_only:
        # Run extraction + generation chunks in parallel
        handles = []
        for i in range(chunks):
            offset = i * chunk_size
            limit = min(chunk_size, total - offset)
            handles.append(run_chunk.spawn(dataset, offset, limit))

        for h in handles:
            result = h.get()
            print(f"  Chunk offset={result['offset']}: "
                  f"AR={result['ar_functional']}/{result['n']}, "
                  f"dLLM={result['dllm_functional']}/{result['n']}")

    # Run probes
    result = run_probe.remote(dataset, chunks, total)
    print("\n" + result)
