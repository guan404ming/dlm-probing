"""Length ablation probes: control for output/prompt length confounds.

Supports three ablation modes:
  --mode output_label  : predict binary output length (above/below median)
  --mode output_matched: subsample to matched output length distributions
  --mode prompt_matched: subsample to matched prompt length distributions

All use the same PCA + StandardScaler + LogisticRegression pipeline.
Reports best AUC per step, layer.

Usage:
  .venv/bin/modal run src/modal_length_ablation_probe.py --dataset jsonschema --mode output_label
  .venv/bin/modal run src/modal_length_ablation_probe.py --dataset gsm8k --model dream --mode output_matched
  .venv/bin/modal run src/modal_length_ablation_probe.py --dataset arc --mode prompt_matched
"""

import modal

app = modal.App("probe-length-ablation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "scikit-learn", "datasets==2.21.0")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)

STEPS = 128
CHECKPOINT_STEPS = sorted([0, 1, 4, 16, 32, 64, STEPS - 1])
N_REGIONS = 4

DATASET_CFGS = {
    "jsonschema": {"gen_length": 256, "total": 272},
    "gsm8k": {"gen_length": 512, "total": 1319},
    "mbpp": {"gen_length": 256, "total": 257},
    "arc": {"gen_length": 256, "total": 1172},
}


def get_reference_lengths(dataset_key, total):
    """Get reference output lengths from dataset."""
    import numpy as np
    from datasets import load_dataset

    if dataset_key == "jsonschema":
        ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
        instances = sorted(list(ds), key=lambda x: x["instance_id"])
        lengths = [len(inst["output"]) for inst in instances[:total]]
    elif dataset_key == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        instances = list(ds)
        lengths = [len(inst["answer"]) for inst in instances[:total]]
    elif dataset_key == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        instances = list(ds)
        lengths = [len(inst["code"]) for inst in instances[:total]]
    elif dataset_key == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        instances = list(ds)
        lengths = [len(inst["question"]) for inst in instances[:total]]
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    return np.array(lengths)


def get_prompt_lengths(dataset_key, total):
    """Get prompt character lengths for each instance."""
    import numpy as np
    from datasets import load_dataset

    if dataset_key == "jsonschema":
        ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
        instances = sorted(list(ds), key=lambda x: x["instance_id"])[:total]
        lengths = [len(inst["schema"]) + len(inst["input"]) for inst in instances]
    elif dataset_key == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        instances = list(ds)[:total]
        lengths = [len(inst["question"]) for inst in instances]
    elif dataset_key == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        instances = sorted(list(ds), key=lambda x: x["task_id"])[:total]
        lengths = [
            len(inst["prompt"]) + sum(len(t) for t in inst["test_list"])
            for inst in instances
        ]
    elif dataset_key == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        instances = list(ds)[:total]
        lengths = [
            len(inst["question"]) + sum(len(t) for t in inst["choices"]["text"])
            for inst in instances
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    return np.array(lengths)


def length_matched_subsample(labels, lengths, n_bins=10, rng_seed=42):
    """Subsample so functional/non-functional have matched length distributions."""
    import numpy as np

    rng = np.random.RandomState(rng_seed)
    bin_edges = np.percentile(lengths, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1

    selected = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (lengths >= lo) & (lengths < hi)
        func_idx = np.where(in_bin & (labels == 1))[0]
        nonfunc_idx = np.where(in_bin & (labels == 0))[0]
        n_keep = min(len(func_idx), len(nonfunc_idx))
        if n_keep > 0:
            selected.extend(rng.choice(func_idx, n_keep, replace=False))
            selected.extend(rng.choice(nonfunc_idx, n_keep, replace=False))

    return np.sort(np.array(selected))


@app.function(
    image=image,
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
)
def run_length_ablation(
    dataset_key: str, model_key: str, n_chunks: int, total: int, mode: str
):
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

    if mode not in ["output_label", "output_matched", "prompt_matched"]:
        raise ValueError(f"Unknown mode: {mode}")

    chunk_size = (total + n_chunks - 1) // n_chunks
    in_dir = f"/results/{dataset_key}_{model_key}"

    all_labels = []
    all_feats = {}

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_labels.append(data["labels"])
        for s in CHECKPOINT_STEPS:
            for r in range(N_REGIONS):
                key = (s, r)
                if key not in all_feats:
                    all_feats[key] = []
                all_feats[key].append(data[f"feat_s{s}_r{r}"])
        print(f"  Loaded {path}: {len(data['labels'])} samples")

    correctness_labels = np.concatenate(all_labels)
    features = {}
    for s in CHECKPOINT_STEPS:
        features[s] = {}
        for r in range(N_REGIONS):
            features[s][r] = np.concatenate(all_feats[(s, r)])

    n_samples = len(correctness_labels)
    n_layers = features[CHECKPOINT_STEPS[0]][0].shape[1]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        "dataset": dataset_key,
        "model": model_key,
        "mode": mode,
        "n_samples": n_samples,
    }

    if mode == "output_label":
        output_lengths = get_reference_lengths(dataset_key, total)
        assert len(output_lengths) == n_samples
        median = np.median(output_lengths)
        length_labels = (output_lengths > median).astype(int)
        print(f"\nOutput length: min={output_lengths.min()}, median={median:.0f}, "
              f"max={output_lengths.max()}, n_long={length_labels.sum()}/{n_samples}")
        results["median_length"] = float(median)
        results["n_long"] = int(length_labels.sum())

        print("\n=== Output length probe ===")
        step_best = {}
        for s in CHECKPOINT_STEPS:
            best_auc = -1
            best_layer = 0
            for layer_idx in range(n_layers):
                X = np.mean(
                    [features[s][r][:, layer_idx, :] for r in range(N_REGIONS)],
                    axis=0,
                )
                aucs = []
                for train_idx, test_idx in skf.split(X, length_labels):
                    clf = make_pipeline(
                        StandardScaler(),
                        PCA(n_components=min(64, X.shape[1])),
                        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
                    )
                    clf.fit(X[train_idx], length_labels[train_idx])
                    prob = clf.predict_proba(X[test_idx])[:, 1]
                    try:
                        aucs.append(roc_auc_score(length_labels[test_idx], prob))
                    except ValueError:
                        aucs.append(0.5)
                mean_auc = np.mean(aucs)
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_layer = layer_idx

            step_best[str(s)] = {"best_auc": round(best_auc, 4), "best_layer": best_layer}
            print(f"  Step {s:>3}: layer={best_layer}, AUC={best_auc:.4f}")

        results["output_length_probe"] = step_best

    else:
        if mode == "output_matched":
            lengths = get_reference_lengths(dataset_key, total)
            name_str = "Output"
        else:  # prompt_matched
            lengths = get_prompt_lengths(dataset_key, total)
            name_str = "Prompt"

        assert len(lengths) == n_samples
        subset_idx = length_matched_subsample(correctness_labels, lengths)
        sub_labels = correctness_labels[subset_idx]

        print(f"\n{name_str}-length matched subsampling:")
        print(f"  Full: {n_samples} samples, {int(correctness_labels.sum())} functional")
        print(f"  Subset: {len(subset_idx)} samples, {int(sub_labels.sum())} functional")

        results["n_full"] = n_samples
        results["n_subset"] = len(subset_idx)
        results["n_func_subset"] = int(sub_labels.sum())

        print(f"\n=== {name_str}-matched correctness probe ===")
        for data_label, use_labels, use_idx in [
            ("full", correctness_labels, np.arange(n_samples)),
            (f"{name_str.lower()}_matched", sub_labels, subset_idx),
        ]:
            step_best = {}
            for s in CHECKPOINT_STEPS:
                best_auc = -1
                best_layer = 0
                for layer_idx in range(n_layers):
                    X_full = np.mean(
                        [features[s][r][:, layer_idx, :] for r in range(N_REGIONS)],
                        axis=0,
                    )
                    X = X_full[use_idx]
                    aucs = []
                    for train_idx, test_idx in skf.split(X, use_labels):
                        clf = make_pipeline(
                            StandardScaler(),
                            PCA(n_components=min(64, X.shape[1])),
                            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
                        )
                        clf.fit(X[train_idx], use_labels[train_idx])
                        prob = clf.predict_proba(X[test_idx])[:, 1]
                        try:
                            aucs.append(roc_auc_score(use_labels[test_idx], prob))
                        except ValueError:
                            aucs.append(0.5)
                    mean_auc = np.mean(aucs)
                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_layer = layer_idx

                step_best[str(s)] = {"best_auc": round(best_auc, 4), "best_layer": best_layer}
                print(f"  [{data_label}] Step {s:>3}: layer={best_layer}, AUC={best_auc:.4f}")

            results[f"{data_label}_probe"] = step_best

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/length_ablation_{mode}.json"
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
    mode: str = "output_label",
):
    if total <= 0:
        total = DATASET_CFGS[dataset]["total"]
    print(f"Length ablation ({mode}): dataset={dataset}, model={model}")
    result = run_length_ablation.remote(dataset, model, chunks, total, mode)
    print("\n" + result)
