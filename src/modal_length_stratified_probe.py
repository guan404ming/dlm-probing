"""Length-stratified correctness probe.

Subsample data so functional and non-functional groups have matched length
distributions (binned matching on reference output length). Re-train the
correctness probe on this balanced subset and report AUC.

Usage:
  .venv/bin/modal run src/modal_length_stratified_probe.py --dataset jsonschema --model llada
"""

import modal

app = modal.App("probe-length-stratified")

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
        # Use question + choices length as reference
        lengths = [len(inst["question"]) for inst in instances[:total]]
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    return np.array(lengths)


def length_matched_subsample(labels, lengths, n_bins=10, rng_seed=42):
    """Subsample so functional/non-functional have matched length distributions.

    Bin lengths into quantiles, then in each bin keep min(n_func, n_nonfunc)
    samples from each class.
    """
    import numpy as np

    rng = np.random.RandomState(rng_seed)
    bin_edges = np.percentile(lengths, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1  # include max

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
    timeout=1800,
    volumes={"/results": RESULTS_VOL},
)
def run_length_stratified(dataset_key: str, model_key: str, n_chunks: int, total: int):
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

    # Load chunks
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

    labels = np.concatenate(all_labels)
    features = {}
    for s in CHECKPOINT_STEPS:
        features[s] = {}
        for r in range(N_REGIONS):
            features[s][r] = np.concatenate(all_feats[(s, r)])

    n_samples = len(labels)
    n_layers = features[CHECKPOINT_STEPS[0]][0].shape[1]

    # Get reference lengths and do matched subsampling
    lengths = get_reference_lengths(dataset_key, total)
    assert len(lengths) == n_samples

    subset_idx = length_matched_subsample(labels, lengths)
    sub_labels = labels[subset_idx]
    sub_lengths = lengths[subset_idx]

    print(f"\nFull dataset: {n_samples} samples, {int(labels.sum())} functional")
    print(f"Length-matched subset: {len(subset_idx)} samples, "
          f"{int(sub_labels.sum())} functional, {int((sub_labels == 0).sum())} non-functional")
    print(f"Subset length stats: func median={np.median(sub_lengths[sub_labels==1]):.0f}, "
          f"nonfunc median={np.median(sub_lengths[sub_labels==0]):.0f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Train probes on full and subset data
    results = {
        "dataset": dataset_key,
        "model": model_key,
        "n_full": n_samples,
        "n_subset": len(subset_idx),
        "n_func_subset": int(sub_labels.sum()),
        "n_nonfunc_subset": int((sub_labels == 0).sum()),
    }

    for data_label, use_labels, use_idx in [
        ("full", labels, np.arange(n_samples)),
        ("length_matched", sub_labels, subset_idx),
    ]:
        print(f"\n=== {data_label} ({len(use_labels)} samples) ===")
        best_auc_overall = -1
        best_layer_overall = 0
        best_step_overall = 0
        step_results = {}

        for s in CHECKPOINT_STEPS:
            best_auc = -1
            best_layer = 0
            for layer_idx in range(n_layers):
                X_full = np.mean([features[s][r][:, layer_idx, :]
                                  for r in range(N_REGIONS)], axis=0)
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

            step_results[str(s)] = {"best_auc": round(best_auc, 4), "best_layer": best_layer}
            if best_auc > best_auc_overall:
                best_auc_overall = best_auc
                best_layer_overall = best_layer
                best_step_overall = s

            print(f"  Step {s:>3}: best_layer={best_layer}, best_auc={best_auc:.4f}")

        print(f"  Overall best: step={best_step_overall}, layer={best_layer_overall}, "
              f"AUC={best_auc_overall:.4f}")

        results[f"{data_label}_probe"] = {
            "overall_best_auc": round(best_auc_overall, 4),
            "overall_best_step": best_step_overall,
            "overall_best_layer": best_layer_overall,
            "per_step": step_results,
        }

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/length_stratified_results.json"
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
):
    if total <= 0:
        total = DATASET_CFGS[dataset]["total"]
    print(f"Length-stratified probe: dataset={dataset}, model={model}, total={total}")
    result = run_length_stratified.remote(dataset, model, chunks, total)
    print("\n" + result)
