"""Length control probe: predict output length instead of correctness.

Trains the same PCA + StandardScaler + LogisticRegression probe pipeline
but with binary length labels (above/below median reference output length)
instead of functional correctness labels. Reports AUC.

If the length probe AUC is close to the correctness probe AUC, the
correctness signal may be confounded with output length.

Usage:
  .venv/bin/modal run src/modal_length_control_probe.py --dataset jsonschema --model llada
  .venv/bin/modal run src/modal_length_control_probe.py --dataset gsm8k --model dream
"""

import modal

app = modal.App("probe-length-control")

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


def compute_length_labels(dataset_key, total):
    """Compute binary length labels (above/below median reference length)."""
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

    lengths = np.array(lengths)
    median = np.median(lengths)
    labels = (lengths > median).astype(int)
    print(f"Length stats: min={lengths.min()}, median={median:.0f}, "
          f"max={lengths.max()}, n_long={labels.sum()}/{len(labels)}")
    return labels


@app.function(
    image=image,
    timeout=1800,
    volumes={"/results": RESULTS_VOL},
)
def run_length_control(dataset_key: str, model_key: str, n_chunks: int, total: int):
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

    # Load chunk features (use correctness labels just for reference)
    all_correctness = []
    all_feats = {}

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_correctness.append(data["labels"])
        for s in CHECKPOINT_STEPS:
            for r in range(N_REGIONS):
                key = (s, r)
                if key not in all_feats:
                    all_feats[key] = []
                all_feats[key].append(data[f"feat_s{s}_r{r}"])
        print(f"  Loaded {path}: {len(data['labels'])} samples")

    correctness_labels = np.concatenate(all_correctness)
    features = {}
    for s in CHECKPOINT_STEPS:
        features[s] = {}
        for r in range(N_REGIONS):
            features[s][r] = np.concatenate(all_feats[(s, r)])

    n_samples = len(correctness_labels)
    n_layers = features[CHECKPOINT_STEPS[0]][0].shape[1]

    # Compute length labels from dataset
    length_labels = compute_length_labels(dataset_key, total)
    assert len(length_labels) == n_samples, (
        f"Length labels ({len(length_labels)}) != samples ({n_samples})"
    )

    print(f"\nSamples: {n_samples}, layers: {n_layers}")
    print(f"Correctness: {int(correctness_labels.sum())}/{n_samples} functional")
    print(f"Length: {int(length_labels.sum())}/{n_samples} above median")

    # Train probes with length labels (same pipeline as correctness probe)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sorted_steps = sorted(CHECKPOINT_STEPS)

    # Also train correctness probe for direct comparison
    print("\n=== Length probe vs Correctness probe (best layer, all regions pooled) ===")

    for label_name, labels in [("length", length_labels), ("correctness", correctness_labels)]:
        best_auc_overall = -1
        best_layer_overall = 0
        best_step_overall = 0

        for s in sorted_steps:
            best_auc_step = -1
            best_layer_step = 0
            for layer_idx in range(n_layers):
                X = np.mean([features[s][r][:, layer_idx, :]
                             for r in range(N_REGIONS)], axis=0)
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
                if mean_auc > best_auc_step:
                    best_auc_step = mean_auc
                    best_layer_step = layer_idx
                if mean_auc > best_auc_overall:
                    best_auc_overall = mean_auc
                    best_layer_overall = layer_idx
                    best_step_overall = s

            print(f"  [{label_name}] Step {s:>3}: best_layer={best_layer_step}, "
                  f"best_auc={best_auc_step:.4f}")

        print(f"  [{label_name}] Overall best: step={best_step_overall}, "
              f"layer={best_layer_overall}, AUC={best_auc_overall:.4f}\n")

    # Save results
    results = {
        "dataset": dataset_key,
        "model": model_key,
        "n_samples": n_samples,
        "n_long": int(length_labels.sum()),
    }

    # Compact results: best AUC per step for both probes
    for label_name, labels in [("length", length_labels), ("correctness", correctness_labels)]:
        step_best = {}
        for s in sorted_steps:
            best_auc = -1
            best_layer = 0
            for layer_idx in range(n_layers):
                X = np.mean([features[s][r][:, layer_idx, :]
                             for r in range(N_REGIONS)], axis=0)
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
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_layer = layer_idx
            step_best[str(s)] = {"best_auc": round(best_auc, 4), "best_layer": best_layer}
        results[f"{label_name}_probe"] = step_best

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/length_control_results.json"
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
    print(f"Length control probe: dataset={dataset}, model={model}, total={total}")
    result = run_length_control.remote(dataset, model, chunks, total)
    print("\n" + result)
