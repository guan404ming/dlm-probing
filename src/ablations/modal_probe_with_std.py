"""Re-run probes and report per-fold AUC with std.

Loads existing hidden states from Modal volume (no GPU needed).
Reports mean +/- std for the main step x layer probing results.

Usage:
  .venv/bin/modal run src/modal_probe_with_std.py --dataset jsonschema --model llada
  .venv/bin/modal run src/modal_probe_with_std.py --dataset gsm8k --model dream
"""

import modal

app = modal.App("probe-with-std")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "scikit-learn")
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


@app.function(
    image=image,
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
)
def run_probe_with_std(dataset_key: str, model_key: str, n_chunks: int, total: int):
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
    n_func = int(labels.sum())
    n_layers = features[CHECKPOINT_STEPS[0]][0].shape[1]
    print(f"\nTotal: {n_samples} samples, {n_func} functional "
          f"({100*n_func/n_samples:.1f}%), {n_layers} layers")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        "dataset": dataset_key,
        "model": model_key,
        "n_samples": n_samples,
        "n_functional": n_func,
    }

    # For each step, find best layer by mean AUC, report mean +/- std
    for s in CHECKPOINT_STEPS:
        best_mean = -1
        best_layer = 0
        best_std = 0
        best_folds = []

        for layer_idx in range(n_layers):
            X = np.mean([features[s][r][:, layer_idx, :]
                         for r in range(N_REGIONS)], axis=0)
            fold_aucs = []
            for train_idx, test_idx in skf.split(X, labels):
                clf = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=min(64, X.shape[1])),
                    LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
                )
                clf.fit(X[train_idx], labels[train_idx])
                prob = clf.predict_proba(X[test_idx])[:, 1]
                try:
                    fold_aucs.append(roc_auc_score(labels[test_idx], prob))
                except ValueError:
                    fold_aucs.append(0.5)

            mean_auc = np.mean(fold_aucs)
            if mean_auc > best_mean:
                best_mean = mean_auc
                best_layer = layer_idx
                best_std = np.std(fold_aucs)
                best_folds = fold_aucs

        print(f"  Step {s:>3}: layer={best_layer}, "
              f"AUC={best_mean:.4f} +/- {best_std:.4f}, "
              f"folds={[round(a, 4) for a in best_folds]}")

        results[f"step_{s}"] = {
            "best_layer": best_layer,
            "mean_auc": round(best_mean, 4),
            "std_auc": round(best_std, 4),
            "fold_aucs": [round(a, 4) for a in best_folds],
        }

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/probe_std_results.json"
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
    print(f"Probe with std: dataset={dataset}, model={model}, total={total}")
    result = run_probe_with_std.remote(dataset, model, chunks, total)
    print("\n" + result)
