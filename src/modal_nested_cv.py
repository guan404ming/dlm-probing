"""Nested cross-validation for rebuttal.

Outer loop: 5-fold stratified CV (same splits as original).
Inner loop: 3-fold CV on training data to select best (layer, region).
Evaluate selected probe on held-out outer test fold.

CPU-only, uses existing hidden states from Modal volume.

Usage:
  .venv/bin/modal run src/modal_nested_cv.py
  .venv/bin/modal run src/modal_nested_cv.py --dataset gsm8k --model dream
"""

import modal

app = modal.App("probe-nested-cv")

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


def load_chunks(dataset_key, model_key, n_chunks, total):
    """Load all chunk npz files, return merged labels and features."""
    import os
    import numpy as np

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

    return labels, features


def train_and_eval(X_train, y_train, X_test, y_test):
    """Train probe pipeline, return AUC on test set."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        PCA(n_components=min(64, X_train.shape[1])),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    try:
        return roc_auc_score(y_test, prob)
    except ValueError:
        return 0.5


@app.function(
    image=image,
    timeout=86400,
    volumes={"/results": RESULTS_VOL},
)
def run_nested_cv(dataset_key: str, model_key: str, n_chunks: int, total: int):
    """Run nested CV for all steps. Compare with original non-nested results."""
    import json
    import os
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    RESULTS_VOL.reload()

    labels, features = load_chunks(dataset_key, model_key, n_chunks, total)
    n_samples = len(labels)
    n_layers = features[CHECKPOINT_STEPS[0]][0].shape[1]
    print(f"\nLoaded: {n_samples} samples, {int(labels.sum())} functional, "
          f"{n_layers} layers")

    # Load original results for comparison
    orig_path = f"/results/{dataset_key}_{model_key}/midstep_probe_results.json"
    with open(orig_path) as f:
        orig = json.load(f)

    outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = {}

    for s in CHECKPOINT_STEPS:
        print(f"\n--- Step {s} ---")

        # Precompute X for all (layer, region) combos: mean-pool regions
        # Same as original: mean across regions per layer
        X_by_layer = []
        for layer_idx in range(n_layers):
            X = np.mean(
                [features[s][r][:, layer_idx, :] for r in range(N_REGIONS)],
                axis=0,
            )
            X_by_layer.append(X)

        # Also prepare per-region features for region selection
        X_by_layer_region = {}
        for layer_idx in range(n_layers):
            for r in range(N_REGIONS):
                X_by_layer_region[(layer_idx, r)] = features[s][r][:, layer_idx, :]

        outer_aucs = []
        selected_configs = []

        for fold_i, (train_idx, test_idx) in enumerate(outer_skf.split(labels, labels)):
            y_train = labels[train_idx]
            y_test = labels[test_idx]

            # Inner CV: select best (layer, region) on training data only
            best_inner_auc = -1
            best_layer = 0
            best_region = None  # None means mean-pooled

            # First try mean-pooled (same as original)
            for layer_idx in range(n_layers):
                X_all = X_by_layer[layer_idx]
                X_tr = X_all[train_idx]

                inner_aucs = []
                for inner_train, inner_val in inner_skf.split(X_tr, y_train):
                    auc = train_and_eval(
                        X_tr[inner_train], y_train[inner_train],
                        X_tr[inner_val], y_train[inner_val],
                    )
                    inner_aucs.append(auc)
                mean_inner = np.mean(inner_aucs)

                if mean_inner > best_inner_auc:
                    best_inner_auc = mean_inner
                    best_layer = layer_idx
                    best_region = None

            # Also try individual regions
            for layer_idx in range(n_layers):
                for r in range(N_REGIONS):
                    X_all = X_by_layer_region[(layer_idx, r)]
                    X_tr = X_all[train_idx]

                    inner_aucs = []
                    for inner_train, inner_val in inner_skf.split(X_tr, y_train):
                        auc = train_and_eval(
                            X_tr[inner_train], y_train[inner_train],
                            X_tr[inner_val], y_train[inner_val],
                        )
                        inner_aucs.append(auc)
                    mean_inner = np.mean(inner_aucs)

                    if mean_inner > best_inner_auc:
                        best_inner_auc = mean_inner
                        best_layer = layer_idx
                        best_region = r

            # Evaluate selected config on outer test fold
            if best_region is None:
                X_all = X_by_layer[best_layer]
            else:
                X_all = X_by_layer_region[(best_layer, best_region)]

            outer_auc = train_and_eval(
                X_all[train_idx], y_train,
                X_all[test_idx], y_test,
            )
            outer_aucs.append(outer_auc)
            selected_configs.append({
                "layer": best_layer,
                "region": best_region,
                "inner_auc": round(best_inner_auc, 4),
                "outer_auc": round(outer_auc, 4),
            })

            print(f"  Fold {fold_i}: selected layer={best_layer}, "
                  f"region={'mean' if best_region is None else best_region}, "
                  f"inner={best_inner_auc:.4f}, outer={outer_auc:.4f}")

        nested_auc = np.mean(outer_aucs)
        nested_std = np.std(outer_aucs)

        # Original best AUC at this step (max across layers, mean-pooled)
        orig_layer_aucs = orig["step_layer_auc"][str(s)]
        orig_best_auc = max(orig_layer_aucs)

        drop = orig_best_auc - nested_auc

        results[str(s)] = {
            "nested_auc": round(nested_auc, 4),
            "nested_std": round(nested_std, 4),
            "original_auc": round(orig_best_auc, 4),
            "drop": round(drop, 4),
            "folds": selected_configs,
        }

        print(f"  Step {s}: nested={nested_auc:.4f} +/- {nested_std:.4f}, "
              f"original={orig_best_auc:.4f}, drop={drop:.4f}")

    # Summary
    all_drops = [results[str(s)]["drop"] for s in CHECKPOINT_STEPS]
    avg_drop = np.mean(all_drops)
    max_drop = max(all_drops)
    print(f"\n=== Summary: {dataset_key}/{model_key} ===")
    print(f"  Avg drop: {avg_drop:.4f}, Max drop: {max_drop:.4f}")

    output = {
        "dataset": dataset_key,
        "model": model_key,
        "n_samples": n_samples,
        "per_step": results,
        "avg_drop": round(avg_drop, 4),
        "max_drop": round(max_drop, 4),
    }

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/nested_cv_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    RESULTS_VOL.commit()

    print(f"Saved to {out_path}")
    return json.dumps(output, indent=2)


@app.local_entrypoint()
def main(
    dataset: str = "jsonschema",
    model: str = "llada",
    chunks: int = 8,
    total: int = 0,
    run_all: bool = False,
):
    if run_all:
        for ds in DATASET_CFGS:
            for mdl in ["llada", "dream"]:
                t = DATASET_CFGS[ds]["total"]
                print(f"\n{'='*60}")
                print(f"Running nested CV: {ds}/{mdl}")
                print(f"{'='*60}")
                result = run_nested_cv.remote(ds, mdl, chunks, t)
                print(result)
    else:
        if total <= 0:
            total = DATASET_CFGS[dataset]["total"]
        print(f"Nested CV: dataset={dataset}, model={model}, total={total}")
        result = run_nested_cv.remote(dataset, model, chunks, total)
        print(result)
