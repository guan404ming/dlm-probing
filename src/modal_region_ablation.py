"""Spatial pooling ablation: test 1, 2, and 4 regions.

Re-runs probes with different numbers of position regions using existing
hidden states. Reports best AUC for each region count.

With 4 extracted regions, we can test:
  1 region: average all 4 regions
  2 regions: average pairs (0,1) and (2,3)
  4 regions: use as-is (original setup)

Usage:
  .venv/bin/modal run src/modal_region_ablation.py --dataset jsonschema --model llada
  .venv/bin/modal run src/modal_region_ablation.py --dataset gsm8k --model dream
"""

import modal

app = modal.App("probe-region-ablation")

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
}


@app.function(
    image=image,
    timeout=86400,
    volumes={"/results": RESULTS_VOL},
)
def run_region_ablation(dataset_key: str, model_key: str, n_chunks: int, total: int):
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

    # Load all chunks
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
    print(f"\nLoaded: {n_samples} samples, {n_func} functional "
          f"({100*n_func/n_samples:.1f}%), {n_layers} layers")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def get_pooled_features(step, layer_idx, n_pool_regions):
        """Pool features into n_pool_regions groups."""
        if n_pool_regions == 1:
            # Average all 4 regions
            return np.mean([features[step][r][:, layer_idx, :]
                            for r in range(4)], axis=0)
        elif n_pool_regions == 2:
            # Average pairs: (0,1) and (2,3)
            r01 = np.mean([features[step][0][:, layer_idx, :],
                           features[step][1][:, layer_idx, :]], axis=0)
            r23 = np.mean([features[step][2][:, layer_idx, :],
                           features[step][3][:, layer_idx, :]], axis=0)
            return np.concatenate([r01, r23], axis=1)
        elif n_pool_regions == 4:
            # Concatenate all 4 regions
            return np.concatenate([features[step][r][:, layer_idx, :]
                                   for r in range(4)], axis=1)
        else:
            raise ValueError(f"Unsupported n_pool_regions: {n_pool_regions}")

    # Test each region count
    region_counts = [1, 2, 4]
    results = {
        "dataset": dataset_key,
        "model": model_key,
        "n_samples": n_samples,
        "n_functional": n_func,
    }

    for n_reg in region_counts:
        print(f"\n=== {n_reg} region(s) ===")
        overall_best_auc = -1
        overall_best_step = 0
        overall_best_layer = 0
        step_results = {}

        for s in CHECKPOINT_STEPS:
            best_auc = -1
            best_layer = 0
            for layer_idx in range(n_layers):
                X = get_pooled_features(s, layer_idx, n_reg)
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

            step_results[str(s)] = {
                "best_auc": round(best_auc, 4),
                "best_layer": best_layer,
            }
            if best_auc > overall_best_auc:
                overall_best_auc = best_auc
                overall_best_step = s
                overall_best_layer = best_layer

            print(f"  Step {s:>3}: best_layer={best_layer}, best_auc={best_auc:.4f}")

        print(f"  Overall best: step={overall_best_step}, layer={overall_best_layer}, "
              f"AUC={overall_best_auc:.4f}")

        results[f"regions_{n_reg}"] = {
            "overall_best_auc": round(overall_best_auc, 4),
            "overall_best_step": overall_best_step,
            "overall_best_layer": overall_best_layer,
            "per_step": step_results,
        }

    # Summary
    print(f"\n{'='*60}")
    print("Summary: best AUC by region count")
    for n_reg in region_counts:
        r = results[f"regions_{n_reg}"]
        print(f"  {n_reg} region(s): AUC={r['overall_best_auc']:.4f} "
              f"(step={r['overall_best_step']}, layer={r['overall_best_layer']})")

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/region_ablation_results.json"
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
    print(f"Region ablation: dataset={dataset}, model={model}, total={total}")
    result = run_region_ablation.remote(dataset, model, chunks, total)
    print("\n" + result)
