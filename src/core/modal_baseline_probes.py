"""Baseline probes for selective generation comparison.

Supports two baseline types:
  --baseline-type shuffle  : train probes with randomly permuted labels (noise floor)
  --baseline-type entropy  : use model's token confidence (entropy/max-prob)

For shuffle: reports per-step best AUC, useful for Figure 2 confidence intervals.
For entropy: re-runs generation to capture confidence statistics, compares to probes.

Usage:
  .venv/bin/modal run src/modal_baseline_probes.py --baseline-type shuffle --dataset jsonschema
  .venv/bin/modal run src/modal_baseline_probes.py --baseline-type entropy --dataset gsm8k --chunks 8
"""

import modal

app = modal.App("probe-baselines")

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)

CHECKPOINT_STEPS = sorted([0, 1, 4, 16, 32, 64, 127])
N_REGIONS = 4

DATASET_TOTALS = {"jsonschema": 272, "gsm8k": 1319, "mbpp": 257, "arc": 1172}

DATASET_CFGS = {
    "jsonschema": {"gen_length": 256, "total": 272},
    "gsm8k": {"gen_length": 512, "total": 1319},
    "mbpp": {"gen_length": 256, "total": 257},
    "arc": {"gen_length": 256, "total": 1172},
}

MODEL_CFGS = {
    "llada": {"name": "GSAI-ML/LLaDA-8B-Instruct", "mask_id": 126336, "temperature": 0.2},
    "dream": {"name": "Dream-org/Dream-v0-Instruct-7B", "mask_id": 151666, "temperature": 0.0},
}


# ============================================================================
# SHUFFLE BASELINE
# ============================================================================

shuffle_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "scikit-learn")
)


@app.function(
    image=shuffle_image,
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
    cpu=8.0,
    memory=16384,
)
def run_shuffle_baseline(dataset_key: str, model_key: str, n_chunks: int = 8):
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

    total = DATASET_TOTALS[dataset_key]
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

    labels = np.concatenate(all_labels)
    features = {s: {r: np.concatenate(all_feats[(s, r)]) for r in range(N_REGIONS)}
                for s in CHECKPOINT_STEPS}
    n_samples = len(labels)
    n_layers = features[CHECKPOINT_STEPS[0]][0].shape[1]
    print(f"Loaded {n_samples} samples, {n_layers} layers, {dataset_key}_{model_key}")

    layer_grid = list(range(0, n_layers, 8))
    if n_layers - 1 not in layer_grid:
        layer_grid.append(n_layers - 1)

    N_SHUFFLES = 3
    rng = np.random.RandomState(0)
    step_results = {}
    for s in CHECKPOINT_STEPS:
        all_aucs_per_shuffle = []
        for shuf in range(N_SHUFFLES):
            shuffled = rng.permutation(labels)
            best = 0.0
            for layer_idx in layer_grid:
                for r in range(N_REGIONS):
                    X = features[s][r][:, layer_idx, :]
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + shuf)
                    fold_aucs = []
                    for train_idx, test_idx in skf.split(X, shuffled):
                        clf = make_pipeline(
                            StandardScaler(),
                            PCA(n_components=min(64, X.shape[1])),
                            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
                        )
                        clf.fit(X[train_idx], shuffled[train_idx])
                        prob = clf.predict_proba(X[test_idx])[:, 1]
                        try:
                            fold_aucs.append(roc_auc_score(shuffled[test_idx], prob))
                        except ValueError:
                            fold_aucs.append(0.5)
                    auc = float(np.mean(fold_aucs))
                    if auc > best:
                        best = auc
            all_aucs_per_shuffle.append(best)
        mean_auc = float(np.mean(all_aucs_per_shuffle))
        std_auc = float(np.std(all_aucs_per_shuffle))
        step_results[s] = {"mean": round(mean_auc, 4), "std": round(std_auc, 4)}
        print(f"  Step {s:>3}: shuffled best-AUC = {mean_auc:.3f} +/- {std_auc:.3f}")

    out = {
        "dataset": dataset_key,
        "model": model_key,
        "n_samples": n_samples,
        "n_shuffles": N_SHUFFLES,
        "layer_grid": layer_grid,
        "step_aucs": {str(k): v for k, v in step_results.items()},
    }
    out_path = f"{in_dir}/shuffle_baseline.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()

    print(f"Saved {out_path}")
    return json.dumps(out, indent=2)


# ============================================================================
# ENTROPY BASELINE (stub for now; original code is ~600 lines, kept separate)
# ============================================================================

def run_entropy_baseline_stub(dataset_key: str, model_key: str, n_chunks: int):
    """Entropy baseline requires full generation, kept in modal_entropy_baseline.py."""
    raise NotImplementedError(
        "Entropy baseline requires generation capability. "
        "Use: .venv/bin/modal run src/modal_entropy_baseline.py"
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

@app.local_entrypoint()
def main(
    baseline_type: str = "shuffle",
    dataset: str = "jsonschema",
    model: str = "llada",
    chunks: int = 8,
):
    if baseline_type == "shuffle":
        print(f"Running shuffle baseline: {dataset}_{model}")
        result = run_shuffle_baseline.remote(dataset, model, chunks)
        print("\n" + result)
    elif baseline_type == "entropy":
        print("For entropy baseline, use: .venv/bin/modal run src/modal_entropy_baseline.py")
        print(f"  --dataset {dataset} --model {model} --chunks {chunks}")
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


@app.local_entrypoint()
def run_all_shuffle(chunks: int = 8):
    """Run shuffle baseline for all dataset/model combinations."""
    handles = []
    for ds in ["jsonschema", "gsm8k", "mbpp", "arc"]:
        for m in ["llada", "dream"]:
            print(f"  Spawning {ds}_{m}")
            handles.append((ds, m, run_shuffle_baseline.spawn(ds, m, chunks)))
    for ds, m, h in handles:
        print(f"  Completing {ds}_{m}")
        h.get()
    print("All shuffle baselines done.")
