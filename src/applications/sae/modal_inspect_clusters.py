"""Stage 3 (Path B): inspect MBPP fail clusters from Stage 2 diagnose output.

Loads the Stage 2 JSON (cluster -> fail_sample_indices) and the MBPP sanitized
test split (sorted by task_id, matching the probe pipeline order), then prints
prompts + ground-truth code for the first few samples of each cluster.

Purpose: human inspection to name the two error modes implied by the
silhouette=0.66 clustering on top fail-leaning SAE features.

Usage:
  .venv/bin/modal run src/applications/sae/modal_inspect_clusters.py
  .venv/bin/modal run src/applications/sae/modal_inspect_clusters.py --n-per-cluster 5
"""

import modal

app = modal.App("sae-inspect-clusters")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)


@app.function(
    image=image,
    timeout=600,
    volumes={"/results": RESULTS_VOL},
)
def inspect(dataset_key: str, model_key: str, n_per_cluster: int):
    import json

    from datasets import load_dataset

    RESULTS_VOL.reload()
    diag_path = f"/results/{dataset_key}_{model_key}/sae_diagnose_stage2.json"
    with open(diag_path) as f:
        diag = json.load(f)

    print(f"Dataset: {diag['dataset']}, Model: {diag['model']}")
    print(f"Step: {diag['step']}, Layer: {diag['sae_layer']}, k={diag['sae_k']}")
    print(f"Samples: {diag['n_samples']} ({diag['n_pass']} pass, "
          f"{diag['n_fail']} fail)")
    print(f"Best K: {diag['best_k']} (silhouette={diag['best_silhouette']})")
    print()

    print("Top 5 fail-leaning features:")
    for r in diag["top_fail_features"][:5]:
        print(
            f"  f{r['feature_id']:>5}: enrichment={r['enrichment']:+.3f} "
            f"p_fail={r['p_fail']:.3f} p_pass={r['p_pass']:.3f}"
        )
    print()

    # Load MBPP in the same order the probe pipeline used
    if dataset_key == "mbpp":
        ds = load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        )
        instances = sorted(list(ds), key=lambda x: x["task_id"])
    else:
        raise ValueError(
            f"Inspection currently only supports mbpp, got {dataset_key}"
        )

    for cluster in diag["clusters"]:
        cid = cluster["cluster"]
        size = cluster["size"]
        feats = [c["feature_id"] for c in cluster["characteristic_features"]]
        print(f"{'=' * 78}")
        print(f"CLUSTER {cid}: size={size}, characteristic features={feats}")
        print(f"{'=' * 78}")
        sample_idxs = cluster["fail_sample_indices"][:n_per_cluster]
        for idx in sample_idxs:
            inst = instances[idx]
            print(f"\n--- sample idx={idx}, task_id={inst['task_id']} ---")
            print(f"PROMPT: {inst['prompt']}")
            print(f"TESTS:")
            for t in inst["test_list"]:
                print(f"  {t}")
            code = inst.get("code", "").strip()
            if code:
                # Truncate long code at ~20 lines
                lines = code.split("\n")
                if len(lines) > 25:
                    code = "\n".join(lines[:25]) + "\n  ..."
                print(f"GROUND TRUTH CODE:\n{code}")
            print()
    return diag


@app.local_entrypoint()
def main(
    dataset: str = "mbpp",
    model: str = "llada",
    n_per_cluster: int = 4,
):
    inspect.remote(dataset, model, n_per_cluster)
