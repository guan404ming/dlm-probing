"""Stage 2 diagnose: error-feature mining on MBPP at (L26, step 64).

Stage 0 showed TopK SAE features (k=160, d_sae=16384) match/beat raw probe
on MBPP at (L26, step 64). We use that cell to:

  1. Encode all MBPP samples through the SAE.
  2. Compute per-feature fail-vs-pass activation enrichment:
        enrichment[i] = P(feature_i active | fail) - P(feature_i active | pass)
     Features with large positive enrichment are "fail-leaning";
     large negative are "pass-leaning".
  3. Cluster fail cases by their activation pattern over the top-N
     fail-leaning features (KMeans on binary/value vectors).
  4. Report cluster sizes, characteristic features, and silhouette.

Output: /results/mbpp_llada/sae_diagnose_stage2.json + sparse activations
saved to volume for later interpretation (Stage 3) and steering (Stage 4).

Usage:
  .venv/bin/modal run src/applications/sae/modal_sae_diagnose.py
"""

import modal

app = modal.App("sae-diagnose-stage2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install(
        "numpy",
        "scikit-learn",
        "torch",
        "huggingface_hub",
    )
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYER = 26
SAE_TRAINER = 2  # k=160, d_sae=16384
SAE_PATH_IN_REPO = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"

TARGET_STEP_DEFAULT = 64  # Stage 0: TopK SAE beats raw on MBPP at this step
N_REGIONS = 4
TOP_N_FEATURES = 20  # candidate error features
N_CLUSTERS = 4  # initial guess; we report silhouette to pick


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_diagnose(
    dataset_key: str, model_key: str, n_chunks: int, total: int,
    target_step: int = TARGET_STEP_DEFAULT,
):
    import json
    import os

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    # ---- Load SAE ----
    ae_local = hf_hub_download(
        repo_id=SAE_REPO,
        filename=f"{SAE_PATH_IN_REPO}/ae.pt",
        cache_dir="/hf-cache",
    )
    cfg_local = hf_hub_download(
        repo_id=SAE_REPO,
        filename=f"{SAE_PATH_IN_REPO}/config.json",
        cache_dir="/hf-cache",
    )
    with open(cfg_local) as f:
        sae_cfg = json.load(f)
    d_in = sae_cfg["trainer"]["activation_dim"]
    d_sae = sae_cfg["trainer"]["dict_size"]
    k = sae_cfg["trainer"]["k"]
    print(f"SAE: d_in={d_in}, d_sae={d_sae}, k={k}, layer={SAE_LAYER}")

    state = torch.load(ae_local, map_location="cpu", weights_only=True)
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    sae.load_from_state_dict(state)

    # ---- Load cached LLaDA hidden states for one (step, all regions) ----
    chunk_size = (total + n_chunks - 1) // n_chunks
    in_dir = f"/results/{dataset_key}_{model_key}"
    all_labels = []
    region_feats = {r: [] for r in range(N_REGIONS)}

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_labels.append(data["labels"])
        for r in range(N_REGIONS):
            region_feats[r].append(data[f"feat_s{target_step}_r{r}"])

    labels = np.concatenate(all_labels).astype(int)
    n_samples = len(labels)
    n_fail = int((labels == 0).sum())
    n_pass = int(labels.sum())
    print(
        f"Loaded {n_samples} {dataset_key} samples at step {target_step}: "
        f"{n_pass} pass, {n_fail} fail"
    )

    # ---- SAE encode and mean across regions ----
    sae_acts = []  # list of (N, d_sae) per region
    for r in range(N_REGIONS):
        feats_r = np.concatenate(region_feats[r])  # (N, n_layers, d_in)
        x = feats_r[:, SAE_LAYER, :].astype(np.float32)
        with torch.no_grad():
            x_t = torch.from_numpy(x).cuda()
            z_topk, _ = sae.encode_both(x_t)
            sae_acts.append(z_topk.cpu().numpy())
    sae_mean = np.mean(sae_acts, axis=0)  # (N, d_sae), still mostly sparse
    print(f"Mean SAE activation shape: {sae_mean.shape}, "
          f"nonzero per sample (mean): {(sae_mean != 0).sum(axis=1).mean():.1f}")

    # ---- Per-feature fail-vs-pass enrichment ----
    active = (sae_mean > 0).astype(np.float32)  # (N, d_sae)
    p_active_fail = active[labels == 0].mean(axis=0)
    p_active_pass = active[labels == 1].mean(axis=0)
    enrichment = p_active_fail - p_active_pass  # (d_sae,)

    # Top fail-leaning (positive) and pass-leaning (negative)
    fail_top = np.argsort(enrichment)[::-1][:TOP_N_FEATURES]
    pass_top = np.argsort(enrichment)[:TOP_N_FEATURES]

    print(f"\nTop {TOP_N_FEATURES} fail-leaning features (enrichment, "
          f"P(active|fail), P(active|pass)):")
    fail_feature_rows = []
    for fid in fail_top:
        row = {
            "feature_id": int(fid),
            "enrichment": round(float(enrichment[fid]), 4),
            "p_fail": round(float(p_active_fail[fid]), 4),
            "p_pass": round(float(p_active_pass[fid]), 4),
        }
        fail_feature_rows.append(row)
        print(
            f"  f{fid:>5}: enr={row['enrichment']:+.3f} "
            f"p_fail={row['p_fail']:.3f} p_pass={row['p_pass']:.3f}"
        )

    pass_feature_rows = []
    for fid in pass_top:
        row = {
            "feature_id": int(fid),
            "enrichment": round(float(enrichment[fid]), 4),
            "p_fail": round(float(p_active_fail[fid]), 4),
            "p_pass": round(float(p_active_pass[fid]), 4),
        }
        pass_feature_rows.append(row)

    # ---- Cluster fail cases by top fail-leaning features ----
    fail_idx = np.where(labels == 0)[0]
    fail_sig = sae_mean[fail_idx][:, fail_top]  # (n_fail, TOP_N)

    # Try several K, pick by silhouette
    cluster_results = []
    for n_clusters in [2, 3, 4, 5]:
        if len(fail_idx) <= n_clusters:
            continue
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(fail_sig)
        try:
            sil = silhouette_score(fail_sig, cluster_ids)
        except ValueError:
            sil = -1.0
        cluster_results.append(
            {
                "n_clusters": n_clusters,
                "silhouette": round(float(sil), 4),
                "sizes": [int((cluster_ids == c).sum()) for c in range(n_clusters)],
            }
        )
        print(f"K={n_clusters}: silhouette={sil:.3f}, "
              f"sizes={cluster_results[-1]['sizes']}")

    best = max(cluster_results, key=lambda r: r["silhouette"])
    print(f"\nBest K by silhouette: {best['n_clusters']} "
          f"(silhouette={best['silhouette']})")

    # Re-run KMeans at best K and report characteristic features per cluster
    km = KMeans(n_clusters=best["n_clusters"], random_state=42, n_init=10)
    cluster_ids = km.fit_predict(fail_sig)
    cluster_summary = []
    for c in range(best["n_clusters"]):
        mask = cluster_ids == c
        if mask.sum() == 0:
            continue
        centroid = fail_sig[mask].mean(axis=0)
        # Top features in this cluster's centroid (within the TOP_N feature subset)
        top_in_cluster = np.argsort(centroid)[::-1][:5]
        chars = [
            {
                "feature_id": int(fail_top[ix]),
                "mean_act": round(float(centroid[ix]), 4),
            }
            for ix in top_in_cluster
        ]
        cluster_summary.append(
            {
                "cluster": int(c),
                "size": int(mask.sum()),
                "fail_sample_indices": [int(i) for i in fail_idx[mask].tolist()],
                "characteristic_features": chars,
            }
        )
        print(
            f"  cluster {c}: size={int(mask.sum())}, "
            f"top features={[ch['feature_id'] for ch in chars]}"
        )

    # ---- Permutation test on best silhouette ----
    N_PERMUTATIONS = 1000
    print(f"\nPermutation test (n={N_PERMUTATIONS})...")
    rng = np.random.RandomState(42)
    null_sils = []
    best_k = best["n_clusters"]
    best_sil = best["silhouette"]
    if best_k >= 2 and len(fail_idx) >= 4:
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(labels)
            f_idx_perm = np.where(perm == 0)[0]
            if len(f_idx_perm) < best_k + 1:
                null_sils.append(-1.0)
                continue
            # Recompute enrichment on permuted labels
            p_f = active[perm == 0].mean(axis=0)
            p_p = active[perm == 1].mean(axis=0)
            enr_perm = p_f - p_p
            ftop_perm = np.argsort(enr_perm)[::-1][:TOP_N_FEATURES]
            sig_perm = sae_mean[f_idx_perm][:, ftop_perm]
            try:
                km_p = KMeans(n_clusters=best_k, random_state=42, n_init=5)
                cids_p = km_p.fit_predict(sig_perm)
                s = float(silhouette_score(sig_perm, cids_p))
            except ValueError:
                s = -1.0
            null_sils.append(s)
        null_arr = np.array(null_sils)
        p_value = float((null_arr >= best_sil).mean())
        print(
            f"Permutation p(silhouette >= {best_sil}) = {p_value:.4f} "
            f"(null mean={null_arr.mean():.3f}, "
            f"null 95th pct={np.percentile(null_arr, 95):.3f})"
        )
    else:
        p_value = None
        null_arr = np.array([])

    # ---- Persist ----
    results = {
        "dataset": dataset_key,
        "model": model_key,
        "sae_repo": SAE_REPO,
        "sae_path": SAE_PATH_IN_REPO,
        "sae_layer": SAE_LAYER,
        "sae_k": k,
        "step": target_step,
        "n_samples": n_samples,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "top_fail_features": fail_feature_rows,
        "top_pass_features": pass_feature_rows,
        "cluster_sweep": cluster_results,
        "best_k": best_k,
        "best_silhouette": best_sil,
        "clusters": cluster_summary,
        "permutation_p": p_value,
        "null_silhouette_mean": float(null_arr.mean()) if len(null_arr) else None,
        "null_silhouette_95pct": (
            float(np.percentile(null_arr, 95)) if len(null_arr) else None
        ),
    }
    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/sae_diagnose_stage2_s{target_step}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save sparse activations for downstream stages
    act_path = f"{out_dir}/sae_acts_step{target_step}_L{SAE_LAYER}.npz"
    np.savez_compressed(
        act_path,
        sae_acts_mean=sae_mean.astype(np.float32),
        labels=labels,
    )
    RESULTS_VOL.commit()
    print(f"\nSaved diagnose summary: {out_path}")
    print(f"Saved SAE activations:   {act_path}")
    return json.dumps(
        {k: v for k, v in results.items() if k != "clusters"},
        indent=2,
    )


class TopKSAE:
    """Minimal loader for dictionary_learning AutoEncoderTopK.

    Accepts state dicts with keys:
      encoder.weight (d_sae, d_in), encoder.bias (d_sae,)
      decoder.weight (d_in, d_sae)
      b_dec (d_in,)
    """

    def __init__(self, d_in, d_sae, k):
        import torch

        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.W_enc = None
        self.b_enc = None
        self.W_dec = None
        self.b_dec = None
        self._torch = torch

    def load_from_state_dict(self, state):
        torch = self._torch

        def pick(*candidates):
            for c in candidates:
                if c in state:
                    return state[c]
            return None

        W_enc = pick("encoder.weight", "W_enc", "encoder.W")
        b_enc = pick("encoder.bias", "b_enc")
        W_dec = pick("decoder.weight", "W_dec", "decoder.W")
        b_dec = pick("b_dec", "decoder.bias", "pre_bias")
        assert W_enc is not None and W_dec is not None, (
            f"Could not locate encoder/decoder weights. Keys: {list(state.keys())}"
        )
        self.W_enc = W_enc.cuda().float()
        self.b_enc = (b_enc.cuda().float() if b_enc is not None
                      else torch.zeros(self.d_sae, device="cuda"))
        self.W_dec = W_dec.cuda().float()
        self.b_dec = (b_dec.cuda().float() if b_dec is not None
                      else torch.zeros(self.d_in, device="cuda"))
        if self.W_enc.shape == (self.d_sae, self.d_in):
            self._enc_op = "in_at_wt"
        elif self.W_enc.shape == (self.d_in, self.d_sae):
            self._enc_op = "in_at_w"
        else:
            raise ValueError(f"Unexpected W_enc shape {self.W_enc.shape}")
        print(f"  W_enc {tuple(self.W_enc.shape)} op={self._enc_op}, "
              f"b_dec norm={self.b_dec.norm().item():.3f}")

    def encode_both(self, x):
        torch = self._torch
        x = x - self.b_dec
        if self._enc_op == "in_at_wt":
            pre = x @ self.W_enc.T + self.b_enc
        else:
            pre = x @ self.W_enc + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        topk_vals = topk_vals.relu()
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse, pre


@app.local_entrypoint()
def main(
    dataset: str = "mbpp",
    model: str = "llada",
    chunks: int = 8,
    total: int = 0,
    step: int = TARGET_STEP_DEFAULT,
):
    totals = {"jsonschema": 272, "gsm8k": 1319, "mbpp": 257, "arc": 1172}
    if total <= 0:
        total = totals[dataset]
    print(
        f"SAE diagnose Stage 2: dataset={dataset}, model={model}, "
        f"step={step}, total={total}"
    )
    summary = run_diagnose.remote(dataset, model, chunks, total, step)
    print("\n" + summary)
