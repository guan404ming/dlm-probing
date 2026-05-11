"""AR baseline: run Stage 2 diagnose on Qwen-2.5-7B (AR analogue of LLaDA).

Mirrors modal_sae_diagnose.py but on the AR side:
  - SAE: AwesomeInterpretability/qwen-topk-sae at resid_post_layer_23/trainer_2
    (d_in=3584, d_sae=16384, k=160 -- matched to LLaDA trainer_2)
  - Hidden states: cached at /results/{dataset}_qwen/chunk_off*.npz under
    key 'feats' of shape (N, 28_layers, 3584). One vector per sample
    (last prompt token), no regions.
  - Labels: Qwen's own correctness (ar_labels), not the LLaDA labels.

Purpose: head-to-head DLM-vs-AR comparison of feature-level failure
signatures (silhouette, top-feature enrichment) on the same task.

Usage:
  .venv/bin/modal run src/applications/sae/modal_sae_diagnose_qwen.py \\
      --dataset mbpp
"""

import modal

app = modal.App("sae-diagnose-qwen")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("numpy", "scikit-learn", "torch", "huggingface_hub")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

SAE_REPO = "AwesomeInterpretability/qwen-topk-sae"
SAE_LAYER = 23  # AR analogue of LLaDA's L26 (both ~80% depth)
SAE_TRAINER = 2  # k=160, d_sae=16384, d_in=3584
SAE_PATH_IN_REPO = (
    f"saes__Qwen_Qwen2.5-7B_top_k/"
    f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
)

TOP_N_FEATURES = 20
N_PERMUTATIONS = 1000  # for permutation test


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_diagnose(dataset_key: str, n_chunks: int, total: int):
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
    print(f"Qwen SAE: d_in={d_in}, d_sae={d_sae}, k={k}, layer={SAE_LAYER}")

    state = torch.load(ae_local, map_location="cpu", weights_only=True)
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    sae.load_from_state_dict(state)

    # ---- Load cached Qwen hidden states ----
    chunk_size = (total + n_chunks - 1) // n_chunks
    in_dir = f"/results/{dataset_key}_qwen"
    all_labels = []
    all_feats = []
    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_labels.append(data["ar_labels"])
        all_feats.append(data["feats"])  # (n, 28, 3584)

    labels = np.concatenate(all_labels).astype(int)
    feats = np.concatenate(all_feats)
    n_samples = len(labels)
    n_pass = int(labels.sum())
    n_fail = n_samples - n_pass
    print(
        f"Loaded {n_samples} {dataset_key} samples: "
        f"{n_pass} pass, {n_fail} fail (Qwen self-correctness)"
    )
    print(f"Feats shape: {feats.shape}")

    # ---- SAE encode at SAE_LAYER ----
    x_in = feats[:, SAE_LAYER, :].astype(np.float32)  # (N, 3584)
    with torch.no_grad():
        x_t = torch.from_numpy(x_in).cuda()
        z_topk, _ = sae.encode_both(x_t)
    sae_acts = z_topk.cpu().numpy()  # (N, d_sae) sparse
    print(
        f"SAE activation shape: {sae_acts.shape}, "
        f"nonzero per sample: {(sae_acts != 0).sum(axis=1).mean():.1f}"
    )

    # ---- Fail-vs-pass enrichment ----
    active = (sae_acts > 0).astype(np.float32)
    p_active_fail = active[labels == 0].mean(axis=0)
    p_active_pass = active[labels == 1].mean(axis=0)
    enrichment = p_active_fail - p_active_pass

    fail_top = np.argsort(enrichment)[::-1][:TOP_N_FEATURES]
    pass_top = np.argsort(enrichment)[:TOP_N_FEATURES]

    print(f"\nTop {TOP_N_FEATURES} fail-leaning features (Qwen):")
    fail_rows = []
    for fid in fail_top:
        row = {
            "feature_id": int(fid),
            "enrichment": round(float(enrichment[fid]), 4),
            "p_fail": round(float(p_active_fail[fid]), 4),
            "p_pass": round(float(p_active_pass[fid]), 4),
        }
        fail_rows.append(row)
        print(
            f"  f{fid:>5}: enr={row['enrichment']:+.3f} "
            f"p_fail={row['p_fail']:.3f} p_pass={row['p_pass']:.3f}"
        )
    pass_rows = []
    for fid in pass_top:
        pass_rows.append({
            "feature_id": int(fid),
            "enrichment": round(float(enrichment[fid]), 4),
            "p_fail": round(float(p_active_fail[fid]), 4),
            "p_pass": round(float(p_active_pass[fid]), 4),
        })

    # ---- Cluster fail cases by top-N feature signature ----
    fail_idx = np.where(labels == 0)[0]
    if len(fail_idx) < 4:
        print(f"Too few fail cases ({len(fail_idx)}) for clustering.")
        cluster_sweep = []
        best_k = 0
        best_sil = 0.0
        clusters_summary = []
    else:
        fail_sig = sae_acts[fail_idx][:, fail_top]  # (n_fail, TOP_N)
        cluster_sweep = []
        for n_clusters in [2, 3, 4, 5]:
            if len(fail_idx) <= n_clusters:
                continue
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cids = km.fit_predict(fail_sig)
            try:
                sil = float(silhouette_score(fail_sig, cids))
            except ValueError:
                sil = -1.0
            sizes = [int((cids == c).sum()) for c in range(n_clusters)]
            cluster_sweep.append({
                "n_clusters": n_clusters,
                "silhouette": round(sil, 4),
                "sizes": sizes,
            })
            print(f"K={n_clusters}: silhouette={sil:.3f}, sizes={sizes}")
        best = max(cluster_sweep, key=lambda r: r["silhouette"])
        best_k = best["n_clusters"]
        best_sil = best["silhouette"]
        print(f"\nBest K: {best_k} (silhouette={best_sil})")

        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cids = km.fit_predict(fail_sig)
        clusters_summary = []
        for c in range(best_k):
            mask = cids == c
            if mask.sum() == 0:
                continue
            centroid = fail_sig[mask].mean(axis=0)
            top_in_cluster = np.argsort(centroid)[::-1][:5]
            chars = [
                {"feature_id": int(fail_top[ix]),
                 "mean_act": round(float(centroid[ix]), 4)}
                for ix in top_in_cluster
            ]
            clusters_summary.append({
                "cluster": int(c),
                "size": int(mask.sum()),
                "characteristic_features": chars,
            })
            print(
                f"  cluster {c}: size={int(mask.sum())}, "
                f"top features={[ch['feature_id'] for ch in chars]}"
            )

    # ---- Permutation test on best silhouette ----
    print(f"\nPermutation test (n={N_PERMUTATIONS})...")
    rng = np.random.RandomState(42)
    null_sils = []
    if best_k >= 2 and len(fail_idx) >= 4:
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(labels)
            f_idx = np.where(perm == 0)[0]
            if len(f_idx) < best_k + 1:
                null_sils.append(-1.0)
                continue
            # Recompute enrichment on permuted labels, use that for clustering
            p_f = active[perm == 0].mean(axis=0)
            p_p = active[perm == 1].mean(axis=0)
            enr_perm = p_f - p_p
            ftop = np.argsort(enr_perm)[::-1][:TOP_N_FEATURES]
            sig = sae_acts[f_idx][:, ftop]
            try:
                km = KMeans(n_clusters=best_k, random_state=42, n_init=5)
                cids = km.fit_predict(sig)
                s = float(silhouette_score(sig, cids))
            except ValueError:
                s = -1.0
            null_sils.append(s)
        null_arr = np.array(null_sils)
        p_value = float((null_arr >= best_sil).mean())
        print(
            f"Permutation p(silhouette >= {best_sil}) = "
            f"{p_value:.4f}  "
            f"(null mean={null_arr.mean():.3f}, "
            f"null 95th pct={np.percentile(null_arr, 95):.3f})"
        )
    else:
        p_value = None
        null_arr = np.array([])

    # ---- Persist ----
    results = {
        "model": "qwen-2.5-7b",
        "dataset": dataset_key,
        "sae_repo": SAE_REPO,
        "sae_path": SAE_PATH_IN_REPO,
        "sae_layer": SAE_LAYER,
        "sae_k": k,
        "n_samples": n_samples,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "top_fail_features": fail_rows,
        "top_pass_features": pass_rows,
        "cluster_sweep": cluster_sweep,
        "best_k": best_k,
        "best_silhouette": best_sil,
        "clusters": clusters_summary,
        "permutation_p": p_value,
        "null_silhouette_mean": float(null_arr.mean()) if len(null_arr) else None,
        "null_silhouette_95pct": (
            float(np.percentile(null_arr, 95)) if len(null_arr) else None
        ),
    }
    out_dir = f"/results/{dataset_key}_qwen"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/sae_diagnose_qwen.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved to {out_path}")
    return json.dumps(
        {k_: v for k_, v in results.items() if k_ != "clusters"}, indent=2,
    )


class TopKSAE:
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

        W_enc = pick("encoder.weight", "W_enc")
        b_enc = pick("encoder.bias", "b_enc")
        W_dec = pick("decoder.weight", "W_dec")
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
    chunks: int = 8,
    total: int = 0,
):
    totals = {"jsonschema": 272, "gsm8k": 1319, "mbpp": 257, "arc": 1172}
    if total <= 0:
        total = totals[dataset]
    print(f"Qwen AR SAE diagnose: dataset={dataset}, total={total}")
    summary = run_diagnose.remote(dataset, chunks, total)
    print("\n" + summary)
