"""Post-hoc selection / top-N / clustering-method sensitivity on LLaDA-MBPP.

Reanalysis of cached plateau residuals (no new generation). For the two
strongest plateau steps (64 and 68), report:

  (a) Top-N sweep: silhouette and permutation p for N in {10, 20, 30, 50}
      under the standard KMeans + permutation-null pipeline.
  (b) Clustering method comparison at N=20: KMeans vs AgglomerativeClustering
      (Ward) vs HDBSCAN. Report silhouette and best K.
  (c) Labels-hidden test at N=20: cluster the full fail+pass set with K=2
      and report Adjusted Rand Index (ARI) and Adjusted Mutual Information
      (AMI) between the unsupervised clusters and the true fail/pass labels.
      Non-zero ARI indicates the feature space carries fail/pass structure
      that does NOT require label-guided selection.

Output: /results/mbpp_llada_dense/sensitivity_analysis.json
"""

import modal

app = modal.App("sae-sensitivity")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install(
        "numpy", "scikit-learn>=1.3", "torch", "huggingface_hub",
    )
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYER = 26
SAE_TRAINER = 2
SAE_PATH = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"

N_REGIONS = 4
STEPS_TO_ANALYZE = [64, 68]
TOP_NS = [10, 20, 30, 50]
N_PERMUTATIONS = 1000


class TopKSAE:
    def __init__(self, d_in, d_sae, k):
        import torch
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self._torch = torch
        self.W_enc = self.b_enc = self.W_dec = self.b_dec = None

    def load(self, state):
        torch = self._torch
        W_enc = state["encoder.weight"]
        b_enc = state.get("encoder.bias")
        W_dec = state["decoder.weight"]
        b_dec = state.get("b_dec")
        if b_dec is None:
            b_dec = state.get("decoder.bias")
        self.W_enc = W_enc.cuda().float()
        self.b_enc = b_enc.cuda().float() if b_enc is not None else torch.zeros(self.d_sae, device="cuda")
        self.W_dec = W_dec.cuda().float()
        self.b_dec = b_dec.cuda().float() if b_dec is not None else torch.zeros(self.d_in, device="cuda")
        self._enc_op = "in_at_wt" if self.W_enc.shape == (self.d_sae, self.d_in) else "in_at_w"

    def encode(self, x):
        torch = self._torch
        x = x - self.b_dec
        pre = (x @ self.W_enc.T + self.b_enc) if self._enc_op == "in_at_wt" else (x @ self.W_enc + self.b_enc)
        tv, ti = pre.topk(self.k, dim=-1)
        tv = tv.relu()
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, ti, tv)
        return sparse


@app.function(
    image=image, gpu="A100", timeout=3600,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_sensitivity(n_chunks: int = 8, total: int = 257):
    import json
    import os

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
    from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

    os.environ["HF_HOME"] = "/hf-cache"
    RESULTS_VOL.reload()

    # SAE
    ae_local = hf_hub_download(repo_id=SAE_REPO, filename=f"{SAE_PATH}/ae.pt", cache_dir="/hf-cache")
    cfg_local = hf_hub_download(repo_id=SAE_REPO, filename=f"{SAE_PATH}/config.json", cache_dir="/hf-cache")
    with open(cfg_local) as f:
        sae_cfg = json.load(f)
    d_in = sae_cfg["trainer"]["activation_dim"]
    d_sae = sae_cfg["trainer"]["dict_size"]
    k = sae_cfg["trainer"]["k"]
    sae = TopKSAE(d_in, d_sae, k)
    sae.load(torch.load(ae_local, map_location="cpu", weights_only=True))
    print(f"SAE loaded: d_in={d_in} d_sae={d_sae} k={k}")

    # Load cached plateau chunks (seed 0)
    in_dir = "/results/mbpp_llada_dense"
    chunk_size = (total + n_chunks - 1) // n_chunks
    all_labels = []
    region_feats = {s: {r: [] for r in range(N_REGIONS)} for s in STEPS_TO_ANALYZE}
    for i in range(n_chunks):
        off = i * chunk_size
        path = f"{in_dir}/chunk_off{off}.npz"
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        data = np.load(path)
        all_labels.append(data["labels"])
        for s in STEPS_TO_ANALYZE:
            for r in range(N_REGIONS):
                region_feats[s][r].append(data[f"feat_s{s}_r{r}"])
    labels = np.concatenate(all_labels).astype(int)
    n_fail = int((labels == 0).sum())
    n_pass = int(labels.sum())
    print(f"loaded {len(labels)}: pass={n_pass} fail={n_fail}")

    results = {"sae_layer": SAE_LAYER, "steps": [], "n_pass": n_pass, "n_fail": n_fail}

    for s in STEPS_TO_ANALYZE:
        print(f"\n=== step {s} ===")
        # Encode all regions
        sae_acts = []
        for r in range(N_REGIONS):
            feats = np.concatenate(region_feats[s][r])
            x = feats[:, SAE_LAYER, :].astype(np.float32)
            with torch.no_grad():
                z = sae.encode(torch.from_numpy(x).cuda()).cpu().numpy()
                sae_acts.append(z)
        sae_mean = np.mean(sae_acts, axis=0)
        active = (sae_mean > 0).astype(np.float32)
        p_fail = active[labels == 0].mean(axis=0)
        p_pass = active[labels == 1].mean(axis=0)
        enr = p_fail - p_pass
        fail_idx = np.where(labels == 0)[0]
        sorted_feats = np.argsort(enr)[::-1]

        step_result = {"step": s, "top_n_sweep": [], "method_compare": [], "labels_hidden": {}}

        # ---- (a) Top-N sweep ----
        for N in TOP_NS:
            top = sorted_feats[:N]
            sig = sae_mean[fail_idx][:, top]
            best = None
            for K in [2, 3, 4, 5]:
                if len(fail_idx) <= K:
                    continue
                km = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(sig)
                try:
                    sl = float(silhouette_score(sig, km))
                except ValueError:
                    sl = -1.0
                if best is None or sl > best["sil"]:
                    best = {"K": K, "sil": sl}
            # Permutation null
            rng = np.random.RandomState(42)
            null_sils = []
            for _ in range(N_PERMUTATIONS):
                perm = rng.permutation(labels)
                fi = np.where(perm == 0)[0]
                if len(fi) < best["K"] + 1:
                    null_sils.append(-1.0)
                    continue
                pf = active[perm == 0].mean(axis=0)
                pp = active[perm == 1].mean(axis=0)
                top_p = np.argsort(pf - pp)[::-1][:N]
                sig_p = sae_mean[fi][:, top_p]
                try:
                    cids = KMeans(n_clusters=best["K"], random_state=42, n_init=5).fit_predict(sig_p)
                    null_sils.append(float(silhouette_score(sig_p, cids)))
                except ValueError:
                    null_sils.append(-1.0)
            null_arr = np.array(null_sils)
            p = float((null_arr >= best["sil"]).mean())
            step_result["top_n_sweep"].append({
                "N": N, "K": best["K"], "silhouette": round(best["sil"], 4),
                "null_mean": round(float(null_arr.mean()), 4),
                "gap": round(best["sil"] - float(null_arr.mean()), 4),
                "p_value": round(p, 4),
            })
            print(f"  N={N:>3}: K={best['K']} sil={best['sil']:.3f} "
                  f"null={null_arr.mean():.3f} gap={best['sil']-null_arr.mean():+.3f} p={p:.4f}")

        # ---- (b) Clustering method at N=20 ----
        top = sorted_feats[:20]
        sig = sae_mean[fail_idx][:, top]
        # KMeans (already done in N=20 sweep but report best K)
        km_best = None
        for K in [2, 3, 4, 5]:
            cids = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(sig)
            try:
                sl = float(silhouette_score(sig, cids))
            except ValueError:
                sl = -1.0
            if km_best is None or sl > km_best["sil"]:
                km_best = {"K": K, "sil": sl}
        step_result["method_compare"].append({"method": "KMeans", "K": km_best["K"], "silhouette": round(km_best["sil"], 4)})
        print(f"  KMeans(K={km_best['K']}): sil={km_best['sil']:.3f}")

        # Agglomerative (Ward)
        ag_best = None
        for K in [2, 3, 4, 5]:
            cids = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(sig)
            try:
                sl = float(silhouette_score(sig, cids))
            except ValueError:
                sl = -1.0
            if ag_best is None or sl > ag_best["sil"]:
                ag_best = {"K": K, "sil": sl}
        step_result["method_compare"].append({"method": "Agglomerative-Ward", "K": ag_best["K"], "silhouette": round(ag_best["sil"], 4)})
        print(f"  Agglomerative(K={ag_best['K']}): sil={ag_best['sil']:.3f}")

        # HDBSCAN (density-based; no need to choose K)
        try:
            hd = HDBSCAN(min_cluster_size=5)
            cids_hd = hd.fit_predict(sig)
            n_clusters_hd = len(set(cids_hd) - {-1})
            n_noise = int((cids_hd == -1).sum())
            if n_clusters_hd >= 2:
                # silhouette on non-noise points
                mask_nn = cids_hd != -1
                sl_hd = float(silhouette_score(sig[mask_nn], cids_hd[mask_nn]))
            else:
                sl_hd = None
            step_result["method_compare"].append({
                "method": "HDBSCAN", "n_clusters": n_clusters_hd, "n_noise": n_noise,
                "silhouette": round(sl_hd, 4) if sl_hd is not None else None,
            })
            print(f"  HDBSCAN: n_clusters={n_clusters_hd} n_noise={n_noise} "
                  f"sil={sl_hd if sl_hd else 'N/A'}")
        except Exception as e:
            print(f"  HDBSCAN failed: {e}")
            step_result["method_compare"].append({"method": "HDBSCAN", "error": str(e)})

        # ---- (c) Labels-hidden test at N=20 ----
        # Cluster fail+pass together with K=2, check correspondence to labels
        sig_all = sae_mean[:, top]  # all samples
        cids_all = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(sig_all)
        ari = float(adjusted_rand_score(labels, cids_all))
        ami = float(adjusted_mutual_info_score(labels, cids_all))
        # Permutation null for ARI
        rng = np.random.RandomState(42)
        null_aris = []
        for _ in range(200):
            perm = rng.permutation(labels)
            null_aris.append(float(adjusted_rand_score(perm, cids_all)))
        null_aris = np.array(null_aris)
        p_ari = float((null_aris >= ari).mean())
        step_result["labels_hidden"] = {
            "N": 20, "K": 2, "ARI": round(ari, 4),
            "ARI_null_mean": round(float(null_aris.mean()), 4),
            "ARI_p": round(p_ari, 4), "AMI": round(ami, 4),
        }
        print(f"  Labels-hidden K=2 on fail+pass: ARI={ari:.4f} "
              f"(null mean={null_aris.mean():.4f}, p={p_ari:.4f}) AMI={ami:.4f}")

        results["steps"].append(step_result)

    out_path = "/results/mbpp_llada_dense/sensitivity_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print(run_sensitivity.remote())
