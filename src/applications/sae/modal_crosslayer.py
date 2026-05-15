"""Cross-layer SAE diagnose on cached LLaDA-MBPP residuals (EMNLP expansion).

For each available DLM-Scope LLaDA Mask-SAE layer (11, 16, 26, 30), encode
the cached LLaDA-MBPP plateau residuals at step 64 and step 68 through the
layer-specific SAE, compute top-20 fail-enrichment + KMeans silhouette +
permutation null. Establishes whether the mid-plateau signal is L26-specific
or distributed across layers.

Output: /results/mbpp_llada_dense/crosslayer_diagnose.json
"""

import modal

app = modal.App("sae-crosslayer")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("torch>=2.0", "numpy", "scikit-learn>=1.3", "huggingface_hub")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYERS = [11, 16, 26, 30]
STEPS = [64, 68]
TOP_N = 20
N_PERMUTATIONS = 500
N_REGIONS = 4


class TopKSAE:
    def __init__(self):
        import torch
        self._torch = torch

    def load(self, state, k, d_in, d_sae):
        import torch
        self.k = k
        self.d_in = d_in
        self.d_sae = d_sae
        W_enc = state["encoder.weight"]
        b_enc = state.get("encoder.bias")
        W_dec = state["decoder.weight"]
        b_dec = state.get("b_dec")
        if b_dec is None:
            b_dec = state.get("decoder.bias")
        self.W_enc = W_enc.cuda().float()
        self.b_enc = b_enc.cuda().float() if b_enc is not None else torch.zeros(d_sae, device="cuda")
        self.W_dec = W_dec.cuda().float()
        self.b_dec = b_dec.cuda().float() if b_dec is not None else torch.zeros(d_in, device="cuda")
        self._enc_op = "in_at_wt" if self.W_enc.shape == (d_sae, d_in) else "in_at_w"

    def encode(self, x):
        import torch
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
def run_crosslayer(n_chunks: int = 8, total: int = 257):
    import json
    import os

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    os.environ["HF_HOME"] = "/hf-cache"
    RESULTS_VOL.reload()

    # Load chunks (per-region feats across all 33 layers)
    in_dir = "/results/mbpp_llada_dense"
    chunk_size = (total + n_chunks - 1) // n_chunks
    all_labels = []
    region_feats = {s: {r: [] for r in range(N_REGIONS)} for s in STEPS}
    for i in range(n_chunks):
        off = i * chunk_size
        path = f"{in_dir}/chunk_off{off}.npz"
        if not os.path.exists(path):
            continue
        d = np.load(path)
        all_labels.append(d["labels"])
        for s in STEPS:
            for r in range(N_REGIONS):
                region_feats[s][r].append(d[f"feat_s{s}_r{r}"])
    labels = np.concatenate(all_labels).astype(int)
    n_fail = int((labels == 0).sum())
    n_pass = int(labels.sum())
    print(f"loaded {len(labels)}: pass={n_pass} fail={n_fail}")

    results = {"sae_repo": SAE_REPO, "n_pass": n_pass, "n_fail": n_fail, "layers": []}

    for sae_layer in SAE_LAYERS:
        print(f"\n========== SAE layer {sae_layer} ==========")
        # Load layer-specific SAE
        sae_path = f"resid_post_layer_{sae_layer}/trainer_2"
        ae = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path}/ae.pt", cache_dir="/hf-cache")
        cfg = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path}/config.json", cache_dir="/hf-cache")
        with open(cfg) as f:
            sae_cfg = json.load(f)
        d_in = sae_cfg["trainer"]["activation_dim"]
        d_sae = sae_cfg["trainer"]["dict_size"]
        k = sae_cfg["trainer"]["k"]
        state = torch.load(ae, map_location="cpu", weights_only=True)
        sae = TopKSAE()
        sae.load(state, k, d_in, d_sae)
        print(f"  SAE: d_in={d_in} d_sae={d_sae} k={k}")

        layer_result = {"sae_layer": sae_layer, "d_sae": d_sae, "steps": []}

        for s in STEPS:
            sae_acts = []
            for r in range(N_REGIONS):
                feats = np.concatenate(region_feats[s][r])  # (N, 33, d_in)
                x = feats[:, sae_layer, :].astype(np.float32)
                with torch.no_grad():
                    z = sae.encode(torch.from_numpy(x).cuda()).cpu().numpy()
                    sae_acts.append(z)
            sae_mean = np.mean(sae_acts, axis=0)
            active = (sae_mean > 0).astype(np.float32)
            fail_idx = np.where(labels == 0)[0]
            p_fail = active[labels == 0].mean(axis=0)
            p_pass = active[labels == 1].mean(axis=0)
            enr = p_fail - p_pass
            top = np.argsort(enr)[::-1][:TOP_N]

            # Silhouette on fails with best K
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
                top_p = np.argsort(pf - pp)[::-1][:TOP_N]
                sig_p = sae_mean[fi][:, top_p]
                try:
                    cids = KMeans(n_clusters=best["K"], random_state=42, n_init=5).fit_predict(sig_p)
                    null_sils.append(float(silhouette_score(sig_p, cids)))
                except ValueError:
                    null_sils.append(-1.0)
            null_arr = np.array(null_sils)
            p_val = float((null_arr >= best["sil"]).mean())
            top_feats = [{"feature_id": int(t), "enrichment": float(enr[t])} for t in top[:5]]
            layer_result["steps"].append({
                "step": s,
                "K": best["K"],
                "silhouette": round(best["sil"], 4),
                "null_mean": round(float(null_arr.mean()), 4),
                "gap": round(best["sil"] - float(null_arr.mean()), 4),
                "p_value": round(p_val, 4),
                "top_5_features": top_feats,
            })
            print(f"  step {s}: sil={best['sil']:.4f} null={null_arr.mean():.4f} "
                  f"gap={best['sil']-null_arr.mean():+.4f} p={p_val:.4f} "
                  f"top1=f{top_feats[0]['feature_id']} (+{top_feats[0]['enrichment']:.3f})")

        results["layers"].append(layer_result)

    out_path = "/results/mbpp_llada_dense/crosslayer_diagnose.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print(run_crosslayer.remote())
