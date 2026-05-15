"""Held-out feature selection silhouette (review W2).

For each LLaDA-MBPP plateau step, split fail samples into 5 stratified folds,
select top-20 fail-enriched features on the held-in 4 folds, compute KMeans
silhouette on the held-out fold (mapped via labels-only selection). Repeat
for permutation null: shuffle labels, re-run the same held-out pipeline.

This addresses the residual selection-bias concern after the existing
permutation null already re-selects features inside each shuffle. The
held-out variant guarantees the silhouette samples are disjoint from the
samples that drove feature selection.

Output: /results/mbpp_llada_dense/heldout_selection.json
"""

import modal

app = modal.App("sae-heldout")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("numpy", "scikit-learn>=1.3", "torch", "huggingface_hub")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYER = 26
SAE_TRAINER = 2
SAE_PATH = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"

STEPS = [48, 52, 56, 60, 64, 68, 72, 76, 80]
N_FOLDS = 5
N_PERMUTATIONS = 500
TOP_N = 20
N_REGIONS = 4


class TopKSAE:
    def __init__(self, d_in, d_sae, k):
        import torch
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self._torch = torch

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
    image=image, gpu="A100", timeout=7200,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_heldout(n_chunks: int = 8, total: int = 257):
    import json
    import os

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import KFold

    os.environ["HF_HOME"] = "/hf-cache"
    RESULTS_VOL.reload()

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

    # Load plateau cache
    in_dir = "/results/mbpp_llada_dense"
    chunk_size = (total + n_chunks - 1) // n_chunks
    all_labels = []
    region_feats = {s: {r: [] for r in range(N_REGIONS)} for s in STEPS}
    for i in range(n_chunks):
        off = i * chunk_size
        path = f"{in_dir}/chunk_off{off}.npz"
        if not os.path.exists(path):
            continue
        data = np.load(path)
        all_labels.append(data["labels"])
        for s in STEPS:
            for r in range(N_REGIONS):
                region_feats[s][r].append(data[f"feat_s{s}_r{r}"])
    labels = np.concatenate(all_labels).astype(int)
    n_fail = int((labels == 0).sum())
    n_pass = int(labels.sum())
    print(f"loaded {len(labels)}: pass={n_pass} fail={n_fail}")

    results = {"sae_layer": SAE_LAYER, "n_pass": n_pass, "n_fail": n_fail,
               "n_folds": N_FOLDS, "top_n": TOP_N, "steps": []}

    for s in STEPS:
        print(f"\n=== step {s} held-out ===")
        sae_acts = []
        for r in range(N_REGIONS):
            feats = np.concatenate(region_feats[s][r])
            x = feats[:, SAE_LAYER, :].astype(np.float32)
            with torch.no_grad():
                z = sae.encode(torch.from_numpy(x).cuda()).cpu().numpy()
                sae_acts.append(z)
        sae_mean = np.mean(sae_acts, axis=0)
        active = (sae_mean > 0).astype(np.float32)
        fail_idx = np.where(labels == 0)[0]
        pass_idx = np.where(labels == 1)[0]

        # Held-out silhouette across folds (on fails)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_sils = []
        for sel_idx, eval_idx in kf.split(fail_idx):
            sel_fail = fail_idx[sel_idx]
            eval_fail = fail_idx[eval_idx]
            # Select top-N on sel_fail vs all passes
            p_fail_sel = active[sel_fail].mean(axis=0)
            p_pass = active[pass_idx].mean(axis=0)
            enr_sel = p_fail_sel - p_pass
            top = np.argsort(enr_sel)[::-1][:TOP_N]
            # Silhouette on eval_fail with selected features
            sig_eval = sae_mean[eval_fail][:, top]
            best = -1.0
            for K in [2, 3, 4, 5]:
                if len(eval_fail) <= K:
                    continue
                km = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(sig_eval)
                try:
                    sl = float(silhouette_score(sig_eval, km))
                    if sl > best:
                        best = sl
                except ValueError:
                    pass
            fold_sils.append(best)
        sil_mean = float(np.mean(fold_sils))
        sil_std = float(np.std(fold_sils))

        # Permutation null: shuffle labels, run same held-out pipeline
        rng = np.random.RandomState(42)
        null_means = []
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(labels)
            fi_p = np.where(perm == 0)[0]
            pi_p = np.where(perm == 1)[0]
            kf2 = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            fold_sils_p = []
            for sel_idx, eval_idx in kf2.split(fi_p):
                sel = fi_p[sel_idx]
                ev = fi_p[eval_idx]
                pf = active[sel].mean(axis=0)
                pp = active[pi_p].mean(axis=0)
                enr_p = pf - pp
                top_p = np.argsort(enr_p)[::-1][:TOP_N]
                sig_e = sae_mean[ev][:, top_p]
                bp = -1.0
                for K in [2, 3, 4, 5]:
                    if len(ev) <= K:
                        continue
                    try:
                        km = KMeans(n_clusters=K, random_state=42, n_init=5).fit_predict(sig_e)
                        sl = float(silhouette_score(sig_e, km))
                        if sl > bp:
                            bp = sl
                    except ValueError:
                        pass
                fold_sils_p.append(bp)
            null_means.append(float(np.mean(fold_sils_p)))
        null_arr = np.array(null_means)
        p_held = float((null_arr >= sil_mean).mean())

        step_result = {
            "step": s,
            "heldout_sil_mean": round(sil_mean, 4),
            "heldout_sil_std": round(sil_std, 4),
            "heldout_null_mean": round(float(null_arr.mean()), 4),
            "heldout_gap": round(sil_mean - float(null_arr.mean()), 4),
            "heldout_p": round(p_held, 4),
            "n_perm": N_PERMUTATIONS,
        }
        print(f"  step {s}: held-out sil={sil_mean:.4f}+/-{sil_std:.4f} "
              f"null={null_arr.mean():.4f} gap={step_result['heldout_gap']} "
              f"p={p_held:.4f}")
        results["steps"].append(step_result)

    out_path = "/results/mbpp_llada_dense/heldout_selection.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    print(run_heldout.remote())
