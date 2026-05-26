"""Alternative-SAE robustness at L26 (reviewer #4).

Re-encode the cached LLaDA-MBPP plateau residuals (step 64/68) through DIFFERENT
SAE dictionaries at the same layer L26 (trainers 0,2,4 of the DLM-Scope repo,
which differ in sparsity k / dictionary), recomputing top-20 fail-enrichment +
KMeans silhouette + permutation null. If the plateau signal and the dominance
of f15601 survive a dictionary swap, the finding is not an artifact of one SAE.

Output: /results/mbpp_llada_dense/altsae_diagnose.json
"""

import modal

app = modal.App("sae-altsae")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git", "curl")
         .pip_install("torch>=2.0", "numpy", "scikit-learn>=1.3", "huggingface_hub"))
RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYER = 26
TRAINERS = [0, 2, 4]
STEPS = [64]
TOP_N = 20
N_PERM = 500
N_REGIONS = 4


@app.function(image=image, gpu="A100", timeout=3600,
              volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL})
def run(n_chunks: int = 8, total: int = 257):
    import json, os, numpy as np, torch
    from huggingface_hub import hf_hub_download
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    os.environ["HF_HOME"] = "/hf-cache"
    RESULTS_VOL.reload()

    in_dir = "/results/mbpp_llada"
    chunk_size = (total + n_chunks - 1) // n_chunks
    all_labels = []
    region_feats = {s: {r: [] for r in range(N_REGIONS)} for s in STEPS}
    for i in range(n_chunks):
        path = f"{in_dir}/chunk_off{i*chunk_size}.npz"
        if not os.path.exists(path):
            continue
        d = np.load(path)
        all_labels.append(d["labels"])
        for s in STEPS:
            for r in range(N_REGIONS):
                region_feats[s][r].append(d[f"feat_s{s}_r{r}"])
    labels = np.concatenate(all_labels).astype(int)
    print(f"loaded {len(labels)} pass={labels.sum()} fail={(labels==0).sum()}")

    def encode(state, k, x):
        We = state["encoder.weight"].cuda().float()
        be = state.get("encoder.bias")
        be = be.cuda().float() if be is not None else torch.zeros(We.shape[0], device="cuda")
        bd = state.get("b_dec", state.get("decoder.bias"))
        d_in = x.shape[1]
        bd = bd.cuda().float() if bd is not None else torch.zeros(d_in, device="cuda")
        xt = torch.from_numpy(x).cuda() - bd
        pre = (xt @ We.T + be) if We.shape[1] == d_in else (xt @ We + be)
        tv, ti = pre.topk(k, dim=-1); tv = tv.relu()
        sp = torch.zeros_like(pre); sp.scatter_(-1, ti, tv)
        return sp.cpu().numpy()

    def diagnose(sae_mean):
        active = (sae_mean > 0).astype(np.float32)
        fi = np.where(labels == 0)[0]
        enr = active[labels == 0].mean(0) - active[labels == 1].mean(0)
        top = np.argsort(enr)[::-1][:TOP_N]
        sig = sae_mean[fi][:, top]
        best = None
        for K in [2, 3, 4, 5]:
            km = KMeans(K, random_state=42, n_init=10).fit_predict(sig)
            try:
                sl = float(silhouette_score(sig, km))
            except ValueError:
                sl = -1.0
            if best is None or sl > best["sil"]:
                best = {"K": K, "sil": sl}
        rng = np.random.RandomState(42); null = []
        for _ in range(N_PERM):
            perm = rng.permutation(labels); fip = np.where(perm == 0)[0]
            if len(fip) < best["K"] + 1:
                null.append(-1.0); continue
            ep = active[perm == 0].mean(0) - active[perm == 1].mean(0)
            tp = np.argsort(ep)[::-1][:TOP_N]
            try:
                c = KMeans(best["K"], random_state=42, n_init=5).fit_predict(sae_mean[fip][:, tp])
                null.append(float(silhouette_score(sae_mean[fip][:, tp], c)))
            except ValueError:
                null.append(-1.0)
        na = np.array(null)
        return best, enr, top, float(na.mean()), float((na >= best["sil"]).mean())

    out = {"sae_repo": SAE_REPO, "layer": SAE_LAYER, "trainers": []}
    for tr in TRAINERS:
        sp = f"resid_post_layer_{SAE_LAYER}/trainer_{tr}"
        ae = hf_hub_download(repo_id=SAE_REPO, filename=f"{sp}/ae.pt", cache_dir="/hf-cache")
        cfg = hf_hub_download(repo_id=SAE_REPO, filename=f"{sp}/config.json", cache_dir="/hf-cache")
        c = json.load(open(cfg)); k = c["trainer"]["k"]; dsae = c["trainer"]["dict_size"]
        state = torch.load(ae, map_location="cpu", weights_only=True)
        print(f"\n== trainer {tr}: k={k} d_sae={dsae} ==")
        tr_res = {"trainer": tr, "k": k, "d_sae": dsae, "steps": []}
        for s in STEPS:
            acts = []
            for r in range(N_REGIONS):
                feats = np.concatenate(region_feats[s][r])
                acts.append(encode(state, k, feats[:, SAE_LAYER, :].astype(np.float32)))
            sae_mean = np.mean(acts, 0)
            best, enr, top, nmean, pval = diagnose(sae_mean)
            tr_res["steps"].append({"step": s, "K": best["K"], "silhouette": round(best["sil"], 4),
                                    "null_mean": round(nmean, 4), "gap": round(best["sil"] - nmean, 4),
                                    "p_value": round(pval, 4),
                                    "top5": [{"feature_id": int(t), "enr": round(float(enr[t]), 4)} for t in top[:5]]})
            print(f"  s{s}: K={best['K']} sil={best['sil']:.3f} gap={best['sil']-nmean:+.3f} "
                  f"p={pval:.4f} top1=f{int(top[0])}(+{enr[top[0]]:.3f})")
        out["trainers"].append(tr_res)
    with open("/results/mbpp_llada/altsae_diagnose.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    return json.dumps(out)


@app.local_entrypoint()
def main():
    print(run.remote())
