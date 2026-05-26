"""Pooling-granularity sensitivity at L26 step 64 (reviewer minor concern).

Region-mean pooling could blur localized signals and manufacture the plateau
cluster structure. We re-run the top-20 fail-enrichment + silhouette + null on
each of the 4 generation regions SEPARATELY (LLaDA-MBPP step 64, L26 trainer 2),
and on the region-mean, to check the structure is not an averaging artifact.

Output: /results/mbpp_llada/pooling_diagnose.json
"""

import modal

app = modal.App("sae-pooling")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git", "curl")
         .pip_install("torch>=2.0", "numpy", "scikit-learn>=1.3", "huggingface_hub"))
RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)
SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_LAYER = 26
STEP = 64
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
    cs = (total + n_chunks - 1) // n_chunks
    labels = []
    rf = {r: [] for r in range(N_REGIONS)}
    for i in range(n_chunks):
        p = f"/results/mbpp_llada/chunk_off{i*cs}.npz"
        if not os.path.exists(p):
            continue
        d = np.load(p); labels.append(d["labels"])
        for r in range(N_REGIONS):
            rf[r].append(d[f"feat_s{STEP}_r{r}"])
    labels = np.concatenate(labels).astype(int)

    sp = f"resid_post_layer_{SAE_LAYER}/trainer_2"
    ae = hf_hub_download(repo_id=SAE_REPO, filename=f"{sp}/ae.pt", cache_dir="/hf-cache")
    cfg = hf_hub_download(repo_id=SAE_REPO, filename=f"{sp}/config.json", cache_dir="/hf-cache")
    k = json.load(open(cfg))["trainer"]["k"]
    st = torch.load(ae, map_location="cpu", weights_only=True)
    We = st["encoder.weight"].cuda().float(); be = st["encoder.bias"].cuda().float()
    bd = st.get("b_dec", st.get("decoder.bias")).cuda().float()

    def enc(x):
        xt = torch.from_numpy(x).cuda() - bd
        pre = (xt @ We.T + be) if We.shape[1] == x.shape[1] else (xt @ We + be)
        tv, ti = pre.topk(k, -1); tv = tv.relu()
        s = torch.zeros_like(pre); s.scatter_(-1, ti, tv); return s.cpu().numpy()

    def diag(sm):
        active = (sm > 0).astype(np.float32); fi = np.where(labels == 0)[0]
        enr = active[labels == 0].mean(0) - active[labels == 1].mean(0)
        top = np.argsort(enr)[::-1][:TOP_N]; sig = sm[fi][:, top]; best = None
        for K in [2, 3, 4, 5]:
            try:
                sl = float(silhouette_score(sig, KMeans(K, random_state=42, n_init=10).fit_predict(sig)))
            except ValueError:
                sl = -1.0
            if best is None or sl > best[1]:
                best = (K, sl)
        rng = np.random.RandomState(42); null = []
        for _ in range(N_PERM):
            perm = rng.permutation(labels); fip = np.where(perm == 0)[0]
            if len(fip) < best[0] + 1:
                null.append(-1.0); continue
            tp = np.argsort(active[perm == 0].mean(0) - active[perm == 1].mean(0))[::-1][:TOP_N]
            try:
                null.append(float(silhouette_score(sm[fip][:, tp], KMeans(best[0], random_state=42, n_init=5).fit_predict(sm[fip][:, tp]))))
            except ValueError:
                null.append(-1.0)
        na = np.array(null)
        return {"K": best[0], "silhouette": round(best[1], 4), "null_mean": round(float(na.mean()), 4),
                "gap": round(best[1] - float(na.mean()), 4), "p_value": round(float((na >= best[1]).mean()), 4),
                "top1": int(top[0]), "top1_enr": round(float(enr[top[0]]), 4)}

    region_acts = {r: enc(np.concatenate(rf[r])[:, SAE_LAYER, :].astype(np.float32)) for r in range(N_REGIONS)}
    out = {"layer": SAE_LAYER, "step": STEP, "per_region": {}, "region_mean": diag(np.mean(list(region_acts.values()), 0))}
    for r in range(N_REGIONS):
        out["per_region"][r] = diag(region_acts[r])
        print(f"region {r}: gap={out['per_region'][r]['gap']:+.3f} p={out['per_region'][r]['p_value']:.3f} top1=f{out['per_region'][r]['top1']}")
    print(f"region-mean: gap={out['region_mean']['gap']:+.3f} p={out['region_mean']['p_value']:.3f} top1=f{out['region_mean']['top1']}")
    with open("/results/mbpp_llada/pooling_diagnose.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    return json.dumps(out)


@app.local_entrypoint()
def main():
    print(run.remote())
