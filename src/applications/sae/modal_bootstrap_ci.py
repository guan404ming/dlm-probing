"""Bootstrap confidence intervals for the Figure 2 signal-to-null gap (reviewer #6).

Dream uses greedy decoding, so re-seeding gives identical generations and no seed
variance; instead we bootstrap the silhouette gap over samples, which works
uniformly for both DLMs and reuses the cached per-cell residuals. For each
(model, task) cell and sparse checkpoint we resample the fail set B times,
recompute the top-20 silhouette, and report the gap (obs silhouette minus
permutation-null mean) with a 95% bootstrap interval.

Output: /results/fig2_bootstrap_ci.json

Usage:
  .venv/bin/modal run src/applications/sae/modal_bootstrap_ci.py
"""

import modal

app = modal.App("sae-bootstrap-ci")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git", "curl")
         .pip_install("torch>=2.0", "numpy", "scikit-learn>=1.3", "huggingface_hub"))
RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

SAE = {
    "llada": {"repo": "AwesomeInterpretability/llada-mask-topk-sae",
              "path": "resid_post_layer_26/trainer_2", "layer": 26},
    "dream": {"repo": "AwesomeInterpretability/dlm-mask-topk-sae",
              "path": "saes_mask_Dream-org_Dream-v0-Base-7B_top_k/resid_post_layer_23/trainer_2", "layer": 23},
}
TOTALS = {"mbpp": 257, "jsonschema": 272, "gsm8k": 1319, "arc": 1172}
STEPS = [4, 16, 32, 64, 127]
TOP_N = 20
N_PERM = 500
N_BOOT = 500
N_REGIONS = 4


@app.function(image=image, gpu="A100", timeout=14400,
              volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL})
def run(n_chunks: int = 8):
    import json, os, math, numpy as np, torch
    from huggingface_hub import hf_hub_download
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    os.environ["HF_HOME"] = "/hf-cache"
    RESULTS_VOL.reload()

    sae_cache = {}
    def get_sae(model):
        if model in sae_cache:
            return sae_cache[model]
        s = SAE[model]
        ae = hf_hub_download(repo_id=s["repo"], filename=f"{s['path']}/ae.pt", cache_dir="/hf-cache")
        cfg = hf_hub_download(repo_id=s["repo"], filename=f"{s['path']}/config.json", cache_dir="/hf-cache")
        k = json.load(open(cfg))["trainer"]["k"]
        st = torch.load(ae, map_location="cpu", weights_only=True)
        We = st["encoder.weight"].cuda().float(); be = st["encoder.bias"].cuda().float()
        bd = st.get("b_dec", st.get("decoder.bias")).cuda().float()
        sae_cache[model] = (We, be, bd, k, s["layer"]); return sae_cache[model]

    def encode(model, x):
        We, be, bd, k, _ = get_sae(model)
        xt = torch.from_numpy(x).cuda() - bd
        pre = (xt @ We.T + be) if We.shape[1] == x.shape[1] else (xt @ We + be)
        tv, ti = pre.topk(k, -1); tv = tv.relu()
        sp = torch.zeros_like(pre); sp.scatter_(-1, ti, tv); return sp.cpu().numpy()

    def gap_ci(sae_mean, labels):
        active = (sae_mean > 0).astype(np.float32)
        fi = np.where(labels == 0)[0]
        if len(fi) < 8:
            return None
        enr = active[labels == 0].mean(0) - active[labels == 1].mean(0)
        top = np.argsort(enr)[::-1][:TOP_N]
        sig = sae_mean[fi][:, top]
        best = None
        for K in [2, 3, 4, 5]:
            try:
                sl = float(silhouette_score(sig, KMeans(K, random_state=42, n_init=10).fit_predict(sig)))
            except ValueError:
                sl = -1.0
            if best is None or sl > best[1]:
                best = (K, sl)
        K, obs = best
        rng = np.random.RandomState(42)
        null = []
        for _ in range(N_PERM):
            perm = rng.permutation(labels); fip = np.where(perm == 0)[0]
            if len(fip) < K + 1:
                null.append(-1.0); continue
            tp = np.argsort(active[perm == 0].mean(0) - active[perm == 1].mean(0))[::-1][:TOP_N]
            try:
                null.append(float(silhouette_score(sae_mean[fip][:, tp], KMeans(K, random_state=42, n_init=5).fit_predict(sae_mean[fip][:, tp]))))
            except ValueError:
                null.append(-1.0)
        nmean = float(np.mean(null)); pval = float((np.array(null) >= obs).mean())
        # bootstrap the observed silhouette over fail samples
        boot = []
        for _ in range(N_BOOT):
            idx = rng.randint(0, len(fi), len(fi))
            bs = sig[idx]
            try:
                boot.append(float(silhouette_score(bs, KMeans(K, random_state=0, n_init=3).fit_predict(bs))) - nmean)
            except ValueError:
                pass
        boot = np.array(boot)
        return {"gap": round(obs - nmean, 4), "K": K, "p_value": round(pval, 4),
                "ci_lo": round(float(np.percentile(boot, 2.5)), 4),
                "ci_hi": round(float(np.percentile(boot, 97.5)), 4)}

    out = {}
    for model in ["llada", "dream"]:
        for task, total in TOTALS.items():
            cell = f"{task}_{model}"
            cs = (total + n_chunks - 1) // n_chunks
            labels = []; rf = {s: {r: [] for r in range(N_REGIONS)} for s in STEPS}
            ok = True
            for i in range(n_chunks):
                p = f"/results/{cell}/chunk_off{i*cs}.npz"
                if not os.path.exists(p):
                    continue
                d = np.load(p); labels.append(d["labels"])
                for s in STEPS:
                    for r in range(N_REGIONS):
                        key = f"feat_s{s}_r{r}"
                        if key in d:
                            rf[s][r].append(d[key])
            if not labels:
                print(f"{cell}: no chunks, skip"); continue
            labels = np.concatenate(labels).astype(int)
            lay = SAE[model]["layer"]
            per = {}
            for s in STEPS:
                if not rf[s][0]:
                    continue
                acts = [encode(model, np.concatenate(rf[s][r])[:, lay, :].astype(np.float32)) for r in range(N_REGIONS)]
                res = gap_ci(np.mean(acts, 0), labels)
                if res:
                    per[s] = res
            out[cell] = {"n_pass": int(labels.sum()), "n_fail": int((labels == 0).sum()), "per_step": per}
            print(f"{cell}: " + " ".join(f"s{s}:{per[s]['gap']:+.3f}[{per[s]['ci_lo']:+.2f},{per[s]['ci_hi']:+.2f}]" for s in per))
    with open("/results/fig2_bootstrap_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    return json.dumps({k: {s: out[k]["per_step"][s]["gap"] for s in out[k]["per_step"]} for k in out})


@app.local_entrypoint()
def main():
    print(run.remote())
