"""Single-feature vs top-N SAE feature vs raw probe AUC comparison.

For each (model, dataset) cell, report:
  - AUC[single]: top-1 fail-enriched SAE feature (1 scalar) -> LR -> AUC
  - AUC[top3], AUC[top5], AUC[top20]: top-N SAE features -> LR
  - AUC[raw]: full hidden state at the SAE layer -> PCA(64) + LR
                (matches the SRW probe pipeline)

All AUCs use 5-fold stratified CV with the same fold splits per cell.
Top-N features are selected on each training fold's enrichment to avoid
data leakage (per-fold feature selection).

Usage:
  .venv/bin/modal run src/applications/sae/modal_sae_auc_compare.py \\
      --model llada --dataset mbpp
  .venv/bin/modal run src/applications/sae/modal_sae_auc_compare.py \\
      --model qwen --dataset jsonschema --chunks 4
"""

import modal

app = modal.App("sae-auc-compare")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("numpy", "scikit-learn", "torch", "huggingface_hub")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Model-specific SAE config
MODEL_SAE = {
    "llada": {
        "repo": "AwesomeInterpretability/llada-mask-topk-sae",
        "path_template": "resid_post_layer_{layer}/trainer_2",
        "layer": 26,
        "d_in": 4096,
    },
    "dream": {
        "repo": "AwesomeInterpretability/dlm-mask-topk-sae",
        "path_template": (
            "saes_mask_Dream-org_Dream-v0-Base-7B_top_k/"
            "resid_post_layer_{layer}/trainer_2"
        ),
        "layer": 23,
        "d_in": 3584,
    },
    "qwen": {
        "repo": "AwesomeInterpretability/qwen-topk-sae",
        "path_template": (
            "saes__Qwen_Qwen2.5-7B_top_k/"
            "resid_post_layer_{layer}/trainer_2"
        ),
        "layer": 23,
        "d_in": 3584,
    },
}

TARGET_STEP = 64  # for LLaDA/Dream (DLM); not used for Qwen
N_REGIONS = 4  # for LLaDA/Dream
TOP_N_LIST = [1, 3, 5, 10, 20]


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_compare(model: str, dataset: str, n_chunks: int, total: int):
    import json
    import os

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    cfg = MODEL_SAE[model]
    sae_path = cfg["path_template"].format(layer=cfg["layer"])

    # ---- Load SAE ----
    ae_local = hf_hub_download(
        repo_id=cfg["repo"], filename=f"{sae_path}/ae.pt",
        cache_dir="/hf-cache",
    )
    sae_cfg_local = hf_hub_download(
        repo_id=cfg["repo"], filename=f"{sae_path}/config.json",
        cache_dir="/hf-cache",
    )
    with open(sae_cfg_local) as f:
        sae_cfg = json.load(f)
    d_in = sae_cfg["trainer"]["activation_dim"]
    d_sae = sae_cfg["trainer"]["dict_size"]
    k = sae_cfg["trainer"]["k"]
    print(f"{model} SAE: d_in={d_in}, d_sae={d_sae}, k={k}, layer={cfg['layer']}")

    state = torch.load(ae_local, map_location="cpu", weights_only=True)
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    sae.load_from_state_dict(state)

    # ---- Load cached hidden states + labels (model-specific) ----
    chunk_size = (total + n_chunks - 1) // n_chunks
    if model == "qwen":
        # AR: cached at /results/{dataset}_qwen, last prompt token, no regions
        in_dir = f"/results/{dataset}_qwen"
        all_labels = []
        all_feats = []
        for i in range(n_chunks):
            offset = i * chunk_size
            path = f"{in_dir}/chunk_off{offset}.npz"
            if not os.path.exists(path):
                print(f"WARNING: missing {path}")
                continue
            d = np.load(path)
            all_labels.append(d["ar_labels"])
            all_feats.append(d["feats"])
        labels = np.concatenate(all_labels).astype(int)
        feats = np.concatenate(all_feats)
        x_raw = feats[:, cfg["layer"], :].astype(np.float32)  # (N, d_in)
    else:
        # DLM: cached at /results/{dataset}_{model}, region-mean at TARGET_STEP
        in_dir = f"/results/{dataset}_{model}"
        all_labels = []
        region_feats = {r: [] for r in range(N_REGIONS)}
        for i in range(n_chunks):
            offset = i * chunk_size
            path = f"{in_dir}/chunk_off{offset}.npz"
            if not os.path.exists(path):
                print(f"WARNING: missing {path}")
                continue
            d = np.load(path)
            all_labels.append(d["labels"])
            for r in range(N_REGIONS):
                region_feats[r].append(d[f"feat_s{TARGET_STEP}_r{r}"])
        labels = np.concatenate(all_labels).astype(int)
        feats_regions = [np.concatenate(region_feats[r]) for r in range(N_REGIONS)]
        # Pool over regions, take SAE layer
        x_layer_regions = np.stack(
            [feats_regions[r][:, cfg["layer"], :] for r in range(N_REGIONS)],
            axis=0,
        )  # (R, N, d_in)
        x_raw = x_layer_regions.mean(axis=0).astype(np.float32)  # (N, d_in)

    n_samples = len(labels)
    n_pass = int(labels.sum())
    n_fail = n_samples - n_pass
    print(
        f"Loaded {n_samples} {dataset}/{model}: "
        f"{n_pass} pass, {n_fail} fail; "
        f"x_raw shape={x_raw.shape}"
    )

    # ---- Encode through SAE (TopK sparse) ----
    with torch.no_grad():
        x_t = torch.from_numpy(x_raw).cuda()
        z_topk, _ = sae.encode_both(x_t)
    sae_acts = z_topk.cpu().numpy()  # (N, d_sae)
    print(f"SAE acts shape={sae_acts.shape}, "
          f"nonzero per sample={(sae_acts > 0).sum(axis=1).mean():.1f}")

    # ---- 5-fold CV: per fold, pick top-N features on train, eval on val ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {n: [] for n in TOP_N_LIST}
    results["raw"] = []

    for fold_idx, (tr, te) in enumerate(skf.split(sae_acts, labels)):
        # Top-N feature selection (per-fold to avoid leakage)
        act_tr = (sae_acts[tr] > 0).astype(np.float32)
        p_fail = act_tr[labels[tr] == 0].mean(axis=0) if (labels[tr] == 0).sum() > 0 else np.zeros(d_sae)
        p_pass = act_tr[labels[tr] == 1].mean(axis=0) if (labels[tr] == 1).sum() > 0 else np.zeros(d_sae)
        enrich = p_fail - p_pass
        top_idx_all = np.argsort(enrich)[::-1]  # most fail-leaning first

        # AUC per top-N
        for n_feat in TOP_N_LIST:
            ftop = top_idx_all[:n_feat]
            X_tr = sae_acts[tr][:, ftop]
            X_te = sae_acts[te][:, ftop]
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
            )
            clf.fit(X_tr, labels[tr])
            prob = clf.predict_proba(X_te)[:, 1]
            try:
                auc = roc_auc_score(labels[te], prob)
            except ValueError:
                auc = 0.5
            # invert if needed: enrich is fail-leaning so high SAE value -> fail
            # But LR will learn the direction. AUC interpreted as P(pass).
            # We want AUC for "predicts label=1 (pass)" -- standard.
            results[n_feat].append(float(auc))

        # Raw probe baseline: PCA(64) + LR on full hidden state at SAE layer
        clf_raw = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(64, x_raw.shape[1], len(tr) - 1)),
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        )
        clf_raw.fit(x_raw[tr], labels[tr])
        prob = clf_raw.predict_proba(x_raw[te])[:, 1]
        try:
            auc = roc_auc_score(labels[te], prob)
        except ValueError:
            auc = 0.5
        results["raw"].append(float(auc))

    # ---- Aggregate ----
    print(f"\n{'='*64}")
    print(f"{model} / {dataset} / L{cfg['layer']} AUC comparison (5-fold mean ± std)")
    print(f"{'='*64}")
    summary = {}
    for k_ in TOP_N_LIST + ["raw"]:
        aucs = np.array(results[k_])
        mean = float(aucs.mean())
        std = float(aucs.std())
        summary[k_] = {"mean": round(mean, 4), "std": round(std, 4)}
        label = f"top-{k_}" if k_ != "raw" else "raw (PCA64)"
        print(f"  {label:>15}: AUC = {mean:.4f} ± {std:.4f}")

    out = {
        "model": model,
        "dataset": dataset,
        "sae_layer": cfg["layer"],
        "sae_k": k,
        "n_samples": n_samples,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "summary": summary,
        "per_fold": {str(k_): results[k_] for k_ in (TOP_N_LIST + ["raw"])},
    }
    out_dir = f"/results/{dataset}_{model}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/sae_auc_compare.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved to {out_path}")
    return json.dumps(summary, indent=2)


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
        assert W_enc is not None and W_dec is not None
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
    model: str = "llada",
    dataset: str = "mbpp",
    chunks: int = 8,
    total: int = 0,
):
    totals = {"jsonschema": 272, "gsm8k": 1319, "mbpp": 257, "arc": 1172}
    if total <= 0:
        total = totals[dataset]
    print(f"AUC compare: model={model}, dataset={dataset}, total={total}")
    summary = run_compare.remote(model, dataset, chunks, total)
    print("\n" + summary)
