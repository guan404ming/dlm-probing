"""Stage 0 sanity check: SAE-feature probe vs raw-hidden-state probe.

Loads cached LLaDA-8B hidden states from the probe-results volume,
encodes them through DLM-Scope's Top-K SAE at layer 26 (in-distribution
for SAE training: dlm_t in [0.05, 0.5] ~ steps 64-122), trains a
logistic probe on SAE features, and compares AUC to a raw-hidden-state
probe at the same (layer, step) cell.

Decision gate:
  SAE AUC >= raw AUC - 0.02  ->  story holds, proceed to diagnose/steer
  SAE AUC clearly worse      ->  need code-specialized SAE (out of scope)

Usage:
  .venv/bin/modal run src/applications/sae/modal_sae_probe.py \\
      --dataset jsonschema --model llada
  .venv/bin/modal run src/applications/sae/modal_sae_probe.py \\
      --dataset mbpp --model llada
"""

import modal

app = modal.App("sae-probe-stage0")

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
SAE_TRAINER = 2  # k=160, d_sae=16384, d_in=4096
SAE_PATH_IN_REPO = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"

# SAE is trained on dlm_t in [0.05, 0.5] ~ steps 64-122 of 128-step schedule.
# Step 0/1/4/16/32 are OOD and excluded from this sanity check.
TEST_STEPS = [64, 127]
N_REGIONS = 4


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run_sae_probe(dataset_key: str, model_key: str, n_chunks: int, total: int):
    import json
    import os

    import numpy as np
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    RESULTS_VOL.reload()
    os.environ["HF_HOME"] = "/hf-cache"

    # ---- Load SAE checkpoint ----
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
    print(f"SAE state_dict keys: {list(state.keys())}")
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    sae.load_from_state_dict(state)

    # ---- Load cached LLaDA hidden states ----
    chunk_size = (total + n_chunks - 1) // n_chunks
    in_dir = f"/results/{dataset_key}_{model_key}"
    all_labels = []
    all_feats = {(s, r): [] for s in TEST_STEPS for r in range(N_REGIONS)}

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_labels.append(data["labels"])
        for s in TEST_STEPS:
            for r in range(N_REGIONS):
                all_feats[(s, r)].append(data[f"feat_s{s}_r{r}"])

    labels = np.concatenate(all_labels)
    features = {
        (s, r): np.concatenate(all_feats[(s, r)])
        for s in TEST_STEPS
        for r in range(N_REGIONS)
    }
    n_samples = len(labels)
    n_func = int(labels.sum())
    n_layers = features[(TEST_STEPS[0], 0)].shape[1]
    print(
        f"Loaded: {n_samples} samples, {n_func} functional "
        f"({100 * n_func / n_samples:.1f}%), {n_layers} layers"
    )

    # ---- Encode L26 hidden states through SAE ----
    # Two encodings per cell to ablate the method:
    #   topk:  sparse (TopK + ReLU), tests pure SAE feature dictionary
    #   dense: pre-TopK encoder output, tests whether TopK gating drops signal
    sae_topk = {}
    sae_dense = {}
    for s in TEST_STEPS:
        for r in range(N_REGIONS):
            x_raw = features[(s, r)][:, SAE_LAYER, :].astype(np.float32)
            with torch.no_grad():
                x_t = torch.from_numpy(x_raw).cuda()
                z_topk, z_dense = sae.encode_both(x_t)
                sae_topk[(s, r)] = z_topk.cpu().numpy()
                sae_dense[(s, r)] = z_dense.cpu().numpy()

    # ---- Train probes: raw L26 vs SAE L26, per (step, region) ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    def cv_auc(X, y, use_pca=True):
        aucs = []
        for tr, te in skf.split(X, y):
            steps_ = [StandardScaler()]
            if use_pca:
                steps_.append(PCA(n_components=min(64, X.shape[1], X.shape[0] - 1)))
            steps_.append(LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
            clf = make_pipeline(*steps_)
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            try:
                aucs.append(roc_auc_score(y[te], prob))
            except ValueError:
                aucs.append(0.5)
        return float(np.mean(aucs)), float(np.std(aucs))

    print(f"\n{'='*92}")
    print(f"Stage 0 SAE sanity check: {dataset_key} / {model_key} / layer {SAE_LAYER}")
    print(f"  raw    = hidden state at L26 (PCA64 + LR)")
    print(f"  topk_p = TopK sparse SAE features (PCA64 + LR)")
    print(f"  topk_n = TopK sparse SAE features (no PCA, LR)")
    print(f"  dense  = pre-TopK encoder output (PCA64 + LR)")
    print(f"{'='*92}")
    print(
        f"{'step':>4} {'region':>6} {'raw':>7} {'topk_p':>7} {'topk_n':>7} "
        f"{'dense':>7} {'best_sae':>8} {'diff':>7} {'verdict':>8}"
    )
    print("-" * 92)

    def emit_row(step, region_label, X_raw, X_topk, X_dense):
        auc_raw, _ = cv_auc(X_raw, labels, use_pca=True)
        auc_topk_p, _ = cv_auc(X_topk, labels, use_pca=True)
        auc_topk_n, _ = cv_auc(X_topk, labels, use_pca=False)
        auc_dense, _ = cv_auc(X_dense, labels, use_pca=True)
        best_sae = max(auc_topk_p, auc_topk_n, auc_dense)
        diff = best_sae - auc_raw
        verdict = "OK" if diff >= -0.02 else "WORSE"
        rows.append(
            {
                "step": step,
                "region": region_label,
                "raw_auc": round(auc_raw, 4),
                "topk_pca_auc": round(auc_topk_p, 4),
                "topk_nopca_auc": round(auc_topk_n, 4),
                "dense_auc": round(auc_dense, 4),
                "best_sae_auc": round(best_sae, 4),
                "diff_best_sae_vs_raw": round(diff, 4),
                "verdict": verdict,
            }
        )
        print(
            f"{step:>4} {str(region_label):>6} {auc_raw:>7.3f} {auc_topk_p:>7.3f} "
            f"{auc_topk_n:>7.3f} {auc_dense:>7.3f} {best_sae:>8.3f} "
            f"{diff:>+7.3f} {verdict:>8}"
        )

    for s in TEST_STEPS:
        for r in range(N_REGIONS):
            emit_row(
                s, r,
                features[(s, r)][:, SAE_LAYER, :],
                sae_topk[(s, r)],
                sae_dense[(s, r)],
            )
        X_raw_mean = np.mean(
            [features[(s, r)][:, SAE_LAYER, :] for r in range(N_REGIONS)], axis=0
        )
        X_topk_mean = np.mean([sae_topk[(s, r)] for r in range(N_REGIONS)], axis=0)
        X_dense_mean = np.mean([sae_dense[(s, r)] for r in range(N_REGIONS)], axis=0)
        emit_row(s, "mean", X_raw_mean, X_topk_mean, X_dense_mean)

    # ---- Persist ----
    results = {
        "dataset": dataset_key,
        "model": model_key,
        "sae_repo": SAE_REPO,
        "sae_path": SAE_PATH_IN_REPO,
        "sae_layer": SAE_LAYER,
        "sae_k": k,
        "sae_d_sae": d_sae,
        "test_steps": TEST_STEPS,
        "n_samples": n_samples,
        "n_functional": n_func,
        "cells": rows,
    }
    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/sae_probe_stage0.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved to {out_path}")
    return json.dumps(results, indent=2)


class TopKSAE:
    """Minimal loader for dictionary_learning AutoEncoderTopK.

    State dict keys used by dictionary_learning:
      encoder.weight (d_sae, d_in), encoder.bias (d_sae,)
      decoder.weight (d_in, d_sae)
      b_dec (d_in,)  pre-encoder bias subtracted from input
    Falls back gracefully if keys differ.
    """

    def __init__(self, d_in, d_sae, k):
        import torch
        import torch.nn as nn

        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.W_enc = None
        self.b_enc = None
        self.W_dec = None
        self.b_dec = None
        self._torch = torch
        self._nn = nn

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

        # Some libs store encoder as (d_sae, d_in) -> input @ W_enc.T
        # Others store it transposed. Detect by shape.
        if self.W_enc.shape == (self.d_sae, self.d_in):
            self._enc_op = "in_at_wt"
        elif self.W_enc.shape == (self.d_in, self.d_sae):
            self._enc_op = "in_at_w"
        else:
            raise ValueError(f"Unexpected W_enc shape {self.W_enc.shape}")
        print(f"  W_enc {tuple(self.W_enc.shape)} op={self._enc_op}, "
              f"b_dec norm={self.b_dec.norm().item():.3f}")

    def eval(self):
        return self

    def encode_both(self, x):
        """Return (sparse_topk, pre_topk_dense)."""
        torch = self._torch
        x = x - self.b_dec
        if self._enc_op == "in_at_wt":
            pre = x @ self.W_enc.T + self.b_enc
        else:
            pre = x @ self.W_enc + self.b_enc
        # Top-K with ReLU on selected entries
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        topk_vals = topk_vals.relu()
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse, pre


@app.local_entrypoint()
def main(
    dataset: str = "jsonschema",
    model: str = "llada",
    chunks: int = 8,
    total: int = 0,
):
    totals = {"jsonschema": 272, "gsm8k": 1319, "mbpp": 257, "arc": 1172}
    if total <= 0:
        total = totals[dataset]
    print(f"SAE Stage 0: dataset={dataset}, model={model}, total={total}")
    result = run_sae_probe.remote(dataset, model, chunks, total)
    print("\n" + result)
