"""Local analysis of all-layer AR-DLM diff vectors.

Reads /tmp/ardlm_alllayers.npz (H_Q, H_D, labels, all 28 layers) and runs:
  1. Per-layer CV AUC for h_Q, h_D, d, normalized variants
  2. Concat-all-layers AUC
  3. Layer-range subsets (early 0-9, mid 10-19, late 20-27)
  4. Best single layer per model
  5. Pairwise layer combinations

Usage:
  .venv/bin/python scripts/analyze_ardlm_layers.py
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def cv_auc(X, y, C=0.01, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=C).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    return float(np.mean(aucs)), float(np.std(aucs))


def main(path="/tmp/ardlm_alllayers.npz"):
    z = np.load(path)
    H_Q, H_D, labels = z["H_Q"], z["H_D"], z["labels"]
    n, n_l, d_h = H_Q.shape
    print(f"n={n}, layers={n_l}, hidden_dim={d_h}, fail={(labels==0).sum()}, pass={(labels==1).sum()}")

    print("\n=== Per-layer CV AUC ===")
    print(f"{'layer':>5s}  {'h_Q':>14s}  {'h_D':>14s}  {'d':>14s}  {'gap':>6s}")
    results = []
    for li in range(n_l):
        hq, hd = H_Q[:, li, :], H_D[:, li, :]
        d = hd - hq
        m_q, s_q = cv_auc(hq, labels)
        m_d, s_d = cv_auc(hd, labels)
        m_dd, s_dd = cv_auc(d, labels)
        gap = m_d - m_q
        results.append({"layer": li, "auc_Q": m_q, "auc_D": m_d, "auc_d": m_dd, "gap": gap})
        print(f"  {li:3d}  {m_q:.3f}±{s_q:.3f}  {m_d:.3f}±{s_d:.3f}  {m_dd:.3f}±{s_dd:.3f}  {gap:+.3f}")

    best_Q = max(results, key=lambda r: r["auc_Q"])
    best_D = max(results, key=lambda r: r["auc_D"])
    best_d = max(results, key=lambda r: r["auc_d"])
    print(f"\nBest layer for h_Q: L{best_Q['layer']} AUC={best_Q['auc_Q']:.3f}")
    print(f"Best layer for h_D: L{best_D['layer']} AUC={best_D['auc_D']:.3f}")
    print(f"Best layer for d:   L{best_d['layer']} AUC={best_d['auc_d']:.3f}")
    print(f"Max (Dream - Qwen) gap: L{max(results, key=lambda r: r['gap'])['layer']} gap={max(r['gap'] for r in results):+.3f}")

    print("\n=== Layer-range subsets ===")
    ranges = {"early (0-9)": list(range(0, 10)),
              "mid (10-19)": list(range(10, 20)),
              "late (20-27)": list(range(20, 28)),
              "all (0-27)": list(range(n_l)),
              "every 4th": list(range(0, n_l, 4))}
    for name, layers in ranges.items():
        hq = H_Q[:, layers, :].reshape(n, -1)
        hd = H_D[:, layers, :].reshape(n, -1)
        d = hd - hq
        m_q, _ = cv_auc(hq, labels)
        m_d, _ = cv_auc(hd, labels)
        m_dd, _ = cv_auc(d, labels)
        print(f"  {name:18s} ({len(layers)} layers, dim={len(layers)*d_h}): h_Q={m_q:.3f}  h_D={m_d:.3f}  d={m_dd:.3f}")

    print("\n=== Concatenated full-stack ===")
    H_Q_flat = H_Q.reshape(n, -1)
    H_D_flat = H_D.reshape(n, -1)
    D_flat = H_D_flat - H_Q_flat
    print(f"dim={H_Q_flat.shape[1]}")
    m_q, _ = cv_auc(H_Q_flat, labels)
    m_d, _ = cv_auc(H_D_flat, labels)
    m_dd, _ = cv_auc(D_flat, labels)
    print(f"  concat-all h_Q: {m_q:.3f}")
    print(f"  concat-all h_D: {m_d:.3f}")
    print(f"  concat-all d:   {m_dd:.3f}")

    out = {
        "n": int(n), "n_layers": int(n_l),
        "per_layer": results,
        "best_Q": best_Q, "best_D": best_D, "best_d": best_d,
    }
    with open("/tmp/ardlm_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved /tmp/ardlm_analysis.json")


if __name__ == "__main__":
    main()
