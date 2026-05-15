"""Cross-generator difficulty test: do Dream L26 features predict Qwen-Instruct's correctness?

If Dream L26 features encode problem difficulty (not Dream-specific generation
correctness), they should predict Qwen-Instruct's pass/fail on the same problems
at AUC similar to predicting Dream's own pass/fail.

Inputs (must exist locally):
  /tmp/ardlm_alllayers.npz                  -- H_Q, H_D (Dream-Base), labels (Dream-Instruct)
  /tmp/qwen_instruct_alllayers.npz          -- H_QI (Qwen-Instruct hidden), labels (Dream)
  /tmp/qwen_instruct_mbpp_labels.json       -- Qwen-Instruct's own pass/fail labels

Outputs:
  Per-layer AUC comparison: Dream labels vs Qwen-Instruct labels using same Dream features.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def cv_auc(X, y, C=0.01, n_splits=5):
    if (y == 0).sum() < n_splits or (y == 1).sum() < n_splits:
        return float("nan"), float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=C).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    z = np.load('/tmp/ardlm_alllayers.npz')
    H_Q, H_D, dream_labels = z['H_Q'], z['H_D'], z['labels']

    z2 = np.load('/tmp/qwen_instruct_alllayers.npz')
    H_QI = z2['H_QI']

    qwen_data = json.load(open('/tmp/qwen_instruct_mbpp_labels.json'))
    qwen_rows = qwen_data['rows']
    print(f"Qwen-Instruct pass rate: {qwen_data['n_pass']}/{qwen_data['n']} = {qwen_data['n_pass']/qwen_data['n']:.2%}")

    # Need to align by sample index. Dream labels are ordered: fail_idxs + pass_idxs (idx into MBPP-sanitized test sorted by task_id)
    # Qwen labels are ordered: idx 0..256 in same sorted order
    # We need to recover the Dream-labels-to-MBPP-idx mapping
    diag_path = "/tmp/sae_diagnose_stage2.json"  # only if we have it
    import os
    if not os.path.exists(diag_path):
        # try modal volume
        import subprocess
        subprocess.run(["/Users/wchiu/Documents/GitHub/dllm-probing/.venv/bin/modal",
                        "volume", "get", "probe-results",
                        "/mbpp_dream/sae_diagnose_stage2.json", diag_path, "--force"],
                       capture_output=True)
    diag = json.load(open(diag_path))
    fail_idxs = []
    for c in diag.get("clusters", []):
        fail_idxs.extend(c.get("fail_sample_indices", []))
    fail_idxs = list(dict.fromkeys(fail_idxs))
    n_total = len(qwen_rows)  # 257
    fail_set = set(fail_idxs)
    pass_idxs = [i for i in range(n_total) if i not in fail_set]
    dream_order = fail_idxs + pass_idxs  # order H_Q/H_D rows correspond to

    # Map Qwen's MBPP-index labels to the dream_order
    qwen_label_by_mbpp_idx = {r['idx']: r['passed'] for r in qwen_rows}
    qwen_labels = np.array([qwen_label_by_mbpp_idx[i] for i in dream_order])
    print(f"Aligned labels: dream_pass={(dream_labels==1).sum()}, qwen_pass={(qwen_labels==1).sum()}")
    # agreement
    agree = (dream_labels == qwen_labels).sum() / len(dream_labels)
    print(f"Per-sample agreement dream/qwen pass/fail: {agree:.3f}")

    print(f"\n{'L':>3s}  {'AUC|Dream':>13s}  {'AUC|Qwen-I':>14s}  {'diff':>8s}")
    for li in range(H_D.shape[1]):
        feats = H_D[:, li, :]
        m_d, _ = cv_auc(feats, dream_labels)
        m_q, _ = cv_auc(feats, qwen_labels)
        d = m_q - m_d
        print(f" {li:2d}  {m_d:.3f}        {m_q:.3f}         {d:+.3f}")

    print(f"\n=== Same comparison with Qwen-Instruct hidden states ===")
    print(f"{'L':>3s}  {'AUC|Dream':>13s}  {'AUC|Qwen-I':>14s}  {'diff':>8s}")
    for li in range(H_QI.shape[1]):
        feats = H_QI[:, li, :]
        m_d, _ = cv_auc(feats, dream_labels)
        m_q, _ = cv_auc(feats, qwen_labels)
        d = m_q - m_d
        print(f" {li:2d}  {m_d:.3f}        {m_q:.3f}         {d:+.3f}")


if __name__ == "__main__":
    main()
