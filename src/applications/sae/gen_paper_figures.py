"""Generate paper figures for Tracing paper from cached diagnose / steering JSONs.

Outputs PDF figures to emnlp_paper/figures/.

Figures produced:
  fig1_trajectory.pdf   -- LLaDA mbpp silhouette + null mean + gap across steps,
                           with SAE in-distribution range shaded and p<0.05 marker.
  fig2_cross.pdf        -- 3-panel cross-condition trajectory (LLaDA mbpp,
                           Dream mbpp, LLaDA jsonschema).
  fig3_steering.pdf     -- Steering negative result summary: 4 conditions
                           (window sweep + multi + reverse) bar chart.
  fig4_feature_drift.pdf -- Top-N feature Jaccard across steps + persistent
                           feature enrichment trajectory.

Usage:
  .venv/bin/python src/applications/sae/gen_paper_figures.py
"""

import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "emnlp_paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("/tmp/paper_figs")
CACHE_DIR.mkdir(exist_ok=True)


def fetch_json(remote: str, local_name: str) -> dict | None:
    """Pull JSON from modal volume via rtk proxy."""
    local = CACHE_DIR / local_name
    if not local.exists():
        r = subprocess.run(
            ["rtk", "proxy", "modal", "volume", "get",
             "probe-results", remote, str(local)],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not local.exists():
            return None
    try:
        return json.load(open(local))
    except Exception:
        return None


def load_trajectory(model: str, dataset: str, steps: list[int]) -> list[dict]:
    out = []
    for s in steps:
        d = fetch_json(
            f"{dataset}_{model}/sae_diagnose_stage2_s{s}.json",
            f"{model}_{dataset}_s{s}.json",
        )
        if d is None:
            continue
        out.append({
            "step": s,
            "silhouette": d.get("best_silhouette", 0.0),
            "null_mean": d.get("null_silhouette_mean") or 0.0,
            "null_95": d.get("null_silhouette_95pct") or 0.0,
            "p_value": d.get("permutation_p"),
            "best_k": d.get("best_k", 0),
            "top_features": d.get("top_fail_features", []),
        })
    return out


def fig1_trajectory():
    """LLaDA mbpp trajectory across all 7 steps."""
    steps = [0, 1, 4, 16, 32, 64, 127]
    traj = load_trajectory("llada", "mbpp", steps)
    xs = [t["step"] for t in traj]
    sil = [t["silhouette"] for t in traj]
    null = [t["null_mean"] for t in traj]
    gap = [s - n for s, n in zip(sil, null)]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    # In-distribution shading (dlm_t in [0.05, 0.5] ~ step 64-122)
    ax.axvspan(64, 122, color="gold", alpha=0.12, zorder=0,
               label="SAE training range")

    ax.plot(xs, sil, "o-", color="#1f5fa8", lw=1.5, ms=4,
            label="observed silhouette")
    ax.plot(xs, null, "s--", color="#999999", lw=1.0, ms=3,
            label="permutation null mean")
    ax.fill_between(xs, null, sil,
                    where=[s > n for s, n in zip(sil, null)],
                    color="#1f5fa8", alpha=0.15, label="gap (signal)")

    # Mark significant step
    sig_x = [t["step"] for t in traj if (t["p_value"] or 1) < 0.05]
    sig_y = [t["silhouette"] for t in traj if (t["p_value"] or 1) < 0.05]
    ax.scatter(sig_x, sig_y, marker="*", s=120, color="#d62728",
               zorder=10, label="$p<0.05$")

    ax.set_xlabel("denoising step")
    ax.set_ylabel("KMeans silhouette on top-20 fail features")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xticks([0, 1, 4, 16, 32, 64, 127])
    ax.set_xticklabels(["0", "1", "4", "16", "32", "64", "127"])
    ax.set_ylim(0.0, 0.85)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title("LLaDA mbpp: failure cluster peaks mid-denoising")

    fig.savefig(FIG_DIR / "fig1_trajectory.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig1_trajectory.pdf'}")


def fig2_cross():
    """3-panel cross-condition trajectory."""
    panels = [
        ("LLaDA mbpp (code)", "llada", "mbpp",
         [0, 1, 4, 16, 32, 64, 127]),
        ("Dream mbpp (code)", "dream", "mbpp", [32, 64, 127]),
        ("LLaDA jsonschema (structural)", "llada", "jsonschema",
         [4, 16, 32, 64, 127]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    for ax, (title, model, dataset, steps) in zip(axes, panels):
        traj = load_trajectory(model, dataset, steps)
        if not traj:
            ax.set_title(f"{title} (no data)")
            continue
        xs = [t["step"] for t in traj]
        sil = [t["silhouette"] for t in traj]
        null = [t["null_mean"] for t in traj]

        ax.axvspan(64, 122, color="gold", alpha=0.12, zorder=0)
        ax.plot(xs, sil, "o-", color="#1f5fa8", lw=1.4, ms=3.5,
                label="silhouette")
        ax.plot(xs, null, "s--", color="#999999", lw=1.0, ms=3,
                label="null mean")

        sig_x = [t["step"] for t in traj if (t["p_value"] or 1) < 0.05]
        sig_y = [t["silhouette"] for t in traj if (t["p_value"] or 1) < 0.05]
        ax.scatter(sig_x, sig_y, marker="*", s=80, color="#d62728", zorder=10)

        ax.set_xlabel("denoising step")
        ax.set_title(title, fontsize=9)
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xticks([0, 1, 4, 16, 32, 64, 127])
        ax.set_xticklabels(["0", "1", "4", "16", "32", "64", "127"], rotation=0)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_ylim(0.0, 0.85)
    axes[0].set_ylabel("silhouette")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=7)
    fig.savefig(FIG_DIR / "fig2_cross.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig2_cross.pdf'}")


def fig3_steering():
    """Steering negative result: window sweep + multi + reverse."""
    conds = [
        ("baseline\nbase rate", None, None, None, "baseline"),
        ("suppress\n@s64", "f15601_a5.0_s64", 5.0, 64, "f15601 suppress"),
        ("suppress\n@s16", "f15601_a5.0_s16", 5.0, 16, "f15601 suppress"),
        ("multi-feat\n@s64", "f15601_8825_2087_11404_9657_a5.0_s64", 5.0, 64,
         "top-5 suppress"),
        ("reverse\n@s64", "f15601_a-5.0_s64", -5.0, 64, "f15601 add"),
    ]
    rows_fp = []  # fail_c1 steer pass rate
    rows_pf = []  # pass regression rate (1 - pass steer rate)
    labels = []
    for label, suffix, alpha, sfrom, _desc in conds:
        labels.append(label)
        if suffix is None:
            rows_fp.append(0.0)  # all fail
            rows_pf.append(0.0)  # all pass kept
            continue
        d = fetch_json(
            f"mbpp_llada/sae_steer_stage4_{suffix}.json",
            f"steer_{suffix}.json",
        )
        if d is None:
            rows_fp.append(np.nan)
            rows_pf.append(np.nan)
            continue
        for r in d["summaries"]:
            if r is None:
                continue
            if "fail_c1" in r["label"]:
                rows_fp.append(r["steer_pass_rate"] * 100)
            elif "pass" in r["label"]:
                rows_pf.append((1 - r["steer_pass_rate"]) * 100)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    xpos = np.arange(len(labels))
    width = 0.38
    bar1 = ax.bar(xpos - width / 2, rows_fp, width, color="#2ca02c",
                  label="fail→pass rescue (%)")
    bar2 = ax.bar(xpos + width / 2, rows_pf, width, color="#d62728",
                  label="pass→fail regression (%)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("rate (%)")
    ax.set_ylim(0, 5)
    ax.set_title("No intervention condition flips fail/pass")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)

    # Annotate zeros
    for x, v in zip(xpos - width / 2, rows_fp):
        if v == 0:
            ax.text(x, 0.15, "0", ha="center", fontsize=7)
    for x, v in zip(xpos + width / 2, rows_pf):
        if v == 0:
            ax.text(x, 0.15, "0", ha="center", fontsize=7)

    fig.savefig(FIG_DIR / "fig3_steering.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig3_steering.pdf'}")


def fig4_feature_drift():
    """Top-N feature Jaccard heatmap + f15601 enrichment trajectory."""
    steps = [0, 1, 4, 16, 32, 64, 127]
    traj = load_trajectory("llada", "mbpp", steps)
    if not traj:
        return
    sets = []
    xs = []
    for t in traj:
        feats = {r["feature_id"] for r in t["top_features"][:20]}
        sets.append(feats)
        xs.append(t["step"])

    # Jaccard matrix
    N = len(xs)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            u = sets[i] | sets[j]
            J[i, j] = len(sets[i] & sets[j]) / len(u) if u else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.4))

    im = ax1.imshow(J, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(N))
    ax1.set_yticks(range(N))
    ax1.set_xticklabels(xs, fontsize=8)
    ax1.set_yticklabels(xs, fontsize=8)
    ax1.set_xlabel("denoising step")
    ax1.set_ylabel("denoising step")
    ax1.set_title("Top-20 fail-feature Jaccard")
    for i in range(N):
        for j in range(N):
            v = J[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v < 0.4 else "black", fontsize=6.5)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard", fontsize=8)

    # f15601 enrichment across steps (plus a couple of comparator features)
    track_ids = [15601, 5561, 11265]
    track_labels = {15601: "f15601 (peak feat)", 5561: "f5561 (early-only)",
                    11265: "f11265 (late-only)"}
    colors = ["#d62728", "#1f5fa8", "#2ca02c"]
    for fid, color in zip(track_ids, colors):
        ys = []
        for t in traj:
            row = next(
                (r for r in t["top_features"] if r["feature_id"] == fid),
                None,
            )
            ys.append(row["enrichment"] if row else None)
        # Plot, skipping None
        valid_x = [x for x, y in zip(xs, ys) if y is not None]
        valid_y = [y for y in ys if y is not None]
        ax2.plot(valid_x, valid_y, "o-", color=color, lw=1.4, ms=4,
                 label=track_labels[fid])
    ax2.axhline(0, color="black", lw=0.5, ls=":")
    ax2.axvspan(64, 122, color="gold", alpha=0.12, zorder=0)
    ax2.set_xlabel("denoising step")
    ax2.set_ylabel("enrichment $P(\\mathrm{fire}|\\mathrm{fail}) - P(\\mathrm{fire}|\\mathrm{pass})$")
    ax2.set_xscale("symlog", linthresh=2)
    ax2.set_xticks([0, 1, 4, 16, 32, 64, 127])
    ax2.set_xticklabels(["0", "1", "4", "16", "32", "64", "127"])
    ax2.set_title("Persistent peak feature emerges at s32")
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=7)
    ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_feature_drift.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig4_feature_drift.pdf'}")


if __name__ == "__main__":
    fig1_trajectory()
    fig2_cross()
    fig3_steering()
    fig4_feature_drift()
    print("\nAll figures saved to:", FIG_DIR)
