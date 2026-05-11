"""Generate paper figures for Tracing paper from cached diagnose / steering JSONs.

Outputs PDF figures to emnlp_paper/figures/.

Figures produced:
  fig1_trajectory.pdf   -- LLaDA mbpp silhouette + null mean + gap across steps.
  fig2_cross.pdf        -- 2x4 signal-to-null gap trajectory grid.
  fig3_steering.pdf     -- Steering negative result summary as a compact table.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PAPER_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PAPER_ROOT / "figures"
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


def checkpoint_positions(traj: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Use categorical checkpoint positions so early steps remain readable."""
    return np.arange(len(traj)), [str(t["step"]) for t in traj]


def fig1_trajectory():
    """LLaDA mbpp trajectory across all 7 steps."""
    steps = [0, 1, 4, 16, 32, 64, 127]
    traj = load_trajectory("llada", "mbpp", steps)
    xs, xlabels = checkpoint_positions(traj)
    sil = [t["silhouette"] for t in traj]
    null = [t["null_mean"] for t in traj]

    fig, ax = plt.subplots(figsize=(3.35, 2.15), constrained_layout=True)

    ax.plot(xs, sil, "o-", color="#1f5fa8", lw=1.5, ms=4,
            label="observed")
    ax.plot(xs, null, "s--", color="#8c8c8c", lw=1.1, ms=3,
            label="null mean")
    ax.fill_between(xs, null, sil,
                    where=[s > n for s, n in zip(sil, null)],
                    color="#1f5fa8", alpha=0.14, label="signal gap")

    # Mark significant step without star glyphs.
    sig_x = [i for i, t in enumerate(traj) if (t["p_value"] or 1) < 0.05]
    sig_y = [t["silhouette"] for t in traj if (t["p_value"] or 1) < 0.05]
    if sig_x:
        ax.scatter(sig_x, sig_y, marker="o", s=58, color="#d62728",
                   edgecolor="white", linewidth=0.7, zorder=10, label="p < 0.05")
        ax.annotate("gap peak", xy=(sig_x[0], sig_y[0]), xytext=(sig_x[0] - 1.1, sig_y[0] + 0.08),
                    arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "#444444"},
                    fontsize=8)

    ax.set_xlabel("denoising checkpoint")
    ax.set_ylabel("KMeans silhouette")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0.0, 0.85)
    legend_handles = [
        Line2D([0], [0], color="#1f5fa8", marker="o", lw=1.5, markersize=4,
               label="observed"),
        Line2D([0], [0], color="#8c8c8c", marker="s", lw=1.1, linestyle="--",
               markersize=4, label="null mean"),
        Patch(facecolor="#1f5fa8", alpha=0.14, edgecolor="none",
              label="signal gap"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor="#d62728",
               markeredgecolor="white", markeredgewidth=0.7, markersize=4,
               label="p < 0.05"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True,
              framealpha=0.92, borderpad=0.3, handlelength=1.4,
              labelspacing=0.25)
    ax.grid(True, axis="y", ls=":", alpha=0.35)

    fig.savefig(FIG_DIR / "fig1_trajectory.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig1_trajectory.pdf'}")


def fig2_cross():
    """2x4 cross-model x cross-task signal-to-null gap matrix."""
    steps = [4, 16, 32, 64, 127]
    datasets = ["mbpp", "jsonschema", "gsm8k", "arc"]
    dataset_titles = {
        "mbpp": "MBPP (code)",
        "jsonschema": "JSON schema",
        "gsm8k": "GSM8K (math)",
        "arc": "ARC (sci. QA)",
    }
    models = ["llada", "dream"]
    fig, axes = plt.subplots(2, 4, figsize=(7.05, 3.05), sharey=True, sharex=True,
                             constrained_layout=True)
    for ri, model in enumerate(models):
        for ci, dataset in enumerate(datasets):
            ax = axes[ri, ci]
            traj = load_trajectory(model, dataset, steps)
            if not traj:
                ax.set_title(f"{model}/{dataset} (no data)")
                continue
            xs, xlabels = checkpoint_positions(traj)
            gap = [t["silhouette"] - t["null_mean"] for t in traj]
            ax.axhline(0, color="#222222", lw=0.7, ls=":")
            ax.plot(xs, gap, "o-", color="#1f5fa8", lw=1.4, ms=3.6)
            ax.fill_between(xs, 0, gap, where=[g >= 0 for g in gap],
                            color="#1f5fa8", alpha=0.10)
            ax.fill_between(xs, 0, gap, where=[g < 0 for g in gap],
                            color="#d62728", alpha=0.08)
            # Mark peak step
            peak_i, peak_gap = max(enumerate(gap), key=lambda x: x[1])
            ax.axvline(peak_i, color="#d62728", lw=0.8, ls=":", alpha=0.65)
            peak_sig = (traj[peak_i]["p_value"] or 1) < 0.05
            ax.scatter([peak_i], [peak_gap], s=38, color="#d62728",
                       edgecolor="#111111" if peak_sig else "white",
                       linewidth=0.9 if peak_sig else 0.6, zorder=9)
            if ri == 0:
                ax.set_title(dataset_titles[dataset], fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{model.title()}\ngap", fontsize=9)
            if ri == 1:
                ax.set_xlabel("step", fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, fontsize=7)
            ax.set_ylim(-0.18, 0.34)
            ax.grid(True, axis="y", ls=":", alpha=0.35)
    fig.savefig(FIG_DIR / "fig2_cross.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig2_cross.pdf'}")


def _fig2_cross_legacy():
    """Original 3-panel (kept for reference)."""
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

        ax.plot(xs, sil, "o-", color="#1f5fa8", lw=1.4, ms=3.5,
                label="silhouette")
        ax.plot(xs, null, "s--", color="#999999", lw=1.0, ms=3,
                label="null mean")

        sig_x = [t["step"] for t in traj if (t["p_value"] or 1) < 0.05]
        sig_y = [t["silhouette"] for t in traj if (t["p_value"] or 1) < 0.05]
        ax.scatter(sig_x, sig_y, marker="o", s=42, color="#d62728",
                   edgecolor="white", linewidth=0.6, zorder=10)

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
        ("baseline", None, None, None, "none"),
        ("suppress f15601", "f15601_a5.0_s64", 5.0, 64, "s64"),
        ("suppress f15601", "f15601_a5.0_s16", 5.0, 16, "s16"),
        ("suppress top-5", "f15601_8825_2087_11404_9657_a5.0_s64", 5.0, 64,
         "s64"),
        ("reverse f15601", "f15601_a-5.0_s64", -5.0, 64, "s64"),
    ]
    table_rows = []
    for label, suffix, alpha, sfrom, window in conds:
        if suffix is None:
            table_rows.append([label, window, "0/8", "0/3"])
            continue
        d = fetch_json(
            f"mbpp_llada/sae_steer_stage4_{suffix}.json",
            f"steer_{suffix}.json",
        )
        if d is None:
            table_rows.append([label, window, "--", "--"])
            continue
        fp = "--"
        pf = "--"
        for r in d["summaries"]:
            if r is None:
                continue
            if "fail_c1" in r["label"]:
                fp = f"{int(round(r['steer_pass_rate'] * r['n']))}/{r['n']}"
            elif "pass" in r["label"]:
                pf = f"{int(round((1 - r['steer_pass_rate']) * r['n']))}/{r['n']}"
        table_rows.append([label, window, fp, pf])

    fig, ax = plt.subplots(figsize=(3.35, 1.9), constrained_layout=True)
    ax.axis("off")
    col_labels = ["condition", "from", "fail→pass", "pass→fail"]
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.38, 0.16, 0.23, 0.23],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.22)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
        elif col in (2, 3):
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_text_props(ha="left")
    ax.text(0.5, 1.03, "No intervention flips correctness labels",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            transform=ax.transAxes)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.85), constrained_layout=True)

    im = ax1.imshow(J, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(N))
    ax1.set_yticks(range(N))
    ax1.set_xticklabels(xs, fontsize=9)
    ax1.set_yticklabels(xs, fontsize=9)
    ax1.set_xlabel("denoising step")
    ax1.set_ylabel("denoising step")
    ax1.set_title("Top-20 fail-feature Jaccard")
    for i in range(N):
        for j in range(N):
            v = J[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v < 0.4 else "black", fontsize=7.5)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard", fontsize=8)

    # f15601 enrichment across steps (plus a couple of comparator features)
    track_ids = [15601, 5561, 11265]
    track_labels = {15601: "f15601 (peak feat)", 5561: "f5561 (early-only)",
                    11265: "f11265 (late-only)"}
    colors = ["#d62728", "#1f5fa8", "#2ca02c"]
    x_pos = np.arange(len(xs))
    x_lookup = {x: i for i, x in enumerate(xs)}
    for fid, color in zip(track_ids, colors):
        ys = []
        for t in traj:
            row = next(
                (r for r in t["top_features"] if r["feature_id"] == fid),
                None,
            )
            ys.append(row["enrichment"] if row else None)
        # Plot, skipping None
        valid_x = [x_lookup[x] for x, y in zip(xs, ys) if y is not None]
        valid_y = [y for y in ys if y is not None]
        ax2.plot(valid_x, valid_y, "o-", color=color, lw=1.4, ms=4,
                 label=track_labels[fid])
    ax2.axhline(0, color="black", lw=0.5, ls=":")
    ax2.set_xlabel("denoising step")
    ax2.set_ylabel("fail-pass enrichment")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(x) for x in xs])
    ax2.set_title("Persistent peak feature emerges at s32")
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=7)
    ax2.grid(True, axis="y", ls=":", alpha=0.35)
    fig.savefig(FIG_DIR / "fig4_feature_drift.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig4_feature_drift.pdf'}")


if __name__ == "__main__":
    fig1_trajectory()
    fig2_cross()
    fig3_steering()
    fig4_feature_drift()
    print("\nAll figures saved to:", FIG_DIR)
