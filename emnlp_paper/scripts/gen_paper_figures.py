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


def _load_dense_seed(seed: int) -> list[dict]:
    """Load merged anchors + plateau + post-plateau sweep for one generation seed."""
    suffix = f"_seed{seed}" if seed > 0 else ""
    parts = [
        f"mbpp_llada_dense_anchors{suffix}",
        f"mbpp_llada_dense{suffix}",
        f"mbpp_llada_dense_post{suffix}",
    ]
    seen, rows = set(), []
    for vol in parts:
        local = CACHE_DIR / f"{vol}.json"
        if not local.exists():
            subprocess.run(
                ["rtk", "proxy", "modal", "volume", "get", "probe-results",
                 f"{vol}/dense_sweep_results.json", str(local)],
                capture_output=True, text=True,
            )
        if not local.exists():
            continue
        try:
            d = json.load(open(local))
        except Exception:
            continue
        for rec in d.get("per_step", []):
            s = rec["step"]
            if s in seen:
                continue
            seen.add(s)
            rows.append({
                "step": s,
                "silhouette": rec["silhouette"],
                "null_mean": rec["null_mean"],
                "p_value": rec.get("p_value"),
            })
    rows.sort(key=lambda r: r["step"])
    return rows


def fig1_trajectory():
    """LLaDA / MBPP 21-step dense trajectory with 3-seed overlay and plateau band."""
    seeds = [_load_dense_seed(i) for i in (0, 1, 2)]
    base = seeds[0]
    if not base:
        print("fig1: no data, skipping")
        return
    steps = [r["step"] for r in base]
    pos = {s: i for i, s in enumerate(steps)}
    xs = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)

    # Plateau band [48, 116]: shaded only, no in-figure label.
    if 48 in pos and 116 in pos:
        ax.axvspan(pos[48] - 0.4, pos[116] + 0.4, color="#f5cf99",
                   alpha=0.45, zorder=0)

    # Faint seed-1 / seed-2 overlays.
    for srows in seeds[1:]:
        if not srows:
            continue
        sx = [pos[r["step"]] for r in srows if r["step"] in pos]
        sy = [r["silhouette"] for r in srows if r["step"] in pos]
        ax.plot(sx, sy, color="#1f5fa8", lw=0.7, alpha=0.32, zorder=2)

    sil = [r["silhouette"] for r in base]
    nul = [r["null_mean"] for r in base]
    ax.plot(xs, nul, "--", color="#8c8c8c", lw=1.0, zorder=3)
    ax.plot(xs, sil, "-", color="#1f5fa8", lw=1.5, zorder=4)

    sig_x = [i for i, r in enumerate(base) if (r["p_value"] or 1) < 0.05]
    sig_y = [sil[i] for i in sig_x]
    if sig_x:
        ax.scatter(sig_x, sig_y, marker="o", s=22, color="#d62728",
                   edgecolor="white", linewidth=0.5, zorder=10)

    ax.set_xlabel("denoising checkpoint")
    ax.set_ylabel("KMeans silhouette (top-20 fail)")
    label_steps = {0, 4, 16, 32, 48, 64, 80, 100, 116, 127}
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) if s in label_steps else "" for s in steps],
                       fontsize=6.5)
    ax.set_ylim(0.0, 0.95)
    ax.grid(True, axis="y", ls=":", alpha=0.35)

    legend_handles = [
        Line2D([0], [0], color="#1f5fa8", lw=1.5, label="seed 0"),
        Line2D([0], [0], color="#1f5fa8", lw=0.7, alpha=0.5,
               label="seed 1-2"),
        Line2D([0], [0], color="#8c8c8c", lw=1.0, linestyle="--",
               label="null mean"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor="#d62728",
               markeredgecolor="white", markeredgewidth=0.5, markersize=4.2,
               label="$p < 0.05$"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.42), frameon=False,
              borderpad=0.2, handlelength=1.4, columnspacing=1.2,
              handletextpad=0.4, fontsize=7)

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
    fig, axes = plt.subplots(2, 4, figsize=(7.05, 3.2), sharey=True, sharex=True,
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
            ax.set_ylim(-0.18, 0.38)
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
    ax1.set_title("(a) Top-20 fail-feature Jaccard")
    for i in range(N):
        for j in range(N):
            v = J[i, j]
            # Pick text color for contrast against viridis at this value.
            txt_color = "white" if v < 0.55 else "black"
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color=txt_color, fontsize=7.5)
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
    ax2.set_title("(b) Persistent peak feature emerges at s32")
    ax2.legend(loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.36), frameon=False,
               borderpad=0.2, handlelength=1.4, columnspacing=1.2,
               handletextpad=0.4, fontsize=7)
    ax2.grid(True, axis="y", ls=":", alpha=0.35)
    fig.savefig(FIG_DIR / "fig4_feature_drift.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig4_feature_drift.pdf'}")


def fig5_auc_compare():
    """Grouped bar chart for the 12 cells x 4 AUC metrics replacing tab:auc_compare."""
    cells = [
        ("LLaDA", "MBPP", [0.59, 0.73, 0.73, 0.82]),
        ("LLaDA", "JSON",  [0.67, 0.66, 0.69, 0.81]),
        ("LLaDA", "GSM8K", [0.69, 0.71, 0.72, 0.77]),
        ("LLaDA", "ARC",   [0.61, 0.65, 0.66, 0.71]),
        ("Dream", "MBPP", [0.70, 0.69, 0.73, 0.82]),
        ("Dream", "JSON",  [0.68, 0.74, 0.78, 0.80]),
        ("Dream", "GSM8K", [0.57, 0.61, 0.66, 0.70]),
        ("Dream", "ARC",   [0.49, 0.54, 0.56, 0.57]),
        ("Qwen",  "MBPP", [0.57, 0.66, 0.59, 0.61]),
        ("Qwen",  "JSON",  [0.59, 0.66, 0.71, 0.75]),
        ("Qwen",  "GSM8K", [0.59, 0.66, 0.71, 0.75]),
        ("Qwen",  "ARC",   [0.56, 0.59, 0.65, 0.74]),
    ]
    metric_labels = ["top-1", "top-5", "top-20", "raw"]
    metric_colors = ["#f5cf99", "#f5a261", "#1f5fa8", "#6c3a8e"]
    fig, ax = plt.subplots(figsize=(7.05, 2.6), constrained_layout=True)
    n_cells = len(cells)
    n_metrics = 4
    width = 0.20
    xs = np.arange(n_cells)
    for m in range(n_metrics):
        vals = [c[2][m] for c in cells]
        ax.bar(xs + (m - 1.5) * width, vals, width=width,
               color=metric_colors[m], edgecolor="white", linewidth=0.3,
               label=metric_labels[m])
    # Star raw bar where raw is the per-cell max.
    for ci, (_, _, vals) in enumerate(cells):
        winner = int(np.argmax(vals))
        ax.scatter([xs[ci] + (winner - 1.5) * width], [vals[winner] + 0.025],
                   marker=(5, 1, 0), s=12, color="#222", zorder=10)
    # X labels: "model / task"
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c[0]}/{c[1]}" for c in cells], rotation=35,
                       ha="right", fontsize=7)
    ax.set_ylabel("AUC at step 64")
    ax.set_ylim(0.4, 0.95)
    ax.axhline(0.5, color="#888", lw=0.5, ls=":")
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.10),
              frameon=False, fontsize=7, columnspacing=1.4,
              handlelength=1.2, handletextpad=0.4)
    fig.savefig(FIG_DIR / "fig5_auc_compare.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig5_auc_compare.pdf'}")


def fig6_dense_compare():
    """LLaDA vs Dream dense MBPP sweep over [48,80] replacing tab:dense_sweep."""
    llada = [
        (48, 0.280, True),  (52, 0.336, True), (56, 0.331, True),
        (60, 0.266, True),  (64, 0.253, False), (68, 0.369, True),
        (72, 0.371, True),  (76, 0.351, True), (80, 0.219, False),
    ]
    dream = [
        (48, 0.105, False), (52, 0.106, False), (56, 0.134, False),
        (60, 0.114, False), (64, 0.107, False), (68, 0.156, False),
        (72, 0.074, False), (76, 0.084, False), (80, 0.043, False),
    ]
    xs = [r[0] for r in llada]
    fig, ax = plt.subplots(figsize=(3.35, 2.0), constrained_layout=True)
    ax.axhline(0, color="#222", lw=0.7, ls=":")
    ax.plot(xs, [r[1] for r in llada], "o-", color="#1f5fa8", lw=1.4, ms=4,
            label="LLaDA-8B")
    ax.plot(xs, [r[1] for r in dream], "s--", color="#a8202a", lw=1.2, ms=3.6,
            label="Dream-7B")
    # Significance markers (filled = p<0.05)
    sig_x = [r[0] for r in llada if r[2]]
    sig_y = [r[1] for r in llada if r[2]]
    ax.scatter(sig_x, sig_y, marker="o", s=70, facecolor="none",
               edgecolor="#d62728", linewidth=1.0, zorder=10,
               label="$p<0.05$")
    ax.set_xlabel("denoising step", labelpad=6.0)
    ax.set_ylabel("signal-to-null gap")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=7)
    ax.set_ylim(-0.02, 0.45)
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    ax.legend(loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.58), frameon=False,
              borderpad=0.2, handlelength=1.4, columnspacing=1.2,
              handletextpad=0.4, fontsize=7)
    fig.savefig(FIG_DIR / "fig6_dense_compare.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig6_dense_compare.pdf'}")


def fig7_topN_sensitivity():
    """Top-N sweep gap vs N for step 64 and step 68 (replaces tab:sensitivity part (a))."""
    Ns = [10, 20, 30, 50]
    s64 = {"gap": [0.34, 0.25, 0.20, 0.12], "p": [0.019, 0.081, 0.110, 0.192]}
    s68 = {"gap": [0.38, 0.37, 0.29, 0.23], "p": [0.015, 0.005, 0.016, 0.009]}
    fig, ax = plt.subplots(figsize=(3.35, 2.0), constrained_layout=True)
    ax.axhline(0, color="#222", lw=0.6, ls=":")
    ax.plot(Ns, s64["gap"], "o-", color="#1f5fa8", lw=1.4, ms=5, label="step 64")
    ax.plot(Ns, s68["gap"], "s--", color="#a8202a", lw=1.4, ms=4.5, label="step 68")
    for x, g, pv in zip(Ns, s64["gap"], s64["p"]):
        if pv < 0.05:
            ax.scatter([x], [g], marker="o", s=72, facecolor="none",
                       edgecolor="#1f5fa8", linewidth=1.0, zorder=10)
    for x, g, pv in zip(Ns, s68["gap"], s68["p"]):
        if pv < 0.05:
            ax.scatter([x], [g], marker="s", s=72, facecolor="none",
                       edgecolor="#a8202a", linewidth=1.0, zorder=10)
    ax.set_xlabel("top-$N$ fail features")
    ax.set_ylabel("signal-to-null gap")
    ax.set_xticks(Ns)
    ax.set_ylim(0.05, 0.45)
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    ax.legend(loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.42), frameon=False,
              borderpad=0.2, handlelength=1.4, columnspacing=1.4,
              handletextpad=0.4, fontsize=7)
    fig.savefig(FIG_DIR / "fig7_topN_sensitivity.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig7_topN_sensitivity.pdf'}")


def fig8_crosslayer():
    """LLaDA-MBPP signal-to-null gap across SAE layers {11,16,26,30} at steps 64/68."""
    src = Path("/tmp/paper_figs_dense/crosslayer_diagnose.json")
    if not src.exists():
        print(f"fig8: missing {src}, skipping")
        return
    data = json.load(open(src))
    layers = [L["sae_layer"] for L in data["layers"]]
    rows = {L["sae_layer"]: {s["step"]: s for s in L["steps"]} for L in data["layers"]}

    fig, ax = plt.subplots(figsize=(3.35, 2.15), constrained_layout=True)
    xs = np.arange(len(layers))
    width = 0.38

    gaps_64 = [rows[L][64]["gap"] for L in layers]
    gaps_68 = [rows[L][68]["gap"] for L in layers]
    p64 = [rows[L][64]["p_value"] for L in layers]
    p68 = [rows[L][68]["p_value"] for L in layers]

    ax.bar(xs - width / 2, gaps_64, width=width, color="#1f5fa8",
           edgecolor="white", linewidth=0.4, label="step 64")
    ax.bar(xs + width / 2, gaps_68, width=width, color="#a8202a",
           edgecolor="white", linewidth=0.4, label="step 68")

    # Mark significance with asterisks above bars
    for i, p in enumerate(p64):
        if p < 0.05:
            ax.text(xs[i] - width / 2, gaps_64[i] + 0.008, "*",
                    ha="center", va="bottom", fontsize=11, color="#1f5fa8")
    for i, p in enumerate(p68):
        if p < 0.05:
            ax.text(xs[i] + width / 2, gaps_68[i] + 0.008, "*",
                    ha="center", va="bottom", fontsize=11, color="#a8202a")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_xlabel("DLM-Scope LLaDA Mask-SAE layer", labelpad=8.0)
    ax.set_ylabel("signal-to-null gap")
    ax.set_ylim(0.0, 0.42)
    ax.axhline(0, color="#888", lw=0.6, ls=":")
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    ax.legend(loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.55), frameon=False,
              borderpad=0.2, handlelength=1.4, columnspacing=1.4,
              handletextpad=0.4, fontsize=7)

    fig.savefig(FIG_DIR / "fig8_crosslayer.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig8_crosslayer.pdf'}")


def fig9_fisher():
    """Horizontal bar of -log10(Fisher combined p) across 8 (model, task) cells."""
    src = Path("/tmp/paper_figs/fisher_per_cell.json")
    if not src.exists():
        print(f"fig9: missing {src}, skipping")
        return
    cells = json.load(open(src))
    # Sort by Fisher p ascending (smallest p first; largest -log10 first)
    cells = sorted(cells, key=lambda c: c["fisher_p"])
    labels = [c["cell"] for c in cells]
    neglog = [-np.log10(c["fisher_p"]) for c in cells]
    sig_mask = [c["fisher_p"] < 0.05 for c in cells]

    fig, ax = plt.subplots(figsize=(3.35, 2.6), constrained_layout=True)
    ypos = np.arange(len(labels))[::-1]  # most significant on top
    colors = ["#1f5fa8" if s else "#bcbcbc" for s in sig_mask]
    ax.barh(ypos, neglog, color=colors, edgecolor="white", linewidth=0.4, height=0.7)

    # Threshold line at -log10(0.05); label moved to caption.
    thresh = -np.log10(0.05)
    ax.axvline(thresh, color="#a8202a", lw=0.9, ls="--")

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("$-\\log_{10}$ Fisher combined $p$", labelpad=6.0)
    ax.set_xlim(0, max(neglog) * 1.20)
    ax.grid(True, axis="x", ls=":", alpha=0.35)

    fig.savefig(FIG_DIR / "fig9_fisher.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig9_fisher.pdf'}")


def fig10_ardlm():
    """Per-layer CV-AUC and cos-similarity for Qwen-Base, Qwen-Instruct, Dream-Base."""
    src = Path("/tmp/paper_figs/ardlm_3way.json")
    if not src.exists():
        print(f"fig10: missing {src}, skipping")
        return
    d = json.load(open(src))
    rows = d["per_layer"]
    L = np.array([r["layer"] for r in rows])
    auc_QB = np.array([r["auc_QB"] for r in rows])
    auc_QI = np.array([r["auc_QI"] for r in rows])
    auc_DB = np.array([r["auc_DB"] for r in rows])
    cos_QB_DB = np.array([r["cos_QB_DB_mean"] for r in rows])
    cos_QB_QI = np.array([r["cos_QB_QI_mean"] for r in rows])
    cos_QI_DB = np.array([r["cos_QI_DB_mean"] for r in rows])

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.35, 3.8),
                                   constrained_layout=True, sharex=True)

    axA.plot(L, auc_QB, "-o", color="#1f5fa8", lw=1.2, ms=3.2,
             label="Qwen-2.5-7B-Base (AR)")
    axA.plot(L, auc_QI, "-s", color="#2e8b3a", lw=1.2, ms=3.2,
             label="Qwen-2.5-7B-Instruct (AR+IFT)")
    axA.plot(L, auc_DB, "-^", color="#a8202a", lw=1.2, ms=3.5,
             label="Dream-7B-Base (DLM)")
    qb_peak = int(L[np.argmax(auc_QB)])
    qi_peak = int(L[np.argmax(auc_QI)])
    db_peak = int(L[np.argmax(auc_DB)])
    for x, c in [(qb_peak, "#1f5fa8"), (qi_peak, "#2e8b3a"), (db_peak, "#a8202a")]:
        axA.axvline(x, color=c, lw=0.5, ls=":", alpha=0.5)
    axA.set_ylabel("5-fold CV-AUC")
    axA.set_ylim(0.55, 0.78)
    axA.grid(True, ls=":", alpha=0.3)
    axA.legend(loc="lower right", fontsize=7, frameon=False,
               handlelength=1.4, handletextpad=0.4, borderpad=0.2)

    axB.plot(L, cos_QB_QI, "-s", color="#2e8b3a", lw=1.2, ms=3.0,
             label="cos(QB, QI)")
    axB.plot(L, cos_QB_DB, "-^", color="#a8202a", lw=1.2, ms=3.0,
             label="cos(QB, DB)")
    axB.plot(L, cos_QI_DB, "-d", color="#8a4ba3", lw=1.2, ms=3.0,
             label="cos(QI, DB)")
    axB.set_xlabel("transformer layer index", labelpad=6.0)
    axB.set_ylabel("per-sample cosine")
    axB.set_ylim(0.0, 1.0)
    axB.set_xticks(np.arange(0, 28, 4))
    axB.grid(True, ls=":", alpha=0.3)
    axB.legend(loc="lower left", fontsize=7, frameon=False,
               handlelength=1.4, handletextpad=0.4, borderpad=0.2)

    fig.savefig(FIG_DIR / "fig10_ardlm.pdf")
    plt.close(fig)
    print(f"saved {FIG_DIR / 'fig10_ardlm.pdf'}")


if __name__ == "__main__":
    fig1_trajectory()
    fig2_cross()
    # fig3_steering() is not cited by the manuscript (Table 1 covers it).
    fig4_feature_drift()
    fig5_auc_compare()
    fig6_dense_compare()
    fig7_topN_sensitivity()
    fig8_crosslayer()
    fig9_fisher()
    fig10_ardlm()
    print("\nAll figures saved to:", FIG_DIR)
