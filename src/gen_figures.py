"""Generate paper figures from midstep probe results.

Usage:
    python src/gen_figures.py

Reads midstep_probe_results.json from Modal volume (must be downloaded first)
or from local /tmp/ cache. Outputs to assets/.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -- Config --
RESULT_FILES = {
    ("jsonschema", "llada"): "/tmp/midstep_jsonschema_llada.json",
    ("jsonschema", "dream"): "/tmp/midstep_jsonschema_dream.json",
    ("gsm8k", "llada"): "/tmp/midstep_gsm8k_llada.json",
    ("gsm8k", "dream"): "/tmp/midstep_gsm8k_dream.json",
    ("mbpp", "llada"): "/tmp/midstep_mbpp_llada.json",
    ("mbpp", "dream"): "/tmp/midstep_mbpp_dream.json",
    ("arc", "llada"): "/tmp/midstep_arc_llada.json",
    ("arc", "dream"): "/tmp/midstep_arc_dream.json",
}

SHUFFLE_FILES = {
    (ds, m): f"/tmp/shuffle_{ds}_{m}.json"
    for ds in ["jsonschema", "gsm8k", "mbpp", "arc"]
    for m in ["llada", "dream"]
}

PANEL_TITLES = {
    ("jsonschema", "llada"): "LLaDA-8B / JSON Schema",
    ("jsonschema", "dream"): "Dream-7B / JSON Schema",
    ("gsm8k", "llada"): "LLaDA-8B / GSM8K",
    ("gsm8k", "dream"): "Dream-7B / GSM8K",
    ("mbpp", "llada"): "LLaDA-8B / MBPP",
    ("mbpp", "dream"): "Dream-7B / MBPP",
    ("arc", "llada"): "LLaDA-8B / ARC",
    ("arc", "dream"): "Dream-7B / ARC",
}

PANEL_ORDER = [
    ("jsonschema", "llada"),
    ("jsonschema", "dream"),
    ("gsm8k", "llada"),
    ("gsm8k", "dream"),
    ("mbpp", "llada"),
    ("mbpp", "dream"),
    ("arc", "llada"),
    ("arc", "dream"),
]


def load_results():
    import os
    data = {}
    for key, path in RESULT_FILES.items():
        with open(path) as f:
            data[key] = json.load(f)
    shuffle = {}
    for key, path in SHUFFLE_FILES.items():
        if os.path.exists(path):
            with open(path) as f:
                shuffle[key] = json.load(f)
    return data, shuffle


def _step_to_frac_label(step, total_steps=128):
    """Approximate fraction of generation tokens unmasked at a given step.

    Both LLaDA (block-based) and Dream (global linear) reach roughly the same
    cumulative fraction of unmasked tokens at the same step index: f ≈ step /
    total_steps. The two schedules differ in WHERE the tokens are unmasked
    (LLaDA fills block-by-block; Dream picks most-confident globally).
    """
    return f"{step} ({100 * step // total_steps}%)"


def _heatmap_grid(data, keys, out_path, nrows=2):
    """Step x Layer AUC heatmap grid."""
    mpl.rcParams.update({"font.size": 8, "font.family": "serif"})

    ncols = len(keys) // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    vmin, vmax = 0.5, 0.85

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        d = data[key]
        steps = d["checkpoint_steps"]
        n_layers = d["n_layers"]
        sla = d["step_layer_auc"]

        matrix = np.zeros((len(steps), n_layers))
        for i, s in enumerate(steps):
            aucs = sla[str(s)]
            matrix[i, :len(aucs)] = aucs

        im = ax.imshow(
            matrix, aspect="auto", cmap="YlOrRd",
            vmin=vmin, vmax=vmax, origin="lower",
            interpolation="nearest",
        )

        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels([_step_to_frac_label(s) for s in steps], fontsize=7)
        ax.set_ylabel("Step (% unmasked)")

        layer_ticks = list(range(0, n_layers, 4))
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels(layer_ticks)
        ax.set_xlabel("Layer")

        ax.set_title(PANEL_TITLES[key], fontsize=9)

        for i in range(len(steps)):
            best_l = int(np.argmax(matrix[i]))
            ax.plot(best_l, i, "k*", markersize=5)

    fig.subplots_adjust(right=0.88, wspace=0.4, hspace=0.55)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="AUC")

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


def fig1_heatmap(data):
    """Main paper: JSON Schema + GSM8K (1x4, full text width)."""
    keys = [
        ("jsonschema", "llada"), ("jsonschema", "dream"),
        ("gsm8k", "llada"), ("gsm8k", "dream"),
    ]
    _heatmap_grid(data, keys, "assets/fig1_heatmap.pdf", nrows=1)


def fig3_heatmap_appendix(data):
    """Appendix: MBPP + ARC (1x4)."""
    keys = [
        ("mbpp", "llada"), ("mbpp", "dream"),
        ("arc", "llada"), ("arc", "dream"),
    ]
    _heatmap_grid(data, keys, "assets/fig3_heatmap_appendix.pdf", nrows=1)


def fig2_auc_curve(data, shuffle, out_path="assets/fig2_auc_curve.pdf"):
    """AUC vs denoising progress, best layer per step, 1x4 grid."""
    import numpy as np
    mpl.rcParams.update({"font.size": 11, "font.family": "serif"})

    datasets = ["jsonschema", "gsm8k", "mbpp", "arc"]
    dataset_labels = {
        "jsonschema": "JSON Schema",
        "gsm8k": "GSM8K",
        "mbpp": "MBPP",
        "arc": "ARC",
    }
    model_styles = {
        "llada": {"color": "#d62728", "marker": "o", "label": "LLaDA-8B"},
        "dream": {"color": "#1f77b4", "marker": "s", "label": "Dream-7B"},
    }

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]

        steps_for_axis = None
        for model in ["llada", "dream"]:
            key = (ds, model)
            d = data[key]
            steps = d["checkpoint_steps"]
            steps_for_axis = steps
            sla = d["step_layer_auc"]

            best_aucs = []
            for s in steps:
                aucs = sla[str(s)]
                best_aucs.append(max(aucs))

            style = model_styles[model]
            ax.plot(
                range(len(steps)), best_aucs,
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2.0, markersize=6,
            )

        # Shuffled-label baseline: averaged across both models for this dataset.
        shufs = []
        for m2 in ["llada", "dream"]:
            if (ds, m2) in shuffle:
                sh = shuffle[(ds, m2)]
                shufs.append([sh["step_aucs"][str(s)]["mean"] for s in steps_for_axis])
        if shufs:
            arr = np.array(shufs).mean(axis=0)
            ax.plot(
                range(len(steps_for_axis)), arr,
                color="#888888", marker="x", linestyle="--",
                label="Shuffled" if ax_idx == len(datasets) - 1 else None,
                linewidth=1.2, markersize=5,
            )

        steps = steps_for_axis

        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([_step_to_frac_label(s) for s in steps],
                           fontsize=8, rotation=30, ha="right")
        ax.set_xlabel("Step (% unmasked)", fontsize=10)
        ax.set_title(dataset_labels[ds], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.45, 0.90)
        ax.tick_params(axis="y", labelsize=9)
        ax.axhline(0.5, color="#cccccc", linestyle=":", linewidth=0.8, zorder=0)

    axes[0].set_ylabel("Best AUC (across layers)", fontsize=10)
    axes[-1].legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    data, shuffle = load_results()
    fig1_heatmap(data)
    fig2_auc_curve(data, shuffle)
    fig3_heatmap_appendix(data)
    print("Done.")
