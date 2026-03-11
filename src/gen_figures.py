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
    data = {}
    for key, path in RESULT_FILES.items():
        with open(path) as f:
            data[key] = json.load(f)
    return data


def _heatmap_grid(data, keys, out_path, nrows=2):
    """Step x Layer AUC heatmap grid."""
    mpl.rcParams.update({"font.size": 8, "font.family": "serif"})

    ncols = len(keys) // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.3 * nrows))
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
        ax.set_yticklabels(steps)
        ax.set_ylabel("Diffusion step")

        layer_ticks = list(range(0, n_layers, 4))
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels(layer_ticks)
        ax.set_xlabel("Layer")

        ax.set_title(PANEL_TITLES[key], fontsize=9)

        for i in range(len(steps)):
            best_l = int(np.argmax(matrix[i]))
            ax.plot(best_l, i, "k*", markersize=5)

    fig.subplots_adjust(right=0.88, wspace=0.3, hspace=0.55)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="AUC")

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


def fig1_heatmap(data):
    """Main paper: JSON Schema + GSM8K (2x2)."""
    keys = [
        ("jsonschema", "llada"), ("jsonschema", "dream"),
        ("gsm8k", "llada"), ("gsm8k", "dream"),
    ]
    _heatmap_grid(data, keys, "assets/fig1_heatmap.pdf", nrows=2)


def fig3_heatmap_appendix(data):
    """Appendix: MBPP + ARC (2x2)."""
    keys = [
        ("mbpp", "llada"), ("mbpp", "dream"),
        ("arc", "llada"), ("arc", "dream"),
    ]
    _heatmap_grid(data, keys, "assets/fig3_heatmap_appendix.pdf", nrows=2)


def fig2_auc_curve(data, out_path="assets/fig2_auc_curve.pdf"):
    """AUC vs diffusion step, best layer per step, 1x4 grid."""
    mpl.rcParams.update({"font.size": 9, "font.family": "serif"})

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

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]

        for model in ["llada", "dream"]:
            key = (ds, model)
            d = data[key]
            steps = d["checkpoint_steps"]
            sla = d["step_layer_auc"]

            # Best AUC per step (across all layers)
            best_aucs = []
            for s in steps:
                aucs = sla[str(s)]
                best_aucs.append(max(aucs))

            style = model_styles[model]
            ax.plot(
                range(len(steps)), best_aucs,
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=1.5, markersize=4,
            )

        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, fontsize=7)
        ax.set_xlabel("Diffusion step")
        ax.set_title(dataset_labels[ds], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.55, 0.90)

    axes[0].set_ylabel("Best AUC (across layers)")
    axes[-1].legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    data = load_results()
    fig1_heatmap(data)
    fig2_auc_curve(data)
    fig3_heatmap_appendix(data)
    print("Done.")
