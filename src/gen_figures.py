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
}

PANEL_TITLES = {
    ("jsonschema", "llada"): "LLaDA-8B / JSON Schema",
    ("jsonschema", "dream"): "Dream-7B / JSON Schema",
    ("gsm8k", "llada"): "LLaDA-8B / GSM8K",
    ("gsm8k", "dream"): "Dream-7B / GSM8K",
}

PANEL_ORDER = [
    ("jsonschema", "llada"),
    ("jsonschema", "dream"),
    ("gsm8k", "llada"),
    ("gsm8k", "dream"),
]


def load_results():
    data = {}
    for key, path in RESULT_FILES.items():
        with open(path) as f:
            data[key] = json.load(f)
    return data


def fig1_heatmap(data, out_path="assets/fig1_heatmap.pdf"):
    """Step x Layer AUC heatmap, 2x2 grid."""
    mpl.rcParams.update({"font.size": 9, "font.family": "serif"})

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    vmin, vmax = 0.5, 0.85

    for idx, key in enumerate(PANEL_ORDER):
        ax = axes[idx // 2][idx % 2]
        d = data[key]
        steps = d["checkpoint_steps"]
        n_layers = d["n_layers"]
        sla = d["step_layer_auc"]

        # Build matrix: rows=steps, cols=layers
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

        # Show every 4th layer
        layer_ticks = list(range(0, n_layers, 4))
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels(layer_ticks)
        ax.set_xlabel("Layer")

        ax.set_title(PANEL_TITLES[key], fontsize=10)

        # Mark best layer per step
        for i in range(len(steps)):
            best_l = int(np.argmax(matrix[i]))
            ax.plot(best_l, i, "k*", markersize=6)

    fig.subplots_adjust(right=0.88, wspace=0.3, hspace=0.45)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="AUC")

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


def fig2_auc_curve(data, out_path="assets/fig2_auc_curve.pdf"):
    """AUC vs diffusion step, key layers only, two models overlaid."""
    mpl.rcParams.update({"font.size": 10, "font.family": "serif"})

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    datasets = ["jsonschema", "gsm8k"]
    dataset_labels = {"jsonschema": "JSON Schema", "gsm8k": "GSM8K"}
    model_styles = {
        "llada": {"color": "#d62728", "marker": "o", "label": "LLaDA-8B"},
        "dream": {"color": "#1f77b4", "marker": "s", "label": "Dream-7B"},
    }

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
                label=style["label"], linewidth=1.5, markersize=5,
            )

        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps)
        ax.set_xlabel("Diffusion step")
        ax.set_title(dataset_labels[ds], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.7, 0.87)

    axes[0].set_ylabel("Best AUC (across layers)")
    axes[1].legend(loc="lower right", fontsize=9)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    data = load_results()
    fig1_heatmap(data)
    fig2_auc_curve(data)
    print("Done.")
