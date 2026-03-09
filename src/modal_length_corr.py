"""Check correlation between output length and correctness labels."""

import modal

app = modal.App("probe-length-corr")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "scikit-learn", "datasets==2.21.0")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)

DATASET_CFGS = {
    "jsonschema": {"total": 272},
    "gsm8k": {"total": 1319},
}


@app.function(
    image=image,
    timeout=600,
    volumes={"/results": RESULTS_VOL},
)
def check_correlation():
    import json
    import os

    import numpy as np
    from datasets import load_dataset
    from scipy.stats import pointbiserialr, pearsonr

    RESULTS_VOL.reload()

    # Load reference lengths
    ds_js = load_dataset("eth-sri/json-mode-eval-extended", split="test")
    js_instances = sorted(list(ds_js), key=lambda x: x["instance_id"])
    js_lengths = np.array([len(inst["output"]) for inst in js_instances])

    ds_gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_instances = list(ds_gsm)
    gsm_lengths = np.array([len(inst["answer"]) for inst in gsm_instances])

    ref_lengths = {"jsonschema": js_lengths, "gsm8k": gsm_lengths}

    for dataset in ["jsonschema", "gsm8k"]:
        for model in ["llada", "dream"]:
            total = DATASET_CFGS[dataset]["total"]
            n_chunks = 8
            chunk_size = (total + n_chunks - 1) // n_chunks
            in_dir = f"/results/{dataset}_{model}"

            all_labels = []
            for i in range(n_chunks):
                offset = i * chunk_size
                path = f"{in_dir}/chunk_off{offset}.npz"
                if os.path.exists(path):
                    data = np.load(path)
                    all_labels.append(data["labels"])

            correctness = np.concatenate(all_labels)
            lengths = ref_lengths[dataset][:len(correctness)]
            median_len = np.median(lengths)
            length_binary = (lengths > median_len).astype(int)

            # Correlations
            corr_binary, p_binary = pearsonr(length_binary, correctness)
            corr_cont, p_cont = pointbiserialr(correctness, lengths)

            print(f"\n=== {dataset}_{model} ===")
            print(f"  N={len(correctness)}, functional={correctness.sum()}/{len(correctness)} "
                  f"({100*correctness.mean():.1f}%)")
            print(f"  Length: median={median_len:.0f}, "
                  f"above_median={length_binary.sum()}/{len(length_binary)}")
            print(f"  Correlation (binary length vs correctness): "
                  f"r={corr_binary:.4f}, p={p_binary:.4f}")
            print(f"  Correlation (continuous length vs correctness): "
                  f"r={corr_cont:.4f}, p={p_cont:.4f}")

            # Contingency table
            tp = ((length_binary == 1) & (correctness == 1)).sum()
            fp = ((length_binary == 1) & (correctness == 0)).sum()
            fn = ((length_binary == 0) & (correctness == 1)).sum()
            tn = ((length_binary == 0) & (correctness == 0)).sum()
            print(f"  Contingency: long+correct={tp}, long+incorrect={fp}, "
                  f"short+correct={fn}, short+incorrect={tn}")
            print(f"  P(correct|long)={tp/(tp+fp):.3f}, "
                  f"P(correct|short)={fn/(fn+tn):.3f}")


@app.local_entrypoint()
def main():
    check_correlation.remote()
