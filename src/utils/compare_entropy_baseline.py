"""Merge probe + entropy baseline results into comparison tables for the paper.

Reads from Modal volume:
  /results/{dataset}_{model}/early_exit_results.json
  /results/{dataset}_{model}/entropy_baseline_results.json

Outputs Markdown tables suitable for:
  - paper/latex appendix table comparing probe vs entropy/maxprob signals
  - per-step AUC comparison (probe vs raw entropy vs raw maxprob vs LR-on-entropy)
  - selective-generation comparison at matched thresholds

Usage:
  .venv/bin/python src/compare_entropy_baseline.py --dataset jsonschema --model llada
  .venv/bin/python src/compare_entropy_baseline.py --all
"""

import argparse
import json
import os
import subprocess
import sys

DATASETS = ["jsonschema", "gsm8k", "mbpp", "arc"]
MODELS = ["llada", "dream"]
CHECKPOINT_STEPS = [0, 1, 4, 16, 32, 64, 127]
THRESHOLDS_REPORT = [0.6, 0.7, 0.8, 0.9]


def fetch(dataset, model, filename, dest_dir="/tmp"):
    """Download a JSON result file from the Modal volume."""
    out_path = os.path.join(dest_dir, f"{dataset}_{model}_{filename}")
    rel = f"{dataset}_{model}/{filename}"
    subprocess.run(
        [".venv/bin/modal", "volume", "get", "probe-results", rel, out_path, "--force"],
        check=False, capture_output=True,
    )
    if not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        return json.load(f)


def per_step_auc_table(dataset, model):
    probe = fetch(dataset, model, "early_exit_results.json")
    ent = fetch(dataset, model, "entropy_baseline_results.json")
    if probe is None or ent is None:
        return f"[skip {dataset}_{model}: missing data]"

    rows = []
    rows.append(f"### {dataset}_{model} per-step AUC")
    rows.append("")
    rows.append("| step | probe | LR-entropy | raw -entropy | raw maxprob |")
    rows.append("|---:|---:|---:|---:|---:|")
    for s in CHECKPOINT_STEPS:
        sk = str(s)
        p = probe["step_aucs"].get(sk, "-")
        lr = ent["lr_step_aucs"].get(sk, "-")
        re_ = ent["raw_auc_neg_entropy"].get(sk, "-")
        rmp = ent["raw_auc_maxprob"].get(sk, "-")
        rows.append(f"| {s} | {p:.3f} | {lr:.3f} | {re_:.3f} | {rmp:.3f} |"
                     if isinstance(p, float) else f"| {s} | {p} | {lr} | {re_} | {rmp} |")
    return "\n".join(rows)


def selective_gen_table(dataset, model):
    probe = fetch(dataset, model, "early_exit_results.json")
    ent = fetch(dataset, model, "entropy_baseline_results.json")
    if probe is None or ent is None:
        return f"[skip {dataset}_{model}: missing data]"

    def find(rows_list, t):
        for r in rows_list:
            if abs(r["threshold"] - t) < 1e-6:
                return r
        return None

    rows = []
    rows.append(f"### {dataset}_{model} selective generation (compute_saved% / accuracy%)")
    rows.append("")
    rows.append("| τ | probe | LR-entropy | raw -entropy | raw maxprob |")
    rows.append("|---:|---:|---:|---:|---:|")
    for t in THRESHOLDS_REPORT:
        p = find(probe["early_exit_results"], t)
        le = find(ent["selective_gen_lr_entropy"], t)
        re_ = find(ent["selective_gen_raw_neg_entropy"], t)
        rmp = find(ent["selective_gen_raw_maxprob"], t)
        cells = []
        for cell in [p, le, re_, rmp]:
            if cell is None:
                cells.append("-")
            else:
                cells.append(f"{cell['compute_saved_pct']:.1f}/{cell['accuracy']:.1f}")
        rows.append(f"| {t:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def summary_one(dataset, model):
    out = []
    out.append(per_step_auc_table(dataset, model))
    out.append("")
    out.append(selective_gen_table(dataset, model))
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for d in DATASETS:
            for m in MODELS:
                print(summary_one(d, m))
                print()
    else:
        if not args.dataset or not args.model:
            print("Specify --dataset and --model, or --all", file=sys.stderr)
            sys.exit(1)
        print(summary_one(args.dataset, args.model))


if __name__ == "__main__":
    main()
