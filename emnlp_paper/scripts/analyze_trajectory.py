"""Trajectory-aware analysis over per-step SAE diagnose results.

Reads /results/{dataset}_{model}/sae_diagnose_stage2_s{step}.json for each
cached step and computes:
  - Silhouette vs step (line chart data)
  - Permutation p vs step
  - Top-N fail feature Jaccard between (i) consecutive steps, (ii) each step vs reference s64
  - Persistence: which features appear in top-20 at >=3 of N steps
  - Cross-step enrichment of persistent features (for the trajectory plot)

Pulls JSON via `modal volume get` into /tmp/traj_<model>/.
Prints tables; saves a summary JSON to /tmp/traj_<model>/summary.json.

Usage:
  .venv/bin/python src/applications/sae/analyze_trajectory.py --model llada
  .venv/bin/python src/applications/sae/analyze_trajectory.py --model dream
"""

import argparse
import json
import os
import subprocess

STEPS_LLADA = [0, 1, 4, 16, 32, 64, 127]
STEPS_DREAM = [0, 1, 4, 16, 32, 64, 127]
TOP_N = 20


def fetch(model: str, dataset: str, steps: list[int]) -> dict[int, dict]:
    out_dir = f"/tmp/traj_{model}"
    os.makedirs(out_dir, exist_ok=True)
    by_step = {}
    for s in steps:
        local_path = f"{out_dir}/{dataset}_s{s}.json"
        if not os.path.exists(local_path):
            remote = f"{dataset}_{model}/sae_diagnose_stage2_s{s}.json"
            r = subprocess.run(
                ["rtk", "proxy", "modal", "volume", "get",
                 "probe-results", remote, local_path],
                capture_output=True, text=True,
            )
            if r.returncode != 0 or not os.path.exists(local_path):
                print(f"  skip s={s}: not yet on volume")
                continue
        try:
            by_step[s] = json.load(open(local_path))
        except Exception as e:
            print(f"  skip s={s}: load failed ({e})")
    return by_step


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llada", choices=["llada", "dream"])
    p.add_argument("--dataset", default="mbpp")
    args = p.parse_args()

    steps = STEPS_LLADA if args.model == "llada" else STEPS_DREAM
    data = fetch(args.model, args.dataset, steps)
    available = sorted(data.keys())
    if not available:
        print("no data; aborting")
        return

    print(f"\n{'='*78}")
    print(f"Trajectory analysis: {args.model} / {args.dataset}")
    print(f"Steps available: {available}")
    print(f"{'='*78}")

    # ---- Silhouette + p trajectory ----
    print(f"\n{'step':>5} {'best_K':>7} {'silhouette':>11} {'null_mean':>10} "
          f"{'gap':>7} {'p':>7} {'top1_fid':>9} {'top1_enr':>9}")
    print("-" * 78)
    traj = []
    for s in available:
        d = data[s]
        sil = d.get("best_silhouette") or 0.0
        null_m = d.get("null_silhouette_mean") or 0.0
        gap = sil - null_m
        pv = d.get("permutation_p")
        pv_str = f"{pv:.3f}" if pv is not None else "-"
        top1 = d["top_fail_features"][0]
        print(
            f"{s:>5} {d['best_k']:>7} {sil:>11.3f} {null_m:>10.3f} "
            f"{gap:>+7.3f} {pv_str:>7} f{top1['feature_id']:>4} "
            f"{top1['enrichment']:>+9.3f}"
        )
        traj.append({
            "step": s,
            "silhouette": sil,
            "null_mean": null_m,
            "gap": gap,
            "p_value": pv,
            "top1_feature": top1["feature_id"],
            "top1_enrichment": top1["enrichment"],
        })

    # ---- Top-N Jaccard between consecutive steps and vs s64 reference ----
    sets = {s: {r["feature_id"] for r in data[s]["top_fail_features"][:TOP_N]}
            for s in available}
    print(f"\nTop-{TOP_N} fail-feature Jaccard between consecutive steps:")
    for a, b in zip(available, available[1:]):
        j = jaccard(sets[a], sets[b])
        shared = sets[a] & sets[b]
        print(f"  s{a:>3} -> s{b:>3}: Jaccard={j:.3f} "
              f"(shared {len(shared)}/{TOP_N})")

    if 64 in available:
        print(f"\nTop-{TOP_N} fail-feature Jaccard vs reference s64:")
        for s in available:
            j = jaccard(sets[s], sets[64])
            shared = sets[s] & sets[64]
            print(f"  s{s:>3}: Jaccard={j:.3f} (shared {len(shared)}/{TOP_N})")

    # ---- Feature persistence (feature in top-N at >= 3 steps) ----
    feat_counts = {}
    for s in available:
        for fid in sets[s]:
            feat_counts[fid] = feat_counts.get(fid, []) + [s]
    persistent = {fid: stps for fid, stps in feat_counts.items() if len(stps) >= 3}
    print(f"\nPersistent fail features (top-{TOP_N} at >= 3 steps):")
    for fid in sorted(persistent, key=lambda f: -len(persistent[f])):
        stps = persistent[fid]
        # Get enrichment trajectory for this feature
        enrs = []
        for s in available:
            top_rows = data[s]["top_fail_features"]
            row = next((r for r in top_rows if r["feature_id"] == fid), None)
            enrs.append(row["enrichment"] if row else None)
        enr_str = " ".join(
            f"s{s:>3}={e:+.2f}" if e is not None else f"s{s:>3}=  -  "
            for s, e in zip(available, enrs)
        )
        print(f"  f{fid:>5}: {len(stps)} steps  {enr_str}")

    # ---- Persist summary ----
    out_path = f"/tmp/traj_{args.model}/summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "trajectory": traj,
            "jaccard_consecutive": [
                {"a": a, "b": b, "jaccard": jaccard(sets[a], sets[b])}
                for a, b in zip(available, available[1:])
            ],
            "persistent_features": [
                {"feature_id": fid, "steps_in_top": persistent[fid]}
                for fid in persistent
            ],
        }, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
