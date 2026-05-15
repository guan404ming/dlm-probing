"""Quick go/no-go: logit-lens of SAE decoder vectors through LLaDA W_U.

For each target SAE feature f, project W_dec[:, f] through the model's
output unembedding W_U and read the top-K promoted vocab tokens. If
tokens are semantically coherent (code/math/assertion tokens for f15601),
the vocab-attribution angle of the EMNLP pivot is viable.

Usage:
  .venv/bin/modal run src/applications/sae/modal_logitlens.py
"""

import modal

app = modal.App("sae-logitlens")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "transformers==4.52.2",
        "huggingface_hub",
    )
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)
HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)

LLADA_NAME = "GSAI-ML/LLaDA-8B-Instruct"
SAE_REPO = "AwesomeInterpretability/llada-mask-topk-sae"
SAE_TRAINER = 2
SAE_LAYER = 26
TOPK_TOKENS = 40
FEATURES = [15601, 3892, 11265, 13087, 2741, 8020, 189, 8825, 3320]


@app.function(
    image=image, gpu="A10G", timeout=1200,
    volumes={"/results": RESULTS_VOL, "/hf-cache": HF_CACHE_VOL},
)
def run():
    import json
    import os
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModel

    os.environ["HF_HOME"] = "/hf-cache"

    sae_path = f"resid_post_layer_{SAE_LAYER}/trainer_{SAE_TRAINER}"
    ae_local = hf_hub_download(repo_id=SAE_REPO, filename=f"{sae_path}/ae.pt", cache_dir="/hf-cache")
    state = torch.load(ae_local, map_location="cpu", weights_only=True)
    W_dec = state["decoder.weight"].float()
    print(f"SAE W_dec shape: {tuple(W_dec.shape)}")

    tok = AutoTokenizer.from_pretrained(LLADA_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLADA_NAME, torch_dtype=torch.float16, trust_remote_code=True,
    ).eval()

    W_U = None
    for name in ["lm_head", "embed_out", "ff_out"]:
        mod = getattr(model, name, None)
        if mod is not None and hasattr(mod, "weight"):
            W_U = mod.weight
            print(f"Using model.{name}.weight, shape={tuple(W_U.shape)}")
            break
    if W_U is None:
        for attr_path in [("model", "transformer", "ff_out"),
                          ("model", "transformer", "wte"),
                          ("transformer", "ff_out")]:
            cur = model
            ok = True
            for a in attr_path:
                if hasattr(cur, a):
                    cur = getattr(cur, a)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "weight"):
                W_U = cur.weight
                print(f"Using model.{'.'.join(attr_path)}.weight, shape={tuple(W_U.shape)}")
                break
    if W_U is None:
        named = [n for n, _ in model.named_parameters() if "embed" in n.lower() or "head" in n.lower() or "ff_out" in n.lower()]
        print("Could not find unembedding. Candidates:")
        for n in named:
            print(f"  {n}")
        raise RuntimeError("No unembedding found")

    W_U = W_U.float().detach()

    out = {"sae_layer": SAE_LAYER, "trainer": SAE_TRAINER, "features": {}}
    for f in FEATURES:
        v = W_dec[:, f]
        scores = W_U @ v
        topk = scores.topk(TOPK_TOKENS)
        botk = scores.topk(15, largest=False)
        top_tokens = [tok.decode([i]) for i in topk.indices.tolist()]
        bot_tokens = [tok.decode([i]) for i in botk.indices.tolist()]
        print(f"\n=== f{f} ===")
        print(f"  top {TOPK_TOKENS}:")
        for t, s in zip(top_tokens, topk.values.tolist()):
            print(f"    {s:+.3f}  {repr(t)}")
        print(f"  bottom 15:")
        for t, s in zip(bot_tokens, botk.values.tolist()):
            print(f"    {s:+.3f}  {repr(t)}")
        out["features"][str(f)] = {
            "top_tokens": top_tokens,
            "top_scores": topk.values.tolist(),
            "bot_tokens": bot_tokens,
            "bot_scores": botk.values.tolist(),
        }

    out_path = "/results/mbpp_llada/sae_logitlens.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    RESULTS_VOL.commit()
    print(f"\nSaved {out_path}")
    return json.dumps({"ok": True, "n_features": len(FEATURES)})


@app.local_entrypoint()
def main():
    print(run.remote())
