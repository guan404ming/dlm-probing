"""Find caption start y for both columns of the fig2 + selective-gen row."""
import fitz
import re

doc = fitz.open("poster.pdf")
page = doc[0]

dashed_y = None
sv_y = None
for blk in page.get_text("dict")["blocks"]:
    if blk.get("type") != 0: continue
    bbox = blk["bbox"]
    text = " ".join(s["text"] for L in blk["lines"] for s in L["spans"])
    text = re.sub(r"\s+", " ", text)
    if dashed_y is None and "Dashed grey" in text:
        dashed_y = bbox[1]
        print(f"Best AUC caption starts at y={bbox[1]:.1f}")
    if sv_y is None and ("denoising steps skipped" in text or "Sv :" in text or "Sv: %" in text or text.lstrip().startswith("Sv")):
        sv_y = bbox[1]
        print(f"Selective Gen caption starts at y={bbox[1]:.1f}")

# also measure table top vs fig top
print()
print("Looking for top of body content in each column:")
# fig: 0.90 label y top, table: "JSON GSM8K MBPP ARC" header y top
for blk in page.get_text("dict")["blocks"]:
    if blk.get("type") != 0: continue
    bbox = blk["bbox"]
    text = " ".join(s["text"] for L in blk["lines"] for s in L["spans"])
    text = re.sub(r"\s+", " ", text)
    if "0.90" in text and bbox[1] < 2100:
        print(f"  fig 0.90 ytick at y={bbox[1]:.1f}")
    if text.strip().startswith("JSON GSM8K MBPP ARC") and bbox[1] < 2200:
        print(f"  table header at y={bbox[1]:.1f}")
