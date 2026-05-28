"""Measure y-center of each footer element vs the footer red bar mid-line."""
import fitz
import re

doc = fitz.open("poster.pdf")
page = doc[0]
h = page.rect.height

# Footer red bar: from y=h-3.0cm to y=h
# 1 cm = 72/2.54 = 28.346 pt
footer_h_pt = 4.0 * 72 / 2.54
footer_top = h - footer_h_pt
footer_mid = h - footer_h_pt / 2
print(f"page h={h:.1f}, footer top={footer_top:.1f}, footer mid={footer_mid:.1f}, bottom={h:.1f}")

# Images in footer region
print()
print("Images in footer:")
for img_info in page.get_images(full=True):
    bbox = page.get_image_bbox(img_info)
    if bbox.y0 > footer_top - 50:
        ymid = (bbox.y0 + bbox.y1) / 2
        offset = ymid - footer_mid
        print(f"  xref={img_info[0]} bbox=({bbox.x0:.0f},{bbox.y0:.0f}) to ({bbox.x1:.0f},{bbox.y1:.0f}) ymid={ymid:.1f} offset_from_mid={offset:+.1f}")

# Text in footer
print()
print("Text in footer:")
for blk in page.get_text("dict")["blocks"]:
    if blk.get("type") != 0: continue
    bbox = blk["bbox"]
    if bbox[1] > footer_top - 50:
        text = " ".join(s["text"] for L in blk["lines"] for s in L["spans"])
        ymid = (bbox[1] + bbox[3]) / 2
        offset = ymid - footer_mid
        print(f"  bbox=({bbox[0]:.0f},{bbox[1]:.0f}) to ({bbox[2]:.0f},{bbox[3]:.0f}) ymid={ymid:.1f} offset={offset:+.1f}  text={text!r}")
