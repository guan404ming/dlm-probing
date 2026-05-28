"""Visual + text verification of poster.pdf.

Checks:
  1. Page-edge clipping: text within 2cm of bottom edge = likely overflow.
  2. Required strings: every block heading must be findable.
  3. Logo regions: render zoomed crops of top-left, top-right, bottom-left
     into PNGs so we can eyeball seal-in-circle alignment.
  4. Empty-region scan: 12x6 grid, count empty cells and tallest empty band.
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).parent
PDF = HERE / "poster.pdf"
CROP_DIR = HERE / "_verify_crops"
CROP_DIR.mkdir(exist_ok=True)

BG = np.array([0xFB, 0xFA, 0xF7], dtype=np.int16)
RED = np.array([0x9F, 0x1D, 0x35], dtype=np.int16)

def near(arr, target, tol):
    return (np.abs(arr.astype(np.int16) - target).sum(axis=-1) < tol)

doc = fitz.open(PDF)
page = doc[0]
w_pt, h_pt = page.rect.width, page.rect.height

# 1. Edge-clipping check using text blocks
EDGE_PT = 80  # ~2.8cm
text = page.get_text("dict")
near_bottom = []
for block in text.get("blocks", []):
    if block.get("type") != 0:
        continue
    bbox = block["bbox"]
    if bbox[3] > h_pt - EDGE_PT:
        snippet = " ".join(s["text"] for line in block["lines"] for s in line["spans"])[:60]
        near_bottom.append((round(bbox[3], 1), snippet))

print("=" * 70)
print("CHECK 1 -- text within 2.8cm of bottom edge (overflow risk)")
if near_bottom:
    for y, snippet in near_bottom[:10]:
        print(f"  y={y:7.1f} pt (page h={h_pt:.0f}) : {snippet!r}")
else:
    print("  none")

# 2. Required strings
required = [
    "Motivation", "Method", "Setup", "Key Findings", "Why It Matters",
    "Takeaways", "Selective Generation", "Controls", "Conclusion",
    "Limitations", "Future Work",
    # last bullet of each bottom block
    "AR baseline", "softmax heuristics", "calibration", "larger scales",
]
all_text = page.get_text()
print()
print("=" * 70)
print("CHECK 2 -- required strings present")
missing = []
for s in required:
    found = s in all_text
    flag = "OK " if found else "MISS"
    print(f"  [{flag}] {s}")
    if not found:
        missing.append(s)

# 3. Render zoom crops of each logo corner
print()
print("=" * 70)
print("CHECK 3 -- rendering logo crops to _verify_crops/")
ZOOM = 1.2
pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
H, W, _ = img.shape

def save_crop(name, y0_pt, y1_pt, x0_pt, x1_pt):
    """Render a region of the page directly via pymupdf clip."""
    clip = fitz.Rect(x0_pt, y0_pt, x1_pt, y1_pt)
    pm = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), clip=clip, alpha=False)
    out = CROP_DIR / f"{name}.png"
    pm.save(out)
    return out, (pm.width, pm.height)

regions_pt = [
    ("01_titlebar_full", 0, h_pt * 0.13, 0, w_pt),
    ("02_ntu_corner",    0, h_pt * 0.11, 0, w_pt * 0.18),
    ("03_cmu_corner",    0, h_pt * 0.11, w_pt * 0.82, w_pt),
    ("04_footer_full",   h_pt * 0.94, h_pt, 0, w_pt),
    ("05_acl_modal",     h_pt * 0.94, h_pt, 0, w_pt * 0.40),
]
for name, y0, y1, x0, x1 in regions_pt:
    out, (pw, ph) = save_crop(name, y0, y1, x0, x1)
    print(f"  saved {out.name}  ({pw}x{ph} px)")

# 4. Grid scan
print()
print("=" * 70)
print("CHECK 4 -- 12x6 grid empty-cell scan")
ROWS, COLS = 12, 6
empty_cells = []
for r in range(ROWS):
    for c in range(COLS):
        y0, y1 = H * r // ROWS, H * (r + 1) // ROWS
        x0, x1 = W * c // COLS, W * (c + 1) // COLS
        cell = img[y0:y1, x0:x1]
        if near(cell, BG, 30).mean() > 0.95:
            empty_cells.append((r, c))
print(f"  empty cells: {len(empty_cells)} / {ROWS*COLS}")
if empty_cells:
    print(f"  {empty_cells}")

# Summary
print()
print("=" * 70)
issues = []
if near_bottom:
    issues.append(f"{len(near_bottom)} text blocks near bottom edge")
if missing:
    issues.append(f"missing strings: {missing}")
if len(empty_cells) > 4:
    issues.append(f"{len(empty_cells)} empty grid cells (>4)")
if issues:
    print("ISSUES: " + " ; ".join(issues))
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
