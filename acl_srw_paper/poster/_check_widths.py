"""Measure leftmost/rightmost red pixel of every horizontal title-bar band."""
import fitz
import numpy as np

doc = fitz.open("poster.pdf")
page = doc[0]
pm = page.get_pixmap(dpi=72)
img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)
RED = np.array([0x9F, 0x1D, 0x35], dtype=np.int16)
mask = (np.abs(img.astype(np.int16) - RED).sum(axis=-1) < 70)
H, W, _ = img.shape

# A row is part of a title bar if it has > 20% red pixels
row_red_frac = mask.mean(axis=1)
is_bar_row = row_red_frac > 0.05

# Find contiguous bands
bands = []
in_band = False
start = None
for y in range(H):
    if is_bar_row[y] and not in_band:
        in_band = True; start = y
    elif not is_bar_row[y] and in_band:
        in_band = False; bands.append((start, y - 1))
if in_band:
    bands.append((start, H - 1))

# Filter to "real" title bars (height 30-200 px)
title_bands = [(a, b) for a, b in bands if 30 <= b - a + 1 <= 200]

print(f"page width: {W} px")
print(f"Found {len(title_bands)} title-bar bands\n")

# For each title bar, find groups of contiguous red columns at mid-row
# but bridge gaps < 100 px (the white text inside)
print(f"{'band y':<12} {'h':<3} {'col 1 left-right (margin)':<30}  {'col 2':<25}  {'col 3':<30}")
for a, b in title_bands:
    mid = (a + b) // 2
    # union of red across all rows in band
    band_red = mask[a:b+1].any(axis=0)
    # bridge gaps < 100 px
    bridged = band_red.copy()
    last_red = -1000
    for x in range(W):
        if band_red[x]:
            if 0 < x - last_red < 100:
                bridged[last_red:x] = True
            last_red = x
    # find contiguous runs in bridged
    diff = np.diff(bridged.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    runs = list(zip(starts.tolist(), ends.tolist()))
    runs_long = [r for r in runs if r[1] - r[0] > 60]
    parts = []
    for s, e in runs_long:
        parts.append(f"{s}-{e-1} (L:{s} R:{W-1-(e-1)})")
    print(f"y={a}-{b:<6} {b-a+1:<3} " + "  ".join(parts))
