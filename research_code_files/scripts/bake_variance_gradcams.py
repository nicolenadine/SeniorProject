#!/usr/bin/env python3
"""
Randomly sample 16 images from the Data/ directory and
generate full-image variance heatmap overlays with bounding
boxes around the quadrant with highest variance.
"""

import cv2, os, numpy as np, pandas as pd, random, tqdm
from pathlib import Path

TILE = 16  # granularity of variance grid
ALPHA = 120  # overlay transparency
NUM_SAMPLES = 16

DATA_DIR = Path("/Users/nicoleparziale/Desktop/SeniorProject/data_and_plots/Data")
OUT_DIR = Path("Application/app/assets/random_gradcam")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Walk and collect image paths ---
all_pngs = list(DATA_DIR.rglob("*.png"))
if len(all_pngs) < NUM_SAMPLES:
    raise ValueError(f"Only found {len(all_pngs)} PNGs in {DATA_DIR}")

# 1. Group files by family
from collections import defaultdict

grouped = defaultdict(list)

for p in all_pngs:
    parts = p.parts
    try:
        # This assumes your folder pattern is like:
        # Data/benign/...  or  Data/malware/<Family>/...
        if "benign" in parts:
            family = "benign"
        elif "malware" in parts:
            malware_index = parts.index("malware")
            family = parts[malware_index + 1]  # folder right after 'malware'
        else:
            print(f"[SKIP] Unrecognized structure: {p}")
            continue
    except Exception as e:
        print(f"[SKIP] {p} — {e}")
        continue

    grouped[family].append(p)


# 2. Sample up to 16 per family
samples = []
for fam, files in grouped.items():
    chosen = random.sample(files, min(16, len(files)))
    samples.extend(chosen)


records = []

for path in tqdm.tqdm(samples):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[SKIP] Could not read {path}")
        continue

    h, w = img.shape
    base_name = path.name

    # --- 2. Compute local variance heatmap ---
    var_grid = img.reshape(h//TILE, TILE, w//TILE, TILE).var(axis=(1,3)).astype('float32')
    var_up = cv2.resize(var_grid, (w, h), interpolation=cv2.INTER_LINEAR)
    var_norm = cv2.normalize(var_up, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(var_norm.astype('uint8'), cv2.COLORMAP_JET)
    heat_rgba = cv2.cvtColor(heat, cv2.COLOR_BGR2BGRA)
    heat_rgba[..., 3] = ALPHA

    # --- 3. Determine highest-variance quadrant and draw rectangle ---
    q_vars = [var_grid[:8,:8].mean(), var_grid[:8,8:].mean(),
              var_grid[8:,:8].mean(), var_grid[8:,8:].mean()]
    q_idx = int(np.argmax(q_vars))
    y0,x0 = [(0,0), (0,128), (128,0), (128,128)][q_idx]
    cv2.rectangle(heat_rgba, (x0,y0), (x0+128, y0+128),
                  color=(255,255,255,255), thickness=2)

    # --- 4. Save base + overlay ---
    cv2.imwrite(str(OUT_DIR / base_name), img)
    cv2.imwrite(str(OUT_DIR / f"{base_name}_var.png"), heat_rgba)

    # --- 5. Save metadata row ---
    # Recompute family based on path just in case
    parts = path.parts
    if parts[1].lower() == "benign":
        family = "benign"
    else:
        family = parts[2]  # e.g. Expiro.BK

    records.append(dict(
        file=base_name,
        full_path=str(path),
        family=family,
        hi_seg=q_idx
    ))

# --- 6. Save metadata ---
pd.DataFrame(records).to_parquet(OUT_DIR / "random_meta.parquet", index=False)
print("✓ Done.")
