# scripts/bake_variance_overlays.py
import cv2, os, numpy as np, tqdm, pandas as pd
from pathlib import Path
from utils.get_test_df import load_test_df

OUT_DIR = Path("Application/app/assets/gradcam_var")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_test_df()
records = []

for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    img = cv2.imread(row.path, cv2.IMREAD_GRAYSCALE)   # assumes 256×256
    h, w = img.shape[:2]

    # raw pixel variance per quadrant
    seg_var = [img[:h//2,:w//2].var(),
               img[:h//2,w//2:].var(),
               img[h//2:,:w//2].var(),
               img[h//2:,w//2:].var()]
    seg_idx = int(np.argmax(seg_var))

    # transparent yellow mask over that quadrant
    overlay = np.zeros((h,w,4), np.uint8)
    y0,x0 = [(0,0), (0,w//2), (h//2,0), (h//2,w//2)][seg_idx]
    overlay[y0:y0+h//2, x0:x0+w//2] = (0,255,255,100)   # B,G,R,A

    base_name = os.path.basename(row.path)
    cv2.imwrite(str(OUT_DIR/base_name), img)
    cv2.imwrite(str(OUT_DIR/f"{base_name}_box{seg_idx}.png"), overlay)

    records.append(dict(file=base_name, label=row.label, sel_seg=seg_idx))

pd.DataFrame(records).to_parquet(OUT_DIR/"var_meta.parquet", index=False)
print("✓ Overlays saved to", OUT_DIR)
