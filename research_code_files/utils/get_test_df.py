# utils/get_test_df.py
import pandas as pd
from pathlib import Path

TEST_FILE = Path("/Users/nicoleparziale/Desktop/SeniorProject/data_and_plots"
                 "/results/full_image_model/data_splits/test_files.txt")


def load_test_df() -> pd.DataFrame:
    rows = []
    with open(TEST_FILE) as f:
        for ln in f:
            p = Path(ln.strip())      # e.g. Data/malware/Enterak.A/…
            parts = p.parts           # ('Data', 'malware', 'Enterak.A', 'file.png')
            cls   = parts[1].lower()  # 'benign' or 'malware'
            if cls == 'benign':
                label, family = 0, ''
            else:                     # malware
                label, family = 1, parts[2]   # 'Enterak.A', 'Expiro.BK', …
            rows.append(dict(path=str(p), label=label, family=family))
    return pd.DataFrame(rows)
