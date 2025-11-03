# data.py
from pathlib import Path
import pandas as pd

def _read_any(path):
    
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df

def load_merged(mat_path="student-mat.csv", por_path="student-por.csv"):
    mat = Path(mat_path); por = Path(por_path)
    for p in (mat, por):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")
    d1 = _read_any(mat)
    d2 = _read_any(por)
    return pd.concat([d1, d2], ignore_index=True)

def winsorize_g2(df):
    g2 = df["G2"]
    q1, q3 = g2.quantile(0.25), g2.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = df.copy()
    out["G2"] = out["G2"].clip(lower=lo, upper=hi)
    return out
