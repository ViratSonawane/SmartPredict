"""Quick offline analysis for SmartPredict.
Run from the project root: python notebooks/quick_analysis.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/metropt_5min.csv", parse_dates=["timestamp"], index_col="timestamp")
print("Rows:", len(df))
print("Date range:", df.index.min(), "to", df.index.max())
print(df[["TP2","TP3","Oil_temperature","Motor_current"]].describe())
for col in ["TP2","Oil_temperature","Motor_current"]:
    ax = df[col].dropna().plot(figsize=(12,3), title=col)
    ax.set_xlabel("Timestamp")
    plt.tight_layout()
    plt.show()
