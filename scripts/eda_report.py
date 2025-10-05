#!/usr/bin/env python
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- plotting utils (pure matplotlib; no seaborn needed) ---
def save_bar(values, labels, title, out_path, color=None, horizontal=False):
    plt.figure(figsize=(7,4))
    if horizontal:
        y = np.arange(len(values))
        plt.barh(y, values, color=color)
        plt.yticks(y, labels)
        plt.xlabel("Value")
    else:
        x = np.arange(len(values))
        plt.bar(x, values, color=color)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_hist(series, title, out_path, bins=40):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return
    plt.figure(figsize=(7,4))
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_corr_heatmap(df_num, out_path, title="Correlation Heatmap"):
    if df_num.shape[1] < 2:
        return
    corr = df_num.corr(numeric_only=True)
    plt.figure(figsize=(7,6))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # show at most first 20 labels to keep legible
    cols = list(corr.columns[:20])
    ticks = np.arange(len(cols))
    plt.xticks(ticks, cols, rotation=90)
    plt.yticks(ticks, cols)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/raw/Dataset.csv")
    ap.add_argument("--outdir", type=str, default="reports")
    ap.add_argument("--target", type=str, default="Default")
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, low_memory=False)
    rows, cols = df.shape

    # ---- target distribution / imbalance
    assert args.target in df.columns, f"Target '{args.target}' not found in columns"
    vc = df[args.target].value_counts(dropna=False)
    vc_pct = (vc / vc.sum()).round(4)

    # Plot target distribution (bar)
    save_bar(
        values=vc.values.tolist(),
        labels=[str(k) for k in vc.index.tolist()],
        title="Target Distribution (counts)",
        out_path=outdir / "target_distribution.png",
        color="#fc8d62"
    )

    # ---- missingness
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss) > 0:
        top = miss.head(15)
        save_bar(
            values=top.values.tolist(),
            labels=top.index.tolist(),
            title="Missing Value Share (Top 15)",
            out_path=outdir / "missing_top15.png",
            color="#66c2a5",
            horizontal=True
        )

    # ---- correlations (numeric only)
    df_num = df.select_dtypes(include=["number"])
    if df_num.shape[1] >= 2:
        save_corr_heatmap(df_num, outdir / "correlation_heatmap.png")

    # ---- a couple of useful distributions (fallback to first numerics if not present)
    preferred = ["Client_Income", "Credit_Amount"]
    numerics = list(df_num.columns)

    for col in preferred:
        if col in df.columns:
            save_hist(df[col], f"{col} Distribution", outdir / f"{col.lower()}_distribution.png")

    # If any preferred missing, fill with first other numerics
    extras = [c for c in numerics if c not in preferred][:2]
    for col in extras:
        save_hist(df[col], f"{col} Distribution", outdir / f"{col.lower()}_distribution.png")

    # ---- write a compact JSON summary you can paste into README if you want exact numbers
    default_rate = float(vc_pct.get(1, vc_pct.get("1", 0.0)))
    summary = {
        "shape": {"rows": int(rows), "cols": int(cols)},
        "target_counts": {str(k): int(v) for k, v in vc.items()},
        "target_share": {str(k): float(v) for k, v in vc_pct.items()},
        "default_rate": default_rate
    }
    with open(outdir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("EDA summary written to:", outdir / "eda_summary.json")
    print("Images saved to:", outdir)

if __name__ == "__main__":
    main()
