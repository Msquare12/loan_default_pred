import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np, pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import precision_recall_fscore_support, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

OUT = Path("reports"); OUT.mkdir(parents=True, exist_ok=True)
art = load("models/latest/model.joblib")
pipe, th0 = art["pipeline"], float(art["threshold"])

X = pd.read_parquet("data/processed/X_test.parquet")
y = pd.read_parquet("data/processed/y_test.parquet")["y"].values
probs = pipe.predict_proba(X)[:,1]

# sweep thresholds
ths = np.linspace(0.01, 0.99, 99)
rows = []
for t in ths:
    pred = (probs >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
df = pd.DataFrame(rows)
df.to_csv(OUT / "threshold_sweep.csv", index=False)

# curves
fpr, tpr, _ = roc_curve(y, probs)
prec, rec, _ = precision_recall_curve(y, probs)
roc_auc = auc(fpr, tpr)
pr_auc = auc(rec, prec)

plt.figure(figsize=(5,4)); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC (AUC={roc_auc:.3f})")
plt.tight_layout(); plt.savefig(OUT / "roc_curve.png", dpi=160); plt.close()

plt.figure(figsize=(5,4)); plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve (AUC={pr_auc:.3f})")
plt.tight_layout(); plt.savefig(OUT / "pr_curve.png", dpi=160); plt.close()

# precision/recall/f1 vs threshold plots
for metric in ["precision","recall","f1"]:
    plt.figure(figsize=(5,4)); plt.plot(df["threshold"], df[metric])
    plt.xlabel("Threshold"); plt.ylabel(metric.capitalize()); plt.title(f"{metric} vs Threshold")
    plt.tight_layout(); plt.savefig(OUT / f"{metric}_vs_threshold.png", dpi=160); plt.close()

print("Saved: reports/threshold_sweep.csv + curves & metric plots")
