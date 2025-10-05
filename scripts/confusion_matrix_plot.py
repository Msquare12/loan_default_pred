import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np, pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

OUT = Path("reports"); OUT.mkdir(parents=True, exist_ok=True)
art = load("models/latest/model.joblib")
pipe, th = art["pipeline"], float(art["threshold"])

X = pd.read_parquet("data/processed/X_test.parquet")
y = pd.read_parquet("data/processed/y_test.parquet")["y"].values
probs = pipe.predict_proba(X)[:,1]
pred = (probs >= th).astype(int)

cm = confusion_matrix(y, pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default","Default"])
disp.plot(values_format="d"); plt.title(f"Confusion Matrix @ threshold={th:.2f}")
plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=160); plt.close()
print("Saved: reports/confusion_matrix.png")
