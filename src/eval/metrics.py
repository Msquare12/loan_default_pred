from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

def compute_metrics(y_true, y_prob, threshold: float=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }

def find_best_threshold_pr_max_f1(y_true, y_prob):
    # sweep thresholds 0..1
    t = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, (y_prob >= th).astype(int), zero_division=0) for th in t]
    best_idx = int(np.argmax(f1s))
    return float(t[best_idx])

def find_best_threshold_cost(y_true, y_prob, cost_fn: float=5.0, cost_fp: float=1.0):
    t = np.linspace(0.01, 0.99, 99)
    def cost(th):
        y_pred = (y_prob >= th).astype(int)
        fn = ((y_true==1) & (y_pred==0)).sum()
        fp = ((y_true==0) & (y_pred==1)).sum()
        return cost_fn*fn + cost_fp*fp
    costs = [cost(th) for th in t]
    return float(t[int(np.argmin(costs))])
