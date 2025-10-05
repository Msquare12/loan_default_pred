import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add repo root to PYTHONPATH

import argparse
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def main(config='configs/default.yaml'):
    art = load('models/latest/model.joblib')
    pipe = art['pipeline']; th = art['threshold']
    import pandas as pd
    X_test = pd.read_parquet('data/processed/X_test.parquet')
    y_test = pd.read_parquet('data/processed/y_test.parquet')['y']
    probs = pipe.predict_proba(X_test)[:,1]
    from src.eval.metrics import compute_metrics
    m = compute_metrics(y_test, probs, th)
    print("Test metrics:", m)

if __name__ == "__main__":
    main()
