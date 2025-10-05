import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add repo root to PYTHONPATH
from src.features.preprocess import ToStringTransformer, NumericCoercer  # ensure classes are importable

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from joblib import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--outdir", type=str, default="models/latest")
    args = ap.parse_args()

    art = load("models/latest/model.joblib")
    pipe = art["pipeline"]

    # load train+val for explaining what the model saw
    X_train = pd.read_parquet("data/processed/X_train.parquet")
    X_val = pd.read_parquet("data/processed/X_val.parquet")
    X = pd.concat([X_train, X_val], axis=0)

    # align columns with what the pipeline expects
    expected_cols = pipe.named_steps["pre"].feature_names_in_
    for c in expected_cols:
        if c not in X.columns:
            X[c] = None
    X = X[expected_cols]

    if len(X) > args.sample:
        X = X.sample(n=args.sample, random_state=42)

    model = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]
    clean = pipe.named_steps.get("clean", None)

    X_proc_input = X.copy()
    if clean is not None:
        X_proc_input = clean.transform(X_proc_input)

    # IMPORTANT: use transform (do NOT refit here)
    X_proc = pre.transform(X_proc_input)

    name = type(model).__name__.lower()
    if "xgb" in name or "xgboost" in name:
        explainer = shap.TreeExplainer(model)
    elif "logisticregression" in name:
        explainer = shap.LinearExplainer(model, X_proc, feature_dependence="independent")
    else:
        explainer = shap.Explainer(model.predict_proba, X_proc)

    sv = explainer(X_proc, check_additivity=False)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # summary bar plot
    plt.figure()
    shap.plots.bar(sv, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_bar.png", dpi=160)
    plt.close()

    # recover feature names after preprocessing
    try:
        num_names = pre.transformers_[0][2] if pre.transformers_ else []
    except Exception:
        num_names = []
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cat_names = list(ohe.get_feature_names_out())
    except Exception:
        cat_names = []

    feat_names = (list(num_names) if num_names else []) + (cat_names if cat_names else [])

    values = sv.values if hasattr(sv, "values") else sv
    mean_abs = np.abs(values).mean(axis=0)
    pd.DataFrame(
        {"feature": feat_names[: len(mean_abs)], "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).to_csv(
        outdir / "feature_importance_shap.csv", index=False
    )

    print("Saved SHAP artifacts to", outdir)


if __name__ == "__main__":
    main()
