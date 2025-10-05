from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.config import load_config
from src.utils.io import ensure_dir, save_artifact
from src.data.dataset import load_raw, split_dataset, save_splits
from src.data.schema import infer_schema
from src.features.preprocess import build_preprocessor, NumericCoercer
from src.eval.metrics import compute_metrics, find_best_threshold_pr_max_f1, find_best_threshold_cost


def make_model(name: str, cfg: Dict[str, Any], use_class_weights: bool):
    if name == "logistic_regression":
        params = cfg["model"]["logistic_regression"]
        class_weight = "balanced" if use_class_weights else None
        clf = LogisticRegression(**params, class_weight=class_weight)
        return clf
    elif name == "xgboost":
        params = cfg["model"]["xgboost"].copy()
        # scale_pos_weight can help with imbalance
        clf = XGBClassifier(**params, tree_method="hist", eval_metric="auc")
        return clf
    else:
        raise ValueError(f"Unknown model name: {name}")

def train(config_path: str):
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    raw = load_raw(cfg["data"]["raw_path"])
    target_col = cfg["data"]["target_col"]
    id_col = cfg["data"]["id_col"]
    # ensure numeric target
    raw[target_col] = raw[target_col].astype(int)

    splits = split_dataset(
        raw, target_col, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["seed"], stratify=cfg["data"]["stratify"]
    )
    save_splits(splits, cfg["data"]["processed_dir"])

    # infer schema from training features only
    feat_df = splits.X_train.drop(columns=[id_col]) if id_col in splits.X_train.columns else splits.X_train
    schema = infer_schema(feat_df, exclude=[id_col] if id_col in splits.X_train.columns else [])
    pre = build_preprocessor(schema, scale_numeric=cfg["preprocessing"]["scale_numeric"], numeric_imputer=cfg["preprocessing"]["numeric_imputer"])

    model = make_model(cfg["model"]["name"], cfg, use_class_weights=cfg["training"]["use_class_weights"])

    clean = NumericCoercer(columns=splits.X_train.columns.tolist())
    pipe = Pipeline([("clean", clean), ("pre", pre), ("clf", model)])

    # cross-validated training (on train only), then refit on train+val
    kf = StratifiedKFold(n_splits=cfg["training"]["cv_folds"], shuffle=True, random_state=cfg["seed"])

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": cfg["model"]["name"],
            "cv_folds": cfg["training"]["cv_folds"],
            "use_class_weights": cfg["training"]["use_class_weights"]
        })

        oof_probs = np.zeros(len(splits.y_train))
        for fold, (tr, va) in enumerate(kf.split(splits.X_train, splits.y_train)):
            pipe.fit(splits.X_train.iloc[tr], splits.y_train.iloc[tr])
            oof_probs[va] = pipe.predict_proba(splits.X_train.iloc[va])[:, 1]

        # Choose threshold
        if cfg["evaluation"]["threshold_strategy"] == "pr_max_f1":
            th = find_best_threshold_pr_max_f1(splits.y_train.values, oof_probs)
        elif cfg["evaluation"]["threshold_strategy"] == "cost_min":
            th = find_best_threshold_cost(
                splits.y_train.values,
                oof_probs,
                cfg["evaluation"]["costs"]["fn"],
                cfg["evaluation"]["costs"]["fp"],
            )
        else:
            th = 0.5

        train_metrics = compute_metrics(splits.y_train.values, oof_probs, th)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"cv_{k}", float(v))

        # Refit on train+val
        X_refit = pd.concat([splits.X_train, splits.X_val], axis=0)
        y_refit = pd.concat([splits.y_train, splits.y_val], axis=0)
        pipe.fit(X_refit, y_refit)

        # Evaluate on test
        test_probs = pipe.predict_proba(splits.X_test)[:, 1]
        test_metrics = compute_metrics(splits.y_test.values, test_probs, th)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", float(v))

        # Persist artifacts
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("models") / ts
        ensure_dir(out_dir)
        from joblib import dump
        dump({"pipeline": pipe, "threshold": th, "schema": schema}, out_dir / "model.joblib")
        # also update "latest"
        latest_dir = Path("models") / "latest"
        if latest_dir.exists():
            import shutil; shutil.rmtree(latest_dir)
        ensure_dir(latest_dir)
        dump({"pipeline": pipe, "threshold": th, "schema": schema}, latest_dir / "model.joblib")

        mlflow.log_artifact(out_dir / "model.joblib")
        mlflow.log_param("chosen_threshold", th)

        print("TRAIN DONE. Test metrics:", test_metrics)
