import argparse
from pathlib import Path
import pandas as pd
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset

try:
    from evidently.report import Report
except ImportError:
    from evidently.report.report import Report

try:
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    # older/newer variants sometimes move presets
    from evidently.presets import DataDriftPreset  # fallback



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="reports/drift_report.html")
    args = ap.parse_args()

    X_train = pd.read_parquet("data/processed/X_train.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet")["y"]
    train = X_train.copy(); train["Default"] = y_train

    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_test = pd.read_parquet("data/processed/y_test.parquet")["y"]
    test = X_test.copy(); test["Default"] = y_test

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train, current_data=test)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    print("Saved drift report to", out)

if __name__ == "__main__":
    main()
