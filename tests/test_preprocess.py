import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.dataset import split_dataset
from src.data.schema import infer_schema
from src.features.preprocess import build_preprocessor, NumericCoercer


def test_preprocess_smoke():
    # Load with better type inference
    df = pd.read_csv("data/raw/Dataset.csv", low_memory=False)

    # Make splits
    splits = split_dataset(
        df, target_col="Default", test_size=0.2, val_size=0.1, seed=42, stratify=True
    )

    # Work on a small subset to avoid OOM in CI/local
    X_small = splits.X_train.sample(n=min(2000, len(splits.X_train)), random_state=42)

    # Drop ID only if present
    drop_cols = [c for c in ["ID"] if c in X_small.columns]
    X_small = X_small.drop(columns=drop_cols)

    # Infer schema from the actual features going into the pipeline
    schema = infer_schema(X_small)

    # Build: numeric coercion -> preprocessing (impute/scale/OHE with string cast inside)
    clean = NumericCoercer()  # robust: figures columns out from the df it receives
    pre = build_preprocessor(schema)
    pipe = Pipeline([("clean", clean), ("pre", pre)])

    # Fit-transform should succeed and preserve row count on the sampled data
    X = pipe.fit_transform(X_small)
    assert X.shape[0] == len(X_small)
