from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from .schema import infer_schema
from typing import Union

@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

def load_raw(path: Union[str, Path]) -> pd.DataFrame:
    # return pd.read_csv(path)
    return pd.read_csv(path, low_memory=False)

def split_dataset(df: pd.DataFrame, target_col: str, test_size: float, val_size: float, seed: int, stratify: bool=True) -> Splits:
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if stratify else None
    )

    # val from train portion
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train if stratify else None
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def _coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns that are mostly numeric strings into real numerics."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            # Try numeric conversion but don't force for non-numeric
            converted = pd.to_numeric(df[c], errors="coerce")
            # If we successfully converted most values, keep it
            if converted.notna().mean() >= 0.9:
                df[c] = converted
    return df


def save_splits(splits: Splits, outdir: Union[str, Path]) -> None:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Coerce mixed-type columns to numeric where appropriate
    Xtr = _coerce_numeric_like(splits.X_train)
    Xva = _coerce_numeric_like(splits.X_val)
    Xte = _coerce_numeric_like(splits.X_test)

    # Parquet write
    Xtr.to_parquet(outdir / "X_train.parquet", index=False)
    Xva.to_parquet(outdir / "X_val.parquet", index=False)
    Xte.to_parquet(outdir / "X_test.parquet", index=False)

    splits.y_train.to_frame("y").to_parquet(outdir / "y_train.parquet", index=False)
    splits.y_val.to_frame("y").to_parquet(outdir / "y_val.parquet", index=False)
    splits.y_test.to_frame("y").to_parquet(outdir / "y_test.parquet", index=False)
