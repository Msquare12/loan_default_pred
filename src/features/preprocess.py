from __future__ import annotations
from typing import Dict, List
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


class NumericCoercer(BaseEstimator, TransformerMixin):
    """Coerces numeric-like object columns to numeric.
    Decides per-column based on >=90% successful to_numeric conversion on fit.
    """
    def __init__(self, columns: List[str] = None, min_success: float = 0.9):
        self.columns = columns
        self.min_success = min_success
        self.coerce_cols_ = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        cols = self.columns or X.columns.tolist()
        cols = [c for c in cols if c in X.columns]
        self.coerce_cols_ = []
        for c in cols:
            s = pd.to_numeric(X[c], errors='coerce')
            success = float(pd.notna(s).mean())
            if success >= self.min_success:
                self.coerce_cols_.append(c)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.coerce_cols_:
            if c in X.columns: 
                X[c] = pd.to_numeric(X[c], errors='coerce')
        return X

class ToStringTransformer(BaseEstimator, TransformerMixin):
    """Cast all columns to string; safe for OHE that requires uniform dtype."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        import pandas as pd
        X = pd.DataFrame(X).copy()
        return X.astype(str)


def build_preprocessor(schema: Dict[str, List[str]], scale_numeric: bool=True, numeric_imputer: str="median"):
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputer))]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))
    # cat_steps = [
    #     ("imputer", SimpleImputer(strategy="most_frequent")),
    #     ("to_str", FunctionTransformer(lambda X: X.astype(str))),
    #     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    # ]
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # ("to_str", FunctionTransformer(lambda X: X.astype(str))),
        ("to_str", ToStringTransformer()),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)) # sparse to avoid huge dense matrices
    ]

    from sklearn.pipeline import Pipeline
    num_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline(cat_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, schema.get("numeric", [])),
            ("cat", cat_pipe, schema.get("categorical", [])),
        ]
    )
    return preprocessor
