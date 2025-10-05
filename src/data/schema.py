from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd

def infer_schema(df: pd.DataFrame, exclude: List[str] | None = None) -> Dict[str, List[str]]:
    exclude = set(exclude or [])
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]
    return {"numeric": numeric_cols, "categorical": cat_cols}
