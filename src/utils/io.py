from typing import Union
from pathlib import Path
import joblib

def ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def save_artifact(obj, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)

def load_artifact(path: Union[str, Path]):
    return joblib.load(path)