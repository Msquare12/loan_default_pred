from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class Config:
    raw: Dict[str, Any]

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
