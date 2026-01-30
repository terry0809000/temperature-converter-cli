import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class Config:
    data: Dict[str, Any]
    preprocess: Dict[str, Any]
    modeling: Dict[str, Any]
    training: Dict[str, Any]
    evaluation: Dict[str, Any]
    output: Dict[str, Any]


def load_config(path: str) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config(
        data=raw.get("data", {}),
        preprocess=raw.get("preprocess", {}),
        modeling=raw.get("modeling", {}),
        training=raw.get("training", {}),
        evaluation=raw.get("evaluation", {}),
        output=raw.get("output", {}),
    )
