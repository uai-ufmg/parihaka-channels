import yaml
from typing import Any, Dict, List


REQUIRED_KEYS: List[str] = ["model", "dataset", "solver"]


def _validate_config_shape(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate config contains required top-level keys and apply minimal defaults.

    This loader is strict: it expects a modern config with `solver` (no aliases).
    """
    if cfg is None:
        raise ValueError("Configuration is empty or could not be parsed")

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required top-level config keys: {missing}")

    # Minimal defaults
    cfg.setdefault("scheduler", {})
    cfg.setdefault("logging", {})

    solver = cfg["solver"]
    solver.setdefault("name", "adamw")
    solver.setdefault("learning_rate", 1e-3)
    solver.setdefault("weight_decay", 0.0)
    solver.setdefault("max_epochs", 100)

    return cfg


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return _validate_config_shape(cfg)