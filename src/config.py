"""Configuration loader — single source of truth for all settings."""

import os
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config, with env-var overrides for deployment flexibility."""
    path = Path(path) if path else _CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Allow env overrides for deployment
    cfg["dask"]["n_workers"] = int(os.getenv("DASK_N_WORKERS", cfg["dask"]["n_workers"]))
    cfg["dask"]["memory_limit"] = os.getenv("DASK_MEMORY_LIMIT", cfg["dask"]["memory_limit"])
    cfg["api"]["host"] = os.getenv("API_HOST", cfg["api"]["host"])
    cfg["api"]["port"] = int(os.getenv("API_PORT", cfg["api"]["port"]))
    cfg["dashboard"]["api_url"] = os.getenv("API_URL", cfg["dashboard"]["api_url"])

    # Resolve relative paths against project root
    for key in ("raw_data", "processed_data", "models", "reports", "logs"):
        p = cfg["paths"][key]
        if not os.path.isabs(p):
            cfg["paths"][key] = str(_PROJECT_ROOT / p)

    return cfg


# Singleton config
_cfg = None


def get_config() -> dict:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg
