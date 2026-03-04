"""Configuration loader for TradingBot."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Environment variables ``BOT_API_KEY`` and ``BOT_API_SECRET`` override the
    values stored in the YAML file so that secrets are never committed.

    Args:
        path: Path to the YAML configuration file.  Falls back to
            ``config.yaml`` next to the project root when *None*.

    Returns:
        Nested dictionary with the full configuration.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    # Allow secrets to come from the environment so they are never stored in
    # source control.
    exchange_cfg = cfg.setdefault("exchange", {})
    exchange_cfg["api_key"] = os.environ.get("BOT_API_KEY", exchange_cfg.get("api_key", ""))
    exchange_cfg["api_secret"] = os.environ.get("BOT_API_SECRET", exchange_cfg.get("api_secret", ""))

    return cfg
