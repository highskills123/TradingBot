"""Shared fixtures for the test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(seed)
    close = np.cumprod(1 + rng.normal(0, 0.002, n)) * 30_000
    spread = close * 0.001
    high = close + rng.uniform(0, spread)
    low = close - rng.uniform(0, spread)
    open_ = close * (1 + rng.normal(0, 0.0005, n))
    volume = rng.uniform(1, 10, n) * 100

    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv()


@pytest.fixture
def features_cfg() -> dict:
    return {
        "ema_fast": 9,
        "ema_slow": 21,
        "ema_trend": 50,
        "rsi_period": 14,
        "atr_period": 14,
        "vwap_period": 20,
        "breakout_periods": 20,
        "volume_avg_periods": 20,
    }
