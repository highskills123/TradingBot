"""Trend-following signal: EMA crossover filtered by ATR volatility.

Entry logic
-----------
* **Long**:  fast EMA crosses above slow EMA **and** price is above the trend
  EMA **and** ATR is above its own 20-period average (i.e. there *is* momentum).
* **Short**: fast EMA crosses below slow EMA **and** price is below the trend
  EMA **and** ATR is above its own 20-period average.

Returns
-------
``"long"`` | ``"short"`` | ``None``
"""
from __future__ import annotations

import pandas as pd


def trend_signal(df: pd.DataFrame, cfg: dict) -> str | None:
    """Generate a trend-following signal from the latest two candles.

    Args:
        df: Feature-enriched OHLCV DataFrame (must contain EMA and ATR columns).
        cfg: The ``features`` section of the bot configuration.

    Returns:
        ``"long"``, ``"short"``, or ``None`` when no signal fires.
    """
    if len(df) < 2:
        return None

    fast = cfg.get("ema_fast", 9)
    slow = cfg.get("ema_slow", 21)
    trend = cfg.get("ema_trend", 50)
    atr_p = cfg.get("atr_period", 14)

    fast_col = f"ema_{fast}"
    slow_col = f"ema_{slow}"
    trend_col = f"ema_{trend}"
    atr_col = f"atr_{atr_p}"

    required = [fast_col, slow_col, trend_col, atr_col]
    if not all(c in df.columns for c in required):
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # Volatility filter: ATR must be above its recent average
    atr_avg = df[atr_col].rolling(20).mean().iloc[-1]
    high_vol = curr[atr_col] > atr_avg

    # Cross detection
    crossed_up = prev[fast_col] <= prev[slow_col] and curr[fast_col] > curr[slow_col]
    crossed_down = prev[fast_col] >= prev[slow_col] and curr[fast_col] < curr[slow_col]

    if crossed_up and curr["close"] > curr[trend_col] and high_vol:
        return "long"
    if crossed_down and curr["close"] < curr[trend_col] and high_vol:
        return "short"

    return None
