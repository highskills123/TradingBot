"""Breakout signal: N-period high/low with volume confirmation.

Entry logic
-----------
* **Long**:  current close sets a new N-period high **and** volume is above
  its N-period average (confirms genuine breakout).
* **Short**: current close sets a new N-period low **and** volume is above
  its N-period average.

Returns
-------
``"long"`` | ``"short"`` | ``None``
"""
from __future__ import annotations

import pandas as pd


def breakout_signal(df: pd.DataFrame, cfg: dict) -> str | None:
    """Generate a breakout signal from the latest candle.

    Args:
        df: Feature-enriched OHLCV DataFrame (must contain rolling high/low
            and volume average columns).
        cfg: The ``features`` section of the bot configuration.

    Returns:
        ``"long"``, ``"short"``, or ``None`` when no signal fires.
    """
    if len(df) < 2:
        return None

    bp = cfg.get("breakout_periods", 20)
    vol_p = cfg.get("volume_avg_periods", 20)

    high_col = f"high_{bp}"
    vol_avg_col = f"vol_avg_{vol_p}"

    # Also need the rolling low (computed on the fly if not pre-computed)
    if high_col not in df.columns:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # Rolling low over the lookback window (excluding current candle)
    low_n = df["close"].rolling(bp).min().iloc[-1]

    volume_confirmed = (
        vol_avg_col in df.columns and curr["volume"] > curr[vol_avg_col]
    )

    # Breakout long: close crosses above the previous N-period high
    if curr["close"] >= prev[high_col] and volume_confirmed:
        return "long"

    # Breakout short: close drops below the N-period low
    if curr["close"] <= low_n and volume_confirmed:
        return "short"

    return None
