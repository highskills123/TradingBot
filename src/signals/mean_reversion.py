"""Mean-reversion signal: deviation from VWAP/EMA filtered by RSI.

Entry logic
-----------
* **Long**:  price is below VWAP by more than ``threshold × ATR`` **and**
  RSI is oversold (< 35).
* **Short**: price is above VWAP by more than ``threshold × ATR`` **and**
  RSI is overbought (> 65).

Returns
-------
``"long"`` | ``"short"`` | ``None``
"""
from __future__ import annotations

import pandas as pd


_DEVIATION_THRESHOLD = 0.5   # multiples of ATR away from VWAP to trigger
_RSI_OVERSOLD = 35
_RSI_OVERBOUGHT = 65


def mean_reversion_signal(df: pd.DataFrame, cfg: dict) -> str | None:
    """Generate a mean-reversion signal from the latest candle.

    Args:
        df: Feature-enriched OHLCV DataFrame (must contain VWAP, ATR and RSI).
        cfg: The ``features`` section of the bot configuration.

    Returns:
        ``"long"``, ``"short"``, or ``None`` when no signal fires.
    """
    if len(df) < 1:
        return None

    rsi_p = cfg.get("rsi_period", 14)
    atr_p = cfg.get("atr_period", 14)
    vwap_p = cfg.get("vwap_period", 20)

    rsi_col = f"rsi_{rsi_p}"
    atr_col = f"atr_{atr_p}"
    vwap_col = f"vwap_{vwap_p}"

    required = [rsi_col, atr_col, vwap_col]
    if not all(c in df.columns for c in required):
        return None

    curr = df.iloc[-1]
    deviation = curr["close"] - curr[vwap_col]
    threshold = _DEVIATION_THRESHOLD * curr[atr_col]

    if deviation < -threshold and curr[rsi_col] < _RSI_OVERSOLD:
        return "long"
    if deviation > threshold and curr[rsi_col] > _RSI_OVERBOUGHT:
        return "short"

    return None
