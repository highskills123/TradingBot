"""Technical-indicator feature engineering.

All functions accept and return a ``pd.DataFrame`` so that they can be chained
easily.  Input DataFrames must have ``open``, ``high``, ``low``, ``close`` and
``volume`` columns (standard OHLCV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (used inside ATR)."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def add_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.DataFrame:
    """Add an EMA column ``ema_{period}`` to *df*."""
    df = df.copy()
    df[f"ema_{period}"] = _ema(df[col], period)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.DataFrame:
    """Add ``rsi_{period}`` column (0–100) to *df*."""
    df = df.copy()
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ``atr_{period}`` (Average True Range) column to *df*."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df[f"atr_{period}"] = _rma(tr, period)
    return df


def add_vwap(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add a rolling VWAP column ``vwap_{period}`` to *df*.

    Uses the typical price ``(H+L+C)/3`` weighted by volume over a rolling
    window, which works on any timeframe without session resets.
    """
    df = df.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    df[f"vwap_{period}"] = (
        tp_vol.rolling(period).sum() / df["volume"].rolling(period).sum()
    )
    return df


def add_volume_avg(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add a simple moving average of volume ``vol_avg_{period}``."""
    df = df.copy()
    df[f"vol_avg_{period}"] = df["volume"].rolling(period).mean()
    return df


# ---------------------------------------------------------------------------
# Composite feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add all configured indicators to an OHLCV DataFrame.

    Args:
        df: Raw OHLCV DataFrame.
        cfg: The ``features`` section of the bot configuration.

    Returns:
        DataFrame enriched with indicator columns.
    """
    feat = cfg

    df = add_ema(df, feat.get("ema_fast", 9))
    df = add_ema(df, feat.get("ema_slow", 21))
    df = add_ema(df, feat.get("ema_trend", 50))
    df = add_rsi(df, feat.get("rsi_period", 14))
    df = add_atr(df, feat.get("atr_period", 14))
    df = add_vwap(df, feat.get("vwap_period", 20))
    df = add_volume_avg(df, feat.get("volume_avg_periods", 20))

    # Highest close over N periods (used by breakout signal)
    bp = feat.get("breakout_periods", 20)
    df[f"high_{bp}"] = df["close"].rolling(bp).max()

    # Drop rows with any NaN indicator (warm-up period)
    df.dropna(inplace=True)

    return df
