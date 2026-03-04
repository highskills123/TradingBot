"""OHLCV data fetching via ccxt."""
from __future__ import annotations

import logging
from typing import Any

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


def build_exchange(cfg: dict[str, Any]) -> ccxt.Exchange:
    """Instantiate a ccxt exchange from configuration.

    Args:
        cfg: The ``exchange`` section of the bot configuration.

    Returns:
        Configured ccxt exchange instance.
    """
    exchange_id: str = cfg.get("name", "binance")
    exchange_class = getattr(ccxt, exchange_id)

    params: dict[str, Any] = {
        "apiKey": cfg.get("api_key", ""),
        "secret": cfg.get("api_secret", ""),
        "enableRateLimit": True,
    }

    exchange: ccxt.Exchange = exchange_class(params)

    if cfg.get("testnet", True):
        if "test" in exchange.urls:
            exchange.set_sandbox_mode(True)
            logger.info("Exchange %s running in sandbox/testnet mode.", exchange_id)
        else:
            logger.warning(
                "Exchange %s has no testnet URL. Running against production.",
                exchange_id,
            )

    return exchange


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "5m",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV candles and return a tidy DataFrame.

    Args:
        exchange: Initialised ccxt exchange.
        symbol: Market symbol, e.g. ``"BTC/USDT"``.
        timeframe: Candle timeframe, e.g. ``"1m"``, ``"5m"``, ``"1h"``.
        limit: Number of candles to retrieve.

    Returns:
        DataFrame with columns ``[open, high, low, close, volume]`` indexed by
        a UTC ``DatetimeIndex``.
    """
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    logger.debug("Fetched %d candles for %s (%s).", len(df), symbol, timeframe)
    return df
