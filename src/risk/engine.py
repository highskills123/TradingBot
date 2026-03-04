"""Risk engine: stop loss, take profit and position sizing.

The engine uses ATR-based stops so that the risk is proportional to current
market volatility rather than a fixed price distance.

Position sizing formula
-----------------------
::

    risk_amount   = capital × risk_per_trade
    stop_distance = atr_stop_multiplier × ATR
    qty           = risk_amount / stop_distance

This guarantees that if the stop is hit the account loses exactly
``risk_per_trade × 100 %`` of capital, regardless of price.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeParams:
    """Parameters calculated by the risk engine for a single trade."""

    direction: str          # "long" | "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float         # in base currency units
    risk_amount: float      # in quote currency (e.g. USDT)


def compute_trade_params(
    direction: str,
    df: pd.DataFrame,
    capital: float,
    cfg_risk: dict,
    cfg_features: dict,
) -> TradeParams | None:
    """Compute stop-loss, take-profit and position size for a new trade.

    Args:
        direction: ``"long"`` or ``"short"``.
        df: Feature-enriched OHLCV DataFrame.  The last row is used.
        capital: Current account balance in quote currency.
        cfg_risk: The ``risk`` section of the bot configuration.
        cfg_features: The ``features`` section of the bot configuration.

    Returns:
        A :class:`TradeParams` instance, or *None* if calculation is not
        possible (e.g. ATR is zero or NaN).
    """
    if df.empty:
        return None

    atr_col = f"atr_{cfg_features.get('atr_period', 14)}"
    if atr_col not in df.columns:
        logger.warning("ATR column %s not found in DataFrame.", atr_col)
        return None

    curr = df.iloc[-1]
    atr = float(curr[atr_col])
    if atr <= 0 or pd.isna(atr):
        logger.warning("ATR is zero or NaN; skipping trade.")
        return None

    entry = float(curr["close"])
    risk_per_trade: float = cfg_risk.get("risk_per_trade", 0.005)
    stop_mult: float = cfg_risk.get("atr_stop_multiplier", 1.5)
    tp_mult: float = cfg_risk.get("atr_tp_multiplier", 2.0)

    stop_distance = stop_mult * atr
    risk_amount = capital * risk_per_trade

    if direction == "long":
        stop_loss = entry - stop_distance
        take_profit = entry + tp_mult * atr
    elif direction == "short":
        stop_loss = entry + stop_distance
        take_profit = entry - tp_mult * atr
    else:
        logger.error("Unknown direction: %s", direction)
        return None

    quantity = risk_amount / stop_distance
    if quantity <= 0:
        return None

    logger.debug(
        "TradeParams: dir=%s entry=%.4f sl=%.4f tp=%.4f qty=%.6f risk=%.2f",
        direction, entry, stop_loss, take_profit, quantity, risk_amount,
    )

    return TradeParams(
        direction=direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        quantity=quantity,
        risk_amount=risk_amount,
    )
