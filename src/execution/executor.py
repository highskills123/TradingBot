"""Order execution with protective guards.

Protection mechanisms
---------------------
* **Max trades per day** – once the daily limit is reached the bot stops
  opening new positions.
* **Cooldown** – a minimum number of seconds must elapse between trades to
  avoid overtrading during volatile microstructure noise.
* **Kill switch** – if the cumulative P&L for the current day exceeds the
  configured loss threshold the bot halts automatically.

Paper-trading mode
------------------
When ``live=False`` no real orders are sent; the executor logs the simulated
fill and returns a synthetic order dict so that the rest of the pipeline
(journal, risk) works identically.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from typing import Any

import ccxt

from src.risk.engine import TradeParams

logger = logging.getLogger(__name__)


class Executor:
    """Manages order submission and protective logic.

    Args:
        exchange: Initialised ccxt exchange (may be a sandbox instance).
        symbol: Market symbol, e.g. ``"BTC/USDT"``.
        cfg_protection: The ``protection`` section of the bot configuration.
        live: When *False* the executor operates in paper-trading mode; no
            real orders are sent to the exchange.
    """

    def __init__(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        cfg_protection: dict[str, Any],
        live: bool = False,
    ) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.live = live

        self._max_trades_per_day: int = cfg_protection.get("max_trades_per_day", 10)
        self._cooldown_seconds: float = cfg_protection.get("cooldown_seconds", 60)
        self._kill_switch_loss_pct: float = cfg_protection.get("kill_switch_loss_pct", 0.05)

        self._trades_today: int = 0
        self._last_trade_time: float = float("-inf")  # no prior trade
        self._trade_date: date = datetime.now(timezone.utc).date()
        self._daily_pnl: float = 0.0
        self._initial_capital: float | None = None
        self._killed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_daily(self, capital: float) -> None:
        """Reset daily counters.  Call at the start of each trading day."""
        today = datetime.now(timezone.utc).date()
        if today != self._trade_date:
            self._trades_today = 0
            self._daily_pnl = 0.0
            self._trade_date = today
            self._killed = False
        self._initial_capital = capital

    def record_pnl(self, pnl: float) -> None:
        """Accumulate realised P&L and trigger kill switch if needed."""
        self._daily_pnl += pnl
        if self._initial_capital and self._initial_capital > 0:
            loss_pct = -self._daily_pnl / self._initial_capital
            if loss_pct >= self._kill_switch_loss_pct:
                logger.warning(
                    "Kill switch triggered: daily loss %.2f%% >= %.2f%%",
                    loss_pct * 100,
                    self._kill_switch_loss_pct * 100,
                )
                self._killed = True

    def can_trade(self) -> bool:
        """Return *True* if a new trade is allowed given all protection rules."""
        if self._killed:
            logger.info("Kill switch active – no new trades.")
            return False

        today = datetime.now(timezone.utc).date()
        if today != self._trade_date:
            self.reset_daily(self._initial_capital or 0.0)

        if self._trades_today >= self._max_trades_per_day:
            logger.info(
                "Daily trade limit reached (%d/%d).",
                self._trades_today,
                self._max_trades_per_day,
            )
            return False

        elapsed = time.monotonic() - self._last_trade_time
        if elapsed < self._cooldown_seconds:
            logger.info(
                "Cooldown active – %.0fs remaining.",
                self._cooldown_seconds - elapsed,
            )
            return False

        return True

    def submit(self, params: TradeParams) -> dict[str, Any]:
        """Submit a market order (or simulate it in paper mode).

        Args:
            params: Trade parameters produced by the risk engine.

        Returns:
            Order dict (real ccxt response or synthetic paper object).

        Raises:
            RuntimeError: If :meth:`can_trade` returns *False*.
        """
        if not self.can_trade():
            raise RuntimeError("Trade not allowed by protection rules.")

        side = "buy" if params.direction == "long" else "sell"

        if self.live:
            order = self.exchange.create_market_order(
                self.symbol, side, params.quantity
            )
            logger.info("Live order submitted: %s", order)
        else:
            order = self._paper_order(side, params)
            logger.info("Paper order simulated: %s", order)

        self._trades_today += 1
        self._last_trade_time = time.monotonic()
        return order

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _paper_order(self, side: str, params: TradeParams) -> dict[str, Any]:
        return {
            "id": f"paper-{int(time.time() * 1000)}",
            "symbol": self.symbol,
            "side": side,
            "type": "market",
            "price": params.entry_price,
            "amount": params.quantity,
            "filled": params.quantity,
            "status": "closed",
            "timestamp": int(time.time() * 1000),
            "paper": True,
        }
