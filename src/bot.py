"""TradingBot – main orchestrator.

Usage (paper trading, default config)::

    python -m src.bot

The bot runs in a continuous loop:

1. Fetch the latest OHLCV candles.
2. Compute technical indicators (features).
3. Ask the selected signal strategy for a trade direction.
4. Run the ML regime filter (optional veto).
5. Calculate risk parameters (entry, stop, TP, size).
6. Submit the order via the executor.
7. Simulate / detect stop-loss and take-profit exits.
8. Record the closed trade in the journal and update expectancy.
9. If rolling expectancy drops below threshold → switch to paper mode.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import ccxt

from src.config import load_config
from src.data.features import build_features
from src.data.fetcher import build_exchange, fetch_ohlcv
from src.execution.executor import Executor
from src.journal.tracker import TradeJournal
from src.ml.regime import RegimeDetector
from src.risk.engine import TradeParams, compute_trade_params
from src.signals.breakout import breakout_signal
from src.signals.mean_reversion import mean_reversion_signal
from src.signals.trend import trend_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

_SIGNAL_MAP = {
    "trend": trend_signal,
    "mean_reversion": mean_reversion_signal,
    "breakout": breakout_signal,
}

# Simulated fee rate (maker/taker round-trip).  Override per exchange.
_FEE_RATE = 0.001  # 0.1 %


class TradingBot:
    """High-level bot that wires all sub-components together.

    Args:
        config_path: Path to ``config.yaml``.  Uses the project default when
            *None*.
        live: Force live-trading mode regardless of config.  When *False*
            (default) the bot runs in paper mode even if ``testnet=False``.
    """

    def __init__(self, config_path: str | None = None, live: bool = False) -> None:
        self.cfg = load_config(config_path)
        self.live = live

        exchange_cfg = self.cfg["exchange"]
        trading_cfg = self.cfg["trading"]
        risk_cfg = self.cfg["risk"]
        protection_cfg = self.cfg["protection"]
        features_cfg = self.cfg["features"]
        ml_cfg = self.cfg.get("ml", {})
        journal_cfg = self.cfg["journal"]

        self.symbol: str = trading_cfg["symbol"]
        self.timeframe: str = trading_cfg["timeframe"]
        self.strategy: str = trading_cfg.get("strategy", "trend")

        self._exchange = build_exchange(exchange_cfg)
        self._features_cfg = features_cfg
        self._risk_cfg = risk_cfg

        self._executor = Executor(
            exchange=self._exchange,
            symbol=self.symbol,
            cfg_protection=protection_cfg,
            live=self.live,
        )

        self._journal = TradeJournal(
            log_path=journal_cfg.get("log_path", "trades.csv"),
            rolling_window=journal_cfg.get("rolling_window", 50),
            min_trades_to_cut=journal_cfg.get("min_trades_to_cut", 20),
            expectancy_threshold=journal_cfg.get("expectancy_threshold", 0.0),
        )

        self._regime: RegimeDetector | None = None
        if ml_cfg.get("enabled", False):
            self._regime = RegimeDetector(
                model_path=ml_cfg.get("regime_model_path", "models/regime_model.pkl"),
                cfg_features=features_cfg,
            )

        # Track open position
        self._open_trade: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, poll_interval: float = 60.0) -> None:
        """Start the main trading loop (blocking).

        Args:
            poll_interval: Seconds to wait between iterations when no new
                candle has closed.
        """
        logger.info(
            "TradingBot started – symbol=%s timeframe=%s strategy=%s live=%s",
            self.symbol, self.timeframe, self.strategy, self.live,
        )

        # Estimate capital from exchange balance (paper mode uses 10 000 USDT)
        capital = self._get_capital()
        self._executor.reset_daily(capital)

        while True:
            try:
                self._tick(capital)
                capital = self._get_capital()
                self._executor.reset_daily(capital)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                break
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BaseError) as exc:
                logger.exception("Exchange error in main loop: %s", exc)
            except RuntimeError as exc:
                logger.exception("Runtime error in main loop: %s", exc)

            time.sleep(poll_interval)

    def step(self, capital: float) -> dict[str, Any] | None:
        """Execute a single bot iteration (useful for testing / backtesting).

        Args:
            capital: Current account balance in quote currency.

        Returns:
            The closed trade dict if a trade was opened and immediately closed
            (paper simulation), otherwise *None*.
        """
        return self._tick(capital)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tick(self, capital: float) -> dict[str, Any] | None:
        # 1. Fetch data
        df = fetch_ohlcv(self._exchange, self.symbol, self.timeframe)

        # 2. Build features
        df = build_features(df, self._features_cfg)
        if df.empty:
            logger.debug("Not enough data for features yet.")
            return None

        # 3. Check open position exit (paper simulation)
        closed_trade = self._check_exit(df)
        if closed_trade:
            return closed_trade

        # 4. Generate signal
        signal_fn = _SIGNAL_MAP.get(self.strategy, trend_signal)
        direction = signal_fn(df, self._features_cfg)
        if direction is None:
            logger.debug("No signal on latest bar.")
            return None

        # 5. ML regime filter
        if self._regime is not None and not self._regime.is_trending(df):
            logger.debug("ML regime filter vetoed trade (ranging market).")
            return None

        # 6. Risk engine
        params = compute_trade_params(
            direction, df, capital, self._risk_cfg, self._features_cfg
        )
        if params is None:
            return None

        # 7. Check halt condition
        if self._journal.should_halt():
            logger.warning("Journal halt triggered – skipping trade.")
            return None

        # 8. Submit order
        try:
            self._executor.submit(params)
        except RuntimeError as exc:
            logger.info("Trade rejected by executor: %s", exc)
            return None

        # Store open trade for exit tracking
        self._open_trade = {
            "params": params,
            "entry_bar": len(df),
        }

        return None

    def _check_exit(self, df) -> dict[str, Any] | None:
        """Simulate stop-loss / take-profit hit for the open position."""
        if self._open_trade is None:
            return None

        params: TradeParams = self._open_trade["params"]
        curr = df.iloc[-1]

        hit_sl = hit_tp = False
        exit_price = float(curr["close"])

        if params.direction == "long":
            if curr["low"] <= params.stop_loss:
                hit_sl = True
                exit_price = params.stop_loss
            elif curr["high"] >= params.take_profit:
                hit_tp = True
                exit_price = params.take_profit
        else:  # short
            if curr["high"] >= params.stop_loss:
                hit_sl = True
                exit_price = params.stop_loss
            elif curr["low"] <= params.take_profit:
                hit_tp = True
                exit_price = params.take_profit

        if not (hit_sl or hit_tp):
            return None

        reason = "stop_loss" if hit_sl else "take_profit"
        fees = exit_price * params.quantity * _FEE_RATE * 2  # round-trip

        trade = self._journal.record(
            symbol=self.symbol,
            direction=params.direction,
            entry_price=params.entry_price,
            exit_price=exit_price,
            quantity=params.quantity,
            fees=fees,
        )
        trade["exit_reason"] = reason
        pnl = trade["pnl"]
        self._executor.record_pnl(pnl)

        logger.info(
            "Position closed via %s: pnl=%.4f",
            reason, pnl,
        )

        # Log rolling expectancy
        stats = self._journal.rolling_stats()
        logger.info(
            "Rolling expectancy: %.4f  win_rate: %.1f%%  drawdown: %.4f",
            stats["expectancy"],
            stats["win_rate"] * 100,
            stats["max_drawdown"],
        )

        self._open_trade = None
        return trade

    def _get_capital(self) -> float:
        if not self.live:
            return 10_000.0  # default paper capital
        try:
            balance = self._exchange.fetch_balance()
            quote = self.symbol.split("/")[1]
            return float(balance["free"].get(quote, 10_000.0))
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            logger.warning("Could not fetch balance: %s", exc)
            return 10_000.0


def main() -> None:
    """CLI entry point."""
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
