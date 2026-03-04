"""Trade journal: logging, expectancy calculation and drawdown tracking.

Every closed trade is recorded to a CSV file and the following metrics are
continuously updated:

* **Win rate** – fraction of winning trades.
* **Avg win / Avg loss** – mean P&L of winning and losing trades.
* **Expectancy** – ``P(win) × AvgWin - P(loss) × AvgLoss``.
  Computed over a rolling window to reflect recent performance.
* **Max drawdown** – peak-to-trough drawdown of the cumulative equity curve.

All monetary values are in quote currency (e.g. USDT).
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_CSV_FIELDS = [
    "timestamp",
    "symbol",
    "direction",
    "entry_price",
    "exit_price",
    "quantity",
    "pnl",
    "pnl_pct",
    "win",
    "fees",
]


class TradeJournal:
    """Persist and analyse closed trades.

    Args:
        log_path: Path to the CSV file where trades are recorded.
        rolling_window: Number of recent trades used for rolling expectancy.
        min_trades_to_cut: Minimum trades before the expectancy check is
            enforced (avoids reacting to statistical noise early on).
        expectancy_threshold: Rolling expectancy below this value triggers a
            warning / paper-mode recommendation.
    """

    def __init__(
        self,
        log_path: str | Path = "trades.csv",
        rolling_window: int = 50,
        min_trades_to_cut: int = 20,
        expectancy_threshold: float = 0.0,
    ) -> None:
        self.log_path = Path(log_path)
        self.rolling_window = rolling_window
        self.min_trades_to_cut = min_trades_to_cut
        self.expectancy_threshold = expectancy_threshold

        self._trades: list[dict[str, Any]] = []
        self._equity_curve: list[float] = []

        # Load existing trades from disk if the file exists
        if self.log_path.exists():
            self._load_from_csv()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        fees: float = 0.0,
    ) -> dict[str, Any]:
        """Record a closed trade and return the enriched trade dict.

        Args:
            symbol: Market symbol.
            direction: ``"long"`` or ``"short"``.
            entry_price: Fill price at entry.
            exit_price: Fill price at exit (close / stop / TP).
            quantity: Position size in base currency.
            fees: Total round-trip fees in quote currency.

        Returns:
            Trade record dict including computed P&L fields.
        """
        if direction == "long":
            raw_pnl = (exit_price - entry_price) * quantity
        else:
            raw_pnl = (entry_price - exit_price) * quantity

        pnl = raw_pnl - fees
        pnl_pct = pnl / (entry_price * quantity) if entry_price * quantity else 0.0
        win = pnl > 0

        trade: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "win": win,
            "fees": fees,
        }

        self._trades.append(trade)
        self._update_equity(pnl)
        self._append_to_csv(trade)

        logger.info(
            "Trade recorded: %s %s %.4f→%.4f pnl=%.4f fees=%.4f win=%s",
            symbol, direction, entry_price, exit_price, pnl, fees, win,
        )

        return trade

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        """Return performance statistics over *all* recorded trades.

        Returns:
            Dictionary with ``win_rate``, ``avg_win``, ``avg_loss``,
            ``expectancy``, ``max_drawdown`` and ``total_pnl``.
        """
        return self._compute_stats(self._trades)

    def rolling_stats(self) -> dict[str, float]:
        """Return performance statistics over the last *rolling_window* trades."""
        window = self._trades[-self.rolling_window :]
        return self._compute_stats(window)

    def should_halt(self) -> bool:
        """Return *True* if the rolling expectancy is below the threshold.

        Only evaluated once at least ``min_trades_to_cut`` trades have been
        recorded so that statistical noise does not cause premature halting.
        """
        if len(self._trades) < self.min_trades_to_cut:
            return False
        s = self.rolling_stats()
        if s["expectancy"] < self.expectancy_threshold:
            logger.warning(
                "Rolling expectancy %.4f < threshold %.4f – recommend paper mode.",
                s["expectancy"],
                self.expectancy_threshold,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(trades: list[dict[str, Any]]) -> dict[str, float]:
        if not trades:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
            }

        wins = [t["pnl"] for t in trades if t["win"]]
        losses = [t["pnl"] for t in trades if not t["win"]]

        total = len(trades)
        p_win = len(wins) / total
        p_loss = len(losses) / total
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        expectancy = p_win * avg_win - p_loss * avg_loss

        # Equity curve drawdown
        pnl_series = pd.Series([t["pnl"] for t in trades]).cumsum()
        rolling_max = pnl_series.cummax()
        drawdowns = rolling_max - pnl_series
        max_drawdown = float(drawdowns.max())

        return {
            "win_rate": p_win,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "max_drawdown": max_drawdown,
            "total_pnl": float(pnl_series.iloc[-1]),
        }

    def _update_equity(self, pnl: float) -> None:
        prev = self._equity_curve[-1] if self._equity_curve else 0.0
        self._equity_curve.append(prev + pnl)

    def _append_to_csv(self, trade: dict[str, Any]) -> None:
        write_header = not self.log_path.exists() or self.log_path.stat().st_size == 0
        with self.log_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: trade[k] for k in _CSV_FIELDS})

    def _load_from_csv(self) -> None:
        try:
            df = pd.read_csv(self.log_path)
            for _, row in df.iterrows():
                self._trades.append(row.to_dict())
                self._update_equity(float(row["pnl"]))
            logger.info("Loaded %d existing trades from %s.", len(self._trades), self.log_path)
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError) as exc:
            logger.warning("Could not load existing trades: %s", exc)
