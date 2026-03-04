"""Tests for src/journal/tracker.py."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from src.journal.tracker import TradeJournal


@pytest.fixture
def journal(tmp_path):
    return TradeJournal(
        log_path=tmp_path / "trades.csv",
        rolling_window=10,
        min_trades_to_cut=5,
        expectancy_threshold=0.0,
    )


def _fill_journal(j: TradeJournal, wins: int, losses: int, win_amount: float = 10.0, loss_amount: float = 5.0):
    """Helper: record a mix of winning and losing trades."""
    for _ in range(wins):
        # long trade that wins
        entry = 100.0
        exit_ = entry + win_amount
        j.record("BTC/USDT", "long", entry, exit_, quantity=1.0, fees=0.0)
    for _ in range(losses):
        # long trade that loses
        entry = 100.0
        exit_ = entry - loss_amount
        j.record("BTC/USDT", "long", entry, exit_, quantity=1.0, fees=0.0)


class TestTradeJournal:
    def test_record_returns_trade_dict(self, journal):
        trade = journal.record("BTC/USDT", "long", 100.0, 110.0, 1.0)
        assert trade["win"] is True
        assert abs(trade["pnl"] - 10.0) < 1e-9

    def test_short_pnl_positive_on_price_drop(self, journal):
        trade = journal.record("BTC/USDT", "short", 100.0, 90.0, 1.0)
        assert trade["win"] is True
        assert abs(trade["pnl"] - 10.0) < 1e-9

    def test_fees_deducted(self, journal):
        trade = journal.record("BTC/USDT", "long", 100.0, 110.0, 1.0, fees=2.0)
        assert abs(trade["pnl"] - 8.0) < 1e-9

    def test_stats_win_rate(self, journal):
        _fill_journal(journal, wins=3, losses=1)
        s = journal.stats()
        assert abs(s["win_rate"] - 0.75) < 1e-9

    def test_stats_expectancy_positive(self, journal):
        # 3 wins of 10, 1 loss of 5  → E = 0.75*10 - 0.25*5 = 6.25
        _fill_journal(journal, wins=3, losses=1, win_amount=10, loss_amount=5)
        s = journal.stats()
        assert s["expectancy"] > 0

    def test_stats_expectancy_formula(self, journal):
        _fill_journal(journal, wins=3, losses=1, win_amount=10.0, loss_amount=5.0)
        s = journal.stats()
        # E = 0.75*10 - 0.25*5 = 6.25
        assert abs(s["expectancy"] - 6.25) < 1e-9

    def test_stats_negative_expectancy(self, journal):
        # 1 win of 2, 3 losses of 10
        _fill_journal(journal, wins=1, losses=3, win_amount=2, loss_amount=10)
        s = journal.stats()
        assert s["expectancy"] < 0

    def test_max_drawdown_non_negative(self, journal):
        _fill_journal(journal, wins=2, losses=2)
        s = journal.stats()
        assert s["max_drawdown"] >= 0

    def test_max_drawdown_with_losing_streak(self, journal):
        for _ in range(5):
            journal.record("BTC/USDT", "long", 100.0, 90.0, 1.0)
        s = journal.stats()
        assert s["max_drawdown"] > 0

    def test_rolling_stats_uses_window(self, journal):
        # Record 15 trades: first 10 losses, last 5 wins
        for _ in range(10):
            journal.record("BTC/USDT", "long", 100.0, 90.0, 1.0)   # losses
        for _ in range(5):
            journal.record("BTC/USDT", "long", 100.0, 120.0, 1.0)  # wins
        rolling = journal.rolling_stats()
        full = journal.stats()
        # Rolling window (last 10) should have more wins than full history
        assert rolling["win_rate"] > full["win_rate"]

    def test_should_halt_false_below_min_trades(self, journal):
        _fill_journal(journal, wins=0, losses=3)
        assert journal.should_halt() is False  # below min_trades_to_cut

    def test_should_halt_true_on_negative_expectancy(self, journal):
        _fill_journal(journal, wins=1, losses=9, win_amount=1, loss_amount=10)
        assert journal.should_halt() is True

    def test_should_halt_false_on_positive_expectancy(self, journal):
        _fill_journal(journal, wins=9, losses=1, win_amount=10, loss_amount=1)
        assert journal.should_halt() is False

    def test_csv_written(self, journal, tmp_path):
        journal.record("BTC/USDT", "long", 100.0, 110.0, 1.0)
        csv_path = tmp_path / "trades.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "BTC/USDT" in content

    def test_persistence_reload(self, tmp_path):
        """Trades written by one journal instance should be loaded by the next."""
        path = tmp_path / "trades.csv"
        j1 = TradeJournal(log_path=path)
        j1.record("BTC/USDT", "long", 100.0, 110.0, 1.0)

        j2 = TradeJournal(log_path=path)
        assert len(j2._trades) == 1

    def test_empty_stats(self, journal):
        s = journal.stats()
        assert s["win_rate"] == 0.0
        assert s["expectancy"] == 0.0
        assert s["total_pnl"] == 0.0
