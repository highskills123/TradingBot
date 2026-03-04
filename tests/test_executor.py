"""Tests for src/execution/executor.py."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.execution.executor import Executor
from src.risk.engine import TradeParams


def _make_params(direction: str = "long") -> TradeParams:
    return TradeParams(
        direction=direction,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=104.0,
        quantity=1.0,
        risk_amount=2.0,
    )


@pytest.fixture
def executor():
    exchange = MagicMock()
    cfg = {
        "max_trades_per_day": 3,
        "cooldown_seconds": 0,
        "kill_switch_loss_pct": 0.05,
    }
    ex = Executor(exchange=exchange, symbol="BTC/USDT", cfg_protection=cfg, live=False)
    ex.reset_daily(10_000.0)
    return ex


class TestExecutor:
    def test_paper_order_returned(self, executor):
        order = executor.submit(_make_params())
        assert order["paper"] is True
        assert order["side"] == "buy"

    def test_short_order_side(self, executor):
        order = executor.submit(_make_params("short"))
        assert order["side"] == "sell"

    def test_trade_count_incremented(self, executor):
        executor.submit(_make_params())
        assert executor._trades_today == 1

    def test_max_trades_per_day_blocks(self, executor):
        for _ in range(3):
            executor.submit(_make_params())
        assert not executor.can_trade()
        with pytest.raises(RuntimeError):
            executor.submit(_make_params())

    def test_cooldown_blocks(self):
        exchange = MagicMock()
        cfg = {
            "max_trades_per_day": 10,
            "cooldown_seconds": 9999,
            "kill_switch_loss_pct": 0.1,
        }
        ex = Executor(exchange=exchange, symbol="BTC/USDT", cfg_protection=cfg, live=False)
        ex.reset_daily(10_000.0)
        # First trade goes through; second is blocked by the cooldown period.
        ex.submit(_make_params())
        assert not ex.can_trade()
        with pytest.raises(RuntimeError):
            ex.submit(_make_params())

    def test_kill_switch_triggers_on_loss(self, executor):
        # Lose 6 % of 10 000 = 600
        executor.record_pnl(-600.0)
        assert executor._killed is True
        assert not executor.can_trade()

    def test_kill_switch_not_triggered_on_small_loss(self, executor):
        executor.record_pnl(-100.0)
        assert executor._killed is False
        assert executor.can_trade()
