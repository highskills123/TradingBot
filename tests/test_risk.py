"""Tests for src/risk/engine.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import build_features
from src.risk.engine import TradeParams, compute_trade_params


@pytest.fixture
def risk_cfg():
    return {
        "risk_per_trade": 0.005,
        "atr_stop_multiplier": 1.5,
        "atr_tp_multiplier": 2.0,
    }


@pytest.fixture
def featured_df(ohlcv, features_cfg):
    return build_features(ohlcv.copy(), features_cfg)


class TestComputeTradeParams:
    def test_long_params(self, featured_df, risk_cfg, features_cfg):
        params = compute_trade_params("long", featured_df, 10_000, risk_cfg, features_cfg)
        assert params is not None
        assert params.direction == "long"
        assert params.stop_loss < params.entry_price
        assert params.take_profit > params.entry_price
        assert params.quantity > 0

    def test_short_params(self, featured_df, risk_cfg, features_cfg):
        params = compute_trade_params("short", featured_df, 10_000, risk_cfg, features_cfg)
        assert params is not None
        assert params.direction == "short"
        assert params.stop_loss > params.entry_price
        assert params.take_profit < params.entry_price
        assert params.quantity > 0

    def test_risk_amount_correct(self, featured_df, risk_cfg, features_cfg):
        capital = 10_000.0
        params = compute_trade_params("long", featured_df, capital, risk_cfg, features_cfg)
        assert params is not None
        expected_risk = capital * risk_cfg["risk_per_trade"]
        assert abs(params.risk_amount - expected_risk) < 0.01

    def test_position_sizing_implies_risk(self, featured_df, risk_cfg, features_cfg):
        """qty × stop_distance should equal risk_amount."""
        params = compute_trade_params("long", featured_df, 10_000, risk_cfg, features_cfg)
        assert params is not None
        atr_col = f"atr_{features_cfg['atr_period']}"
        atr = float(featured_df[atr_col].iloc[-1])
        stop_distance = risk_cfg["atr_stop_multiplier"] * atr
        implied_risk = params.quantity * stop_distance
        assert abs(implied_risk - params.risk_amount) < 0.01

    def test_returns_none_on_empty_df(self, risk_cfg, features_cfg):
        result = compute_trade_params("long", pd.DataFrame(), 10_000, risk_cfg, features_cfg)
        assert result is None

    def test_returns_none_on_missing_atr(self, ohlcv, risk_cfg, features_cfg):
        result = compute_trade_params("long", ohlcv, 10_000, risk_cfg, features_cfg)
        assert result is None

    def test_returns_none_on_unknown_direction(self, featured_df, risk_cfg, features_cfg):
        result = compute_trade_params("sideways", featured_df, 10_000, risk_cfg, features_cfg)
        assert result is None

    def test_risk_reward_ratio(self, featured_df, risk_cfg, features_cfg):
        """TP distance should be > SL distance (positive expectancy ratio)."""
        params = compute_trade_params("long", featured_df, 10_000, risk_cfg, features_cfg)
        assert params is not None
        sl_dist = abs(params.entry_price - params.stop_loss)
        tp_dist = abs(params.entry_price - params.take_profit)
        ratio = tp_dist / sl_dist
        expected_ratio = risk_cfg["atr_tp_multiplier"] / risk_cfg["atr_stop_multiplier"]
        assert abs(ratio - expected_ratio) < 0.01
