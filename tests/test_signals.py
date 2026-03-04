"""Tests for src/signals/."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import build_features
from src.signals.breakout import breakout_signal
from src.signals.mean_reversion import mean_reversion_signal
from src.signals.trend import trend_signal


@pytest.fixture
def featured_df(ohlcv, features_cfg):
    return build_features(ohlcv.copy(), features_cfg)


class TestTrendSignal:
    def test_returns_none_or_valid(self, featured_df, features_cfg):
        result = trend_signal(featured_df, features_cfg)
        assert result in (None, "long", "short")

    def test_returns_none_on_empty(self, features_cfg):
        result = trend_signal(pd.DataFrame(), features_cfg)
        assert result is None

    def test_returns_none_on_single_row(self, featured_df, features_cfg):
        result = trend_signal(featured_df.iloc[:1], features_cfg)
        assert result is None

    def test_long_signal_conditions(self, features_cfg):
        """Manually construct a DataFrame that should trigger a long signal."""
        n = 100
        rng = np.random.default_rng(0)
        close = np.ones(n) * 100.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(n) * 500,
            }
        )
        df = build_features(df, features_cfg)

        if len(df) < 2:
            pytest.skip("Not enough rows after warm-up.")

        # Force a cross-up in the last two rows
        fast = f"ema_{features_cfg['ema_fast']}"
        slow = f"ema_{features_cfg['ema_slow']}"
        trend = f"ema_{features_cfg['ema_trend']}"
        atr = f"atr_{features_cfg['atr_period']}"

        df = df.copy()
        df.iloc[-2, df.columns.get_loc(fast)] = 99.0
        df.iloc[-2, df.columns.get_loc(slow)] = 100.0
        df.iloc[-1, df.columns.get_loc(fast)] = 101.0
        df.iloc[-1, df.columns.get_loc(slow)] = 100.0
        df.iloc[-1, df.columns.get_loc("close")] = 105.0
        df.iloc[-1, df.columns.get_loc(trend)] = 100.0
        # Make ATR above its average
        df[atr] = df[atr] * 2
        df.iloc[-1, df.columns.get_loc(atr)] = df[atr].mean() * 2

        result = trend_signal(df, features_cfg)
        assert result == "long"

    def test_missing_columns_returns_none(self, ohlcv, features_cfg):
        result = trend_signal(ohlcv, features_cfg)
        assert result is None


class TestMeanReversionSignal:
    def test_returns_none_or_valid(self, featured_df, features_cfg):
        result = mean_reversion_signal(featured_df, features_cfg)
        assert result in (None, "long", "short")

    def test_returns_none_on_empty(self, features_cfg):
        result = mean_reversion_signal(pd.DataFrame(), features_cfg)
        assert result is None

    def test_long_when_oversold_below_vwap(self, featured_df, features_cfg):
        """Force conditions for a long mean-reversion signal."""
        df = featured_df.copy()
        rsi_col = f"rsi_{features_cfg['rsi_period']}"
        vwap_col = f"vwap_{features_cfg['vwap_period']}"
        atr_col = f"atr_{features_cfg['atr_period']}"

        df.iloc[-1, df.columns.get_loc(rsi_col)] = 25.0          # oversold
        df.iloc[-1, df.columns.get_loc("close")] = 100.0
        df.iloc[-1, df.columns.get_loc(vwap_col)] = 105.0        # above close
        df.iloc[-1, df.columns.get_loc(atr_col)] = 1.0           # ATR

        result = mean_reversion_signal(df, features_cfg)
        assert result == "long"

    def test_short_when_overbought_above_vwap(self, featured_df, features_cfg):
        """Force conditions for a short mean-reversion signal."""
        df = featured_df.copy()
        rsi_col = f"rsi_{features_cfg['rsi_period']}"
        vwap_col = f"vwap_{features_cfg['vwap_period']}"
        atr_col = f"atr_{features_cfg['atr_period']}"

        df.iloc[-1, df.columns.get_loc(rsi_col)] = 75.0          # overbought
        df.iloc[-1, df.columns.get_loc("close")] = 110.0
        df.iloc[-1, df.columns.get_loc(vwap_col)] = 105.0        # below close
        df.iloc[-1, df.columns.get_loc(atr_col)] = 1.0

        result = mean_reversion_signal(df, features_cfg)
        assert result == "short"


class TestBreakoutSignal:
    def test_returns_none_or_valid(self, featured_df, features_cfg):
        result = breakout_signal(featured_df, features_cfg)
        assert result in (None, "long", "short")

    def test_returns_none_on_single_row(self, featured_df, features_cfg):
        result = breakout_signal(featured_df.iloc[:1], features_cfg)
        assert result is None

    def test_long_on_breakout_with_volume(self, featured_df, features_cfg):
        """Force a long breakout: close equals previous N-period high and volume spikes."""
        df = featured_df.copy()
        bp = features_cfg["breakout_periods"]
        vol_p = features_cfg["volume_avg_periods"]
        high_col = f"high_{bp}"
        vol_col = f"vol_avg_{vol_p}"

        current_high = float(df[high_col].iloc[-1])
        # Make the previous bar's high_N slightly below current close
        df.iloc[-2, df.columns.get_loc(high_col)] = current_high * 0.99
        df.iloc[-1, df.columns.get_loc("close")] = current_high
        df.iloc[-1, df.columns.get_loc("volume")] = df[vol_col].iloc[-1] * 3

        result = breakout_signal(df, features_cfg)
        assert result == "long"
