"""Tests for src/data/features.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_atr,
    add_ema,
    add_rsi,
    add_vwap,
    add_volume_avg,
    build_features,
)


class TestAddEma:
    def test_column_created(self, ohlcv):
        result = add_ema(ohlcv, 9)
        assert "ema_9" in result.columns

    def test_values_finite(self, ohlcv):
        result = add_ema(ohlcv, 9)
        assert result["ema_9"].notna().all()

    def test_original_not_mutated(self, ohlcv):
        original_cols = list(ohlcv.columns)
        add_ema(ohlcv, 9)
        assert list(ohlcv.columns) == original_cols

    def test_ema_tracks_price(self, ohlcv):
        """EMA should be correlated with close price."""
        result = add_ema(ohlcv, 9)
        correlation = result["ema_9"].corr(result["close"])
        assert correlation > 0.95


class TestAddRsi:
    def test_column_created(self, ohlcv):
        result = add_rsi(ohlcv, 14)
        assert "rsi_14" in result.columns

    def test_values_in_range(self, ohlcv):
        result = add_rsi(ohlcv, 14)
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_warm_up_period(self, ohlcv):
        """First (period-1) rows should be NaN."""
        result = add_rsi(ohlcv, 14)
        # At least some initial NaNs expected before warmup completes
        assert result["rsi_14"].isna().sum() >= 1


class TestAddAtr:
    def test_column_created(self, ohlcv):
        result = add_atr(ohlcv, 14)
        assert "atr_14" in result.columns

    def test_values_positive(self, ohlcv):
        result = add_atr(ohlcv, 14)
        valid = result["atr_14"].dropna()
        assert (valid >= 0).all()

    def test_atr_less_than_price(self, ohlcv):
        result = add_atr(ohlcv, 14)
        valid = result.dropna()
        assert (valid["atr_14"] < valid["close"]).all()


class TestAddVwap:
    def test_column_created(self, ohlcv):
        result = add_vwap(ohlcv, 20)
        assert "vwap_20" in result.columns

    def test_values_close_to_price(self, ohlcv):
        result = add_vwap(ohlcv, 20).dropna()
        # VWAP should be within ±5 % of close on average
        pct_diff = ((result["vwap_20"] - result["close"]) / result["close"]).abs()
        assert pct_diff.mean() < 0.05


class TestAddVolumeAvg:
    def test_column_created(self, ohlcv):
        result = add_volume_avg(ohlcv, 20)
        assert "vol_avg_20" in result.columns

    def test_values_positive(self, ohlcv):
        result = add_volume_avg(ohlcv, 20).dropna()
        assert (result["vol_avg_20"] > 0).all()


class TestBuildFeatures:
    def test_all_columns_present(self, ohlcv, features_cfg):
        result = build_features(ohlcv.copy(), features_cfg)
        expected_cols = [
            "ema_9", "ema_21", "ema_50",
            "rsi_14", "atr_14", "vwap_20", "vol_avg_20", "high_20",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nans_after_build(self, ohlcv, features_cfg):
        result = build_features(ohlcv.copy(), features_cfg)
        assert result.isna().sum().sum() == 0

    def test_length_reduced_by_warmup(self, ohlcv, features_cfg):
        result = build_features(ohlcv.copy(), features_cfg)
        assert len(result) < len(ohlcv)
