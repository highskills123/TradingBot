"""ML-based market regime detector and trade filter.

The regime detector classifies each bar as **trending** or **ranging** using a
lightweight Random-Forest classifier trained on momentum / volatility features.
It is used purely as a *filter*: the bot's rule-based signals still decide
direction, but the ML model can veto low-confidence trades.

Features used
-------------
* EMA slope (fast EMA change over N bars)
* ATR relative to price (volatility normalised)
* RSI
* Volume ratio (volume / rolling-average volume)
* Price position relative to trend EMA

Training
--------
Labels are generated heuristically:

* A bar is "trending" (label 1) when the slow EMA slope is steep and ATR is
  elevated.
* Otherwise it is "ranging" (label 0).

This labelling strategy is intentionally simple and can be replaced by a
more rigorous walk-forward labelling scheme in production.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_FEATURE_COLS = [
    "ema_slope",
    "atr_ratio",
    "rsi",
    "vol_ratio",
    "price_vs_trend",
]


class RegimeDetector:
    """Classifies the current market as trending (1) or ranging (0).

    Args:
        model_path: Where to save / load the trained sklearn model.
        cfg_features: The ``features`` section of the bot configuration.
    """

    def __init__(
        self,
        model_path: str | Path = "models/regime_model.pkl",
        cfg_features: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.cfg = cfg_features or {}
        self._model: RandomForestClassifier | None = None
        self._scaler: StandardScaler | None = None

        if self.model_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Train the regime classifier on historical feature data.

        Args:
            df: Feature-enriched OHLCV DataFrame.  Must contain the indicator
                columns produced by :mod:`src.data.features`.
        """
        features_df = self._extract_features(df)
        labels = self._generate_labels(df, features_df)

        valid = features_df.notna().all(axis=1) & labels.notna()
        X = features_df[valid].values
        y = labels[valid].values

        if len(X) < 50:
            logger.warning("Too few samples (%d) to train regime model.", len(X))
            return

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        self._model.fit(X_scaled, y)
        logger.info("Regime model trained on %d samples.", len(X))
        self._save()

    def predict(self, df: pd.DataFrame) -> int:
        """Predict the regime of the most recent bar.

        Args:
            df: Feature-enriched OHLCV DataFrame (at least the last bar).

        Returns:
            ``1`` for trending, ``0`` for ranging, ``-1`` if model is not
            available (treated as "no filter – allow trade").
        """
        if self._model is None or self._scaler is None:
            return -1  # no model – pass through

        features_df = self._extract_features(df)
        last = features_df.iloc[[-1]]
        if last.isna().any(axis=1).iloc[0]:
            return -1

        X = self._scaler.transform(last.values)
        return int(self._model.predict(X)[0])

    def is_trending(self, df: pd.DataFrame) -> bool:
        """Return *True* when the market is classified as trending."""
        return self.predict(df) != 0  # 1 (trending) or -1 (no model) → allow

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.cfg.get("ema_fast", 9)
        slow = self.cfg.get("ema_slow", 21)
        trend = self.cfg.get("ema_trend", 50)
        rsi_p = self.cfg.get("rsi_period", 14)
        atr_p = self.cfg.get("atr_period", 14)
        vol_p = self.cfg.get("volume_avg_periods", 20)

        feat = pd.DataFrame(index=df.index)

        # EMA slope: change in slow EMA over 3 bars, normalised by price
        if f"ema_{slow}" in df.columns:
            feat["ema_slope"] = df[f"ema_{slow}"].diff(3) / df["close"]
        else:
            feat["ema_slope"] = np.nan

        # ATR relative to price
        if f"atr_{atr_p}" in df.columns:
            feat["atr_ratio"] = df[f"atr_{atr_p}"] / df["close"]
        else:
            feat["atr_ratio"] = np.nan

        # RSI (normalised to 0–1)
        if f"rsi_{rsi_p}" in df.columns:
            feat["rsi"] = df[f"rsi_{rsi_p}"] / 100.0
        else:
            feat["rsi"] = np.nan

        # Volume ratio
        if f"vol_avg_{vol_p}" in df.columns:
            feat["vol_ratio"] = df["volume"] / df[f"vol_avg_{vol_p}"].replace(0, np.nan)
        else:
            feat["vol_ratio"] = np.nan

        # Price vs trend EMA
        if f"ema_{trend}" in df.columns:
            feat["price_vs_trend"] = (df["close"] - df[f"ema_{trend}"]) / df["close"]
        else:
            feat["price_vs_trend"] = np.nan

        return feat

    @staticmethod
    def _generate_labels(
        df: pd.DataFrame, features_df: pd.DataFrame
    ) -> pd.Series:
        """Heuristic labels: trending when EMA slope is steep and ATR elevated."""
        if "ema_slope" not in features_df.columns or "atr_ratio" not in features_df.columns:
            return pd.Series(np.nan, index=df.index)

        slope_threshold = features_df["ema_slope"].abs().quantile(0.6)
        atr_threshold = features_df["atr_ratio"].quantile(0.5)

        trending = (
            (features_df["ema_slope"].abs() > slope_threshold)
            & (features_df["atr_ratio"] > atr_threshold)
        ).astype(int)
        return trending

    def _save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "scaler": self._scaler}, self.model_path)
        logger.info("Regime model saved to %s.", self.model_path)

    def _load(self) -> None:
        try:
            payload: dict = joblib.load(self.model_path)
            self._model = payload["model"]
            self._scaler = payload["scaler"]
            logger.info("Regime model loaded from %s.", self.model_path)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.warning("Could not load regime model: %s", exc)
