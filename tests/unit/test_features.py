"""Tests for feature engineering pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agent.data.features import compute_features
from src.agent.models import FeatureVector


class TestComputeFeatures:
    def test_returns_valid_feature_vector(self, sample_bars_df):
        result = compute_features(sample_bars_df, "AAPL")

        assert isinstance(result, FeatureVector)
        assert result.symbol == "AAPL"

    def test_returns_correct_symbol(self, sample_bars_df):
        result = compute_features(sample_bars_df, "TSLA")

        assert result.symbol == "TSLA"

    def test_all_features_are_finite(self, sample_bars_df):
        result = compute_features(sample_bars_df, "AAPL")

        for name, val in zip(FeatureVector.feature_names(), result.to_array()):
            assert np.isfinite(val), f"Feature '{name}' has non-finite value: {val}"

    def test_rsi_computed_with_enough_bars(self, sample_bars_df):
        """100-bar fixture has enough data for RSI-14 (needs >= 15)."""
        result = compute_features(sample_bars_df, "AAPL")

        assert result.rsi_14 != 50.0
        assert 0.0 <= result.rsi_14 <= 100.0

    def test_sma50_computed_with_100_bars(self, sample_bars_df):
        """100-bar fixture has enough data for SMA-50."""
        result = compute_features(sample_bars_df, "AAPL")

        assert result.sma_50 > 0.0


class TestInsufficientData:
    def test_single_bar_returns_defaults(self):
        df = pd.DataFrame({
            "open": [150.0],
            "high": [151.0],
            "low": [149.0],
            "close": [150.5],
            "volume": [100_000.0],
            "vwap": [150.2],
        })

        result = compute_features(df, "AAPL")

        assert isinstance(result, FeatureVector)
        assert result.symbol == "AAPL"
        assert result.rsi_14 == 50.0
        assert result.volume_ratio == 1.0
        assert result.returns_1 == 0.0

    def test_empty_dataframe_returns_defaults(self):
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "vwap"]
        )

        result = compute_features(df, "TSLA")

        assert result.symbol == "TSLA"
        assert result.returns_1 == 0.0
        assert result.rsi_14 == 50.0

    def test_fewer_than_50_bars_sma50_defaults_to_zero(self):
        """SMA-50 requires >= 50 bars; with only 30, it should stay at 0."""
        np.random.seed(42)
        n = 30
        dates = pd.date_range(
            "2024-01-02 09:30", periods=n, freq="5min", tz="US/Eastern"
        )
        close = 150.0 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "open": close + np.random.randn(n) * 0.2,
                "high": close + np.abs(np.random.randn(n) * 0.3),
                "low": close - np.abs(np.random.randn(n) * 0.3),
                "close": close,
                "volume": np.random.randint(10_000, 500_000, n).astype(float),
                "vwap": close,
            },
            index=dates,
        )

        result = compute_features(df, "MSFT")

        assert result.sma_50 == 0.0
        assert result.returns_1 != 0.0  # still computed with enough bars
