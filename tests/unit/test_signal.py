"""Tests for signal generation."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.agent.config import StrategyConfig
from src.agent.models import FeatureVector, Signal, SignalAction, Side
from src.agent.signal.engine import SignalEngine
from src.agent.signal.ml_model import MLSignalModel
from src.agent.signal.technical import TechnicalSignalGenerator


@pytest.fixture
def strategy_cfg() -> StrategyConfig:
    return StrategyConfig(
        lookback_bars=20,
        volume_surge_ratio=1.5,
        ml_confidence_threshold=0.6,
    )


@pytest.fixture
def flat_bars_df() -> pd.DataFrame:
    """Bars with no breakout — price stays within a very tight range."""
    np.random.seed(99)
    n = 30
    dates = pd.date_range(
        "2024-01-02 09:30", periods=n, freq="5min", tz="US/Eastern"
    )
    close = np.full(n, 150.0) + np.random.randn(n) * 0.01
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": np.full(n, 100_000.0),
            "vwap": close,
        },
        index=dates,
    )


@pytest.fixture
def flat_features() -> FeatureVector:
    return FeatureVector(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc),
        returns_1=0.0,
        returns_5=0.0,
        rsi_14=50.0,
        volume_ratio=1.0,
        atr_14=0.5,
    )


class TestTechnicalSignalGenerator:
    def test_returns_hold_for_flat_market(
        self, strategy_cfg, flat_bars_df, flat_features
    ):
        gen = TechnicalSignalGenerator(strategy_cfg)

        signal = gen.evaluate(flat_bars_df, flat_features)

        assert signal.action == SignalAction.HOLD
        assert signal.side == Side.FLAT

    def test_returns_hold_for_insufficient_bars(self, strategy_cfg, flat_features):
        tiny_df = pd.DataFrame(
            {
                "open": [150.0],
                "high": [151.0],
                "low": [149.0],
                "close": [150.0],
                "volume": [100_000.0],
            }
        )
        gen = TechnicalSignalGenerator(strategy_cfg)

        signal = gen.evaluate(tiny_df, flat_features)

        assert signal.action == SignalAction.HOLD

    def test_returns_hold_for_empty_bars(self, strategy_cfg, flat_features):
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        gen = TechnicalSignalGenerator(strategy_cfg)

        signal = gen.evaluate(empty_df, flat_features)

        assert signal.action == SignalAction.HOLD


class TestSignalEngine:
    def test_flat_market_produces_hold(
        self, strategy_cfg, flat_bars_df, flat_features
    ):
        engine = SignalEngine(strategy_cfg)

        signal = engine.generate_signal("AAPL", flat_bars_df, flat_features)

        assert signal.action == SignalAction.HOLD

    def test_respects_confidence_threshold(
        self, strategy_cfg, flat_bars_df, flat_features
    ):
        """Low-confidence signals should not pass the threshold filter."""
        engine = SignalEngine(strategy_cfg)

        signal = engine.generate_signal("AAPL", flat_bars_df, flat_features)

        if signal.action != SignalAction.HOLD:
            assert signal.confidence >= strategy_cfg.ml_confidence_threshold

    def test_sentiment_can_veto_long_signal(self, strategy_cfg, flat_bars_df):
        class _MockTech:
            def evaluate(self, bars_df, features, quote=None):
                return Signal(
                    symbol=features.symbol,
                    timestamp=features.timestamp,
                    action=SignalAction.BUY,
                    side=Side.LONG,
                    confidence=0.9,
                    rule_score=0.9,
                    features=features,
                    reason="mock long",
                )

        features = FeatureVector(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc),
            atr_14=1.0,
            sentiment_score=-0.8,
        )
        engine = SignalEngine(
            strategy_cfg,
            technical=_MockTech(),
            sentiment_negative_threshold=-0.2,
        )

        signal = engine.generate_signal("AAPL", flat_bars_df, features)
        assert signal.action == SignalAction.HOLD
        assert "sentiment veto" in signal.reason
        assert signal.metadata.get("sentiment_effect") == "veto_long"

    def test_sentiment_can_boost_aligned_signal(self, strategy_cfg, flat_bars_df):
        class _MockTech:
            def evaluate(self, bars_df, features, quote=None):
                return Signal(
                    symbol=features.symbol,
                    timestamp=features.timestamp,
                    action=SignalAction.BUY,
                    side=Side.LONG,
                    confidence=0.6,
                    rule_score=0.6,
                    features=features,
                    reason="mock long",
                )

        features = FeatureVector(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc),
            atr_14=1.0,
            sentiment_score=0.9,
        )
        engine = SignalEngine(
            strategy_cfg,
            technical=_MockTech(),
            sentiment_confidence_boost=0.1,
        )

        signal = engine.generate_signal("AAPL", flat_bars_df, features)
        assert signal.action == SignalAction.BUY
        assert signal.confidence > 0.6
        assert signal.metadata.get("sentiment_effect") == "aligned_boost"


class TestMLSignalModel:
    def test_is_trained_false_before_training(self):
        model = MLSignalModel()

        assert model.is_trained is False

    def test_predict_raises_before_training(self, flat_features):
        model = MLSignalModel()

        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(flat_features)

    def test_is_trained_true_after_training(self):
        model = MLSignalModel()
        np.random.seed(42)
        X = np.random.randn(100, 11)
        y = np.random.randint(0, 3, 100)

        model.train(X, y)

        assert model.is_trained is True

    def test_predict_returns_side_and_confidence(self):
        model = MLSignalModel()
        np.random.seed(42)
        X = np.random.randn(100, 11)
        y = np.random.randint(0, 3, 100)
        model.train(X, y)

        features = FeatureVector(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc),
        )
        side, confidence = model.predict(features)

        assert isinstance(side, Side)
        assert 0.0 <= confidence <= 1.0
