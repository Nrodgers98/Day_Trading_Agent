"""Tests for configuration loading and validation."""
from __future__ import annotations

import pytest

from src.agent.config import AppConfig, TradingConfig, load_config, _deep_merge


class TestLoadConfig:
    def test_load_default_yaml_returns_app_config(self):
        cfg = load_config("config/default.yaml")

        assert isinstance(cfg, AppConfig)
        assert cfg.trading.mode == "paper"
        assert cfg.risk.max_concurrent_positions == 10
        assert cfg.strategy.lookback_bars == 20
        assert cfg.improvement.optimize_for == "risk_adjusted_return"
        assert cfg.improvement.autonomy_mode == "manual"
        assert cfg.improvement.evaluate_with_backtest is False
        assert cfg.sentiment.provider in {"none", "finbert"}

    def test_load_with_overrides(self):
        cfg = load_config(
            "config/default.yaml",
            overrides={
                "strategy": {"lookback_bars": 50},
                "improvement": {"autonomy_mode": "autonomous_nonprod"},
            },
        )

        assert cfg.strategy.lookback_bars == 50
        assert cfg.trading.mode == "paper"
        assert cfg.improvement.autonomy_mode == "autonomous_nonprod"

    def test_load_nonexistent_file_uses_pydantic_defaults(self):
        cfg = load_config("nonexistent_file_that_does_not_exist.yaml")

        assert isinstance(cfg, AppConfig)
        assert cfg.trading.mode == "paper"
        assert cfg.risk.max_risk_per_trade_pct == 0.005


class TestLiveTradingSafety:
    def test_live_mode_with_enable_live_false_raises_value_error(self):
        with pytest.raises(ValueError, match="enable_live is False"):
            AppConfig(
                trading=TradingConfig(mode="live", enable_live=False),
            )

    def test_paper_mode_with_enable_live_false_is_allowed(self):
        cfg = AppConfig(
            trading=TradingConfig(mode="paper", enable_live=False),
        )

        assert cfg.trading.mode == "paper"

    def test_backtest_mode_with_enable_live_false_is_allowed(self):
        cfg = AppConfig(
            trading=TradingConfig(mode="backtest", enable_live=False),
        )

        assert cfg.trading.mode == "backtest"


class TestDeepMerge:
    def test_flat_keys_are_merged(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = _deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_dicts_are_recursively_merged(self):
        base = {"x": {"a": 1, "b": 2}, "y": 10}
        override = {"x": {"b": 99, "c": 3}}

        result = _deep_merge(base, override)

        assert result["x"] == {"a": 1, "b": 99, "c": 3}
        assert result["y"] == 10

    def test_override_replaces_non_dict_with_dict(self):
        base = {"x": 42}
        override = {"x": {"nested": True}}

        result = _deep_merge(base, override)

        assert result["x"] == {"nested": True}

    def test_override_replaces_dict_with_scalar(self):
        base = {"x": {"nested": True}}
        override = {"x": 42}

        result = _deep_merge(base, override)

        assert result["x"] == 42

    def test_empty_override_returns_base_unchanged(self):
        base = {"a": 1, "b": {"c": 2}}

        result = _deep_merge(base, {})

        assert result == {"a": 1, "b": {"c": 2}}


class TestImprovementValidation:
    def test_invalid_improvement_chunk_overlap_raises(self):
        with pytest.raises(ValueError, match="rag_chunk_overlap_lines"):
            AppConfig(
                improvement={
                    "rag_max_chunk_lines": 20,
                    "rag_chunk_overlap_lines": 20,
                }
            )

    def test_improvement_observe_modes_requires_values(self):
        with pytest.raises(ValueError, match="observe_modes"):
            AppConfig(
                improvement={
                    "observe_modes": [],
                }
            )


class TestSentimentValidation:
    def test_invalid_sentiment_threshold_range_raises(self):
        with pytest.raises(ValueError, match="positive_threshold"):
            AppConfig(
                sentiment={
                    "enabled": True,
                    "provider": "finbert",
                    "positive_threshold": 1.5,
                }
            )
