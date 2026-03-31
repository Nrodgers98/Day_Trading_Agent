"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.agent.config import (
    AppConfig, TradingConfig, UniverseConfig, SessionConfig,
    StrategyConfig, RiskConfig, AlpacaConfig, BacktestConfig,
    MonitoringConfig,
)
from src.agent.models import (
    Bar, Quote, FeatureVector, Signal, SignalAction, Side,
    OrderRequest, OrderResult, OrderStatus, OrderType,
    AccountInfo, Position, RiskVerdict, TimeInForce,
)


@pytest.fixture
def sample_config() -> AppConfig:
    return AppConfig(
        trading=TradingConfig(mode="paper", enable_live=False),
        risk=RiskConfig(
            max_risk_per_trade_pct=0.005,
            max_daily_drawdown_pct=0.02,
            max_concurrent_positions=10,
            daily_trade_cap=30,
        ),
        strategy=StrategyConfig(
            lookback_bars=20,
            volume_surge_ratio=1.5,
            ml_confidence_threshold=0.6,
        ),
    )


@pytest.fixture
def sample_bars_df() -> pd.DataFrame:
    """Generate 100 bars of synthetic OHLCV data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="US/Eastern")
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(10000, 500000, n).astype(float)
    vwap = (high + low + close) / 3.0

    return pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": vwap,
    }).set_index("timestamp")


@pytest.fixture
def sample_features() -> FeatureVector:
    return FeatureVector(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc),
        returns_1=0.002,
        returns_5=0.005,
        vwap_distance=0.001,
        rsi_14=55.0,
        ema_9=150.5,
        ema_21=150.0,
        sma_50=149.0,
        atr_14=1.5,
        volume_ratio=1.8,
        trend_slope=0.05,
    )


@pytest.fixture
def sample_account() -> AccountInfo:
    return AccountInfo(
        equity=100_000.0,
        cash=50_000.0,
        buying_power=100_000.0,
        portfolio_value=100_000.0,
        day_trade_count=0,
        pdt_flag=False,
        status="ACTIVE",
    )


@pytest.fixture
def sample_quote() -> Quote:
    return Quote(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc),
        bid_price=150.00,
        bid_size=100,
        ask_price=150.05,
        ask_size=200,
    )


@pytest.fixture
def sample_order_request() -> OrderRequest:
    return OrderRequest(
        symbol="AAPL",
        side=Side.LONG,
        qty=10.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )
