"""
Core domain models — immutable value objects and enums used across every layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


# ── Market Data ───────────────────────────────────────────────────────


class Bar(BaseModel):
    """OHLCV bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None


class Quote(BaseModel):
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        return self.spread / mid if mid > 0 else float("inf")


# ── Features ──────────────────────────────────────────────────────────


class FeatureVector(BaseModel):
    """Computed features for a single symbol at a point in time."""
    symbol: str
    timestamp: datetime
    returns_1: float = 0.0
    returns_5: float = 0.0
    vwap_distance: float = 0.0
    rsi_14: float = 50.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    sma_50: float = 0.0
    atr_14: float = 0.0
    volume_ratio: float = 1.0
    trend_slope: float = 0.0
    sentiment_score: float | None = None
    extra: dict[str, float] = Field(default_factory=dict)

    def to_array(self) -> list[float]:
        """Flatten numeric features for ML input."""
        return [
            self.returns_1,
            self.returns_5,
            self.vwap_distance,
            self.rsi_14,
            self.ema_9,
            self.ema_21,
            self.sma_50,
            self.atr_14,
            self.volume_ratio,
            self.trend_slope,
            self.sentiment_score or 0.0,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "returns_1", "returns_5", "vwap_distance", "rsi_14",
            "ema_9", "ema_21", "sma_50", "atr_14",
            "volume_ratio", "trend_slope", "sentiment_score",
        ]


# ── Signals ───────────────────────────────────────────────────────────


class Signal(BaseModel):
    """Output of the signal engine for one symbol."""
    symbol: str
    timestamp: datetime
    action: SignalAction
    side: Side
    confidence: float = 0.0
    ml_score: float = 0.0
    rule_score: float = 0.0
    features: FeatureVector | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Orders ────────────────────────────────────────────────────────────


class OrderRequest(BaseModel):
    """Intent to place an order (pre–risk-check)."""
    symbol: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal: Signal | None = None


class OrderResult(BaseModel):
    """Broker-confirmed order state."""
    broker_order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    side: Side = Side.FLAT
    qty: float = 0.0
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    order_type: OrderType = OrderType.MARKET
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


# ── Positions ─────────────────────────────────────────────────────────


class Position(BaseModel):
    symbol: str
    side: Side
    qty: float
    avg_entry_price: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None
    entry_time: datetime | None = None


# ── Account ───────────────────────────────────────────────────────────


class AccountInfo(BaseModel):
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    portfolio_value: float = 0.0
    day_trade_count: int = 0
    pdt_flag: bool = False
    status: str = "ACTIVE"


# ── Risk Decision ────────────────────────────────────────────────────


class RiskVerdict(BaseModel):
    """Output of pre-trade risk check."""
    approved: bool
    adjusted_qty: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    checks_passed: dict[str, bool] = Field(default_factory=dict)


# ── Audit ─────────────────────────────────────────────────────────────


class AuditRecord(BaseModel):
    """Full decision trace for a single signal evaluation."""
    timestamp: datetime
    symbol: str
    raw_features: dict[str, float] = Field(default_factory=dict)
    signal: Signal | None = None
    risk_verdict: RiskVerdict | None = None
    order_request: OrderRequest | None = None
    order_result: OrderResult | None = None
    notes: str = ""
