"""
Rule-based technical signal generator — momentum breakout strategy.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from src.agent.config import StrategyConfig
from src.agent.models import (
    FeatureVector,
    Quote,
    Signal,
    SignalAction,
    Side,
)

logger = logging.getLogger(__name__)


class TechnicalSignalGenerator:
    """Generates signals from price-action rules defined in *StrategyConfig*."""

    def __init__(self, config: StrategyConfig, spread_guard_pct: float = 0.001) -> None:
        self._cfg = config
        self._spread_guard_pct = spread_guard_pct

    # ── public API ────────────────────────────────────────────────────

    def evaluate(
        self,
        bars_df: pd.DataFrame,
        features: FeatureVector,
        latest_quote: Quote | None = None,
    ) -> Signal:
        """Score current conditions and return a *Signal*."""
        now = datetime.now(tz=timezone.utc)
        symbol = features.symbol
        hold = self._hold_signal(symbol, now, features)

        if bars_df.empty or len(bars_df) < self._cfg.lookback_bars:
            return hold

        close = float(bars_df["close"].iloc[-1])
        lookback = bars_df.iloc[-(self._cfg.lookback_bars + 1) : -1]
        highest_high = float(lookback["high"].max())
        lowest_low = float(lookback["low"].min())

        if latest_quote and not self._spread_ok(latest_quote):
            return hold.model_copy(update={"reason": "spread too wide"})

        long_conditions = self._check_long(close, highest_high, features, bars_df)
        short_conditions = self._check_short(close, lowest_low, features, bars_df)

        long_score = sum(long_conditions.values())
        short_score = sum(short_conditions.values())
        total = len(long_conditions)

        if long_score > short_score and long_conditions["breakout"]:
            confidence = long_score / total
            reasons = [k for k, v in long_conditions.items() if v]
            return Signal(
                symbol=symbol,
                timestamp=now,
                action=SignalAction.BUY,
                side=Side.LONG,
                confidence=confidence,
                rule_score=confidence,
                features=features,
                reason=f"long breakout ({', '.join(reasons)})",
            )

        if short_score > long_score and short_conditions["breakout"]:
            confidence = short_score / total
            reasons = [k for k, v in short_conditions.items() if v]
            return Signal(
                symbol=symbol,
                timestamp=now,
                action=SignalAction.SELL,
                side=Side.SHORT,
                confidence=confidence,
                rule_score=confidence,
                features=features,
                reason=f"short breakout ({', '.join(reasons)})",
            )

        return hold

    # ── condition checkers ────────────────────────────────────────────

    def _check_long(
        self,
        close: float,
        highest_high: float,
        features: FeatureVector,
        bars_df: pd.DataFrame,
    ) -> dict[str, bool]:
        rsi_lo, rsi_hi = self._cfg.rsi_long_range
        conditions: dict[str, bool] = {
            "breakout": close > highest_high,
            "volume_surge": features.volume_ratio > self._cfg.volume_surge_ratio,
            "rsi_range": rsi_lo <= features.rsi_14 <= rsi_hi,
        }
        if "vwap" in bars_df.columns:
            vwap = bars_df["vwap"].iloc[-1]
            if pd.notna(vwap) and vwap > 0:
                conditions["above_vwap"] = close > float(vwap)
        return conditions

    def _check_short(
        self,
        close: float,
        lowest_low: float,
        features: FeatureVector,
        bars_df: pd.DataFrame,
    ) -> dict[str, bool]:
        rsi_lo, rsi_hi = self._cfg.rsi_short_range
        conditions: dict[str, bool] = {
            "breakout": close < lowest_low,
            "volume_surge": features.volume_ratio > self._cfg.volume_surge_ratio,
            "rsi_range": rsi_lo <= features.rsi_14 <= rsi_hi,
        }
        if "vwap" in bars_df.columns:
            vwap = bars_df["vwap"].iloc[-1]
            if pd.notna(vwap) and vwap > 0:
                conditions["below_vwap"] = close < float(vwap)
        return conditions

    # ── helpers ───────────────────────────────────────────────────────

    def _spread_ok(self, quote: Quote) -> bool:
        return quote.spread_pct <= self._spread_guard_pct

    def _hold_signal(
        self, symbol: str, ts: datetime, features: FeatureVector
    ) -> Signal:
        return Signal(
            symbol=symbol,
            timestamp=ts,
            action=SignalAction.HOLD,
            side=Side.FLAT,
            confidence=0.0,
            rule_score=0.0,
            features=features,
            reason="no breakout",
        )
