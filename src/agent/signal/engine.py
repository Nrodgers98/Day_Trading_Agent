"""
Signal ensemble engine — blends technical rules with ML predictions.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.agent.config import StrategyConfig
from src.agent.models import (
    FeatureVector,
    Quote,
    Signal,
    SignalAction,
    Side,
)
from src.agent.signal.ml_model import MLSignalModel
from src.agent.signal.technical import TechnicalSignalGenerator

logger = logging.getLogger(__name__)

_TIMEFRAME_RE = re.compile(r"^(\d+)(m|h|d)$")

_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440}


def _timeframe_to_minutes(tf: str) -> int:
    m = _TIMEFRAME_RE.match(tf)
    if not m:
        return 5
    return int(m.group(1)) * _UNIT_MINUTES[m.group(2)]


class SignalEngine:
    """Weighted ensemble of rule-based and ML signal generators."""

    def __init__(
        self,
        config: StrategyConfig,
        technical: TechnicalSignalGenerator | None = None,
        ml_model: MLSignalModel | None = None,
        *,
        technical_weight: float = 0.4,
        ml_weight: float = 0.6,
        spread_guard_pct: float = 0.001,
        sentiment_positive_threshold: float = 0.25,
        sentiment_negative_threshold: float = -0.25,
        sentiment_confidence_boost: float = 0.05,
    ) -> None:
        self._cfg = config
        self._tech = technical or TechnicalSignalGenerator(config, spread_guard_pct)
        self._ml = ml_model or MLSignalModel()
        self._w_tech = technical_weight
        self._w_ml = ml_weight
        self._sent_pos = sentiment_positive_threshold
        self._sent_neg = sentiment_negative_threshold
        self._sent_boost = sentiment_confidence_boost

        self._bar_minutes = _timeframe_to_minutes(config.timeframe)
        self._cooldown_delta = timedelta(
            minutes=config.cooldown_bars * self._bar_minutes
        )

        self._last_trade_time: dict[str, datetime] = {}
        self._daily_trade_counts: dict[str, int] = defaultdict(int)
        self._current_trade_day: str = ""

    # ── public API ────────────────────────────────────────────────────

    def generate_signal(
        self,
        symbol: str,
        bars_df: pd.DataFrame,
        features: FeatureVector,
        quote: Quote | None = None,
    ) -> Signal:
        now = datetime.now(tz=timezone.utc)
        self._roll_day(now)

        if self._in_cooldown(symbol, now):
            return self._hold(symbol, now, features, "cooldown active")

        if self._daily_trade_counts[symbol] >= self._cfg.max_trades_per_day:
            return self._hold(symbol, now, features, "max daily trades reached")

        tech_signal = self._tech.evaluate(bars_df, features, quote)

        if not self._ml.is_trained:
            final_signal = self._apply_sentiment_overlay(
                self._finalize_technical_only(tech_signal, now),
                now,
            )
            if final_signal.action != SignalAction.HOLD:
                self._record_trade(symbol, now)
            return final_signal

        ml_side, ml_conf = self._ml.predict(features)
        blended = self._blend(tech_signal, ml_side, ml_conf)

        if blended.confidence < self._cfg.ml_confidence_threshold:
            return self._hold(
                symbol,
                now,
                features,
                f"confidence {blended.confidence:.2f} below "
                f"threshold {self._cfg.ml_confidence_threshold:.2f}",
            )

        final_signal = self._apply_sentiment_overlay(blended, now)
        if final_signal.action != SignalAction.HOLD:
            self._record_trade(symbol, now)
        return final_signal

    # ── blending ──────────────────────────────────────────────────────

    def _blend(
        self,
        tech: Signal,
        ml_side: Side,
        ml_conf: float,
    ) -> Signal:
        tech_conf = tech.confidence
        combined = self._w_tech * tech_conf + self._w_ml * ml_conf

        if tech.side == ml_side:
            action = tech.action
            side = tech.side
        elif ml_conf > tech_conf:
            side = ml_side
            action = _side_to_action(ml_side)
        else:
            side = tech.side
            action = tech.action

        if side == Side.FLAT:
            action = SignalAction.HOLD

        return Signal(
            symbol=tech.symbol,
            timestamp=tech.timestamp,
            action=action,
            side=side,
            confidence=round(combined, 4),
            rule_score=round(tech_conf, 4),
            ml_score=round(ml_conf, 4),
            features=tech.features,
            reason=f"ensemble tech={tech_conf:.2f} ml={ml_conf:.2f}",
        )

    def _finalize_technical_only(self, tech: Signal, now: datetime) -> Signal:
        """When no ML model is available, use technical signal directly."""
        if tech.action == SignalAction.HOLD:
            return tech

        if tech.confidence < self._cfg.ml_confidence_threshold:
            return self._hold(
                tech.symbol,
                now,
                tech.features,
                f"tech-only confidence {tech.confidence:.2f} below threshold",
            )

        return tech.model_copy(
            update={"reason": f"tech-only: {tech.reason}"},
        )

    def _apply_sentiment_overlay(self, signal: Signal, now: datetime) -> Signal:
        if signal.action == SignalAction.HOLD or signal.features is None:
            return signal.model_copy(
                update={
                    "metadata": {
                        **signal.metadata,
                        "sentiment_effect": "none",
                        "sentiment_score": signal.features.sentiment_score if signal.features else None,
                    }
                }
            )

        sentiment = signal.features.sentiment_score
        if sentiment is None:
            return signal.model_copy(
                update={
                    "metadata": {
                        **signal.metadata,
                        "sentiment_effect": "unavailable",
                        "sentiment_score": None,
                    }
                }
            )

        if signal.action == SignalAction.BUY and sentiment <= self._sent_neg:
            return self._hold(
                signal.symbol,
                now,
                signal.features,
                (
                    f"sentiment veto for long "
                    f"({sentiment:.3f} <= {self._sent_neg:.3f})"
                ),
            ).model_copy(
                update={
                    "metadata": {
                        **signal.metadata,
                        "sentiment_effect": "veto_long",
                        "sentiment_score": sentiment,
                    }
                }
            )
        if signal.action == SignalAction.SELL and sentiment >= self._sent_pos:
            return self._hold(
                signal.symbol,
                now,
                signal.features,
                (
                    f"sentiment veto for short "
                    f"({sentiment:.3f} >= {self._sent_pos:.3f})"
                ),
            ).model_copy(
                update={
                    "metadata": {
                        **signal.metadata,
                        "sentiment_effect": "veto_short",
                        "sentiment_score": sentiment,
                    }
                }
            )

        aligned = (
            (signal.action == SignalAction.BUY and sentiment > 0)
            or (signal.action == SignalAction.SELL and sentiment < 0)
        )
        if aligned:
            new_conf = min(1.0, signal.confidence + self._sent_boost * min(1.0, abs(sentiment)))
            return signal.model_copy(
                update={
                    "confidence": round(new_conf, 4),
                    "reason": f"{signal.reason}; sentiment_aligned={sentiment:.3f}",
                    "metadata": {
                        **signal.metadata,
                        "sentiment_effect": "aligned_boost",
                        "sentiment_score": sentiment,
                    },
                }
            )

        return signal.model_copy(
            update={
                "reason": f"{signal.reason}; sentiment={sentiment:.3f}",
                "metadata": {
                    **signal.metadata,
                    "sentiment_effect": "context_only",
                    "sentiment_score": sentiment,
                },
            }
        )

    # ── cooldown / trade tracking ─────────────────────────────────────

    def _in_cooldown(self, symbol: str, now: datetime) -> bool:
        last = self._last_trade_time.get(symbol)
        if last is None:
            return False
        return (now - last) < self._cooldown_delta

    def _record_trade(self, symbol: str, now: datetime) -> None:
        self._last_trade_time[symbol] = now
        self._daily_trade_counts[symbol] += 1

    def _roll_day(self, now: datetime) -> None:
        today = now.strftime("%Y-%m-%d")
        if today != self._current_trade_day:
            self._current_trade_day = today
            self._daily_trade_counts.clear()

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _hold(
        symbol: str,
        ts: datetime,
        features: FeatureVector | None,
        reason: str,
    ) -> Signal:
        return Signal(
            symbol=symbol,
            timestamp=ts,
            action=SignalAction.HOLD,
            side=Side.FLAT,
            confidence=0.0,
            features=features,
            reason=reason,
        )


def _side_to_action(side: Side) -> SignalAction:
    if side == Side.LONG:
        return SignalAction.BUY
    if side == Side.SHORT:
        return SignalAction.SELL
    return SignalAction.HOLD
