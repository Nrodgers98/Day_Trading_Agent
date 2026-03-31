"""Alert system — drawdown, order-failure, and data-gap monitoring."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import ClassVar

from src.agent.config import MonitoringConfig

logger = logging.getLogger("trading_agent")

_COOLDOWN = timedelta(minutes=5)


class AlertManager:
    """Monitors key health metrics and emits throttled structured alerts."""

    DRAWDOWN: ClassVar[str] = "drawdown"
    ORDER_FAILURE: ClassVar[str] = "order_failure"
    DATA_GAP: ClassVar[str] = "data_gap"

    def __init__(self, config: MonitoringConfig) -> None:
        self._config = config
        self._last_alert: dict[str, datetime] = {}

    def _should_alert(self, alert_type: str, now: datetime | None = None) -> bool:
        now = now or datetime.utcnow()
        last = self._last_alert.get(alert_type)
        if last is not None and (now - last) < _COOLDOWN:
            return False
        self._last_alert[alert_type] = now
        return True

    def check_drawdown(
        self,
        current_equity: float,
        starting_equity: float,
    ) -> None:
        """Log a CRITICAL alert if daily drawdown exceeds the configured limit."""
        if not self._config.alert_on_drawdown or starting_equity <= 0:
            return

        drawdown_pct = (starting_equity - current_equity) / starting_equity

        if drawdown_pct <= 0:
            return

        if drawdown_pct >= 0.02 and self._should_alert(self.DRAWDOWN):
            logger.critical(
                "Drawdown alert: %.2f%% drawdown (equity $%.2f -> $%.2f)",
                drawdown_pct * 100,
                starting_equity,
                current_equity,
                extra={
                    "alert_type": self.DRAWDOWN,
                    "drawdown_pct": round(drawdown_pct, 6),
                    "current_equity": current_equity,
                    "starting_equity": starting_equity,
                },
            )

    def check_order_failures(self, consecutive: int) -> None:
        """Log a CRITICAL alert if consecutive order failures exceed the configured max."""
        if not self._config.alert_on_order_failure:
            return

        if consecutive > self._config.max_consecutive_failures and self._should_alert(
            self.ORDER_FAILURE,
        ):
            logger.critical(
                "Order failure alert: %d consecutive failures (max %d)",
                consecutive,
                self._config.max_consecutive_failures,
                extra={
                    "alert_type": self.ORDER_FAILURE,
                    "consecutive_failures": consecutive,
                    "max_allowed": self._config.max_consecutive_failures,
                },
            )

    def check_data_gap(
        self,
        symbol: str,
        last_bar_time: datetime,
        now: datetime,
        max_gap_minutes: int = 10,
    ) -> None:
        """Log a CRITICAL alert if the gap since the last bar exceeds the threshold."""
        if not self._config.alert_on_data_gap:
            return

        gap = now - last_bar_time
        if gap > timedelta(minutes=max_gap_minutes) and self._should_alert(
            self.DATA_GAP,
        ):
            logger.critical(
                "Data gap alert: %s has no data for %.1f minutes (max %d)",
                symbol,
                gap.total_seconds() / 60,
                max_gap_minutes,
                extra={
                    "alert_type": self.DATA_GAP,
                    "symbol": symbol,
                    "gap_minutes": round(gap.total_seconds() / 60, 2),
                    "max_gap_minutes": max_gap_minutes,
                    "last_bar_time": last_bar_time.isoformat(),
                },
            )
