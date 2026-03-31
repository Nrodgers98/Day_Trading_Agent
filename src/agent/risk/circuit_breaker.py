from __future__ import annotations

import logging

from src.agent.config import MonitoringConfig, RiskConfig

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, risk_cfg: RiskConfig, mon_cfg: MonitoringConfig) -> None:
        self._risk_cfg = risk_cfg
        self._mon_cfg = mon_cfg

        self.starting_equity: float = 0.0
        self.daily_pnl: float = 0.0
        self.trade_count_today: int = 0
        self.consecutive_failures: int = 0
        self.kill_switch_active: bool = False
        self._kill_reason: str = ""

    # ── checks ────────────────────────────────────────────────────────

    def check_daily_drawdown(self, current_equity: float) -> bool:
        """Return *True* if the daily drawdown limit has been breached."""
        if self.starting_equity <= 0:
            return False
        drawdown = (self.starting_equity - current_equity) / self.starting_equity
        return drawdown >= self._risk_cfg.max_daily_drawdown_pct

    def check_trade_cap(self) -> bool:
        """Return *True* if the daily trade cap has been reached."""
        return self.trade_count_today >= self._risk_cfg.daily_trade_cap

    # ── failure / success tracking ────────────────────────────────────

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures > self._mon_cfg.max_consecutive_failures:
            self.activate_kill_switch(
                f"Consecutive failures ({self.consecutive_failures}) "
                f"exceeded limit ({self._mon_cfg.max_consecutive_failures})"
            )

    def record_success(self) -> None:
        self.consecutive_failures = 0

    # ── kill switch ───────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str) -> None:
        self.kill_switch_active = True
        self._kill_reason = reason
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def is_halted(self) -> bool:
        return self.kill_switch_active

    # ── daily reset ───────────────────────────────────────────────────

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.trade_count_today = 0
        self.consecutive_failures = 0
        self.kill_switch_active = False
        self._kill_reason = ""
