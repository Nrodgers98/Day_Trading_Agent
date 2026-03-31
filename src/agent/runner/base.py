"""Abstract base runner — lifecycle hooks and session/market-hour helpers."""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, time, timedelta
from typing import Any

import pandas_market_calendars as mcal
import pytz

from src.agent.config import AppConfig

logger = logging.getLogger("trading_agent")

_TIMEFRAME_RE = re.compile(r"^(\d+)(m|h|d)$")
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}


def _timeframe_to_seconds(tf: str) -> int:
    m = _TIMEFRAME_RE.match(tf)
    if not m:
        return 300
    return int(m.group(1)) * _UNIT_SECONDS[m.group(2)]


def _parse_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


class BaseRunner(ABC):
    """Defines the run lifecycle: ``setup → run_loop → shutdown``.

    Subclasses implement the three abstract hooks while inheriting session
    timing and market-calendar helpers.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._tz = pytz.timezone(config.session.timezone)
        self._nyse = mcal.get_calendar("NYSE")

        self._market_open = _parse_time(config.session.market_open)
        self._market_close = _parse_time(config.session.market_close)
        self._opening_guard = timedelta(minutes=config.session.opening_guard_minutes)
        self._closing_guard = timedelta(minutes=config.session.closing_guard_minutes)
        self._eod_flatten_time = (
            _parse_time(config.session.eod_flatten_time)
            if config.session.eod_flatten
            else None
        )

        self._loop_interval_s = _timeframe_to_seconds(config.strategy.timeframe)

    # ── abstract hooks ────────────────────────────────────────────────

    @abstractmethod
    async def setup(self) -> None: ...

    @abstractmethod
    async def run_loop(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    # ── lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Entry point: setup → run_loop → shutdown (guaranteed)."""
        logger.info(
            "Runner starting | mode=%s interval=%ds",
            self._config.trading.mode,
            self._loop_interval_s,
        )
        try:
            await self.setup()
            await self.run_loop()
        finally:
            await self.shutdown()
            logger.info("Runner shut down cleanly")

    # ── session / market helpers ──────────────────────────────────────

    def _now_eastern(self) -> datetime:
        return datetime.now(tz=self._tz)

    def _check_session(self) -> bool:
        """True when the current time falls within the guarded trading window."""
        now = self._now_eastern()
        t = now.time()

        open_dt = datetime.combine(now.date(), self._market_open)
        guarded_open = (open_dt + self._opening_guard).time()

        close_dt = datetime.combine(now.date(), self._market_close)
        guarded_close = (close_dt - self._closing_guard).time()

        return guarded_open <= t <= guarded_close

    def _should_flatten(self) -> bool:
        """True when the current time is at or past the EOD flatten time."""
        if self._eod_flatten_time is None:
            return False
        return self._now_eastern().time() >= self._eod_flatten_time

    def _is_market_open(self) -> bool:
        """True when today is a NYSE trading day and we are within market hours."""
        now = self._now_eastern()
        today_str = now.strftime("%Y-%m-%d")
        schedule = self._nyse.schedule(start_date=today_str, end_date=today_str)

        if schedule.empty:
            return False

        market_open_utc = schedule.iloc[0]["market_open"].to_pydatetime()
        market_close_utc = schedule.iloc[0]["market_close"].to_pydatetime()
        now_utc = now.astimezone(pytz.utc)

        return market_open_utc <= now_utc <= market_close_utc
