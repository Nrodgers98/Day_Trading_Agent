"""
Async Alpaca v2 market-data and trading-API client.

All timestamps are normalised to US/Eastern.
Uses httpx + tenacity for retries with exponential backoff and jitter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import pytz
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.agent.config import AlpacaConfig
from src.agent.models import AccountInfo, Position, Quote, Side

logger = logging.getLogger(__name__)

EASTERN = pytz.timezone("US/Eastern")


def _is_retryable(exc: BaseException) -> bool:
    """Only retry on transient transport errors and server-side / throttle codes."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return False


# ── Rate limiter ──────────────────────────────────────────────────────


class _RateLimiter:
    """Sliding-window rate limiter (async-safe)."""

    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        self._max_calls = max_calls
        self._period = period
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._calls = [t for t in self._calls if now - t < self._period]
            if len(self._calls) >= self._max_calls:
                wait_for = self._period - (now - self._calls[0])
                if wait_for > 0:
                    logger.warning("Rate limit reached — sleeping %.2fs", wait_for)
                    await asyncio.sleep(wait_for)
                    now = time.monotonic()
                    self._calls = [
                        t for t in self._calls if now - t < self._period
                    ]
            self._calls.append(time.monotonic())


# ── Client ────────────────────────────────────────────────────────────


class AlpacaMarketDataClient:
    """Async client wrapping Alpaca v2 data and trading REST APIs."""

    def __init__(self, config: AlpacaConfig) -> None:
        self._cfg = config
        headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.secret_key,
        }
        self._data = httpx.AsyncClient(
            base_url=config.data_url,
            headers=headers,
            timeout=config.api_timeout_seconds,
        )
        self._trading = httpx.AsyncClient(
            base_url=config.base_url,
            headers=headers,
            timeout=config.api_timeout_seconds,
        )
        self._limiter = _RateLimiter(config.rate_limit_calls_per_minute)

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._data.aclose()
        await self._trading.aclose()

    async def __aenter__(self) -> AlpacaMarketDataClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Internal HTTP helpers ─────────────────────────────────────────

    async def _data_get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Any:
        @retry(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=self._cfg.retry_base_delay,
                max=self._cfg.retry_max_delay,
            ),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _call() -> Any:
            await self._limiter.acquire()
            resp = await self._data.get(path, params=params)
            resp.raise_for_status()
            return resp.json()

        return await _call()

    async def _trading_get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Any:
        @retry(
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=self._cfg.retry_base_delay,
                max=self._cfg.retry_max_delay,
            ),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _call() -> Any:
            await self._limiter.acquire()
            resp = await self._trading.get(path, params=params)
            resp.raise_for_status()
            return resp.json()

        return await _call()

    # ── Timestamp normalisation ───────────────────────────────────────

    @staticmethod
    def _to_eastern(ts: str | datetime) -> datetime:
        dt = pd.Timestamp(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert(EASTERN).to_pydatetime()

    # ── Market Data API ───────────────────────────────────────────────

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "5Min",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        limit: int = 1_000,
        adjustment: str = "raw",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars with automatic pagination.

        Parameters
        ----------
        symbol:     Ticker symbol (e.g. ``"AAPL"``).
        timeframe:  Alpaca timeframe string — ``"1Min"``, ``"5Min"``, ``"15Min"``, etc.
        start/end:  ISO-8601 strings or datetime objects.
        limit:      Max bars per page (Alpaca caps at 10 000).
        adjustment: ``"raw"`` | ``"split"`` | ``"dividend"`` | ``"all"``.
        """
        params: dict[str, Any] = {
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": adjustment,
        }
        if start is not None:
            params["start"] = (
                start.isoformat() if isinstance(start, datetime) else start
            )
        if end is not None:
            params["end"] = (
                end.isoformat() if isinstance(end, datetime) else end
            )

        all_bars: list[dict[str, Any]] = []
        page_token: str | None = None
        seen_tokens: set[str] = set()
        page_count = 0
        max_pages = 10_000

        while True:
            if page_token:
                if page_token in seen_tokens:
                    logger.error(
                        "Pagination token repeated for %s; breaking to avoid infinite loop",
                        symbol,
                    )
                    break
                seen_tokens.add(page_token)
                params["page_token"] = page_token

            body = await self._data_get(f"/v2/stocks/{symbol}/bars", params)
            raw_bars: list[dict[str, Any]] = body.get("bars") or []

            for b in raw_bars:
                all_bars.append(
                    {
                        "timestamp": self._to_eastern(b["t"]),
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low": float(b["l"]),
                        "close": float(b["c"]),
                        "volume": float(b["v"]),
                        "vwap": float(b["vw"]) if "vw" in b else None,
                        "trade_count": int(b["n"]) if "n" in b else None,
                    }
                )

            page_token = body.get("next_page_token")
            page_count += 1
            if page_count >= max_pages:
                logger.error(
                    "Exceeded max pagination pages (%d) for %s; breaking",
                    max_pages,
                    symbol,
                )
                break
            if not page_token:
                break

        if not all_bars:
            return pd.DataFrame(
                columns=[
                    "timestamp", "open", "high", "low",
                    "close", "volume", "vwap", "trade_count",
                ]
            )

        df = pd.DataFrame(all_bars)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    async def get_latest_quote(self, symbol: str) -> Quote:
        """Fetch the latest NBBO quote for *symbol*."""
        body = await self._data_get(f"/v2/stocks/{symbol}/quotes/latest")
        q = body["quote"]
        return Quote(
            symbol=symbol,
            timestamp=self._to_eastern(q["t"]),
            bid_price=float(q["bp"]),
            bid_size=float(q["bs"]),
            ask_price=float(q["ap"]),
            ask_size=float(q["as"]),
        )

    async def get_news(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fetch recent news articles for a symbol from Alpaca data API."""
        start_dt = start or (datetime.now(tz=timezone.utc) - timedelta(hours=8))
        end_dt = end or datetime.now(tz=timezone.utc)
        params = {
            "symbols": symbol,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "limit": limit,
            "sort": "desc",
        }
        body = await self._data_get("/v1beta1/news", params=params)
        news = body.get("news")
        return news if isinstance(news, list) else []

    # ── Trading API ───────────────────────────────────────────────────

    async def get_account(self) -> AccountInfo:
        """Return current account equity, cash, and PDT status."""
        data = await self._trading_get("/v2/account")
        return AccountInfo(
            equity=float(data.get("equity", 0)),
            cash=float(data.get("cash", 0)),
            buying_power=float(data.get("buying_power", 0)),
            portfolio_value=float(data.get("portfolio_value", 0)),
            day_trade_count=int(data.get("daytrade_count", 0)),
            pdt_flag=data.get("pattern_day_trader", False),
            status=data.get("status", "ACTIVE"),
        )

    async def get_assets(
        self,
        status: str = "active",
        asset_class: str = "us_equity",
    ) -> list[dict[str, Any]]:
        """Return all assets matching *status* and *asset_class*."""
        return await self._trading_get(
            "/v2/assets", {"status": status, "asset_class": asset_class}
        )

    async def get_positions(self) -> list[Position]:
        """Return every open position in the account."""
        raw: list[dict[str, Any]] = await self._trading_get("/v2/positions")
        positions: list[Position] = []
        for p in raw:
            side_str = p.get("side", "long")
            positions.append(
                Position(
                    symbol=p["symbol"],
                    side=Side.LONG if side_str == "long" else Side.SHORT,
                    qty=abs(float(p.get("qty", 0))),
                    avg_entry_price=float(p.get("avg_entry_price", 0)),
                    current_price=float(p.get("current_price", 0)),
                    market_value=float(p.get("market_value", 0)),
                    unrealized_pnl=float(p.get("unrealized_pl", 0)),
                    unrealized_pnl_pct=float(p.get("unrealized_plpc", 0)),
                )
            )
        return positions

    async def check_asset(self, symbol: str) -> dict[str, Any]:
        """Return asset details for *symbol*.  Raises on 404."""
        return await self._trading_get(f"/v2/assets/{symbol}")
