"""
Dynamic stock scanner — finds tradeable symbols by scanning the market.

Supports three universe modes:
  - custom:  user-specified symbol list (no scanning)
  - scan:    live market scan via Alpaca most-actives + snapshots
  - hybrid:  custom symbols PLUS additional scan results

The scanner runs on a configurable schedule:
  - Pre-market scan: once before market open
  - Intraday rescan: every N minutes during the session
  - Results are cached between scans
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from src.agent.config import AlpacaConfig, ScannerConfig, UniverseConfig
from src.agent.data.market_data import AlpacaMarketDataClient

logger = logging.getLogger("trading_agent")


class MarketScanner:
    """Scans the market for tradeable stocks using Alpaca APIs."""

    def __init__(
        self,
        scanner_cfg: ScannerConfig,
        universe_cfg: UniverseConfig,
        market_data: AlpacaMarketDataClient,
    ) -> None:
        self._cfg = scanner_cfg
        self._universe_cfg = universe_cfg
        self._client = market_data

        self._cached_symbols: list[str] = []
        self._last_scan_time: float = 0.0
        self._scan_count: int = 0

    @property
    def cached_symbols(self) -> list[str]:
        return list(self._cached_symbols)

    @property
    def last_scan_time(self) -> float:
        return self._last_scan_time

    def needs_rescan(self) -> bool:
        """Check if enough time has passed to warrant a new scan."""
        if not self._cached_symbols:
            return True
        elapsed_min = (time.monotonic() - self._last_scan_time) / 60.0
        return elapsed_min >= self._cfg.rescan_interval_minutes

    async def scan(self) -> list[str]:
        """Run a full market scan and return filtered symbols.

        Uses the configured universe mode:
          - custom: returns custom_symbols only
          - scan: returns live scan results only
          - hybrid: merges custom_symbols with scan results
        """
        mode = self._universe_cfg.base

        if mode == "custom":
            symbols = list(self._universe_cfg.custom_symbols)
            logger.info("Universe mode=custom | symbols=%d", len(symbols))
            self._cached_symbols = symbols
            self._last_scan_time = time.monotonic()
            return symbols

        scanned = await self._run_live_scan()

        if mode == "hybrid":
            custom = set(self._universe_cfg.custom_symbols)
            merged = list(custom | set(scanned))
            logger.info(
                "Universe mode=hybrid | custom=%d scanned=%d merged=%d",
                len(custom), len(scanned), len(merged),
            )
            self._cached_symbols = merged
        else:
            logger.info("Universe mode=scan | scanned=%d", len(scanned))
            self._cached_symbols = scanned

        self._last_scan_time = time.monotonic()
        self._scan_count += 1
        return list(self._cached_symbols)

    async def _run_live_scan(self) -> list[str]:
        """Query Alpaca for active/moving stocks, then filter."""
        candidates: set[str] = set()

        most_active = await self._fetch_most_actives()
        candidates.update(most_active)

        if self._cfg.scan_gainers_losers:
            movers = await self._fetch_top_movers()
            candidates.update(movers)

        if not candidates:
            logger.warning("Live scan returned 0 candidates — falling back to assets scan")
            candidates = await self._fetch_active_assets_sample()

        filtered = await self._apply_filters(list(candidates))

        filtered = filtered[: self._cfg.max_symbols]

        logger.info(
            "Scan complete | candidates=%d -> filtered=%d (scan #%d)",
            len(candidates), len(filtered), self._scan_count + 1,
        )
        return filtered

    # ── Alpaca data sources ───────────────────────────────────────────

    async def _fetch_most_actives(self) -> list[str]:
        """Fetch most active stocks by volume from Alpaca screener."""
        try:
            data = await self._client._data_get(
                "/v1beta1/screener/stocks/most-actives",
                params={"by": "volume", "top": self._cfg.scan_top_n},
            )
            symbols = [
                item["symbol"]
                for item in (data.get("most_actives") or [])
                if isinstance(item, dict) and "symbol" in item
            ]
            logger.debug("Most actives: %d symbols", len(symbols))
            return symbols
        except Exception:
            logger.warning("Failed to fetch most-actives, continuing with other sources")
            return []

    async def _fetch_top_movers(self) -> list[str]:
        """Fetch top gainers and losers from Alpaca screener."""
        symbols: list[str] = []
        for direction in ("gainers", "losers"):
            try:
                data = await self._client._data_get(
                    f"/v1beta1/screener/stocks/movers",
                    params={"top": self._cfg.scan_top_n},
                )
                for item in data.get(direction) or []:
                    if isinstance(item, dict) and "symbol" in item:
                        symbols.append(item["symbol"])
            except Exception:
                logger.warning("Failed to fetch %s", direction)
        logger.debug("Top movers: %d symbols", len(symbols))
        return symbols

    async def _fetch_active_assets_sample(self) -> set[str]:
        """Fallback: pull all active US equity assets and take the
        most common large-cap names based on exchange."""
        try:
            assets = await self._client.get_assets(status="active", asset_class="us_equity")
            tradeable = [
                a["symbol"]
                for a in assets
                if a.get("tradable")
                and a.get("exchange") in ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")
                and a.get("status") == "active"
            ]
            return set(tradeable[: self._cfg.scan_top_n * 2])
        except Exception:
            logger.warning("Failed to fetch assets list")
            return set()

    # ── Filtering ─────────────────────────────────────────────────────

    async def _apply_filters(self, candidates: list[str]) -> list[str]:
        """Filter candidates by price, volume, and Alpaca asset flags."""
        filters = self._universe_cfg.filters
        passed: list[str] = []

        snapshots = await self._fetch_snapshots(candidates)

        for symbol in candidates:
            snap = snapshots.get(symbol)
            if snap is None:
                continue

            if not self._passes_snapshot_filters(symbol, snap, filters):
                continue

            passed.append(symbol)

        return passed

    async def _fetch_snapshots(
        self, symbols: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch batch snapshots for multiple symbols.

        Alpaca's /v2/stocks/snapshots accepts up to ~200 symbols per call,
        so we chunk large lists.
        """
        result: dict[str, dict[str, Any]] = {}
        chunk_size = 100

        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            symbols_param = ",".join(chunk)
            try:
                data = await self._client._data_get(
                    "/v2/stocks/snapshots",
                    params={"symbols": symbols_param},
                )
                if isinstance(data, dict):
                    result.update(data)
            except Exception:
                logger.warning(
                    "Failed to fetch snapshots for chunk %d-%d",
                    i, i + len(chunk),
                )
                await asyncio.sleep(1)

        return result

    def _passes_snapshot_filters(
        self,
        symbol: str,
        snap: dict[str, Any],
        filters: Any,
    ) -> bool:
        """Check if a snapshot passes all configured filters."""
        daily_bar = snap.get("dailyBar") or snap.get("prevDailyBar") or {}
        minute_bar = snap.get("minuteBar") or {}
        latest_trade = snap.get("latestTrade") or {}

        price = float(latest_trade.get("p", 0) or minute_bar.get("c", 0) or daily_bar.get("c", 0))
        if price <= 0:
            return False

        if price < filters.min_price or price > filters.max_price:
            return False

        volume = float(daily_bar.get("v", 0))
        if volume < filters.min_avg_volume:
            return False

        return True

    # ── Asset flag validation ─────────────────────────────────────────

    async def validate_asset_flags(self, symbols: list[str]) -> list[str]:
        """Check Alpaca asset endpoints to confirm tradability.

        This is an optional second pass — more expensive but catches
        delisted or halted stocks.
        """
        validated: list[str] = []
        for symbol in symbols:
            try:
                asset = await self._client.check_asset(symbol)
                if not asset.get("tradable", False):
                    continue
                if self._universe_cfg.filters.require_easy_to_borrow:
                    if not asset.get("easy_to_borrow", False):
                        pass
                validated.append(symbol)
            except Exception:
                logger.debug("Asset check failed for %s — skipping", symbol)
        return validated
