"""Tests for the MarketScanner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.agent.config import ScannerConfig, UniverseConfig, UniverseFilters
from src.agent.data.scanner import MarketScanner


@pytest.fixture
def scanner_config() -> ScannerConfig:
    return ScannerConfig(
        enabled=True,
        rescan_interval_minutes=15,
        scan_top_n=20,
        scan_gainers_losers=True,
        max_symbols=10,
    )


@pytest.fixture
def custom_universe() -> UniverseConfig:
    return UniverseConfig(
        base="custom",
        custom_symbols=["AAPL", "MSFT", "GOOGL"],
    )


@pytest.fixture
def scan_universe() -> UniverseConfig:
    return UniverseConfig(
        base="scan",
        custom_symbols=[],
        filters=UniverseFilters(min_price=5.0, max_price=500.0, min_avg_volume=500_000),
    )


@pytest.fixture
def hybrid_universe() -> UniverseConfig:
    return UniverseConfig(
        base="hybrid",
        custom_symbols=["AAPL", "TSLA"],
        filters=UniverseFilters(min_price=5.0, max_price=500.0, min_avg_volume=500_000),
    )


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client._data_get = AsyncMock()
    client.get_assets = AsyncMock(return_value=[])
    client.check_asset = AsyncMock(return_value={"tradable": True, "easy_to_borrow": True})
    return client


def _snap(price: float, volume: float) -> dict:
    """Build a minimal Alpaca snapshot dict."""
    return {
        "latestTrade": {"p": price},
        "dailyBar": {"c": price, "v": volume},
        "minuteBar": {"c": price},
    }


# ── Custom mode ──────────────────────────────────────────────────────


class TestCustomMode:
    @pytest.mark.asyncio
    async def test_returns_custom_symbols_only(self, scanner_config, custom_universe, mock_client):
        scanner = MarketScanner(scanner_config, custom_universe, mock_client)
        result = await scanner.scan()
        assert result == ["AAPL", "MSFT", "GOOGL"]
        mock_client._data_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_caches_result(self, scanner_config, custom_universe, mock_client):
        scanner = MarketScanner(scanner_config, custom_universe, mock_client)
        await scanner.scan()
        assert scanner.cached_symbols == ["AAPL", "MSFT", "GOOGL"]
        assert not scanner.needs_rescan()


# ── Scan mode ────────────────────────────────────────────────────────


class TestScanMode:
    @pytest.mark.asyncio
    async def test_scan_calls_most_actives(self, scanner_config, scan_universe, mock_client):
        movers_resp = {"gainers": [{"symbol": "XYZ"}], "losers": [{"symbol": "ABC"}]}
        mock_client._data_get.side_effect = [
            {"most_actives": [{"symbol": "NVDA"}, {"symbol": "AMD"}]},
            movers_resp,
            movers_resp,
            {"NVDA": _snap(180, 2_000_000), "AMD": _snap(120, 1_500_000),
             "XYZ": _snap(25, 800_000), "ABC": _snap(45, 600_000)},
        ]
        scanner = MarketScanner(scanner_config, scan_universe, mock_client)
        result = await scanner.scan()
        assert "NVDA" in result
        assert "AMD" in result

    @pytest.mark.asyncio
    async def test_filters_by_price(self, scanner_config, scan_universe, mock_client):
        scan_universe.filters.max_price = 100.0
        empty_movers = {"gainers": [], "losers": []}
        mock_client._data_get.side_effect = [
            {"most_actives": [{"symbol": "NVDA"}, {"symbol": "CHEAP"}]},
            empty_movers,
            empty_movers,
            {"NVDA": _snap(180, 2_000_000), "CHEAP": _snap(8, 1_000_000)},
        ]
        scanner = MarketScanner(scanner_config, scan_universe, mock_client)
        result = await scanner.scan()
        assert "NVDA" not in result
        assert "CHEAP" in result

    @pytest.mark.asyncio
    async def test_filters_by_volume(self, scanner_config, scan_universe, mock_client):
        empty_movers = {"gainers": [], "losers": []}
        mock_client._data_get.side_effect = [
            {"most_actives": [{"symbol": "HIGH_VOL"}, {"symbol": "LOW_VOL"}]},
            empty_movers,
            empty_movers,
            {"HIGH_VOL": _snap(50, 2_000_000), "LOW_VOL": _snap(50, 100)},
        ]
        scanner = MarketScanner(scanner_config, scan_universe, mock_client)
        result = await scanner.scan()
        assert "HIGH_VOL" in result
        assert "LOW_VOL" not in result

    @pytest.mark.asyncio
    async def test_respects_max_symbols(self, scanner_config, scan_universe, mock_client):
        scanner_config.max_symbols = 2
        actives = [{"symbol": f"S{i}"} for i in range(10)]
        snaps = {f"S{i}": _snap(50, 2_000_000) for i in range(10)}
        empty_movers = {"gainers": [], "losers": []}
        mock_client._data_get.side_effect = [
            {"most_actives": actives},
            empty_movers,
            empty_movers,
            snaps,
        ]
        scanner = MarketScanner(scanner_config, scan_universe, mock_client)
        result = await scanner.scan()
        assert len(result) <= 2


# ── Hybrid mode ──────────────────────────────────────────────────────


class TestHybridMode:
    @pytest.mark.asyncio
    async def test_merges_custom_with_scanned(self, scanner_config, hybrid_universe, mock_client):
        empty_movers = {"gainers": [], "losers": []}
        mock_client._data_get.side_effect = [
            {"most_actives": [{"symbol": "NVDA"}, {"symbol": "AAPL"}]},
            empty_movers,
            empty_movers,
            {"NVDA": _snap(130, 3_000_000), "AAPL": _snap(175, 5_000_000)},
        ]
        scanner = MarketScanner(scanner_config, hybrid_universe, mock_client)
        result = await scanner.scan()
        assert "AAPL" in result
        assert "TSLA" in result
        assert "NVDA" in result


# ── Rescan logic ─────────────────────────────────────────────────────


class TestRescanLogic:
    def test_needs_rescan_when_empty(self, scanner_config, scan_universe, mock_client):
        scanner = MarketScanner(scanner_config, scan_universe, mock_client)
        assert scanner.needs_rescan()

    @pytest.mark.asyncio
    async def test_no_rescan_immediately_after_scan(self, scanner_config, custom_universe, mock_client):
        scanner = MarketScanner(scanner_config, custom_universe, mock_client)
        await scanner.scan()
        assert not scanner.needs_rescan()
