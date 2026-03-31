"""Integration tests for Alpaca execution client (mocked HTTP)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.agent.config import AlpacaConfig
from src.agent.execution.alpaca_client import (
    AlpacaAPIError,
    AlpacaExecutionClient,
    AlpacaValidationError,
)
from src.agent.models import (
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)


@pytest.fixture
def alpaca_cfg() -> AlpacaConfig:
    return AlpacaConfig(
        api_key="test-key",
        secret_key="test-secret",
        base_url="https://paper-api.alpaca.markets",
        max_retries=1,
    )


@pytest.fixture
def mock_order_json() -> dict:
    return {
        "id": "broker-order-123",
        "client_order_id": "client-123",
        "symbol": "AAPL",
        "side": "buy",
        "qty": "10",
        "filled_qty": "0",
        "filled_avg_price": "0",
        "status": "accepted",
        "type": "market",
        "submitted_at": "2024-01-02T10:30:00Z",
        "filled_at": None,
    }


def _make_response(status_code: int, json_data: dict | list | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


class TestSubmitOrder:
    @pytest.mark.asyncio
    async def test_returns_order_result_on_success(self, alpaca_cfg, mock_order_json):
        client = AlpacaExecutionClient(alpaca_cfg)
        client._client.post = AsyncMock(
            return_value=_make_response(200, mock_order_json)
        )

        order = OrderRequest(
            symbol="AAPL",
            side=Side.LONG,
            qty=10.0,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id="client-123",
        )

        result = await client.submit_order(order)

        assert isinstance(result, OrderResult)
        assert result.broker_order_id == "broker-order-123"
        assert result.symbol == "AAPL"
        assert result.side == Side.LONG
        assert result.status == OrderStatus.ACCEPTED
        await client.close()

    @pytest.mark.asyncio
    async def test_maps_sell_side_correctly(self, alpaca_cfg):
        client = AlpacaExecutionClient(alpaca_cfg)
        sell_json = {
            "id": "broker-456",
            "client_order_id": "client-456",
            "symbol": "TSLA",
            "side": "sell",
            "qty": "5",
            "filled_qty": "0",
            "filled_avg_price": "0",
            "status": "accepted",
            "type": "market",
            "submitted_at": None,
            "filled_at": None,
        }
        client._client.post = AsyncMock(
            return_value=_make_response(200, sell_json)
        )

        order = OrderRequest(
            symbol="TSLA", side=Side.SHORT, qty=5.0
        )

        result = await client.submit_order(order)

        assert result.side == Side.SHORT
        await client.close()


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_returns_true_on_204(self, alpaca_cfg):
        client = AlpacaExecutionClient(alpaca_cfg)
        client._client.delete = AsyncMock(
            return_value=_make_response(204)
        )

        result = await client.cancel_order("broker-order-123")

        assert result is True
        await client.close()

    @pytest.mark.asyncio
    async def test_cancel_returns_false_on_404(self, alpaca_cfg):
        client = AlpacaExecutionClient(alpaca_cfg)
        client._client.delete = AsyncMock(
            return_value=_make_response(404)
        )

        result = await client.cancel_order("nonexistent-id")

        assert result is False
        await client.close()


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_422_raises_validation_error(self, alpaca_cfg):
        client = AlpacaExecutionClient(alpaca_cfg)
        client._client.post = AsyncMock(
            return_value=_make_response(422, text="Invalid order parameters")
        )

        order = OrderRequest(symbol="AAPL", side=Side.LONG, qty=10.0)

        with pytest.raises(AlpacaValidationError) as exc_info:
            await client.submit_order(order)

        assert exc_info.value.status_code == 422
        await client.close()

    @pytest.mark.asyncio
    async def test_generic_error_raises_alpaca_api_error(self, alpaca_cfg):
        client = AlpacaExecutionClient(alpaca_cfg)
        client._client.post = AsyncMock(
            return_value=_make_response(500, text="Internal Server Error")
        )

        order = OrderRequest(symbol="AAPL", side=Side.LONG, qty=10.0)

        with pytest.raises(AlpacaAPIError) as exc_info:
            await client.submit_order(order)

        assert exc_info.value.status_code == 500
        await client.close()

    @pytest.mark.asyncio
    async def test_flat_side_raises_value_error(self, alpaca_cfg):
        client = AlpacaExecutionClient(alpaca_cfg)

        order = OrderRequest(symbol="AAPL", side=Side.FLAT, qty=10.0)

        with pytest.raises(ValueError, match="Cannot map side"):
            await client.submit_order(order)

        await client.close()
