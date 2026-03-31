"""Alpaca order execution client with retry logic and idempotent submissions."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.agent.config import AlpacaConfig
from src.agent.models import (
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)

logger = logging.getLogger(__name__)

_SIDE_TO_ALPACA = {Side.LONG: "buy", Side.SHORT: "sell"}

_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIAL_FILL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "pending_new": OrderStatus.PENDING,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
}


class AlpacaAPIError(Exception):
    """Raised when the Alpaca API returns a non-success response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Alpaca API {status_code}: {detail}")


class AlpacaRateLimitError(AlpacaAPIError):
    """Raised on HTTP 429 — eligible for automatic retry."""


class AlpacaForbiddenError(AlpacaAPIError):
    """Raised on HTTP 403 — authentication or permission issue."""


class AlpacaValidationError(AlpacaAPIError):
    """Raised on HTTP 422 — malformed or invalid order parameters."""


def _parse_order(data: dict[str, Any]) -> OrderResult:
    """Map an Alpaca order JSON payload to an OrderResult."""
    alpaca_side = data.get("side", "")
    if alpaca_side == "buy":
        side = Side.LONG
    elif alpaca_side == "sell":
        side = Side.SHORT
    else:
        side = Side.FLAT

    return OrderResult(
        broker_order_id=data.get("id", ""),
        client_order_id=data.get("client_order_id", ""),
        symbol=data.get("symbol", ""),
        side=side,
        qty=float(data.get("qty") or 0),
        filled_qty=float(data.get("filled_qty") or 0),
        filled_avg_price=float(data.get("filled_avg_price") or 0),
        status=_STATUS_MAP.get(data.get("status", ""), OrderStatus.PENDING),
        order_type=OrderType(data.get("type", "market")),
        submitted_at=data.get("submitted_at"),
        filled_at=data.get("filled_at"),
        raw=data,
    )


class AlpacaExecutionClient:
    """Async Alpaca trading client with idempotent order submission and retries."""

    def __init__(self, config: AlpacaConfig) -> None:
        self._cfg = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "APCA-API-KEY-ID": config.api_key,
                "APCA-API-SECRET-KEY": config.secret_key,
            },
            timeout=httpx.Timeout(config.api_timeout_seconds),
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ── internal helpers ──────────────────────────────────────────────

    def _retry_decorator(self):
        return retry(
            retry=retry_if_exception_type(AlpacaRateLimitError),
            stop=stop_after_attempt(self._cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=self._cfg.retry_base_delay,
                max=self._cfg.retry_max_delay,
            ),
            reraise=True,
        )

    def _raise_for_error(self, response: httpx.Response) -> None:
        if response.status_code in (200, 201, 204):
            return
        detail = response.text
        if response.status_code == 403:
            raise AlpacaForbiddenError(403, detail)
        if response.status_code == 422:
            raise AlpacaValidationError(422, detail)
        if response.status_code == 429:
            raise AlpacaRateLimitError(429, detail)
        raise AlpacaAPIError(response.status_code, detail)

    # ── public API ────────────────────────────────────────────────────

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit an order to Alpaca with idempotent client_order_id."""

        alpaca_side = _SIDE_TO_ALPACA.get(order.side)
        if alpaca_side is None:
            raise ValueError(f"Cannot map side {order.side!r} to an Alpaca order side")

        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": alpaca_side,
            "type": order.order_type.value,
            "time_in_force": order.time_in_force.value,
            "client_order_id": order.client_order_id,
        }
        if order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)
        if order.stop_price is not None:
            payload["stop_price"] = str(order.stop_price)

        @self._retry_decorator()
        async def _do_submit() -> OrderResult:
            logger.info(
                "Submitting order %s %s %s qty=%s",
                order.client_order_id,
                alpaca_side,
                order.symbol,
                order.qty,
            )
            resp = await self._client.post("/v2/orders", json=payload)
            self._raise_for_error(resp)
            result = _parse_order(resp.json())
            logger.info(
                "Order accepted broker_id=%s status=%s",
                result.broker_order_id,
                result.status.value,
            )
            return result

        return await _do_submit()

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel a single order. Returns True if successfully cancelled."""

        @self._retry_decorator()
        async def _do_cancel() -> bool:
            resp = await self._client.delete(f"/v2/orders/{broker_order_id}")
            if resp.status_code == 204:
                logger.info("Cancelled order %s", broker_order_id)
                return True
            if resp.status_code == 404:
                logger.warning("Order %s not found for cancellation", broker_order_id)
                return False
            if resp.status_code == 422:
                detail = resp.text.lower()
                if 'already in "filled" state' in detail:
                    # Benign race: broker filled the order before cancellation arrived.
                    logger.info(
                        "Cancel skipped for %s; order already filled",
                        broker_order_id,
                    )
                    return False
            self._raise_for_error(resp)
            return True

        return await _do_cancel()

    async def cancel_all_orders(self) -> int:
        """Cancel every open order. Returns the number of orders cancelled."""

        @self._retry_decorator()
        async def _do_cancel_all() -> int:
            resp = await self._client.delete("/v2/orders")
            self._raise_for_error(resp)
            cancelled: list[dict[str, Any]] = resp.json()
            count = len(cancelled)
            logger.info("Cancelled %d open orders", count)
            return count

        return await _do_cancel_all()

    async def get_order(self, broker_order_id: str) -> OrderResult:
        """Fetch current state of an order by its broker-assigned ID."""

        @self._retry_decorator()
        async def _do_get() -> OrderResult:
            resp = await self._client.get(f"/v2/orders/{broker_order_id}")
            self._raise_for_error(resp)
            return _parse_order(resp.json())

        return await _do_get()

    async def close_position(self, symbol: str) -> OrderResult:
        """Close an open position for the given symbol."""

        @self._retry_decorator()
        async def _do_close() -> OrderResult:
            resp = await self._client.delete(f"/v2/positions/{symbol}")
            self._raise_for_error(resp)
            result = _parse_order(resp.json())
            logger.info("Closed position %s broker_id=%s", symbol, result.broker_order_id)
            return result

        return await _do_close()

    async def close_all_positions(self) -> list[OrderResult]:
        """Liquidate all open positions. Returns the resulting close orders."""

        @self._retry_decorator()
        async def _do_close_all() -> list[OrderResult]:
            resp = await self._client.delete(
                "/v2/positions", params={"cancel_orders": "true"}
            )
            self._raise_for_error(resp)
            results = [_parse_order(item) for item in resp.json()]
            logger.info("Closed %d positions", len(results))
            return results

        return await _do_close_all()
