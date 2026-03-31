"""Order lifecycle manager — tracks, refreshes, and garbage-collects orders."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.agent.execution.alpaca_client import AlpacaExecutionClient
from src.agent.models import OrderRequest, OrderResult, OrderStatus
from src.agent.risk.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
)


class OrderManager:
    """Wraps AlpacaExecutionClient with local tracking and circuit-breaker integration."""

    def __init__(
        self,
        client: AlpacaExecutionClient,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        self._client = client
        self._cb = circuit_breaker
        self._open_orders: dict[str, OrderResult] = {}

    # ── public API ────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order via the execution client and begin tracking it."""
        result = await self._client.submit_order(request)
        self._track(result)
        logger.info(
            "order_placed | client_order_id=%s broker_id=%s symbol=%s side=%s qty=%s status=%s",
            result.client_order_id,
            result.broker_order_id,
            result.symbol,
            result.side.value,
            result.qty,
            result.status.value,
        )
        if result.status == OrderStatus.REJECTED:
            self._cb.record_failure()
        return result

    async def check_order_status(self, client_order_id: str) -> OrderResult:
        """Refresh an order's status from the broker and update local state."""
        existing = self._open_orders.get(client_order_id)
        if existing is None:
            raise KeyError(f"No tracked order with client_order_id={client_order_id}")

        refreshed = await self._client.get_order(existing.broker_order_id)
        old_status = existing.status
        self._track(refreshed)

        if old_status != refreshed.status:
            logger.info(
                "order_status_change | client_order_id=%s %s -> %s",
                client_order_id,
                old_status.value,
                refreshed.status.value,
            )
            self._apply_terminal_effects(refreshed)

        if refreshed.status == OrderStatus.PARTIAL_FILL:
            await self.handle_partial_fill(refreshed)

        return refreshed

    async def handle_partial_fill(self, order: OrderResult) -> None:
        """Log partial fill details for monitoring."""
        logger.warning(
            "partial_fill | client_order_id=%s symbol=%s filled=%s/%s avg_price=%s",
            order.client_order_id,
            order.symbol,
            order.filled_qty,
            order.qty,
            order.filled_avg_price,
        )

    async def cancel_stale_orders(self, max_age_seconds: int = 300) -> list[str]:
        """Cancel orders that have been open longer than *max_age_seconds*."""
        now = datetime.now(tz=timezone.utc)
        cancelled: list[str] = []

        for cid, order in list(self._open_orders.items()):
            if order.status in _TERMINAL_STATUSES:
                continue
            if order.submitted_at is None:
                continue
            submitted = order.submitted_at
            if submitted.tzinfo is None:
                submitted = submitted.replace(tzinfo=timezone.utc)
            age = (now - submitted).total_seconds()
            if age > max_age_seconds:
                success = await self._client.cancel_order(order.broker_order_id)
                if success:
                    logger.info(
                        "stale_order_cancelled | client_order_id=%s age_s=%.0f",
                        cid,
                        age,
                    )
                    order.status = OrderStatus.CANCELED
                    cancelled.append(cid)
                else:
                    # If broker did not cancel (e.g., already filled), refresh status so
                    # we stop retrying stale cancellations on terminal orders.
                    try:
                        refreshed = await self._client.get_order(order.broker_order_id)
                        self._track(refreshed)
                        if refreshed.status in _TERMINAL_STATUSES:
                            logger.info(
                                "stale_order_resolved | client_order_id=%s broker_id=%s status=%s",
                                cid,
                                order.broker_order_id,
                                refreshed.status.value,
                            )
                    except Exception:
                        logger.debug(
                            "Could not refresh stale order status | client_order_id=%s broker_id=%s",
                            cid,
                            order.broker_order_id,
                            exc_info=True,
                        )

        return cancelled

    def get_open_orders(self) -> list[OrderResult]:
        """Return all non-terminal orders currently being tracked."""
        return [
            o for o in self._open_orders.values() if o.status not in _TERMINAL_STATUSES
        ]

    # ── internals ─────────────────────────────────────────────────────

    def _track(self, order: OrderResult) -> None:
        self._open_orders[order.client_order_id] = order

    def _apply_terminal_effects(self, order: OrderResult) -> None:
        if order.status == OrderStatus.FILLED:
            self._cb.record_success()
            self._cb.trade_count_today += 1
        elif order.status in (OrderStatus.REJECTED, OrderStatus.EXPIRED):
            self._cb.record_failure()
