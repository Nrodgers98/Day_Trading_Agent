"""
Order fill simulator for backtesting — applies slippage and commission.
"""

from __future__ import annotations

from src.agent.models import Bar, OrderRequest, OrderResult, OrderStatus, Side


class FillSimulator:
    """Simulates market-order fills at the bar's open price with realistic
    slippage and per-share commission."""

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.0,
    ) -> None:
        self._slippage_bps = slippage_bps
        self._commission_per_share = commission_per_share

    @property
    def commission_per_share(self) -> float:
        return self._commission_per_share

    def calculate_slippage(self, price: float, side: Side) -> float:
        """Return signed slippage: positive for buys (worse fill),
        negative for sells (worse fill)."""
        slip = price * (self._slippage_bps / 10_000)
        return slip if side == Side.LONG else -slip

    def simulate_fill(self, order: OrderRequest, bar: Bar) -> OrderResult:
        """Fill *order* at *bar*.open ± slippage and return a FILLED result."""
        base_price = bar.open
        slippage = self.calculate_slippage(base_price, order.side)
        fill_price = base_price + slippage

        return OrderResult(
            broker_order_id=f"bt-{order.client_order_id}",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            filled_qty=order.qty,
            filled_avg_price=round(fill_price, 6),
            status=OrderStatus.FILLED,
            order_type=order.order_type,
            submitted_at=bar.timestamp,
            filled_at=bar.timestamp,
        )
