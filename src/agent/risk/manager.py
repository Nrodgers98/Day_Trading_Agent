from __future__ import annotations

import logging

from src.agent.config import MonitoringConfig, RiskConfig
from src.agent.models import (
    AccountInfo,
    FeatureVector,
    OrderRequest,
    OrderResult,
    Position,
    Quote,
    RiskVerdict,
    Side,
)
from src.agent.risk.circuit_breaker import CircuitBreaker
from src.agent.risk.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        risk_cfg: RiskConfig,
        mon_cfg: MonitoringConfig | None = None,
        *,
        fractional: bool = False,
    ) -> None:
        self._cfg = risk_cfg
        self._mon_cfg = mon_cfg or MonitoringConfig()
        self.sizer = PositionSizer(risk_cfg, fractional=fractional)
        self.breaker = CircuitBreaker(risk_cfg, self._mon_cfg)

    # ── pre-trade ─────────────────────────────────────────────────────

    def pre_trade_check(
        self,
        order: OrderRequest,
        account: AccountInfo,
        positions: list[Position],
        features: FeatureVector,
        quote: Quote | None = None,
    ) -> RiskVerdict:
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        price = quote.mid_price if quote else features.extra.get("price", 0.0)

        checks["kill_switch"] = not self.breaker.is_halted()
        if not checks["kill_switch"]:
            reasons.append("Kill switch is active")

        checks["daily_drawdown"] = not self.breaker.check_daily_drawdown(
            account.equity
        )
        if not checks["daily_drawdown"]:
            reasons.append("Daily drawdown limit breached")

        checks["trade_cap"] = not self.breaker.check_trade_cap()
        if not checks["trade_cap"]:
            reasons.append("Daily trade cap reached")

        checks["max_positions"] = len(positions) < self._cfg.max_concurrent_positions
        if not checks["max_positions"]:
            reasons.append(
                f"Max concurrent positions ({self._cfg.max_concurrent_positions}) reached"
            )

        existing_exposure = sum(
            p.market_value for p in positions if p.symbol == order.symbol
        )
        new_notional = order.qty * price if price else 0.0
        concentration_limit = account.equity * self._cfg.max_symbol_concentration_pct
        checks["symbol_concentration"] = (
            existing_exposure + new_notional <= concentration_limit
        )
        if not checks["symbol_concentration"]:
            reasons.append("Symbol concentration limit exceeded")

        total_market_value = sum(abs(p.market_value) for p in positions)
        gross_limit = self._cfg.max_gross_exposure_pct * account.equity
        checks["gross_exposure"] = total_market_value + new_notional <= gross_limit
        if not checks["gross_exposure"]:
            reasons.append("Gross exposure limit exceeded")

        if quote is not None:
            checks["spread_guard"] = quote.spread_pct < self._cfg.spread_guard_pct
            if not checks["spread_guard"]:
                reasons.append(
                    f"Spread too wide ({quote.spread_pct:.4f} >= {self._cfg.spread_guard_pct})"
                )
        else:
            checks["spread_guard"] = True

        checks["buying_power"] = account.buying_power >= order.qty * price
        if not checks["buying_power"]:
            reasons.append("Insufficient buying power")

        approved = all(checks.values())

        adjusted_qty = 0.0
        if approved and price > 0 and features.atr_14 > 0:
            adjusted_qty = self.sizer.calculate_size(
                account.equity, features.atr_14, price, order.side
            )

        return RiskVerdict(
            approved=approved and adjusted_qty > 0,
            adjusted_qty=adjusted_qty,
            reasons=reasons,
            checks_passed=checks,
        )

    # ── exit levels ───────────────────────────────────────────────────

    def calculate_exit_levels(
        self,
        entry_price: float,
        atr: float,
        side: Side,
    ) -> dict[str, float]:
        if side == Side.LONG:
            stop_loss = entry_price - atr * self._cfg.stop_loss_atr_mult
            take_profit = entry_price + atr * self._cfg.take_profit_atr_mult
            trailing_stop = entry_price - atr * self._cfg.trailing_stop_atr_mult
        else:
            stop_loss = entry_price + atr * self._cfg.stop_loss_atr_mult
            take_profit = entry_price - atr * self._cfg.take_profit_atr_mult
            trailing_stop = entry_price + atr * self._cfg.trailing_stop_atr_mult

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop": trailing_stop,
        }

    # ── post-trade ────────────────────────────────────────────────────

    def post_trade_validate(
        self,
        order_result: OrderResult,
        expected: OrderRequest,
    ) -> list[str]:
        warnings: list[str] = []

        if order_result.symbol != expected.symbol:
            warnings.append(
                f"Symbol mismatch: expected {expected.symbol}, got {order_result.symbol}"
            )

        if order_result.filled_qty != expected.qty:
            warnings.append(
                f"Qty mismatch: expected {expected.qty}, filled {order_result.filled_qty}"
            )

        if order_result.side != expected.side:
            warnings.append(
                f"Side mismatch: expected {expected.side.value}, got {order_result.side.value}"
            )

        if expected.limit_price and order_result.filled_avg_price:
            slippage_bps = (
                abs(order_result.filled_avg_price - expected.limit_price)
                / expected.limit_price
                * 10_000
            )
            if slippage_bps > self._cfg.slippage_bps:
                warnings.append(
                    f"Excessive slippage: {slippage_bps:.1f} bps "
                    f"(limit {self._cfg.slippage_bps} bps)"
                )

        return warnings
