from __future__ import annotations

import math

from src.agent.config import RiskConfig
from src.agent.models import Side


class PositionSizer:
    def __init__(self, cfg: RiskConfig, *, fractional: bool = False) -> None:
        self._cfg = cfg
        self._fractional = fractional

    def calculate_size(
        self,
        equity: float,
        atr: float,
        price: float,
        side: Side,
    ) -> float:
        if price <= 0 or atr <= 0:
            return 0.0

        risk_amount = equity * self._cfg.max_risk_per_trade_pct
        stop_distance = atr * self._cfg.stop_loss_atr_mult
        if stop_distance <= 0:
            return 0.0

        qty = risk_amount / stop_distance

        max_notional = equity * self._cfg.max_symbol_concentration_pct
        max_qty = max_notional / price
        qty = min(qty, max_qty)

        if not self._fractional:
            qty = float(math.floor(qty))

        return max(qty, 0.0)
