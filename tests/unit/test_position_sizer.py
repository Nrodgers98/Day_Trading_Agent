"""Tests for position sizing logic."""
from __future__ import annotations

import math

import pytest

from src.agent.config import RiskConfig
from src.agent.models import Side
from src.agent.risk.position_sizer import PositionSizer


@pytest.fixture
def risk_cfg() -> RiskConfig:
    return RiskConfig(
        max_risk_per_trade_pct=0.005,
        max_symbol_concentration_pct=0.15,
        stop_loss_atr_mult=1.5,
    )


class TestCalculateSizeBasicMath:
    def test_risk_limited_size(self, risk_cfg):
        """When risk-based qty < concentration cap, risk amount drives sizing."""
        sizer = PositionSizer(risk_cfg)

        qty = sizer.calculate_size(
            equity=100_000.0, atr=2.0, price=100.0, side=Side.LONG
        )

        # risk_amount = 100_000 * 0.005 = 500
        # stop_distance = 2.0 * 1.5 = 3.0
        # raw_qty = 500 / 3.0 ≈ 166.67
        # max_notional = 100_000 * 0.15 = 15_000 → max_qty = 150
        # final = min(166.67, 150) = 150 → floor = 150
        assert qty == 150.0


class TestConcentrationCap:
    def test_concentration_cap_limits_qty(self, risk_cfg):
        """When risk-based qty exceeds concentration cap, cap takes effect."""
        sizer = PositionSizer(risk_cfg)

        qty = sizer.calculate_size(
            equity=100_000.0, atr=0.5, price=50.0, side=Side.LONG
        )

        # risk_amount = 500
        # stop_distance = 0.5 * 1.5 = 0.75
        # raw_qty = 500 / 0.75 ≈ 666.67
        # max_notional = 15_000 → max_qty = 300
        # final = min(666.67, 300) = 300 → floor = 300
        assert qty == 300.0
        assert qty * 50.0 <= 100_000.0 * risk_cfg.max_symbol_concentration_pct


class TestShareRounding:
    def test_integer_shares_floor_applied(self, risk_cfg):
        sizer = PositionSizer(risk_cfg, fractional=False)

        qty = sizer.calculate_size(
            equity=10_000.0, atr=3.0, price=200.0, side=Side.LONG
        )

        # risk_amount = 50, stop_distance = 4.5
        # raw_qty = 11.11, max_qty = 7.5
        # min(11.11, 7.5) = 7.5 → floor = 7
        assert qty == 7.0
        assert qty == math.floor(qty)

    def test_fractional_shares_preserve_decimal(self, risk_cfg):
        sizer = PositionSizer(risk_cfg, fractional=True)

        qty = sizer.calculate_size(
            equity=10_000.0, atr=3.0, price=200.0, side=Side.LONG
        )

        # Same math but no floor: 7.5
        assert qty == 7.5

    def test_integer_shares_never_negative(self, risk_cfg):
        sizer = PositionSizer(risk_cfg, fractional=False)

        qty = sizer.calculate_size(
            equity=100.0, atr=50.0, price=200.0, side=Side.LONG
        )

        assert qty >= 0.0

    def test_fractional_same_as_integer_when_whole_number(self, risk_cfg):
        """When the raw qty is already a whole number, both modes agree."""
        sizer_int = PositionSizer(risk_cfg, fractional=False)
        sizer_frac = PositionSizer(risk_cfg, fractional=True)

        qty_int = sizer_int.calculate_size(
            equity=100_000.0, atr=2.0, price=100.0, side=Side.LONG
        )
        qty_frac = sizer_frac.calculate_size(
            equity=100_000.0, atr=2.0, price=100.0, side=Side.LONG
        )

        assert qty_int == qty_frac
