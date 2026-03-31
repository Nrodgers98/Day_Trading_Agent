"""Tests for risk management components."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agent.config import MonitoringConfig, RiskConfig
from src.agent.models import (
    AccountInfo,
    FeatureVector,
    OrderRequest,
    OrderType,
    Position,
    Quote,
    RiskVerdict,
    Side,
    TimeInForce,
)
from src.agent.risk.circuit_breaker import CircuitBreaker
from src.agent.risk.manager import RiskManager
from src.agent.risk.position_sizer import PositionSizer


@pytest.fixture
def risk_cfg() -> RiskConfig:
    return RiskConfig(
        max_risk_per_trade_pct=0.005,
        max_daily_drawdown_pct=0.02,
        max_concurrent_positions=10,
        max_symbol_concentration_pct=0.15,
        daily_trade_cap=30,
        stop_loss_atr_mult=1.5,
        take_profit_atr_mult=2.5,
        trailing_stop_atr_mult=1.0,
    )


@pytest.fixture
def mon_cfg() -> MonitoringConfig:
    return MonitoringConfig(max_consecutive_failures=3)


@pytest.fixture
def risk_manager(risk_cfg, mon_cfg) -> RiskManager:
    return RiskManager(risk_cfg, mon_cfg)


# ── PositionSizer ─────────────────────────────────────────────────────


class TestPositionSizer:
    def test_calculate_size_returns_positive(self, risk_cfg):
        sizer = PositionSizer(risk_cfg)

        qty = sizer.calculate_size(
            equity=100_000.0, atr=1.5, price=150.0, side=Side.LONG
        )

        assert qty > 0

    def test_returns_zero_for_zero_atr(self, risk_cfg):
        sizer = PositionSizer(risk_cfg)

        assert sizer.calculate_size(100_000.0, 0.0, 150.0, Side.LONG) == 0.0

    def test_returns_zero_for_negative_atr(self, risk_cfg):
        sizer = PositionSizer(risk_cfg)

        assert sizer.calculate_size(100_000.0, -1.0, 150.0, Side.LONG) == 0.0

    def test_returns_zero_for_zero_price(self, risk_cfg):
        sizer = PositionSizer(risk_cfg)

        assert sizer.calculate_size(100_000.0, 1.5, 0.0, Side.LONG) == 0.0


# ── CircuitBreaker ────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_check_daily_drawdown_detects_breach(self, risk_cfg, mon_cfg):
        cb = CircuitBreaker(risk_cfg, mon_cfg)
        cb.starting_equity = 100_000.0

        breached = cb.check_daily_drawdown(97_000.0)  # 3% > 2% limit

        assert breached is True

    def test_check_daily_drawdown_no_breach(self, risk_cfg, mon_cfg):
        cb = CircuitBreaker(risk_cfg, mon_cfg)
        cb.starting_equity = 100_000.0

        breached = cb.check_daily_drawdown(99_000.0)  # 1% < 2% limit

        assert breached is False

    def test_kill_switch_activates_after_consecutive_failures(self, risk_cfg, mon_cfg):
        cb = CircuitBreaker(risk_cfg, mon_cfg)

        assert not cb.is_halted()

        for _ in range(mon_cfg.max_consecutive_failures + 1):
            cb.record_failure()

        assert cb.is_halted()

    def test_success_resets_consecutive_failure_count(self, risk_cfg, mon_cfg):
        cb = CircuitBreaker(risk_cfg, mon_cfg)
        cb.record_failure()
        cb.record_failure()

        cb.record_success()

        assert cb.consecutive_failures == 0
        assert not cb.is_halted()

    def test_check_trade_cap_reached(self, risk_cfg, mon_cfg):
        cb = CircuitBreaker(risk_cfg, mon_cfg)
        cb.trade_count_today = risk_cfg.daily_trade_cap

        assert cb.check_trade_cap() is True

    def test_check_trade_cap_not_reached(self, risk_cfg, mon_cfg):
        cb = CircuitBreaker(risk_cfg, mon_cfg)
        cb.trade_count_today = 0

        assert cb.check_trade_cap() is False

    def test_reset_daily_clears_all_counters(self, risk_cfg, mon_cfg):
        cb = CircuitBreaker(risk_cfg, mon_cfg)
        cb.trade_count_today = 15
        cb.consecutive_failures = 2
        cb.activate_kill_switch("test")

        cb.reset_daily()

        assert cb.trade_count_today == 0
        assert cb.consecutive_failures == 0
        assert not cb.is_halted()


# ── RiskManager ───────────────────────────────────────────────────────


class TestRiskManager:
    def test_pre_trade_check_approves_valid_order(
        self, risk_manager, sample_account, sample_features, sample_quote
    ):
        risk_manager.breaker.starting_equity = sample_account.equity
        order = OrderRequest(
            symbol="AAPL",
            side=Side.LONG,
            qty=10.0,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
        )

        verdict = risk_manager.pre_trade_check(
            order=order,
            account=sample_account,
            positions=[],
            features=sample_features,
            quote=sample_quote,
        )

        assert isinstance(verdict, RiskVerdict)
        assert verdict.approved is True
        assert verdict.adjusted_qty > 0

    def test_rejects_when_max_positions_reached(
        self, risk_manager, sample_account, sample_features, sample_quote
    ):
        risk_manager.breaker.starting_equity = sample_account.equity
        positions = [
            Position(
                symbol=f"SYM{i}",
                side=Side.LONG,
                qty=10.0,
                avg_entry_price=100.0,
                market_value=1_000.0,
            )
            for i in range(10)
        ]
        order = OrderRequest(symbol="AAPL", side=Side.LONG, qty=10.0)

        verdict = risk_manager.pre_trade_check(
            order=order,
            account=sample_account,
            positions=positions,
            features=sample_features,
            quote=sample_quote,
        )

        assert verdict.approved is False
        assert any("Max concurrent" in r for r in verdict.reasons)

    def test_rejects_when_daily_drawdown_breached(
        self, risk_manager, sample_features, sample_quote
    ):
        risk_manager.breaker.starting_equity = 100_000.0
        account = AccountInfo(
            equity=97_000.0,
            cash=50_000.0,
            buying_power=100_000.0,
            portfolio_value=97_000.0,
        )
        order = OrderRequest(symbol="AAPL", side=Side.LONG, qty=10.0)

        verdict = risk_manager.pre_trade_check(
            order=order,
            account=account,
            positions=[],
            features=sample_features,
            quote=sample_quote,
        )

        assert verdict.approved is False
        assert any("drawdown" in r.lower() for r in verdict.reasons)

    def test_rejects_when_kill_switch_active(
        self, risk_manager, sample_account, sample_features, sample_quote
    ):
        risk_manager.breaker.starting_equity = sample_account.equity
        risk_manager.breaker.activate_kill_switch("test kill")
        order = OrderRequest(symbol="AAPL", side=Side.LONG, qty=10.0)

        verdict = risk_manager.pre_trade_check(
            order=order,
            account=sample_account,
            positions=[],
            features=sample_features,
            quote=sample_quote,
        )

        assert verdict.approved is False
        assert any("Kill switch" in r for r in verdict.reasons)

    def test_calculate_exit_levels_long(self, risk_manager):
        levels = risk_manager.calculate_exit_levels(
            entry_price=150.0, atr=2.0, side=Side.LONG
        )

        assert levels["stop_loss"] < 150.0
        assert levels["take_profit"] > 150.0
        assert levels["trailing_stop"] < 150.0
        assert levels["stop_loss"] == pytest.approx(150.0 - 2.0 * 1.5)
        assert levels["take_profit"] == pytest.approx(150.0 + 2.0 * 2.5)

    def test_calculate_exit_levels_short(self, risk_manager):
        levels = risk_manager.calculate_exit_levels(
            entry_price=150.0, atr=2.0, side=Side.SHORT
        )

        assert levels["stop_loss"] > 150.0
        assert levels["take_profit"] < 150.0
        assert levels["stop_loss"] == pytest.approx(150.0 + 2.0 * 1.5)
        assert levels["take_profit"] == pytest.approx(150.0 - 2.0 * 2.5)
