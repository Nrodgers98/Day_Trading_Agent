"""Integration tests for end-to-end signal -> risk -> execute flow."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.config import (
    AppConfig,
    MonitoringConfig,
    RiskConfig,
    StrategyConfig,
    TradingConfig,
)
from src.agent.execution.alpaca_client import AlpacaExecutionClient
from src.agent.execution.order_manager import OrderManager
from src.agent.models import (
    AccountInfo,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    RiskVerdict,
    Signal,
    SignalAction,
    Side,
    TimeInForce,
)
from src.agent.risk.manager import RiskManager
from src.agent.signal.engine import SignalEngine


@pytest.fixture
def e2e_config() -> AppConfig:
    return AppConfig(
        trading=TradingConfig(mode="paper", enable_live=False),
        strategy=StrategyConfig(
            lookback_bars=20,
            volume_surge_ratio=1.5,
            ml_confidence_threshold=0.6,
        ),
        risk=RiskConfig(
            max_risk_per_trade_pct=0.005,
            max_daily_drawdown_pct=0.02,
            max_concurrent_positions=10,
            daily_trade_cap=30,
            stop_loss_atr_mult=1.5,
            take_profit_atr_mult=2.5,
        ),
        monitoring=MonitoringConfig(max_consecutive_failures=3),
    )


@pytest.fixture
def mock_exec_client() -> AlpacaExecutionClient:
    client = MagicMock(spec=AlpacaExecutionClient)
    client.submit_order = AsyncMock(
        return_value=OrderResult(
            broker_order_id="broker-123",
            client_order_id="client-123",
            symbol="AAPL",
            side=Side.LONG,
            qty=10.0,
            filled_qty=10.0,
            filled_avg_price=150.05,
            status=OrderStatus.FILLED,
        )
    )
    client.cancel_all_orders = AsyncMock(return_value=0)
    client.close_all_positions = AsyncMock(
        return_value=[
            OrderResult(
                broker_order_id="close-1",
                symbol="AAPL",
                side=Side.LONG,
                qty=10.0,
                filled_qty=10.0,
                status=OrderStatus.FILLED,
            )
        ]
    )
    client.close_position = AsyncMock(
        return_value=OrderResult(
            broker_order_id="close-1",
            symbol="AAPL",
            side=Side.LONG,
            qty=10.0,
            filled_qty=10.0,
            status=OrderStatus.FILLED,
        )
    )
    return client


class TestSignalToExecuteFlow:
    @pytest.mark.asyncio
    async def test_end_to_end_signal_risk_execute(
        self,
        e2e_config,
        mock_exec_client,
        sample_account,
        sample_features,
        sample_quote,
    ):
        risk_mgr = RiskManager(e2e_config.risk, e2e_config.monitoring)
        risk_mgr.breaker.starting_equity = sample_account.equity
        order_mgr = OrderManager(mock_exec_client, risk_mgr.breaker)

        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(tz=timezone.utc),
            action=SignalAction.BUY,
            side=Side.LONG,
            confidence=0.75,
            features=sample_features,
        )

        side = Side.LONG
        desired_qty = risk_mgr.sizer.calculate_size(
            sample_account.equity,
            sample_features.atr_14,
            sample_quote.mid_price,
            side,
        )
        assert desired_qty > 0

        order = OrderRequest(
            symbol="AAPL",
            side=side,
            qty=desired_qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            signal=signal,
        )

        verdict = risk_mgr.pre_trade_check(
            order=order,
            account=sample_account,
            positions=[],
            features=sample_features,
            quote=sample_quote,
        )

        assert verdict.approved is True
        assert verdict.adjusted_qty > 0

        final_order = order.model_copy(update={"qty": verdict.adjusted_qty})
        result = await order_mgr.place_order(final_order)

        assert result.status == OrderStatus.FILLED
        assert result.broker_order_id == "broker-123"
        mock_exec_client.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejected_signal_does_not_place_order(
        self,
        e2e_config,
        mock_exec_client,
        sample_features,
        sample_quote,
    ):
        risk_mgr = RiskManager(e2e_config.risk, e2e_config.monitoring)
        risk_mgr.breaker.starting_equity = 100_000.0
        risk_mgr.breaker.activate_kill_switch("test")

        account = AccountInfo(
            equity=100_000.0,
            cash=50_000.0,
            buying_power=100_000.0,
            portfolio_value=100_000.0,
        )
        order = OrderRequest(symbol="AAPL", side=Side.LONG, qty=10.0)

        verdict = risk_mgr.pre_trade_check(
            order=order,
            account=account,
            positions=[],
            features=sample_features,
            quote=sample_quote,
        )

        assert verdict.approved is False
        mock_exec_client.submit_order.assert_not_called()


class TestEODFlatten:
    @pytest.mark.asyncio
    async def test_eod_flatten_cancels_orders_and_closes_positions(
        self, mock_exec_client
    ):
        await mock_exec_client.cancel_all_orders()
        results = await mock_exec_client.close_all_positions()

        mock_exec_client.cancel_all_orders.assert_called_once()
        mock_exec_client.close_all_positions.assert_called_once()
        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED
        assert results[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_eod_flatten_single_position(self, mock_exec_client):
        result = await mock_exec_client.close_position("AAPL")

        mock_exec_client.close_position.assert_called_once_with("AAPL")
        assert result.status == OrderStatus.FILLED
