"""Live trading runner — inherits PaperRunner with live-credential safeguards."""

from __future__ import annotations

import logging
import os

from src.agent.config import AlpacaConfig, AppConfig
from src.agent.data.market_data import AlpacaMarketDataClient
from src.agent.data.sentiment import SentimentProvider
from src.agent.execution.alpaca_client import AlpacaExecutionClient
from src.agent.execution.order_manager import OrderManager
from src.agent.monitoring.logger import setup_component_logger, setup_logging
from src.agent.runner.paper import PaperRunner

logger = logging.getLogger("trading_agent")


class LiveRunner(PaperRunner):
    """Runs the identical pipeline as :class:`PaperRunner` but against
    a **live** Alpaca brokerage account.

    Safety gates enforced during ``setup()``:

    1. ``config.trading.enable_live`` must be ``True``
    2. The environment variable ``ENABLE_LIVE_TRADING`` must equal ``"true"``
    3. Live API credentials (``ALPACA_LIVE_API_KEY`` / ``ALPACA_LIVE_SECRET_KEY``)
       must be present in the environment.
    """

    async def setup(self) -> None:
        setup_logging(self._config.monitoring, log_name="live")
        self._sentiment_logger = setup_component_logger(
            self._config.monitoring,
            logger_name="trading_agent.sentiment",
            file_name="agent_sentiment.log",
        )

        if not self._config.trading.enable_live:
            raise RuntimeError(
                "Live trading requires config trading.enable_live=true"
            )

        env_flag = os.getenv("ENABLE_LIVE_TRADING", "false").lower()
        if env_flag != "true":
            raise RuntimeError(
                "Live trading requires env var ENABLE_LIVE_TRADING=true"
            )

        live_key = os.getenv("ALPACA_LIVE_API_KEY", "")
        live_secret = os.getenv("ALPACA_LIVE_SECRET_KEY", "")
        if not live_key or not live_secret:
            raise RuntimeError(
                "ALPACA_LIVE_API_KEY and ALPACA_LIVE_SECRET_KEY must be set"
            )

        live_base_url = os.getenv(
            "ALPACA_LIVE_BASE_URL", "https://api.alpaca.markets"
        )

        logger.warning(
            "*** LIVE TRADING MODE *** — real money is at risk. "
            "base_url=%s",
            live_base_url,
        )

        live_alpaca = AlpacaConfig(
            api_key=live_key,
            secret_key=live_secret,
            base_url=live_base_url,
            data_url=self._config.alpaca.data_url,
            api_timeout_seconds=self._config.alpaca.api_timeout_seconds,
            max_retries=self._config.alpaca.max_retries,
            retry_base_delay=self._config.alpaca.retry_base_delay,
            retry_max_delay=self._config.alpaca.retry_max_delay,
            rate_limit_calls_per_minute=self._config.alpaca.rate_limit_calls_per_minute,
        )

        self._market_data = AlpacaMarketDataClient(live_alpaca)
        self._exec_client = AlpacaExecutionClient(live_alpaca)

        from src.agent.monitoring.alerts import AlertManager
        from src.agent.monitoring.audit import AuditLogger
        from src.agent.monitoring.reports import DailyReportGenerator
        from src.agent.execution.reconciliation import PositionReconciler
        from src.agent.risk.manager import RiskManager
        from src.agent.signal.engine import SignalEngine
        from src.agent.signal.ml_model import MLSignalModel
        from src.agent.runner.paper import _DEFAULT_MODEL_FILE

        self._risk_mgr = RiskManager(
            self._config.risk,
            self._config.monitoring,
            fractional=self._config.trading.enable_fractional_shares,
        )

        self._ml_model = MLSignalModel()
        if _DEFAULT_MODEL_FILE.exists():
            self._ml_model.load(str(_DEFAULT_MODEL_FILE))
            logger.info("Loaded ML model from %s", _DEFAULT_MODEL_FILE)

        self._signal_engine = SignalEngine(
            self._config.strategy,
            ml_model=self._ml_model if self._ml_model.is_trained else None,
            spread_guard_pct=self._config.risk.spread_guard_pct,
            sentiment_positive_threshold=self._config.sentiment.positive_threshold,
            sentiment_negative_threshold=self._config.sentiment.negative_threshold,
            sentiment_confidence_boost=self._config.sentiment.confidence_boost,
        )

        self._order_mgr = OrderManager(
            self._exec_client,
            self._risk_mgr.breaker,
        )

        self._audit = AuditLogger(self._config.monitoring.log_dir)
        self._alerts = AlertManager(self._config.monitoring)
        self._reconciler = PositionReconciler()
        self._report_gen = DailyReportGenerator()
        self._sentiment = SentimentProvider(self._config.sentiment, self._market_data)

        account = await self._market_data.get_account()
        self._starting_equity = account.equity
        self._risk_mgr.breaker.starting_equity = account.equity

        logger.warning(
            "LIVE account connected | equity=%.2f cash=%.2f buying_power=%.2f status=%s",
            account.equity,
            account.cash,
            account.buying_power,
            account.status,
        )
