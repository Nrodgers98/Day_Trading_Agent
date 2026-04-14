"""Paper trading runner — full intraday loop against Alpaca paper accounts."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agent.config import AppConfig
from src.agent.data.features import compute_features
from src.agent.data.market_data import AlpacaMarketDataClient
from src.agent.data.scanner import MarketScanner
from src.agent.data.sentiment import SentimentProvider
from src.agent.execution.alpaca_client import AlpacaExecutionClient
from src.agent.execution.order_manager import OrderManager
from src.agent.execution.reconciliation import PositionReconciler
from src.agent.models import (
    AuditRecord,
    OrderRequest,
    Signal,
    SignalAction,
    Side,
)
from src.agent.monitoring.alerts import AlertManager
from src.agent.monitoring.audit import AuditLogger
from src.agent.monitoring.logger import setup_component_logger, setup_logging
from src.agent.monitoring.reports import DailyReportGenerator
from src.agent.risk.manager import RiskManager
from src.agent.runner.base import BaseRunner
from src.agent.signal.engine import SignalEngine
from src.agent.signal.ml_model import MLSignalModel

logger = logging.getLogger("trading_agent")

_MODEL_DIR = Path("models")
_DEFAULT_MODEL_FILE = _MODEL_DIR / "signal_model.joblib"

_RECONCILE_EVERY_N_CYCLES = 10


class PaperRunner(BaseRunner):
    """Runs the full signal → risk → execution pipeline against Alpaca paper."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)

        self._market_data: AlpacaMarketDataClient | None = None
        self._exec_client: AlpacaExecutionClient | None = None
        self._order_mgr: OrderManager | None = None
        self._signal_engine: SignalEngine | None = None
        self._risk_mgr: RiskManager | None = None
        self._audit: AuditLogger | None = None
        self._alerts: AlertManager | None = None
        self._reconciler: PositionReconciler | None = None
        self._report_gen: DailyReportGenerator | None = None
        self._ml_model: MLSignalModel | None = None
        self._scanner: MarketScanner | None = None
        self._sentiment: SentimentProvider | None = None
        self._sentiment_logger: logging.Logger | None = None

        self._starting_equity: float = 0.0
        self._cycle_count: int = 0
        self._trades_today: list[dict[str, Any]] = []
        self._equity_curve: list[dict[str, Any]] = []
        self._running: bool = False
        # Calendar date (session tz) for which _trades_today / _equity_curve apply; used for per-day reports.
        self._active_report_date: str | None = None
        # Session date (YYYY-MM-DD) for which we already ran EOD flatten (once per day).
        self._eod_flatten_done_session_date: str | None = None

    # ── setup ─────────────────────────────────────────────────────────

    async def setup(self) -> None:
        root_logger = setup_logging(self._config.monitoring, log_name="paper")
        self._sentiment_logger = setup_component_logger(
            self._config.monitoring,
            logger_name="trading_agent.sentiment",
            file_name="agent_sentiment.log",
        )
        logger.info("Setting up PaperRunner")

        self._market_data = AlpacaMarketDataClient(self._config.alpaca)
        self._exec_client = AlpacaExecutionClient(self._config.alpaca)

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

        self._scanner = MarketScanner(
            self._config.scanner,
            self._config.universe,
            self._market_data,
        )
        self._sentiment = SentimentProvider(self._config.sentiment, self._market_data)

        self._audit = AuditLogger(self._config.monitoring.log_dir)
        self._alerts = AlertManager(self._config.monitoring)
        self._reconciler = PositionReconciler()
        self._report_gen = DailyReportGenerator()

        account = await self._market_data.get_account()
        self._starting_equity = account.equity
        self._risk_mgr.breaker.starting_equity = account.equity

        logger.info(
            "Account connected | equity=%.2f cash=%.2f buying_power=%.2f status=%s",
            account.equity,
            account.cash,
            account.buying_power,
            account.status,
        )

    # ── main loop ─────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        self._running = True
        logger.info("Entering main trading loop (interval=%ds)", self._loop_interval_s)

        if (
            self._config.universe.base != "custom"
            and self._config.scanner.premarket_scan
            and self._scanner is not None
        ):
            try:
                symbols = await self._scanner.scan()
                logger.info(
                    "Pre-market scan complete | watchlist=%d symbols: %s",
                    len(symbols),
                    ", ".join(symbols[:15]) + ("..." if len(symbols) > 15 else ""),
                )
            except Exception:
                logger.exception("Pre-market scan failed — will retry on first cycle")

        while self._running:
            self._maybe_rollover_daily_report()

            cycle_start = asyncio.get_event_loop().time()

            if not self._is_market_open():
                logger.debug("Market closed — sleeping")
                await asyncio.sleep(self._loop_interval_s)
                continue

            # Flatten must run even after closing_guard ends the "new trades" window,
            # otherwise we never reach the code that was below _check_session() and
            # positions can carry overnight unintentionally.
            await self._maybe_eod_flatten()

            if not self._check_session():
                logger.debug("Outside guarded session window — sleeping")
                await asyncio.sleep(self._loop_interval_s)
                continue

            if self._risk_mgr.breaker.is_halted():
                logger.warning("Circuit breaker halted — skipping cycle")
                await asyncio.sleep(self._loop_interval_s)
                continue

            try:
                await self._run_cycle()
            except Exception:
                logger.exception("Error during trading cycle")
                self._risk_mgr.breaker.record_failure()
                self._alerts.check_order_failures(
                    self._risk_mgr.breaker.consecutive_failures,
                )

            self._cycle_count += 1
            if self._cycle_count % _RECONCILE_EVERY_N_CYCLES == 0:
                await self._reconcile()

            elapsed = asyncio.get_event_loop().time() - cycle_start
            sleep_for = max(0.0, self._loop_interval_s - elapsed)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    # ── single cycle ──────────────────────────────────────────────────

    async def _run_cycle(self) -> None:
        assert self._market_data is not None
        assert self._signal_engine is not None
        assert self._risk_mgr is not None
        assert self._order_mgr is not None
        assert self._audit is not None
        assert self._alerts is not None

        account = await self._market_data.get_account()
        positions = await self._market_data.get_positions()
        self._alerts.check_drawdown(account.equity, self._starting_equity)

        self._equity_curve.append({
            "timestamp": datetime.now(tz=self._tz).isoformat(),
            "equity": round(account.equity, 2),
        })

        symbols = await self._get_universe()
        timeframe = self._config.strategy.timeframe.replace("m", "Min").replace("h", "Hour")
        lookback = self._config.strategy.lookback_bars

        for symbol in symbols:
            try:
                bars_df = await self._market_data.get_bars(
                    symbol, timeframe=timeframe, limit=lookback,
                )
                if bars_df.empty:
                    continue

                features = compute_features(bars_df, symbol)
                if self._sentiment is not None and self._config.sentiment.enabled:
                    sentiment = await self._sentiment.get_sentiment_score(symbol)
                    features = features.model_copy(update={"sentiment_score": sentiment})

                quote = None
                try:
                    quote = await self._market_data.get_latest_quote(symbol)
                except Exception:
                    logger.debug("Could not fetch quote for %s", symbol)

                signal = self._signal_engine.generate_signal(
                    symbol, bars_df, features, quote,
                )

                audit = AuditRecord(
                    timestamp=datetime.now(tz=self._tz),
                    symbol=symbol,
                    raw_features=self._build_audit_features(features),
                    signal=signal,
                )

                sentiment_effect = (
                    signal.metadata.get("sentiment_effect")
                    if signal.metadata
                    else None
                )
                sentiment_score = (
                    signal.metadata.get("sentiment_score")
                    if signal.metadata
                    else features.sentiment_score
                )
                logger.info(
                    "signal_evaluated | symbol=%s action=%s side=%s confidence=%.3f sentiment=%s sentiment_effect=%s reason=%s",
                    signal.symbol,
                    signal.action.value,
                    signal.side.value,
                    signal.confidence,
                    "None" if sentiment_score is None else f"{float(sentiment_score):.4f}",
                    sentiment_effect or "none",
                    signal.reason,
                )
                if self._sentiment_logger is not None:
                    self._sentiment_logger.info(
                        "sentiment_signal",
                        extra={
                            "symbol": signal.symbol,
                            "action": signal.action.value,
                            "side": signal.side.value,
                            "confidence": round(signal.confidence, 4),
                            "sentiment_score": sentiment_score,
                            "sentiment_effect": sentiment_effect or "none",
                            "reason": signal.reason,
                            "provider": self._config.sentiment.provider,
                            "news_source": self._config.sentiment.news_source,
                            "sentiment_enabled": self._config.sentiment.enabled,
                        },
                    )

                if signal.action in (SignalAction.BUY, SignalAction.SELL):
                    await self._handle_signal(
                        signal, features, account, positions, quote, audit,
                    )
                elif signal.action == SignalAction.CLOSE:
                    has_position = any(p.symbol == symbol for p in positions)
                    if has_position:
                        result = await self._exec_client.close_position(symbol)
                        audit.order_result = result
                        logger.info("Closed position %s", symbol)

                self._audit.log_decision(audit)

            except Exception:
                logger.exception("Error processing symbol %s", symbol)

        await self._order_mgr.cancel_stale_orders()

    @staticmethod
    def _build_audit_features(features: Any) -> dict[str, float]:
        payload: dict[str, float] = {
            "returns_1": float(features.returns_1),
            "returns_5": float(features.returns_5),
            "vwap_distance": float(features.vwap_distance),
            "rsi_14": float(features.rsi_14),
            "ema_9": float(features.ema_9),
            "ema_21": float(features.ema_21),
            "sma_50": float(features.sma_50),
            "atr_14": float(features.atr_14),
            "volume_ratio": float(features.volume_ratio),
            "trend_slope": float(features.trend_slope),
        }
        if features.sentiment_score is not None:
            payload["sentiment_score"] = float(features.sentiment_score)
        for k, v in features.extra.items():
            try:
                payload[k] = float(v)
            except (TypeError, ValueError):
                continue
        return payload

    async def _handle_signal(
        self,
        signal: Signal,
        features: Any,
        account: Any,
        positions: list,
        quote: Any,
        audit: AuditRecord,
    ) -> None:
        assert self._risk_mgr is not None
        assert self._order_mgr is not None

        side = Side.LONG if signal.action == SignalAction.BUY else Side.SHORT

        if side == Side.SHORT and not self._config.trading.enable_shorting:
            logger.debug("Shorting disabled — skipping %s", signal.symbol)
            return

        price = quote.mid_price if quote else features.extra.get("price", 0.0)
        if features.atr_14 <= 0 or price <= 0:
            return

        desired_qty = self._risk_mgr.sizer.calculate_size(
            account.equity, features.atr_14, price, side,
        )
        if desired_qty <= 0:
            return

        order = OrderRequest(
            symbol=signal.symbol,
            side=side,
            qty=desired_qty,
            signal=signal,
        )

        verdict = self._risk_mgr.pre_trade_check(
            order, account, positions, features, quote,
        )
        audit.risk_verdict = verdict

        if not verdict.approved:
            logger.info(
                "Risk rejected %s %s: %s",
                signal.symbol,
                side.value,
                ", ".join(verdict.reasons),
            )
            return

        final_order = order.model_copy(update={"qty": verdict.adjusted_qty})
        audit.order_request = final_order

        result = await self._order_mgr.place_order(final_order)
        audit.order_result = result

        self._trades_today.append({
            "timestamp": datetime.now(tz=self._tz).isoformat(),
            "symbol": signal.symbol,
            "side": side.value,
            "qty": verdict.adjusted_qty,
            "signal_confidence": signal.confidence,
        })

    # ── EOD flatten ───────────────────────────────────────────────────

    async def _maybe_eod_flatten(self) -> None:
        """Close all positions once per session day after eod_flatten_time while RTH is open."""
        if not self._config.session.eod_flatten or self._eod_flatten_time is None:
            return
        if not self._should_flatten():
            return
        today = self._session_date_str()
        if self._eod_flatten_done_session_date == today:
            return
        if self._exec_client is None:
            return
        await self._flatten_positions()
        self._eod_flatten_done_session_date = today

    async def _flatten_positions(self) -> None:
        assert self._exec_client is not None
        logger.info("EOD flatten triggered — closing all positions")
        try:
            await self._exec_client.cancel_all_orders()
            results = await self._exec_client.close_all_positions()
            logger.info("Flattened %d positions", len(results))
        except Exception:
            logger.exception("Error during EOD flatten")

    # ── reconciliation ────────────────────────────────────────────────

    async def _reconcile(self) -> None:
        assert self._market_data is not None
        assert self._reconciler is not None
        try:
            broker_positions = await self._market_data.get_positions()
            self._reconciler.reconcile([], broker_positions)
        except Exception:
            logger.exception("Error during reconciliation")

    # ── universe ──────────────────────────────────────────────────────

    async def _get_universe(self) -> list[str]:
        """Return the current list of symbols to trade.

        When universe.base is 'custom', returns the fixed list from config.
        When 'scan' or 'hybrid', delegates to the MarketScanner which
        caches results and re-scans on the configured interval.
        """
        assert self._scanner is not None

        if self._config.universe.base == "custom":
            return list(self._config.universe.custom_symbols)

        if self._scanner.needs_rescan():
            try:
                symbols = await self._scanner.scan()
                logger.info("Scanner returned %d symbols", len(symbols))
                return symbols
            except Exception:
                logger.exception("Scanner failed — using cached symbols")
                cached = self._scanner.cached_symbols
                if cached:
                    return cached
                return list(self._config.universe.custom_symbols)

        return self._scanner.cached_symbols

    # ── daily reports (session calendar) ──────────────────────────────

    def _session_date_str(self) -> str:
        return self._now_eastern().strftime("%Y-%m-%d")

    def _maybe_rollover_daily_report(self) -> None:
        """When session calendar day advances, persist the prior day and reset buffers."""
        if self._report_gen is None or not self._config.monitoring.daily_report:
            return

        current = self._session_date_str()
        if self._active_report_date is None:
            self._active_report_date = current
            return
        if current == self._active_report_date:
            return

        completed_date = self._active_report_date
        trades_snap = list(self._trades_today)
        equity_snap = list(self._equity_curve)
        self._trades_today.clear()
        self._equity_curve.clear()
        self._active_report_date = current

        try:
            self._write_daily_report(completed_date, trades=trades_snap, equity_curve=equity_snap)
        except Exception:
            logger.exception(
                "Error generating daily report for completed session date %s",
                completed_date,
            )

    def _write_daily_report(
        self,
        report_date: str,
        *,
        trades: list[dict[str, Any]] | None = None,
        equity_curve: list[dict[str, Any]] | None = None,
    ) -> str:
        assert self._report_gen is not None
        t = self._trades_today if trades is None else trades
        e = self._equity_curve if equity_curve is None else equity_curve
        return self._report_gen.generate(
            report_date,
            t,
            e,
            {
                "log_dir": self._config.monitoring.log_dir,
                "report_format": self._config.monitoring.report_format,
                "report_max_equity_points": self._config.monitoring.report_max_equity_points,
            },
        )

    # ── shutdown ──────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        logger.info("Shutting down PaperRunner")
        self._running = False

        if self._exec_client is not None:
            try:
                await self._exec_client.cancel_all_orders()
            except Exception:
                logger.exception("Error cancelling orders during shutdown")

            if self._config.session.eod_flatten:
                try:
                    await self._exec_client.close_all_positions()
                except Exception:
                    logger.exception("Error closing positions during shutdown")

        if self._report_gen is not None:
            try:
                if self._config.monitoring.daily_report:
                    report_date = self._active_report_date or self._session_date_str()
                else:
                    # Single end-of-run report (legacy) when daily reports are disabled.
                    report_date = self._session_date_str()
                report_path = self._write_daily_report(report_date)
                logger.info("Daily report saved to %s", report_path)
            except Exception:
                logger.exception("Error generating daily report")

        if self._market_data is not None:
            await self._market_data.close()
        if self._exec_client is not None:
            await self._exec_client.close()

        logger.info(
            "PaperRunner shutdown complete | cycles=%d trades=%d",
            self._cycle_count,
            len(self._trades_today),
        )
