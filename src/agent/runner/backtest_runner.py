"""Backtest execution runner — loads data, runs engine, reports results."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent.backtest.analytics import BacktestAnalytics
from src.agent.backtest.engine import BacktestEngine, BacktestResult
from src.agent.config import AppConfig, load_config
from src.agent.data.market_data import AlpacaMarketDataClient

logger = logging.getLogger("trading_agent")

_OUTPUT_DIR = Path("output")


class BacktestRunner:
    """Orchestrates a full backtest: data loading → engine → analytics → report."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._engine = BacktestEngine(config)
        self._analytics = BacktestAnalytics()
        self._output_dir = _OUTPUT_DIR / datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ── public API ────────────────────────────────────────────────────

    async def run(self) -> None:
        """Load historical data, run the backtest, compute analytics,
        print and save the results."""
        symbols = self._resolve_symbols()
        logger.info(
            "Starting backtest | symbols=%d start=%s end=%s capital=%.2f",
            len(symbols),
            self._config.backtest.start_date,
            self._config.backtest.end_date,
            self._config.backtest.initial_capital,
        )

        bars_data = await self._load_data(symbols)

        result = await self._engine.run(symbols, bars_data)

        metrics = self._analytics.compute(result)

        report_text = self._analytics.generate_report(
            metrics, format=self._config.monitoring.report_format,
        )
        print(report_text)

        self._save_results(result, metrics)

        logger.info(
            "Backtest complete | pnl=%.2f return=%.2f%% sharpe=%.4f trades=%d",
            metrics["total_pnl"],
            metrics["total_return_pct"],
            metrics["sharpe_ratio"],
            metrics["total_trades"],
        )

    async def run_walk_forward(self) -> None:
        """Run walk-forward validation over the configured date range."""
        symbols = self._resolve_symbols()
        logger.info(
            "Starting walk-forward | symbols=%d train=%dd validate=%dd step=%dd",
            len(symbols),
            self._config.backtest.walk_forward.train_days,
            self._config.backtest.walk_forward.validate_days,
            self._config.backtest.walk_forward.step_days,
        )

        bars_data = await self._load_data(symbols)

        results = await self._engine.run_walk_forward(symbols, bars_data)

        self._output_dir.mkdir(parents=True, exist_ok=True)

        all_metrics: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            m = self._analytics.compute(result)
            all_metrics.append(m)

            report_text = self._analytics.generate_report(
                m, format=self._config.monitoring.report_format,
            )
            print(f"\n{'='*60}")
            print(f"Walk-forward window {i + 1}/{len(results)}")
            print(f"{'='*60}")
            print(report_text)

            window_dir = self._output_dir / f"window_{i:03d}"
            window_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(result, m, output_dir=window_dir)

        summary_path = self._output_dir / "walk_forward_summary.json"
        summary_path.write_text(
            json.dumps(all_metrics, indent=2, default=str), encoding="utf-8",
        )
        logger.info(
            "Walk-forward complete | windows=%d summary=%s",
            len(results),
            summary_path,
        )

    async def evaluate(self, walk_forward: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run a deterministic evaluation and return metrics.

        Returns:
            Tuple of (single-run metrics, walk-forward metrics list).
            If walk_forward=False, the second element is [].
        """
        symbols = self._resolve_symbols()
        bars_data = await self._load_data(symbols)

        result = await self._engine.run(symbols, bars_data)
        base_metrics = self._analytics.compute(result)

        wf_metrics: list[dict[str, Any]] = []
        if walk_forward:
            results = await self._engine.run_walk_forward(symbols, bars_data)
            for window in results:
                wf_metrics.append(self._analytics.compute(window))

        return base_metrics, wf_metrics

    # ── data loading ──────────────────────────────────────────────────

    async def _load_data(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch historical bars from Alpaca for each symbol."""
        client = AlpacaMarketDataClient(self._config.alpaca)
        bars_data: dict[str, pd.DataFrame] = {}

        tf = self._config.strategy.timeframe.replace("m", "Min").replace("h", "Hour")

        try:
            for symbol in symbols:
                logger.info("Loading data for %s", symbol)
                df = await client.get_bars(
                    symbol,
                    timeframe=tf,
                    start=self._config.backtest.start_date,
                    end=self._config.backtest.end_date,
                )
                bars_data[symbol] = df
                logger.info("Loaded %d bars for %s", len(df), symbol)
        finally:
            await client.close()

        return bars_data

    # ── symbol resolution ─────────────────────────────────────────────

    def _resolve_symbols(self) -> list[str]:
        if self._config.universe.custom_symbols:
            return self._config.universe.custom_symbols
        return []

    # ── output ────────────────────────────────────────────────────────

    def _save_results(
        self,
        result: BacktestResult,
        metrics: dict[str, Any],
        *,
        output_dir: Path | None = None,
    ) -> None:
        out = output_dir or self._output_dir
        out.mkdir(parents=True, exist_ok=True)

        (out / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8",
        )

        (out / "trades.json").write_text(
            json.dumps(result.trades, indent=2, default=str), encoding="utf-8",
        )

        (out / "equity_curve.json").write_text(
            json.dumps(result.equity_curve, indent=2, default=str),
            encoding="utf-8",
        )

        (out / "daily_pnl.json").write_text(
            json.dumps(result.daily_pnl, indent=2, default=str),
            encoding="utf-8",
        )

        (out / "config_snapshot.json").write_text(
            json.dumps(result.config_snapshot, indent=2, default=str),
            encoding="utf-8",
        )

        logger.info("Results saved to %s", out.resolve())
