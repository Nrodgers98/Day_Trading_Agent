"""
Performance analytics — computes key metrics from a BacktestResult.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any

import numpy as np

from src.agent.backtest.engine import BacktestResult

_MARKET_MINUTES_PER_DAY = 390  # 09:30–16:00 = 6.5 h


class BacktestAnalytics:
    """Stateless calculator that turns a BacktestResult into a rich
    metrics dictionary and optional formatted report."""

    def compute(self, result: BacktestResult) -> dict[str, Any]:
        """Return a flat metrics dict with per-symbol breakdown."""
        trades = result.trades
        equity_curve = result.equity_curve
        daily_pnl = result.daily_pnl

        total_trades = len(trades)
        total_pnl = sum(t["pnl"] for t in trades)

        initial_eq = equity_curve[0]["equity"] if equity_curve else 0.0
        final_eq = equity_curve[-1]["equity"] if equity_curve else 0.0
        total_return_pct = (
            ((final_eq / initial_eq) - 1.0) * 100.0
            if initial_eq > 0
            else 0.0
        )

        wins = [t for t in trades if t["pnl"] > 0]
        win_rate = len(wins) / total_trades * 100.0 if total_trades else 0.0

        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        sharpe_ratio = self._sharpe(daily_pnl)
        max_drawdown, max_dd_duration = self._drawdown(equity_curve)

        avg_hold = (
            float(np.mean([t["hold_minutes"] for t in trades]))
            if trades
            else 0.0
        )

        num_days = len(daily_pnl)
        avg_trades_day = total_trades / num_days if num_days else 0.0

        total_hold_min = sum(t["hold_minutes"] for t in trades)
        total_market_min = num_days * _MARKET_MINUTES_PER_DAY
        exposure_pct = (
            min(total_hold_min / total_market_min * 100.0, 100.0)
            if total_market_min > 0
            else 0.0
        )

        per_symbol = self._per_symbol(trades)

        return {
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_duration_days": max_dd_duration,
            "avg_hold_minutes": round(avg_hold, 1),
            "total_trades": total_trades,
            "avg_trades_per_day": round(avg_trades_day, 2),
            "exposure_pct": round(exposure_pct, 2),
            "per_symbol_breakdown": per_symbol,
        }

    def generate_report(
        self,
        metrics: dict[str, Any],
        format: str = "json",
    ) -> str:
        """Render *metrics* as json, csv, or markdown."""
        if format == "csv":
            return self._to_csv(metrics)
        if format == "markdown":
            return self._to_markdown(metrics)
        return json.dumps(metrics, indent=2, default=str)

    # ── internal computations ─────────────────────────────────────────

    @staticmethod
    def _sharpe(daily_pnl: list[dict[str, Any]]) -> float:
        if len(daily_pnl) < 2:
            return 0.0
        returns = [
            d["pnl"] / max(d["equity"] - d["pnl"], 1.0)
            for d in daily_pnl
        ]
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))
        if std_r == 0:
            return 0.0
        return mean_r / std_r * float(np.sqrt(252))

    @staticmethod
    def _drawdown(
        equity_curve: list[dict[str, Any]],
    ) -> tuple[float, int]:
        if not equity_curve:
            return 0.0, 0

        equities = [e["equity"] for e in equity_curve]
        peak = equities[0]
        max_dd = 0.0
        max_duration = 0
        dd_start: datetime | None = None

        for i, eq in enumerate(equities):
            if eq >= peak:
                peak = eq
                if dd_start is not None:
                    ts = datetime.fromisoformat(
                        equity_curve[i]["timestamp"],
                    )
                    duration = (ts - dd_start).days
                    max_duration = max(max_duration, duration)
                    dd_start = None
            else:
                dd_pct = (peak - eq) / peak * 100.0
                if dd_pct > max_dd:
                    max_dd = dd_pct
                if dd_start is None:
                    dd_start = datetime.fromisoformat(
                        equity_curve[i]["timestamp"],
                    )

        if dd_start is not None:
            last_ts = datetime.fromisoformat(
                equity_curve[-1]["timestamp"],
            )
            max_duration = max(max_duration, (last_ts - dd_start).days)

        return max_dd, max_duration

    @staticmethod
    def _per_symbol(
        trades: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for t in trades:
            sym = t["symbol"]
            if sym not in buckets:
                buckets[sym] = {"pnl": 0.0, "trades": 0, "wins": 0}
            buckets[sym]["pnl"] += t["pnl"]
            buckets[sym]["trades"] += 1
            if t["pnl"] > 0:
                buckets[sym]["wins"] += 1
        return {
            sym: {
                "pnl": round(d["pnl"], 2),
                "trades": d["trades"],
                "win_rate": (
                    round(d["wins"] / d["trades"] * 100.0, 2)
                    if d["trades"]
                    else 0.0
                ),
            }
            for sym, d in buckets.items()
        }

    # ── formatters ────────────────────────────────────────────────────

    @staticmethod
    def _to_csv(metrics: dict[str, Any]) -> str:
        buf = io.StringIO()
        flat = {
            k: v for k, v in metrics.items() if k != "per_symbol_breakdown"
        }
        writer = csv.DictWriter(buf, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
        return buf.getvalue()

    @staticmethod
    def _to_markdown(metrics: dict[str, Any]) -> str:
        lines = [
            "# Backtest Report",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in metrics.items():
            if k == "per_symbol_breakdown":
                continue
            lines.append(f"| {k} | {v} |")

        breakdown = metrics.get("per_symbol_breakdown", {})
        if breakdown:
            lines += [
                "",
                "## Per-Symbol Breakdown",
                "",
                "| Symbol | PnL | Trades | Win Rate |",
                "|--------|-----|--------|----------|",
            ]
            for sym, d in breakdown.items():
                lines.append(
                    f"| {sym} | {d['pnl']} | {d['trades']} | {d['win_rate']}% |"
                )

        return "\n".join(lines)
