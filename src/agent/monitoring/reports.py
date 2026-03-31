"""Daily report generator — PnL, win rate, drawdown, and trade summaries."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


class DailyReportGenerator:
    """Produces end-of-day performance reports in JSON, CSV, or Markdown."""

    def generate(
        self,
        date: str,
        trades: list[dict[str, Any]],
        equity_curve: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> str:
        """Build and save a daily report, returning the absolute path of the saved file."""
        log_dir = Path(config.get("log_dir", "logs"))
        report_format: str = config.get("report_format", "json")
        report_dir = log_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        summary = self._compute_summary(date, trades, equity_curve)

        ext = {"json": "json", "csv": "csv", "markdown": "md"}.get(report_format, "json")
        output_path = report_dir / f"report_{date}.{ext}"

        content = {
            "json": self._to_json,
            "csv": self._to_csv,
            "markdown": self._to_markdown,
        }.get(report_format, self._to_json)(summary, trades)

        output_path.write_text(content, encoding="utf-8")
        return str(output_path.resolve())

    # ── private helpers ──────────────────────────────────────────────

    @staticmethod
    def _compute_summary(
        date: str,
        trades: list[dict[str, Any]],
        equity_curve: list[dict[str, Any]],
    ) -> dict[str, Any]:
        total_pnl = sum(t.get("pnl", 0.0) for t in trades)
        trade_count = len(trades)
        winners = [t for t in trades if t.get("pnl", 0.0) > 0]
        win_rate = len(winners) / trade_count if trade_count else 0.0

        max_drawdown = 0.0
        peak = 0.0
        for point in equity_curve:
            equity = point.get("equity", 0.0)
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_drawdown:
                    max_drawdown = dd

        return {
            "date": date,
            "total_pnl": round(total_pnl, 2),
            "trade_count": trade_count,
            "win_count": len(winners),
            "loss_count": trade_count - len(winners),
            "win_rate": round(win_rate, 4),
            "max_drawdown_pct": round(max_drawdown, 6),
        }

    @staticmethod
    def _to_json(summary: dict[str, Any], trades: list[dict[str, Any]]) -> str:
        return json.dumps({"summary": summary, "trades": trades}, indent=2, default=str)

    @staticmethod
    def _to_csv(summary: dict[str, Any], trades: list[dict[str, Any]]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)

        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])

        writer.writerow([])
        if trades:
            writer.writerow(trades[0].keys())
            for trade in trades:
                writer.writerow(trade.values())

        return buf.getvalue()

    @staticmethod
    def _to_markdown(summary: dict[str, Any], trades: list[dict[str, Any]]) -> str:
        lines: list[str] = [
            f"# Daily Report — {summary['date']}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
        ]
        for key, value in summary.items():
            lines.append(f"| {key} | {value} |")

        lines += ["", "## Trades", ""]
        if trades:
            headers = list(trades[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for trade in trades:
                row = " | ".join(str(trade.get(h, "")) for h in headers)
                lines.append(f"| {row} |")
        else:
            lines.append("_No trades._")

        lines.append("")
        return "\n".join(lines)
