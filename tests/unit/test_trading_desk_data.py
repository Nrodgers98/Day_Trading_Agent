"""Dashboard data helpers for multi-day reports."""

from __future__ import annotations

from dashboard.utils.data_loader import (
    daily_reports_summary_dataframe,
    stitched_equity_curve_from_reports,
    trades_concat_from_reports,
)


def test_daily_reports_summary_and_stitched_equity() -> None:
    reports = [
        {
            "date": "2026-04-01",
            "path": None,
            "data": {
                "summary": {
                    "date": "2026-04-01",
                    "total_pnl": 100.0,
                    "trade_count": 2,
                    "win_rate": 0.5,
                    "max_drawdown_pct": 0.01,
                },
                "equity_curve": [
                    {"timestamp": "2026-04-01T10:00:00-04:00", "equity": 100_000.0},
                    {"timestamp": "2026-04-01T11:00:00-04:00", "equity": 100_050.0},
                ],
                "trades": [{"symbol": "A", "pnl": 50.0}],
            },
        },
        {
            "date": "2026-04-02",
            "path": None,
            "data": {
                "summary": {
                    "date": "2026-04-02",
                    "total_pnl": -20.0,
                    "trade_count": 1,
                    "win_rate": 0.0,
                    "max_drawdown_pct": 0.02,
                },
                "trades": [],
            },
        },
    ]
    df = daily_reports_summary_dataframe(reports)
    assert len(df) == 2
    assert df.iloc[-1]["date"].strftime("%Y-%m-%d") == "2026-04-02"
    assert abs(float(df.iloc[-1]["cumulative_pnl"]) - 80.0) < 1e-6

    eq = stitched_equity_curve_from_reports(reports)
    assert len(eq) == 2
    assert float(eq.iloc[-1]["equity"]) == 100_050.0

    trades = trades_concat_from_reports(reports)
    assert len(trades) == 1
    assert trades[0]["symbol"] == "A"
