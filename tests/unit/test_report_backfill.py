"""Tests for log-based daily report backfill."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.agent.monitoring.report_backfill import (
    _ACCOUNT_EQUITY_RE,
    _ORDER_PLACED_RE,
    _session_date_str,
    _trade_from_order_placed_message,
    backfill_reports_from_logs,
)


def test_order_placed_regex() -> None:
    msg = (
        "order_placed | client_order_id=abc broker_id=brk symbol=NVDA "
        "side=long qty=1.5 status=filled"
    )
    assert _ORDER_PLACED_RE.search(msg)


def test_trade_from_order_placed_message() -> None:
    msg = (
        "order_placed | client_order_id=c1 broker_id=b1 symbol=AAPL "
        "side=short qty=2 status=pending_new"
    )
    t = _trade_from_order_placed_message(msg, "2026-04-02T10:00:00-04:00")
    assert t is not None
    assert t["symbol"] == "AAPL"
    assert t["side"] == "short"
    assert t["qty"] == 2.0
    assert t["status"] == "pending_new"


def test_account_equity_regex() -> None:
    assert _ACCOUNT_EQUITY_RE.search("Account connected | equity=100000.00 cash=50000.00")


def test_session_date_str_eastern() -> None:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("US/Eastern")
    assert _session_date_str("2026-04-02T23:00:00-04:00", tz) == "2026-04-02"
    assert _session_date_str("2026-04-03T02:00:00+00:00", tz) == "2026-04-02"


def test_backfill_from_agent_log(tmp_path: Path) -> None:
    log_dir = tmp_path
    log_dir.mkdir(parents=True, exist_ok=True)
    agent = log_dir / "agent_paper.log"
    agent.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-02T15:30:00-04:00",
                "level": "INFO",
                "logger": "trading_agent",
                "message": (
                    "order_placed | client_order_id=x broker_id=y symbol=QQQ "
                    "side=long qty=1.0 status=filled"
                ),
            },
        )
        + "\n"
        + json.dumps(
            {
                "timestamp": "2026-04-02T15:31:00-04:00",
                "level": "INFO",
                "logger": "trading_agent",
                "message": "Account connected | equity=100001.00 cash=90000.00 buying_power=200000.00 status=ACTIVE",
            },
        )
        + "\n"
        + json.dumps(
            {
                "timestamp": "2026-04-02T15:32:00-04:00",
                "level": "INFO",
                "logger": "trading_agent",
                "message": (
                    "signal_evaluated | symbol=QQQ action=buy side=long confidence=0.75 "
                    "sentiment=None sentiment_effect=none reason=breakout"
                ),
            },
        )
        + "\n",
        encoding="utf-8",
    )

    out = backfill_reports_from_logs(
        log_dir=log_dir,
        start=date(2026, 4, 2),
        end=date(2026, 4, 2),
        session_tz="US/Eastern",
        mode="paper",
        dry_run=False,
        force=True,
        report_format="json",
    )
    assert "2026-04-02" in out["written"]
    report = json.loads((log_dir / "reports" / "report_2026-04-02.json").read_text(encoding="utf-8"))
    assert report["summary"]["trade_count"] == 1
    assert report["summary"]["date"] == "2026-04-02"
    assert len(report["trades"]) == 1
    assert report["trades"][0]["symbol"] == "QQQ"
    assert report["summary"].get("log_signals_buy") == 1
    assert len(report["summary"]) >= 5
    assert "equity_curve" in report
    assert isinstance(report["equity_curve"], list)
    assert len(report["equity_curve"]) >= 1


def test_backfill_skips_existing_without_force(tmp_path: Path) -> None:
    log_dir = tmp_path
    (log_dir / "reports").mkdir(parents=True)
    existing = log_dir / "reports" / "report_2026-04-03.json"
    existing.write_text('{"summary": {"date": "2026-04-03"}, "trades": []}', encoding="utf-8")

    out = backfill_reports_from_logs(
        log_dir=log_dir,
        start=date(2026, 4, 3),
        end=date(2026, 4, 3),
        session_tz="US/Eastern",
        mode="paper",
        force=False,
    )
    assert "2026-04-03" in out["skipped"]
    assert not out["written"]


def test_only_days_with_log_activity(tmp_path: Path) -> None:
    log_dir = tmp_path
    out = backfill_reports_from_logs(
        log_dir=log_dir,
        start=date(2026, 4, 1),
        end=date(2026, 4, 5),
        session_tz="US/Eastern",
        mode="paper",
        force=True,
        only_days_with_log_activity=True,
    )
    assert not out["written"]
    assert len(out["skipped"]) == 5
