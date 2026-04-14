"""Tests for audit JSONL ingestion helpers."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.agent.monitoring.audit_ingest import (
    collect_audit_summary_for_session_date,
    collect_audit_trades_by_session_date,
    parse_audit_line_for_trade,
    session_date_str,
)


def test_session_date_str_respects_tz() -> None:
    tz = ZoneInfo("US/Eastern")
    dt = datetime.fromisoformat("2026-04-03T02:00:00+00:00")
    assert session_date_str(dt, tz) == "2026-04-02"


def test_collect_audit_trades_buckets_by_session_date_not_filename(tmp_path: Path) -> None:
    """Row filed under ``audit_2026-04-01`` but timestamp Eastern April 2 → report day 2."""
    tz = ZoneInfo("US/Eastern")
    log_dir = tmp_path
    wrong_name = log_dir / "audit_2026-04-01.jsonl"
    line = {
        "timestamp": "2026-04-02T14:00:00-04:00",
        "symbol": "QQQ",
        "order_result": {
            "symbol": "QQQ",
            "side": "long",
            "qty": 2.0,
            "status": "filled",
            "client_order_id": "c1",
            "broker_order_id": "b1",
        },
        "signal": {"action": "buy", "confidence": 0.7},
    }
    wrong_name.write_text(json.dumps(line) + "\n", encoding="utf-8")

    by_day = collect_audit_trades_by_session_date(
        log_dir, date(2026, 4, 1), date(2026, 4, 3), tz,
    )
    assert "2026-04-02" in by_day
    assert len(by_day["2026-04-02"]) == 1
    assert by_day["2026-04-02"][0]["symbol"] == "QQQ"
    assert "2026-04-01" not in by_day or not by_day.get("2026-04-01")


def test_parse_audit_line_loose_nested_enums() -> None:
    line = json.dumps(
        {
            "timestamp": "2026-04-02T10:00:00-04:00",
            "symbol": "X",
            "order_result": {
                "symbol": "X",
                "side": {"value": "long"},
                "qty": 1.0,
                "status": {"value": "filled"},
                "client_order_id": "a",
                "broker_order_id": "b",
            },
        },
    )
    ts, row = parse_audit_line_for_trade(line)
    assert ts is not None and row is not None
    assert row["side"] == "long"
    assert row["status"] == "filled"


def test_collect_audit_summary_counts_risk_reject(tmp_path: Path) -> None:
    tz = ZoneInfo("US/Eastern")
    p = tmp_path / "audit_2026-04-05.jsonl"
    p.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-05T10:00:00-04:00",
                "symbol": "A",
                "signal": {"action": "buy"},
                "risk_verdict": {"approved": False, "reasons": ["cap"]},
            },
        )
        + "\n",
        encoding="utf-8",
    )
    s = collect_audit_summary_for_session_date(tmp_path, "2026-04-05", tz)
    assert s["audit_records"] == 1
    assert s["audit_risk_rejections"] == 1
