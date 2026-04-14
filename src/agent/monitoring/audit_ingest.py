"""Parse audit JSONL for session-dated trade rows and summary aggregates."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.agent.models import AuditRecord


def _aware_in_tz(dt: datetime, tz: ZoneInfo) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def session_date_str(dt: datetime, tz: ZoneInfo) -> str:
    return _aware_in_tz(dt, tz).strftime("%Y-%m-%d")


def _parse_ts(val: Any) -> datetime | None:
    if val is None:
        return None
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_audit_line_for_trade(line: str) -> tuple[datetime | None, dict[str, Any] | None]:
    """Return (timestamp, trade_row) if the line has a non-null ``order_result``."""
    line = line.strip()
    if not line:
        return None, None

    try:
        rec = AuditRecord.model_validate_json(line)
    except (ValueError, json.JSONDecodeError):
        rec = None

    if rec is not None:
        if rec.order_result is None:
            return rec.timestamp, None
        or_ = rec.order_result
        conf = float(rec.signal.confidence) if rec.signal else 0.0
        row = {
            "timestamp": rec.timestamp.isoformat(),
            "symbol": or_.symbol,
            "side": or_.side.value,
            "qty": float(or_.qty),
            "signal_confidence": conf,
            "client_order_id": or_.client_order_id,
            "broker_order_id": or_.broker_order_id,
            "status": or_.status.value,
        }
        return rec.timestamp, row

    try:
        raw: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        return None, None

    ts = _parse_ts(raw.get("timestamp"))
    if ts is None:
        return None, None

    or_ = raw.get("order_result")
    if not isinstance(or_, dict):
        return ts, None

    sig = raw.get("signal") if isinstance(raw.get("signal"), dict) else {}
    conf = float(sig.get("confidence", 0.0) or 0.0)
    side_raw = or_.get("side", "flat")
    if isinstance(side_raw, dict):
        side = str(side_raw.get("value", "flat"))
    else:
        side = str(side_raw)
    status_raw = or_.get("status", "")
    if isinstance(status_raw, dict):
        status = str(status_raw.get("value", ""))
    else:
        status = str(status_raw)
    row = {
        "timestamp": ts.isoformat(),
        "symbol": str(or_.get("symbol", raw.get("symbol", ""))),
        "side": str(side),
        "qty": float(or_.get("qty", 0.0) or 0.0),
        "signal_confidence": conf,
        "client_order_id": str(or_.get("client_order_id", "")),
        "broker_order_id": str(or_.get("broker_order_id", "")),
        "status": str(status).lower(),
    }
    return ts, row


def collect_audit_trades_by_session_date(
    log_dir: Path,
    start: date,
    end: date,
    tz: ZoneInfo,
) -> dict[str, list[dict[str, Any]]]:
    """Bucket trade rows from every ``audit_*.jsonl`` by session calendar date in *tz*.

    Filenames like ``audit_2026-04-02.jsonl`` can contain rows that belong to an
    adjacent calendar day when timestamps cross midnight in UTC vs US/Eastern,
    so we always bucket by parsed timestamp — not by filename.
    """
    start_s, end_s = start.isoformat(), end.isoformat()
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: dict[str, set[str]] = defaultdict(set)

    for path in sorted(log_dir.glob("audit_*.jsonl")):
        if not path.is_file():
            continue
        try:
            fh = path.open(encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                ts, trade = parse_audit_line_for_trade(line)
                if ts is None or trade is None:
                    continue
                d = session_date_str(ts, tz)
                if not (start_s <= d <= end_s):
                    continue
                dedupe = str(
                    trade.get("client_order_id")
                    or trade.get("broker_order_id")
                    or f"{trade.get('timestamp')}|{trade.get('symbol')}|{trade.get('side')}|{trade.get('qty')}"
                )
                if dedupe in seen_keys[d]:
                    continue
                seen_keys[d].add(dedupe)
                by_day[d].append(trade)

    return by_day


def collect_audit_summary_for_session_date(
    log_dir: Path,
    day: str,
    tz: ZoneInfo,
) -> dict[str, Any]:
    """Aggregate audit JSONL rows for one session date (raw JSON, schema-tolerant)."""
    n_records = 0
    n_order_result = 0
    n_filled = 0
    n_rejected = 0
    n_risk_reject = 0
    n_signal_buy = 0
    n_signal_sell = 0
    n_signal_hold = 0

    for path in sorted(log_dir.glob("audit_*.jsonl")):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _parse_ts(raw.get("timestamp"))
                if ts is None:
                    continue
                if session_date_str(ts, tz) != day:
                    continue

                n_records += 1

                sig = raw.get("signal") if isinstance(raw.get("signal"), dict) else {}
                act_raw = sig.get("action", "hold")
                if isinstance(act_raw, dict):
                    action = str(act_raw.get("value", "hold")).lower()
                else:
                    action = str(act_raw).lower()
                if action == "buy":
                    n_signal_buy += 1
                elif action == "sell":
                    n_signal_sell += 1
                else:
                    n_signal_hold += 1

                rv = raw.get("risk_verdict")
                if isinstance(rv, dict) and rv.get("approved") is False:
                    n_risk_reject += 1

                or_ = raw.get("order_result")
                if isinstance(or_, dict):
                    n_order_result += 1
                    st_raw = or_.get("status", "")
                    if isinstance(st_raw, dict):
                        st = str(st_raw.get("value", "")).lower()
                    else:
                        st = str(st_raw).lower()
                    if "filled" in st:
                        n_filled += 1
                    if "rejected" in st:
                        n_rejected += 1

    return {
        "audit_records": n_records,
        "audit_order_results": n_order_result,
        "audit_filled_orders": n_filled,
        "audit_rejected_orders": n_rejected,
        "audit_risk_rejections": n_risk_reject,
        "audit_signals_buy": n_signal_buy,
        "audit_signals_sell": n_signal_sell,
        "audit_signals_hold": n_signal_hold,
    }
