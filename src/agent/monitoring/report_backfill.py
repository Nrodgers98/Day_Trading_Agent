"""Reconstruct daily JSON reports from agent logs and audit JSONL."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from src.agent.monitoring.audit_ingest import (
    collect_audit_summary_for_session_date,
    collect_audit_trades_by_session_date,
)
from src.agent.monitoring.reports import DailyReportGenerator

_ORDER_PLACED_RE = re.compile(
    r"order_placed \| client_order_id=(\S+) broker_id=(\S+) symbol=(\S+) "
    r"side=(\S+) qty=([\d.]+) status=(\S+)"
)
_ACCOUNT_EQUITY_RE = re.compile(r"Account connected \| equity=([\d.]+)")
_SIGNAL_EVAL_RE = re.compile(
    r"signal_evaluated \| symbol=(\S+) action=(buy|sell|hold|close)\b",
    re.IGNORECASE,
)


def _session_date_str(ts_raw: str, tz: ZoneInfo) -> str:
    """Calendar YYYY-MM-DD in *tz* for an ISO8601 timestamp string."""
    raw = ts_raw.strip()
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz).strftime("%Y-%m-%d")


def _agent_log_paths(log_dir: Path, mode: Literal["paper", "live"]) -> list[Path]:
    """Oldest rotated chunk first, then the active log (newest)."""
    stem = f"agent_{mode}.log"
    backups = sorted(
        log_dir.glob(f"{stem}.*"),
        key=lambda p: int(p.name.split(".")[-1]),
    )
    paths = [p for p in backups if p.is_file()]
    main = log_dir / stem
    if main.is_file():
        paths.append(main)
    return paths


def _all_json_log_files(log_dir: Path, modes: list[Literal["paper", "live"]]) -> list[Path]:
    """Paper/live agent logs plus any other top-level ``*.log`` / ``*.log.N`` JSONL files.

    ``order_placed`` lines are emitted by the execution logger, which may not be a
    child of ``trading_agent``; scanning every log file catches them when present.
    """
    seen: set[Path] = set()
    ordered: list[Path] = []
    for mode in modes:
        for p in _agent_log_paths(log_dir, mode):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                ordered.append(p)

    for pattern in ("*.log", "*.log.*"):
        for p in sorted(log_dir.glob(pattern)):
            if not p.is_file():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            # Skip obvious non-runner logs (no structured agent messages expected).
            if p.name.startswith("improvement_") or p.name.startswith("agent_sentiment"):
                continue
            seen.add(rp)
            ordered.append(p)
    return ordered


def _modes_arg(mode: Literal["paper", "live", "both"]) -> list[Literal["paper", "live"]]:
    if mode == "both":
        return ["paper", "live"]
    return [mode]


def _trade_from_order_placed_message(message: str, ts_iso: str) -> dict[str, Any] | None:
    m = _ORDER_PLACED_RE.search(message)
    if not m:
        return None
    _cid, _bid, sym, side, qty_s, status_s = m.groups()
    return {
        "timestamp": ts_iso,
        "symbol": sym,
        "side": side,
        "qty": float(qty_s),
        "signal_confidence": 0.0,
        "status": status_s,
    }


def _equity_from_log_obj(obj: dict[str, Any]) -> list[tuple[str, float]]:
    """Return (timestamp_iso, equity) pairs extractable from one JSON log row."""
    ts = str(obj.get("timestamp", ""))
    msg = str(obj.get("message", ""))
    pts: list[tuple[str, float]] = []

    m = _ACCOUNT_EQUITY_RE.search(msg)
    if m and ts:
        pts.append((ts, float(m.group(1))))

    if obj.get("alert_type") == "drawdown":
        try:
            eq = float(obj.get("current_equity", 0.0))
        except (TypeError, ValueError):
            eq = 0.0
        if ts and eq > 0:
            pts.append((ts, eq))
    return pts


def _scan_agent_logs_for_range(
    log_dir: Path,
    modes: list[Literal["paper", "live"]],
    start: date,
    end: date,
    tz: ZoneInfo,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
]:
    """Bucket trades (from order_placed), equity points, and log activity by session date."""
    trades_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    equity_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    activity_by_day: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "log_signals_buy": 0,
            "log_signals_sell": 0,
            "log_signals_hold": 0,
            "log_signals_close": 0,
            "log_risk_rejections": 0,
        },
    )

    for path in _all_json_log_files(log_dir, modes):
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_raw = str(obj.get("timestamp", ""))
                d = _session_date_str(ts_raw, tz)
                if not d or not (start.isoformat() <= d <= end.isoformat()):
                    continue
                msg = str(obj.get("message", ""))
                trade = _trade_from_order_placed_message(msg, ts_raw)
                if trade is not None:
                    trades_by_day[d].append(trade)
                if "Risk rejected" in msg:
                    activity_by_day[d]["log_risk_rejections"] += 1
                m_sig = _SIGNAL_EVAL_RE.search(msg)
                if m_sig:
                    act = str(m_sig.group(2)).lower()
                    if act == "buy":
                        activity_by_day[d]["log_signals_buy"] += 1
                    elif act == "sell":
                        activity_by_day[d]["log_signals_sell"] += 1
                    elif act == "close":
                        activity_by_day[d]["log_signals_close"] += 1
                    else:
                        activity_by_day[d]["log_signals_hold"] += 1
                for ts_eq, eq in _equity_from_log_obj(obj):
                    d_eq = _session_date_str(ts_eq, tz)
                    if d_eq and start.isoformat() <= d_eq <= end.isoformat():
                        equity_by_day[d_eq].append({"timestamp": ts_eq, "equity": round(eq, 2)})

    for d in equity_by_day:
        equity_by_day[d].sort(key=lambda x: str(x.get("timestamp", "")))

    return trades_by_day, equity_by_day, activity_by_day


def _merge_activity_into_json_report(
    report_path: Path,
    *,
    session_date: str,
    log_dir: Path,
    tz: ZoneInfo,
    activity: dict[str, int],
) -> None:
    """Augment a generated JSON report with audit + log activity (non-trade metrics)."""
    audit_summary = collect_audit_summary_for_session_date(log_dir, session_date, tz)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.setdefault("summary", {})
    summary.update({k: v for k, v in audit_summary.items() if v})
    for k, v in activity.items():
        if v:
            summary[k] = int(v)
    report_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _report_stem_ext(report_format: str) -> str:
    return {"json": "json", "csv": "csv", "markdown": "md"}.get(report_format, "json")


def backfill_reports_from_logs(
    *,
    log_dir: Path,
    start: date,
    end: date,
    session_tz: str = "US/Eastern",
    mode: Literal["paper", "live", "both"] = "paper",
    dry_run: bool = False,
    force: bool = False,
    only_days_with_log_activity: bool = False,
    report_format: str = "json",
) -> dict[str, Any]:
    """Build ``report_YYYY-MM-DD`` files from audit JSONL and/or agent JSONL logs.

    **Trades** in ``trades[]`` come only from broker submissions (``order_result``
    in audit JSONL, or ``order_placed`` in logs). If the bot never passed risk,
    those stay empty even when many signals were evaluated.

    **Summary** (JSON reports only): after generation, merges per-day
    ``audit_*`` aggregates and **log** counts (``signal_evaluated`` buy/sell/hold,
    ``Risk rejected``) so activity is visible when ``trade_count`` is 0.

    Equity curve: ``Account connected`` and drawdown CRITICAL rows (``extra``
    fields) from agent logs — per-cycle equity is not logged historically, so
    drawdown may be approximate when few points exist.
    """
    tz = ZoneInfo(session_tz)
    modes = _modes_arg(mode)

    trades_by_day, equity_by_day, activity_by_day = _scan_agent_logs_for_range(
        log_dir, modes, start, end, tz,
    )
    audit_trades_by_day = collect_audit_trades_by_session_date(log_dir, start, end, tz)

    written: list[str] = []
    skipped: list[str] = []
    gen = DailyReportGenerator()
    ext = _report_stem_ext(report_format)

    d = start
    while d <= end:
        ds = d.isoformat()
        report_path = log_dir / "reports" / f"report_{ds}.{ext}"
        if report_path.exists() and not force:
            skipped.append(ds)
            d += timedelta(days=1)
            continue

        audit_path = log_dir / f"audit_{ds}.jsonl"
        audit_trades = audit_trades_by_day.get(ds, [])
        log_trades = trades_by_day.get(ds, [])
        trades = audit_trades if audit_trades else log_trades

        equity_curve = list(equity_by_day.get(ds, []))

        has_audit = audit_path.is_file() or bool(audit_trades)
        act = activity_by_day.get(ds, {})
        has_log_activity = bool(log_trades or equity_curve or sum(act.values()) > 0)
        if only_days_with_log_activity and not has_audit and not has_log_activity:
            skipped.append(f"{ds} (no activity)")
            d += timedelta(days=1)
            continue

        cfg = {"log_dir": str(log_dir), "report_format": report_format}
        if dry_run:
            written.append(
                f"{ds} (dry-run: trades={len(trades)} equity_pts={len(equity_curve)} "
                f"log_activity={dict(act)})"
            )
        else:
            gen.generate(ds, trades, equity_curve, cfg)
            if ext == "json":
                _merge_activity_into_json_report(
                    report_path,
                    session_date=ds,
                    log_dir=log_dir,
                    tz=tz,
                    activity=dict(act),
                )
            written.append(ds)

        d += timedelta(days=1)

    return {
        "written": written,
        "skipped": skipped,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "session_tz": session_tz,
        "modes": modes,
    }
