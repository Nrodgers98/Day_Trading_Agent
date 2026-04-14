"""Shared data loading utilities for the Streamlit dashboard.

Reads backtest output files, audit logs, daily reports, and config.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"


# ── Backtest results ─────────────────────────────────────────────────


def list_backtest_runs() -> list[dict[str, Any]]:
    """Return a list of available backtest run directories, newest first."""
    if not OUTPUT_DIR.exists():
        return []

    runs: list[dict[str, Any]] = []
    for d in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        metrics_path = d / "metrics.json"
        wf_path = d / "walk_forward_summary.json"
        has_metrics = metrics_path.exists()
        has_walk_forward = wf_path.exists()
        if has_metrics or has_walk_forward:
            metrics = (
                _load_json(metrics_path)
                if has_metrics
                else _aggregate_walk_forward_metrics(_load_json(wf_path))
            )
            runs.append({
                "name": d.name,
                "path": d,
                "metrics": metrics,
                "has_walk_forward": has_walk_forward,
            })
    return runs


def load_backtest_result(run_path: Path) -> dict[str, Any]:
    """Load all artifacts for a single backtest run."""
    result: dict[str, Any] = {}

    for name in ("metrics", "trades", "equity_curve", "daily_pnl", "config_snapshot"):
        fpath = run_path / f"{name}.json"
        result[name] = _load_json(fpath) if fpath.exists() else None

    wf_path = run_path / "walk_forward_summary.json"
    if wf_path.exists():
        result["walk_forward"] = _load_json(wf_path)
        if result.get("metrics") is None:
            result["metrics"] = _aggregate_walk_forward_metrics(result["walk_forward"])

    window_dirs = sorted(run_path.glob("window_*"))
    if window_dirs:
        result["windows"] = []
        for wd in window_dirs:
            result["windows"].append(load_backtest_result(wd))

    return result


def equity_curve_to_df(equity_curve: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert equity curve list to a DataFrame with datetime index."""
    if not equity_curve:
        return pd.DataFrame(columns=["timestamp", "equity"])
    df = pd.DataFrame(equity_curve)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def trades_to_df(trades: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert trades list to a DataFrame."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def daily_pnl_to_df(daily_pnl: list[dict[str, Any]]) -> pd.DataFrame:
    if not daily_pnl:
        return pd.DataFrame(columns=["date", "pnl", "equity"])
    df = pd.DataFrame(daily_pnl)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_drawdown_series(equity_curve_df: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown percentage from equity curve."""
    if equity_curve_df.empty or "equity" not in equity_curve_df.columns:
        return pd.DataFrame(columns=["timestamp", "drawdown_pct"])

    df = equity_curve_df.copy()
    df["peak"] = df["equity"].cummax()
    df["drawdown_pct"] = (df["peak"] - df["equity"]) / df["peak"] * 100.0
    return df[["timestamp", "drawdown_pct"]]


# ── Audit logs ────────────────────────────────────────────────────────


def list_audit_dates() -> list[str]:
    """Return available audit log dates, newest first."""
    if not LOGS_DIR.exists():
        return []
    dates: list[str] = []
    for f in sorted(LOGS_DIR.glob("audit_*.jsonl"), reverse=True):
        date_str = f.stem.replace("audit_", "")
        dates.append(date_str)
    return dates


def load_audit_log(date: str, symbol: str | None = None) -> list[dict[str, Any]]:
    """Load audit records for a given date."""
    fpath = LOGS_DIR / f"audit_{date}.jsonl"
    if not fpath.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if symbol and record.get("symbol") != symbol:
                    continue
                records.append(record)
            except json.JSONDecodeError:
                continue
    return records


def audit_records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten audit records into a DataFrame suitable for display."""
    if not records:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for r in records:
        signal = r.get("signal") or {}
        verdict = r.get("risk_verdict") or {}
        order = r.get("order_result") or {}
        rows.append({
            "timestamp": r.get("timestamp", ""),
            "symbol": r.get("symbol", ""),
            "action": signal.get("action", ""),
            "side": signal.get("side", ""),
            "confidence": signal.get("confidence", 0.0),
            "ml_score": signal.get("ml_score", 0.0),
            "rule_score": signal.get("rule_score", 0.0),
            "reason": signal.get("reason", ""),
            "risk_approved": verdict.get("approved", None),
            "risk_reasons": ", ".join(verdict.get("reasons", [])),
            "adjusted_qty": verdict.get("adjusted_qty", 0.0),
            "order_status": order.get("status", ""),
            "filled_qty": order.get("filled_qty", 0.0),
            "filled_price": order.get("filled_avg_price", 0.0),
        })

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# ── Daily reports ─────────────────────────────────────────────────────


def list_daily_reports() -> list[dict[str, Any]]:
    """Return available daily reports, newest first."""
    reports_dir = LOGS_DIR / "reports"
    if not reports_dir.exists():
        return []

    reports: list[dict[str, Any]] = []
    for f in sorted(reports_dir.glob("report_*"), reverse=True):
        try:
            data = _load_json(f) if f.suffix == ".json" else {"raw": f.read_text()}
            reports.append({"date": f.stem.replace("report_", ""), "path": f, "data": data})
        except Exception:
            continue
    return reports


def daily_reports_summary_dataframe(
    reports: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """One row per report date (chronological), from ``summary`` blocks."""
    items = reports if reports is not None else list_daily_reports()
    rows: list[dict[str, Any]] = []
    for item in items:
        date_str = str(item.get("date", ""))
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        row: dict[str, Any] = {
            "date": pd.to_datetime(date_str, errors="coerce"),
            "total_pnl": float(summary.get("total_pnl", 0.0) or 0.0),
            "trade_count": int(summary.get("trade_count", 0) or 0),
            "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
            "max_drawdown_pct": float(summary.get("max_drawdown_pct", 0.0) or 0.0),
            "win_count": int(summary.get("win_count", 0) or 0),
            "loss_count": int(summary.get("loss_count", 0) or 0),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["cumulative_pnl"] = df["total_pnl"].cumsum()
    df["rolling_7d_pnl"] = df["total_pnl"].rolling(7, min_periods=1).sum()
    df["rolling_7d_win_rate"] = df["win_rate"].rolling(7, min_periods=1).mean()
    return df


def trades_concat_from_reports(
    reports: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """All ``trades`` entries from JSON reports (order not guaranteed)."""
    items = reports if reports is not None else list_daily_reports()
    out: list[dict[str, Any]] = []
    for item in items:
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        trades = data.get("trades")
        if isinstance(trades, list):
            out.extend(t for t in trades if isinstance(t, dict))
    return out


def stitched_equity_curve_from_reports(
    reports: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Concatenate ``equity_curve`` arrays from JSON reports; sorted by timestamp."""
    items = reports if reports is not None else list_daily_reports()
    points: list[dict[str, Any]] = []
    for item in items:
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        curve = data.get("equity_curve")
        if not isinstance(curve, list):
            continue
        for pt in curve:
            if not isinstance(pt, dict):
                continue
            ts = pt.get("timestamp")
            if not ts:
                continue
            try:
                eq = float(pt.get("equity", 0.0) or 0.0)
            except (TypeError, ValueError):
                eq = 0.0
            points.append({"timestamp": ts, "equity": eq})
    df = pd.DataFrame(points)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "equity"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Application log ──────────────────────────────────────────────────


def tail_app_log(n_lines: int = 100, level_filter: str | None = None) -> list[dict[str, Any]]:
    """Read the last N lines across structured JSON app logs."""
    log_paths = sorted(LOGS_DIR.glob("agent*.log"))
    if not log_paths:
        return []

    entries: list[dict[str, Any]] = []
    per_file_tail = max(n_lines, 200)
    for log_path in log_paths:
        try:
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()[-per_file_tail:]
        except Exception:
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if level_filter and entry.get("level", "") != level_filter:
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                continue

    entries.sort(key=lambda e: e.get("timestamp", ""))
    return entries[-n_lines:]


# ── Config ────────────────────────────────────────────────────────────


def load_yaml_config(name: str = "default.yaml") -> dict[str, Any]:
    """Load a YAML config file."""
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml_config(name: str, data: dict[str, Any]) -> None:
    """Write a config dict to ``config/<name>``. Overwrites the file."""
    path = CONFIG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def list_config_files() -> list[str]:
    """Return available config file names."""
    if not CONFIG_DIR.exists():
        return []
    return [f.name for f in sorted(CONFIG_DIR.glob("*.yaml"))]


# ── Helpers ───────────────────────────────────────────────────────────


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _aggregate_walk_forward_metrics(wf_metrics: Any) -> dict[str, Any]:
    """Build a lightweight aggregate metrics dict from walk-forward windows."""
    if not isinstance(wf_metrics, list) or not wf_metrics:
        return {}

    numeric_keys = [
        "total_pnl",
        "total_return_pct",
        "win_rate",
        "profit_factor",
        "sharpe_ratio",
        "max_drawdown",
        "avg_hold_minutes",
        "total_trades",
        "avg_trades_per_day",
        "exposure_pct",
    ]

    out: dict[str, Any] = {}
    count = float(len(wf_metrics))
    for key in numeric_keys:
        vals: list[float] = []
        for row in wf_metrics:
            if not isinstance(row, dict):
                continue
            try:
                vals.append(float(row.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        if vals:
            out[key] = sum(vals) / count
    out["source"] = "walk_forward_aggregate"
    out["windows"] = len(wf_metrics)
    return out
