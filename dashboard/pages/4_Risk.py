"""Risk Monitor — rejection analysis, drawdown tracking, alert log."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.utils.charts import (
    daily_pnl_chart,
    drawdown_chart,
    rejection_reasons_pie,
)
from dashboard.utils.data_loader import (
    audit_records_to_df,
    compute_drawdown_series,
    daily_pnl_to_df,
    equity_curve_to_df,
    list_audit_dates,
    list_backtest_runs,
    load_audit_log,
    load_backtest_result,
    tail_app_log,
)

st.set_page_config(page_title="Risk Monitor", page_icon="$", layout="wide")
st.title("Risk Monitor")

# ── Drawdown tracking ─────────────────────────────────────────────────

st.subheader("Drawdown Analysis")

runs = list_backtest_runs()
if runs:
    run_names = [r["name"] for r in runs]
    sel_run_idx = st.selectbox(
        "Backtest Run (for drawdown data)",
        range(len(run_names)),
        format_func=lambda i: run_names[i],
        key="risk_run",
    )
    result = load_backtest_result(runs[sel_run_idx]["path"])
    metrics = result.get("metrics", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
    c2.metric("DD Duration", f"{metrics.get('max_drawdown_duration_days', 0)} days")
    c3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.4f}")
    c4.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

    eq_data = result.get("equity_curve")
    if eq_data:
        eq_df = equity_curve_to_df(eq_data)
        dd_df = compute_drawdown_series(eq_df)
        if not dd_df.empty:
            st.plotly_chart(drawdown_chart(dd_df), use_container_width=True)

    pnl_data = result.get("daily_pnl")
    if pnl_data:
        pnl_df = daily_pnl_to_df(pnl_data)
        if not pnl_df.empty:
            losing_days = len(pnl_df[pnl_df["pnl"] < 0])
            winning_days = len(pnl_df[pnl_df["pnl"] >= 0])
            worst_day = pnl_df["pnl"].min()
            best_day = pnl_df["pnl"].max()

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Winning Days", winning_days)
            d2.metric("Losing Days", losing_days)
            d3.metric("Best Day", f"${best_day:,.2f}")
            d4.metric("Worst Day", f"${worst_day:,.2f}")

            st.plotly_chart(daily_pnl_chart(pnl_df), use_container_width=True)
else:
    st.info("No backtest data available for drawdown analysis.")

st.divider()

# ── Risk rejection analysis ───────────────────────────────────────────

st.subheader("Risk Rejection Analysis")

dates = list_audit_dates()
if dates:
    sel_date = st.selectbox("Audit Date", dates, key="risk_audit_date")
    records = load_audit_log(sel_date)

    if records:
        total_signals = len(records)
        rejected = [r for r in records if r.get("risk_verdict") and not r["risk_verdict"].get("approved")]
        approved = [r for r in records if r.get("risk_verdict") and r["risk_verdict"].get("approved")]
        no_verdict = [r for r in records if not r.get("risk_verdict")]

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total Signals", total_signals)
        r2.metric("Approved", len(approved))
        r3.metric("Rejected", len(rejected))
        r4.metric("No Verdict (HOLD)", len(no_verdict))

        col_pie, col_detail = st.columns([1, 1])

        with col_pie:
            st.plotly_chart(rejection_reasons_pie(records), use_container_width=True)

        with col_detail:
            st.markdown("**Risk Check Pass Rates**")
            check_stats: dict[str, dict[str, int]] = {}
            for r in records:
                verdict = r.get("risk_verdict") or {}
                for check, passed in verdict.get("checks_passed", {}).items():
                    if check not in check_stats:
                        check_stats[check] = {"passed": 0, "failed": 0}
                    if passed:
                        check_stats[check]["passed"] += 1
                    else:
                        check_stats[check]["failed"] += 1

            if check_stats:
                check_rows = []
                for check, counts in check_stats.items():
                    total = counts["passed"] + counts["failed"]
                    rate = counts["passed"] / total * 100 if total > 0 else 0
                    check_rows.append({
                        "Check": check,
                        "Passed": counts["passed"],
                        "Failed": counts["failed"],
                        "Pass Rate": f"{rate:.1f}%",
                    })
                st.dataframe(check_rows, use_container_width=True, hide_index=True)

        # ── Rejected signals table ─────────────────────────────
        if rejected:
            st.divider()
            st.subheader("Rejected Signals")
            rej_rows = []
            for r in rejected[-50:]:
                signal = r.get("signal") or {}
                verdict = r.get("risk_verdict") or {}
                rej_rows.append({
                    "Time": r.get("timestamp", "")[:19],
                    "Symbol": r.get("symbol", ""),
                    "Action": signal.get("action", ""),
                    "Confidence": f"{signal.get('confidence', 0):.2f}",
                    "Reasons": ", ".join(verdict.get("reasons", [])),
                })
            st.dataframe(rej_rows, use_container_width=True, hide_index=True)
    else:
        st.info(f"No audit records for {sel_date}.")
else:
    st.info("No audit logs available.")

st.divider()

# ── Symbol concentration ──────────────────────────────────────────────

st.subheader("Symbol Concentration")

if runs:
    result = load_backtest_result(runs[0]["path"])
    breakdown = result.get("metrics", {}).get("per_symbol_breakdown", {})
    if breakdown:
        import plotly.graph_objects as go
        from dashboard.utils.charts import COLORS, PLOTLY_TEMPLATE

        total_trades = sum(d["trades"] for d in breakdown.values())
        labels = list(breakdown.keys())
        values = [breakdown[s]["trades"] for s in labels]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="label+percent",
        )])
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Trade Concentration by Symbol",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Alert log ─────────────────────────────────────────────────────────

st.subheader("Recent Alerts & Warnings")

alerts = tail_app_log(n_lines=200, level_filter=None)
critical_alerts = [e for e in alerts if e.get("level") in ("WARNING", "ERROR", "CRITICAL")]

if critical_alerts:
    for entry in reversed(critical_alerts[-30:]):
        level = entry.get("level", "")
        msg = entry.get("message", "")
        ts = entry.get("timestamp", "")[:19]
        if level == "CRITICAL":
            st.error(f"**{ts}** — {msg}")
        elif level == "ERROR":
            st.error(f"**{ts}** — {msg}")
        else:
            st.warning(f"**{ts}** — {msg}")
else:
    st.success("No warnings or errors in recent logs.")
