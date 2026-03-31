"""Backtest Analyzer — interactive exploration of backtest results."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.utils.charts import (
    cumulative_pnl_chart,
    daily_pnl_chart,
    drawdown_chart,
    equity_curve_chart,
    hold_time_histogram,
    per_symbol_pnl_chart,
    per_symbol_winrate_chart,
    pnl_histogram,
    trades_by_hour_chart,
    walk_forward_comparison,
)
from dashboard.utils.data_loader import (
    compute_drawdown_series,
    daily_pnl_to_df,
    equity_curve_to_df,
    list_backtest_runs,
    load_backtest_result,
    trades_to_df,
)

st.set_page_config(page_title="Backtest Analyzer", page_icon="$", layout="wide")
st.title("Backtest Analyzer")

# ── Run selector ──────────────────────────────────────────────────────

runs = list_backtest_runs()

if not runs:
    st.info("No backtest results found. Run a backtest first:")
    st.code("python scripts/run_backtest.py --config config/backtest.yaml")
    st.stop()

run_names = [r["name"] for r in runs]
selected_idx = st.sidebar.selectbox(
    "Select Backtest Run",
    range(len(run_names)),
    format_func=lambda i: run_names[i],
)
selected_run = runs[selected_idx]
result = load_backtest_result(selected_run["path"])
metrics = result.get("metrics", {})

# ── Summary metrics ───────────────────────────────────────────────────

st.subheader("Performance Summary")

if metrics:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total PnL", f"${metrics.get('total_pnl', 0):,.2f}")
    c2.metric("Return", f"{metrics.get('total_return_pct', 0):.2f}%")
    c3.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
    c4.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.4f}")
    c5.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
    c6.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")

    c7, c8, c9, c10, c11, c12 = st.columns(6)
    c7.metric("Total Trades", metrics.get("total_trades", 0))
    c8.metric("Trades/Day", f"{metrics.get('avg_trades_per_day', 0):.1f}")
    c9.metric("Avg Hold", f"{metrics.get('avg_hold_minutes', 0):.0f} min")
    c10.metric("Exposure", f"{metrics.get('exposure_pct', 0):.1f}%")
    c11.metric("DD Duration", f"{metrics.get('max_drawdown_duration_days', 0)}d")
    c12.metric("Run", selected_run["name"][:15])

st.divider()

# ── Equity curve and drawdown ─────────────────────────────────────────

eq_data = result.get("equity_curve")
if eq_data:
    eq_df = equity_curve_to_df(eq_data)
    if not eq_df.empty:
        st.plotly_chart(equity_curve_chart(eq_df), use_container_width=True)

        dd_df = compute_drawdown_series(eq_df)
        if not dd_df.empty:
            st.plotly_chart(drawdown_chart(dd_df), use_container_width=True)

# ── Daily PnL ─────────────────────────────────────────────────────────

pnl_data = result.get("daily_pnl")
if pnl_data:
    pnl_df = daily_pnl_to_df(pnl_data)
    if not pnl_df.empty:
        st.plotly_chart(daily_pnl_chart(pnl_df), use_container_width=True)

st.divider()

# ── Trade analysis ────────────────────────────────────────────────────

st.subheader("Trade Analysis")

trades_data = result.get("trades")
trades_df = trades_to_df(trades_data) if trades_data else None

if trades_df is not None and not trades_df.empty:
    tab1, tab2, tab3 = st.tabs(["Distribution", "Cumulative", "Timing"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(pnl_histogram(trades_df), use_container_width=True)
        with col_b:
            st.plotly_chart(hold_time_histogram(trades_df), use_container_width=True)

    with tab2:
        st.plotly_chart(cumulative_pnl_chart(trades_df), use_container_width=True)

    with tab3:
        st.plotly_chart(trades_by_hour_chart(trades_df), use_container_width=True)
else:
    st.info("No trades in this backtest run.")

st.divider()

# ── Per-symbol breakdown ──────────────────────────────────────────────

st.subheader("Per-Symbol Breakdown")

breakdown = metrics.get("per_symbol_breakdown", {}) if metrics else {}
if breakdown:
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(per_symbol_pnl_chart(breakdown), use_container_width=True)
    with col_b:
        st.plotly_chart(per_symbol_winrate_chart(breakdown), use_container_width=True)

    st.dataframe(
        [
            {
                "Symbol": sym,
                "PnL": f"${d['pnl']:,.2f}",
                "Trades": d["trades"],
                "Win Rate": f"{d['win_rate']:.1f}%",
            }
            for sym, d in breakdown.items()
        ],
        use_container_width=True,
        hide_index=True,
    )

# ── Walk-forward results ─────────────────────────────────────────────

wf_data = result.get("walk_forward")
if wf_data:
    st.divider()
    st.subheader("Walk-Forward Validation")
    st.plotly_chart(walk_forward_comparison(wf_data), use_container_width=True)

    wf_summary = []
    for i, wm in enumerate(wf_data):
        wf_summary.append({
            "Window": f"W{i+1}",
            "Return %": f"{wm.get('total_return_pct', 0):.2f}%",
            "Sharpe": f"{wm.get('sharpe_ratio', 0):.4f}",
            "Win Rate": f"{wm.get('win_rate', 0):.1f}%",
            "Max DD": f"{wm.get('max_drawdown', 0):.2f}%",
            "Trades": wm.get("total_trades", 0),
        })
    st.dataframe(wf_summary, use_container_width=True, hide_index=True)

# ── Raw trades table ──────────────────────────────────────────────────

if trades_df is not None and not trades_df.empty:
    st.divider()
    with st.expander("Raw Trade Log", expanded=False):
        display_df = trades_df.copy()
        for col in ("entry_time", "exit_time"):
            if col in display_df.columns:
                display_df[col] = display_df[col].dt.strftime("%Y-%m-%d %H:%M")
        if "pnl" in display_df.columns:
            display_df["pnl"] = display_df["pnl"].map(lambda x: f"${x:,.2f}")
        if "entry_price" in display_df.columns:
            display_df["entry_price"] = display_df["entry_price"].map(lambda x: f"${x:,.2f}")
        if "exit_price" in display_df.columns:
            display_df["exit_price"] = display_df["exit_price"].map(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Config snapshot ───────────────────────────────────────────────────

config_snap = result.get("config_snapshot")
if config_snap:
    with st.expander("Configuration Used", expanded=False):
        st.json(config_snap)
