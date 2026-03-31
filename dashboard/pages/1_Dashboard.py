"""Overview dashboard — key metrics, account status, latest performance."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from dashboard.utils.alpaca_reader import fetch_account, fetch_positions, get_alpaca_credentials
from dashboard.utils.charts import daily_pnl_chart, equity_curve_chart
from dashboard.utils.data_loader import (
    daily_pnl_to_df,
    equity_curve_to_df,
    list_backtest_runs,
    load_backtest_result,
    tail_app_log,
)

st.set_page_config(page_title="Dashboard", page_icon="$", layout="wide")
st.title("Dashboard")


# ── Alpaca account section ────────────────────────────────────────────

st.subheader("Alpaca Account")

creds = get_alpaca_credentials()
if creds:
    account = fetch_account()
    if account:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Equity", f"${float(account.get('equity', 0)):,.2f}")
        c2.metric("Cash", f"${float(account.get('cash', 0)):,.2f}")
        c3.metric("Buying Power", f"${float(account.get('buying_power', 0)):,.2f}")
        c4.metric("Day Trades", account.get("daytrade_count", 0))
        c5.metric("Status", account.get("status", "UNKNOWN"))

        if account.get("pattern_day_trader"):
            st.warning("Pattern Day Trader flag is active on this account.")

        positions = fetch_positions()
        if positions:
            st.markdown(f"**Open Positions:** {len(positions)}")
            pos_data = []
            for p in positions:
                pos_data.append({
                    "Symbol": p.get("symbol", ""),
                    "Side": p.get("side", ""),
                    "Qty": float(p.get("qty", 0)),
                    "Entry Price": f"${float(p.get('avg_entry_price', 0)):,.2f}",
                    "Current Price": f"${float(p.get('current_price', 0)):,.2f}",
                    "P&L": f"${float(p.get('unrealized_pl', 0)):,.2f}",
                    "P&L %": f"{float(p.get('unrealized_plpc', 0)) * 100:.2f}%",
                })
            st.dataframe(pos_data, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")
    else:
        st.error("Could not connect to Alpaca. Check your API keys.")
else:
    st.info("Alpaca API keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")

st.divider()

# ── Latest backtest section ───────────────────────────────────────────

st.subheader("Latest Backtest Performance")

runs = list_backtest_runs()
if runs:
    latest = runs[0]
    result = load_backtest_result(latest["path"])
    metrics = result.get("metrics", {})

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total PnL", f"${metrics.get('total_pnl', 0):,.2f}")
        c2.metric("Return", f"{metrics.get('total_return_pct', 0):.2f}%")
        c3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.4f}")
        c4.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
        c6.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
        c7.metric("Total Trades", metrics.get("total_trades", 0))
        c8.metric("Avg Hold", f"{metrics.get('avg_hold_minutes', 0):.0f} min")

    eq_data = result.get("equity_curve")
    if eq_data:
        eq_df = equity_curve_to_df(eq_data)
        if not eq_df.empty:
            st.plotly_chart(equity_curve_chart(eq_df), use_container_width=True)

    pnl_data = result.get("daily_pnl")
    if pnl_data:
        pnl_df = daily_pnl_to_df(pnl_data)
        if not pnl_df.empty:
            st.plotly_chart(daily_pnl_chart(pnl_df), use_container_width=True)
else:
    st.info("No backtest results found. Run a backtest first: `python scripts/run_backtest.py`")

st.divider()

# ── Recent log entries ────────────────────────────────────────────────

st.subheader("Recent Activity")

log_entries = tail_app_log(n_lines=50)
if log_entries:
    critical_entries = [e for e in log_entries if e.get("level") in ("ERROR", "CRITICAL")]
    if critical_entries:
        st.error(f"{len(critical_entries)} error(s) in recent logs")
        for entry in critical_entries[-5:]:
            st.code(f"[{entry.get('timestamp', '')}] {entry.get('message', '')}")

    with st.expander("All Recent Log Entries", expanded=False):
        for entry in reversed(log_entries[-20:]):
            level = entry.get("level", "INFO")
            msg = entry.get("message", "")
            ts = entry.get("timestamp", "")[:19]
            if level in ("ERROR", "CRITICAL"):
                st.markdown(f"**:red[{ts} {level}]** {msg}")
            elif level == "WARNING":
                st.markdown(f"**:orange[{ts} {level}]** {msg}")
            else:
                st.markdown(f"`{ts}` {msg}")
else:
    st.info("No log entries found.")
