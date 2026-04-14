"""Day Trading Agent — Streamlit Dashboard

Launch with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from dashboard.utils.data_loader import list_backtest_runs, list_daily_reports
from dashboard.utils.alpaca_reader import fetch_account, get_alpaca_credentials

st.set_page_config(
    page_title="Day Trading Agent",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("## Day Trading Agent")
st.sidebar.caption("AI-powered intraday trading system")
st.sidebar.divider()

# ── Landing page ──────────────────────────────────────────────────────

st.title("Day Trading Agent")
st.caption("Autonomous intraday trading system for US equities via Alpaca")

col1, col2, col3 = st.columns(3)

runs = list_backtest_runs()
reports = list_daily_reports()

with col1:
    st.metric("Backtest Runs", len(runs))

with col2:
    st.metric("Daily Reports", len(reports))

with col3:
    creds = get_alpaca_credentials()
    if creds:
        account = fetch_account()
        if account:
            status = account.get("status", "UNKNOWN")
            st.metric("Alpaca Account", status)
        else:
            st.metric("Alpaca Account", "Error")
    else:
        st.metric("Alpaca Account", "Not Configured")

st.divider()

st.markdown("""
### Quick Navigation

Use the sidebar to navigate between pages:

- **Dashboard** — Key metrics, account status, latest performance
- **Backtest** — Analyze backtest results with interactive charts
- **Trades** — Browse trade history and audit logs
- **Risk** — Monitor risk metrics and rejection patterns
- **Config** — View current configuration settings
- **Trading desk** — Session PnL, stitched equity, cumulative progress, LLM desk review
""")

if runs:
    st.divider()
    st.subheader("Latest Backtest")
    latest = runs[0]
    metrics = latest["metrics"]
    if metrics:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total PnL", f"${metrics.get('total_pnl', 0):,.2f}")
        c2.metric("Return", f"{metrics.get('total_return_pct', 0):.2f}%")
        c3.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
        c4.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
        c5.metric("Max DD", f"{metrics.get('max_drawdown', 0):.2f}%")
        c6.metric("Trades", metrics.get("total_trades", 0))

st.sidebar.divider()
st.sidebar.caption("Educational/research use only. Not financial advice.")
