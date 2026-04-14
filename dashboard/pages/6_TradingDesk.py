"""Trading desk — account progress from daily reports + broker snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from dashboard.utils.alpaca_reader import fetch_account, fetch_positions, get_alpaca_credentials
from dashboard.utils.charts import (
    cumulative_daily_pnl_chart,
    daily_pnl_chart,
    drawdown_chart,
    equity_curve_chart,
    pnl_histogram,
    rolling_daily_pnl_chart,
)
from dashboard.utils.data_loader import (
    compute_drawdown_series,
    daily_reports_summary_dataframe,
    list_daily_reports,
    stitched_equity_curve_from_reports,
    trades_concat_from_reports,
    trades_to_df,
)
from src.agent.config import load_config
from src.agent.improvement.episodes import EpisodeDatasetBuilder
from src.agent.improvement.llm_advisor import ImprovementLLMAdvisor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="Trading Desk", page_icon="$", layout="wide")
st.title("Trading desk")
st.caption("Paper/live progress from session reports and live broker snapshot")

reports_all = list_daily_reports()
lookback = st.sidebar.slider("Report lookback (days)", min_value=7, max_value=365, value=60)
reports = reports_all[:lookback] if reports_all else []

daily_df = daily_reports_summary_dataframe(reports)
eq_df = stitched_equity_curve_from_reports(reports)
trades_raw = trades_concat_from_reports(reports)
trades_df = trades_to_df(trades_raw)

# ── Headline row ─────────────────────────────────────────────────────

creds = get_alpaca_credentials()
h1, h2, h3, h4, h5 = st.columns(5)
if creds and (acct := fetch_account()):
    h1.metric("Equity", f"${float(acct.get('equity', 0)):,.2f}")
    h2.metric("Cash", f"${float(acct.get('cash', 0)):,.2f}")
    h3.metric("Buying power", f"${float(acct.get('buying_power', 0)):,.2f}")
    h4.metric("Day trades", acct.get("daytrade_count", "—"))
    h5.metric("Status", str(acct.get("status", "—")))
else:
    h1.info("Alpaca not configured — broker tiles unavailable.")

if daily_df.shape[0]:
    last = daily_df.iloc[-1]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Sessions in view", int(daily_df.shape[0]))
    s2.metric("Last session PnL", f"${float(last['total_pnl']):,.2f}")
    s3.metric("Last session trades", int(last["trade_count"]))
    s4.metric("Cumulative PnL (view)", f"${float(last['cumulative_pnl']):,.2f}")
else:
    st.warning("No daily JSON reports in `logs/reports/` for this lookback.")

st.divider()

# ── Charts ───────────────────────────────────────────────────────────

if not eq_df.empty:
    st.subheader("Account equity (stitched from reports)")
    st.plotly_chart(equity_curve_chart(eq_df), use_container_width=True)
    dd_df = compute_drawdown_series(eq_df)
    if not dd_df.empty:
        st.plotly_chart(drawdown_chart(dd_df), use_container_width=True)
elif reports:
    st.info("Reports have no `equity_curve` yet — run the agent with daily JSON reports, or backfill.")

if not daily_df.empty:
    st.subheader("Daily session PnL")
    pnl_for_bar = daily_df.rename(columns={"total_pnl": "pnl"})[["date", "pnl"]]
    st.plotly_chart(daily_pnl_chart(pnl_for_bar), use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(cumulative_daily_pnl_chart(daily_df), use_container_width=True)
    with c_right:
        st.plotly_chart(rolling_daily_pnl_chart(daily_df), use_container_width=True)

if not trades_df.empty and "pnl" in trades_df.columns:
    st.subheader("Trade PnL distribution (aggregated reports)")
    st.plotly_chart(pnl_histogram(trades_df), use_container_width=True)

# ── Open positions ────────────────────────────────────────────────────

if creds:
    st.subheader("Open positions")
    pos = fetch_positions()
    if pos:
        st.dataframe(pos, use_container_width=True, hide_index=True)
    else:
        st.caption("No open positions.")

st.divider()

# ── LLM desk review (read-only) ─────────────────────────────────────

st.subheader("Desk review (LLM)")
st.caption(
    "Uses `OPENAI_API_KEY` or `IMPROVEMENT_LLM_API_KEY` and improvement LLM settings from "
    "`config/default.yaml`. Does not apply config changes — use the improvement loop for that."
)

if st.button("Generate desk review", type="primary"):
    cfg_path = PROJECT_ROOT / "config" / "default.yaml"
    if not cfg_path.is_file():
        st.error("config/default.yaml not found.")
    else:
        try:
            cfg = load_config(cfg_path)
        except Exception as e:
            st.error(f"Could not load config: {e}")
            st.stop()
        builder = EpisodeDatasetBuilder(
            cfg.monitoring.log_dir,
            observe_modes=cfg.improvement.observe_modes,
            session_timezone=cfg.session.timezone,
        )
        episodes = builder.build(cfg.improvement.analysis_lookback_days)
        if not episodes:
            st.warning("No episodes from reports — nothing to send to the LLM.")
        else:
            with st.spinner("Calling LLM…"):
                advisor = ImprovementLLMAdvisor(cfg)
                proposals, meta = advisor.propose_from_episodes(episodes, [])
            if meta.get("errors"):
                with st.expander("LLM errors", expanded=True):
                    # st.write(list[str]) renders as a odd single-column table; JSON is readable.
                    st.code(json.dumps(meta["errors"], indent=2), language="json")
            if meta.get("desk_review_markdown"):
                st.markdown(meta["desk_review_markdown"])
            else:
                st.info("No markdown returned — check API key and model response.")
            if proposals:
                with st.expander("Structured suggestions (not applied here)", expanded=False):
                    for p in proposals:
                        st.json(p.model_dump(mode="json"))
