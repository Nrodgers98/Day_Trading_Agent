"""Trade Journal — browse and filter trades from backtests and audit logs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.utils.charts import (
    cumulative_pnl_chart,
    hold_time_histogram,
    pnl_histogram,
    trades_by_hour_chart,
)
from dashboard.utils.data_loader import (
    audit_records_to_df,
    list_audit_dates,
    list_backtest_runs,
    load_audit_log,
    load_backtest_result,
    trades_to_df,
)

st.set_page_config(page_title="Trade Journal", page_icon="$", layout="wide")
st.title("Trade Journal")

tab_bt, tab_audit = st.tabs(["Backtest Trades", "Live/Paper Audit Log"])

# ── Tab 1: Backtest trades ────────────────────────────────────────────

with tab_bt:
    runs = list_backtest_runs()

    if not runs:
        st.info("No backtest results available.")
    else:
        run_names = [r["name"] for r in runs]
        sel = st.selectbox("Backtest Run", run_names, key="bt_run")
        sel_run = runs[run_names.index(sel)]
        result = load_backtest_result(sel_run["path"])
        trades_data = result.get("trades", [])
        trades_df = trades_to_df(trades_data)

        if trades_df.empty:
            st.info("No trades in this backtest run.")
        else:
            # ── Filters ──────────────────────────────────────────
            st.subheader("Filters")
            fc1, fc2, fc3 = st.columns(3)

            symbols = sorted(trades_df["symbol"].unique()) if "symbol" in trades_df.columns else []
            with fc1:
                sel_symbols = st.multiselect("Symbols", symbols, default=symbols, key="bt_sym")
            with fc2:
                sides = sorted(trades_df["side"].unique()) if "side" in trades_df.columns else []
                sel_sides = st.multiselect("Side", sides, default=sides, key="bt_side")
            with fc3:
                pnl_range = st.slider(
                    "PnL Range ($)",
                    float(trades_df["pnl"].min()) if "pnl" in trades_df.columns else -1000.0,
                    float(trades_df["pnl"].max()) if "pnl" in trades_df.columns else 1000.0,
                    (
                        float(trades_df["pnl"].min()) if "pnl" in trades_df.columns else -1000.0,
                        float(trades_df["pnl"].max()) if "pnl" in trades_df.columns else 1000.0,
                    ),
                    key="bt_pnl",
                )

            filtered = trades_df.copy()
            if sel_symbols and "symbol" in filtered.columns:
                filtered = filtered[filtered["symbol"].isin(sel_symbols)]
            if sel_sides and "side" in filtered.columns:
                filtered = filtered[filtered["side"].isin(sel_sides)]
            if "pnl" in filtered.columns:
                filtered = filtered[
                    (filtered["pnl"] >= pnl_range[0]) & (filtered["pnl"] <= pnl_range[1])
                ]

            # ── Summary ──────────────────────────────────────────
            st.divider()
            total = len(filtered)
            wins = len(filtered[filtered["pnl"] > 0]) if "pnl" in filtered.columns else 0
            total_pnl = filtered["pnl"].sum() if "pnl" in filtered.columns else 0

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Filtered Trades", total)
            mc2.metric("Winners", wins)
            mc3.metric("Win Rate", f"{wins / total * 100:.1f}%" if total > 0 else "—")
            mc4.metric("Total PnL", f"${total_pnl:,.2f}")

            # ── Charts ───────────────────────────────────────────
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(pnl_histogram(filtered), use_container_width=True)
            with col_b:
                st.plotly_chart(hold_time_histogram(filtered), use_container_width=True)

            col_c, col_d = st.columns(2)
            with col_c:
                st.plotly_chart(cumulative_pnl_chart(filtered), use_container_width=True)
            with col_d:
                st.plotly_chart(trades_by_hour_chart(filtered), use_container_width=True)

            # ── Table ────────────────────────────────────────────
            st.divider()
            st.subheader("Trade Log")
            display = filtered.copy()
            for col in ("entry_time", "exit_time"):
                if col in display.columns:
                    display[col] = display[col].dt.strftime("%Y-%m-%d %H:%M")
            for col in ("pnl", "entry_price", "exit_price"):
                if col in display.columns:
                    display[col] = display[col].map(lambda x: f"${x:,.2f}")
            if "hold_minutes" in display.columns:
                display["hold_minutes"] = display["hold_minutes"].map(lambda x: f"{x:.0f}m")
            st.dataframe(display, use_container_width=True, hide_index=True)

# ── Tab 2: Audit log ─────────────────────────────────────────────────

with tab_audit:
    dates = list_audit_dates()

    if not dates:
        st.info("No audit logs found. Run paper trading to generate audit data.")
    else:
        ac1, ac2 = st.columns(2)
        with ac1:
            sel_date = st.selectbox("Date", dates, key="audit_date")
        with ac2:
            sym_filter = st.text_input(
                "Symbol filter (blank for all)", "", key="audit_sym"
            ).strip().upper() or None

        records = load_audit_log(sel_date, symbol=sym_filter)

        if not records:
            st.info(f"No audit records for {sel_date}" + (f" / {sym_filter}" if sym_filter else ""))
        else:
            audit_df = audit_records_to_df(records)

            st.metric("Total Decisions", len(records))

            # ── Action breakdown ─────────────────────────────────
            if "action" in audit_df.columns:
                action_counts = audit_df["action"].value_counts()
                ac_cols = st.columns(len(action_counts))
                for i, (action, count) in enumerate(action_counts.items()):
                    ac_cols[i].metric(action.upper() if action else "NONE", count)

            st.divider()

            # ── Decisions table ───────────────────────────────────
            st.subheader("Decision Log")
            display_cols = [
                "timestamp", "symbol", "action", "side", "confidence",
                "risk_approved", "risk_reasons", "order_status",
            ]
            available_cols = [c for c in display_cols if c in audit_df.columns]
            if available_cols:
                disp = audit_df[available_cols].copy()
                if "timestamp" in disp.columns:
                    disp["timestamp"] = disp["timestamp"].dt.strftime("%H:%M:%S")
                st.dataframe(disp, use_container_width=True, hide_index=True)

            # ── Drill-down ────────────────────────────────────────
            st.divider()
            st.subheader("Decision Detail")
            if len(records) > 0:
                idx = st.number_input(
                    "Record index", 0, len(records) - 1, 0, key="audit_idx"
                )
                st.json(records[idx])
