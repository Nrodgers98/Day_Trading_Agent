"""Reusable Plotly chart builders for the trading dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def _resolve_plotly_template() -> str | None:
    """Return a Plotly template compatible with the installed Plotly version."""
    # Some template bundles can include trace keys unsupported by the current
    # Plotly package (e.g. `heatmapgl`). Sanitize template data keys first.
    valid_data_keys = set(go.layout.template.Data()._valid_props)

    for candidate in ("plotly_dark", "plotly", "simple_white"):
        try:
            raw_template = pio.templates[candidate].to_plotly_json()
            template_data = raw_template.get("data")
            if isinstance(template_data, dict):
                raw_template["data"] = {
                    k: v for k, v in template_data.items() if k in valid_data_keys
                }

            compat_name = f"{candidate}_compat"
            pio.templates[compat_name] = go.layout.Template(raw_template)
            go.Figure().update_layout(template=compat_name)
            return compat_name
        except Exception:
            continue

    for candidate in ("none",):
        try:
            go.Figure().update_layout(template=candidate)
            return candidate
        except Exception:
            continue
    return None


PLOTLY_TEMPLATE = _resolve_plotly_template()

COLORS = {
    "green": "#00d4aa",
    "red": "#ff4b6e",
    "blue": "#4da6ff",
    "yellow": "#ffd700",
    "gray": "#636e82",
    "purple": "#bb86fc",
    "orange": "#ff9f43",
}


def equity_curve_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of equity over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["equity"],
        mode="lines",
        name="Equity",
        line=dict(color=COLORS["green"], width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.08)",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    return fig


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """Area chart of drawdown percentage over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=-df["drawdown_pct"],
        mode="lines",
        name="Drawdown",
        line=dict(color=COLORS["red"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,75,110,0.15)",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=280,
        margin=dict(l=60, r=20, t=50, b=40),
        yaxis=dict(ticksuffix="%"),
    )
    return fig


def daily_pnl_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of daily PnL."""
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in df["pnl"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["pnl"],
        marker_color=colors,
        name="Daily PnL",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Daily PnL",
        xaxis_title="Date",
        yaxis_title="PnL ($)",
        height=350,
        margin=dict(l=60, r=20, t=50, b=40),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    return fig


def pnl_histogram(trades_df: pd.DataFrame) -> go.Figure:
    """Histogram of trade PnL distribution."""
    if trades_df.empty or "pnl" not in trades_df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trades_df["pnl"],
        nbinsx=50,
        marker_color=COLORS["blue"],
        opacity=0.8,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["gray"])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Trade PnL Distribution",
        xaxis_title="PnL ($)",
        yaxis_title="Count",
        height=350,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def hold_time_histogram(trades_df: pd.DataFrame) -> go.Figure:
    """Histogram of trade hold times."""
    if trades_df.empty or "hold_minutes" not in trades_df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trades_df["hold_minutes"],
        nbinsx=40,
        marker_color=COLORS["purple"],
        opacity=0.8,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Hold Time Distribution",
        xaxis_title="Minutes",
        yaxis_title="Count",
        height=350,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def per_symbol_pnl_chart(breakdown: dict[str, dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart of PnL by symbol."""
    symbols = list(breakdown.keys())
    pnls = [breakdown[s]["pnl"] for s in symbols]
    colors = [COLORS["green"] if p >= 0 else COLORS["red"] for p in pnls]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pnls,
        y=symbols,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="PnL by Symbol",
        xaxis_title="PnL ($)",
        height=max(300, len(symbols) * 35),
        margin=dict(l=80, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    return fig


def per_symbol_winrate_chart(breakdown: dict[str, dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart of win rate by symbol."""
    symbols = list(breakdown.keys())
    win_rates = [breakdown[s]["win_rate"] for s in symbols]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=win_rates,
        y=symbols,
        orientation="h",
        marker_color=COLORS["blue"],
    ))
    fig.add_vline(x=50, line_dash="dash", line_color=COLORS["gray"])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Win Rate by Symbol",
        xaxis_title="Win Rate (%)",
        height=max(300, len(symbols) * 35),
        margin=dict(l=80, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 100], ticksuffix="%"),
    )
    return fig


def trades_by_hour_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Bar chart of trade count by hour of day."""
    if trades_df.empty or "entry_time" not in trades_df.columns:
        return go.Figure()

    df = trades_df.copy()
    df["hour"] = df["entry_time"].dt.hour
    hourly = df.groupby("hour").size().reindex(range(0, 24), fill_value=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly.index,
        y=hourly.values,
        marker_color=COLORS["orange"],
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Trades by Hour of Day",
        xaxis_title="Hour (ET)",
        yaxis_title="Trade Count",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def cumulative_pnl_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Cumulative PnL line chart from trade history."""
    if trades_df.empty or "pnl" not in trades_df.columns:
        return go.Figure()

    df = trades_df.sort_values("exit_time").copy()
    df["cumulative_pnl"] = df["pnl"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["exit_time"],
        y=df["cumulative_pnl"],
        mode="lines",
        line=dict(color=COLORS["green"], width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.08)",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Cumulative PnL",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL ($)",
        height=350,
        margin=dict(l=60, r=20, t=50, b=40),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    return fig


def walk_forward_comparison(wf_metrics: list[dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing walk-forward window metrics."""
    if not wf_metrics:
        return go.Figure()

    windows = [f"W{i+1}" for i in range(len(wf_metrics))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Return %",
        x=windows,
        y=[m.get("total_return_pct", 0) for m in wf_metrics],
        marker_color=COLORS["green"],
    ))
    fig.add_trace(go.Bar(
        name="Max DD %",
        x=windows,
        y=[-m.get("max_drawdown", 0) for m in wf_metrics],
        marker_color=COLORS["red"],
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Walk-Forward Window Comparison",
        barmode="group",
        xaxis_title="Window",
        yaxis_title="Percentage (%)",
        height=350,
        margin=dict(l=60, r=20, t=50, b=40),
        yaxis=dict(ticksuffix="%"),
    )
    return fig


def rejection_reasons_pie(records: list[dict[str, Any]]) -> go.Figure:
    """Pie chart of risk rejection reasons from audit records."""
    reasons: dict[str, int] = {}
    for r in records:
        verdict = r.get("risk_verdict") or {}
        if verdict.get("approved"):
            continue
        for reason in verdict.get("reasons", []):
            reasons[reason] = reasons.get(reason, 0) + 1

    if not reasons:
        return go.Figure().update_layout(
            template=PLOTLY_TEMPLATE,
            title="No Rejections Found",
            height=300,
        )

    labels = list(reasons.keys())
    values = list(reasons.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=[
            COLORS["red"], COLORS["orange"], COLORS["yellow"],
            COLORS["purple"], COLORS["blue"], COLORS["gray"],
        ]),
    )])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Risk Rejection Reasons",
        height=350,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig
