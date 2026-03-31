"""Configuration Viewer — read-only display of YAML config files."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.utils.data_loader import list_config_files, load_yaml_config


def _render_config_section(name: str, data: dict) -> None:
    """Render a config section as a clean key-value display."""
    rows = []
    for k, v in data.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append({"Parameter": f"{k}.{k2}", "Value": str(v2)})
        else:
            rows.append({"Parameter": k, "Value": str(v)})
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _find_diffs(
    a: dict, b: dict, prefix: str = ""
) -> list[tuple[str, object, object]]:
    """Recursively find differences between two config dicts."""
    diffs: list[tuple[str, object, object]] = []
    all_keys = set(list(a.keys()) + list(b.keys()))

    for key in sorted(all_keys):
        path = f"{prefix}.{key}" if prefix else key
        va = a.get(key)
        vb = b.get(key)

        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(_find_diffs(va, vb, path))
        elif va != vb:
            diffs.append((path, va, vb))

    return diffs


# ── Page setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Configuration", page_icon="$", layout="wide")
st.title("Configuration Viewer")
st.caption("Read-only view of the current configuration files.")

config_files = list_config_files()

if not config_files:
    st.info("No configuration files found in config/ directory.")
    st.stop()

# ── Config file selector ──────────────────────────────────────────────

selected_file = st.sidebar.selectbox("Config File", config_files)
config = load_yaml_config(selected_file)

if not config:
    st.warning(f"Config file `{selected_file}` is empty or could not be loaded.")
    st.stop()

# ── Structured view ───────────────────────────────────────────────────

st.subheader(f"`config/{selected_file}`")

sections = list(config.keys())

for section in sections:
    section_data = config[section]
    with st.expander(f"{section}", expanded=True):
        if isinstance(section_data, dict):
            _render_config_section(section, section_data)
        else:
            st.write(section_data)

st.divider()

# ── Risk parameters highlight ─────────────────────────────────────────

risk_cfg = config.get("risk", {})
if risk_cfg:
    st.subheader("Risk Parameters Summary")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric(
        "Max Risk/Trade",
        f"{risk_cfg.get('max_risk_per_trade_pct', 0) * 100:.1f}%"
    )
    rc2.metric(
        "Max Daily DD",
        f"{risk_cfg.get('max_daily_drawdown_pct', 0) * 100:.1f}%"
    )
    rc3.metric(
        "Max Exposure",
        f"{risk_cfg.get('max_gross_exposure_pct', 0) * 100:.0f}%"
    )
    rc4.metric(
        "Max Positions",
        risk_cfg.get("max_concurrent_positions", "—")
    )

    rc5, rc6, rc7, rc8 = st.columns(4)
    rc5.metric("SL (ATR mult)", risk_cfg.get("stop_loss_atr_mult", "—"))
    rc6.metric("TP (ATR mult)", risk_cfg.get("take_profit_atr_mult", "—"))
    rc7.metric("Trailing Stop", risk_cfg.get("trailing_stop_atr_mult", "—"))
    rc8.metric("Daily Trade Cap", risk_cfg.get("daily_trade_cap", "—"))

# ── Strategy parameters highlight ─────────────────────────────────────

strategy_cfg = config.get("strategy", {})
if strategy_cfg:
    st.subheader("Strategy Parameters Summary")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Strategy", strategy_cfg.get("name", "—"))
    sc2.metric("Timeframe", strategy_cfg.get("timeframe", "—"))
    sc3.metric("Lookback Bars", strategy_cfg.get("lookback_bars", "—"))
    sc4.metric("ML Threshold", strategy_cfg.get("ml_confidence_threshold", "—"))

    sc5, sc6, sc7, sc8 = st.columns(4)
    sc5.metric("Vol Surge Ratio", strategy_cfg.get("volume_surge_ratio", "—"))
    sc6.metric("Cooldown Bars", strategy_cfg.get("cooldown_bars", "—"))
    sc7.metric("Max Trades/Day", strategy_cfg.get("max_trades_per_day", "—"))
    sc8.metric("Max Hold (min)", strategy_cfg.get("max_hold_minutes", "—"))

# ── Session parameters ────────────────────────────────────────────────

session_cfg = config.get("session", {})
if session_cfg:
    st.subheader("Session Parameters")
    ss1, ss2, ss3, ss4 = st.columns(4)
    ss1.metric("Timezone", session_cfg.get("timezone", "—"))
    ss2.metric("Market Open", session_cfg.get("market_open", "—"))
    ss3.metric("Market Close", session_cfg.get("market_close", "—"))
    ss4.metric("EOD Flatten", session_cfg.get("eod_flatten_time", "—"))

    ss5, ss6, ss7, ss8 = st.columns(4)
    ss5.metric("Opening Guard", f"{session_cfg.get('opening_guard_minutes', 0)} min")
    ss6.metric("Closing Guard", f"{session_cfg.get('closing_guard_minutes', 0)} min")
    ss7.metric("EOD Flatten", "ON" if session_cfg.get("eod_flatten") else "OFF")
    ss8.metric("Pre-Market", "ON" if session_cfg.get("enable_premarket") else "OFF")

st.divider()

# ── Raw YAML ──────────────────────────────────────────────────────────

with st.expander("Raw YAML", expanded=False):
    st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")

# ── Compare configs ───────────────────────────────────────────────────

if len(config_files) > 1:
    st.divider()
    st.subheader("Compare Configurations")

    cc1, cc2 = st.columns(2)
    with cc1:
        file_a = st.selectbox("Left", config_files, index=0, key="cmp_a")
    with cc2:
        file_b = st.selectbox(
            "Right",
            config_files,
            index=min(1, len(config_files) - 1),
            key="cmp_b",
        )

    if file_a != file_b:
        cfg_a = load_yaml_config(file_a)
        cfg_b = load_yaml_config(file_b)

        diffs = _find_diffs(cfg_a, cfg_b)
        if diffs:
            st.markdown(f"**{len(diffs)} difference(s) found:**")
            diff_rows = [
                {"Path": path, f"{file_a}": str(va), f"{file_b}": str(vb)}
                for path, va, vb in diffs
            ]
            st.dataframe(diff_rows, use_container_width=True, hide_index=True)
        else:
            st.success("Configurations are identical.")
    else:
        st.info("Select two different files to compare.")
