"""Configuration viewer and editor for YAML files under ``config/``."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.utils.data_loader import (
    list_config_files,
    load_yaml_config,
    save_yaml_config,
)
from src.agent.config import AppConfig

# Paths (dot-separated) rendered as select boxes — matches Pydantic ``Literal`` fields.
SELECT_OPTIONS: dict[str, tuple[str, ...]] = {
    "trading.mode": ("paper", "live", "backtest"),
    "universe.base": ("custom", "scan", "hybrid"),
    "sentiment.provider": ("none", "finbert"),
    "sentiment.news_source": ("alpaca_news", "newsapi"),
    "monitoring.report_format": ("json", "csv", "markdown"),
    "improvement.autonomy_mode": ("manual", "autonomous_nonprod", "autonomous"),
    "improvement.optimize_for": ("risk_adjusted_return", "raw_pnl", "stability"),
}


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


def _render_editable_node(node: dict[str, Any], prefix: tuple[str, ...]) -> None:
    """Render widgets for one mapping; mutates ``node`` in place (draft config)."""
    scope = st.session_state.cfg_draft_name
    for key in sorted(node.keys()):
        val = node[key]
        path = prefix + (key,)
        path_str = ".".join(path)

        if isinstance(val, dict):
            with st.expander(path_str, expanded=len(path) <= 1):
                _render_editable_node(val, path)
            continue

        wid = f"cfg__{path_str.replace('.', '_')}__{scope}"

        if isinstance(val, bool):
            node[key] = st.toggle(path_str, value=val, key=wid)
        elif path_str in SELECT_OPTIONS:
            opts = SELECT_OPTIONS[path_str]
            index = opts.index(val) if val in opts else 0
            node[key] = st.selectbox(path_str, opts, index=index, key=wid)
        elif path_str == "improvement.observe_modes":
            cur = val if isinstance(val, list) else ["paper"]
            sel = st.multiselect(path_str, ["paper", "live"], default=cur, key=wid)
            node[key] = sel if sel else ["paper"]
        elif path_str == "universe.custom_symbols":
            lines_txt = "\n".join(val) if isinstance(val, list) else ""
            raw = st.text_area(
                f"{path_str} (one symbol per line)",
                value=lines_txt,
                height=min(200, 24 + 18 * max(1, lines_txt.count("\n") + 1)),
                key=wid,
            )
            node[key] = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        elif isinstance(val, list):
            if path_str in ("strategy.rsi_long_range", "strategy.rsi_short_range") and len(val) == 2:
                c1, c2 = st.columns(2)
                with c1:
                    a = st.number_input(f"{path_str}[0]", value=float(val[0]), key=wid + "_a")
                with c2:
                    b = st.number_input(f"{path_str}[1]", value=float(val[1]), key=wid + "_b")
                node[key] = [a, b]
            else:
                raw = st.text_area(path_str + " (JSON list)", value=json.dumps(val), key=wid)
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        node[key] = parsed
                    else:
                        st.caption("Expected a JSON array — values unchanged.")
                except json.JSONDecodeError:
                    st.caption("Invalid JSON — fix the list or reload from disk.")
        elif isinstance(val, int):
            node[key] = int(st.number_input(path_str, value=int(val), step=1, key=wid))
        elif isinstance(val, float):
            node[key] = float(
                st.number_input(path_str, value=float(val), format="%.8g", key=wid)
            )
        else:
            node[key] = st.text_input(path_str, value=str(val), key=wid)


# ── Page setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Configuration", page_icon="$", layout="wide")
st.title("Configuration")
st.caption("Inspect and edit YAML under `config/`. Saving validates against the agent schema then overwrites the file.")

config_files = list_config_files()

if not config_files:
    st.info("No configuration files found in config/ directory.")
    st.stop()

selected_file = st.sidebar.selectbox("Config file", config_files)

if st.sidebar.button("Reload from disk", help="Discard unsaved edits for the selected file."):
    st.session_state.cfg_draft = copy.deepcopy(load_yaml_config(selected_file))
    st.session_state.cfg_draft_name = selected_file
    st.rerun()

if (
    st.session_state.get("cfg_draft_name") != selected_file
    or "cfg_draft" not in st.session_state
):
    st.session_state.cfg_draft = copy.deepcopy(load_yaml_config(selected_file))
    st.session_state.cfg_draft_name = selected_file

config: dict[str, Any] = st.session_state.cfg_draft

if not config:
    st.warning(f"Config file `{selected_file}` is empty or could not be loaded.")
    st.stop()

tab_edit, tab_overview, tab_raw, tab_compare = st.tabs(
    ["Edit", "Overview", "Raw YAML", "Compare"]
)

# ── Edit ──────────────────────────────────────────────────────────────

with tab_edit:
    st.subheader(f"Edit `config/{selected_file}`")
    st.warning(
        "Saving rewrites the entire YAML file. Inline comments in the file are **not** preserved."
    )
    _render_editable_node(config, ())

    b1, b2 = st.columns(2)
    with b1:
        save_clicked = st.button("Save to disk", type="primary")
    with b2:
        if st.button("Reset draft to last saved"):
            st.session_state.cfg_draft = copy.deepcopy(load_yaml_config(selected_file))
            st.session_state.cfg_draft_name = selected_file
            st.rerun()

    if save_clicked:
        try:
            AppConfig(**st.session_state.cfg_draft)
        except ValidationError as err:
            st.error("Config does not match the agent schema (fix errors below).")
            st.code(str(err), language="text")
        else:
            save_yaml_config(selected_file, st.session_state.cfg_draft)
            st.success(f"Saved `config/{selected_file}`.")

# ── Overview (reflects current draft) ─────────────────────────────────

with tab_overview:
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

    risk_cfg = config.get("risk", {})
    if risk_cfg:
        st.subheader("Risk parameters summary")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(
            "Max risk/trade",
            f"{risk_cfg.get('max_risk_per_trade_pct', 0) * 100:.1f}%",
        )
        rc2.metric(
            "Max daily DD",
            f"{risk_cfg.get('max_daily_drawdown_pct', 0) * 100:.1f}%",
        )
        rc3.metric(
            "Max exposure",
            f"{risk_cfg.get('max_gross_exposure_pct', 0) * 100:.0f}%",
        )
        rc4.metric("Max positions", risk_cfg.get("max_concurrent_positions", "—"))

        rc5, rc6, rc7, rc8 = st.columns(4)
        rc5.metric("SL (ATR mult)", risk_cfg.get("stop_loss_atr_mult", "—"))
        rc6.metric("TP (ATR mult)", risk_cfg.get("take_profit_atr_mult", "—"))
        rc7.metric("Trailing stop", risk_cfg.get("trailing_stop_atr_mult", "—"))
        rc8.metric("Daily trade cap", risk_cfg.get("daily_trade_cap", "—"))

    strategy_cfg = config.get("strategy", {})
    if strategy_cfg:
        st.subheader("Strategy parameters summary")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Strategy", strategy_cfg.get("name", "—"))
        sc2.metric("Timeframe", strategy_cfg.get("timeframe", "—"))
        sc3.metric("Lookback bars", strategy_cfg.get("lookback_bars", "—"))
        sc4.metric("ML threshold", strategy_cfg.get("ml_confidence_threshold", "—"))

        sc5, sc6, sc7, sc8 = st.columns(4)
        sc5.metric("Vol surge ratio", strategy_cfg.get("volume_surge_ratio", "—"))
        sc6.metric("Cooldown bars", strategy_cfg.get("cooldown_bars", "—"))
        sc7.metric("Max trades/day", strategy_cfg.get("max_trades_per_day", "—"))
        sc8.metric("Max hold (min)", strategy_cfg.get("max_hold_minutes", "—"))

    session_cfg = config.get("session", {})
    if session_cfg:
        st.subheader("Session parameters")
        ss1, ss2, ss3, ss4 = st.columns(4)
        ss1.metric("Timezone", session_cfg.get("timezone", "—"))
        ss2.metric("Market open", session_cfg.get("market_open", "—"))
        ss3.metric("Market close", session_cfg.get("market_close", "—"))
        ss4.metric("EOD flatten", session_cfg.get("eod_flatten_time", "—"))

        ss5, ss6, ss7, ss8 = st.columns(4)
        ss5.metric("Opening guard", f"{session_cfg.get('opening_guard_minutes', 0)} min")
        ss6.metric("Closing guard", f"{session_cfg.get('closing_guard_minutes', 0)} min")
        ss7.metric("EOD flatten", "ON" if session_cfg.get("eod_flatten") else "OFF")
        ss8.metric("Pre-market", "ON" if session_cfg.get("enable_premarket") else "OFF")

    imp_cfg = config.get("improvement", {})
    if imp_cfg:
        st.subheader("Improvement loop")
        i1, i2, i3 = st.columns(3)
        i1.metric("Enabled", "ON" if imp_cfg.get("enabled") else "OFF")
        i2.metric("Autonomy", str(imp_cfg.get("autonomy_mode", "—")))
        i3.metric("Dry run", "ON" if imp_cfg.get("dry_run") else "OFF")

# ── Raw YAML ──────────────────────────────────────────────────────────

with tab_raw:
    st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")

# ── Compare configs ───────────────────────────────────────────────────

with tab_compare:
    if len(config_files) > 1:
        st.subheader("Compare configurations (on disk)")

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
    else:
        st.info("Add a second YAML file under `config/` to enable comparison.")
