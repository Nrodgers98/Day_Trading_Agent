"""
Core backtest engine — walks through historical bars using the same
SignalEngine / RiskManager / FeatureVector pipeline as the live system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from itertools import groupby
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, Field

from src.agent.backtest.simulator import FillSimulator
from src.agent.config import AppConfig
from src.agent.data.features import compute_features
from src.agent.models import (
    AccountInfo,
    Bar,
    OrderRequest,
    Position,
    Side,
    SignalAction,
)
from src.agent.risk.manager import RiskManager
from src.agent.signal.engine import SignalEngine

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────


class BacktestResult(BaseModel):
    """Immutable output produced by a single backtest run."""

    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    daily_pnl: list[dict[str, Any]] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


# ── Internal position tracker ────────────────────────────────────────


@dataclass
class _OpenPosition:
    symbol: str
    side: Side
    qty: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────


def _parse_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


# ── Engine ────────────────────────────────────────────────────────────


class BacktestEngine:
    """Walk-forward capable backtest engine that reuses the live trading
    pipeline (SignalEngine, RiskManager, compute_features)."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._bt_cfg = config.backtest
        self._session_cfg = config.session

        self._simulator = FillSimulator(
            slippage_bps=self._bt_cfg.slippage_bps,
            commission_per_share=self._bt_cfg.commission_per_share,
        )
        self._signal_engine = SignalEngine(config.strategy)
        self._risk_manager = RiskManager(
            config.risk,
            config.monitoring,
            fractional=config.trading.enable_fractional_shares,
        )

        self._tz = ZoneInfo(config.session.timezone)

        open_dt = datetime.combine(
            datetime.today(),
            _parse_time(config.session.market_open),
        ) + timedelta(minutes=config.session.opening_guard_minutes)
        close_dt = datetime.combine(
            datetime.today(),
            _parse_time(config.session.market_close),
        ) - timedelta(minutes=config.session.closing_guard_minutes)

        self._session_open = open_dt.time()
        self._session_close = close_dt.time()
        self._eod_flatten_time: time | None = (
            _parse_time(config.session.eod_flatten_time)
            if config.session.eod_flatten
            else None
        )

    # ── public API ────────────────────────────────────────────────────

    async def run(
        self,
        symbols: list[str],
        bars_data: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Run a full backtest over *bars_data* (no look-ahead)."""
        cash: float = self._bt_cfg.initial_capital
        positions: dict[str, _OpenPosition] = {}
        equity_curve: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        daily_pnl: list[dict[str, Any]] = []

        sorted_data = self._prepare_data(symbols, bars_data)
        timeline = self._build_timeline(symbols, sorted_data)

        prev_date: object = None
        day_start_equity: float = cash
        flattened_today = False
        last_ts: datetime | None = None

        for ts, group_iter in groupby(timeline, key=lambda x: x[0]):
            bar_entries = list(group_iter)
            last_ts = ts
            local_ts = self._localize(ts)
            current_date = local_ts.date()

            # ── day roll ──────────────────────────────────────────────
            if current_date != prev_date:
                if prev_date is not None:
                    end_eq = self._compute_equity(cash, positions)
                    daily_pnl.append({
                        "date": prev_date.isoformat(),
                        "pnl": round(end_eq - day_start_equity, 2),
                        "equity": round(end_eq, 2),
                    })
                day_start_equity = self._compute_equity(cash, positions)
                prev_date = current_date
                flattened_today = False
                self._signal_engine._roll_day(
                    ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                )

            # ── update current prices ─────────────────────────────────
            current_bars: dict[str, Bar] = {}
            for _, sym, idx in bar_entries:
                bar = self._row_to_bar(sorted_data[sym].iloc[idx], sym)
                current_bars[sym] = bar
                if sym in positions:
                    positions[sym].current_price = bar.close

            # ── session guard ─────────────────────────────────────────
            if not self._in_session(local_ts):
                equity_curve.append(self._eq_point(ts, cash, positions))
                continue

            # ── EOD flatten ───────────────────────────────────────────
            if self._should_flatten(local_ts) and not flattened_today:
                flattened_today = True
                for sym in list(positions):
                    pos = positions.pop(sym)
                    cash, trade = self._close_position(
                        pos, pos.current_price, ts, cash,
                    )
                    trades.append(trade)

            if flattened_today:
                equity_curve.append(self._eq_point(ts, cash, positions))
                continue

            # ── max-hold auto-close ───────────────────────────────────
            max_hold = self._config.strategy.max_hold_minutes
            for sym in list(positions):
                pos = positions[sym]
                held_min = (ts - pos.entry_time).total_seconds() / 60
                if held_min >= max_hold:
                    exit_px = (
                        current_bars[sym].open
                        if sym in current_bars
                        else pos.current_price
                    )
                    cash, trade = self._close_position(
                        positions.pop(sym), exit_px, ts, cash,
                    )
                    trades.append(trade)

            # ── signal loop (per symbol with bar at this timestamp) ───
            for _, symbol, row_idx in bar_entries:
                current_bar = current_bars[symbol]

                # Only use data BEFORE this bar (no look-ahead)
                lookback_df = sorted_data[symbol].iloc[:row_idx]
                if len(lookback_df) < 2:
                    continue

                features = compute_features(lookback_df, symbol)
                features.extra["price"] = current_bar.open

                signal = self._signal_engine.generate_signal(
                    symbol, lookback_df, features,
                )

                if signal.action == SignalAction.HOLD:
                    continue

                # ── CLOSE signal ──────────────────────────────────────
                if signal.action == SignalAction.CLOSE:
                    if symbol in positions:
                        cash, trade = self._close_position(
                            positions.pop(symbol),
                            current_bar.open,
                            ts,
                            cash,
                        )
                        trades.append(trade)
                    continue

                # ── BUY / SELL signal ─────────────────────────────────
                target_side = (
                    Side.LONG
                    if signal.action == SignalAction.BUY
                    else Side.SHORT
                )

                if symbol in positions and positions[symbol].side == target_side:
                    continue  # already positioned in same direction

                if (
                    target_side == Side.SHORT
                    and not self._config.trading.enable_shorting
                ):
                    continue

                # close opposite position first
                if symbol in positions:
                    cash, trade = self._close_position(
                        positions.pop(symbol),
                        current_bar.open,
                        ts,
                        cash,
                    )
                    trades.append(trade)

                cash = self._try_open(
                    symbol,
                    target_side,
                    signal,
                    features,
                    current_bar,
                    ts,
                    cash,
                    positions,
                )

            equity_curve.append(self._eq_point(ts, cash, positions))

        # ── close remaining positions at last known price ─────────────
        if positions and last_ts is not None:
            for sym in list(positions):
                pos = positions.pop(sym)
                cash, trade = self._close_position(
                    pos, pos.current_price, last_ts, cash,
                )
                trades.append(trade)

        if prev_date is not None:
            end_eq = self._compute_equity(cash, positions)
            daily_pnl.append({
                "date": prev_date.isoformat(),
                "pnl": round(end_eq - day_start_equity, 2),
                "equity": round(end_eq, 2),
            })

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            daily_pnl=daily_pnl,
            config_snapshot=self._config.model_dump(),
        )

    async def run_walk_forward(
        self,
        symbols: list[str],
        bars_data: dict[str, pd.DataFrame],
    ) -> list[BacktestResult]:
        """Run walk-forward validation: slide a train/validate window
        across the data and return one BacktestResult per out-of-sample
        validation period."""
        wf = self._bt_cfg.walk_forward
        all_dates = self._collect_trading_dates(symbols, bars_data)

        if len(all_dates) < wf.train_days + wf.validate_days:
            logger.warning(
                "Insufficient data for walk-forward (%d dates, need %d+%d)",
                len(all_dates),
                wf.train_days,
                wf.validate_days,
            )
            return [await self.run(symbols, bars_data)]

        results: list[BacktestResult] = []
        start = 0
        total_windows = (
            (len(all_dates) - (wf.train_days + wf.validate_days)) // wf.step_days
        ) + 1

        while start + wf.train_days + wf.validate_days <= len(all_dates):
            val_start = all_dates[start + wf.train_days]
            val_end_idx = min(
                start + wf.train_days + wf.validate_days - 1,
                len(all_dates) - 1,
            )
            val_end = all_dates[val_end_idx]
            window_idx = len(results) + 1
            logger.info(
                "Walk-forward window %d/%d | validate=%s..%s",
                window_idx,
                total_windows,
                val_start,
                val_end,
            )

            window = self._slice_bars(symbols, bars_data, val_start, val_end)
            engine = BacktestEngine(self._config)
            result = await engine.run(symbols, window)
            results.append(result)
            logger.info(
                "Completed window %d/%d | trades=%d",
                window_idx,
                total_windows,
                len(result.trades),
            )

            start += wf.step_days

        return results

    # ── position management ───────────────────────────────────────────

    def _try_open(
        self,
        symbol: str,
        side: Side,
        signal: Any,
        features: Any,
        bar: Bar,
        ts: datetime,
        cash: float,
        positions: dict[str, _OpenPosition],
    ) -> float:
        """Attempt to open a new position; returns updated cash."""
        account = self._build_account(cash, positions)
        price = bar.open

        if features.atr_14 <= 0 or price <= 0:
            return cash

        desired_qty = self._risk_manager.sizer.calculate_size(
            account.equity, features.atr_14, price, side,
        )
        if desired_qty <= 0:
            return cash

        order = OrderRequest(
            symbol=symbol, side=side, qty=desired_qty, signal=signal,
        )
        positions_list = self._positions_list(positions)
        verdict = self._risk_manager.pre_trade_check(
            order, account, positions_list, features,
        )
        if not verdict.approved or verdict.adjusted_qty <= 0:
            return cash

        fill_order = order.model_copy(update={"qty": verdict.adjusted_qty})
        result = self._simulator.simulate_fill(fill_order, bar)

        commission = self._simulator.commission_per_share * result.filled_qty
        if side == Side.LONG:
            cash -= result.filled_qty * result.filled_avg_price
        else:
            cash += result.filled_qty * result.filled_avg_price
        cash -= commission

        positions[symbol] = _OpenPosition(
            symbol=symbol,
            side=side,
            qty=result.filled_qty,
            entry_price=result.filled_avg_price,
            entry_time=ts,
            current_price=bar.close,
        )
        return cash

    def _close_position(
        self,
        pos: _OpenPosition,
        exit_price: float,
        exit_time: datetime,
        cash: float,
    ) -> tuple[float, dict[str, Any]]:
        """Close *pos* at *exit_price* with slippage; returns (cash, trade)."""
        close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
        slippage = self._simulator.calculate_slippage(exit_price, close_side)
        adjusted_exit = exit_price + slippage

        exit_commission = self._simulator.commission_per_share * pos.qty
        entry_commission = self._simulator.commission_per_share * pos.qty

        if pos.side == Side.LONG:
            cash += pos.qty * adjusted_exit
            gross_pnl = (adjusted_exit - pos.entry_price) * pos.qty
        else:
            cash -= pos.qty * adjusted_exit
            gross_pnl = (pos.entry_price - adjusted_exit) * pos.qty
        cash -= exit_commission

        net_pnl = gross_pnl - entry_commission - exit_commission
        hold_secs = (exit_time - pos.entry_time).total_seconds()

        trade: dict[str, Any] = {
            "entry_time": pos.entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "symbol": pos.symbol,
            "side": pos.side.value,
            "qty": pos.qty,
            "entry_price": round(pos.entry_price, 4),
            "exit_price": round(adjusted_exit, 4),
            "pnl": round(net_pnl, 2),
            "hold_minutes": round(hold_secs / 60, 1),
        }
        return cash, trade

    # ── data preparation ──────────────────────────────────────────────

    @staticmethod
    def _prepare_data(
        symbols: list[str],
        bars_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = bars_data[sym].copy()
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)
            out[sym] = df
        return out

    @staticmethod
    def _build_timeline(
        symbols: list[str],
        sorted_data: dict[str, pd.DataFrame],
    ) -> list[tuple[datetime, str, int]]:
        timeline: list[tuple[datetime, str, int]] = []
        for sym in symbols:
            df = sorted_data[sym]
            ts_col = df["timestamp"] if "timestamp" in df.columns else df.index
            for i, ts in enumerate(ts_col):
                timeline.append((ts, sym, i))
        timeline.sort(key=lambda x: x[0])
        return timeline

    @staticmethod
    def _collect_trading_dates(
        symbols: list[str],
        bars_data: dict[str, pd.DataFrame],
    ) -> list[Any]:
        dates: set[Any] = set()
        for sym in symbols:
            df = bars_data[sym]
            if "timestamp" in df.columns:
                ds = pd.to_datetime(df["timestamp"]).dt.date
            else:
                ds = pd.to_datetime(df.index).date
            dates.update(ds)
        return sorted(dates)

    @staticmethod
    def _slice_bars(
        symbols: list[str],
        bars_data: dict[str, pd.DataFrame],
        start_date: Any,
        end_date: Any,
    ) -> dict[str, pd.DataFrame]:
        window: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = bars_data[sym]
            if "timestamp" in df.columns:
                bar_dates = pd.to_datetime(df["timestamp"]).dt.date
            else:
                bar_dates = pd.to_datetime(df.index).date
            mask = (bar_dates >= start_date) & (bar_dates <= end_date)
            window[sym] = df.loc[mask].reset_index(drop=True)
        return window

    # ── session helpers ───────────────────────────────────────────────

    def _localize(self, ts: datetime) -> datetime:
        if ts.tzinfo is not None:
            return ts.astimezone(self._tz)
        return ts

    def _in_session(self, local_ts: datetime) -> bool:
        t = local_ts.time()
        return self._session_open <= t <= self._session_close

    def _should_flatten(self, local_ts: datetime) -> bool:
        if self._eod_flatten_time is None:
            return False
        return local_ts.time() >= self._eod_flatten_time

    # ── equity helpers ────────────────────────────────────────────────

    @staticmethod
    def _compute_equity(
        cash: float,
        positions: dict[str, _OpenPosition],
    ) -> float:
        equity = cash
        for pos in positions.values():
            if pos.side == Side.LONG:
                equity += pos.qty * pos.current_price
            else:
                equity -= pos.qty * pos.current_price
        return equity

    def _eq_point(
        self,
        ts: datetime,
        cash: float,
        positions: dict[str, _OpenPosition],
    ) -> dict[str, Any]:
        return {
            "timestamp": ts.isoformat(),
            "equity": round(self._compute_equity(cash, positions), 2),
        }

    def _build_account(
        self,
        cash: float,
        positions: dict[str, _OpenPosition],
    ) -> AccountInfo:
        equity = self._compute_equity(cash, positions)
        return AccountInfo(
            equity=equity,
            cash=cash,
            buying_power=cash,
            portfolio_value=equity,
        )

    @staticmethod
    def _positions_list(
        positions: dict[str, _OpenPosition],
    ) -> list[Position]:
        result: list[Position] = []
        for pos in positions.values():
            if pos.side == Side.LONG:
                mv = pos.qty * pos.current_price
                upnl = (pos.current_price - pos.entry_price) * pos.qty
            else:
                mv = -(pos.qty * pos.current_price)
                upnl = (pos.entry_price - pos.current_price) * pos.qty
            result.append(
                Position(
                    symbol=pos.symbol,
                    side=pos.side,
                    qty=pos.qty,
                    avg_entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    market_value=mv,
                    unrealized_pnl=upnl,
                    entry_time=pos.entry_time,
                )
            )
        return result

    @staticmethod
    def _row_to_bar(row: pd.Series, symbol: str) -> Bar:
        return Bar(
            symbol=symbol,
            timestamp=row["timestamp"] if "timestamp" in row.index else row.name,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            vwap=(
                float(row["vwap"])
                if "vwap" in row.index and pd.notna(row["vwap"])
                else None
            ),
        )
