"""
Feature engineering pipeline — turns OHLCV bars into a FeatureVector.

All indicators are computed with the ``ta`` library except the trend-slope,
which uses a simple NumPy linear-regression fit over the last 10 EMA-21 values.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange

from src.agent.models import FeatureVector

logger = logging.getLogger(__name__)

MIN_BARS_REQUIRED = 2


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    """Return the last value of *series*, falling back to *default* on NaN."""
    if series.empty:
        return default
    val = series.iloc[-1]
    return float(val) if pd.notna(val) else default


def compute_features(bars_df: pd.DataFrame, symbol: str) -> FeatureVector:
    """Compute all technical features from an OHLCV DataFrame.

    Returns a ``FeatureVector`` with sensible defaults when there is
    insufficient data (fewer than ``MIN_BARS_REQUIRED`` rows).
    """
    now = datetime.now(tz=timezone.utc)

    if bars_df.empty or len(bars_df) < MIN_BARS_REQUIRED:
        logger.warning(
            "Not enough bars for %s (got %d, need %d)",
            symbol,
            len(bars_df) if not bars_df.empty else 0,
            MIN_BARS_REQUIRED,
        )
        return FeatureVector(symbol=symbol, timestamp=now)

    df = bars_df.copy()
    close: pd.Series = df["close"]
    high: pd.Series = df["high"]
    low: pd.Series = df["low"]
    volume: pd.Series = df["volume"]
    n = len(df)

    latest_ts: datetime = (
        df["timestamp"].iloc[-1] if "timestamp" in df.columns else now
    )

    # ── Returns ───────────────────────────────────────────────────────
    prev_1 = close.iloc[-2]
    returns_1 = float(close.iloc[-1] / prev_1 - 1.0) if prev_1 != 0 else 0.0

    returns_5 = 0.0
    if n >= 6:
        prev_5 = close.iloc[-6]
        returns_5 = float(close.iloc[-1] / prev_5 - 1.0) if prev_5 != 0 else 0.0

    # ── VWAP distance ─────────────────────────────────────────────────
    vwap_distance = 0.0
    if "vwap" in df.columns:
        latest_vwap = df["vwap"].iloc[-1]
        if pd.notna(latest_vwap) and latest_vwap > 0:
            vwap_distance = float(
                (close.iloc[-1] - latest_vwap) / latest_vwap
            )

    # ── RSI-14 ────────────────────────────────────────────────────────
    rsi_val = 50.0
    if n >= 15:
        rsi_val = _safe_last(
            RSIIndicator(close=close, window=14).rsi(), default=50.0
        )

    # ── EMA-9 ─────────────────────────────────────────────────────────
    ema9_val = 0.0
    if n >= 9:
        ema9_val = _safe_last(
            EMAIndicator(close=close, window=9).ema_indicator()
        )

    # ── EMA-21 ────────────────────────────────────────────────────────
    ema21_series: pd.Series | None = None
    ema21_val = 0.0
    if n >= 21:
        ema21_series = EMAIndicator(close=close, window=21).ema_indicator()
        ema21_val = _safe_last(ema21_series)

    # ── SMA-50 ────────────────────────────────────────────────────────
    sma50_val = 0.0
    if n >= 50:
        sma50_val = _safe_last(
            SMAIndicator(close=close, window=50).sma_indicator()
        )

    # ── ATR-14 ────────────────────────────────────────────────────────
    atr_val = 0.0
    if n >= 15:
        atr_val = _safe_last(
            AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
        )

    # ── Volume ratio (current bar / 20-bar average) ──────────────────
    volume_ratio = 1.0
    if n >= 20:
        avg_vol = float(volume.iloc[-20:].mean())
        if avg_vol > 0:
            volume_ratio = float(volume.iloc[-1] / avg_vol)

    # ── Trend slope (linear-regression slope of EMA-21 over 10 bars) ─
    trend_slope = 0.0
    if ema21_series is not None and n >= 31:
        slope_window = ema21_series.iloc[-10:].dropna()
        if len(slope_window) >= 2:
            x = np.arange(len(slope_window), dtype=np.float64)
            y = slope_window.to_numpy(dtype=np.float64)
            x_mean = x.mean()
            denom = float(np.sum((x - x_mean) ** 2))
            if denom > 0:
                trend_slope = float(
                    np.sum((x - x_mean) * (y - y.mean())) / denom
                )

    return FeatureVector(
        symbol=symbol,
        timestamp=latest_ts,
        returns_1=returns_1,
        returns_5=returns_5,
        vwap_distance=vwap_distance,
        rsi_14=rsi_val,
        ema_9=ema9_val,
        ema_21=ema21_val,
        sma_50=sma50_val,
        atr_14=atr_val,
        volume_ratio=volume_ratio,
        trend_slope=trend_slope,
    )
