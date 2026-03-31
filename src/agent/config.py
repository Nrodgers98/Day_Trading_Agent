"""
Centralised configuration using Pydantic v2 models.

Loads from YAML files with .env overrides for secrets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

load_dotenv()

# ── Sub-models ────────────────────────────────────────────────────────


class TradingConfig(BaseModel):
    mode: Literal["paper", "live", "backtest"] = "paper"
    enable_live: bool = False
    enable_shorting: bool = False
    enable_fractional_shares: bool = False


class UniverseFilters(BaseModel):
    min_price: float = 5.0
    max_price: float = 500.0
    min_avg_volume: int = 1_000_000
    require_tradable: bool = True
    require_easy_to_borrow: bool = True


class UniverseConfig(BaseModel):
    base: Literal["custom", "scan", "hybrid"] = "scan"
    custom_symbols: list[str] = Field(default_factory=list)
    filters: UniverseFilters = Field(default_factory=UniverseFilters)


class ScannerConfig(BaseModel):
    enabled: bool = True
    rescan_interval_minutes: int = 15
    premarket_scan: bool = True
    scan_top_n: int = 50
    scan_gainers_losers: bool = True
    max_symbols: int = 30
    min_gap_pct: float = 2.0
    min_relative_volume: float = 1.5


class SessionConfig(BaseModel):
    timezone: str = "US/Eastern"
    market_open: str = "09:30"
    market_close: str = "16:00"
    enable_premarket: bool = False
    enable_afterhours: bool = False
    opening_guard_minutes: int = 5
    closing_guard_minutes: int = 15
    eod_flatten: bool = True
    eod_flatten_time: str = "15:45"


class StrategyConfig(BaseModel):
    name: str = "momentum_breakout"
    timeframe: str = "5m"
    confirmation_timeframe: str = "15m"
    lookback_bars: int = 20
    volume_surge_ratio: float = 1.5
    rsi_long_range: list[float] = Field(default_factory=lambda: [40.0, 70.0])
    rsi_short_range: list[float] = Field(default_factory=lambda: [30.0, 60.0])
    ml_confidence_threshold: float = 0.60
    cooldown_bars: int = 6
    max_trades_per_day: int = 20
    max_hold_minutes: int = 120


class SentimentConfig(BaseModel):
    enabled: bool = False
    provider: Literal["none", "finbert"] = "none"
    news_source: Literal["alpaca_news", "newsapi"] = "alpaca_news"
    model_name: str = "ProsusAI/finbert"
    lookback_minutes: int = 120
    max_headlines: int = 12
    cache_ttl_seconds: int = 300
    positive_threshold: float = 0.25
    negative_threshold: float = -0.25
    confidence_boost: float = 0.05

    @field_validator("lookback_minutes", "max_headlines", "cache_ttl_seconds")
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("sentiment numeric values must be > 0")
        return v

    @field_validator("positive_threshold", "negative_threshold", "confidence_boost")
    @classmethod
    def _sentiment_ranges(cls, v: float, info) -> float:
        if info.field_name in {"positive_threshold", "negative_threshold"}:
            if v < -1.0 or v > 1.0:
                raise ValueError(f"{info.field_name} must be within [-1.0, 1.0]")
        if info.field_name == "confidence_boost" and (v < 0.0 or v > 0.5):
            raise ValueError("confidence_boost must be within [0.0, 0.5]")
        return v


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = 0.005
    max_daily_drawdown_pct: float = 0.02
    max_gross_exposure_pct: float = 0.80
    max_concurrent_positions: int = 10
    max_symbol_concentration_pct: float = 0.15
    daily_trade_cap: int = 30
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 2.5
    trailing_stop_atr_mult: float = 1.0
    enable_trailing_stop: bool = True
    spread_guard_pct: float = 0.001
    slippage_bps: float = 5.0


class AlpacaConfig(BaseModel):
    api_key: str = Field(default="")
    secret_key: str = Field(default="")
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    api_timeout_seconds: int = 10
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    rate_limit_calls_per_minute: int = 200
    use_websocket: bool = False

    @field_validator("base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, v: str) -> str:
        """Normalize trading base URL to host-only form.

        Users sometimes set ALPACA_BASE_URL to .../v2, while the clients
        already include /v2 in request paths. Strip trailing /v2 to avoid
        duplicated URLs like /v2/v2/account.
        """
        if not isinstance(v, str):
            return v
        url = v.rstrip("/")
        if url.endswith("/v2"):
            url = url[:-3]
        return url

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: dict) -> dict:
        values.setdefault("api_key", os.getenv("ALPACA_API_KEY", ""))
        values.setdefault("secret_key", os.getenv("ALPACA_SECRET_KEY", ""))
        values.setdefault(
            "base_url",
            os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        )
        values.setdefault(
            "data_url",
            os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets"),
        )
        return values


class WalkForwardConfig(BaseModel):
    train_days: int = 252
    validate_days: int = 63
    step_days: int = 21

    @field_validator("train_days", "validate_days", "step_days")
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("walk_forward values must be > 0")
        return v


class BacktestConfig(BaseModel):
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    initial_capital: float = 100_000.0
    commission_per_share: float = 0.0
    slippage_bps: float = 5.0
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)


class MonitoringConfig(BaseModel):
    log_level: str = "INFO"
    log_dir: str = "logs"
    audit_log: bool = True
    daily_report: bool = True
    report_format: Literal["json", "csv", "markdown"] = "json"
    alert_on_drawdown: bool = True
    alert_on_order_failure: bool = True
    alert_on_data_gap: bool = True
    max_consecutive_failures: int = 3


class ImprovementGatesConfig(BaseModel):
    max_drawdown_worsen_pct: float = 0.2
    min_sharpe_delta: float = 0.05
    max_trades_delta_pct: float = 0.30
    min_profit_factor_delta: float = 0.0
    min_canary_sessions: int = 3
    rollback_underperformance_sessions: int = 2


class ImprovementConfig(BaseModel):
    enabled: bool = False
    autonomy_mode: Literal["manual", "autonomous_nonprod", "autonomous"] = "manual"
    optimize_for: Literal["risk_adjusted_return", "raw_pnl", "stability"] = "risk_adjusted_return"
    analysis_lookback_days: int = 7
    evaluation_timeout_seconds: int = 1800
    proposal_cooldown_minutes: int = 60
    max_proposals_per_run: int = 3
    candidate_dir: str = "output/improvement"
    rag_top_k: int = 6
    rag_max_chunk_lines: int = 60
    rag_chunk_overlap_lines: int = 10
    allow_code_patches: bool = False
    evaluate_with_backtest: bool = False
    dry_run: bool = True
    observe_modes: list[Literal["paper", "live"]] = Field(default_factory=lambda: ["paper"])
    gates: ImprovementGatesConfig = Field(default_factory=ImprovementGatesConfig)

    @field_validator(
        "analysis_lookback_days",
        "evaluation_timeout_seconds",
        "proposal_cooldown_minutes",
        "max_proposals_per_run",
        "rag_top_k",
        "rag_max_chunk_lines",
    )
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("improvement numeric values must be > 0")
        return v

    @field_validator("rag_chunk_overlap_lines")
    @classmethod
    def _chunk_overlap_bounds(cls, v: int, info) -> int:
        max_chunk = info.data.get("rag_max_chunk_lines", 60)
        if v < 0 or v >= max_chunk:
            raise ValueError("rag_chunk_overlap_lines must be >= 0 and < rag_max_chunk_lines")
        return v

    @field_validator("observe_modes")
    @classmethod
    def _validate_observe_modes(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("observe_modes must contain at least one mode")
        deduped = list(dict.fromkeys(v))
        return deduped


# ── Root Config ───────────────────────────────────────────────────────


class AppConfig(BaseModel):
    trading: TradingConfig = Field(default_factory=TradingConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    improvement: ImprovementConfig = Field(default_factory=ImprovementConfig)

    @model_validator(mode="after")
    def _safety_checks(self) -> "AppConfig":
        if self.trading.mode == "live" and not self.trading.enable_live:
            raise ValueError(
                "Live trading requested but enable_live is False. "
                "Set trading.enable_live=true AND ENABLE_LIVE_TRADING=true in env."
            )
        env_live = os.getenv("ENABLE_LIVE_TRADING", "false").lower()
        if self.trading.mode == "live" and env_live != "true":
            raise ValueError(
                "Live trading requires ENABLE_LIVE_TRADING=true in environment."
            )
        return self


def load_config(path: str | Path = "config/default.yaml", overrides: dict | None = None) -> AppConfig:
    """Load configuration from YAML with optional dict overrides."""
    cfg_path = Path(path)
    data: dict = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}

    if overrides:
        _deep_merge(data, overrides)

    return AppConfig(**data)


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base
