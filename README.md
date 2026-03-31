# Day Trading Agent

> **⚠️ DISCLAIMER: This software is for educational and research purposes only. It is NOT financial advice. Trading stocks involves significant risk of loss. Past performance does not guarantee future results. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.**

An AI-powered autonomous day trading system for US equities, built exclusively for the [Alpaca](https://alpaca.markets/) brokerage API.

## Features

- **Momentum breakout strategy** with technical + ML signal generation
- **XGBoost classifier** with probability calibration for signal confidence
- **Comprehensive risk management**: per-trade stops, daily drawdown limits, circuit breakers, kill switch
- **Alpaca-native**: paper trading, live trading (feature-flagged), full order lifecycle management
- **Backtesting engine** with walk-forward validation and realistic slippage modeling
- **Full audit trail**: every signal decision logged as structured JSON
- **Session-aware**: US market hours, holiday calendar, opening/closing guards, EOD flatten
- **Streamlit dashboard**: interactive backtest analysis, trade journal, risk monitoring, config viewer

## Architecture

```
Data Ingestion → Feature Pipeline → Signal Engine → Risk Manager → Execution → Monitoring
                                    (Technical +    (Pre-trade     (Alpaca      (Audit +
                                     ML ensemble)    checks +       REST API)    Alerts)
                                                     sizing)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system diagram.

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your Alpaca paper trading API keys
```

### 3. Run Backtest

```bash
python scripts/run_backtest.py --config config/backtest.yaml

# With walk-forward validation
python scripts/run_backtest.py --config config/backtest.yaml --walk-forward
```

### 4. Run Paper Trading

```bash
python scripts/run_paper.py --config config/default.yaml
```

### 5. Run Live Trading (Extreme Caution)

```bash
# Requires THREE safety gates:
# 1. ENABLE_LIVE_TRADING=true in .env
# 2. enable_live: true in config YAML
# 3. --confirm flag

ENABLE_LIVE_TRADING=true python scripts/run_live.py --confirm
```

### 6. Launch Streamlit Dashboard

Run from the project root after activating your virtual environment:

```bash
# Linux/Mac
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\activate

streamlit run dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501).

The app includes 5 pages:
- Dashboard
- Backtest Analyzer
- Trade Journal
- Risk Monitor
- Configuration Viewer

Optional flags:

```bash
# Change port
streamlit run dashboard/app.py --server.port 8502

# Expose on local network
streamlit run dashboard/app.py --server.address 0.0.0.0
```

### 7. Run Autonomous Improvement Loop

```bash
# Advisor/manual mode (default, dry-run)
python scripts/run_improvement_loop.py --config config/default.yaml --enable

# Non-production autonomous apply (writes candidate YAML artifacts)
python scripts/run_improvement_loop.py --config config/default.yaml --enable --autonomy-mode autonomous_nonprod

# Fully autonomous apply (writes into config/default.yaml) - use with caution
python scripts/run_improvement_loop.py --config config/default.yaml --enable --autonomy-mode autonomous --apply
```

The loop uses:
- `logs/audit_*.jsonl` and `logs/reports/report_*.json` for reward episodes.
- A lightweight local RAG index over strategy/risk/execution/config/runbook files.
- Backtest + walk-forward gates before any apply action.

## Project Structure

```
Day_Trading_Agent/
├── config/
│   ├── default.yaml          # Production config (paper trading)
│   └── backtest.yaml         # Backtest-specific overrides
├── src/agent/
│   ├── config.py             # Pydantic configuration models
│   ├── models.py             # Core domain models (signals, orders, positions)
│   ├── data/
│   │   ├── market_data.py    # Alpaca market data client
│   │   ├── features.py       # Technical feature engineering
│   │   └── sentiment.py      # Optional sentiment scoring
│   ├── signal/
│   │   ├── technical.py      # Rule-based breakout signals
│   │   ├── ml_model.py       # XGBoost ML classifier
│   │   └── engine.py         # Signal ensemble engine
│   ├── risk/
│   │   ├── position_sizer.py # ATR-based position sizing
│   │   ├── circuit_breaker.py# Drawdown stops, kill switch
│   │   └── manager.py        # Pre/post-trade risk checks
│   ├── execution/
│   │   ├── alpaca_client.py  # Alpaca order REST client
│   │   ├── order_manager.py  # Order lifecycle tracking
│   │   └── reconciliation.py # Position reconciliation
│   ├── backtest/
│   │   ├── simulator.py      # Fill simulation with slippage
│   │   ├── engine.py         # Backtest engine + walk-forward
│   │   └── analytics.py      # Performance metrics
│   ├── monitoring/
│   │   ├── logger.py         # Structured JSON logging
│   │   ├── audit.py          # Decision audit trail
│   │   ├── alerts.py         # Alerting system
│   │   └── reports.py        # Daily report generation
│   └── runner/
│       ├── base.py           # Abstract runner with session logic
│       ├── paper.py          # Paper trading runner
│       ├── live.py           # Live trading runner
│       └── backtest_runner.py# Backtest execution runner
├── dashboard/
│   ├── app.py                # Streamlit main entry point
│   ├── pages/                # Multi-page app (Dashboard, Backtest, Trades, Risk, Config)
│   └── utils/                # Data loaders, chart builders, Alpaca reader
├── scripts/
│   ├── run_backtest.py       # Backtest entry point
│   ├── run_paper.py          # Paper trading entry point
│   ├── run_live.py           # Live trading entry point
│   └── run_improvement_loop.py # Autonomous improvement runner
├── tests/
│   ├── unit/                 # Unit tests for all modules
│   └── integration/          # Integration tests with mocked Alpaca
├── docs/
│   ├── ARCHITECTURE.md       # System architecture
│   ├── RUNBOOK.md            # Operations guide
│   └── GO_LIVE_CHECKLIST.md  # Pre-deployment checklist
├── .env.example              # Environment variable template
└── requirements.txt          # Python dependencies
```

## Strategy: Momentum Breakout

| Parameter | Default | Description |
|-----------|---------|-------------|
| Timeframe | 5-minute bars | Primary candle resolution |
| Lookback | 20 bars | Breakout window |
| Volume surge | 1.5x average | Confirmation threshold |
| RSI filter (long) | 40–70 | Avoid overbought entries |
| ML confidence | 0.60 | Minimum ensemble score |
| Max hold | 120 minutes | Intraday time limit |

**Entry**: Price breaks above/below N-bar high/low with volume surge, VWAP confirmation, RSI in range, ML probability above threshold.

**Exit**: ATR-based stop-loss (1.5x), take-profit (2.5x), optional trailing stop (1x ATR), time stop, EOD flatten.

## Risk Controls

| Control | Default | Description |
|---------|---------|-------------|
| Risk per trade | 0.5% equity | Maximum loss per position |
| Daily drawdown | 2% equity | Kill switch trigger |
| Max positions | 10 | Concurrent position limit |
| Daily trade cap | 30 | Maximum trades per day |
| Concentration | 15% equity | Per-symbol exposure limit |
| Spread guard | 0.10% | Reject wide-spread stocks |
| EOD flatten | 15:45 ET | Close all before market close |
| Opening guard | 5 minutes | Skip volatile open |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# With coverage
python -m pytest tests/ --cov=src/agent --cov-report=term-missing
```

## Configuration

All behavior is controlled via `config/default.yaml` and environment variables. See the config file for all available options with descriptions.

The autonomous loop is configured under `improvement.*`:
- `enabled`, `autonomy_mode`, `optimize_for`
- `observe_modes` to choose runtime sources (`paper`, `live`)
- RAG knobs (`rag_top_k`, chunk sizes)
- Safety gates (`min_sharpe_delta`, `max_drawdown_worsen_pct`, trade-delta limit)
- `dry_run` for safe validation before applying any change

News sentiment can be enabled under `sentiment.*`:
- `sentiment.enabled: true`
- `sentiment.provider: finbert`
- `sentiment.news_source: alpaca_news | newsapi`
- tunable lookback, cache TTL, and veto thresholds

When enabled, the runner fetches recent headlines and scores them with FinBERT (`ProsusAI/finbert`) before signal generation.
If `sentiment.news_source` is `newsapi`, set `NEWS_API_KEY` in `.env`.

**Secrets** are loaded exclusively from environment variables (`.env` file). Never commit API keys.

## Known Limitations

- **Survivorship bias**: Backtests use current S&P 500 constituents; historical composition is not tracked
- **Latency**: REST polling adds latency vs WebSocket streaming (WebSocket support is stubbed but not implemented)
- **Sentiment**: News sentiment is a placeholder; wire in your preferred news API
- **Single strategy**: Only momentum breakout is implemented; mean reversion is noted as a future addition
- **No options/futures**: US equities only (NYSE/NASDAQ/AMEX)
- **PDT rule**: The bot does not enforce the Pattern Day Trader rule; monitor your account if under $25k

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and data flow
- [Runbook](docs/RUNBOOK.md) — Operations guide and troubleshooting
- [Go-Live Checklist](docs/GO_LIVE_CHECKLIST.md) — Pre-deployment validation steps

## License

This project is provided as-is for educational and research purposes. See disclaimer above.
