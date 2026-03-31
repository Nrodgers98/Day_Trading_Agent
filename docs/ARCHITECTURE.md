# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DAY TRADING AGENT                               │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ Scheduler│──>│  DATA    │──>│ FEATURE  │──>│  SIGNAL ENGINE │  │
│  │ (clock)  │   │ INGEST   │   │ PIPELINE │   │  (ML + Rules)  │  │
│  └──────────┘   └──────────┘   └──────────┘   └───────┬────────┘  │
│                       │                                │            │
│                       │ Alpaca Market                  │ Signals    │
│                       │ Data API                       ▼            │
│                       │              ┌──────────────────────────┐   │
│                       │              │     RISK MANAGER         │   │
│                       │              │  • Pre-trade checks      │   │
│                       │              │  • Position sizing       │   │
│                       │              │  • Circuit breakers      │   │
│                       │              │  • Kill switch           │   │
│                       │              └───────────┬──────────────┘   │
│                       │                          │ Approved Orders  │
│                       │                          ▼                  │
│                       │              ┌──────────────────────────┐   │
│  ┌──────────────┐     │              │   EXECUTION ENGINE       │   │
│  │  MONITORING  │◄────┼──────────────│  • Alpaca REST client    │   │
│  │  • Audit log │     │              │  • Order manager         │   │
│  │  • Alerts    │     │              │  • Fill handling         │   │
│  │  • Reports   │     │              │  • Reconciliation        │   │
│  │  • Metrics   │     │              └──────────────────────────┘   │
│  └──────────────┘     │                          │                  │
│                       │                          ▼                  │
│                       │              ┌──────────────────────────┐   │
│                       └──────────────│      ALPACA API          │   │
│                                      │  (Paper / Live)          │   │
│                                      └──────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    BACKTEST ENGINE                            │   │
│  │   Historical Data → Features → Signals → Simulated Fills    │   │
│  │   Walk-Forward Validation │ Analytics │ Reports              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### Data Layer (`src/agent/data/`)
- **market_data.py**: Async Alpaca API client (bars, quotes, account, assets, positions)
- **features.py**: Technical feature computation (RSI, EMA, ATR, VWAP distance, volume ratio)
- **sentiment.py**: Optional news sentiment scoring (graceful degradation)

### Signal Layer (`src/agent/signal/`)
- **technical.py**: Rule-based momentum breakout signal generator
- **ml_model.py**: XGBoost classifier with probability calibration
- **engine.py**: Weighted ensemble combining technical + ML signals

### Risk Layer (`src/agent/risk/`)
- **position_sizer.py**: ATR/volatility-based position sizing
- **circuit_breaker.py**: Daily drawdown stop, trade caps, kill switch
- **manager.py**: Pre-trade validation, exit level calculation, post-trade checks

### Execution Layer (`src/agent/execution/`)
- **alpaca_client.py**: Alpaca order REST client with retries and rate limiting
- **order_manager.py**: Order lifecycle tracking, partial fill handling
- **reconciliation.py**: Expected vs actual position reconciliation

### Backtest Layer (`src/agent/backtest/`)
- **simulator.py**: Realistic fill simulation with slippage
- **engine.py**: Event-driven backtest with walk-forward support
- **analytics.py**: Performance metrics (Sharpe, drawdown, profit factor)

### Monitoring Layer (`src/agent/monitoring/`)
- **logger.py**: Structured JSON logging with rotation
- **audit.py**: Decision audit trail (JSONL per day)
- **alerts.py**: Throttled alerting for drawdown, failures, data gaps
- **reports.py**: Daily report generation (JSON/CSV/Markdown)

### Runner Layer (`src/agent/runner/`)
- **base.py**: Abstract runner with session/calendar logic
- **paper.py**: Paper trading main loop
- **live.py**: Live trading (inherits paper, adds safety gates)
- **backtest_runner.py**: Backtest execution and reporting

## Data Flow

1. **Clock tick** → Scheduler triggers at configured interval
2. **Data fetch** → Bars + quotes from Alpaca Market Data API
3. **Feature computation** → Technical indicators + optional sentiment
4. **Signal generation** → Technical rules + ML model → ensemble score
5. **Risk check** → Pre-trade validation against all constraints
6. **Order placement** → Submit to Alpaca with idempotent client_order_id
7. **Fill handling** → Track status, handle partials, update positions
8. **Audit** → Log complete decision trace
9. **Reconciliation** → Periodic position sync with broker

## Key Design Decisions

- **Async throughout**: All I/O operations use `httpx` async client
- **Config-driven**: All behavior controlled via YAML + env vars
- **No LLM in hot path**: ML inference is lightweight scikit-learn/XGBoost
- **Fail-safe defaults**: Kill switch, EOD flatten, paper mode ON
- **Idempotent orders**: Every order has a unique `client_order_id`
- **Walk-forward backtest**: Prevents overfitting with rolling windows
