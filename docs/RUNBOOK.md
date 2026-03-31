# Runbook

## Prerequisites

- Python 3.11+
- Alpaca account (paper or live)
- API keys configured in `.env`

## Setup

```bash
# Clone and install
cd Day_Trading_Agent
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Alpaca API keys
```

## Daily Operations

### Starting Paper Trading

```bash
python scripts/run_paper.py --config config/default.yaml
```

The bot will:
1. Connect to Alpaca paper account
2. Verify account status and buying power
3. Screen universe for tradable symbols
4. Enter main trading loop (runs until market close or SIGINT)
5. Flatten positions at EOD (15:45 ET by default)
6. Generate daily report

### Running a Backtest

```bash
# Standard backtest
python scripts/run_backtest.py --config config/backtest.yaml

# Walk-forward validation
python scripts/run_backtest.py --config config/backtest.yaml --walk-forward
```

### Running the Improvement Loop

```bash
# Manual proposal mode (default)
python scripts/run_improvement_loop.py --config config/default.yaml --enable

# Manual mode + explicit backtest evaluation gate
python scripts/run_improvement_loop.py --config config/default.yaml --enable --evaluate-backtest

# Non-production autonomous mode
python scripts/run_improvement_loop.py --config config/default.yaml --enable --autonomy-mode autonomous_nonprod

# Fully autonomous mode (applies accepted config patches)
python scripts/run_improvement_loop.py --config config/default.yaml --enable --autonomy-mode autonomous --apply
```

Operational notes:
1. Keep `improvement.dry_run: true` until evaluation quality is trusted.
2. Use `autonomous_nonprod` before `autonomous`.
3. By default, proposals are generated from observed paper/live logs without running evaluator backtests.
4. Enable `--evaluate-backtest` (or set `improvement.evaluate_with_backtest: true`) when you explicitly want backtest/walk-forward acceptance gating.
5. Only accepted proposals are applied; rejections are artifacted and logged.

### Starting Live Trading (CAUTION)

```bash
# Requires three safety gates:
# 1. ENABLE_LIVE_TRADING=true in .env
# 2. enable_live: true in config
# 3. --confirm flag on command line

ENABLE_LIVE_TRADING=true python scripts/run_live.py --config config/default.yaml --confirm
```

## Monitoring

### Log Files

```
logs/
├── agent_paper.log        # Paper runner structured JSON log
├── agent_backtest.log     # Backtest runner structured JSON log
├── agent_live.log         # Live runner structured JSON log
├── agent_sentiment.log    # Per-signal sentiment diagnostics
├── audit_YYYY-MM-DD.jsonl # Decision audit trail
├── improvement_YYYY-MM-DD.jsonl # Improvement-loop audit trail
└── reports/
    └── report_YYYY-MM-DD.json  # Daily performance report
```

### Key Log Patterns to Watch

```bash
# Errors
rg '"level":"ERROR"' logs/agent*.log

# Kill switch activations
rg 'kill_switch' logs/agent*.log

# Order rejections
rg '"status":"rejected"' logs/agent*.log

# Drawdown alerts
rg 'drawdown' logs/agent*.log

# Improvement-loop events
rg '"event_type"' logs/improvement_*.jsonl
```

## Troubleshooting

### Bot Won't Start
1. Check `.env` has valid API keys
2. Verify Alpaca account is active: `curl -H "APCA-API-KEY-ID: $KEY" https://paper-api.alpaca.markets/v2/account`
3. Check config YAML is valid: `python -c "from src.agent.config import load_config; print(load_config())"`

### No Trades Being Placed
1. Check if market is open (US/Eastern 09:30–16:00)
2. Check opening guard period (first 5 minutes)
3. Check circuit breaker status in logs
4. Verify ML model is loaded (`models/signal_model.joblib`)
5. Check signal confidence vs threshold (default 0.60)

### Kill Switch Activated
1. Check logs for the trigger reason
2. All open orders will be canceled, positions flattened
3. Bot will stop trading for the rest of the day
4. Restart the bot next trading day (kill switch resets daily)

### Position Reconciliation Mismatches
1. Check `logs/agent.log` for reconciliation warnings
2. Compare local state with Alpaca dashboard
3. Manually close any orphaned positions via Alpaca UI if needed

## Emergency Procedures

### Manual Kill Switch
```bash
# Send SIGINT to the process
kill -2 <pid>
# or Ctrl+C in terminal
```
The shutdown handler will:
1. Cancel all open orders
2. Close all positions
3. Generate final report

### Manual Position Flatten via Alpaca
```bash
# Cancel all orders
curl -X DELETE -H "APCA-API-KEY-ID: $KEY" -H "APCA-API-SECRET-KEY: $SECRET" \
  https://paper-api.alpaca.markets/v2/orders

# Close all positions
curl -X DELETE -H "APCA-API-KEY-ID: $KEY" -H "APCA-API-SECRET-KEY: $SECRET" \
  https://paper-api.alpaca.markets/v2/positions
```

## Configuration Tuning

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `risk.max_risk_per_trade_pct` | 0.5% | 0.1–2% | Risk per trade as % of equity |
| `risk.max_daily_drawdown_pct` | 2% | 1–5% | Daily loss limit |
| `strategy.ml_confidence_threshold` | 0.60 | 0.50–0.80 | Higher = fewer but higher-conviction trades |
| `strategy.volume_surge_ratio` | 1.5 | 1.2–3.0 | Volume confirmation threshold |
| `strategy.lookback_bars` | 20 | 10–50 | Breakout window length |
| `session.opening_guard_minutes` | 5 | 0–30 | Skip volatile open |
| `session.eod_flatten_time` | 15:45 | 15:30–15:55 | When to start closing positions |
