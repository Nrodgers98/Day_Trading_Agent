# Go-Live Checklist — Alpaca + US Equities

## Pre-Deployment (Backtest Phase)

- [ ] Backtest completed with >= 1 year of data
- [ ] Walk-forward validation shows consistent performance across windows
- [ ] Sharpe ratio > 1.0 (after slippage and realistic assumptions)
- [ ] Max drawdown within acceptable limits (< 10% recommended)
- [ ] Profit factor > 1.5
- [ ] No look-ahead bias confirmed (features use only past data)
- [ ] Slippage assumptions documented and realistic (5+ bps)
- [ ] Survivorship bias documented (using current S&P 500 list)

## Paper Trading Phase (Minimum 2–4 Weeks)

- [ ] Paper account set up on Alpaca
- [ ] Bot runs for full trading sessions without crashes
- [ ] Order placement and fill handling verified
- [ ] Position reconciliation shows no discrepancies
- [ ] EOD flatten works correctly (no overnight positions)
- [ ] Circuit breakers trigger correctly on simulated drawdown
- [ ] Kill switch activates and flattens positions
- [ ] Daily reports generated and reviewed
- [ ] Audit logs are complete and parseable
- [ ] Alert system fires appropriately
- [ ] No memory leaks over multi-day runs
- [ ] Paper PnL roughly matches backtest expectations (within reason)
- [ ] PDT rule awareness verified (if account < $25k)

## Alpaca-Specific Checks

- [ ] API keys rotated and stored securely (env vars only)
- [ ] Account status is ACTIVE
- [ ] Buying power is sufficient for intended position sizes
- [ ] Paper vs live base URLs are correct
- [ ] Rate limiting is respected (200 calls/min for free tier)
- [ ] All traded symbols verified as `tradable=true` on Alpaca
- [ ] Short selling only attempted on `shortable=true` and `easy_to_borrow=true` assets
- [ ] Fractional shares only used when `fractionable=true`
- [ ] `client_order_id` idempotency verified

## Risk Configuration Review

- [ ] `max_risk_per_trade_pct` set appropriately (recommend <= 1%)
- [ ] `max_daily_drawdown_pct` set (recommend 2%)
- [ ] `max_concurrent_positions` reasonable for account size
- [ ] `daily_trade_cap` set to avoid excessive trading
- [ ] Stop-loss and take-profit levels validated
- [ ] Trailing stop tested in paper
- [ ] Spread guard prevents trading illiquid stocks
- [ ] Concentration limit prevents overexposure to single name

## Live Deployment

- [ ] `ENABLE_LIVE_TRADING=true` set in environment
- [ ] `trading.enable_live: true` set in config
- [ ] Live API keys configured (`ALPACA_LIVE_API_KEY`, `ALPACA_LIVE_SECRET_KEY`)
- [ ] Live base URL: `https://api.alpaca.markets`
- [ ] Start with REDUCED position sizes (e.g., 25% of intended)
- [ ] Monitor first full trading day manually
- [ ] Verify fills match expected prices (check slippage)
- [ ] Verify daily PnL tracking is accurate
- [ ] Verify reconciliation matches Alpaca dashboard
- [ ] Set up external monitoring (uptime check for the process)
- [ ] Document emergency contact / manual kill procedure
- [ ] Inform relevant parties that live trading is active

## Ongoing Operations

- [ ] Review daily reports every trading day
- [ ] Weekly review of cumulative performance
- [ ] Monthly review of strategy parameters
- [ ] Re-run walk-forward backtest quarterly with new data
- [ ] Monitor for regime changes (volatility, correlation shifts)
- [ ] Keep dependencies updated (security patches)
- [ ] Rotate API keys periodically
- [ ] Maintain audit log backups

## Red Flags — Stop and Investigate

- Daily drawdown > 3% for 2+ consecutive days
- Win rate drops below 35% over 50+ trades
- Sharpe ratio turns negative over 1-month rolling window
- Reconciliation mismatches persist
- Multiple consecutive order rejections
- Unexpected overnight positions
- API errors > 5% of requests
