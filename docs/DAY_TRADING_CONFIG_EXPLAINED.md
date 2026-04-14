# Day trading settings explained (plain English)

This guide walks through **`config/default.yaml`** the way a trader would read it: what each knob does, and what usually happens if you turn it up or down. Values shown are **examples** from your file—yours may differ.

---

## `trading` — how the bot is allowed to trade

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `trading.mode` | **paper** = practice money at Alpaca paper; **live** = real brokerage (extra safety gates apply); **backtest** = historical simulation. | Live needs keys, flags, and serious risk review. |
| `trading.enable_live` | Must be **true** before the live runner will even start. | Extra safety so you don’t hit real money by mistake. |
| `trading.enable_shorting` | **true** = the bot may place short sales; **false** = sell signals that imply shorts are skipped before risk checks. | Turning shorts off often **reduces** “sell” ideas that would have been real orders. |
| `trading.enable_fractional_shares` | **true** = position sizes can be fractional shares where supported. | Helps small accounts size cleanly; depends on broker rules. |

---

## `universe` — which stocks are even in play

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `universe.base` | **custom** = only your list; **scan** = bot picks names from the market scanner; **hybrid** = mix of both. | Scan = more names, more API work; custom = full control. |
| `universe.custom_symbols` | Your fixed ticker list when using custom/hybrid. | Empty `[]` means “use scanner side only” for hybrid/scan. |
| `universe.filters.min_price` / `max_price` | Ignore stocks cheaper than **min** or pricier than **max**. | Tighter band = fewer tiny / mega caps. |
| `universe.filters.min_avg_volume` | Stock must have at least this **average daily volume** (roughly “how tradable / liquid it is”). | Higher = usually calmer fills, fewer exotic names. |
| `universe.filters.require_tradable` | Broker must say the asset is tradable. | Keeps out broken or restricted symbols. |
| `universe.filters.require_easy_to_borrow` | For shorts, stock must be **easy to borrow** (if you enable shorting). | Reduces short borrow headaches. |

---

## `scanner` — how the bot builds a watchlist (when `universe.base` is scan or hybrid)

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `scanner.enabled` | Master switch for the scanner feature. | **false** = no dynamic list from scanner (depends on universe mode). |
| `scanner.rescan_interval_minutes` | How often to refresh the watchlist during the day. | Lower = fresher list, more API calls. |
| `scanner.premarket_scan` | Run an initial scan before the regular session. | Helps have symbols ready at the open. |
| `scanner.scan_top_n` | How many “raw” candidates to pull from the market before filtering. | Bigger pool before your caps (`max_symbols`) cut it down. |
| `scanner.scan_gainers_losers` | Include movers (gainers/losers style lists), not only one type. | More variety; can be noisier. |
| `scanner.max_symbols` | **Hard cap** on how many symbols the bot will watch after scanning. | Lower = fewer symbols per cycle, less load. |
| `scanner.min_gap_pct` | Minimum **gap %** (how much it opened away from prior close) to count as “interesting.” | Higher = fewer names, more “in play” stocks. |
| `scanner.min_relative_volume` | Minimum **relative volume** vs a longer average (roughly “is today unusually active?”). | Higher = stricter “busy stock” filter. |

---

## `session` — clock, trading window, and end-of-day behavior

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `session.timezone` | Calendar used for “what day is it” and session times (e.g. **US/Eastern**). | Should match how you think about the US market day. |
| `session.market_open` / `market_close` | Regular session bounds used for scheduling logic. | Usually keep at exchange hours unless you know why you’re changing them. |
| `session.enable_premarket` / `enable_afterhours` | Whether to allow trading **outside** regular hours. | Off = classic RTH day-trading style. |
| `session.opening_guard_minutes` | **No new trades** for this many minutes **after** the open (avoids opening chaos). | Larger = fewer early whipsaws, fewer early entries. |
| `session.closing_guard_minutes` | **No new trades** this many minutes **before** the official close (last slice of the day is “exit only” in spirit). | Larger = you stop adding risk earlier; pairs with flatten time. |
| `session.eod_flatten` | If **true**, the runner tries to **close all positions** near the end of the day. | **false** = positions can stay open overnight (unless you close elsewhere). |
| `session.eod_flatten_time` | **Earliest** clock time (in `timezone`) to start that flatten. | Earlier = flatter earlier. The runner triggers flatten **after** the “no new trades” window begins (same clock neighborhood as `closing_guard`), while the exchange is still in regular hours—so positions are not left open simply because the bot stopped scanning for entries. |

---

## `strategy` — how entries are scored (momentum / ML stack)

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `strategy.name` | Label for the strategy style (here: momentum breakout). | Mostly documentation unless code branches on it. |
| `strategy.timeframe` | Bar size the bot reads most often (e.g. **5m** = five-minute candles). | Drives loop timing and how “fast” signals react. |
| `strategy.confirmation_timeframe` | Intended **higher timeframe** for extra trend context (e.g. 15m). | Present in config; **confirm in code** that your signal path uses it for entries. |
| `strategy.lookback_bars` | How many bars define the recent **high/low channel** for breakouts. | Longer = slower, smoother breakouts; shorter = more sensitive. |
| `strategy.volume_surge_ratio` | Today’s volume must be this many times “normal” to count as a **volume surge**. | Higher = fewer signals, stronger volume requirement. |
| `strategy.rsi_long_range` / `rsi_short_range` | RSI must sit in this band to count as healthy for **long** vs **short** setups. | Tighter = fewer valid RSI regimes. |
| `strategy.ml_confidence_threshold` | When the ML model is used, its confidence must be **at least** this to treat the idea as strong enough (combined with rules). | **Higher** = fewer trades, “pickier”; **lower** = more trades, noisier. |
| `strategy.cooldown_bars` | After a trade on a symbol, wait this many **bars of the strategy timeframe** before that symbol can fire again. | Larger = less churn, fewer re-entries on the same name. |
| `strategy.max_trades_per_day` | Per **symbol**, cap how many times you’ll trade it in a day (engine-level throttle). | Lower = less overtrading one ticker. |
| `strategy.max_hold_minutes` | **Backtest** engine uses this to force exits after N minutes in a simulated position. | In **live/paper**, confirm whether your path uses the same rule; otherwise treat as **simulation-focused** unless wired in runner. |

---

## `sentiment` — news / text mood overlay

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `sentiment.enabled` | Turn sentiment scoring **on** or **off**. | Off = faster, no news API load; on = extra veto/boost layer. |
| `sentiment.provider` | Which sentiment engine (**none**, **finbert**, etc.). | FinBERT = ML on headlines; needs model/API setup. |
| `sentiment.news_source` | Where headlines come from (**alpaca_news**, **newsapi**, …). | Must match your API keys and subscription. |
| `sentiment.model_name` | Which Hugging Face–style model name to load for FinBERT. | Change only if you know the model is compatible. |
| `sentiment.lookback_minutes` | How far back to search news for each symbol. | Longer = more context, slower / more data. |
| `sentiment.max_headlines` | Cap headlines scored per request. | Lower = faster, less news influence. |
| `sentiment.cache_ttl_seconds` | Don’t refetch/score the same symbol’s news more often than this. | Saves API quota. |
| `sentiment.positive_threshold` / `negative_threshold` | Mood score must cross these to count as clearly **bullish** or **bearish** enough to matter. | Wider gap = fewer sentiment vetoes/boosts. |
| `sentiment.confidence_boost` | How much to **bump** the signal confidence when news aligns with the trade idea. | Larger = stronger “news tailwind” effect (within validated limits). |

---

## `risk` — sizing, limits, and “how much pain is allowed”

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `risk.max_risk_per_trade_pct` | Each new position risks about this **fraction of account equity** (via sizing vs stop distance conceptually). | **Lower** = smaller lines; **higher** = bigger swings. |
| `risk.max_daily_drawdown_pct` | If account drops this much from the **day’s starting equity**, protective logic can trip (alerts / breaker behavior—see code paths). | Lower = tighter daily stop. |
| `risk.max_gross_exposure_pct` | **Total** long + short exposure cap vs equity (gross, not net). | Lower = fewer simultaneous full-size positions; often causes **“Risk rejected: Gross exposure limit”** when many names are on. |
| `risk.max_concurrent_positions` | Hard cap on how many open positions at once. | Lower = more selective portfolio. |
| `risk.max_symbol_concentration_pct` | No single symbol may use more than this fraction of equity. | Lower = more **concentration rejections** when a name wants a big line. |
| `risk.daily_trade_cap` | Max **round-trip style** trades per day the risk layer will allow (circuit breaker / cap—see implementation). | Lower = stops firing after fewer attempts. |
| `risk.stop_loss_atr_mult` / `take_profit_atr_mult` / `trailing_stop_atr_mult` | **Theoretical** stop / target distances in multiples of **ATR** (average true range = volatility ruler). | Used for analytics / some engines; **paper runner may not attach these as live bracket orders**—check execution code if you expect auto-stops at the broker. |
| `risk.enable_trailing_stop` | Whether trailing-stop style logic is enabled where the code supports it. | Pairs with trailing ATR multiple when implemented. |
| `risk.spread_guard_pct` | If bid–ask spread is wider than this **fraction of price**, skip trading that quote (too expensive to cross). | Lower = stricter; more “spread too wide” skips. |
| `risk.slippage_bps` | Assumed execution slippage in **basis points** (100 bps = 1%) for modeling / checks. | Used in risk/backtest style math, not the literal broker fee. |

---

## `alpaca` — talking to the broker and data APIs

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `alpaca.api_timeout_seconds` | How long to wait for a single HTTP call before giving up. | Higher = fewer timeouts on slow networks; slower failure detection. |
| `alpaca.max_retries` / `retry_base_delay` / `retry_max_delay` | How many times to retry failed calls and how delays grow (backoff). | More retries = more resilience, slower worst case. |
| `alpaca.rate_limit_calls_per_minute` | Soft guard so the bot doesn’t spam the API past a sane pace. | Lower = safer for limits; possibly slower scans. |
| `alpaca.use_websocket` | **true** = live streaming where supported; **false** = mostly REST polling. | Websockets need extra setup and stability handling. |

*(API keys and base URLs are usually loaded from **environment** / `.env`; the YAML `alpaca` block still holds timeouts and behavior.)*

---

## `backtest` — simulation-only (not live paper trading)

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `backtest.start_date` / `end_date` | Historical window for simulations. | Longer = more data, slower runs. |
| `backtest.initial_capital` | Starting account size in the simulator. | Scales PnL % vs dollars. |
| `backtest.commission_per_share` | Simulated per-share commission (often **0** for Alpaca-style commission-free). | Raise to stress-test costs. |
| `backtest.slippage_bps` | Simulated slippage in the backtester. | Higher = more pessimistic fills. |
| `backtest.walk_forward.train_days` | Length of each **training** segment in walk-forward analysis. | Bigger = more history per fit, fewer steps. |
| `backtest.walk_forward.validate_days` | Out-of-sample test window after each train. | Standard ML-style walk-forward. |
| `backtest.walk_forward.step_days` | How far the window slides forward each step. | Smaller = more steps, heavier compute. |

---

## `monitoring` — logs, reports, and alerts

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `monitoring.log_level` | How chatty logs are (**DEBUG**, **INFO**, **WARNING**, …). | **DEBUG** = huge files; **WARNING** = quieter. |
| `monitoring.log_dir` | Folder for JSON logs and reports. | Point elsewhere if you use a dedicated disk. |
| `monitoring.audit_log` | Write per-decision **audit JSONL** (`audit_YYYY-MM-DD.jsonl`). | **true** = much better forensics and backfill quality. |
| `monitoring.daily_report` | When **true**, the runner writes **per-calendar-day** reports while it runs (and on shutdown). | **false** = legacy “one dump at end” style depending on version. |
| `monitoring.report_format` | **json**, **csv**, or **markdown** daily report files. | JSON is easiest for tooling. |
| `monitoring.alert_on_drawdown` | Emit serious alerts when daily drawdown crosses a threshold (see alerts code). | Turn off only if you accept silent drawdowns. |
| `monitoring.alert_on_order_failure` | Alert when repeated order failures stack up. | Helps catch broker / connectivity issues. |
| `monitoring.alert_on_data_gap` | Alert when price data goes stale vs expectations. | Helps catch feed outages. |
| `monitoring.max_consecutive_failures` | How many failures in a row before the order-failure alert fires. | Lower = more sensitive. |

---

## `improvement` — autonomous “suggest / maybe apply config” loop

| Setting | In plain English | If you change it |
|--------|-------------------|------------------|
| `improvement.enabled` | Master switch for the improvement runner. | **false** = no automatic proposal pipeline. |
| `improvement.autonomy_mode` | **manual** = human gate; **autonomous_nonprod** = write candidate YAMLs; **autonomous** = can patch live config (dangerous). | Treat autonomous modes like production deploys. |
| `improvement.optimize_for` | What “better” means to the heuristic scorer (**risk**, **raw PnL**, **stability**). | Changes which proposals are favored when logic branches on it. |
| `improvement.analysis_lookback_days` | How many recent **daily report** files the loop considers. | Longer = more history, slower / more context. |
| `improvement.evaluation_timeout_seconds` | Max seconds for optional **backtest** evaluation of a proposal. | Larger = deeper tests, slower runs. |
| `improvement.proposal_cooldown_minutes` | Minimum spacing between improvement events to avoid spamming. | Larger = calmer, fewer proposals. |
| `improvement.max_proposals_per_run` | Cap proposals per invocation. | Lower = less noise per run. |
| `improvement.candidate_dir` | Where candidate YAML / artifacts are written. | Point to a writable folder. |
| `improvement.rag_*` | Sizes for the local “find relevant code/docs chunks” index. | Tweaks retrieval granularity for proposals. |
| `improvement.allow_code_patches` | Whether code-change proposals are even allowed. | **false** = config-only suggestions. |
| `improvement.evaluate_with_backtest` | If **true**, proposals are backtest-gated before auto-apply paths. | Strongly recommended before any autonomous apply. |
| `improvement.dry_run` | If **true**, don’t write live config even if “autonomous.” | Safety rail. |
| `improvement.observe_modes` | Which **`agent_*.log`** files failures are mined from (**paper**, **live**, or both). | Match how you actually run the bot. |
| `improvement.gates.*` | Numeric **accept/reject** rules for backtested proposals (Sharpe delta, drawdown worsening, trade count swing, etc.). | Stricter gates = fewer auto-changes, safer. |

---

## Quick mental model

1. **`universe` + `scanner`** decide *who* is on the radar.  
2. **`session`** decides *when* the bot is allowed to work and *when* to flatten.  
3. **`strategy` + `sentiment`** decide *whether* a stock gets a buy/sell idea.  
4. **`risk`** decides *whether that idea becomes a real order* and *how big* it is.  
5. **`monitoring` + `improvement`** decide *what gets recorded* and *whether the system tries to tune itself*.

If something “feels” off (too few trades, too many rejects), start with **`risk.*`**, **`strategy.ml_confidence_threshold`**, **`session` guards**, and **`trading.enable_shorting`**—those usually explain day-to-day behavior before you touch obscure keys.
