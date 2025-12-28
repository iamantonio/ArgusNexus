# Changelog

All notable changes to ArgusNexus V4-Core will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- PR-1: RiskManager integration for Portfolio Trader
- PR-2: Catastrophic Stop Enforcement
- PR-3: Trade Logging Completeness
- PR-3.1: Database migration for validity tracking

---

## [2025-12-27] - Bidirectional Trading & Monitoring

### SHORT Capability (v6.1)

**Goal:** Enable profitable trading in bear markets with bidirectional strategy.

#### Added
- SHORT entry signals when price breaks below 55-period Donchian low
- Bearish trend filter: price must be below 200 SMA
- Inverse Chandelier Exit: `Lowest_Low_Since_Entry + (3 × ATR)`
- Position direction tracking (`is_short_position` flag)
- Bearish setup grading in multi-timeframe analysis
- 50% position sizing for shorts (conservative risk management)

#### Changed
- `donchian.py` - Added SHORT entry/exit logic, inverse ratchet tracking
- `professional.py` - Added bearish grading, SHORT_POSITION_MULTIPLIER
- `engine.py` - Added position direction tracking, `lowest_low_since_entry`

---

### SHORT Simulation for Paper Trading

**Goal:** Enable end-to-end SHORT trading in paper mode without requiring margin exchange.

#### Added
- `ShortPosition` dataclass in `src/execution/paper.py`:
  - Tracks symbol, quantity, entry_price, collateral, opened_at
- Simulated margin trading with 100% collateral requirement
- Inverse P&L calculation: `(entry_price - exit_price) * quantity`
- `signal_to_order_side()` helper function in `src/engine.py`:
  - `long` → BUY, `short` → SELL
  - `exit_long` → SELL, `exit_short` → BUY
- New executor methods:
  - `_open_short()` - Opens short with collateral lock
  - `_close_short()` - Buys to cover, returns collateral + P&L
  - `has_short_position()`, `get_short_position()`
  - `get_realized_short_pnl()`, `get_unrealized_short_pnl()`
  - `get_all_short_positions()`

#### Changed
- `execute()` now routes to appropriate handler based on position state
- `get_position_size()` returns negative for short positions
- `get_total_value()` includes unrealized short P&L
- `reset_balance()` clears short positions

#### Backtest Results (6 months, $10k capital)
| Metric | LONG | SHORT | Combined |
|--------|------|-------|----------|
| Trades | 53 | 61 | 114 |
| Win Rate | 34% | 21% | 27% |
| Total P&L | -$1,145 | -$3,967 | -$5,112 |
| Best Trade | +$706 | +$481 | - |

**Note:** 2024 was bullish (BTC $45k→$100k), so shorts underperformed. Strategy needs tuning for counter-trend entries.

#### Files Modified
- `src/execution/paper.py` - Full SHORT simulation
- `src/engine.py` - Signal-to-order mapping, quantity fixes

---

### Live Trader SHORT Support

**Goal:** Enable SHORT positions in live/paper unified trader with proper stop management.

#### Added
- `is_short_position` field in `SymbolPosition` dataclass
- `lowest_low_since_entry` for SHORT chandelier tracking
- `signal_type` field in `SymbolSignal` (long/short/exit_long/exit_short)
- `_open_short_position()` helper method for SHORT entries
- `_close_short_position()` helper method for cover orders

#### Changed
- `SymbolPosition.calculate_hard_stop()` - Now calculates stop ABOVE entry for shorts
- `SymbolPosition.to_dict()/from_dict()` - Include new SHORT fields
- Stop check logic - Inverse comparisons for SHORT positions (>= vs <=)
- Chandelier ratchet - Tracks lowest low (ratchets DOWN) for shorts
- `execute_trade()` - Routes to appropriate handler based on signal type
- P&L calculation - Inverse for shorts: `(entry - exit) * qty`

#### Files Modified
- `scripts/live_unified_trader.py` - Full SHORT position support

---

### System Monitoring & Discord Alerts

**Goal:** Never miss a paper trader outage - immediate Discord notifications.

#### Added
- `/api/system/status` endpoint - Returns trader health (process + log freshness)
- `scripts/watchdog.sh` - Cron-based monitoring with Discord webhook alerts
- Dashboard "Trader" status indicator - Real-time status in header (polls every 30s)
- Cron job: `*/2 * * * *` - Checks every 2 minutes

#### Alert Types
- **DOWN** - Paper trader process not running (red)
- **STALE** - Process running but log stale >2 min (yellow)
- **RECOVERED** - Trader back online after outage (green)

#### Files Added
- `src/api/routes/system.py` - Health check endpoint
- `scripts/watchdog.sh` - Watchdog script with Discord integration

#### Files Modified
- `src/api/main.py` - Added system router
- `static/index.html` - Added trader status indicator + CSS + JS polling

---

### Social Card Improvements

#### Changed
- Decision cards now generate as 1200x630 PNG images (was SVG)
- Optimized for Twitter/X rich preview cards
- Added Pillow dependency for image generation

#### Files Added
- `src/api/card_image.py` - PNG generation with Pillow

#### Files Modified
- `src/api/routes/public.py` - Updated to serve PNG cards
- `requirements.txt` - Added Pillow>=10.0.0

---

## [2025-12-19] - Hardening Sprint

### PR-1: RiskManager Integration for Portfolio Trader

**Goal:** No order path without RiskManager approval.

#### Added
- `RiskManager.evaluate()` call before every order execution in `live_portfolio_trader.py`
- `TradeRequest` and `RiskPortfolioState` builders in portfolio trader
- Risk check results logged to decision records
- Fail-closed behavior: if `evaluate()` throws, order is blocked

#### Changed
- `execute_order()` now requires risk approval before proceeding
- Orders blocked by risk are logged with rejection reason

#### Files Modified
- `scripts/live_portfolio_trader.py` - Added risk gate integration
- `src/risk/manager.py` - Added `RiskConfigError` exception class
- `tests/unit/test_portfolio_risk_integration.py` - 9 acceptance tests

---

### PR-2: Catastrophic Stop Enforcement

**Goal:** Independent guardian loop with 15% stop, fail-closed on price feed errors.

#### Added
- `portfolio.catastrophic_stop_pct` config (default: 15%)
- `portfolio.guardian_check_interval_seconds` config (default: 60s)
- Guardian loop in `live_portfolio_trader.py`:
  - `_guardian_loop()` - async task checking price vs stop every 60s
  - `_get_cost_basis()` - retrieves WAC from state
  - `_compute_catastrophic_stop()` - calculates stop price from WAC
  - `_fetch_current_price()` - gets current price from data source
  - `_emergency_exit()` - triggers exit + halt + CRITICAL alert
- Cost basis (WAC) tracking in portfolio state JSON
- Durable halt: `trading_halted` + `halt_reason` persisted to state file

#### Changed
- `execute_order()` updates cost basis on BUY orders
- `save_state()` includes `btc_cost_basis` and `btc_total_cost`
- `run()` starts guardian task on startup

#### Files Modified
- `config.yaml` - Added `portfolio` section
- `scripts/live_portfolio_trader.py` - Added guardian loop and cost basis tracking
- `tests/unit/test_catastrophic_stop.py` - 16 acceptance tests

---

### PR-3: Trade Logging Completeness

**Goal:** Every closed trade auditable to the penny, invalid trades marked with reason.

#### Added
- `is_valid` column in trades table (INTEGER DEFAULT 1)
- `invalid_reason` column in trades table (TEXT)
- `is_valid` and `invalid_reason` fields in `Trade` dataclass
- Validation logic in `close_trade()`:
  - Detects None commission → marks invalid with `missing_commission`
  - Detects None slippage → marks invalid with `missing_slippage`
  - Uses 0 for calculation but preserves invalidity flag

#### Changed
- `close_trade()` signature: `commission` and `slippage` now `Optional[Decimal]`
- `close_trade()` logs `[INVALID]` suffix when data quality issues detected
- Trade `to_dict()` includes `is_valid` and `invalid_reason`

#### Files Modified
- `src/truth/schema.py` - Added validity columns to Trade dataclass and SQL schema
- `src/truth/logger.py` - Added validation logic in `close_trade()`
- `tests/unit/test_trade_logging_completeness.py` - 10 acceptance tests

---

### PR-3.1: Database Migration for Validity Tracking

**Goal:** Apply PR-3 schema changes to live database.

#### Added
- `is_valid` column to existing `trades` table (ALTER TABLE)
- `invalid_reason` column to existing `trades` table (ALTER TABLE)

#### Files Modified
- `data/v4_live_paper.db` - Schema migrated

#### Evidence
- Most recent closed trade verified with all required fields:
  - entry_timestamp, exit_timestamp, entry_price, exit_price, quantity
  - total_commission (non-NULL), total_slippage (computed, non-NULL)
  - net_pnl (non-NULL), duration_seconds (non-NULL), is_winner (non-NULL)
  - is_valid=1, invalid_reason=NULL

---

## Test Coverage

| PR | Tests | Status |
|----|-------|--------|
| PR-1 | 9 | PASS |
| PR-2 | 16 | PASS |
| PR-3 | 10 | PASS |
| **Total** | **35** | **ALL PASS** |

---

## Paper Dry Run - ACTIVE

**Launched:** 2025-12-19T22:19:07Z
**Status:** IN PROGRESS (2/10 real valid trades)

### Configuration
- **Symbols:** BTC-USD, ETH-USD only (SOL-USD excluded)
- **Max concentration:** 30% enforced
- **Guardian loop:** 60s interval, 15% catastrophic stop
- **Evaluation interval:** 1 day
- **Auto-halt triggers:** emergency_exit, risk exception, price feed failure

### Pre-flight Checks (all passed)
- [x] trading_halted=FALSE (cleared stale halt from pre-PR-1)
- [x] Config loaded (all required sections)
- [x] RiskManager active (risk_gate_enabled=TRUE)
- [x] Guardian loop active (interval=60s)
- [x] Max concentration=30%
- [x] Discord alert sent (DRY_RUN_START)
- [x] RUN_START record logged to database

### Current Metrics (2 real valid trades)
| Metric | Value |
|--------|-------|
| Real valid trades | 2 / 10 |
| From hardened portfolio_manager | 1 |
| Net PnL | -$115.27 |
| Win rate | 50% (1W/1L) |
| Avg hold time | 12.2 hours |
| Avg slippage | $0.00 |
| Invalid trades | 0 |
| Halts | 0 (during dry run) |

### Next Steps
- Continue monitoring until 10 valid closed trades
- Portfolio trader evaluates daily at 00:05 UTC
- Report back when target reached or halt occurs
