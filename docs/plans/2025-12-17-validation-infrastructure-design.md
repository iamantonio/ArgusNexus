# ArgusNexus V4 Validation Infrastructure Design

**Date:** 2025-12-17
**Author:** Tony (Lead Engineer) + Agent Team
**Status:** Approved for Implementation

---

## Executive Summary

This document defines two parallel workstreams to reach production-ready validation:

1. **Replay Runner** - Fast path to 100+ validated trades (Gate A)
2. **Bracket Orders** - On-exchange TP/SL protection (safety foundation)

Both must be implemented before live trading with real capital.

---

## Validation Gates

| Gate | Requirement | Metric |
|------|-------------|--------|
| **A (Fast)** | 100+ replay trades through production engine | Strategy + logging + risk math validated |
| **B (Real-time)** | 10-20 live paper trades | Execution + operational behavior validated |

---

# Part 1: Replay Runner

## 1.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REPLAY RUNNER ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

                         CLI Entry Point
                              │
                    scripts/replay_runner.py
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │   Data   │       │   Run    │       │  Report  │
    │  Loader  │       │  Engine  │       │ Generator│
    └────┬─────┘       └────┬─────┘       └────┬─────┘
         │                  │                  │
    ┌────▼─────┐       ┌────▼─────┐       ┌────▼─────┐
    │ Coinbase │       │ Existing │       │  JSON +  │
    │ API/Cache│       │ Engine   │       │ Console  │
    └──────────┘       │ + Risk   │       └──────────┘
                       │ + Truth  │
                       └──────────┘

Key principle: Replay uses the SAME engine.run_tick() path as live trading.
```

## 1.2 Data Source

- **Primary:** Coinbase API (canonical source)
- **Cache:** `data/cache/{symbol}_{start}_{end}_4h.csv`
- **Symbols:** BTC-USD, ETH-USD, SOL-USD (core trio)
- **CSV abandoned:** Volume=0 in existing CSVs makes them structurally incompatible

## 1.3 CLI Specification

```bash
python scripts/replay_runner.py \
    --symbols BTC-USD,ETH-USD,SOL-USD \
    --start 2024-01-01 \
    --end 2025-12-17 \
    --data-source cache \
    --capital-mode independent \
    --starting-capital 500 \
    --run-mode replay_backtest \
    --report-format json
```

**Debug-only flags:**
```bash
--skip-volume-filter   # Tags run as data_integrity=degraded_skip_volume
```

## 1.4 Run Metadata Schema

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    run_mode TEXT NOT NULL,           -- replay_backtest | paper_live
    capital_mode TEXT,                -- independent | portfolio
    market_time_basis TEXT,           -- candle_timestamp | wall_clock
    data_integrity TEXT,              -- ok | degraded_skip_volume
    created_by TEXT,                  -- replay_runner | live_paper_trader
    git_sha TEXT,
    config_hash TEXT,
    symbols TEXT,
    start_date TEXT,
    end_date TEXT,
    started_at TEXT,
    completed_at TEXT,
    trades_opened INT DEFAULT 0,
    trades_closed INT DEFAULT 0,
    notes TEXT
);

-- Alterations to existing tables
ALTER TABLE decisions ADD COLUMN run_id TEXT REFERENCES runs(run_id);
ALTER TABLE orders ADD COLUMN run_id TEXT REFERENCES runs(run_id);
ALTER TABLE trades ADD COLUMN run_id TEXT REFERENCES runs(run_id);

CREATE INDEX idx_decisions_run_id ON decisions(run_id);
CREATE INDEX idx_orders_run_id ON orders(run_id);
CREATE INDEX idx_trades_run_id ON trades(run_id);
```

## 1.5 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| One engine per symbol | Yes | Prevents state bleed (has_open_position is single boolean) |
| Rolling window | 300 bars | O(N) not O(N²), same as live |
| Timestamp source | `as_of` parameter | Deterministic, no wall-clock in replay |
| Run metadata | Nullable run_id | Don't break existing data |
| Capital mode | Independent for Gate A | Validates per-symbol, portfolio mode later |

## 1.6 Report Metrics

```
TRADES
  Opened / Closed / Win Rate / Avg Win / Avg Loss
  R:R Actual / Expectancy / Profit Factor

RISK
  Max Drawdown ($ and %) / Peak Equity / Final Equity
  Max Consecutive Losses (count and $)

DATA QUALITY
  Candles fetched / Gaps detected / Volume anomalies

GLASS BOX
  "Why" Lookup: avg / p95 / max latency
  Audit Chain Coverage: % of trades with full decision trail
```

---

# Part 2: Bracket Orders (On-Exchange Protection)

## 2.1 Goal

When entering a position, the exchange must immediately have TP and SL orders.
No "stop stored in SQLite." If protection can't be confirmed, fail-closed.

## 2.2 Coinbase API Contract

**Payload structure:**
```json
{
  "client_order_id": "ANX4_abc12345_BTCUSD_20250115T1200_7f3a2b1c",
  "product_id": "BTC-USD",
  "side": "BUY",
  "order_configuration": {
    "limit_limit_gtc": {
      "base_size": "0.00123",
      "limit_price": "104500.00",
      "post_only": false
    }
  },
  "attached_order_configuration": {
    "trigger_bracket_gtc": {
      "limit_price": "112860.00",
      "stop_trigger_price": "102361.00"
    }
  }
}
```

## 2.3 Execution Flow Sequence Diagram

```
┌────────┐  ┌────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐
│ Engine │  │ Sizing │  │ Executor │  │Coinbase│  │  Truth   │
└───┬────┘  └───┬────┘  └────┬─────┘  └───┬────┘  └────┬─────┘
    │           │            │            │            │
    │ LONG signal            │            │            │
    ├──────────>│            │            │            │
    │           │            │            │            │
    │  Calculate bracket-aware size       │            │
    │<──────────│            │            │            │
    │           │            │            │            │
    │  Check for duplicate (client_order_id)          │
    ├───────────────────────>│            │            │
    │           │            │            │            │
    │  Build bracket payload │            │            │
    ├───────────────────────>│            │            │
    │           │            │            │            │
    │           │   create_order()        │            │
    │           │            ├───────────>│            │
    │           │            │            │            │
    │           │   order_id + success    │            │
    │           │            │<───────────│            │
    │           │            │            │            │
    │           │   list_orders() [verify attached]   │
    │           │            ├───────────>│            │
    │           │            │            │            │
    │           │   2 attached orders     │            │
    │           │            │<───────────│            │
    │           │            │            │            │
    │  Verification PASSED   │            │            │
    │<───────────────────────│            │            │
    │           │            │            │            │
    │  Log decision + order + trade       │            │
    ├─────────────────────────────────────────────────>│
    │           │            │            │            │
    │  state = ENTRY_PENDING │            │            │
    │           │            │            │            │
```

## 2.4 Engine State Machine

```
                         ┌─────────┐
                         │  IDLE   │
                         └────┬────┘
                              │ LONG signal + sizing OK
                              │ place_bracket_order()
                              ▼
                    ┌─────────────────┐
                    │  ENTRY_PENDING  │◄──────────────────┐
                    └────────┬────────┘                   │
                             │                            │
              ┌──────────────┼──────────────┐            │
              │              │              │            │
              ▼              ▼              ▼            │
         [FILLED]      [CANCELLED]     [FAILED]         │
              │              │              │            │
              ▼              │              ▼            │
     ┌────────────────┐     │         ┌──────────┐      │
     │ POSITION_OPEN  │     │         │FAIL_CLOSED│     │
     └───────┬────────┘     │         └──────────┘      │
             │              │                            │
             │ TP/SL fills  │                            │
             │ (on exchange)│                            │
             ▼              │                            │
    ┌────────────────┐      │                            │
    │  RECONCILE     │      │                            │
    │  (detect exit) │      │                            │
    └───────┬────────┘      │                            │
            │               │                            │
            │ finalize_exit()                            │
            ▼               │                            │
         ┌─────────┐        │                            │
         │  IDLE   │◄───────┴────────────────────────────┘
         └─────────┘
```

## 2.5 Position Sizing (Bracket-Aware)

Coinbase's `trigger_bracket_gtc` applies a **5% buffer** on stop-limit fills (hypothesis - confirm with live order).

**Sizing formula:**
```python
worst_case_fill = stop_trigger_price * (1 - 0.05)
worst_case_risk_per_unit = entry_price - worst_case_fill
risk_budget = capital * 0.01  # 1% risk
quantity = risk_budget / worst_case_risk_per_unit
```

**Example ($500 account):**
| Input | Value |
|-------|-------|
| Entry | $104,500 |
| Stop trigger | $102,361 |
| Worst-case fill (5%) | $97,243 |
| Risk budget (1%) | $5 |
| Position size | 0.000689 BTC |

## 2.6 Price Rounding (Directional)

| Level | Direction | Reason |
|-------|-----------|--------|
| Entry limit (BUY) | FLOOR | Never pay more |
| Take-profit (SELL) | FLOOR | Hit slightly earlier |
| Stop trigger (SELL) | CEIL | Trigger sooner is safer |

## 2.7 Verification Requirements

| Check | Requirement | Fail Action |
|-------|-------------|-------------|
| Parent created | `order_id` in response | FAILED |
| 2 attached orders | `originating_order_id == parent` | FAIL_CLOSED + emergency close |
| Fill confirmed | `status == FILLED` | PENDING (poll again) |

## 2.8 Bracket Exit Reconciliation

When bracket TP/SL fills on Coinbase, the engine must:
1. Detect position is flat (broker-as-truth)
2. Query attached orders to find which filled
3. Query fills API for actual exit price/fees
4. Finalize trade via **single finalize_exit() path**

**All exits call one function:**
- Signal exit (Chandelier)
- Bracket TP filled
- Bracket SL filled
- Manual exit
- Emergency close

## 2.9 Duplicate Prevention (client_order_id)

**Format:** `ANX4_{run8}_{symbol}_{ts}_{hash8}`

**Check order:**
1. Local state (in_flight_order)
2. Database (orders.client_order_id)
3. Coinbase API (list_orders)

**On restart:** Query Coinbase for orders with `ANX4_{run8}_` prefix, hydrate state.

## 2.10 Single-Loop Architecture

```python
while not shutdown:
    # 1. Safety first
    engine.reconcile_position()

    # 2. Advance pending orders
    if engine.state == ENTRY_PENDING:
        check_entry_order()

    # 3. Strategy only on new candle
    if new_candle_detected():  # Data-driven, not wall-clock
        engine.run_tick(data)

    sleep(30)  # Watcher interval
```

---

# Part 3: Schema Changes Summary

## New Tables

```sql
CREATE TABLE runs (...);  -- See 1.4
```

## Altered Tables

```sql
-- decisions
ADD COLUMN run_id TEXT;

-- orders
ADD COLUMN run_id TEXT;
ADD COLUMN client_order_id TEXT;
ADD COLUMN order_state TEXT;
ADD COLUMN attached_order_ids TEXT;
ADD COLUMN bracket_verified BOOLEAN;

-- trades
ADD COLUMN run_id TEXT;
ADD COLUMN entry_client_order_id TEXT;
ADD COLUMN bracket_tp_order_id TEXT;
ADD COLUMN bracket_sl_order_id TEXT;
```

## New Indexes

```sql
CREATE INDEX idx_decisions_run_id ON decisions(run_id);
CREATE INDEX idx_orders_run_id ON orders(run_id);
CREATE INDEX idx_trades_run_id ON trades(run_id);
CREATE UNIQUE INDEX idx_orders_client_order_id ON orders(client_order_id);
```

---

# Part 4: Files to Create/Modify

## Replay Runner

| File | Action |
|------|--------|
| `src/replay/__init__.py` | CREATE |
| `src/replay/data_loader.py` | CREATE |
| `src/replay/runner.py` | CREATE |
| `src/replay/report.py` | CREATE |
| `scripts/replay_runner.py` | CREATE |
| `src/engine.py` | MODIFY - add `as_of` parameter |
| `src/truth/logger.py` | MODIFY - remove `utcnow()` fallbacks |
| `src/truth/schema.py` | MODIFY - add runs table |

## Bracket Orders

| File | Action |
|------|--------|
| `src/execution/bracket.py` | CREATE |
| `src/execution/bracket_config.py` | CREATE |
| `src/execution/bracket_executor.py` | CREATE |
| `src/execution/product_specs.py` | CREATE |
| `src/execution/schema.py` | MODIFY - add ExecutionMode.BRACKET_ATOMIC |
| `src/engine.py` | MODIFY - state machine, sizing integration |

---

# Part 5: Open Questions (To Confirm)

1. **5% stop buffer:** Hypothesis based on Coinbase docs. Confirm with small live test.
2. **Attached order visibility:** Does create response include attached IDs, or must we poll?
3. **Fills API pagination:** For exit P&L, need to handle cursor if many fills.

---

# Appendix: Devil's Advocate Summary

| Risk | Mitigation |
|------|------------|
| Replay with bad data = fake validation | Coinbase API only, volume validation |
| Bracket legs missing after fill = naked | Verification + emergency close |
| Restart duplicates orders | client_order_id idempotency |
| Wall-clock timestamps in replay | `as_of` threading, no `utcnow()` fallbacks |
| Multiple finalize paths drift | Single `finalize_exit()` for all exit types |
| Thread concurrency bugs | Single-loop architecture |
| 5% buffer is wrong | Treat as hypothesis, size conservatively |
