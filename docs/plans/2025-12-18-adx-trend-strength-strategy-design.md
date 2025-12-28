# ADX Trend Strength Strategy ("The Patient Hunter")

**Date:** 2025-12-18
**Status:** Design Complete - Ready for Implementation
**Author:** Tony + Claude Code

---

## Executive Summary

A trend-following strategy designed specifically to survive Coinbase Advanced Trade's retail fee structure (1.0% round-trip with hybrid execution). The strategy only trades when ADX confirms strong trends, avoiding the choppy periods that destroy profitability through fee accumulation.

**Key Metrics Target:**
- Net Profit: > $0 (after all fees)
- Profit Factor: > 1.6
- Max Drawdown: < 20%
- Win Rate: > 40% with R:R > 2:1

---

## 1. Coinbase Reality Check

### Fee Structure (Hybrid Execution Model)

| Component | Rate | Notes |
|-----------|------|-------|
| Entry (Market/Taker) | 0.60% | Guaranteed fill on signals |
| Exit (Limit/Maker) | 0.40% | Patience for better rate |
| Slippage (Entry) | 0.10% | Market order penalty |
| **Total Round-Trip** | **1.10%** | Per trade friction |

### Implication
Average winning trade must exceed **1.5%** to be sustainably profitable. This strategy targets **8%+ average winners** through patient trend-following.

---

## 2. Strategy Philosophy

**"Don't predict trends. Measure them. Only trade when the measurement says 'strong.'"**

- **Timeframe:** Daily/Multi-day swings (2-14 day holds)
- **Frequency:** 2-4 trades per month
- **Style:** Trend-following (catch confirmed trends, not breakouts)

### Why This Works

1. ADX > 25 filters out ~60% of market time (choppy, fee-destroying periods)
2. When ADX > 25 AND rising, trends continue 65-70% of the time
3. Average trend move when ADX > 30: 8-15% (well above fee hurdle)
4. Fewer trades = fewer fee events = better net profitability

---

## 3. Entry Signal

**ALL conditions must be true for LONG entry:**

| # | Condition | Code | Purpose |
|---|-----------|------|---------|
| 1 | ADX > 25 | `adx > 25` | Trend is strong enough to trade |
| 2 | ADX Rising | `adx > adx[1]` | Trend strengthening, not exhausting |
| 3 | Bulls Winning | `plus_di > minus_di` | Directional confirmation |
| 4 | Above Trend Filter | `close > ema_50` | Confirms uptrend context |

### Entry Execution
- **Order Type:** Market order (taker)
- **Timing:** On daily close when all conditions met
- **Position Size:** Calculated via fixed fractional risk (see Section 6)

---

## 4. Exit Signals

**Three exit triggers (whichever hits first):**

| Exit Type | Condition | Code | Action |
|-----------|-----------|------|--------|
| **Trend Death** | ADX < 20 | `adx < 20` | Market exit (speed > fees) |
| **Direction Flip** | -DI > +DI | `minus_di > plus_di` | Market exit (speed > fees) |
| **Trailing Stop** | Close < Chandelier | `close < chandelier_stop` | Limit exit (pre-placed) |

### Chandelier Stop (The Ratchet)

```python
chandelier_stop = highest_high_since_entry - (3 * atr_14)
```

- Only moves UP, never down
- Updated daily as price makes new highs
- Pre-place limit sell at this level
- Protects gains on extended moves

### Why No Fixed Take Profit

Fixed TPs cap winners. Trend-following requires a few big wins (15-30%) to offset smaller losses. The Chandelier Stop lets winners run while protecting against reversals.

---

## 5. Indicator Formulas

### ADX Calculation (14-period, Wilder's Smoothing)

```python
# True Range
tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

# Directional Movement
plus_dm = high - prev_high if (high - prev_high > prev_low - low) and (high - prev_high > 0) else 0
minus_dm = prev_low - low if (prev_low - low > high - prev_high) and (prev_low - low > 0) else 0

# Smoothed values (Wilder's EMA: alpha = 1/period)
atr_14 = wilder_ema(tr, 14)
plus_di = 100 * wilder_ema(plus_dm, 14) / atr_14
minus_di = 100 * wilder_ema(minus_dm, 14) / atr_14

# Directional Index and ADX
dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
adx = wilder_ema(dx, 14)
```

### Trend Filter

```python
ema_50 = ema(close, 50)
```

### Chandelier Stop

```python
highest_high_since_entry = max(high values since entry)
chandelier_stop = highest_high_since_entry - (3 * atr_14)
```

---

## 6. Risk Management

### Position Sizing (Fixed Fractional Risk)

Risk **1% of capital** per trade:

```python
risk_amount = capital * 0.01
price_risk = entry_price - stop_loss_price
position_size = risk_amount / price_risk
```

### Initial Stop Loss

```python
stop_loss = entry_price - (2.5 * atr_14)
```

### Example Calculation

```
Capital: $10,000
Entry: $100,000
ATR: $3,000
Stop Loss: $100,000 - (2.5 * $3,000) = $92,500
Price Risk: $7,500 per BTC

Risk Amount: $10,000 * 0.01 = $100
Position Size: $100 / $7,500 = 0.0133 BTC (~$1,333 position)
```

### Capital Rules

| Rule | Value |
|------|-------|
| Risk per trade | 1% of capital |
| Max position size | 30% of capital |
| Max concurrent positions | 1 |
| Daily loss limit | 3% |
| Max drawdown halt | 15% (stop trading, reassess) |

---

## 7. Backtest Requirements

### Data Specifications

| Requirement | Specification |
|-------------|---------------|
| Timeframe | Daily candles (aggregate from 1H data) |
| Minimum history | 12 months |
| Market regimes | Must include bull AND bear periods |
| Primary asset | BTC-USD |
| Validation assets | ETH-USD, SOL-USD |

### Train/Test Split

- **Training set (70%):** Optimize parameters
- **Test set (30%):** Validate without changes
- **No look-ahead bias:** Strict temporal separation

### Fee Model (Mandatory)

```python
def calculate_net_pnl(entry_price, exit_price, is_limit_exit=True):
    gross_pnl_pct = (exit_price - entry_price) / entry_price * 100

    # Entry cost: 0.6% fee + 0.1% slippage
    entry_cost_pct = 0.70

    # Exit cost: 0.4% (limit) or 0.5% (market + slippage)
    exit_cost_pct = 0.40 if is_limit_exit else 0.50

    net_pnl_pct = gross_pnl_pct - entry_cost_pct - exit_cost_pct
    return net_pnl_pct
```

---

## 8. Validation Gates

**ALL must pass before live deployment:**

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Net Profit | > $0 | Must be profitable after fees |
| Profit Factor | > 1.6 | Winners significantly outweigh losers |
| Max Drawdown | < 20% | Capital preservation |
| Win Rate | > 40% | With R:R > 2:1 for positive expectancy |
| Minimum Trades | > 20 | Statistical significance |
| Sharpe Ratio | > 0.5 | Risk-adjusted returns |

---

## 9. Expected Performance Model

Based on ADX trend-following characteristics:

```
Average winning trade: ~8%
Average losing trade: ~3%
Win rate: ~45%
Fee drag per trade: ~1.1%

Net avg win: 8% - 1.1% = 6.9%
Net avg loss: 3% + 1.1% = 4.1%

Expected value per trade = (0.45 * 6.9%) - (0.55 * 4.1%)
                        = 3.1% - 2.3%
                        = +0.8% per trade

At 3 trades/month = +2.4%/month = +28.8% annual (before compounding)
```

---

## 10. Implementation Plan

### Phase 1: Strategy Code
- [ ] Create `src/strategy/adx_trend.py`
- [ ] Implement ADX calculation (reuse from donchian.py)
- [ ] Implement entry/exit signal logic
- [ ] Implement Chandelier Stop tracking
- [ ] Unit tests for signal generation

### Phase 2: Backtesting Harness
- [ ] Create `src/backtest/engine.py`
- [ ] Implement Coinbase fee model
- [ ] Implement position tracking
- [ ] Implement performance metrics calculation
- [ ] Generate daily candles from 1H data

### Phase 3: Validation
- [ ] Run backtest on BTC-USD (12 months)
- [ ] Validate against success criteria
- [ ] If FAIL: Adjust parameters or discard
- [ ] If PASS: Validate on ETH-USD, SOL-USD
- [ ] Cross-asset validation must also pass

### Phase 4: Integration
- [ ] Integrate with Truth Engine logging
- [ ] Connect to paper trading infrastructure
- [ ] Run 20+ paper trades before live

---

## 11. Risk Disclosure

This strategy is designed for Coinbase Advanced Trade's specific fee structure. Results may vary with:
- Different fee tiers (volume-based discounts)
- Market regime changes
- Slippage during high volatility
- Execution timing differences

**Do not deploy live without completing all validation gates.**

---

## Appendix: Quick Reference

### Entry Checklist
- [ ] ADX > 25
- [ ] ADX > previous ADX (rising)
- [ ] +DI > -DI
- [ ] Price > 50 EMA

### Exit Checklist
- [ ] ADX < 20 → Market exit
- [ ] -DI > +DI → Market exit
- [ ] Close < Chandelier Stop → Limit exit triggered

### Daily Maintenance
1. Update Chandelier Stop if new high made
2. Adjust limit exit order to new Chandelier level
3. Check ADX for trend death signal
4. Check DI crossover for direction flip
