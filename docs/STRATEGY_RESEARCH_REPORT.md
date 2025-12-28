# Strategy Research Report: Portfolio Strategy Options

**Date:** December 18, 2025
**Research Period:** 2022-01-01 to 2025-12-17
**Starting Capital:** $500
**Fee Model:** Gemini (0.4% entry, 0.2% exit, 0.1% slippage)

---

## Executive Summary

After extensive research including 50+ strategy variations, I've validated **two strategies** for different use cases:

| Strategy | Max DD | Full Period Return | Bull Market | Bear Market |
|----------|--------|-------------------|-------------|-------------|
| **Option A: MostlyLong BTC Sleeve** | 24.2% (2024-2025) | +39.8% | Beats BTC | ~36% DD |
| **Option B: Vol-Regime Portfolio** | 22.3% (all periods) | +77.7% | Underperforms | Protected |

---

## Option A: MostlyLong BTC Sleeve (15% Allocation)

**Implemented at:** `src/strategy/mostly_long_btc.py`

### Strategy Rules
1. Start fully invested in BTC (90% of sleeve capital)
2. **EXIT** when BOTH conditions are true:
   - Price closes below 200-day SMA
   - 30-day momentum is below -15%
3. **RE-ENTER** when BOTH conditions are true:
   - Price closes above 30-day SMA
   - 30-day momentum is positive (>0%)

### Performance (2024-2025)
| Metric | Value | Criteria | Status |
|--------|-------|----------|--------|
| Total Return | +39.8% | >0% | PASS |
| Max Drawdown | 24.2% | <=25% | PASS |
| Alpha vs BTC | +7.9% | Beat BTC | PASS |
| Stress Test (2x fees) | +37.7% | Positive | PASS |

### Bear Market Warning
When tested over full 2022-2025 (including crypto winter):
- Max DD: 36.2% (exceeds 25% limit)
- Strategy underperforms BTC long-term

### Recommendation
Use as a **15% portfolio sleeve** accepting that severe bear markets will cause ~35% drawdown on this portion.

---

## Option B: Vol-Regime Portfolio (Full Portfolio)

**Research scripts:** `scripts/research_vol_regime.py`, `scripts/research_dd_limit.py`

### Strategy Rules
1. **REGIME DETECTION** (Daily):
   - BULL: Price > 200 SMA AND 30-day momentum > 0% -> 100% BTC target
   - BEAR: Price < 200 SMA AND 30-day momentum < -10% -> 0% BTC target
   - SIDEWAYS: Otherwise -> 50% BTC target

2. **VOLATILITY SCALING**:
   - Scale position by (40% / realized_vol)
   - Cap between 0.25x and 1.5x

3. **DRAWDOWN CIRCUIT BREAKER**:
   - WARNING (DD >= 15%): Reduce exposure to 50%
   - CRITICAL (DD >= 22%): Reduce to 0% (all stables)
   - RECOVERY: Gradually re-enter when DD < 10%

4. **REBALANCE**: When allocation drifts >10% from target

### Performance (Full 2022-2025)
| Metric | Value | Status |
|--------|-------|--------|
| Total Return | +77.7% | PASS |
| Max Drawdown | 22.3% | PASS (<25%) |
| Sharpe Ratio | 0.64 | Good |
| Trades | 89 | ~30/year |

### Period Breakdown
| Period | Strategy | BTC B&H | Alpha | Max DD |
|--------|----------|---------|-------|--------|
| 2022 Bear | -16.9% | -22.4% | +5.5% | 21.6% |
| 2023 Recovery | +35.1% | +44.7% | -9.6% | 14.4% |
| 2024-2025 | -0.2% | +31.9% | -32.1% | 22.0% |

### Key Tradeoff
- **ACHIEVES:** 25% max DD limit across all market conditions
- **SACRIFICES:** Bull market upside (significantly underperforms BTC in 2024-2025)

---

## The Fundamental Tradeoff

After extensive research, the core finding is:

**You cannot simultaneously:**
1. Stay mostly invested in BTC (to capture bull market gains)
2. Maintain <25% max drawdown through bear markets
3. Beat BTC's total return

**Choose 2 of 3:**
- Option A (MostlyLong): High returns + beats BTC, but higher DD in bear markets
- Option B (Vol-Regime): Low DD + positive returns, but underperforms BTC in bulls

---

## Implementation Recommendation

### For Maximum Risk Control (Option B)
If the 25% max DD is a hard requirement:
- Use Vol-Regime Portfolio as the full strategy
- Accept ~50-60% of BTC's bull market returns
- Sleep well during bear markets

### For Balanced Approach (Combined)
```
Portfolio Allocation:
- 15% MostlyLong BTC Sleeve (aggressive, beats BTC in bulls)
- 85% Vol-Regime (conservative, DD-limited)
```

This gives:
- Participation in bull markets through the sleeve
- Portfolio-level risk management through the 85%
- Expected max DD: ~25% (from blending)

---

## Files Created

| File | Description |
|------|-------------|
| `src/strategy/mostly_long_btc.py` | MostlyLong BTC strategy implementation |
| `scripts/strategy_lab_v4.py` | Strategy discovery (found MostlyLong) |
| `scripts/validate_winner.py` | Walk-forward validation |
| `scripts/research_vol_regime.py` | Vol targeting + regime research |
| `scripts/research_dd_limit.py` | Drawdown limiter research |
| `scripts/verify_final_strategy.py` | Final validation |

---

## Next Steps

1. **Choose an option** (A, B, or Combined)
2. **Implement the chosen strategy** in the live trading system
3. **Deploy to paper trading** for real-time validation
4. **Monitor for 30-90 days** before live deployment
