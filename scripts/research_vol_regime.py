#!/usr/bin/env python3
"""
Research: Volatility Targeting + Regime-Based Exposure

Goal: Enforce 25% max portfolio drawdown by dynamically adjusting
BTC exposure based on:
1. Realized volatility (scale inversely)
2. Market regime (bull/bear/sideways)

Portfolio Structure:
- MostlyLong BTC Sleeve: 15% (fixed, uses its own signals)
- Dynamic Sleeve: 85% (this research) - rotates BTC/stables

The dynamic sleeve aims to:
- Be fully invested in BTC during low-vol bull markets
- Reduce to 50% BTC during high-vol periods
- Reduce to 0% BTC (100% stables) during bear regime
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass


GEMINI_FEES = {"entry": 0.004, "exit": 0.002, "slippage": 0.001}


async def fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    time_window = 300 * 86400
    all_candles = []
    current_end = end_ts

    async with aiohttp.ClientSession() as session:
        while current_end > start_ts:
            params = {"start": str(max(start_ts, current_end - time_window)), "end": str(current_end), "granularity": "ONE_DAY", "limit": "300"}
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    break
                data = await response.json()
                candles = data.get("candles", [])
                if not candles:
                    break
                all_candles.extend(candles)
                current_end = int(candles[-1]["start"]) - 1
                await asyncio.sleep(0.1)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def calc_realized_vol(returns: pd.Series, period: int = 30) -> pd.Series:
    """Realized volatility (annualized)"""
    return returns.rolling(window=period).std() * np.sqrt(252) * 100


def calc_momentum(series: pd.Series, period: int) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


@dataclass
class RegimeState:
    """Market regime classification"""
    regime: str  # "bull", "bear", "sideways"
    vol_regime: str  # "low", "normal", "high"
    btc_allocation: float  # 0.0 to 1.0
    reason: str


def classify_regime(
    price: float,
    sma_50: float,
    sma_200: float,
    momentum_30: float,
    realized_vol: float,
    vol_target: float = 50.0  # Target annual vol
) -> RegimeState:
    """
    Classify market regime and determine BTC allocation.

    Regime Rules:
    - BULL: Price > SMA200 AND momentum > 0
    - BEAR: Price < SMA200 AND momentum < -10%
    - SIDEWAYS: Everything else

    Vol-Adjusted Allocation:
    - Base allocation from regime
    - Scale by vol_target / realized_vol (capped)
    """
    # Regime classification
    if price > sma_200 and momentum_30 > 0:
        regime = "bull"
        base_alloc = 1.0  # 100% BTC in bull
    elif price < sma_200 and momentum_30 < -10:
        regime = "bear"
        base_alloc = 0.0  # 0% BTC in bear (all stables)
    else:
        regime = "sideways"
        base_alloc = 0.5  # 50% BTC in sideways

    # Volatility scaling
    if realized_vol > 0:
        vol_scalar = vol_target / realized_vol
        vol_scalar = max(0.25, min(1.5, vol_scalar))  # Cap between 0.25x and 1.5x
    else:
        vol_scalar = 1.0

    # Vol regime
    if realized_vol < 40:
        vol_regime = "low"
    elif realized_vol > 80:
        vol_regime = "high"
    else:
        vol_regime = "normal"

    # Final allocation
    final_alloc = base_alloc * vol_scalar
    final_alloc = max(0.0, min(1.0, final_alloc))  # Clamp to 0-100%

    reason = f"{regime.upper()} regime, {vol_regime} vol ({realized_vol:.1f}%), vol_scalar={vol_scalar:.2f}"

    return RegimeState(
        regime=regime,
        vol_regime=vol_regime,
        btc_allocation=final_alloc,
        reason=reason
    )


def run_vol_regime_strategy(
    df: pd.DataFrame,
    vol_target: float = 50.0,
    rebalance_threshold: float = 0.10,  # Rebalance if allocation drifts >10%
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> Dict:
    """
    Run volatility-targeting regime-based strategy.

    This manages the "dynamic sleeve" (85% of portfolio).
    The remaining 15% is managed by MostlyLong BTC strategy.
    """
    min_bars = 205  # Need 200 SMA

    if len(df) < min_bars + 50:
        return None

    # Calculate indicators
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['sma_50'] = calc_sma(df['close'], 50)
    df['sma_200'] = calc_sma(df['close'], 200)
    df['momentum_30'] = calc_momentum(df['close'], 30)
    df['realized_vol'] = calc_realized_vol(df['returns'], 30)

    capital = initial_capital
    btc_qty = 0.0
    current_alloc = 0.0
    trades = []
    equity_curve = []
    regime_history = []

    for i in range(min_bars, len(df)):
        close = float(df['close'].iloc[i])
        sma_50 = float(df['sma_50'].iloc[i])
        sma_200 = float(df['sma_200'].iloc[i])
        momentum = float(df['momentum_30'].iloc[i])
        vol = float(df['realized_vol'].iloc[i])

        # Get regime state
        state = classify_regime(close, sma_50, sma_200, momentum, vol, vol_target)
        regime_history.append({
            'date': df.index[i],
            'regime': state.regime,
            'vol_regime': state.vol_regime,
            'target_alloc': state.btc_allocation
        })

        # Calculate current allocation
        btc_value = btc_qty * close
        total_value = capital + btc_value
        current_alloc = btc_value / total_value if total_value > 0 else 0

        # Check if rebalance needed
        alloc_diff = abs(state.btc_allocation - current_alloc)

        if alloc_diff > rebalance_threshold:
            target_btc_value = total_value * state.btc_allocation

            if target_btc_value > btc_value:
                # Buy more BTC
                buy_amount = target_btc_value - btc_value
                buy_price = close * (1 + fees["slippage"])
                fee = buy_amount * fees["entry"]
                btc_bought = (buy_amount - fee) / buy_price
                btc_qty += btc_bought
                capital -= buy_amount
                trades.append({
                    "action": "BUY",
                    "amount": buy_amount,
                    "price": buy_price,
                    "reason": state.reason
                })
            else:
                # Sell BTC
                sell_btc_value = btc_value - target_btc_value
                sell_price = close * (1 - fees["slippage"])
                sell_qty = sell_btc_value / close
                fee = sell_btc_value * fees["exit"]
                capital += sell_btc_value - fee
                btc_qty -= sell_qty
                trades.append({
                    "action": "SELL",
                    "amount": sell_btc_value,
                    "price": sell_price,
                    "reason": state.reason
                })

        # Track equity
        equity = capital + (btc_qty * close)
        equity_curve.append(equity)

    # Final equity
    final_close = float(df['close'].iloc[-1])
    final_equity = capital + (btc_qty * final_close)

    # Metrics
    equity_series = pd.Series(equity_curve)
    max_dd = float(((equity_series.cummax() - equity_series) / equity_series.cummax() * 100).max())
    returns = equity_series.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0

    # BTC return for comparison
    btc_return = (float(df['close'].iloc[-1]) / float(df['close'].iloc[min_bars]) - 1) * 100

    total_return = (final_equity / initial_capital - 1) * 100

    # Regime breakdown
    regime_df = pd.DataFrame(regime_history)
    regime_counts = regime_df['regime'].value_counts()

    return {
        "return": total_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "btc_return": btc_return,
        "alpha": total_return - btc_return,
        "trades": len(trades),
        "final_capital": final_equity,
        "regime_breakdown": regime_counts.to_dict(),
        "avg_allocation": regime_df['target_alloc'].mean() * 100
    }


async def main():
    print("="*70)
    print("VOLATILITY TARGETING + REGIME RESEARCH")
    print("="*70)

    # Load full history
    print("\nLoading BTC data 2022-2025...")
    df = await fetch_ohlcv("BTC-USD", "2022-01-01", "2025-12-17")
    print(f"Loaded {len(df)} bars")

    # Test different vol targets
    print("\n" + "="*70)
    print("VOLATILITY TARGET SWEEP")
    print("="*70)
    print(f"{'Vol Target':<12} {'Return':>10} {'MaxDD':>8} {'Alpha':>10} {'Sharpe':>8} {'Trades':>8} {'Avg Alloc':>10}")
    print("-"*70)

    best_result = None
    best_dd = float('inf')

    for vol_target in [30, 40, 50, 60, 80]:
        result = run_vol_regime_strategy(df, vol_target=vol_target)
        if result:
            print(f"{vol_target}%{'':<9} {result['return']:>+9.1f}% {result['max_dd']:>7.1f}% {result['alpha']:>+9.1f}% {result['sharpe']:>7.2f} {result['trades']:>8} {result['avg_allocation']:>9.1f}%")

            if result['max_dd'] < best_dd and result['return'] > 0:
                best_dd = result['max_dd']
                best_result = result
                best_vol_target = vol_target

    # Test different rebalance thresholds
    print("\n" + "="*70)
    print("REBALANCE THRESHOLD SWEEP (Vol Target = 50%)")
    print("="*70)
    print(f"{'Threshold':<12} {'Return':>10} {'MaxDD':>8} {'Trades':>8}")
    print("-"*45)

    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25]:
        result = run_vol_regime_strategy(df, vol_target=50, rebalance_threshold=threshold)
        if result:
            print(f"{threshold*100:.0f}%{'':<9} {result['return']:>+9.1f}% {result['max_dd']:>7.1f}% {result['trades']:>8}")

    # Test on different periods
    print("\n" + "="*70)
    print("PERIOD ANALYSIS (Vol Target = 50%, Threshold = 10%)")
    print("="*70)

    periods = [
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("2023 Recovery", "2023-01-01", "2023-12-31"),
        ("2024-2025", "2024-01-01", "2025-12-17"),
        ("Full 2022-2025", "2022-01-01", "2025-12-17"),
    ]

    print(f"{'Period':<20} {'Return':>10} {'BTC B&H':>10} {'Alpha':>10} {'MaxDD':>8} {'Regime Breakdown'}")
    print("-"*80)

    for name, start, end in periods:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        period_df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if len(period_df) < 210:
            print(f"{name:<20} INSUFFICIENT DATA")
            continue

        result = run_vol_regime_strategy(period_df, vol_target=50, rebalance_threshold=0.10)
        if result:
            regime_str = ", ".join([f"{k}:{v}" for k, v in result['regime_breakdown'].items()])
            print(f"{name:<20} {result['return']:>+9.1f}% {result['btc_return']:>+9.1f}% {result['alpha']:>+9.1f}% {result['max_dd']:>7.1f}% {regime_str}")

    # Combined portfolio simulation
    print("\n" + "="*70)
    print("COMBINED PORTFOLIO (15% MostlyLong + 85% VolRegime)")
    print("="*70)

    # Simulate combined portfolio
    await simulate_combined_portfolio(df)


async def simulate_combined_portfolio(df: pd.DataFrame):
    """
    Simulate the full portfolio:
    - 15% MostlyLong BTC sleeve (from our validated strategy)
    - 85% Dynamic vol/regime sleeve
    """
    min_bars = 205
    initial_capital = 500.0
    fees = GEMINI_FEES

    # Calculate indicators
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['sma_30'] = calc_sma(df['close'], 30)
    df['sma_50'] = calc_sma(df['close'], 50)
    df['sma_200'] = calc_sma(df['close'], 200)
    df['momentum_30'] = calc_momentum(df['close'], 30)
    df['realized_vol'] = calc_realized_vol(df['returns'], 30)

    # Portfolio state
    # MostlyLong sleeve (15%)
    ml_capital = initial_capital * 0.15
    ml_btc_qty = 0.0
    ml_has_position = False

    # Vol/Regime sleeve (85%)
    vr_capital = initial_capital * 0.85
    vr_btc_qty = 0.0
    vr_current_alloc = 0.0

    equity_curve = []
    vol_target = 50.0
    rebalance_threshold = 0.10

    for i in range(min_bars, len(df)):
        close = float(df['close'].iloc[i])
        sma_30 = float(df['sma_30'].iloc[i])
        sma_50 = float(df['sma_50'].iloc[i])
        sma_200 = float(df['sma_200'].iloc[i])
        momentum = float(df['momentum_30'].iloc[i])
        vol = float(df['realized_vol'].iloc[i])

        # === MostlyLong Sleeve (15%) ===
        # Entry: price > 30 SMA AND momentum > 0
        # Exit: price < 200 SMA AND momentum < -15%

        if not ml_has_position:
            if close > sma_30 and momentum > 0:
                # Enter position
                entry_price = close * (1 + fees["slippage"])
                fee_rate = fees["entry"]
                ml_btc_qty = (ml_capital * 0.90) / (entry_price * (1 + fee_rate))
                ml_has_position = True
        else:
            if close < sma_200 and momentum < -15:
                # Exit position
                exit_price = close * (1 - fees["slippage"])
                gross = ml_btc_qty * exit_price
                fee = gross * fees["exit"]
                ml_capital = ml_capital + gross - fee - (ml_btc_qty * entry_price * fees["entry"])
                ml_btc_qty = 0.0
                ml_has_position = False

        # === Vol/Regime Sleeve (85%) ===
        state = classify_regime(close, sma_50, sma_200, momentum, vol, vol_target)

        vr_btc_value = vr_btc_qty * close
        vr_total = vr_capital + vr_btc_value
        vr_current_alloc = vr_btc_value / vr_total if vr_total > 0 else 0

        alloc_diff = abs(state.btc_allocation - vr_current_alloc)

        if alloc_diff > rebalance_threshold:
            target_btc_value = vr_total * state.btc_allocation

            if target_btc_value > vr_btc_value:
                buy_amount = target_btc_value - vr_btc_value
                buy_price = close * (1 + fees["slippage"])
                fee = buy_amount * fees["entry"]
                vr_btc_qty += (buy_amount - fee) / buy_price
                vr_capital -= buy_amount
            else:
                sell_value = vr_btc_value - target_btc_value
                sell_qty = sell_value / close
                fee = sell_value * fees["exit"]
                vr_capital += sell_value - fee
                vr_btc_qty -= sell_qty

        # Calculate total portfolio equity
        ml_equity = ml_capital + (ml_btc_qty * close) if not ml_has_position else ml_capital + (ml_btc_qty * close)
        vr_equity = vr_capital + (vr_btc_qty * close)
        total_equity = ml_equity + vr_equity
        equity_curve.append(total_equity)

    # Final metrics
    equity_series = pd.Series(equity_curve)
    final_equity = equity_series.iloc[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    max_dd = float(((equity_series.cummax() - equity_series) / equity_series.cummax() * 100).max())
    returns = equity_series.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0
    btc_return = (float(df['close'].iloc[-1]) / float(df['close'].iloc[min_bars]) - 1) * 100

    print(f"\nCombined Portfolio Results (2022-2025):")
    print(f"  Total Return: {total_return:+.1f}%")
    print(f"  Max Drawdown: {max_dd:.1f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  BTC Buy & Hold: {btc_return:+.1f}%")
    print(f"  Alpha: {total_return - btc_return:+.1f}%")
    print(f"  Final Capital: ${final_equity:.2f}")

    # Criteria check
    print("\n" + "="*70)
    print("CRITERIA CHECK")
    print("="*70)

    checks = [
        ("Max DD <= 25%", max_dd <= 25, f"{max_dd:.1f}%"),
        ("Positive Return", total_return > 0, f"{total_return:+.1f}%"),
        ("Sharpe > 0.5", sharpe > 0.5, f"{sharpe:.2f}"),
    ]

    all_pass = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {value}")

    if all_pass:
        print("\n*** ALL CRITERIA MET ***")
    else:
        print("\n*** NEEDS MORE WORK ***")


if __name__ == "__main__":
    asyncio.run(main())
