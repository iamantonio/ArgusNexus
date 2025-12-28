#!/usr/bin/env python3
"""
Verify the final portfolio strategy configuration.

Configuration:
- DD Warning: 15%
- DD Critical: 22%
- Vol Target: 40%
- Regime-based allocation
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict


GEMINI_FEES = {"entry": 0.004, "exit": 0.002, "slippage": 0.001}
GEMINI_FEES_2X = {"entry": 0.008, "exit": 0.004, "slippage": 0.002}


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
    return returns.rolling(window=period).std() * np.sqrt(252) * 100


def calc_momentum(series: pd.Series, period: int) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


def run_final_strategy(
    df: pd.DataFrame,
    warning_dd: float = 15.0,
    critical_dd: float = 22.0,
    vol_target: float = 40.0,
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> Dict:
    """Run the final validated strategy."""
    min_bars = 205

    if len(df) < min_bars + 50:
        return None

    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['sma_30'] = calc_sma(df['close'], 30)
    df['sma_50'] = calc_sma(df['close'], 50)
    df['sma_200'] = calc_sma(df['close'], 200)
    df['momentum_30'] = calc_momentum(df['close'], 30)
    df['realized_vol'] = calc_realized_vol(df['returns'], 30)

    capital = initial_capital
    btc_qty = 0.0
    equity_curve = []
    high_water_mark = initial_capital
    trades = 0
    recovery_mode = False

    for i in range(min_bars, len(df)):
        close = float(df['close'].iloc[i])
        sma_50 = float(df['sma_50'].iloc[i])
        sma_200 = float(df['sma_200'].iloc[i])
        momentum = float(df['momentum_30'].iloc[i])
        vol = float(df['realized_vol'].iloc[i])

        btc_value = btc_qty * close
        current_equity = capital + btc_value
        high_water_mark = max(high_water_mark, current_equity)
        current_dd = (1 - current_equity / high_water_mark) * 100

        # DD multiplier
        if current_dd >= critical_dd:
            dd_mult = 0.0
            recovery_mode = True
        elif current_dd >= warning_dd:
            dd_mult = 0.5
        else:
            dd_mult = 1.0

        if recovery_mode:
            if current_dd < 10:
                dd_mult = 1.0
                recovery_mode = False
            elif current_dd < 15:
                dd_mult = 0.5

        # Regime
        if close > sma_200 and momentum > 0:
            base_alloc = 1.0
        elif close < sma_200 and momentum < -10:
            base_alloc = 0.0
        else:
            base_alloc = 0.5

        # Vol scaling
        vol_scalar = vol_target / vol if vol > 0 else 1.0
        vol_scalar = max(0.25, min(1.5, vol_scalar))

        target_alloc = max(0.0, min(1.0, base_alloc * vol_scalar * dd_mult))

        total_value = capital + btc_value
        current_alloc = btc_value / total_value if total_value > 0 else 0

        if abs(target_alloc - current_alloc) > 0.10:
            target_btc = total_value * target_alloc

            if target_btc > btc_value:
                buy = min(target_btc - btc_value, capital)
                if buy > 0:
                    price = close * (1 + fees["slippage"])
                    fee = buy * fees["entry"]
                    btc_qty += (buy - fee) / price
                    capital -= buy
                    trades += 1
            else:
                sell = btc_value - target_btc
                if sell > 0 and btc_qty > 0:
                    qty = min(sell / close, btc_qty)
                    val = qty * close
                    fee = val * fees["exit"]
                    capital += val - fee
                    btc_qty -= qty
                    trades += 1

        equity_curve.append(capital + btc_qty * close)

    equity = pd.Series(equity_curve)
    final = equity.iloc[-1]
    ret = (final / initial_capital - 1) * 100
    dd = float(((equity.cummax() - equity) / equity.cummax() * 100).max())
    rets = equity.pct_change().dropna()
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() > 0 else 0
    btc_ret = (float(df['close'].iloc[-1]) / float(df['close'].iloc[min_bars]) - 1) * 100

    return {
        "return": ret,
        "max_dd": dd,
        "sharpe": sharpe,
        "btc_return": btc_ret,
        "alpha": ret - btc_ret,
        "trades": trades,
        "final": final
    }


async def main():
    print("="*70)
    print("FINAL STRATEGY VALIDATION")
    print("="*70)
    print("\nConfiguration:")
    print("  DD Warning: 15%")
    print("  DD Critical: 22%")
    print("  Vol Target: 40%")
    print("  Regime: Bull/Bear/Sideways based on SMA200 + Momentum")

    df = await fetch_ohlcv("BTC-USD", "2022-01-01", "2025-12-17")
    print(f"\nLoaded {len(df)} bars")

    # Walk-forward analysis
    print("\n" + "="*70)
    print("WALK-FORWARD ANALYSIS")
    print("="*70)

    periods = [
        ("2022 Bear Market", "2022-01-01", "2022-12-31"),
        ("2023 Recovery", "2023-01-01", "2023-12-31"),
        ("2024-2025 Bull", "2024-01-01", "2025-12-17"),
        ("Full 2022-2025", "2022-01-01", "2025-12-17"),
    ]

    print(f"{'Period':<20} {'Return':>10} {'BTC B&H':>10} {'Alpha':>10} {'MaxDD':>8} {'Sharpe':>8} {'Pass':>6}")
    print("-"*75)

    all_pass = True
    for name, start, end in periods:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        period_df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if len(period_df) < 210:
            print(f"{name:<20} INSUFFICIENT DATA")
            continue

        result = run_final_strategy(period_df)
        if result:
            # Pass: positive return OR (positive alpha with DD < 25%)
            passes = (result['return'] > 0 and result['max_dd'] <= 25) or \
                     (result['alpha'] > 0 and result['max_dd'] < 22)
            status = "YES" if passes else "NO"
            if not passes:
                all_pass = False

            print(f"{name:<20} {result['return']:>+9.1f}% {result['btc_return']:>+9.1f}% {result['alpha']:>+9.1f}% {result['max_dd']:>7.1f}% {result['sharpe']:>7.2f} {status:>6}")

    # Stress test with 2x fees
    print("\n" + "="*70)
    print("STRESS TEST (2x Fees)")
    print("="*70)

    for name, start, end in periods:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        period_df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if len(period_df) < 210:
            continue

        result = run_final_strategy(period_df, fees=GEMINI_FEES_2X)
        if result:
            print(f"{name:<20} {result['return']:>+9.1f}% MaxDD: {result['max_dd']:>6.1f}%")

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    full_result = run_final_strategy(df)
    recent_df = df[df.index >= pd.Timestamp("2024-01-01")]
    recent_result = run_final_strategy(recent_df)

    print(f"\nFull Period (2022-2025):")
    print(f"  Return: {full_result['return']:+.1f}%")
    print(f"  Max DD: {full_result['max_dd']:.1f}%")
    print(f"  Sharpe: {full_result['sharpe']:.2f}")
    print(f"  Alpha: {full_result['alpha']:+.1f}%")
    print(f"  Trades: {full_result['trades']}")

    print(f"\nRecent Period (2024-2025):")
    print(f"  Return: {recent_result['return']:+.1f}%")
    print(f"  Max DD: {recent_result['max_dd']:.1f}%")
    print(f"  Alpha: {recent_result['alpha']:+.1f}%")

    # Criteria check
    criteria = []

    if full_result['max_dd'] <= 25:
        criteria.append((True, f"Max DD under 25%: {full_result['max_dd']:.1f}%"))
    else:
        criteria.append((False, f"Max DD over 25%: {full_result['max_dd']:.1f}%"))

    if full_result['return'] > 0:
        criteria.append((True, f"Positive return: {full_result['return']:+.1f}%"))
    else:
        criteria.append((False, f"Negative return: {full_result['return']:+.1f}%"))

    if recent_result['alpha'] > 0 or recent_result['max_dd'] < 20:
        criteria.append((True, f"Recent alpha or lower risk: alpha={recent_result['alpha']:+.1f}%, DD={recent_result['max_dd']:.1f}%"))
    else:
        criteria.append((False, f"Doesn't beat BTC recently: alpha={recent_result['alpha']:+.1f}%"))

    print("\nCriteria:")
    all_criteria_pass = True
    for passed, msg in criteria:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_criteria_pass = False
        print(f"  [{status}] {msg}")

    if all_criteria_pass:
        print("\n" + "="*70)
        print("STRATEGY VALIDATED FOR DEPLOYMENT")
        print("="*70)
        print("""
PORTFOLIO STRATEGY: Vol-Targeting Regime Rotation with DD Limiter

Rules:
1. REGIME DETECTION (Daily):
   - BULL: Price > 200 SMA AND 30-day momentum > 0% -> 100% BTC target
   - BEAR: Price < 200 SMA AND 30-day momentum < -10% -> 0% BTC target
   - SIDEWAYS: Otherwise -> 50% BTC target

2. VOLATILITY SCALING:
   - Scale position by (40% / realized_vol)
   - Cap between 0.25x and 1.5x

3. DRAWDOWN CIRCUIT BREAKER:
   - WARNING (DD >= 15%): Reduce exposure to 50%
   - CRITICAL (DD >= 22%): Reduce to 0% (all stables)
   - RECOVERY: Gradually re-enter when DD < 10%

4. REBALANCE: When allocation drifts >10% from target

Fee Model: Gemini (0.4% entry, 0.2% exit, 0.1% slippage)
        """)
    else:
        print("\n*** STRATEGY NEEDS MORE WORK ***")


if __name__ == "__main__":
    asyncio.run(main())
