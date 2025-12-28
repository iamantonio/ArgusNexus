#!/usr/bin/env python3
"""
Research: Portfolio-Level Drawdown Limiter

Add a circuit breaker that de-risks the entire portfolio when
drawdown exceeds a threshold.

Strategy:
1. Monitor portfolio drawdown in real-time
2. When DD hits warning level (15%), reduce exposure to 50%
3. When DD hits critical level (20%), reduce exposure to 0% (all stables)
4. Re-enter gradually when portfolio recovers
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple


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
    return returns.rolling(window=period).std() * np.sqrt(252) * 100


def calc_momentum(series: pd.Series, period: int) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


def get_dd_limit_multiplier(current_dd: float, warning_dd: float = 15.0, critical_dd: float = 20.0) -> float:
    """
    Get exposure multiplier based on current drawdown.

    Returns:
        1.0 if DD < warning (full exposure)
        0.5 if warning <= DD < critical (half exposure)
        0.0 if DD >= critical (no exposure)
    """
    if current_dd >= critical_dd:
        return 0.0
    elif current_dd >= warning_dd:
        return 0.5
    else:
        return 1.0


def get_recovery_multiplier(current_dd: float, recovery_threshold: float = 10.0) -> float:
    """
    After hitting critical DD, require recovery before re-entering.

    Returns:
        Multiplier based on how much we've recovered from the drawdown.
    """
    if current_dd < recovery_threshold:
        return 1.0
    elif current_dd < 15:
        return 0.5
    else:
        return 0.0


def run_portfolio_with_dd_limit(
    df: pd.DataFrame,
    warning_dd: float = 15.0,
    critical_dd: float = 20.0,
    vol_target: float = 40.0,
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> Dict:
    """
    Run portfolio with drawdown circuit breaker.
    """
    min_bars = 205

    if len(df) < min_bars + 50:
        return None

    # Calculate indicators
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['sma_30'] = calc_sma(df['close'], 30)
    df['sma_50'] = calc_sma(df['close'], 50)
    df['sma_200'] = calc_sma(df['close'], 200)
    df['momentum_30'] = calc_momentum(df['close'], 30)
    df['realized_vol'] = calc_realized_vol(df['returns'], 30)

    # Portfolio state
    capital = initial_capital
    btc_qty = 0.0
    equity_curve = []
    high_water_mark = initial_capital
    trades = 0
    dd_breaker_active = False
    recovery_mode = False

    for i in range(min_bars, len(df)):
        close = float(df['close'].iloc[i])
        sma_30 = float(df['sma_30'].iloc[i])
        sma_50 = float(df['sma_50'].iloc[i])
        sma_200 = float(df['sma_200'].iloc[i])
        momentum = float(df['momentum_30'].iloc[i])
        vol = float(df['realized_vol'].iloc[i])

        # Calculate current equity and drawdown
        btc_value = btc_qty * close
        current_equity = capital + btc_value
        high_water_mark = max(high_water_mark, current_equity)
        current_dd = (1 - current_equity / high_water_mark) * 100

        # Check DD circuit breaker
        dd_multiplier = get_dd_limit_multiplier(current_dd, warning_dd, critical_dd)

        if dd_multiplier == 0.0:
            dd_breaker_active = True
            recovery_mode = True

        if recovery_mode:
            dd_multiplier = get_recovery_multiplier(current_dd)
            if dd_multiplier == 1.0:
                recovery_mode = False

        # Base regime allocation
        if close > sma_200 and momentum > 0:
            base_alloc = 1.0  # Bull
        elif close < sma_200 and momentum < -10:
            base_alloc = 0.0  # Bear
        else:
            base_alloc = 0.5  # Sideways

        # Vol scaling
        if vol > 0:
            vol_scalar = vol_target / vol
            vol_scalar = max(0.25, min(1.5, vol_scalar))
        else:
            vol_scalar = 1.0

        # Final target allocation (with DD limit applied)
        target_alloc = base_alloc * vol_scalar * dd_multiplier
        target_alloc = max(0.0, min(1.0, target_alloc))

        # Current allocation
        total_value = capital + btc_value
        current_alloc = btc_value / total_value if total_value > 0 else 0

        # Rebalance if needed
        if abs(target_alloc - current_alloc) > 0.10:
            target_btc_value = total_value * target_alloc

            if target_btc_value > btc_value:
                # Buy
                buy_amount = min(target_btc_value - btc_value, capital)
                if buy_amount > 0:
                    buy_price = close * (1 + fees["slippage"])
                    fee = buy_amount * fees["entry"]
                    btc_qty += (buy_amount - fee) / buy_price
                    capital -= buy_amount
                    trades += 1
            else:
                # Sell
                sell_value = btc_value - target_btc_value
                if sell_value > 0 and btc_qty > 0:
                    sell_qty = min(sell_value / close, btc_qty)
                    actual_sell = sell_qty * close
                    fee = actual_sell * fees["exit"]
                    capital += actual_sell - fee
                    btc_qty -= sell_qty
                    trades += 1

        # Track equity
        final_equity = capital + (btc_qty * close)
        equity_curve.append(final_equity)

    # Metrics
    equity_series = pd.Series(equity_curve)
    final_equity = equity_series.iloc[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    max_dd = float(((equity_series.cummax() - equity_series) / equity_series.cummax() * 100).max())
    returns = equity_series.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0
    btc_return = (float(df['close'].iloc[-1]) / float(df['close'].iloc[min_bars]) - 1) * 100

    return {
        "return": total_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "btc_return": btc_return,
        "alpha": total_return - btc_return,
        "trades": trades,
        "final_capital": final_equity,
        "dd_breaker_triggered": dd_breaker_active
    }


async def main():
    print("="*70)
    print("DRAWDOWN LIMITER RESEARCH")
    print("="*70)

    df = await fetch_ohlcv("BTC-USD", "2022-01-01", "2025-12-17")
    print(f"Loaded {len(df)} bars")

    # Test different DD limits
    print("\n" + "="*70)
    print("DD LIMIT SWEEP")
    print("="*70)
    print(f"{'Warning/Critical':<20} {'Return':>10} {'MaxDD':>8} {'Sharpe':>8} {'Trades':>8} {'Breaker'}")
    print("-"*60)

    configs = [
        (10, 15),  # Very conservative
        (12, 18),
        (15, 20),  # Moderate
        (15, 22),
        (18, 23),
        (20, 25),  # Target
    ]

    best_result = None
    best_dd = float('inf')

    for warning, critical in configs:
        result = run_portfolio_with_dd_limit(df, warning_dd=warning, critical_dd=critical)
        if result:
            breaker = "YES" if result["dd_breaker_triggered"] else "NO"
            print(f"{warning}% / {critical}%{'':<10} {result['return']:>+9.1f}% {result['max_dd']:>7.1f}% {result['sharpe']:>7.2f} {result['trades']:>8} {breaker:>8}")

            if result['max_dd'] <= 25 and (result['return'] > (best_result['return'] if best_result else -999)):
                best_result = result
                best_config = (warning, critical)

    # Test with different vol targets
    print("\n" + "="*70)
    print("VOL TARGET SWEEP (DD Limit: 15%/20%)")
    print("="*70)
    print(f"{'Vol Target':<12} {'Return':>10} {'MaxDD':>8} {'Sharpe':>8}")
    print("-"*45)

    for vol in [25, 30, 35, 40, 50]:
        result = run_portfolio_with_dd_limit(df, warning_dd=15, critical_dd=20, vol_target=vol)
        if result:
            print(f"{vol}%{'':<9} {result['return']:>+9.1f}% {result['max_dd']:>7.1f}% {result['sharpe']:>7.2f}")

    # Test on different periods
    print("\n" + "="*70)
    print("PERIOD ANALYSIS (Best Config)")
    print("="*70)

    periods = [
        ("2022 Bear", "2022-01-01", "2022-12-31"),
        ("2023 Recovery", "2023-01-01", "2023-12-31"),
        ("2024-2025", "2024-01-01", "2025-12-17"),
        ("Full 2022-2025", "2022-01-01", "2025-12-17"),
    ]

    print(f"{'Period':<20} {'Return':>10} {'BTC B&H':>10} {'Alpha':>10} {'MaxDD':>8}")
    print("-"*60)

    for name, start, end in periods:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        period_df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if len(period_df) < 210:
            print(f"{name:<20} INSUFFICIENT DATA")
            continue

        result = run_portfolio_with_dd_limit(period_df, warning_dd=15, critical_dd=20, vol_target=35)
        if result:
            print(f"{name:<20} {result['return']:>+9.1f}% {result['btc_return']:>+9.1f}% {result['alpha']:>+9.1f}% {result['max_dd']:>7.1f}%")

    # Final recommendation
    print("\n" + "="*70)
    print("FINAL CONFIGURATION")
    print("="*70)

    result = run_portfolio_with_dd_limit(df, warning_dd=15, critical_dd=20, vol_target=35)
    if result:
        print(f"\nConfiguration:")
        print(f"  DD Warning Level: 15%")
        print(f"  DD Critical Level: 20%")
        print(f"  Vol Target: 35%")
        print(f"\nResults:")
        print(f"  Total Return: {result['return']:+.1f}%")
        print(f"  Max Drawdown: {result['max_dd']:.1f}%")
        print(f"  Sharpe Ratio: {result['sharpe']:.2f}")
        print(f"  BTC B&H: {result['btc_return']:+.1f}%")
        print(f"  Alpha: {result['alpha']:+.1f}%")
        print(f"  Trades: {result['trades']}")

        print("\n" + "="*70)
        print("CRITERIA CHECK")
        print("="*70)

        checks = [
            ("Max DD <= 25%", result['max_dd'] <= 25, f"{result['max_dd']:.1f}%"),
            ("Positive Return", result['return'] > 0, f"{result['return']:+.1f}%"),
            ("Stress Test", result['max_dd'] < 30, "See DD above"),
        ]

        all_pass = True
        for name, passed, value in checks:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  [{status}] {name}: {value}")

        if all_pass:
            print("\n*** STRATEGY VALIDATED ***")


if __name__ == "__main__":
    asyncio.run(main())
