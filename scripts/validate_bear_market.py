#!/usr/bin/env python3
"""
Bear Market Stress Test - validate strategy survives 2022-2023 crypto winter.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict


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


def calc_momentum(series: pd.Series, period: int) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


def run_mostly_long(
    df: pd.DataFrame,
    exit_sma: int = 200,
    exit_momentum: int = 30,
    reentry_sma: int = 30,
    position_size: float = 0.90,
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
):
    """Run the MostlyLong strategy."""
    min_bars = max(exit_sma, exit_momentum, reentry_sma) + 5

    if len(df) < min_bars + 50:
        return None

    sma_exit = calc_sma(df["close"], exit_sma)
    sma_reentry = calc_sma(df["close"], reentry_sma)
    momentum = calc_momentum(df["close"], exit_momentum)

    capital = initial_capital
    position = None
    trades = []
    equity_curve = []

    # Start with position
    entry_price = float(df["close"].iloc[min_bars]) * (1 + fees["slippage"])
    alloc = capital * position_size
    qty = alloc / (entry_price * (1 + fees["entry"]))
    position = {"entry_price": entry_price, "qty": qty, "entry_time": df.index[min_bars]}

    for i in range(min_bars, len(df)):
        close = float(df["close"].iloc[i])
        current_sma_exit = sma_exit.iloc[i]
        current_sma_reentry = sma_reentry.iloc[i]
        current_mom = momentum.iloc[i]

        if position is not None:
            # Exit condition
            if close < current_sma_exit and current_mom < -15:
                fill_price = close * (1 - fees["slippage"])
                gross_pnl = (fill_price - position["entry_price"]) * position["qty"]
                entry_fee = position["entry_price"] * position["qty"] * fees["entry"]
                exit_fee = fill_price * position["qty"] * fees["exit"]
                net_pnl = gross_pnl - entry_fee - exit_fee
                trades.append({"net_pnl": net_pnl, "exit_price": fill_price, "entry_price": position["entry_price"]})
                capital += net_pnl
                position = None
        else:
            # Re-entry condition
            if close > current_sma_reentry and current_mom > 0:
                alloc = capital * position_size
                entry_price = close * (1 + fees["slippage"])
                qty = alloc / (entry_price * (1 + fees["entry"]))
                position = {"entry_price": entry_price, "qty": qty}

        # Track equity
        if position:
            equity = (capital - position["entry_price"] * position["qty"]) + (close * position["qty"])
        else:
            equity = capital
        equity_curve.append(equity)

    # Close remaining
    if position:
        close = float(df["close"].iloc[-1])
        fill_price = close * (1 - fees["slippage"])
        gross_pnl = (fill_price - position["entry_price"]) * position["qty"]
        entry_fee = position["entry_price"] * position["qty"] * fees["entry"]
        exit_fee = fill_price * position["qty"] * fees["exit"]
        net_pnl = gross_pnl - entry_fee - exit_fee
        trades.append({"net_pnl": net_pnl})
        capital += net_pnl

    # BTC return
    btc_return = (float(df["close"].iloc[-1]) / float(df["close"].iloc[min_bars]) - 1) * 100

    # Metrics
    total_return = (capital / initial_capital - 1) * 100
    equity = pd.Series(equity_curve)
    max_dd = float(((equity.cummax() - equity) / equity.cummax() * 100).max())
    returns = equity.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0

    return {
        "return": total_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "btc_return": btc_return,
        "alpha": total_return - btc_return,
        "trades": len(trades),
        "final_capital": capital
    }


async def main():
    print("="*70)
    print("BEAR MARKET STRESS TEST: 2022-2023 Crypto Winter")
    print("="*70)

    # Load bear market data
    print("\nLoading BTC data 2022-2023...")
    df = await fetch_ohlcv("BTC-USD", "2022-01-01", "2023-12-31")
    print(f"Loaded {len(df)} bars")

    # Test periods
    periods = [
        ("2022 Bear Market", "2022-01-01", "2022-12-31"),
        ("2023 Recovery", "2023-01-01", "2023-12-31"),
        ("Full 2022-2023", "2022-01-01", "2023-12-31"),
        ("Crash Period (May-Nov 2022)", "2022-05-01", "2022-11-30"),
    ]

    print("\n" + "="*70)
    print("BEAR MARKET PERFORMANCE")
    print("="*70)
    print(f"{'Period':<30} {'Return':>10} {'BTC B&H':>10} {'Alpha':>10} {'MaxDD':>8} {'Trades':>7}")
    print("-"*75)

    for name, start, end in periods:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        period_df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if len(period_df) < 210:
            print(f"{name:<30} INSUFFICIENT DATA ({len(period_df)} bars)")
            continue

        result = run_mostly_long(period_df, 200, 30, 30, 0.90, GEMINI_FEES)

        if result:
            print(f"{name:<30} {result['return']:>+9.1f}% {result['btc_return']:>+9.1f}% {result['alpha']:>+9.1f}% {result['max_dd']:>7.1f}% {result['trades']:>7}")

    # Now test full history 2022-2025
    print("\n" + "="*70)
    print("FULL HISTORY TEST: 2022-2025")
    print("="*70)

    df_full = await fetch_ohlcv("BTC-USD", "2022-01-01", "2025-12-17")
    print(f"Loaded {len(df_full)} bars (full history)")

    result = run_mostly_long(df_full, 200, 30, 30, 0.90, GEMINI_FEES)
    if result:
        print(f"\nFull 2022-2025 Results:")
        print(f"  Return: {result['return']:+.1f}%")
        print(f"  BTC B&H: {result['btc_return']:+.1f}%")
        print(f"  Alpha: {result['alpha']:+.1f}%")
        print(f"  Max Drawdown: {result['max_dd']:.1f}%")
        print(f"  Sharpe: {result['sharpe']:.2f}")
        print(f"  Trades: {result['trades']}")
        print(f"  Final Capital: ${result['final_capital']:.2f}")

    # Verdict
    print("\n" + "="*70)
    print("BEAR MARKET VERDICT")
    print("="*70)

    if result:
        passed = []
        failed = []

        if result["max_dd"] <= 50:
            passed.append(f"Survived bear market: {result['max_dd']:.1f}% max DD (limit: 50%)")
        else:
            failed.append(f"Too much drawdown: {result['max_dd']:.1f}%")

        if result["alpha"] > 0:
            passed.append(f"Beat buy-and-hold: {result['alpha']:+.1f}% alpha")
        else:
            failed.append(f"Underperformed: {result['alpha']:+.1f}% alpha")

        if result["return"] > -30:
            passed.append(f"Preserved capital: {result['return']:+.1f}% (vs -30% threshold)")
        else:
            failed.append(f"Too much loss: {result['return']:+.1f}%")

        print("\nPASSED:")
        for p in passed:
            print(f"  + {p}")

        if failed:
            print("\nFAILED:")
            for f in failed:
                print(f"  - {f}")

        if not failed:
            print("\n*** STRATEGY SURVIVES BEAR MARKET ***")


if __name__ == "__main__":
    asyncio.run(main())
