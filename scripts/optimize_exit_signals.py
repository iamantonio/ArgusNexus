#!/usr/bin/env python3
"""
Optimize exit signals to reduce drawdown in bear markets.
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
    exit_momentum_period: int = 30,
    exit_momentum_threshold: float = -15,
    reentry_sma: int = 30,
    position_size: float = 0.90,
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
):
    """Run the MostlyLong strategy with configurable parameters."""
    min_bars = max(exit_sma, exit_momentum_period, reentry_sma) + 5

    if len(df) < min_bars + 50:
        return None

    sma_exit = calc_sma(df["close"], exit_sma)
    sma_reentry = calc_sma(df["close"], reentry_sma)
    momentum = calc_momentum(df["close"], exit_momentum_period)

    capital = initial_capital
    position = None
    trades = []
    equity_curve = []

    # Start with position
    entry_price = float(df["close"].iloc[min_bars]) * (1 + fees["slippage"])
    alloc = capital * position_size
    qty = alloc / (entry_price * (1 + fees["entry"]))
    position = {"entry_price": entry_price, "qty": qty}

    for i in range(min_bars, len(df)):
        close = float(df["close"].iloc[i])
        current_sma_exit = sma_exit.iloc[i]
        current_sma_reentry = sma_reentry.iloc[i]
        current_mom = momentum.iloc[i]

        if position is not None:
            # Exit condition
            if close < current_sma_exit and current_mom < exit_momentum_threshold:
                fill_price = close * (1 - fees["slippage"])
                gross_pnl = (fill_price - position["entry_price"]) * position["qty"]
                entry_fee = position["entry_price"] * position["qty"] * fees["entry"]
                exit_fee = fill_price * position["qty"] * fees["exit"]
                net_pnl = gross_pnl - entry_fee - exit_fee
                trades.append({"net_pnl": net_pnl})
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
        "trades": len(trades)
    }


async def main():
    print("="*70)
    print("EXIT SIGNAL OPTIMIZATION")
    print("="*70)

    # Load full history
    print("\nLoading BTC data 2022-2025...")
    df_full = await fetch_ohlcv("BTC-USD", "2022-01-01", "2025-12-17")
    print(f"Loaded {len(df_full)} bars")

    # Load 2024-2025 for recent performance check
    df_recent = df_full[df_full.index >= pd.Timestamp("2024-01-01")]
    print(f"Recent period (2024-2025): {len(df_recent)} bars")

    # Grid search for optimal parameters
    print("\n" + "="*70)
    print("PARAMETER GRID SEARCH")
    print("="*70)
    print(f"{'Exit SMA':<10} {'Mom Period':<12} {'Mom Thresh':<12} {'Full Return':>12} {'Full DD':>10} {'Recent Alpha':>14}")
    print("-"*70)

    results = []
    for exit_sma in [50, 100, 150, 200]:
        for mom_period in [14, 21, 30]:
            for mom_thresh in [-10, -15, -20]:
                full_result = run_mostly_long(df_full, exit_sma, mom_period, mom_thresh, 30)
                recent_result = run_mostly_long(df_recent, exit_sma, mom_period, mom_thresh, 30)

                if full_result and recent_result:
                    results.append({
                        "exit_sma": exit_sma,
                        "mom_period": mom_period,
                        "mom_thresh": mom_thresh,
                        "full_return": full_result["return"],
                        "full_dd": full_result["max_dd"],
                        "full_alpha": full_result["alpha"],
                        "recent_return": recent_result["return"],
                        "recent_dd": recent_result["max_dd"],
                        "recent_alpha": recent_result["alpha"],
                        "trades": full_result["trades"]
                    })

                    print(f"{exit_sma:<10} {mom_period:<12} {mom_thresh:<12} {full_result['return']:>+11.1f}% {full_result['max_dd']:>9.1f}% {recent_result['alpha']:>+13.1f}%")

    # Find best configs
    print("\n" + "="*70)
    print("TOP CONFIGURATIONS (by Max DD)")
    print("="*70)

    # Filter for max DD under 30% and sort by recent alpha
    valid = [r for r in results if r["full_dd"] <= 30]
    valid.sort(key=lambda x: x["recent_alpha"], reverse=True)

    print(f"{'Config':<25} {'Full Return':>12} {'Full DD':>10} {'Recent Alpha':>14} {'Trades':>8}")
    print("-"*70)

    for r in valid[:10]:
        config = f"SMA{r['exit_sma']}_M{r['mom_period']}_T{r['mom_thresh']}"
        print(f"{config:<25} {r['full_return']:>+11.1f}% {r['full_dd']:>9.1f}% {r['recent_alpha']:>+13.1f}% {r['trades']:>8}")

    # Best config analysis
    if valid:
        best = valid[0]
        print("\n" + "="*70)
        print(f"BEST CONFIG: SMA{best['exit_sma']}_M{best['mom_period']}_T{best['mom_thresh']}")
        print("="*70)
        print(f"\nFull Period (2022-2025):")
        print(f"  Return: {best['full_return']:+.1f}%")
        print(f"  Max DD: {best['full_dd']:.1f}%")
        print(f"  Alpha vs BTC: {best['full_alpha']:+.1f}%")
        print(f"\nRecent Period (2024-2025):")
        print(f"  Return: {best['recent_return']:+.1f}%")
        print(f"  Max DD: {best['recent_dd']:.1f}%")
        print(f"  Alpha vs BTC: {best['recent_alpha']:+.1f}%")
        print(f"\nTotal Trades: {best['trades']}")

        # Criteria check
        print("\n" + "="*70)
        print("CRITERIA CHECK")
        print("="*70)

        criteria = []
        if best["full_dd"] <= 25:
            criteria.append(("Max DD <= 25%", True, f"{best['full_dd']:.1f}%"))
        else:
            criteria.append(("Max DD <= 25%", False, f"{best['full_dd']:.1f}%"))

        if best["recent_alpha"] > 0:
            criteria.append(("Beats BTC (2024-2025)", True, f"{best['recent_alpha']:+.1f}%"))
        else:
            criteria.append(("Beats BTC (2024-2025)", False, f"{best['recent_alpha']:+.1f}%"))

        if best["full_return"] > 0:
            criteria.append(("Positive full-period return", True, f"{best['full_return']:+.1f}%"))
        else:
            criteria.append(("Positive full-period return", False, f"{best['full_return']:+.1f}%"))

        for name, passed, value in criteria:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
