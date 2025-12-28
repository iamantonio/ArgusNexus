#!/usr/bin/env python3
"""
Backtest Combined Portfolio Strategy

Uses the Portfolio Manager to combine:
- Vol-Regime Core (85%)
- MostlyLong Sleeve (15%)

With portfolio-level DD circuit breaker (15%/22%/10%/15%).

Validates:
- Max DD <= 25%
- Positive return
- Sensible rebalance frequency
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio.portfolio_manager import (
    PortfolioManager,
    PortfolioState,
    DDState
)


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


def run_backtest(
    df: pd.DataFrame,
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> Dict:
    """
    Run backtest using the Portfolio Manager.
    """
    min_bars = 205

    if len(df) < min_bars + 50:
        return None

    # Initialize Portfolio Manager
    pm = PortfolioManager()

    # Initialize state
    state = PortfolioState(
        total_equity=Decimal(str(initial_capital)),
        btc_qty=Decimal("0"),
        cash=Decimal(str(initial_capital)),
        high_water_mark=Decimal(str(initial_capital)),
        dd_state=DDState.NORMAL,
        recovery_mode=False,
        sleeve_in_position=False,
        sleeve_entry_price=None,
        last_update=None
    )

    equity_curve = []
    dd_curve = []
    trade_log = []
    dd_state_history = []

    for i in range(min_bars, len(df)):
        # Get historical data up to current bar (no look-ahead)
        hist_df = df.iloc[:i+1]
        current_price = Decimal(str(hist_df.iloc[-1]['close']))
        current_time = hist_df.index[-1]

        # Evaluate portfolio
        order, new_state = pm.evaluate(
            hist_df,
            state,
            current_price,
            timestamp=current_time
        )

        # Execute order if needed
        if order.action != "HOLD":
            new_state = pm.execute_order(order, new_state, current_price, fees)
            trade_log.append({
                "timestamp": current_time,
                "action": order.action,
                "btc_delta": float(order.btc_qty_delta),
                "target_alloc": float(order.target_alloc_pct),
                "reason": order.reason,
                "dd_state": new_state.dd_state.value
            })

        # Update equity
        btc_value = new_state.btc_qty * current_price
        equity = new_state.cash + btc_value
        new_state.total_equity = equity

        equity_curve.append(float(equity))
        dd_curve.append(float(order.context.get("current_dd", 0)))
        dd_state_history.append(new_state.dd_state.value)

        state = new_state

    # Calculate metrics
    equity_series = pd.Series(equity_curve)
    final_equity = equity_series.iloc[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    max_dd = float(((equity_series.cummax() - equity_series) / equity_series.cummax() * 100).max())

    returns = equity_series.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0

    btc_return = (float(df['close'].iloc[-1]) / float(df['close'].iloc[min_bars]) - 1) * 100

    # DD state breakdown
    dd_states = pd.Series(dd_state_history)
    state_counts = dd_states.value_counts().to_dict()

    return {
        "return": total_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "btc_return": btc_return,
        "alpha": total_return - btc_return,
        "trades": len(trade_log),
        "final_capital": final_equity,
        "dd_state_breakdown": state_counts,
        "trade_log": trade_log[-10:]  # Last 10 trades for inspection
    }


async def main():
    print("="*70)
    print("COMBINED PORTFOLIO BACKTEST")
    print("="*70)

    # Create PM to show actual config
    pm_test = PortfolioManager()
    print("\nConfiguration (from PortfolioManager):")
    print(f"  - Core weight: {pm_test.core_weight}")
    print(f"  - Sleeve weight: {pm_test.sleeve_weight}")
    print(f"  - DD Warning: {pm_test.dd_warning}%")
    print(f"  - DD Critical: {pm_test.dd_critical}%")
    print(f"  - Rebalance threshold: {pm_test.rebalance_threshold}")

    # Load data
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

    print(f"{'Period':<20} {'Return':>10} {'BTC B&H':>10} {'Alpha':>10} {'MaxDD':>8} {'Sharpe':>8} {'Trades':>8}")
    print("-"*80)

    for name, start, end in periods:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        period_df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if len(period_df) < 210:
            print(f"{name:<20} INSUFFICIENT DATA")
            continue

        result = run_backtest(period_df)
        if result:
            print(f"{name:<20} {result['return']:>+9.1f}% {result['btc_return']:>+9.1f}% {result['alpha']:>+9.1f}% {result['max_dd']:>7.1f}% {result['sharpe']:>7.2f} {result['trades']:>8}")

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

        result = run_backtest(period_df, fees=GEMINI_FEES_2X)
        if result:
            print(f"{name:<20} {result['return']:>+9.1f}% MaxDD: {result['max_dd']:>6.1f}%")

    # Full period details
    print("\n" + "="*70)
    print("FULL PERIOD DETAILS (2022-2025)")
    print("="*70)

    result = run_backtest(df)
    if result:
        print(f"\nPerformance:")
        print(f"  Total Return: {result['return']:+.1f}%")
        print(f"  Max Drawdown: {result['max_dd']:.1f}%")
        print(f"  Sharpe Ratio: {result['sharpe']:.2f}")
        print(f"  BTC Buy & Hold: {result['btc_return']:+.1f}%")
        print(f"  Alpha: {result['alpha']:+.1f}%")
        print(f"  Total Trades: {result['trades']}")
        print(f"  Final Capital: ${result['final_capital']:.2f}")

        print(f"\nDD State Breakdown:")
        for state, count in result['dd_state_breakdown'].items():
            pct = count / sum(result['dd_state_breakdown'].values()) * 100
            print(f"  {state}: {count} bars ({pct:.1f}%)")

        print(f"\nLast 10 Trades:")
        for trade in result['trade_log']:
            print(f"  {trade['timestamp'].strftime('%Y-%m-%d')}: {trade['action']} "
                  f"target={trade['target_alloc']:.1f}% ({trade['dd_state']})")

    # Criteria check
    print("\n" + "="*70)
    print("CRITERIA CHECK")
    print("="*70)

    checks = [
        ("Max DD <= 25%", result['max_dd'] <= 25, f"{result['max_dd']:.1f}%"),
        ("Positive Return", result['return'] > 0, f"{result['return']:+.1f}%"),
        ("Sharpe > 0.5", result['sharpe'] > 0.5, f"{result['sharpe']:.2f}"),
        ("Trades per year < 50", result['trades'] / 3 < 50, f"{result['trades']/3:.0f}/year"),
    ]

    all_pass = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {value}")

    if all_pass:
        print("\n" + "="*70)
        print("PORTFOLIO STRATEGY VALIDATED")
        print("="*70)
    else:
        print("\n*** NEEDS ADJUSTMENT ***")


if __name__ == "__main__":
    asyncio.run(main())
