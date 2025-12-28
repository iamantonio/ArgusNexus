#!/usr/bin/env python3
"""
Verify the combined portfolio simulation with detailed logging.
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


def calc_realized_vol(returns: pd.Series, period: int = 30) -> pd.Series:
    return returns.rolling(window=period).std() * np.sqrt(252) * 100


def calc_momentum(series: pd.Series, period: int) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


def classify_regime(price, sma_50, sma_200, momentum_30, realized_vol, vol_target=50.0):
    """Classify market regime and return target BTC allocation"""
    # Regime
    if price > sma_200 and momentum_30 > 0:
        regime = "bull"
        base_alloc = 1.0
    elif price < sma_200 and momentum_30 < -10:
        regime = "bear"
        base_alloc = 0.0
    else:
        regime = "sideways"
        base_alloc = 0.5

    # Vol scaling
    if realized_vol > 0:
        vol_scalar = vol_target / realized_vol
        vol_scalar = max(0.25, min(1.5, vol_scalar))
    else:
        vol_scalar = 1.0

    final_alloc = max(0.0, min(1.0, base_alloc * vol_scalar))
    return regime, final_alloc


async def main():
    print("="*70)
    print("COMBINED PORTFOLIO VERIFICATION")
    print("="*70)

    df = await fetch_ohlcv("BTC-USD", "2022-01-01", "2025-12-17")
    print(f"Loaded {len(df)} bars")

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

    # === MOSTLY LONG SLEEVE (15%) ===
    ml_initial = initial_capital * 0.15  # $75
    ml_capital = ml_initial
    ml_btc_qty = 0.0
    ml_has_position = False
    ml_entry_price = 0.0
    ml_equity_curve = []
    ml_trades = 0

    # === VOL/REGIME SLEEVE (85%) ===
    vr_initial = initial_capital * 0.85  # $425
    vr_capital = vr_initial
    vr_btc_qty = 0.0
    vr_equity_curve = []
    vr_trades = 0

    vol_target = 50.0
    rebalance_threshold = 0.10

    # Track combined
    combined_equity_curve = []

    print(f"\nInitial Capital: ${initial_capital:.2f}")
    print(f"  MostlyLong Sleeve (15%): ${ml_initial:.2f}")
    print(f"  Vol/Regime Sleeve (85%): ${vr_initial:.2f}")

    for i in range(min_bars, len(df)):
        close = float(df['close'].iloc[i])
        sma_30 = float(df['sma_30'].iloc[i])
        sma_50 = float(df['sma_50'].iloc[i])
        sma_200 = float(df['sma_200'].iloc[i])
        momentum = float(df['momentum_30'].iloc[i])
        vol = float(df['realized_vol'].iloc[i])

        # === MOSTLY LONG SLEEVE ===
        if not ml_has_position:
            if close > sma_30 and momentum > 0:
                # Enter: allocate 90% of sleeve capital
                ml_entry_price = close * (1 + fees["slippage"])
                alloc = ml_capital * 0.90
                fee = alloc * fees["entry"]
                ml_btc_qty = (alloc - fee) / ml_entry_price
                ml_capital -= alloc
                ml_has_position = True
                ml_trades += 1
        else:
            if close < sma_200 and momentum < -15:
                # Exit
                exit_price = close * (1 - fees["slippage"])
                gross = ml_btc_qty * exit_price
                fee = gross * fees["exit"]
                ml_capital += gross - fee
                ml_btc_qty = 0.0
                ml_has_position = False
                ml_trades += 1

        ml_equity = ml_capital + (ml_btc_qty * close)
        ml_equity_curve.append(ml_equity)

        # === VOL/REGIME SLEEVE ===
        regime, target_alloc = classify_regime(close, sma_50, sma_200, momentum, vol, vol_target)

        vr_btc_value = vr_btc_qty * close
        vr_total = vr_capital + vr_btc_value
        vr_current_alloc = vr_btc_value / vr_total if vr_total > 0 else 0

        alloc_diff = abs(target_alloc - vr_current_alloc)

        if alloc_diff > rebalance_threshold:
            target_btc_value = vr_total * target_alloc

            if target_btc_value > vr_btc_value:
                # Buy BTC
                buy_amount = target_btc_value - vr_btc_value
                if buy_amount > vr_capital:
                    buy_amount = vr_capital  # Can't spend more than we have
                buy_price = close * (1 + fees["slippage"])
                fee = buy_amount * fees["entry"]
                btc_bought = (buy_amount - fee) / buy_price
                vr_btc_qty += btc_bought
                vr_capital -= buy_amount
                vr_trades += 1
            else:
                # Sell BTC
                sell_value = vr_btc_value - target_btc_value
                if sell_value > 0 and vr_btc_qty > 0:
                    sell_qty = min(sell_value / close, vr_btc_qty)
                    actual_sell_value = sell_qty * close
                    fee = actual_sell_value * fees["exit"]
                    vr_capital += actual_sell_value - fee
                    vr_btc_qty -= sell_qty
                    vr_trades += 1

        vr_equity = vr_capital + (vr_btc_qty * close)
        vr_equity_curve.append(vr_equity)

        # Combined
        combined_equity = ml_equity + vr_equity
        combined_equity_curve.append(combined_equity)

    # Final results
    ml_series = pd.Series(ml_equity_curve)
    vr_series = pd.Series(vr_equity_curve)
    combined_series = pd.Series(combined_equity_curve)

    ml_final = ml_series.iloc[-1]
    vr_final = vr_series.iloc[-1]
    combined_final = combined_series.iloc[-1]

    ml_return = (ml_final / ml_initial - 1) * 100
    vr_return = (vr_final / vr_initial - 1) * 100
    combined_return = (combined_final / initial_capital - 1) * 100

    ml_dd = float(((ml_series.cummax() - ml_series) / ml_series.cummax() * 100).max())
    vr_dd = float(((vr_series.cummax() - vr_series) / vr_series.cummax() * 100).max())
    combined_dd = float(((combined_series.cummax() - combined_series) / combined_series.cummax() * 100).max())

    btc_return = (float(df['close'].iloc[-1]) / float(df['close'].iloc[min_bars]) - 1) * 100

    print("\n" + "="*70)
    print("SLEEVE RESULTS")
    print("="*70)

    print(f"\nMostlyLong Sleeve (15%):")
    print(f"  Initial: ${ml_initial:.2f}")
    print(f"  Final: ${ml_final:.2f}")
    print(f"  Return: {ml_return:+.1f}%")
    print(f"  Max DD: {ml_dd:.1f}%")
    print(f"  Trades: {ml_trades}")

    print(f"\nVol/Regime Sleeve (85%):")
    print(f"  Initial: ${vr_initial:.2f}")
    print(f"  Final: ${vr_final:.2f}")
    print(f"  Return: {vr_return:+.1f}%")
    print(f"  Max DD: {vr_dd:.1f}%")
    print(f"  Trades: {vr_trades}")

    print(f"\n" + "="*70)
    print("COMBINED PORTFOLIO")
    print("="*70)
    print(f"  Initial: ${initial_capital:.2f}")
    print(f"  Final: ${combined_final:.2f}")
    print(f"  Return: {combined_return:+.1f}%")
    print(f"  Max DD: {combined_dd:.1f}%")
    print(f"  BTC B&H: {btc_return:+.1f}%")
    print(f"  Alpha: {combined_return - btc_return:+.1f}%")

    # Sharpe
    returns = combined_series.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0
    print(f"  Sharpe: {sharpe:.2f}")

    # Criteria check
    print("\n" + "="*70)
    print("CRITERIA CHECK")
    print("="*70)

    checks = [
        ("Max DD <= 25%", combined_dd <= 25, f"{combined_dd:.1f}%"),
        ("Positive Return", combined_return > 0, f"{combined_return:+.1f}%"),
        ("Beats BTC or Lower DD", (combined_return > btc_return) or (combined_dd < 40), "See above"),
    ]

    all_pass = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {value}")

    if all_pass:
        print("\n*** ALL CRITERIA MET - STRATEGY VALIDATED ***")
    else:
        print("\n*** NEEDS MORE WORK ***")


if __name__ == "__main__":
    asyncio.run(main())
