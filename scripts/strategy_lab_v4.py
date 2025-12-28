#!/usr/bin/env python3
"""
Strategy Lab V4 - "Mostly Long" Approach

Key insight: BTC +100% over period means sitting in cash kills returns.
Try staying invested with only emergency exits.

Strategies:
1. BTC Buy-and-Hold (baseline)
2. Mostly Long - only exit on extreme trend breakdown
3. Risk-Parity Multi-Asset - stay invested, rotate based on risk
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
INITIAL_CAPITAL = 500.0

GEMINI_FEES = {"entry": 0.004, "exit": 0.002, "slippage": 0.001}
GEMINI_FEES_2X = {"entry": 0.008, "exit": 0.004, "slippage": 0.002}


# =============================================================================
# DATA FETCHING
# =============================================================================

async def fetch_ohlcv(symbol: str, start_date: str, end_date: str, granularity: str = "ONE_DAY") -> pd.DataFrame:
    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    time_window = 300 * 86400
    all_candles = []
    current_end = end_ts

    async with aiohttp.ClientSession() as session:
        while current_end > start_ts:
            params = {"start": str(max(start_ts, current_end - time_window)), "end": str(current_end), "granularity": granularity, "limit": "300"}
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


async def load_all_data(assets: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    data = {}
    for symbol in assets:
        print(f"  Loading {symbol}...", end=" ")
        df = await fetch_ohlcv(symbol, start_date, end_date)
        if not df.empty:
            data[symbol] = df
            print(f"{len(df)} bars")
        else:
            print("FAILED")
    return data


# =============================================================================
# INDICATORS
# =============================================================================

def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_momentum(series: pd.Series, period: int) -> pd.Series:
    return (series / series.shift(period) - 1) * 100

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# =============================================================================
# RESULT CLASS
# =============================================================================

@dataclass
class BacktestResult:
    strategy_name: str
    trades: List[dict]
    equity_curve: List[float]
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_fees: float
    btc_buy_hold_return: float
    alpha_vs_btc: float
    time_in_market_pct: float

    def passes_criteria(self) -> Tuple[bool, List[str]]:
        failures = []
        if self.max_drawdown_pct > 25.0:
            failures.append(f"Drawdown {self.max_drawdown_pct:.1f}% > 25% limit")
        if self.total_return_pct <= 0:
            failures.append(f"Negative return")

        # Criteria: Beat BTC OR (DD < 15% AND return > BTC*0.5)
        beats_btc = self.total_return_pct >= self.btc_buy_hold_return
        acceptable_tradeoff = (self.max_drawdown_pct < 15.0 and
                              self.total_return_pct > self.btc_buy_hold_return * 0.5)

        if not beats_btc and not acceptable_tradeoff:
            failures.append(f"Doesn't beat BTC or have acceptable risk/return tradeoff")

        return len(failures) == 0, failures

    def summary(self) -> str:
        passed, failures = self.passes_criteria()
        status = "PASS" if passed else "FAIL"
        return f"""
{'='*70}
{self.strategy_name} | {status}
{'='*70}
Return: {self.total_return_pct:+.2f}% | BTC B&H: {self.btc_buy_hold_return:+.2f}% | Alpha: {self.alpha_vs_btc:+.2f}%
Trades: {self.total_trades} | Win Rate: {self.win_rate:.1f}% | PF: {self.profit_factor:.2f}
Max DD: {self.max_drawdown_pct:.1f}% | Sharpe: {self.sharpe_ratio:.2f} | Time in Market: {self.time_in_market_pct:.1f}%
Fees: ${self.total_fees:.2f}
{'-'*70}
{chr(10).join(failures) if failures else 'ALL CRITERIA PASSED'}
{'='*70}
"""


# =============================================================================
# STRATEGY 1: BTC Buy and Hold (Baseline)
# =============================================================================

def run_btc_buy_hold(
    data: Dict[str, pd.DataFrame],
    position_size: float = 0.95,
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> BacktestResult:
    """Simple buy and hold."""
    df = data.get("BTC-USD")
    if df is None:
        return None

    # Buy on first day
    capital = initial_capital
    alloc = capital * position_size
    entry_price = float(df["close"].iloc[0]) * (1 + fees["slippage"])
    qty = alloc / (entry_price * (1 + fees["entry"]))
    entry_fee = entry_price * qty * fees["entry"]

    # Track equity
    equity_curve = []
    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        equity = (capital - alloc) + (price * qty)
        equity_curve.append(equity)

    # Final value
    exit_price = float(df["close"].iloc[-1]) * (1 - fees["slippage"])
    exit_fee = exit_price * qty * fees["exit"]
    final_value = (capital - alloc) + (exit_price * qty) - exit_fee

    total_return = (final_value / initial_capital - 1) * 100
    total_fees = entry_fee + exit_fee

    equity = pd.Series(equity_curve)
    max_dd = float(((equity.cummax() - equity) / equity.cummax() * 100).max())
    returns = equity.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0

    return BacktestResult(
        strategy_name="BTC_BuyHold",
        trades=[{"type": "buy_hold"}],
        equity_curve=equity_curve,
        total_return_pct=total_return,
        total_trades=1,
        win_rate=100.0 if total_return > 0 else 0.0,
        profit_factor=float('inf') if total_return > 0 else 0.0,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        total_fees=total_fees,
        btc_buy_hold_return=total_return,
        alpha_vs_btc=0.0,
        time_in_market_pct=100.0
    )


# =============================================================================
# STRATEGY 2: Mostly Long BTC with Emergency Exit
# =============================================================================

def run_mostly_long(
    data: Dict[str, pd.DataFrame],
    symbol: str = "BTC-USD",
    exit_sma: int = 100,  # Exit only below long SMA
    exit_momentum: int = 30,  # AND negative momentum
    reentry_sma: int = 50,  # Re-enter above shorter SMA
    position_size: float = 0.90,
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> BacktestResult:
    """
    Mostly Long: Stay invested unless extreme conditions.

    Exit only when: price < 100 SMA AND 30-day momentum < -15%
    Re-enter when: price > 50 SMA
    """
    name = f"MostlyLong_{symbol}_Exit{exit_sma}_Reentry{reentry_sma}"

    df = data.get(symbol)
    if df is None:
        return None

    min_bars = max(exit_sma, exit_momentum, reentry_sma) + 5

    sma_exit = calc_sma(df["close"], exit_sma)
    sma_reentry = calc_sma(df["close"], reentry_sma)
    momentum = calc_momentum(df["close"], exit_momentum)

    capital = initial_capital
    position = None
    trades = []
    equity_curve = []
    days_in_market = 0

    # Start with a position
    entry_price = float(df["close"].iloc[min_bars]) * (1 + fees["slippage"])
    alloc = capital * position_size
    qty = alloc / (entry_price * (1 + fees["entry"]))
    position = {"entry_price": entry_price, "qty": qty, "entry_time": df.index[min_bars]}

    for i in range(min_bars, len(df)):
        date = df.index[i]
        close = float(df["close"].iloc[i])
        current_sma_exit = sma_exit.iloc[i]
        current_sma_reentry = sma_reentry.iloc[i]
        current_mom = momentum.iloc[i]

        if position is not None:
            days_in_market += 1

            # Emergency exit: below long SMA AND strong negative momentum
            if close < current_sma_exit and current_mom < -15:
                fill_price = close * (1 - fees["slippage"])
                gross_pnl = (fill_price - position["entry_price"]) * position["qty"]
                entry_fee = position["entry_price"] * position["qty"] * fees["entry"]
                exit_fee = fill_price * position["qty"] * fees["exit"]
                net_pnl = gross_pnl - entry_fee - exit_fee

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": date,
                    "entry_price": position["entry_price"],
                    "exit_price": fill_price,
                    "net_pnl": net_pnl,
                    "fees": entry_fee + exit_fee,
                    "reason": "emergency_exit"
                })

                capital += net_pnl
                position = None

        else:
            # Re-entry: back above reentry SMA
            if close > current_sma_reentry and current_mom > 0:
                alloc = capital * position_size
                entry_price = close * (1 + fees["slippage"])
                qty = alloc / (entry_price * (1 + fees["entry"]))
                position = {"entry_price": entry_price, "qty": qty, "entry_time": date}

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
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": df.index[-1],
            "entry_price": position["entry_price"],
            "exit_price": fill_price,
            "net_pnl": net_pnl,
            "fees": entry_fee + exit_fee,
            "reason": "end"
        })
        capital += net_pnl

    # BTC return from same start point
    btc_return = (float(df["close"].iloc[-1]) / float(df["close"].iloc[min_bars]) - 1) * 100

    # Metrics
    total_return = (capital / initial_capital - 1) * 100
    total_trades = len(trades)
    winners = len([t for t in trades if t["net_pnl"] > 0])
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_loss = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    equity = pd.Series(equity_curve) if equity_curve else pd.Series([initial_capital])
    max_dd = float(((equity.cummax() - equity) / equity.cummax() * 100).max())
    returns = equity.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0
    total_fees = sum(t["fees"] for t in trades)
    time_in_market = (days_in_market / len(equity_curve) * 100) if equity_curve else 0

    return BacktestResult(
        strategy_name=name,
        trades=trades,
        equity_curve=equity_curve,
        total_return_pct=total_return,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        total_fees=total_fees,
        btc_buy_hold_return=btc_return,
        alpha_vs_btc=total_return - btc_return,
        time_in_market_pct=time_in_market
    )


# =============================================================================
# STRATEGY 3: Risk-Parity Rotation (Always Invested)
# =============================================================================

def run_risk_parity_rotation(
    data: Dict[str, pd.DataFrame],
    vol_period: int = 20,
    rebalance_days: int = 14,
    total_allocation: float = 0.80,  # 80% in crypto
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> BacktestResult:
    """
    Risk-Parity: Always invested, allocate inversely to volatility.

    - Calculate 20-day realized vol for each asset
    - Weight inversely to volatility (lower vol = higher weight)
    - Rebalance every 2 weeks
    """
    name = f"RiskParity_V{vol_period}_R{rebalance_days}"

    all_dates = set()
    for df in data.values():
        all_dates.update(df.index.tolist())
    dates = sorted(all_dates)

    min_bars = vol_period + 10

    # Precompute volatilities
    vol_data = {}
    for symbol, df in data.items():
        vol_data[symbol] = df["close"].pct_change().rolling(vol_period).std() * np.sqrt(252)

    capital = initial_capital
    cash = initial_capital
    positions: Dict[str, dict] = {}  # symbol -> {qty, entry_price}
    trades = []
    equity_curve = []
    last_rebalance = None
    days_in_market = 0

    for date in dates[min_bars:]:
        should_rebalance = last_rebalance is None or (date - last_rebalance).days >= rebalance_days

        if should_rebalance:
            # Calculate target weights (inverse vol)
            vols = {}
            for symbol, df in data.items():
                if date not in df.index or date not in vol_data[symbol].index:
                    continue
                v = vol_data[symbol].loc[date]
                if pd.notna(v) and v > 0:
                    vols[symbol] = v

            if not vols:
                continue

            # Inverse vol weights
            inv_vols = {s: 1/v for s, v in vols.items()}
            total_inv = sum(inv_vols.values())
            weights = {s: iv/total_inv for s, iv in inv_vols.items()}

            # Calculate current equity
            equity = cash
            for s, p in positions.items():
                if s in data and date in data[s].index:
                    equity += float(data[s].loc[date, "close"]) * p["qty"]

            target_alloc = equity * total_allocation

            # Close positions that are over-allocated or not in weights
            for symbol in list(positions.keys()):
                if symbol not in weights or symbol not in data or date not in data[symbol].index:
                    # Close fully
                    if symbol in data and date in data[symbol].index:
                        price = float(data[symbol].loc[date, "close"])
                        fill = price * (1 - fees["slippage"])
                        pos = positions[symbol]
                        gross = (fill - pos["entry_price"]) * pos["qty"]
                        fee = pos["entry_price"] * pos["qty"] * fees["entry"] + fill * pos["qty"] * fees["exit"]
                        trades.append({"symbol": symbol, "net_pnl": gross - fee, "fees": fee})
                        cash += fill * pos["qty"]
                        del positions[symbol]

            # Adjust positions to target
            for symbol, weight in weights.items():
                if symbol not in data or date not in data[symbol].index:
                    continue

                price = float(data[symbol].loc[date, "close"])
                target_value = target_alloc * weight

                if symbol in positions:
                    current_value = price * positions[symbol]["qty"]
                    diff = target_value - current_value

                    if abs(diff) > target_value * 0.1:  # Only rebalance if >10% off
                        if diff > 0:
                            # Buy more
                            buy_value = min(diff, cash * 0.95)
                            if buy_value > 5:
                                fill = price * (1 + fees["slippage"])
                                add_qty = buy_value / (fill * (1 + fees["entry"]))
                                positions[symbol]["qty"] += add_qty
                                cash -= buy_value
                        else:
                            # Sell some
                            sell_qty = min(-diff / price, positions[symbol]["qty"])
                            if sell_qty > 0:
                                fill = price * (1 - fees["slippage"])
                                cash += fill * sell_qty
                                positions[symbol]["qty"] -= sell_qty
                else:
                    # New position
                    buy_value = min(target_value, cash * 0.95)
                    if buy_value > 5:
                        fill = price * (1 + fees["slippage"])
                        qty = buy_value / (fill * (1 + fees["entry"]))
                        positions[symbol] = {"qty": qty, "entry_price": fill}
                        cash -= buy_value

            last_rebalance = date

        # Track equity
        equity = cash
        for s, p in positions.items():
            if s in data and date in data[s].index:
                equity += float(data[s].loc[date, "close"]) * p["qty"]
                days_in_market += 1
        equity_curve.append(equity)

    # Close all
    final_date = dates[-1]
    for symbol in list(positions.keys()):
        if symbol in data and final_date in data[symbol].index:
            price = float(data[symbol].loc[final_date, "close"])
            fill = price * (1 - fees["slippage"])
            pos = positions[symbol]
            gross = (fill - pos["entry_price"]) * pos["qty"]
            fee = pos["entry_price"] * pos["qty"] * fees["entry"] + fill * pos["qty"] * fees["exit"]
            trades.append({"symbol": symbol, "net_pnl": gross - fee, "fees": fee})
            cash += fill * pos["qty"]

    capital = cash

    # BTC return
    btc_df = data.get("BTC-USD")
    btc_return = 0.0
    if btc_df is not None:
        valid = [d for d in dates[min_bars:] if d in btc_df.index]
        if len(valid) >= 2:
            btc_return = (float(btc_df.loc[valid[-1], "close"]) / float(btc_df.loc[valid[0], "close"]) - 1) * 100

    # Metrics
    total_return = (capital / initial_capital - 1) * 100
    total_trades = len(trades)
    winners = len([t for t in trades if t.get("net_pnl", 0) > 0])
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    gross_profit = sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) > 0)
    gross_loss = abs(sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    equity = pd.Series(equity_curve) if equity_curve else pd.Series([initial_capital])
    max_dd = float(((equity.cummax() - equity) / equity.cummax() * 100).max())
    returns = equity.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0
    total_fees = sum(t.get("fees", 0) for t in trades)
    time_in_market = (days_in_market / len(equity_curve) / len(positions) * 100) if equity_curve and positions else 0

    return BacktestResult(
        strategy_name=name,
        trades=trades,
        equity_curve=equity_curve,
        total_return_pct=total_return,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        total_fees=total_fees,
        btc_buy_hold_return=btc_return,
        alpha_vs_btc=total_return - btc_return,
        time_in_market_pct=time_in_market
    )


# =============================================================================
# MAIN
# =============================================================================

async def run_lab():
    print("\n" + "="*70)
    print("STRATEGY LAB V4 - Mostly Long Approach")
    print("="*70)

    print("\nLoading data...")
    data = await load_all_data(ASSETS, "2024-01-01", "2025-12-17")

    if not data:
        print("ERROR: No data loaded!")
        return

    results = []

    # 1. BTC Buy and Hold (Baseline)
    print("\n" + "="*70)
    print("BASELINE: BTC Buy and Hold")
    print("="*70)
    result = run_btc_buy_hold(data, 0.95, GEMINI_FEES)
    if result:
        print(result.summary())
        stress = run_btc_buy_hold(data, 0.95, GEMINI_FEES_2X)
        stress_pass = stress.total_return_pct > 0 if stress else False
        print(f"Stress test: {stress.total_return_pct:+.1f}% {'PASS' if stress_pass else 'FAIL'}\n")
        results.append((result.strategy_name, result, stress_pass))

    # 2. Mostly Long variants
    print("\n" + "="*70)
    print("MOSTLY LONG STRATEGIES")
    print("="*70)

    for exit_sma in [100, 150, 200]:
        for reentry_sma in [30, 50]:
            result = run_mostly_long(data, "BTC-USD", exit_sma, 30, reentry_sma, 0.90, GEMINI_FEES)
            if result:
                print(result.summary())
                stress = run_mostly_long(data, "BTC-USD", exit_sma, 30, reentry_sma, 0.90, GEMINI_FEES_2X)
                stress_pass = stress.total_return_pct > 0 if stress else False
                print(f"Stress test: {stress.total_return_pct:+.1f}% {'PASS' if stress_pass else 'FAIL'}\n")
                results.append((result.strategy_name, result, stress_pass))

    # 3. Risk Parity
    print("\n" + "="*70)
    print("RISK PARITY STRATEGIES")
    print("="*70)

    for vol_period in [14, 20, 30]:
        for rebalance in [7, 14]:
            result = run_risk_parity_rotation(data, vol_period, rebalance, 0.80, GEMINI_FEES)
            if result:
                print(result.summary())
                stress = run_risk_parity_rotation(data, vol_period, rebalance, 0.80, GEMINI_FEES_2X)
                stress_pass = stress.total_return_pct > 0 if stress else False
                print(f"Stress test: {stress.total_return_pct:+.1f}% {'PASS' if stress_pass else 'FAIL'}\n")
                results.append((result.strategy_name, result, stress_pass))

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Strategy':<45} {'Return':>10} {'MaxDD':>8} {'Sharpe':>8} {'Stress':>8} {'Pass':>6}")
    print("-"*80)

    for name, result, stress_pass in results:
        passed, _ = result.passes_criteria()
        overall = "YES" if (passed and stress_pass) else "NO"
        print(f"{name:<45} {result.total_return_pct:>+9.1f}% {result.max_drawdown_pct:>7.1f}% {result.sharpe_ratio:>7.2f} {'PASS' if stress_pass else 'FAIL':>8} {overall:>6}")

    # Find winners
    print("\n" + "="*70)
    passing = [(n, r, s) for n, r, s in results if r.passes_criteria()[0] and s]
    if passing:
        print(f"PASSING STRATEGIES: {len(passing)}")
        for name, result, _ in passing:
            print(f"  - {name}: {result.total_return_pct:+.1f}% return, {result.max_drawdown_pct:.1f}% DD")

        best = max(passing, key=lambda x: x[1].sharpe_ratio)
        print(f"\nBEST BY SHARPE: {best[0]}")
        print(best[1].summary())
    else:
        print("NO STRATEGIES PASSED ALL CRITERIA")
        if results:
            best = max(results, key=lambda x: (x[1].total_return_pct / max(x[1].max_drawdown_pct, 1)))
            print(f"\nBest risk-adjusted: {best[0]}")
            print(f"  Return: {best[1].total_return_pct:+.1f}%, DD: {best[1].max_drawdown_pct:.1f}%")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_lab())
