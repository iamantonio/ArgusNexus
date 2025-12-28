#!/usr/bin/env python3
"""
Strategy Lab V3 - Trend Filter & Dual Momentum

Key changes:
- Full period testing (not just train/test split)
- Add SMA trend filter to avoid chop
- Dual momentum (absolute + relative)
- Simple BTC-only with trend filter baseline
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

MAX_PORTFOLIO_DRAWDOWN = 0.25


# =============================================================================
# DATA FETCHING
# =============================================================================

async def fetch_ohlcv(symbol: str, start_date: str, end_date: str, granularity: str = "ONE_DAY") -> pd.DataFrame:
    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    time_window = 300 * 86400 if granularity == "ONE_DAY" else 300 * 4 * 3600
    all_candles = []
    current_end = end_ts

    async with aiohttp.ClientSession() as session:
        while current_end > start_ts:
            params = {
                "start": str(max(start_ts, current_end - time_window)),
                "end": str(current_end),
                "granularity": granularity,
                "limit": "300"
            }
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


async def load_all_data(assets: List[str], start_date: str, end_date: str, granularity: str = "ONE_DAY") -> Dict[str, pd.DataFrame]:
    data = {}
    for symbol in assets:
        print(f"  Loading {symbol}...", end=" ")
        df = await fetch_ohlcv(symbol, start_date, end_date, granularity)
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

def calc_momentum(series: pd.Series, period: int = 20) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


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
            failures.append(f"Negative return: {self.total_return_pct:.1f}%")

        # Beat BTC OR have materially lower drawdown
        beats_btc = self.total_return_pct >= self.btc_buy_hold_return
        # Lower drawdown = less than half BTC's return as proxy for "smoother"
        has_better_risk = self.max_drawdown_pct < 15.0 and self.total_return_pct > self.btc_buy_hold_return * 0.5

        if not beats_btc and not has_better_risk:
            failures.append(f"Doesn't beat BTC ({self.btc_buy_hold_return:.1f}%) or have better risk-adjusted returns")

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
# STRATEGY 1: Simple SMA Trend Filter
# =============================================================================

def run_sma_trend_filter(
    data: Dict[str, pd.DataFrame],
    symbol: str = "BTC-USD",
    sma_period: int = 50,
    position_size: float = 0.80,  # 80% of capital
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> BacktestResult:
    """
    Simple SMA trend filter:
    - Above SMA = hold position
    - Below SMA = go to cash
    """
    name = f"SMA{sma_period}_TrendFilter_{symbol}"

    df = data.get(symbol)
    if df is None or len(df) < sma_period + 10:
        return None

    sma = calc_sma(df["close"], sma_period)

    # State
    capital = initial_capital
    position = None  # None or {"qty": x, "entry_price": x, "entry_time": x}
    trades = []
    equity_curve = []
    days_in_market = 0

    for i in range(sma_period + 1, len(df)):
        date = df.index[i]
        close = float(df["close"].iloc[i])
        current_sma = sma.iloc[i]

        # Entry: price above SMA and not in position
        if close > current_sma and position is None:
            alloc = capital * position_size
            fill_price = close * (1 + fees["slippage"])
            qty = alloc / (fill_price * (1 + fees["entry"]))
            position = {
                "qty": qty,
                "entry_price": fill_price,
                "entry_time": date
            }

        # Exit: price below SMA and in position
        elif close < current_sma and position is not None:
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
                "qty": position["qty"],
                "net_pnl": net_pnl,
                "fees": entry_fee + exit_fee
            })

            capital += net_pnl
            position = None

        # Track equity
        if position is not None:
            equity = capital - (position["entry_price"] * position["qty"]) + (close * position["qty"])
            days_in_market += 1
        else:
            equity = capital
        equity_curve.append(equity)

    # Close any remaining position
    if position is not None:
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
            "qty": position["qty"],
            "net_pnl": net_pnl,
            "fees": entry_fee + exit_fee
        })
        capital += net_pnl

    # BTC buy and hold
    btc_return = (float(df["close"].iloc[-1]) / float(df["close"].iloc[sma_period]) - 1) * 100

    # Metrics
    total_return = (capital / initial_capital - 1) * 100
    total_trades = len(trades)
    winners = len([t for t in trades if t["net_pnl"] > 0])
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_loss = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    equity = pd.Series(equity_curve)
    rolling_max = equity.cummax()
    drawdown = (rolling_max - equity) / rolling_max * 100
    max_dd = float(drawdown.max())

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
# STRATEGY 2: Dual Momentum
# =============================================================================

def run_dual_momentum(
    data: Dict[str, pd.DataFrame],
    lookback: int = 30,
    sma_period: int = 50,
    rebalance_days: int = 7,
    position_size: float = 0.50,
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> BacktestResult:
    """
    Dual Momentum:
    1. Absolute momentum: only go long if momentum > 0
    2. Relative momentum: pick the best asset
    3. Trend filter: must be above SMA
    """
    name = f"DualMom_L{lookback}_SMA{sma_period}"

    # Get all dates
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index.tolist())
    dates = sorted(all_dates)

    min_bars = max(lookback, sma_period) + 10

    # Precompute indicators
    momentum_data = {}
    sma_data = {}
    for symbol, df in data.items():
        momentum_data[symbol] = calc_momentum(df["close"], lookback)
        sma_data[symbol] = calc_sma(df["close"], sma_period)

    # State
    capital = initial_capital
    cash = initial_capital
    position = None  # {"symbol": x, "qty": x, "entry_price": x, "entry_time": x}
    trades = []
    equity_curve = []
    last_rebalance = None
    days_in_market = 0

    for date in dates[min_bars:]:
        # Check rebalance
        should_rebalance = last_rebalance is None or (date - last_rebalance).days >= rebalance_days

        if should_rebalance:
            # Find best asset
            candidates = []
            for symbol, df in data.items():
                if date not in df.index:
                    continue
                if date not in momentum_data[symbol].index:
                    continue
                if date not in sma_data[symbol].index:
                    continue

                close = float(df.loc[date, "close"])
                mom = momentum_data[symbol].loc[date]
                sma = sma_data[symbol].loc[date]

                # Dual filters: positive momentum AND above SMA
                if pd.notna(mom) and pd.notna(sma) and mom > 0 and close > sma:
                    candidates.append((symbol, mom, close))

            # Sort by momentum
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0] if candidates else None

            # Close current position if different asset or no valid candidate
            if position is not None:
                should_close = (best is None) or (best[0] != position["symbol"])
                if should_close:
                    df = data[position["symbol"]]
                    if date in df.index:
                        close = float(df.loc[date, "close"])
                        fill_price = close * (1 - fees["slippage"])
                        gross_pnl = (fill_price - position["entry_price"]) * position["qty"]
                        entry_fee = position["entry_price"] * position["qty"] * fees["entry"]
                        exit_fee = fill_price * position["qty"] * fees["exit"]
                        net_pnl = gross_pnl - entry_fee - exit_fee

                        trades.append({
                            "symbol": position["symbol"],
                            "entry_time": position["entry_time"],
                            "exit_time": date,
                            "entry_price": position["entry_price"],
                            "exit_price": fill_price,
                            "qty": position["qty"],
                            "net_pnl": net_pnl,
                            "fees": entry_fee + exit_fee
                        })

                        cash += fill_price * position["qty"]
                        position = None

            # Open new position
            if best is not None and position is None:
                symbol, _, close = best
                alloc = cash * position_size
                if alloc > 10:
                    fill_price = close * (1 + fees["slippage"])
                    qty = alloc / (fill_price * (1 + fees["entry"]))
                    cash -= alloc
                    position = {
                        "symbol": symbol,
                        "qty": qty,
                        "entry_price": fill_price,
                        "entry_time": date
                    }

            last_rebalance = date

        # Track equity
        if position is not None:
            df = data[position["symbol"]]
            if date in df.index:
                current_price = float(df.loc[date, "close"])
                equity = cash + (current_price * position["qty"])
                days_in_market += 1
            else:
                equity = cash + (position["entry_price"] * position["qty"])
        else:
            equity = cash
        equity_curve.append(equity)

    # Close remaining
    if position is not None:
        df = data[position["symbol"]]
        date = dates[-1]
        if date in df.index:
            close = float(df.loc[date, "close"])
            fill_price = close * (1 - fees["slippage"])
            gross_pnl = (fill_price - position["entry_price"]) * position["qty"]
            entry_fee = position["entry_price"] * position["qty"] * fees["entry"]
            exit_fee = fill_price * position["qty"] * fees["exit"]
            net_pnl = gross_pnl - entry_fee - exit_fee

            trades.append({
                "symbol": position["symbol"],
                "entry_time": position["entry_time"],
                "exit_time": date,
                "entry_price": position["entry_price"],
                "exit_price": fill_price,
                "qty": position["qty"],
                "net_pnl": net_pnl,
                "fees": entry_fee + exit_fee
            })
            cash += fill_price * position["qty"]

    capital = cash

    # BTC buy and hold
    btc_df = data.get("BTC-USD")
    btc_return = 0.0
    if btc_df is not None:
        valid_dates = [d for d in dates[min_bars:] if d in btc_df.index]
        if len(valid_dates) >= 2:
            btc_return = (float(btc_df.loc[valid_dates[-1], "close"]) /
                         float(btc_df.loc[valid_dates[0], "close"]) - 1) * 100

    # Metrics
    total_return = (capital / initial_capital - 1) * 100
    total_trades = len(trades)
    winners = len([t for t in trades if t["net_pnl"] > 0])
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_loss = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    equity = pd.Series(equity_curve) if equity_curve else pd.Series([initial_capital])
    rolling_max = equity.cummax()
    drawdown = (rolling_max - equity) / rolling_max * 100
    max_dd = float(drawdown.max())

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
# STRATEGY 3: Adaptive Momentum with Volatility Filter
# =============================================================================

def run_adaptive_momentum(
    data: Dict[str, pd.DataFrame],
    momentum_period: int = 20,
    vol_period: int = 20,
    vol_threshold: float = 60.0,  # Only trade when vol < threshold
    sma_period: int = 50,
    rebalance_days: int = 7,
    position_size: float = 0.50,
    fees: Dict[str, float] = GEMINI_FEES,
    initial_capital: float = 500.0
) -> BacktestResult:
    """
    Adaptive Momentum with Volatility Filter:
    - Only trade when volatility is below threshold (avoid crazy periods)
    - Use momentum to pick best asset
    - Trend filter with SMA
    """
    name = f"AdaptiveMom_M{momentum_period}_Vol{vol_threshold}"

    # Get all dates
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index.tolist())
    dates = sorted(all_dates)

    min_bars = max(momentum_period, vol_period, sma_period) + 10

    # Precompute
    momentum_data = {}
    sma_data = {}
    vol_data = {}
    for symbol, df in data.items():
        momentum_data[symbol] = calc_momentum(df["close"], momentum_period)
        sma_data[symbol] = calc_sma(df["close"], sma_period)
        vol_data[symbol] = df["close"].pct_change().rolling(vol_period).std() * np.sqrt(252) * 100

    # State
    capital = initial_capital
    cash = initial_capital
    position = None
    trades = []
    equity_curve = []
    last_rebalance = None
    days_in_market = 0

    for date in dates[min_bars:]:
        should_rebalance = last_rebalance is None or (date - last_rebalance).days >= rebalance_days

        if should_rebalance:
            candidates = []
            for symbol, df in data.items():
                if date not in df.index:
                    continue

                close = float(df.loc[date, "close"])
                mom = momentum_data[symbol].get(date, np.nan) if date in momentum_data[symbol].index else np.nan
                sma = sma_data[symbol].get(date, np.nan) if date in sma_data[symbol].index else np.nan
                vol = vol_data[symbol].get(date, np.nan) if date in vol_data[symbol].index else np.nan

                # Filters: positive momentum, above SMA, volatility below threshold
                if pd.notna(mom) and pd.notna(sma) and pd.notna(vol):
                    if mom > 0 and close > sma and vol < vol_threshold:
                        candidates.append((symbol, mom, close))

            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0] if candidates else None

            # Close if needed
            if position is not None:
                should_close = (best is None) or (best[0] != position["symbol"])
                if should_close:
                    df = data[position["symbol"]]
                    if date in df.index:
                        close = float(df.loc[date, "close"])
                        fill_price = close * (1 - fees["slippage"])
                        gross_pnl = (fill_price - position["entry_price"]) * position["qty"]
                        entry_fee = position["entry_price"] * position["qty"] * fees["entry"]
                        exit_fee = fill_price * position["qty"] * fees["exit"]
                        net_pnl = gross_pnl - entry_fee - exit_fee
                        trades.append({
                            "symbol": position["symbol"],
                            "entry_time": position["entry_time"],
                            "exit_time": date,
                            "net_pnl": net_pnl,
                            "fees": entry_fee + exit_fee
                        })
                        cash += fill_price * position["qty"]
                        position = None

            # Open if valid
            if best is not None and position is None:
                symbol, _, close = best
                alloc = cash * position_size
                if alloc > 10:
                    fill_price = close * (1 + fees["slippage"])
                    qty = alloc / (fill_price * (1 + fees["entry"]))
                    cash -= alloc
                    position = {
                        "symbol": symbol,
                        "qty": qty,
                        "entry_price": fill_price,
                        "entry_time": date
                    }

            last_rebalance = date

        # Equity
        if position is not None:
            df = data[position["symbol"]]
            if date in df.index:
                equity = cash + (float(df.loc[date, "close"]) * position["qty"])
                days_in_market += 1
            else:
                equity = cash + (position["entry_price"] * position["qty"])
        else:
            equity = cash
        equity_curve.append(equity)

    # Close remaining
    if position is not None:
        df = data[position["symbol"]]
        date = dates[-1]
        if date in df.index:
            close = float(df.loc[date, "close"])
            fill_price = close * (1 - fees["slippage"])
            gross_pnl = (fill_price - position["entry_price"]) * position["qty"]
            entry_fee = position["entry_price"] * position["qty"] * fees["entry"]
            exit_fee = fill_price * position["qty"] * fees["exit"]
            net_pnl = gross_pnl - entry_fee - exit_fee
            trades.append({
                "symbol": position["symbol"],
                "entry_time": position["entry_time"],
                "exit_time": date,
                "net_pnl": net_pnl,
                "fees": entry_fee + exit_fee
            })
            cash += fill_price * position["qty"]

    capital = cash

    # BTC return
    btc_df = data.get("BTC-USD")
    btc_return = 0.0
    if btc_df is not None:
        valid_dates = [d for d in dates[min_bars:] if d in btc_df.index]
        if len(valid_dates) >= 2:
            btc_return = (float(btc_df.loc[valid_dates[-1], "close"]) /
                         float(btc_df.loc[valid_dates[0], "close"]) - 1) * 100

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
# MAIN
# =============================================================================

async def run_lab():
    print("\n" + "="*70)
    print("STRATEGY LAB V3 - Trend Filter & Dual Momentum")
    print("="*70)

    print("\nLoading data (full period: Jan 2024 - Dec 2025)...")
    data = await load_all_data(ASSETS, "2024-01-01", "2025-12-17", "ONE_DAY")

    if not data:
        print("ERROR: No data loaded!")
        return

    results = []

    # Test SMA trend filters
    print("\n" + "="*70)
    print("SMA TREND FILTER STRATEGIES")
    print("="*70)

    for sma in [30, 50, 100, 200]:
        for symbol in ["BTC-USD"]:
            result = run_sma_trend_filter(data, symbol, sma, 0.80, GEMINI_FEES)
            if result:
                print(result.summary())

                # Stress test
                stress = run_sma_trend_filter(data, symbol, sma, 0.80, GEMINI_FEES_2X)
                stress_pass = stress.total_return_pct > 0 if stress else False
                print(f"Stress test (2x fees): {stress.total_return_pct:+.1f}% {'PASS' if stress_pass else 'FAIL'}\n")

                results.append((result.strategy_name, result, stress_pass))

    # Test Dual Momentum
    print("\n" + "="*70)
    print("DUAL MOMENTUM STRATEGIES")
    print("="*70)

    for lookback in [20, 30, 40]:
        for sma in [30, 50]:
            result = run_dual_momentum(data, lookback, sma, 7, 0.50, GEMINI_FEES)
            if result:
                print(result.summary())

                stress = run_dual_momentum(data, lookback, sma, 7, 0.50, GEMINI_FEES_2X)
                stress_pass = stress.total_return_pct > 0 if stress else False
                print(f"Stress test (2x fees): {stress.total_return_pct:+.1f}% {'PASS' if stress_pass else 'FAIL'}\n")

                results.append((result.strategy_name, result, stress_pass))

    # Test Adaptive Momentum
    print("\n" + "="*70)
    print("ADAPTIVE MOMENTUM STRATEGIES")
    print("="*70)

    for vol_thresh in [50, 60, 80]:
        result = run_adaptive_momentum(data, 20, 20, vol_thresh, 50, 7, 0.50, GEMINI_FEES)
        if result:
            print(result.summary())

            stress = run_adaptive_momentum(data, 20, 20, vol_thresh, 50, 7, 0.50, GEMINI_FEES_2X)
            stress_pass = stress.total_return_pct > 0 if stress else False
            print(f"Stress test (2x fees): {stress.total_return_pct:+.1f}% {'PASS' if stress_pass else 'FAIL'}\n")

            results.append((result.strategy_name, result, stress_pass))

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Strategy':<40} {'Return':>10} {'MaxDD':>8} {'Sharpe':>8} {'Stress':>8} {'Pass':>6}")
    print("-"*70)

    for name, result, stress_pass in results:
        passed, _ = result.passes_criteria()
        overall = "YES" if (passed and stress_pass) else "NO"
        print(f"{name:<40} {result.total_return_pct:>+9.1f}% {result.max_drawdown_pct:>7.1f}% {result.sharpe_ratio:>7.2f} {'PASS' if stress_pass else 'FAIL':>8} {overall:>6}")

    # Find best
    print("\n" + "="*70)
    passing = [(n, r, s) for n, r, s in results if r.passes_criteria()[0] and s]
    if passing:
        print(f"PASSING STRATEGIES: {len(passing)}")
        best = max(passing, key=lambda x: x[1].sharpe_ratio)
        print(f"\nBEST BY SHARPE: {best[0]}")
        print(best[1].summary())
    else:
        print("NO STRATEGIES PASSED ALL CRITERIA")
        # Show best overall anyway
        if results:
            best = max(results, key=lambda x: x[1].total_return_pct)
            print(f"\nBest by return: {best[0]} at {best[1].total_return_pct:+.1f}%")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_lab())
