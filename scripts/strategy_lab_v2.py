#!/usr/bin/env python3
"""
Strategy Lab V2 - Iteration with improved risk management

Key changes:
- Reduced position sizes (15-20%)
- Volatility-scaled sizing
- Rotational "always deployed" strategy
- Trailing profit locks
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod


# =============================================================================
# CONFIGURATION
# =============================================================================

ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
TRAIN_END = "2024-09-01"
TEST_START = "2024-09-01"
INITIAL_CAPITAL = 500.0

GEMINI_FEES = {"entry": 0.004, "exit": 0.002, "slippage": 0.001}
GEMINI_FEES_2X = {"entry": 0.008, "exit": 0.004, "slippage": 0.002}  # Stress test

MAX_PORTFOLIO_DRAWDOWN = 0.25
MAX_POSITION_SIZE = 0.20  # Reduced from 0.30
MAX_POSITIONS = 3


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

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_momentum(series: pd.Series, period: int = 20) -> pd.Series:
    return (series / series.shift(period) - 1) * 100

def calc_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    return series.pct_change().rolling(window=period).std() * np.sqrt(252) * 100


# =============================================================================
# TRADE & RESULT CLASSES
# =============================================================================

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestResult:
    strategy_name: str
    period: str
    trades: List[Trade]
    equity_curve: List[float]
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_fees: float
    btc_buy_hold_return: float
    alpha_vs_btc: float

    def passes_criteria(self) -> Tuple[bool, List[str]]:
        failures = []
        if self.max_drawdown_pct > 25.0:
            failures.append(f"Drawdown {self.max_drawdown_pct:.1f}% > 25% limit")
        if self.total_return_pct <= 0:
            failures.append(f"Negative return: {self.total_return_pct:.1f}%")
        beats_btc = self.total_return_pct >= self.btc_buy_hold_return
        lower_dd = self.max_drawdown_pct < (self.btc_buy_hold_return * 0.5)
        if not beats_btc and not lower_dd:
            failures.append(f"Doesn't beat BTC ({self.btc_buy_hold_return:.1f}%) or have better risk")
        return len(failures) == 0, failures

    def summary(self) -> str:
        passed, failures = self.passes_criteria()
        status = "PASS" if passed else "FAIL"
        return f"""
{'='*70}
{self.strategy_name} | {self.period.upper()} | {status}
{'='*70}
Return: {self.total_return_pct:+.2f}% | BTC: {self.btc_buy_hold_return:+.2f}% | Alpha: {self.alpha_vs_btc:+.2f}%
Trades: {self.total_trades} | Win Rate: {self.win_rate:.1f}% | PF: {self.profit_factor:.2f}
Max DD: {self.max_drawdown_pct:.1f}% | Sharpe: {self.sharpe_ratio:.2f} | Sortino: {self.sortino_ratio:.2f}
Fees: ${self.total_fees:.2f}
{'-'*70}
{chr(10).join(failures) if failures else 'ALL CRITERIA PASSED'}
{'='*70}
"""


# =============================================================================
# BACKTESTER V2
# =============================================================================

class BacktesterV2:
    """Enhanced backtester with volatility-scaled sizing and trailing stops."""

    def __init__(
        self,
        name: str,
        fees: Dict[str, float],
        initial_capital: float = 500.0,
        max_positions: int = 3,
        base_position_size: float = 0.20,
        max_drawdown: float = 0.25,
        use_vol_scaling: bool = True,
        target_vol: float = 30.0  # Target 30% annualized vol per position
    ):
        self.name = name
        self.fees = fees
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.base_position_size = base_position_size
        self.max_drawdown = max_drawdown
        self.use_vol_scaling = use_vol_scaling
        self.target_vol = target_vol

        # State
        self.capital = initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.peak_equity = initial_capital
        self.halted = False

    def _vol_adjusted_size(self, current_vol: float) -> float:
        """Scale position size inversely to volatility."""
        if not self.use_vol_scaling or current_vol <= 0:
            return self.base_position_size

        # If vol is higher than target, reduce position size
        # If vol is lower than target, increase (but cap at base)
        scale = min(1.0, self.target_vol / current_vol)
        return self.base_position_size * scale

    def _calculate_equity(self, date: datetime, data: Dict[str, pd.DataFrame]) -> float:
        equity = self.capital
        for symbol, pos in self.positions.items():
            if symbol in data and date in data[symbol].index:
                current_price = float(data[symbol].loc[date, "close"])
                unrealized = (current_price - pos["entry_price"]) * pos["quantity"]
                equity += unrealized
        return equity


# =============================================================================
# STRATEGY: Volatility-Adjusted Momentum
# =============================================================================

class VolAdjustedMomentum:
    """
    Momentum with volatility-adjusted position sizing.

    - Rank assets by momentum
    - Size positions inversely to volatility
    - Use wide ATR stops
    - Trail stops to lock in profits
    """

    def __init__(
        self,
        fees: Dict[str, float],
        momentum_period: int = 30,
        vol_period: int = 20,
        atr_period: int = 14,
        atr_mult: float = 3.0,
        target_vol: float = 30.0,  # Target annualized vol
        base_size: float = 0.20,
        trail_trigger: float = 0.15,  # Start trailing after 15% profit
        trail_distance: float = 0.10,  # Trail at 10% below high
    ):
        self.name = f"VolAdj_M{momentum_period}_TV{target_vol}_ATR{atr_mult}"
        self.fees = fees
        self.momentum_period = momentum_period
        self.vol_period = vol_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.target_vol = target_vol
        self.base_size = base_size
        self.trail_trigger = trail_trigger
        self.trail_distance = trail_distance
        self.min_bars = max(momentum_period, vol_period, atr_period) + 10

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 500.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_drawdown: float = 0.25
    ) -> BacktestResult:
        """Run the strategy."""

        # Get dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]

        # Precompute indicators
        indicators = {}
        for symbol, df in data.items():
            indicators[symbol] = {
                "momentum": calc_momentum(df["close"], self.momentum_period),
                "vol": calc_volatility(df["close"], self.vol_period),
                "atr": calc_atr(df, self.atr_period),
                "sma": calc_sma(df["close"], 50)
            }

        # State
        capital = initial_capital
        positions: Dict[str, dict] = {}
        trades: List[Trade] = []
        equity_curve: List[float] = []
        peak_equity = initial_capital
        halted = False

        # Main loop
        for date in dates[self.min_bars:]:
            if halted:
                break

            # Update existing positions
            for symbol in list(positions.keys()):
                if symbol not in data or date not in data[symbol].index:
                    continue

                pos = positions[symbol]
                bar = data[symbol].loc[date]
                current_price = float(bar["close"])
                current_low = float(bar["low"])
                current_high = float(bar["high"])

                # Update highest price since entry
                if current_high > pos["highest"]:
                    pos["highest"] = current_high

                    # Trail stop if we've hit profit trigger
                    pnl_pct = (current_high - pos["entry_price"]) / pos["entry_price"]
                    if pnl_pct >= self.trail_trigger:
                        new_stop = current_high * (1 - self.trail_distance)
                        if new_stop > pos["stop_loss"]:
                            pos["stop_loss"] = new_stop

                # Check stops
                if current_low <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"] * (1 - self.fees["slippage"])
                    gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                    fees = pos["entry_price"] * pos["quantity"] * self.fees["entry"] + \
                           exit_price * pos["quantity"] * self.fees["exit"]
                    net_pnl = gross_pnl - fees

                    trades.append(Trade(
                        symbol=symbol,
                        entry_time=pos["entry_time"],
                        exit_time=date,
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        quantity=pos["quantity"],
                        gross_pnl=gross_pnl,
                        fees=fees,
                        net_pnl=net_pnl,
                        exit_reason="stop_loss",
                        bars_held=(date - pos["entry_time"]).days
                    ))
                    capital += net_pnl
                    del positions[symbol]

            # Look for new entries if we have capacity
            if len(positions) < 3:
                candidates = []

                for symbol, df in data.items():
                    if symbol in positions:
                        continue
                    if date not in df.index:
                        continue

                    idx = df.index.get_loc(date)
                    if idx < self.min_bars:
                        continue

                    ind = indicators[symbol]
                    momentum = ind["momentum"].iloc[idx]
                    vol = ind["vol"].iloc[idx]
                    atr = ind["atr"].iloc[idx]
                    sma = ind["sma"].iloc[idx]
                    close = float(df["close"].iloc[idx])

                    # Entry conditions:
                    # 1. Positive momentum
                    # 2. Above 50-day SMA (trend filter)
                    if momentum > 0 and close > sma:
                        # Vol-adjusted size
                        if vol > 0:
                            size_mult = min(1.0, self.target_vol / vol)
                        else:
                            size_mult = 1.0

                        position_size = self.base_size * size_mult
                        stop_price = close - (self.atr_mult * atr)

                        candidates.append({
                            "symbol": symbol,
                            "momentum": momentum,
                            "size": position_size,
                            "stop": stop_price,
                            "price": close
                        })

                # Sort by momentum (strongest first)
                candidates.sort(key=lambda x: x["momentum"], reverse=True)

                # Take entries
                for c in candidates[:3 - len(positions)]:
                    alloc = capital * c["size"]
                    fill_price = c["price"] * (1 + self.fees["slippage"])
                    qty = alloc / (fill_price * (1 + self.fees["entry"]))

                    positions[c["symbol"]] = {
                        "entry_time": date,
                        "entry_price": fill_price,
                        "quantity": qty,
                        "stop_loss": c["stop"],
                        "highest": c["price"]
                    }

            # Update equity
            equity = capital
            for symbol, pos in positions.items():
                if symbol in data and date in data[symbol].index:
                    current = float(data[symbol].loc[date, "close"])
                    equity += (current - pos["entry_price"]) * pos["quantity"]
            equity_curve.append(equity)

            # Check drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            if dd > max_drawdown:
                # Close all positions
                for symbol in list(positions.keys()):
                    if symbol in data and date in data[symbol].index:
                        price = float(data[symbol].loc[date, "close"])
                        pos = positions[symbol]
                        exit_price = price * (1 - self.fees["slippage"])
                        gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                        fees = pos["entry_price"] * pos["quantity"] * self.fees["entry"] + \
                               exit_price * pos["quantity"] * self.fees["exit"]
                        net_pnl = gross_pnl - fees
                        trades.append(Trade(
                            symbol=symbol,
                            entry_time=pos["entry_time"],
                            exit_time=date,
                            entry_price=pos["entry_price"],
                            exit_price=exit_price,
                            quantity=pos["quantity"],
                            gross_pnl=gross_pnl,
                            fees=fees,
                            net_pnl=net_pnl,
                            exit_reason="drawdown_limit",
                            bars_held=(date - pos["entry_time"]).days
                        ))
                        capital += net_pnl
                        del positions[symbol]
                halted = True

        # Close remaining positions
        if not halted and dates:
            final_date = dates[-1]
            for symbol in list(positions.keys()):
                if symbol in data and final_date in data[symbol].index:
                    price = float(data[symbol].loc[final_date, "close"])
                    pos = positions[symbol]
                    exit_price = price * (1 - self.fees["slippage"])
                    gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                    fees = pos["entry_price"] * pos["quantity"] * self.fees["entry"] + \
                           exit_price * pos["quantity"] * self.fees["exit"]
                    net_pnl = gross_pnl - fees
                    trades.append(Trade(
                        symbol=symbol,
                        entry_time=pos["entry_time"],
                        exit_time=final_date,
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        quantity=pos["quantity"],
                        gross_pnl=gross_pnl,
                        fees=fees,
                        net_pnl=net_pnl,
                        exit_reason="end_of_period",
                        bars_held=(final_date - pos["entry_time"]).days
                    ))
                    capital += net_pnl

        # Calculate BTC buy-and-hold
        btc_return = 0.0
        if "BTC-USD" in data and len(dates) >= 2:
            btc_df = data["BTC-USD"]
            valid_dates = [d for d in dates if d in btc_df.index]
            if len(valid_dates) >= 2:
                btc_return = (float(btc_df.loc[valid_dates[-1], "close"]) /
                             float(btc_df.loc[valid_dates[0], "close"]) - 1) * 100

        # Generate result
        total_return = (capital / initial_capital - 1) * 100
        total_trades = len(trades)
        winners = len([t for t in trades if t.net_pnl > 0])
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        equity = pd.Series(equity_curve) if equity_curve else pd.Series([initial_capital])
        rolling_max = equity.cummax()
        drawdown = (rolling_max - equity) / rolling_max * 100
        max_dd = float(drawdown.max())

        returns = equity.pct_change().dropna()
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0
        downside = returns[returns < 0].std()
        sortino = float((returns.mean() / downside) * np.sqrt(252)) if downside > 0 else 0

        total_fees = sum(t.fees for t in trades)

        return BacktestResult(
            strategy_name=self.name,
            period="backtest",
            trades=trades,
            equity_curve=equity_curve,
            total_return_pct=total_return,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            total_fees=total_fees,
            btc_buy_hold_return=btc_return,
            alpha_vs_btc=total_return - btc_return
        )


# =============================================================================
# STRATEGY: Rotational Momentum (Always Invested)
# =============================================================================

class RotationalMomentum:
    """
    Always stay invested in the strongest momentum asset.

    - Rank assets by momentum weekly
    - Always hold the top asset(s)
    - Rotate when momentum ranking changes
    """

    def __init__(
        self,
        fees: Dict[str, float],
        momentum_period: int = 20,
        rebalance_days: int = 7,  # Weekly rebalance
        num_holdings: int = 1,  # How many to hold
        position_size: float = 0.50,  # 50% of capital in crypto
    ):
        self.name = f"Rotational_M{momentum_period}_R{rebalance_days}_N{num_holdings}"
        self.fees = fees
        self.momentum_period = momentum_period
        self.rebalance_days = rebalance_days
        self.num_holdings = num_holdings
        self.position_size = position_size
        self.min_bars = momentum_period + 10

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 500.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_drawdown: float = 0.25
    ) -> BacktestResult:
        """Run the strategy."""

        # Get dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]

        # Precompute momentum
        momentum_data = {}
        for symbol, df in data.items():
            momentum_data[symbol] = calc_momentum(df["close"], self.momentum_period)

        # State
        capital = initial_capital
        cash = initial_capital
        positions: Dict[str, dict] = {}
        trades: List[Trade] = []
        equity_curve: List[float] = []
        peak_equity = initial_capital
        halted = False
        last_rebalance = None

        for date in dates[self.min_bars:]:
            if halted:
                break

            # Check if we need to rebalance
            should_rebalance = (
                last_rebalance is None or
                (date - last_rebalance).days >= self.rebalance_days
            )

            if should_rebalance:
                # Rank assets by momentum
                rankings = []
                for symbol, df in data.items():
                    if date not in df.index:
                        continue
                    if date not in momentum_data[symbol].index:
                        continue
                    mom = momentum_data[symbol].loc[date]
                    if pd.notna(mom):
                        rankings.append((symbol, mom))

                rankings.sort(key=lambda x: x[1], reverse=True)
                top_assets = [r[0] for r in rankings[:self.num_holdings] if r[1] > 0]

                # Close positions not in top
                for symbol in list(positions.keys()):
                    if symbol not in top_assets:
                        if symbol in data and date in data[symbol].index:
                            price = float(data[symbol].loc[date, "close"])
                            pos = positions[symbol]
                            exit_price = price * (1 - self.fees["slippage"])
                            gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                            fees = pos["entry_price"] * pos["quantity"] * self.fees["entry"] + \
                                   exit_price * pos["quantity"] * self.fees["exit"]
                            net_pnl = gross_pnl - fees
                            trades.append(Trade(
                                symbol=symbol,
                                entry_time=pos["entry_time"],
                                exit_time=date,
                                entry_price=pos["entry_price"],
                                exit_price=exit_price,
                                quantity=pos["quantity"],
                                gross_pnl=gross_pnl,
                                fees=fees,
                                net_pnl=net_pnl,
                                exit_reason="rotation",
                                bars_held=(date - pos["entry_time"]).days
                            ))
                            cash += exit_price * pos["quantity"]
                            del positions[symbol]

                # Open new positions
                for symbol in top_assets:
                    if symbol not in positions:
                        if symbol in data and date in data[symbol].index:
                            # Calculate total equity for sizing
                            equity = cash
                            for s, p in positions.items():
                                if s in data and date in data[s].index:
                                    equity += float(data[s].loc[date, "close"]) * p["quantity"]

                            alloc = equity * self.position_size / self.num_holdings
                            if alloc > cash:
                                alloc = cash * 0.95  # Keep some reserve

                            if alloc > 10:  # Minimum trade size
                                price = float(data[symbol].loc[date, "close"])
                                fill_price = price * (1 + self.fees["slippage"])
                                qty = alloc / (fill_price * (1 + self.fees["entry"]))
                                cash -= alloc

                                positions[symbol] = {
                                    "entry_time": date,
                                    "entry_price": fill_price,
                                    "quantity": qty
                                }

                last_rebalance = date

            # Calculate equity
            equity = cash
            for symbol, pos in positions.items():
                if symbol in data and date in data[symbol].index:
                    equity += float(data[symbol].loc[date, "close"]) * pos["quantity"]
            equity_curve.append(equity)

            # Check drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            if dd > max_drawdown:
                for symbol in list(positions.keys()):
                    if symbol in data and date in data[symbol].index:
                        price = float(data[symbol].loc[date, "close"])
                        pos = positions[symbol]
                        exit_price = price * (1 - self.fees["slippage"])
                        gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                        fees = pos["entry_price"] * pos["quantity"] * self.fees["entry"] + \
                               exit_price * pos["quantity"] * self.fees["exit"]
                        net_pnl = gross_pnl - fees
                        trades.append(Trade(
                            symbol=symbol,
                            entry_time=pos["entry_time"],
                            exit_time=date,
                            entry_price=pos["entry_price"],
                            exit_price=exit_price,
                            quantity=pos["quantity"],
                            gross_pnl=gross_pnl,
                            fees=fees,
                            net_pnl=net_pnl,
                            exit_reason="drawdown_limit",
                            bars_held=(date - pos["entry_time"]).days
                        ))
                        cash += exit_price * pos["quantity"]
                        del positions[symbol]
                halted = True

        # Close remaining
        if not halted and dates:
            final_date = dates[-1]
            for symbol in list(positions.keys()):
                if symbol in data and final_date in data[symbol].index:
                    price = float(data[symbol].loc[final_date, "close"])
                    pos = positions[symbol]
                    exit_price = price * (1 - self.fees["slippage"])
                    gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                    fees = pos["entry_price"] * pos["quantity"] * self.fees["entry"] + \
                           exit_price * pos["quantity"] * self.fees["exit"]
                    net_pnl = gross_pnl - fees
                    trades.append(Trade(
                        symbol=symbol,
                        entry_time=pos["entry_time"],
                        exit_time=final_date,
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        quantity=pos["quantity"],
                        gross_pnl=gross_pnl,
                        fees=fees,
                        net_pnl=net_pnl,
                        exit_reason="end_of_period",
                        bars_held=(final_date - pos["entry_time"]).days
                    ))
                    cash += exit_price * pos["quantity"]

        capital = cash

        # Calculate BTC buy-and-hold
        btc_return = 0.0
        if "BTC-USD" in data and len(dates) >= 2:
            btc_df = data["BTC-USD"]
            valid_dates = [d for d in dates if d in btc_df.index]
            if len(valid_dates) >= 2:
                btc_return = (float(btc_df.loc[valid_dates[-1], "close"]) /
                             float(btc_df.loc[valid_dates[0], "close"]) - 1) * 100

        # Generate result
        total_return = (capital / initial_capital - 1) * 100
        total_trades = len(trades)
        winners = len([t for t in trades if t.net_pnl > 0])
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        equity = pd.Series(equity_curve) if equity_curve else pd.Series([initial_capital])
        rolling_max = equity.cummax()
        drawdown = (rolling_max - equity) / rolling_max * 100
        max_dd = float(drawdown.max())

        returns = equity.pct_change().dropna()
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0
        downside = returns[returns < 0].std()
        sortino = float((returns.mean() / downside) * np.sqrt(252)) if downside > 0 else 0

        total_fees = sum(t.fees for t in trades)

        return BacktestResult(
            strategy_name=self.name,
            period="backtest",
            trades=trades,
            equity_curve=equity_curve,
            total_return_pct=total_return,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            total_fees=total_fees,
            btc_buy_hold_return=btc_return,
            alpha_vs_btc=total_return - btc_return
        )


# =============================================================================
# MAIN
# =============================================================================

async def run_lab():
    print("\n" + "="*70)
    print("STRATEGY LAB V2 - Improved Risk Management")
    print("="*70)

    print("\nLoading data...")
    data = await load_all_data(ASSETS, "2024-01-01", "2025-12-17", "ONE_DAY")

    if not data:
        print("ERROR: No data loaded!")
        return

    # Strategies to test
    strategies = [
        # Vol-adjusted momentum variants
        VolAdjustedMomentum(GEMINI_FEES, momentum_period=30, target_vol=30, atr_mult=3.0, base_size=0.20),
        VolAdjustedMomentum(GEMINI_FEES, momentum_period=20, target_vol=40, atr_mult=2.5, base_size=0.15),
        VolAdjustedMomentum(GEMINI_FEES, momentum_period=40, target_vol=25, atr_mult=4.0, base_size=0.20),

        # Rotational momentum variants
        RotationalMomentum(GEMINI_FEES, momentum_period=20, rebalance_days=7, num_holdings=1, position_size=0.40),
        RotationalMomentum(GEMINI_FEES, momentum_period=30, rebalance_days=14, num_holdings=1, position_size=0.50),
        RotationalMomentum(GEMINI_FEES, momentum_period=20, rebalance_days=7, num_holdings=2, position_size=0.60),
    ]

    results = []

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Testing: {strategy.name}")
        print("="*70)

        # Train
        print("\n--- TRAIN (Jan-Aug 2024) ---")
        train_result = strategy.run(data, INITIAL_CAPITAL, end_date=TRAIN_END)
        train_result.period = "train"
        print(train_result.summary())

        # Test
        print("\n--- TEST (Sep 2024 - Dec 2025) ---")
        test_result = strategy.run(data, INITIAL_CAPITAL, start_date=TEST_START)
        test_result.period = "test"
        print(test_result.summary())

        # Stress test with 2x fees
        print("\n--- STRESS TEST (2x Fees) ---")
        if isinstance(strategy, VolAdjustedMomentum):
            stress_strat = VolAdjustedMomentum(
                GEMINI_FEES_2X,
                momentum_period=strategy.momentum_period,
                vol_period=strategy.vol_period,
                atr_period=strategy.atr_period,
                atr_mult=strategy.atr_mult,
                target_vol=strategy.target_vol,
                base_size=strategy.base_size,
                trail_trigger=strategy.trail_trigger,
                trail_distance=strategy.trail_distance
            )
        else:
            stress_strat = RotationalMomentum(
                GEMINI_FEES_2X,
                momentum_period=strategy.momentum_period,
                rebalance_days=strategy.rebalance_days,
                num_holdings=strategy.num_holdings,
                position_size=strategy.position_size
            )

        stress_result = stress_strat.run(data, INITIAL_CAPITAL, start_date=TEST_START)
        stress_result.period = "stress"
        stress_pass = stress_result.total_return_pct > 0
        print(f"Stress test (2x fees): {stress_result.total_return_pct:+.1f}% {'PASS' if stress_pass else 'FAIL'}")

        results.append((strategy.name, train_result, test_result, stress_result))

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Strategy':<45} {'Train':>8} {'Test':>8} {'Stress':>8} {'Pass':>6}")
    print("-"*70)

    for name, train, test, stress in results:
        train_ok, _ = train.passes_criteria()
        test_ok, _ = test.passes_criteria()
        stress_ok = stress.total_return_pct > 0
        overall = "YES" if (train_ok and test_ok and stress_ok) else "NO"
        print(f"{name:<45} {train.total_return_pct:>+7.1f}% {test.total_return_pct:>+7.1f}% {stress.total_return_pct:>+7.1f}% {overall:>6}")

    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_lab())
