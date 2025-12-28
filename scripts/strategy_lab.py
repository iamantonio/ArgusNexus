#!/usr/bin/env python3
"""
Strategy Lab - Systematic Strategy Development & Validation

Framework for iterating through strategies with:
- Train/test split validation
- Walk-forward analysis
- Cost stress testing (2x fees)
- BTC buy-and-hold comparison
- Proper risk metrics

Usage:
    python scripts/strategy_lab.py
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]  # Start focused, expand if working
TRAIN_END = "2024-09-01"  # Train on Jan-Aug 2024
TEST_START = "2024-09-01"  # Test on Sep 2024 - Dec 2025
INITIAL_CAPITAL = 500.0

# Fee models
GEMINI_FEES = {"entry": 0.004, "exit": 0.002, "slippage": 0.001}  # 0.4%/0.2% + 0.1% slip
COINBASE_FEES = {"entry": 0.006, "exit": 0.004, "slippage": 0.001}  # 0.6%/0.4% + 0.1% slip

# Risk limits
MAX_PORTFOLIO_DRAWDOWN = 0.25  # 25% hard limit
MAX_POSITION_SIZE = 0.30  # 30% per position
MAX_POSITIONS = 3


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # "long" or "short"
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestResult:
    strategy_name: str
    period: str  # "train" or "test"
    trades: List[Trade]
    equity_curve: List[float]

    # Core metrics
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float

    # Risk metrics
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Cost analysis
    total_fees: float
    fee_drag_pct: float  # fees as % of gross profits

    # Comparison
    btc_buy_hold_return: float
    alpha_vs_btc: float  # strategy return - btc return

    def passes_criteria(self) -> Tuple[bool, List[str]]:
        """Check if result passes all success criteria."""
        failures = []

        # Drawdown limit
        if self.max_drawdown_pct > 25.0:
            failures.append(f"Drawdown {self.max_drawdown_pct:.1f}% > 25% limit")

        # Must be net profitable
        if self.total_return_pct <= 0:
            failures.append(f"Negative return: {self.total_return_pct:.1f}%")

        # Beat BTC or have materially lower drawdown
        beats_btc = self.total_return_pct >= self.btc_buy_hold_return
        lower_dd = self.max_drawdown_pct < (self.btc_buy_hold_return * 0.5)  # Half the volatility
        if not beats_btc and not lower_dd:
            failures.append(f"Doesn't beat BTC ({self.btc_buy_hold_return:.1f}%) and no DD advantage")

        return len(failures) == 0, failures

    def summary(self) -> str:
        passed, failures = self.passes_criteria()
        status = "PASS" if passed else "FAIL"

        return f"""
{'='*70}
{self.strategy_name} | {self.period.upper()} | {status}
{'='*70}
Return: {self.total_return_pct:+.2f}% | BTC B&H: {self.btc_buy_hold_return:+.2f}% | Alpha: {self.alpha_vs_btc:+.2f}%
Trades: {self.total_trades} | Win Rate: {self.win_rate:.1f}% | Profit Factor: {self.profit_factor:.2f}
Max DD: {self.max_drawdown_pct:.1f}% | Sharpe: {self.sharpe_ratio:.2f} | Sortino: {self.sortino_ratio:.2f}
Fees: ${self.total_fees:.2f} ({self.fee_drag_pct:.1f}% of gross)
{'-'*70}
{chr(10).join(failures) if failures else 'All criteria passed'}
{'='*70}
"""


# =============================================================================
# DATA FETCHING
# =============================================================================

async def fetch_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    granularity: str = "ONE_DAY"
) -> pd.DataFrame:
    """Fetch OHLCV data from Coinbase."""
    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles"

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Time window based on granularity
    if granularity == "ONE_DAY":
        time_window = 300 * 86400
    elif granularity == "FOUR_HOUR":
        time_window = 300 * 4 * 3600
    else:
        time_window = 300 * 3600

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
    df = df.set_index("timestamp")

    return df[["open", "high", "low", "close", "volume"]]


async def load_all_data(
    assets: List[str],
    start_date: str,
    end_date: str,
    granularity: str = "ONE_DAY"
) -> Dict[str, pd.DataFrame]:
    """Load data for all assets."""
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
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_bollinger(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper = mid + (std_dev * std)
    lower = mid - (std_dev * std)
    return upper, mid, lower

def calc_momentum(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate momentum as rate of change."""
    return (series / series.shift(period) - 1) * 100

def calc_donchian(df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
    """Calculate Donchian channel high/low."""
    high = df["high"].rolling(window=period).max()
    low = df["low"].rolling(window=period).min()
    return high, low


# =============================================================================
# STRATEGY BASE CLASS
# =============================================================================

class Strategy(ABC):
    """Base class for all strategies."""

    def __init__(self, name: str, fees: Dict[str, float]):
        self.name = name
        self.fees = fees

    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals for all assets.

        Returns DataFrame with columns:
        - timestamp (index)
        - symbol
        - signal: 1 (long), -1 (short), 0 (flat)
        - strength: signal strength for ranking (optional)
        - stop_loss: suggested stop price
        """
        pass

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Minimum bars needed for indicators."""
        pass


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """
    Portfolio backtester with proper position management.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 500.0,
        max_positions: int = 3,
        max_position_size: float = 0.30,
        max_drawdown: float = 0.25
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown

        # State
        self.capital = initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.peak_equity = initial_capital
        self.halted = False  # Stop trading if drawdown limit hit

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """Run backtest on data."""

        # Get common date range
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if start_date:
            start_dt = pd.Timestamp(start_date)
            dates = [d for d in dates if d >= start_dt]
        if end_date:
            end_dt = pd.Timestamp(end_date)
            dates = [d for d in dates if d <= end_dt]

        # Generate signals
        signals_df = self.strategy.generate_signals(data)

        # Main loop
        for i, date in enumerate(dates[self.strategy.min_bars:], start=self.strategy.min_bars):
            if self.halted:
                break

            # Update existing positions
            self._update_positions(date, data)

            # Check for new entries
            if len(self.positions) < self.max_positions:
                self._check_entries(date, signals_df, data)

            # Update equity curve
            equity = self._calculate_equity(date, data)
            self.equity_curve.append(equity)

            # Check drawdown limit
            if equity > self.peak_equity:
                self.peak_equity = equity
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self._close_all_positions(date, data, "drawdown_limit")
                self.halted = True

        # Close remaining positions
        if not self.halted and dates:
            self._close_all_positions(dates[-1], data, "end_of_period")

        return self._generate_result(data, dates)

    def _update_positions(self, date: datetime, data: Dict[str, pd.DataFrame]):
        """Check stops and update positions."""
        to_close = []

        for symbol, pos in self.positions.items():
            if symbol not in data or date not in data[symbol].index:
                continue

            bar = data[symbol].loc[date]
            low = float(bar["low"])

            # Check stop loss
            if low <= pos["stop_loss"]:
                to_close.append((symbol, pos["stop_loss"], "stop_loss"))

        for symbol, price, reason in to_close:
            self._close_position(symbol, price, date, reason)

    def _check_entries(self, date: datetime, signals: pd.DataFrame, data: Dict[str, pd.DataFrame]):
        """Check for new entry signals."""
        if date not in signals.index:
            return

        day_signals = signals.loc[[date]] if isinstance(signals.loc[date], pd.Series) else signals.loc[[date]]

        # Filter for long signals only (no shorting for now)
        entries = day_signals[day_signals["signal"] == 1].copy()

        if entries.empty:
            return

        # Sort by strength if available
        if "strength" in entries.columns:
            entries = entries.sort_values("strength", ascending=False)

        # Take entries up to max positions
        available_slots = self.max_positions - len(self.positions)

        for _, row in entries.head(available_slots).iterrows():
            symbol = row["symbol"]
            if symbol in self.positions:
                continue
            if symbol not in data or date not in data[symbol].index:
                continue

            price = float(data[symbol].loc[date, "close"])
            stop_loss = float(row.get("stop_loss", price * 0.95))

            self._open_position(symbol, price, stop_loss, date)

    def _open_position(self, symbol: str, price: float, stop_loss: float, date: datetime):
        """Open a new position."""
        # Position sizing
        available = self.capital * self.max_position_size

        # Apply entry costs
        entry_cost = price * (self.strategy.fees["entry"] + self.strategy.fees["slippage"])
        fill_price = price * (1 + self.strategy.fees["slippage"])

        quantity = available / (fill_price * (1 + self.strategy.fees["entry"]))

        self.positions[symbol] = {
            "entry_time": date,
            "entry_price": fill_price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "capital_used": available
        }

    def _close_position(self, symbol: str, price: float, date: datetime, reason: str):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Apply exit costs
        fill_price = price * (1 - self.strategy.fees["slippage"])

        gross_pnl = (fill_price - pos["entry_price"]) * pos["quantity"]
        entry_fees = pos["entry_price"] * pos["quantity"] * self.strategy.fees["entry"]
        exit_fees = fill_price * pos["quantity"] * self.strategy.fees["exit"]
        total_fees = entry_fees + exit_fees
        net_pnl = gross_pnl - total_fees

        bars_held = (date - pos["entry_time"]).days

        trade = Trade(
            symbol=symbol,
            entry_time=pos["entry_time"],
            exit_time=date,
            entry_price=pos["entry_price"],
            exit_price=fill_price,
            quantity=pos["quantity"],
            side="long",
            gross_pnl=gross_pnl,
            fees=total_fees,
            net_pnl=net_pnl,
            exit_reason=reason,
            bars_held=bars_held
        )
        self.trades.append(trade)

        self.capital += net_pnl
        del self.positions[symbol]

    def _close_all_positions(self, date: datetime, data: Dict[str, pd.DataFrame], reason: str):
        """Close all open positions."""
        for symbol in list(self.positions.keys()):
            if symbol in data and date in data[symbol].index:
                price = float(data[symbol].loc[date, "close"])
                self._close_position(symbol, price, date, reason)

    def _calculate_equity(self, date: datetime, data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current equity including unrealized P&L."""
        equity = self.capital

        for symbol, pos in self.positions.items():
            if symbol in data and date in data[symbol].index:
                current_price = float(data[symbol].loc[date, "close"])
                unrealized = (current_price - pos["entry_price"]) * pos["quantity"]
                equity += unrealized

        return equity

    def _generate_result(self, data: Dict[str, pd.DataFrame], dates: List[datetime]) -> BacktestResult:
        """Generate backtest result."""

        # Calculate BTC buy-and-hold
        btc_return = 0.0
        if "BTC-USD" in data and len(dates) >= 2:
            btc_df = data["BTC-USD"]
            valid_dates = [d for d in dates if d in btc_df.index]
            if len(valid_dates) >= 2:
                start_price = float(btc_df.loc[valid_dates[0], "close"])
                end_price = float(btc_df.loc[valid_dates[-1], "close"])
                btc_return = (end_price / start_price - 1) * 100

        # Core metrics
        total_return = (self.capital / self.initial_capital - 1) * 100
        total_trades = len(self.trades)
        winners = len([t for t in self.trades if t.net_pnl > 0])
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Drawdown
        equity = pd.Series(self.equity_curve) if self.equity_curve else pd.Series([self.initial_capital])
        rolling_max = equity.cummax()
        drawdown = (rolling_max - equity) / rolling_max * 100
        max_dd = float(drawdown.max())

        # Sharpe & Sortino
        returns = equity.pct_change().dropna()
        sharpe = 0.0
        sortino = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))
            downside = returns[returns < 0].std()
            if downside > 0:
                sortino = float((returns.mean() / downside) * np.sqrt(252))

        # Calmar
        calmar = total_return / max_dd if max_dd > 0 else 0.0

        # Fees
        total_fees = sum(t.fees for t in self.trades)
        gross_pnl = sum(t.gross_pnl for t in self.trades)
        fee_drag = (total_fees / gross_pnl * 100) if gross_pnl > 0 else 0

        return BacktestResult(
            strategy_name=self.strategy.name,
            period="backtest",
            trades=self.trades,
            equity_curve=self.equity_curve,
            total_return_pct=total_return,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_fees=total_fees,
            fee_drag_pct=fee_drag,
            btc_buy_hold_return=btc_return,
            alpha_vs_btc=total_return - btc_return
        )


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

class DailyTrendFollowing(Strategy):
    """
    Daily trend following with wide stops.

    Entry: Price breaks above N-day high AND above 50-day SMA
    Exit: 2.5x ATR trailing stop
    """

    def __init__(
        self,
        fees: Dict[str, float],
        breakout_period: int = 20,
        trend_period: int = 50,
        atr_period: int = 14,
        atr_mult: float = 2.5
    ):
        super().__init__(f"DailyTrend_B{breakout_period}_T{trend_period}_ATR{atr_mult}", fees)
        self.breakout_period = breakout_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    @property
    def min_bars(self) -> int:
        return max(self.breakout_period, self.trend_period) + 5

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        signals = []

        for symbol, df in data.items():
            if len(df) < self.min_bars:
                continue

            # Calculate indicators
            sma = calc_sma(df["close"], self.trend_period)
            atr = calc_atr(df, self.atr_period)
            high_channel, _ = calc_donchian(df, self.breakout_period)

            for i in range(self.min_bars, len(df)):
                date = df.index[i]
                close = df["close"].iloc[i]
                prev_high = high_channel.iloc[i-1]
                trend = sma.iloc[i]
                current_atr = atr.iloc[i]

                # Entry: breakout above previous high AND above trend
                signal = 0
                strength = 0.0
                stop_loss = close - (self.atr_mult * current_atr)

                if close > prev_high and close > trend:
                    signal = 1
                    strength = (close - prev_high) / prev_high  # Breakout strength

                signals.append({
                    "timestamp": date,
                    "symbol": symbol,
                    "signal": signal,
                    "strength": strength,
                    "stop_loss": stop_loss
                })

        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame(signals)
        df = df.set_index("timestamp")
        return df


class MomentumRanking(Strategy):
    """
    Cross-sectional momentum: buy assets with strongest recent performance.

    Rank assets by 20-day momentum.
    Go long top N assets.
    Exit when momentum turns negative or drops out of top rank.
    """

    def __init__(
        self,
        fees: Dict[str, float],
        lookback: int = 20,
        atr_period: int = 14,
        atr_mult: float = 2.0
    ):
        super().__init__(f"Momentum_L{lookback}_ATR{atr_mult}", fees)
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    @property
    def min_bars(self) -> int:
        return max(self.lookback, self.atr_period) + 5

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        signals = []

        # Get all common dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        for date in dates[self.min_bars:]:
            # Calculate momentum for each asset
            momentums = []

            for symbol, df in data.items():
                if date not in df.index:
                    continue

                idx = df.index.get_loc(date)
                if idx < self.lookback:
                    continue

                # Momentum = return over lookback period
                current_price = df["close"].iloc[idx]
                past_price = df["close"].iloc[idx - self.lookback]
                momentum = (current_price / past_price - 1) * 100

                # ATR for stop
                atr = calc_atr(df.iloc[:idx+1], self.atr_period).iloc[-1]
                stop_loss = current_price - (self.atr_mult * atr)

                momentums.append({
                    "symbol": symbol,
                    "momentum": momentum,
                    "stop_loss": stop_loss
                })

            # Rank by momentum - only signal positive momentum
            for m in momentums:
                signal = 1 if m["momentum"] > 0 else 0
                signals.append({
                    "timestamp": date,
                    "symbol": m["symbol"],
                    "signal": signal,
                    "strength": m["momentum"],
                    "stop_loss": m["stop_loss"]
                })

        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame(signals)
        df = df.set_index("timestamp")
        return df


class MeanReversionRSI(Strategy):
    """
    Mean reversion using RSI oversold conditions.

    Entry: RSI < 30 AND price above 200-day SMA (buy dips in uptrend)
    Exit: RSI > 70 OR tight stop loss
    """

    def __init__(
        self,
        fees: Dict[str, float],
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        trend_period: int = 200,
        atr_period: int = 14,
        atr_mult: float = 1.5
    ):
        super().__init__(f"MeanRevert_RSI{rsi_oversold}/{rsi_overbought}", fees)
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    @property
    def min_bars(self) -> int:
        return self.trend_period + 10

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        signals = []

        for symbol, df in data.items():
            if len(df) < self.min_bars:
                continue

            rsi = calc_rsi(df["close"], self.rsi_period)
            sma = calc_sma(df["close"], self.trend_period)
            atr = calc_atr(df, self.atr_period)

            for i in range(self.min_bars, len(df)):
                date = df.index[i]
                close = df["close"].iloc[i]
                current_rsi = rsi.iloc[i]
                trend = sma.iloc[i]
                current_atr = atr.iloc[i]

                signal = 0
                strength = 0.0
                stop_loss = close - (self.atr_mult * current_atr)

                # Entry: oversold in uptrend
                if current_rsi < self.rsi_oversold and close > trend:
                    signal = 1
                    strength = (self.rsi_oversold - current_rsi)  # More oversold = stronger

                signals.append({
                    "timestamp": date,
                    "symbol": symbol,
                    "signal": signal,
                    "strength": strength,
                    "stop_loss": stop_loss
                })

        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame(signals)
        df = df.set_index("timestamp")
        return df


class HybridTrendMomentum(Strategy):
    """
    Hybrid: Trend filter + Momentum ranking + Volatility sizing

    1. Only trade assets in uptrend (above 50-day SMA)
    2. Rank by momentum (20-day return)
    3. Take top assets with positive momentum
    4. Wide ATR stops
    """

    def __init__(
        self,
        fees: Dict[str, float],
        trend_period: int = 50,
        momentum_period: int = 20,
        atr_period: int = 14,
        atr_mult: float = 3.0  # Wide stops
    ):
        super().__init__(f"Hybrid_T{trend_period}_M{momentum_period}_ATR{atr_mult}", fees)
        self.trend_period = trend_period
        self.momentum_period = momentum_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    @property
    def min_bars(self) -> int:
        return max(self.trend_period, self.momentum_period) + 10

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        signals = []

        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        for date in dates[self.min_bars:]:
            candidates = []

            for symbol, df in data.items():
                if date not in df.index:
                    continue

                idx = df.index.get_loc(date)
                if idx < self.min_bars:
                    continue

                close = df["close"].iloc[idx]

                # Trend filter
                sma = calc_sma(df["close"], self.trend_period).iloc[idx]
                if close <= sma:
                    # Not in uptrend - no signal
                    signals.append({
                        "timestamp": date,
                        "symbol": symbol,
                        "signal": 0,
                        "strength": 0,
                        "stop_loss": close * 0.9
                    })
                    continue

                # Momentum
                past_price = df["close"].iloc[idx - self.momentum_period]
                momentum = (close / past_price - 1) * 100

                if momentum <= 0:
                    # Negative momentum - no signal
                    signals.append({
                        "timestamp": date,
                        "symbol": symbol,
                        "signal": 0,
                        "strength": 0,
                        "stop_loss": close * 0.9
                    })
                    continue

                # ATR stop
                atr = calc_atr(df.iloc[:idx+1], self.atr_period).iloc[-1]
                stop_loss = close - (self.atr_mult * atr)

                candidates.append({
                    "timestamp": date,
                    "symbol": symbol,
                    "signal": 1,
                    "strength": momentum,
                    "stop_loss": stop_loss
                })

            signals.extend(candidates)

        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame(signals)
        df = df.set_index("timestamp")
        return df


# =============================================================================
# MAIN
# =============================================================================

async def run_strategy_lab():
    """Run the strategy lab."""

    print("\n" + "="*70)
    print("STRATEGY LAB - Systematic Strategy Development")
    print("="*70)

    # Load data
    print("\nLoading data...")
    data = await load_all_data(
        ASSETS,
        start_date="2024-01-01",
        end_date="2025-12-17",
        granularity="ONE_DAY"
    )

    if not data:
        print("ERROR: No data loaded!")
        return

    # Define strategies to test
    strategies = [
        DailyTrendFollowing(GEMINI_FEES, breakout_period=20, atr_mult=2.5),
        DailyTrendFollowing(GEMINI_FEES, breakout_period=30, atr_mult=3.0),
        MomentumRanking(GEMINI_FEES, lookback=20, atr_mult=2.5),
        MomentumRanking(GEMINI_FEES, lookback=30, atr_mult=3.0),
        MeanReversionRSI(GEMINI_FEES, rsi_oversold=25, atr_mult=2.0),
        HybridTrendMomentum(GEMINI_FEES, trend_period=50, momentum_period=20, atr_mult=3.0),
        HybridTrendMomentum(GEMINI_FEES, trend_period=30, momentum_period=10, atr_mult=2.5),
    ]

    results = []

    # Test each strategy
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Testing: {strategy.name}")
        print("="*70)

        # Train period
        print("\n--- TRAIN PERIOD (Jan-Aug 2024) ---")
        bt_train = Backtester(
            strategy,
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_POSITIONS,
            max_position_size=MAX_POSITION_SIZE,
            max_drawdown=MAX_PORTFOLIO_DRAWDOWN
        )
        train_result = bt_train.run(data, end_date=TRAIN_END)
        train_result.period = "train"
        print(train_result.summary())

        # Test period
        print("\n--- TEST PERIOD (Sep 2024 - Dec 2025) ---")
        bt_test = Backtester(
            strategy,
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_POSITIONS,
            max_position_size=MAX_POSITION_SIZE,
            max_drawdown=MAX_PORTFOLIO_DRAWDOWN
        )
        test_result = bt_test.run(data, start_date=TEST_START)
        test_result.period = "test"
        print(test_result.summary())

        results.append((strategy.name, train_result, test_result))

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Strategy':<45} {'Train':>10} {'Test':>10} {'Pass':>6}")
    print("-"*70)

    for name, train, test in results:
        train_pass, _ = train.passes_criteria()
        test_pass, _ = test.passes_criteria()
        overall = "YES" if (train_pass and test_pass) else "NO"
        print(f"{name:<45} {train.total_return_pct:>+9.1f}% {test.total_return_pct:>+9.1f}% {overall:>6}")

    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_strategy_lab())
