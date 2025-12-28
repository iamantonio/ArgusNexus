#!/usr/bin/env python3
"""
Multi-Asset Portfolio Backtester - Professional Turtle-Style

Tests the multi-asset breakout strategy with:
- Watchlist scanning (BTC, ETH, SOL, AVAX, LINK)
- Volatility-weighted position sizing
- Maximum 3 concurrent positions
- 30% capital per position
- Gemini fee model

Usage:
    python scripts/backtest_multi_asset.py --capital 500 --start 2024-01-01 --end 2024-12-15
"""

import sys
import argparse
import asyncio
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import aiohttp
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.coinbase_survivor import CoinbaseSurvivorStrategy, Signal


@dataclass
class Position:
    """Track an open position"""
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    trailing_stop: float
    highest_high: float
    capital_allocated: float


@dataclass
class ClosedTrade:
    """Record of a completed trade"""
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
class PortfolioResult:
    """Multi-asset backtest results"""
    trades: List[ClosedTrade]
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    gross_pnl: float
    total_fees: float
    net_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: Optional[float]
    trades_per_asset: Dict[str, int]
    pnl_per_asset: Dict[str, float]

    def summary(self) -> str:
        status = "PASS" if self.net_pnl > 0 and self.win_rate > 40 else "FAIL"

        asset_breakdown = "\n".join([
            f"    {sym}: {count} trades, ${self.pnl_per_asset.get(sym, 0):.2f} P&L"
            for sym, count in sorted(self.trades_per_asset.items(), key=lambda x: -x[1])
        ])

        return f"""
================================================================================
MULTI-ASSET PORTFOLIO BACKTEST - {status}
================================================================================
Initial Capital: ${self.initial_capital:,.2f}
Final Capital: ${self.final_capital:,.2f}
Total Return: {self.total_return_pct:.2f}%

--- TRADE STATISTICS ---
Total Trades: {self.total_trades}
Winners: {self.winners} | Losers: {self.losers}
Win Rate: {self.win_rate:.1f}%

--- P&L BREAKDOWN ---
Gross P&L: ${self.gross_pnl:,.2f}
Total Fees: ${self.total_fees:,.2f}
Net P&L: ${self.net_pnl:,.2f}

--- RISK METRICS ---
Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.1f}%)
Sharpe Ratio: {f'{self.sharpe_ratio:.2f}' if self.sharpe_ratio else 'N/A'}

--- PER-ASSET BREAKDOWN ---
{asset_breakdown}

================================================================================
"""


class MultiAssetBacktester:
    """
    Professional multi-asset backtester with Turtle-style rules.

    Rules:
    - Scan watchlist for breakout signals
    - Max 3 concurrent positions
    - 30% capital per position (90% max deployed)
    - Volatility-weighted sizing (normalize by ATR)
    - ATR trailing stops
    """

    WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"]
    MAX_POSITIONS = 3
    CAPITAL_PER_POSITION = 0.30

    # Gemini fee model
    ENTRY_FEE = 0.004   # 0.40% taker
    EXIT_FEE = 0.002    # 0.20% maker
    SLIPPAGE = 0.001    # 0.10%

    def __init__(self, initial_capital: float = 500.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.equity_curve: List[float] = [initial_capital]

        # Strategy instance (shared params)
        self.strategy = CoinbaseSurvivorStrategy(
            breakout_period=15,
            trend_period=50,
            atr_period=14,
            trailing_atr_mult=1.5
        )

    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        granularity: str = "FOUR_HOUR"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from Coinbase."""
        url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles"

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

        all_candles = []
        current_end = end_ts

        # Adjust time window based on granularity
        if granularity == "FOUR_HOUR":
            time_window = 300 * 4 * 3600  # 300 4-hour bars
        else:
            time_window = 300 * 86400  # 300 daily bars

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
                        print(f"Error fetching {symbol}: {response.status}")
                        break

                    data = await response.json()
                    candles = data.get("candles", [])

                    if not candles:
                        break

                    all_candles.extend(candles)
                    current_end = int(candles[-1]["start"]) - 1

                    await asyncio.sleep(0.1)  # Rate limiting

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        df = df.set_index("timestamp")

        return df[["open", "high", "low", "close", "volume"]]

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period + 1:
            return df["high"].iloc[-1] - df["low"].iloc[-1]

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        return float(np.mean(tr[-period:]))

    def calculate_position_size(
        self,
        price: float,
        atr: float,
        available_capital: float
    ) -> float:
        """
        Calculate volatility-weighted position size.

        Uses ATR to normalize position sizes across assets.
        Higher volatility = smaller position.
        """
        # Base allocation
        base_allocation = available_capital * self.CAPITAL_PER_POSITION

        # Volatility adjustment (target 2% daily move risk)
        target_volatility = 0.02
        actual_volatility = atr / price
        vol_adjustment = min(1.0, target_volatility / actual_volatility) if actual_volatility > 0 else 1.0

        adjusted_allocation = base_allocation * vol_adjustment

        # Calculate quantity
        quantity = adjusted_allocation / price

        return quantity

    def get_available_capital(self) -> float:
        """Calculate capital available for new positions."""
        allocated = sum(p.capital_allocated for p in self.positions.values())
        return self.capital - allocated

    async def run(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-15"
    ) -> PortfolioResult:
        """Run multi-asset backtest."""
        print(f"\n{'='*60}")
        print("MULTI-ASSET PORTFOLIO BACKTEST")
        print(f"{'='*60}")
        print(f"Capital: ${self.initial_capital:,.2f}")
        print(f"Watchlist: {', '.join(self.WATCHLIST)}")
        print(f"Max Positions: {self.MAX_POSITIONS}")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*60}\n")

        # Fetch data for all assets
        print("Fetching historical data...")
        asset_data: Dict[str, pd.DataFrame] = {}

        for symbol in self.WATCHLIST:
            print(f"  Loading {symbol}...", end=" ")
            df = await self.fetch_historical_data(symbol, start_date, end_date)
            if not df.empty:
                asset_data[symbol] = df
                print(f"{len(df)} bars")
            else:
                print("FAILED")

        if not asset_data:
            print("ERROR: No data loaded!")
            return self._generate_result()

        # Align all data to common dates
        all_dates = set()
        for df in asset_data.values():
            all_dates.update(df.index.tolist())

        common_dates = sorted(all_dates)
        print(f"\nBacktesting across {len(common_dates)} days...")

        # Main backtest loop
        min_history = 50  # Need 50 bars for indicators

        for i, date in enumerate(common_dates[min_history:], start=min_history):
            # Update existing positions (check stops)
            await self._update_positions(date, asset_data)

            # Scan for new opportunities if we have capacity
            if len(self.positions) < self.MAX_POSITIONS:
                await self._scan_for_entries(date, i, asset_data)

            # Update equity curve
            self._update_equity(date, asset_data)

            if (i - min_history) % 50 == 0:
                print(f"  Day {i - min_history}/{len(common_dates) - min_history} | "
                      f"Positions: {len(self.positions)} | "
                      f"Trades: {len(self.closed_trades)} | "
                      f"Equity: ${self.equity_curve[-1]:,.2f}")

        # Close any remaining positions
        await self._close_all_positions(common_dates[-1], asset_data)

        return self._generate_result()

    async def _update_positions(
        self,
        date: datetime,
        asset_data: Dict[str, pd.DataFrame]
    ):
        """Check and update existing positions."""
        positions_to_close = []

        for symbol, pos in self.positions.items():
            if symbol not in asset_data:
                continue

            df = asset_data[symbol]
            if date not in df.index:
                continue

            bar = df.loc[date]
            current_price = float(bar["close"])
            current_high = float(bar["high"])
            current_low = float(bar["low"])

            # Update highest high
            if current_high > pos.highest_high:
                pos.highest_high = current_high
                # Update trailing stop
                atr = self.calculate_atr(df.loc[:date], 14)
                pos.trailing_stop = pos.highest_high - (1.5 * atr)

            # Check stop loss
            exit_reason = None
            exit_price = current_price

            if current_low <= pos.stop_loss:
                exit_reason = "initial_stop"
                exit_price = pos.stop_loss
            elif current_low <= pos.trailing_stop:
                exit_reason = "trailing_stop"
                exit_price = pos.trailing_stop

            if exit_reason:
                positions_to_close.append((symbol, exit_price, exit_reason, date))

        # Close positions
        for symbol, exit_price, reason, exit_date in positions_to_close:
            self._close_position(symbol, exit_price, reason, exit_date)

    async def _scan_for_entries(
        self,
        date: datetime,
        bar_idx: int,
        asset_data: Dict[str, pd.DataFrame]
    ):
        """Scan watchlist for entry signals."""
        signals = []

        for symbol in self.WATCHLIST:
            # Skip if already in position
            if symbol in self.positions:
                continue

            if symbol not in asset_data:
                continue

            df = asset_data[symbol]
            if date not in df.index:
                continue

            # Get historical data up to this point
            historical = df.loc[:date]
            if len(historical) < self.strategy.min_bars:
                continue

            # Evaluate signal
            result = self.strategy.evaluate(historical)

            if result.signal == Signal.LONG:
                # Calculate signal strength (how far above breakout)
                current_price = float(historical["close"].iloc[-1])
                breakout_level = float(historical["high"].rolling(15).max().iloc[-2])
                strength = (current_price - breakout_level) / breakout_level

                signals.append({
                    "symbol": symbol,
                    "price": current_price,
                    "strength": strength,
                    "atr": self.calculate_atr(historical, 14),
                    "stop_loss": result.context.stop_loss_price,
                    "date": date
                })

        # Sort by strength (strongest breakout first)
        signals.sort(key=lambda x: x["strength"], reverse=True)

        # Take positions up to max
        available_slots = self.MAX_POSITIONS - len(self.positions)
        available_capital = self.get_available_capital()

        for signal in signals[:available_slots]:
            if available_capital < self.initial_capital * 0.10:  # Keep 10% reserve
                break

            self._open_position(
                symbol=signal["symbol"],
                price=signal["price"],
                atr=signal["atr"],
                stop_loss=float(signal["stop_loss"]),
                date=signal["date"],
                available_capital=available_capital
            )

            available_capital = self.get_available_capital()

    def _open_position(
        self,
        symbol: str,
        price: float,
        atr: float,
        stop_loss: float,
        date: datetime,
        available_capital: float
    ):
        """Open a new position."""
        quantity = self.calculate_position_size(price, atr, available_capital)

        # Apply entry costs
        entry_cost = price * quantity * (self.ENTRY_FEE + self.SLIPPAGE)
        fill_price = price * (1 + self.SLIPPAGE)

        capital_used = fill_price * quantity + entry_cost

        position = Position(
            symbol=symbol,
            entry_time=date,
            entry_price=fill_price,
            quantity=quantity,
            stop_loss=stop_loss,
            trailing_stop=price - (1.5 * atr),
            highest_high=price,
            capital_allocated=capital_used
        )

        self.positions[symbol] = position
        print(f"  ENTRY: {symbol} @ ${fill_price:,.2f} | Qty: {quantity:.6f} | Stop: ${stop_loss:,.2f}")

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
        exit_date: datetime
    ):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Calculate P&L
        gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        entry_fees = pos.entry_price * pos.quantity * (self.ENTRY_FEE + self.SLIPPAGE)
        exit_fees = exit_price * pos.quantity * self.EXIT_FEE
        total_fees = entry_fees + exit_fees
        net_pnl = gross_pnl - total_fees

        # Calculate bars held
        bars_held = (exit_date - pos.entry_time).days

        # Record trade
        trade = ClosedTrade(
            symbol=symbol,
            entry_time=pos.entry_time,
            exit_time=exit_date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            gross_pnl=gross_pnl,
            fees=total_fees,
            net_pnl=net_pnl,
            exit_reason=reason,
            bars_held=bars_held
        )
        self.closed_trades.append(trade)

        # Update capital
        self.capital += net_pnl

        # Remove position
        del self.positions[symbol]

        result = "WIN" if net_pnl > 0 else "LOSS"
        print(f"  EXIT: {symbol} @ ${exit_price:,.2f} | {reason} | {result} ${net_pnl:+,.2f}")

    async def _close_all_positions(
        self,
        date: datetime,
        asset_data: Dict[str, pd.DataFrame]
    ):
        """Close all remaining positions at end of backtest."""
        for symbol in list(self.positions.keys()):
            if symbol in asset_data and date in asset_data[symbol].index:
                price = float(asset_data[symbol].loc[date, "close"])
                self._close_position(symbol, price, "end_of_backtest", date)

    def _update_equity(self, date: datetime, asset_data: Dict[str, pd.DataFrame]):
        """Update equity curve with unrealized P&L."""
        unrealized = 0.0

        for symbol, pos in self.positions.items():
            if symbol in asset_data and date in asset_data[symbol].index:
                current_price = float(asset_data[symbol].loc[date, "close"])
                unrealized += (current_price - pos.entry_price) * pos.quantity

        self.equity_curve.append(self.capital + unrealized)

    def _generate_result(self) -> PortfolioResult:
        """Generate final backtest results."""
        if not self.closed_trades:
            return PortfolioResult(
                trades=[],
                total_trades=0,
                winners=0,
                losers=0,
                win_rate=0.0,
                gross_pnl=0.0,
                total_fees=0.0,
                net_pnl=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                initial_capital=self.initial_capital,
                final_capital=self.capital,
                total_return_pct=0.0,
                sharpe_ratio=None,
                trades_per_asset={},
                pnl_per_asset={}
            )

        # Basic stats
        total_trades = len(self.closed_trades)
        winners = len([t for t in self.closed_trades if t.net_pnl > 0])
        losers = total_trades - winners
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        # P&L
        gross_pnl = sum(t.gross_pnl for t in self.closed_trades)
        total_fees = sum(t.fees for t in self.closed_trades)
        net_pnl = sum(t.net_pnl for t in self.closed_trades)

        # Drawdown
        equity = pd.Series(self.equity_curve)
        rolling_max = equity.cummax()
        drawdown = rolling_max - equity
        max_drawdown = float(drawdown.max())
        max_drawdown_pct = float((drawdown / rolling_max).max() * 100)

        # Sharpe
        returns = equity.pct_change().dropna()
        sharpe = None
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))

        # Per-asset breakdown
        trades_per_asset: Dict[str, int] = {}
        pnl_per_asset: Dict[str, float] = {}

        for trade in self.closed_trades:
            sym = trade.symbol
            trades_per_asset[sym] = trades_per_asset.get(sym, 0) + 1
            pnl_per_asset[sym] = pnl_per_asset.get(sym, 0.0) + trade.net_pnl

        return PortfolioResult(
            trades=self.closed_trades,
            total_trades=total_trades,
            winners=winners,
            losers=losers,
            win_rate=win_rate,
            gross_pnl=gross_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_return_pct=((self.capital - self.initial_capital) / self.initial_capital * 100),
            sharpe_ratio=sharpe,
            trades_per_asset=trades_per_asset,
            pnl_per_asset=pnl_per_asset
        )


async def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Portfolio Backtester")
    parser.add_argument("--capital", type=float, default=500.0, help="Starting capital")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-15", help="End date")
    args = parser.parse_args()

    backtester = MultiAssetBacktester(initial_capital=args.capital)
    result = await backtester.run(start_date=args.start, end_date=args.end)

    print(result.summary())

    # Verdict
    if result.net_pnl > 0 and result.win_rate > 40:
        print("VERDICT: STRATEGY VIABLE FOR DEPLOYMENT")
    else:
        print("VERDICT: NEEDS OPTIMIZATION")


if __name__ == "__main__":
    asyncio.run(main())
