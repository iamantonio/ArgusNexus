"""
Backtesting Engine with Coinbase Fee Model

Designed for realistic backtesting that accounts for Coinbase's retail fee structure.
All results show both GROSS and NET P&L to verify fee impact.

Fee Model (Hybrid Execution):
- Entry: 0.6% (Market/Taker) + 0.1% slippage = 0.7%
- Exit: 0.4% (Limit/Maker) = 0.4%
- Total Round-Trip: 1.1%

Usage:
    engine = BacktestEngine(strategy, fee_model='coinbase_hybrid')
    result = engine.run(df)
    print(result.summary())
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
from enum import Enum
import pandas as pd
import numpy as np


class FeeModel(Enum):
    """Fee model configurations"""
    COINBASE_HYBRID = "coinbase_hybrid"     # Market entry, Limit exit
    COINBASE_TAKER = "coinbase_taker"       # All market orders
    COINBASE_MAKER = "coinbase_maker"       # All limit orders
    GEMINI_HYBRID = "gemini_hybrid"         # Gemini ActiveTrader (market entry, limit exit)
    KRAKEN_HYBRID = "kraken_hybrid"         # Kraken Pro (market entry, limit exit)
    ZERO = "zero"                           # For comparison


FEE_CONFIGS = {
    FeeModel.COINBASE_HYBRID: {
        "entry_fee": Decimal("0.006"),       # 0.6% taker
        "entry_slippage": Decimal("0.001"),  # 0.1% slippage
        "exit_fee": Decimal("0.004"),        # 0.4% maker
        "exit_slippage": Decimal("0.0"),     # No slippage on limit
    },
    FeeModel.COINBASE_TAKER: {
        "entry_fee": Decimal("0.006"),
        "entry_slippage": Decimal("0.001"),
        "exit_fee": Decimal("0.006"),
        "exit_slippage": Decimal("0.001"),
    },
    FeeModel.COINBASE_MAKER: {
        "entry_fee": Decimal("0.004"),
        "entry_slippage": Decimal("0.0"),
        "exit_fee": Decimal("0.004"),
        "exit_slippage": Decimal("0.0"),
    },
    FeeModel.GEMINI_HYBRID: {
        # Gemini ActiveTrader (Tier 0: <$10K volume)
        "entry_fee": Decimal("0.004"),       # 0.40% taker
        "entry_slippage": Decimal("0.001"),  # 0.1% slippage
        "exit_fee": Decimal("0.002"),        # 0.20% maker
        "exit_slippage": Decimal("0.0"),     # No slippage on limit
    },
    FeeModel.KRAKEN_HYBRID: {
        # Kraken Pro (Tier 0: <$10K volume)
        "entry_fee": Decimal("0.004"),       # 0.40% taker
        "entry_slippage": Decimal("0.001"),  # 0.1% slippage
        "exit_fee": Decimal("0.0025"),       # 0.25% maker
        "exit_slippage": Decimal("0.0"),     # No slippage on limit
    },
    FeeModel.ZERO: {
        "entry_fee": Decimal("0.0"),
        "entry_slippage": Decimal("0.0"),
        "exit_fee": Decimal("0.0"),
        "exit_slippage": Decimal("0.0"),
    },
}


@dataclass
class Trade:
    """Record of a completed trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    position_size: Decimal
    side: str  # "long" or "short"

    # Gross P&L (before fees)
    gross_pnl: Decimal
    gross_pnl_pct: Decimal

    # Costs
    entry_cost: Decimal
    exit_cost: Decimal
    total_cost: Decimal

    # Net P&L (after fees)
    net_pnl: Decimal
    net_pnl_pct: Decimal

    # Metadata
    exit_reason: str
    bars_held: int
    highest_high: Optional[Decimal] = None
    lowest_low: Optional[Decimal] = None
    max_favorable_excursion: Optional[Decimal] = None
    max_adverse_excursion: Optional[Decimal] = None


@dataclass
class BacktestResult:
    """Complete backtest results with metrics"""
    # Trade list
    trades: List[Trade]

    # Summary metrics
    total_trades: int
    winners: int
    losers: int
    win_rate: Decimal

    # P&L metrics
    gross_profit: Decimal
    gross_loss: Decimal
    total_fees: Decimal
    net_profit: Decimal

    # Ratios
    profit_factor: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    avg_win_loss_ratio: Decimal
    expectancy: Decimal

    # Risk metrics
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    sharpe_ratio: Optional[Decimal]

    # Time metrics
    avg_bars_held: Decimal
    total_bars: int
    time_in_market_pct: Decimal

    # Configuration
    initial_capital: Decimal
    final_capital: Decimal
    total_return_pct: Decimal
    fee_model: str

    def summary(self) -> str:
        """Generate human-readable summary"""
        passed_checks = []
        failed_checks = []

        # Check validation gates
        if self.net_profit > 0:
            passed_checks.append(f"Net Profit: ${float(self.net_profit):.2f}")
        else:
            failed_checks.append(f"Net Profit: ${float(self.net_profit):.2f} (NEED > $0)")

        if self.profit_factor > Decimal("1.6"):
            passed_checks.append(f"Profit Factor: {float(self.profit_factor):.2f}")
        else:
            failed_checks.append(f"Profit Factor: {float(self.profit_factor):.2f} (NEED > 1.6)")

        if self.max_drawdown_pct < Decimal("20"):
            passed_checks.append(f"Max Drawdown: {float(self.max_drawdown_pct):.1f}%")
        else:
            failed_checks.append(f"Max Drawdown: {float(self.max_drawdown_pct):.1f}% (NEED < 20%)")

        if self.win_rate > Decimal("40"):
            passed_checks.append(f"Win Rate: {float(self.win_rate):.1f}%")
        else:
            failed_checks.append(f"Win Rate: {float(self.win_rate):.1f}% (NEED > 40%)")

        if self.total_trades >= 20:
            passed_checks.append(f"Sample Size: {self.total_trades} trades")
        else:
            failed_checks.append(f"Sample Size: {self.total_trades} trades (NEED >= 20)")

        status = "PASS" if len(failed_checks) == 0 else "FAIL"

        return f"""
================================================================================
BACKTEST RESULTS - {status}
================================================================================
Fee Model: {self.fee_model}
Period: {self.total_bars} bars
Initial Capital: ${float(self.initial_capital):,.2f}
Final Capital: ${float(self.final_capital):,.2f}
Total Return: {float(self.total_return_pct):.2f}%

--- TRADE STATISTICS ---
Total Trades: {self.total_trades}
Winners: {self.winners} | Losers: {self.losers}
Win Rate: {float(self.win_rate):.1f}%
Avg Win: ${float(self.avg_win):.2f} ({float(self.avg_win / self.initial_capital * 100):.2f}%)
Avg Loss: ${float(self.avg_loss):.2f} ({float(self.avg_loss / self.initial_capital * 100):.2f}%)
Avg Win/Loss Ratio: {float(self.avg_win_loss_ratio):.2f}
Avg Bars Held: {float(self.avg_bars_held):.1f}
Time in Market: {float(self.time_in_market_pct):.1f}%

--- P&L BREAKDOWN ---
Gross Profit: ${float(self.gross_profit):,.2f}
Gross Loss: ${float(self.gross_loss):,.2f}
Total Fees: ${float(self.total_fees):,.2f}
Net Profit: ${float(self.net_profit):,.2f}

--- RISK METRICS ---
Profit Factor: {float(self.profit_factor):.2f}
Max Drawdown: ${float(self.max_drawdown):,.2f} ({float(self.max_drawdown_pct):.1f}%)
Expectancy: ${float(self.expectancy):.2f} per trade
Sharpe Ratio: {f'{float(self.sharpe_ratio):.2f}' if self.sharpe_ratio is not None else 'N/A'}

--- VALIDATION GATES ---
PASSED:
{chr(10).join('  [+] ' + c for c in passed_checks) if passed_checks else '  (none)'}

FAILED:
{chr(10).join('  [-] ' + c for c in failed_checks) if failed_checks else '  (none)'}

STATUS: {status}
================================================================================
"""

    def passed_all_gates(self) -> bool:
        """Check if all validation gates passed"""
        return (
            self.net_profit > 0 and
            self.profit_factor > Decimal("1.6") and
            self.max_drawdown_pct < Decimal("20") and
            self.win_rate > Decimal("40") and
            self.total_trades >= 20
        )


class BacktestEngine:
    """
    Event-driven backtesting engine with Coinbase fee model.
    """

    def __init__(
        self,
        strategy,
        fee_model: FeeModel = FeeModel.COINBASE_HYBRID,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.30
    ):
        self.strategy = strategy
        self.fee_model = fee_model
        self.fee_config = FEE_CONFIGS[fee_model]
        self.initial_capital = Decimal(str(initial_capital))
        self.risk_per_trade = Decimal(str(risk_per_trade))
        self.max_position_pct = Decimal(str(max_position_pct))

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on provided data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be datetime

        Returns:
            BacktestResult with full metrics
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')

        trades: List[Trade] = []
        capital = self.initial_capital
        equity_curve = [float(capital)]

        # Position state
        in_position = False
        is_short_position = False  # v6.1: SHORT support
        entry_price: Optional[Decimal] = None
        entry_time: Optional[datetime] = None
        position_size: Optional[Decimal] = None
        stop_loss: Optional[Decimal] = None
        highest_high_since_entry: Optional[Decimal] = None
        lowest_low_since_entry: Optional[Decimal] = None
        entry_bar_idx: int = 0
        bars_in_position = 0

        # Iterate through bars
        min_bars = self.strategy.min_bars
        for i in range(min_bars, len(df)):
            current_bar = df.iloc[i]
            historical_df = df.iloc[:i+1]
            bar_time = df.index[i]
            current_price = Decimal(str(current_bar['close']))
            current_high = Decimal(str(current_bar['high']))
            current_low = Decimal(str(current_bar['low']))

            # Update position tracking
            if in_position:
                bars_in_position += 1
                if current_high > highest_high_since_entry:
                    highest_high_since_entry = current_high
                if current_low < lowest_low_since_entry:
                    lowest_low_since_entry = current_low

            # Get signal
            signal_result = self.strategy.evaluate(
                historical_df,
                timestamp=bar_time,
                has_open_position=in_position,
                entry_price=entry_price,
                highest_high_since_entry=highest_high_since_entry,
                # v6.1: SHORT support
                is_short_position=is_short_position,
                lowest_low_since_entry=lowest_low_since_entry
            )

            signal = signal_result.signal.value

            # Process signals
            if signal == "long" and not in_position:
                # Calculate position size
                stop_loss = signal_result.context.stop_loss_price
                risk_amount = current_price - stop_loss
                if risk_amount > 0:
                    risk_capital = capital * self.risk_per_trade
                    position_size = risk_capital / risk_amount

                    # Cap at max position
                    max_size = (capital * self.max_position_pct) / current_price
                    position_size = min(position_size, max_size)

                    # Apply entry cost
                    entry_cost_rate = self.fee_config["entry_fee"] + self.fee_config["entry_slippage"]
                    entry_cost = current_price * position_size * entry_cost_rate

                    # FIXED: Deduct entry cost from capital (was missing - inflated backtest P&L)
                    capital -= entry_cost

                    # Enter position
                    in_position = True
                    entry_price = current_price
                    entry_time = bar_time
                    entry_bar_idx = i
                    highest_high_since_entry = current_high
                    lowest_low_since_entry = current_low
                    bars_in_position = 0

            # v6.1: SHORT entry handling
            elif signal == "short" and not in_position:
                # Calculate position size (stop is ABOVE entry for shorts)
                stop_loss = signal_result.context.stop_loss_price
                risk_amount = stop_loss - current_price  # Inverse: stop above entry
                if risk_amount > 0:
                    risk_capital = capital * self.risk_per_trade
                    position_size = risk_capital / risk_amount

                    # Cap at max position (50% of normal for shorts - conservative)
                    max_size = (capital * self.max_position_pct * Decimal("0.5")) / current_price
                    position_size = min(position_size, max_size)

                    # Apply entry cost
                    entry_cost_rate = self.fee_config["entry_fee"] + self.fee_config["entry_slippage"]
                    entry_cost = current_price * position_size * entry_cost_rate
                    capital -= entry_cost

                    # Enter SHORT position
                    in_position = True
                    is_short_position = True
                    entry_price = current_price
                    entry_time = bar_time
                    entry_bar_idx = i
                    highest_high_since_entry = current_high
                    lowest_low_since_entry = current_low
                    bars_in_position = 0

            elif signal == "exit_long" and in_position and not is_short_position:
                # Calculate exit
                exit_price = current_price
                exit_time = bar_time
                exit_reason = signal_result.context.exit_reason.value

                # Gross P&L
                gross_pnl = (exit_price - entry_price) * position_size
                gross_pnl_pct = (exit_price - entry_price) / entry_price * 100

                # Costs
                entry_cost_rate = self.fee_config["entry_fee"] + self.fee_config["entry_slippage"]
                exit_cost_rate = self.fee_config["exit_fee"] + self.fee_config["exit_slippage"]
                entry_cost = entry_price * position_size * entry_cost_rate
                exit_cost = exit_price * position_size * exit_cost_rate
                total_cost = entry_cost + exit_cost

                # Net P&L
                net_pnl = gross_pnl - total_cost
                net_pnl_pct = net_pnl / (entry_price * position_size) * 100

                # MFE/MAE
                mfe = (highest_high_since_entry - entry_price) / entry_price * 100
                mae = (entry_price - lowest_low_since_entry) / entry_price * 100

                # Record trade
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size=position_size,
                    side="long",
                    gross_pnl=gross_pnl,
                    gross_pnl_pct=gross_pnl_pct,
                    entry_cost=entry_cost,
                    exit_cost=exit_cost,
                    total_cost=total_cost,
                    net_pnl=net_pnl,
                    net_pnl_pct=net_pnl_pct,
                    exit_reason=exit_reason,
                    bars_held=bars_in_position,
                    highest_high=highest_high_since_entry,
                    lowest_low=lowest_low_since_entry,
                    max_favorable_excursion=mfe,
                    max_adverse_excursion=mae
                )
                trades.append(trade)

                # Update capital
                capital += net_pnl

                # Reset position state
                in_position = False
                is_short_position = False
                entry_price = None
                entry_time = None
                position_size = None
                stop_loss = None
                highest_high_since_entry = None
                lowest_low_since_entry = None

            # v6.1: SHORT exit handling
            elif signal == "exit_short" and in_position and is_short_position:
                # Calculate exit
                exit_price = current_price
                exit_time = bar_time
                exit_reason = signal_result.context.exit_reason.value

                # Gross P&L (INVERSE for shorts: profit when price drops)
                gross_pnl = (entry_price - exit_price) * position_size
                gross_pnl_pct = (entry_price - exit_price) / entry_price * 100

                # Costs
                entry_cost_rate = self.fee_config["entry_fee"] + self.fee_config["entry_slippage"]
                exit_cost_rate = self.fee_config["exit_fee"] + self.fee_config["exit_slippage"]
                entry_cost = entry_price * position_size * entry_cost_rate
                exit_cost = exit_price * position_size * exit_cost_rate
                total_cost = entry_cost + exit_cost

                # Net P&L
                net_pnl = gross_pnl - total_cost
                net_pnl_pct = net_pnl / (entry_price * position_size) * 100

                # MFE/MAE (INVERSE for shorts: MFE is lowest low, MAE is highest high)
                mfe = (entry_price - lowest_low_since_entry) / entry_price * 100
                mae = (highest_high_since_entry - entry_price) / entry_price * 100

                # Record trade
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size=position_size,
                    side="short",
                    gross_pnl=gross_pnl,
                    gross_pnl_pct=gross_pnl_pct,
                    entry_cost=entry_cost,
                    exit_cost=exit_cost,
                    total_cost=total_cost,
                    net_pnl=net_pnl,
                    net_pnl_pct=net_pnl_pct,
                    exit_reason=exit_reason,
                    bars_held=bars_in_position,
                    highest_high=highest_high_since_entry,
                    lowest_low=lowest_low_since_entry,
                    max_favorable_excursion=mfe,
                    max_adverse_excursion=mae
                )
                trades.append(trade)

                # Update capital
                capital += net_pnl

                # Reset position state
                in_position = False
                is_short_position = False
                entry_price = None
                entry_time = None
                position_size = None
                stop_loss = None
                highest_high_since_entry = None
                lowest_low_since_entry = None

            # Track equity
            if in_position:
                # v6.1: Inverse unrealized P&L for shorts
                if is_short_position:
                    unrealized = (entry_price - current_price) * position_size
                else:
                    unrealized = (current_price - entry_price) * position_size
                equity_curve.append(float(capital + unrealized))
            else:
                equity_curve.append(float(capital))

        # Calculate results
        return self._calculate_results(trades, equity_curve, len(df))

    def _calculate_results(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        total_bars: int
    ) -> BacktestResult:
        """Calculate backtest metrics from trade list"""

        if not trades:
            return BacktestResult(
                trades=[],
                total_trades=0,
                winners=0,
                losers=0,
                win_rate=Decimal("0"),
                gross_profit=Decimal("0"),
                gross_loss=Decimal("0"),
                total_fees=Decimal("0"),
                net_profit=Decimal("0"),
                profit_factor=Decimal("0"),
                avg_win=Decimal("0"),
                avg_loss=Decimal("0"),
                avg_win_loss_ratio=Decimal("0"),
                expectancy=Decimal("0"),
                max_drawdown=Decimal("0"),
                max_drawdown_pct=Decimal("0"),
                sharpe_ratio=None,
                avg_bars_held=Decimal("0"),
                total_bars=total_bars,
                time_in_market_pct=Decimal("0"),
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return_pct=Decimal("0"),
                fee_model=self.fee_model.value
            )

        # Basic counts
        total_trades = len(trades)
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]
        win_count = len(winners)
        loss_count = len(losers)
        win_rate = Decimal(str(win_count / total_trades * 100)) if total_trades > 0 else Decimal("0")

        # P&L
        gross_profit = sum(t.gross_pnl for t in winners) if winners else Decimal("0")
        gross_loss = abs(sum(t.gross_pnl for t in losers)) if losers else Decimal("0")
        total_fees = sum(t.total_cost for t in trades)
        net_profit = sum(t.net_pnl for t in trades)

        # Averages
        avg_win = gross_profit / win_count if win_count > 0 else Decimal("0")
        avg_loss = gross_loss / loss_count if loss_count > 0 else Decimal("0")
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else Decimal("0")

        # Profit factor
        net_wins = sum(t.net_pnl for t in winners) if winners else Decimal("0")
        net_losses = abs(sum(t.net_pnl for t in losers)) if losers else Decimal("0")
        profit_factor = net_wins / net_losses if net_losses > 0 else Decimal("999")

        # Expectancy
        expectancy = net_profit / total_trades if total_trades > 0 else Decimal("0")

        # Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = rolling_max - equity_series
        max_drawdown = Decimal(str(drawdown.max()))
        max_drawdown_pct = Decimal(str((drawdown / rolling_max).max() * 100))

        # Sharpe (simplified - daily returns assumed)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = Decimal(str((returns.mean() / returns.std()) * np.sqrt(252)))
            else:
                sharpe = None
        else:
            sharpe = None

        # Time metrics
        total_bars_held = sum(t.bars_held for t in trades)
        avg_bars_held = Decimal(str(total_bars_held / total_trades)) if total_trades > 0 else Decimal("0")
        time_in_market_pct = Decimal(str(total_bars_held / total_bars * 100)) if total_bars > 0 else Decimal("0")

        # Capital
        final_capital = self.initial_capital + net_profit
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        return BacktestResult(
            trades=trades,
            total_trades=total_trades,
            winners=win_count,
            losers=loss_count,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            total_fees=total_fees,
            net_profit=net_profit,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_loss_ratio=avg_win_loss_ratio,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            avg_bars_held=avg_bars_held,
            total_bars=total_bars,
            time_in_market_pct=time_in_market_pct,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return_pct,
            fee_model=self.fee_model.value
        )


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly or sub-daily data to daily candles.

    Args:
        df: DataFrame with OHLCV columns and datetime index

    Returns:
        Daily OHLCV DataFrame
    """
    df = df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')

    # Aggregate to daily
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return daily
