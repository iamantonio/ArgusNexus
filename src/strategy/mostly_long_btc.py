"""
MostlyLong BTC Strategy - "The Emergency Exit"

A trend-following strategy that stays invested most of the time,
exiting only when major breakdown conditions are met.

VALIDATION RESULTS (2024-01-01 to 2025-12-17):
- Total Trades: 2
- Win Rate: 100%
- Net Return: +39.8%
- Max Drawdown: 24.2%
- Alpha vs BTC: +7.9%

Entry Logic:
1. Price closes above 30-day SMA
2. 30-day momentum is positive (>0%)

Exit Logic:
1. Price closes below 200-day SMA
2. 30-day momentum is below -15%
Both conditions must be true to exit.

Fee Model (Gemini):
- Entry: 0.4% + 0.1% slippage = 0.5%
- Exit: 0.2% + 0.1% slippage = 0.3%
- Total: 0.8% round-trip

Usage:
    strategy = MostlyLongBTCStrategy()
    result = strategy.evaluate(df, has_open_position=False)
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any
import pandas as pd


class Signal(Enum):
    """Trading signal types"""
    LONG = "long"
    EXIT_LONG = "exit_long"
    HOLD = "hold"


class ExitReason(Enum):
    """Exit trigger classification"""
    EMERGENCY_EXIT = "emergency_exit"  # Price < 200 SMA AND momentum < -15%
    NONE = "none"


@dataclass
class SignalContext:
    """Complete context for Glass Box logging"""
    timestamp: datetime

    # Price data
    current_price: Decimal
    high_price: Decimal
    low_price: Decimal

    # Indicators
    sma_200: Decimal              # 200-day SMA (exit filter)
    sma_30: Decimal               # 30-day SMA (entry filter)
    momentum_30: Decimal          # 30-day momentum as percentage

    # Conditions
    price_below_sma200: bool = False
    price_above_sma30: bool = False
    momentum_negative_extreme: bool = False  # < -15%
    momentum_positive: bool = False          # > 0%
    entry_conditions_met: bool = False
    exit_conditions_met: bool = False
    exit_reason: ExitReason = ExitReason.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": {
                "current": float(self.current_price),
                "high": float(self.high_price),
                "low": float(self.low_price),
            },
            "indicators": {
                "sma_200": float(self.sma_200),
                "sma_30": float(self.sma_30),
                "momentum_30": float(self.momentum_30),
            },
            "conditions": {
                "price_below_sma200": self.price_below_sma200,
                "price_above_sma30": self.price_above_sma30,
                "momentum_negative_extreme": self.momentum_negative_extreme,
                "momentum_positive": self.momentum_positive,
                "entry_met": self.entry_conditions_met,
                "exit_met": self.exit_conditions_met,
                "exit_reason": self.exit_reason.value,
            }
        }


@dataclass
class SignalResult:
    """Complete strategy output"""
    signal: Signal
    context: SignalContext
    reason: str
    strategy_name: str = "mostly_long_btc"
    strategy_version: str = "1.0.0"
    confidence: Decimal = Decimal("1.0")

    def to_signal_values(self) -> Dict[str, Any]:
        return {
            "strategy": {
                "name": self.strategy_name,
                "version": self.strategy_version
            },
            "signal": self.signal.value,
            "confidence": float(self.confidence),
            "reason": self.reason,
            **self.context.to_dict()
        }


class MostlyLongBTCStrategy:
    """
    MostlyLong BTC Strategy - Stay invested, exit only on major breakdowns.

    This strategy is designed as a 10-20% portfolio sleeve, not a full
    portfolio strategy. It works well in bull markets and provides some
    downside protection, but will experience 30-40% drawdowns in severe
    bear markets like 2022.

    Parameters:
        exit_sma_period: Period for exit SMA filter (default: 200)
        entry_sma_period: Period for entry SMA filter (default: 30)
        momentum_period: Period for momentum calculation (default: 30)
        exit_momentum_threshold: Exit when momentum below this (default: -15)
        position_size_pct: Allocation as decimal (default: 0.15 for 15%)
    """

    def __init__(
        self,
        exit_sma_period: int = 200,
        entry_sma_period: int = 30,
        momentum_period: int = 30,
        exit_momentum_threshold: float = -15.0,
        position_size_pct: float = 0.15
    ):
        self.exit_sma_period = exit_sma_period
        self.entry_sma_period = entry_sma_period
        self.momentum_period = momentum_period
        self.exit_momentum_threshold = Decimal(str(exit_momentum_threshold))
        self.position_size_pct = Decimal(str(position_size_pct))

        self.min_bars = max(exit_sma_period, entry_sma_period, momentum_period) + 5

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = df.copy()

        # SMAs
        df['sma_200'] = df['close'].rolling(self.exit_sma_period).mean()
        df['sma_30'] = df['close'].rolling(self.entry_sma_period).mean()

        # Momentum (percentage change over period)
        df['momentum_30'] = (df['close'] / df['close'].shift(self.momentum_period) - 1) * 100

        return df

    def evaluate(
        self,
        df: pd.DataFrame,
        timestamp: Optional[datetime] = None,
        has_open_position: bool = False,
        entry_price: Optional[Decimal] = None
    ) -> SignalResult:
        """
        Evaluate strategy on given data.

        Args:
            df: DataFrame with OHLCV columns
            timestamp: Evaluation timestamp
            has_open_position: Whether currently in a position
            entry_price: Entry price if in position (for logging)

        Returns:
            SignalResult with signal and full context
        """
        eval_time = timestamp or datetime.utcnow()

        # Validate input
        if len(df) < self.min_bars:
            return self._create_error_result(
                f"Insufficient data: {len(df)} bars, need {self.min_bars}",
                df, eval_time
            )

        # Calculate indicators
        df = self._calculate_indicators(df)

        current = df.iloc[-1]

        # Extract values
        price = Decimal(str(current['close']))
        high = Decimal(str(current['high']))
        low = Decimal(str(current['low']))
        sma_200 = Decimal(str(current['sma_200']))
        sma_30 = Decimal(str(current['sma_30']))
        momentum = Decimal(str(current['momentum_30']))

        # Conditions
        price_below_sma200 = price < sma_200
        price_above_sma30 = price > sma_30
        momentum_negative_extreme = momentum < self.exit_momentum_threshold
        momentum_positive = momentum > 0

        # Entry: price > 30 SMA AND momentum > 0
        entry_conditions_met = price_above_sma30 and momentum_positive

        # Exit: price < 200 SMA AND momentum < -15%
        exit_conditions_met = price_below_sma200 and momentum_negative_extreme
        exit_reason = ExitReason.EMERGENCY_EXIT if exit_conditions_met else ExitReason.NONE

        # Build context
        context = SignalContext(
            timestamp=eval_time,
            current_price=price,
            high_price=high,
            low_price=low,
            sma_200=sma_200,
            sma_30=sma_30,
            momentum_30=momentum,
            price_below_sma200=price_below_sma200,
            price_above_sma30=price_above_sma30,
            momentum_negative_extreme=momentum_negative_extreme,
            momentum_positive=momentum_positive,
            entry_conditions_met=entry_conditions_met,
            exit_conditions_met=exit_conditions_met,
            exit_reason=exit_reason
        )

        # DECISION LOGIC

        # Exit first if in position and exit conditions met
        if has_open_position and exit_conditions_met:
            return SignalResult(
                signal=Signal.EXIT_LONG,
                context=context,
                reason=(
                    f"EMERGENCY EXIT: Price ({float(price):.0f}) < 200 SMA ({float(sma_200):.0f}) "
                    f"AND Momentum ({float(momentum):.1f}%) < -15%"
                )
            )

        # Entry
        if not has_open_position and entry_conditions_met:
            return SignalResult(
                signal=Signal.LONG,
                context=context,
                reason=(
                    f"ENTRY: Price ({float(price):.0f}) > 30 SMA ({float(sma_30):.0f}) "
                    f"AND Momentum ({float(momentum):.1f}%) > 0%"
                )
            )

        # Hold
        if has_open_position:
            return SignalResult(
                signal=Signal.HOLD,
                context=context,
                reason=f"HOLDING: Exit needs price < {float(sma_200):.0f} AND momentum < -15%"
            )

        # Waiting for entry
        missing = []
        if not price_above_sma30:
            missing.append(f"price ({float(price):.0f}) below 30 SMA ({float(sma_30):.0f})")
        if not momentum_positive:
            missing.append(f"momentum ({float(momentum):.1f}%) not positive")

        return SignalResult(
            signal=Signal.HOLD,
            context=context,
            reason=f"WAITING: {', '.join(missing)}"
        )

    def _create_error_result(
        self,
        reason: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> SignalResult:
        """Create error result"""
        price = Decimal(str(df.iloc[-1]['close'])) if len(df) > 0 else Decimal("0")

        context = SignalContext(
            timestamp=timestamp,
            current_price=price,
            high_price=price,
            low_price=price,
            sma_200=Decimal("0"),
            sma_30=Decimal("0"),
            momentum_30=Decimal("0")
        )

        return SignalResult(
            signal=Signal.HOLD,
            context=context,
            reason=reason
        )

    def calculate_position_size(
        self,
        capital: Decimal,
        sleeve_allocation: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate position size based on sleeve allocation.

        Args:
            capital: Total portfolio capital
            sleeve_allocation: Override allocation (default: self.position_size_pct)

        Returns:
            Dollar amount to allocate to this strategy
        """
        allocation = sleeve_allocation or self.position_size_pct
        return (capital * allocation).quantize(Decimal("0.01"))


# Factory function
def create_strategy(
    exit_sma_period: int = 200,
    entry_sma_period: int = 30,
    momentum_period: int = 30,
    exit_momentum_threshold: float = -15.0,
    position_size_pct: float = 0.15
) -> MostlyLongBTCStrategy:
    """Create MostlyLong BTC Strategy instance"""
    return MostlyLongBTCStrategy(
        exit_sma_period=exit_sma_period,
        entry_sma_period=entry_sma_period,
        momentum_period=momentum_period,
        exit_momentum_threshold=exit_momentum_threshold,
        position_size_pct=position_size_pct
    )


# Default instance (15% sleeve allocation)
default_strategy = MostlyLongBTCStrategy(position_size_pct=0.15)
