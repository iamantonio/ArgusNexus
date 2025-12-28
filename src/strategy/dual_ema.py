"""
Dual EMA Crossover Strategy - V4 Core Strategy

THE ONLY STRATEGY IN V4. THE BRAIN THAT WITNESSES.

This isn't just a calculator - it's a witness.
Every signal comes with full context for the Glass Box.

Rules:
- Fast EMA (12) crosses above Slow EMA (26) → LONG signal
- Fast EMA crosses below Slow EMA → EXIT signal (close long)
- Stop loss: 2x ATR below entry
- Take profit: 3x ATR above entry (1:1.5 R:R)
- Position size: Risk 1% of capital per trade

NO ML. NO AI. Just math. Just truth.

Usage:
    strategy = DualEMACrossover()
    result = strategy.evaluate(df)  # DataFrame with OHLCV

    if result.signal == Signal.LONG:
        # Log the FULL context to Truth Engine
        truth_logger.log_decision(
            signal_values=result.to_signal_values(),
            ...
        )
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


class Signal(Enum):
    """Trading signal types"""
    LONG = "long"           # Open long position
    EXIT_LONG = "exit_long" # Close long position
    SHORT = "short"         # Open short position (future)
    EXIT_SHORT = "exit_short"  # Close short position (future)
    HOLD = "hold"           # No action


class CrossoverType(Enum):
    """EMA crossover classification"""
    BULLISH = "bullish"     # Fast crossed above slow
    BEARISH = "bearish"     # Fast crossed below slow
    NONE = "none"           # No crossover


@dataclass
class SignalContext:
    """
    The Witness Record - What the strategy saw at decision time.

    This is THE critical data for the Glass Box.
    Every value here gets logged to the decisions table.
    """
    # Timestamp of evaluation
    timestamp: datetime

    # Current price data
    current_price: Decimal
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal

    # EMA values - THE CORE INDICATORS
    fast_ema: Decimal           # 12-period EMA
    slow_ema: Decimal           # 26-period EMA
    ema_diff: Decimal           # fast_ema - slow_ema
    ema_diff_percent: Decimal   # (fast - slow) / slow * 100

    # Previous bar EMA values (for crossover detection)
    prev_fast_ema: Decimal
    prev_slow_ema: Decimal
    prev_ema_diff: Decimal

    # Crossover detection
    crossover_type: CrossoverType
    crossover_bars_ago: int     # How many bars since last crossover

    # ATR - Volatility measure for stops
    atr: Decimal                # 14-period ATR
    atr_percent: Decimal        # ATR as % of price

    # Calculated thresholds (at signal time)
    stop_loss_price: Optional[Decimal] = None    # Entry - 2*ATR
    take_profit_price: Optional[Decimal] = None  # Entry + 3*ATR
    risk_amount: Optional[Decimal] = None        # Distance to stop in $
    reward_amount: Optional[Decimal] = None      # Distance to TP in $
    risk_reward_ratio: Optional[Decimal] = None  # reward / risk

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Truth Engine logging"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "current_price": float(self.current_price),
            "ohlcv": {
                "open": float(self.open_price),
                "high": float(self.high_price),
                "low": float(self.low_price),
                "close": float(self.close_price),
                "volume": float(self.volume)
            },
            "ema": {
                "fast_ema_12": float(self.fast_ema),
                "slow_ema_26": float(self.slow_ema),
                "ema_diff": float(self.ema_diff),
                "ema_diff_percent": float(self.ema_diff_percent),
                "prev_fast_ema": float(self.prev_fast_ema),
                "prev_slow_ema": float(self.prev_slow_ema),
                "prev_ema_diff": float(self.prev_ema_diff)
            },
            "crossover": {
                "type": self.crossover_type.value,
                "bars_ago": self.crossover_bars_ago
            },
            "atr": {
                "atr_14": float(self.atr),
                "atr_percent": float(self.atr_percent)
            },
            "thresholds": {
                "stop_loss_price": float(self.stop_loss_price) if self.stop_loss_price else None,
                "take_profit_price": float(self.take_profit_price) if self.take_profit_price else None,
                "risk_amount": float(self.risk_amount) if self.risk_amount else None,
                "reward_amount": float(self.reward_amount) if self.reward_amount else None,
                "risk_reward_ratio": float(self.risk_reward_ratio) if self.risk_reward_ratio else None
            }
        }


@dataclass
class SignalResult:
    """
    The complete output of a strategy evaluation.

    This is what gets passed to the Truth Engine.
    Signal + Context + Reason = Full Glass Box Record
    """
    # The decision
    signal: Signal

    # The witness record - everything the strategy saw
    context: SignalContext

    # Human-readable explanation
    reason: str

    # Strategy metadata
    strategy_name: str = "dual_ema_crossover"
    strategy_version: str = "1.0.0"

    # Confidence (for future use, always 1.0 for mechanical strategy)
    confidence: Decimal = Decimal("1.0")

    def to_signal_values(self) -> Dict[str, Any]:
        """
        Format for Truth Engine's signal_values field.

        This is THE data that answers "why did this trade happen?"
        """
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


class DualEMACrossover:
    """
    Dual EMA Crossover Strategy - The V4 Brain

    A mechanical, deterministic strategy that witnesses everything it sees.

    Parameters:
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        atr_period: ATR period for volatility (default: 14)
        atr_stop_multiplier: ATR multiplier for stop loss (default: 2.0)
        atr_tp_multiplier: ATR multiplier for take profit (default: 3.0)

    The 12/26 periods align with standard MACD settings.
    Boring. Proven. No magic.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        atr_tp_multiplier: float = 3.0
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = Decimal(str(atr_stop_multiplier))
        self.atr_tp_multiplier = Decimal(str(atr_tp_multiplier))

        # Minimum bars needed for valid signals
        self.min_bars = max(slow_period, atr_period) + 2

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range

        True Range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        # True Range is max of all three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is EMA of True Range
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def _detect_crossover(
        self,
        fast_ema: float,
        slow_ema: float,
        prev_fast_ema: float,
        prev_slow_ema: float
    ) -> CrossoverType:
        """
        Detect EMA crossover

        Bullish: Fast was below slow, now above
        Bearish: Fast was above slow, now below
        """
        was_below = prev_fast_ema < prev_slow_ema
        was_above = prev_fast_ema > prev_slow_ema
        now_above = fast_ema > slow_ema
        now_below = fast_ema < slow_ema

        if was_below and now_above:
            return CrossoverType.BULLISH
        elif was_above and now_below:
            return CrossoverType.BEARISH
        else:
            return CrossoverType.NONE

    def _find_bars_since_crossover(self, df: pd.DataFrame) -> int:
        """Find how many bars since the last crossover"""
        if len(df) < 2:
            return 0

        fast_ema = self._calculate_ema(df['close'], self.fast_period)
        slow_ema = self._calculate_ema(df['close'], self.slow_period)

        # Calculate where crossovers occurred
        above = fast_ema > slow_ema
        crossover_points = above.ne(above.shift(1))

        # Find last crossover
        crossover_indices = crossover_points[crossover_points].index
        if len(crossover_indices) == 0:
            return len(df)

        last_crossover_idx = crossover_indices[-1]
        bars_since = len(df) - df.index.get_loc(last_crossover_idx) - 1

        return bars_since

    def evaluate(
        self,
        df: pd.DataFrame,
        timestamp: Optional[datetime] = None,
        has_open_position: bool = False
    ) -> SignalResult:
        """
        Evaluate the strategy on the given data.

        This is THE function that generates witnessed signals.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Must have at least self.min_bars rows
            timestamp: Optional timestamp (defaults to now)
            has_open_position: Whether we currently have an open long position

        Returns:
            SignalResult with full context for Truth Engine logging
        """
        eval_time = timestamp or datetime.utcnow()

        # Validate input
        if len(df) < self.min_bars:
            return self._create_result(
                signal=Signal.HOLD,
                reason=f"Insufficient data: {len(df)} bars, need {self.min_bars}",
                df=df,
                timestamp=eval_time
            )

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return self._create_result(
                signal=Signal.HOLD,
                reason=f"Missing columns: {missing}",
                df=df,
                timestamp=eval_time
            )

        # Calculate indicators
        df = df.copy()
        df['fast_ema'] = self._calculate_ema(df['close'], self.fast_period)
        df['slow_ema'] = self._calculate_ema(df['close'], self.slow_period)
        df['atr'] = self._calculate_atr(df, self.atr_period)

        # Get current and previous values
        current = df.iloc[-1]
        previous = df.iloc[-2]

        # Extract values
        current_price = Decimal(str(current['close']))
        fast_ema = Decimal(str(current['fast_ema']))
        slow_ema = Decimal(str(current['slow_ema']))
        prev_fast_ema = Decimal(str(previous['fast_ema']))
        prev_slow_ema = Decimal(str(previous['slow_ema']))
        atr = Decimal(str(current['atr']))

        # Calculate derived values
        ema_diff = fast_ema - slow_ema
        prev_ema_diff = prev_fast_ema - prev_slow_ema
        ema_diff_percent = (ema_diff / slow_ema * 100) if slow_ema != 0 else Decimal("0")
        atr_percent = (atr / current_price * 100) if current_price != 0 else Decimal("0")

        # Detect crossover
        crossover = self._detect_crossover(
            float(fast_ema), float(slow_ema),
            float(prev_fast_ema), float(prev_slow_ema)
        )

        # Find bars since last crossover
        bars_since_crossover = self._find_bars_since_crossover(df)

        # Calculate stop/TP thresholds (for potential entry)
        stop_loss_price = current_price - (atr * self.atr_stop_multiplier)
        take_profit_price = current_price + (atr * self.atr_tp_multiplier)
        risk_amount = current_price - stop_loss_price
        reward_amount = take_profit_price - current_price
        risk_reward_ratio = reward_amount / risk_amount if risk_amount != 0 else Decimal("0")

        # Build context (THE WITNESS RECORD)
        context = SignalContext(
            timestamp=eval_time,
            current_price=current_price,
            open_price=Decimal(str(current['open'])),
            high_price=Decimal(str(current['high'])),
            low_price=Decimal(str(current['low'])),
            close_price=Decimal(str(current['close'])),
            volume=Decimal(str(current['volume'])),
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            ema_diff=ema_diff,
            ema_diff_percent=ema_diff_percent,
            prev_fast_ema=prev_fast_ema,
            prev_slow_ema=prev_slow_ema,
            prev_ema_diff=prev_ema_diff,
            crossover_type=crossover,
            crossover_bars_ago=bars_since_crossover,
            atr=atr,
            atr_percent=atr_percent,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=risk_reward_ratio
        )

        # DECISION LOGIC
        # ==============

        # Case 1: Bullish crossover and no position → LONG
        if crossover == CrossoverType.BULLISH and not has_open_position:
            return SignalResult(
                signal=Signal.LONG,
                context=context,
                reason=(
                    f"Bullish EMA crossover: Fast EMA ({float(fast_ema):.2f}) "
                    f"crossed above Slow EMA ({float(slow_ema):.2f}). "
                    f"Entry at {float(current_price):.2f}, "
                    f"Stop: {float(stop_loss_price):.2f}, "
                    f"TP: {float(take_profit_price):.2f}, "
                    f"R:R = 1:{float(risk_reward_ratio):.1f}"
                )
            )

        # Case 2: Bearish crossover and have position → EXIT
        if crossover == CrossoverType.BEARISH and has_open_position:
            return SignalResult(
                signal=Signal.EXIT_LONG,
                context=context,
                reason=(
                    f"Bearish EMA crossover: Fast EMA ({float(fast_ema):.2f}) "
                    f"crossed below Slow EMA ({float(slow_ema):.2f}). "
                    f"Signal exit at {float(current_price):.2f}."
                )
            )

        # Case 3: Already in position, fast still above slow → HOLD
        if has_open_position and fast_ema > slow_ema:
            return SignalResult(
                signal=Signal.HOLD,
                context=context,
                reason=(
                    f"Holding position: Fast EMA ({float(fast_ema):.2f}) "
                    f"still above Slow EMA ({float(slow_ema):.2f}). "
                    f"Trend intact."
                )
            )

        # Case 4: No position, fast below slow → HOLD (wait for crossover)
        if not has_open_position and fast_ema < slow_ema:
            return SignalResult(
                signal=Signal.HOLD,
                context=context,
                reason=(
                    f"No signal: Fast EMA ({float(fast_ema):.2f}) "
                    f"below Slow EMA ({float(slow_ema):.2f}). "
                    f"Waiting for bullish crossover."
                )
            )

        # Case 5: No position, fast above slow but no fresh crossover → HOLD
        if not has_open_position and fast_ema > slow_ema:
            return SignalResult(
                signal=Signal.HOLD,
                context=context,
                reason=(
                    f"No signal: Fast EMA ({float(fast_ema):.2f}) "
                    f"above Slow EMA ({float(slow_ema):.2f}), "
                    f"but crossover was {bars_since_crossover} bars ago. "
                    f"Waiting for fresh crossover."
                )
            )

        # Default: HOLD
        return SignalResult(
            signal=Signal.HOLD,
            context=context,
            reason="No actionable signal"
        )

    def _create_result(
        self,
        signal: Signal,
        reason: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> SignalResult:
        """Create a SignalResult with minimal context (for error cases)"""
        # Use last row if available, otherwise zeros
        if len(df) > 0:
            current = df.iloc[-1]
            current_price = Decimal(str(current.get('close', 0)))
            open_price = Decimal(str(current.get('open', 0)))
            high_price = Decimal(str(current.get('high', 0)))
            low_price = Decimal(str(current.get('low', 0)))
            volume = Decimal(str(current.get('volume', 0)))
        else:
            current_price = open_price = high_price = low_price = volume = Decimal("0")

        context = SignalContext(
            timestamp=timestamp,
            current_price=current_price,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=current_price,
            volume=volume,
            fast_ema=Decimal("0"),
            slow_ema=Decimal("0"),
            ema_diff=Decimal("0"),
            ema_diff_percent=Decimal("0"),
            prev_fast_ema=Decimal("0"),
            prev_slow_ema=Decimal("0"),
            prev_ema_diff=Decimal("0"),
            crossover_type=CrossoverType.NONE,
            crossover_bars_ago=0,
            atr=Decimal("0"),
            atr_percent=Decimal("0")
        )

        return SignalResult(
            signal=signal,
            context=context,
            reason=reason
        )

    def calculate_position_size(
        self,
        capital: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal,
        risk_percent: Decimal = Decimal("0.01")  # 1% risk per trade
    ) -> Decimal:
        """
        Calculate position size based on risk management.

        Risk 1% of capital per trade.
        Position size = (Capital * Risk%) / (Entry - Stop)

        Args:
            capital: Total trading capital
            entry_price: Expected entry price
            stop_loss_price: Stop loss price
            risk_percent: Percentage of capital to risk (default 1%)

        Returns:
            Position size in base currency
        """
        risk_per_trade = capital * risk_percent
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            return Decimal("0")

        position_size = risk_per_trade / price_risk

        return position_size.quantize(Decimal("0.00000001"))  # 8 decimal places for crypto


# =============================================================================
# Convenience Functions
# =============================================================================

def create_strategy(
    fast_period: int = 12,
    slow_period: int = 26,
    atr_period: int = 14,
    stop_multiplier: float = 2.0,
    tp_multiplier: float = 3.0
) -> DualEMACrossover:
    """Factory function to create a strategy instance"""
    return DualEMACrossover(
        fast_period=fast_period,
        slow_period=slow_period,
        atr_period=atr_period,
        atr_stop_multiplier=stop_multiplier,
        atr_tp_multiplier=tp_multiplier
    )


# Default strategy instance
default_strategy = DualEMACrossover()
