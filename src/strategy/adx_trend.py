"""
ADX Trend Strength Strategy - "The Patient Hunter"

Designed specifically to survive Coinbase Advanced Trade's retail fee structure.
Only trades when ADX confirms strong trends, avoiding fee-destroying chop.

Entry (ALL must be true):
- ADX > 25 (strong trend)
- ADX rising (trend strengthening)
- +DI > -DI (bulls winning)
- Price > 50 EMA (uptrend context)

Exit (any triggers):
- ADX < 20 (trend death)
- -DI > +DI (direction flip)
- Close < Chandelier Stop (trailing stop hit)

Fee Model Target:
- 1.0% round-trip (hybrid: market entry, limit exit)
- Average winner must exceed 1.5% to be profitable
- Strategy targets 8%+ average winners

Usage:
    strategy = ADXTrendStrategy()
    result = strategy.evaluate(df, has_open_position=False)
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


class Signal(Enum):
    """Trading signal types"""
    LONG = "long"
    EXIT_LONG = "exit_long"
    HOLD = "hold"


class ExitReason(Enum):
    """Exit trigger classification"""
    TREND_DEATH = "trend_death"          # ADX < 20
    DIRECTION_FLIP = "direction_flip"    # -DI > +DI
    CHANDELIER_STOP = "chandelier_stop"  # Price < trailing stop
    NONE = "none"


@dataclass
class SignalContext:
    """
    The Witness Record - What the strategy saw at decision time.
    Glass Box compliant: every value gets logged.
    """
    timestamp: datetime

    # Price data
    current_price: Decimal
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal

    # ADX System - THE CORE
    adx: Decimal                    # Trend strength (0-100)
    adx_prev: Decimal               # Previous ADX (for rising check)
    adx_rising: bool                # Is ADX increasing?
    plus_di: Decimal                # Bullish directional indicator
    minus_di: Decimal               # Bearish directional indicator
    di_bullish: bool                # Is +DI > -DI?

    # Trend Filter
    ema_50: Decimal                 # 50-period EMA
    above_ema_50: bool              # Price > EMA 50?

    # ATR for stops
    atr: Decimal
    atr_percent: Decimal

    # Chandelier Exit
    highest_high_since_entry: Optional[Decimal] = None
    chandelier_stop: Optional[Decimal] = None
    chandelier_triggered: bool = False

    # Entry thresholds
    stop_loss_price: Optional[Decimal] = None
    risk_amount: Optional[Decimal] = None

    # Signal conditions summary
    entry_conditions_met: bool = False
    exit_reason: ExitReason = ExitReason.NONE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Truth Engine logging"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": {
                "current": float(self.current_price),
                "open": float(self.open_price),
                "high": float(self.high_price),
                "low": float(self.low_price),
                "close": float(self.close_price),
                "volume": float(self.volume)
            },
            "adx_system": {
                "adx": float(self.adx),
                "adx_prev": float(self.adx_prev),
                "adx_rising": self.adx_rising,
                "plus_di": float(self.plus_di),
                "minus_di": float(self.minus_di),
                "di_bullish": self.di_bullish
            },
            "trend_filter": {
                "ema_50": float(self.ema_50),
                "above_ema_50": self.above_ema_50
            },
            "atr": {
                "value": float(self.atr),
                "percent": float(self.atr_percent)
            },
            "chandelier": {
                "highest_high": float(self.highest_high_since_entry) if self.highest_high_since_entry else None,
                "stop": float(self.chandelier_stop) if self.chandelier_stop else None,
                "triggered": self.chandelier_triggered
            },
            "entry_thresholds": {
                "stop_loss": float(self.stop_loss_price) if self.stop_loss_price else None,
                "risk_amount": float(self.risk_amount) if self.risk_amount else None
            },
            "conditions": {
                "entry_met": self.entry_conditions_met,
                "exit_reason": self.exit_reason.value
            }
        }


@dataclass
class SignalResult:
    """Complete output of strategy evaluation."""
    signal: Signal
    context: SignalContext
    reason: str
    strategy_name: str = "adx_trend_strength"
    strategy_version: str = "1.0.0"
    confidence: Decimal = Decimal("1.0")

    def to_signal_values(self) -> Dict[str, Any]:
        """Format for Truth Engine's signal_values field."""
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


class ADXTrendStrategy:
    """
    ADX Trend Strength Strategy - "The Patient Hunter"

    Parameters:
        adx_period: Period for ADX calculation (default: 14)
        adx_entry_threshold: Minimum ADX for entry (default: 25)
        adx_exit_threshold: ADX level that signals trend death (default: 20)
        ema_period: Period for trend filter EMA (default: 50)
        atr_period: Period for ATR calculation (default: 14)
        atr_stop_multiplier: ATR multiplier for initial stop (default: 2.5)
        chandelier_multiplier: ATR multiplier for trailing stop (default: 3.0)
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_entry_threshold: float = 25.0,
        adx_exit_threshold: float = 20.0,
        ema_period: int = 50,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.5,
        chandelier_multiplier: float = 3.0
    ):
        self.adx_period = adx_period
        self.adx_entry_threshold = Decimal(str(adx_entry_threshold))
        self.adx_exit_threshold = Decimal(str(adx_exit_threshold))
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = Decimal(str(atr_stop_multiplier))
        self.chandelier_multiplier = Decimal(str(chandelier_multiplier))

        # Minimum bars needed
        self.min_bars = max(adx_period * 2, ema_period) + 5

    def _calculate_adx_system(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX, +DI, -DI using Wilder's smoothing.
        """
        df = df.copy()
        period = self.adx_period

        # True Range
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        # Create Series with proper index to maintain alignment
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index
        )

        # Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / period

        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        # DX and ADX
        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)
        dx = 100 * di_diff / di_sum.replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        df['atr'] = atr
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['adx'] = adx

        return df

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    def evaluate(
        self,
        df: pd.DataFrame,
        timestamp: Optional[datetime] = None,
        has_open_position: bool = False,
        entry_price: Optional[Decimal] = None,
        highest_high_since_entry: Optional[Decimal] = None
    ) -> SignalResult:
        """
        Evaluate the ADX Trend Strength strategy.

        Args:
            df: DataFrame with OHLCV columns
            timestamp: Evaluation timestamp
            has_open_position: Whether we have an open long position
            entry_price: Entry price if in position
            highest_high_since_entry: Highest high since entry for Chandelier

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
        df = df.copy()
        df = self._calculate_adx_system(df)
        df['ema_50'] = self._calculate_ema(df['close'], self.ema_period)

        # Get current and previous values
        current = df.iloc[-1]
        previous = df.iloc[-2]

        # Extract values
        current_price = Decimal(str(current['close']))
        current_high = Decimal(str(current['high']))
        adx = Decimal(str(current['adx'])) if pd.notna(current['adx']) else Decimal("0")
        adx_prev = Decimal(str(previous['adx'])) if pd.notna(previous['adx']) else Decimal("0")
        plus_di = Decimal(str(current['plus_di'])) if pd.notna(current['plus_di']) else Decimal("0")
        minus_di = Decimal(str(current['minus_di'])) if pd.notna(current['minus_di']) else Decimal("0")
        ema_50 = Decimal(str(current['ema_50'])) if pd.notna(current['ema_50']) else Decimal("0")
        atr = Decimal(str(current['atr'])) if pd.notna(current['atr']) else Decimal("0")

        # Derived conditions
        adx_rising = adx > adx_prev
        di_bullish = plus_di > minus_di
        above_ema_50 = current_price > ema_50
        atr_percent = (atr / current_price * 100) if current_price > 0 else Decimal("0")

        # Entry conditions
        adx_strong = adx > self.adx_entry_threshold
        entry_conditions_met = adx_strong and adx_rising and di_bullish and above_ema_50

        # Calculate stop loss for potential entry
        stop_loss_price = current_price - (atr * self.atr_stop_multiplier)
        risk_amount = current_price - stop_loss_price

        # Chandelier Stop calculation
        chandelier_stop: Optional[Decimal] = None
        chandelier_triggered = False

        if has_open_position and highest_high_since_entry is not None:
            chandelier_stop = highest_high_since_entry - (atr * self.chandelier_multiplier)
            chandelier_triggered = current_price < chandelier_stop

        # Determine exit reason
        exit_reason = ExitReason.NONE
        if has_open_position:
            if adx < self.adx_exit_threshold:
                exit_reason = ExitReason.TREND_DEATH
            elif minus_di > plus_di:
                exit_reason = ExitReason.DIRECTION_FLIP
            elif chandelier_triggered:
                exit_reason = ExitReason.CHANDELIER_STOP

        # Build context
        context = SignalContext(
            timestamp=eval_time,
            current_price=current_price,
            open_price=Decimal(str(current['open'])),
            high_price=current_high,
            low_price=Decimal(str(current['low'])),
            close_price=current_price,
            volume=Decimal(str(current['volume'])),
            adx=adx,
            adx_prev=adx_prev,
            adx_rising=adx_rising,
            plus_di=plus_di,
            minus_di=minus_di,
            di_bullish=di_bullish,
            ema_50=ema_50,
            above_ema_50=above_ema_50,
            atr=atr,
            atr_percent=atr_percent,
            highest_high_since_entry=highest_high_since_entry,
            chandelier_stop=chandelier_stop,
            chandelier_triggered=chandelier_triggered,
            stop_loss_price=stop_loss_price,
            risk_amount=risk_amount,
            entry_conditions_met=entry_conditions_met,
            exit_reason=exit_reason
        )

        # DECISION LOGIC

        # Case 1: Exit signals (check first if in position)
        if has_open_position and exit_reason != ExitReason.NONE:
            reason_text = {
                ExitReason.TREND_DEATH: f"TREND DEATH: ADX ({float(adx):.1f}) dropped below {float(self.adx_exit_threshold)}",
                ExitReason.DIRECTION_FLIP: f"DIRECTION FLIP: -DI ({float(minus_di):.1f}) crossed above +DI ({float(plus_di):.1f})",
                ExitReason.CHANDELIER_STOP: f"CHANDELIER STOP: Price ({float(current_price):.2f}) fell below stop ({float(chandelier_stop):.2f})"
            }
            return SignalResult(
                signal=Signal.EXIT_LONG,
                context=context,
                reason=reason_text[exit_reason]
            )

        # Case 2: Entry signal
        if not has_open_position and entry_conditions_met:
            return SignalResult(
                signal=Signal.LONG,
                context=context,
                reason=(
                    f"ENTRY: ADX={float(adx):.1f} (>{float(self.adx_entry_threshold)}), "
                    f"Rising={adx_rising}, +DI={float(plus_di):.1f} > -DI={float(minus_di):.1f}, "
                    f"Price ({float(current_price):.2f}) > EMA50 ({float(ema_50):.2f}). "
                    f"Stop: {float(stop_loss_price):.2f}"
                )
            )

        # Case 3: Holding position
        if has_open_position:
            chandelier_info = f" Chandelier: {float(chandelier_stop):.2f}" if chandelier_stop else ""
            return SignalResult(
                signal=Signal.HOLD,
                context=context,
                reason=(
                    f"HOLDING: ADX={float(adx):.1f}, +DI > -DI.{chandelier_info}"
                )
            )

        # Case 4: No position, waiting for entry
        missing = []
        if not adx_strong:
            missing.append(f"ADX ({float(adx):.1f}) < {float(self.adx_entry_threshold)}")
        if not adx_rising:
            missing.append("ADX falling")
        if not di_bullish:
            missing.append(f"-DI ({float(minus_di):.1f}) > +DI ({float(plus_di):.1f})")
        if not above_ema_50:
            missing.append(f"Price < EMA50")

        return SignalResult(
            signal=Signal.HOLD,
            context=context,
            reason=f"WAITING: Missing conditions: {', '.join(missing)}"
        )

    def _create_error_result(
        self,
        reason: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> SignalResult:
        """Create SignalResult for error cases"""
        if len(df) > 0:
            current = df.iloc[-1]
            price = Decimal(str(current.get('close', 0)))
        else:
            price = Decimal("0")

        context = SignalContext(
            timestamp=timestamp,
            current_price=price,
            open_price=price,
            high_price=price,
            low_price=price,
            close_price=price,
            volume=Decimal("0"),
            adx=Decimal("0"),
            adx_prev=Decimal("0"),
            adx_rising=False,
            plus_di=Decimal("0"),
            minus_di=Decimal("0"),
            di_bullish=False,
            ema_50=Decimal("0"),
            above_ema_50=False,
            atr=Decimal("0"),
            atr_percent=Decimal("0")
        )

        return SignalResult(
            signal=Signal.HOLD,
            context=context,
            reason=reason
        )

    def calculate_position_size(
        self,
        capital: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal,
        risk_percent: Decimal = Decimal("0.01")
    ) -> Decimal:
        """
        Calculate position size based on fixed fractional risk.
        Risk 1% of capital per trade.
        """
        risk_per_trade = capital * risk_percent
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            return Decimal("0")

        position_size = risk_per_trade / price_risk
        return position_size.quantize(Decimal("0.00000001"))


# Factory function
def create_strategy(
    adx_period: int = 14,
    adx_entry_threshold: float = 25.0,
    adx_exit_threshold: float = 20.0,
    ema_period: int = 50,
    atr_period: int = 14,
    atr_stop_multiplier: float = 2.5,
    chandelier_multiplier: float = 3.0
) -> ADXTrendStrategy:
    """Factory function to create ADX Trend Strategy"""
    return ADXTrendStrategy(
        adx_period=adx_period,
        adx_entry_threshold=adx_entry_threshold,
        adx_exit_threshold=adx_exit_threshold,
        ema_period=ema_period,
        atr_period=atr_period,
        atr_stop_multiplier=atr_stop_multiplier,
        chandelier_multiplier=chandelier_multiplier
    )


# Default strategy instance
default_strategy = ADXTrendStrategy()
