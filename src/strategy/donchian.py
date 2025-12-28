"""
Donchian Channel Breakout Strategy - TURTLE-4 (v6.1) "The Ratchet + Shorts"

EVOLUTION: TURTLE-4 v5.0 protected gains in bull markets with Chandelier Exit.
v6.1 adds SHORT capability to profit in bear markets too.

The 'TURTLE-4' Logic (LONGS - The Ratchet):
- ENTRY: Price > Highest High of last 55 periods → LONG
- TREND FILTER: Price > 200-period SMA
- VOLUME FILTER: Breakout volume > 1.5x 20-period average
- CHANDELIER EXIT: Highest High Since Entry - (3 * ATR)
  * The Ratchet only moves UP, never down
  * Locks in gains as price rises
- BACKUP EXIT: 55-period low (safety net)

The 'TURTLE-4' Logic (SHORTS - The Inverse Ratchet) - NEW v6.1:
- ENTRY: Price < Lowest Low of last 55 periods → SHORT
- TREND FILTER: Price < 200-period SMA (INVERTED)
- VOLUME FILTER: Breakdown volume > 1.5x 20-period average
- INVERSE CHANDELIER EXIT: Lowest Low Since Entry + (3 * ATR)
  * The Inverse Ratchet only moves DOWN, never up
  * Locks in gains as price falls
- BACKUP EXIT: 55-period high (safety net)

Why v6.1 (Shorts):
- December 2025 pullback: BTC dropped 10.8%, ETH dropped 20.2%
- Long-only strategy sat flat while market fell
- With shorts, we capture profits in BOTH directions

"Trade both sides of the market, but always with discipline."

History:
- v1.0 (Turtle): -$848 (0% win rate, commission death)
- v2.0 (Sniper): -$618 (+$27 gross, 0.04% moves)
- v3.0 (TURTLE-2): -$302 (+$206 gross, 1 win, 0.38% moves)
- v4.0 (TURTLE-3): +$23,612 in 2024, -$3,872 in 2025 (Diamond Hands got crushed)
- v5.0 (TURTLE-4): The Ratchet - protect the gains
- v6.1 (TURTLE-4): The Inverse Ratchet - profit in bear markets

Usage:
    strategy = DonchianBreakout()  # TURTLE-4 v6.1

    # LONG position
    result = strategy.evaluate(df, has_open_position=True,
                               entry_price=Decimal("100000"),
                               highest_high_since_entry=Decimal("126000"))

    # SHORT position
    result = strategy.evaluate(df, has_open_position=True,
                               entry_price=Decimal("95000"),
                               is_short_position=True,
                               lowest_low_since_entry=Decimal("84000"))
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
    SHORT = "short"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"


class BreakoutType(Enum):
    """Breakout classification"""
    UPPER = "upper_breakout"    # Price broke above upper channel
    LOWER = "lower_breakout"    # Price broke below lower channel
    NONE = "none"               # No breakout


@dataclass
class SignalContext:
    """
    The Witness Record - What the strategy saw at decision time.
    Glass Box compliant: every value gets logged.
    """
    timestamp: datetime

    # Current price data
    current_price: Decimal
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal

    # Donchian Channel values - THE CORE
    upper_channel: Decimal      # Highest high of entry_period
    lower_channel: Decimal      # Lowest low of entry_period
    exit_channel: Decimal       # Lowest low of exit_period (for exits)
    channel_width: Decimal      # upper - lower
    channel_width_pct: Decimal  # width as % of price

    # Breakout detection
    breakout_type: BreakoutType
    distance_from_upper: Decimal    # How far price is from upper channel
    distance_from_lower: Decimal    # How far price is from lower channel

    # ATR for stops
    atr: Decimal
    atr_percent: Decimal

    # SNIPER FILTERS (v2.0)
    sma_200: Optional[Decimal] = None       # 200-period SMA (trend filter)
    above_sma_200: bool = False             # Is price above 200 SMA?
    volume_avg_20: Optional[Decimal] = None # 20-period average volume
    volume_ratio: Optional[Decimal] = None  # Current volume / avg volume
    volume_confirmed: bool = False          # Is volume > 1.5x average?
    filters_passed: bool = False            # Did all SNIPER filters pass?

    # MARKET REGIME (v6.0 - Trend Strength)
    adx: Optional[Decimal] = None           # Trend strength (0-100)
    trend_strong: bool = True               # Is ADX > threshold?

    # CHANDELIER EXIT (v5.0 - The Ratchet)
    highest_high_since_entry: Optional[Decimal] = None  # Tracked HH since position opened
    chandelier_stop: Optional[Decimal] = None           # HH - (3 * ATR) - the ratchet
    chandelier_triggered: bool = False                  # Did price fall below chandelier?

    # SHORT CAPABILITY (v6.1 - Inverse Ratchet)
    lowest_low_since_entry: Optional[Decimal] = None    # Tracked LL since short opened
    short_chandelier_stop: Optional[Decimal] = None     # LL + (3 * ATR) - inverse ratchet
    short_chandelier_triggered: bool = False            # Did price rise above short chandelier?
    below_sma_200: bool = False                         # Is price BELOW 200 SMA? (bearish filter)
    is_short_position: bool = False                     # Is this a short position?

    # Calculated thresholds
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    risk_amount: Optional[Decimal] = None
    reward_amount: Optional[Decimal] = None
    risk_reward_ratio: Optional[Decimal] = None

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
            "donchian": {
                "upper_channel": float(self.upper_channel),
                "lower_channel": float(self.lower_channel),
                "exit_channel": float(self.exit_channel),
                "channel_width": float(self.channel_width),
                "channel_width_pct": float(self.channel_width_pct)
            },
            "breakout": {
                "type": self.breakout_type.value,
                "distance_from_upper": float(self.distance_from_upper),
                "distance_from_lower": float(self.distance_from_lower)
            },
            "atr": {
                "atr_14": float(self.atr),
                "atr_percent": float(self.atr_percent)
            },
            "sniper_filters": {
                "sma_200": float(self.sma_200) if self.sma_200 else None,
                "above_sma_200": self.above_sma_200,
                "volume_avg_20": float(self.volume_avg_20) if self.volume_avg_20 else None,
                "volume_ratio": float(self.volume_ratio) if self.volume_ratio else None,
                "volume_confirmed": self.volume_confirmed,
                "filters_passed": self.filters_passed
            },
            "market_regime": {
                "adx_14": float(self.adx) if self.adx else None,
                "trend_strong": self.trend_strong
            },
            "chandelier_exit": {
                "highest_high_since_entry": float(self.highest_high_since_entry) if self.highest_high_since_entry else None,
                "chandelier_stop": float(self.chandelier_stop) if self.chandelier_stop else None,
                "chandelier_triggered": self.chandelier_triggered
            },
            "short_chandelier_exit": {
                "lowest_low_since_entry": float(self.lowest_low_since_entry) if self.lowest_low_since_entry else None,
                "short_chandelier_stop": float(self.short_chandelier_stop) if self.short_chandelier_stop else None,
                "short_chandelier_triggered": self.short_chandelier_triggered,
                "below_sma_200": self.below_sma_200,
                "is_short_position": self.is_short_position
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
    """Complete output of strategy evaluation."""
    signal: Signal
    context: SignalContext
    reason: str
    strategy_name: str = "donchian_turtle4"
    strategy_version: str = "5.0.0"
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


class DonchianBreakout:
    """
    Donchian Channel Breakout Strategy - TURTLE-4 (v5.0) "The Ratchet"

    Trend following with Chandelier Exit to protect gains.

    TURTLE-4 System (v5.0):
        - Entry: 55-period high breakout (same as TURTLE-3)
        - Trend Filter: Price must be above 200 SMA (same)
        - Volume Filter: Breakout volume must be > 1.5x 20-period average (same)
        - CHANDELIER EXIT (NEW): Highest High Since Entry - (3 * ATR)
          * Stop only ratchets UP, never down
          * Exits when price closes below the ratcheted stop
        - Backup Exit: 55-period low breakdown (safety net)

    Parameters:
        entry_period: Period for upper channel (default: 55)
        exit_period: Period for exit channel backup (default: 55)
        atr_period: ATR period for stops (default: 14)
        atr_stop_multiplier: ATR multiplier for initial stop loss (default: 2.0)
        atr_tp_multiplier: ATR multiplier for take profit (default: 4.0)
        chandelier_atr_multiplier: ATR multiplier for chandelier exit (default: 3.0)
        trend_period: Period for trend SMA filter (default: 200)
        volume_avg_period: Period for volume average (default: 20)
        volume_multiplier: Required volume vs average (default: 1.5)
    """

    def __init__(
        self,
        entry_period: int = 55,  # Major breakout only
        exit_period: int = 55,   # Backup exit (Chandelier usually hits first)
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        atr_tp_multiplier: float = 4.0,
        chandelier_atr_multiplier: float = 3.0,  # THE RATCHET
        trend_period: int = 200,
        volume_avg_period: int = 20,
        volume_multiplier: float = 1.5,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        use_adx_filter: bool = False  # Disabled by default for backward compatibility
    ):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = Decimal(str(atr_stop_multiplier))
        self.atr_tp_multiplier = Decimal(str(atr_tp_multiplier))
        self.chandelier_atr_multiplier = Decimal(str(chandelier_atr_multiplier))

        # SNIPER filters (v2.0)
        self.trend_period = trend_period
        self.volume_avg_period = volume_avg_period
        self.volume_multiplier = Decimal(str(volume_multiplier))

        # MARKET REGIME (v6.0)
        self.adx_period = adx_period
        self.adx_threshold = Decimal(str(adx_threshold))
        self.use_adx_filter = use_adx_filter

        # Minimum bars needed (must include 200 SMA and ADX warm-up)
        self.min_bars = max(entry_period, atr_period, trend_period, adx_period * 2) + 2

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) using pure pandas.
        Standard Wilder's method.
        """
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']

        # Directional Movement
        # NOTE: plus_dm = current high - previous high (upward movement)
        #       minus_dm = previous low - current low (downward movement)
        # FIXED: Was using low.diff(-1) which looks FORWARD (future data leak!)
        plus_dm = high.diff()
        minus_dm = low.shift(1) - low  # Correct: previous low - current low
        
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed values using Wilder's EMA
        tr_s = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_dm_s = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()

        # Directional Indicators
        plus_di = 100 * (plus_dm_s / tr_s)
        minus_di = 100 * (minus_dm_s / tr_s)

        # Directional Index (DX)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        
        # ADX: Smoothed DX
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def _calculate_donchian(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian Channels.

        Upper: Highest high of entry_period
        Lower: Lowest low of entry_period
        Exit: Lowest low of exit_period
        """
        df = df.copy()

        # Entry channel (20-period by default)
        df['upper_channel'] = df['high'].rolling(window=self.entry_period).max()
        df['lower_channel'] = df['low'].rolling(window=self.entry_period).min()

        # Exit channel (20-period by default in SNIPER) - only the low for exiting longs
        df['exit_channel'] = df['low'].rolling(window=self.exit_period).min()

        return df

    def _calculate_sniper_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SNIPER filter indicators (v2.0).

        - 200-period SMA for trend filter
        - 20-period average volume for volume filter
        """
        df = df.copy()

        # Trend filter: 200-period Simple Moving Average
        df['sma_200'] = df['close'].rolling(window=self.trend_period).mean()

        # Volume filter: 20-period average volume
        df['volume_avg_20'] = df['volume'].rolling(window=self.volume_avg_period).mean()

        return df

    def evaluate(
        self,
        df: pd.DataFrame,
        timestamp: Optional[datetime] = None,
        has_open_position: bool = False,
        entry_price: Optional[Decimal] = None,
        highest_high_since_entry: Optional[Decimal] = None,
        # SHORT CAPABILITY (v6.1)
        is_short_position: bool = False,
        lowest_low_since_entry: Optional[Decimal] = None,
        # v6.2: Use external daily SMA for proper 200-day trend filter
        daily_sma_200: Optional[Decimal] = None
    ) -> SignalResult:
        """
        Evaluate the Donchian Breakout strategy with TURTLE-4 Chandelier Exit.

        The Logic (LONGS):
        1. No position + Upper Breakout + SNIPER filters pass → LONG
        2. Has LONG position + Price closes below Chandelier Stop → EXIT_LONG (PRIMARY)
        3. Has LONG position + Price breaks below 55-period low → EXIT_LONG (BACKUP)

        The Logic (SHORTS - v6.1):
        4. No position + Lower Breakout + BEARISH filters pass → SHORT
        5. Has SHORT position + Price closes above Short Chandelier Stop → EXIT_SHORT (PRIMARY)
        6. Has SHORT position + Price breaks above 55-period high → EXIT_SHORT (BACKUP)
        7. Otherwise → HOLD

        SNIPER Filters for LONGS (must ALL pass):
        - Trend Filter: Price > 200 SMA
        - Volume Filter: Current volume > 1.5x 20-period average

        BEARISH Filters for SHORTS (must ALL pass):
        - Trend Filter: Price < 200 SMA (INVERTED)
        - Volume Filter: Current volume > 1.5x 20-period average

        CHANDELIER EXIT (v5.0 - LONGS):
        - Chandelier Stop = Highest High Since Entry - (3 * ATR)
        - The stop only ratchets UP, never down

        INVERSE CHANDELIER EXIT (v6.1 - SHORTS):
        - Short Chandelier Stop = Lowest Low Since Entry + (3 * ATR)
        - The stop only ratchets DOWN, never up
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

        # Calculate indicators
        df = df.copy()
        df = self._calculate_donchian(df)
        df = self._calculate_sniper_filters(df)  # SNIPER v2.0
        df['atr'] = self._calculate_atr(df, self.atr_period)
        df['adx'] = self._calculate_adx(df, self.adx_period)  # MARKET REGIME v6.0

        # Get current and previous values
        current = df.iloc[-1]
        previous = df.iloc[-2]

        # Extract values
        current_price = Decimal(str(current['close']))
        current_high = Decimal(str(current['high']))
        current_low = Decimal(str(current['low']))
        current_volume = Decimal(str(current['volume']))

        upper_channel = Decimal(str(previous['upper_channel']))  # Use PREVIOUS bar's channel
        lower_channel = Decimal(str(previous['lower_channel']))
        exit_channel = Decimal(str(previous['exit_channel']))
        atr = Decimal(str(current['atr']))
        adx = Decimal(str(current['adx'])) if pd.notna(current['adx']) else None

        # SNIPER filter values
        sma_200 = Decimal(str(current['sma_200'])) if pd.notna(current['sma_200']) else None
        volume_avg_20 = Decimal(str(current['volume_avg_20'])) if pd.notna(current['volume_avg_20']) else None

        # v6.2: Use external daily SMA if provided (proper 200-DAY trend filter)
        # This ensures all timeframes use the same daily 200-day SMA for trend filtering
        if daily_sma_200 is not None:
            sma_200 = daily_sma_200

        # SNIPER filter checks (LONGS)
        above_sma_200 = current_price > sma_200 if sma_200 else False
        volume_ratio = (current_volume / volume_avg_20) if volume_avg_20 and volume_avg_20 > 0 else Decimal("0")
        volume_confirmed = volume_ratio >= self.volume_multiplier if volume_avg_20 else False

        # BEARISH filter checks (SHORTS - v6.1)
        below_sma_200 = current_price < sma_200 if sma_200 else False
        short_filters_passed = below_sma_200 and volume_confirmed

        # MARKET REGIME checks
        trend_strong = adx >= self.adx_threshold if adx else True
        if not self.use_adx_filter:
            trend_strong = True  # Always strong if filter disabled

        filters_passed = above_sma_200 and volume_confirmed and trend_strong

        # Channel metrics
        channel_width = upper_channel - lower_channel
        channel_width_pct = (channel_width / current_price * 100) if current_price > 0 else Decimal("0")
        atr_percent = (atr / current_price * 100) if current_price > 0 else Decimal("0")

        # Distance from channels
        distance_from_upper = current_high - upper_channel
        distance_from_lower = current_low - lower_channel

        # Detect breakout
        # UPPER BREAKOUT: Current high exceeds previous upper channel
        is_upper_breakout = current_high > upper_channel
        # LOWER BREAKOUT (for exits): Current low drops below exit channel
        is_lower_breakout = current_low < exit_channel

        if is_upper_breakout:
            breakout_type = BreakoutType.UPPER
        elif is_lower_breakout:
            breakout_type = BreakoutType.LOWER
        else:
            breakout_type = BreakoutType.NONE

        # Calculate stops (for potential entry)
        stop_loss_price = current_price - (atr * self.atr_stop_multiplier)
        take_profit_price = current_price + (atr * self.atr_tp_multiplier)
        risk_amount = current_price - stop_loss_price
        reward_amount = take_profit_price - current_price
        risk_reward_ratio = reward_amount / risk_amount if risk_amount != 0 else Decimal("0")

        # =====================================================================
        # CHANDELIER EXIT CALCULATION (v5.0 - The Ratchet) - LONGS
        # =====================================================================
        chandelier_stop: Optional[Decimal] = None
        chandelier_triggered = False

        if has_open_position and not is_short_position and highest_high_since_entry is not None:
            # The Ratchet: Chandelier Stop = Highest High Since Entry - (3 * ATR)
            chandelier_stop = highest_high_since_entry - (atr * self.chandelier_atr_multiplier)

            # Check if price has fallen below the chandelier stop
            # We use CLOSE price for chandelier (not low) to avoid whipsaws
            chandelier_triggered = current_price < chandelier_stop

        # =====================================================================
        # SHORT CHANDELIER EXIT CALCULATION (v6.1 - Inverse Ratchet) - SHORTS
        # =====================================================================
        short_chandelier_stop: Optional[Decimal] = None
        short_chandelier_triggered = False

        if has_open_position and is_short_position and lowest_low_since_entry is not None:
            # Inverse Ratchet: Short Chandelier Stop = Lowest Low Since Entry + (3 * ATR)
            short_chandelier_stop = lowest_low_since_entry + (atr * self.chandelier_atr_multiplier)

            # Check if price has risen above the short chandelier stop
            # We use CLOSE price for chandelier (not high) to avoid whipsaws
            short_chandelier_triggered = current_price > short_chandelier_stop

        # Build context (with SNIPER filter values and Chandelier Exit)
        context = SignalContext(
            timestamp=eval_time,
            current_price=current_price,
            open_price=Decimal(str(current['open'])),
            high_price=current_high,
            low_price=current_low,
            close_price=current_price,
            volume=current_volume,
            upper_channel=upper_channel,
            lower_channel=lower_channel,
            exit_channel=exit_channel,
            channel_width=channel_width,
            channel_width_pct=channel_width_pct,
            breakout_type=breakout_type,
            distance_from_upper=distance_from_upper,
            distance_from_lower=distance_from_lower,
            atr=atr,
            atr_percent=atr_percent,
            # SNIPER filter context (v2.0)
            sma_200=sma_200,
            above_sma_200=above_sma_200,
            volume_avg_20=volume_avg_20,
            volume_ratio=volume_ratio,
            volume_confirmed=volume_confirmed,
            filters_passed=filters_passed,
            # CHANDELIER EXIT context (v5.0)
            highest_high_since_entry=highest_high_since_entry,
            chandelier_stop=chandelier_stop,
            chandelier_triggered=chandelier_triggered,
            # SHORT CHANDELIER EXIT context (v6.1)
            lowest_low_since_entry=lowest_low_since_entry,
            short_chandelier_stop=short_chandelier_stop,
            short_chandelier_triggered=short_chandelier_triggered,
            below_sma_200=below_sma_200,
            is_short_position=is_short_position,
            # Thresholds
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=risk_reward_ratio
        )

        # =====================================================================
        # DECISION LOGIC - TURTLE-4 (v6.1) with Chandelier Exit + SHORT Capability
        # =====================================================================

        # =====================================================================
        # EXIT SIGNALS FIRST (protect capital)
        # =====================================================================

        # Case A: SHORT CHANDELIER EXIT (PRIMARY) - Inverse Ratchet triggered
        if has_open_position and is_short_position and short_chandelier_triggered and short_chandelier_stop is not None:
            # Calculate P&L info for short (inverted: entry - current)
            pnl_info = ""
            if entry_price is not None:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                pnl_info = f" P&L: {float(pnl_pct):+.2f}%."

            return SignalResult(
                signal=Signal.EXIT_SHORT,
                context=context,
                reason=(
                    f"SHORT CHANDELIER EXIT! Close ({float(current_price):.2f}) rose above "
                    f"Inverse Ratchet Stop ({float(short_chandelier_stop):.2f}). "
                    f"LL since entry: {float(lowest_low_since_entry):.2f}, ATR: {float(atr):.2f}.{pnl_info}"
                ),
                confidence=Decimal("0.9")  # High confidence - inverse ratchet triggered
            )

        # Case B: BACKUP EXIT for SHORTS - Upper breakout while short
        if breakout_type == BreakoutType.UPPER and has_open_position and is_short_position:
            pnl_info = ""
            if entry_price is not None:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                pnl_info = f" P&L: {float(pnl_pct):+.2f}%."

            return SignalResult(
                signal=Signal.EXIT_SHORT,
                context=context,
                reason=(
                    f"SHORT BACKUP EXIT! Price high ({float(current_high):.2f}) broke above "
                    f"{self.entry_period}-period high ({float(upper_channel):.2f}). "
                    f"Inverse Chandelier didn't trigger first. Closing at {float(current_price):.2f}.{pnl_info}"
                )
            )

        # =====================================================================
        # LONG ENTRY AND EXIT SIGNALS
        # =====================================================================

        # Case 1: Upper breakout + no position + SNIPER filters pass → LONG
        if breakout_type == BreakoutType.UPPER and not has_open_position:
            # Check SNIPER and MARKET REGIME filters
            if not filters_passed:
                # Build rejection reason
                rejection_reasons = []
                if not above_sma_200:
                    rejection_reasons.append(
                        f"TREND FILTER FAILED: Price ({float(current_price):.2f}) < SMA200 ({float(sma_200):.2f})" if sma_200 else "No SMA200 data"
                    )
                if not volume_confirmed:
                    rejection_reasons.append(
                        f"VOLUME FILTER FAILED: Volume ratio ({float(volume_ratio):.2f}x) < {float(self.volume_multiplier)}x required"
                    )
                if self.use_adx_filter and not trend_strong:
                    rejection_reasons.append(
                        f"ADX FILTER FAILED: ADX ({float(adx):.1f}) < {float(self.adx_threshold)} threshold" if adx else "No ADX data"
                    )

                return SignalResult(
                    signal=Signal.HOLD,
                    context=context,
                    reason=(
                        f"SNIPER REJECTED: Breakout detected but filters failed. "
                        + " | ".join(rejection_reasons)
                    ),
                    confidence=Decimal("0.3")  # Low confidence - breakout but filtered
                )

            # All filters passed → SNIPER ENTRY
            adx_info = f" ADX ✓ ({float(adx):.1f})" if adx else ""
            return SignalResult(
                signal=Signal.LONG,
                context=context,
                reason=(
                    f"SNIPER ENTRY! Breakout at {float(current_high):.2f} > {float(upper_channel):.2f}. "
                    f"TREND ✓ (above SMA200 {float(sma_200):.2f}). "
                    f"VOLUME ✓ ({float(volume_ratio):.1f}x avg).{adx_info} "
                    f"Stop: {float(stop_loss_price):.2f}, R:R = 1:{float(risk_reward_ratio):.1f}"
                ),
                confidence=Decimal("0.8")  # High confidence - all filters passed
            )

        # =====================================================================
        # SHORT ENTRY SIGNAL (v6.1)
        # =====================================================================

        # Case 1.5: Lower breakout + no position + BEARISH filters pass → SHORT
        if breakout_type == BreakoutType.LOWER and not has_open_position:
            # Check BEARISH filters (inverted trend + volume)
            if not short_filters_passed:
                # Build rejection reason
                rejection_reasons = []
                if not below_sma_200:
                    rejection_reasons.append(
                        f"BEARISH TREND FILTER FAILED: Price ({float(current_price):.2f}) > SMA200 ({float(sma_200):.2f})" if sma_200 else "No SMA200 data"
                    )
                if not volume_confirmed:
                    rejection_reasons.append(
                        f"VOLUME FILTER FAILED: Volume ratio ({float(volume_ratio):.2f}x) < {float(self.volume_multiplier)}x required"
                    )

                return SignalResult(
                    signal=Signal.HOLD,
                    context=context,
                    reason=(
                        f"SHORT REJECTED: Lower breakout detected but bearish filters failed. "
                        + " | ".join(rejection_reasons)
                    ),
                    confidence=Decimal("0.3")  # Low confidence - breakout but filtered
                )

            # Calculate short-specific stops (inverted)
            short_stop_loss_price = current_price + (atr * self.atr_stop_multiplier)
            short_take_profit_price = current_price - (atr * self.atr_tp_multiplier)
            short_risk_amount = short_stop_loss_price - current_price
            short_reward_amount = current_price - short_take_profit_price
            short_risk_reward_ratio = short_reward_amount / short_risk_amount if short_risk_amount != 0 else Decimal("0")

            # All bearish filters passed → SHORT ENTRY
            return SignalResult(
                signal=Signal.SHORT,
                context=context,
                reason=(
                    f"SHORT ENTRY! Breakdown at {float(current_low):.2f} < {float(lower_channel):.2f}. "
                    f"BEARISH ✓ (below SMA200 {float(sma_200):.2f}). "
                    f"VOLUME ✓ ({float(volume_ratio):.1f}x avg). "
                    f"Stop: {float(short_stop_loss_price):.2f}, R:R = 1:{float(short_risk_reward_ratio):.1f}"
                ),
                confidence=Decimal("0.8")  # High confidence - all bearish filters passed
            )

        # Case 2: CHANDELIER EXIT (PRIMARY) - The Ratchet triggered
        if has_open_position and chandelier_triggered and chandelier_stop is not None:
            # Calculate P&L info if we have entry price
            pnl_info = ""
            if entry_price is not None:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                pnl_info = f" P&L: {float(pnl_pct):+.2f}%."

            return SignalResult(
                signal=Signal.EXIT_LONG,
                context=context,
                reason=(
                    f"CHANDELIER EXIT! Close ({float(current_price):.2f}) fell below "
                    f"Ratchet Stop ({float(chandelier_stop):.2f}). "
                    f"HH since entry: {float(highest_high_since_entry):.2f}, ATR: {float(atr):.2f}.{pnl_info}"
                ),
                confidence=Decimal("0.9")  # High confidence - ratchet triggered
            )

        # Case 3: BACKUP EXIT - Lower breakout (below 55-period low) - LONGS ONLY
        if breakout_type == BreakoutType.LOWER and has_open_position and not is_short_position:
            return SignalResult(
                signal=Signal.EXIT_LONG,
                context=context,
                reason=(
                    f"BACKUP EXIT! Price low ({float(current_low):.2f}) dropped below "
                    f"{self.exit_period}-period low ({float(exit_channel):.2f}). "
                    f"Chandelier didn't trigger first. Closing at {float(current_price):.2f}."
                )
            )

        # Case 4: Has position, holding → HOLD (ride the trend, show chandelier status)
        if has_open_position:
            if is_short_position:
                # SHORT position - show inverse chandelier info
                chandelier_info = ""
                if short_chandelier_stop is not None:
                    distance_to_stop = short_chandelier_stop - current_price
                    distance_pct = (distance_to_stop / current_price) * 100
                    chandelier_info = f" Short Chandelier: {float(short_chandelier_stop):.2f} ({float(distance_pct):.1f}% above)."

                return SignalResult(
                    signal=Signal.HOLD,
                    context=context,
                    reason=(
                        f"Holding SHORT: Price ({float(current_price):.2f}) below "
                        f"upper channel ({float(upper_channel):.2f}).{chandelier_info}"
                    )
                )
            else:
                # LONG position - show regular chandelier info
                chandelier_info = ""
                if chandelier_stop is not None:
                    distance_to_stop = current_price - chandelier_stop
                    distance_pct = (distance_to_stop / current_price) * 100
                    chandelier_info = f" Chandelier: {float(chandelier_stop):.2f} ({float(distance_pct):.1f}% below)."

                return SignalResult(
                    signal=Signal.HOLD,
                    context=context,
                    reason=(
                        f"Holding LONG: Price ({float(current_price):.2f}) above "
                        f"exit channel ({float(exit_channel):.2f}).{chandelier_info}"
                    )
                )

        # Case 5: No position, no breakout → HOLD (wait for breakout)
        return SignalResult(
            signal=Signal.HOLD,
            context=context,
            reason=(
                f"Waiting: Price ({float(current_price):.2f}) between channels. "
                f"Upper: {float(upper_channel):.2f}, Lower: {float(lower_channel):.2f}. "
                f"Need breakout to enter."
            )
        )

    def _create_result(
        self,
        signal: Signal,
        reason: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> SignalResult:
        """Create SignalResult with minimal context (for error cases)"""
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
            upper_channel=Decimal("0"),
            lower_channel=Decimal("0"),
            exit_channel=Decimal("0"),
            channel_width=Decimal("0"),
            channel_width_pct=Decimal("0"),
            breakout_type=BreakoutType.NONE,
            distance_from_upper=Decimal("0"),
            distance_from_lower=Decimal("0"),
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
        risk_percent: Decimal = Decimal("0.01")
    ) -> Decimal:
        """
        Calculate position size based on risk.
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
    entry_period: int = 55,  # Major breakout only
    exit_period: int = 55,   # Backup exit (Chandelier usually hits first)
    atr_period: int = 14,
    stop_multiplier: float = 2.0,
    tp_multiplier: float = 4.0,
    chandelier_multiplier: float = 3.0,  # THE RATCHET
    trend_period: int = 200,
    volume_avg_period: int = 20,
    volume_multiplier: float = 1.5,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    use_adx_filter: bool = False
) -> DonchianBreakout:
    """Factory function to create Donchian TURTLE-4 strategy (v6.0) - The Ratchet + ADX"""
    return DonchianBreakout(
        entry_period=entry_period,
        exit_period=exit_period,
        atr_period=atr_period,
        atr_stop_multiplier=stop_multiplier,
        atr_tp_multiplier=tp_multiplier,
        chandelier_atr_multiplier=chandelier_multiplier,
        trend_period=trend_period,
        volume_avg_period=volume_avg_period,
        volume_multiplier=volume_multiplier,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        use_adx_filter=use_adx_filter
    )


# Default strategy instance (TURTLE-4 v5.0 - The Ratchet)
default_strategy = DonchianBreakout()
