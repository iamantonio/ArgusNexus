"""
Professional Multi-Timeframe Trading System

"Trade like a professional human, but 24/7 with faster decisions."

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    ARGUS PROFESSIONAL MODE                   │
├─────────────────────────────────────────────────────────────┤
│  CONTEXT LAYER (Daily)     - Overall trend, key levels      │
│  SETUP LAYER (4h)          - Valid Donchian setup forming?  │
│  TIMING LAYER (1h)         - Entry trigger confirmation     │
│  EXECUTION                 - Only A+ setups, all aligned    │
└─────────────────────────────────────────────────────────────┘

Setup Grades:
- A+ : All 3 timeframes aligned, strong trend, volume confirmed
- A  : 2 timeframes aligned, decent trend
- B  : 1 timeframe signal, caution advised
- C  : Conflicting signals, no trade

Position Sizing by Conviction:
- A+ : 1.5x base position (high conviction)
- A  : 1.0x base position (standard)
- B  : 0.5x base position (reduced)
- C  : 0.0x (no trade)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import logging

from .donchian import DonchianBreakout, SignalResult, Signal

logger = logging.getLogger(__name__)


class SetupGrade(Enum):
    """Setup quality grades - determines position sizing"""
    A_PLUS = "A+"   # All aligned, high conviction → 1.5x position
    A = "A"         # 2 TFs aligned, good setup → 1.0x position
    B = "B"         # 1 TF signal, caution → 0.5x position
    C = "C"         # Conflicting/no signal → 0x (no trade)


class TrendBias(Enum):
    """Overall market trend direction"""
    STRONG_BULLISH = "strong_bullish"   # Clear uptrend, above all MAs
    BULLISH = "bullish"                  # Uptrend, above key MAs
    NEUTRAL = "neutral"                  # Ranging, no clear direction
    BEARISH = "bearish"                  # Downtrend, below key MAs
    STRONG_BEARISH = "strong_bearish"   # Clear downtrend


@dataclass
class KeyLevel:
    """Support/Resistance level"""
    price: Decimal
    level_type: str  # "support" or "resistance"
    strength: float  # 0-1, based on touches/importance
    timeframe: str   # Which TF identified this level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": float(self.price),
            "type": self.level_type,
            "strength": self.strength,
            "timeframe": self.timeframe
        }


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe"""
    timeframe: str
    trend_bias: TrendBias
    signal: Signal
    signal_strength: float  # 0-1 confidence in the signal

    # Price structure
    current_price: Decimal
    sma_20: Optional[Decimal] = None
    sma_50: Optional[Decimal] = None
    sma_200: Optional[Decimal] = None

    # Momentum
    rsi: Optional[float] = None
    adx: Optional[float] = None

    # Donchian
    upper_channel: Optional[Decimal] = None
    lower_channel: Optional[Decimal] = None
    channel_width_pct: Optional[float] = None

    # Volume
    volume_ratio: Optional[float] = None  # vs 20-period average

    # Key levels near current price
    nearby_support: Optional[Decimal] = None
    nearby_resistance: Optional[Decimal] = None

    # Raw signal result from Donchian strategy
    donchian_result: Optional[SignalResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "trend_bias": self.trend_bias.value,
            "signal": self.signal.value,
            "signal_strength": self.signal_strength,
            "current_price": float(self.current_price),
            "moving_averages": {
                "sma_20": float(self.sma_20) if self.sma_20 else None,
                "sma_50": float(self.sma_50) if self.sma_50 else None,
                "sma_200": float(self.sma_200) if self.sma_200 else None,
            },
            "momentum": {
                "rsi": self.rsi,
                "adx": self.adx,
            },
            "donchian": {
                "upper_channel": float(self.upper_channel) if self.upper_channel else None,
                "lower_channel": float(self.lower_channel) if self.lower_channel else None,
                "channel_width_pct": self.channel_width_pct,
            },
            "volume_ratio": self.volume_ratio,
            "key_levels": {
                "nearby_support": float(self.nearby_support) if self.nearby_support else None,
                "nearby_resistance": float(self.nearby_resistance) if self.nearby_resistance else None,
            }
        }


@dataclass
class ProfessionalSetup:
    """
    Complete professional setup analysis across all timeframes.
    This is the final output that determines if we trade.
    """
    symbol: str
    timestamp: datetime

    # The grade - this determines everything
    grade: SetupGrade

    # Position sizing multiplier based on grade
    position_multiplier: float

    # Final trading decision
    signal: Signal

    # Why this grade was assigned
    grade_reason: str

    # Individual timeframe analyses
    context_layer: TimeframeAnalysis   # Daily
    setup_layer: TimeframeAnalysis     # 4h
    timing_layer: TimeframeAnalysis    # 1h

    # Alignment metrics
    trend_alignment: bool      # Do all TFs agree on direction?
    momentum_alignment: bool   # Is momentum confirming?
    volume_confirmation: bool  # Is volume supporting the move?

    # Key levels context
    key_levels: List[KeyLevel] = field(default_factory=list)
    distance_to_support_pct: Optional[float] = None
    distance_to_resistance_pct: Optional[float] = None

    # Risk context
    suggested_stop: Optional[Decimal] = None
    suggested_target: Optional[Decimal] = None
    risk_reward_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "grade": self.grade.value,
            "position_multiplier": self.position_multiplier,
            "signal": self.signal.value,
            "grade_reason": self.grade_reason,
            "layers": {
                "context_daily": self.context_layer.to_dict(),
                "setup_4h": self.setup_layer.to_dict(),
                "timing_1h": self.timing_layer.to_dict(),
            },
            "alignment": {
                "trend": self.trend_alignment,
                "momentum": self.momentum_alignment,
                "volume": self.volume_confirmation,
            },
            "key_levels": [kl.to_dict() for kl in self.key_levels],
            "distance_to_support_pct": self.distance_to_support_pct,
            "distance_to_resistance_pct": self.distance_to_resistance_pct,
            "risk": {
                "stop": float(self.suggested_stop) if self.suggested_stop else None,
                "target": float(self.suggested_target) if self.suggested_target else None,
                "risk_reward": self.risk_reward_ratio,
            }
        }


class MultiTimeframeAnalyzer:
    """
    Professional multi-timeframe trading analyzer.

    Thinks like a human professional trader:
    1. Check Daily for overall bias and key levels
    2. Check 4h for setup formation
    3. Check 1h for entry timing
    4. Only trade when all layers align

    Interval mapping (24/7 crypto):
    - "daily" context   → Updated every 4 hours
    - "4h" setup        → Updated every hour
    - "1h" timing       → Updated every 15 minutes
    """

    # Timeframe hierarchy
    CONTEXT_TF = "1d"   # Daily for context
    SETUP_TF = "4h"     # 4-hour for setups
    TIMING_TF = "1h"    # 1-hour for timing

    # Position multipliers by grade
    GRADE_MULTIPLIERS = {
        SetupGrade.A_PLUS: 1.5,
        SetupGrade.A: 1.0,
        SetupGrade.B: 0.5,
        SetupGrade.C: 0.0,
    }

    # SHORT position sizing reduction (v6.1)
    # Shorts use 50% of normal position size due to unlimited loss potential
    SHORT_POSITION_MULTIPLIER = 0.5

    def __init__(
        self,
        # Donchian parameters (used on all timeframes)
        entry_period: int = 55,
        exit_period: int = 55,
        atr_period: int = 14,
        chandelier_multiplier: float = 3.0,
        # Trend filter
        trend_sma_period: int = 200,
        # Volume filter
        volume_avg_period: int = 20,
        volume_multiplier: float = 1.5,
        # ADX filter
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        # Grade thresholds
        min_volume_for_a_plus: float = 2.0,  # 2x average volume for A+
        min_adx_for_a_plus: float = 30.0,    # Strong trend for A+
    ):
        # Store parameters
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.chandelier_multiplier = chandelier_multiplier
        self.trend_sma_period = trend_sma_period
        self.volume_avg_period = volume_avg_period
        self.volume_multiplier = volume_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.min_volume_for_a_plus = min_volume_for_a_plus
        self.min_adx_for_a_plus = min_adx_for_a_plus

        # Create strategy instances for each timeframe
        self.strategies = {
            self.CONTEXT_TF: DonchianBreakout(
                entry_period=entry_period,
                exit_period=exit_period,
                atr_period=atr_period,
                chandelier_atr_multiplier=chandelier_multiplier,
                trend_period=trend_sma_period,
                volume_avg_period=volume_avg_period,
                volume_multiplier=volume_multiplier,
                adx_period=adx_period,
                adx_threshold=adx_threshold,
                use_adx_filter=True
            ),
            self.SETUP_TF: DonchianBreakout(
                entry_period=entry_period,
                exit_period=exit_period,
                atr_period=atr_period,
                chandelier_atr_multiplier=chandelier_multiplier,
                trend_period=trend_sma_period,
                volume_avg_period=volume_avg_period,
                volume_multiplier=volume_multiplier,
                adx_period=adx_period,
                adx_threshold=adx_threshold,
                use_adx_filter=True
            ),
            self.TIMING_TF: DonchianBreakout(
                entry_period=entry_period,
                exit_period=exit_period,
                atr_period=atr_period,
                chandelier_atr_multiplier=chandelier_multiplier,
                trend_period=trend_sma_period,
                volume_avg_period=volume_avg_period,
                volume_multiplier=volume_multiplier,
                adx_period=adx_period,
                adx_threshold=adx_threshold,
                use_adx_filter=True
            ),
        }

        logger.info(f"MultiTimeframeAnalyzer initialized: {self.CONTEXT_TF}/{self.SETUP_TF}/{self.TIMING_TF}")

    def analyze_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        has_position: bool = False,
        entry_price: Optional[Decimal] = None,
        highest_high_since_entry: Optional[Decimal] = None,
        # SHORT CAPABILITY (v6.1)
        is_short_position: bool = False,
        lowest_low_since_entry: Optional[Decimal] = None,
        # v6.2: External daily SMA for proper 200-day trend filter
        daily_sma_200: Optional[Decimal] = None
    ) -> TimeframeAnalysis:
        """
        Analyze a single timeframe.

        Args:
            df: OHLCV DataFrame for this timeframe
            timeframe: "1d", "4h", or "1h"
            has_position: Whether we have an open position
            entry_price: Entry price if we have a position
            highest_high_since_entry: Highest high since entry (for chandelier)

        Returns:
            TimeframeAnalysis with all relevant metrics
        """
        if len(df) < 200:
            logger.warning(f"Insufficient data for {timeframe}: {len(df)} bars")
            return self._empty_analysis(timeframe, df)

        strategy = self.strategies.get(timeframe, self.strategies[self.TIMING_TF])

        # Get Donchian signal
        result = strategy.evaluate(
            df=df,
            has_open_position=has_position,
            entry_price=entry_price,
            highest_high_since_entry=highest_high_since_entry,
            # SHORT CAPABILITY (v6.1)
            is_short_position=is_short_position,
            lowest_low_since_entry=lowest_low_since_entry,
            # v6.2: Use daily SMA for proper 200-day trend filter
            daily_sma_200=daily_sma_200
        )

        # Extract key metrics
        ctx = result.context
        current_price = ctx.current_price

        # Calculate trend bias
        trend_bias = self._determine_trend_bias(
            current_price=current_price,
            sma_20=self._calc_sma(df, 20),
            sma_50=self._calc_sma(df, 50),
            sma_200=ctx.sma_200,
            adx=float(ctx.adx) if ctx.adx else None
        )

        # Calculate signal strength (0-1)
        signal_strength = self._calculate_signal_strength(result, ctx)

        # Find nearby support/resistance
        support, resistance = self._find_nearby_levels(df, current_price)

        return TimeframeAnalysis(
            timeframe=timeframe,
            trend_bias=trend_bias,
            signal=result.signal,
            signal_strength=signal_strength,
            current_price=current_price,
            sma_20=self._calc_sma(df, 20),
            sma_50=self._calc_sma(df, 50),
            sma_200=ctx.sma_200,
            rsi=self._calc_rsi(df),
            adx=float(ctx.adx) if ctx.adx else None,
            upper_channel=ctx.upper_channel,
            lower_channel=ctx.lower_channel,
            channel_width_pct=float(ctx.channel_width_pct) if ctx.channel_width_pct else None,
            volume_ratio=float(ctx.volume_ratio) if ctx.volume_ratio else None,
            nearby_support=support,
            nearby_resistance=resistance,
            donchian_result=result
        )

    def analyze(
        self,
        symbol: str,
        df_daily: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        has_position: bool = False,
        entry_price: Optional[Decimal] = None,
        highest_high_since_entry: Optional[Decimal] = None,
        # SHORT CAPABILITY (v6.1)
        is_short_position: bool = False,
        lowest_low_since_entry: Optional[Decimal] = None
    ) -> ProfessionalSetup:
        """
        Complete multi-timeframe professional analysis.

        This is the main entry point. Analyzes all three timeframes
        and produces a graded setup with position sizing recommendation.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            df_daily: Daily OHLCV data (200+ bars ideal)
            df_4h: 4-hour OHLCV data (200+ bars ideal)
            df_1h: 1-hour OHLCV data (200+ bars ideal)
            has_position: Whether we have an open position
            entry_price: Entry price if we have a position
            highest_high_since_entry: For chandelier stop tracking

        Returns:
            ProfessionalSetup with grade, position multiplier, and full analysis
        """
        timestamp = datetime.utcnow()

        # v6.2: Calculate daily 200-day SMA once and pass to all timeframes
        # This ensures the trend filter uses the proper 200-DAY SMA, not 200-period on smaller TFs
        daily_sma_200 = self._calc_sma(df_daily, 200)
        if daily_sma_200 is not None:
            logger.debug(f"Using daily 200-day SMA: {daily_sma_200}")

        # Analyze each timeframe (all using the same daily SMA for trend filter)
        context_analysis = self.analyze_timeframe(
            df_daily, self.CONTEXT_TF, has_position, entry_price, highest_high_since_entry,
            is_short_position, lowest_low_since_entry, daily_sma_200
        )
        setup_analysis = self.analyze_timeframe(
            df_4h, self.SETUP_TF, has_position, entry_price, highest_high_since_entry,
            is_short_position, lowest_low_since_entry, daily_sma_200
        )
        timing_analysis = self.analyze_timeframe(
            df_1h, self.TIMING_TF, has_position, entry_price, highest_high_since_entry,
            is_short_position, lowest_low_since_entry, daily_sma_200
        )

        # Determine alignment
        trend_alignment = self._check_trend_alignment(
            context_analysis.trend_bias,
            setup_analysis.trend_bias,
            timing_analysis.trend_bias
        )

        momentum_alignment = self._check_momentum_alignment(
            context_analysis, setup_analysis, timing_analysis
        )

        volume_confirmation = self._check_volume_confirmation(
            setup_analysis, timing_analysis
        )

        # Grade the setup
        grade, grade_reason = self._grade_setup(
            context=context_analysis,
            setup=setup_analysis,
            timing=timing_analysis,
            trend_alignment=trend_alignment,
            momentum_alignment=momentum_alignment,
            volume_confirmation=volume_confirmation,
            has_position=has_position
        )

        # Determine final signal
        final_signal = self._determine_final_signal(
            grade=grade,
            context=context_analysis,
            setup=setup_analysis,
            timing=timing_analysis,
            has_position=has_position
        )

        # Extract key levels from all timeframes
        key_levels = self._extract_key_levels(
            context_analysis, setup_analysis, timing_analysis
        )

        # Calculate distances to key levels
        current_price = timing_analysis.current_price
        dist_support, dist_resistance = self._calc_level_distances(
            current_price, key_levels
        )

        # Get risk parameters from timing layer (most precise)
        suggested_stop = None
        suggested_target = None
        risk_reward = None

        if timing_analysis.donchian_result:
            ctx = timing_analysis.donchian_result.context
            suggested_stop = ctx.stop_loss_price
            suggested_target = ctx.take_profit_price
            if ctx.risk_reward_ratio:
                risk_reward = float(ctx.risk_reward_ratio)

        return ProfessionalSetup(
            symbol=symbol,
            timestamp=timestamp,
            grade=grade,
            position_multiplier=self.GRADE_MULTIPLIERS[grade],
            signal=final_signal,
            grade_reason=grade_reason,
            context_layer=context_analysis,
            setup_layer=setup_analysis,
            timing_layer=timing_analysis,
            trend_alignment=trend_alignment,
            momentum_alignment=momentum_alignment,
            volume_confirmation=volume_confirmation,
            key_levels=key_levels,
            distance_to_support_pct=dist_support,
            distance_to_resistance_pct=dist_resistance,
            suggested_stop=suggested_stop,
            suggested_target=suggested_target,
            risk_reward_ratio=risk_reward
        )

    def _determine_trend_bias(
        self,
        current_price: Decimal,
        sma_20: Optional[Decimal],
        sma_50: Optional[Decimal],
        sma_200: Optional[Decimal],
        adx: Optional[float]
    ) -> TrendBias:
        """Determine trend bias based on price vs moving averages."""
        if not sma_200:
            return TrendBias.NEUTRAL

        above_200 = current_price > sma_200
        above_50 = current_price > sma_50 if sma_50 else above_200
        above_20 = current_price > sma_20 if sma_20 else above_50

        # Strong trend requires ADX > 25
        strong_trend = adx and adx > 25

        if above_20 and above_50 and above_200:
            return TrendBias.STRONG_BULLISH if strong_trend else TrendBias.BULLISH
        elif not above_20 and not above_50 and not above_200:
            return TrendBias.STRONG_BEARISH if strong_trend else TrendBias.BEARISH
        else:
            return TrendBias.NEUTRAL

    def _calculate_signal_strength(
        self,
        result: SignalResult,
        ctx
    ) -> float:
        """Calculate signal strength 0-1 based on multiple factors."""
        strength = 0.0
        factors = 0

        # Base: Signal type
        if result.signal in (Signal.LONG, Signal.SHORT):
            strength += 0.5
            factors += 1
        elif result.signal in (Signal.EXIT_LONG, Signal.EXIT_SHORT):
            strength += 0.3
            factors += 1
        else:
            strength += 0.1
            factors += 1

        # Trend filter passed
        if ctx.above_sma_200:
            strength += 0.2
            factors += 1

        # Volume confirmed
        if ctx.volume_confirmed:
            strength += 0.2
            factors += 1

        # ADX shows strong trend
        if ctx.trend_strong:
            strength += 0.1
            factors += 1

        return min(1.0, strength)

    def _check_trend_alignment(
        self,
        context_bias: TrendBias,
        setup_bias: TrendBias,
        timing_bias: TrendBias
    ) -> bool:
        """Check if all timeframes agree on trend direction."""
        bullish_biases = {TrendBias.BULLISH, TrendBias.STRONG_BULLISH}
        bearish_biases = {TrendBias.BEARISH, TrendBias.STRONG_BEARISH}

        # All bullish
        if all(b in bullish_biases for b in [context_bias, setup_bias, timing_bias]):
            return True

        # All bearish
        if all(b in bearish_biases for b in [context_bias, setup_bias, timing_bias]):
            return True

        # At least 2 agree and none strongly oppose
        biases = [context_bias, setup_bias, timing_bias]
        bullish_count = sum(1 for b in biases if b in bullish_biases)
        bearish_count = sum(1 for b in biases if b in bearish_biases)

        if bullish_count >= 2 and bearish_count == 0:
            return True
        if bearish_count >= 2 and bullish_count == 0:
            return True

        return False

    def _check_momentum_alignment(
        self,
        context: TimeframeAnalysis,
        setup: TimeframeAnalysis,
        timing: TimeframeAnalysis
    ) -> bool:
        """Check if momentum indicators are aligned."""
        # RSI alignment: all above 50 (bullish) or all below 50 (bearish)
        rsi_values = [a.rsi for a in [context, setup, timing] if a.rsi]

        if len(rsi_values) >= 2:
            all_bullish_rsi = all(r > 50 for r in rsi_values)
            all_bearish_rsi = all(r < 50 for r in rsi_values)
            if all_bullish_rsi or all_bearish_rsi:
                return True

        # ADX alignment: at least 2 show strong trend
        adx_values = [a.adx for a in [context, setup, timing] if a.adx]
        if len(adx_values) >= 2:
            strong_count = sum(1 for a in adx_values if a > 25)
            if strong_count >= 2:
                return True

        return False

    def _check_volume_confirmation(
        self,
        setup: TimeframeAnalysis,
        timing: TimeframeAnalysis
    ) -> bool:
        """Check if volume supports the move."""
        # Need above-average volume on at least one of setup/timing
        setup_vol = setup.volume_ratio or 0
        timing_vol = timing.volume_ratio or 0

        return setup_vol > 1.0 or timing_vol > 1.0

    def _grade_setup(
        self,
        context: TimeframeAnalysis,
        setup: TimeframeAnalysis,
        timing: TimeframeAnalysis,
        trend_alignment: bool,
        momentum_alignment: bool,
        volume_confirmation: bool,
        has_position: bool
    ) -> Tuple[SetupGrade, str]:
        """
        Grade the setup quality.

        A+ Requirements:
        - All 3 TFs show same signal direction
        - Trend alignment across all TFs
        - Volume > 2x average on timing
        - ADX > 30 on setup

        A Requirements:
        - 2 TFs agree on direction
        - Trend mostly aligned
        - Volume confirms

        B Requirements:
        - 1 TF shows signal
        - Some alignment

        C: Everything else (no trade)
        """
        # Count active signals
        signals = [context.signal, setup.signal, timing.signal]
        long_count = sum(1 for s in signals if s == Signal.LONG)
        short_count = sum(1 for s in signals if s == Signal.SHORT)  # v6.1: SHORT signals
        exit_count = sum(1 for s in signals if s in (Signal.EXIT_LONG, Signal.EXIT_SHORT))

        # For exits, we're more lenient (protect capital)
        if has_position and exit_count >= 1:
            if exit_count >= 2:
                return SetupGrade.A, "Multiple exit signals - protect capital"
            return SetupGrade.B, "Single exit signal - consider exiting"

        # For entries, we're strict (need at least one signal)
        if long_count == 0 and short_count == 0:
            return SetupGrade.C, "No entry signals across timeframes"

        # Determine primary direction (prefer shorts in bearish trend, longs in bullish)
        is_bearish_setup = short_count > long_count

        # Check for A+ (all aligned, strong conviction)
        timing_vol = timing.volume_ratio or 0
        setup_adx = setup.adx or 0

        # Use the dominant signal count (long or short)
        signal_count = short_count if is_bearish_setup else long_count
        direction = "SHORT" if is_bearish_setup else "LONG"

        if (signal_count >= 2 and
            trend_alignment and
            momentum_alignment and
            volume_confirmation and
            timing_vol >= self.min_volume_for_a_plus and
            setup_adx >= self.min_adx_for_a_plus):
            return SetupGrade.A_PLUS, f"{direction} A+: All aligned: {signal_count}/3 signals, strong trend (ADX {setup_adx:.0f}), high volume ({timing_vol:.1f}x)"

        # Check for A (2 aligned, decent setup)
        if signal_count >= 2 and trend_alignment and (momentum_alignment or volume_confirmation):
            reason = f"{direction} A: Good setup: {signal_count}/3 signals"
            if momentum_alignment:
                reason += ", momentum aligned"
            if volume_confirmation:
                reason += ", volume confirmed"
            return SetupGrade.A, reason

        # Check for B (1 signal, some alignment)
        if signal_count >= 1 and (trend_alignment or momentum_alignment):
            return SetupGrade.B, f"{direction} B: Partial setup: {signal_count}/3 signals, limited alignment - reduced size"

        # C - no trade
        reasons = []
        if not trend_alignment:
            reasons.append("conflicting trends")
        if not momentum_alignment:
            reasons.append("weak momentum")
        if not volume_confirmation:
            reasons.append("low volume")

        return SetupGrade.C, f"No trade: {', '.join(reasons) or 'insufficient alignment'}"

    def _determine_final_signal(
        self,
        grade: SetupGrade,
        context: TimeframeAnalysis,
        setup: TimeframeAnalysis,
        timing: TimeframeAnalysis,
        has_position: bool
    ) -> Signal:
        """Determine the final trading signal based on grade and analysis."""
        # C grade = no trade
        if grade == SetupGrade.C:
            return Signal.HOLD

        # Check for exit signals first (protect capital)
        if has_position:
            if timing.signal in (Signal.EXIT_LONG, Signal.EXIT_SHORT):
                return timing.signal
            if setup.signal in (Signal.EXIT_LONG, Signal.EXIT_SHORT):
                return setup.signal

        # For LONG entries, timing layer is the trigger
        if timing.signal == Signal.LONG:
            return Signal.LONG

        # For SHORT entries (v6.1), timing layer is the trigger
        if timing.signal == Signal.SHORT:
            return Signal.SHORT

        # If timing isn't ready but setup is, hold for better entry
        if setup.signal == Signal.LONG and timing.signal == Signal.HOLD:
            return Signal.HOLD  # Wait for timing layer confirmation

        # If setup shows SHORT but timing isn't ready, hold for better entry
        if setup.signal == Signal.SHORT and timing.signal == Signal.HOLD:
            return Signal.HOLD  # Wait for timing layer confirmation

        return Signal.HOLD

    def _find_nearby_levels(
        self,
        df: pd.DataFrame,
        current_price: Decimal
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Find nearby support and resistance from recent price action."""
        if len(df) < 20:
            return None, None

        # Use recent swing highs/lows
        highs = df['high'].tail(50).values
        lows = df['low'].tail(50).values

        price_float = float(current_price)

        # Find resistance (nearest high above current price)
        resistance_candidates = [h for h in highs if h > price_float]
        resistance = Decimal(str(min(resistance_candidates))) if resistance_candidates else None

        # Find support (nearest low below current price)
        support_candidates = [l for l in lows if l < price_float]
        support = Decimal(str(max(support_candidates))) if support_candidates else None

        return support, resistance

    def _extract_key_levels(
        self,
        context: TimeframeAnalysis,
        setup: TimeframeAnalysis,
        timing: TimeframeAnalysis
    ) -> List[KeyLevel]:
        """Extract key support/resistance levels from all timeframes."""
        levels = []

        # Donchian channels are natural S/R
        for analysis, tf_name in [(context, "daily"), (setup, "4h"), (timing, "1h")]:
            if analysis.upper_channel:
                levels.append(KeyLevel(
                    price=analysis.upper_channel,
                    level_type="resistance",
                    strength=0.8 if tf_name == "daily" else 0.6 if tf_name == "4h" else 0.4,
                    timeframe=tf_name
                ))
            if analysis.lower_channel:
                levels.append(KeyLevel(
                    price=analysis.lower_channel,
                    level_type="support",
                    strength=0.8 if tf_name == "daily" else 0.6 if tf_name == "4h" else 0.4,
                    timeframe=tf_name
                ))

            # SMAs are also key levels
            if analysis.sma_200:
                levels.append(KeyLevel(
                    price=analysis.sma_200,
                    level_type="support" if analysis.current_price > analysis.sma_200 else "resistance",
                    strength=0.9,  # 200 SMA is very significant
                    timeframe=tf_name
                ))

        # Sort by price
        levels.sort(key=lambda x: x.price)
        return levels

    def _calc_level_distances(
        self,
        current_price: Decimal,
        levels: List[KeyLevel]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate percentage distance to nearest support/resistance."""
        if not levels:
            return None, None

        supports = [l for l in levels if l.level_type == "support" and l.price < current_price]
        resistances = [l for l in levels if l.level_type == "resistance" and l.price > current_price]

        dist_support = None
        dist_resistance = None

        if supports:
            nearest_support = max(supports, key=lambda x: x.price)
            dist_support = float((current_price - nearest_support.price) / current_price * 100)

        if resistances:
            nearest_resistance = min(resistances, key=lambda x: x.price)
            dist_resistance = float((nearest_resistance.price - current_price) / current_price * 100)

        return dist_support, dist_resistance

    def _calc_sma(self, df: pd.DataFrame, period: int) -> Optional[Decimal]:
        """Calculate SMA for the given period."""
        if len(df) < period:
            return None
        return Decimal(str(df['close'].tail(period).mean()))

    def _calc_rsi(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate RSI."""
        if len(df) < period + 1:
            return None

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

    def _empty_analysis(self, timeframe: str, df: pd.DataFrame) -> TimeframeAnalysis:
        """Return empty analysis when data is insufficient."""
        current_price = Decimal(str(df['close'].iloc[-1])) if len(df) > 0 else Decimal("0")
        return TimeframeAnalysis(
            timeframe=timeframe,
            trend_bias=TrendBias.NEUTRAL,
            signal=Signal.HOLD,
            signal_strength=0.0,
            current_price=current_price
        )
