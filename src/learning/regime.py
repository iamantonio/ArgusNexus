"""
Market Regime Detection Module

Identifies the current market regime and provides adaptive parameters.
Research shows that strategies perform differently across regimes:
- High Volatility: Wider stops, smaller positions, faster exits
- Strong Trend: Let winners run, trail stops, larger positions
- Ranging: Tighter exits, counter-trend bias, smaller positions

Based on research into:
- Renaissance Technologies' regime-aware trading
- VIX-based volatility regimes
- ADX trend strength classification
- Academic research on regime-switching models

"The market is a different animal in different regimes."
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import json

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    # Volatility-based
    EXTREME_VOLATILITY = "extreme_volatility"  # Crisis mode
    HIGH_VOLATILITY = "high_volatility"        # Elevated risk
    NORMAL_VOLATILITY = "normal_volatility"    # Standard conditions
    LOW_VOLATILITY = "low_volatility"          # Compression, potential breakout

    # Trend-based
    STRONG_UPTREND = "strong_uptrend"          # Bull market
    WEAK_UPTREND = "weak_uptrend"              # Grinding higher
    RANGING = "ranging"                         # No clear direction
    WEAK_DOWNTREND = "weak_downtrend"          # Grinding lower
    STRONG_DOWNTREND = "strong_downtrend"      # Bear market

    # Special conditions
    BREAKOUT = "breakout"                       # Breaking out of range
    BREAKDOWN = "breakdown"                     # Breaking down from range
    CAPITULATION = "capitulation"               # Panic selling
    EUPHORIA = "euphoria"                       # Extreme bullishness

    UNKNOWN = "unknown"


@dataclass
class RegimeParameters:
    """
    Adaptive trading parameters based on current regime.

    These parameters adjust trading behavior to match market conditions.
    """
    # Position sizing multiplier (1.0 = normal)
    position_size_multiplier: float = 1.0

    # Stop loss adjustments
    stop_atr_multiplier: float = 3.0       # ATR multiplier for stops
    stop_widen_factor: float = 1.0         # Additional widening in volatile markets

    # Take profit adjustments
    take_profit_atr_multiplier: float = 4.0
    trail_stop_enabled: bool = True
    trail_stop_atr: float = 2.5

    # Entry filters
    min_adx_for_entry: float = 20.0        # Minimum trend strength
    require_trend_confirmation: bool = True

    # Risk adjustments
    max_daily_trades: int = 5
    max_drawdown_exit_pct: float = 5.0     # Force exit if DD exceeds this

    # Regime-specific flags
    prefer_counter_trend: bool = False
    allow_new_positions: bool = True
    aggressive_trailing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_size_multiplier": self.position_size_multiplier,
            "stop_atr_multiplier": self.stop_atr_multiplier,
            "stop_widen_factor": self.stop_widen_factor,
            "take_profit_atr_multiplier": self.take_profit_atr_multiplier,
            "trail_stop_enabled": self.trail_stop_enabled,
            "trail_stop_atr": self.trail_stop_atr,
            "min_adx_for_entry": self.min_adx_for_entry,
            "require_trend_confirmation": self.require_trend_confirmation,
            "max_daily_trades": self.max_daily_trades,
            "max_drawdown_exit_pct": self.max_drawdown_exit_pct,
            "prefer_counter_trend": self.prefer_counter_trend,
            "allow_new_positions": self.allow_new_positions,
            "aggressive_trailing": self.aggressive_trailing,
        }


# Regime-specific parameter presets
REGIME_PARAMETERS: Dict[MarketRegime, RegimeParameters] = {
    # =========================================================================
    # VOLATILITY REGIMES
    # =========================================================================
    MarketRegime.EXTREME_VOLATILITY: RegimeParameters(
        position_size_multiplier=0.25,      # Quarter size
        stop_atr_multiplier=5.0,            # Very wide stops
        stop_widen_factor=1.5,
        take_profit_atr_multiplier=3.0,     # Take profits quickly
        trail_stop_enabled=True,
        trail_stop_atr=4.0,
        min_adx_for_entry=35.0,             # Only strong trends
        require_trend_confirmation=True,
        max_daily_trades=2,                 # Limit exposure
        max_drawdown_exit_pct=3.0,
        prefer_counter_trend=False,
        allow_new_positions=False,          # No new positions in crisis
        aggressive_trailing=True,
    ),

    MarketRegime.HIGH_VOLATILITY: RegimeParameters(
        position_size_multiplier=0.5,       # Half size
        stop_atr_multiplier=4.0,            # Wider stops
        stop_widen_factor=1.25,
        take_profit_atr_multiplier=4.0,
        trail_stop_enabled=True,
        trail_stop_atr=3.0,
        min_adx_for_entry=25.0,
        require_trend_confirmation=True,
        max_daily_trades=3,
        max_drawdown_exit_pct=4.0,
        prefer_counter_trend=False,
        allow_new_positions=True,
        aggressive_trailing=True,
    ),

    MarketRegime.NORMAL_VOLATILITY: RegimeParameters(
        position_size_multiplier=1.0,       # Full size
        stop_atr_multiplier=3.0,            # Standard stops
        stop_widen_factor=1.0,
        take_profit_atr_multiplier=4.0,
        trail_stop_enabled=True,
        trail_stop_atr=2.5,
        min_adx_for_entry=20.0,
        require_trend_confirmation=True,
        max_daily_trades=5,
        max_drawdown_exit_pct=5.0,
        prefer_counter_trend=False,
        allow_new_positions=True,
        aggressive_trailing=False,
    ),

    MarketRegime.LOW_VOLATILITY: RegimeParameters(
        position_size_multiplier=1.25,      # Slightly larger (compression often precedes moves)
        stop_atr_multiplier=2.5,            # Tighter stops (low vol = small moves)
        stop_widen_factor=1.0,
        take_profit_atr_multiplier=5.0,     # Let it run if breakout
        trail_stop_enabled=True,
        trail_stop_atr=2.0,
        min_adx_for_entry=15.0,             # Lower threshold
        require_trend_confirmation=False,   # Breakouts start in low vol
        max_daily_trades=5,
        max_drawdown_exit_pct=5.0,
        prefer_counter_trend=False,
        allow_new_positions=True,
        aggressive_trailing=False,
    ),

    # =========================================================================
    # TREND REGIMES
    # =========================================================================
    MarketRegime.STRONG_UPTREND: RegimeParameters(
        position_size_multiplier=1.25,      # Larger size in strong trends
        stop_atr_multiplier=3.5,            # Give room to run
        stop_widen_factor=1.0,
        take_profit_atr_multiplier=6.0,     # Let winners run
        trail_stop_enabled=True,
        trail_stop_atr=2.5,
        min_adx_for_entry=25.0,
        require_trend_confirmation=False,   # Already confirmed
        max_daily_trades=5,
        max_drawdown_exit_pct=5.0,
        prefer_counter_trend=False,
        allow_new_positions=True,
        aggressive_trailing=False,          # Let it run
    ),

    MarketRegime.WEAK_UPTREND: RegimeParameters(
        position_size_multiplier=0.75,
        stop_atr_multiplier=3.0,
        stop_widen_factor=1.0,
        take_profit_atr_multiplier=4.0,
        trail_stop_enabled=True,
        trail_stop_atr=2.5,
        min_adx_for_entry=20.0,
        require_trend_confirmation=True,
        max_daily_trades=4,
        max_drawdown_exit_pct=4.0,
        prefer_counter_trend=False,
        allow_new_positions=True,
        aggressive_trailing=False,
    ),

    MarketRegime.RANGING: RegimeParameters(
        position_size_multiplier=0.5,       # Smaller in ranges (chop)
        stop_atr_multiplier=2.5,            # Tighter stops
        stop_widen_factor=0.8,
        take_profit_atr_multiplier=2.5,     # Take profits quickly
        trail_stop_enabled=False,           # No trailing in ranges
        trail_stop_atr=2.0,
        min_adx_for_entry=15.0,
        require_trend_confirmation=False,
        max_daily_trades=3,
        max_drawdown_exit_pct=3.0,
        prefer_counter_trend=True,          # Fade the extremes
        allow_new_positions=True,
        aggressive_trailing=False,
    ),

    MarketRegime.WEAK_DOWNTREND: RegimeParameters(
        position_size_multiplier=0.5,       # Defensive
        stop_atr_multiplier=3.0,
        stop_widen_factor=1.0,
        take_profit_atr_multiplier=3.0,
        trail_stop_enabled=True,
        trail_stop_atr=2.5,
        min_adx_for_entry=20.0,
        require_trend_confirmation=True,
        max_daily_trades=3,
        max_drawdown_exit_pct=3.0,
        prefer_counter_trend=False,
        allow_new_positions=True,
        aggressive_trailing=True,           # Quick exits
    ),

    MarketRegime.STRONG_DOWNTREND: RegimeParameters(
        position_size_multiplier=0.25,      # Very defensive
        stop_atr_multiplier=4.0,            # Wide stops for volatility
        stop_widen_factor=1.25,
        take_profit_atr_multiplier=2.5,     # Quick profits
        trail_stop_enabled=True,
        trail_stop_atr=3.0,
        min_adx_for_entry=30.0,             # Only strong signals
        require_trend_confirmation=True,
        max_daily_trades=2,
        max_drawdown_exit_pct=2.0,
        prefer_counter_trend=False,
        allow_new_positions=False,          # Cash is king in bear markets
        aggressive_trailing=True,
    ),

    # =========================================================================
    # SPECIAL CONDITIONS
    # =========================================================================
    MarketRegime.BREAKOUT: RegimeParameters(
        position_size_multiplier=1.0,
        stop_atr_multiplier=3.0,
        stop_widen_factor=1.0,
        take_profit_atr_multiplier=5.0,     # Let breakouts run
        trail_stop_enabled=True,
        trail_stop_atr=2.0,
        min_adx_for_entry=20.0,
        require_trend_confirmation=False,
        max_daily_trades=5,
        max_drawdown_exit_pct=5.0,
        prefer_counter_trend=False,
        allow_new_positions=True,
        aggressive_trailing=False,
    ),

    MarketRegime.BREAKDOWN: RegimeParameters(
        position_size_multiplier=0.5,
        stop_atr_multiplier=4.0,
        stop_widen_factor=1.25,
        take_profit_atr_multiplier=3.0,
        trail_stop_enabled=True,
        trail_stop_atr=3.0,
        min_adx_for_entry=25.0,
        require_trend_confirmation=True,
        max_daily_trades=2,
        max_drawdown_exit_pct=3.0,
        prefer_counter_trend=False,
        allow_new_positions=False,
        aggressive_trailing=True,
    ),

    MarketRegime.CAPITULATION: RegimeParameters(
        position_size_multiplier=0.0,       # NO trading during panic
        stop_atr_multiplier=5.0,
        stop_widen_factor=2.0,
        take_profit_atr_multiplier=2.0,
        trail_stop_enabled=True,
        trail_stop_atr=4.0,
        min_adx_for_entry=50.0,
        require_trend_confirmation=True,
        max_daily_trades=0,
        max_drawdown_exit_pct=1.0,
        prefer_counter_trend=False,
        allow_new_positions=False,
        aggressive_trailing=True,
    ),

    MarketRegime.EUPHORIA: RegimeParameters(
        position_size_multiplier=0.5,       # Defensive near tops
        stop_atr_multiplier=4.0,
        stop_widen_factor=1.0,
        take_profit_atr_multiplier=3.0,     # Take profits
        trail_stop_enabled=True,
        trail_stop_atr=2.0,
        min_adx_for_entry=30.0,
        require_trend_confirmation=True,
        max_daily_trades=2,
        max_drawdown_exit_pct=3.0,
        prefer_counter_trend=True,          # Consider shorts
        allow_new_positions=True,
        aggressive_trailing=True,
    ),

    MarketRegime.UNKNOWN: RegimeParameters(
        position_size_multiplier=0.5,       # Defensive when unsure
        stop_atr_multiplier=3.0,
        stop_widen_factor=1.0,
        take_profit_atr_multiplier=3.0,
        trail_stop_enabled=True,
        trail_stop_atr=2.5,
        min_adx_for_entry=25.0,
        require_trend_confirmation=True,
        max_daily_trades=3,
        max_drawdown_exit_pct=3.0,
        prefer_counter_trend=False,
        allow_new_positions=True,
        aggressive_trailing=False,
    ),
}


@dataclass
class RegimeState:
    """Current regime state with confidence and history"""
    current_regime: MarketRegime
    volatility_regime: MarketRegime
    trend_regime: MarketRegime
    confidence: float                        # 0-1 confidence in detection
    detected_at: datetime
    parameters: RegimeParameters

    # Indicators used for detection
    atr_percent: float                       # ATR as % of price
    adx: float                               # Trend strength
    rsi: float                               # Momentum
    price_vs_sma20: float                    # Price relative to 20 SMA
    price_vs_sma50: float                    # Price relative to 50 SMA
    volume_ratio: float                      # Current volume vs average

    # Recent history
    regime_duration_hours: float = 0         # How long in current regime
    previous_regime: Optional[MarketRegime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_regime": self.current_regime.value,
            "volatility_regime": self.volatility_regime.value,
            "trend_regime": self.trend_regime.value,
            "confidence": round(self.confidence, 2),
            "detected_at": self.detected_at.isoformat(),
            "parameters": self.parameters.to_dict(),
            "indicators": {
                "atr_percent": round(self.atr_percent, 4),
                "adx": round(self.adx, 2),
                "rsi": round(self.rsi, 2),
                "price_vs_sma20": round(self.price_vs_sma20, 4),
                "price_vs_sma50": round(self.price_vs_sma50, 4),
                "volume_ratio": round(self.volume_ratio, 2),
            },
            "regime_duration_hours": round(self.regime_duration_hours, 1),
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
        }


class RegimeDetector:
    """
    Detects current market regime from price data and indicators.

    Uses multiple signals:
    1. ATR-based volatility classification
    2. ADX-based trend strength
    3. Price relative to moving averages
    4. Volume analysis
    5. RSI for momentum/extremes
    """

    # Volatility thresholds (ATR as % of price)
    EXTREME_VOL_THRESHOLD = 5.0    # ATR > 5% of price
    HIGH_VOL_THRESHOLD = 3.0       # ATR > 3%
    LOW_VOL_THRESHOLD = 1.0        # ATR < 1%

    # Trend strength thresholds (ADX)
    STRONG_TREND_ADX = 30
    WEAK_TREND_ADX = 20
    RANGING_ADX = 15

    # RSI extremes
    OVERBOUGHT_RSI = 70
    OVERSOLD_RSI = 30
    EXTREME_OVERBOUGHT = 80
    EXTREME_OVERSOLD = 20

    def __init__(self):
        self._last_regime: Optional[RegimeState] = None
        self._regime_start: Optional[datetime] = None

    def detect(
        self,
        current_price: float,
        atr: float,
        adx: float,
        rsi: float = 50.0,
        sma_20: Optional[float] = None,
        sma_50: Optional[float] = None,
        volume_ratio: float = 1.0,
        recent_high: Optional[float] = None,
        recent_low: Optional[float] = None,
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            current_price: Current price
            atr: Average True Range
            adx: Average Directional Index
            rsi: Relative Strength Index
            sma_20: 20-period Simple Moving Average
            sma_50: 50-period Simple Moving Average
            volume_ratio: Current volume / average volume
            recent_high: Recent N-period high (for breakout detection)
            recent_low: Recent N-period low (for breakdown detection)

        Returns:
            RegimeState with detected regime and adaptive parameters
        """
        now = datetime.now(timezone.utc)

        # Calculate derived metrics
        atr_percent = (atr / current_price * 100) if current_price > 0 else 0

        price_vs_sma20 = 0.0
        if sma_20 and sma_20 > 0:
            price_vs_sma20 = (current_price - sma_20) / sma_20

        price_vs_sma50 = 0.0
        if sma_50 and sma_50 > 0:
            price_vs_sma50 = (current_price - sma_50) / sma_50

        # Step 1: Detect volatility regime
        volatility_regime = self._detect_volatility_regime(atr_percent, volume_ratio)

        # Step 2: Detect trend regime
        trend_regime = self._detect_trend_regime(
            adx, price_vs_sma20, price_vs_sma50, rsi
        )

        # Step 3: Check for special conditions
        special_regime = self._detect_special_conditions(
            atr_percent, rsi, volume_ratio,
            current_price, recent_high, recent_low
        )

        # Step 4: Combine regimes into final classification
        current_regime, confidence = self._combine_regimes(
            volatility_regime, trend_regime, special_regime
        )

        # Step 5: Get parameters for this regime
        parameters = REGIME_PARAMETERS.get(
            current_regime, REGIME_PARAMETERS[MarketRegime.UNKNOWN]
        )

        # Track regime duration
        regime_duration_hours = 0.0
        previous_regime = None

        if self._last_regime:
            previous_regime = self._last_regime.current_regime
            if previous_regime == current_regime and self._regime_start:
                regime_duration_hours = (now - self._regime_start).total_seconds() / 3600
            else:
                self._regime_start = now
        else:
            self._regime_start = now

        state = RegimeState(
            current_regime=current_regime,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            confidence=confidence,
            detected_at=now,
            parameters=parameters,
            atr_percent=atr_percent,
            adx=adx,
            rsi=rsi,
            price_vs_sma20=price_vs_sma20,
            price_vs_sma50=price_vs_sma50,
            volume_ratio=volume_ratio,
            regime_duration_hours=regime_duration_hours,
            previous_regime=previous_regime,
        )

        self._last_regime = state
        return state

    def _detect_volatility_regime(
        self, atr_percent: float, volume_ratio: float
    ) -> MarketRegime:
        """Detect volatility-based regime"""
        if atr_percent >= self.EXTREME_VOL_THRESHOLD:
            return MarketRegime.EXTREME_VOLATILITY
        elif atr_percent >= self.HIGH_VOL_THRESHOLD:
            return MarketRegime.HIGH_VOLATILITY
        elif atr_percent <= self.LOW_VOL_THRESHOLD:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.NORMAL_VOLATILITY

    def _detect_trend_regime(
        self,
        adx: float,
        price_vs_sma20: float,
        price_vs_sma50: float,
        rsi: float
    ) -> MarketRegime:
        """Detect trend-based regime"""
        # Determine trend direction from price vs SMAs
        is_bullish = price_vs_sma20 > 0 and price_vs_sma50 > 0
        is_bearish = price_vs_sma20 < 0 and price_vs_sma50 < 0

        # Classify by trend strength
        if adx >= self.STRONG_TREND_ADX:
            if is_bullish:
                return MarketRegime.STRONG_UPTREND
            elif is_bearish:
                return MarketRegime.STRONG_DOWNTREND
            else:
                # Strong ADX but mixed signals - probably transitioning
                return MarketRegime.RANGING

        elif adx >= self.WEAK_TREND_ADX:
            if is_bullish:
                return MarketRegime.WEAK_UPTREND
            elif is_bearish:
                return MarketRegime.WEAK_DOWNTREND
            else:
                return MarketRegime.RANGING

        else:
            return MarketRegime.RANGING

    def _detect_special_conditions(
        self,
        atr_percent: float,
        rsi: float,
        volume_ratio: float,
        current_price: float,
        recent_high: Optional[float],
        recent_low: Optional[float]
    ) -> Optional[MarketRegime]:
        """Detect special market conditions"""

        # Capitulation: Extreme oversold + high volume + high volatility
        if (rsi <= self.EXTREME_OVERSOLD and
            volume_ratio >= 2.0 and
            atr_percent >= self.HIGH_VOL_THRESHOLD):
            return MarketRegime.CAPITULATION

        # Euphoria: Extreme overbought + high volume
        if (rsi >= self.EXTREME_OVERBOUGHT and
            volume_ratio >= 1.5):
            return MarketRegime.EUPHORIA

        # Breakout: Price at recent high with volume
        if recent_high and current_price >= recent_high * 0.99 and volume_ratio >= 1.3:
            return MarketRegime.BREAKOUT

        # Breakdown: Price at recent low with volume
        if recent_low and current_price <= recent_low * 1.01 and volume_ratio >= 1.3:
            return MarketRegime.BREAKDOWN

        return None

    def _combine_regimes(
        self,
        volatility_regime: MarketRegime,
        trend_regime: MarketRegime,
        special_regime: Optional[MarketRegime]
    ) -> Tuple[MarketRegime, float]:
        """Combine different regime signals into final classification"""

        # Special conditions take precedence
        if special_regime:
            return special_regime, 0.9

        # Extreme volatility overrides everything
        if volatility_regime == MarketRegime.EXTREME_VOLATILITY:
            return MarketRegime.EXTREME_VOLATILITY, 0.95

        # High volatility + strong downtrend = dangerous
        if (volatility_regime == MarketRegime.HIGH_VOLATILITY and
            trend_regime == MarketRegime.STRONG_DOWNTREND):
            return MarketRegime.STRONG_DOWNTREND, 0.85

        # Otherwise, trend regime takes precedence with volatility adjustment
        confidence = 0.75

        # Adjust confidence based on volatility
        if volatility_regime in [MarketRegime.NORMAL_VOLATILITY, MarketRegime.LOW_VOLATILITY]:
            confidence = 0.85

        return trend_regime, confidence

    def get_regime_description(self, regime: MarketRegime) -> str:
        """Get human-readable description of a regime"""
        descriptions = {
            MarketRegime.EXTREME_VOLATILITY: "DANGER: Extreme volatility - reduce exposure",
            MarketRegime.HIGH_VOLATILITY: "Elevated volatility - widen stops, reduce size",
            MarketRegime.NORMAL_VOLATILITY: "Normal conditions - standard parameters",
            MarketRegime.LOW_VOLATILITY: "Low volatility - watch for breakout",
            MarketRegime.STRONG_UPTREND: "Strong uptrend - let winners run",
            MarketRegime.WEAK_UPTREND: "Weak uptrend - be selective",
            MarketRegime.RANGING: "Ranging market - quick profits, tight stops",
            MarketRegime.WEAK_DOWNTREND: "Weak downtrend - defensive mode",
            MarketRegime.STRONG_DOWNTREND: "Strong downtrend - preserve capital",
            MarketRegime.BREAKOUT: "Breakout in progress - ride the momentum",
            MarketRegime.BREAKDOWN: "Breakdown in progress - avoid longs",
            MarketRegime.CAPITULATION: "CAPITULATION - stay out, wait for reversal",
            MarketRegime.EUPHORIA: "EUPHORIA - extreme caution, consider exits",
            MarketRegime.UNKNOWN: "Unknown conditions - defensive stance",
        }
        return descriptions.get(regime, "Unknown regime")


# Convenience function
def detect_regime(
    current_price: float,
    atr: float,
    adx: float,
    rsi: float = 50.0,
    **kwargs
) -> RegimeState:
    """
    Convenience function to detect current market regime.

    Example:
        state = detect_regime(
            current_price=50000,
            atr=1500,
            adx=25,
            rsi=55
        )
        print(f"Regime: {state.current_regime.value}")
        print(f"Position size: {state.parameters.position_size_multiplier}")
    """
    detector = RegimeDetector()
    return detector.detect(current_price, atr, adx, rsi, **kwargs)
