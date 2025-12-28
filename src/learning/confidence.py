"""
Confidence Scoring System

Scores trading signals based on:
1. Historical performance in similar conditions
2. Lessons from the Reflexion Layer
3. Current market regime
4. Signal strength and quality metrics

Research basis:
- Kelly Criterion for position sizing
- Signal quality filtering (reduce noise trades)
- Ensemble methods for confidence estimation

"Conviction without evidence is the enemy of returns."
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import math

from .regime import MarketRegime, RegimeState, REGIME_PARAMETERS

logger = logging.getLogger(__name__)


class SignalQuality(Enum):
    """Signal quality classification"""
    EXCELLENT = "excellent"    # High confidence, take full size
    GOOD = "good"              # Moderate confidence
    MARGINAL = "marginal"      # Low confidence, reduce size
    POOR = "poor"              # Very low confidence, skip or minimal
    SKIP = "skip"              # Do not trade


@dataclass
class ConfidenceScore:
    """
    Confidence score for a trading signal.

    Combines multiple factors into a final recommendation.
    """
    # Overall scores
    final_score: float              # 0-1, overall confidence
    quality: SignalQuality          # Quality classification
    position_size_factor: float     # 0-2, multiplier for position size

    # Component scores (each 0-1)
    signal_strength: float          # Raw signal strength
    regime_alignment: float         # How well signal fits regime
    historical_success: float       # Past performance in similar conditions
    lesson_adjustment: float        # Adjustment from Reflexion Layer

    # Recommendations
    recommended_action: str         # "execute", "reduce_size", "skip", "wait"
    warnings: List[str]             # Warning messages
    supporting_factors: List[str]   # Positive factors
    risk_factors: List[str]         # Negative factors

    # Context
    regime: MarketRegime
    scored_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_score": round(self.final_score, 3),
            "quality": self.quality.value,
            "position_size_factor": round(self.position_size_factor, 2),
            "components": {
                "signal_strength": round(self.signal_strength, 3),
                "regime_alignment": round(self.regime_alignment, 3),
                "historical_success": round(self.historical_success, 3),
                "lesson_adjustment": round(self.lesson_adjustment, 3),
            },
            "recommended_action": self.recommended_action,
            "warnings": self.warnings,
            "supporting_factors": self.supporting_factors,
            "risk_factors": self.risk_factors,
            "regime": self.regime.value,
            "scored_at": self.scored_at.isoformat(),
        }


@dataclass
class HistoricalPerformance:
    """Historical performance for similar signals"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.5
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 1.0
    expectancy: float = 0.0          # Expected value per trade

    def calculate_metrics(self):
        """Calculate derived metrics"""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        else:
            self.win_rate = 0.5  # Assume neutral

        # Profit factor
        if self.avg_loss_pct != 0:
            self.profit_factor = abs(self.avg_win_pct / self.avg_loss_pct) if self.avg_loss_pct else 1.0
        else:
            self.profit_factor = 1.0

        # Expectancy
        self.expectancy = (self.win_rate * self.avg_win_pct) + ((1 - self.win_rate) * self.avg_loss_pct)


class ConfidenceScorer:
    """
    Scores trading signals for confidence level.

    Uses a multi-factor model:
    1. Signal Strength (40%): Raw signal quality
    2. Regime Alignment (25%): Does signal match regime?
    3. Historical Success (20%): Past performance
    4. Lesson Adjustment (15%): Reflexion layer insights
    """

    # Weights for final score
    WEIGHT_SIGNAL = 0.40
    WEIGHT_REGIME = 0.25
    WEIGHT_HISTORY = 0.20
    WEIGHT_LESSONS = 0.15

    # Quality thresholds
    EXCELLENT_THRESHOLD = 0.80
    GOOD_THRESHOLD = 0.60
    MARGINAL_THRESHOLD = 0.40
    POOR_THRESHOLD = 0.25

    # Position sizing based on Kelly Criterion (capped)
    MAX_KELLY_FRACTION = 0.25  # Never risk more than 25% Kelly

    def __init__(
        self,
        db_path: Optional[str] = None,
        min_trades_for_history: int = 5
    ):
        """
        Initialize the confidence scorer.

        Args:
            db_path: Path to Truth Engine database for historical lookup
            min_trades_for_history: Minimum trades needed for historical stats
        """
        self.db_path = db_path
        self.min_trades_for_history = min_trades_for_history

    def score_signal(
        self,
        signal_type: str,                    # "long", "short", "exit"
        signal_values: Dict[str, Any],       # Raw signal data
        regime_state: RegimeState,           # Current market regime
        lessons: Optional[Dict[str, Any]] = None,  # From ReflexionEngine
        historical_performance: Optional[HistoricalPerformance] = None,
    ) -> ConfidenceScore:
        """
        Score a trading signal for confidence.

        Args:
            signal_type: Type of signal (long/short/exit)
            signal_values: Raw signal data from strategy
            regime_state: Current market regime state
            lessons: Lessons from Reflexion Layer
            historical_performance: Optional pre-computed historical stats

        Returns:
            ConfidenceScore with recommendations
        """
        warnings = []
        supporting = []
        risk_factors = []

        # Component 1: Signal Strength
        signal_strength = self._score_signal_strength(signal_values, signal_type)
        if signal_strength >= 0.7:
            supporting.append(f"Strong signal ({signal_strength:.0%})")
        elif signal_strength < 0.4:
            risk_factors.append(f"Weak signal ({signal_strength:.0%})")

        # Component 2: Regime Alignment
        regime_alignment = self._score_regime_alignment(
            signal_type, regime_state, signal_values
        )
        if regime_alignment >= 0.7:
            supporting.append(f"Regime supports {signal_type}")
        elif regime_alignment < 0.4:
            warnings.append(f"Signal conflicts with regime ({regime_state.current_regime.value})")

        # Component 3: Historical Success
        if historical_performance is None:
            historical_performance = HistoricalPerformance()
            historical_success = 0.5  # Neutral if no history
        else:
            historical_success = self._score_historical(historical_performance)
            if historical_success >= 0.6:
                supporting.append(f"Historical win rate: {historical_performance.win_rate:.0%}")
            elif historical_success < 0.4:
                risk_factors.append(f"Poor historical performance ({historical_performance.win_rate:.0%})")

        # Component 4: Lesson Adjustment
        lesson_adjustment = 1.0  # Neutral
        if lessons:
            lesson_adjustment = self._apply_lessons(lessons, signal_type)
            if lesson_adjustment < 0.8:
                warnings.extend(lessons.get("warnings", []))
            if lessons.get("cautionary_lessons"):
                risk_factors.append(f"{len(lessons['cautionary_lessons'])} cautionary lessons apply")
            if lessons.get("supporting_lessons"):
                supporting.append(f"{len(lessons['supporting_lessons'])} supporting lessons apply")

        # Calculate final score
        final_score = (
            signal_strength * self.WEIGHT_SIGNAL +
            regime_alignment * self.WEIGHT_REGIME +
            historical_success * self.WEIGHT_HISTORY +
            lesson_adjustment * self.WEIGHT_LESSONS
        )

        # Apply regime multiplier
        regime_params = regime_state.parameters
        final_score *= regime_params.position_size_multiplier

        # Clamp to 0-1
        final_score = max(0.0, min(1.0, final_score))

        # Determine quality
        quality = self._classify_quality(final_score)

        # Calculate position size factor
        position_size_factor = self._calculate_position_size(
            final_score, historical_performance, regime_params
        )

        # Determine recommended action
        recommended_action = self._get_recommendation(
            quality, regime_state, signal_type, warnings
        )

        # Add regime-specific warnings
        if not regime_params.allow_new_positions and signal_type in ["long", "short"]:
            warnings.append(f"New positions blocked in {regime_state.current_regime.value}")
            recommended_action = "skip"

        return ConfidenceScore(
            final_score=final_score,
            quality=quality,
            position_size_factor=position_size_factor,
            signal_strength=signal_strength,
            regime_alignment=regime_alignment,
            historical_success=historical_success,
            lesson_adjustment=lesson_adjustment,
            recommended_action=recommended_action,
            warnings=warnings,
            supporting_factors=supporting,
            risk_factors=risk_factors,
            regime=regime_state.current_regime,
            scored_at=datetime.now(timezone.utc),
        )

    def _score_signal_strength(
        self,
        signal_values: Dict[str, Any],
        signal_type: str
    ) -> float:
        """Score raw signal strength from 0-1"""
        score = 0.5  # Start neutral

        # ADX contribution (trend strength)
        adx = signal_values.get("adx", 0)
        if adx >= 30:
            score += 0.2
        elif adx >= 20:
            score += 0.1
        elif adx < 15:
            score -= 0.1

        # Check for crossover confirmation
        if signal_values.get("crossover_confirmed"):
            score += 0.15

        # Check price vs channels
        current_price = signal_values.get("current_price", 0)
        entry_channel = signal_values.get("entry_channel", 0)

        if signal_type == "long" and current_price and entry_channel:
            if current_price >= entry_channel:
                score += 0.1  # At breakout level

        # ATR-based volatility check
        atr = signal_values.get("atr", 0)
        if atr and current_price:
            atr_pct = (atr / current_price) * 100
            if 1.0 <= atr_pct <= 3.0:  # Sweet spot
                score += 0.1
            elif atr_pct > 5.0:
                score -= 0.15  # Too volatile

        # RSI contribution
        rsi = signal_values.get("rsi", 50)
        if signal_type == "long":
            if 30 <= rsi <= 50:  # Oversold but not extreme
                score += 0.1
            elif rsi > 70:
                score -= 0.1  # Overbought
        elif signal_type == "short":
            if 50 <= rsi <= 70:
                score += 0.1
            elif rsi < 30:
                score -= 0.1

        return max(0.0, min(1.0, score))

    def _score_regime_alignment(
        self,
        signal_type: str,
        regime_state: RegimeState,
        signal_values: Dict[str, Any]
    ) -> float:
        """Score how well signal aligns with current regime"""
        regime = regime_state.current_regime
        score = 0.5  # Neutral

        # Trend alignment
        if signal_type == "long":
            if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND, MarketRegime.BREAKOUT]:
                score += 0.3
            elif regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.BREAKDOWN, MarketRegime.CAPITULATION]:
                score -= 0.4
            elif regime == MarketRegime.RANGING:
                score -= 0.1  # Slightly negative for trend signals in ranges

        elif signal_type == "short":
            if regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND, MarketRegime.BREAKDOWN]:
                score += 0.3
            elif regime in [MarketRegime.STRONG_UPTREND, MarketRegime.BREAKOUT, MarketRegime.EUPHORIA]:
                score -= 0.3
            elif regime == MarketRegime.RANGING:
                score -= 0.1

        # Volatility alignment
        vol_regime = regime_state.volatility_regime
        if vol_regime == MarketRegime.EXTREME_VOLATILITY:
            score -= 0.2  # Reduce confidence in extreme vol
        elif vol_regime == MarketRegime.HIGH_VOLATILITY:
            score -= 0.1
        elif vol_regime == MarketRegime.LOW_VOLATILITY:
            score += 0.1  # Breakouts from low vol are often good

        # Regime confidence affects our confidence
        score *= regime_state.confidence

        return max(0.0, min(1.0, score))

    def _score_historical(self, perf: HistoricalPerformance) -> float:
        """Score based on historical performance"""
        if perf.total_trades < self.min_trades_for_history:
            return 0.5  # Not enough data

        score = 0.5

        # Win rate contribution
        if perf.win_rate >= 0.6:
            score += 0.2
        elif perf.win_rate >= 0.5:
            score += 0.1
        elif perf.win_rate < 0.4:
            score -= 0.2

        # Profit factor contribution
        if perf.profit_factor >= 2.0:
            score += 0.2
        elif perf.profit_factor >= 1.5:
            score += 0.1
        elif perf.profit_factor < 1.0:
            score -= 0.2

        # Expectancy contribution
        if perf.expectancy > 0.02:  # 2% expected
            score += 0.1
        elif perf.expectancy < -0.01:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _apply_lessons(self, lessons: Dict[str, Any], signal_type: str) -> float:
        """Apply lessons from Reflexion Layer"""
        # Start with the pre-computed adjustment
        base_adjustment = lessons.get("adjusted_confidence", 1.0)

        # Factor in size adjustment
        size_adj = lessons.get("size_adjustment", 1.0)

        # Combine adjustments
        combined = (base_adjustment + size_adj) / 2

        return max(0.3, min(1.5, combined))

    def _classify_quality(self, score: float) -> SignalQuality:
        """Classify score into quality tier"""
        if score >= self.EXCELLENT_THRESHOLD:
            return SignalQuality.EXCELLENT
        elif score >= self.GOOD_THRESHOLD:
            return SignalQuality.GOOD
        elif score >= self.MARGINAL_THRESHOLD:
            return SignalQuality.MARGINAL
        elif score >= self.POOR_THRESHOLD:
            return SignalQuality.POOR
        else:
            return SignalQuality.SKIP

    def _calculate_position_size(
        self,
        score: float,
        historical: HistoricalPerformance,
        regime_params
    ) -> float:
        """Calculate position size factor using modified Kelly"""

        # Base position from score
        base_size = score

        # Apply Kelly Criterion if we have history
        if historical and historical.total_trades >= self.min_trades_for_history:
            # Kelly = (bp - q) / b
            # where b = odds, p = win rate, q = 1-p
            if historical.avg_loss_pct != 0:
                b = abs(historical.avg_win_pct / historical.avg_loss_pct)
            else:
                b = 1.0

            p = historical.win_rate
            q = 1 - p

            kelly = (b * p - q) / b if b > 0 else 0

            # Cap Kelly fraction
            kelly = max(0, min(self.MAX_KELLY_FRACTION, kelly))

            # Blend Kelly with score-based sizing
            base_size = (base_size + kelly * 2) / 2

        # Apply regime multiplier
        base_size *= regime_params.position_size_multiplier

        # Apply quality floor
        quality = self._classify_quality(score)
        if quality == SignalQuality.SKIP:
            return 0.0
        elif quality == SignalQuality.POOR:
            base_size = min(base_size, 0.25)
        elif quality == SignalQuality.MARGINAL:
            base_size = min(base_size, 0.5)

        return max(0.0, min(2.0, base_size))

    def _get_recommendation(
        self,
        quality: SignalQuality,
        regime_state: RegimeState,
        signal_type: str,
        warnings: List[str]
    ) -> str:
        """Get recommended action based on analysis"""

        if quality == SignalQuality.SKIP:
            return "skip"

        if quality == SignalQuality.POOR:
            return "skip"

        if len(warnings) >= 3:
            return "wait"

        if quality == SignalQuality.MARGINAL:
            return "reduce_size"

        if quality in [SignalQuality.GOOD, SignalQuality.EXCELLENT]:
            return "execute"

        return "wait"


# Convenience function
def score_signal(
    signal_type: str,
    signal_values: Dict[str, Any],
    regime_state: RegimeState,
    **kwargs
) -> ConfidenceScore:
    """
    Convenience function to score a trading signal.

    Example:
        score = score_signal(
            signal_type="long",
            signal_values={"adx": 28, "atr": 1500, "current_price": 50000},
            regime_state=regime_state
        )
        print(f"Confidence: {score.final_score:.0%}")
        print(f"Action: {score.recommended_action}")
    """
    scorer = ConfidenceScorer()
    return scorer.score_signal(signal_type, signal_values, regime_state, **kwargs)
