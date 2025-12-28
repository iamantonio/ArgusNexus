"""
Multi-Timeframe Engine - Simultaneous Strategy Evaluation

Runs multiple timeframes (1h, 4h, 1d) on the same symbol simultaneously
and aggregates signals using conflict resolution rules.

Key Components:
- TimeframeUnit: Single timeframe evaluation unit
- SignalAggregator: Resolves conflicts between timeframe signals
- MultiTimeframeEngine: Coordinates multiple timeframes for one symbol

Signal Conflict Resolution:
- Aligned signals (both LONG or both SHORT) = Execute with high confidence
- One active, one HOLD = Execute the active signal
- Conflicting signals (LONG vs SHORT) = HOLD until aligned
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import pandas as pd
import uuid


logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types from strategy evaluation."""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    CLOSE = "close"


class ConflictResolution(Enum):
    """How to resolve conflicting signals between timeframes."""
    HIGHER_TIMEFRAME_WINS = "higher_timeframe_wins"  # 4h overrides 1h
    REQUIRE_ALIGNMENT = "require_alignment"          # Both must agree
    WEIGHTED_AVERAGE = "weighted_average"            # Use weights


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe evaluation."""
    timeframe: str                       # "1h", "4h", "1d"
    signal: SignalType
    confidence: float                    # 0.0 - 1.0
    timestamp: datetime
    signal_values: Dict[str, Any]        # Full strategy output
    weight: float = 1.0                  # Contribution weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "timeframe": self.timeframe,
            "signal": self.signal.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "weight": self.weight,
            "signal_values": self.signal_values
        }


@dataclass
class AggregatedSignal:
    """Result of multi-timeframe signal aggregation."""
    signal: SignalType
    confidence: float
    is_aligned: bool                     # All timeframes agree
    contributing_timeframes: List[str]
    conflict_resolution_used: Optional[str] = None
    individual_signals: Dict[str, TimeframeSignal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "is_aligned": self.is_aligned,
            "contributing_timeframes": self.contributing_timeframes,
            "conflict_resolution_used": self.conflict_resolution_used,
            "individual_signals": {
                tf: sig.to_dict()
                for tf, sig in self.individual_signals.items()
            }
        }


class SignalAggregator:
    """
    Aggregates signals from multiple timeframes into a single action.

    Conflict Resolution Matrix:
    | 1h Signal | 4h Signal | Result   | Reason              |
    |-----------|-----------|----------|---------------------|
    | LONG      | LONG      | LONG     | Aligned - strong    |
    | LONG      | HOLD      | LONG     | 1h leading          |
    | LONG      | SHORT     | HOLD     | Conflict - wait     |
    | SHORT     | SHORT     | SHORT    | Aligned - strong    |
    | SHORT     | HOLD      | SHORT    | 1h leading          |
    | SHORT     | LONG      | HOLD     | Conflict - wait     |
    | HOLD      | LONG      | LONG     | 4h leading          |
    | HOLD      | SHORT     | SHORT    | 4h leading          |
    | HOLD      | HOLD      | HOLD     | No signal           |
    """

    # Conflict resolution matrix
    # (lower_tf_signal, higher_tf_signal) -> resolved_signal
    RESOLUTION_MATRIX = {
        # Aligned signals - strong
        (SignalType.LONG, SignalType.LONG): SignalType.LONG,
        (SignalType.SHORT, SignalType.SHORT): SignalType.SHORT,
        (SignalType.HOLD, SignalType.HOLD): SignalType.HOLD,
        (SignalType.CLOSE, SignalType.CLOSE): SignalType.CLOSE,

        # One active, one hold - use active
        (SignalType.LONG, SignalType.HOLD): SignalType.LONG,
        (SignalType.SHORT, SignalType.HOLD): SignalType.SHORT,
        (SignalType.HOLD, SignalType.LONG): SignalType.LONG,
        (SignalType.HOLD, SignalType.SHORT): SignalType.SHORT,

        # Conflicting signals - wait
        (SignalType.LONG, SignalType.SHORT): SignalType.HOLD,
        (SignalType.SHORT, SignalType.LONG): SignalType.HOLD,

        # Close signals
        (SignalType.CLOSE, SignalType.HOLD): SignalType.CLOSE,
        (SignalType.HOLD, SignalType.CLOSE): SignalType.CLOSE,
        (SignalType.CLOSE, SignalType.LONG): SignalType.CLOSE,  # Close takes priority
        (SignalType.CLOSE, SignalType.SHORT): SignalType.CLOSE,
        (SignalType.LONG, SignalType.CLOSE): SignalType.CLOSE,
        (SignalType.SHORT, SignalType.CLOSE): SignalType.CLOSE,
    }

    # Timeframe hierarchy (lower index = faster)
    TIMEFRAME_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    def __init__(
        self,
        resolution_mode: ConflictResolution = ConflictResolution.REQUIRE_ALIGNMENT
    ):
        """
        Initialize signal aggregator.

        Args:
            resolution_mode: How to resolve conflicting signals.
        """
        self.resolution_mode = resolution_mode

    def aggregate(
        self,
        signals: Dict[str, TimeframeSignal]
    ) -> AggregatedSignal:
        """
        Aggregate multiple timeframe signals into single action.

        Args:
            signals: Dict of timeframe -> TimeframeSignal

        Returns:
            AggregatedSignal with resolved action
        """
        if not signals:
            return AggregatedSignal(
                signal=SignalType.HOLD,
                confidence=0.0,
                is_aligned=True,
                contributing_timeframes=[],
                conflict_resolution_used=None,
                individual_signals={}
            )

        # Sort timeframes by hierarchy
        sorted_tfs = sorted(
            signals.keys(),
            key=lambda tf: self._get_timeframe_rank(tf)
        )

        # Check if all signals are aligned
        signal_types = {s.signal for s in signals.values()}
        is_aligned = len(signal_types) == 1

        if is_aligned:
            # All agree - use the signal with weighted confidence
            final_signal = list(signal_types)[0]
            weighted_conf = self._weighted_confidence(signals)

            return AggregatedSignal(
                signal=final_signal,
                confidence=weighted_conf,
                is_aligned=True,
                contributing_timeframes=sorted_tfs,
                conflict_resolution_used=None,
                individual_signals=signals
            )

        # Not aligned - apply resolution mode
        if self.resolution_mode == ConflictResolution.HIGHER_TIMEFRAME_WINS:
            return self._resolve_higher_wins(signals, sorted_tfs)
        elif self.resolution_mode == ConflictResolution.WEIGHTED_AVERAGE:
            return self._resolve_weighted(signals, sorted_tfs)
        else:  # REQUIRE_ALIGNMENT
            return self._resolve_require_alignment(signals, sorted_tfs)

    def _resolve_higher_wins(
        self,
        signals: Dict[str, TimeframeSignal],
        sorted_tfs: List[str]
    ) -> AggregatedSignal:
        """Higher timeframe signal takes priority."""
        # Get highest timeframe signal (last in sorted list)
        highest_tf = sorted_tfs[-1]
        highest_signal = signals[highest_tf]

        return AggregatedSignal(
            signal=highest_signal.signal,
            confidence=highest_signal.confidence * 0.8,  # Reduce for conflict
            is_aligned=False,
            contributing_timeframes=[highest_tf],
            conflict_resolution_used="higher_timeframe_wins",
            individual_signals=signals
        )

    def _resolve_weighted(
        self,
        signals: Dict[str, TimeframeSignal],
        sorted_tfs: List[str]
    ) -> AggregatedSignal:
        """Use weighted voting to determine signal."""
        # Tally weighted votes
        votes: Dict[SignalType, float] = {}
        for tf in sorted_tfs:
            sig = signals[tf]
            votes[sig.signal] = votes.get(sig.signal, 0) + sig.weight * sig.confidence

        # Winner takes all
        if not votes:
            final_signal = SignalType.HOLD
        else:
            final_signal = max(votes.keys(), key=lambda s: votes[s])

        # Confidence based on vote share
        total_weight = sum(votes.values())
        confidence = votes.get(final_signal, 0) / total_weight if total_weight > 0 else 0.0

        return AggregatedSignal(
            signal=final_signal,
            confidence=confidence,
            is_aligned=False,
            contributing_timeframes=sorted_tfs,
            conflict_resolution_used="weighted_average",
            individual_signals=signals
        )

    def _resolve_require_alignment(
        self,
        signals: Dict[str, TimeframeSignal],
        sorted_tfs: List[str]
    ) -> AggregatedSignal:
        """
        Require alignment - HOLD if any conflict.

        Uses resolution matrix for pair-wise resolution.
        """
        if len(sorted_tfs) < 2:
            sig = signals[sorted_tfs[0]]
            return AggregatedSignal(
                signal=sig.signal,
                confidence=sig.confidence,
                is_aligned=True,
                contributing_timeframes=sorted_tfs,
                individual_signals=signals
            )

        # Pair-wise resolution from lowest to highest
        current_signal = signals[sorted_tfs[0]].signal

        for i in range(1, len(sorted_tfs)):
            higher_tf = sorted_tfs[i]
            higher_signal = signals[higher_tf].signal

            # Look up in resolution matrix
            key = (current_signal, higher_signal)
            if key in self.RESOLUTION_MATRIX:
                current_signal = self.RESOLUTION_MATRIX[key]
            else:
                # Unknown combination - default to HOLD
                logger.warning(f"Unknown signal combination: {key}, defaulting to HOLD")
                current_signal = SignalType.HOLD

        # Calculate confidence (lower if conflict occurred)
        avg_conf = self._weighted_confidence(signals)
        has_conflict = any(
            signals[tf].signal != current_signal
            for tf in sorted_tfs
        )
        final_conf = avg_conf * (0.7 if has_conflict else 1.0)

        return AggregatedSignal(
            signal=current_signal,
            confidence=final_conf,
            is_aligned=not has_conflict,
            contributing_timeframes=sorted_tfs,
            conflict_resolution_used="require_alignment" if has_conflict else None,
            individual_signals=signals
        )

    def _weighted_confidence(self, signals: Dict[str, TimeframeSignal]) -> float:
        """Calculate weighted average confidence."""
        total_weight = sum(s.weight for s in signals.values())
        if total_weight == 0:
            return 0.0

        return sum(s.confidence * s.weight for s in signals.values()) / total_weight

    def _get_timeframe_rank(self, tf: str) -> int:
        """Get numeric rank for timeframe (lower = faster)."""
        try:
            return self.TIMEFRAME_ORDER.index(tf)
        except ValueError:
            # Unknown timeframe - put at end
            return len(self.TIMEFRAME_ORDER)


@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe."""
    interval: str                        # "1h", "4h"
    weight: float                        # Contribution weight (0.0 - 1.0)
    strategy: str                        # Strategy name
    enabled: bool = True


@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe engine."""
    enabled: bool = True
    timeframes: List[TimeframeConfig] = field(default_factory=list)
    conflict_resolution: ConflictResolution = ConflictResolution.REQUIRE_ALIGNMENT

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiTimeframeConfig":
        """Create from config dictionary."""
        timeframes = [
            TimeframeConfig(
                interval=tf.get("interval", "4h"),
                weight=tf.get("weight", 1.0),
                strategy=tf.get("strategy", "donchian"),
                enabled=tf.get("enabled", True)
            )
            for tf in data.get("timeframes", [])
        ]

        resolution_str = data.get("conflict_resolution", "require_alignment")
        resolution = ConflictResolution(resolution_str)

        return cls(
            enabled=data.get("enabled", True),
            timeframes=timeframes,
            conflict_resolution=resolution
        )


class TimeframeUnit:
    """
    Single timeframe evaluation unit.

    Holds state for one timeframe of one symbol.
    """

    def __init__(
        self,
        symbol: str,
        interval: str,
        weight: float,
        strategy_evaluator: Callable[[pd.DataFrame], Dict[str, Any]]
    ):
        """
        Initialize timeframe unit.

        Args:
            symbol: Trading pair (e.g., "BTC-USD").
            interval: Timeframe interval (e.g., "1h", "4h").
            weight: Contribution weight for aggregation.
            strategy_evaluator: Function that evaluates strategy and returns signal dict.
        """
        self.symbol = symbol
        self.interval = interval
        self.weight = weight
        self.strategy_evaluator = strategy_evaluator

        self.last_signal: Optional[TimeframeSignal] = None
        self.last_eval_time: Optional[datetime] = None
        self.eval_count: int = 0

    def evaluate(self, data: pd.DataFrame) -> TimeframeSignal:
        """
        Evaluate strategy on data and return signal.

        Args:
            data: OHLCV DataFrame for this timeframe.

        Returns:
            TimeframeSignal with evaluation result.
        """
        now = datetime.utcnow()

        # Call strategy evaluator
        result = self.strategy_evaluator(data)

        # Extract signal type
        signal_str = result.get("signal", "hold").lower()
        try:
            signal_type = SignalType(signal_str)
        except ValueError:
            logger.warning(f"Unknown signal type '{signal_str}', using HOLD")
            signal_type = SignalType.HOLD

        # Extract confidence
        confidence = result.get("confidence", 0.5)

        signal = TimeframeSignal(
            timeframe=self.interval,
            signal=signal_type,
            confidence=confidence,
            timestamp=now,
            signal_values=result,
            weight=self.weight
        )

        self.last_signal = signal
        self.last_eval_time = now
        self.eval_count += 1

        logger.debug(
            f"{self.symbol} {self.interval}: {signal_type.value} "
            f"(confidence: {confidence:.2f})"
        )

        return signal


class MultiTimeframeEngine:
    """
    Coordinates multiple timeframes for one symbol.

    Runs strategy on each timeframe and aggregates signals.
    """

    def __init__(
        self,
        symbol: str,
        units: List[TimeframeUnit],
        aggregator: Optional[SignalAggregator] = None
    ):
        """
        Initialize multi-timeframe engine.

        Args:
            symbol: Trading pair.
            units: List of TimeframeUnit instances.
            aggregator: Signal aggregator (uses default if not provided).
        """
        self.symbol = symbol
        self.units = {u.interval: u for u in units}
        self.aggregator = aggregator or SignalAggregator()

        self.last_aggregated: Optional[AggregatedSignal] = None
        self.eval_count: int = 0

    def add_unit(self, unit: TimeframeUnit) -> None:
        """Add a timeframe unit."""
        self.units[unit.interval] = unit

    def remove_unit(self, interval: str) -> None:
        """Remove a timeframe unit."""
        self.units.pop(interval, None)

    def evaluate(
        self,
        data_by_timeframe: Dict[str, pd.DataFrame]
    ) -> AggregatedSignal:
        """
        Evaluate all timeframes and aggregate signals.

        Args:
            data_by_timeframe: Dict of interval -> DataFrame

        Returns:
            AggregatedSignal with final action.
        """
        signals: Dict[str, TimeframeSignal] = {}

        for interval, unit in self.units.items():
            if interval not in data_by_timeframe:
                logger.warning(f"No data for {self.symbol} {interval}, skipping")
                continue

            data = data_by_timeframe[interval]
            if data.empty:
                logger.warning(f"Empty data for {self.symbol} {interval}, skipping")
                continue

            try:
                signal = unit.evaluate(data)
                signals[interval] = signal
            except Exception as e:
                logger.error(f"Error evaluating {self.symbol} {interval}: {e}")
                continue

        # Aggregate signals
        aggregated = self.aggregator.aggregate(signals)
        self.last_aggregated = aggregated
        self.eval_count += 1

        logger.info(
            f"{self.symbol} MTF: {aggregated.signal.value} "
            f"(aligned: {aggregated.is_aligned}, conf: {aggregated.confidence:.2f})"
        )

        return aggregated

    def get_status(self) -> Dict[str, Any]:
        """Get current status for logging/display."""
        return {
            "symbol": self.symbol,
            "timeframes": list(self.units.keys()),
            "eval_count": self.eval_count,
            "last_signal": self.last_aggregated.to_dict() if self.last_aggregated else None,
            "unit_status": {
                interval: {
                    "last_eval": unit.last_eval_time.isoformat() if unit.last_eval_time else None,
                    "last_signal": unit.last_signal.signal.value if unit.last_signal else None,
                    "eval_count": unit.eval_count
                }
                for interval, unit in self.units.items()
            }
        }


def create_mtf_signal_id() -> str:
    """Generate unique ID for multi-timeframe signal."""
    return str(uuid.uuid4())
