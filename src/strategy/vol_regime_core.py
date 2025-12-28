"""
Vol-Regime Core Strategy

Computes target BTC allocation based on:
1. Regime detection (Bull/Bear/Sideways)
2. Volatility scaling

This module does NOT execute trades - it only computes the desired allocation.
The Portfolio Manager is responsible for combining this with the sleeve
and executing a single consolidated order.

IMPORTANT: DD scaling is NOT applied here. It's applied at the portfolio level
to avoid double-de-risking.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class VolRegimeContext:
    """Complete context for Glass Box logging"""
    timestamp: datetime

    # Price data
    current_price: Decimal
    sma_200: Decimal
    momentum_30: Decimal
    realized_vol_30: Decimal

    # Regime classification
    regime: str  # "bull", "bear", "sideways"
    base_alloc: Decimal  # Before vol scaling
    vol_scalar: Decimal  # Vol adjustment factor

    # Final output
    target_alloc: Decimal  # Final target allocation [0..1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": float(self.current_price),
            "indicators": {
                "sma_200": float(self.sma_200),
                "momentum_30": float(self.momentum_30),
                "realized_vol_30": float(self.realized_vol_30),
            },
            "regime": {
                "classification": self.regime,
                "base_alloc": float(self.base_alloc),
                "vol_scalar": float(self.vol_scalar),
            },
            "target_alloc": float(self.target_alloc),
        }


@dataclass
class VolRegimeResult:
    """Strategy output - allocation target only, no signals"""
    target_alloc: float  # Target BTC allocation [0..1]
    context: VolRegimeContext
    reason: str
    strategy_name: str = "vol_regime_core"
    strategy_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": {
                "name": self.strategy_name,
                "version": self.strategy_version
            },
            "target_alloc": self.target_alloc,
            "reason": self.reason,
            **self.context.to_dict()
        }


class VolRegimeCoreStrategy:
    """
    Vol-Regime Core Strategy - computes target BTC allocation.

    Regime Rules:
    - BULL: Price > SMA200 AND momentum_30 > 0 -> base_alloc = 1.0
    - BEAR: Price < SMA200 AND momentum_30 < -10 -> base_alloc = 0.0
    - SIDEWAYS: Otherwise -> base_alloc = 0.5

    Vol Scaling:
    - vol_scalar = (vol_target / realized_vol_30), clamped to [0.25, 1.5]
    - target_alloc = base_alloc * vol_scalar, clamped to [0..1]

    Parameters:
        vol_target: Target annualized volatility (default: 40%)
        vol_scalar_min: Minimum vol scalar (default: 0.25)
        vol_scalar_max: Maximum vol scalar (default: 1.5)
        bear_momentum_threshold: Momentum threshold for bear regime (default: -10%)
    """

    def __init__(
        self,
        vol_target: float = 40.0,
        vol_scalar_min: float = 0.25,
        vol_scalar_max: float = 1.5,
        bear_momentum_threshold: float = -10.0
    ):
        self.vol_target = Decimal(str(vol_target))
        self.vol_scalar_min = Decimal(str(vol_scalar_min))
        self.vol_scalar_max = Decimal(str(vol_scalar_max))
        self.bear_momentum_threshold = Decimal(str(bear_momentum_threshold))

        self.min_bars = 205  # Need 200 SMA + buffer

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required indicators"""
        df = df.copy()

        # SMA 200
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # 30-day momentum (percentage change)
        df['momentum_30'] = (df['close'] / df['close'].shift(30) - 1) * 100

        # Realized volatility (30-day, annualized)
        returns = df['close'].pct_change()
        df['realized_vol_30'] = returns.rolling(window=30).std() * np.sqrt(252) * 100

        return df

    def _classify_regime(
        self,
        price: Decimal,
        sma_200: Decimal,
        momentum_30: Decimal
    ) -> tuple[str, Decimal]:
        """
        Classify market regime.

        Returns:
            (regime_name, base_allocation)
        """
        if price > sma_200 and momentum_30 > 0:
            return "bull", Decimal("1.0")
        elif price < sma_200 and momentum_30 < self.bear_momentum_threshold:
            return "bear", Decimal("0.0")
        else:
            return "sideways", Decimal("0.5")

    def _calculate_vol_scalar(self, realized_vol: Decimal) -> Decimal:
        """
        Calculate volatility scaling factor.

        vol_scalar = vol_target / realized_vol, clamped to [min, max]
        """
        if realized_vol <= 0:
            return Decimal("1.0")

        vol_scalar = self.vol_target / realized_vol
        vol_scalar = max(self.vol_scalar_min, min(self.vol_scalar_max, vol_scalar))
        return vol_scalar

    def evaluate(
        self,
        df: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> VolRegimeResult:
        """
        Evaluate strategy and return target allocation.

        Args:
            df: DataFrame with OHLCV columns
            timestamp: Evaluation timestamp

        Returns:
            VolRegimeResult with target allocation and context
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
        sma_200 = Decimal(str(current['sma_200']))
        momentum_30 = Decimal(str(current['momentum_30']))
        realized_vol = Decimal(str(current['realized_vol_30']))

        # Classify regime
        regime, base_alloc = self._classify_regime(price, sma_200, momentum_30)

        # Calculate vol scalar
        vol_scalar = self._calculate_vol_scalar(realized_vol)

        # Calculate target allocation (clamped to [0..1])
        target_alloc = base_alloc * vol_scalar
        target_alloc = max(Decimal("0.0"), min(Decimal("1.0"), target_alloc))

        # Build context
        context = VolRegimeContext(
            timestamp=eval_time,
            current_price=price,
            sma_200=sma_200,
            momentum_30=momentum_30,
            realized_vol_30=realized_vol,
            regime=regime,
            base_alloc=base_alloc,
            vol_scalar=vol_scalar,
            target_alloc=target_alloc
        )

        # Build reason string
        reason = (
            f"{regime.upper()} regime (price {'>' if price > sma_200 else '<'} SMA200, "
            f"mom={float(momentum_30):.1f}%), "
            f"vol_scalar={float(vol_scalar):.2f} "
            f"-> target_alloc={float(target_alloc):.1%}"
        )

        return VolRegimeResult(
            target_alloc=float(target_alloc),
            context=context,
            reason=reason
        )

    def _create_error_result(
        self,
        reason: str,
        df: pd.DataFrame,
        timestamp: datetime
    ) -> VolRegimeResult:
        """Create error result with zero allocation"""
        price = Decimal(str(df.iloc[-1]['close'])) if len(df) > 0 else Decimal("0")

        context = VolRegimeContext(
            timestamp=timestamp,
            current_price=price,
            sma_200=Decimal("0"),
            momentum_30=Decimal("0"),
            realized_vol_30=Decimal("0"),
            regime="unknown",
            base_alloc=Decimal("0"),
            vol_scalar=Decimal("1"),
            target_alloc=Decimal("0")
        )

        return VolRegimeResult(
            target_alloc=0.0,
            context=context,
            reason=reason
        )


# Factory function
def create_strategy(
    vol_target: float = 40.0,
    vol_scalar_min: float = 0.25,
    vol_scalar_max: float = 1.5,
    bear_momentum_threshold: float = -10.0
) -> VolRegimeCoreStrategy:
    """Create Vol-Regime Core Strategy instance"""
    return VolRegimeCoreStrategy(
        vol_target=vol_target,
        vol_scalar_min=vol_scalar_min,
        vol_scalar_max=vol_scalar_max,
        bear_momentum_threshold=bear_momentum_threshold
    )


# Default instance
default_strategy = VolRegimeCoreStrategy()
