"""
Liquidity Monitor - Real-time Liquidity Tracking

Tracks bid-ask spreads and volume to detect low-liquidity conditions.
Used by SessionManager to make informed decisions about position sizing.

Key metrics:
- Spread (basis points): Lower is better
- Volume ratio: Current volume / 24h average
- Depth score: Order book depth (if available)
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
import logging

from .schema import LiquidityMetrics, MarketSession, SessionConfig


logger = logging.getLogger(__name__)


class LiquidityMonitor:
    """
    Monitors real-time liquidity for trading symbols.

    Tracks spread and volume metrics to detect when liquidity
    drops below acceptable thresholds.
    """

    def __init__(self, config: Optional[SessionConfig] = None):
        """
        Initialize liquidity monitor.

        Args:
            config: Session configuration with liquidity thresholds.
        """
        self.config = config or SessionConfig()
        self._metrics: Dict[str, LiquidityMetrics] = {}
        self._volume_24h: Dict[str, Decimal] = {}  # Cache 24h volume
        self._last_update: Dict[str, datetime] = {}

    def update(
        self,
        symbol: str,
        bid: Decimal,
        ask: Decimal,
        volume: Decimal,
        volume_24h: Optional[Decimal] = None,
        depth_score: Optional[Decimal] = None,
        session: Optional[MarketSession] = None,
        timestamp: Optional[datetime] = None
    ) -> LiquidityMetrics:
        """
        Update liquidity metrics for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD").
            bid: Current best bid price.
            ask: Current best ask price.
            volume: Current period volume.
            volume_24h: 24-hour volume (will cache if provided).
            depth_score: Order book depth score 0-100 (optional).
            session: Current market session (optional).
            timestamp: Override timestamp (for testing).

        Returns:
            Updated LiquidityMetrics.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Cache 24h volume if provided
        if volume_24h is not None:
            self._volume_24h[symbol] = volume_24h

        # Get cached 24h volume or use current as fallback
        vol_24h = self._volume_24h.get(symbol, volume)

        # Calculate spread in basis points
        mid_price = (bid + ask) / 2
        spread = ask - bid
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else Decimal("0")

        # Calculate volume ratio
        volume_ratio = (volume / vol_24h * 24) if vol_24h > 0 else Decimal("1.0")

        # Determine if in dead zone
        is_dead = session == MarketSession.DEAD_ZONE if session else False

        metrics = LiquidityMetrics(
            symbol=symbol,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            spread_bps=spread_bps,
            volume_24h=vol_24h,
            volume_current=volume,
            volume_ratio=volume_ratio,
            depth_score=depth_score,
            session=session,
            is_dead_zone=is_dead
        )

        self._metrics[symbol] = metrics
        self._last_update[symbol] = timestamp

        return metrics

    def get_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """
        Get current liquidity metrics for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            LiquidityMetrics or None if not tracked.
        """
        return self._metrics.get(symbol)

    def get_all_metrics(self) -> Dict[str, LiquidityMetrics]:
        """Get all tracked liquidity metrics."""
        return self._metrics.copy()

    def is_liquid(
        self,
        symbol: str,
        max_spread_bps: Optional[float] = None,
        min_volume_ratio: Optional[float] = None
    ) -> bool:
        """
        Check if a symbol meets minimum liquidity thresholds.

        Args:
            symbol: Trading pair.
            max_spread_bps: Override max spread threshold.
            min_volume_ratio: Override min volume ratio threshold.

        Returns:
            True if liquid, False if not or not tracked.
        """
        metrics = self._metrics.get(symbol)
        if metrics is None:
            logger.warning(f"No liquidity data for {symbol}")
            return False

        spread_threshold = Decimal(str(max_spread_bps or self.config.max_spread_bps))
        volume_threshold = Decimal(str(min_volume_ratio or self.config.min_liquidity_ratio))

        spread_ok = metrics.spread_bps < spread_threshold
        volume_ok = metrics.volume_ratio > volume_threshold

        if not spread_ok:
            logger.debug(
                f"{symbol} spread too wide: {metrics.spread_bps:.1f} bps > {spread_threshold} bps"
            )

        if not volume_ok:
            logger.debug(
                f"{symbol} volume too low: {metrics.volume_ratio:.2f} < {volume_threshold}"
            )

        return spread_ok and volume_ok

    def is_all_liquid(self, symbols: Optional[List[str]] = None) -> bool:
        """
        Check if all tracked symbols (or specified) are liquid.

        Args:
            symbols: Specific symbols to check. Uses all if not provided.

        Returns:
            True if all liquid, False otherwise.
        """
        check_symbols = symbols or list(self._metrics.keys())

        if not check_symbols:
            return True  # No symbols to check

        return all(self.is_liquid(s) for s in check_symbols)

    def get_liquidity_score(self, symbol: str) -> float:
        """
        Get normalized liquidity score for a symbol (0.0 - 1.0).

        Higher is better:
        - 1.0 = Excellent liquidity
        - 0.5 = Marginal
        - 0.0 = Illiquid

        Args:
            symbol: Trading pair.

        Returns:
            Liquidity score between 0.0 and 1.0.
        """
        metrics = self._metrics.get(symbol)
        if metrics is None:
            return 0.0

        # Score based on spread (0-50 bps maps to 1.0-0.0)
        spread_score = max(0.0, 1.0 - float(metrics.spread_bps) / 50.0)

        # Score based on volume ratio (0.3-2.0 maps to 0.0-1.0)
        vol_ratio = float(metrics.volume_ratio)
        if vol_ratio < 0.3:
            volume_score = 0.0
        elif vol_ratio > 2.0:
            volume_score = 1.0
        else:
            volume_score = (vol_ratio - 0.3) / 1.7

        # Combined score (weighted average)
        return 0.6 * spread_score + 0.4 * volume_score

    def get_aggregate_score(self, symbols: Optional[List[str]] = None) -> float:
        """
        Get aggregate liquidity score across all/specified symbols.

        Args:
            symbols: Specific symbols to check. Uses all if not provided.

        Returns:
            Average liquidity score (0.0 - 1.0).
        """
        check_symbols = symbols or list(self._metrics.keys())

        if not check_symbols:
            return 1.0  # No symbols = assume liquid

        scores = [self.get_liquidity_score(s) for s in check_symbols]
        return sum(scores) / len(scores)

    def get_position_adjustment(self, symbol: str) -> Decimal:
        """
        Get position size adjustment based on liquidity.

        Lower liquidity = smaller position size.

        Args:
            symbol: Trading pair.

        Returns:
            Multiplier (0.0 - 1.0) for position sizing.
        """
        score = self.get_liquidity_score(symbol)

        # Map score to multiplier
        # Score 1.0 -> 1.0 multiplier
        # Score 0.5 -> 0.7 multiplier
        # Score 0.0 -> 0.3 multiplier (minimum, never 0)
        multiplier = 0.3 + (score * 0.7)

        return Decimal(str(round(multiplier, 2)))

    def is_stale(self, symbol: str, max_age_seconds: int = 300) -> bool:
        """
        Check if liquidity data for a symbol is stale.

        Args:
            symbol: Trading pair.
            max_age_seconds: Maximum age before considered stale.

        Returns:
            True if stale or not tracked, False if fresh.
        """
        last_update = self._last_update.get(symbol)
        if last_update is None:
            return True

        age = datetime.utcnow() - last_update
        return age.total_seconds() > max_age_seconds

    def get_stale_symbols(self, max_age_seconds: int = 300) -> List[str]:
        """
        Get list of symbols with stale data.

        Args:
            max_age_seconds: Maximum age before considered stale.

        Returns:
            List of symbol names with stale data.
        """
        return [s for s in self._metrics.keys() if self.is_stale(s, max_age_seconds)]

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached metrics.

        Args:
            symbol: Specific symbol to clear. Clears all if not provided.
        """
        if symbol:
            self._metrics.pop(symbol, None)
            self._volume_24h.pop(symbol, None)
            self._last_update.pop(symbol, None)
        else:
            self._metrics.clear()
            self._volume_24h.clear()
            self._last_update.clear()

    def get_summary(self) -> Dict:
        """
        Get summary of all liquidity data for logging/display.

        Returns:
            Dictionary with liquidity summary.
        """
        symbols = list(self._metrics.keys())

        return {
            "tracked_symbols": symbols,
            "symbol_count": len(symbols),
            "aggregate_score": self.get_aggregate_score(),
            "all_liquid": self.is_all_liquid(),
            "stale_count": len(self.get_stale_symbols()),
            "by_symbol": {
                s: {
                    "spread_bps": float(m.spread_bps),
                    "volume_ratio": float(m.volume_ratio),
                    "is_liquid": m.is_liquid,
                    "score": self.get_liquidity_score(s)
                }
                for s, m in self._metrics.items()
            }
        }
