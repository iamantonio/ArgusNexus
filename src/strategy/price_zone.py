"""
Price Zone Filter - Distribution Zone Detection

Prevents entries in distribution zones where win rates collapse.
Based on backtest analysis showing:
- Below 60% of ATH: 35% win rate (accumulation zone)
- 60-90% of ATH: 22% win rate (caution zone)
- Above 90% of ATH: 8% win rate (distribution zone - NO TRADE)

For BTC specifically:
- Below $60k: accumulation (35% win rate)
- $60k-$90k: caution (22% win rate)
- Above $90k: distribution (8% win rate) → C-grade override

Usage:
    zone_filter = PriceZoneFilter()
    zone = zone_filter.get_zone("BTC-USD", current_price=95000)
    if zone.is_distribution:
        grade = "C"  # Override to no-trade
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class PriceZone:
    """Price zone classification result."""
    symbol: str
    current_price: float
    ath_price: Optional[float]
    ath_ratio: Optional[float]  # current_price / ath_price

    zone: str  # "accumulation", "caution", "distribution", "unknown"
    is_distribution: bool  # True = auto-downgrade to C
    is_caution: bool  # True = consider tightening filters

    reason: str
    btc_context: Optional[str] = None  # BTC market regime context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "ath_price": self.ath_price,
            "ath_ratio": self.ath_ratio,
            "zone": self.zone,
            "is_distribution": self.is_distribution,
            "is_caution": self.is_caution,
            "reason": self.reason,
            "btc_context": self.btc_context,
        }


class PriceZoneFilter:
    """
    Filters trades based on price zone relative to ATH.

    Zone Thresholds (from backtest analysis):
    - Accumulation: < 60% of ATH (best win rate)
    - Caution: 60-90% of ATH (reduced win rate)
    - Distribution: > 90% of ATH (very low win rate - NO TRADE)

    BTC Absolute Thresholds (from 2024-2025 data):
    - Below $60k: accumulation zone
    - $60k-$90k: caution zone
    - Above $90k: distribution zone
    """

    # Zone thresholds (as ratio of ATH)
    ACCUMULATION_MAX = 0.60  # Below 60% of ATH = accumulation
    CAUTION_MAX = 0.90       # 60-90% of ATH = caution
    # Above 90% = distribution

    # BTC absolute thresholds (fallback when ATH not available)
    BTC_ACCUMULATION_MAX = 60000
    BTC_CAUTION_MAX = 90000

    # Known ATH prices (fallback cache, updated periodically)
    # These are approximate and should be refreshed via API
    KNOWN_ATH = {
        "BTC-USD": 108000,   # ~Dec 2024 high
        "ETH-USD": 4800,     # Nov 2021
        "XRP-USD": 3.40,     # Jan 2018
        "SOL-USD": 260,      # Nov 2021
        "DOGE-USD": 0.74,    # May 2021
        "ADA-USD": 3.10,     # Sep 2021
        "BCH-USD": 3786,     # Dec 2017 (Tony's data point)
        "LINK-USD": 53,      # May 2021
    }

    def __init__(self):
        self.ath_cache: Dict[str, float] = self.KNOWN_ATH.copy()
        self.last_refresh: Optional[datetime] = None

    def get_zone(
        self,
        symbol: str,
        current_price: float,
        btc_price: Optional[float] = None
    ) -> PriceZone:
        """
        Determine price zone for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            current_price: Current price
            btc_price: Optional BTC price for market regime context

        Returns:
            PriceZone with zone classification and trade recommendation
        """
        ath_price = self.ath_cache.get(symbol)

        # Calculate ATH ratio if we have ATH data
        ath_ratio = None
        if ath_price and ath_price > 0:
            ath_ratio = current_price / ath_price

        # Special handling for BTC with absolute thresholds
        if "BTC" in symbol:
            return self._classify_btc(current_price, ath_price, ath_ratio)

        # For other assets, use ATH ratio
        if ath_ratio is not None:
            return self._classify_by_ratio(symbol, current_price, ath_price, ath_ratio, btc_price)

        # Unknown ATH - use BTC context if available
        return self._classify_unknown(symbol, current_price, btc_price)

    def _classify_btc(
        self,
        current_price: float,
        ath_price: Optional[float],
        ath_ratio: Optional[float]
    ) -> PriceZone:
        """Classify BTC using both absolute and ratio thresholds."""

        # Use absolute thresholds (most reliable for BTC)
        if current_price < self.BTC_ACCUMULATION_MAX:
            return PriceZone(
                symbol="BTC-USD",
                current_price=current_price,
                ath_price=ath_price,
                ath_ratio=ath_ratio,
                zone="accumulation",
                is_distribution=False,
                is_caution=False,
                reason=f"BTC below ${self.BTC_ACCUMULATION_MAX/1000:.0f}k accumulation zone (35% win rate)",
            )
        elif current_price < self.BTC_CAUTION_MAX:
            return PriceZone(
                symbol="BTC-USD",
                current_price=current_price,
                ath_price=ath_price,
                ath_ratio=ath_ratio,
                zone="caution",
                is_distribution=False,
                is_caution=True,
                reason=f"BTC in ${self.BTC_ACCUMULATION_MAX/1000:.0f}k-${self.BTC_CAUTION_MAX/1000:.0f}k caution zone (22% win rate)",
            )
        else:
            return PriceZone(
                symbol="BTC-USD",
                current_price=current_price,
                ath_price=ath_price,
                ath_ratio=ath_ratio,
                zone="distribution",
                is_distribution=True,
                is_caution=True,
                reason=f"BTC above ${self.BTC_CAUTION_MAX/1000:.0f}k distribution zone (8% win rate) - NO TRADE",
            )

    def _classify_by_ratio(
        self,
        symbol: str,
        current_price: float,
        ath_price: float,
        ath_ratio: float,
        btc_price: Optional[float]
    ) -> PriceZone:
        """Classify by ATH ratio."""

        btc_context = None
        if btc_price:
            if btc_price > self.BTC_CAUTION_MAX:
                btc_context = "BTC in distribution zone - extra caution"
            elif btc_price > self.BTC_ACCUMULATION_MAX:
                btc_context = "BTC in caution zone"

        if ath_ratio < self.ACCUMULATION_MAX:
            return PriceZone(
                symbol=symbol,
                current_price=current_price,
                ath_price=ath_price,
                ath_ratio=ath_ratio,
                zone="accumulation",
                is_distribution=False,
                is_caution=False,
                reason=f"{symbol} at {ath_ratio:.0%} of ATH - accumulation zone",
                btc_context=btc_context,
            )
        elif ath_ratio < self.CAUTION_MAX:
            return PriceZone(
                symbol=symbol,
                current_price=current_price,
                ath_price=ath_price,
                ath_ratio=ath_ratio,
                zone="caution",
                is_distribution=False,
                is_caution=True,
                reason=f"{symbol} at {ath_ratio:.0%} of ATH - caution zone",
                btc_context=btc_context,
            )
        else:
            return PriceZone(
                symbol=symbol,
                current_price=current_price,
                ath_price=ath_price,
                ath_ratio=ath_ratio,
                zone="distribution",
                is_distribution=True,
                is_caution=True,
                reason=f"{symbol} at {ath_ratio:.0%} of ATH - distribution zone risk - NO TRADE",
                btc_context=btc_context,
            )

    def _classify_unknown(
        self,
        symbol: str,
        current_price: float,
        btc_price: Optional[float]
    ) -> PriceZone:
        """Classify when ATH is unknown - use BTC as proxy."""

        # If BTC is in distribution, be cautious with everything
        if btc_price and btc_price > self.BTC_CAUTION_MAX:
            return PriceZone(
                symbol=symbol,
                current_price=current_price,
                ath_price=None,
                ath_ratio=None,
                zone="caution",
                is_distribution=False,
                is_caution=True,
                reason=f"ATH unknown, but BTC in distribution zone - caution",
                btc_context=f"BTC at ${btc_price:,.0f} (distribution)",
            )

        return PriceZone(
            symbol=symbol,
            current_price=current_price,
            ath_price=None,
            ath_ratio=None,
            zone="unknown",
            is_distribution=False,
            is_caution=False,
            reason=f"ATH unknown for {symbol} - no zone filter applied",
            btc_context=f"BTC at ${btc_price:,.0f}" if btc_price else None,
        )

    def update_ath(self, symbol: str, ath_price: float) -> None:
        """Update ATH cache for a symbol."""
        self.ath_cache[symbol] = ath_price
        logger.info(f"Updated ATH for {symbol}: ${ath_price:,.2f}")

    def should_block_trade(
        self,
        symbol: str,
        current_price: float,
        btc_price: Optional[float] = None
    ) -> tuple[bool, str]:
        """
        Quick check if trade should be blocked due to distribution zone.

        Returns:
            (should_block, reason)
        """
        zone = self.get_zone(symbol, current_price, btc_price)
        return zone.is_distribution, zone.reason


# Singleton instance for easy access
_price_zone_filter: Optional[PriceZoneFilter] = None

def get_price_zone_filter() -> PriceZoneFilter:
    """Get or create the singleton PriceZoneFilter instance."""
    global _price_zone_filter
    if _price_zone_filter is None:
        _price_zone_filter = PriceZoneFilter()
    return _price_zone_filter
