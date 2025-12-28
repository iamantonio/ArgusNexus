"""
Coinbase Market Scanner - The All-Seeing Eye

Scans the Coinbase market for TURTLE-4 breakout signals.

Architecture (from Technical Research 2025-12-16):
- Rate Limit: 8 requests/second (safe margin below 10/s limit)
- Concurrency: AsyncIO with Semaphore
- Efficiency: 350 candles per API call (no pagination needed)
- Filtering: Excludes stablecoins (USDT-USD, USDC-USD, etc.)

FAST MODE (Default):
- Fetches all products with volume_24h
- Sorts by 24h volume descending
- Scans only Top 100 liquid assets
- Reduces API load by ~85%
- Ensures we only trade where we can exit

Performance:
- Scans Top 100 in ~15 seconds
- Minimal rate limit retries
- No IP bans

Usage:
    python src/scanner.py              # Fast mode (Top 100)
    python src/scanner.py --full       # Full market scan

Output:
    TURTLE-4 Breakouts detected with signal context.
"""

import asyncio
import aiohttp
import time
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Any
from pathlib import Path

import pandas as pd

# Add src to path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from strategy.donchian import DonchianBreakout, Signal, SignalResult


# =============================================================================
# CONSTANTS
# =============================================================================

# Stablecoins to exclude (flat charts, not tradable for momentum)
STABLECOINS = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "GUSD",
    "FRAX", "LUSD", "SUSD", "PYUSD", "EURC", "USDD", "FDUSD",
    "PAX", "USDJ", "UST", "MIM", "TRIBE", "FEI", "CEUR", "CUSD"
}

# Granularity enum values for Coinbase API
class Granularity(str, Enum):
    ONE_MINUTE = "ONE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    TWO_HOUR = "TWO_HOUR"
    FOUR_HOUR = "FOUR_HOUR"
    SIX_HOUR = "SIX_HOUR"
    ONE_DAY = "ONE_DAY"

    def to_seconds(self) -> int:
        """Convert granularity to seconds."""
        mapping = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "FOUR_HOUR": 14400,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400
        }
        return mapping[self.value]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScanResult:
    """Result of scanning a single product."""
    product_id: str
    signal: Signal
    reason: str
    price: Decimal = Decimal("0")
    upper_channel: Decimal = Decimal("0")
    volume_ratio: Optional[Decimal] = None
    sma_200: Optional[Decimal] = None
    candles_fetched: int = 0
    error: Optional[str] = None
    scan_time_ms: float = 0.0


@dataclass
class MarketScanResult:
    """Result of full market scan."""
    results: List[ScanResult] = field(default_factory=list)
    breakouts: List[ScanResult] = field(default_factory=list)
    total_products: int = 0
    successful: int = 0
    failed: int = 0
    scan_duration_seconds: float = 0.0


# =============================================================================
# COINBASE MARKET SCANNER
# =============================================================================

class CoinbaseMarketScanner:
    """
    High-performance market scanner for Coinbase Advanced Trade.

    Rate-limited async scanner using Section 6.2 pattern:
    - AsyncIO with Semaphore for concurrency control
    - 8 requests/second safe rate
    - Exponential backoff on 429 errors
    """

    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"
    PUBLIC_BASE_URL = "https://api.coinbase.com/api/v3/brokerage/market"

    def __init__(
        self,
        requests_per_second: int = 8,
        use_public_api: bool = True,
        verbose: bool = True
    ):
        """
        Initialize scanner.

        Args:
            requests_per_second: Max requests per second (default: 8)
            use_public_api: Use public endpoints (no auth needed)
            verbose: Print progress
        """
        self.requests_per_second = requests_per_second
        self.use_public_api = use_public_api
        self.verbose = verbose

        # Rate limiting
        self.semaphore = asyncio.Semaphore(requests_per_second)
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0

        # Strategy
        self.strategy = DonchianBreakout()

        # Stats
        self.request_count = 0
        self.retry_count = 0

    def _log(self, message: str):
        """Print if verbose."""
        if self.verbose:
            print(message)

    async def _rate_limited_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Make a rate-limited HTTP GET request with exponential backoff.

        Args:
            session: aiohttp session
            url: Request URL
            params: Query parameters
            max_retries: Max retry attempts on 429

        Returns:
            JSON response or None on failure
        """
        async with self.semaphore:
            # Enforce minimum interval
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)

            self.last_request_time = time.time()
            self.request_count += 1

            for attempt in range(max_retries):
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            return await response.json()

                        elif response.status == 429:
                            # Rate limited - exponential backoff
                            retry_after = response.headers.get("Retry-After", "2")
                            try:
                                wait_time = float(retry_after)
                            except ValueError:
                                wait_time = 2 ** attempt

                            self.retry_count += 1
                            self._log(f"  Rate limited, waiting {wait_time:.1f}s...")
                            await asyncio.sleep(wait_time)
                            continue

                        elif response.status == 404:
                            # Product not found / delisted
                            return None

                        else:
                            text = await response.text()
                            self._log(f"  HTTP {response.status}: {text[:100]}")
                            return None

                except asyncio.TimeoutError:
                    self._log(f"  Timeout on attempt {attempt + 1}")
                    await asyncio.sleep(1)

                except Exception as e:
                    self._log(f"  Error: {e}")
                    return None

            return None

    async def get_products(
        self,
        session: aiohttp.ClientSession
    ) -> List[Dict]:
        """
        Fetch all tradable SPOT products.

        Returns:
            List of product dictionaries
        """
        if self.use_public_api:
            url = f"{self.PUBLIC_BASE_URL}/products"
        else:
            url = f"{self.BASE_URL}/products"

        params = {
            "product_type": "SPOT",
            "get_tradability_status": "true"
        }

        response = await self._rate_limited_request(session, url, params)

        if response and "products" in response:
            return response["products"]
        return []

    async def get_candles(
        self,
        session: aiohttp.ClientSession,
        product_id: str,
        granularity: Granularity = Granularity.FOUR_HOUR,
        limit: int = 300
    ) -> Optional[List[Dict]]:
        """
        Fetch candles for a single product.

        Args:
            session: aiohttp session
            product_id: Trading pair (e.g., "BTC-USD")
            granularity: Candle timeframe
            limit: Number of candles (max 350)

        Returns:
            List of candle dictionaries or None
        """
        # Calculate time range
        end_time = int(time.time())
        start_time = end_time - (limit * granularity.to_seconds())

        if self.use_public_api:
            url = f"{self.PUBLIC_BASE_URL}/products/{product_id}/candles"
        else:
            url = f"{self.BASE_URL}/products/{product_id}/candles"

        params = {
            "start": str(start_time),
            "end": str(end_time),
            "granularity": granularity.value,
            "limit": str(min(limit, 350))
        }

        response = await self._rate_limited_request(session, url, params)

        if response and "candles" in response:
            return response["candles"]
        return None

    def _filter_tradable_products(
        self,
        products: List[Dict],
        top_n: Optional[int] = None
    ) -> List[str]:
        """
        Filter products to only tradable, non-stablecoin pairs.

        FAST MODE (top_n specified):
        - Sorts by 24h volume descending
        - Returns only top N liquid assets
        - Ensures we trade where we can exit

        Args:
            products: Raw product list from API
            top_n: If specified, return only top N by volume (FAST MODE)

        Returns:
            List of product IDs to scan (sorted by volume if top_n)
        """
        tradable = []

        for p in products:
            product_id = p.get("product_id", "")
            base = p.get("base_currency_id", "").upper()
            quote = p.get("quote_currency_id", "").upper()

            # Skip stablecoins
            if base in STABLECOINS:
                continue

            # Skip non-USD quote (for now)
            if quote not in ("USD", "USDC"):
                continue

            # Skip disabled/view-only
            if p.get("is_disabled", False):
                continue
            if p.get("trading_disabled", False):
                continue
            if p.get("view_only", False):
                continue

            # Product type must be SPOT
            if p.get("product_type", "") != "SPOT":
                continue

            # Get volume for sorting
            try:
                volume_24h = float(p.get("volume_24h", 0) or 0)
            except (ValueError, TypeError):
                volume_24h = 0.0

            tradable.append({
                "product_id": product_id,
                "volume_24h": volume_24h
            })

        # Sort by 24h volume (descending) - most liquid first
        tradable.sort(key=lambda x: x["volume_24h"], reverse=True)

        # Apply FAST MODE cutoff if specified
        if top_n is not None:
            tradable = tradable[:top_n]
            self._log(f"  FAST MODE: Filtered to Top {top_n} by 24h volume")

        # Return just product IDs (now sorted by liquidity)
        return [p["product_id"] for p in tradable]

    def _candles_to_dataframe(self, candles: List[Dict]) -> pd.DataFrame:
        """
        Convert Coinbase candles to pandas DataFrame.

        Coinbase format: {start, open, high, low, close, volume}
        """
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)

        # Convert start timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s")

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort chronologically (Coinbase returns newest first)
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    async def _scan_product(
        self,
        session: aiohttp.ClientSession,
        product_id: str,
        granularity: Granularity,
        candle_limit: int
    ) -> ScanResult:
        """
        Scan a single product for TURTLE-4 breakout.

        Args:
            session: aiohttp session
            product_id: Trading pair
            granularity: Candle timeframe
            candle_limit: Number of candles to fetch

        Returns:
            ScanResult with signal and context
        """
        start_time = time.time()

        # Fetch candles
        candles = await self.get_candles(
            session, product_id, granularity, candle_limit
        )

        if not candles:
            return ScanResult(
                product_id=product_id,
                signal=Signal.HOLD,
                reason="No candles returned",
                error="fetch_failed",
                scan_time_ms=(time.time() - start_time) * 1000
            )

        # Convert to DataFrame
        df = self._candles_to_dataframe(candles)

        if len(df) < self.strategy.min_bars:
            return ScanResult(
                product_id=product_id,
                signal=Signal.HOLD,
                reason=f"Insufficient data ({len(df)} bars, need {self.strategy.min_bars})",
                candles_fetched=len(df),
                error="insufficient_data",
                scan_time_ms=(time.time() - start_time) * 1000
            )

        # Evaluate strategy (no open position - looking for entries)
        try:
            result: SignalResult = self.strategy.evaluate(
                df=df,
                timestamp=datetime.now(timezone.utc),
                has_open_position=False
            )

            return ScanResult(
                product_id=product_id,
                signal=result.signal,
                reason=result.reason,
                price=result.context.current_price,
                upper_channel=result.context.upper_channel,
                volume_ratio=result.context.volume_ratio,
                sma_200=result.context.sma_200,
                candles_fetched=len(df),
                scan_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ScanResult(
                product_id=product_id,
                signal=Signal.HOLD,
                reason=f"Strategy error: {e}",
                candles_fetched=len(df),
                error=str(e),
                scan_time_ms=(time.time() - start_time) * 1000
            )

    async def scan_market(
        self,
        granularity: Granularity = Granularity.FOUR_HOUR,
        candle_limit: int = 300,
        top_n: Optional[int] = 100
    ) -> MarketScanResult:
        """
        Scan Coinbase market for TURTLE-4 breakouts.

        Args:
            granularity: Candle timeframe (default: 4H)
            candle_limit: Candles per product (default: 300)
            top_n: Scan only top N by 24h volume (FAST MODE, default: 100)
                   Set to None for full market scan

        Returns:
            MarketScanResult with all breakouts
        """
        start_time = time.time()
        result = MarketScanResult()

        mode = "FAST MODE" if top_n else "FULL SCAN"

        self._log("=" * 60)
        self._log(f"COINBASE MARKET SCANNER - THE ALL-SEEING EYE [{mode}]")
        self._log("=" * 60)
        self._log(f"Granularity: {granularity.value}")
        self._log(f"Candles per asset: {candle_limit}")
        self._log(f"Rate limit: {self.requests_per_second} req/sec")
        if top_n:
            self._log(f"Volume Filter: Top {top_n} liquid assets")
        self._log("")

        # Configure timeout
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Step 1: Get all products
            self._log("Fetching product list...")
            products = await self.get_products(session)
            self._log(f"  Found {len(products)} total products")

            # Step 2: Filter tradable products (with volume sort + FAST MODE cutoff)
            product_ids = self._filter_tradable_products(products, top_n=top_n)
            result.total_products = len(product_ids)
            self._log(f"  Tradable (excl. stablecoins): {len(product_ids)}")
            self._log("")

            # Step 3: Scan products concurrently
            self._log(f"Scanning {len(product_ids)} products for TURTLE-4 breakouts...")
            self._log("")

            tasks = [
                self._scan_product(session, pid, granularity, candle_limit)
                for pid in product_ids
            ]

            scan_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Step 4: Process results
            for sr in scan_results:
                if isinstance(sr, Exception):
                    result.failed += 1
                    continue

                if sr.error:
                    result.failed += 1
                else:
                    result.successful += 1

                result.results.append(sr)

                # Collect breakouts (LONG signals)
                if sr.signal == Signal.LONG:
                    result.breakouts.append(sr)

        result.scan_duration_seconds = time.time() - start_time

        self._log("")
        self._log("=" * 60)
        self._log("SCAN COMPLETE")
        self._log("=" * 60)
        self._log(f"Duration: {result.scan_duration_seconds:.1f} seconds")
        self._log(f"Products scanned: {result.successful}/{result.total_products}")
        self._log(f"Failed: {result.failed}")
        self._log(f"API requests: {self.request_count}")
        self._log(f"Rate limit retries: {self.retry_count}")
        self._log("")

        return result


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_breakouts(result: MarketScanResult):
    """Display TURTLE-4 breakouts in a clean format."""
    print("")
    print("=" * 70)
    print(" TURTLE-4 BREAKOUTS DETECTED ".center(70, "="))
    print("=" * 70)
    print("")

    if not result.breakouts:
        print("  No breakouts detected in this scan.")
        print("")
        print("  This means no assets are currently breaking above their")
        print("  55-period high with confirmed trend and volume filters.")
        print("")
    else:
        print(f"  Found {len(result.breakouts)} TURTLE-4 breakout signal(s):")
        print("")

        for i, b in enumerate(result.breakouts, 1):
            print(f"  [{i}] {b.product_id}")
            print(f"      Price: ${float(b.price):,.2f}")
            print(f"      Upper Channel (55-period high): ${float(b.upper_channel):,.2f}")
            if b.volume_ratio:
                print(f"      Volume: {float(b.volume_ratio):.1f}x average")
            if b.sma_200:
                print(f"      SMA(200): ${float(b.sma_200):,.2f}")
            print(f"      Reason: {b.reason[:80]}...")
            print("")

    print("=" * 70)


def display_near_breakouts(result: MarketScanResult, threshold_pct: float = 2.0):
    """Display assets near breakout (within threshold of upper channel)."""
    near_breakouts = []

    for sr in result.results:
        if sr.signal != Signal.LONG and sr.price > 0 and sr.upper_channel > 0:
            distance_pct = ((sr.upper_channel - sr.price) / sr.price) * 100
            if 0 < distance_pct <= threshold_pct:
                near_breakouts.append((sr, distance_pct))

    if near_breakouts:
        # Sort by distance (closest first)
        near_breakouts.sort(key=lambda x: x[1])

        print("")
        print(f" NEAR BREAKOUT (within {threshold_pct}% of upper channel) ".center(70, "-"))
        print("")

        for sr, dist in near_breakouts[:10]:  # Top 10
            print(f"  {sr.product_id}: ${float(sr.price):,.2f} ({dist:.1f}% below channel)")

        print("")


# =============================================================================
# MAIN
# =============================================================================

async def main(full_scan: bool = False, top_n: int = 100):
    """
    Run the All-Seeing Eye scanner.

    Args:
        full_scan: If True, scan entire market (slow). Default: False (FAST MODE)
        top_n: Number of top liquid assets to scan in FAST MODE (default: 100)
    """
    mode = "FULL SCAN" if full_scan else f"FAST MODE (Top {top_n})"

    print("")
    print(" THE ALL-SEEING EYE ".center(60, "*"))
    print(" Coinbase Market Scanner v1.0 ".center(60, " "))
    print(f" TURTLE-4 Breakout Detection ".center(60, " "))
    print(f" {mode} ".center(60, " "))
    print("*" * 60)
    print("")

    # Initialize scanner
    scanner = CoinbaseMarketScanner(
        requests_per_second=8,
        use_public_api=True,
        verbose=True
    )

    # Scan market for 4H breakouts
    result = await scanner.scan_market(
        granularity=Granularity.FOUR_HOUR,
        candle_limit=300,
        top_n=None if full_scan else top_n
    )

    # Display results
    display_breakouts(result)
    display_near_breakouts(result, threshold_pct=3.0)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Coinbase Market Scanner - TURTLE-4 Breakout Detection"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full market scan (slow, ~60+ seconds). Default: FAST MODE (Top 100)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of top liquid assets to scan in FAST MODE (default: 100)"
    )

    args = parser.parse_args()

    asyncio.run(main(full_scan=args.full, top_n=args.top))
