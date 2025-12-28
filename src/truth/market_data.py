"""
Real-Time Market Data Fetcher for LLM Verification

Fetches current prices, recent candles, calculates technical indicators,
and searches for real-time news to provide full market context.

Usage:
    from src.truth.market_data import MarketDataFetcher

    fetcher = MarketDataFetcher()
    snapshot = await fetcher.get_market_snapshot("BTC-USD")
    # Returns: {
    #     "current_price": 98500.0,
    #     "price_change_24h": 2.5,
    #     "volume_24h": 1234567890,
    #     "indicators": {...},
    #     "recent_news": [...],
    #     ...
    # }
"""

import os
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import json

logger = logging.getLogger(__name__)

COINBASE_BASE_URL = "https://api.coinbase.com/api/v3/brokerage/market/products"
TAVILY_API_URL = "https://api.tavily.com/search"
# LunarCrush API v4 - coins list endpoint (works with Individual subscription)
LUNARCRUSH_API_URL = "https://lunarcrush.com/api4/public/coins/list/v1"

# Symbol to asset name mapping for better search queries
SYMBOL_NAMES = {
    "BTC-USD": "Bitcoin BTC",
    "ETH-USD": "Ethereum ETH",
    "SOL-USD": "Solana SOL",
    "XRP-USD": "XRP Ripple",
    "DOGE-USD": "Dogecoin DOGE",
    "ADA-USD": "Cardano ADA",
    "AVAX-USD": "Avalanche AVAX",
    "DOT-USD": "Polkadot DOT",
    "LINK-USD": "Chainlink LINK",
    "MATIC-USD": "Polygon MATIC",
    "UNI-USD": "Uniswap UNI",
    "LTC-USD": "Litecoin LTC",
    "BCH-USD": "Bitcoin Cash BCH",
}


@dataclass
class NewsItem:
    """A single news item from web search"""
    title: str
    snippet: str
    url: str
    published: str  # Relative time like "2 hours ago"
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SocialMetrics:
    """LunarCrush social sentiment data"""
    galaxy_score: float          # 0-100, overall health
    alt_rank: int                # Rank vs other cryptos (1 = best)
    social_dominance: float      # % of social volume vs all crypto
    social_volume: int           # Number of social posts
    social_sentiment: float      # -1 to 1
    bullish_posts: int
    bearish_posts: int
    trending_topics: List[str]   # Related trending topics
    top_influencer_posts: List[Dict[str, str]]  # Recent influential posts

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def sentiment_label(self) -> str:
        if self.social_sentiment > 0.3:
            return "bullish"
        elif self.social_sentiment < -0.3:
            return "bearish"
        elif self.bullish_posts > self.bearish_posts * 1.5:
            return "leaning_bullish"
        elif self.bearish_posts > self.bullish_posts * 1.5:
            return "leaning_bearish"
        return "neutral"


@dataclass
class TechnicalIndicators:
    """Technical indicator snapshot"""
    ema_12: float
    ema_26: float
    ema_50: float
    ema_200: float
    atr_14: float
    rsi_14: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    volatility_24h: float  # Percent
    trend_direction: str   # "bullish", "bearish", "sideways"
    trend_strength: str    # "strong", "moderate", "weak"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketSnapshot:
    """Complete market snapshot for a symbol"""
    symbol: str
    timestamp: str

    # Current price data
    current_price: float
    bid: float
    ask: float
    spread_pct: float

    # 24h data
    price_change_24h: float  # Percent
    volume_24h: float
    high_24h: float
    low_24h: float

    # Technical indicators
    indicators: TechnicalIndicators

    # Recent price action (last 20 candles summary)
    recent_high: float
    recent_low: float
    recent_range_pct: float
    candles_up: int
    candles_down: int

    # Decision verification helpers
    price_vs_ema_50: str  # "above", "below"
    price_vs_ema_200: str
    rsi_zone: str  # "oversold", "neutral", "overbought"
    bb_position: str  # "below_lower", "lower_half", "upper_half", "above_upper"

    # Real-time news and sentiment
    recent_news: List[NewsItem] = field(default_factory=list)
    news_sentiment: str = "neutral"  # "bullish", "bearish", "neutral", "mixed"

    # LunarCrush social metrics
    social_metrics: Optional[SocialMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["indicators"] = self.indicators.to_dict()
        result["recent_news"] = [n.to_dict() for n in self.recent_news]
        if self.social_metrics:
            result["social_metrics"] = self.social_metrics.to_dict()
        return result

    def to_llm_context(self) -> str:
        """Format snapshot as readable context for the LLM"""
        lines = [
            f"=== REAL-TIME MARKET DATA: {self.symbol} ===",
            f"Timestamp: {self.timestamp} (RIGHT NOW)",
            "",
            "CURRENT PRICE:",
            f"  Price: ${self.current_price:,.2f}",
            f"  24h Change: {self.price_change_24h:+.2f}%",
            f"  24h Range: ${self.low_24h:,.2f} - ${self.high_24h:,.2f}",
            f"  24h Volume: ${self.volume_24h:,.0f}",
            f"  Spread: {self.spread_pct:.3f}%",
            "",
            "TECHNICAL INDICATORS:",
            f"  EMA 12: ${self.indicators.ema_12:,.2f}",
            f"  EMA 26: ${self.indicators.ema_26:,.2f}",
            f"  EMA 50: ${self.indicators.ema_50:,.2f} (price {self.price_vs_ema_50})",
            f"  EMA 200: ${self.indicators.ema_200:,.2f} (price {self.price_vs_ema_200})",
            f"  RSI 14: {self.indicators.rsi_14:.1f} ({self.rsi_zone})",
            f"  ATR 14: ${self.indicators.atr_14:,.2f}",
            f"  Bollinger Bands: ${self.indicators.bb_lower:,.2f} / ${self.indicators.bb_middle:,.2f} / ${self.indicators.bb_upper:,.2f}",
            f"  BB Position: {self.bb_position}",
            f"  24h Volatility: {self.indicators.volatility_24h:.2f}%",
            "",
            "TREND ANALYSIS:",
            f"  Direction: {self.indicators.trend_direction}",
            f"  Strength: {self.indicators.trend_strength}",
            f"  Recent 20 candles: {self.candles_up} up / {self.candles_down} down",
            f"  Recent range: {self.recent_range_pct:.2f}%",
        ]

        # Add LunarCrush social metrics if available
        if self.social_metrics:
            sm = self.social_metrics
            lines.append("")
            lines.append("SOCIAL SENTIMENT (LunarCrush - REAL-TIME):")
            lines.append(f"  Galaxy Score: {sm.galaxy_score:.0f}/100 (overall health)")
            lines.append(f"  AltRank: #{sm.alt_rank} (vs all cryptos)")
            lines.append(f"  Social Sentiment: {sm.social_sentiment:+.2f} ({sm.sentiment_label})")
            lines.append(f"  Social Volume: {sm.social_volume:,} posts")
            lines.append(f"  Bullish/Bearish Posts: {sm.bullish_posts:,} / {sm.bearish_posts:,}")
            lines.append(f"  Social Dominance: {sm.social_dominance:.2f}%")

            if sm.trending_topics:
                lines.append(f"  Trending Topics: {', '.join(sm.trending_topics[:5])}")

            if sm.top_influencer_posts:
                lines.append("  Top Social Posts:")
                for post in sm.top_influencer_posts[:3]:
                    lines.append(f"    - @{post.get('creator', 'unknown')}: {post.get('text', '')[:100]}...")

        # Add news section if available
        if self.recent_news:
            lines.append("")
            lines.append(f"BREAKING NEWS (sentiment: {self.news_sentiment}):")
            for i, news in enumerate(self.recent_news[:5], 1):
                lines.append(f"  {i}. [{news.source}] {news.title}")
                lines.append(f"     {news.snippet[:150]}...")

        return "\n".join(lines)


class MarketDataFetcher:
    """
    Fetches real-time market data from multiple sources for LLM verification.

    Sources:
    - Coinbase: Price data, candles, technical indicators
    - LunarCrush: Social sentiment, Galaxy Score, trending posts
    - Tavily: Breaking news and web search
    """

    def __init__(
        self,
        lunarcrush_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None
    ):
        self._session: Optional[aiohttp.ClientSession] = None
        self.lunarcrush_api_key = lunarcrush_api_key or os.environ.get("LUNARCRUSH_API_KEY")
        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_lunarcrush_data(self, symbol: str) -> Optional[SocialMetrics]:
        """
        Fetch social sentiment data from LunarCrush API.

        Uses the coins/list endpoint which works with Individual subscription.
        """
        if not self.lunarcrush_api_key:
            logger.warning("LUNARCRUSH_API_KEY not set - skipping social data")
            return None

        # Convert symbol to LunarCrush ticker format (BTC-USD -> BTC)
        ticker = symbol.replace("-USD", "").upper()
        session = await self._get_session()

        try:
            # Use coins/list endpoint - works with Individual subscription
            headers = {"Authorization": f"Bearer {self.lunarcrush_api_key}"}

            async with session.get(LUNARCRUSH_API_URL, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(f"LunarCrush API returned {response.status}: {error_text[:200]}")
                    return None

                data = await response.json()
                coins = data.get("data", [])

                # Find our coin in the list
                coin_data = None
                for coin in coins:
                    if coin.get("symbol", "").upper() == ticker:
                        coin_data = coin
                        break

                if not coin_data:
                    logger.warning(f"Coin {ticker} not found in LunarCrush data")
                    return None

                # Extract metrics from coins/list response
                galaxy_score = float(coin_data.get("galaxy_score", 50))
                alt_rank = int(coin_data.get("alt_rank", 999))
                social_dominance = float(coin_data.get("social_dominance", 0))
                social_volume = int(coin_data.get("social_volume_24h", 0))
                sentiment = float(coin_data.get("sentiment", 50))
                interactions = int(coin_data.get("interactions_24h", 0))

                # Estimate bullish/bearish from sentiment (0-100 scale)
                # sentiment 50 = neutral, >50 = bullish, <50 = bearish
                if sentiment > 50:
                    bullish_ratio = (sentiment - 50) / 50  # 0-1
                    bullish = int(social_volume * (0.5 + bullish_ratio * 0.5))
                    bearish = social_volume - bullish
                else:
                    bearish_ratio = (50 - sentiment) / 50  # 0-1
                    bearish = int(social_volume * (0.5 + bearish_ratio * 0.5))
                    bullish = social_volume - bearish

                # Categories as trending topics
                categories = coin_data.get("categories", "")
                trending_topics = [c.strip() for c in categories.split(",") if c.strip()][:5]

                return SocialMetrics(
                    galaxy_score=galaxy_score,
                    alt_rank=alt_rank,
                    social_dominance=social_dominance,
                    social_volume=social_volume,
                    social_sentiment=(sentiment - 50) / 50,  # Convert 0-100 to -1 to 1
                    bullish_posts=bullish,
                    bearish_posts=bearish,
                    trending_topics=trending_topics,
                    top_influencer_posts=[]  # Not available in coins/list
                )

        except Exception as e:
            logger.error(f"LunarCrush API error: {e}")
            return None

    async def search_news(self, symbol: str, limit: int = 5) -> List[NewsItem]:
        """
        Search for recent news using Tavily API.

        Focuses on TODAY's news and breaking information.
        """
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not set - skipping news search")
            return []

        session = await self._get_session()
        asset_name = SYMBOL_NAMES.get(symbol, symbol.replace("-USD", ""))

        try:
            # Search for today's news
            query = f"{asset_name} crypto news today breaking"

            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "topic": "news",
                "days": 1,  # Only today
                "max_results": limit,
                "include_answer": False
            }

            async with session.post(TAVILY_API_URL, json=payload) as response:
                if response.status != 200:
                    logger.warning(f"Tavily API returned {response.status}")
                    return []

                data = await response.json()
                results = data.get("results", [])

                news_items = []
                for r in results:
                    news_items.append(NewsItem(
                        title=r.get("title", ""),
                        snippet=r.get("content", r.get("snippet", ""))[:300],
                        url=r.get("url", ""),
                        published=r.get("published_date", "today"),
                        source=r.get("source", "unknown")
                    ))

                return news_items

        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Fetch current price data from Coinbase"""
        session = await self._get_session()
        url = f"{COINBASE_BASE_URL}/{symbol}"

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "price": float(data.get("price", 0)),
                        "bid": float(data.get("bid", 0) or data.get("price", 0)),
                        "ask": float(data.get("ask", 0) or data.get("price", 0)),
                        "volume_24h": float(data.get("volume_24h", 0)),
                        "price_change_24h": float(data.get("price_percentage_change_24h", 0)),
                    }
                else:
                    logger.warning(f"Coinbase API returned {response.status} for {symbol}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to fetch price for {symbol}: {e}")
            return {}

    async def get_candles(
        self,
        symbol: str,
        granularity: str = "ONE_HOUR",
        limit: int = 200
    ) -> List[Dict[str, float]]:
        """Fetch OHLCV candle data from Coinbase"""
        session = await self._get_session()
        url = f"{COINBASE_BASE_URL}/{symbol}/candles"

        # Calculate time range
        granularity_seconds = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600,
            "FOUR_HOUR": 14400,
            "ONE_DAY": 86400
        }

        end_time = int(datetime.now(timezone.utc).timestamp())
        start_time = end_time - (limit * granularity_seconds.get(granularity, 3600))

        params = {
            "start": str(start_time),
            "end": str(end_time),
            "granularity": granularity,
            "limit": str(limit)
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    candles = data.get("candles", [])

                    # Format and sort by timestamp
                    formatted = [{
                        "timestamp": int(c["start"]),
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": float(c["volume"])
                    } for c in candles]

                    # Sort ascending by timestamp
                    formatted.sort(key=lambda x: x["timestamp"])
                    return formatted
                else:
                    logger.warning(f"Coinbase candles API returned {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return []

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral default

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        if len(gains) < period:
            return 50.0

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(self, candles: List[Dict], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(candles) < period + 1:
            return 0

        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = candles[i-1]["close"]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0

        return sum(true_ranges[-period:]) / period

    def _calculate_bollinger_bands(
        self,
        prices: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current = prices[-1] if prices else 0
            return current, current, current

        recent = prices[-period:]
        middle = sum(recent) / period

        variance = sum((p - middle) ** 2 for p in recent) / period
        std = variance ** 0.5

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return upper, middle, lower

    def _calculate_indicators(self, candles: List[Dict]) -> TechnicalIndicators:
        """Calculate all technical indicators from candle data"""
        if not candles:
            return TechnicalIndicators(
                ema_12=0, ema_26=0, ema_50=0, ema_200=0,
                atr_14=0, rsi_14=50, bb_upper=0, bb_middle=0, bb_lower=0,
                volatility_24h=0, trend_direction="unknown", trend_strength="unknown"
            )

        closes = [c["close"] for c in candles]

        # EMAs
        ema_12 = self._calculate_ema(closes, 12)
        ema_26 = self._calculate_ema(closes, 26)
        ema_50 = self._calculate_ema(closes, 50)
        ema_200 = self._calculate_ema(closes, min(200, len(closes)))

        # RSI
        rsi_14 = self._calculate_rsi(closes, 14)

        # ATR
        atr_14 = self._calculate_atr(candles, 14)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20, 2.0)

        # 24h volatility (using last 24 1-hour candles)
        if len(candles) >= 24:
            recent_24 = closes[-24:]
            high_24 = max(recent_24)
            low_24 = min(recent_24)
            volatility_24h = ((high_24 - low_24) / low_24) * 100 if low_24 > 0 else 0
        else:
            volatility_24h = 0

        # Trend direction and strength
        current_price = closes[-1]

        if ema_12 > ema_26 > ema_50:
            trend_direction = "bullish"
        elif ema_12 < ema_26 < ema_50:
            trend_direction = "bearish"
        else:
            trend_direction = "sideways"

        # Trend strength based on EMA spread
        ema_spread = abs(ema_12 - ema_50) / ema_50 * 100 if ema_50 > 0 else 0
        if ema_spread > 3:
            trend_strength = "strong"
        elif ema_spread > 1:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"

        return TechnicalIndicators(
            ema_12=round(ema_12, 2),
            ema_26=round(ema_26, 2),
            ema_50=round(ema_50, 2),
            ema_200=round(ema_200, 2),
            atr_14=round(atr_14, 2),
            rsi_14=round(rsi_14, 1),
            bb_upper=round(bb_upper, 2),
            bb_middle=round(bb_middle, 2),
            bb_lower=round(bb_lower, 2),
            volatility_24h=round(volatility_24h, 2),
            trend_direction=trend_direction,
            trend_strength=trend_strength
        )

    async def get_market_snapshot(
        self,
        symbol: str,
        include_social: bool = True,
        include_news: bool = True
    ) -> MarketSnapshot:
        """
        Get a complete market snapshot for LLM context.

        Fetches:
        - Current price and candles from Coinbase
        - Technical indicators (EMA, RSI, ATR, Bollinger)
        - Social sentiment from LunarCrush (if API key available)
        - Breaking news from Tavily (if API key available)
        """
        import asyncio

        # Fetch all data concurrently for speed
        tasks = [
            self.get_current_price(symbol),
            self.get_candles(symbol, "ONE_HOUR", 200),
        ]

        if include_social:
            tasks.append(self.get_lunarcrush_data(symbol))
        if include_news:
            tasks.append(self.search_news(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results
        price_data = results[0] if not isinstance(results[0], Exception) else {}
        candles = results[1] if not isinstance(results[1], Exception) else []

        social_metrics = None
        news_items = []

        if include_social and len(results) > 2:
            if not isinstance(results[2], Exception):
                social_metrics = results[2]

        if include_news:
            news_idx = 3 if include_social else 2
            if len(results) > news_idx and not isinstance(results[news_idx], Exception):
                news_items = results[news_idx]

        if not price_data or not candles:
            raise ValueError(f"Failed to fetch market data for {symbol}")

        # Calculate indicators
        indicators = self._calculate_indicators(candles)

        # Extract price data
        current_price = price_data.get("price", 0)
        bid = price_data.get("bid", current_price)
        ask = price_data.get("ask", current_price)
        spread_pct = ((ask - bid) / current_price * 100) if current_price > 0 else 0

        # 24h high/low from candles
        if len(candles) >= 24:
            recent_24 = candles[-24:]
            high_24h = max(c["high"] for c in recent_24)
            low_24h = min(c["low"] for c in recent_24)
        else:
            high_24h = max(c["high"] for c in candles) if candles else current_price
            low_24h = min(c["low"] for c in candles) if candles else current_price

        # Recent 20 candles analysis
        recent_20 = candles[-20:] if len(candles) >= 20 else candles
        recent_high = max(c["high"] for c in recent_20)
        recent_low = min(c["low"] for c in recent_20)
        recent_range_pct = ((recent_high - recent_low) / recent_low * 100) if recent_low > 0 else 0

        candles_up = sum(1 for c in recent_20 if c["close"] > c["open"])
        candles_down = len(recent_20) - candles_up

        # Position helpers
        price_vs_ema_50 = "above" if current_price > indicators.ema_50 else "below"
        price_vs_ema_200 = "above" if current_price > indicators.ema_200 else "below"

        if indicators.rsi_14 < 30:
            rsi_zone = "oversold"
        elif indicators.rsi_14 > 70:
            rsi_zone = "overbought"
        else:
            rsi_zone = "neutral"

        if current_price < indicators.bb_lower:
            bb_position = "below_lower"
        elif current_price < indicators.bb_middle:
            bb_position = "lower_half"
        elif current_price < indicators.bb_upper:
            bb_position = "upper_half"
        else:
            bb_position = "above_upper"

        # Determine news sentiment from headlines (simple heuristic)
        news_sentiment = "neutral"
        if news_items:
            bullish_words = ["surge", "rally", "breakout", "soar", "bull", "high", "gains", "up"]
            bearish_words = ["crash", "drop", "fall", "bear", "low", "losses", "down", "plunge"]

            bull_count = sum(1 for n in news_items if any(w in n.title.lower() for w in bullish_words))
            bear_count = sum(1 for n in news_items if any(w in n.title.lower() for w in bearish_words))

            if bull_count > bear_count:
                news_sentiment = "bullish"
            elif bear_count > bull_count:
                news_sentiment = "bearish"
            elif bull_count > 0 and bear_count > 0:
                news_sentiment = "mixed"

        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            current_price=current_price,
            bid=bid,
            ask=ask,
            spread_pct=round(spread_pct, 4),
            price_change_24h=price_data.get("price_change_24h", 0),
            volume_24h=price_data.get("volume_24h", 0),
            high_24h=high_24h,
            low_24h=low_24h,
            indicators=indicators,
            recent_high=recent_high,
            recent_low=recent_low,
            recent_range_pct=round(recent_range_pct, 2),
            candles_up=candles_up,
            candles_down=candles_down,
            price_vs_ema_50=price_vs_ema_50,
            price_vs_ema_200=price_vs_ema_200,
            rsi_zone=rsi_zone,
            bb_position=bb_position,
            recent_news=news_items,
            news_sentiment=news_sentiment,
            social_metrics=social_metrics
        )

    async def verify_decision(
        self,
        symbol: str,
        decision_price: float,
        decision_type: str,  # "long", "short", "close", "hold"
        decision_timestamp: str
    ) -> Dict[str, Any]:
        """
        Verify a past decision against current market state.

        Returns analysis of whether the decision was validated by market movement.
        """
        snapshot = await self.get_market_snapshot(symbol)

        current_price = snapshot.current_price
        price_change_pct = ((current_price - decision_price) / decision_price * 100) if decision_price > 0 else 0

        # Determine if decision was validated
        if decision_type in ["long", "signal_long", "buy"]:
            if price_change_pct > 1:
                validation = "VALIDATED - price rose since long signal"
                grade = "good"
            elif price_change_pct < -2:
                validation = "INVALIDATED - price fell since long signal"
                grade = "poor"
            else:
                validation = "NEUTRAL - insufficient movement to judge"
                grade = "neutral"

        elif decision_type in ["short", "signal_short", "sell", "close", "signal_close"]:
            if price_change_pct < -1:
                validation = "VALIDATED - price fell since close/short signal"
                grade = "good"
            elif price_change_pct > 2:
                validation = "INVALIDATED - price rose since close/short signal (missed upside)"
                grade = "poor"
            else:
                validation = "NEUTRAL - insufficient movement to judge"
                grade = "neutral"

        elif decision_type in ["hold", "no_signal", "no_action"]:
            if abs(price_change_pct) < 2:
                validation = "VALIDATED - market was indeed sideways"
                grade = "good"
            else:
                validation = f"MISSED OPPORTUNITY - market moved {abs(price_change_pct):.1f}%"
                grade = "neutral"
        else:
            validation = "Unable to validate - unknown decision type"
            grade = "unknown"

        return {
            "decision_price": decision_price,
            "current_price": current_price,
            "price_change_pct": round(price_change_pct, 2),
            "validation": validation,
            "grade": grade,
            "time_since_decision": decision_timestamp,
            "market_snapshot": snapshot.to_dict()
        }


# Convenience function for one-off use
async def get_market_snapshot(symbol: str) -> MarketSnapshot:
    """Get market snapshot for a symbol"""
    fetcher = MarketDataFetcher()
    try:
        return await fetcher.get_market_snapshot(symbol)
    finally:
        await fetcher.close()
