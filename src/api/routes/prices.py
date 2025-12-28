"""
Price Endpoints
Coinbase API proxy for live prices and candles.
"""

import aiohttp
from fastapi import APIRouter, Query
from datetime import datetime, timezone
from typing import Optional

router = APIRouter()


@router.get("/prices/{symbol}")
async def get_price(symbol: str):
    """
    Get current price for a symbol.
    Proxies Coinbase API to avoid CORS issues.
    """
    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "symbol": symbol,
                        "price": float(data.get("price", 0)),
                        "bid": float(data.get("quote_increment", 0)),
                        "volume_24h": float(data.get("volume_24h", 0)),
                        "price_change_24h": float(data.get("price_percentage_change_24h", 0)),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                return {"error": f"API returned {response.status}"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/candles/{symbol}")
async def get_candles(
    symbol: str,
    granularity: str = Query(default="FOUR_HOUR", regex="^(ONE_MINUTE|FIVE_MINUTE|FIFTEEN_MINUTE|ONE_HOUR|FOUR_HOUR|ONE_DAY)$"),
    limit: int = Query(default=150, le=300)
):
    """
    Get OHLCV candle data for charting.

    Granularity options: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, FOUR_HOUR, ONE_DAY
    """
    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles"

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
    start_time = end_time - (limit * granularity_seconds.get(granularity, 14400))

    params = {
        "start": str(start_time),
        "end": str(end_time),
        "granularity": granularity,
        "limit": str(limit)
    }

    try:
        async with aiohttp.ClientSession() as session:
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

                    return {
                        "symbol": symbol,
                        "granularity": granularity,
                        "candles": formatted
                    }
                return {"error": f"API returned {response.status}"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/ticker")
async def get_ticker():
    """
    Get ticker data for multiple symbols.
    Used for the price ticker strip.
    """
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "BCH-USD", "LINK-USD"]
    results = []

    async with aiohttp.ClientSession() as session:
        for symbol in symbols:
            url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}"
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        results.append({
                            "symbol": symbol.replace("-USD", ""),
                            "full_symbol": symbol,
                            "price": float(data.get("price", 0)),
                            "change_24h": float(data.get("price_percentage_change_24h", 0))
                        })
            except:
                results.append({
                    "symbol": symbol.replace("-USD", ""),
                    "full_symbol": symbol,
                    "price": 0,
                    "change_24h": 0
                })

    return {
        "tickers": results,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
