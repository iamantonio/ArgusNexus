"""
V4 WebSocket Manager - Real-time Price Sidecar

Listens to Coinbase Advanced Trade WebSocket (v3) ticker channel.
Provides real-time price updates to the TradingEngine for instant stops.

Principles:
- Sidecar: Polling remains the source of truth; WS is a speed-up.
- Non-blocking: Uses asyncio to run alongside the main fleet.
- Efficient: Subscribes only to relevant symbols.
"""

import asyncio
import json
import logging
import time
import websockets
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages real-time ticker data from Coinbase.
    
    Provides a way for TradingEngine to subscribe to price updates.
    """
    WS_URL = "wss://advanced-trade-ws.coinbase.com"

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.prices: Dict[str, Decimal] = {}
        self.callbacks: List[Callable[[str, Decimal], Awaitable[None]]] = []
        self._stop_event = asyncio.Event()
        self._running_task: Optional[asyncio.Task] = None

    def register_callback(self, callback: Callable[[str, Decimal], Awaitable[None]]):
        """Register a function to be called on every price update."""
        self.callbacks.append(callback)

    async def start(self):
        """Start the WebSocket listener task."""
        if self._running_task:
            return
        self._stop_event.clear()
        self._running_task = asyncio.create_task(self._run_loop())
        logger.info(f"WebSocketManager started for {self.symbols}")

    async def stop(self):
        """Stop the WebSocket listener task."""
        self._stop_event.set()
        if self._running_task:
            await self._running_task
            self._running_task = None
        logger.info("WebSocketManager stopped")

    async def _run_loop(self):
        """Main reconnection loop for WebSocket."""
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.WS_URL) as ws:
                    # Subscribe to ticker
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": self.symbols,
                        "channel": "ticker",
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to ticker for {self.symbols}")

                    while not self._stop_event.is_set():
                        try:
                            # Use timeout to check stop_event periodically
                            msg_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            msg = json.loads(msg_raw)
                            
                            if msg.get("channel") == "ticker" and msg.get("events"):
                                for event in msg["events"]:
                                    for update in event.get("tickers", []):
                                        symbol = update.get("product_id")
                                        price_str = update.get("price")
                                        if symbol and price_str:
                                            price = Decimal(price_str)
                                            self.prices[symbol] = price
                                            
                                            # Fire callbacks
                                            for cb in self.callbacks:
                                                await cb(symbol, price)
                                                
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"WebSocket message error: {e}")
                            break # Trigger reconnect
                            
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.warning(f"WebSocket connection lost ({e}). Reconnecting in 5s...")
                    await asyncio.sleep(5)

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """Get the latest cached price for a symbol."""
        return self.prices.get(symbol)
