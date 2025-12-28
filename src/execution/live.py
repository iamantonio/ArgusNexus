"""
V4 Async Live Executor - High-Performance Order Execution

Wraps the Coinbase Advanced Trade API using aiohttp and ES256 JWT signing.

CRITICAL: This places REAL orders with REAL money.

Key V4 Upgrade:
- Asynchronous connection pooling via aiohttp
- ES256 JWT authentication for Cloud API Keys (v3)
- Non-blocking execution for fleet scalability
- Precise slippage tracking and fee logging
"""

import logging
import time
import jwt
import aiohttp
import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional, List
from cryptography.hazmat.primitives import serialization
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from .base import Executor
from .schema import (
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    OrderSide,
    OrderType,
)

logger = logging.getLogger(__name__)

class LiveExecutor(Executor):
    """
    Executes real orders via Coinbase Advanced Trade API (Async).

    ⚠️ WARNING: This executor places REAL orders with REAL money.
    """

    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"

    def __init__(self, key_name: str, key_secret: str, session: Optional[aiohttp.ClientSession] = None):
        """
        Initialize live executor.

        Args:
            key_name: API Key Name (organizations/{org}/apiKeys/{id})
            key_secret: PEM-encoded EC Private Key
            session: Optional shared aiohttp session
        """
        super().__init__(mode=ExecutionMode.LIVE)
        
        self.key_name = key_name
        self.key_secret = key_secret
        self.session = session
        self._own_session = False

        self.logger.warning(
            "🔴 ASYNC LIVE EXECUTOR INITIALIZED - ALL ORDERS ARE REAL 🔴"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            self._own_session = True
        return self.session

    def _generate_jwt(self, method: str, path: str) -> str:
        """
        Generate ES256 JWT for Coinbase Cloud authentication.
        
        Implementation based on CDP authentication specification:
        - Algorithm: ES256
        - Claims: iss (cdp), nbf (now), exp (now+60s), sub (key_name), uri (METHOD PATH)
        """
        import os
        payload = {
            "iss": "cdp",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 60,
            "sub": self.key_name,
        }
        # Format: "METHOD /api/v3/brokerage/..."
        payload["uri"] = f"{method.upper()} {path}"

        token = jwt.encode(
            payload,
            self.key_secret,
            algorithm="ES256",
            headers={
                "kid": self.key_name, 
                "nonce": os.urandom(16).hex(),
                "typ": "JWT"
            }
        )
        return token

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception), # Broad for now, can refine to specific errors
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def _request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict:
        """Make a signed async request to Coinbase with automatic retries."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/{path.lstrip('/')}"
        
        # Use full path for JWT URI claim
        jwt_path = f"/api/v3/brokerage/{path.lstrip('/')}"
        token = self._generate_jwt(method, jwt_path)
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        async with session.request(method, url, json=data, headers=headers) as response:
            if response.status >= 400:
                text = await response.text()
                raise Exception(f"Coinbase API Error {response.status}: {text}")
            return await response.json()

    async def execute(self, order: OrderRequest) -> ExecutionResult:
        """Execute a real order (Async)."""
        self.logger.info(
            f"🔴 LIVE ORDER: {order.side.value.upper()} "
            f"{order.quantity} {order.symbol} @ expected {order.expected_price}"
        )

        try:
            order_config = self._build_order_config(order)
            
            # Client order ID for idempotency
            client_order_id = f"v4_{int(time.time())}_{order.symbol.replace('-','')}"
            
            payload = {
                "client_order_id": client_order_id,
                "product_id": order.symbol,
                "side": order.side.value.upper(),
                "order_configuration": order_config
            }

            response = await self._request("POST", "orders", payload)
            
            # Coinbase Advanced Trade returns a wrap around the success/error
            result = self._parse_response(response, order)
            self.record_execution(result)

            if result.is_success:
                self.logger.info(
                    f"✅ LIVE FILL: {result.external_id} @ {result.fill_price} "
                    f"(slippage: {result.slippage_pct:.4f}%, fee: ${result.fee})"
                )
            else:
                self.logger.error(f"❌ LIVE ORDER FAILED: {result.message}")

            return result

        except Exception as e:
            self.logger.error(f"❌ LIVE ORDER EXCEPTION: {e}")
            result = ExecutionResult.failed(
                order_request=order,
                reason=str(e),
                details={"exception_type": type(e).__name__}
            )
            self.record_execution(result)
            return result

    def _build_order_config(self, order: OrderRequest) -> Dict[str, Any]:
        """Build Coinbase order configuration."""
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                quote_size = order.expected_price * order.quantity
                return {"market_market_ioc": {"quote_size": str(quote_size)}}
            else:
                return {"market_market_ioc": {"base_size": str(order.quantity)}}
        else:
            limit_price = order.limit_price or order.expected_price
            return {
                "limit_limit_gtc": {
                    "base_size": str(order.quantity),
                    "limit_price": str(limit_price),
                    "post_only": False
                }
            }

    def _parse_response(self, response: Dict[str, Any], order: OrderRequest) -> ExecutionResult:
        """Parse Coinbase response."""
        # Success check
        if not response.get('success', False):
            err = response.get('error_response', {})
            msg = err.get('message', 'Unknown error')
            return ExecutionResult.failed(order, reason=msg, details=response)

        success_data = response.get('success_response', {})
        order_id = success_data.get('order_id')
        
        # In V3, the immediate response might not have fill details
        # We assume pending unless filled_size is returned
        return ExecutionResult(
            status=ExecutionStatus.PENDING,
            external_id=order_id,
            message="Order placed, awaiting fill",
            order_request=order,
            details=response
        )

    async def get_balance(self, currency: str = "USD") -> Decimal:
        """Get available balance (Async)."""
        try:
            response = await self._request("GET", "accounts")
            accounts = response.get('accounts', [])
            for acct in accounts:
                if acct.get('currency') == currency:
                    return Decimal(str(acct.get('available_balance', {}).get('value', '0')))
            return Decimal("0")
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return Decimal("0")

    async def cancel_order(self, external_id: str) -> ExecutionResult:
        """Cancel an open order (Async)."""
        try:
            payload = {"order_ids": [external_id]}
            response = await self._request("POST", "orders/batch_cancel", payload)
            
            # Check results
            results = response.get('results', [])
            if results and results[0].get('success'):
                return ExecutionResult(
                    status=ExecutionStatus.CANCELLED,
                    external_id=external_id,
                    message="Order cancelled"
                )
            return ExecutionResult.failed(None, reason="Cancel failed", details=response)
        except Exception as e:
            return ExecutionResult.failed(None, reason=str(e))

    async def close(self):
        """Close the connection session."""
        if self._own_session and self.session:
            await self.session.close()

    # =========================================================================
    # POSITION TRACKING METHODS - Required for position reconciliation
    # =========================================================================

    def has_position(self, symbol: str) -> bool:
        """
        Check if we have an open position in the given symbol.

        For Coinbase, we check the balance of the base currency.
        """
        # This is a sync wrapper that checks cached balance
        # For accurate live data, use async get_balance
        base_currency = symbol.split("-")[0]
        # Note: This returns False by default since we can't make async calls
        # Position reconciliation should use async methods for live trading
        self.logger.warning(
            f"has_position called synchronously for {symbol}. "
            "For accurate results, use async get_balance."
        )
        return False  # Conservative default - assumes no position

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position details for a symbol.

        For live trading, this requires an async API call.
        Returns None in sync context.
        """
        base_currency = symbol.split("-")[0]
        self.logger.warning(
            f"get_position called synchronously for {symbol}. "
            "For accurate results, use async get_live_position."
        )
        return None  # Conservative default - no position data

    def get_position_size(self, symbol: str) -> Decimal:
        """
        Get the position size for a symbol.

        For live trading, this requires an async API call.
        Returns 0 in sync context.
        """
        base_currency = symbol.split("-")[0]
        self.logger.warning(
            f"get_position_size called synchronously for {symbol}. "
            "For accurate results, use async get_balance."
        )
        return Decimal("0")  # Conservative default

    async def get_live_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position details from Coinbase (async).

        This is the preferred method for live trading.
        """
        base_currency = symbol.split("-")[0]
        try:
            balance = await self.get_balance(base_currency)
            if balance > Decimal("0"):
                return {
                    "symbol": symbol,
                    "quantity": balance,
                    "side": "buy"  # Coinbase spot = long only
                }
            return None
        except Exception as e:
            self.logger.error(f"Failed to get live position for {symbol}: {e}")
            return None

    async def has_live_position(self, symbol: str) -> bool:
        """
        Check if we have a position (async).

        This is the preferred method for live trading.
        """
        position = await self.get_live_position(symbol)
        return position is not None

import os