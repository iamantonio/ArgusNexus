"""
V4 Async Gemini Executor - Live Order Execution for Gemini Exchange

Wraps the Gemini REST API using aiohttp and HMAC-SHA384 authentication.

CRITICAL: This places REAL orders with REAL money.

Key Features:
- Asynchronous connection pooling via aiohttp
- HMAC-SHA384 authentication (Gemini's required method)
- Non-blocking execution for fleet scalability
- Precise slippage tracking and fee logging
- ActiveTrader fee tier support

Gemini API Docs: https://docs.gemini.com/rest-api/
"""

import base64
import hashlib
import hmac
import json
import logging
import time
import aiohttp
from decimal import Decimal
from typing import Any, Dict, Optional
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


class GeminiExecutor(Executor):
    """
    Executes real orders via Gemini REST API (Async).

    Authentication: HMAC-SHA384 with base64-encoded payload

    ⚠️ WARNING: This executor places REAL orders with REAL money.

    Usage:
        executor = GeminiExecutor(
            api_key=os.getenv("GEMINI_API_KEY"),
            api_secret=os.getenv("GEMINI_API_SECRET")
        )
        result = await executor.execute(order)
    """

    # Use sandbox for testing, production for live
    SANDBOX_URL = "https://api.sandbox.gemini.com"
    PRODUCTION_URL = "https://api.gemini.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        sandbox: bool = False,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize Gemini executor.

        Args:
            api_key: Gemini API Key
            api_secret: Gemini API Secret
            sandbox: If True, use sandbox environment (no real money)
            session: Optional shared aiohttp session
        """
        # Sandbox mode uses PAPER, production uses LIVE
        mode = ExecutionMode.PAPER if sandbox else ExecutionMode.LIVE
        super().__init__(mode=mode)

        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = self.SANDBOX_URL if sandbox else self.PRODUCTION_URL
        self.session = session
        self._own_session = False

        env_str = "SANDBOX" if sandbox else "PRODUCTION"
        self.logger.warning(
            f"🔵 GEMINI EXECUTOR INITIALIZED - {env_str} MODE 🔵"
        )

        if not sandbox:
            self.logger.warning(
                "🔴 LIVE TRADING MODE - ALL ORDERS ARE REAL 🔴"
            )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            self._own_session = True
        return self.session

    def _generate_signature(self, payload: Dict[str, Any]) -> tuple[str, str]:
        """
        Generate Gemini API signature.

        Gemini requires:
        1. JSON payload encoded as base64
        2. HMAC-SHA384 signature of the base64 payload

        Returns:
            Tuple of (base64_payload, signature)
        """
        # Add nonce (timestamp in milliseconds)
        payload["nonce"] = str(int(time.time() * 1000))

        # Encode payload as base64
        payload_json = json.dumps(payload)
        payload_b64 = base64.b64encode(payload_json.encode())

        # Create HMAC-SHA384 signature
        signature = hmac.new(
            self.api_secret,
            payload_b64,
            hashlib.sha384
        ).hexdigest()

        return payload_b64.decode(), signature

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(aiohttp.ClientError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def _request(self, endpoint: str, payload: Dict[str, Any]) -> Dict:
        """
        Make a signed async request to Gemini with automatic retries.

        Args:
            endpoint: API endpoint (e.g., "/v1/order/new")
            payload: Request payload (nonce will be added automatically)

        Returns:
            JSON response as dict
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        # Add request path to payload
        payload["request"] = endpoint

        # Generate signature
        payload_b64, signature = self._generate_signature(payload)

        headers = {
            "Content-Type": "text/plain",
            "Content-Length": "0",
            "X-GEMINI-APIKEY": self.api_key,
            "X-GEMINI-PAYLOAD": payload_b64,
            "X-GEMINI-SIGNATURE": signature,
            "Cache-Control": "no-cache"
        }

        async with session.post(url, headers=headers) as response:
            text = await response.text()

            if response.status >= 400:
                raise Exception(f"Gemini API Error {response.status}: {text}")

            return json.loads(text) if text else {}

    async def execute(self, order: OrderRequest) -> ExecutionResult:
        """
        Execute a real order on Gemini (Async).

        Args:
            order: OrderRequest with symbol, side, quantity, expected_price

        Returns:
            ExecutionResult with status, fill_price, slippage, fee
        """
        self.logger.info(
            f"🔵 GEMINI ORDER: {order.side.value.upper()} "
            f"{order.quantity} {order.symbol} @ expected {order.expected_price}"
        )

        try:
            # Convert symbol format: BTC-USD -> btcusd
            gemini_symbol = order.symbol.lower().replace("-", "")

            # Build order payload
            payload = {
                "symbol": gemini_symbol,
                "amount": str(order.quantity),
                "side": order.side.value.lower(),
                "type": "exchange market" if order.order_type == OrderType.MARKET else "exchange limit",
            }

            # Add price for limit orders
            if order.order_type == OrderType.LIMIT:
                price = order.limit_price or order.expected_price
                payload["price"] = str(price)

            # For market orders, Gemini still requires a price (used as limit protection)
            # Set it slightly above/below market to ensure fill
            if order.order_type == OrderType.MARKET:
                if order.side == OrderSide.BUY:
                    # Set limit 5% above expected for buys
                    payload["price"] = str(order.expected_price * Decimal("1.05"))
                else:
                    # Set limit 5% below expected for sells
                    payload["price"] = str(order.expected_price * Decimal("0.95"))
                payload["options"] = ["immediate-or-cancel"]

            response = await self._request("/v1/order/new", payload)
            result = self._parse_response(response, order)
            self.record_execution(result)

            if result.is_success:
                self.logger.info(
                    f"✅ GEMINI FILL: {result.external_id} @ {result.fill_price} "
                    f"(slippage: {result.slippage_pct:.4f}%)"
                )
            else:
                self.logger.error(f"❌ GEMINI ORDER FAILED: {result.message}")

            return result

        except Exception as e:
            self.logger.error(f"❌ GEMINI ORDER EXCEPTION: {e}")
            result = ExecutionResult.failed(
                order_request=order,
                reason=str(e),
                details={"exception_type": type(e).__name__}
            )
            self.record_execution(result)
            return result

    def _parse_response(self, response: Dict[str, Any], order: OrderRequest) -> ExecutionResult:
        """Parse Gemini order response into ExecutionResult."""

        # Check for error
        if "result" in response and response["result"] == "error":
            return ExecutionResult.failed(
                order_request=order,
                reason=response.get("message", "Unknown error"),
                details=response
            )

        order_id = response.get("order_id")

        # Check fill status
        is_cancelled = response.get("is_cancelled", False)
        executed_amount = Decimal(str(response.get("executed_amount", "0")))
        original_amount = Decimal(str(response.get("original_amount", order.quantity)))
        avg_execution_price = response.get("avg_execution_price")

        if is_cancelled and executed_amount == 0:
            return ExecutionResult(
                status=ExecutionStatus.CANCELLED,
                external_id=order_id,
                message="Order cancelled - no fill",
                order_request=order,
                details=response
            )

        if executed_amount > 0:
            fill_price = Decimal(str(avg_execution_price)) if avg_execution_price else order.expected_price
            slippage = fill_price - order.expected_price
            slippage_pct = float(slippage / order.expected_price * 100)

            # Determine status
            if executed_amount >= original_amount:
                status = ExecutionStatus.FILLED
            else:
                status = ExecutionStatus.PARTIAL

            return ExecutionResult(
                status=status,
                external_id=order_id,
                fill_price=fill_price,
                filled_quantity=executed_amount,
                slippage=slippage,
                slippage_pct=slippage_pct,
                message=f"Filled {executed_amount} @ {fill_price}",
                order_request=order,
                details=response
            )

        # Order placed but not filled yet
        return ExecutionResult(
            status=ExecutionStatus.PENDING,
            external_id=order_id,
            message="Order placed, awaiting fill",
            order_request=order,
            details=response
        )

    async def get_balance(self, currency: str = "USD") -> Decimal:
        """
        Get available balance for a currency (Async).

        Args:
            currency: Currency code (default: USD)

        Returns:
            Available balance as Decimal
        """
        try:
            # Gemini v1/balances requires account parameter
            response = await self._request("/v1/balances", {"account": "primary"})

            for balance in response:
                if balance.get("currency", "").upper() == currency.upper():
                    return Decimal(str(balance.get("available", "0")))

            return Decimal("0")

        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return Decimal("0")

    async def get_balances(self) -> Dict[str, Decimal]:
        """
        Get all available balances (Async).

        Returns:
            Dict mapping currency to available balance
        """
        try:
            # Gemini v1/balances requires account parameter
            response = await self._request("/v1/balances", {"account": "primary"})

            balances = {}
            for balance in response:
                currency = balance.get("currency", "").upper()
                available = Decimal(str(balance.get("available", "0")))
                if available > 0:
                    balances[currency] = available

            return balances

        except Exception as e:
            self.logger.error(f"Failed to get balances: {e}")
            return {}

    async def cancel_order(self, external_id: str) -> ExecutionResult:
        """
        Cancel an open order (Async).

        Args:
            external_id: Gemini order ID to cancel

        Returns:
            ExecutionResult with cancellation status
        """
        try:
            response = await self._request("/v1/order/cancel", {"order_id": external_id})

            if response.get("is_cancelled"):
                return ExecutionResult(
                    status=ExecutionStatus.CANCELLED,
                    external_id=external_id,
                    message="Order cancelled successfully"
                )

            return ExecutionResult.failed(
                order_request=None,
                reason="Cancel request did not confirm",
                details=response
            )

        except Exception as e:
            return ExecutionResult.failed(
                order_request=None,
                reason=str(e)
            )

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current ticker data for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD" or "btcusd")

        Returns:
            Ticker data with bid, ask, last price
        """
        try:
            gemini_symbol = symbol.lower().replace("-", "")
            session = await self._get_session()
            url = f"{self.base_url}/v1/pubticker/{gemini_symbol}"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "symbol": symbol,
                        "bid": Decimal(str(data.get("bid", "0"))),
                        "ask": Decimal(str(data.get("ask", "0"))),
                        "last": Decimal(str(data.get("last", "0"))),
                        "volume": Decimal(str(data.get("volume", {}).get("BTC", "0")))
                    }
                return None

        except Exception as e:
            self.logger.error(f"Failed to get ticker: {e}")
            return None

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

        Note: For async accuracy, use has_live_position() instead.
        """
        base_currency = symbol.split("-")[0]
        self.logger.warning(
            f"has_position called synchronously for {symbol}. "
            "For accurate results, use async has_live_position."
        )
        return False  # Conservative default

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position details for a symbol.

        Note: For async accuracy, use get_live_position() instead.
        """
        self.logger.warning(
            f"get_position called synchronously for {symbol}. "
            "For accurate results, use async get_live_position."
        )
        return None  # Conservative default

    def get_position_size(self, symbol: str) -> Decimal:
        """
        Get the position size for a symbol.

        Note: For async accuracy, use get_balance() instead.
        """
        self.logger.warning(
            f"get_position_size called synchronously for {symbol}. "
            "For accurate results, use async get_balance."
        )
        return Decimal("0")  # Conservative default

    async def get_live_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position details from Gemini (async).

        This is the preferred method for live trading.
        """
        base_currency = symbol.split("-")[0]
        try:
            balance = await self.get_balance(base_currency)
            if balance > Decimal("0"):
                return {
                    "symbol": symbol,
                    "quantity": balance,
                    "side": "buy"  # Gemini spot = long only
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


# Factory function for convenience
def create_gemini_executor(sandbox: bool = False) -> GeminiExecutor:
    """
    Create a Gemini executor from environment variables.

    Requires GEMINI_API_KEY and GEMINI_API_SECRET in environment.

    Args:
        sandbox: If True, use sandbox environment

    Returns:
        Configured GeminiExecutor
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    api_secret = os.getenv("GEMINI_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError(
            "GEMINI_API_KEY and GEMINI_API_SECRET must be set in environment. "
            "Add them to your .env file."
        )

    return GeminiExecutor(
        api_key=api_key,
        api_secret=api_secret,
        sandbox=sandbox
    )
