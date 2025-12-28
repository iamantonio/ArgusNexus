"""
V4 Paper Executor - Realistic Simulation with Slippage

The critical upgrade from V3: DON'T fill at the expected price!

V3 Problem:
- Paper trades filled at exact expected price
- Backtests looked perfect
- Live trades bled from slippage
- No way to know until real money was lost

V4 Solution:
- Simulate realistic slippage on EVERY trade
- Track slippage metrics for monitoring
- If simulated slippage exceeds model, strategy is suspect

Slippage Model:
- Base spread: 0.01% (bid-ask spread)
- Random noise: 0-0.03% (market microstructure)
- Size impact: 0.01% per $10k notional (larger orders move price)
- Fee: 0.4% (Coinbase maker fee)

SHORT Simulation (v6.1):
- Simulated margin trading for backtesting shorts
- Collateral requirement: 100% of notional (conservative)
- P&L: (entry_price - exit_price) * quantity
- Tracks entry price for proper P&L calculation
"""

import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from .base import Executor


@dataclass
class ShortPosition:
    """Tracks a simulated short position."""
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    collateral: Decimal  # USD locked as margin
    opened_at: datetime


from .schema import (
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    OrderSide,
    OrderType,
)


class PaperExecutor(Executor):
    """
    Simulates order execution with realistic slippage.

    Key Features:
    - Realistic slippage model (spread + noise + size impact)
    - Fee simulation (configurable, default 0.4%)
    - Balance tracking
    - Execution history for analysis

    Usage:
        executor = PaperExecutor(
            starting_balance=Decimal("10000"),
            base_slippage_pct=Decimal("0.01"),  # 0.01% base spread
            fee_rate=Decimal("0.004")           # 0.4% maker fee
        )

        order = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            expected_price=Decimal("100000")
        )

        result = executor.execute(order)
        print(f"Filled at {result.fill_price}, slippage: {result.slippage_pct}%")
    """

    def __init__(
        self,
        starting_balance: Decimal = Decimal("10000"),
        base_slippage_pct: Decimal = Decimal("0.01"),   # 0.01% base
        noise_slippage_pct: Decimal = Decimal("0.03"),  # 0-0.03% random
        size_impact_per_10k: Decimal = Decimal("0.01"), # 0.01% per $10k
        fee_rate: Decimal = Decimal("0.004"),           # 0.4% fee
    ):
        """
        Initialize paper executor.

        Args:
            starting_balance: Initial balance in USD
            base_slippage_pct: Fixed spread component (%)
            noise_slippage_pct: Max random slippage (%)
            size_impact_per_10k: Slippage per $10k notional (%)
            fee_rate: Transaction fee rate (0.004 = 0.4%)
        """
        super().__init__(mode=ExecutionMode.PAPER)

        self.base_slippage_pct = base_slippage_pct
        self.noise_slippage_pct = noise_slippage_pct
        self.size_impact_per_10k = size_impact_per_10k
        self.fee_rate = fee_rate

        # Virtual balances (for long positions)
        self._balances: Dict[str, Decimal] = {"USD": starting_balance}
        self._starting_balance = starting_balance

        # SHORT position tracking (v6.1)
        # Maps symbol -> ShortPosition
        self._short_positions: Dict[str, ShortPosition] = {}
        self._realized_short_pnl: Decimal = Decimal("0")

        self.logger.info(
            f"Paper executor initialized: "
            f"balance=${starting_balance}, "
            f"base_slippage={base_slippage_pct}%, "
            f"fee={fee_rate * 100}% "
            f"[SHORT simulation ENABLED]"
        )

    async def execute(self, order: OrderRequest) -> ExecutionResult:
        """
        Execute a simulated order with realistic slippage (Async).

        Supports both LONG and SHORT positions:
        - BUY: Opens long OR closes short
        - SELL: Closes long OR opens short (simulated margin)

        Slippage calculation:
        1. Base spread: fixed bid-ask spread
        2. Random noise: market microstructure
        3. Size impact: larger orders move price more

        Args:
            order: OrderRequest with trade details

        Returns:
            ExecutionResult with fill_price, slippage, fee
        """
        base_currency = order.symbol.split("-")[0]
        has_long = self._balances.get(base_currency, Decimal("0")) > Decimal("0")
        has_short = order.symbol in self._short_positions

        self.logger.info(
            f"Paper order: {order.side.value.upper()} "
            f"{order.quantity} {order.symbol} @ expected {order.expected_price}"
        )

        # Determine order type
        if order.side == OrderSide.BUY:
            if has_short:
                # Closing a short position
                return await self._close_short(order)
            else:
                # Opening a long position
                return await self._open_long(order)
        else:  # SELL
            if has_long:
                # Closing a long position
                return await self._close_long(order)
            else:
                # Opening a short position (simulated margin)
                return await self._open_short(order)

    async def _open_long(self, order: OrderRequest) -> ExecutionResult:
        """Open a long position (standard BUY)."""
        required = order.notional_value * (1 + self.fee_rate)
        if self._balances.get("USD", Decimal("0")) < required:
            result = ExecutionResult.failed(
                order_request=order,
                reason=f"Insufficient balance: need ${required:.2f}, "
                       f"have ${self._balances.get('USD', 0):.2f}"
            )
            self.record_execution(result)
            return result

        fill_price = self._calculate_fill_price(order)
        notional = fill_price * order.quantity
        fee = notional * self.fee_rate
        external_id = f"paper_{uuid.uuid4().hex[:12]}"
        base_currency = order.symbol.split("-")[0]

        self._balances["USD"] -= (notional + fee)
        self._balances[base_currency] = self._balances.get(
            base_currency, Decimal("0")
        ) + order.quantity

        result = ExecutionResult.filled(
            order_request=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            fee=fee,
            external_id=external_id,
            details={
                "position_type": "LONG",
                "action": "OPEN",
                "balance_after": {
                    "USD": float(self._balances.get("USD", 0)),
                    base_currency: float(self._balances.get(base_currency, 0))
                }
            }
        )
        self.record_execution(result)
        self.logger.info(
            f"Paper fill [LONG OPEN]: {external_id} @ {fill_price} "
            f"(slippage: {result.slippage_pct:.4f}%, fee: ${fee:.4f})"
        )
        return result

    async def _close_long(self, order: OrderRequest) -> ExecutionResult:
        """Close a long position (standard SELL)."""
        base_currency = order.symbol.split("-")[0]
        current_base = self._balances.get(base_currency, Decimal("0"))

        if current_base < order.quantity:
            result = ExecutionResult.failed(
                order_request=order,
                reason=f"Insufficient {base_currency}: "
                       f"need {order.quantity}, have {current_base}"
            )
            self.record_execution(result)
            return result

        fill_price = self._calculate_fill_price(order)
        notional = fill_price * order.quantity
        fee = notional * self.fee_rate
        external_id = f"paper_{uuid.uuid4().hex[:12]}"

        self._balances[base_currency] -= order.quantity
        self._balances["USD"] += (notional - fee)

        result = ExecutionResult.filled(
            order_request=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            fee=fee,
            external_id=external_id,
            details={
                "position_type": "LONG",
                "action": "CLOSE",
                "balance_after": {
                    "USD": float(self._balances.get("USD", 0)),
                    base_currency: float(self._balances.get(base_currency, 0))
                }
            }
        )
        self.record_execution(result)
        self.logger.info(
            f"Paper fill [LONG CLOSE]: {external_id} @ {fill_price} "
            f"(slippage: {result.slippage_pct:.4f}%, fee: ${fee:.4f})"
        )
        return result

    async def _open_short(self, order: OrderRequest) -> ExecutionResult:
        """
        Open a short position (simulated margin trading).

        Simulates:
        - Borrowing the asset
        - Selling it at current price
        - Locking USD as collateral (100% of notional)
        """
        # Check if we already have a short position in this symbol
        if order.symbol in self._short_positions:
            result = ExecutionResult.failed(
                order_request=order,
                reason=f"Already have short position in {order.symbol}"
            )
            self.record_execution(result)
            return result

        # Calculate collateral requirement (100% of notional + fees)
        fill_price = self._calculate_fill_price(order)
        notional = fill_price * order.quantity
        fee = notional * self.fee_rate
        collateral_required = notional + fee  # 100% margin

        if self._balances.get("USD", Decimal("0")) < collateral_required:
            result = ExecutionResult.failed(
                order_request=order,
                reason=f"Insufficient collateral for short: need ${collateral_required:.2f}, "
                       f"have ${self._balances.get('USD', 0):.2f}"
            )
            self.record_execution(result)
            return result

        external_id = f"paper_short_{uuid.uuid4().hex[:12]}"

        # Lock collateral
        self._balances["USD"] -= collateral_required

        # Create short position
        self._short_positions[order.symbol] = ShortPosition(
            symbol=order.symbol,
            quantity=order.quantity,
            entry_price=fill_price,
            collateral=collateral_required,
            opened_at=datetime.utcnow()
        )

        result = ExecutionResult.filled(
            order_request=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            fee=fee,
            external_id=external_id,
            details={
                "position_type": "SHORT",
                "action": "OPEN",
                "collateral_locked": float(collateral_required),
                "entry_price": float(fill_price),
                "balance_after": {
                    "USD": float(self._balances.get("USD", 0))
                }
            }
        )
        self.record_execution(result)
        self.logger.info(
            f"Paper fill [SHORT OPEN]: {external_id} @ {fill_price} "
            f"(collateral: ${collateral_required:.2f}, fee: ${fee:.4f})"
        )
        return result

    async def _close_short(self, order: OrderRequest) -> ExecutionResult:
        """
        Close a short position (buy to cover).

        Calculates realized P&L:
        - Profit = (entry_price - exit_price) * quantity - fees
        """
        if order.symbol not in self._short_positions:
            result = ExecutionResult.failed(
                order_request=order,
                reason=f"No short position to close in {order.symbol}"
            )
            self.record_execution(result)
            return result

        short_pos = self._short_positions[order.symbol]

        # Calculate fill price and costs
        fill_price = self._calculate_fill_price(order)
        notional = fill_price * order.quantity
        fee = notional * self.fee_rate
        external_id = f"paper_cover_{uuid.uuid4().hex[:12]}"

        # Calculate P&L: (entry - exit) * quantity
        # Shorts profit when price goes DOWN
        gross_pnl = (short_pos.entry_price - fill_price) * short_pos.quantity
        net_pnl = gross_pnl - fee  # Subtract exit fee (entry fee already in collateral)

        # Return collateral + P&L
        returned_amount = short_pos.collateral + net_pnl
        self._balances["USD"] += returned_amount
        self._realized_short_pnl += net_pnl

        # Remove short position
        del self._short_positions[order.symbol]

        result = ExecutionResult.filled(
            order_request=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            fee=fee,
            external_id=external_id,
            details={
                "position_type": "SHORT",
                "action": "CLOSE",
                "entry_price": float(short_pos.entry_price),
                "exit_price": float(fill_price),
                "gross_pnl": float(gross_pnl),
                "net_pnl": float(net_pnl),
                "collateral_returned": float(short_pos.collateral),
                "total_returned": float(returned_amount),
                "balance_after": {
                    "USD": float(self._balances.get("USD", 0))
                }
            }
        )
        self.record_execution(result)

        pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
        self.logger.info(
            f"Paper fill [SHORT CLOSE]: {external_id} @ {fill_price} "
            f"(entry: {short_pos.entry_price}, P&L: {pnl_str})"
        )
        return result

    def _calculate_fill_price(self, order: OrderRequest) -> Decimal:
        """
        Calculate realistic fill price with slippage.

        Components:
        1. Base spread: Always present (bid-ask spread)
        2. Random noise: Market microstructure (0 to noise_max)
        3. Size impact: Larger orders get worse fills

        For BUYS: fill_price > expected (we pay more)
        For SELLS: fill_price < expected (we receive less)
        """
        expected = order.expected_price
        notional = float(order.notional_value)

        # Component 1: Base spread
        base_pct = float(self.base_slippage_pct) / 100

        # Component 2: Random noise (uniform distribution)
        noise_pct = random.uniform(0, float(self.noise_slippage_pct)) / 100

        # Component 3: Size impact (linear with notional value)
        # Impact per $10k notional
        size_impact_pct = (notional / 10000) * float(self.size_impact_per_10k) / 100

        # Total slippage percentage
        total_slippage_pct = base_pct + noise_pct + size_impact_pct

        # Apply slippage based on side
        if order.side == OrderSide.BUY:
            # Buys: we pay MORE than expected
            fill_price = expected * (1 + Decimal(str(total_slippage_pct)))
        else:
            # Sells: we receive LESS than expected
            fill_price = expected * (1 - Decimal(str(total_slippage_pct)))

        # Round to 2 decimal places (standard for USD)
        return fill_price.quantize(Decimal("0.01"))

    async def get_balance(self, currency: str = "USD") -> Decimal:
        """Get available balance for a currency (Async)."""
        return self._balances.get(currency, Decimal("0"))

    async def set_balance(self, currency: str, amount: Decimal) -> None:
        """
        Set balance for a currency (Async).

        Used for state synchronization when loading persisted portfolio state.
        Does NOT record an execution (this is a state sync, not a trade).

        Args:
            currency: Currency code (e.g., "USD", "BTC")
            amount: New balance amount
        """
        old_balance = self._balances.get(currency, Decimal("0"))
        self._balances[currency] = amount
        self.logger.debug(
            f"Balance sync: {currency} {float(old_balance):.6f} -> {float(amount):.6f}"
        )

    def has_position(self, symbol: str) -> bool:
        """
        Check if we have an open position in the given symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            True if long or short position exists
        """
        base_currency = symbol.split("-")[0]
        has_long = self._balances.get(base_currency, Decimal("0")) > Decimal("0")
        has_short = symbol in self._short_positions
        return has_long or has_short

    def has_short_position(self, symbol: str) -> bool:
        """Check if we have a short position in this symbol."""
        return symbol in self._short_positions

    def get_short_position(self, symbol: str) -> Optional[ShortPosition]:
        """Get short position details if exists."""
        return self._short_positions.get(symbol)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position details for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            Dict with position details if position exists, None otherwise
            {
                "symbol": "BTC-USD",
                "quantity": Decimal("0.1"),
                "side": "buy" | "sell",  # sell = short position
                "entry_price": Decimal (for shorts only)
            }
        """
        base_currency = symbol.split("-")[0]

        # Check for short position first
        if symbol in self._short_positions:
            short_pos = self._short_positions[symbol]
            return {
                "symbol": symbol,
                "quantity": short_pos.quantity,
                "side": "sell",  # Short position
                "entry_price": short_pos.entry_price,
                "collateral": short_pos.collateral,
                "opened_at": short_pos.opened_at.isoformat()
            }

        # Check for long position
        quantity = self._balances.get(base_currency, Decimal("0"))
        if quantity > Decimal("0"):
            return {
                "symbol": symbol,
                "quantity": quantity,
                "side": "buy"  # Long position
            }

        return None

    def get_position_size(self, symbol: str) -> Decimal:
        """
        Get the position size for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            Position quantity (positive for long, negative for short, 0 if flat)
        """
        # Check for short position first
        if symbol in self._short_positions:
            return -self._short_positions[symbol].quantity  # Negative for shorts

        base_currency = symbol.split("-")[0]
        return self._balances.get(base_currency, Decimal("0"))

    def get_all_balances(self) -> Dict[str, Decimal]:
        """Get all balances (long positions only)."""
        return self._balances.copy()

    def get_all_short_positions(self) -> Dict[str, ShortPosition]:
        """Get all open short positions."""
        return self._short_positions.copy()

    async def cancel_order(self, external_id: str) -> ExecutionResult:
        """
        Cancel not supported - paper orders fill immediately (Async).

        Returns:
            ExecutionResult with FAILED status
        """
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            message="Cancel not supported - paper orders fill immediately",
            external_id=external_id
        )

    def reset_balance(self, amount: Optional[Decimal] = None) -> None:
        """
        Reset balance to starting or specified amount.

        Also clears all short positions.

        Args:
            amount: New balance (uses starting_balance if None)
        """
        new_balance = amount or self._starting_balance
        self._balances = {"USD": new_balance}
        self._short_positions = {}
        self._realized_short_pnl = Decimal("0")
        self.logger.info(f"Balance reset to ${new_balance} (shorts cleared)")

    def get_pnl(self) -> Decimal:
        """
        Calculate paper trading P&L.

        Simple: current USD balance - starting balance.
        Does not account for unrealized gains in held assets or open shorts.

        Returns:
            Realized P&L in USD
        """
        return self._balances.get("USD", Decimal("0")) - self._starting_balance

    def get_realized_short_pnl(self) -> Decimal:
        """Get total realized P&L from closed short positions."""
        return self._realized_short_pnl

    def get_total_value(self, prices: Dict[str, Decimal]) -> Decimal:
        """
        Calculate total portfolio value including unrealized short P&L.

        Args:
            prices: Current prices for held assets (e.g., {"BTC": 100000})

        Returns:
            Total portfolio value in USD
        """
        total = self._balances.get("USD", Decimal("0"))

        # Add long position values
        for currency, balance in self._balances.items():
            if currency != "USD" and balance > 0:
                if currency in prices:
                    total += balance * prices[currency]
                else:
                    self.logger.warning(
                        f"No price for {currency}, excluding from total value"
                    )

        # Add unrealized short P&L
        # Short P&L = (entry_price - current_price) * quantity
        for symbol, short_pos in self._short_positions.items():
            base_currency = symbol.split("-")[0]
            if base_currency in prices:
                current_price = prices[base_currency]
                unrealized_pnl = (short_pos.entry_price - current_price) * short_pos.quantity
                # Collateral is already subtracted from USD, add back with P&L
                total += short_pos.collateral + unrealized_pnl
            else:
                # If no price, just add collateral (assume breakeven)
                total += short_pos.collateral
                self.logger.warning(
                    f"No price for {symbol} short, assuming breakeven"
                )

        return total

    def get_unrealized_short_pnl(self, prices: Dict[str, Decimal]) -> Decimal:
        """
        Calculate unrealized P&L from open short positions.

        Args:
            prices: Current prices (e.g., {"BTC": 100000})

        Returns:
            Unrealized P&L from shorts (positive = profit, negative = loss)
        """
        total_pnl = Decimal("0")
        for symbol, short_pos in self._short_positions.items():
            base_currency = symbol.split("-")[0]
            if base_currency in prices:
                current_price = prices[base_currency]
                pnl = (short_pos.entry_price - current_price) * short_pos.quantity
                total_pnl += pnl
        return total_pnl
