"""
V4 Executor Base Class - The Immutable Safety Lock

The execution mode (PAPER vs LIVE) is set ONCE at initialization
and cannot be changed. This is the safety lock.

No "accidental" switching from paper to live mid-session.
No runtime configuration that could flip real money on.
"""

import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .schema import (
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    SlippageStats,
)


class ExecutionModeImmutableError(Exception):
    """Raised when attempting to change execution mode after initialization."""
    pass


class Executor(ABC):
    """
    Abstract base class for order executors.

    The Safety Lock:
    - ExecutionMode is set ONCE at __init__ and is IMMUTABLE
    - Attempting to change mode raises ExecutionModeImmutableError
    - Mode is exposed as read-only property

    Implementations:
    - PaperExecutor: Simulates fills with realistic slippage
    - LiveExecutor: Real orders via Coinbase API

    Usage:
        # Create executor with mode locked
        executor = PaperExecutor(mode=ExecutionMode.PAPER, ...)

        # Execute order
        result = executor.execute(order_request)

        # Check slippage
        if result.slippage_pct > 0.1:
            logger.warning("High slippage detected!")

        # Get aggregate stats
        stats = executor.get_slippage_stats()
    """

    def __init__(self, mode: ExecutionMode):
        """
        Initialize executor with IMMUTABLE mode.

        Args:
            mode: PAPER or LIVE - cannot be changed after init

        Raises:
            ValueError: If mode is not a valid ExecutionMode
        """
        if not isinstance(mode, ExecutionMode):
            raise ValueError(f"mode must be ExecutionMode, got {type(mode)}")

        self._mode = mode
        self._mode_locked = True  # Flag to prevent any modification
        self._execution_history: List[ExecutionResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Log mode at initialization for audit trail
        mode_emoji = "📋" if mode == ExecutionMode.PAPER else "💰"
        self.logger.info(f"{mode_emoji} Executor initialized in {mode.value.upper()} mode")

        if mode == ExecutionMode.LIVE:
            self.logger.warning(
                "⚠️ LIVE TRADING MODE - Real money at risk! "
                "Double-check all configurations before proceeding."
            )

    @property
    def mode(self) -> ExecutionMode:
        """
        Get execution mode (read-only).

        Returns:
            ExecutionMode: PAPER or LIVE
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        """
        Prevent mode modification.

        Raises:
            ExecutionModeImmutableError: Always - mode cannot be changed
        """
        raise ExecutionModeImmutableError(
            f"Execution mode is IMMUTABLE. "
            f"Current mode: {self._mode.value}. "
            f"Create a new executor instance to change modes."
        )

    @property
    def is_paper(self) -> bool:
        """True if in paper trading mode."""
        return self._mode == ExecutionMode.PAPER

    @property
    def is_live(self) -> bool:
        """True if in live trading mode."""
        return self._mode == ExecutionMode.LIVE

    @abstractmethod
    async def execute(self, order: OrderRequest) -> ExecutionResult:
        """
        Execute an order (Async).

        This is THE core method. Implementations must:
        1. Place the order (simulated or real)
        2. Calculate slippage (fill_price - expected_price)
        3. Return ExecutionResult with full details
        4. Record result in execution history

        Args:
            order: OrderRequest with symbol, side, quantity, expected_price

        Returns:
            ExecutionResult with status, fill_price, slippage, fee, external_id
        """
        pass

    @abstractmethod
    async def get_balance(self, currency: str = "USD") -> Decimal:
        """
        Get available balance for a currency (Async).

        Args:
            currency: Currency code (default: USD)

        Returns:
            Available balance as Decimal
        """
        pass

    @abstractmethod
    async def cancel_order(self, external_id: str) -> ExecutionResult:
        """
        Cancel an open order (Async).

        Args:
            external_id: Exchange order ID to cancel

        Returns:
            ExecutionResult with cancellation status
        """
        pass

    # =========================================================================
    # POSITION TRACKING METHODS - Required for position reconciliation
    # =========================================================================

    @abstractmethod
    def has_position(self, symbol: str) -> bool:
        """
        Check if we have an open position in the given symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            True if position exists with quantity > 0
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position details for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            Dict with position details if position exists, None otherwise
            Expected format:
            {
                "symbol": "BTC-USD",
                "quantity": Decimal("0.1"),
                "side": "buy"
            }
        """
        pass

    @abstractmethod
    def get_position_size(self, symbol: str) -> Decimal:
        """
        Get the position size for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            Position quantity (Decimal("0") if no position)
        """
        pass

    def record_execution(self, result: ExecutionResult) -> None:
        """
        Record execution in history for slippage tracking.

        Called automatically by implementations after execute().

        Args:
            result: ExecutionResult to record
        """
        self._execution_history.append(result)

    def get_execution_history(self) -> List[ExecutionResult]:
        """
        Get all executions since initialization.

        Returns:
            List of ExecutionResult objects
        """
        return self._execution_history.copy()

    def get_slippage_stats(self) -> SlippageStats:
        """
        Calculate aggregate slippage statistics.

        This is THE monitoring tool. If avg_slippage_pct exceeds model
        assumption (typically 0.05%), strategy edge is being consumed.

        Returns:
            SlippageStats with total, avg, max slippage metrics
        """
        filled = [
            r for r in self._execution_history
            if r.status in (ExecutionStatus.FILLED, ExecutionStatus.PARTIAL)
            and r.slippage is not None
        ]

        if not filled:
            return SlippageStats.empty()

        total_slippage = sum(abs(r.slippage) for r in filled)
        total_fees = sum(r.fee or Decimal("0") for r in filled)

        # Find worst slippage
        worst = max(filled, key=lambda r: abs(r.slippage))
        max_slippage = abs(worst.slippage)
        max_slippage_pct = abs(worst.slippage_pct or 0.0)

        avg_slippage = total_slippage / len(filled)
        avg_slippage_pct = sum(abs(r.slippage_pct or 0.0) for r in filled) / len(filled)

        return SlippageStats(
            total_trades=len(filled),
            total_slippage=total_slippage,
            avg_slippage=avg_slippage,
            avg_slippage_pct=avg_slippage_pct,
            max_slippage=max_slippage,
            max_slippage_pct=max_slippage_pct,
            total_fees=total_fees,
            worst_trade_id=worst.external_id
        )

    def clear_history(self) -> int:
        """
        Clear execution history.

        Returns:
            Number of records cleared
        """
        count = len(self._execution_history)
        self._execution_history.clear()
        self.logger.info(f"Cleared {count} execution records")
        return count
