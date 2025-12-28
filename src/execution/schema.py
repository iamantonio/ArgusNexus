"""
V4 Execution Schema - Observable Order Execution

The Glass Box principle: Every fill, every fee, every slippage is explainable.
Slippage tracking is THE critical upgrade from V3.

In V3: Backtests assumed perfect fills, live bled from slippage.
In V4: ExecutionResult tracks slippage on EVERY trade. If slippage exceeds
       our modeled assumption, we know immediately and can stop trading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional


class ExecutionMode(Enum):
    """
    Execution mode - IMMUTABLE after initialization.

    This is the safety lock. You cannot switch from PAPER to LIVE at runtime.
    Mode must be set once at executor creation and never changed.
    """
    PAPER = "paper"
    LIVE = "live"


class ExecutionStatus(Enum):
    """Status of an order execution attempt."""
    PENDING = "pending"      # Order submitted, awaiting fill
    FILLED = "filled"        # Order completely filled
    PARTIAL = "partial"      # Order partially filled
    FAILED = "failed"        # Order rejected or error
    CANCELLED = "cancelled"  # Order cancelled


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class OrderRequest:
    """
    Input to the executor - what order are we placing?

    Contains everything needed to execute an order.
    This is passed INTO execution, ExecutionResult comes OUT.
    """
    symbol: str                     # e.g., "BTC-USD"
    side: OrderSide
    quantity: Decimal
    expected_price: Decimal         # THE KEY - what we EXPECT to fill at
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[Decimal] = None  # For LIMIT orders
    time_in_force: str = "GTC"      # GTC, IOC, FOK
    client_order_id: Optional[str] = None  # For correlation

    def __post_init__(self):
        """Validate and set defaults."""
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            self.limit_price = self.expected_price

    @property
    def notional_value(self) -> Decimal:
        """Total order value at expected price."""
        return self.expected_price * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "expected_price": str(self.expected_price),
            "order_type": self.order_type.value,
            "limit_price": str(self.limit_price) if self.limit_price else None,
            "time_in_force": self.time_in_force,
            "client_order_id": self.client_order_id,
            "notional_value": str(self.notional_value)
        }


@dataclass
class ExecutionResult:
    """
    The complete result of order execution.

    This is THE Glass Box output for execution. We track:
    - status: What happened? (FILLED, FAILED, PARTIAL)
    - fill_price: What price did we ACTUALLY get?
    - slippage: fill_price - expected_price (THE CRITICAL METRIC)
    - fee: Transaction costs
    - external_id: Exchange's order ID for audit trail

    The Slippage Promise:
    "Why did this trade underperform backtest?" -> ExecutionResult.slippage tells you.
    If slippage consistently exceeds model (0.05% default), strategy is invalid.

    Usage:
        result = executor.execute(order_request)

        if result.status == ExecutionStatus.FILLED:
            print(f"Filled at {result.fill_price}")
            print(f"Slippage: {result.slippage_pct:.4f}%")

            # Alert if slippage exceeds model
            if result.slippage_pct > 0.1:  # 0.1% threshold
                print("WARNING: Slippage exceeds model assumption!")

        # Log to Truth Engine
        truth_logger.log_order(execution_result=result.to_dict())
    """
    status: ExecutionStatus
    fill_price: Optional[Decimal] = None
    fill_quantity: Optional[Decimal] = None
    slippage: Optional[Decimal] = None      # Absolute: fill_price - expected_price
    slippage_pct: Optional[float] = None    # Percentage: (slippage / expected_price) * 100
    fee: Optional[Decimal] = None
    external_id: Optional[str] = None       # Exchange order ID
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    order_request: Optional[OrderRequest] = None  # Original request for context
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """True if order filled (fully or partially)."""
        return self.status in (ExecutionStatus.FILLED, ExecutionStatus.PARTIAL)

    @property
    def is_complete(self) -> bool:
        """True if order is in terminal state."""
        return self.status in (
            ExecutionStatus.FILLED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED
        )

    @property
    def net_proceeds(self) -> Optional[Decimal]:
        """Net value after fees (for sells) or net cost (for buys)."""
        if self.fill_price is None or self.fill_quantity is None:
            return None
        gross = self.fill_price * self.fill_quantity
        fee = self.fee or Decimal("0")
        return gross - fee

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict for Truth Engine logging.

        This is THE bridge to TruthLogger.log_order(execution_result=...).
        """
        return {
            "status": self.status.value,
            "fill_price": str(self.fill_price) if self.fill_price else None,
            "fill_quantity": str(self.fill_quantity) if self.fill_quantity else None,
            "slippage": str(self.slippage) if self.slippage else None,
            "slippage_pct": self.slippage_pct,
            "fee": str(self.fee) if self.fee else None,
            "external_id": self.external_id,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "is_success": self.is_success,
            "net_proceeds": str(self.net_proceeds) if self.net_proceeds else None,
            "order_request": self.order_request.to_dict() if self.order_request else None,
            "details": self.details
        }

    @classmethod
    def filled(
        cls,
        order_request: OrderRequest,
        fill_price: Decimal,
        fill_quantity: Decimal,
        fee: Decimal,
        external_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> "ExecutionResult":
        """
        Factory for successful fills.

        Automatically calculates slippage from expected vs actual price.
        """
        # Calculate slippage
        slippage = fill_price - order_request.expected_price

        # For sells, negative slippage is bad (got less than expected)
        # For buys, positive slippage is bad (paid more than expected)
        if order_request.side == OrderSide.SELL:
            slippage = -slippage  # Normalize: positive = bad for both

        slippage_pct = float(slippage / order_request.expected_price * 100)

        return cls(
            status=ExecutionStatus.FILLED,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            slippage=slippage,
            slippage_pct=slippage_pct,
            fee=fee,
            external_id=external_id,
            message="Order filled",
            order_request=order_request,
            details=details or {}
        )

    @classmethod
    def partial(
        cls,
        order_request: OrderRequest,
        fill_price: Decimal,
        fill_quantity: Decimal,
        fee: Decimal,
        external_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> "ExecutionResult":
        """Factory for partial fills."""
        slippage = fill_price - order_request.expected_price
        if order_request.side == OrderSide.SELL:
            slippage = -slippage
        slippage_pct = float(slippage / order_request.expected_price * 100)

        return cls(
            status=ExecutionStatus.PARTIAL,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            slippage=slippage,
            slippage_pct=slippage_pct,
            fee=fee,
            external_id=external_id,
            message=f"Partial fill: {fill_quantity}/{order_request.quantity}",
            order_request=order_request,
            details=details or {}
        )

    @classmethod
    def failed(
        cls,
        order_request: OrderRequest,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> "ExecutionResult":
        """Factory for failed orders."""
        return cls(
            status=ExecutionStatus.FAILED,
            message=reason,
            order_request=order_request,
            details=details or {"error": reason}
        )


@dataclass
class SlippageStats:
    """
    Aggregate slippage statistics for monitoring.

    If avg_slippage_pct consistently exceeds our model assumption (0.05%),
    the strategy edge is being eaten by execution costs.

    Usage:
        stats = executor.get_slippage_stats()
        if stats.avg_slippage_pct > 0.1:
            print("ALERT: Slippage exceeds model - consider pausing trading")
    """
    total_trades: int
    total_slippage: Decimal              # Sum of all slippage
    avg_slippage: Decimal                # Average slippage per trade
    avg_slippage_pct: float              # Average slippage percentage
    max_slippage: Decimal                # Worst slippage
    max_slippage_pct: float              # Worst slippage percentage
    total_fees: Decimal                  # Total fees paid
    worst_trade_id: Optional[str] = None # External ID of worst slippage trade

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "total_trades": self.total_trades,
            "total_slippage": str(self.total_slippage),
            "avg_slippage": str(self.avg_slippage),
            "avg_slippage_pct": self.avg_slippage_pct,
            "max_slippage": str(self.max_slippage),
            "max_slippage_pct": self.max_slippage_pct,
            "total_fees": str(self.total_fees),
            "worst_trade_id": self.worst_trade_id
        }

    @classmethod
    def empty(cls) -> "SlippageStats":
        """Factory for empty stats."""
        return cls(
            total_trades=0,
            total_slippage=Decimal("0"),
            avg_slippage=Decimal("0"),
            avg_slippage_pct=0.0,
            max_slippage=Decimal("0"),
            max_slippage_pct=0.0,
            total_fees=Decimal("0")
        )
