"""
V4 Execution Module - Observable Order Execution with Slippage Tracking

The Glass Box promise: Every fill, every fee, every slippage is tracked.

Key Types:
- ExecutionMode: PAPER or LIVE (immutable after init)
- ExecutionResult: status, fill_price, SLIPPAGE, fee, external_id
- OrderRequest: symbol, side, quantity, expected_price

Executors:
- PaperExecutor: Simulates fills with realistic slippage
- LiveExecutor: Real orders via Coinbase API

The Slippage Promise:
- V3 Problem: Backtests assumed perfect fills, live bled from slippage
- V4 Solution: Track slippage on EVERY trade, alert if exceeds model

Usage:
    from execution import PaperExecutor, OrderRequest, OrderSide

    executor = PaperExecutor(starting_balance=Decimal("10000"))

    order = OrderRequest(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=Decimal("0.01"),
        expected_price=Decimal("100000")
    )

    result = executor.execute(order)

    print(f"Filled at {result.fill_price}")
    print(f"Slippage: {result.slippage_pct:.4f}%")  # THE KEY METRIC

    # Check aggregate slippage
    stats = executor.get_slippage_stats()
    if stats.avg_slippage_pct > 0.1:
        print("WARNING: Slippage exceeds model assumption!")

    # Log to Truth Engine
    truth_logger.log_order(execution_result=result.to_dict())
"""

from .schema import (
    ExecutionMode,
    ExecutionStatus,
    ExecutionResult,
    OrderRequest,
    OrderSide,
    OrderType,
    SlippageStats,
)
from .base import Executor, ExecutionModeImmutableError
from .paper import PaperExecutor
from .live import LiveExecutor
from .gemini import GeminiExecutor, create_gemini_executor

__all__ = [
    # Schema
    "ExecutionMode",
    "ExecutionStatus",
    "ExecutionResult",
    "OrderRequest",
    "OrderSide",
    "OrderType",
    "SlippageStats",
    # Base
    "Executor",
    "ExecutionModeImmutableError",
    # Implementations
    "PaperExecutor",
    "LiveExecutor",
    "GeminiExecutor",
    "create_gemini_executor",
]
