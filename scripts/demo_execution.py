#!/usr/bin/env python3
"""
Demo: Execution Module + Slippage Tracking + Truth Engine Integration

Shows how the V4 execution system tracks slippage on every trade.

The Glass Box promise in action:
- Every fill shows expected vs actual price
- Slippage is calculated and tracked
- Aggregate stats reveal if model assumption is violated

Run: python scripts/demo_execution.py
"""

import sys
from pathlib import Path
from decimal import Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from execution import (
    PaperExecutor,
    OrderRequest,
    OrderSide,
    OrderType,
    ExecutionMode,
    ExecutionModeImmutableError,
)
from truth.logger import TruthLogger
from truth.schema import DecisionResult, OrderSide as TruthOrderSide


def demo_paper_execution():
    """Demonstrate paper execution with slippage tracking."""
    print()
    print("=" * 60)
    print("PAPER EXECUTOR DEMO: Slippage Tracking")
    print("=" * 60)

    # Create paper executor
    executor = PaperExecutor(
        starting_balance=Decimal("10000"),
        base_slippage_pct=Decimal("0.02"),      # 0.02% base spread
        noise_slippage_pct=Decimal("0.03"),     # 0-0.03% random
        size_impact_per_10k=Decimal("0.01"),    # 0.01% per $10k
        fee_rate=Decimal("0.004")               # 0.4% fee
    )

    print(f"\nExecutor Mode: {executor.mode.value} (immutable)")
    print(f"Starting Balance: ${executor.get_balance('USD')}")
    print()

    # Execute multiple trades
    trades = [
        ("BTC-USD", OrderSide.BUY, "0.01", "100000"),
        ("BTC-USD", OrderSide.BUY, "0.02", "100500"),
        ("ETH-USD", OrderSide.BUY, "0.5", "4000"),
        ("BTC-USD", OrderSide.SELL, "0.01", "101000"),
    ]

    print("-" * 60)
    print("Executing trades with slippage simulation...")
    print("-" * 60)

    for symbol, side, qty, price in trades:
        order = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=Decimal(qty),
            expected_price=Decimal(price)
        )

        result = executor.execute(order)

        print(f"\n{side.value.upper()} {qty} {symbol}")
        print(f"  Expected: ${price}")
        print(f"  Filled:   ${result.fill_price}")
        print(f"  Slippage: {result.slippage_pct:.4f}%")
        print(f"  Fee:      ${result.fee:.4f}")
        print(f"  Order ID: {result.external_id}")

    # Show aggregate stats
    print()
    print("-" * 60)
    print("SLIPPAGE STATISTICS (The Key V4 Metric)")
    print("-" * 60)

    stats = executor.get_slippage_stats()
    print(f"\n  Total Trades:    {stats.total_trades}")
    print(f"  Total Slippage:  ${stats.total_slippage:.4f}")
    print(f"  Avg Slippage:    ${stats.avg_slippage:.4f}")
    print(f"  Avg Slippage %:  {stats.avg_slippage_pct:.4f}%")
    print(f"  Max Slippage:    ${stats.max_slippage:.4f}")
    print(f"  Max Slippage %:  {stats.max_slippage_pct:.4f}%")
    print(f"  Total Fees:      ${stats.total_fees:.4f}")
    print(f"  Worst Trade:     {stats.worst_trade_id}")

    # Check against model assumption
    MODEL_SLIPPAGE_PCT = 0.05  # Our backtest assumes 0.05% slippage

    print()
    if stats.avg_slippage_pct > MODEL_SLIPPAGE_PCT:
        print(f"  ⚠️ WARNING: Avg slippage ({stats.avg_slippage_pct:.4f}%) "
              f"exceeds model assumption ({MODEL_SLIPPAGE_PCT}%)")
        print("  Strategy edge may be consumed by execution costs!")
    else:
        print(f"  ✅ Slippage within model assumption ({MODEL_SLIPPAGE_PCT}%)")

    # Final balances
    print()
    print("-" * 60)
    print("Final Balances")
    print("-" * 60)
    for currency, balance in executor.get_all_balances().items():
        print(f"  {currency}: {balance}")

    print(f"\nP&L (realized): ${executor.get_pnl():.2f}")


def demo_immutable_mode():
    """Demonstrate that execution mode is immutable."""
    print()
    print("=" * 60)
    print("SAFETY LOCK DEMO: Immutable Execution Mode")
    print("=" * 60)

    executor = PaperExecutor(starting_balance=Decimal("1000"))

    print(f"\nExecutor created in {executor.mode.value} mode")
    print("Attempting to change mode to LIVE...")

    try:
        executor.mode = ExecutionMode.LIVE
        print("  FAILED - mode should not be changeable!")
    except ExecutionModeImmutableError as e:
        print(f"  ✅ Blocked: {e}")

    print("\nThe Safety Lock works. Mode cannot be changed at runtime.")


def demo_truth_engine_integration():
    """Demonstrate integration with Truth Engine."""
    print()
    print("=" * 60)
    print("TRUTH ENGINE INTEGRATION")
    print("=" * 60)

    # Initialize components
    db_path = Path(__file__).parent.parent / "data" / "v4_live_paper.db"
    truth_logger = TruthLogger(str(db_path))

    executor = PaperExecutor(starting_balance=Decimal("10000"))

    # Execute a trade
    order = OrderRequest(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=Decimal("0.01"),
        expected_price=Decimal("100000")
    )

    result = executor.execute(order)

    print(f"\nTrade executed:")
    print(f"  Order ID: {result.external_id}")
    print(f"  Slippage: {result.slippage_pct:.4f}%")

    # Log to Truth Engine
    print("\nLogging to Truth Engine...")

    # First, log a decision (normally comes from strategy)
    decision = truth_logger.log_decision(
        symbol="BTC-USD",
        strategy_name="dual_ema_crossover",
        signal_values={
            "action": "buy",
            "confidence": 0.75,
            "ema_diff": 150.5
        },
        risk_checks={
            "approved": True,
            "checks_passed": 10,
            "summary": "All risk checks passed"
        },
        result=DecisionResult.SIGNAL_LONG,
        result_reason="Bullish crossover detected"
    )

    # Log the order with execution details
    logged_order = truth_logger.log_order(
        decision_id=decision.decision_id,
        symbol="BTC-USD",
        side=TruthOrderSide.BUY,
        quantity=Decimal("0.01"),
        requested_price=Decimal("100000")
    )

    # Update with fill
    truth_logger.update_order_fill(
        order_id=logged_order.order_id,
        fill_price=result.fill_price,
        fill_quantity=result.fill_quantity,
        exchange_order_id=result.external_id,
        commission=result.fee
    )

    print(f"  Decision ID: {decision.decision_id}")
    print(f"  Order ID: {logged_order.order_id}")

    # Query the full audit trail
    print("\nQuerying audit trail...")
    recent = truth_logger.get_recent_decisions(symbol="BTC-USD", limit=1)
    if recent:
        d = recent[0]
        print(f"  Strategy: {d['strategy_name']}")
        print(f"  Result: {d['result']}")
        print(f"  Reason: {d['result_reason']}")

    print()
    print("The Glass Box: Full audit trail from signal to fill to slippage.")


def demo_slippage_comparison():
    """Show how slippage affects P&L over many trades."""
    print()
    print("=" * 60)
    print("SLIPPAGE IMPACT ANALYSIS")
    print("=" * 60)
    print("\nComparing: No slippage vs Realistic slippage")
    print("-" * 60)

    # Perfect execution (V3 style - unrealistic)
    perfect = PaperExecutor(
        starting_balance=Decimal("10000"),
        base_slippage_pct=Decimal("0"),
        noise_slippage_pct=Decimal("0"),
        size_impact_per_10k=Decimal("0"),
        fee_rate=Decimal("0.004")  # Still charge fees
    )

    # Realistic execution (V4 style)
    realistic = PaperExecutor(
        starting_balance=Decimal("10000"),
        base_slippage_pct=Decimal("0.02"),
        noise_slippage_pct=Decimal("0.03"),
        size_impact_per_10k=Decimal("0.01"),
        fee_rate=Decimal("0.004")
    )

    # Simulate 10 round-trip trades
    print("\nSimulating 10 round-trip trades...")

    for i in range(10):
        # Buy
        buy = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            expected_price=Decimal("100000")
        )
        perfect.execute(buy)
        realistic.execute(buy)

        # Sell at 1% profit
        sell = OrderRequest(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            expected_price=Decimal("101000")  # 1% profit target
        )
        perfect.execute(sell)
        realistic.execute(sell)

    # Compare results
    perfect_stats = perfect.get_slippage_stats()
    realistic_stats = realistic.get_slippage_stats()

    print(f"\nPerfect Execution (V3 Style):")
    print(f"  Total Slippage: ${perfect_stats.total_slippage:.2f}")
    print(f"  Total Fees: ${perfect_stats.total_fees:.2f}")
    print(f"  Ending Balance: ${perfect.get_balance('USD'):.2f}")
    print(f"  Net P&L: ${perfect.get_pnl():.2f}")

    print(f"\nRealistic Execution (V4 Style):")
    print(f"  Total Slippage: ${realistic_stats.total_slippage:.2f}")
    print(f"  Total Fees: ${realistic_stats.total_fees:.2f}")
    print(f"  Ending Balance: ${realistic.get_balance('USD'):.2f}")
    print(f"  Net P&L: ${realistic.get_pnl():.2f}")

    # The difference
    pnl_diff = perfect.get_pnl() - realistic.get_pnl()
    print(f"\n  Slippage Cost: ${pnl_diff:.2f} over 10 round trips")
    print(f"  Per Trade: ${pnl_diff / 20:.2f}")
    print()
    print("This is why V4 tracks slippage. The 'edge' that looks great")
    print("in backtest gets eaten by execution costs in reality.")


def main():
    print()
    print("=" * 60)
    print("V4 DEMO: Execution Module - Slippage Tracking")
    print("=" * 60)

    demo_paper_execution()
    demo_immutable_mode()
    demo_truth_engine_integration()
    demo_slippage_comparison()

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("The V4 Execution Promise:")
    print("  - Every fill tracks expected vs actual price")
    print("  - Slippage is calculated on EVERY trade")
    print("  - Aggregate stats reveal model vs reality gap")
    print("  - Truth Engine logs full execution audit trail")
    print()


if __name__ == "__main__":
    main()
