#!/usr/bin/env python3
"""
Golden Run Test - The Heart Beats

This is THE integration test for V4.
It verifies the complete flow: Strategy → Risk → Decision → Execution → Trade

Requirements:
1. Feed one dataframe row simulating a perfect Bullish Crossover
2. Run the engine
3. Assert the database contains exactly:
   - 1 Decision row (Approved)
   - 1 Order row (Filled)
   - 1 Trade row (Open)

If this test passes, the Glass Box heart is beating.
If this test fails, something in the pipeline is broken.

Run: python scripts/golden_run.py
"""

import sys
import os
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

# Import the components
from engine import TradingEngine
from strategy.dual_ema import DualEMACrossover
from risk import RiskManager, RiskConfig
from execution import PaperExecutor
from truth.logger import TruthLogger
from truth.schema import DecisionResult


def create_bullish_crossover_data() -> pd.DataFrame:
    """
    Create a dataframe that will trigger a Bullish Crossover.

    The Dual EMA strategy looks for (in _detect_crossover):
    - PREVIOUS bar: fast_ema < slow_ema
    - CURRENT bar: fast_ema > slow_ema

    The crossover must happen ON THE CURRENT BAR (last bar).
    If the crossover happened on a previous bar, we get CrossoverType.NONE.

    Strategy:
    - Bars 0-47: Sideways with slight decline, keeps fast < slow
    - Bar 48 (previous): Still has fast < slow
    - Bar 49 (current): BIG spike pulls fast > slow → BULLISH CROSSOVER
    """
    # Generate 50 rows of price data
    rows = 50
    base_price = 100000.0

    data = []

    for i in range(rows):
        # The key insight: We need fast < slow through bar 48,
        # then fast > slow at bar 49. The spike must be ONLY on bar 49.

        if i < 48:
            # Slight downtrend - keeps fast EMA below slow EMA
            # Decline slowly from 100000 to about 95000
            price = base_price - (i * 105)  # 100000 -> 95040
        elif i == 48:
            # Bar 48 (previous bar): still declining, fast still < slow
            price = base_price - (48 * 105)  # ~94960
        else:
            # Bar 49 (CURRENT bar): MASSIVE spike to force crossover NOW
            # Need to overwhelm the slow EMA's inertia
            price = 110000  # Big spike to pull fast EMA above slow

        # Add variation for ATR (but keep trends clear)
        high = price * 1.003
        low = price * 0.997
        close = price
        open_price = price * 0.999

        data.append({
            "timestamp": datetime.now(timezone.utc) - timedelta(minutes=(rows - i)),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000000
        })

    df = pd.DataFrame(data)
    return df


def run_golden_test():
    """
    The Golden Test.

    This is what success looks like:
    1. Strategy sees bullish crossover → SignalResult with BUY
    2. Risk approves → RiskResult with approved=True
    3. Decision logged → 1 row in decisions table
    4. Executor fills → ExecutionResult with FILLED
    5. Order logged → 1 row in orders table
    6. Trade logged → 1 row in trades table
    """
    print()
    print("=" * 60)
    print("GOLDEN RUN: The Heart Beats")
    print("=" * 60)
    print()

    # Use a fresh test database
    db_path = Path(__file__).parent.parent / "data" / "golden_test.db"
    if db_path.exists():
        db_path.unlink()  # Start fresh

    # Initialize components
    print("Initializing components...")

    strategy = DualEMACrossover(
        fast_period=12,
        slow_period=26,
        atr_period=14
    )

    risk_config = RiskConfig(
        daily_loss_limit_pct=2.0,
        max_drawdown_pct=5.0,
        min_risk_reward_ratio=1.5,  # Our strategy gives 1.5:1 (3x ATR TP / 2x ATR stop)
        max_asset_concentration_pct=50.0,  # Allow larger positions for test
        max_trades_per_hour=10
    )
    risk_manager = RiskManager(risk_config)

    executor = PaperExecutor(
        starting_balance=Decimal("10000"),
        base_slippage_pct=Decimal("0.02"),
        fee_rate=Decimal("0.004")
    )

    truth_logger = TruthLogger(str(db_path))

    # Create the engine
    print("Creating TradingEngine...")
    engine = TradingEngine(
        strategy=strategy,
        risk_manager=risk_manager,
        executor=executor,
        truth_logger=truth_logger,
        symbol="BTC-USD",
        capital=Decimal("10000")
    )

    # Create bullish crossover data
    print("Creating bullish crossover data...")
    df = create_bullish_crossover_data()
    print(f"  Data rows: {len(df)}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Run one tick
    print("\nRunning engine tick...")
    result = engine.run_tick(df)

    # Verify the result
    print("\n" + "-" * 60)
    print("VERIFICATION")
    print("-" * 60)

    # Check the result
    print(f"\nEngine result: {result}")

    # Query the database
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Count decisions
    decisions = conn.execute("SELECT * FROM decisions").fetchall()
    decision_count = len(decisions)
    print(f"\nDecisions in database: {decision_count}")

    if decisions:
        d = decisions[0]
        print(f"  Decision ID: {d['decision_id']}")
        print(f"  Result: {d['result']}")
        print(f"  Reason: {d['result_reason']}")

    # Count orders
    orders = conn.execute("SELECT * FROM orders").fetchall()
    order_count = len(orders)
    print(f"\nOrders in database: {order_count}")

    if orders:
        o = orders[0]
        print(f"  Order ID: {o['order_id']}")
        print(f"  Status: {o['status']}")
        print(f"  Fill price: {o['fill_price']}")

    # Count trades
    trades = conn.execute("SELECT * FROM trades").fetchall()
    trade_count = len(trades)
    print(f"\nTrades in database: {trade_count}")

    if trades:
        t = trades[0]
        print(f"  Trade ID: {t['trade_id']}")
        print(f"  Status: {t['status']}")
        print(f"  Entry price: {t['entry_price']}")

    conn.close()

    # ASSERTIONS - The Golden Test
    print("\n" + "=" * 60)
    print("ASSERTIONS")
    print("=" * 60)

    failures = []

    # Assert 1 Decision
    if decision_count != 1:
        failures.append(f"Expected 1 Decision, got {decision_count}")
    else:
        print("✅ 1 Decision row")
        # Check it was approved
        if decisions[0]['result'] != 'signal_long':
            failures.append(f"Expected Decision result 'signal_long', got '{decisions[0]['result']}'")
        else:
            print("✅ Decision was APPROVED (signal_long)")

    # Assert 1 Order
    if order_count != 1:
        failures.append(f"Expected 1 Order, got {order_count}")
    else:
        print("✅ 1 Order row")
        # Check it was filled
        if orders[0]['status'] != 'filled':
            failures.append(f"Expected Order status 'filled', got '{orders[0]['status']}'")
        else:
            print("✅ Order was FILLED")

    # Assert 1 Trade
    if trade_count != 1:
        failures.append(f"Expected 1 Trade, got {trade_count}")
    else:
        print("✅ 1 Trade row")
        # Check status is open
        if trades[0]['status'] != 'open':
            failures.append(f"Expected Trade status 'open', got '{trades[0]['status']}'")
        else:
            print("✅ Trade is OPEN")

    # Final verdict
    print("\n" + "=" * 60)
    if failures:
        print("❌ GOLDEN RUN FAILED")
        print("=" * 60)
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✅ GOLDEN RUN PASSED")
        print("=" * 60)
        print("\nThe heart beats. The Glass Box lives.")
        print("Every decision logged. Every trade tracked.")
        sys.exit(0)


if __name__ == "__main__":
    run_golden_test()
