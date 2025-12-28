#!/usr/bin/env python3
"""
Demo: Strategy + Truth Engine Integration

Shows how the Dual EMA strategy generates witnessed signals
that flow directly into the Truth Engine for logging.

This is THE workflow:
1. Strategy evaluates market data
2. Strategy returns SignalResult with FULL context
3. SignalResult.to_signal_values() feeds directly into TruthLogger.log_decision()

Run: python scripts/demo_strategy.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.dual_ema import DualEMACrossover, Signal
from truth.logger import TruthLogger
from truth.schema import DecisionResult


def generate_sample_data(
    num_bars: int = 100,
    start_price: float = 50000.0,
    trend: str = "bullish"
) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)

    # Generate price movement
    if trend == "bullish":
        drift = 0.001  # Slight upward drift
    elif trend == "bearish":
        drift = -0.001
    else:
        drift = 0

    returns = np.random.normal(drift, 0.02, num_bars)
    prices = start_price * np.cumprod(1 + returns)

    # Generate OHLCV
    data = {
        'timestamp': [datetime.utcnow() - timedelta(minutes=5*(num_bars-i)) for i in range(num_bars)],
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, num_bars)),
        'high': prices * (1 + np.random.uniform(0, 0.01, num_bars)),
        'low': prices * (1 - np.random.uniform(0, 0.01, num_bars)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, num_bars)
    }

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df


def main():
    print("=" * 60)
    print("V4 DEMO: Strategy → Truth Engine Integration")
    print("=" * 60)
    print()

    # Initialize
    strategy = DualEMACrossover()
    truth_logger = TruthLogger(str(Path(__file__).parent.parent / "data" / "v4_live_paper.db"))

    print(f"Strategy: {strategy.__class__.__name__}")
    print(f"  Fast EMA: {strategy.fast_period}")
    print(f"  Slow EMA: {strategy.slow_period}")
    print(f"  ATR Period: {strategy.atr_period}")
    print(f"  Stop Multiplier: {strategy.atr_stop_multiplier}x ATR")
    print(f"  TP Multiplier: {strategy.atr_tp_multiplier}x ATR")
    print()

    # Generate sample data with a bullish trend (to trigger crossover)
    print("Generating sample market data (100 bars, bullish trend)...")
    df = generate_sample_data(num_bars=100, start_price=50000, trend="bullish")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
    print()

    # Evaluate strategy
    print("Evaluating strategy...")
    result = strategy.evaluate(df, has_open_position=False)

    print()
    print("-" * 60)
    print("SIGNAL RESULT (The Glass Box Output)")
    print("-" * 60)
    print(f"Signal: {result.signal.value.upper()}")
    print(f"Reason: {result.reason}")
    print()

    # Show the full context (what gets logged)
    ctx = result.context
    print("Context (What the strategy witnessed):")
    print(f"  Timestamp: {ctx.timestamp}")
    print(f"  Current Price: ${float(ctx.current_price):.2f}")
    print()
    print("  EMA Values:")
    print(f"    Fast EMA (12): ${float(ctx.fast_ema):.2f}")
    print(f"    Slow EMA (26): ${float(ctx.slow_ema):.2f}")
    print(f"    EMA Diff: ${float(ctx.ema_diff):.2f} ({float(ctx.ema_diff_percent):.3f}%)")
    print(f"    Crossover Type: {ctx.crossover_type.value}")
    print(f"    Bars Since Crossover: {ctx.crossover_bars_ago}")
    print()
    print("  ATR (Volatility):")
    print(f"    ATR (14): ${float(ctx.atr):.2f}")
    print(f"    ATR %: {float(ctx.atr_percent):.2f}%")
    print()

    if ctx.stop_loss_price:
        print("  Calculated Thresholds:")
        print(f"    Stop Loss: ${float(ctx.stop_loss_price):.2f}")
        print(f"    Take Profit: ${float(ctx.take_profit_price):.2f}")
        print(f"    Risk Amount: ${float(ctx.risk_amount):.2f}")
        print(f"    Reward Amount: ${float(ctx.reward_amount):.2f}")
        print(f"    Risk:Reward: 1:{float(ctx.risk_reward_ratio):.1f}")

    print()
    print("-" * 60)
    print("TRUTH ENGINE LOGGING")
    print("-" * 60)

    # Map Signal to DecisionResult
    if result.signal == Signal.LONG:
        decision_result = DecisionResult.SIGNAL_LONG
    elif result.signal == Signal.EXIT_LONG:
        decision_result = DecisionResult.SIGNAL_CLOSE
    else:
        decision_result = DecisionResult.NO_SIGNAL

    # Log to Truth Engine
    print("Logging decision to Truth Engine...")

    decision = truth_logger.log_decision(
        symbol="BTC-USD",
        strategy_name=result.strategy_name,
        signal_values=result.to_signal_values(),
        risk_checks={},  # Would be populated by risk module
        result=decision_result,
        result_reason=result.reason
    )

    print(f"  Decision ID: {decision.decision_id}")
    print(f"  Logged at: {decision.timestamp}")
    print()

    # Show what was logged
    print("Verifying logged data...")
    logged = truth_logger.get_recent_decisions(symbol="BTC-USD", limit=1)
    if logged:
        print("  ✅ Decision successfully logged to database!")
        print()
        print("  Signal Values (JSON in database):")
        sv = logged[0]['signal_values']
        print(f"    Strategy: {sv['strategy']['name']} v{sv['strategy']['version']}")
        print(f"    Signal: {sv['signal']}")
        print(f"    Fast EMA: {sv['ema']['fast_ema_12']:.2f}")
        print(f"    Slow EMA: {sv['ema']['slow_ema_26']:.2f}")
        print(f"    Crossover: {sv['crossover']['type']}")
    else:
        print("  ❌ Failed to verify logged decision")

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("The Glass Box promise fulfilled:")
    print("  - Strategy evaluated market data")
    print("  - Full context captured (EMA values, ATR, thresholds)")
    print("  - Decision logged to Truth Engine")
    print("  - Query: 'Why did this signal fire?' → Answer in database")
    print()


if __name__ == "__main__":
    main()
