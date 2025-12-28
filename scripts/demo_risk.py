#!/usr/bin/env python3
"""
Demo: Risk Manager + Truth Engine Integration

Shows how the 10-layer risk system evaluates trades and produces
observable results for the Truth Engine.

The Glass Box promise in action:
- Every check shows what it evaluated
- Every rejection explains exactly why
- Full audit trail for post-trade analysis

Run: python scripts/demo_risk.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk import (
    RiskManager,
    RiskConfig,
    RiskResult,
    TradeRequest,
    PortfolioState,
)
from truth.logger import TruthLogger
from truth.schema import DecisionResult


def create_healthy_portfolio() -> PortfolioState:
    """Create a portfolio in good standing - all checks should pass."""
    return PortfolioState(
        total_capital=Decimal("10000"),
        cash_available=Decimal("8000"),
        daily_pnl=Decimal("50"),
        daily_pnl_percent=0.5,
        total_pnl=Decimal("200"),
        total_pnl_percent=2.0,
        open_positions={
            "ETH-USD": {"value": 2000, "quantity": 0.5}
        },
        recent_trades_count=2,
        circuit_breaker_triggered=False,
        trading_halted=False
    )


def create_stressed_portfolio() -> PortfolioState:
    """Create a portfolio under stress - some checks should fail."""
    return PortfolioState(
        total_capital=Decimal("10000"),
        cash_available=Decimal("3000"),
        daily_pnl=Decimal("-250"),
        daily_pnl_percent=-2.5,  # Exceeds 2% daily loss limit
        total_pnl=Decimal("-400"),
        total_pnl_percent=-4.0,
        open_positions={
            "BTC-USD": {"value": 3500, "quantity": 0.035},
            "ETH-USD": {"value": 3500, "quantity": 0.9}
        },
        recent_trades_count=4,
        circuit_breaker_triggered=False,
        trading_halted=False
    )


def create_good_trade() -> TradeRequest:
    """Create a trade request with good R:R ratio."""
    return TradeRequest(
        symbol="BTC-USD",
        side="buy",
        quantity=Decimal("0.01"),
        entry_price=Decimal("100000"),
        stop_loss_price=Decimal("98000"),   # 2% stop
        take_profit_price=Decimal("106000"), # 6% target = 3:1 R:R
        strategy_name="dual_ema_crossover",
        confidence=0.75
    )


def create_bad_trade() -> TradeRequest:
    """Create a trade request with poor R:R ratio."""
    return TradeRequest(
        symbol="BTC-USD",
        side="buy",
        quantity=Decimal("0.05"),
        entry_price=Decimal("100000"),
        stop_loss_price=Decimal("95000"),    # 5% stop = $250 risk
        take_profit_price=Decimal("101000"), # 1% target = $50 reward = 0.2 R:R
        strategy_name="dual_ema_crossover",
        confidence=0.3
    )


def print_risk_result(result: RiskResult, title: str):
    """Pretty print a risk evaluation result."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print()

    if result.approved:
        print("RESULT: APPROVED")
    else:
        print("RESULT: REJECTED")
        print(f"REASON: {result.rejection_reason}")

    print()
    print("-" * 60)
    print("ALL CHECKS (10-Layer Risk Gate)")
    print("-" * 60)

    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        icon = " " if check.passed else "X"
        print(f"  [{icon}] {check.name.value:30} {status}")
        print(f"      Threshold: {check.threshold}")
        print(f"      Actual:    {check.actual}")
        if not check.passed:
            print(f"      Reason:    {check.reason}")
        print()

    print("-" * 60)
    print(f"Summary: {len(result.passed_checks)}/10 checks passed")
    print("-" * 60)


def main():
    print()
    print("=" * 60)
    print("V4 DEMO: Risk Manager - Observable 10-Layer System")
    print("=" * 60)
    print()

    # Initialize
    config = RiskConfig(
        daily_loss_limit_pct=2.0,
        max_drawdown_pct=5.0,
        min_risk_reward_ratio=2.0,
        max_asset_concentration_pct=30.0,
        max_trades_per_hour=5
    )
    risk_manager = RiskManager(config)

    print("Risk Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print()

    # =========================================================================
    # Scenario 1: Good trade on healthy portfolio
    # =========================================================================
    print("\n" + "=" * 60)
    print("SCENARIO 1: Good Trade + Healthy Portfolio")
    print("Expected: APPROVED (all checks pass)")
    print("=" * 60)

    portfolio = create_healthy_portfolio()
    trade = create_good_trade()

    print(f"\nPortfolio State:")
    print(f"  Capital: ${portfolio.total_capital}")
    print(f"  Daily P&L: ${portfolio.daily_pnl} ({portfolio.daily_pnl_percent}%)")
    print(f"  Open Positions: {portfolio.position_count}")
    print(f"  Recent Trades: {portfolio.recent_trades_count}/hr")

    print(f"\nTrade Request:")
    print(f"  {trade.side.upper()} {trade.quantity} {trade.symbol}")
    print(f"  Entry: ${trade.entry_price}")
    print(f"  Stop Loss: ${trade.stop_loss_price}")
    print(f"  Take Profit: ${trade.take_profit_price}")
    print(f"  R:R Ratio: {trade.risk_reward_ratio:.2f}:1")

    result = risk_manager.evaluate(trade, portfolio)
    print_risk_result(result, "SCENARIO 1 RESULT")

    # =========================================================================
    # Scenario 2: Good trade on stressed portfolio
    # =========================================================================
    print("\n" + "=" * 60)
    print("SCENARIO 2: Good Trade + Stressed Portfolio")
    print("Expected: REJECTED (daily loss limit exceeded)")
    print("=" * 60)

    stressed_portfolio = create_stressed_portfolio()
    trade = create_good_trade()

    print(f"\nPortfolio State:")
    print(f"  Capital: ${stressed_portfolio.total_capital}")
    print(f"  Daily P&L: ${stressed_portfolio.daily_pnl} ({stressed_portfolio.daily_pnl_percent}%)")
    print(f"  Total P&L: ${stressed_portfolio.total_pnl} ({stressed_portfolio.total_pnl_percent}%)")
    print(f"  Open Positions: {stressed_portfolio.position_count}")

    result = risk_manager.evaluate(trade, stressed_portfolio)
    print_risk_result(result, "SCENARIO 2 RESULT")

    # =========================================================================
    # Scenario 3: Bad trade (poor R:R) on healthy portfolio
    # =========================================================================
    print("\n" + "=" * 60)
    print("SCENARIO 3: Bad Trade (Poor R:R) + Healthy Portfolio")
    print("Expected: REJECTED (R:R ratio too low)")
    print("=" * 60)

    portfolio = create_healthy_portfolio()
    bad_trade = create_bad_trade()

    print(f"\nTrade Request:")
    print(f"  {bad_trade.side.upper()} {bad_trade.quantity} {bad_trade.symbol}")
    print(f"  Entry: ${bad_trade.entry_price}")
    print(f"  Stop Loss: ${bad_trade.stop_loss_price} ({(1 - float(bad_trade.stop_loss_price / bad_trade.entry_price)) * 100:.1f}% risk)")
    print(f"  Take Profit: ${bad_trade.take_profit_price} ({(float(bad_trade.take_profit_price / bad_trade.entry_price) - 1) * 100:.1f}% reward)")
    print(f"  R:R Ratio: {bad_trade.risk_reward_ratio:.2f}:1 (POOR!)")

    result = risk_manager.evaluate(bad_trade, portfolio)
    print_risk_result(result, "SCENARIO 3 RESULT")

    # =========================================================================
    # Integration with Truth Engine
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRUTH ENGINE INTEGRATION")
    print("=" * 60)

    # Initialize Truth Engine
    db_path = Path(__file__).parent.parent / "data" / "v4_live_paper.db"
    truth_logger = TruthLogger(str(db_path))

    # Re-evaluate a trade for logging
    portfolio = create_healthy_portfolio()
    trade = create_good_trade()
    result = risk_manager.evaluate(trade, portfolio)

    print("\nLogging decision to Truth Engine...")
    print(f"  Risk Result: {'APPROVED' if result.approved else 'REJECTED'}")
    print(f"  Checks Passed: {len(result.passed_checks)}/10")

    # This is THE integration - RiskResult.to_dict() feeds directly into TruthLogger
    decision = truth_logger.log_decision(
        symbol=trade.symbol,
        strategy_name=trade.strategy_name,
        signal_values={
            "trade_request": trade.to_dict(),
            "confidence": trade.confidence
        },
        risk_checks=result.to_dict(),  # THE GLASS BOX OUTPUT
        result=DecisionResult.SIGNAL_LONG if result.approved else DecisionResult.RISK_REJECTED,
        result_reason="Trade approved" if result.approved else result.rejection_reason
    )

    print(f"\n  Decision ID: {decision.decision_id}")
    print(f"  Timestamp: {decision.timestamp}")

    # Verify it was logged
    print("\nVerifying logged data...")
    logged = truth_logger.get_recent_decisions(symbol="BTC-USD", limit=1)
    if logged:
        print("  Decision logged to database!")
        rc = logged[0]['risk_checks']
        print(f"  Approved: {rc['approved']}")
        print(f"  Checks Summary: {rc['summary']['passed']}/{rc['summary']['total_checks']} passed")
    else:
        print("  Failed to verify logged decision")

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("The Glass Box promise fulfilled:")
    print("  - Every check shows threshold vs actual")
    print("  - Every rejection explains exactly WHY")
    print("  - Full breakdown logged to Truth Engine")
    print("  - Query: 'Why was this trade blocked?' -> Answer in database")
    print()


if __name__ == "__main__":
    main()
