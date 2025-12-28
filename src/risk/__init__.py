"""
V4 Risk Module - Observable 10-Layer Risk System

The Glass Box promise: Every risk decision is explainable.

Usage:
    from risk import RiskManager, RiskConfig, TradeRequest, PortfolioState

    # Create manager with config
    config = RiskConfig(daily_loss_limit_pct=2.0, min_risk_reward_ratio=2.0)
    risk_manager = RiskManager(config)

    # Evaluate a trade
    result = risk_manager.evaluate(trade_request, portfolio_state)

    if result.approved:
        # Proceed with trade
        print("Trade approved - all 10 checks passed")
    else:
        # Trade blocked - full explanation available
        print(f"Blocked: {result.rejection_reason}")
        print(f"Failed check: {result.first_failure.name.value}")

    # Log to Truth Engine
    truth_logger.log_decision(risk_checks=result.to_dict())
"""

from .schema import (
    PortfolioState,
    RiskCheckName,
    RiskCheckResult,
    RiskResult,
    TradeRequest,
)
from .manager import RiskConfig, RiskConfigError, RiskManager

__all__ = [
    "RiskManager",
    "RiskConfig",
    "RiskConfigError",
    "RiskResult",
    "RiskCheckResult",
    "RiskCheckName",
    "TradeRequest",
    "PortfolioState",
]
