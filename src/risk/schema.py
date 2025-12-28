"""
V4 Risk System Schema - Observable Risk Results

The Glass Box principle: Every risk decision must be explainable.
No more "trade blocked" without knowing exactly WHY.

RiskResult replaces the old boolean return type.
Old: def can_open_trade(...) -> bool
New: def can_open_trade(...) -> RiskResult
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskCheckName(Enum):
    """All risk checks in the system - the 11-layer defense (Layer 0 = Session)."""
    SESSION_STATE = "session_state"          # Layer 0: Session/time-based checks
    PORTFOLIO_RISK = "portfolio_risk"        # Layer 0.5: Portfolio-level checks
    TRADING_HALTED = "trading_halted"
    TRADE_FREQUENCY = "trade_frequency"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CIRCUIT_BREAKER = "circuit_breaker"
    CIRCUIT_BREAKER_COOLDOWN = "circuit_breaker_cooldown"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    ASSET_CONCENTRATION = "asset_concentration"
    CORRELATED_EXPOSURE = "correlated_exposure"
    LEVERAGE_LIMIT = "leverage_limit"


@dataclass
class RiskCheckResult:
    """
    Result of a single risk check.

    This is the atomic unit of risk observability.
    Every check tells its story: what it checked, what it found, pass or fail.

    Example:
        RiskCheckResult(
            name=RiskCheckName.DAILY_LOSS_LIMIT,
            passed=False,
            reason="Daily loss exceeds limit",
            details={
                "current_daily_loss_pct": -2.3,
                "daily_loss_limit_pct": -2.0,
                "excess_pct": 0.3
            },
            threshold="2.0%",
            actual="-2.3%"
        )
    """
    name: RiskCheckName
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[str] = None  # Human-readable threshold
    actual: Optional[str] = None     # Human-readable actual value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization / Truth Engine logging."""
        return {
            "name": self.name.value,
            "passed": self.passed,
            "reason": self.reason,
            "threshold": self.threshold,
            "actual": self.actual,
            "details": self.details
        }


@dataclass
class RiskResult:
    """
    The complete result of risk evaluation.

    This replaces the old boolean return. Now we know:
    - approved: Can we trade? (the old boolean)
    - rejection_reason: If no, why? (human-readable)
    - checks: Full breakdown of every check performed
    - first_failure: Which check failed first (if any)

    The Glass Box Promise:
    "Why was this trade blocked?" -> RiskResult tells you exactly.
    "Why was this trade allowed?" -> RiskResult shows all checks passed.

    Usage:
        result = risk_manager.evaluate(trade_request)
        if not result.approved:
            print(f"Trade blocked: {result.rejection_reason}")
            print(f"Failed check: {result.first_failure.name.value}")
            for check in result.checks:
                print(f"  {check.name.value}: {'PASS' if check.passed else 'FAIL'}")
    """
    approved: bool
    rejection_reason: Optional[str]
    checks: List[RiskCheckResult]
    first_failure: Optional[RiskCheckResult] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def all_passed(self) -> bool:
        """True if all checks passed."""
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> List[RiskCheckResult]:
        """List of checks that failed."""
        return [check for check in self.checks if not check.passed]

    @property
    def passed_checks(self) -> List[RiskCheckResult]:
        """List of checks that passed."""
        return [check for check in self.checks if check.passed]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict for Truth Engine logging.

        This is THE bridge to TruthLogger.log_decision(risk_checks=...).
        """
        return {
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_checks": len(self.checks),
                "passed": len(self.passed_checks),
                "failed": len(self.failed_checks)
            },
            "first_failure": self.first_failure.to_dict() if self.first_failure else None,
            "checks": {check.name.value: check.to_dict() for check in self.checks}
        }

    @classmethod
    def approved_result(cls, checks: List[RiskCheckResult]) -> "RiskResult":
        """Factory for approved trades - all checks passed."""
        return cls(
            approved=True,
            rejection_reason=None,
            checks=checks,
            first_failure=None
        )

    @classmethod
    def rejected_result(
        cls,
        checks: List[RiskCheckResult],
        first_failure: RiskCheckResult
    ) -> "RiskResult":
        """Factory for rejected trades - at least one check failed."""
        return cls(
            approved=False,
            rejection_reason=first_failure.reason,
            checks=checks,
            first_failure=first_failure
        )


@dataclass
class TradeRequest:
    """
    Input to the risk manager - what trade are we evaluating?

    Contains everything the risk manager needs to evaluate a trade.
    This is passed INTO risk evaluation, RiskResult comes OUT.
    """
    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    entry_price: Decimal
    stop_loss_price: Decimal
    take_profit_price: Decimal
    strategy_name: str
    confidence: float = 0.0
    is_exit: bool = False  # Explicit flag for exit orders

    @property
    def risk_amount(self) -> Decimal:
        """Dollar risk if stop loss hit."""
        if self.side == "buy":
            return (self.entry_price - self.stop_loss_price) * self.quantity
        else:
            return (self.stop_loss_price - self.entry_price) * self.quantity

    @property
    def reward_amount(self) -> Decimal:
        """Dollar reward if take profit hit."""
        if self.side == "buy":
            return (self.take_profit_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.take_profit_price) * self.quantity

    @property
    def risk_reward_ratio(self) -> float:
        """R:R ratio (reward / risk)."""
        if self.risk_amount == 0:
            return 0.0
        return float(self.reward_amount / self.risk_amount)

    @property
    def position_value(self) -> Decimal:
        """Total position value."""
        return self.entry_price * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "stop_loss_price": str(self.stop_loss_price),
            "take_profit_price": str(self.take_profit_price),
            "strategy_name": self.strategy_name,
            "confidence": self.confidence,
            "calculated": {
                "risk_amount": str(self.risk_amount),
                "reward_amount": str(self.reward_amount),
                "risk_reward_ratio": round(self.risk_reward_ratio, 2),
                "position_value": str(self.position_value)
            }
        }


@dataclass
class PortfolioState:
    """
    Current portfolio state for risk calculations.

    The risk manager needs to know current exposure to make decisions.
    This is passed alongside TradeRequest.
    """
    total_capital: Decimal
    cash_available: Decimal
    daily_pnl: Decimal
    daily_pnl_percent: float
    total_pnl: Decimal
    total_pnl_percent: float  # From high water mark
    open_positions: Dict[str, Any]  # symbol -> position details
    recent_trades_count: int  # Trades in last hour
    circuit_breaker_triggered: bool = False
    circuit_breaker_triggered_at: Optional[datetime] = None
    trading_halted: bool = False

    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len(self.open_positions)

    @property
    def total_exposure(self) -> Decimal:
        """Total $ value of open positions."""
        return sum(
            Decimal(str(pos.get("value", 0)))
            for pos in self.open_positions.values()
        )

    @property
    def exposure_percent(self) -> float:
        """Exposure as % of capital."""
        if self.total_capital == 0:
            return 0.0
        return float(self.total_exposure / self.total_capital * 100)

    def get_asset_exposure(self, symbol: str) -> Decimal:
        """Get exposure for a specific asset."""
        if symbol in self.open_positions:
            return Decimal(str(self.open_positions[symbol].get("value", 0)))
        return Decimal("0")

    def get_asset_exposure_percent(self, symbol: str) -> float:
        """Get exposure for a specific asset as % of capital."""
        if self.total_capital == 0:
            return 0.0
        return float(self.get_asset_exposure(symbol) / self.total_capital * 100)
