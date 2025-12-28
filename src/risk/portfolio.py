"""
Portfolio Risk Aggregator - Cross-Asset Risk Coordination

Enforces portfolio-level limits that individual RiskManagers cannot see:
- Total portfolio exposure
- Correlation group limits (BTC+ETH together)
- Combined daily loss limit across all assets
- Cross-asset circuit breaker

This is Layer 0 - runs BEFORE individual asset risk checks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set
import logging


logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskConfig:
    """Configuration for portfolio-level risk limits."""
    # Exposure limits
    max_portfolio_exposure_pct: float = 80.0      # Max total exposure
    max_correlation_group_pct: float = 50.0       # Max per correlation group

    # Loss limits
    max_combined_daily_loss_pct: float = 5.0      # Combined daily loss
    cross_asset_circuit_breaker_pct: float = 10.0 # Portfolio drop triggers halt

    # Circuit breaker settings
    circuit_breaker_window_minutes: int = 60      # Lookback window
    circuit_breaker_cooldown_minutes: int = 30    # Cooldown after trigger

    # Correlation groups - assets that move together
    correlation_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "large_cap": ["BTC-USD", "ETH-USD"],
        "layer2": ["SOL-USD", "MATIC-USD"],
        "defi": ["UNI-USD", "AAVE-USD"],
    })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioRiskConfig":
        """Create from config dictionary."""
        return cls(
            max_portfolio_exposure_pct=data.get("max_portfolio_exposure_pct", 80.0),
            max_correlation_group_pct=data.get("max_correlation_group_pct", 50.0),
            max_combined_daily_loss_pct=data.get("max_combined_daily_loss_pct", 5.0),
            cross_asset_circuit_breaker_pct=data.get("cross_asset_circuit_breaker_pct", 10.0),
            circuit_breaker_window_minutes=data.get("circuit_breaker_window_minutes", 60),
            circuit_breaker_cooldown_minutes=data.get("circuit_breaker_cooldown_minutes", 30),
            correlation_groups=data.get("correlation_groups", {
                "large_cap": ["BTC-USD", "ETH-USD"]
            })
        )


@dataclass
class AssetRiskState:
    """Per-asset risk tracking."""
    symbol: str
    position_value: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl_today: Decimal = Decimal("0")
    trades_today: int = 0
    last_update: Optional[datetime] = None

    @property
    def total_pnl_today(self) -> Decimal:
        """Total P&L including unrealized."""
        return self.realized_pnl_today + self.unrealized_pnl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "symbol": self.symbol,
            "position_value": str(self.position_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl_today": str(self.realized_pnl_today),
            "total_pnl_today": str(self.total_pnl_today),
            "trades_today": self.trades_today,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }


@dataclass
class PortfolioRiskCheckResult:
    """Result of a portfolio-level risk check."""
    name: str
    passed: bool
    reason: str
    threshold: Optional[str] = None
    actual: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "name": self.name,
            "passed": self.passed,
            "reason": self.reason,
            "threshold": self.threshold,
            "actual": self.actual,
            "details": self.details
        }


@dataclass
class PortfolioRiskResult:
    """Complete result of portfolio risk evaluation."""
    approved: bool
    rejection_reason: Optional[str]
    checks: List[PortfolioRiskCheckResult]
    first_failure: Optional[PortfolioRiskCheckResult] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for Truth Engine logging."""
        return {
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_checks": len(self.checks),
                "passed": len([c for c in self.checks if c.passed]),
                "failed": len([c for c in self.checks if not c.passed])
            },
            "first_failure": self.first_failure.to_dict() if self.first_failure else None,
            "checks": {c.name: c.to_dict() for c in self.checks}
        }


class PortfolioRiskAggregator:
    """
    Cross-asset risk coordination.

    Runs BEFORE individual RiskManager checks (Layer 0).
    Enforces portfolio-level limits that span all assets.
    """

    def __init__(
        self,
        config: PortfolioRiskConfig,
        total_capital: Decimal
    ):
        """
        Initialize portfolio risk aggregator.

        Args:
            config: Portfolio risk configuration.
            total_capital: Total portfolio capital.
        """
        self.config = config
        self.total_capital = total_capital

        # Per-asset state tracking
        self.asset_states: Dict[str, AssetRiskState] = {}

        # Portfolio-level state
        self.high_water_mark: Decimal = total_capital
        self.portfolio_value_history: List[tuple] = []  # (timestamp, value)
        self.circuit_breaker_triggered: bool = False
        self.circuit_breaker_triggered_at: Optional[datetime] = None
        self.trading_halted: bool = False

        # Daily tracking (reset at session_reset_hour)
        self.daily_start_value: Decimal = total_capital
        self.daily_reset_time: Optional[datetime] = None

    def update_position(
        self,
        symbol: str,
        position_value: Decimal,
        unrealized_pnl: Decimal = Decimal("0"),
        realized_pnl: Optional[Decimal] = None
    ) -> None:
        """
        Update position state for an asset.

        Args:
            symbol: Trading pair.
            position_value: Current position value.
            unrealized_pnl: Unrealized P&L.
            realized_pnl: Realized P&L to add (None = no change).
        """
        now = datetime.utcnow()

        if symbol not in self.asset_states:
            self.asset_states[symbol] = AssetRiskState(symbol=symbol)

        state = self.asset_states[symbol]
        state.position_value = position_value
        state.unrealized_pnl = unrealized_pnl
        state.last_update = now

        if realized_pnl is not None:
            state.realized_pnl_today += realized_pnl
            state.trades_today += 1

        # Update portfolio value history
        total_value = self.get_portfolio_value()
        self.portfolio_value_history.append((now, total_value))

        # Trim history to window
        cutoff = now - timedelta(minutes=self.config.circuit_breaker_window_minutes)
        self.portfolio_value_history = [
            (t, v) for t, v in self.portfolio_value_history if t >= cutoff
        ]

        # Update high water mark
        if total_value > self.high_water_mark:
            self.high_water_mark = total_value

    def get_portfolio_value(self) -> Decimal:
        """Get current total portfolio value."""
        return self.total_capital + sum(
            s.unrealized_pnl for s in self.asset_states.values()
        )

    def get_total_exposure(self) -> Decimal:
        """Get total position exposure across all assets."""
        return sum(
            abs(s.position_value) for s in self.asset_states.values()
        )

    def get_exposure_percent(self) -> float:
        """Get total exposure as percentage of capital."""
        if self.total_capital == 0:
            return 0.0
        return float(self.get_total_exposure() / self.total_capital * 100)

    def get_correlation_group(self, symbol: str) -> Optional[str]:
        """Find which correlation group a symbol belongs to."""
        for group_name, symbols in self.config.correlation_groups.items():
            if symbol in symbols:
                return group_name
        return None

    def get_group_exposure(self, group_name: str) -> Decimal:
        """Get total exposure for a correlation group."""
        symbols = self.config.correlation_groups.get(group_name, [])
        return sum(
            abs(self.asset_states[s].position_value)
            for s in symbols
            if s in self.asset_states
        )

    def get_combined_daily_pnl(self) -> Decimal:
        """Get combined daily P&L across all assets."""
        return sum(s.total_pnl_today for s in self.asset_states.values())

    def get_combined_daily_pnl_percent(self) -> float:
        """Get combined daily P&L as percentage."""
        if self.daily_start_value == 0:
            return 0.0
        return float(self.get_combined_daily_pnl() / self.daily_start_value * 100)

    def can_open_position(
        self,
        symbol: str,
        proposed_value: Decimal
    ) -> PortfolioRiskResult:
        """
        Portfolio-level check before individual risk checks.

        Args:
            symbol: Symbol for proposed trade.
            proposed_value: Value of proposed position.

        Returns:
            PortfolioRiskResult with approval/rejection.
        """
        checks: List[PortfolioRiskCheckResult] = []
        first_failure: Optional[PortfolioRiskCheckResult] = None

        # Check 1: Trading halted
        check = self._check_trading_halted()
        checks.append(check)
        if not check.passed and first_failure is None:
            first_failure = check

        # Check 2: Circuit breaker cooldown
        check = self._check_circuit_breaker_cooldown()
        checks.append(check)
        if not check.passed and first_failure is None:
            first_failure = check

        # Check 3: Total exposure limit
        check = self._check_total_exposure(proposed_value)
        checks.append(check)
        if not check.passed and first_failure is None:
            first_failure = check

        # Check 4: Correlation group limit
        check = self._check_correlation_limit(symbol, proposed_value)
        checks.append(check)
        if not check.passed and first_failure is None:
            first_failure = check

        # Check 5: Combined daily loss limit
        check = self._check_combined_daily_loss()
        checks.append(check)
        if not check.passed and first_failure is None:
            first_failure = check

        # Check 6: Portfolio circuit breaker
        check = self._check_portfolio_circuit_breaker()
        checks.append(check)
        if not check.passed and first_failure is None:
            first_failure = check

        approved = first_failure is None

        return PortfolioRiskResult(
            approved=approved,
            rejection_reason=first_failure.reason if first_failure else None,
            checks=checks,
            first_failure=first_failure
        )

    def _check_trading_halted(self) -> PortfolioRiskCheckResult:
        """Check if trading is halted."""
        return PortfolioRiskCheckResult(
            name="portfolio_trading_halted",
            passed=not self.trading_halted,
            reason="Portfolio trading halted" if self.trading_halted else "Trading active",
            threshold="False",
            actual=str(self.trading_halted)
        )

    def _check_circuit_breaker_cooldown(self) -> PortfolioRiskCheckResult:
        """Check if in circuit breaker cooldown."""
        if not self.circuit_breaker_triggered:
            return PortfolioRiskCheckResult(
                name="portfolio_circuit_breaker_cooldown",
                passed=True,
                reason="No circuit breaker active"
            )

        now = datetime.utcnow()
        cooldown_end = self.circuit_breaker_triggered_at + timedelta(
            minutes=self.config.circuit_breaker_cooldown_minutes
        )

        if now >= cooldown_end:
            # Cooldown expired
            self.circuit_breaker_triggered = False
            self.circuit_breaker_triggered_at = None
            return PortfolioRiskCheckResult(
                name="portfolio_circuit_breaker_cooldown",
                passed=True,
                reason="Circuit breaker cooldown expired"
            )

        remaining = (cooldown_end - now).total_seconds() / 60
        return PortfolioRiskCheckResult(
            name="portfolio_circuit_breaker_cooldown",
            passed=False,
            reason=f"Circuit breaker cooldown: {remaining:.0f} minutes remaining",
            threshold=f"{self.config.circuit_breaker_cooldown_minutes} min",
            actual=f"{remaining:.0f} min remaining"
        )

    def _check_total_exposure(self, proposed_value: Decimal) -> PortfolioRiskCheckResult:
        """Check total portfolio exposure limit."""
        current_exposure = self.get_total_exposure()
        new_exposure = current_exposure + abs(proposed_value)
        new_exposure_pct = float(new_exposure / self.total_capital * 100)

        passed = new_exposure_pct <= self.config.max_portfolio_exposure_pct

        return PortfolioRiskCheckResult(
            name="portfolio_total_exposure",
            passed=passed,
            reason=f"Total exposure {'within' if passed else 'exceeds'} limit",
            threshold=f"{self.config.max_portfolio_exposure_pct}%",
            actual=f"{new_exposure_pct:.1f}%",
            details={
                "current_exposure": str(current_exposure),
                "proposed_addition": str(proposed_value),
                "new_total": str(new_exposure)
            }
        )

    def _check_correlation_limit(
        self,
        symbol: str,
        proposed_value: Decimal
    ) -> PortfolioRiskCheckResult:
        """Check correlation group exposure limit."""
        group = self.get_correlation_group(symbol)

        if group is None:
            return PortfolioRiskCheckResult(
                name="portfolio_correlation_limit",
                passed=True,
                reason=f"{symbol} not in any correlation group"
            )

        current_group_exposure = self.get_group_exposure(group)
        new_group_exposure = current_group_exposure + abs(proposed_value)
        new_group_pct = float(new_group_exposure / self.total_capital * 100)

        passed = new_group_pct <= self.config.max_correlation_group_pct

        return PortfolioRiskCheckResult(
            name="portfolio_correlation_limit",
            passed=passed,
            reason=f"Correlation group '{group}' {'within' if passed else 'exceeds'} limit",
            threshold=f"{self.config.max_correlation_group_pct}%",
            actual=f"{new_group_pct:.1f}%",
            details={
                "group": group,
                "group_symbols": self.config.correlation_groups[group],
                "current_group_exposure": str(current_group_exposure),
                "new_group_exposure": str(new_group_exposure)
            }
        )

    def _check_combined_daily_loss(self) -> PortfolioRiskCheckResult:
        """Check combined daily loss limit across all assets."""
        daily_loss_pct = self.get_combined_daily_pnl_percent()

        # Loss is negative, so check if it's worse than limit
        passed = daily_loss_pct >= -self.config.max_combined_daily_loss_pct

        return PortfolioRiskCheckResult(
            name="portfolio_combined_daily_loss",
            passed=passed,
            reason=f"Combined daily loss {'within' if passed else 'exceeds'} limit",
            threshold=f"-{self.config.max_combined_daily_loss_pct}%",
            actual=f"{daily_loss_pct:.2f}%",
            details={
                "combined_pnl": str(self.get_combined_daily_pnl()),
                "by_asset": {
                    s: str(state.total_pnl_today)
                    for s, state in self.asset_states.items()
                }
            }
        )

    def _check_portfolio_circuit_breaker(self) -> PortfolioRiskCheckResult:
        """Check for sudden portfolio value drop."""
        if len(self.portfolio_value_history) < 2:
            return PortfolioRiskCheckResult(
                name="portfolio_circuit_breaker",
                passed=True,
                reason="Insufficient history for circuit breaker check"
            )

        # Get oldest value in window
        oldest_time, oldest_value = self.portfolio_value_history[0]
        current_value = self.get_portfolio_value()

        if oldest_value == 0:
            return PortfolioRiskCheckResult(
                name="portfolio_circuit_breaker",
                passed=True,
                reason="Cannot calculate circuit breaker (zero starting value)"
            )

        change_pct = float((current_value - oldest_value) / oldest_value * 100)

        # Check if drop exceeds threshold
        passed = change_pct >= -self.config.cross_asset_circuit_breaker_pct

        if not passed:
            # Trigger circuit breaker
            self.circuit_breaker_triggered = True
            self.circuit_breaker_triggered_at = datetime.utcnow()
            logger.warning(
                f"PORTFOLIO CIRCUIT BREAKER TRIGGERED: {change_pct:.1f}% drop "
                f"in {self.config.circuit_breaker_window_minutes} minutes"
            )

        return PortfolioRiskCheckResult(
            name="portfolio_circuit_breaker",
            passed=passed,
            reason=f"Portfolio {'stable' if passed else 'dropped significantly'}",
            threshold=f"-{self.config.cross_asset_circuit_breaker_pct}%",
            actual=f"{change_pct:.2f}%",
            details={
                "window_minutes": self.config.circuit_breaker_window_minutes,
                "start_value": str(oldest_value),
                "current_value": str(current_value)
            }
        )

    def reset_daily(self) -> None:
        """Reset daily tracking (call at session_reset_hour)."""
        now = datetime.utcnow()

        for state in self.asset_states.values():
            state.realized_pnl_today = Decimal("0")
            state.trades_today = 0

        self.daily_start_value = self.get_portfolio_value()
        self.daily_reset_time = now

        logger.info(
            f"Daily reset complete. Starting value: {self.daily_start_value}"
        )

    def halt_trading(self, reason: str) -> None:
        """Halt all portfolio trading."""
        self.trading_halted = True
        logger.warning(f"PORTFOLIO TRADING HALTED: {reason}")

    def resume_trading(self) -> None:
        """Resume portfolio trading."""
        self.trading_halted = False
        self.circuit_breaker_triggered = False
        self.circuit_breaker_triggered_at = None
        logger.info("Portfolio trading resumed")

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get full portfolio state for logging."""
        return {
            "total_capital": str(self.total_capital),
            "portfolio_value": str(self.get_portfolio_value()),
            "high_water_mark": str(self.high_water_mark),
            "total_exposure": str(self.get_total_exposure()),
            "exposure_percent": self.get_exposure_percent(),
            "combined_daily_pnl": str(self.get_combined_daily_pnl()),
            "combined_daily_pnl_pct": self.get_combined_daily_pnl_percent(),
            "trading_halted": self.trading_halted,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "asset_count": len(self.asset_states),
            "assets": {s: state.to_dict() for s, state in self.asset_states.items()}
        }
