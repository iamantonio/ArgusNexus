"""
V4 Risk Manager - Observable 10-Layer Risk System

Harvested from V3's src/trading/risk.py and refactored for observability.
Every check now returns RiskCheckResult instead of bool.

The Glass Box Promise:
- Every "No" is explainable
- Every "Yes" shows all checks passed
- Full audit trail for the Truth Engine

Usage:
    risk_manager = RiskManager(config)
    result = risk_manager.evaluate(trade_request, portfolio_state)

    if result.approved:
        # Proceed with trade
        logger.log_decision(risk_checks=result.to_dict())
    else:
        # Trade blocked
        print(f"Blocked by: {result.first_failure.name.value}")
        print(f"Reason: {result.rejection_reason}")
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .schema import (
    PortfolioState,
    RiskCheckName,
    RiskCheckResult,
    RiskResult,
    TradeRequest,
)

logger = logging.getLogger(__name__)


import yaml
from pathlib import Path

class RiskConfigError(Exception):
    """Raised when risk configuration is invalid or missing."""
    pass


@dataclass
class RiskConfig:
    """
    Risk configuration - all thresholds in one place.

    FAIL CLOSED PRINCIPLE:
    - Missing config file → RiskConfigError (not defaults)
    - Invalid values → RiskConfigError (not silent correction)
    - Parse errors → RiskConfigError (not empty config)

    Defaults exist ONLY for unit tests with explicit construction.
    Production code MUST use load_from_yaml() which validates.
    """
    risk_per_trade_pct: float = 1.0
    daily_loss_limit_pct: float = 2.0
    max_trades_per_hour: int = 5
    max_drawdown_pct: float = 5.0
    circuit_breaker_pct: float = 8.0
    circuit_breaker_window_minutes: int = 60
    circuit_breaker_cooldown_minutes: int = 30
    min_risk_reward_ratio: float = 2.0
    max_asset_concentration_pct: float = 30.0
    max_correlated_exposure_pct: float = 50.0
    max_leverage_per_position: float = 1.0
    max_portfolio_leverage: float = 1.0

    def __post_init__(self):
        """Validate config values - fail closed on invalid."""
        self._validate()

    def _validate(self):
        """
        Validate all config values are sane.

        Raises RiskConfigError on any invalid value.
        """
        errors = []

        # All percentage limits must be positive
        if self.risk_per_trade_pct <= 0:
            errors.append(f"risk_per_trade_pct must be > 0, got {self.risk_per_trade_pct}")
        if self.daily_loss_limit_pct <= 0:
            errors.append(f"daily_loss_limit_pct must be > 0, got {self.daily_loss_limit_pct}")
        if self.max_drawdown_pct <= 0:
            errors.append(f"max_drawdown_pct must be > 0, got {self.max_drawdown_pct}")
        if self.circuit_breaker_pct <= 0:
            errors.append(f"circuit_breaker_pct must be > 0, got {self.circuit_breaker_pct}")
        if self.max_asset_concentration_pct <= 0:
            errors.append(f"max_asset_concentration_pct must be > 0, got {self.max_asset_concentration_pct}")
        if self.max_correlated_exposure_pct <= 0:
            errors.append(f"max_correlated_exposure_pct must be > 0, got {self.max_correlated_exposure_pct}")

        # Leverage limits must be positive
        if self.max_leverage_per_position <= 0:
            errors.append(f"max_leverage_per_position must be > 0, got {self.max_leverage_per_position}")
        if self.max_portfolio_leverage <= 0:
            errors.append(f"max_portfolio_leverage must be > 0, got {self.max_portfolio_leverage}")

        # R:R must be positive (0 is invalid)
        if self.min_risk_reward_ratio <= 0:
            errors.append(f"min_risk_reward_ratio must be > 0, got {self.min_risk_reward_ratio}")

        # Trade limits must be positive integers
        if self.max_trades_per_hour <= 0:
            errors.append(f"max_trades_per_hour must be > 0, got {self.max_trades_per_hour}")
        if self.circuit_breaker_window_minutes <= 0:
            errors.append(f"circuit_breaker_window_minutes must be > 0, got {self.circuit_breaker_window_minutes}")
        if self.circuit_breaker_cooldown_minutes <= 0:
            errors.append(f"circuit_breaker_cooldown_minutes must be > 0, got {self.circuit_breaker_cooldown_minutes}")

        # Concentration can't exceed 100%
        if self.max_asset_concentration_pct > 100:
            errors.append(f"max_asset_concentration_pct must be <= 100, got {self.max_asset_concentration_pct}")
        if self.max_correlated_exposure_pct > 100:
            errors.append(f"max_correlated_exposure_pct must be <= 100, got {self.max_correlated_exposure_pct}")

        if errors:
            raise RiskConfigError(f"Invalid risk configuration: {'; '.join(errors)}")

    @classmethod
    def load_from_yaml(cls, path: str = "config.yaml") -> 'RiskConfig':
        """
        Load risk settings from YAML file.

        FAIL CLOSED: Raises RiskConfigError if:
        - File doesn't exist
        - File can't be parsed
        - 'risk' section missing
        - Any value is invalid

        This is the ONLY way production code should load config.
        """
        config_path = Path(path)

        # FAIL CLOSED: Missing file is an error, not "use defaults"
        if not config_path.exists():
            raise RiskConfigError(
                f"Risk config file not found: {path}. "
                f"Cannot run without explicit risk configuration."
            )

        try:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RiskConfigError(f"Failed to parse risk config {path}: {e}")

        if full_config is None:
            raise RiskConfigError(f"Risk config file is empty: {path}")

        risk_data = full_config.get('risk')
        if risk_data is None:
            raise RiskConfigError(
                f"No 'risk' section in config file: {path}. "
                f"Risk configuration is required."
            )

        # Let __post_init__ validate the values
        try:
            return cls(**risk_data)
        except TypeError as e:
            raise RiskConfigError(f"Invalid risk config structure: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "max_trades_per_hour": self.max_trades_per_hour,
            "max_drawdown_pct": self.max_drawdown_pct,
            "circuit_breaker_pct": self.circuit_breaker_pct,
            "circuit_breaker_window_minutes": self.circuit_breaker_window_minutes,
            "circuit_breaker_cooldown_minutes": self.circuit_breaker_cooldown_minutes,
            "min_risk_reward_ratio": self.min_risk_reward_ratio,
            "max_asset_concentration_pct": self.max_asset_concentration_pct,
            "max_correlated_exposure_pct": self.max_correlated_exposure_pct,
            "max_leverage_per_position": self.max_leverage_per_position,
            "max_portfolio_leverage": self.max_portfolio_leverage
        }


class RiskManager:
    """
    The 10-Layer Risk System - Observable Edition.

    Each layer is a guardian. All must approve for a trade to proceed.
    Every check returns full context for the Glass Box.

    Layer Order (fail-fast):
    1. Trading Halted - Is trading paused?
    2. Trade Frequency - Too many trades recently?
    3. Daily Loss Limit - Have we lost too much today?
    4. Drawdown Limit - Are we in significant drawdown?
    5. Circuit Breaker - Did market move too fast?
    6. Circuit Breaker Cooldown - Are we in cooldown period?
    7. Risk/Reward Ratio - Is the trade worth the risk?
    8. Asset Concentration - Too much in one asset?
    9. Correlated Exposure - Too much in correlated assets?
    10. Leverage Limit - Within leverage bounds?
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize with configuration.

        FAIL CLOSED: Config is REQUIRED, not optional.
        Passing None will raise TypeError (no default fallback).

        Production code must explicitly load config via RiskConfig.load_from_yaml().
        """
        if config is None:
            raise TypeError(
                "RiskManager requires explicit RiskConfig. "
                "Use RiskConfig.load_from_yaml() to load from file, "
                "or RiskConfig(...) with explicit values for tests."
            )
        self.config = config
        self._correlation_groups = self._build_correlation_groups()

    def _build_correlation_groups(self) -> Dict[str, List[str]]:
        """
        Define which assets are correlated.
        Correlated assets share risk - if one dumps, they all dump.
        """
        return {
            "btc_ecosystem": ["BTC-USD", "WBTC-USD"],
            "eth_ecosystem": ["ETH-USD", "STETH-USD"],
            "large_cap": ["BTC-USD", "ETH-USD"],
            "defi": ["AAVE-USD", "UNI-USD", "LINK-USD"],
            "layer2": ["MATIC-USD", "OP-USD", "ARB-USD"],
            "meme": ["DOGE-USD", "SHIB-USD"],
        }

    def evaluate(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskResult:
        """
        Run all 10 risk checks and return complete result.

        This is THE entry point for risk evaluation.
        Returns RiskResult with full observability.

        Args:
            request: The trade being evaluated
            portfolio: Current portfolio state

        Returns:
            RiskResult with approved/rejected status and all check details
        """
        checks: List[RiskCheckResult] = []
        first_failure: Optional[RiskCheckResult] = None

        # Run checks in order - fail fast but capture all for logging
        check_methods = [
            (RiskCheckName.TRADING_HALTED, self._check_trading_halted),
            (RiskCheckName.TRADE_FREQUENCY, self._check_trade_frequency),
            (RiskCheckName.DAILY_LOSS_LIMIT, self._check_daily_loss_limit),
            (RiskCheckName.DRAWDOWN_LIMIT, self._check_drawdown_limit),
            (RiskCheckName.CIRCUIT_BREAKER, self._check_circuit_breaker),
            (RiskCheckName.CIRCUIT_BREAKER_COOLDOWN, self._check_circuit_breaker_cooldown),
            (RiskCheckName.RISK_REWARD_RATIO, self._check_risk_reward_ratio),
            (RiskCheckName.ASSET_CONCENTRATION, self._check_asset_concentration),
            (RiskCheckName.CORRELATED_EXPOSURE, self._check_correlated_exposure),
            (RiskCheckName.LEVERAGE_LIMIT, self._check_leverage_limit),
        ]

        for check_name, check_method in check_methods:
            # EXIT BYPASS LOGIC (Suggestion 3)
            # Defensive exits should bypass opening-oriented filters
            if request.is_exit and check_name not in [RiskCheckName.TRADING_HALTED]:
                result = RiskCheckResult(
                    name=check_name,
                    passed=True,
                    reason="Bypassing for EXIT order",
                    details={"is_exit": True},
                    threshold="N/A",
                    actual="EXIT"
                )
            else:
                result = check_method(request, portfolio)
            
            checks.append(result)

            # Track first failure
            if not result.passed and first_failure is None:
                first_failure = result
                logger.warning(
                    f"Risk check FAILED: {check_name.value} - {result.reason}"
                )

        # Build final result
        if first_failure:
            return RiskResult.rejected_result(checks, first_failure)
        else:
            logger.info(
                f"All risk checks PASSED for {request.symbol} {request.side}"
            )
            return RiskResult.approved_result(checks)

    # =========================================================================
    # Layer 1: Trading Halted
    # =========================================================================
    def _check_trading_halted(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 1: Is trading halted?

        Trading can be halted manually or by system events.
        When halted, no new positions can be opened.
        """
        if portfolio.trading_halted:
            return RiskCheckResult(
                name=RiskCheckName.TRADING_HALTED,
                passed=False,
                reason="Trading is currently halted",
                details={
                    "halted": True,
                    "message": "Trading has been manually halted or paused by system"
                },
                threshold="Trading must be active",
                actual="Trading halted"
            )

        return RiskCheckResult(
            name=RiskCheckName.TRADING_HALTED,
            passed=True,
            reason="Trading is active",
            details={"halted": False},
            threshold="Trading must be active",
            actual="Trading active"
        )

    # =========================================================================
    # Layer 2: Trade Frequency
    # =========================================================================
    def _check_trade_frequency(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 2: Are we trading too frequently?

        Prevents overtrading and excessive fees.
        Also helps avoid emotional/revenge trading.
        """
        recent_trades = portfolio.recent_trades_count
        max_trades = self.config.max_trades_per_hour

        if recent_trades >= max_trades:
            return RiskCheckResult(
                name=RiskCheckName.TRADE_FREQUENCY,
                passed=False,
                reason=f"Trade frequency limit reached: {recent_trades}/{max_trades} trades/hour",
                details={
                    "recent_trades": recent_trades,
                    "limit": max_trades,
                    "window": "1 hour"
                },
                threshold=f"Max {max_trades} trades/hour",
                actual=f"{recent_trades} trades in last hour"
            )

        return RiskCheckResult(
            name=RiskCheckName.TRADE_FREQUENCY,
            passed=True,
            reason=f"Trade frequency OK: {recent_trades}/{max_trades} trades/hour",
            details={
                "recent_trades": recent_trades,
                "limit": max_trades,
                "remaining": max_trades - recent_trades
            },
            threshold=f"Max {max_trades} trades/hour",
            actual=f"{recent_trades} trades in last hour"
        )

    # =========================================================================
    # Layer 3: Daily Loss Limit
    # =========================================================================
    def _check_daily_loss_limit(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 3: Have we lost too much today?

        The daily loss limit is a hard stop.
        When hit, we stop trading for the day to prevent tilt.
        """
        daily_loss_pct = portfolio.daily_pnl_percent
        limit_pct = -self.config.daily_loss_limit_pct  # Negative for loss

        if daily_loss_pct <= limit_pct:
            return RiskCheckResult(
                name=RiskCheckName.DAILY_LOSS_LIMIT,
                passed=False,
                reason=f"Daily loss limit exceeded: {daily_loss_pct:.2f}% <= {limit_pct:.2f}%",
                details={
                    "daily_pnl": str(portfolio.daily_pnl),
                    "daily_pnl_pct": daily_loss_pct,
                    "limit_pct": limit_pct,
                    "excess_pct": abs(daily_loss_pct - limit_pct)
                },
                threshold=f"Max {self.config.daily_loss_limit_pct}% daily loss",
                actual=f"{abs(daily_loss_pct):.2f}% loss today"
            )

        return RiskCheckResult(
            name=RiskCheckName.DAILY_LOSS_LIMIT,
            passed=True,
            reason=f"Daily P&L within limit: {daily_loss_pct:.2f}%",
            details={
                "daily_pnl": str(portfolio.daily_pnl),
                "daily_pnl_pct": daily_loss_pct,
                "limit_pct": limit_pct,
                "headroom_pct": abs(limit_pct - daily_loss_pct)
            },
            threshold=f"Max {self.config.daily_loss_limit_pct}% daily loss",
            actual=f"{daily_loss_pct:.2f}% P&L today"
        )

    # =========================================================================
    # Layer 4: Drawdown Limit
    # =========================================================================
    def _check_drawdown_limit(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 4: Are we in significant drawdown?

        Drawdown is measured from high water mark.
        Large drawdowns require reducing risk, not adding to it.
        """
        drawdown_pct = portfolio.total_pnl_percent
        limit_pct = -self.config.max_drawdown_pct  # Negative for drawdown

        if drawdown_pct <= limit_pct:
            return RiskCheckResult(
                name=RiskCheckName.DRAWDOWN_LIMIT,
                passed=False,
                reason=f"Drawdown limit exceeded: {drawdown_pct:.2f}% <= {limit_pct:.2f}%",
                details={
                    "current_drawdown_pct": drawdown_pct,
                    "limit_pct": limit_pct,
                    "total_pnl": str(portfolio.total_pnl),
                    "excess_pct": abs(drawdown_pct - limit_pct)
                },
                threshold=f"Max {self.config.max_drawdown_pct}% drawdown",
                actual=f"{abs(drawdown_pct):.2f}% drawdown"
            )

        return RiskCheckResult(
            name=RiskCheckName.DRAWDOWN_LIMIT,
            passed=True,
            reason=f"Drawdown within limit: {drawdown_pct:.2f}%",
            details={
                "current_drawdown_pct": drawdown_pct,
                "limit_pct": limit_pct,
                "headroom_pct": abs(limit_pct - drawdown_pct)
            },
            threshold=f"Max {self.config.max_drawdown_pct}% drawdown",
            actual=f"{abs(drawdown_pct) if drawdown_pct < 0 else 0:.2f}% drawdown"
        )

    # =========================================================================
    # Layer 5: Circuit Breaker
    # =========================================================================
    def _check_circuit_breaker(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 5: Did the market move too fast?

        Circuit breaker triggers on large moves in short timeframes.
        This protects against flash crashes and extreme volatility.
        """
        if portfolio.circuit_breaker_triggered:
            return RiskCheckResult(
                name=RiskCheckName.CIRCUIT_BREAKER,
                passed=False,
                reason="Circuit breaker is active - market moved too fast",
                details={
                    "triggered": True,
                    "triggered_at": portfolio.circuit_breaker_triggered_at.isoformat()
                        if portfolio.circuit_breaker_triggered_at else None,
                    "threshold_pct": self.config.circuit_breaker_pct,
                    "window_minutes": self.config.circuit_breaker_window_minutes
                },
                threshold=f"No {self.config.circuit_breaker_pct}%+ move in {self.config.circuit_breaker_window_minutes}min",
                actual="Circuit breaker triggered"
            )

        return RiskCheckResult(
            name=RiskCheckName.CIRCUIT_BREAKER,
            passed=True,
            reason="No circuit breaker event detected",
            details={
                "triggered": False,
                "threshold_pct": self.config.circuit_breaker_pct,
                "window_minutes": self.config.circuit_breaker_window_minutes
            },
            threshold=f"No {self.config.circuit_breaker_pct}%+ move in {self.config.circuit_breaker_window_minutes}min",
            actual="Market stable"
        )

    # =========================================================================
    # Layer 6: Circuit Breaker Cooldown
    # =========================================================================
    def _check_circuit_breaker_cooldown(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 6: Are we still in cooldown after circuit breaker?

        After circuit breaker triggers, we wait before trading again.
        This prevents jumping back in during continued volatility.
        """
        if not portfolio.circuit_breaker_triggered_at:
            return RiskCheckResult(
                name=RiskCheckName.CIRCUIT_BREAKER_COOLDOWN,
                passed=True,
                reason="No circuit breaker cooldown active",
                details={"in_cooldown": False},
                threshold=f"{self.config.circuit_breaker_cooldown_minutes}min cooldown after trigger",
                actual="No recent trigger"
            )

        cooldown_end = portfolio.circuit_breaker_triggered_at + timedelta(
            minutes=self.config.circuit_breaker_cooldown_minutes
        )
        now = datetime.utcnow()

        if now < cooldown_end:
            remaining_seconds = (cooldown_end - now).total_seconds()
            remaining_minutes = remaining_seconds / 60

            return RiskCheckResult(
                name=RiskCheckName.CIRCUIT_BREAKER_COOLDOWN,
                passed=False,
                reason=f"Circuit breaker cooldown active: {remaining_minutes:.1f}min remaining",
                details={
                    "in_cooldown": True,
                    "triggered_at": portfolio.circuit_breaker_triggered_at.isoformat(),
                    "cooldown_end": cooldown_end.isoformat(),
                    "remaining_minutes": remaining_minutes
                },
                threshold=f"{self.config.circuit_breaker_cooldown_minutes}min cooldown",
                actual=f"{remaining_minutes:.1f}min remaining"
            )

        return RiskCheckResult(
            name=RiskCheckName.CIRCUIT_BREAKER_COOLDOWN,
            passed=True,
            reason="Circuit breaker cooldown has expired",
            details={
                "in_cooldown": False,
                "triggered_at": portfolio.circuit_breaker_triggered_at.isoformat(),
                "cooldown_end": cooldown_end.isoformat(),
                "expired_ago_minutes": (now - cooldown_end).total_seconds() / 60
            },
            threshold=f"{self.config.circuit_breaker_cooldown_minutes}min cooldown",
            actual="Cooldown expired"
        )

    # =========================================================================
    # Layer 7: Risk/Reward Ratio
    # =========================================================================
    def _check_risk_reward_ratio(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 7: Is this trade worth the risk?

        We only take trades where potential reward exceeds potential risk.
        Default minimum is 2:1 (reward:risk).

        DEVIL'S ADVOCATE: Should we check R:R for exits?
        PUSHBACK: NO. Exits (especially stops) must NEVER be blocked by 
        profitability filters. We only apply this check to OPENING orders.
        """
        if request.is_exit:
            return RiskCheckResult(
                name=RiskCheckName.RISK_REWARD_RATIO,
                passed=True,
                reason="Bypassing R:R check for EXIT order",
                details={"is_exit": True},
                threshold=f"Min {self.config.min_risk_reward_ratio}:1 R:R",
                actual="EXIT (Bypass)"
            )

        rr_ratio = request.risk_reward_ratio
        min_ratio = self.config.min_risk_reward_ratio

        if rr_ratio < min_ratio:
            return RiskCheckResult(
                name=RiskCheckName.RISK_REWARD_RATIO,
                passed=False,
                reason=f"Risk/Reward ratio too low: {rr_ratio:.2f} < {min_ratio:.2f}",
                details={
                    "risk_reward_ratio": rr_ratio,
                    "min_ratio": min_ratio,
                    "risk_amount": str(request.risk_amount),
                    "reward_amount": str(request.reward_amount),
                    "entry_price": str(request.entry_price),
                    "stop_loss": str(request.stop_loss_price),
                    "take_profit": str(request.take_profit_price)
                },
                threshold=f"Min {min_ratio}:1 R:R ratio",
                actual=f"{rr_ratio:.2f}:1 R:R"
            )

        return RiskCheckResult(
            name=RiskCheckName.RISK_REWARD_RATIO,
            passed=True,
            reason=f"Risk/Reward ratio acceptable: {rr_ratio:.2f}",
            details={
                "risk_reward_ratio": rr_ratio,
                "min_ratio": min_ratio,
                "risk_amount": str(request.risk_amount),
                "reward_amount": str(request.reward_amount)
            },
            threshold=f"Min {min_ratio}:1 R:R ratio",
            actual=f"{rr_ratio:.2f}:1 R:R"
        )

    # =========================================================================
    # Layer 8: Asset Concentration
    # =========================================================================
    def _check_asset_concentration(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 8: Are we too concentrated in one asset?

        Don't put all eggs in one basket.
        Max exposure to single asset is configurable (default 30%).
        """
        # Current exposure in this asset
        current_exposure = portfolio.get_asset_exposure(request.symbol)
        new_exposure = current_exposure + request.position_value
        total_capital = portfolio.total_capital

        if total_capital == 0:
            return RiskCheckResult(
                name=RiskCheckName.ASSET_CONCENTRATION,
                passed=False,
                reason="Cannot calculate concentration: zero capital",
                details={"error": "total_capital is zero"},
                threshold=f"Max {self.config.max_asset_concentration_pct}% per asset",
                actual="Unable to calculate"
            )

        new_concentration_pct = float(new_exposure / total_capital * 100)
        max_concentration = self.config.max_asset_concentration_pct

        if new_concentration_pct > max_concentration:
            return RiskCheckResult(
                name=RiskCheckName.ASSET_CONCENTRATION,
                passed=False,
                reason=f"Asset concentration too high: {new_concentration_pct:.1f}% > {max_concentration}%",
                details={
                    "symbol": request.symbol,
                    "current_exposure": str(current_exposure),
                    "new_trade_value": str(request.position_value),
                    "projected_exposure": str(new_exposure),
                    "concentration_pct": new_concentration_pct,
                    "limit_pct": max_concentration,
                    "total_capital": str(total_capital)
                },
                threshold=f"Max {max_concentration}% per asset",
                actual=f"{new_concentration_pct:.1f}% in {request.symbol}"
            )

        return RiskCheckResult(
            name=RiskCheckName.ASSET_CONCENTRATION,
            passed=True,
            reason=f"Asset concentration OK: {new_concentration_pct:.1f}% in {request.symbol}",
            details={
                "symbol": request.symbol,
                "concentration_pct": new_concentration_pct,
                "limit_pct": max_concentration,
                "headroom_pct": max_concentration - new_concentration_pct
            },
            threshold=f"Max {max_concentration}% per asset",
            actual=f"{new_concentration_pct:.1f}% in {request.symbol}"
        )

    # =========================================================================
    # Layer 9: Correlated Exposure
    # =========================================================================
    def _check_correlated_exposure(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 9: Are we too exposed to correlated assets?

        BTC and ETH often move together. DOGE and SHIB move together.
        Total exposure to correlated assets should be limited.
        """
        # Find which correlation groups this asset belongs to
        symbol = request.symbol
        related_groups = []
        for group_name, symbols in self._correlation_groups.items():
            if symbol in symbols:
                related_groups.append((group_name, symbols))

        if not related_groups:
            return RiskCheckResult(
                name=RiskCheckName.CORRELATED_EXPOSURE,
                passed=True,
                reason=f"{symbol} has no defined correlated assets",
                details={
                    "symbol": symbol,
                    "correlated_groups": [],
                    "note": "No correlation group defined for this asset"
                },
                threshold=f"Max {self.config.max_correlated_exposure_pct}% correlated exposure",
                actual="No correlated assets"
            )

        total_capital = portfolio.total_capital
        if total_capital == 0:
            return RiskCheckResult(
                name=RiskCheckName.CORRELATED_EXPOSURE,
                passed=False,
                reason="Cannot calculate correlation exposure: zero capital",
                details={"error": "total_capital is zero"},
                threshold=f"Max {self.config.max_correlated_exposure_pct}%",
                actual="Unable to calculate"
            )

        # Calculate exposure to each correlated group
        max_group_exposure_pct = 0.0
        worst_group = None
        group_details = {}

        for group_name, symbols in related_groups:
            group_exposure = Decimal("0")
            for s in symbols:
                group_exposure += portfolio.get_asset_exposure(s)

            # Add new trade if it's in this group
            new_exposure = group_exposure + request.position_value
            exposure_pct = float(new_exposure / total_capital * 100)
            group_details[group_name] = {
                "symbols": symbols,
                "current_exposure": str(group_exposure),
                "projected_exposure": str(new_exposure),
                "exposure_pct": exposure_pct
            }

            if exposure_pct > max_group_exposure_pct:
                max_group_exposure_pct = exposure_pct
                worst_group = group_name

        max_allowed = self.config.max_correlated_exposure_pct

        if max_group_exposure_pct > max_allowed:
            return RiskCheckResult(
                name=RiskCheckName.CORRELATED_EXPOSURE,
                passed=False,
                reason=f"Correlated exposure too high in {worst_group}: {max_group_exposure_pct:.1f}% > {max_allowed}%",
                details={
                    "symbol": symbol,
                    "worst_group": worst_group,
                    "max_exposure_pct": max_group_exposure_pct,
                    "limit_pct": max_allowed,
                    "groups": group_details
                },
                threshold=f"Max {max_allowed}% correlated exposure",
                actual=f"{max_group_exposure_pct:.1f}% in {worst_group}"
            )

        return RiskCheckResult(
            name=RiskCheckName.CORRELATED_EXPOSURE,
            passed=True,
            reason=f"Correlated exposure OK: max {max_group_exposure_pct:.1f}%",
            details={
                "symbol": symbol,
                "max_exposure_pct": max_group_exposure_pct,
                "limit_pct": max_allowed,
                "groups": group_details
            },
            threshold=f"Max {max_allowed}% correlated exposure",
            actual=f"{max_group_exposure_pct:.1f}% max correlation"
        )

    # =========================================================================
    # Layer 10: Leverage Limit
    # =========================================================================
    def _check_leverage_limit(
        self,
        request: TradeRequest,
        portfolio: PortfolioState
    ) -> RiskCheckResult:
        """
        Layer 10: Are we within leverage limits?

        For V4, we start with no leverage (1x max).
        This check exists for future expansion.
        """
        total_capital = portfolio.total_capital
        if total_capital == 0:
            return RiskCheckResult(
                name=RiskCheckName.LEVERAGE_LIMIT,
                passed=False,
                reason="Cannot calculate leverage: zero capital",
                details={"error": "total_capital is zero"},
                threshold=f"Max {self.config.max_portfolio_leverage}x leverage",
                actual="Unable to calculate"
            )

        # Current total exposure
        current_exposure = portfolio.total_exposure
        new_exposure = current_exposure + request.position_value
        leverage = float(new_exposure / total_capital)

        max_leverage = self.config.max_portfolio_leverage

        if leverage > max_leverage:
            return RiskCheckResult(
                name=RiskCheckName.LEVERAGE_LIMIT,
                passed=False,
                reason=f"Portfolio leverage too high: {leverage:.2f}x > {max_leverage}x",
                details={
                    "current_exposure": str(current_exposure),
                    "new_trade_value": str(request.position_value),
                    "projected_exposure": str(new_exposure),
                    "total_capital": str(total_capital),
                    "leverage": leverage,
                    "max_leverage": max_leverage
                },
                threshold=f"Max {max_leverage}x leverage",
                actual=f"{leverage:.2f}x leverage"
            )

        return RiskCheckResult(
            name=RiskCheckName.LEVERAGE_LIMIT,
            passed=True,
            reason=f"Leverage within limit: {leverage:.2f}x",
            details={
                "leverage": leverage,
                "max_leverage": max_leverage,
                "headroom": max_leverage - leverage
            },
            threshold=f"Max {max_leverage}x leverage",
            actual=f"{leverage:.2f}x leverage"
        )
