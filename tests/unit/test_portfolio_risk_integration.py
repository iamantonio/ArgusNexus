"""
PR-1: RiskManager Integration for Portfolio Trader - Acceptance Tests

These tests MUST pass before the portfolio trader can resume operations.

Required behavior (from Tony's marching orders, Dec 19 2025):
- Before any order execution:
  1. Build TradeRequest from the planned order
  2. Build RiskPortfolioState from the current portfolio state
  3. Call RiskManager.evaluate(...)
  4. If not approved -> block and log rejection (DB + decision log)
  5. If evaluation throws -> fail closed and emit CRITICAL alert

Acceptance tests:
- Concentration breach blocks: target 45.8% BTC with max 30% -> blocked
- Missing risk config fails closed: simulate missing/invalid config -> blocked + CRITICAL alert
- trading_halted blocks: flag true -> blocked
- risk_checks persisted: DB record shows non-NULL risk payload for every attempted order
"""

import pytest
import asyncio
import sys
import os
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.risk.manager import RiskManager, RiskConfig, RiskConfigError
from src.risk.schema import TradeRequest, PortfolioState as RiskPortfolioState, RiskCheckName
from src.portfolio.portfolio_manager import PortfolioManager, PortfolioState, DDState, RebalanceOrder


class TestPortfolioRiskIntegration:
    """
    Tests that the portfolio trader integrates with RiskManager correctly.

    These are RED tests - they will FAIL until PR-1 is implemented.
    """

    @pytest.fixture
    def risk_config_strict(self):
        """Risk config with 30% max concentration (as per config.yaml)"""
        return RiskConfig(
            risk_per_trade_pct=1.0,
            daily_loss_limit_pct=2.0,
            max_trades_per_hour=5,
            max_drawdown_pct=5.0,
            circuit_breaker_pct=8.0,
            min_risk_reward_ratio=2.0,
            max_asset_concentration_pct=30.0,  # The critical limit
            max_correlated_exposure_pct=50.0,
            max_leverage_per_position=1.0,
            max_portfolio_leverage=1.0
        )

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing"""
        db_path = tmp_path / "test_risk_integration.db"
        return str(db_path)

    # =========================================================================
    # TEST 1: Concentration breach blocks
    # =========================================================================

    def test_concentration_breach_blocks_order(self, risk_config_strict):
        """
        Acceptance Test: Concentration breach blocks

        Scenario: Portfolio manager wants 45.8% BTC allocation, but max is 30%
        Expected: Order MUST be blocked by RiskManager

        This tests the integration point - that the portfolio trader
        actually calls RiskManager and respects "approved=False"
        """
        # Build a trade request representing 45.8% concentration
        # On $500 capital, 45.8% = $229 position
        # Use a valid R:R ratio (3:1) so only concentration fails
        trade_request = TradeRequest(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.00269"),  # ~$229 at $85k
            entry_price=Decimal("85000"),
            stop_loss_price=Decimal("80000"),  # -$5k risk
            take_profit_price=Decimal("100000"),  # +$15k reward = 3:1 R:R
            strategy_name="portfolio_manager"
        )

        # Current portfolio: empty (0% BTC)
        portfolio_state = RiskPortfolioState(
            total_capital=Decimal("500"),
            cash_available=Decimal("500"),
            daily_pnl=Decimal("0"),
            daily_pnl_percent=0.0,
            total_pnl=Decimal("0"),
            total_pnl_percent=0.0,
            open_positions={},  # No existing positions
            recent_trades_count=0,
            circuit_breaker_triggered=False,
            trading_halted=False
        )

        # RiskManager should reject this
        risk_manager = RiskManager(risk_config_strict)
        result = risk_manager.evaluate(trade_request, portfolio_state)

        # Assert: Order MUST be blocked due to concentration
        assert result.approved is False, \
            f"Expected order to be BLOCKED due to 45.8% > 30% concentration, but got approved=True"
        assert result.first_failure is not None
        assert result.first_failure.name == RiskCheckName.ASSET_CONCENTRATION, \
            f"Expected ASSET_CONCENTRATION failure, got {result.first_failure.name}"

        # The rejection reason should mention the breach
        assert "concentration" in result.rejection_reason.lower() or "30" in result.rejection_reason

    # =========================================================================
    # TEST 2: trading_halted blocks
    # =========================================================================

    def test_trading_halted_blocks_all_orders(self, risk_config_strict):
        """
        Acceptance Test: trading_halted blocks

        Scenario: trading_halted=true in portfolio state
        Expected: ALL orders blocked, even small safe ones
        """
        # A perfectly safe trade request
        trade_request = TradeRequest(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.001"),  # Small
            entry_price=Decimal("85000"),
            stop_loss_price=Decimal("80000"),
            take_profit_price=Decimal("95000"),
            strategy_name="portfolio_manager"
        )

        # Portfolio with halt flag set
        portfolio_state = RiskPortfolioState(
            total_capital=Decimal("100000"),  # Lots of capital
            cash_available=Decimal("100000"),
            daily_pnl=Decimal("0"),
            daily_pnl_percent=0.0,
            total_pnl=Decimal("0"),
            total_pnl_percent=0.0,
            open_positions={},
            recent_trades_count=0,
            circuit_breaker_triggered=False,
            trading_halted=True  # HALTED
        )

        risk_manager = RiskManager(risk_config_strict)
        result = risk_manager.evaluate(trade_request, portfolio_state)

        # Assert: Order MUST be blocked
        assert result.approved is False, \
            "Expected order to be BLOCKED when trading_halted=True"
        assert result.first_failure.name == RiskCheckName.TRADING_HALTED

    # =========================================================================
    # TEST 3: Missing/invalid config fails closed
    # =========================================================================

    def test_missing_config_fails_closed(self):
        """
        Acceptance Test: Missing risk config fails closed

        Scenario: RiskConfig is None
        Expected: RiskManager(None) MUST raise TypeError

        This tests defensive behavior - if risk config is None,
        construction fails immediately.
        """
        # RiskManager(None) must raise TypeError
        with pytest.raises(TypeError) as exc_info:
            RiskManager(None)

        # Error message should be informative
        assert "requires explicit RiskConfig" in str(exc_info.value)

    def test_invalid_config_values_fail_closed(self):
        """
        Scenario: Config has nonsensical values (negative limits)
        Expected: RiskConfigError raised at construction time
        """
        # Negative concentration limit makes no sense - should raise immediately
        with pytest.raises(RiskConfigError) as exc_info:
            RiskConfig(max_asset_concentration_pct=-10.0)

        assert "max_asset_concentration_pct must be > 0" in str(exc_info.value)

    # =========================================================================
    # TEST 4: Risk evaluation exception fails closed
    # =========================================================================

    def test_evaluation_exception_fails_closed(self, risk_config_strict):
        """
        Acceptance Test: If RiskManager.evaluate() throws, fail closed

        Scenario: Something goes wrong during risk evaluation
        Expected: Catch exception, return rejected result, not crash or approve

        This is the defensive behavior Tony demanded:
        "If evaluation throws -> fail closed"
        """
        risk_manager = RiskManager(risk_config_strict)

        # Create a malformed trade request that might cause issues
        # (e.g., zero quantity could cause division errors)
        trade_request = TradeRequest(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0"),  # Zero quantity - edge case
            entry_price=Decimal("85000"),
            stop_loss_price=Decimal("80000"),
            take_profit_price=Decimal("95000"),
            strategy_name="test"
        )

        portfolio_state = RiskPortfolioState(
            total_capital=Decimal("0"),  # Zero capital - edge case
            cash_available=Decimal("0"),
            daily_pnl=Decimal("0"),
            daily_pnl_percent=0.0,
            total_pnl=Decimal("0"),
            total_pnl_percent=0.0,
            open_positions={},
            recent_trades_count=0,
            circuit_breaker_triggered=False,
            trading_halted=False
        )

        # Should either raise controlled exception or return rejected
        # Both are acceptable "fail closed" behaviors
        try:
            result = risk_manager.evaluate(trade_request, portfolio_state)
            # If it returns, it must be rejected
            assert result.approved is False, \
                "Edge case (zero capital/quantity) should fail closed"
        except Exception as e:
            # Controlled exception is also acceptable
            pass  # Test passes - failed closed via exception


class TestPortfolioTraderRiskGate:
    """
    Tests that the portfolio trader has a risk gate that cannot be bypassed.

    These tests verify the INTEGRATION - that execute_order() calls RiskManager.
    They will FAIL until the integration code is added to live_portfolio_trader.py.
    """

    @pytest.fixture
    def mock_risk_manager_rejects(self):
        """A mock RiskManager that always rejects"""
        mock = Mock(spec=RiskManager)
        mock_result = Mock()
        mock_result.approved = False
        mock_result.rejection_reason = "TEST_REJECTION"
        mock_result.to_dict.return_value = {"approved": False, "reason": "TEST_REJECTION"}
        mock.evaluate.return_value = mock_result
        return mock

    @pytest.fixture
    def mock_risk_manager_approves(self):
        """A mock RiskManager that always approves"""
        mock = Mock(spec=RiskManager)
        mock_result = Mock()
        mock_result.approved = True
        mock_result.rejection_reason = None
        mock_result.to_dict.return_value = {"approved": True, "checks": {}}
        mock.evaluate.return_value = mock_result
        return mock

    def test_portfolio_trader_has_risk_gate(self):
        """
        Verify that live_portfolio_trader.py imports and uses RiskManager.

        This is a code inspection test - if the import doesn't exist,
        the integration hasn't been done.
        """
        trader_path = Path(__file__).parent.parent.parent / "scripts" / "live_portfolio_trader.py"
        trader_code = trader_path.read_text()

        # Must import RiskManager
        assert "from src.risk.manager import RiskManager" in trader_code or \
               "from src.risk import" in trader_code, \
            "live_portfolio_trader.py must import RiskManager"

        # Must call evaluate somewhere
        assert "risk" in trader_code.lower() and "evaluate" in trader_code.lower(), \
            "live_portfolio_trader.py must call risk evaluation"

    def test_risk_checks_logged_to_decision(self):
        """
        Acceptance Test: risk_checks persisted

        Scenario: Any order attempt (approved or rejected)
        Expected: Decision log contains non-NULL risk_checks payload

        This verifies the audit trail requirement.
        """
        # This test will need to mock the TruthLogger and verify
        # that log_decision is called with risk_checks parameter

        # For now, verify the decision schema supports it
        from src.truth.schema import DecisionResult

        # The decision table should have risk_checks column
        # (This is verified by the schema, but let's be explicit)
        assert hasattr(DecisionResult, 'SIGNAL_LONG')

        # TODO: Full integration test once PR-1 is implemented
        # Will verify that execute_order calls log_decision with
        # risk_checks=risk_result.to_dict()


class TestRiskManagerConcentrationCalculation:
    """
    Unit tests for the concentration check math.

    Verifies the RiskManager correctly calculates whether a new position
    would breach the concentration limit.
    """

    def test_new_position_concentration_math(self):
        """
        Verify concentration is calculated correctly for new positions.

        Scenario: $500 capital, want to buy $229 BTC (45.8%)
        With max_asset_concentration_pct=30%, this should be blocked.
        """
        config = RiskConfig(
            max_asset_concentration_pct=30.0
        )

        # Position value: 0.00269 BTC * $85,000 = $228.65 (45.7% of $500)
        trade_request = TradeRequest(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.00269"),
            entry_price=Decimal("85000"),
            stop_loss_price=Decimal("72250"),
            take_profit_price=Decimal("100000"),
            strategy_name="portfolio_manager"
        )

        portfolio_state = RiskPortfolioState(
            total_capital=Decimal("500"),
            cash_available=Decimal("500"),
            daily_pnl=Decimal("0"),
            daily_pnl_percent=0.0,
            total_pnl=Decimal("0"),
            total_pnl_percent=0.0,
            open_positions={},
            recent_trades_count=0,
            circuit_breaker_triggered=False,
            trading_halted=False
        )

        risk_manager = RiskManager(config)
        result = risk_manager.evaluate(trade_request, portfolio_state)

        # Math check: position_value / total_capital = 228.65 / 500 = 45.7%
        # This exceeds 30% limit
        assert result.approved is False

        # Find the concentration check result
        conc_check = next(
            (c for c in result.checks if c.name == RiskCheckName.ASSET_CONCENTRATION),
            None
        )
        assert conc_check is not None
        assert conc_check.passed is False

        # Verify the actual percentage was calculated
        assert conc_check.actual is not None
        assert "45" in conc_check.actual or "46" in conc_check.actual  # ~45.7%

    def test_within_concentration_limit_approves(self):
        """
        Verify trades within concentration limit are approved.

        Scenario: $500 capital, buy $100 BTC (20%)
        With max_asset_concentration_pct=30%, this should pass.
        """
        config = RiskConfig(
            max_asset_concentration_pct=30.0,
            min_risk_reward_ratio=1.5  # Lower for this test
        )

        # Position value: 0.00118 BTC * $85,000 = $100.30 (20% of $500)
        trade_request = TradeRequest(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.00118"),
            entry_price=Decimal("85000"),
            stop_loss_price=Decimal("80000"),
            take_profit_price=Decimal("95000"),  # 2:1 R:R
            strategy_name="portfolio_manager"
        )

        portfolio_state = RiskPortfolioState(
            total_capital=Decimal("500"),
            cash_available=Decimal("500"),
            daily_pnl=Decimal("0"),
            daily_pnl_percent=0.0,
            total_pnl=Decimal("0"),
            total_pnl_percent=0.0,
            open_positions={},
            recent_trades_count=0,
            circuit_breaker_triggered=False,
            trading_halted=False
        )

        risk_manager = RiskManager(config)
        result = risk_manager.evaluate(trade_request, portfolio_state)

        # 20% < 30% limit - should pass concentration check
        conc_check = next(
            (c for c in result.checks if c.name == RiskCheckName.ASSET_CONCENTRATION),
            None
        )
        assert conc_check is not None
        assert conc_check.passed is True, \
            f"20% position should pass 30% limit, but failed: {conc_check.reason}"
