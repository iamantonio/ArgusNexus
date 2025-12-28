import pytest
from decimal import Decimal
from datetime import datetime
from src.risk.manager import RiskManager, RiskConfig
from src.risk.schema import TradeRequest, PortfolioState, RiskCheckName

class TestRiskManager:
    @pytest.fixture
    def config(self):
        return RiskConfig(
            risk_per_trade_pct=1.0,
            daily_loss_limit_pct=2.0,
            max_trades_per_hour=5
        )

    @pytest.fixture
    def manager(self, config):
        return RiskManager(config)

    @pytest.fixture
    def trade_request(self):
        return TradeRequest(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            take_profit_price=Decimal("53000"),
            strategy_name="test_strat"
        )

    @pytest.fixture
    def portfolio_state(self):
        return PortfolioState(
            total_capital=Decimal("100000"),
            cash_available=Decimal("100000"),
            daily_pnl=Decimal("0"),
            daily_pnl_percent=0.0,
            total_pnl=Decimal("0"),
            total_pnl_percent=0.0,
            open_positions={},
            recent_trades_count=0,
            circuit_breaker_triggered=False,
            trading_halted=False
        )

    def test_approve_valid_trade(self, manager, trade_request, portfolio_state):
        result = manager.evaluate(trade_request, portfolio_state)
        assert result.approved is True
        assert len(result.checks) == 10
        assert all(c.passed for c in result.checks)

    def test_reject_daily_loss_limit(self, manager, trade_request, portfolio_state):
        portfolio_state.daily_pnl_percent = -3.0  # Exceeds 2% limit
        result = manager.evaluate(trade_request, portfolio_state)
        assert result.approved is False
        assert result.first_failure.name == RiskCheckName.DAILY_LOSS_LIMIT

    def test_reject_trade_frequency(self, manager, trade_request, portfolio_state):
        portfolio_state.recent_trades_count = 6  # Exceeds 5 limit
        result = manager.evaluate(trade_request, portfolio_state)
        assert result.approved is False
        assert result.first_failure.name == RiskCheckName.TRADE_FREQUENCY

    def test_reject_risk_reward_ratio(self, manager, trade_request, portfolio_state):
        trade_request.take_profit_price = Decimal("50500") # 1:0.5 R:R
        result = manager.evaluate(trade_request, portfolio_state)
        assert result.approved is False
        assert result.first_failure.name == RiskCheckName.RISK_REWARD_RATIO

    def test_reject_trading_halted(self, manager, trade_request, portfolio_state):
        portfolio_state.trading_halted = True
        result = manager.evaluate(trade_request, portfolio_state)
        assert result.approved is False
        assert result.first_failure.name == RiskCheckName.TRADING_HALTED

    def test_bypass_rr_for_exits(self, manager, trade_request, portfolio_state):
        trade_request.is_exit = True
        trade_request.take_profit_price = Decimal("50500") # Normally fails 2:1 R:R
        result = manager.evaluate(trade_request, portfolio_state)
        assert result.approved is True
        # Check that RR check passed (bypassed)
        rr_check = next(c for c in result.checks if c.name == RiskCheckName.RISK_REWARD_RATIO)
        assert rr_check.passed is True
        assert "Bypassing" in rr_check.reason
