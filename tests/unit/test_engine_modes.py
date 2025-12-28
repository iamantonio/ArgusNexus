import pytest
import pandas as pd
from decimal import Decimal
from datetime import datetime, timezone
from src.engine import TradingEngine, TradingMode
from src.strategy.donchian import DonchianBreakout, Signal, SignalResult, SignalContext, BreakoutType
from src.risk.manager import RiskManager, RiskConfig
from src.execution.paper import PaperExecutor
from unittest.mock import Mock, AsyncMock

class TestEngineModes:
    """Tests for TradingEngine modes (NORMAL, EXIT_ONLY, HALTED)."""

    @pytest.fixture
    def mock_truth_logger(self):
        logger = Mock()
        logger.log_decision = AsyncMock()
        logger.log_order = AsyncMock()
        logger.open_trade = AsyncMock()
        return logger

    @pytest.fixture
    def engine(self, mock_truth_logger):
        strategy = DonchianBreakout()
        risk_manager = RiskManager(RiskConfig())
        executor = PaperExecutor()
        return TradingEngine(
            strategy=strategy,
            risk_manager=risk_manager,
            executor=executor,
            truth_logger=mock_truth_logger,
            symbol="BTC-USD",
            capital=Decimal("10000")
        )

    @pytest.fixture
    def signal_long(self):
        context = SignalContext(
            timestamp=datetime.now(timezone.utc),
            current_price=Decimal("100000"),
            open_price=Decimal("100000"),
            high_price=Decimal("100000"),
            low_price=Decimal("100000"),
            close_price=Decimal("100000"),
            volume=Decimal("100000"),
            upper_channel=Decimal("90000"),
            lower_channel=Decimal("80000"),
            exit_channel=Decimal("85000"),
            channel_width=Decimal("10000"),
            channel_width_pct=Decimal("10"),
            breakout_type=BreakoutType.UPPER,
            distance_from_upper=Decimal("10000"),
            distance_from_lower=Decimal("20000"),
            atr=Decimal("1000"),
            atr_percent=Decimal("1"),
            stop_loss_price=Decimal("98000"),
            take_profit_price=Decimal("105000"),
            risk_amount=Decimal("2000"),
            reward_amount=Decimal("5000"),
            risk_reward_ratio=Decimal("2.5")
        )
        return SignalResult(signal=Signal.LONG, context=context, reason="Breakout")

    @pytest.fixture
    def signal_exit(self):
        context = SignalContext(
            timestamp=datetime.now(timezone.utc),
            current_price=Decimal("95000"),
            open_price=Decimal("95000"),
            high_price=Decimal("95000"),
            low_price=Decimal("95000"),
            close_price=Decimal("95000"),
            volume=Decimal("100000"),
            upper_channel=Decimal("110000"),
            lower_channel=Decimal("90000"),
            exit_channel=Decimal("96000"),
            channel_width=Decimal("20000"),
            channel_width_pct=Decimal("20"),
            breakout_type=BreakoutType.LOWER,
            distance_from_upper=Decimal("-15000"),
            distance_from_lower=Decimal("5000"),
            atr=Decimal("1000"),
            atr_percent=Decimal("1"),
            stop_loss_price=Decimal("96000"),
            take_profit_price=Decimal("90000"),
            risk_amount=Decimal("1000"),
            reward_amount=Decimal("5000"),
            risk_reward_ratio=Decimal("5.0")
        )
        return SignalResult(signal=Signal.EXIT_LONG, context=context, reason="Stop hit")

    @pytest.mark.asyncio
    async def test_exit_only_blocks_long(self, engine, signal_long):
        """Verify EXIT_ONLY blocks a LONG signal."""
        engine.mode = TradingMode.EXIT_ONLY
        engine.strategy.evaluate = Mock(return_value=signal_long)
        
        # Mock candle data
        df = pd.DataFrame({'close': [100000], 'high': [100000]})
        
        result = await engine.run_tick(df)
        
        assert result.action_taken == "blocked"
        assert "EXIT-ONLY" in str(engine.truth_logger.log_decision.call_args[1]['result_reason'])

    @pytest.mark.asyncio
    async def test_exit_only_allows_exit(self, engine, signal_exit):
        """Verify EXIT_ONLY allows an EXIT signal to proceed."""
        engine.mode = TradingMode.EXIT_ONLY
        engine.has_open_position = True
        engine.entry_price = Decimal("100000")
        engine.strategy.evaluate = Mock(return_value=signal_exit)
        
        # Seed position in executor
        engine.executor._balances["BTC"] = Decimal("0.1")
        engine.executor.get_balance = AsyncMock(side_effect=lambda curr: Decimal("10000") if curr == "USD" else Decimal("0.1"))
        
        df = pd.DataFrame({'close': [95000], 'high': [95000]})
        
        result = await engine.run_tick(df)
        
        # Note: In TradingEngine.run_tick, an EXIT signal actually goes through risk evaluation
        # and execution. If approved, action_taken will be "executed" or "blocked" (by risk).
        # We just want to see it didn't get caught by the STEP 0 EXIT_ONLY filter.
        assert result.action_taken != "blocked"
        # It will likely be "executed" if risk approves
        assert result.action_taken in ["executed", "blocked"] 
        # Verify it wasn't the "EXIT-ONLY mode active" rejection
        reason = engine.truth_logger.log_decision.call_args[1]['result_reason']
        assert "EXIT-ONLY" not in reason

    @pytest.mark.asyncio
    async def test_halted_blocks_everything(self, engine, signal_exit):
        """Verify HALTED blocks an EXIT signal."""
        engine.mode = TradingMode.HALTED
        engine.halt_reason = "Critical Error"
        
        df = pd.DataFrame({'close': [95000], 'high': [95000]})
        
        result = await engine.run_tick(df)
        
        assert result.action_taken == "fail_closed"
        assert "HALTED" in str(engine.truth_logger.log_decision.call_args[1]['result_reason'])
