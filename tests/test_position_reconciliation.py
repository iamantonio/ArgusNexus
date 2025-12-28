"""
Verification Gate Test Suite - Position Reconciliation (Async)
=======================================================

Run with: PYTHONPATH=. ./venv/bin/pytest tests/test_position_reconciliation.py -v
"""

import pytest
import sqlite3
import uuid
import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import pandas as pd
import os

# Set feature flag before imports
os.environ["POSITION_RECONCILIATION_ENABLED"] = "true"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from truth.logger import TruthLogger
from truth.schema import OrderSide, DecisionResult
from engine import TradingEngine, PositionState, ReconciliationResult, POSITION_RECONCILIATION_ENABLED
from strategy.donchian import DonchianBreakout
from risk import RiskManager, RiskConfig
from execution import PaperExecutor


class TestPositionReconciliation:
    """Verification gate tests for position reconciliation (Async)."""

    @pytest.fixture
    async def truth_logger(self):
        """Create a TruthLogger with temp database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        logger = TruthLogger(db_path)
        await logger.initialize()
        yield logger
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def sample_candles(self):
        """Create sample candle data for testing."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=200)
        data = []
        price = 100.0
        for i in range(200):
            price = price * (1 + (0.01 if i % 3 == 0 else -0.005))
            data.append({
                'timestamp': base_time + timedelta(hours=i * 4),
                'open': price * 0.99, 'high': price * 1.02,
                'low': price * 0.98, 'close': price, 'volume': 1000.0
            })
        return pd.DataFrame(data)

    @pytest.fixture
    async def trading_engine(self, truth_logger):
        """Create a trading engine for testing."""
        strategy = DonchianBreakout()
        risk_config = RiskConfig()
        risk_manager = RiskManager(risk_config)
        executor = PaperExecutor(starting_balance=Decimal("10000"))
        return TradingEngine(strategy, risk_manager, executor, truth_logger, "TEST-USD", Decimal("10000"))

    async def seed_open_position(self, truth_logger, symbol="TEST-USD", entry_price="100.00", executor=None):
        decision = await truth_logger.log_decision(symbol, "donchian_breakout", {"signal": "long"}, {}, DecisionResult.SIGNAL_LONG, "Test entry")
        order = await truth_logger.log_order(decision.decision_id, symbol, OrderSide.BUY, Decimal("10"), Decimal(entry_price))
        await truth_logger.update_order_fill(order.order_id, Decimal(entry_price), Decimal("10"))
        trade_id = await truth_logger.open_trade(symbol, OrderSide.BUY, order.order_id, Decimal(entry_price), Decimal("10"), Decimal(str(float(entry_price) * 0.95)), Decimal(str(float(entry_price) * 1.10)), "donchian_breakout", datetime.now(timezone.utc) - timedelta(hours=20))
        if executor is not None:
            executor._balances[symbol.split("-")[0]] = Decimal("10")
        return trade_id

    @pytest.mark.asyncio
    async def test_hydration_loads_existing_position(self, trading_engine, truth_logger, sample_candles):
        trade_id = await self.seed_open_position(truth_logger, "TEST-USD", "100.00", executor=trading_engine.executor)
        result = await trading_engine.hydrate_position_state(sample_candles)
        assert result.success and result.position_found and trading_engine.has_open_position
        assert trading_engine.entry_price == Decimal("100.00") and trading_engine.open_trade_id == trade_id

    @pytest.mark.asyncio
    async def test_hydration_no_position_returns_clean(self, trading_engine, sample_candles):
        result = await trading_engine.hydrate_position_state(sample_candles)
        assert result.success and not result.position_found and not trading_engine.has_open_position

    @pytest.mark.asyncio
    async def test_fail_closed_blocks_trading(self, trading_engine, sample_candles):
        await trading_engine.enter_fail_closed("Test failure")
        result = await trading_engine.run_tick(sample_candles)
        assert result.action_taken == "fail_closed" and trading_engine.fail_closed

    @pytest.mark.asyncio
    async def test_mismatch_triggers_fail_closed(self, trading_engine, truth_logger):
        await self.seed_open_position(truth_logger, "TEST-USD", "100.00")
        result = await trading_engine.reconcile_position()
        assert result.requires_fail_closed and trading_engine.fail_closed


class TestBrokerDBReconciliationHarness:
    """CRITICAL TEST HARNESS - Broker vs DB State Reconciliation (Async)."""

    @pytest.fixture
    async def truth_logger(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        logger = TruthLogger(db_path)
        await logger.initialize()
        yield logger
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def sample_candles(self):
        base_time = datetime.now(timezone.utc) - timedelta(hours=30)
        data = []
        price = 100.0
        for i in range(200):
            price = price * (1 + (0.01 if i % 3 == 0 else -0.005))
            data.append({
                'timestamp': base_time + timedelta(hours=i * 4),
                'open': price * 0.99, 'high': price * 1.02,
                'low': price * 0.98, 'close': price, 'volume': 1000.0
            })
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_notifier(self):
        return Mock()

    def create_engine_with_mocked_executor(self, truth_logger, mock_notifier, broker_position=None):
        strategy = DonchianBreakout()
        risk_manager = RiskManager(RiskConfig())
        executor = Mock()
        fut_bal = asyncio.Future()
        fut_bal.set_result(Decimal("10000"))
        executor.get_balance.return_value = fut_bal
        executor.get_position_size.return_value = Decimal("10") if broker_position else Decimal("0")
        executor.has_position.return_value = broker_position is not None
        executor.get_position.return_value = broker_position
        return TradingEngine(strategy, risk_manager, executor, truth_logger, "TEST-USD", Decimal("10000"), mock_notifier)

    async def seed_open_position(self, truth_logger, symbol="TEST-USD", entry_price="100.00"):
        decision = await truth_logger.log_decision(symbol, "donchian_breakout", {"signal": "long"}, {}, DecisionResult.SIGNAL_LONG, "Test entry")
        order = await truth_logger.log_order(decision.decision_id, symbol, OrderSide.BUY, Decimal("10"), Decimal(entry_price))
        await truth_logger.update_order_fill(order.order_id, Decimal(entry_price), Decimal("10"))
        return await truth_logger.open_trade(symbol, OrderSide.BUY, order.order_id, Decimal(entry_price), Decimal("10"), Decimal("95"), Decimal("110"), "donchian_breakout", datetime.now(timezone.utc) - timedelta(hours=20))

    @pytest.mark.asyncio
    async def test_scenario_1_broker_has_position_db_empty(self, truth_logger, mock_notifier):
        broker_position = {"symbol": "TEST-USD", "quantity": Decimal("10"), "entry_price": Decimal("100"), "side": "buy"}
        engine = self.create_engine_with_mocked_executor(truth_logger, mock_notifier, broker_position=broker_position)
        result = await engine.reconcile_position()
        assert not result.success and result.requires_fail_closed and engine.fail_closed

    @pytest.mark.asyncio
    async def test_scenario_2_broker_empty_db_has_position(self, truth_logger, mock_notifier):
        await self.seed_open_position(truth_logger, "TEST-USD", "100.00")
        engine = self.create_engine_with_mocked_executor(truth_logger, mock_notifier, broker_position=None)
        result = await engine.reconcile_position()
        assert not result.success and result.requires_fail_closed and engine.fail_closed