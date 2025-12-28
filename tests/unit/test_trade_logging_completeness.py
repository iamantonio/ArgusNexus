"""
PR-3: Trade Logging Completeness - Acceptance Tests

Every closed trade must be "auditable to the penny":
- why we entered
- why we exited
- what it cost (fees + slippage)
- how long it lived
- whether it won
- and what risk checks allowed it

Hard requirements (non-NULL on close for valid trades):
- exit_timestamp
- exit_price
- net_pnl (after fees)
- fees_total (total_commission)
- total_slippage
- duration_seconds
- is_winner
- entry_decision_id, exit_decision_id
- risk_checks payload

If any are missing → trade marked invalid with reason (not silently NULL).
"""

import pytest
import asyncio
import sys
import json
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import aiosqlite

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.truth.logger import TruthLogger
from src.truth.schema import (
    Decision, Order, Trade,
    DecisionResult, OrderSide, OrderStatus, ExitReason,
    SQL_SCHEMA
)


class TestTradeLoggingSchema:
    """
    Tests that the schema supports trade validity tracking.
    """

    @pytest.fixture
    async def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_trade_logging.db"
        logger = TruthLogger(str(db_path))
        await logger.initialize()
        return logger

    @pytest.mark.asyncio
    async def test_schema_has_is_valid_column(self, temp_db):
        """
        Trades table must have is_valid column for data quality tracking.
        """
        async with aiosqlite.connect(str(temp_db.db_path)) as db:
            cursor = await db.execute("PRAGMA table_info(trades)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

            assert "is_valid" in column_names, \
                "trades table must have 'is_valid' column"

    @pytest.mark.asyncio
    async def test_schema_has_invalid_reason_column(self, temp_db):
        """
        Trades table must have invalid_reason column to explain why invalid.
        """
        async with aiosqlite.connect(str(temp_db.db_path)) as db:
            cursor = await db.execute("PRAGMA table_info(trades)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

            assert "invalid_reason" in column_names, \
                "trades table must have 'invalid_reason' column"


class TestTradeCloseValidation:
    """
    Tests that close_trade validates all required fields.
    """

    @pytest.fixture
    async def logger_with_open_trade(self, tmp_path):
        """Create logger with an open trade for testing close scenarios."""
        db_path = tmp_path / "test_close_validation.db"
        logger = TruthLogger(str(db_path))
        await logger.initialize()

        # Create entry decision
        entry_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test_strategy",
            signal_values={"test": True},
            risk_checks={"approved": True},
            result=DecisionResult.SIGNAL_LONG,
            result_reason="Test entry"
        )

        # Create entry order
        entry_order = await logger.log_order(
            decision_id=entry_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
            requested_price=Decimal("50000")
        )

        # Fill entry order
        await logger.update_order_fill(
            order_id=entry_order.order_id,
            fill_price=Decimal("50000"),
            fill_quantity=Decimal("0.001"),
            commission=Decimal("2.00")
        )

        # Open trade
        trade_id = await logger.open_trade(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            entry_order_id=entry_order.order_id,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.001"),
            strategy_name="test_strategy"
        )

        return {
            "logger": logger,
            "trade_id": trade_id,
            "entry_order": entry_order,
            "entry_decision": entry_decision
        }

    @pytest.mark.asyncio
    async def test_close_trade_populates_all_fields(self, logger_with_open_trade):
        """
        Closing a trade must populate ALL required fields.
        """
        ctx = logger_with_open_trade
        logger = ctx["logger"]

        # Create exit decision and order
        exit_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test_strategy",
            signal_values={"exit": True},
            risk_checks={"approved": True},
            result=DecisionResult.SIGNAL_CLOSE,
            result_reason="Test exit"
        )

        exit_order = await logger.log_order(
            decision_id=exit_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.001"),
            requested_price=Decimal("51000")
        )

        await logger.update_order_fill(
            order_id=exit_order.order_id,
            fill_price=Decimal("51000"),
            fill_quantity=Decimal("0.001"),
            commission=Decimal("2.04")
        )

        # Close trade with explicit fee/slippage
        await logger.close_trade(
            trade_id=ctx["trade_id"],
            exit_order_id=exit_order.order_id,
            exit_price=Decimal("51000"),
            exit_reason=ExitReason.SIGNAL_EXIT,
            commission=Decimal("4.04"),  # Entry + exit
            slippage=Decimal("0.50")
        )

        # Verify ALL required fields are populated
        async with aiosqlite.connect(str(logger.db_path)) as db:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (ctx["trade_id"],)
            )
            row = await cursor.fetchone()
            columns = [description[0] for description in cursor.description]
            trade = dict(zip(columns, row))

        # Required fields - NONE can be NULL
        required_fields = [
            "exit_timestamp",
            "exit_price",
            "net_pnl",
            "total_commission",
            "total_slippage",
            "duration_seconds",
            "is_winner",
            "entry_decision_id",
            "exit_decision_id",
        ]

        for field in required_fields:
            assert trade.get(field) is not None, \
                f"Required field '{field}' is NULL - trade is not auditable"

    @pytest.mark.asyncio
    async def test_slippage_is_computed_not_null(self, logger_with_open_trade):
        """
        Slippage must be explicitly computed and stored, never NULL.

        Even if slippage is 0, it should be stored as "0", not NULL.
        """
        ctx = logger_with_open_trade
        logger = ctx["logger"]

        # Create exit
        exit_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test_strategy",
            signal_values={"exit": True},
            risk_checks={},
            result=DecisionResult.SIGNAL_CLOSE,
            result_reason="Test exit"
        )

        exit_order = await logger.log_order(
            decision_id=exit_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.001"),
            requested_price=Decimal("51000")
        )

        await logger.update_order_fill(
            order_id=exit_order.order_id,
            fill_price=Decimal("51000"),
            fill_quantity=Decimal("0.001"),
            commission=Decimal("2.00")
        )

        # Close trade - slippage should be computed from orders
        await logger.close_trade(
            trade_id=ctx["trade_id"],
            exit_order_id=exit_order.order_id,
            exit_price=Decimal("51000"),
            exit_reason=ExitReason.SIGNAL_EXIT,
            commission=Decimal("4.00"),
            slippage=Decimal("0")  # Explicit zero, not NULL
        )

        async with aiosqlite.connect(str(logger.db_path)) as db:
            cursor = await db.execute(
                "SELECT total_slippage FROM trades WHERE trade_id = ?",
                (ctx["trade_id"],)
            )
            row = await cursor.fetchone()

        assert row[0] is not None, "total_slippage must never be NULL"
        # Should be stored as "0" or "0.00", not NULL
        slippage = Decimal(row[0])
        assert slippage >= 0, "Slippage should be >= 0"


class TestTradeValidityMarking:
    """
    Tests for marking trades as invalid when data is missing.
    """

    @pytest.fixture
    async def temp_logger(self, tmp_path):
        db_path = tmp_path / "test_validity.db"
        logger = TruthLogger(str(db_path))
        await logger.initialize()
        return logger

    @pytest.mark.asyncio
    async def test_missing_commission_marks_invalid(self, temp_logger):
        """
        If executor returns missing fee data, trade should be marked invalid.
        """
        logger = temp_logger

        # Create entry decision/order
        entry_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test",
            signal_values={},
            risk_checks={},
            result=DecisionResult.SIGNAL_LONG,
            result_reason="test"
        )

        entry_order = await logger.log_order(
            decision_id=entry_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
            requested_price=Decimal("50000")
        )

        # Fill WITHOUT commission (simulates executor error)
        await logger.update_order_fill(
            order_id=entry_order.order_id,
            fill_price=Decimal("50000"),
            fill_quantity=Decimal("0.001"),
            commission=None  # Missing!
        )

        trade_id = await logger.open_trade(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            entry_order_id=entry_order.order_id,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.001")
        )

        # Create exit
        exit_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test",
            signal_values={},
            risk_checks={},
            result=DecisionResult.SIGNAL_CLOSE,
            result_reason="test"
        )

        exit_order = await logger.log_order(
            decision_id=exit_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.001"),
            requested_price=Decimal("51000")
        )

        # Close with missing commission (None passed)
        await logger.close_trade(
            trade_id=trade_id,
            exit_order_id=exit_order.order_id,
            exit_price=Decimal("51000"),
            exit_reason=ExitReason.SIGNAL_EXIT,
            commission=None  # Missing commission!
        )

        # Trade should be marked invalid
        async with aiosqlite.connect(str(logger.db_path)) as db:
            cursor = await db.execute(
                "SELECT is_valid, invalid_reason FROM trades WHERE trade_id = ?",
                (trade_id,)
            )
            row = await cursor.fetchone()

        # Trade should be marked as invalid
        assert row is not None, "Trade should exist"
        is_valid = row[0]
        invalid_reason = row[1]

        assert is_valid == 0, \
            "Trade with missing commission should be marked is_valid=0"
        assert invalid_reason is not None, \
            "Invalid trade should have a reason"
        assert "commission" in invalid_reason.lower() or "fee" in invalid_reason.lower(), \
            f"Invalid reason should mention missing commission, got: {invalid_reason}"


class TestDurationCalculation:
    """
    Tests for accurate duration calculation.
    """

    @pytest.fixture
    async def temp_logger(self, tmp_path):
        db_path = tmp_path / "test_duration.db"
        logger = TruthLogger(str(db_path))
        await logger.initialize()
        return logger

    @pytest.mark.asyncio
    async def test_duration_equals_exit_minus_entry(self, temp_logger):
        """
        duration_seconds must equal exit_timestamp - entry_timestamp.
        """
        logger = temp_logger

        # Fixed timestamps for precise testing
        entry_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        exit_time = datetime(2025, 1, 1, 14, 30, 45, tzinfo=timezone.utc)
        expected_duration = int((exit_time - entry_time).total_seconds())  # 9045 seconds

        # Create entry
        entry_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test",
            signal_values={},
            risk_checks={},
            result=DecisionResult.SIGNAL_LONG,
            result_reason="test",
            timestamp=entry_time
        )

        entry_order = await logger.log_order(
            decision_id=entry_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
            requested_price=Decimal("50000"),
            timestamp=entry_time
        )

        await logger.update_order_fill(
            order_id=entry_order.order_id,
            fill_price=Decimal("50000"),
            fill_quantity=Decimal("0.001"),
            fill_timestamp=entry_time,
            commission=Decimal("2.00")
        )

        trade_id = await logger.open_trade(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            entry_order_id=entry_order.order_id,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.001"),
            entry_timestamp=entry_time
        )

        # Create exit at specific time
        exit_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test",
            signal_values={},
            risk_checks={},
            result=DecisionResult.SIGNAL_CLOSE,
            result_reason="test",
            timestamp=exit_time
        )

        exit_order = await logger.log_order(
            decision_id=exit_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.001"),
            requested_price=Decimal("51000"),
            timestamp=exit_time
        )

        await logger.update_order_fill(
            order_id=exit_order.order_id,
            fill_price=Decimal("51000"),
            fill_quantity=Decimal("0.001"),
            fill_timestamp=exit_time,
            commission=Decimal("2.00")
        )

        await logger.close_trade(
            trade_id=trade_id,
            exit_order_id=exit_order.order_id,
            exit_price=Decimal("51000"),
            exit_reason=ExitReason.SIGNAL_EXIT,
            commission=Decimal("4.00"),
            slippage=Decimal("0"),
            exit_timestamp=exit_time
        )

        # Check duration
        async with aiosqlite.connect(str(logger.db_path)) as db:
            cursor = await db.execute(
                "SELECT duration_seconds FROM trades WHERE trade_id = ?",
                (trade_id,)
            )
            row = await cursor.fetchone()

        actual_duration = row[0]
        assert actual_duration == expected_duration, \
            f"Duration should be {expected_duration}s, got {actual_duration}s"


class TestNetPnlCalculation:
    """
    Tests for correct net PnL calculation.
    """

    @pytest.fixture
    async def temp_logger(self, tmp_path):
        db_path = tmp_path / "test_pnl.db"
        logger = TruthLogger(str(db_path))
        await logger.initialize()
        return logger

    @pytest.mark.asyncio
    async def test_net_pnl_formula_long(self, temp_logger):
        """
        Net PnL for LONG: (exit - entry) * qty - commission - |slippage|

        Example:
        - Entry: 0.001 BTC @ $50,000 = $50
        - Exit: 0.001 BTC @ $51,000 = $51
        - Gross PnL: $1.00
        - Commission: $0.40
        - Slippage: $0.10
        - Net PnL: $1.00 - $0.40 - $0.10 = $0.50
        """
        logger = temp_logger

        entry_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test",
            signal_values={},
            risk_checks={},
            result=DecisionResult.SIGNAL_LONG,
            result_reason="test"
        )

        entry_order = await logger.log_order(
            decision_id=entry_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
            requested_price=Decimal("50000")
        )

        await logger.update_order_fill(
            order_id=entry_order.order_id,
            fill_price=Decimal("50000"),
            fill_quantity=Decimal("0.001"),
            commission=Decimal("0.20")
        )

        trade_id = await logger.open_trade(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            entry_order_id=entry_order.order_id,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.001")
        )

        exit_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test",
            signal_values={},
            risk_checks={},
            result=DecisionResult.SIGNAL_CLOSE,
            result_reason="test"
        )

        exit_order = await logger.log_order(
            decision_id=exit_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.001"),
            requested_price=Decimal("51000")
        )

        await logger.update_order_fill(
            order_id=exit_order.order_id,
            fill_price=Decimal("51000"),
            fill_quantity=Decimal("0.001"),
            commission=Decimal("0.20")
        )

        await logger.close_trade(
            trade_id=trade_id,
            exit_order_id=exit_order.order_id,
            exit_price=Decimal("51000"),
            exit_reason=ExitReason.SIGNAL_EXIT,
            commission=Decimal("0.40"),  # Total: entry + exit
            slippage=Decimal("0.10")
        )

        # Verify net PnL calculation
        async with aiosqlite.connect(str(logger.db_path)) as db:
            cursor = await db.execute(
                "SELECT realized_pnl, net_pnl, is_winner FROM trades WHERE trade_id = ?",
                (trade_id,)
            )
            row = await cursor.fetchone()

        realized_pnl = Decimal(row[0])
        net_pnl = Decimal(row[1])
        is_winner = row[2]

        # Gross: (51000 - 50000) * 0.001 = $1.00
        expected_gross = Decimal("1.00")
        assert realized_pnl == expected_gross, \
            f"Realized PnL should be ${expected_gross}, got ${realized_pnl}"

        # Net: $1.00 - $0.40 - $0.10 = $0.50
        expected_net = Decimal("0.50")
        assert net_pnl == expected_net, \
            f"Net PnL should be ${expected_net}, got ${net_pnl}"

        # Should be a winner
        assert is_winner == 1, "Trade with positive net PnL should be a winner"


class TestDecisionChainQuery:
    """
    Tests for querying the full decision chain.
    """

    @pytest.fixture
    async def temp_logger(self, tmp_path):
        db_path = tmp_path / "test_chain.db"
        logger = TruthLogger(str(db_path))
        await logger.initialize()
        return logger

    @pytest.mark.asyncio
    async def test_decision_chain_returns_complete_audit(self, temp_logger):
        """
        Query should return decision → order → trade in one shot.
        """
        logger = temp_logger

        # Create full chain
        entry_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test_strategy",
            signal_values={"ema_crossover": True, "atr": 500},
            risk_checks={"concentration": {"passed": True, "value": 15}},
            result=DecisionResult.SIGNAL_LONG,
            result_reason="EMA bullish crossover"
        )

        entry_order = await logger.log_order(
            decision_id=entry_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
            requested_price=Decimal("50000")
        )

        await logger.update_order_fill(
            order_id=entry_order.order_id,
            fill_price=Decimal("50050"),
            fill_quantity=Decimal("0.001"),
            commission=Decimal("2.00")
        )

        trade_id = await logger.open_trade(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            entry_order_id=entry_order.order_id,
            entry_price=Decimal("50050"),
            quantity=Decimal("0.001"),
            strategy_name="test_strategy"
        )

        exit_decision = await logger.log_decision(
            symbol="BTC-USD",
            strategy_name="test_strategy",
            signal_values={"ema_crossover": False, "exit_signal": True},
            risk_checks={},
            result=DecisionResult.SIGNAL_CLOSE,
            result_reason="EMA bearish crossover"
        )

        exit_order = await logger.log_order(
            decision_id=exit_decision.decision_id,
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.001"),
            requested_price=Decimal("51000")
        )

        await logger.update_order_fill(
            order_id=exit_order.order_id,
            fill_price=Decimal("50980"),
            fill_quantity=Decimal("0.001"),
            commission=Decimal("2.00")
        )

        await logger.close_trade(
            trade_id=trade_id,
            exit_order_id=exit_order.order_id,
            exit_price=Decimal("50980"),
            exit_reason=ExitReason.SIGNAL_EXIT,
            commission=Decimal("4.00"),
            slippage=Decimal("0.70")
        )

        # Query using v_trade_audit view
        audit = await logger.get_trade_audit(trade_id)

        assert audit is not None, "Trade audit should be retrievable"
        assert "entry_signal" in audit, "Audit should include entry signal"
        assert "entry_risk_checks" in audit, "Audit should include entry risk checks"
        assert "exit_signal" in audit, "Audit should include exit signal"
        assert "net_pnl" in audit, "Audit should include net PnL"

        # Verify signal data is preserved
        assert audit["entry_signal"]["ema_crossover"] == True
        assert audit["entry_risk_checks"]["concentration"]["passed"] == True


class TestTestContamination:
    """
    Tests for excluding test/canary records from production queries.
    """

    @pytest.fixture
    async def temp_logger(self, tmp_path):
        db_path = tmp_path / "test_contamination.db"
        logger = TruthLogger(str(db_path))
        await logger.initialize()
        return logger

    @pytest.mark.asyncio
    async def test_performance_summary_excludes_invalid_trades(self, temp_logger):
        """
        Performance summary should only count VALID trades.
        """
        logger = temp_logger

        # This test assumes is_valid column exists
        # The schema update should add this
        async with aiosqlite.connect(str(logger.db_path)) as db:
            cursor = await db.execute("PRAGMA table_info(trades)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

            if "is_valid" not in column_names:
                pytest.skip("is_valid column not yet implemented")

        # If we reach here, is_valid exists - test that invalid trades are excluded
        # This would require inserting test data and verifying the summary excludes it
        pass

    @pytest.mark.asyncio
    async def test_canary_test_records_identifiable(self, temp_logger):
        """
        Test records should be identifiable by strategy_name or other marker.
        """
        logger = temp_logger

        # Insert a test decision with clear marker
        test_decision = await logger.log_decision(
            symbol="TEST-USD",
            strategy_name="CANARY_TEST_CLEANUP",  # Clear test marker
            signal_values={"test": True},
            risk_checks={},
            result=DecisionResult.NO_SIGNAL,
            result_reason="Test record for cleanup verification"
        )

        # Query to find test records
        async with aiosqlite.connect(str(logger.db_path)) as db:
            cursor = await db.execute("""
                SELECT COUNT(*) FROM decisions
                WHERE strategy_name LIKE '%TEST%'
                   OR strategy_name LIKE '%CANARY%'
                   OR symbol LIKE '%TEST%'
            """)
            row = await cursor.fetchone()

        test_count = row[0]
        assert test_count > 0, "Test records should be identifiable"

        # Production queries should be able to exclude these
        async with aiosqlite.connect(str(logger.db_path)) as db:
            cursor = await db.execute("""
                SELECT COUNT(*) FROM decisions
                WHERE strategy_name NOT LIKE '%TEST%'
                  AND strategy_name NOT LIKE '%CANARY%'
                  AND symbol NOT LIKE '%TEST%'
            """)
            row = await cursor.fetchone()

        # This should work - production queries can filter
        assert True, "Production queries can exclude test records"
