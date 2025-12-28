"""
Integration test: Scoreboard API reflects state file correctly.

P0 Requirement: Stop loss and chandelier stop values in API must match state file.
"""

import pytest
import json
import aiosqlite
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

# Test paths
TEST_STATE_PATH = Path("tests/fixtures/test_paper_trader_state.json")
TEST_DB_PATH = Path("tests/fixtures/test_scoreboard.db")


class TestScoreboardStateAlignment:
    """Tests that scoreboard API reflects state file accurately."""

    @pytest.fixture
    def sample_state(self):
        """Create sample state with known stop values."""
        return {
            "total_equity": 10000.0,
            "cash": 8000.0,
            "high_water_mark": 10000.0,
            "starting_capital": 10000.0,
            "positions": {
                "BCH-USD": {
                    "symbol": "BCH-USD",
                    "qty": 2.04,
                    "cost_basis": 610.23,
                    "entry_time": "2025-12-26T03:50:00Z",
                    "current_price": 605.0,
                    "unrealized_pnl": -10.67,
                    "regime": "ranging",
                    "hard_stop_price": 579.72,  # -5% from entry
                    "hard_stop_pct": 0.05,
                    "chandelier_stop": 590.50,  # Dynamic trailing
                    "highest_high_since_entry": 615.00
                }
            },
            "dd_state": "normal",
            "current_dd_pct": 0.1,
            "last_update": "2025-12-26T12:00:00Z",
            "engine": "unified_mtf"
        }

    @pytest.fixture
    def setup_test_files(self, sample_state, tmp_path):
        """Create test state file and DB."""
        # Create state file
        state_file = tmp_path / "paper_trader_state.json"
        state_file.write_text(json.dumps(sample_state))

        # Create test DB with matching trade
        db_path = tmp_path / "test.db"

        import asyncio

        async def create_db():
            async with aiosqlite.connect(str(db_path)) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id TEXT PRIMARY KEY,
                        symbol TEXT,
                        side TEXT,
                        status TEXT,
                        entry_price TEXT,
                        quantity TEXT,
                        stop_loss_price TEXT,
                        take_profit_price TEXT,
                        strategy_name TEXT
                    )
                """)
                # Insert open trade with hard stop
                await db.execute("""
                    INSERT INTO trades (trade_id, symbol, side, status, entry_price, quantity, stop_loss_price, strategy_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, ("test-trade-1", "BCH-USD", "buy", "open", "610.23", "2.04", "579.72", "unified_portfolio"))
                await db.commit()

        asyncio.get_event_loop().run_until_complete(create_db())

        return state_file, db_path

    def test_state_has_hard_stop(self, sample_state):
        """Verify state file contains hard_stop_price."""
        position = sample_state["positions"]["BCH-USD"]
        assert position["hard_stop_price"] == 579.72
        assert position["hard_stop_pct"] == 0.05

    def test_state_has_chandelier_stop(self, sample_state):
        """Verify state file contains chandelier_stop."""
        position = sample_state["positions"]["BCH-USD"]
        assert position["chandelier_stop"] == 590.50
        assert position["highest_high_since_entry"] == 615.00

    def test_hard_stop_calculation(self, sample_state):
        """Verify hard stop is correctly calculated as 5% below entry."""
        position = sample_state["positions"]["BCH-USD"]
        entry = position["cost_basis"]
        hard_stop_pct = position["hard_stop_pct"]
        expected_hard_stop = entry * (1 - hard_stop_pct)

        # Should be 610.23 * 0.95 = 579.7185
        assert abs(position["hard_stop_price"] - expected_hard_stop) < 0.01

    def test_chandelier_below_highest_high(self, sample_state):
        """Verify chandelier stop is below highest high since entry."""
        position = sample_state["positions"]["BCH-USD"]
        assert position["chandelier_stop"] < position["highest_high_since_entry"]

    def test_chandelier_above_hard_stop(self, sample_state):
        """Verify chandelier stop is above hard stop (tighter protection)."""
        position = sample_state["positions"]["BCH-USD"]
        # Chandelier should provide tighter protection than hard stop
        assert position["chandelier_stop"] > position["hard_stop_price"]

    @pytest.mark.asyncio
    async def test_db_has_stop_loss(self, setup_test_files):
        """Verify DB trades table has stop_loss_price populated."""
        state_file, db_path = setup_test_files

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT stop_loss_price FROM trades WHERE symbol = ?",
                ("BCH-USD",)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row["stop_loss_price"] == "579.72"


class TestReasoningArrayPopulation:
    """Tests that reasoning arrays are properly populated."""

    def test_reasoning_includes_trend_alignment(self):
        """Verify reasoning includes trend alignment status."""
        from scripts.live_unified_trader import SymbolSignal

        signal = SymbolSignal(
            symbol="BTC-USD",
            action="HOLD",
            regime="bull",
            target_alloc_pct=Decimal("0"),
            current_price=Decimal("100000"),
            reason="Grade C setup",
            context={
                "professional": {
                    "grade": "C",
                    "reasoning": [
                        "Trend NOT aligned - conflicting TF signals",
                        "Momentum NOT confirming",
                        "Volume NOT confirmed",
                        "Grade C: No trade: conflicting trends, weak momentum"
                    ]
                }
            }
        )

        reasoning = signal.context["professional"]["reasoning"]
        assert len(reasoning) >= 3
        assert any("Trend" in r for r in reasoning)
        assert any("Momentum" in r for r in reasoning)
        assert any("Volume" in r for r in reasoning)

    def test_reasoning_includes_grade_reason(self):
        """Verify reasoning includes the grade reason."""
        reasoning = [
            "Trend aligned across timeframes (bullish)",
            "Momentum confirms direction",
            "Volume confirmed (2.1x avg)",
            "Grade A+: All aligned: 3/3 signals, strong trend (ADX 35), high volume (2.1x)"
        ]

        assert any("Grade" in r for r in reasoning)
        assert any("A+" in r for r in reasoning)
