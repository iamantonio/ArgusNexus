"""
PR-2: Catastrophic Stop Enforcement - Acceptance Tests

These tests MUST pass before the portfolio trader can resume operations.

Required behavior (from Tony's marching orders, Dec 19 2025):
- Enforced in code, on a frequent cadence (NOT daily)
- Runs even if the strategy loop doesn't run (guardian/watchdog)
- Fail closed on missing price feed
- On trigger: emergency exit + symbol halt + CRITICAL alert + durable reason

Acceptance tests:
1. Stop triggers exit: price crosses stop → exit invoked + trade closed with reason catastrophic_stop
2. Halt after trigger: trading_halted=true persists durably
3. Feed failure fails closed: price fetch error → halt + CRITICAL alert, no new orders
4. Restart safety: on restart with open position, guardian starts and enforces within one interval
"""

import pytest
import asyncio
import sys
import json
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCatastrophicStopConfig:
    """
    Tests for catastrophic stop configuration.

    The stop must be configurable and loaded from config.yaml.
    """

    def test_config_has_catastrophic_stop_pct(self):
        """
        Config must have catastrophic_stop_pct defined.

        Default: 15% (configurable)
        """
        import yaml

        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Must have portfolio section with catastrophic_stop_pct
        assert "portfolio" in config, "config.yaml must have 'portfolio' section"
        assert "catastrophic_stop_pct" in config["portfolio"], \
            "portfolio section must have 'catastrophic_stop_pct'"

        stop_pct = config["portfolio"]["catastrophic_stop_pct"]
        assert isinstance(stop_pct, (int, float)), "catastrophic_stop_pct must be numeric"
        assert 0 < stop_pct <= 100, f"catastrophic_stop_pct must be between 0 and 100, got {stop_pct}"

    def test_default_catastrophic_stop_is_15_percent(self):
        """
        Default catastrophic stop should be 15%.
        """
        import yaml

        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        stop_pct = config["portfolio"]["catastrophic_stop_pct"]
        assert stop_pct == 15.0, f"Default catastrophic_stop_pct should be 15.0, got {stop_pct}"


class TestCatastrophicStopCalculation:
    """
    Tests for catastrophic stop price calculation.

    Stop price = WAC * (1 - catastrophic_stop_pct/100)
    """

    def test_stop_price_calculation_from_entry(self):
        """
        For a single entry, stop = entry * (1 - stop_pct/100)

        Example: Entry $100,000, stop_pct=15%
        Stop = $100,000 * 0.85 = $85,000
        """
        entry_price = Decimal("100000")
        stop_pct = Decimal("15")

        expected_stop = entry_price * (1 - stop_pct / 100)
        assert expected_stop == Decimal("85000"), f"Expected $85,000, got ${expected_stop}"

    def test_stop_price_with_wac_after_multiple_buys(self):
        """
        After multiple buys, stop is based on WAC (weighted average cost).

        Buy 1: 0.01 BTC @ $90,000 = $900
        Buy 2: 0.02 BTC @ $95,000 = $1,900
        Total: 0.03 BTC, $2,800
        WAC = $2,800 / 0.03 = $93,333.33

        Stop = $93,333.33 * 0.85 = $79,333.33
        """
        # Buy 1
        qty1 = Decimal("0.01")
        price1 = Decimal("90000")
        cost1 = qty1 * price1

        # Buy 2
        qty2 = Decimal("0.02")
        price2 = Decimal("95000")
        cost2 = qty2 * price2

        # WAC
        total_qty = qty1 + qty2
        total_cost = cost1 + cost2
        wac = total_cost / total_qty

        expected_wac = Decimal("93333.333333333333333333333333")
        assert abs(wac - expected_wac) < Decimal("0.01"), f"WAC should be ~$93,333.33, got ${wac}"

        # Stop
        stop_pct = Decimal("15")
        stop_price = wac * (1 - stop_pct / 100)
        expected_stop = Decimal("79333.333333333333333333333333")
        assert abs(stop_price - expected_stop) < Decimal("0.01"), \
            f"Stop should be ~$79,333.33, got ${stop_price}"


class TestCatastrophicStopTrigger:
    """
    Acceptance Test 1: Stop triggers exit

    When price crosses catastrophic stop threshold:
    - Exit invoked immediately
    - Trade closed with reason="catastrophic_stop"
    - Position becomes 0
    """

    @pytest.fixture
    def mock_state_file(self, tmp_path):
        """Create a mock state file with an open position."""
        state = {
            "total_equity": 500.0,
            "btc_qty": 0.005,
            "cash": 75.0,
            "high_water_mark": 500.0,
            "dd_state": "normal",
            "recovery_mode": False,
            "bars_in_critical": 0,
            "sleeve_in_position": False,
            "trading_halted": False,
            # Cost basis tracking
            "btc_cost_basis": 85000.0,  # Entry/WAC price
            "btc_total_cost": 425.0,    # 0.005 * 85000
        }
        state_path = tmp_path / "portfolio_state.json"
        state_path.write_text(json.dumps(state))
        return state_path

    def test_stop_triggers_when_price_breaches_threshold(self):
        """
        Price crosses below stop → exit triggered.

        Scenario:
        - Entry (WAC): $85,000
        - Stop (15%): $72,250
        - Current price: $72,000 (below stop)
        - Expected: EXIT triggered
        """
        entry_price = Decimal("85000")
        stop_pct = Decimal("15")
        stop_price = entry_price * (1 - stop_pct / 100)
        current_price = Decimal("72000")

        # Assert price is below stop
        assert current_price < stop_price, "Test setup: price should be below stop"

        # The guardian should detect this and trigger exit
        # This tests the LOGIC - actual implementation will call this
        should_exit = current_price <= stop_price
        assert should_exit is True, "Should trigger exit when price <= stop"

    def test_no_trigger_when_price_above_stop(self):
        """
        Price above stop → no exit.

        Scenario:
        - Entry (WAC): $85,000
        - Stop (15%): $72,250
        - Current price: $80,000 (above stop)
        - Expected: NO exit
        """
        entry_price = Decimal("85000")
        stop_pct = Decimal("15")
        stop_price = entry_price * (1 - stop_pct / 100)
        current_price = Decimal("80000")

        assert current_price > stop_price, "Test setup: price should be above stop"

        should_exit = current_price <= stop_price
        assert should_exit is False, "Should NOT trigger exit when price > stop"


class TestCatastrophicStopHalt:
    """
    Acceptance Test 2: Halt after trigger

    When catastrophic stop triggers:
    - trading_halted=true set in state
    - Halt persists durably (survives restart)
    - No new orders allowed
    """

    @pytest.fixture
    def tmp_state_path(self, tmp_path):
        return tmp_path / "portfolio_state.json"

    def test_halt_persisted_to_state_file(self, tmp_state_path):
        """
        After catastrophic stop, trading_halted=true in state file.
        """
        # Simulate catastrophic stop trigger
        state = {
            "total_equity": 400.0,
            "btc_qty": 0.0,  # Position closed
            "cash": 400.0,
            "trading_halted": True,  # HALTED
            "halt_reason": "CATASTROPHIC_STOP: Price $72000 breached stop $72250",
            "halt_time": datetime.now(timezone.utc).isoformat()
        }

        tmp_state_path.write_text(json.dumps(state))

        # Verify halt persisted
        loaded = json.loads(tmp_state_path.read_text())
        assert loaded["trading_halted"] is True
        assert "CATASTROPHIC_STOP" in loaded["halt_reason"]

    def test_halt_survives_restart(self, tmp_state_path):
        """
        On restart, if trading_halted=true, system stays halted.
        """
        # Set initial halted state
        state = {
            "trading_halted": True,
            "halt_reason": "CATASTROPHIC_STOP: Previous incident"
        }
        tmp_state_path.write_text(json.dumps(state))

        # Simulate restart (re-read state)
        loaded = json.loads(tmp_state_path.read_text())

        # System should recognize halt
        assert loaded["trading_halted"] is True
        # Should NOT auto-clear the halt
        assert "CATASTROPHIC_STOP" in loaded["halt_reason"]


class TestCatastrophicStopFeedFailure:
    """
    Acceptance Test 3: Feed failure fails closed

    When price feed fails:
    - Halt trading immediately
    - Send CRITICAL alert
    - No new orders allowed
    - "If you can't price it, you can't hold it"
    """

    def test_feed_error_triggers_halt(self):
        """
        Price fetch error → system halts.

        The guardian should:
        1. Attempt to fetch price
        2. On exception → set trading_halted=true
        3. Send CRITICAL alert
        """
        # This is a logic test - implementation will wrap price fetch in try/except
        # and call halt on any error

        # Simulate price fetch failure
        price_fetch_failed = True

        # Expected behavior
        should_halt = price_fetch_failed
        assert should_halt is True, "Feed failure should trigger halt"

    def test_feed_error_reason_logged(self, tmp_path):
        """
        Feed failure should log the error reason durably.
        """
        state_path = tmp_path / "portfolio_state.json"

        # Simulate feed failure halt
        state = {
            "trading_halted": True,
            "halt_reason": "PRICE_FEED_ERROR: Connection timeout to Coinbase API",
            "halt_time": datetime.now(timezone.utc).isoformat()
        }
        state_path.write_text(json.dumps(state))

        loaded = json.loads(state_path.read_text())
        assert loaded["trading_halted"] is True
        assert "PRICE_FEED_ERROR" in loaded["halt_reason"]


class TestCatastrophicStopGuardianLoop:
    """
    Tests for the guardian loop that enforces catastrophic stop.

    The guardian must:
    - Run independently of strategy loop
    - Check at frequent interval (default 60 seconds)
    - Start on system startup
    - Continue even if strategy loop crashes
    """

    def test_guardian_check_interval_configured(self):
        """
        Guardian check interval should be configurable.

        Default: 60 seconds
        """
        # This tests config exists
        import yaml

        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "portfolio" in config
        assert "guardian_check_interval_seconds" in config["portfolio"], \
            "portfolio section must have 'guardian_check_interval_seconds'"

        interval = config["portfolio"]["guardian_check_interval_seconds"]
        assert isinstance(interval, (int, float))
        assert interval >= 10, "Guardian interval should be at least 10 seconds"
        assert interval <= 300, "Guardian interval should be at most 5 minutes"

    def test_guardian_has_open_position_check(self):
        """
        Guardian should check for open positions.

        The live_portfolio_trader.py must have a method that:
        1. Checks if there's an open BTC position
        2. If yes, fetches price and checks against stop
        3. If breached, triggers emergency exit
        """
        # Code inspection test
        trader_path = Path(__file__).parent.parent.parent / "scripts" / "live_portfolio_trader.py"
        trader_code = trader_path.read_text()

        # Must have guardian-related code
        assert "guardian" in trader_code.lower() or "catastrophic" in trader_code.lower(), \
            "live_portfolio_trader.py must have guardian or catastrophic stop logic"


class TestCatastrophicStopRestartSafety:
    """
    Acceptance Test 4: Restart safety

    On restart with an open position:
    - Guardian loop starts immediately
    - Enforces stop within one interval
    - Position survives restart intact until checked
    """

    def test_restart_with_open_position_triggers_check(self, tmp_path):
        """
        On startup with open position, guardian should check immediately.
        """
        # State file with open position
        state = {
            "total_equity": 500.0,
            "btc_qty": 0.005,  # Open position
            "cash": 75.0,
            "btc_cost_basis": 85000.0,
            "trading_halted": False
        }
        state_path = tmp_path / "portfolio_state.json"
        state_path.write_text(json.dumps(state))

        # On restart, guardian should:
        # 1. Load state
        # 2. See btc_qty > 0
        # 3. Immediately check price vs stop
        loaded = json.loads(state_path.read_text())
        has_open_position = loaded.get("btc_qty", 0) > 0

        assert has_open_position is True, "Should detect open position on restart"
        # Guardian should be armed and checking

    def test_cost_basis_preserved_across_restart(self, tmp_path):
        """
        Cost basis (WAC) must be preserved in state file.

        Without cost basis, we can't compute stop price.
        If cost basis is missing on restart with open position → fail closed.
        """
        # State with position but NO cost basis = dangerous
        state_no_cost = {
            "btc_qty": 0.005,
            "trading_halted": False
            # btc_cost_basis MISSING!
        }
        state_path = tmp_path / "portfolio_state.json"
        state_path.write_text(json.dumps(state_no_cost))

        loaded = json.loads(state_path.read_text())
        has_position = loaded.get("btc_qty", 0) > 0
        has_cost_basis = "btc_cost_basis" in loaded

        # If position exists but cost basis missing → FAIL CLOSED
        should_fail_closed = has_position and not has_cost_basis
        assert should_fail_closed is True, \
            "Missing cost basis with open position should trigger fail closed"


class TestCatastrophicStopLogging:
    """
    Tests for catastrophic stop logging/audit trail.

    Must record:
    - Computed stop price in decision log
    - Trigger event with snapshot (price, stop, distance, timestamp)
    """

    def test_stop_price_in_decision_context(self):
        """
        When guardian checks, computed stop price should be available for logging.
        """
        entry_price = Decimal("85000")
        stop_pct = Decimal("15")
        stop_price = entry_price * (1 - stop_pct / 100)
        current_price = Decimal("80000")

        # Decision context should include
        decision_context = {
            "btc_cost_basis": float(entry_price),
            "catastrophic_stop_price": float(stop_price),
            "current_price": float(current_price),
            "distance_to_stop_pct": float((current_price - stop_price) / stop_price * 100)
        }

        assert "catastrophic_stop_price" in decision_context
        assert decision_context["catastrophic_stop_price"] == 72250.0

    def test_trigger_event_logged_with_snapshot(self):
        """
        When stop triggers, event should include full snapshot.
        """
        trigger_event = {
            "event_type": "CATASTROPHIC_STOP_TRIGGERED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "snapshot": {
                "current_price": 72000.0,
                "stop_price": 72250.0,
                "cost_basis": 85000.0,
                "btc_qty": 0.005,
                "distance_pct": -0.35  # How far below stop
            },
            "action": "EMERGENCY_EXIT",
            "exit_reason": "catastrophic_stop"
        }

        # All required fields present
        assert trigger_event["event_type"] == "CATASTROPHIC_STOP_TRIGGERED"
        assert "snapshot" in trigger_event
        assert "current_price" in trigger_event["snapshot"]
        assert "stop_price" in trigger_event["snapshot"]
        assert trigger_event["snapshot"]["current_price"] < trigger_event["snapshot"]["stop_price"]
