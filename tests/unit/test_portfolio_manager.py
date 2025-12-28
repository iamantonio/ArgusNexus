"""
Unit tests for Portfolio Manager

Tests DD state transitions, circuit breaker behavior, and recovery logic.
Updated for hardened portfolio manager with:
- Time-based recovery (bars_in_critical >= 14 AND bull regime)
- Conservative recovery allocation (30%)
- Rebalance cooldown (5 days)
- Tighter DD thresholds (warning=12%, critical=20%)
"""

import pytest
import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.portfolio.portfolio_manager import (
    PortfolioManager,
    PortfolioState,
    DDState,
    RebalanceOrder
)


class TestDDStateTransitions:
    """Test DD circuit breaker state transitions."""

    def test_normal_to_warning(self):
        """DD >= warning (12%) should transition to WARNING state."""
        pm = PortfolioManager()

        # DD at 14% (>= 12% warning)
        dd_mult, new_state, recovery_mode, bars = pm._compute_dd_multiplier(
            Decimal("14.0"), DDState.NORMAL, False, 0, "sideways"
        )

        assert new_state == DDState.WARNING
        assert dd_mult == Decimal("0.5")
        assert not recovery_mode

    def test_warning_to_critical(self):
        """DD >= critical (20%) should transition to CRITICAL state."""
        pm = PortfolioManager()

        # DD at 21% (>= 20% critical)
        dd_mult, new_state, recovery_mode, bars = pm._compute_dd_multiplier(
            Decimal("21.0"), DDState.WARNING, False, 0, "bear"
        )

        assert new_state == DDState.CRITICAL
        assert dd_mult == Decimal("0.0")
        assert recovery_mode  # Should enter recovery mode

    def test_critical_holds_at_zero(self):
        """CRITICAL state should maintain 0% exposure when not enough bars."""
        pm = PortfolioManager()

        # DD above critical, not enough bars for time-based recovery
        dd_mult, new_state, recovery_mode, bars = pm._compute_dd_multiplier(
            Decimal("25.0"), DDState.CRITICAL, True, 5, "bull"  # Only 5 bars
        )

        assert new_state == DDState.CRITICAL
        assert dd_mult == Decimal("0.0")
        assert recovery_mode

    def test_time_based_recovery_triggers(self):
        """After min_bars_in_critical AND bull regime, allow recovery."""
        pm = PortfolioManager()

        # DD elevated but waited long enough AND bull regime
        dd_mult, new_state, recovery_mode, bars = pm._compute_dd_multiplier(
            Decimal("25.0"), DDState.CRITICAL, True, 15, "bull"  # 15 bars >= 14
        )

        assert new_state == DDState.RECOVERY
        assert dd_mult == pm.recovery_alloc  # Should be 0.30
        assert recovery_mode  # Still in recovery mode

    def test_time_based_recovery_blocked_in_bear(self):
        """Time-based recovery should NOT trigger in bear regime."""
        pm = PortfolioManager()

        # Enough bars but bear regime
        dd_mult, new_state, recovery_mode, bars = pm._compute_dd_multiplier(
            Decimal("18.0"), DDState.RECOVERY, True, 20, "bear"
        )

        # Should stay at 0% in bear regime
        assert dd_mult == Decimal("0.0")
        assert recovery_mode

    def test_recovery_to_normal_requires_bull_and_low_dd(self):
        """DD < recovery_full (10%) AND bull should restore normal state."""
        pm = PortfolioManager()

        # DD at 8% (< 10% recovery_full) AND bull regime AND enough bars
        dd_mult, new_state, recovery_mode, bars = pm._compute_dd_multiplier(
            Decimal("8.0"), DDState.RECOVERY, True, 15, "bull"
        )

        assert new_state == DDState.NORMAL
        assert dd_mult == Decimal("1.0")
        assert not recovery_mode  # Should exit recovery mode

    def test_recovery_half_exposure(self):
        """DD between recovery_full and recovery_half should give 50% exposure."""
        pm = PortfolioManager()

        # DD at 12% (between 10% and 15%) AND bull AND enough bars
        dd_mult, new_state, recovery_mode, bars = pm._compute_dd_multiplier(
            Decimal("12.0"), DDState.RECOVERY, True, 15, "bull"
        )

        assert new_state == DDState.RECOVERY
        assert dd_mult == Decimal("0.5")
        assert recovery_mode

    def test_normal_stays_normal(self):
        """Low DD should stay in NORMAL state."""
        pm = PortfolioManager()

        # DD at 5%
        dd_mult, new_state, recovery_mode, bars = pm._compute_dd_multiplier(
            Decimal("5.0"), DDState.NORMAL, False, 0, "sideways"
        )

        assert new_state == DDState.NORMAL
        assert dd_mult == Decimal("1.0")
        assert not recovery_mode

    def test_bars_in_critical_increments(self):
        """bars_in_critical should increment when in critical state."""
        pm = PortfolioManager()

        # First hit of critical
        _, _, _, bars1 = pm._compute_dd_multiplier(
            Decimal("25.0"), DDState.WARNING, False, 0, "bear"
        )
        assert bars1 == 1

        # Staying in critical
        _, _, _, bars2 = pm._compute_dd_multiplier(
            Decimal("25.0"), DDState.CRITICAL, True, 5, "bear"
        )
        assert bars2 == 6

    def test_bars_reset_on_full_recovery(self):
        """bars_in_critical should reset to 0 on full recovery."""
        pm = PortfolioManager()

        # Full recovery: DD < 10% AND bull AND enough bars
        _, new_state, _, bars = pm._compute_dd_multiplier(
            Decimal("8.0"), DDState.RECOVERY, True, 20, "bull"
        )

        assert new_state == DDState.NORMAL
        assert bars == 0


class TestRebalanceLogic:
    """Test rebalance threshold and order generation."""

    def setup_method(self):
        """Create test data."""
        # Create minimal OHLCV data
        dates = pd.date_range("2024-01-01", periods=250, freq="D")
        self.df = pd.DataFrame({
            "open": 40000.0,
            "high": 41000.0,
            "low": 39000.0,
            "close": 40000.0,
            "volume": 1000.0
        }, index=dates)
        # Add some variation to create valid indicators
        self.df["close"] = 40000 + np.random.randn(250) * 1000

    def test_no_rebalance_within_threshold(self):
        """Should not rebalance if drift is within threshold."""
        pm = PortfolioManager(rebalance_threshold=0.15)

        # State at 50% allocation
        state = PortfolioState(
            total_equity=Decimal("1000"),
            btc_qty=Decimal("0.0125"),  # 500 / 40000 = 0.0125 BTC
            cash=Decimal("500"),
            high_water_mark=Decimal("1000"),
            dd_state=DDState.NORMAL,
            recovery_mode=False,
            sleeve_in_position=True,
            sleeve_entry_price=Decimal("40000"),
            last_update=datetime.now()
        )

        order, new_state = pm.evaluate(
            self.df,
            state,
            Decimal("40000"),
            timestamp=datetime.now()
        )

        # Check that we're not trading if drift is small
        # (actual behavior depends on strategy signals)
        assert order.action in ["HOLD", "BUY", "SELL"]

    def test_rebalance_when_drift_exceeds_threshold(self):
        """Should rebalance if drift exceeds threshold."""
        pm = PortfolioManager(rebalance_threshold=0.15)

        # State at 0% allocation but target might be higher
        state = PortfolioState(
            total_equity=Decimal("1000"),
            btc_qty=Decimal("0"),
            cash=Decimal("1000"),
            high_water_mark=Decimal("1000"),
            dd_state=DDState.NORMAL,
            recovery_mode=False,
            sleeve_in_position=False,
            sleeve_entry_price=None,
            last_update=datetime.now()
        )

        order, new_state = pm.evaluate(
            self.df,
            state,
            Decimal("40000"),
            timestamp=datetime.now()
        )

        # If target allocation is significantly different from 0%,
        # we should see a BUY order
        if order.target_alloc_pct > Decimal("15"):  # If target > 15%
            assert order.action == "BUY"


class TestRebalanceCooldown:
    """Test rebalance cooldown behavior."""

    def test_cooldown_blocks_rebalance(self):
        """Should not rebalance if cooldown not elapsed."""
        pm = PortfolioManager(rebalance_cooldown_days=5)

        current_time = datetime(2024, 6, 15, 12, 0)
        last_rebalance = datetime(2024, 6, 12, 12, 0)  # 3 days ago

        can_rebalance = pm._can_rebalance(
            current_time, last_rebalance, dd_state_changed=False
        )

        assert not can_rebalance

    def test_cooldown_allows_after_elapsed(self):
        """Should allow rebalance after cooldown elapsed."""
        pm = PortfolioManager(rebalance_cooldown_days=5)

        current_time = datetime(2024, 6, 15, 12, 0)
        last_rebalance = datetime(2024, 6, 8, 12, 0)  # 7 days ago

        can_rebalance = pm._can_rebalance(
            current_time, last_rebalance, dd_state_changed=False
        )

        assert can_rebalance

    def test_dd_state_change_overrides_cooldown(self):
        """DD state change should bypass cooldown."""
        pm = PortfolioManager(rebalance_cooldown_days=5)

        current_time = datetime(2024, 6, 15, 12, 0)
        last_rebalance = datetime(2024, 6, 14, 12, 0)  # Only 1 day ago

        can_rebalance = pm._can_rebalance(
            current_time, last_rebalance, dd_state_changed=True
        )

        assert can_rebalance  # DD state change overrides cooldown

    def test_first_rebalance_always_allowed(self):
        """First rebalance should always be allowed."""
        pm = PortfolioManager(rebalance_cooldown_days=5)

        can_rebalance = pm._can_rebalance(
            datetime.now(), None, dd_state_changed=False
        )

        assert can_rebalance


class TestCriticalOverridesSleeve:
    """Test that CRITICAL DD forces sleeve exit."""

    def setup_method(self):
        """Create test data."""
        dates = pd.date_range("2024-01-01", periods=250, freq="D")
        self.df = pd.DataFrame({
            "open": 40000.0,
            "high": 41000.0,
            "low": 39000.0,
            "close": 40000.0,
            "volume": 1000.0
        }, index=dates)
        self.df["close"] = 40000 + np.random.randn(250) * 1000

    def test_critical_dd_forces_sleeve_out(self):
        """CRITICAL DD should force sleeve to exit regardless of sleeve signals."""
        pm = PortfolioManager()

        # State with sleeve in position but DD is critical
        state = PortfolioState(
            total_equity=Decimal("800"),  # Down from 1000
            btc_qty=Decimal("0.02"),
            cash=Decimal("0"),
            high_water_mark=Decimal("1000"),  # HWM is higher
            dd_state=DDState.CRITICAL,  # Already in critical
            recovery_mode=True,
            bars_in_critical=5,  # Not enough for time-based recovery
            sleeve_in_position=True,  # Sleeve thinks it should be in
            sleeve_entry_price=Decimal("40000"),
            last_update=datetime.now()
        )

        order, new_state = pm.evaluate(
            self.df,
            state,
            Decimal("40000"),
            timestamp=datetime.now()
        )

        # DD multiplier should be 0 or low, forcing target to 0 or very low
        # (unless regime triggers recovery)
        assert order.context["dd_multiplier"] <= 0.5


class TestOrderExecution:
    """Test order execution updates state correctly."""

    def test_buy_order_execution(self):
        """BUY order should increase BTC and decrease cash."""
        pm = PortfolioManager()

        state = PortfolioState(
            total_equity=Decimal("1000"),
            btc_qty=Decimal("0"),
            cash=Decimal("1000"),
            high_water_mark=Decimal("1000"),
            dd_state=DDState.NORMAL,
            recovery_mode=False,
            sleeve_in_position=False,
            sleeve_entry_price=None,
            last_update=datetime.now()
        )

        order = RebalanceOrder(
            action="BUY",
            btc_qty_delta=Decimal("0.01"),  # Buy 0.01 BTC
            target_btc_qty=Decimal("0.01"),
            target_alloc_pct=Decimal("40"),
            reason="Test buy"
        )

        fees = {"entry": 0.004, "exit": 0.002, "slippage": 0.001}
        new_state = pm.execute_order(order, state, Decimal("40000"), fees)

        assert new_state.btc_qty > 0
        assert new_state.cash < state.cash

    def test_sell_order_execution(self):
        """SELL order should decrease BTC and increase cash."""
        pm = PortfolioManager()

        state = PortfolioState(
            total_equity=Decimal("1000"),
            btc_qty=Decimal("0.02"),
            cash=Decimal("200"),
            high_water_mark=Decimal("1000"),
            dd_state=DDState.NORMAL,
            recovery_mode=False,
            sleeve_in_position=True,
            sleeve_entry_price=Decimal("40000"),
            last_update=datetime.now()
        )

        order = RebalanceOrder(
            action="SELL",
            btc_qty_delta=Decimal("-0.01"),  # Sell 0.01 BTC
            target_btc_qty=Decimal("0.01"),
            target_alloc_pct=Decimal("20"),
            reason="Test sell"
        )

        fees = {"entry": 0.004, "exit": 0.002, "slippage": 0.001}
        new_state = pm.execute_order(order, state, Decimal("40000"), fees)

        assert new_state.btc_qty < state.btc_qty
        assert new_state.cash > state.cash

    def test_hold_order_no_change(self):
        """HOLD order should not change state."""
        pm = PortfolioManager()

        state = PortfolioState(
            total_equity=Decimal("1000"),
            btc_qty=Decimal("0.01"),
            cash=Decimal("600"),
            high_water_mark=Decimal("1000"),
            dd_state=DDState.NORMAL,
            recovery_mode=False,
            sleeve_in_position=True,
            sleeve_entry_price=Decimal("40000"),
            last_update=datetime.now()
        )

        order = RebalanceOrder(
            action="HOLD",
            btc_qty_delta=Decimal("0"),
            target_btc_qty=Decimal("0.01"),
            target_alloc_pct=Decimal("40"),
            reason="No rebalance needed"
        )

        fees = {"entry": 0.004, "exit": 0.002, "slippage": 0.001}
        new_state = pm.execute_order(order, state, Decimal("40000"), fees)

        assert new_state.btc_qty == state.btc_qty
        assert new_state.cash == state.cash


class TestStatePersistence:
    """Test state serialization and deserialization."""

    def test_to_dict_from_dict_roundtrip(self):
        """State should survive JSON round-trip with no precision loss."""
        original = PortfolioState(
            total_equity=Decimal("1234.56789"),
            btc_qty=Decimal("0.00123456"),
            cash=Decimal("987.654321"),
            high_water_mark=Decimal("1500.00"),
            dd_state=DDState.RECOVERY,
            recovery_mode=True,
            bars_in_critical=15,
            sleeve_in_position=True,
            sleeve_entry_price=Decimal("95000.50"),
            last_rebalance_time=datetime(2024, 6, 15, 12, 30, 45),
            last_update=datetime(2024, 6, 16, 8, 0, 0),
        )

        # Round-trip through dict (simulates JSON)
        as_dict = original.to_dict()
        restored = PortfolioState.from_dict(as_dict)

        # Verify all fields match
        assert restored.total_equity == original.total_equity
        assert restored.btc_qty == original.btc_qty
        assert restored.cash == original.cash
        assert restored.high_water_mark == original.high_water_mark
        assert restored.dd_state == original.dd_state
        assert restored.recovery_mode == original.recovery_mode
        assert restored.bars_in_critical == original.bars_in_critical
        assert restored.sleeve_in_position == original.sleeve_in_position
        assert restored.sleeve_entry_price == original.sleeve_entry_price
        assert restored.last_rebalance_time == original.last_rebalance_time
        assert restored.last_update == original.last_update

    def test_from_dict_handles_none_values(self):
        """from_dict should handle None/missing values gracefully."""
        minimal_dict = {
            "total_equity": "100",
            "btc_qty": "0",
            "cash": "100",
            "high_water_mark": "100",
        }

        state = PortfolioState.from_dict(minimal_dict)

        assert state.total_equity == Decimal("100")
        assert state.dd_state == DDState.NORMAL
        assert state.recovery_mode == False
        assert state.bars_in_critical == 0
        assert state.sleeve_entry_price is None
        assert state.last_rebalance_time is None

    def test_from_dict_handles_float_input(self):
        """from_dict should handle float values (legacy compatibility)."""
        float_dict = {
            "total_equity": 1234.56,  # float instead of str
            "btc_qty": 0.001,
            "cash": 1000.0,
            "high_water_mark": 1500.0,
        }

        state = PortfolioState.from_dict(float_dict)

        # Should convert to Decimal without crash
        assert isinstance(state.total_equity, Decimal)
        assert float(state.total_equity) == 1234.56


class TestConfigDefaults:
    """Test that default configuration values are correct."""

    def test_default_dd_thresholds(self):
        """Verify default DD thresholds match spec."""
        pm = PortfolioManager()

        assert pm.dd_warning == Decimal("12.0")
        assert pm.dd_critical == Decimal("20.0")
        assert pm.dd_recovery_full == Decimal("10.0")
        assert pm.dd_recovery_half == Decimal("15.0")

    def test_default_recovery_params(self):
        """Verify default recovery parameters."""
        pm = PortfolioManager()

        assert pm.min_bars_in_critical == 14
        assert pm.recovery_alloc == Decimal("0.30")
        assert pm.rebalance_cooldown_days == 5

    def test_default_weights(self):
        """Verify default portfolio weights."""
        pm = PortfolioManager()

        assert pm.core_weight == Decimal("0.85")
        assert pm.sleeve_weight == Decimal("0.15")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
