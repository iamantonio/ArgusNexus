import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone
from src.strategy.donchian import DonchianBreakout, Signal

class TestDonchianBreakout:
    @pytest.fixture
    def strategy(self):
        return DonchianBreakout(
            entry_period=20,
            exit_period=20,
            trend_period=50
        )

    @pytest.fixture
    def trending_up_data(self):
        # Create 100 bars of trending up data
        dates = pd.date_range(start="2025-01-01", periods=100, freq="4H")
        data = {
            'timestamp': dates,
            'open': np.linspace(100, 200, 100),
            'high': np.linspace(102, 202, 100),
            'low': np.linspace(98, 198, 100),
            'close': np.linspace(100, 200, 100),
            'volume': [1000] * 100
        }
        return pd.DataFrame(data)

    def test_insufficient_data(self, strategy):
        df = pd.DataFrame({'close': [100, 101]})
        result = strategy.evaluate(df)
        assert result.signal == Signal.HOLD
        assert "Insufficient data" in result.reason

    def test_long_breakout(self, strategy, trending_up_data):
        # Last bar breaks above the 20-period high and is above 50 SMA
        # We need to make sure the last bar is a clear breakout
        df = trending_up_data.copy()
        df.loc[99, 'high'] = 300
        df.loc[99, 'close'] = 290
        df.loc[99, 'volume'] = 5000 # High volume
        
        result = strategy.evaluate(df)
        assert result.signal == Signal.LONG
        assert "SNIPER ENTRY" in result.reason

    def test_chandelier_exit(self, strategy, trending_up_data):
        # Simulate a position with a high watermark
        df = trending_up_data.copy()
        # Drop price sharply in the last bar
        df.loc[99, 'close'] = 150 # Was around 200
        
        result = strategy.evaluate(
            df, 
            has_open_position=True, 
            entry_price=Decimal("180"),
            highest_high_since_entry=Decimal("202")
        )
        assert result.signal == Signal.EXIT_LONG
        assert "CHANDELIER EXIT" in result.reason

    def test_position_sizing(self, strategy):
        capital = Decimal("10000")
        entry = Decimal("100")
        stop = Decimal("90")
        # 1% of 10000 is 100. Price risk is 10. Size should be 10.
        size = strategy.calculate_position_size(capital, entry, stop, Decimal("0.01"))
        assert size == Decimal("10")

    # =========================================================================
    # SHORT POSITION TESTS (v6.1 - Bidirectional Trading)
    # =========================================================================

    @pytest.fixture
    def trending_down_data(self):
        """Create 100 bars of trending down data for SHORT tests."""
        dates = pd.date_range(start="2025-01-01", periods=100, freq="4H")
        data = {
            'timestamp': dates,
            'open': np.linspace(200, 100, 100),
            'high': np.linspace(202, 102, 100),
            'low': np.linspace(198, 98, 100),
            'close': np.linspace(200, 100, 100),
            'volume': [1000] * 100
        }
        return pd.DataFrame(data)

    def test_short_breakdown(self, strategy, trending_down_data):
        """
        Test SHORT entry on lower channel breakdown.

        v6.1 SHORT Entry Rules:
        - Price breaks below 55-period Donchian Channel low
        - Price must be below 200 SMA (bearish trend filter)
        - Volume must be > 1.5x 20-period average
        """
        df = trending_down_data.copy()
        # Last bar breaks below the 20-period low and is below 50 SMA
        # Force a clear breakdown with high volume
        df.loc[99, 'low'] = 50  # Break below channel
        df.loc[99, 'close'] = 55
        df.loc[99, 'volume'] = 5000  # High volume confirms

        result = strategy.evaluate(df)
        assert result.signal == Signal.SHORT
        assert "SHORT ENTRY" in result.reason or "SNIPER" in result.reason

    def test_inverse_chandelier_exit(self, strategy, trending_down_data):
        """
        Test inverse Chandelier exit for SHORT positions.

        v6.1 Inverse Ratchet Logic:
        - Short Chandelier Stop = Lowest Low Since Entry + (3 × ATR)
        - The Inverse Ratchet only moves DOWN, never up
        - Triggers EXIT_SHORT when price rises above the stop
        """
        df = trending_down_data.copy()
        # Price rises sharply in last bar (short squeeze scenario)
        df.loc[99, 'close'] = 180  # Was around 100, now spiking up
        df.loc[99, 'high'] = 185

        result = strategy.evaluate(
            df,
            has_open_position=True,
            is_short_position=True,
            entry_price=Decimal("150"),
            lowest_low_since_entry=Decimal("95")  # Price went as low as 95
        )
        assert result.signal == Signal.EXIT_SHORT
        assert "INVERSE CHANDELIER" in result.reason or "EXIT" in result.reason

    def test_short_hold_in_downtrend(self, strategy, trending_down_data):
        """
        Test that SHORT position holds when price continues falling.

        The inverse ratchet should NOT trigger when price stays below
        the Lowest Low Since Entry + (3 × ATR) threshold.
        """
        df = trending_down_data.copy()
        # Price continues to fall (good for shorts)
        df.loc[99, 'close'] = 90
        df.loc[99, 'low'] = 88

        result = strategy.evaluate(
            df,
            has_open_position=True,
            is_short_position=True,
            entry_price=Decimal("150"),
            lowest_low_since_entry=Decimal("90")  # New low set
        )
        assert result.signal == Signal.HOLD
        assert "SHORT" in result.reason or "Holding" in result.reason

    def test_short_backup_exit_on_upper_breakout(self, strategy, trending_down_data):
        """
        Test backup exit for SHORTs: 55-period high breakout.

        If inverse chandelier doesn't trigger first, breaking above
        the upper channel should still exit the SHORT position.
        """
        df = trending_down_data.copy()
        # Price explodes upward, breaking upper channel
        df.loc[99, 'high'] = 250  # Above any previous high
        df.loc[99, 'close'] = 245

        result = strategy.evaluate(
            df,
            has_open_position=True,
            is_short_position=True,
            entry_price=Decimal("150"),
            lowest_low_since_entry=Decimal("100")
        )
        # Should trigger either inverse chandelier or backup exit
        assert result.signal == Signal.EXIT_SHORT
