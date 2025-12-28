"""
Unit Tests for Session Management Module

Tests session detection, dead zone handling, and liquidity monitoring.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import pytz

from src.session import (
    SessionManager,
    SessionConfig,
    MarketSession,
    SessionState,
    SessionEventType,
    LiquidityMonitor,
    LiquidityMetrics
)


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_get_current_session_asia(self):
        """Test Asia session detection (00:00-08:00 UTC)."""
        manager = SessionManager()

        # 03:00 UTC = Asia
        test_time = datetime(2024, 1, 15, 3, 0, 0, tzinfo=pytz.UTC)
        assert manager.get_current_session(test_time) == MarketSession.ASIA

        # 07:59 UTC = still Asia
        test_time = datetime(2024, 1, 15, 7, 59, 0, tzinfo=pytz.UTC)
        assert manager.get_current_session(test_time) == MarketSession.ASIA

    def test_get_current_session_asia_london_overlap(self):
        """Test Asia/London overlap (08:00-09:00 UTC)."""
        manager = SessionManager()

        test_time = datetime(2024, 1, 15, 8, 30, 0, tzinfo=pytz.UTC)
        assert manager.get_current_session(test_time) == MarketSession.ASIA_LONDON_OVERLAP

    def test_get_current_session_london(self):
        """Test London session (09:00-13:00 UTC)."""
        manager = SessionManager()

        test_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=pytz.UTC)
        assert manager.get_current_session(test_time) == MarketSession.LONDON

    def test_get_current_session_london_ny_overlap(self):
        """Test London/NY overlap - highest liquidity (13:00-16:00 UTC)."""
        manager = SessionManager()

        test_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
        assert manager.get_current_session(test_time) == MarketSession.LONDON_NY_OVERLAP

    def test_get_current_session_new_york(self):
        """Test New York session (16:00-22:00 UTC)."""
        manager = SessionManager()

        test_time = datetime(2024, 1, 15, 18, 0, 0, tzinfo=pytz.UTC)
        assert manager.get_current_session(test_time) == MarketSession.NEW_YORK

    def test_get_current_session_dead_zone(self):
        """Test dead zone detection (22:00-00:00 UTC)."""
        manager = SessionManager()

        # 22:00 UTC = dead zone
        test_time = datetime(2024, 1, 15, 22, 0, 0, tzinfo=pytz.UTC)
        assert manager.get_current_session(test_time) == MarketSession.DEAD_ZONE

        # 23:30 UTC = still dead zone
        test_time = datetime(2024, 1, 15, 23, 30, 0, tzinfo=pytz.UTC)
        assert manager.get_current_session(test_time) == MarketSession.DEAD_ZONE

    def test_is_dead_zone(self):
        """Test is_dead_zone helper."""
        manager = SessionManager()

        # Dead zone
        dead_time = datetime(2024, 1, 15, 22, 30, 0, tzinfo=pytz.UTC)
        assert manager.is_dead_zone(dead_time) is True

        # Not dead zone
        active_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
        assert manager.is_dead_zone(active_time) is False

    def test_session_multipliers(self):
        """Test position size multipliers by session."""
        manager = SessionManager()

        # Dead zone = 0.0 (no trading)
        dead_time = datetime(2024, 1, 15, 22, 30, 0, tzinfo=pytz.UTC)
        assert manager.get_position_size_multiplier(dead_time) == 0.0

        # London/NY overlap = 1.0 (best liquidity)
        overlap_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
        assert manager.get_position_size_multiplier(overlap_time) == 1.0

        # Asia = 0.7 (moderate)
        asia_time = datetime(2024, 1, 15, 3, 0, 0, tzinfo=pytz.UTC)
        assert manager.get_position_size_multiplier(asia_time) == 0.7

    def test_time_to_next_session(self):
        """Test calculating time to next session."""
        manager = SessionManager()

        # At 10:00 UTC (London), next session is London/NY overlap at 13:00
        test_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=pytz.UTC)
        next_session, time_delta = manager.get_time_to_next_session(test_time)

        assert next_session == MarketSession.LONDON_NY_OVERLAP
        assert time_delta.total_seconds() == 3 * 3600  # 3 hours

    def test_session_transition_detection(self):
        """Test detection of session transitions."""
        manager = SessionManager()

        # Initial state - no event
        time1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=pytz.UTC)
        event = manager.check_session_transition(time1)
        assert event is None  # First check, no transition

        # Move to different session
        time2 = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
        event = manager.check_session_transition(time2)
        assert event is not None
        assert event.session == MarketSession.LONDON_NY_OVERLAP
        assert event.previous_session == MarketSession.LONDON

    def test_dead_zone_enter_event(self):
        """Test dead zone enter event type."""
        manager = SessionManager()

        # Start in NY session
        time1 = datetime(2024, 1, 15, 20, 0, 0, tzinfo=pytz.UTC)
        manager.check_session_transition(time1)

        # Move to dead zone
        time2 = datetime(2024, 1, 15, 22, 30, 0, tzinfo=pytz.UTC)
        event = manager.check_session_transition(time2)

        assert event is not None
        assert event.event_type == SessionEventType.DEAD_ZONE_ENTER
        assert event.session == MarketSession.DEAD_ZONE

    def test_dead_zone_exit_event(self):
        """Test dead zone exit event type."""
        manager = SessionManager()

        # Start in dead zone
        time1 = datetime(2024, 1, 15, 22, 30, 0, tzinfo=pytz.UTC)
        manager.check_session_transition(time1)

        # Move to Asia (exit dead zone)
        time2 = datetime(2024, 1, 16, 1, 0, 0, tzinfo=pytz.UTC)
        event = manager.check_session_transition(time2)

        assert event is not None
        assert event.event_type == SessionEventType.DEAD_ZONE_EXIT
        assert event.session == MarketSession.ASIA

    def test_should_pause_trading(self):
        """Test trading pause logic."""
        manager = SessionManager()

        # Active session - no pause
        active_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
        should_pause, reason = manager.should_pause_trading(active_time)
        assert should_pause is False

        # Dead zone - should pause
        dead_time = datetime(2024, 1, 15, 22, 30, 0, tzinfo=pytz.UTC)
        should_pause, reason = manager.should_pause_trading(dead_time)
        assert should_pause is True
        assert "dead zone" in reason.lower()

    def test_session_state_changes(self):
        """Test manual session state changes."""
        manager = SessionManager()

        # Default state is ACTIVE
        assert manager.get_session_state() == SessionState.ACTIVE

        # Pause trading
        event = manager.set_session_state(SessionState.PAUSED)
        assert event.state == SessionState.PAUSED
        assert event.previous_state == SessionState.ACTIVE
        assert manager.get_session_state() == SessionState.PAUSED

        # Paused state should pause trading
        should_pause, _ = manager.should_pause_trading()
        assert should_pause is True

    def test_session_context(self):
        """Test getting full session context."""
        manager = SessionManager()

        test_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
        context = manager.get_session_context(now=test_time)

        assert context.session == MarketSession.LONDON_NY_OVERLAP
        assert context.state == SessionState.ACTIVE
        assert context.is_dead_zone is False
        assert context.position_size_multiplier == 1.0
        assert context.time_to_next_session_minutes > 0


class TestLiquidityMonitor:
    """Tests for LiquidityMonitor class."""

    def test_update_and_get_metrics(self):
        """Test updating and retrieving liquidity metrics."""
        monitor = LiquidityMonitor()

        metrics = monitor.update(
            symbol="BTC-USD",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            volume=Decimal("1000"),
            volume_24h=Decimal("24000")
        )

        assert metrics.symbol == "BTC-USD"
        assert metrics.spread_bps == Decimal("2")  # 10/50005 * 10000 ≈ 2 bps

        # Retrieve same metrics
        retrieved = monitor.get_metrics("BTC-USD")
        assert retrieved is not None
        assert retrieved.symbol == "BTC-USD"

    def test_is_liquid(self):
        """Test liquidity threshold checking."""
        monitor = LiquidityMonitor()

        # Good liquidity
        monitor.update(
            symbol="BTC-USD",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            volume=Decimal("1000"),
            volume_24h=Decimal("24000")  # 1000/24000*24 = 1.0 ratio
        )
        assert monitor.is_liquid("BTC-USD") is True

        # Bad liquidity - wide spread
        monitor.update(
            symbol="SHITCOIN-USD",
            bid=Decimal("1.00"),
            ask=Decimal("1.10"),  # 10% spread = 1000 bps
            volume=Decimal("100"),
            volume_24h=Decimal("2400")
        )
        assert monitor.is_liquid("SHITCOIN-USD") is False

    def test_liquidity_score(self):
        """Test liquidity scoring."""
        monitor = LiquidityMonitor()

        # Excellent liquidity
        monitor.update(
            symbol="BTC-USD",
            bid=Decimal("50000"),
            ask=Decimal("50005"),  # 1 bps spread
            volume=Decimal("2000"),
            volume_24h=Decimal("24000")  # 2x normal volume
        )
        score = monitor.get_liquidity_score("BTC-USD")
        assert score > 0.8  # Should be high

        # Poor liquidity
        monitor.update(
            symbol="ILLIQUID-USD",
            bid=Decimal("100"),
            ask=Decimal("105"),  # 500 bps spread
            volume=Decimal("10"),
            volume_24h=Decimal("2400")  # 0.1x normal volume
        )
        score = monitor.get_liquidity_score("ILLIQUID-USD")
        assert score < 0.3  # Should be low

    def test_position_adjustment(self):
        """Test position size adjustment based on liquidity."""
        monitor = LiquidityMonitor()

        # Good liquidity = higher multiplier
        monitor.update(
            symbol="BTC-USD",
            bid=Decimal("50000"),
            ask=Decimal("50005"),
            volume=Decimal("2000"),
            volume_24h=Decimal("24000")
        )
        adj = monitor.get_position_adjustment("BTC-USD")
        assert adj >= Decimal("0.7")

        # Unknown symbol = minimum multiplier
        adj = monitor.get_position_adjustment("UNKNOWN-USD")
        assert adj == Decimal("0.3")

    def test_stale_data_detection(self):
        """Test detection of stale liquidity data."""
        monitor = LiquidityMonitor()

        # Fresh data
        monitor.update(
            symbol="BTC-USD",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            volume=Decimal("1000"),
            volume_24h=Decimal("24000")
        )
        assert monitor.is_stale("BTC-USD", max_age_seconds=60) is False

        # Stale check with 0 second max age
        assert monitor.is_stale("BTC-USD", max_age_seconds=0) is True

        # Unknown symbol is always stale
        assert monitor.is_stale("UNKNOWN-USD") is True

    def test_aggregate_score(self):
        """Test aggregate liquidity score across symbols."""
        monitor = LiquidityMonitor()

        # Add two symbols
        monitor.update("BTC-USD", Decimal("50000"), Decimal("50005"),
                       Decimal("2000"), Decimal("24000"))
        monitor.update("ETH-USD", Decimal("3000"), Decimal("3002"),
                       Decimal("1500"), Decimal("18000"))

        score = monitor.get_aggregate_score()
        assert 0.0 <= score <= 1.0

        # Check specific symbols
        score = monitor.get_aggregate_score(["BTC-USD"])
        assert 0.0 <= score <= 1.0


class TestSignalAggregator:
    """Tests for SignalAggregator conflict resolution."""

    def test_aligned_long_signals(self):
        """Test when all timeframes agree on LONG."""
        from src.engine import SignalAggregator, TimeframeSignal, SignalType

        aggregator = SignalAggregator()

        signals = {
            "1h": TimeframeSignal(
                timeframe="1h",
                signal=SignalType.LONG,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                signal_values={},
                weight=0.4
            ),
            "4h": TimeframeSignal(
                timeframe="4h",
                signal=SignalType.LONG,
                confidence=0.9,
                timestamp=datetime.utcnow(),
                signal_values={},
                weight=0.6
            )
        }

        result = aggregator.aggregate(signals)

        assert result.signal == SignalType.LONG
        assert result.is_aligned is True
        assert result.confidence > 0.8  # Weighted average

    def test_conflicting_signals_hold(self):
        """Test that conflicting signals result in HOLD."""
        from src.engine import SignalAggregator, TimeframeSignal, SignalType

        aggregator = SignalAggregator()

        signals = {
            "1h": TimeframeSignal(
                timeframe="1h",
                signal=SignalType.LONG,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                signal_values={},
                weight=0.4
            ),
            "4h": TimeframeSignal(
                timeframe="4h",
                signal=SignalType.SHORT,  # Conflict!
                confidence=0.9,
                timestamp=datetime.utcnow(),
                signal_values={},
                weight=0.6
            )
        }

        result = aggregator.aggregate(signals)

        assert result.signal == SignalType.HOLD
        assert result.is_aligned is False
        assert result.conflict_resolution_used is not None

    def test_one_active_one_hold(self):
        """Test when one timeframe is active, other is HOLD."""
        from src.engine import SignalAggregator, TimeframeSignal, SignalType

        aggregator = SignalAggregator()

        signals = {
            "1h": TimeframeSignal(
                timeframe="1h",
                signal=SignalType.LONG,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                signal_values={},
                weight=0.4
            ),
            "4h": TimeframeSignal(
                timeframe="4h",
                signal=SignalType.HOLD,
                confidence=0.5,
                timestamp=datetime.utcnow(),
                signal_values={},
                weight=0.6
            )
        }

        result = aggregator.aggregate(signals)

        # Should follow the active signal
        assert result.signal == SignalType.LONG

    def test_close_signal_priority(self):
        """Test that CLOSE signal takes priority."""
        from src.engine import SignalAggregator, TimeframeSignal, SignalType

        aggregator = SignalAggregator()

        signals = {
            "1h": TimeframeSignal(
                timeframe="1h",
                signal=SignalType.LONG,
                confidence=0.8,
                timestamp=datetime.utcnow(),
                signal_values={},
                weight=0.4
            ),
            "4h": TimeframeSignal(
                timeframe="4h",
                signal=SignalType.CLOSE,
                confidence=0.9,
                timestamp=datetime.utcnow(),
                signal_values={},
                weight=0.6
            )
        }

        result = aggregator.aggregate(signals)

        # CLOSE takes priority
        assert result.signal == SignalType.CLOSE


class TestPortfolioRiskAggregator:
    """Tests for PortfolioRiskAggregator."""

    def test_total_exposure_limit(self):
        """Test total portfolio exposure limit."""
        from src.risk.portfolio import PortfolioRiskAggregator, PortfolioRiskConfig

        config = PortfolioRiskConfig(max_portfolio_exposure_pct=80.0)
        aggregator = PortfolioRiskAggregator(config, Decimal("10000"))

        # Add existing position (50% exposure)
        aggregator.update_position("BTC-USD", Decimal("5000"))

        # Try to add another 40% - should fail (total 90% > 80%)
        result = aggregator.can_open_position("ETH-USD", Decimal("4000"))
        assert result.approved is False

        # Try to add 25% - should pass (total 75% < 80%)
        result = aggregator.can_open_position("ETH-USD", Decimal("2500"))
        assert result.approved is True

    def test_correlation_group_limit(self):
        """Test correlation group exposure limit."""
        from src.risk.portfolio import PortfolioRiskAggregator, PortfolioRiskConfig

        config = PortfolioRiskConfig(
            max_portfolio_exposure_pct=100.0,  # Don't trigger total limit
            max_correlation_group_pct=50.0,
            correlation_groups={"large_cap": ["BTC-USD", "ETH-USD"]}
        )
        aggregator = PortfolioRiskAggregator(config, Decimal("10000"))

        # Add BTC position (40% exposure)
        aggregator.update_position("BTC-USD", Decimal("4000"))

        # Try to add ETH for 20% - should fail (group total 60% > 50%)
        result = aggregator.can_open_position("ETH-USD", Decimal("2000"))
        assert result.approved is False

        # Try to add ETH for 5% - should pass (group total 45% < 50%)
        result = aggregator.can_open_position("ETH-USD", Decimal("500"))
        assert result.approved is True

    def test_combined_daily_loss_limit(self):
        """Test combined daily loss limit across assets."""
        from src.risk.portfolio import PortfolioRiskAggregator, PortfolioRiskConfig

        config = PortfolioRiskConfig(max_combined_daily_loss_pct=5.0)
        aggregator = PortfolioRiskAggregator(config, Decimal("10000"))

        # Add losses to multiple assets
        aggregator.update_position("BTC-USD", Decimal("1000"),
                                   unrealized_pnl=Decimal("-200"),
                                   realized_pnl=Decimal("-100"))
        aggregator.update_position("ETH-USD", Decimal("1000"),
                                   unrealized_pnl=Decimal("-150"),
                                   realized_pnl=Decimal("-100"))

        # Total loss = -550 = 5.5% > 5% limit
        result = aggregator.can_open_position("SOL-USD", Decimal("500"))
        assert result.approved is False

    def test_portfolio_state(self):
        """Test portfolio state reporting."""
        from src.risk.portfolio import PortfolioRiskAggregator, PortfolioRiskConfig

        config = PortfolioRiskConfig()
        aggregator = PortfolioRiskAggregator(config, Decimal("10000"))

        aggregator.update_position("BTC-USD", Decimal("3000"),
                                   unrealized_pnl=Decimal("100"))
        aggregator.update_position("ETH-USD", Decimal("2000"),
                                   unrealized_pnl=Decimal("-50"))

        state = aggregator.get_portfolio_state()

        assert state["total_capital"] == "10000"
        assert state["total_exposure"] == "5000"
        assert float(state["exposure_percent"]) == 50.0
        assert state["asset_count"] == 2
