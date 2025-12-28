"""
Session Manager - Time-Aware Trading Coordination

Detects current market session and manages trading state based on
global market hours. Crypto runs 24/7 but liquidity follows
traditional equity market patterns.

Key responsibilities:
- Detect current session (Asia/London/NY/Dead Zone)
- Track session state (Active/Reduced/Paused/Halted)
- Calculate time to next session
- Provide position size multipliers based on liquidity
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import pytz
import logging

from .schema import (
    MarketSession,
    SessionState,
    SessionEventType,
    SessionConfig,
    SessionContext,
    SessionEvent,
    LiquidityMetrics
)


logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages trading session awareness for 24/7 perpetual trading.

    All times are internally handled in UTC. Session definitions:
    - ASIA: 00:00-08:00 UTC
    - ASIA_LONDON_OVERLAP: 08:00-09:00 UTC
    - LONDON: 08:00-16:00 UTC
    - LONDON_NY_OVERLAP: 13:00-16:00 UTC
    - NEW_YORK: 13:00-22:00 UTC
    - DEAD_ZONE: 22:00-00:00 UTC
    """

    # Session hour ranges (UTC)
    # Format: (start_hour, end_hour) - end_hour is exclusive
    SESSION_HOURS = {
        MarketSession.DEAD_ZONE: (22, 24),        # 22:00-00:00
        MarketSession.ASIA: (0, 8),               # 00:00-08:00
        MarketSession.ASIA_LONDON_OVERLAP: (8, 9),  # 08:00-09:00
        MarketSession.LONDON: (9, 13),            # 09:00-13:00 (after overlap, before NY)
        MarketSession.LONDON_NY_OVERLAP: (13, 16),  # 13:00-16:00
        MarketSession.NEW_YORK: (16, 22),         # 16:00-22:00 (after overlap)
    }

    # Position size multipliers by session
    # v6.3: 24/7 crypto trading - all sessions trade at full size
    # Crypto markets have sufficient liquidity around the clock
    SESSION_MULTIPLIERS = {
        MarketSession.DEAD_ZONE: 1.0,             # 24/7 trading enabled
        MarketSession.ASIA: 1.0,                  # Full size
        MarketSession.ASIA_LONDON_OVERLAP: 1.0,   # Full size
        MarketSession.LONDON: 1.0,                # Full size
        MarketSession.LONDON_NY_OVERLAP: 1.0,     # Full size
        MarketSession.NEW_YORK: 1.0,              # Full size
    }

    def __init__(self, config: Optional[SessionConfig] = None):
        """
        Initialize session manager.

        Args:
            config: Session configuration. Uses defaults if not provided.
        """
        self.config = config or SessionConfig()
        self._current_session: Optional[MarketSession] = None
        self._current_state: SessionState = SessionState.ACTIVE
        self._last_event: Optional[SessionEvent] = None
        self._utc = pytz.UTC

    def get_current_session(self, now: Optional[datetime] = None) -> MarketSession:
        """
        Get the current market session based on UTC time.

        Args:
            now: Override current time (for testing). Uses UTC now if not provided.

        Returns:
            Current MarketSession enum value.
        """
        if now is None:
            now = datetime.now(self._utc)
        elif now.tzinfo is None:
            now = self._utc.localize(now)

        hour = now.hour

        # Check dead zone first (spans midnight)
        if hour >= self.config.dead_zone_start_hour:
            return MarketSession.DEAD_ZONE

        # 00:00-08:00 = Asia
        if 0 <= hour < 8:
            return MarketSession.ASIA

        # 08:00-09:00 = Asia/London overlap
        if 8 <= hour < 9:
            return MarketSession.ASIA_LONDON_OVERLAP

        # 09:00-13:00 = London only
        if 9 <= hour < 13:
            return MarketSession.LONDON

        # 13:00-16:00 = London/NY overlap (best liquidity)
        if 13 <= hour < 16:
            return MarketSession.LONDON_NY_OVERLAP

        # 16:00-22:00 = NY only (extends to dead_zone_start_hour if > 22)
        ny_end = max(22, self.config.dead_zone_start_hour)
        if 16 <= hour < ny_end:
            return MarketSession.NEW_YORK

        # Fallback (shouldn't reach here if dead_zone_start_hour >= 24)
        return MarketSession.DEAD_ZONE

    def is_dead_zone(self, now: Optional[datetime] = None) -> bool:
        """
        Check if currently in dead zone (low liquidity period).

        Args:
            now: Override current time (for testing).

        Returns:
            True if in dead zone, False otherwise.
        """
        return self.get_current_session(now) == MarketSession.DEAD_ZONE

    def get_session_state(self) -> SessionState:
        """Get current session state."""
        return self._current_state

    def set_session_state(self, state: SessionState) -> SessionEvent:
        """
        Update session state and return event for logging.

        Args:
            state: New session state.

        Returns:
            SessionEvent for Truth Engine logging.
        """
        previous_state = self._current_state
        self._current_state = state

        event = SessionEvent(
            event_id=SessionEvent.generate_id(),
            timestamp=datetime.now(self._utc),
            event_type=SessionEventType.STATE_CHANGE,
            session=self.get_current_session(),
            state=state,
            previous_state=previous_state
        )

        self._last_event = event
        logger.info(f"Session state changed: {previous_state.value} -> {state.value}")

        return event

    def get_position_size_multiplier(self, now: Optional[datetime] = None) -> float:
        """
        Get position size multiplier based on current session.

        Higher multiplier = more liquid session = can trade larger.

        Args:
            now: Override current time (for testing).

        Returns:
            Multiplier between 0.0 and 1.0.
        """
        session = self.get_current_session(now)
        return self.SESSION_MULTIPLIERS.get(session, 0.5)

    def get_time_to_next_session(self, now: Optional[datetime] = None) -> Tuple[MarketSession, timedelta]:
        """
        Calculate time until the next session begins.

        Args:
            now: Override current time (for testing).

        Returns:
            Tuple of (next_session, time_until).
        """
        if now is None:
            now = datetime.now(self._utc)
        elif now.tzinfo is None:
            now = self._utc.localize(now)

        current = self.get_current_session(now)
        current_hour = now.hour

        # Define session order and transitions
        session_order = [
            (MarketSession.ASIA, 0),
            (MarketSession.ASIA_LONDON_OVERLAP, 8),
            (MarketSession.LONDON, 9),
            (MarketSession.LONDON_NY_OVERLAP, 13),
            (MarketSession.NEW_YORK, 16),
            (MarketSession.DEAD_ZONE, 22),
        ]

        # Find next session
        for session, start_hour in session_order:
            if start_hour > current_hour:
                next_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
                return (session, next_start - now)

        # Wrap to next day (Asia starts at 00:00)
        next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (MarketSession.ASIA, next_day - now)

    def get_session_context(
        self,
        liquidity_metrics: Optional[Dict[str, LiquidityMetrics]] = None,
        now: Optional[datetime] = None
    ) -> SessionContext:
        """
        Get full session context for logging with decisions.

        Args:
            liquidity_metrics: Current liquidity for tracked symbols.
            now: Override current time (for testing).

        Returns:
            SessionContext dataclass for Truth Engine.
        """
        session = self.get_current_session(now)
        is_dead = session == MarketSession.DEAD_ZONE
        next_session, time_to_next = self.get_time_to_next_session(now)

        # Calculate liquidity score from metrics
        liquidity_score = 1.0
        if liquidity_metrics:
            scores = [
                1.0 if m.is_liquid else float(m.volume_ratio)
                for m in liquidity_metrics.values()
            ]
            if scores:
                liquidity_score = sum(scores) / len(scores)

        # Adjust multiplier based on state
        multiplier = self.get_position_size_multiplier(now)
        if self._current_state == SessionState.REDUCED:
            multiplier *= 0.5
        elif self._current_state == SessionState.PAUSED:
            multiplier = 0.0
        elif self._current_state == SessionState.HALTED:
            multiplier = 0.0

        return SessionContext(
            session=session,
            state=self._current_state,
            is_dead_zone=is_dead,
            time_to_next_session_minutes=int(time_to_next.total_seconds() / 60),
            liquidity_score=liquidity_score,
            position_size_multiplier=multiplier
        )

    def check_session_transition(
        self,
        now: Optional[datetime] = None
    ) -> Optional[SessionEvent]:
        """
        Check if session has changed and return event if so.

        Call this periodically (e.g., every tick) to detect transitions.

        Args:
            now: Override current time (for testing).

        Returns:
            SessionEvent if transition occurred, None otherwise.
        """
        new_session = self.get_current_session(now)

        if self._current_session is None:
            # First check - initialize without event
            self._current_session = new_session
            return None

        if new_session != self._current_session:
            previous = self._current_session
            self._current_session = new_session

            # Determine event type
            if new_session == MarketSession.DEAD_ZONE:
                event_type = SessionEventType.DEAD_ZONE_ENTER
            elif previous == MarketSession.DEAD_ZONE:
                event_type = SessionEventType.DEAD_ZONE_EXIT
            else:
                event_type = SessionEventType.TRANSITION

            # v6.3: 24/7 crypto trading - no auto-pause on dead zone
            # Keep state ACTIVE for all session transitions
            if self._current_state != SessionState.HALTED:
                self._current_state = SessionState.ACTIVE

            event = SessionEvent(
                event_id=SessionEvent.generate_id(),
                timestamp=datetime.now(self._utc) if now is None else now,
                event_type=event_type,
                session=new_session,
                previous_session=previous,
                state=self._current_state
            )

            self._last_event = event
            logger.info(
                f"Session transition: {previous.value} -> {new_session.value} "
                f"(state: {self._current_state.value})"
            )

            return event

        return None

    def should_pause_trading(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if trading should be paused.

        Args:
            now: Override current time (for testing).

        Returns:
            Tuple of (should_pause, reason).
        """
        if self._current_state == SessionState.HALTED:
            return (True, "Session state is HALTED")

        if self._current_state == SessionState.PAUSED:
            return (True, "Session state is PAUSED")

        # v6.3: 24/7 crypto trading - dead zone no longer pauses trading
        # Crypto markets have sufficient liquidity around the clock

        return (False, "")

    def get_session_info(self, now: Optional[datetime] = None) -> Dict:
        """
        Get human-readable session info for display/logging.

        Args:
            now: Override current time (for testing).

        Returns:
            Dictionary with session information.
        """
        if now is None:
            now = datetime.now(self._utc)

        session = self.get_current_session(now)
        next_session, time_to_next = self.get_time_to_next_session(now)

        return {
            "current_time_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
            "current_session": session.value,
            "session_state": self._current_state.value,
            "is_dead_zone": session == MarketSession.DEAD_ZONE,
            "next_session": next_session.value,
            "time_to_next": str(time_to_next).split(".")[0],  # Remove microseconds
            "position_multiplier": self.get_position_size_multiplier(now),
            "trading_allowed": not self.should_pause_trading(now)[0]
        }
