"""
Session Management Schema - 24-Hour Perpetual Trading

Defines the session types, states, and metrics for time-aware trading.
Crypto markets run 24/7 but liquidity follows traditional market hours.

Session Definitions (UTC):
- ASIA: 00:00-08:00 UTC (Tokyo/Singapore/Hong Kong)
- LONDON: 08:00-16:00 UTC (London/Frankfurt)
- NEW_YORK: 13:00-22:00 UTC (NYC, overlaps with London 13:00-16:00)
- DEAD_ZONE: 22:00-00:00 UTC (low liquidity, auto-pause)
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import uuid


class MarketSession(Enum):
    """
    Global trading sessions based on traditional market hours.

    Even crypto follows these patterns - volume and volatility
    spike during equity market hours.
    """
    ASIA = "asia"                      # 00:00-08:00 UTC
    ASIA_LONDON_OVERLAP = "asia_london"  # 08:00-09:00 UTC
    LONDON = "london"                  # 08:00-16:00 UTC
    LONDON_NY_OVERLAP = "london_ny"    # 13:00-16:00 UTC (highest liquidity)
    NEW_YORK = "new_york"              # 13:00-22:00 UTC
    DEAD_ZONE = "dead_zone"            # 22:00-00:00 UTC (auto-pause)


class SessionState(Enum):
    """
    Trading state during a session.

    Controls what trading actions are allowed.
    """
    ACTIVE = "active"       # Normal trading - all actions allowed
    REDUCED = "reduced"     # Reduced position sizing (low liquidity)
    PAUSED = "paused"       # No new entries, exits only
    HALTED = "halted"       # Emergency halt - no trading


class SessionEventType(Enum):
    """Types of session events for logging."""
    TRANSITION = "transition"           # Session changed
    DEAD_ZONE_ENTER = "dead_zone_enter" # Entered dead zone
    DEAD_ZONE_EXIT = "dead_zone_exit"   # Exited dead zone
    LIQUIDITY_LOW = "liquidity_low"     # Liquidity dropped below threshold
    LIQUIDITY_RESTORED = "liquidity_restored"  # Liquidity restored
    STATE_CHANGE = "state_change"       # SessionState changed


@dataclass
class LiquidityMetrics:
    """
    Real-time liquidity metrics for a symbol.

    Used to detect dead zones and adjust position sizing.
    """
    symbol: str
    timestamp: datetime

    # Spread metrics
    bid: Decimal
    ask: Decimal
    spread_bps: Decimal              # Bid-ask spread in basis points

    # Volume metrics
    volume_24h: Decimal              # 24-hour volume
    volume_current: Decimal          # Current period volume
    volume_ratio: Decimal            # Current / 24h average (1.0 = normal)

    # Order book depth (optional, if available)
    depth_score: Optional[Decimal] = None  # 0-100 scale

    # Session context
    session: Optional[MarketSession] = None
    is_dead_zone: bool = False

    @property
    def is_liquid(self) -> bool:
        """Check if liquidity meets minimum thresholds."""
        # Default thresholds: spread < 50 bps, volume > 30% of average
        return self.spread_bps < Decimal("50") and self.volume_ratio > Decimal("0.3")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": str(self.bid),
            "ask": str(self.ask),
            "spread_bps": str(self.spread_bps),
            "volume_24h": str(self.volume_24h),
            "volume_current": str(self.volume_current),
            "volume_ratio": str(self.volume_ratio),
            "depth_score": str(self.depth_score) if self.depth_score else None,
            "session": self.session.value if self.session else None,
            "is_dead_zone": self.is_dead_zone,
            "is_liquid": self.is_liquid
        }


@dataclass
class SessionConfig:
    """
    Configuration for session management.

    Loaded from config.yaml.
    """
    timezone: str = "UTC"

    # Dead zone hours (UTC)
    dead_zone_start_hour: int = 22
    dead_zone_end_hour: int = 0

    # Liquidity thresholds
    min_liquidity_ratio: float = 0.3   # Volume must be 30%+ of 24h average
    max_spread_bps: float = 50.0       # Max spread in basis points

    # Session reset (when daily limits reset)
    session_reset_hour: int = 0        # UTC hour

    # Notification settings
    notify_on_dead_zone: bool = True
    notify_on_session_change: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionConfig":
        """Create from config dictionary."""
        return cls(
            timezone=data.get("timezone", "UTC"),
            dead_zone_start_hour=data.get("dead_zone_start_hour", 22),
            dead_zone_end_hour=data.get("dead_zone_end_hour", 0),
            min_liquidity_ratio=data.get("min_liquidity_ratio", 0.3),
            max_spread_bps=data.get("max_spread_bps", 50.0),
            session_reset_hour=data.get("session_reset_hour", 0),
            notify_on_dead_zone=data.get("notify_on_dead_zone", True),
            notify_on_session_change=data.get("notify_on_session_change", False)
        )


@dataclass
class SessionEvent:
    """
    Records a session-related event for Truth Engine.

    Logged to session_events table.
    """
    event_id: str
    timestamp: datetime
    event_type: SessionEventType
    session: MarketSession
    previous_session: Optional[MarketSession] = None
    state: SessionState = SessionState.ACTIVE
    previous_state: Optional[SessionState] = None
    liquidity_metrics: Optional[Dict[str, LiquidityMetrics]] = None
    affected_symbols: Optional[List[str]] = None
    actions_taken: Optional[List[str]] = None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for database storage."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "session": self.session.value,
            "previous_session": self.previous_session.value if self.previous_session else None,
            "state": self.state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "liquidity_metrics": json.dumps({
                k: v.to_dict() for k, v in self.liquidity_metrics.items()
            }) if self.liquidity_metrics else None,
            "affected_symbols": json.dumps(self.affected_symbols) if self.affected_symbols else None,
            "actions_taken": json.dumps(self.actions_taken) if self.actions_taken else None
        }


@dataclass
class SessionContext:
    """
    Session context to attach to trading decisions.

    This gets logged with every Decision in the Truth Engine.
    """
    session: MarketSession
    state: SessionState
    is_dead_zone: bool
    time_to_next_session_minutes: int
    liquidity_score: float              # 0.0-1.0
    position_size_multiplier: float     # 1.0 = full size, 0.5 = half

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "session": self.session.value,
            "state": self.state.value,
            "is_dead_zone": self.is_dead_zone,
            "time_to_next_session_minutes": self.time_to_next_session_minutes,
            "liquidity_score": self.liquidity_score,
            "position_size_multiplier": self.position_size_multiplier
        }


# =============================================================================
# SQL Schema for session_events table
# =============================================================================

SESSION_SQL_SCHEMA = """
-- Session Events Table: Track session transitions and state changes
-- Part of the Truth Engine - every session event is logged

CREATE TABLE IF NOT EXISTS session_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,          -- SessionEventType enum value
    session TEXT NOT NULL,             -- MarketSession enum value
    previous_session TEXT,
    state TEXT NOT NULL,               -- SessionState enum value
    previous_state TEXT,
    liquidity_metrics TEXT,            -- JSON: {symbol: LiquidityMetrics}
    affected_symbols TEXT,             -- JSON array of symbols
    actions_taken TEXT                 -- JSON array of actions
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_session_events_timestamp ON session_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_session_events_type ON session_events(event_type);
CREATE INDEX IF NOT EXISTS idx_session_events_session ON session_events(session);

-- Multi-timeframe signals table
CREATE TABLE IF NOT EXISTS multi_timeframe_signals (
    signal_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,           -- '1h', '4h', '1d'
    signal TEXT NOT NULL,              -- 'long', 'short', 'hold'
    confidence REAL,
    signal_values TEXT,                -- JSON: strategy output
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE INDEX IF NOT EXISTS idx_mtf_decision ON multi_timeframe_signals(decision_id);
CREATE INDEX IF NOT EXISTS idx_mtf_symbol ON multi_timeframe_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_mtf_timeframe ON multi_timeframe_signals(timeframe);
"""
