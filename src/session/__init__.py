"""
Session Management Module - 24-Hour Perpetual Trading

Provides session awareness for time-based trading decisions:
- Session detection (Asia/London/NY/Dead Zone)
- Liquidity monitoring
- Position size adjustments based on market conditions

Usage:
    from src.session import SessionManager, LiquidityMonitor, MarketSession

    session_mgr = SessionManager()
    if session_mgr.is_dead_zone():
        print("Trading paused - low liquidity period")
"""

from .schema import (
    MarketSession,
    SessionState,
    SessionEventType,
    SessionConfig,
    SessionContext,
    SessionEvent,
    LiquidityMetrics,
    SESSION_SQL_SCHEMA
)

from .manager import SessionManager
from .liquidity import LiquidityMonitor


__all__ = [
    # Enums
    "MarketSession",
    "SessionState",
    "SessionEventType",

    # Dataclasses
    "SessionConfig",
    "SessionContext",
    "SessionEvent",
    "LiquidityMetrics",

    # Classes
    "SessionManager",
    "LiquidityMonitor",

    # SQL
    "SESSION_SQL_SCHEMA"
]
