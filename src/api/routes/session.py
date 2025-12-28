"""
Session Status API Route

Provides current trading session information:
- Current session (Asia/London/NY/Dead Zone)
- Session state (active/paused)
- Position size multiplier
- Time until next session
"""

from fastapi import APIRouter
from datetime import datetime, timezone

# Import SessionManager
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from session import SessionManager, MarketSession

router = APIRouter()

# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get or create SessionManager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


@router.get("/session")
async def get_session_status():
    """
    Get current trading session status.

    Returns:
        - session: Current session name (asia, london, new_york, dead_zone, etc.)
        - session_display: Human-readable session name
        - state: Session state (active, paused)
        - is_dead_zone: Whether currently in dead zone
        - is_trading_allowed: Whether new entries are allowed
        - position_multiplier: Current position size multiplier (0.0-1.0)
        - next_session: Name of next session
        - next_session_in: Time until next session (formatted string)
        - next_session_seconds: Seconds until next session
        - timestamp: Current UTC timestamp
    """
    sm = get_session_manager()

    # Get current session info
    session_info = sm.get_session_info()
    session = sm.get_current_session()

    # Check if trading is allowed
    should_pause, pause_reason = sm.should_pause_trading()

    # Format session display name
    session_display_map = {
        MarketSession.ASIA: "Asia Session",
        MarketSession.ASIA_LONDON_OVERLAP: "Asia/London Overlap",
        MarketSession.LONDON: "London Session",
        MarketSession.LONDON_NY_OVERLAP: "London/NY Overlap",
        MarketSession.NEW_YORK: "New York Session",
        MarketSession.DEAD_ZONE: "Dead Zone",
    }

    # Format next session time
    next_session_seconds = session_info.get("next_session_in_seconds", 0)
    if next_session_seconds > 0:
        hours = int(next_session_seconds // 3600)
        minutes = int((next_session_seconds % 3600) // 60)
        if hours > 0:
            next_session_formatted = f"{hours}h {minutes}m"
        else:
            next_session_formatted = f"{minutes}m"
    else:
        next_session_formatted = "now"

    return {
        "session": session.value,
        "session_display": session_display_map.get(session, session.value),
        "state": session_info.get("state", "active"),
        "is_dead_zone": session_info.get("is_dead_zone", False),
        "is_trading_allowed": not should_pause,
        "pause_reason": pause_reason if should_pause else None,
        "position_multiplier": session_info.get("position_multiplier", 1.0),
        "next_session": session_info.get("next_session", "unknown"),
        "next_session_in": next_session_formatted,
        "next_session_seconds": next_session_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
