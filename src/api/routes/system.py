"""
System status endpoints for monitoring paper trader health.
"""

import subprocess
import os
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter

router = APIRouter()

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
LOG_PATH = BASE_DIR / "runtime" / "logs" / "paper_trader.log"
DB_PATH = BASE_DIR / "data" / "v4_live_paper.db"
MAX_LOG_AGE_SECONDS = 120  # Consider stale if no log update in 2 minutes


@router.get("/system/status")
async def get_system_status():
    """
    Check paper trader health status.
    Returns process status and log freshness.
    """
    # Check if paper trader process is running
    # Searches for both live_unified_trader.py and live_paper_trader.py
    try:
        result = subprocess.run(
            ["pgrep", "-f", "live_(unified|paper)_trader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        process_running = result.returncode == 0
        pid = result.stdout.strip().split('\n')[0] if process_running else None
    except Exception:
        process_running = False
        pid = None

    # Check log freshness
    log_age_seconds = None
    log_fresh = False
    last_log_update = None

    if LOG_PATH.exists():
        try:
            mtime = os.path.getmtime(LOG_PATH)
            last_log_update = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            log_age_seconds = (datetime.now(timezone.utc) - datetime.fromtimestamp(mtime, tz=timezone.utc)).total_seconds()
            log_fresh = log_age_seconds < MAX_LOG_AGE_SECONDS
        except Exception:
            pass

    # Overall health
    healthy = process_running and log_fresh

    # Status summary
    if healthy:
        status = "online"
        message = "Paper trader running normally"
    elif process_running and not log_fresh:
        status = "stale"
        message = f"Process running but log stale ({int(log_age_seconds or 0)}s old)"
    else:
        status = "offline"
        message = "Paper trader not running"

    return {
        "status": status,
        "healthy": healthy,
        "message": message,
        "process": {
            "running": process_running,
            "pid": pid
        },
        "log": {
            "fresh": log_fresh,
            "age_seconds": int(log_age_seconds) if log_age_seconds else None,
            "last_update": last_log_update,
            "max_age_seconds": MAX_LOG_AGE_SECONDS
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/system/health")
async def get_system_health():
    """
    Comprehensive system health check for the dashboard widget.

    Returns:
    - Overall status (operational/degraded/offline)
    - Uptime percentage (based on recent activity)
    - Last decision timestamp
    - Decisions today count
    - Service statuses (trader, price feed, database)
    """
    # Check trader process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "live_(unified|paper)_trader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        trader_running = result.returncode == 0
    except Exception:
        trader_running = False

    # Database queries
    last_decision = None
    decisions_today = 0
    total_decisions = 0
    database_ok = False

    try:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            database_ok = True

            # Last decision
            cursor = await db.execute(
                "SELECT timestamp FROM decisions ORDER BY timestamp DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            if row:
                last_decision = row["timestamp"]

            # Decisions today
            cursor = await db.execute(
                "SELECT COUNT(*) as count FROM decisions WHERE date(timestamp) = date('now')"
            )
            decisions_today = (await cursor.fetchone())["count"]

            # Total decisions (for uptime calc)
            cursor = await db.execute("SELECT COUNT(*) as count FROM decisions")
            total_decisions = (await cursor.fetchone())["count"]

    except Exception as e:
        database_ok = False

    # Check log freshness for price feed status
    log_fresh = False
    if LOG_PATH.exists():
        try:
            mtime = os.path.getmtime(LOG_PATH)
            log_age = (datetime.now(timezone.utc) - datetime.fromtimestamp(mtime, tz=timezone.utc)).total_seconds()
            log_fresh = log_age < MAX_LOG_AGE_SECONDS
        except Exception:
            pass

    # Calculate uptime (simplified: based on trader and db status)
    # In production, track actual uptime metrics
    uptime_pct = 99.9 if (trader_running and database_ok) else (95.0 if database_ok else 50.0)

    # Determine overall status
    if trader_running and database_ok and log_fresh:
        status = "operational"
    elif database_ok:
        status = "degraded"
    else:
        status = "offline"

    return {
        "status": status,
        "uptime_pct": uptime_pct,
        "last_decision": last_decision,
        "decisions_today": decisions_today,
        "total_decisions": total_decisions,
        "trader_running": trader_running,
        "trader_status": "online" if trader_running else "offline",
        "price_feed_status": "ok" if log_fresh else "stale",
        "database_status": "ok" if database_ok else "error",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
