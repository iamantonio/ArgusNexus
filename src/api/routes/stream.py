"""
Server-Sent Events (SSE) Stream for Real-time Dashboard Updates

Pushes dashboard data to connected clients without polling.
Uses the same data sources as the REST endpoints.
"""

import asyncio
import json
import aiosqlite
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pathlib import Path
from typing import AsyncGenerator

from .metrics import get_scoreboard, get_balance, get_daily_performance

router = APIRouter()
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "v4_live_paper.db"


async def fetch_decisions(limit: int = 5):
    """Fetch recent decisions directly from database."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        query = """
            SELECT decision_id, timestamp, symbol, strategy_name, result, result_reason,
                   signal_values, risk_checks, market_context
            FROM decisions
            ORDER BY timestamp DESC
            LIMIT ?
        """

        cursor = await db.execute(query, [limit])
        rows = await cursor.fetchall()

        decisions = []
        for row in rows:
            d = dict(row)
            # Parse JSON fields
            try:
                d["signal_values"] = json.loads(d["signal_values"]) if d["signal_values"] else {}
            except:
                d["signal_values"] = {}
            try:
                d["risk_checks"] = json.loads(d["risk_checks"]) if d["risk_checks"] else {}
            except:
                d["risk_checks"] = {}
            try:
                d["market_context"] = json.loads(d["market_context"]) if d["market_context"] else {}
            except:
                d["market_context"] = {}
            decisions.append(d)

        return decisions


async def get_dashboard_data() -> dict:
    """Fetch all dashboard data using existing endpoint functions."""
    try:
        # Fetch all data in parallel
        balance_task = get_balance()
        scoreboard_task = get_scoreboard()
        decisions_task = fetch_decisions(limit=5)
        daily_perf_task = get_daily_performance(days=7)

        balance, scoreboard, decisions, daily_perf = await asyncio.gather(
            balance_task,
            scoreboard_task,
            decisions_task,
            daily_perf_task,
            return_exceptions=True
        )

        # Handle any errors gracefully
        if isinstance(balance, Exception):
            balance = {"total_equity": 0, "pnl": 0, "pnl_pct": 0}
        if isinstance(scoreboard, Exception):
            scoreboard = {"positions": [], "wins": 0, "losses": 0}
        if isinstance(decisions, Exception):
            decisions = []
        if isinstance(daily_perf, Exception):
            daily_perf = []

        return {
            "balance": balance,
            "scoreboard": scoreboard,
            "decisions": decisions,
            "dailyPerf": daily_perf
        }
    except Exception as e:
        return {"error": str(e)}


async def event_generator() -> AsyncGenerator[str, None]:
    """Generate SSE events with dashboard data."""
    last_data_hash = None

    while True:
        try:
            data = await get_dashboard_data()

            # Check for errors
            if "error" in data:
                yield f"event: error\ndata: {json.dumps(data)}\n\n"
                await asyncio.sleep(5)
                continue

            # Simple hash to detect changes (exclude timestamp fields)
            def strip_timestamps(obj):
                if isinstance(obj, dict):
                    return {k: strip_timestamps(v) for k, v in obj.items() if 'timestamp' not in k.lower()}
                elif isinstance(obj, list):
                    return [strip_timestamps(item) for item in obj]
                return obj

            data_for_hash = strip_timestamps(data)
            data_str = json.dumps(data_for_hash, sort_keys=True, default=str)
            data_hash = hash(data_str)

            # Only send if data changed (or first time)
            if data_hash != last_data_hash:
                last_data_hash = data_hash
                yield f"data: {json.dumps(data, default=str)}\n\n"
            else:
                # Send heartbeat to keep connection alive
                yield f": heartbeat\n\n"

            # Check for updates every 2 seconds
            await asyncio.sleep(2)

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(5)


@router.get("/stream")
async def stream_dashboard():
    """
    Server-Sent Events stream for real-time dashboard updates.

    Connect with EventSource in JavaScript:
    ```js
    const es = new EventSource('/api/stream');
    es.onmessage = (e) => updateDashboard(JSON.parse(e.data));
    ```
    """
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
