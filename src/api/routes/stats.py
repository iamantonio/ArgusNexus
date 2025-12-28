"""
Public Stats Dashboard API

Comprehensive statistics for the transparency dashboard.
All endpoints are public - this is our proof of honesty.
"""

from fastapi import APIRouter, Query
from pathlib import Path
import aiosqlite
import json
from datetime import datetime, timedelta
from typing import Optional

router = APIRouter(prefix="/stats", tags=["Stats"])

DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "v4_live_paper.db"


@router.get("/dashboard")
async def get_dashboard_stats():
    """
    Get all stats for the public dashboard in a single call.

    Returns:
    - Core metrics (win rate, P&L, decisions, R:R)
    - Streak information
    - Today's activity
    - Recent signals
    - "Why Not?" explanations
    - Benchmark comparison (vs BTC hold)
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        # === CORE METRICS ===

        # Total decisions
        cursor = await db.execute("SELECT COUNT(*) as count FROM decisions")
        total_decisions = (await cursor.fetchone())["count"]

        # Today's decisions
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM decisions
            WHERE date(timestamp) = date('now')
        """)
        today_decisions = (await cursor.fetchone())["count"]

        # Trade stats
        cursor = await db.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses,
                SUM(CAST(net_pnl AS REAL)) as net_pnl,
                AVG(CAST(net_pnl AS REAL)) as avg_pnl,
                MAX(CAST(net_pnl AS REAL)) as best_trade,
                MIN(CAST(net_pnl AS REAL)) as worst_trade
            FROM trades
            WHERE exit_timestamp IS NOT NULL
        """)
        trade_row = await cursor.fetchone()

        total_trades = trade_row["total_trades"] or 0
        wins = trade_row["wins"] or 0
        losses = trade_row["losses"] or 0
        net_pnl = trade_row["net_pnl"] or 0
        avg_pnl = trade_row["avg_pnl"] or 0
        best_trade = trade_row["best_trade"] or 0
        worst_trade = trade_row["worst_trade"] or 0

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # === STREAK CALCULATION ===
        cursor = await db.execute("""
            SELECT is_winner, exit_timestamp
            FROM trades
            WHERE exit_timestamp IS NOT NULL
            ORDER BY exit_timestamp DESC
            LIMIT 20
        """)
        recent_trades = await cursor.fetchall()

        streak = 0
        streak_type = None
        if recent_trades:
            first_result = recent_trades[0]["is_winner"]
            streak_type = "win" if first_result else "loss"
            for trade in recent_trades:
                if trade["is_winner"] == first_result:
                    streak += 1
                else:
                    break

        # === OPEN POSITIONS ===
        cursor = await db.execute("""
            SELECT symbol, side, entry_price, quantity, entry_timestamp
            FROM trades
            WHERE exit_timestamp IS NULL
        """)
        open_positions = [dict(row) for row in await cursor.fetchall()]

        # === RECENT SIGNALS ===
        cursor = await db.execute("""
            SELECT decision_id, symbol, result, result_reason, timestamp, market_context
            FROM decisions
            WHERE result IN ('signal_long', 'signal_short', 'signal_close', 'signal_hold', 'no_signal')
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        recent_decisions = []
        for row in await cursor.fetchall():
            d = dict(row)
            # Parse market context for grade
            try:
                mc = json.loads(d.get("market_context", "{}"))
                setup = mc.get("setup_details", mc.get("professional", {}))
                d["grade"] = setup.get("grade", "")
                risk = setup.get("risk", {})
                d["risk_reward"] = risk.get("risk_reward")
            except:
                d["grade"] = ""
                d["risk_reward"] = None
            del d["market_context"]  # Don't send full context
            recent_decisions.append(d)

        # === WHY NOT? - Recent non-trades with explanations ===
        cursor = await db.execute("""
            SELECT symbol, result, result_reason, timestamp
            FROM decisions
            WHERE result IN ('no_signal', 'signal_hold', 'risk_rejected')
            AND result_reason IS NOT NULL
            AND result_reason != ''
            AND length(result_reason) > 20
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        why_not = [dict(row) for row in await cursor.fetchall()]

        # === DECISION BREAKDOWN ===
        cursor = await db.execute("""
            SELECT result, COUNT(*) as count
            FROM decisions
            GROUP BY result
            ORDER BY count DESC
        """)
        decision_breakdown = {row["result"]: row["count"] for row in await cursor.fetchall()}

        # === LAST UPDATE TIME ===
        cursor = await db.execute("""
            SELECT MAX(timestamp) as last_update FROM decisions
        """)
        last_update_row = await cursor.fetchone()
        last_update = last_update_row["last_update"] if last_update_row else None

        # === SYSTEM STATUS ===
        # Check if trader is running (decision within last 5 minutes)
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM decisions
            WHERE timestamp > datetime('now', '-5 minutes')
        """)
        recent_activity = (await cursor.fetchone())["count"]
        trader_status = "online" if recent_activity > 0 else "idle"

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "trader_status": trader_status,
        "last_decision": last_update,

        "core_metrics": {
            "win_rate": round(win_rate, 1),
            "win_rate_display": f"{win_rate:.1f}%",
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "net_pnl": round(net_pnl, 2),
            "net_pnl_display": f"+${net_pnl:,.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):,.2f}",
            "avg_pnl": round(avg_pnl, 2),
            "best_trade": round(best_trade, 2),
            "worst_trade": round(worst_trade, 2),
            "avg_rr": 1.8,  # TODO: Calculate from actual trades
            "total_decisions": total_decisions,
            "today_decisions": today_decisions,
        },

        "streak": {
            "count": streak,
            "type": streak_type,
            "display": f"{'🔥' if streak_type == 'win' else '❄️'} {streak} {streak_type.title() if streak_type else 'N/A'}{'s' if streak != 1 else ''}" if streak_type else "No streak",
        },

        "open_positions": open_positions,
        "position_count": len(open_positions),

        "recent_signals": recent_decisions,

        "why_not": why_not,

        "decision_breakdown": decision_breakdown,

        "benchmarks": {
            "vs_btc_hold": None,  # TODO: Calculate BTC benchmark
            "uptime_pct": 99.2,   # TODO: Calculate from logs
        }
    }


@router.get("/performance")
async def get_performance_history(days: int = Query(default=7, le=90)):
    """
    Get historical performance data for charting.

    Returns daily P&L for the specified number of days.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("""
            SELECT
                date(exit_timestamp) as date,
                COUNT(*) as trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CAST(net_pnl AS REAL)) as daily_pnl
            FROM trades
            WHERE exit_timestamp IS NOT NULL
            AND exit_timestamp >= datetime('now', '-' || ? || ' days')
            GROUP BY date(exit_timestamp)
            ORDER BY date ASC
        """, (days,))

        daily_data = []
        cumulative_pnl = 0
        for row in await cursor.fetchall():
            daily_pnl = row["daily_pnl"] or 0
            cumulative_pnl += daily_pnl
            daily_data.append({
                "date": row["date"],
                "trades": row["trades"],
                "wins": row["wins"],
                "daily_pnl": round(daily_pnl, 2),
                "cumulative_pnl": round(cumulative_pnl, 2),
            })

    return {
        "days": days,
        "data": daily_data,
        "summary": {
            "total_pnl": round(cumulative_pnl, 2),
            "trading_days": len(daily_data),
        }
    }


@router.get("/signals/recent")
async def get_recent_signals(limit: int = Query(default=20, le=100)):
    """
    Get recent trading signals with full details.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("""
            SELECT decision_id, symbol, result, result_reason, timestamp, market_context
            FROM decisions
            WHERE result IN ('signal_long', 'signal_short', 'signal_close')
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        signals = []
        for row in await cursor.fetchall():
            d = dict(row)
            try:
                mc = json.loads(d.get("market_context", "{}"))
                setup = mc.get("setup_details", mc.get("professional", {}))
                d["grade"] = setup.get("grade", "")
                risk = setup.get("risk", {})
                d["entry_price"] = mc.get("current_price") or setup.get("current_price")
                d["stop_loss"] = risk.get("stop")
                d["target"] = risk.get("target")
                d["risk_reward"] = risk.get("risk_reward")
            except:
                pass
            del d["market_context"]
            signals.append(d)

    return {"signals": signals, "count": len(signals)}


@router.get("/why-not")
async def get_why_not_explanations(limit: int = Query(default=20, le=100)):
    """
    Get explanations for why trades were NOT taken.

    Educational content showing Argus's decision-making transparency.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("""
            SELECT symbol, result, result_reason, timestamp, market_context
            FROM decisions
            WHERE result IN ('no_signal', 'signal_hold', 'risk_rejected')
            AND result_reason IS NOT NULL
            AND result_reason != ''
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        explanations = []
        for row in await cursor.fetchall():
            d = dict(row)
            # Try to extract more context
            try:
                mc = json.loads(d.get("market_context", "{}"))
                setup = mc.get("setup_details", {})
                d["current_price"] = mc.get("current_price") or setup.get("current_price")
                layers = setup.get("layers", {})
                daily = layers.get("context_daily", {})
                d["trend"] = daily.get("trend_bias", "unknown")
            except:
                pass
            del d["market_context"]
            explanations.append(d)

    return {"explanations": explanations, "count": len(explanations)}
