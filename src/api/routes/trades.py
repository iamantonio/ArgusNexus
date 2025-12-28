"""
Trade Endpoints
Open and closed positions from Truth Engine.
"""

import aiosqlite
from fastapi import APIRouter, Query
from pathlib import Path
from typing import Optional

router = APIRouter()
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "v4_live_paper.db"


@router.get("/trades/open")
async def get_open_trades(strategy: Optional[str] = None):
    """Get all open positions."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        query = """
            SELECT
                trade_id, symbol, side, entry_timestamp, entry_price,
                quantity, stop_loss_price, take_profit_price, strategy_name
            FROM trades
            WHERE status = 'open'
        """
        params = []

        if strategy and strategy != "all":
            query += " AND strategy_name = ?"
            params.append(strategy)

        query += " ORDER BY entry_timestamp DESC"

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        return [{
            "trade_id": row["trade_id"],
            "symbol": row["symbol"],
            "side": row["side"],
            "entry_timestamp": row["entry_timestamp"],
            "entry_price": float(row["entry_price"]) if row["entry_price"] else 0,
            "quantity": float(row["quantity"]) if row["quantity"] else 0,
            "stop_loss": float(row["stop_loss_price"]) if row["stop_loss_price"] else None,
            "take_profit": float(row["take_profit_price"]) if row["take_profit_price"] else None,
            "strategy_name": row["strategy_name"]
        } for row in rows]


@router.get("/trades/closed")
async def get_closed_trades(
    strategy: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0)
):
    """Get closed trade history with pagination."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        query = """
            SELECT
                trade_id, symbol, side, entry_timestamp, exit_timestamp,
                entry_price, exit_price, quantity, realized_pnl, net_pnl,
                exit_reason, is_winner, strategy_name, duration_seconds
            FROM trades
            WHERE status = 'closed'
        """
        params = []

        if strategy and strategy != "all":
            query += " AND strategy_name = ?"
            params.append(strategy)

        query += " ORDER BY exit_timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        return [{
            "trade_id": row["trade_id"],
            "symbol": row["symbol"],
            "side": row["side"],
            "entry_timestamp": row["entry_timestamp"],
            "exit_timestamp": row["exit_timestamp"],
            "entry_price": float(row["entry_price"]) if row["entry_price"] else 0,
            "exit_price": float(row["exit_price"]) if row["exit_price"] else 0,
            "quantity": float(row["quantity"]) if row["quantity"] else 0,
            "realized_pnl": float(row["realized_pnl"]) if row["realized_pnl"] else 0,
            "net_pnl": float(row["net_pnl"]) if row["net_pnl"] else 0,
            "exit_reason": row["exit_reason"],
            "is_winner": bool(row["is_winner"]),
            "strategy_name": row["strategy_name"],
            "duration_seconds": row["duration_seconds"]
        } for row in rows]


@router.get("/trades/{trade_id}")
async def get_trade_detail(trade_id: str):
    """Get full trade details with audit trail."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("""
            SELECT * FROM trades WHERE trade_id = ?
        """, (trade_id,))
        trade = await cursor.fetchone()

        if not trade:
            return {"error": "Trade not found"}

        return dict(trade)
