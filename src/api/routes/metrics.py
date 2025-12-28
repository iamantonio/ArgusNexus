"""
Scoreboard / Metrics Endpoints
Aggregated P&L, win rate, and fleet statistics.
"""

import aiosqlite
import json
from fastapi import APIRouter
from pathlib import Path
from typing import Optional
import aiohttp
from datetime import datetime, timezone
from decimal import Decimal

router = APIRouter()
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "v4_live_paper.db"
RUNTIME_DIR = Path(__file__).parent.parent.parent.parent / "runtime"


async def fetch_current_price(symbol: str) -> float:
    """Fetch current price from Coinbase."""
    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get("price", 0))
    except Exception:
        pass
    return 0.0


@router.get("/scoreboard")
async def get_scoreboard(strategy: Optional[str] = None):
    """
    Get aggregated scoreboard metrics.

    Returns realized P&L, unrealized P&L, win rate, and active position count.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        # Build WHERE clause for strategy filter
        strategy_filter = ""
        params = []
        if strategy and strategy != "all":
            strategy_filter = " AND strategy_name = ?"
            params.append(strategy)

        # Get closed trade stats
        cursor = await db.execute(f"""
            SELECT
                COALESCE(SUM(CAST(realized_pnl AS REAL)), 0) as realized_pnl,
                COALESCE(SUM(CAST(net_pnl AS REAL)), 0) as net_pnl,
                COUNT(*) as total_closed,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses
            FROM trades
            WHERE status = 'closed'{strategy_filter}
        """, params)
        closed_stats = await cursor.fetchone()

        # Get open positions
        cursor = await db.execute(f"""
            SELECT trade_id, symbol, side, entry_price, quantity, stop_loss_price, take_profit_price, strategy_name
            FROM trades
            WHERE status = 'open'{strategy_filter}
        """, params)
        open_positions = await cursor.fetchall()

        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        positions_data = []

        for pos in open_positions:
            pos_dict = dict(pos)
            symbol = pos_dict["symbol"]
            entry_price = float(pos_dict["entry_price"])
            quantity = float(pos_dict["quantity"])
            side = pos_dict["side"]

            current_price = await fetch_current_price(symbol)

            if current_price > 0:
                if side == "buy":
                    pnl = (current_price - entry_price) * quantity
                else:
                    pnl = (entry_price - current_price) * quantity

                pnl_pct = ((current_price - entry_price) / entry_price * 100) if side == "buy" else ((entry_price - current_price) / entry_price * 100)

                # Get chandelier stop from state file (updated each tick)
                chandelier_stop = None
                try:
                    state_path = RUNTIME_DIR / "paper_trader_state.json"
                    if state_path.exists():
                        state_data = json.loads(state_path.read_text())
                        positions = state_data.get("positions", {})
                        if symbol in positions:
                            chandelier_stop = positions[symbol].get("chandelier_stop")
                except Exception:
                    pass  # State file may not exist or be malformed

                unrealized_pnl += pnl
                positions_data.append({
                    "trade_id": pos_dict["trade_id"],
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "quantity": quantity,
                    "stop_loss": float(pos_dict["stop_loss_price"]) if pos_dict["stop_loss_price"] else None,
                    "chandelier_stop": chandelier_stop,  # P0: Dynamic trailing stop
                    "take_profit": float(pos_dict["take_profit_price"]) if pos_dict["take_profit_price"] else None,
                    "unrealized_pnl": round(pnl, 2),
                    "unrealized_pnl_pct": round(pnl_pct, 2),
                    "strategy_name": pos_dict["strategy_name"]
                })

        realized = closed_stats["realized_pnl"] or 0
        total_closed = closed_stats["total_closed"] or 0
        wins = closed_stats["wins"] or 0
        losses = closed_stats["losses"] or 0
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0

        return {
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(realized + unrealized_pnl, 2),
            "win_rate": round(win_rate, 1),
            "wins": wins,
            "losses": losses,
            "total_closed": total_closed,
            "active_positions": len(positions_data),
            "positions": positions_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.get("/performance/daily")
async def get_daily_performance(days: int = 30, strategy: Optional[str] = None):
    """Get daily P&L breakdown."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        strategy_filter = ""
        params = [days]
        if strategy and strategy != "all":
            strategy_filter = " AND strategy_name = ?"
            params.append(strategy)

        cursor = await db.execute(f"""
            SELECT
                DATE(exit_timestamp) as date,
                COUNT(*) as trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses,
                ROUND(SUM(CAST(net_pnl AS REAL)), 2) as net_pnl
            FROM trades
            WHERE status = 'closed'
              AND exit_timestamp >= datetime('now', '-' || ? || ' days'){strategy_filter}
            GROUP BY DATE(exit_timestamp)
            ORDER BY date DESC
        """, params)

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


@router.get("/balance")
async def get_balance():
    """
    Get portfolio balance.

    Checks for paper trader state first, then unified state, falls back to per-symbol state files.
    """
    try:
        # Check for paper trader state first (learning-enabled paper trader)
        paper_state_path = RUNTIME_DIR / "paper_trader_state.json"
        if paper_state_path.exists():
            state = json.loads(paper_state_path.read_text())
            positions = []

            for symbol, pos_data in state.get("positions", {}).items():
                qty = float(pos_data.get("qty", 0))
                if qty > 0:
                    current_price = await fetch_current_price(symbol)
                    position_value = qty * current_price if current_price > 0 else 0
                    cost_basis = float(pos_data.get("cost_basis", 0))
                    pnl_pct = ((current_price - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0

                    positions.append({
                        "symbol": symbol,
                        "quantity": round(qty, 8),
                        "current_price": round(current_price, 2),
                        "value": round(position_value, 2),
                        "cost_basis": round(cost_basis, 2),
                        "unrealized_pnl": round(position_value - (qty * cost_basis), 2),
                        "unrealized_pnl_pct": round(pnl_pct, 2),
                        "regime": pos_data.get("regime", "unknown")
                    })

            total_equity = float(state.get("total_equity", 0))
            hwm = float(state.get("high_water_mark", state.get("starting_capital", 10000)))

            return {
                "total_cash": round(float(state.get("cash", 0)), 2),
                "total_equity": round(total_equity, 2),
                "high_water_mark": round(hwm, 2),
                "pnl": round(total_equity - hwm, 2),
                "pnl_pct": round((total_equity - hwm) / hwm * 100, 2) if hwm > 0 else 0,
                "dd_pct": round(float(state.get("current_dd_pct", 0)), 2),
                "dd_state": state.get("dd_state", "normal"),
                "position_count": len(positions),
                "positions": positions,
                "engine": "paper_learning",
                "symbol": state.get("symbol", "BTC-USD"),
                "interval": state.get("interval", "1h"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # Check for unified portfolio state second
        unified_state_path = RUNTIME_DIR / "unified_portfolio_state.json"
        if unified_state_path.exists():
            state = json.loads(unified_state_path.read_text())
            positions = []

            for symbol, pos_data in state.get("positions", {}).items():
                qty = float(pos_data.get("qty", 0))
                if qty > 0:
                    current_price = await fetch_current_price(symbol)
                    position_value = qty * current_price if current_price > 0 else 0
                    cost_basis = float(pos_data.get("cost_basis", 0))
                    pnl_pct = ((current_price - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0

                    positions.append({
                        "symbol": symbol,
                        "quantity": round(qty, 8),
                        "current_price": round(current_price, 2),
                        "value": round(position_value, 2),
                        "cost_basis": round(cost_basis, 2),
                        "unrealized_pnl": round(position_value - (qty * cost_basis), 2),
                        "unrealized_pnl_pct": round(pnl_pct, 2),
                        "regime": pos_data.get("regime", "unknown")
                    })

            total_equity = float(state.get("total_equity", 0))
            hwm = float(state.get("high_water_mark", 0))

            return {
                "total_cash": round(float(state.get("cash", 0)), 2),
                "total_equity": round(total_equity, 2),
                "high_water_mark": round(hwm, 2),
                "pnl": round(total_equity - hwm, 2),
                "pnl_pct": round((total_equity - hwm) / hwm * 100, 2) if hwm > 0 else 0,
                "dd_pct": round(float(state.get("current_dd_pct", 0)), 2),
                "dd_state": state.get("dd_state", "normal"),
                "position_count": len(positions),
                "positions": positions,
                "engine": "unified",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # Fallback: Find all per-symbol state files
        state_files = list(RUNTIME_DIR.glob("portfolio_state_*.json"))

        if not state_files:
            return {
                "error": "No portfolio state files found",
                "cash": 0,
                "positions": [],
                "total_equity": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        total_cash = 0.0
        total_equity = 0.0
        total_hwm = 0.0
        positions = []

        for state_file in state_files:
            try:
                state = json.loads(state_file.read_text())

                # Extract symbol from filename (e.g., portfolio_state_btc_usd.json -> BTC-USD)
                symbol_safe = state_file.stem.replace("portfolio_state_", "")
                symbol = symbol_safe.upper().replace("_", "-")

                qty = float(state.get("btc_qty", 0))
                cash = float(state.get("cash", 0))
                hwm = float(state.get("high_water_mark", 500))

                current_price = await fetch_current_price(symbol)
                position_value = qty * current_price if current_price > 0 else 0

                total_cash += cash
                total_equity += cash + position_value
                total_hwm += hwm

                if qty > 0:
                    positions.append({
                        "symbol": symbol,
                        "quantity": round(qty, 8),
                        "current_price": round(current_price, 2),
                        "value": round(position_value, 2),
                        "cash": round(cash, 2),
                        "equity": round(cash + position_value, 2),
                        "dd_state": state.get("dd_state", "unknown"),
                        "cost_basis": state.get("btc_cost_basis")
                    })
            except Exception as e:
                continue

        return {
            "total_cash": round(total_cash, 2),
            "total_equity": round(total_equity, 2),
            "high_water_mark": round(total_hwm, 2),
            "pnl": round(total_equity - total_hwm, 2),
            "pnl_pct": round((total_equity - total_hwm) / total_hwm * 100, 2) if total_hwm > 0 else 0,
            "position_count": len(positions),
            "positions": positions,
            "engine": "multi-process",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
