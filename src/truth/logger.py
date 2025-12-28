"""
Truth Logger - The Glass Box Core

Records all trading decisions and outcomes to the Truth Engine database.
This is the ONLY way data enters the Truth Engine tables.

PRINCIPLE: If it's not logged here, it didn't happen.

Usage:
    logger = TruthLogger("path/to/v4.db")
    logger.initialize()  # Creates tables if needed

    # Log a decision (every strategy evaluation)
    decision = logger.log_decision(
        symbol="BTC-USD",
        strategy_name="dual_ema_crossover",
        signal_values={"fast_ema": 50123.45, "slow_ema": 50100.00, ...},
        risk_checks={"daily_loss_limit": {"passed": True, ...}, ...},
        result=DecisionResult.SIGNAL_LONG,
        result_reason="Bullish EMA crossover with favorable risk/reward"
    )

    # Log an order (when decision leads to execution)
    order = logger.log_order(
        decision_id=decision.decision_id,
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=Decimal("0.001"),
        requested_price=Decimal("50150.00")
    )

    # Update order on fill
    logger.update_order_fill(
        order_id=order.order_id,
        fill_price=Decimal("50155.00"),
        fill_quantity=Decimal("0.001"),
        commission=Decimal("0.50")
    )

    # Log completed trade
    trade = logger.log_trade(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        entry_order_id=entry_order.order_id,
        exit_order_id=exit_order.order_id,
        ...
    )
"""

import aiosqlite
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import asynccontextmanager

from .schema import (
    Decision, Order, Trade,
    DecisionResult, OrderSide, OrderStatus, ExitReason,
    SQL_SCHEMA, decimal_to_str, str_to_decimal
)

# Lazy import for Reflexion Engine to avoid circular imports
_reflexion_engine = None


logger = logging.getLogger(__name__)


class TruthLogger:
    """
    The Glass Box Core - All truth flows through here.

    Asynchronous aiosqlite logger for the V4 Truth Engine.
    Every decision, order, and trade is recorded with full context.
    """

    def __init__(self, db_path: str):
        """
        Initialize the Truth Logger.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def _get_connection(self):
        """Async context manager for database connections"""
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            try:
                yield db
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error(f"Database error: {e}")
                raise

    async def initialize(self) -> None:
        """
        Initialize the database schema and enable WAL mode.

        Creates all tables, indexes, and views if they don't exist.
        Safe to call multiple times.
        """
        async with self._get_connection() as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.executescript(SQL_SCHEMA)
            logger.info(f"Truth Engine initialized (WAL mode) at {self.db_path}")

    async def reset(self) -> None:
        """
        Reset the Truth Engine database for a fresh backtest.

        DANGER: This deletes ALL data in the database.
        Use only for backtesting - never in production.

        Clears all tables while preserving the schema.
        """
        async with self._get_connection() as db:
            # Delete in order to respect foreign keys (if enabled)
            await db.execute("DELETE FROM trades")
            await db.execute("DELETE FROM orders")
            await db.execute("DELETE FROM decisions")
            logger.warning(f"Truth Engine RESET - all data cleared at {self.db_path}")

    # =========================================================================
    # DECISION LOGGING
    # =========================================================================

    async def log_decision(
        self,
        symbol: str,
        strategy_name: str,
        signal_values: Dict[str, Any],
        risk_checks: Dict[str, Any],
        result: DecisionResult,
        result_reason: str,
        market_context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        generate_insight: bool = False
    ) -> Decision:
        """
        Log a trading decision (Async).

        This should be called EVERY time the strategy evaluates.

        Args:
            generate_insight: If True, generate LLM insight using GPT-5.2.
                             Requires OPENAI_API_KEY environment variable.
        """
        decision = Decision(
            decision_id=Decision.generate_id(),
            timestamp=timestamp or datetime.utcnow(),
            symbol=symbol,
            strategy_name=strategy_name,
            signal_values=signal_values,
            risk_checks=risk_checks,
            market_context=market_context or {},
            result=result,
            result_reason=result_reason
        )

        # Generate LLM insight if requested
        if generate_insight:
            try:
                from .llm_insights import LLMInsightGenerator
                generator = LLMInsightGenerator()
                if generator.is_available:
                    insight = await generator.analyze_decision(decision.to_dict())
                    decision.llm_insight = insight.to_dict()
                    logger.debug(f"LLM insight generated for {decision.decision_id}")
            except Exception as e:
                logger.warning(f"Failed to generate LLM insight: {e}")

        async with self._get_connection() as db:
            data = decision.to_dict()
            await db.execute("""
                INSERT INTO decisions (
                    decision_id, timestamp, symbol, strategy_name,
                    signal_values, risk_checks, market_context,
                    result, result_reason, order_id, llm_insight
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["decision_id"],
                data["timestamp"],
                data["symbol"],
                data["strategy_name"],
                data["signal_values"],
                data["risk_checks"],
                data["market_context"],
                data["result"],
                data["result_reason"],
                data["order_id"],
                data["llm_insight"]
            ))

        logger.debug(f"Decision logged: {decision.decision_id} -> {result.value}")
        return decision

    async def link_decision_to_order(self, decision_id: str, order_id: str) -> None:
        """Link a decision to its resulting order"""
        async with self._get_connection() as db:
            await db.execute(
                "UPDATE decisions SET order_id = ? WHERE decision_id = ?",
                (order_id, decision_id)
            )

    # =========================================================================
    # ORDER LOGGING
    # =========================================================================

    async def log_order(
        self,
        decision_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        requested_price: Optional[Decimal] = None,
        timestamp: Optional[datetime] = None
    ) -> Order:
        """
        Log an order placement (Async).
        """
        order = Order(
            order_id=Order.generate_id(),
            decision_id=decision_id,
            timestamp=timestamp or datetime.utcnow(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            requested_price=requested_price,
            status=OrderStatus.PENDING
        )

        async with self._get_connection() as db:
            data = order.to_dict()
            await db.execute("""
                INSERT INTO orders (
                    order_id, decision_id, timestamp, symbol, side, quantity,
                    requested_price, fill_price, fill_quantity, fill_timestamp,
                    status, exchange_order_id, slippage_amount, slippage_percent,
                    commission, commission_asset, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["order_id"],
                data["decision_id"],
                data["timestamp"],
                data["symbol"],
                data["side"],
                data["quantity"],
                data["requested_price"],
                data["fill_price"],
                data["fill_quantity"],
                data["fill_timestamp"],
                data["status"],
                data["exchange_order_id"],
                data["slippage_amount"],
                data["slippage_percent"],
                data["commission"],
                data["commission_asset"],
                data["error_message"]
            ))

        # Link decision to order
        await self.link_decision_to_order(decision_id, order.order_id)

        logger.debug(f"Order logged: {order.order_id} for decision {decision_id}")
        return order

    async def update_order_fill(
        self,
        order_id: str,
        fill_price: Decimal,
        fill_quantity: Decimal,
        fill_timestamp: Optional[datetime] = None,
        exchange_order_id: Optional[str] = None,
        commission: Optional[Decimal] = None,
        commission_asset: Optional[str] = None
    ) -> None:
        """
        Update an order with fill information (Async).
        """
        fill_ts = fill_timestamp or datetime.utcnow()

        async with self._get_connection() as db:
            # Get requested price for slippage calculation
            cursor = await db.execute(
                "SELECT requested_price FROM orders WHERE order_id = ?",
                (order_id,)
            )
            row = await cursor.fetchone()

            slippage_amount = None
            slippage_percent = None

            if row and row["requested_price"]:
                requested = Decimal(row["requested_price"])
                slippage_amount = fill_price - requested
                if requested != 0:
                    slippage_percent = (slippage_amount / requested) * 100

            await db.execute("""
                UPDATE orders SET
                    fill_price = ?,
                    fill_quantity = ?,
                    fill_timestamp = ?,
                    status = ?,
                    exchange_order_id = ?,
                    slippage_amount = ?,
                    slippage_percent = ?,
                    commission = ?,
                    commission_asset = ?
                WHERE order_id = ?
            """, (
                str(fill_price),
                str(fill_quantity),
                fill_ts.isoformat(),
                OrderStatus.FILLED.value,
                exchange_order_id,
                decimal_to_str(slippage_amount),
                decimal_to_str(slippage_percent),
                decimal_to_str(commission),
                commission_asset,
                order_id
            ))

        logger.debug(f"Order filled: {order_id} at {fill_price}")

    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update order status (Async)"""
        async with self._get_connection() as db:
            await db.execute("""
                UPDATE orders SET status = ?, error_message = ?
                WHERE order_id = ?
            """, (status.value, error_message, order_id))

    # =========================================================================
    # TRADE LOGGING
    # =========================================================================

    async def log_trade(
        self,
        symbol: str,
        side: OrderSide,
        entry_order_id: str,
        exit_order_id: str,
        entry_decision_id: str,
        exit_decision_id: str,
        entry_timestamp: datetime,
        exit_timestamp: datetime,
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
        total_commission: Decimal,
        total_slippage: Decimal,
        exit_reason: ExitReason
    ) -> Trade:
        """
        Log a completed trade (Async).
        """
        # Calculate P&L
        if side == OrderSide.BUY:  # Long trade
            realized_pnl = (exit_price - entry_price) * quantity
        else:  # Short trade
            realized_pnl = (entry_price - exit_price) * quantity

        entry_value = entry_price * quantity
        realized_pnl_percent = (realized_pnl / entry_value) * 100 if entry_value != 0 else Decimal("0")

        net_pnl = realized_pnl - total_commission - abs(total_slippage)
        is_winner = net_pnl > 0

        duration = int((exit_timestamp - entry_timestamp).total_seconds())

        trade = Trade(
            trade_id=Trade.generate_id(),
            symbol=symbol,
            side=side,
            entry_order_id=entry_order_id,
            exit_order_id=exit_order_id,
            entry_decision_id=entry_decision_id,
            exit_decision_id=exit_decision_id,
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
            duration_seconds=duration,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            realized_pnl=realized_pnl,
            realized_pnl_percent=realized_pnl_percent,
            total_commission=total_commission,
            total_slippage=total_slippage,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            is_winner=is_winner
        )

        async with self._get_connection() as db:
            data = trade.to_dict()
            await db.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, entry_order_id, exit_order_id,
                    entry_decision_id, exit_decision_id, entry_timestamp,
                    exit_timestamp, duration_seconds, entry_price, exit_price,
                    quantity, realized_pnl, realized_pnl_percent,
                    total_commission, total_slippage, net_pnl,
                    exit_reason, is_winner
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["trade_id"],
                data["symbol"],
                data["side"],
                data["entry_order_id"],
                data["exit_order_id"],
                data["entry_decision_id"],
                data["exit_decision_id"],
                data["entry_timestamp"],
                data["exit_timestamp"],
                data["duration_seconds"],
                data["entry_price"],
                data["exit_price"],
                data["quantity"],
                data["realized_pnl"],
                data["realized_pnl_percent"],
                data["total_commission"],
                data["total_slippage"],
                data["net_pnl"],
                data["exit_reason"],
                1 if data["is_winner"] else 0
            ))

        status = "WIN" if is_winner else "LOSS"
        logger.info(f"Trade logged: {trade.trade_id} | {symbol} | {status} | P&L: {net_pnl}")
        return trade

    async def open_trade(
        self,
        symbol: str,
        side: OrderSide,
        entry_order_id: str,
        entry_price: Decimal,
        quantity: Decimal,
        stop_loss_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
        strategy_name: Optional[str] = None,
        entry_timestamp: Optional[datetime] = None
    ) -> str:
        """
        Log an OPEN trade (Async).
        """
        import uuid
        trade_id = str(uuid.uuid4())
        ts = entry_timestamp or datetime.utcnow()

        async with self._get_connection() as db:
            cursor = await db.execute(
                "SELECT decision_id FROM orders WHERE order_id = ?",
                (entry_order_id,)
            )
            row = await cursor.fetchone()
            entry_decision_id = row["decision_id"] if row else None

            await db.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, status,
                    entry_order_id, entry_decision_id,
                    entry_timestamp, entry_price, quantity,
                    stop_loss_price, take_profit_price, strategy_name
                ) VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                symbol,
                side.value,
                entry_order_id,
                entry_decision_id,
                ts.isoformat(),
                str(entry_price),
                str(quantity),
                str(stop_loss_price) if stop_loss_price else None,
                str(take_profit_price) if take_profit_price else None,
                strategy_name
            ))

        logger.info(f"Trade OPENED: {trade_id} | {symbol} | {side.value} @ {entry_price}")
        return trade_id

    async def close_trade(
        self,
        trade_id: str,
        exit_order_id: str,
        exit_price: Decimal,
        exit_reason: ExitReason,
        commission: Optional[Decimal] = None,
        slippage: Optional[Decimal] = None,
        exit_timestamp: Optional[datetime] = None
    ) -> None:
        """
        Close an open trade (Async).

        PR-3: If commission or slippage is None, trade is marked invalid.
        "If we can't compute PnL reliably, we mark it as invalid."
        """
        ts = exit_timestamp or datetime.utcnow()

        # PR-3: Data quality validation
        is_valid = True
        invalid_reasons = []

        if commission is None:
            is_valid = False
            invalid_reasons.append("missing_commission")
            commission = Decimal("0")  # Use 0 for calculation but mark invalid

        if slippage is None:
            is_valid = False
            invalid_reasons.append("missing_slippage")
            slippage = Decimal("0")  # Use 0 for calculation but mark invalid

        invalid_reason = "|".join(invalid_reasons) if invalid_reasons else None

        async with self._get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,)
            )
            row = await cursor.fetchone()

            if not row:
                logger.error(f"Trade not found: {trade_id}")
                return

            entry_price = Decimal(row["entry_price"])
            quantity = Decimal(row["quantity"])
            side = row["side"]
            entry_ts = datetime.fromisoformat(row["entry_timestamp"])

            if side == "buy":  # Long
                realized_pnl = (exit_price - entry_price) * quantity
            else:  # Short
                realized_pnl = (entry_price - exit_price) * quantity

            entry_value = entry_price * quantity
            realized_pnl_percent = (realized_pnl / entry_value * 100) if entry_value != 0 else Decimal("0")
            net_pnl = realized_pnl - commission - abs(slippage)
            is_winner = 1 if net_pnl > 0 else 0
            duration = int((ts - entry_ts).total_seconds())

            cursor = await db.execute(
                "SELECT decision_id FROM orders WHERE order_id = ?",
                (exit_order_id,)
            )
            exit_row = await cursor.fetchone()
            exit_decision_id = exit_row["decision_id"] if exit_row else None

            await db.execute("""
                UPDATE trades SET
                    status = 'closed',
                    exit_order_id = ?,
                    exit_decision_id = ?,
                    exit_timestamp = ?,
                    exit_price = ?,
                    duration_seconds = ?,
                    realized_pnl = ?,
                    realized_pnl_percent = ?,
                    total_commission = ?,
                    total_slippage = ?,
                    net_pnl = ?,
                    exit_reason = ?,
                    is_winner = ?,
                    is_valid = ?,
                    invalid_reason = ?
                WHERE trade_id = ?
            """, (
                exit_order_id,
                exit_decision_id,
                ts.isoformat(),
                str(exit_price),
                duration,
                str(realized_pnl),
                str(realized_pnl_percent),
                str(commission),
                str(slippage),
                str(net_pnl),
                exit_reason.value,
                is_winner,
                1 if is_valid else 0,
                invalid_reason,
                trade_id
            ))

        status = "WIN" if is_winner else "LOSS"
        validity = "" if is_valid else " [INVALID]"
        logger.info(f"Trade CLOSED: {trade_id} | {status} | P&L: {net_pnl}{validity}")

        # Trigger reflection generation (async, non-blocking)
        try:
            await self._trigger_reflection(trade_id)
        except Exception as e:
            logger.warning(f"Failed to trigger reflection for {trade_id}: {e}")

    # =========================================================================
    # QUERY METHODS - Async
    # =========================================================================

    async def get_trade_audit(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get full audit trail (Async)"""
        async with self._get_connection() as db:
            cursor = await db.execute("""
                SELECT * FROM v_trade_audit WHERE trade_id = ?
            """, (trade_id,))
            row = await cursor.fetchone()

            if row:
                result = dict(row)
                result["entry_signal"] = json.loads(result["entry_signal"])
                result["entry_risk_checks"] = json.loads(result["entry_risk_checks"])
                result["exit_signal"] = json.loads(result["exit_signal"])
                return result
            return None

    async def get_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get a single decision by ID (Async)"""
        async with self._get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE decision_id = ?",
                (decision_id,)
            )
            row = await cursor.fetchone()

            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get("signal_values"):
                    result["signal_values"] = json.loads(result["signal_values"])
                if result.get("risk_checks"):
                    result["risk_checks"] = json.loads(result["risk_checks"])
                return result
            return None

    async def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get a single trade by ID (Async)"""
        async with self._get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,)
            )
            row = await cursor.fetchone()

            if row:
                return dict(row)
            return None

    async def get_daily_pnl(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily P&L summary (Async)"""
        async with self._get_connection() as db:
            cursor = await db.execute("""
                SELECT * FROM v_daily_pnl LIMIT ?
            """, (days,))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_risk_rejections(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get risk rejection analysis (Async)"""
        async with self._get_connection() as db:
            cursor = await db.execute("""
                SELECT * FROM v_risk_rejections
                WHERE rejection_date >= date('now', ?)
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_recent_decisions(
        self,
        symbol: Optional[str] = None,
        result: Optional[DecisionResult] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent decisions (Async)"""
        query = "SELECT * FROM decisions WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if result:
            query += " AND result = ?"
            params.append(result.value)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with self._get_connection() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["signal_values"] = json.loads(d["signal_values"])
                d["risk_checks"] = json.loads(d["risk_checks"])
                if d["market_context"]:
                    d["market_context"] = json.loads(d["market_context"])
                if d.get("llm_insight"):
                    d["llm_insight"] = json.loads(d["llm_insight"])
                results.append(d)
            return results

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary (Async)"""
        async with self._get_connection() as db:
            cursor = await db.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses,
                    ROUND(100.0 * SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate,
                    SUM(CAST(realized_pnl AS REAL)) as gross_pnl,
                    SUM(CAST(net_pnl AS REAL)) as net_pnl,
                    AVG(CAST(net_pnl AS REAL)) as avg_pnl_per_trade,
                    MAX(CAST(net_pnl AS REAL)) as best_trade,
                    MIN(CAST(net_pnl AS REAL)) as worst_trade,
                    AVG(duration_seconds) as avg_duration_seconds
                FROM trades
            """)
            row = await cursor.fetchone()

            return dict(row) if row else {}

    # =========================================================================
    # POSITION STATE HYDRATION - Async
    # =========================================================================

    async def get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get open position from the Truth Engine (Async).
        """
        async with self._get_connection() as db:
            cursor = await db.execute("""
                SELECT
                    trade_id,
                    symbol,
                    side,
                    entry_order_id,
                    entry_decision_id,
                    entry_timestamp,
                    entry_price,
                    quantity,
                    stop_loss_price,
                    take_profit_price,
                    strategy_name
                FROM trades
                WHERE symbol = ? AND status = 'open'
                ORDER BY entry_timestamp DESC
                LIMIT 1
            """, (symbol,))
            row = await cursor.fetchone()

            if row:
                result = dict(row)
                for field in ['entry_price', 'quantity', 'stop_loss_price', 'take_profit_price']:
                    if result.get(field):
                        result[field] = Decimal(str(result[field]))
                return result
            return None

    async def get_all_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions (Async)"""
        async with self._get_connection() as db:
            cursor = await db.execute("""
                SELECT
                    trade_id,
                    symbol,
                    side,
                    entry_order_id,
                    entry_decision_id,
                    entry_timestamp,
                    entry_price,
                    quantity,
                    stop_loss_price,
                    take_profit_price,
                    strategy_name
                FROM trades
                WHERE status = 'open'
                ORDER BY entry_timestamp DESC
            """)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                result = dict(row)
                for field in ['entry_price', 'quantity', 'stop_loss_price', 'take_profit_price']:
                    if result.get(field):
                        result[field] = Decimal(str(result[field]))
                results.append(result)
            return results

    async def log_reconciliation_event(
        self,
        symbol: str,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "INFO"
    ) -> None:
        """Log a position reconciliation event (Async)"""
        await self.log_decision(
            symbol=symbol,
            strategy_name="position_reconciliation",
            signal_values={
                "event_type": event_type,
                "severity": severity,
                "details": details
            },
            risk_checks={},
            result=DecisionResult.NO_SIGNAL,
            result_reason=f"[{severity}] Position Reconciliation: {event_type}"
        )

    # =========================================================================
    # PHANTOM TRADE TRACKING - Validation Layer
    # =========================================================================

    async def log_phantom_trade(
        self,
        decision_id: str,
        symbol: str,
        timestamp: datetime,
        setup_grade: str,
        signal_type: str,
        hypothetical_entry: float,
        chandelier_stop_price: Optional[float] = None,
        hard_stop_price: Optional[float] = None
    ) -> str:
        """
        Log a phantom (hypothetical) trade for validation.

        Called when we decide NOT to trade a B/C grade setup.
        Later, we backfill the outcome to see if we should have traded it.

        Args:
            decision_id: The HOLD decision this phantom relates to
            symbol: Trading pair
            timestamp: When we would have entered
            setup_grade: A+, A, B, C
            signal_type: long or short
            hypothetical_entry: Price we would have entered at
            chandelier_stop_price: Where Chandelier exit would be
            hard_stop_price: Where hard stop would be

        Returns:
            phantom_id: UUID of the created phantom trade
        """
        import uuid
        phantom_id = str(uuid.uuid4())

        async with self._get_connection() as db:
            await db.execute("""
                INSERT INTO phantom_trades (
                    phantom_id, decision_id, symbol, timestamp,
                    setup_grade, signal_type, hypothetical_entry,
                    chandelier_stop_price, hard_stop_price,
                    verdict, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """, (
                phantom_id, decision_id, symbol, timestamp.isoformat(),
                setup_grade, signal_type, hypothetical_entry,
                chandelier_stop_price, hard_stop_price,
                datetime.utcnow().isoformat()
            ))
            await db.commit()

        logger.info(f"Phantom trade logged: {phantom_id} | {symbol} | {setup_grade} grade | entry ${hypothetical_entry:.2f}")
        return phantom_id

    async def update_phantom_outcome(
        self,
        phantom_id: str,
        price_after_1h: Optional[float] = None,
        price_after_4h: Optional[float] = None,
        price_after_24h: Optional[float] = None,
        price_after_48h: Optional[float] = None,
        would_hit_chandelier: Optional[bool] = None,
        would_hit_hard_stop: Optional[bool] = None,
        chandelier_hit_time: Optional[datetime] = None,
        hard_stop_hit_time: Optional[datetime] = None,
        max_favorable_excursion: Optional[float] = None,
        max_adverse_excursion: Optional[float] = None,
        phantom_exit_price: Optional[float] = None,
        phantom_pnl_percent: Optional[float] = None,
        duration_to_exit_hours: Optional[float] = None,
        verdict: Optional[str] = None,
        verdict_reason: Optional[str] = None
    ) -> None:
        """
        Update a phantom trade with outcome data.

        Called by background job after collecting price data.
        """
        async with self._get_connection() as db:
            await db.execute("""
                UPDATE phantom_trades SET
                    price_after_1h = COALESCE(?, price_after_1h),
                    price_after_4h = COALESCE(?, price_after_4h),
                    price_after_24h = COALESCE(?, price_after_24h),
                    price_after_48h = COALESCE(?, price_after_48h),
                    would_hit_chandelier = COALESCE(?, would_hit_chandelier),
                    would_hit_hard_stop = COALESCE(?, would_hit_hard_stop),
                    chandelier_hit_time = COALESCE(?, chandelier_hit_time),
                    hard_stop_hit_time = COALESCE(?, hard_stop_hit_time),
                    max_favorable_excursion = COALESCE(?, max_favorable_excursion),
                    max_adverse_excursion = COALESCE(?, max_adverse_excursion),
                    phantom_exit_price = COALESCE(?, phantom_exit_price),
                    phantom_pnl_percent = COALESCE(?, phantom_pnl_percent),
                    duration_to_exit_hours = COALESCE(?, duration_to_exit_hours),
                    verdict = COALESCE(?, verdict),
                    verdict_reason = COALESCE(?, verdict_reason),
                    updated_at = ?
                WHERE phantom_id = ?
            """, (
                price_after_1h, price_after_4h, price_after_24h, price_after_48h,
                1 if would_hit_chandelier else 0 if would_hit_chandelier is not None else None,
                1 if would_hit_hard_stop else 0 if would_hit_hard_stop is not None else None,
                chandelier_hit_time.isoformat() if chandelier_hit_time else None,
                hard_stop_hit_time.isoformat() if hard_stop_hit_time else None,
                max_favorable_excursion, max_adverse_excursion,
                phantom_exit_price, phantom_pnl_percent, duration_to_exit_hours,
                verdict, verdict_reason,
                datetime.utcnow().isoformat(),
                phantom_id
            ))
            await db.commit()

        if verdict:
            logger.info(f"Phantom {phantom_id} verdict: {verdict} ({verdict_reason})")

    async def get_pending_phantoms(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get phantom trades that need outcome backfill."""
        async with self._get_connection() as db:
            cursor = await db.execute("""
                SELECT * FROM phantom_trades
                WHERE verdict = 'pending'
                ORDER BY timestamp ASC
                LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_phantom_summary_by_grade(self) -> List[Dict[str, Any]]:
        """Get phantom trade summary grouped by setup grade."""
        async with self._get_connection() as db:
            cursor = await db.execute("""
                SELECT
                    setup_grade,
                    COUNT(*) as total_phantoms,
                    SUM(CASE WHEN verdict = 'regret' THEN 1 ELSE 0 END) as regrets,
                    SUM(CASE WHEN verdict = 'relief' THEN 1 ELSE 0 END) as reliefs,
                    SUM(CASE WHEN verdict = 'pending' THEN 1 ELSE 0 END) as pending,
                    ROUND(100.0 * SUM(CASE WHEN verdict = 'regret' THEN 1 ELSE 0 END) /
                          NULLIF(SUM(CASE WHEN verdict != 'pending' THEN 1 ELSE 0 END), 0), 1) as regret_rate,
                    ROUND(AVG(CASE WHEN verdict != 'pending' THEN phantom_pnl_percent END), 2) as avg_phantom_pnl,
                    ROUND(AVG(CASE WHEN verdict != 'pending' THEN duration_to_exit_hours END), 1) as avg_duration_hours
                FROM phantom_trades
                GROUP BY setup_grade
                ORDER BY setup_grade
            """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # REFLEXION LAYER INTEGRATION
    # =========================================================================

    def _get_reflexion_engine(self):
        """Lazy-load the Reflexion Engine"""
        global _reflexion_engine
        if _reflexion_engine is None:
            try:
                from ..learning.reflexion import ReflexionEngine
                _reflexion_engine = ReflexionEngine(str(self.db_path))
                logger.info("ReflexionEngine initialized for learning")
            except ImportError as e:
                logger.warning(f"Reflexion module not available: {e}")
                return None
        return _reflexion_engine

    async def _trigger_reflection(self, trade_id: str) -> None:
        """
        Trigger reflection generation for a closed trade.

        This is the learning loop entry point - called automatically
        when a trade closes.
        """
        engine = self._get_reflexion_engine()
        if engine is None:
            return

        try:
            # Initialize reflexion tables if needed
            await engine.initialize()

            # Generate reflection asynchronously
            reflection = await engine.reflect_on_trade(trade_id)
            if reflection:
                logger.info(
                    f"LEARNED: [{reflection.reflection_type.value}] "
                    f"{reflection.lesson_learned} (confidence: {reflection.confidence:.0%})"
                )
        except Exception as e:
            logger.error(f"Reflection generation failed for {trade_id}: {e}")

    async def get_relevant_lessons(
        self,
        symbol: str,
        signals: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query relevant lessons for a new trading decision.

        This should be called BEFORE making a trade to incorporate
        lessons from similar past situations.

        Args:
            symbol: Symbol being traded
            signals: Current signal values
            market_context: Current market state

        Returns:
            Dict with:
            - adjusted_confidence: Confidence adjustment factor
            - size_adjustment: Position size multiplier
            - warnings: List of warning messages
            - supporting_lessons: Lessons that support the trade
            - cautionary_lessons: Lessons that caution against it
        """
        engine = self._get_reflexion_engine()
        if engine is None:
            return {
                "adjusted_confidence": 1.0,
                "size_adjustment": 1.0,
                "warnings": [],
                "supporting_lessons": [],
                "cautionary_lessons": [],
                "lesson_count": 0
            }

        try:
            await engine.initialize()
            return await engine.apply_lessons_to_decision(
                symbol=symbol,
                current_signals=signals,
                current_market=market_context,
                proposed_action="long"  # Will be enhanced later
            )
        except Exception as e:
            logger.error(f"Failed to query lessons: {e}")
            return {
                "adjusted_confidence": 1.0,
                "size_adjustment": 1.0,
                "warnings": [f"Lesson query failed: {e}"],
                "supporting_lessons": [],
                "cautionary_lessons": [],
                "lesson_count": 0
            }

    async def get_learning_summary(
        self,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get a summary of lessons learned.

        Useful for periodic review of what Argus has learned.
        """
        engine = self._get_reflexion_engine()
        if engine is None:
            return {"error": "Reflexion Engine not available"}

        try:
            await engine.initialize()
            return await engine.get_lessons_summary(
                symbol=symbol,
                days=days,
                min_confidence=0.5
            )
        except Exception as e:
            logger.error(f"Failed to get learning summary: {e}")
            return {"error": str(e)}

    async def backfill_reflections(self, days: int = 30) -> int:
        """
        Generate reflections for past trades that don't have them.

        Useful for bootstrapping Argus's memory from historical trades.

        Returns:
            Number of reflections generated
        """
        engine = self._get_reflexion_engine()
        if engine is None:
            logger.warning("Cannot backfill - Reflexion Engine not available")
            return 0

        try:
            await engine.initialize()
            return await engine.backfill_reflections(days=days)
        except Exception as e:
            logger.error(f"Backfill failed: {e}")
            return 0

    # =========================================================================
    # LLM INSIGHT METHODS
    # =========================================================================

    async def generate_insight_for_decision(
        self,
        decision_id: str,
        force_regenerate: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Generate LLM insight for an existing decision (on-demand).

        Args:
            decision_id: The decision ID to generate insight for
            force_regenerate: If True, regenerate even if insight exists

        Returns:
            The generated insight dict, or None if generation failed
        """
        async with self._get_connection() as db:
            # Fetch the decision
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE decision_id = ?",
                (decision_id,)
            )
            row = await cursor.fetchone()

            if not row:
                logger.warning(f"Decision not found: {decision_id}")
                return None

            decision = dict(row)

            # Check if insight already exists
            if decision.get("llm_insight") and not force_regenerate:
                return json.loads(decision["llm_insight"])

            # Parse JSON fields
            decision["signal_values"] = json.loads(decision["signal_values"])
            decision["risk_checks"] = json.loads(decision["risk_checks"])
            if decision["market_context"]:
                decision["market_context"] = json.loads(decision["market_context"])

        # Generate insight
        try:
            from .llm_insights import LLMInsightGenerator
            generator = LLMInsightGenerator()
            if not generator.is_available:
                logger.warning("LLM insights not available - check OPENAI_API_KEY")
                return None

            insight = await generator.analyze_decision(decision)
            insight_dict = insight.to_dict()

            # Save to database
            async with self._get_connection() as db:
                await db.execute(
                    "UPDATE decisions SET llm_insight = ? WHERE decision_id = ?",
                    (json.dumps(insight_dict), decision_id)
                )

            logger.info(f"LLM insight generated for decision {decision_id}")
            return insight_dict

        except Exception as e:
            logger.error(f"Failed to generate insight for {decision_id}: {e}")
            return None

    async def get_decision_with_insight(
        self,
        decision_id: str,
        generate_if_missing: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get a decision with its LLM insight, optionally generating if missing.

        Args:
            decision_id: The decision ID to fetch
            generate_if_missing: If True and no insight exists, generate one

        Returns:
            Decision dict with parsed JSON fields and insight
        """
        async with self._get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE decision_id = ?",
                (decision_id,)
            )
            row = await cursor.fetchone()

            if not row:
                return None

            decision = dict(row)
            decision["signal_values"] = json.loads(decision["signal_values"])
            decision["risk_checks"] = json.loads(decision["risk_checks"])
            if decision["market_context"]:
                decision["market_context"] = json.loads(decision["market_context"])
            if decision.get("llm_insight"):
                decision["llm_insight"] = json.loads(decision["llm_insight"])

        # Generate insight if missing and requested
        if not decision.get("llm_insight") and generate_if_missing:
            insight = await self.generate_insight_for_decision(decision_id)
            if insight:
                decision["llm_insight"] = insight

        return decision
