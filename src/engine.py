"""
V4 Trading Engine - The Orchestrator

This is the heart. It doesn't contain logic; it contains FLOW.
It is the pipe that connects the organs:
- Strategy (the Brain)
- Risk Manager (the Conscience)
- Executor (the Hands)
- Truth Logger (the Memory)

The run_tick() method executes THE sequence:
1. Ask the Brain → SignalResult
2. Check the Conscience → RiskResult
3. Log the Decision (even if rejected!)
4. Move the Hands → ExecutionResult
5. Log the Action
6. Alert the Commander → Discord notification

If the test passes, the heart beats.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

# Import both strategies - engine is strategy-agnostic
from strategy.dual_ema import DualEMACrossover, SignalResult as DualEMASignalResult
from strategy.dual_ema import Signal as DualEMASignal
from strategy.donchian import DonchianBreakout, SignalResult as DonchianSignalResult
from strategy.donchian import Signal as DonchianSignal

# Use common references (both strategies have compatible types)
Signal = DualEMASignal  # Both strategies use identical Signal enums
SignalResult = DualEMASignalResult  # Both have same structure
from risk import RiskManager, RiskResult, TradeRequest, PortfolioState
from execution import PaperExecutor, OrderRequest, OrderSide, ExecutionResult, ExecutionStatus
from truth.logger import TruthLogger
from truth.schema import DecisionResult, OrderSide as TruthOrderSide
from notifier import DiscordNotifier
import os


logger = logging.getLogger(__name__)


# Feature flag for position reconciliation
POSITION_RECONCILIATION_ENABLED = os.environ.get("POSITION_RECONCILIATION_ENABLED", "true").lower() == "true"


class TradingMode(Enum):
    """Execution state of the engine."""
    NORMAL = "normal"           # Full trading enabled
    EXIT_ONLY = "exit_only"     # Manage existing trades, block new ones
    HALTED = "halted"           # All trading blocked (Fail-Closed)


def signal_to_order_side(signal_value: str, is_short_position: bool = False) -> OrderSide:
    """
    Map signal value to the correct order side (v6.1 SHORT support).

    Signal to Order mapping:
    - "long"       → BUY  (open long position)
    - "short"      → SELL (open short position / borrow & sell)
    - "exit_long"  → SELL (close long / sell holdings)
    - "exit_short" → BUY  (close short / buy to cover)
    - "exit"       → depends on is_short_position
    """
    if signal_value == "long":
        return OrderSide.BUY
    elif signal_value == "short":
        return OrderSide.SELL
    elif signal_value == "exit_long":
        return OrderSide.SELL
    elif signal_value == "exit_short":
        return OrderSide.BUY
    elif signal_value == "exit":
        # Generic exit - depends on current position direction
        return OrderSide.BUY if is_short_position else OrderSide.SELL
    else:
        # Default to SELL for unknown exit-like signals
        return OrderSide.SELL


def signal_to_truth_side(signal_value: str, is_short_position: bool = False) -> TruthOrderSide:
    """Map signal to truth logger order side."""
    order_side = signal_to_order_side(signal_value, is_short_position)
    return TruthOrderSide.BUY if order_side == OrderSide.BUY else TruthOrderSide.SELL


@dataclass
class PositionState:
    """Hydrated position state from Truth Engine."""
    trade_id: str
    symbol: str
    side: str
    entry_price: Decimal
    quantity: Decimal
    stop_loss_price: Optional[Decimal]
    take_profit_price: Optional[Decimal]
    entry_timestamp: str
    highest_high_since_entry: Optional[Decimal] = None
    # SHORT capability (v6.1)
    lowest_low_since_entry: Optional[Decimal] = None

    @property
    def is_short(self) -> bool:
        """Return True if this is a short position."""
        return self.side.lower() in ("sell", "short")

    def has_required_fields(self) -> bool:
        """Check if all required fields for exits are populated."""
        required = [
            self.trade_id,
            self.entry_price,
            self.quantity,
            self.stop_loss_price  # Stop is required for exit logic
        ]
        return all(f is not None for f in required)

    def get_null_fields(self) -> list:
        """Get list of fields that are null but required."""
        nulls = []
        if self.entry_price is None:
            nulls.append("entry_price")
        if self.quantity is None:
            nulls.append("quantity")
        if self.stop_loss_price is None:
            nulls.append("stop_loss_price")
        # For longs, need highest_high; for shorts, need lowest_low
        if not self.is_short and self.highest_high_since_entry is None:
            nulls.append("highest_high_since_entry")
        if self.is_short and self.lowest_low_since_entry is None:
            nulls.append("lowest_low_since_entry")
        return nulls


@dataclass
class ReconciliationResult:
    """Result of position state reconciliation."""
    success: bool
    position_found: bool
    position_state: Optional[PositionState]
    error_message: Optional[str] = None
    requires_fail_closed: bool = False


@dataclass
class TickResult:
    """Result of a single tick through the engine."""
    timestamp: datetime
    signal: Optional[SignalResult]
    risk_result: Optional[RiskResult]
    execution_result: Optional[ExecutionResult]
    decision_id: Optional[str]
    order_id: Optional[str]
    trade_id: Optional[str]
    action_taken: str  # "none", "blocked", "executed"
    lesson_feedback: Optional[Dict[str, Any]] = None  # Learning feedback from past trades


class TradingEngine:
    """
    The V4 Trading Engine - The Glass Box Orchestrator.

    This class doesn't contain trading LOGIC - it contains trading FLOW.
    Each component handles its domain:
    - Strategy: When to trade
    - Risk: Whether we CAN trade
    - Executor: HOW to trade
    - Truth Logger: Recording everything

    The engine just wires them together in the right sequence.
    """

    def __init__(
        self,
        strategy,  # DualEMACrossover or DonchianBreakout
        risk_manager: RiskManager,
        executor: PaperExecutor,
        truth_logger: TruthLogger,
        symbol: str,
        capital: Decimal,
        notifier: Optional[DiscordNotifier] = None
    ):
        """
        Initialize the trading engine.

        Args:
            strategy: The Brain - generates signals
            risk_manager: The Conscience - approves/rejects trades
            executor: The Hands - executes orders
            truth_logger: The Memory - records everything
            symbol: Trading pair (e.g., "BTC-USD")
            capital: Total capital for position sizing
            notifier: The Voice - Discord alerts (optional)
        """
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.executor = executor
        self.truth_logger = truth_logger
        self.symbol = symbol
        self.capital = capital
        self.notifier = notifier

        # Detect strategy name from instance
        if isinstance(strategy, DonchianBreakout):
            self.strategy_name = "donchian_breakout"
        else:
            self.strategy_name = "dual_ema_crossover"

        # Track state
        self.has_open_position = False
        self.daily_pnl = Decimal("0")
        self.total_pnl = Decimal("0")
        self.trades_today = 0

        # High water mark for drawdown calculation
        # FIXED: Was missing - caused incorrect drawdown calculations
        self.high_water_mark = capital

        # TURTLE-4: Track position data for Chandelier Exit
        self.entry_price: Optional[Decimal] = None
        self.highest_high_since_entry: Optional[Decimal] = None
        self.current_stop_price: Optional[Decimal] = None

        # Position reconciliation state
        self.mode = TradingMode.NORMAL
        self.halt_reason: Optional[str] = None
        self.open_trade_id: Optional[str] = None

        logger.info(f"TradingEngine initialized for {symbol} with {self.strategy_name}")
        logger.info(f"Position Reconciliation: {'ENABLED' if POSITION_RECONCILIATION_ENABLED else 'DISABLED'}")

    @property
    def fail_closed(self) -> bool:
        """Compatibility property for old fail_closed checks."""
        return self.mode == TradingMode.HALTED

    # =========================================================================
    # POSITION STATE HYDRATION & RECONCILIATION
    # =========================================================================

    async def hydrate_position_state(self, candle_data: Optional[pd.DataFrame] = None) -> ReconciliationResult:
        """
        Hydrate position state by querying BROKER first, then DB for audit (Async).

        GAP 1 FIX: Broker is the source of truth, not DB.

        This MUST be called on startup before any trading logic runs.
        Populates all required fields including highest_high_since_entry.

        State comparison logic:
        - Broker open, DB empty → fail-closed (CRITICAL mismatch)
        - Broker empty, DB open → fail-closed (CRITICAL mismatch)
        - Both open & match → hydrate from DB, recompute exits, validate
        - Both empty → normal "no_position_found"

        Args:
            candle_data: Historical candle data to compute highest high since entry

        Returns:
            ReconciliationResult with hydrated state or error
        """
        if not POSITION_RECONCILIATION_ENABLED:
            # GAP 5 FIX: Disabled reconciliation is UNSAFE - fail-closed, don't return success
            error_msg = "Position reconciliation DISABLED - UNSAFE: cannot verify broker/DB state"
            logger.critical(f"[HYDRATE] {error_msg}")

            await self.truth_logger.log_reconciliation_event(
                symbol=self.symbol,
                event_type="RECONCILIATION_DISABLED",
                details={"error": error_msg},
                severity="CRITICAL"
            )

            await self.enter_fail_closed(error_msg)

            return ReconciliationResult(
                success=False,
                position_found=False,
                position_state=None,
                error_message=error_msg,
                requires_fail_closed=True
            )

        logger.info(f"[HYDRATE] Loading position state for {self.symbol}...")

        # GAP 1 FIX: Query BROKER first (source of truth)
        broker_has_position = self.executor.has_position(self.symbol)
        broker_position = self.executor.get_position(self.symbol) if broker_has_position else None

        # Query DB for audit/comparison
        db_position = await self.truth_logger.get_open_position(self.symbol)
        db_has_position = db_position is not None

        logger.info(f"[HYDRATE] Broker has position: {broker_has_position}, DB has position: {db_has_position}")

        # STATE COMPARISON - Four cases
        # Case 1: Broker has position, DB doesn't → CRITICAL mismatch
        if broker_has_position and not db_has_position:
            error_msg = f"BROKER has position but DB shows none - orphan position detected"
            logger.error(f"[HYDRATE] CRITICAL: {error_msg}")

            await self.truth_logger.log_reconciliation_event(
                symbol=self.symbol,
                event_type="MISMATCH",
                details={
                    "broker_position": True,
                    "broker_quantity": str(broker_position.get('quantity')) if broker_position else None,
                    "db_position": False,
                    "error": error_msg
                },
                severity="CRITICAL"
            )

            await self.enter_fail_closed(error_msg)

            return ReconciliationResult(
                success=False,
                position_found=True,  # Broker has it
                position_state=None,
                error_message=error_msg,
                requires_fail_closed=True
            )

        # Case 2: Broker empty, DB has position → CRITICAL mismatch
        if not broker_has_position and db_has_position:
            error_msg = f"DB has position but BROKER shows none - ghost position detected"
            logger.error(f"[HYDRATE] CRITICAL: {error_msg}")

            await self.truth_logger.log_reconciliation_event(
                symbol=self.symbol,
                event_type="MISMATCH",
                details={
                    "broker_position": False,
                    "db_position": True,
                    "db_trade_id": db_position.get('trade_id'),
                    "error": error_msg
                },
                severity="CRITICAL"
            )

            await self.enter_fail_closed(error_msg)

            return ReconciliationResult(
                success=False,
                position_found=True,  # DB has it
                position_state=None,
                error_message=error_msg,
                requires_fail_closed=True
            )

        # Case 3: Both empty → No position, clean state
        if not broker_has_position and not db_has_position:
            logger.info(f"[HYDRATE] No open position found for {self.symbol}")
            await self.truth_logger.log_reconciliation_event(
                symbol=self.symbol,
                event_type="HYDRATE",
                details={
                    "result": "no_position_found",
                    "broker_position": False,
                    "db_position": False
                },
                severity="INFO"
            )
            return ReconciliationResult(
                success=True,
                position_found=False,
                position_state=None
            )

        # Case 4: Both have position → Hydrate from DB (has full trade details)
        logger.info(f"[HYDRATE] Both broker and DB have position - proceeding with hydration")

        # Build position state
        position = PositionState(
            trade_id=db_position['trade_id'],
            symbol=db_position['symbol'],
            side=db_position['side'],
            entry_price=db_position['entry_price'],
            quantity=db_position['quantity'],
            stop_loss_price=db_position.get('stop_loss_price'),
            take_profit_price=db_position.get('take_profit_price'),
            entry_timestamp=db_position['entry_timestamp']
        )

        # Compute highest_high_since_entry from candle data
        if candle_data is not None and len(candle_data) > 0:
            entry_ts = pd.to_datetime(position.entry_timestamp)
            # Filter candles since entry
            if 'timestamp' in candle_data.columns:
                candles_since_entry = candle_data[candle_data['timestamp'] >= entry_ts]
            else:
                # Assume index is datetime
                candles_since_entry = candle_data[candle_data.index >= entry_ts]

            if len(candles_since_entry) > 0:
                highest_high = Decimal(str(candles_since_entry['high'].max()))
                position.highest_high_since_entry = highest_high
                logger.info(f"[HYDRATE] Computed highest_high_since_entry: ${highest_high}")
            else:
                # GAP 4: This will trigger fail-closed via null_fields check below
                logger.error("[HYDRATE] No candles found since entry - WILL FAIL-CLOSED (cannot arm chandelier exit)")
        else:
            # GAP 4: This will trigger fail-closed via null_fields check below
            logger.error("[HYDRATE] No candle data provided - WILL FAIL-CLOSED (cannot arm exits)")

        # Check for required fields
        null_fields = position.get_null_fields()
        if null_fields:
            error_msg = f"Position has NULL required fields: {null_fields}"
            logger.error(f"[HYDRATE] CRITICAL: {error_msg}")

            await self.truth_logger.log_reconciliation_event(
                symbol=self.symbol,
                event_type="HYDRATE_FAILED",
                details={
                    "trade_id": position.trade_id,
                    "null_fields": null_fields,
                    "error": error_msg
                },
                severity="CRITICAL"
            )

            # GAP 3 FIX: ENFORCE fail-closed immediately, don't just flag it
            await self.enter_fail_closed(error_msg)

            return ReconciliationResult(
                success=False,
                position_found=True,
                position_state=position,
                error_message=error_msg,
                requires_fail_closed=True
            )

        # Success - populate engine state
        self.has_open_position = True
        self.entry_price = position.entry_price
        self.highest_high_since_entry = position.highest_high_since_entry
        self.lowest_low_since_entry = getattr(position, 'lowest_low_since_entry', None)  # v6.1
        self.is_short_position = position.is_short  # v6.1
        self.current_stop_price = position.stop_loss_price
        self.open_trade_id = position.trade_id

        logger.info(f"[HYDRATE] SUCCESS - Position loaded:")
        logger.info(f"  Trade ID: {position.trade_id}")
        logger.info(f"  Side: {'SHORT' if self.is_short_position else 'LONG'}")
        logger.info(f"  Entry: ${position.entry_price}")
        logger.info(f"  Stop: ${position.stop_loss_price}")
        if self.is_short_position:
            logger.info(f"  Lowest Low: ${self.lowest_low_since_entry}")
        else:
            logger.info(f"  Highest High: ${position.highest_high_since_entry}")

        await self.truth_logger.log_reconciliation_event(
            symbol=self.symbol,
            event_type="HYDRATE",
            details={
                "trade_id": position.trade_id,
                "entry_price": str(position.entry_price),
                "stop_loss": str(position.stop_loss_price),
                "highest_high": str(position.highest_high_since_entry),
                "result": "success",
                "broker_position": True,
                "db_position": True,
                "broker_quantity": str(broker_position.get('quantity')) if broker_position else None
            },
            severity="INFO"
        )

        return ReconciliationResult(
            success=True,
            position_found=True,
            position_state=position
        )

    async def reconcile_position(self) -> ReconciliationResult:
        """
        Reconcile engine state against broker and DB (Async).

        GAP 2 FIX: Queries broker internally, doesn't trust caller input.

        Called at startup and periodically during trading.
        Detects and handles state mismatches between broker, DB, and engine.

        Returns:
            ReconciliationResult with reconciliation outcome
        """
        if not POSITION_RECONCILIATION_ENABLED:
            # GAP 5 FIX: Disabled reconciliation is UNSAFE - fail-closed
            error_msg = "Position reconciliation DISABLED - UNSAFE: cannot verify state"
            logger.critical(f"[RECONCILE] {error_msg}")
            await self.enter_fail_closed(error_msg)
            return ReconciliationResult(
                success=False,
                position_found=False,
                position_state=None,
                error_message=error_msg,
                requires_fail_closed=True
            )

        # GAP 2 FIX: Query broker directly, don't trust caller
        broker_has_position = self.executor.has_position(self.symbol)
        broker_position = self.executor.get_position(self.symbol) if broker_has_position else None

        db_position = await self.truth_logger.get_open_position(self.symbol)
        db_has_position = db_position is not None
        engine_has_position = self.has_open_position

        logger.info(f"[RECONCILE] Broker: {broker_has_position}, DB: {db_has_position}, Engine: {engine_has_position}")

        # Check for mismatches
        if broker_has_position and not db_has_position:
            # Broker has position but DB doesn't - ANOMALY
            error_msg = "Broker has position but DB shows none - orphan position"
            logger.error(f"[RECONCILE] CRITICAL: {error_msg}")

            await self.truth_logger.log_reconciliation_event(
                symbol=self.symbol,
                event_type="MISMATCH",
                details={
                    "broker_position": True,
                    "broker_quantity": str(broker_position.get('quantity')) if broker_position else None,
                    "db_position": False,
                    "engine_position": engine_has_position,
                    "error": error_msg
                },
                severity="CRITICAL"
            )

            # GAP 3 FIX: ENFORCE fail-closed immediately, don't just flag it
            await self.enter_fail_closed(error_msg)

            return ReconciliationResult(
                success=False,
                position_found=True,  # Broker has it
                position_state=None,
                error_message=error_msg,
                requires_fail_closed=True
            )

        if db_has_position and not engine_has_position:
            # DB has position but engine doesn't know - needs hydration
            error_msg = "DB has position but engine not tracking it"
            logger.error(f"[RECONCILE] CRITICAL: {error_msg}")

            await self.truth_logger.log_reconciliation_event(
                symbol=self.symbol,
                event_type="MISMATCH",
                details={
                    "broker_position": broker_has_position,
                    "broker_quantity": str(broker_position.get('quantity')) if broker_position else None,
                    "db_position": True,
                    "engine_position": False,
                    "trade_id": db_position['trade_id'],
                    "error": error_msg
                },
                severity="CRITICAL"
            )

            # GAP 3 FIX: ENFORCE fail-closed immediately, don't just flag it
            await self.enter_fail_closed(error_msg)

            return ReconciliationResult(
                success=False,
                position_found=True,
                position_state=None,
                error_message=error_msg,
                requires_fail_closed=True
            )

        if engine_has_position and not db_has_position:
            # Engine thinks it has position but DB doesn't - close state
            logger.warning("[RECONCILE] Engine has position but DB shows closed - clearing engine state")

            await self.truth_logger.log_reconciliation_event(
                symbol=self.symbol,
                event_type="STATE_CLEARED",
                details={
                    "reason": "DB shows no open position",
                    "engine_entry_price": str(self.entry_price) if self.entry_price else None
                },
                severity="WARNING"
            )

            self.has_open_position = False
            self.entry_price = None
            self.highest_high_since_entry = None
            self.lowest_low_since_entry = None  # v6.1: SHORT support
            self.is_short_position = False      # v6.1: SHORT support
            self.open_trade_id = None

        return ReconciliationResult(
            success=True,
            position_found=db_has_position,
            position_state=None
        )

    async def enter_fail_closed(self, reason: str) -> None:
        """
        Enter HALTED mode. Blocks all trading until resolved (Async).

        Args:
            reason: Human-readable reason for fail-closed state
        """
        self.mode = TradingMode.HALTED
        self.halt_reason = reason

        logger.critical(f"[HALTED] Trading BLOCKED: {reason}")

        await self.truth_logger.log_reconciliation_event(
            symbol=self.symbol,
            event_type="FAIL_CLOSED",
            details={
                "reason": reason,
                "action": "trading_halted"
            },
            severity="CRITICAL"
        )

        # Send Discord alert if notifier available
        if self.notifier:
            try:
                self.notifier.send_risk_alert(
                    f"**🚨 FAIL-CLOSED: {self.symbol}**\n\n"
                    f"Trading HALTED due to position state error:\n"
                    f"```{reason}```\n\n"
                    f"Manual intervention required.",
                    severity="critical"
                )
            except Exception as e:
                logger.error(f"Failed to send Discord alert: {e}")

    async def run_tick(self, data: pd.DataFrame) -> TickResult:
        """
        Run one tick through the engine (Async).

        THE SEQUENCE (do not change order):
        0. Check mode - BLOCK if HALTED, FILTER if EXIT_ONLY
        1. Ask the Brain - strategy.evaluate(data)
        2. Check the Conscience - risk.evaluate(trade_request)
        3. Log the Decision - truth.log_decision() ALWAYS
        4. Move the Hands - executor.execute() IF APPROVED
        5. Log the Action - truth.log_order(), truth.log_trade()

        Args:
            data: DataFrame with OHLCV data for analysis

        Returns:
            TickResult with all outcomes
        """
        timestamp = datetime.utcnow()
        result = TickResult(
            timestamp=timestamp,
            signal=None,
            risk_result=None,
            execution_result=None,
            decision_id=None,
            order_id=None,
            trade_id=None,
            action_taken="none"
        )

        # =====================================================================
        # STEP 0: Check mode (NON-NEGOTIABLE)
        # =====================================================================
        if self.mode == TradingMode.HALTED:
            logger.warning(f"[HALTED] Trading blocked: {self.halt_reason}")
            # Get current price from data if available
            current_price = float(data.iloc[-1]['close']) if len(data) > 0 else None
            # Log the blocked decision
            decision = await self.truth_logger.log_decision(
                symbol=self.symbol,
                strategy_name=self.strategy_name,
                signal_values={"mode": "HALTED", "reason": self.halt_reason},
                risk_checks={},
                result=DecisionResult.RISK_REJECTED,
                result_reason=f"Trading HALTED. All operations suspended. Reason: {self.halt_reason}",
                market_context={"current_price": current_price} if current_price else None
            )
            result.decision_id = decision.decision_id
            result.action_taken = "fail_closed"
            return result

        # =====================================================================
        # TURTLE-4: Update highest high / lowest low since entry (The Ratchet)
        # =====================================================================
        if self.has_open_position and len(data) > 0:
            current_high = Decimal(str(data.iloc[-1]['high']))
            current_low = Decimal(str(data.iloc[-1]['low']))

            if getattr(self, 'is_short_position', False):
                # SHORT: Track lowest low (Inverse Ratchet - v6.1)
                if self.lowest_low_since_entry is None:
                    self.lowest_low_since_entry = current_low
                else:
                    # The Inverse Ratchet only moves DOWN, never up
                    self.lowest_low_since_entry = min(self.lowest_low_since_entry, current_low)
            else:
                # LONG: Track highest high (Original Ratchet)
                if self.highest_high_since_entry is None:
                    self.highest_high_since_entry = current_high
                else:
                    # The Ratchet only moves UP, never down
                    self.highest_high_since_entry = max(self.highest_high_since_entry, current_high)

        # =====================================================================
        # STEP 1: Ask the Brain
        # =====================================================================
        logger.info("Step 1: Asking the Brain (Strategy)...")

        signal = self.strategy.evaluate(
            df=data,
            timestamp=timestamp,
            has_open_position=self.has_open_position,
            entry_price=self.entry_price,
            highest_high_since_entry=self.highest_high_since_entry,
            # SHORT capability (v6.1)
            is_short_position=getattr(self, 'is_short_position', False),
            lowest_low_since_entry=getattr(self, 'lowest_low_since_entry', None)
        )
        result.signal = signal

        # Update emergency stop price from context
        if self.has_open_position:
            # Use appropriate Chandelier Stop based on position direction
            if getattr(self, 'is_short_position', False):
                # SHORT: Use inverse chandelier or fallback
                self.current_stop_price = signal.context.short_chandelier_stop or signal.context.stop_loss_price
            else:
                # LONG: Use regular chandelier or fallback
                self.current_stop_price = signal.context.chandelier_stop or signal.context.stop_loss_price
            logger.info(f"  Emergency stop armed at ${self.current_stop_price:,.2f}")

        logger.info(f"  Signal: {signal.signal.value}, Confidence: {signal.confidence}")

        # EXIT ONLY FILTER (Suggestion 2)
        if self.mode == TradingMode.EXIT_ONLY and signal.signal.value in ["long", "short"]:
            logger.warning(f"[EXIT-ONLY] Entry signal {signal.signal.value} BLOCKED.")
            decision = await self.truth_logger.log_decision(
                symbol=self.symbol,
                strategy_name=self.strategy_name,
                signal_values=signal.to_signal_values(),
                risk_checks={},
                result=DecisionResult.RISK_REJECTED,
                result_reason="EXIT-ONLY mode active: new entries blocked.",
                market_context={"current_price": float(signal.context.current_price)} if signal.context.current_price else None
            )
            result.decision_id = decision.decision_id
            result.action_taken = "blocked"
            return result

        # If no action, log decision and return
        # Compare by value to support different strategy Signal enums
        if signal.signal.value == "hold":
            # Build descriptive hold reason from signal context
            ctx = signal.context
            hold_details = []
            if ctx.current_price:
                hold_details.append(f"price ${float(ctx.current_price):,.2f}")
            if hasattr(ctx, 'ema_short') and ctx.ema_short:
                hold_details.append(f"EMA trend: {'above' if ctx.current_price > ctx.ema_long else 'below'}")
            hold_info = f" ({', '.join(hold_details)})" if hold_details else ""

            # Log the HOLD decision
            decision = await self.truth_logger.log_decision(
                symbol=self.symbol,
                strategy_name=self.strategy_name,
                signal_values=signal.to_signal_values(),
                risk_checks={},
                result=DecisionResult.NO_SIGNAL,
                result_reason=f"No trade signal. Market conditions do not meet entry/exit criteria.{hold_info}",
                market_context={"current_price": float(signal.context.current_price)} if signal.context.current_price else None
            )
            result.decision_id = decision.decision_id
            result.action_taken = "none"
            logger.info("  No action - HOLD signal")
            return result

        # =====================================================================
        # STEP 1.5: Query Lessons from Memory (Learning Feedback Loop)
        # =====================================================================
        lesson_feedback = None
        size_adjustment = Decimal("1.0")  # Default: no adjustment

        if signal.signal.value in ["long", "short"]:
            logger.info("Step 1.5: Querying lessons from memory...")
            try:
                # Build market context from signal
                market_context = {
                    "price": float(signal.context.current_price),
                    "atr": float(signal.context.atr) if signal.context.atr else 0,
                    "volume_ratio": float(signal.context.volume_ratio) if hasattr(signal.context, 'volume_ratio') and signal.context.volume_ratio else 0,
                }

                lesson_feedback = await self.truth_logger.get_relevant_lessons(
                    symbol=self.symbol,
                    signals=signal.to_signal_values(),
                    market_context=market_context
                )
                result.lesson_feedback = lesson_feedback

                # Apply lessons to this decision
                if lesson_feedback:
                    size_adjustment = Decimal(str(lesson_feedback.get("size_adjustment", 1.0)))
                    warnings = lesson_feedback.get("warnings", [])
                    lesson_count = lesson_feedback.get("lesson_count", 0)
                    support_ratio = lesson_feedback.get("support_ratio", 0.5)

                    logger.info(f"  Lessons found: {lesson_count}, Support ratio: {support_ratio:.0%}")
                    if size_adjustment < 1.0:
                        logger.warning(f"  ⚠️ Position size reduced to {size_adjustment:.0%} based on past lessons")
                    if warnings:
                        for w in warnings[:3]:
                            logger.warning(f"  ⚠️ {w}")
            except Exception as e:
                logger.error(f"  Lesson query failed: {e}")
                lesson_feedback = {"error": str(e)}

        # =====================================================================
        # STEP 2: Check the Conscience
        # =====================================================================
        logger.info("Step 2: Checking the Conscience (Risk)...")

        # Build trade request from signal
        entry_price = signal.context.current_price
        stop_loss = signal.context.stop_loss_price
        take_profit = signal.context.take_profit_price

        # POSITION SIZING LOGIC
        if signal.signal.value in ["long", "short"]:
            # Entry: Calculate size based on risk
            quantity = self.strategy.calculate_position_size(
                capital=self.capital,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                risk_percent=Decimal("0.01")  # Default 1%
            )
            
            # HARD CAP: Max 1x Leverage (Suggestion 4)
            # Never commit more than 95% of available capital to a single trade
            max_quantity = (self.capital * Decimal("0.95")) / entry_price
            if quantity > max_quantity:
                logger.info(f"  Position size capped: {quantity} -> {max_quantity} (1x Leverage limit)")
                quantity = max_quantity

            # LEARNING FEEDBACK: Apply size adjustment from past lessons
            if size_adjustment < Decimal("1.0"):
                original_quantity = quantity
                quantity = quantity * size_adjustment
                logger.info(f"  📚 Learning adjustment: {original_quantity} -> {quantity} ({size_adjustment:.0%} of original)")
        else:
            # Exit: Use current position size (absolute value, shorts return negative)
            quantity = abs(self.executor.get_position_size(self.symbol))

        trade_request = TradeRequest(
            symbol=self.symbol,
            side="buy" if signal_to_order_side(signal.signal.value, getattr(self, 'is_short_position', False)) == OrderSide.BUY else "sell",
            quantity=quantity,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            strategy_name=self.strategy_name,
            confidence=signal.confidence,
            is_exit=signal.signal.value in ["exit_long", "exit_short", "exit"]
        )

        # Build portfolio state
        # FIXED: Calculate drawdown from high water mark, not from starting capital
        # Drawdown = (current_equity - HWM) / HWM * 100 (negative when in drawdown)
        current_equity = self.capital + self.total_pnl
        drawdown_pct = 0.0
        if self.high_water_mark > 0:
            drawdown_pct = float((current_equity - self.high_water_mark) / self.high_water_mark * 100)

        portfolio = PortfolioState(
            total_capital=self.capital,
            cash_available=await self.executor.get_balance("USD"),
            daily_pnl=self.daily_pnl,
            daily_pnl_percent=float(self.daily_pnl / self.capital * 100) if self.capital > 0 else 0,
            total_pnl=self.total_pnl,
            total_pnl_percent=drawdown_pct,  # FIXED: Now uses drawdown from HWM
            open_positions={},
            recent_trades_count=self.trades_today,
            circuit_breaker_triggered=False,
            trading_halted=False
        )

        risk_result = self.risk_manager.evaluate(trade_request, portfolio)
        result.risk_result = risk_result

        logger.info(f"  Risk: {'APPROVED' if risk_result.approved else 'REJECTED'}")
        if not risk_result.approved:
            logger.info(f"  Reason: {risk_result.rejection_reason}")

        # =====================================================================
        # STEP 3: Log the Decision (ALWAYS - even if rejected!)
        # =====================================================================
        logger.info("Step 3: Logging the Decision...")

        if risk_result.approved:
            ctx = signal.context
            price_info = f" @ ${float(ctx.current_price):,.2f}" if ctx.current_price else ""

            # FIXED: Properly handle all signal types including exits
            if signal.signal.value == "long":
                decision_result = DecisionResult.SIGNAL_LONG
                sl_info = f", SL ${float(ctx.stop_loss_price):,.2f}" if ctx.stop_loss_price else ""
                result_reason = f"Opening LONG position{price_info}{sl_info}. Confidence: {signal.confidence:.0%}. All risk checks passed."
            elif signal.signal.value == "short":
                decision_result = DecisionResult.SIGNAL_SHORT
                sl_info = f", SL ${float(ctx.stop_loss_price):,.2f}" if ctx.stop_loss_price else ""
                result_reason = f"Opening SHORT position{price_info}{sl_info}. Confidence: {signal.confidence:.0%}. All risk checks passed."
            elif signal.signal.value in ["exit_long", "exit_short", "exit"]:
                # FIXED: Exit signals now properly logged with full context
                decision_result = DecisionResult.SIGNAL_CLOSE

                # Build rich exit context
                exit_context_parts = [f"Closing position{price_info}"]

                # Add Chandelier Exit context if available
                if hasattr(ctx, 'chandelier_triggered') and ctx.chandelier_triggered:
                    chandelier_stop = getattr(ctx, 'chandelier_stop', None)
                    if chandelier_stop:
                        exit_context_parts.append(f"Chandelier Exit triggered (stop: ${float(chandelier_stop):,.2f})")
                    else:
                        exit_context_parts.append("Chandelier Exit triggered")

                # Add highest high since entry
                if hasattr(ctx, 'highest_high_since_entry') and ctx.highest_high_since_entry:
                    exit_context_parts.append(f"HH since entry: ${float(ctx.highest_high_since_entry):,.2f}")

                # Add signal reason
                exit_context_parts.append(f"Reason: {signal.reason}")
                exit_context_parts.append(f"Confidence: {signal.confidence:.0%}")

                result_reason = ". ".join(exit_context_parts)
            else:
                decision_result = DecisionResult.NO_SIGNAL
                result_reason = f"Unknown signal type: {signal.signal.value}"
        else:
            decision_result = DecisionResult.RISK_REJECTED
            result_reason = f"Trade BLOCKED by risk management. {risk_result.rejection_reason}"

        # Build enhanced market context for exits
        market_context = {"current_price": float(signal.context.current_price)} if signal.context.current_price else {}

        # Add exit-specific context if this is an exit signal
        if signal.signal.value in ["exit_long", "exit_short", "exit"]:
            ctx = signal.context
            if hasattr(ctx, 'chandelier_stop') and ctx.chandelier_stop:
                market_context["chandelier_stop"] = float(ctx.chandelier_stop)
            if hasattr(ctx, 'chandelier_triggered'):
                market_context["chandelier_triggered"] = ctx.chandelier_triggered
            if hasattr(ctx, 'highest_high_since_entry') and ctx.highest_high_since_entry:
                market_context["highest_high_since_entry"] = float(ctx.highest_high_since_entry)
            if hasattr(ctx, 'atr') and ctx.atr:
                market_context["atr"] = float(ctx.atr)

        # Add learning feedback to market context
        if lesson_feedback:
            market_context["learning_feedback"] = {
                "lesson_count": lesson_feedback.get("lesson_count", 0),
                "size_adjustment": float(lesson_feedback.get("size_adjustment", 1.0)),
                "support_ratio": lesson_feedback.get("support_ratio", 0.5),
                "warnings": lesson_feedback.get("warnings", [])[:3],
                "regime": lesson_feedback.get("regime", "unknown")
            }

        decision = await self.truth_logger.log_decision(
            symbol=self.symbol,
            strategy_name=self.strategy_name,
            signal_values=signal.to_signal_values(),
            risk_checks=risk_result.to_dict(),
            result=decision_result,
            result_reason=result_reason,
            market_context=market_context
        )
        result.decision_id = decision.decision_id

        logger.info(f"  Decision logged: {decision.decision_id}")

        # If rejected, stop here
        if not risk_result.approved:
            result.action_taken = "blocked"
            logger.info("  Trade blocked by risk manager")
            return result

        # =====================================================================
        # STEP 4: Move the Hands
        # =====================================================================
        logger.info("Step 4: Moving the Hands (Execution)...")

        order_request = OrderRequest(
            symbol=self.symbol,
            side=signal_to_order_side(signal.signal.value, getattr(self, 'is_short_position', False)),
            quantity=quantity,
            expected_price=entry_price
        )

        exec_result = await self.executor.execute(order_request)
        result.execution_result = exec_result

        logger.info(f"  Execution: {exec_result.status.value}")
        if exec_result.fill_price:
            logger.info(f"  Fill: {exec_result.fill_price}, Slippage: {exec_result.slippage_pct:.4f}%")

        # =====================================================================
        # STEP 5: Log the Action
        # =====================================================================
        logger.info("Step 5: Logging the Action...")

        # Log order
        order = await self.truth_logger.log_order(
            decision_id=decision.decision_id,
            symbol=self.symbol,
            side=signal_to_truth_side(signal.signal.value, getattr(self, 'is_short_position', False)),
            quantity=quantity,
            requested_price=entry_price
        )
        result.order_id = order.order_id

        # Update order with fill if successful
        if exec_result.status == ExecutionStatus.FILLED:
            await self.truth_logger.update_order_fill(
                order_id=order.order_id,
                fill_price=exec_result.fill_price,
                fill_quantity=exec_result.fill_quantity,
                exchange_order_id=exec_result.external_id,
                commission=exec_result.fee
            )

            # LIFECYCLE MANAGEMENT (Fix for 0% Win Rate)
            if signal.signal.value in ["exit_long", "exit_short", "exit"] and self.open_trade_id:
                # This is an EXIT - Close the existing trade record
                from truth.schema import ExitReason as TruthExitReason
                await self.truth_logger.close_trade(
                    trade_id=self.open_trade_id,
                    exit_order_id=order.order_id,
                    exit_price=exec_result.fill_price,
                    exit_reason=TruthExitReason.SIGNAL_EXIT, # Assume signal exit unless triggered by emergency
                    commission=exec_result.fee, # Note: Total commission usually summed by logger
                    slippage=exec_result.slippage
                )
                logger.info(f"  Trade CLOSED: {self.open_trade_id}")
                self.open_trade_id = None
                self.has_open_position = False
                self.entry_price = None
                self.highest_high_since_entry = None
                self.lowest_low_since_entry = None   # v6.1: SHORT support
                self.is_short_position = False        # v6.1: SHORT support
                self.current_stop_price = None
            else:
                # This is an ENTRY - Open a new trade record
                trade_id = await self.truth_logger.open_trade(
                    symbol=self.symbol,
                    side=signal_to_truth_side(signal.signal.value, False),  # New position, not short yet
                    entry_order_id=order.order_id,
                    entry_price=exec_result.fill_price,
                    quantity=exec_result.fill_quantity,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    strategy_name=self.strategy_name
                )
                self.open_trade_id = trade_id
                self.has_open_position = True
                self.entry_price = exec_result.fill_price
                self.is_short_position = (signal.signal.value == "short")  # v6.1: Track direction

                # Set initial tracking values based on position direction
                if len(data) > 0:
                    if self.is_short_position:
                        # SHORT: Initialize lowest low for inverse ratchet
                        self.lowest_low_since_entry = Decimal(str(data.iloc[-1]['low']))
                        self.highest_high_since_entry = None
                        logger.info(f"  Trade OPENED (SHORT): {trade_id}")
                    else:
                        # LONG: Initialize highest high for ratchet
                        self.highest_high_since_entry = Decimal(str(data.iloc[-1]['high']))
                        self.lowest_low_since_entry = None
                        logger.info(f"  Trade OPENED (LONG): {trade_id}")

            self.trades_today += 1
            logger.info(f"  Order logged: {order.order_id}")

            # =====================================================================
            # STEP 6: Alert the Commander (Discord)
            # =====================================================================
            if self.notifier:
                logger.info("Step 6: Alerting Command (Discord)...")
                try:
                    # v6.1: Correct side for all signal types
                    if signal.signal.value == "long":
                        alert_side = "BUY (LONG)"
                    elif signal.signal.value == "short":
                        alert_side = "SELL (SHORT)"
                    elif signal.signal.value == "exit_long":
                        alert_side = "SELL (Exit LONG)"
                    elif signal.signal.value == "exit_short":
                        alert_side = "BUY (Cover SHORT)"
                    else:
                        alert_side = "SELL"

                    self.notifier.send_trade_alert(
                        symbol=self.symbol,
                        side=alert_side,
                        entry_price=float(exec_result.fill_price),
                        size=float(exec_result.fill_quantity),
                        stop_loss=float(stop_loss),
                        take_profit=float(take_profit) if take_profit else None
                    )
                    logger.info("  Discord alert sent")
                except Exception as e:
                    logger.warning(f"  Discord alert failed: {e}")

        result.action_taken = "executed"
        logger.info("Tick complete.")

        return result

    async def close_position(
        self,
        exit_price: Decimal,
        exit_reason: str = "Strategy exit signal"
    ) -> Optional[str]:
        """
        Close the current open position (Async).

        Args:
            exit_price: The price at which the position was closed
            exit_reason: Why the position was closed

        Returns:
            trade_id if closed successfully, None otherwise
        """
        if not self.has_open_position or not self.entry_price or not self.open_trade_id:
            logger.warning("No open position to close")
            return None

        # Calculate P&L (v6.1: Handle shorts - quantity is negative for shorts)
        raw_quantity = self.executor.get_position_size(self.symbol)
        quantity = abs(raw_quantity)  # Order quantity must be positive
        is_short = getattr(self, 'is_short_position', False) or raw_quantity < 0

        # P&L calculation differs for shorts
        if is_short:
            # Short P&L: profit when price goes DOWN
            pnl = (self.entry_price - exit_price) * quantity
        else:
            # Long P&L: profit when price goes UP
            pnl = (exit_price - self.entry_price) * quantity
        pnl_percent = float(pnl / (self.entry_price * quantity) * 100) if self.entry_price else 0

        logger.info(f"Closing {'SHORT' if is_short else 'LONG'} position: Entry={self.entry_price}, Exit={exit_price}, P&L={pnl}")

        # Update DB via Truth Engine
        exit_decision = await self.truth_logger.log_decision(
            symbol=self.symbol,
            strategy_name=self.strategy_name,
            signal_values={"exit_price": float(exit_price), "reason": exit_reason, "is_short": is_short},
            risk_checks={},
            result=DecisionResult.SIGNAL_CLOSE,
            result_reason=exit_reason,
            market_context={"current_price": float(exit_price)}
        )

        # Exit order side: BUY to cover shorts, SELL to close longs
        exit_side = TruthOrderSide.BUY if is_short else TruthOrderSide.SELL
        exit_order = await self.truth_logger.log_order(
            decision_id=exit_decision.decision_id,
            symbol=self.symbol,
            side=exit_side,
            quantity=quantity,
            requested_price=exit_price
        )

        # Update order fill (assume instant fill for emergency/market close)
        await self.truth_logger.update_order_fill(
            order_id=exit_order.order_id,
            fill_price=exit_price,
            fill_quantity=quantity
        )

        # Close the trade record
        from truth.schema import ExitReason as TruthExitReason
        await self.truth_logger.close_trade(
            trade_id=self.open_trade_id,
            exit_order_id=exit_order.order_id,
            exit_price=exit_price,
            exit_reason=TruthExitReason.STOP_LOSS if "Stop" in exit_reason else TruthExitReason.SIGNAL_EXIT
        )

        # Update in-memory state
        entry_price_for_alert = self.entry_price
        self.has_open_position = False
        self.entry_price = None
        self.highest_high_since_entry = None
        self.lowest_low_since_entry = None   # v6.1: SHORT support
        self.is_short_position = False        # v6.1: SHORT support
        self.current_stop_price = None
        self.open_trade_id = None

        # Update daily/total P&L
        self.daily_pnl += pnl
        self.total_pnl += pnl

        # Update high water mark for drawdown calculation
        # FIXED: Was missing - HWM only increases, never decreases
        current_equity = self.capital + self.total_pnl
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        # Send Discord alert
        if self.notifier:
            try:
                self.notifier.send_exit_alert(
                    symbol=self.symbol,
                    side="SHORT" if is_short else "LONG",  # v6.1: Correct side
                    entry_price=float(entry_price_for_alert),
                    exit_price=float(exit_price),
                    pnl=float(pnl),
                    pnl_percent=pnl_percent
                )
                logger.info("  Discord exit alert sent")
            except Exception as e:
                logger.error(f"  Discord exit alert failed: {e}")

        return "closed"

    async def update_price_and_check_stops(self, price: Decimal) -> bool:
        """
        Instant price update and emergency stop check (Async).
        
        Called by WebSocket sidecar. Checks if current price violates
        existing stop-loss or chandelier ratchet.
        
        Returns True if emergency exit was triggered.
        """
        if not self.has_open_position or self.mode == TradingMode.HALTED:
            return False

        is_short = getattr(self, 'is_short_position', False)

        if is_short:
            # SHORT: Update Lowest Low (Inverse Ratchet - v6.1)
            if self.lowest_low_since_entry and price < self.lowest_low_since_entry:
                self.lowest_low_since_entry = price
                return False

            # SHORT: Stop triggers when price goes UP above stop
            if self.current_stop_price and price >= self.current_stop_price:
                logger.warning(f"🚨 EMERGENCY STOP TRIGGERED (SHORT): Price ${price:,.2f} >= Stop ${self.current_stop_price:,.2f}")
                await self.close_position(
                    exit_price=price,
                    exit_reason="Emergency Stop (WebSocket)"
                )
                return True
        else:
            # LONG: Update Highest High (Original Ratchet)
            if self.highest_high_since_entry and price > self.highest_high_since_entry:
                self.highest_high_since_entry = price
                return False

            # LONG: Stop triggers when price goes DOWN below stop
            if self.current_stop_price and price <= self.current_stop_price:
                logger.warning(f"🚨 EMERGENCY STOP TRIGGERED (LONG): Price ${price:,.2f} <= Stop ${self.current_stop_price:,.2f}")
                await self.close_position(
                    exit_price=price,
                    exit_reason="Emergency Stop (WebSocket)"
                )
                return True

        return False
