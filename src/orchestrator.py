"""
Trading Orchestrator - Central Coordinator for 24/7 Perpetual Trading

Coordinates:
- Session awareness (dead zone detection, auto-pause)
- Multi-timeframe strategy evaluation
- Portfolio-level risk aggregation
- Cross-asset coordination

This is the main entry point for perpetual trading operations.
Replaces the simple loop in live_paper_trader.py with full orchestration.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from src.session import (
    SessionManager,
    SessionConfig,
    SessionState,
    SessionContext,
    SessionEvent,
    MarketSession,
    LiquidityMonitor,
    LiquidityMetrics
)
from src.engine import (
    MultiTimeframeEngine,
    TimeframeUnit,
    SignalAggregator,
    AggregatedSignal,
    SignalType,
    ConflictResolution,
    MultiTimeframeConfig
)
from src.risk.portfolio import (
    PortfolioRiskAggregator,
    PortfolioRiskConfig,
    PortfolioRiskResult
)
from src.truth.schema import Decision, DecisionResult


logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the trading orchestrator."""
    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])

    # Timing
    tick_interval_seconds: int = 60         # How often to check
    dead_zone_check_interval_seconds: int = 300  # Check every 5 min in dead zone

    # Session config
    session: SessionConfig = field(default_factory=SessionConfig)

    # Multi-timeframe config
    multi_timeframe: MultiTimeframeConfig = field(default_factory=MultiTimeframeConfig)

    # Portfolio risk config
    portfolio_risk: PortfolioRiskConfig = field(default_factory=PortfolioRiskConfig)

    # Capital
    total_capital: Decimal = Decimal("20000")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorConfig":
        """Create from config dictionary."""
        return cls(
            symbols=data.get("symbols", ["BTC-USD", "ETH-USD"]),
            tick_interval_seconds=data.get("tick_interval_seconds", 60),
            dead_zone_check_interval_seconds=data.get("dead_zone_check_interval_seconds", 300),
            session=SessionConfig.from_dict(data.get("session", {})),
            multi_timeframe=MultiTimeframeConfig.from_dict(data.get("multi_timeframe", {})),
            portfolio_risk=PortfolioRiskConfig.from_dict(data.get("portfolio_risk", {})),
            total_capital=Decimal(str(data.get("total_capital", 20000)))
        )


@dataclass
class TickResult:
    """Result of a single orchestration tick."""
    symbol: str
    timestamp: datetime
    session_context: SessionContext
    mtf_signal: Optional[AggregatedSignal]
    portfolio_check: Optional[PortfolioRiskResult]
    decision: Optional[Decision]
    action_taken: str                   # "trade", "hold", "blocked", "paused"
    details: Dict[str, Any] = field(default_factory=dict)


class TradingOrchestrator:
    """
    Central coordinator for 24/7 perpetual trading.

    Responsibilities:
    - Session-aware scheduling
    - Multi-timeframe coordination across assets
    - Portfolio-level risk aggregation
    - Dead zone handling
    - Fail-closed enforcement
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        data_fetcher: Callable[[str, str], pd.DataFrame],
        trade_executor: Optional[Callable[[str, str, Decimal], bool]] = None,
        notifier: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize trading orchestrator.

        Args:
            config: Orchestrator configuration.
            data_fetcher: Function to fetch OHLCV data: (symbol, interval) -> DataFrame
            trade_executor: Function to execute trades: (symbol, side, quantity) -> success
            notifier: Function to send notifications: (message) -> None
        """
        self.config = config
        self.data_fetcher = data_fetcher
        self.trade_executor = trade_executor
        self.notifier = notifier

        # Core components
        self.session_manager = SessionManager(config.session)
        self.liquidity_monitor = LiquidityMonitor(config.session)
        self.portfolio_risk = PortfolioRiskAggregator(
            config.portfolio_risk,
            config.total_capital
        )

        # Multi-timeframe engines per symbol
        self.mtf_engines: Dict[str, MultiTimeframeEngine] = {}

        # State
        self._shutdown = False
        self._paused = False
        self._last_session_event: Optional[SessionEvent] = None
        self._tick_count = 0
        self._decisions: List[Decision] = []

    def setup_mtf_engine(
        self,
        symbol: str,
        strategy_evaluator: Callable[[pd.DataFrame], Dict[str, Any]]
    ) -> None:
        """
        Set up multi-timeframe engine for a symbol.

        Args:
            symbol: Trading pair.
            strategy_evaluator: Function to evaluate strategy on data.
        """
        units = []
        for tf_config in self.config.multi_timeframe.timeframes:
            if not tf_config.enabled:
                continue

            unit = TimeframeUnit(
                symbol=symbol,
                interval=tf_config.interval,
                weight=tf_config.weight,
                strategy_evaluator=strategy_evaluator
            )
            units.append(unit)

        aggregator = SignalAggregator(self.config.multi_timeframe.conflict_resolution)
        engine = MultiTimeframeEngine(symbol, units, aggregator)
        self.mtf_engines[symbol] = engine

        logger.info(f"MTF engine setup for {symbol}: {[u.interval for u in units]}")

    async def run(self) -> None:
        """
        Main orchestration loop.

        Runs until shutdown() is called.
        """
        logger.info("Orchestrator starting...")
        await self._notify("Trading orchestrator starting up")

        # Initial session check
        session_event = self.session_manager.check_session_transition()
        if session_event:
            await self._handle_session_event(session_event)

        while not self._shutdown:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Error in orchestration tick: {e}", exc_info=True)
                await self._notify(f"Orchestration error: {e}")
                # Continue after error - fail-safe, not fail-stop for non-critical errors

            await self._wait_for_next_tick()

        logger.info("Orchestrator shutdown complete")
        await self._notify("Trading orchestrator shut down")

    async def _tick(self) -> List[TickResult]:
        """
        Single orchestration tick - evaluate all symbols.

        Returns:
            List of TickResult for each symbol.
        """
        self._tick_count += 1
        now = datetime.utcnow()
        results: List[TickResult] = []

        # Check for session transition
        session_event = self.session_manager.check_session_transition()
        if session_event:
            await self._handle_session_event(session_event)

        # Get session context
        liquidity_metrics = self.liquidity_monitor.get_all_metrics()
        session_context = self.session_manager.get_session_context(liquidity_metrics)

        # Check if trading should be paused
        should_pause, pause_reason = self.session_manager.should_pause_trading()
        if should_pause:
            logger.info(f"Trading paused: {pause_reason}")
            for symbol in self.config.symbols:
                result = TickResult(
                    symbol=symbol,
                    timestamp=now,
                    session_context=session_context,
                    mtf_signal=None,
                    portfolio_check=None,
                    decision=self._create_paused_decision(symbol, session_context, pause_reason),
                    action_taken="paused",
                    details={"reason": pause_reason}
                )
                results.append(result)
            return results

        # Evaluate each symbol
        for symbol in self.config.symbols:
            try:
                result = await self._evaluate_symbol(symbol, session_context)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}", exc_info=True)
                result = TickResult(
                    symbol=symbol,
                    timestamp=now,
                    session_context=session_context,
                    mtf_signal=None,
                    portfolio_check=None,
                    decision=None,
                    action_taken="error",
                    details={"error": str(e)}
                )
                results.append(result)

        return results

    async def _evaluate_symbol(
        self,
        symbol: str,
        session_context: SessionContext
    ) -> TickResult:
        """
        Evaluate a single symbol.

        Flow:
        1. Check portfolio-level limits
        2. Fetch data for all timeframes
        3. Run multi-timeframe engine
        4. Execute trade if approved
        """
        now = datetime.utcnow()

        # Step 1: Portfolio-level pre-check
        # Use a placeholder value for the check (actual trade size would come from strategy)
        proposed_value = self.config.total_capital * Decimal("0.1")  # 10% position
        portfolio_result = self.portfolio_risk.can_open_position(symbol, proposed_value)

        if not portfolio_result.approved:
            logger.info(f"{symbol}: Blocked by portfolio risk - {portfolio_result.rejection_reason}")
            decision = self._create_blocked_decision(
                symbol, session_context, portfolio_result
            )
            return TickResult(
                symbol=symbol,
                timestamp=now,
                session_context=session_context,
                mtf_signal=None,
                portfolio_check=portfolio_result,
                decision=decision,
                action_taken="blocked",
                details={"reason": portfolio_result.rejection_reason}
            )

        # Step 2: Fetch data for all timeframes
        mtf_engine = self.mtf_engines.get(symbol)
        if mtf_engine is None:
            logger.warning(f"No MTF engine for {symbol}")
            return TickResult(
                symbol=symbol,
                timestamp=now,
                session_context=session_context,
                mtf_signal=None,
                portfolio_check=portfolio_result,
                decision=None,
                action_taken="error",
                details={"error": "No MTF engine configured"}
            )

        data_by_tf: Dict[str, pd.DataFrame] = {}
        for interval in mtf_engine.units.keys():
            try:
                df = self.data_fetcher(symbol, interval)
                data_by_tf[interval] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} {interval} data: {e}")

        if not data_by_tf:
            return TickResult(
                symbol=symbol,
                timestamp=now,
                session_context=session_context,
                mtf_signal=None,
                portfolio_check=portfolio_result,
                decision=None,
                action_taken="error",
                details={"error": "No data available"}
            )

        # Step 3: Run multi-timeframe engine
        mtf_signal = mtf_engine.evaluate(data_by_tf)

        # Determine action based on signal
        if mtf_signal.signal == SignalType.HOLD:
            action = "hold"
            decision = self._create_hold_decision(symbol, session_context, mtf_signal, portfolio_result)
        elif mtf_signal.signal in (SignalType.LONG, SignalType.SHORT):
            if not mtf_signal.is_aligned:
                # Signals conflict - wait
                action = "conflict"
                decision = self._create_conflict_decision(symbol, session_context, mtf_signal, portfolio_result)
            else:
                # Execute trade
                action = "trade"
                decision = self._create_trade_decision(symbol, session_context, mtf_signal, portfolio_result)
                if self.trade_executor:
                    side = "buy" if mtf_signal.signal == SignalType.LONG else "sell"
                    # Calculate position size with session multiplier
                    quantity = self._calculate_position_size(
                        symbol,
                        session_context.position_size_multiplier
                    )
                    await asyncio.to_thread(self.trade_executor, symbol, side, quantity)
        elif mtf_signal.signal == SignalType.CLOSE:
            action = "close"
            decision = self._create_close_decision(symbol, session_context, mtf_signal, portfolio_result)
            if self.trade_executor:
                await asyncio.to_thread(self.trade_executor, symbol, "close", Decimal("0"))
        else:
            action = "unknown"
            decision = None

        return TickResult(
            symbol=symbol,
            timestamp=now,
            session_context=session_context,
            mtf_signal=mtf_signal,
            portfolio_check=portfolio_result,
            decision=decision,
            action_taken=action
        )

    async def _handle_session_event(self, event: SessionEvent) -> None:
        """Handle a session transition event."""
        self._last_session_event = event

        msg = f"Session: {event.previous_session.value if event.previous_session else 'None'} -> {event.session.value}"

        if event.session == MarketSession.DEAD_ZONE:
            await self._notify(f"Entering dead zone - trading paused until 00:00 UTC")
        elif event.previous_session == MarketSession.DEAD_ZONE:
            await self._notify(f"Exiting dead zone - trading resumed")
        else:
            logger.info(msg)

    async def _wait_for_next_tick(self) -> None:
        """Wait for the next tick interval."""
        if self.session_manager.is_dead_zone():
            # Longer interval during dead zone
            interval = self.config.dead_zone_check_interval_seconds
        else:
            interval = self.config.tick_interval_seconds

        await asyncio.sleep(interval)

    async def _notify(self, message: str) -> None:
        """Send notification if notifier is configured."""
        if self.notifier:
            try:
                await asyncio.to_thread(self.notifier, message)
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    def _calculate_position_size(
        self,
        symbol: str,
        session_multiplier: float
    ) -> Decimal:
        """Calculate position size with session adjustment."""
        base_size = self.config.total_capital * Decimal("0.1")  # 10% base
        adjusted = base_size * Decimal(str(session_multiplier))

        # Get liquidity adjustment
        liquidity_adj = self.liquidity_monitor.get_position_adjustment(symbol)
        final_size = adjusted * liquidity_adj

        return final_size

    def _create_paused_decision(
        self,
        symbol: str,
        session_context: SessionContext,
        reason: str
    ) -> Decision:
        """Create a decision record for paused trading."""
        return Decision(
            decision_id=Decision.generate_id(),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            strategy_name="orchestrator",
            signal_values={},
            risk_checks={},
            result=DecisionResult.SESSION_PAUSED,
            result_reason=reason,
            session_context=session_context.to_dict()
        )

    def _create_blocked_decision(
        self,
        symbol: str,
        session_context: SessionContext,
        portfolio_result: PortfolioRiskResult
    ) -> Decision:
        """Create a decision record for portfolio-blocked trade."""
        return Decision(
            decision_id=Decision.generate_id(),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            strategy_name="orchestrator",
            signal_values={},
            risk_checks={},
            result=DecisionResult.PORTFOLIO_BLOCKED,
            result_reason=portfolio_result.rejection_reason or "Portfolio limit exceeded",
            session_context=session_context.to_dict(),
            portfolio_risk=portfolio_result.to_dict()
        )

    def _create_hold_decision(
        self,
        symbol: str,
        session_context: SessionContext,
        mtf_signal: AggregatedSignal,
        portfolio_result: PortfolioRiskResult
    ) -> Decision:
        """Create a decision record for hold signal."""
        return Decision(
            decision_id=Decision.generate_id(),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            strategy_name="orchestrator",
            signal_values={},
            risk_checks={},
            result=DecisionResult.SIGNAL_HOLD,
            result_reason="Multi-timeframe signal: HOLD",
            session_context=session_context.to_dict(),
            mtf_aggregation=mtf_signal.to_dict(),
            portfolio_risk=portfolio_result.to_dict()
        )

    def _create_conflict_decision(
        self,
        symbol: str,
        session_context: SessionContext,
        mtf_signal: AggregatedSignal,
        portfolio_result: PortfolioRiskResult
    ) -> Decision:
        """Create a decision record for conflicting timeframe signals."""
        return Decision(
            decision_id=Decision.generate_id(),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            strategy_name="orchestrator",
            signal_values={},
            risk_checks={},
            result=DecisionResult.MTF_CONFLICT,
            result_reason="Multi-timeframe signals not aligned - waiting",
            session_context=session_context.to_dict(),
            mtf_aggregation=mtf_signal.to_dict(),
            portfolio_risk=portfolio_result.to_dict()
        )

    def _create_trade_decision(
        self,
        symbol: str,
        session_context: SessionContext,
        mtf_signal: AggregatedSignal,
        portfolio_result: PortfolioRiskResult
    ) -> Decision:
        """Create a decision record for trade execution."""
        result = DecisionResult.SIGNAL_LONG if mtf_signal.signal == SignalType.LONG else DecisionResult.SIGNAL_SHORT
        return Decision(
            decision_id=Decision.generate_id(),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            strategy_name="orchestrator",
            signal_values={},
            risk_checks={},
            result=result,
            result_reason=f"Multi-timeframe aligned: {mtf_signal.signal.value}",
            session_context=session_context.to_dict(),
            mtf_aggregation=mtf_signal.to_dict(),
            portfolio_risk=portfolio_result.to_dict()
        )

    def _create_close_decision(
        self,
        symbol: str,
        session_context: SessionContext,
        mtf_signal: AggregatedSignal,
        portfolio_result: PortfolioRiskResult
    ) -> Decision:
        """Create a decision record for position close."""
        return Decision(
            decision_id=Decision.generate_id(),
            timestamp=datetime.utcnow(),
            symbol=symbol,
            strategy_name="orchestrator",
            signal_values={},
            risk_checks={},
            result=DecisionResult.SIGNAL_CLOSE,
            result_reason="Multi-timeframe signal: CLOSE",
            session_context=session_context.to_dict(),
            mtf_aggregation=mtf_signal.to_dict(),
            portfolio_risk=portfolio_result.to_dict()
        )

    def shutdown(self) -> None:
        """Signal graceful shutdown."""
        logger.info("Shutdown requested")
        self._shutdown = True

    def pause(self) -> None:
        """Pause trading (can resume)."""
        self._paused = True
        self.session_manager.set_session_state(SessionState.PAUSED)

    def resume(self) -> None:
        """Resume trading."""
        self._paused = False
        self.session_manager.set_session_state(SessionState.ACTIVE)

    def halt(self, reason: str) -> None:
        """Emergency halt (requires manual resume)."""
        self._paused = True
        self.session_manager.set_session_state(SessionState.HALTED)
        self.portfolio_risk.halt_trading(reason)
        logger.warning(f"EMERGENCY HALT: {reason}")

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        session_info = self.session_manager.get_session_info()
        portfolio_state = self.portfolio_risk.get_portfolio_state()

        return {
            "running": not self._shutdown,
            "paused": self._paused,
            "tick_count": self._tick_count,
            "symbols": self.config.symbols,
            "session": session_info,
            "portfolio": portfolio_state,
            "mtf_engines": {
                symbol: engine.get_status()
                for symbol, engine in self.mtf_engines.items()
            },
            "liquidity": self.liquidity_monitor.get_summary()
        }
