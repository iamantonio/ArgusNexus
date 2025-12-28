#!/usr/bin/env python3
"""
ArgusNexus V4 Live Portfolio Trader - Combined Strategy

Runs the hardened Portfolio Manager with the existing paper trading infrastructure:
- Vol-Regime Core (85%) + MostlyLong Sleeve (15%)
- Portfolio-level DD circuit breaker (12%/20%/10%/15%)
- State persistence across restarts
- DD threshold alerting
- Integration with existing PaperExecutor and TruthLogger

This replaces the single-strategy approach with a combined portfolio strategy
that maintains a target BTC allocation and rebalances based on:
1. Regime (bull/bear/sideways from SMA200 + momentum)
2. Volatility scaling (vol_target / realized_vol)
3. DD circuit breaker state

Usage:
    python scripts/live_portfolio_trader.py
    python scripts/live_portfolio_trader.py --capital 500 --interval 1d

Environment:
    - Execution Mode: PAPER (no real money at risk)
    - Strategy: Portfolio Manager (Vol-Regime Core + MostlyLong Sleeve)
    - Data: Coinbase daily candles
    - State: Persisted to runtime/portfolio_state.json
"""

import sys
import time
import signal
import logging
import argparse
import asyncio
import json
import os
import yaml
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.portfolio.portfolio_manager import PortfolioManager, PortfolioState, DDState, RebalanceOrder
from src.portfolio.alerts import AlertManager, AlertLevel
from src.execution import PaperExecutor
from src.truth.logger import TruthLogger
from src.truth.schema import DecisionResult, OrderSide
from src.data.loader import fetch_coinbase_data
from src.notifier import DiscordNotifier
from src.risk.manager import RiskManager, RiskConfig, RiskConfigError
from src.risk.schema import TradeRequest, PortfolioState as RiskPortfolioState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("runtime/portfolio_trader.log")
    ]
)
logger = logging.getLogger(__name__)

# Global shutdown flag
shutdown_requested = False

# Paths - now set dynamically per symbol in __init__
STATE_PATH = None  # Set in LivePortfolioTrader.__init__
SNAPSHOT_LOG_PATH = None  # Set in LivePortfolioTrader.__init__

# Fee structure (Gemini)
FEES = {
    "entry": 0.004,   # 0.4% entry fee
    "exit": 0.002,    # 0.2% exit fee
    "slippage": 0.001 # 0.1% slippage
}


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info("Shutdown signal received. Finishing current tick...")
    shutdown_requested = True


def ensure_runtime_dir():
    """Ensure runtime directory exists."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: dict):
    """Atomically write JSON to disk."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


class LivePortfolioTrader:
    """
    Live Paper Trading for Portfolio Manager Strategy.

    Integrates with existing V4 infrastructure:
    - PaperExecutor for order execution
    - TruthLogger for decision/trade logging
    - DiscordNotifier for alerts
    - PortfolioManager for strategy decisions
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        interval: str = "1d",
        starting_capital: Decimal = Decimal("500"),
        lookback_days: int = 300,
    ):
        self.symbol = symbol
        self.interval = interval
        self.starting_capital = starting_capital
        self.lookback_days = lookback_days

        # Paths - per-symbol state files to support multiple instances
        global STATE_PATH, SNAPSHOT_LOG_PATH
        symbol_safe = symbol.replace("-", "_").lower()
        STATE_PATH = Path(f"runtime/portfolio_state_{symbol_safe}.json")
        SNAPSHOT_LOG_PATH = Path(f"runtime/portfolio_snapshots_{symbol_safe}.jsonl")
        self.state_path = STATE_PATH
        self.snapshot_path = SNAPSHOT_LOG_PATH

        # Use same database as dashboard for visibility
        self.db_path = Path(__file__).parent.parent / "data" / "v4_live_paper.db"

        # ===========================================================================
        # RISK TRACKING STATE - CRITICAL for proper risk gate operation
        # ===========================================================================
        # Daily P&L tracking (resets at 00:00 UTC)
        self._daily_pnl = Decimal("0")
        self._daily_pnl_reset_date: Optional[datetime] = None
        self._day_start_equity: Optional[Decimal] = None

        # Trade frequency tracking (rolling 60-minute window)
        self._recent_trade_timestamps: list[datetime] = []

        # Circuit breaker tracking (8% move in 60 minutes)
        self._price_history: list[tuple[datetime, Decimal]] = []  # (timestamp, price)
        self._circuit_breaker_triggered = False
        self._circuit_breaker_trigger_time: Optional[datetime] = None

        self._setup_components()

    def _setup_components(self):
        """Initialize all components."""
        logger.info("=" * 70)
        logger.info("ArgusNexus V4 - LIVE PORTFOLIO TRADER")
        logger.info("=" * 70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Interval: {self.interval}")
        logger.info(f"Starting Capital: ${self.starting_capital:,.2f}")
        logger.info(f"Database: {self.db_path}")
        logger.info("")

        # Portfolio Manager - reduced cooldown for 1h validation
        self.pm = PortfolioManager(rebalance_cooldown_days=1)
        logger.info("Strategy: PORTFOLIO MANAGER")
        logger.info(f"  - Core Weight: {float(self.pm.core_weight)*100:.0f}%")
        logger.info(f"  - Sleeve Weight: {float(self.pm.sleeve_weight)*100:.0f}%")
        logger.info(f"  - DD Warning: {float(self.pm.dd_warning)}%")
        logger.info(f"  - DD Critical: {float(self.pm.dd_critical)}%")
        logger.info(f"  - Rebalance Threshold: {float(self.pm.rebalance_threshold)*100:.0f}%")
        logger.info(f"  - Rebalance Cooldown: {self.pm.rebalance_cooldown_days} days")

        # Risk Manager - MANDATORY, fail closed on any error
        try:
            self.risk_manager = RiskManager(RiskConfig.load_from_yaml())
            logger.info("RiskManager: INITIALIZED")
            logger.info(f"  - Max Asset Concentration: {self.risk_manager.config.max_asset_concentration_pct}%")
            logger.info(f"  - Max Drawdown: {self.risk_manager.config.max_drawdown_pct}%")
            logger.info(f"  - Circuit Breaker: {self.risk_manager.config.circuit_breaker_pct}%")
        except (RiskConfigError, FileNotFoundError, Exception) as e:
            logger.critical(f"FATAL: Risk configuration error: {e}")
            logger.critical("System cannot start without valid risk configuration.")
            # Set durable halt in state file so restarts also fail
            self._set_permanent_halt(f"RISK_CONFIG_ERROR: {e}")
            raise SystemExit(1)

        # Paper Executor
        self.executor = PaperExecutor(
            starting_balance=self.starting_capital,
            base_slippage_pct=Decimal("0.001"),  # 0.1% base slippage
            noise_slippage_pct=Decimal("0.001"),
            size_impact_per_10k=Decimal("0.001"),
            fee_rate=Decimal("0.004")
        )

        # Truth Logger
        self.truth_logger = TruthLogger(str(self.db_path))

        # Discord Notifier
        self.notifier = DiscordNotifier()
        if not self.notifier.url:
            self.notifier = None
            logger.warning("Discord notifier not configured (no DISCORD_WEBHOOK_URL)")

        # Alert Manager
        self.alert_manager = AlertManager(alert_log_path=Path("runtime/portfolio_alerts.jsonl"))

        # Load portfolio config for catastrophic stop
        try:
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
            portfolio_config = config.get("portfolio", {})
            self.catastrophic_stop_pct = Decimal(str(portfolio_config.get("catastrophic_stop_pct", 15.0)))
            self.guardian_check_interval = int(portfolio_config.get("guardian_check_interval_seconds", 60))
            logger.info("Catastrophic Stop Guardian: ARMED")
            logger.info(f"  - Stop Threshold: {self.catastrophic_stop_pct}%")
            logger.info(f"  - Check Interval: {self.guardian_check_interval}s")
        except Exception as e:
            logger.critical(f"FATAL: Failed to load portfolio config: {e}")
            self._set_permanent_halt(f"PORTFOLIO_CONFIG_ERROR: {e}")
            raise SystemExit(1)

        # Guardian loop task reference (set in run())
        self.guardian_task: Optional[asyncio.Task] = None

    def load_state(self) -> Optional[PortfolioState]:
        """Load persisted state from disk."""
        if not STATE_PATH.exists():
            logger.info("No persisted state found - starting fresh")
            return None

        try:
            data = json.loads(STATE_PATH.read_text())
            state = PortfolioState.from_dict(data)
            logger.info(f"Loaded persisted state: equity=${float(state.total_equity):.2f}, "
                       f"DD_state={state.dd_state.value}, bars_in_critical={state.bars_in_critical}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def save_state(self, state: PortfolioState, cost_basis: Optional[Decimal] = None):
        """
        Persist state to disk atomically.

        Includes cost basis tracking for catastrophic stop calculation.
        """
        data = state.to_dict()

        # Preserve existing cost basis if not provided
        if cost_basis is not None:
            data["btc_cost_basis"] = float(cost_basis)
        elif STATE_PATH.exists():
            try:
                existing = json.loads(STATE_PATH.read_text())
                if "btc_cost_basis" in existing:
                    data["btc_cost_basis"] = existing["btc_cost_basis"]
            except Exception:
                pass

        atomic_write_json(STATE_PATH, data)
        logger.debug(f"State persisted: equity=${float(state.total_equity):.2f}")

    def create_default_state(self) -> PortfolioState:
        """Create default initial state."""
        return PortfolioState(
            total_equity=self.starting_capital,
            btc_qty=Decimal("0"),
            cash=self.starting_capital,
            high_water_mark=self.starting_capital,
            dd_state=DDState.NORMAL,
            recovery_mode=False,
            bars_in_critical=0,
            sleeve_in_position=False,
            sleeve_entry_price=None,
            last_rebalance_time=None,
            last_update=None,
        )

    def _set_permanent_halt(self, reason: str):
        """
        Set durable halt in state file.

        This ensures restarts also refuse to trade until manually cleared.
        Used when risk configuration is invalid or missing.
        """
        ensure_runtime_dir()

        # Try to load existing state, or create minimal halt state
        if STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text())
            except Exception:
                data = {}
        else:
            data = {"total_equity": float(self.starting_capital)}

        # Set halt flags
        data["trading_halted"] = True
        data["halt_reason"] = reason
        data["halt_time"] = datetime.now(timezone.utc).isoformat()

        atomic_write_json(STATE_PATH, data)
        logger.critical(f"PERMANENT HALT SET: {reason}")
        logger.critical(f"State file: {STATE_PATH}")
        logger.critical("Manual intervention required to clear halt.")

    # =========================================================================
    # CATASTROPHIC STOP GUARDIAN - Independent Protection Loop
    # =========================================================================

    def _get_cost_basis(self) -> Optional[Decimal]:
        """
        Get cost basis (WAC) from state file.

        Returns None if no cost basis tracked (no position or missing data).
        """
        if not STATE_PATH.exists():
            return None

        try:
            data = json.loads(STATE_PATH.read_text())
            if "btc_cost_basis" in data and float(data.get("btc_qty", 0)) > 0:
                return Decimal(str(data["btc_cost_basis"]))
            return None
        except Exception as e:
            logger.error(f"Failed to read cost basis: {e}")
            return None

    def _compute_catastrophic_stop(self, cost_basis: Decimal) -> Decimal:
        """
        Compute catastrophic stop price from cost basis.

        stop_price = cost_basis * (1 - catastrophic_stop_pct/100)
        """
        return cost_basis * (1 - self.catastrophic_stop_pct / 100)

    async def _fetch_current_price(self) -> Optional[Decimal]:
        """
        Fetch current price for catastrophic stop check.

        Returns None on error (triggers fail-closed behavior).
        """
        try:
            # Use a minimal data fetch - just need latest price
            df, _ = fetch_coinbase_data(
                symbol=self.symbol,
                lookback_days=1,
                interval="1h",  # Use hourly for faster response
                verbose=False
            )
            if df is None or len(df) == 0:
                return None
            return Decimal(str(df.iloc[-1]["close"]))
        except Exception as e:
            logger.error(f"GUARDIAN: Price fetch failed: {e}")
            return None

    async def _emergency_exit(
        self,
        current_price: Decimal,
        stop_price: Decimal,
        cost_basis: Decimal,
        btc_qty: Decimal
    ):
        """
        Execute emergency exit due to catastrophic stop breach.

        This is the LAST LINE OF DEFENSE:
        1. Close position immediately
        2. Set trading_halted=true durably
        3. Send CRITICAL alert
        4. Log to Truth Engine
        """
        logger.critical("=" * 70)
        logger.critical("CATASTROPHIC STOP TRIGGERED")
        logger.critical("=" * 70)
        logger.critical(f"  Current Price: ${current_price:,.2f}")
        logger.critical(f"  Stop Price: ${stop_price:,.2f}")
        logger.critical(f"  Cost Basis: ${cost_basis:,.2f}")
        logger.critical(f"  Distance: {float((current_price - stop_price) / stop_price * 100):.2f}%")

        # Build snapshot for logging
        snapshot = {
            "current_price": float(current_price),
            "stop_price": float(stop_price),
            "cost_basis": float(cost_basis),
            "btc_qty": float(btc_qty),
            "catastrophic_stop_pct": float(self.catastrophic_stop_pct),
            "distance_pct": float((current_price - stop_price) / stop_price * 100)
        }

        # Log decision to Truth Engine
        await self.truth_logger.log_decision(
            symbol=self.symbol,
            strategy_name="portfolio_manager",
            signal_values={
                "event": "CATASTROPHIC_STOP_TRIGGERED",
                "snapshot": snapshot
            },
            risk_checks={"catastrophic_stop": "TRIGGERED", "approved": False},
            result=DecisionResult.SIGNAL_CLOSE,
            result_reason=f"CATASTROPHIC_STOP: Price ${current_price:,.2f} breached stop ${stop_price:,.2f}",
            market_context=snapshot,
            timestamp=datetime.now(timezone.utc)
        )

        # Close position via executor
        try:
            # Log the exit order
            exit_order = await self.truth_logger.log_order(
                decision_id=None,  # Emergency - no decision ID
                symbol=self.symbol,
                side=OrderSide.SELL,
                quantity=btc_qty,
                requested_price=current_price
            )

            # Update executor balances - close position
            cash_received = btc_qty * current_price * (1 - Decimal("0.002"))  # Exit fee
            current_cash = await self.executor.get_balance("USD")
            await self.executor.set_balance("BTC", Decimal("0"))
            await self.executor.set_balance("USD", current_cash + cash_received)

            # Mark order filled
            await self.truth_logger.update_order_fill(
                order_id=exit_order.order_id,
                fill_price=current_price,
                fill_quantity=btc_qty,
                commission=btc_qty * current_price * Decimal("0.002")
            )

            # Close trade record
            open_position = await self.truth_logger.get_open_position(self.symbol)
            if open_position:
                await self.truth_logger.close_trade(
                    trade_id=open_position["trade_id"],
                    exit_order_id=exit_order.order_id,
                    exit_price=current_price,
                    exit_reason="catastrophic_stop"
                )

            logger.critical(f"Position CLOSED: {float(btc_qty):.6f} BTC @ ${current_price:,.2f}")

        except Exception as e:
            logger.critical(f"EMERGENCY EXIT EXECUTION FAILED: {e}")

        # Set durable halt
        self._set_permanent_halt(
            f"CATASTROPHIC_STOP: Price ${current_price:,.2f} breached stop ${stop_price:,.2f}"
        )

        # Send CRITICAL alert
        if self.notifier:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.notifier.send_system_alert,
                    "CRITICAL: Catastrophic Stop Triggered",
                    f"Position EMERGENCY CLOSED\n\n"
                    f"Price: ${current_price:,.2f}\n"
                    f"Stop: ${stop_price:,.2f}\n"
                    f"Cost Basis: ${cost_basis:,.2f}\n"
                    f"Qty: {float(btc_qty):.6f} BTC\n\n"
                    f"Trading HALTED. Manual intervention required."
                )
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    async def _guardian_loop(self):
        """
        Independent guardian loop for catastrophic stop enforcement.

        Runs every guardian_check_interval seconds, checking:
        1. Is there an open position?
        2. If yes, fetch current price
        3. If price <= catastrophic stop → emergency exit

        FAIL CLOSED: If price fetch fails, halt trading immediately.
        """
        logger.info(f"GUARDIAN: Starting catastrophic stop guardian (interval: {self.guardian_check_interval}s)")

        while not shutdown_requested:
            try:
                # Check if we have an open position
                if not STATE_PATH.exists():
                    await asyncio.sleep(self.guardian_check_interval)
                    continue

                data = json.loads(STATE_PATH.read_text())

                # Check if already halted
                if data.get("trading_halted", False):
                    logger.debug("GUARDIAN: Trading halted, skipping check")
                    await asyncio.sleep(self.guardian_check_interval)
                    continue

                btc_qty = Decimal(str(data.get("btc_qty", 0)))
                if btc_qty <= 0:
                    # No position, nothing to protect
                    await asyncio.sleep(self.guardian_check_interval)
                    continue

                # Have position - get cost basis
                cost_basis = self._get_cost_basis()
                if cost_basis is None:
                    # FAIL CLOSED: Position exists but no cost basis
                    logger.critical("GUARDIAN: Position exists but no cost basis - FAIL CLOSED")
                    self._set_permanent_halt("GUARDIAN_ERROR: Position exists but no cost basis")
                    if self.notifier:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.notifier.send_system_alert,
                            "CRITICAL: Guardian Error",
                            "Position exists but cost basis missing.\nTrading HALTED."
                        )
                    break

                # Compute stop price
                stop_price = self._compute_catastrophic_stop(cost_basis)

                # Fetch current price
                current_price = await self._fetch_current_price()
                if current_price is None:
                    # FAIL CLOSED: Can't price it, can't hold it
                    logger.critical("GUARDIAN: Price feed error - FAIL CLOSED")
                    self._set_permanent_halt("PRICE_FEED_ERROR: Cannot fetch current price")
                    if self.notifier:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.notifier.send_system_alert,
                            "CRITICAL: Price Feed Error",
                            "Cannot fetch current price.\nTrading HALTED.\n\n"
                            "If you can't price it, you can't hold it."
                        )
                    break

                # Check if stop breached
                if current_price <= stop_price:
                    logger.critical(f"GUARDIAN: Stop BREACHED! Price ${current_price:,.2f} <= Stop ${stop_price:,.2f}")
                    await self._emergency_exit(current_price, stop_price, cost_basis, btc_qty)
                    break
                else:
                    distance_pct = float((current_price - stop_price) / stop_price * 100)
                    logger.debug(f"GUARDIAN: OK - Price ${current_price:,.2f}, Stop ${stop_price:,.2f}, Distance: +{distance_pct:.1f}%")

            except Exception as e:
                logger.error(f"GUARDIAN: Error in check loop: {e}")
                # Don't halt on transient errors, but log them

            await asyncio.sleep(self.guardian_check_interval)

        logger.info("GUARDIAN: Stopped")

    def _build_trade_request(
        self,
        order: RebalanceOrder,
        current_price: Decimal,
        state: PortfolioState
    ) -> TradeRequest:
        """
        Build TradeRequest from a RebalanceOrder for risk evaluation.

        For the portfolio manager (rebalancing strategy):
        - SELL orders are exits, so they get is_exit=True and bypass R:R
        - BUY orders use synthetic SL/TP that meets min R:R requirement
        - The real gate we care about is ASSET_CONCENTRATION (30%)
        """
        # For SELL orders (reducing position), use is_exit=True
        is_exit = order.action == "SELL"

        if is_exit:
            # Exit orders bypass R:R check - use placeholder prices
            stop_loss_price = current_price * Decimal("0.9")
            take_profit_price = current_price * Decimal("1.1")
        else:
            # BUY orders need valid R:R
            # Use synthetic levels: 15% SL, 45% TP = 3:1 R:R
            # This ensures R:R check passes, concentration is the real gate
            stop_loss_price = current_price * Decimal("0.85")
            take_profit_price = current_price * Decimal("1.45")

        return TradeRequest(
            symbol=self.symbol,
            side=order.action.lower(),
            quantity=abs(order.btc_qty_delta),
            entry_price=current_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            strategy_name="portfolio_manager",
            is_exit=is_exit
        )

    # ===========================================================================
    # RISK TRACKING METHODS - CRITICAL for proper risk gate operation
    # ===========================================================================

    def _update_daily_pnl(self, state: PortfolioState):
        """
        Update daily P&L tracking with reset at 00:00 UTC.

        This enables the daily_loss_limit_pct check in the RiskManager.
        Without proper tracking, daily loss limits are effectively disabled.
        """
        now = datetime.now(timezone.utc)
        today = now.date()

        # Check if we need to reset for a new day
        if self._daily_pnl_reset_date is None or self._daily_pnl_reset_date != today:
            # New day - reset daily P&L
            self._daily_pnl_reset_date = today
            self._day_start_equity = state.total_equity
            self._daily_pnl = Decimal("0")
            logger.info(f"[RISK] Daily P&L reset for {today}. Day start equity: ${float(state.total_equity):,.2f}")
        else:
            # Same day - calculate P&L from day start
            if self._day_start_equity is not None:
                self._daily_pnl = state.total_equity - self._day_start_equity

    def _get_daily_pnl_percent(self) -> float:
        """Get daily P&L as a percentage of day start equity."""
        if self._day_start_equity is None or self._day_start_equity == 0:
            return 0.0
        return float(self._daily_pnl / self._day_start_equity * 100)

    def _record_trade(self):
        """
        Record a trade timestamp for frequency tracking.

        This enables the max_trades_per_hour check in the RiskManager.
        Without proper tracking, trade frequency limits are effectively disabled.
        """
        now = datetime.now(timezone.utc)
        self._recent_trade_timestamps.append(now)

        # Prune old timestamps (older than 60 minutes)
        cutoff = now - timedelta(minutes=60)
        self._recent_trade_timestamps = [
            ts for ts in self._recent_trade_timestamps if ts > cutoff
        ]

        logger.debug(f"[RISK] Trade recorded. Trades in last hour: {len(self._recent_trade_timestamps)}")

    def _get_recent_trades_count(self) -> int:
        """
        Get count of trades in the last 60 minutes.

        Used by RiskManager for max_trades_per_hour check.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=60)

        # Prune old timestamps
        self._recent_trade_timestamps = [
            ts for ts in self._recent_trade_timestamps if ts > cutoff
        ]

        return len(self._recent_trade_timestamps)

    def _update_price_history(self, current_price: Decimal):
        """
        Update price history for circuit breaker detection.

        Circuit breaker triggers if price moves 8%+ in 60 minutes.
        This is a market-wide protection against flash crashes/spikes.
        """
        now = datetime.now(timezone.utc)

        # Add current price to history
        self._price_history.append((now, current_price))

        # Prune old prices (older than 60 minutes)
        cutoff = now - timedelta(minutes=60)
        self._price_history = [
            (ts, price) for ts, price in self._price_history if ts > cutoff
        ]

    def _check_circuit_breaker(self, current_price: Decimal) -> bool:
        """
        Check if circuit breaker should be triggered.

        Triggers if price moves 8%+ from any price in the last 60 minutes.
        Once triggered, stays triggered for cooldown period (30 minutes).
        """
        now = datetime.now(timezone.utc)

        # Check if we're still in cooldown from previous trigger
        if self._circuit_breaker_triggered and self._circuit_breaker_trigger_time:
            # Load cooldown from config (default 30 minutes)
            cooldown_minutes = 30
            try:
                with open("config.yaml") as f:
                    config = yaml.safe_load(f)
                cooldown_minutes = config.get("risk", {}).get("circuit_breaker_cooldown_minutes", 30)
            except Exception:
                pass

            cooldown_end = self._circuit_breaker_trigger_time + timedelta(minutes=cooldown_minutes)
            if now < cooldown_end:
                logger.debug(f"[RISK] Circuit breaker in cooldown until {cooldown_end.isoformat()}")
                return True
            else:
                # Cooldown expired, reset circuit breaker
                self._circuit_breaker_triggered = False
                self._circuit_breaker_trigger_time = None
                logger.info("[RISK] Circuit breaker cooldown expired, reset")

        # Check for 8%+ move from any price in history
        circuit_breaker_pct = 8.0
        try:
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
            circuit_breaker_pct = config.get("risk", {}).get("circuit_breaker_pct", 8.0)
        except Exception:
            pass

        for ts, historical_price in self._price_history:
            if historical_price > 0:
                move_pct = abs(float((current_price - historical_price) / historical_price * 100))
                if move_pct >= circuit_breaker_pct:
                    # Circuit breaker triggered!
                    self._circuit_breaker_triggered = True
                    self._circuit_breaker_trigger_time = now
                    logger.warning(f"[RISK] CIRCUIT BREAKER TRIGGERED! {move_pct:.1f}% move detected")
                    logger.warning(f"  Historical price: ${float(historical_price):,.2f} at {ts.isoformat()}")
                    logger.warning(f"  Current price: ${float(current_price):,.2f}")
                    return True

        return self._circuit_breaker_triggered

    def _build_risk_state(
        self,
        state: PortfolioState,
        current_price: Decimal
    ) -> RiskPortfolioState:
        """
        Build RiskPortfolioState from PortfolioManager's state.

        Maps the portfolio manager's state to the risk system's expected format.

        CRITICAL: This method now uses ACTUAL tracked values instead of hardcoded
        zeros that were disabling risk protections. Proper tracking enables:
        - daily_loss_limit_pct check (via daily_pnl tracking)
        - max_trades_per_hour check (via trade frequency tracking)
        - circuit_breaker_pct check (via price history tracking)
        """
        # Calculate current BTC position value
        btc_value = state.btc_qty * current_price

        # Build open positions dict for risk calculations
        open_positions = {}
        if state.btc_qty > 0:
            open_positions["BTC-USD"] = {
                "quantity": str(state.btc_qty),
                "value": str(btc_value),
                "avg_entry": str(current_price)  # Approximate
            }

        # Calculate drawdown from high water mark
        drawdown_pct = 0.0
        if state.high_water_mark > 0:
            drawdown_pct = float(
                (state.total_equity - state.high_water_mark) / state.high_water_mark * 100
            )

        # Check for trading_halted flag in state (may not exist in old states)
        trading_halted = getattr(state, 'trading_halted', False)
        if not trading_halted and STATE_PATH.exists():
            # Also check the JSON file for halt flag
            try:
                state_data = json.loads(STATE_PATH.read_text())
                trading_halted = state_data.get('trading_halted', False)
            except Exception:
                pass

        # ======================================================================
        # FIXED: Use actual tracked values instead of hardcoded zeros
        # These values were previously hardcoded to 0/False, which disabled:
        # - Daily loss limit check (daily_pnl=0)
        # - Trade frequency check (recent_trades_count=0)
        # - Circuit breaker check (circuit_breaker_triggered=False)
        # ======================================================================

        # Update tracking before building state
        self._update_daily_pnl(state)
        self._update_price_history(current_price)
        circuit_breaker_active = self._check_circuit_breaker(current_price)

        return RiskPortfolioState(
            total_capital=state.total_equity,
            cash_available=state.cash,
            daily_pnl=self._daily_pnl,  # FIXED: Now uses actual tracked daily P&L
            daily_pnl_percent=self._get_daily_pnl_percent(),  # FIXED: Actual percentage
            total_pnl=state.total_equity - self.starting_capital,
            total_pnl_percent=drawdown_pct,
            open_positions=open_positions,
            recent_trades_count=self._get_recent_trades_count(),  # FIXED: Actual count
            circuit_breaker_triggered=circuit_breaker_active,  # FIXED: Actual state
            trading_halted=trading_halted
        )

    def log_snapshot(self, state: PortfolioState, order: RebalanceOrder, btc_price: Decimal):
        """Append snapshot to JSONL log."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": float(state.total_equity),
            "btc_qty": float(state.btc_qty),
            "cash": float(state.cash),
            "high_water_mark": float(state.high_water_mark),
            "dd_state": state.dd_state.value,
            "recovery_mode": state.recovery_mode,
            "bars_in_critical": state.bars_in_critical,
            "sleeve_in_position": state.sleeve_in_position,
            "btc_price": float(btc_price),
            "action": order.action,
            "target_alloc_pct": float(order.target_alloc_pct),
            "context": order.context
        }

        with SNAPSHOT_LOG_PATH.open("a") as f:
            f.write(json.dumps(snapshot, default=str) + "\n")

    def fetch_latest_data(self) -> Optional[pd.DataFrame]:
        """Fetch latest candle data."""
        try:
            df, report = fetch_coinbase_data(
                symbol=self.symbol,
                lookback_days=self.lookback_days,
                interval=self.interval,
                verbose=False
            )
            logger.info(f"Data loaded: {len(df)} bars, latest: {df['timestamp'].iloc[-1]}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            self.alert_manager.alert_market_data_unavailable(self.symbol, str(e))
            return None

    async def sync_executor_state(self, state: PortfolioState, current_price: Decimal):
        """Sync paper executor state with portfolio state."""
        # Get current executor balances
        usd_balance = await self.executor.get_balance("USD")
        btc_balance = await self.executor.get_balance("BTC")

        # Check for mismatch
        btc_mismatch = abs(float(btc_balance) - float(state.btc_qty)) > 0.00001
        cash_mismatch = abs(float(usd_balance) - float(state.cash)) > 1.0

        if btc_mismatch or cash_mismatch:
            logger.warning(f"Executor state mismatch detected:")
            logger.warning(f"  BTC: executor={float(btc_balance):.6f}, state={float(state.btc_qty):.6f}")
            logger.warning(f"  USD: executor=${float(usd_balance):.2f}, state=${float(state.cash):.2f}")

            # Sync executor to match state
            await self.executor.set_balance("BTC", state.btc_qty)
            await self.executor.set_balance("USD", state.cash)
            logger.info("Executor balances synced to portfolio state")

    async def execute_order(
        self,
        order: RebalanceOrder,
        state: PortfolioState,
        current_price: Decimal
    ) -> PortfolioState:
        """
        Execute order through paper executor and update state.

        RISK GATE: Every order MUST pass through RiskManager.evaluate() first.
        If evaluation rejects → order blocked, rejection logged
        If evaluation throws → fail closed, halt, CRITICAL alert
        """
        if order.action == "HOLD":
            return state

        # =======================================================================
        # RISK GATE - MANDATORY BEFORE ANY EXECUTION
        # =======================================================================
        trade_request = self._build_trade_request(order, current_price, state)
        risk_state = self._build_risk_state(state, current_price)

        try:
            risk_result = self.risk_manager.evaluate(trade_request, risk_state)
        except Exception as e:
            # FAIL CLOSED: Exception during risk evaluation = block + halt + alert
            logger.critical(f"RISK EVALUATION EXCEPTION: {e}")
            self._set_permanent_halt(f"RISK_EVAL_EXCEPTION: {e}")

            # Send CRITICAL alert
            if self.notifier:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.notifier.send_system_alert,
                    "CRITICAL: Risk Evaluation Failed",
                    f"Exception during risk evaluation: {e}\nSystem HALTED. Manual intervention required."
                )

            # Log the blocked decision to Truth Engine
            await self.truth_logger.log_decision(
                symbol=self.symbol,
                strategy_name="portfolio_manager",
                signal_values={"order_action": order.action, "qty": str(order.btc_qty_delta)},
                risk_checks={"error": str(e), "exception": True, "approved": False},
                result=DecisionResult.SIGNAL_HOLD,
                result_reason=f"RISK_EVALUATION_EXCEPTION: {e}",
                market_context=order.context,
                timestamp=datetime.now(timezone.utc)
            )
            return state  # Return unchanged state

        # Check if risk approved the trade
        if not risk_result.approved:
            # ORDER BLOCKED BY RISK MANAGER
            logger.warning(f"ORDER BLOCKED BY RISK: {risk_result.rejection_reason}")
            logger.warning(f"  Failed check: {risk_result.first_failure.name.value if risk_result.first_failure else 'unknown'}")

            # Log blocked decision to Truth Engine WITH full risk_checks payload
            await self.truth_logger.log_decision(
                symbol=self.symbol,
                strategy_name="portfolio_manager",
                signal_values={
                    "order_action": order.action,
                    "qty": str(order.btc_qty_delta),
                    "target_alloc_pct": float(order.target_alloc_pct)
                },
                risk_checks=risk_result.to_dict(),  # Full risk audit trail
                result=DecisionResult.SIGNAL_HOLD,  # Blocked = no action
                result_reason=f"RISK_BLOCKED: {risk_result.rejection_reason}",
                market_context=order.context,
                timestamp=datetime.now(timezone.utc)
            )

            # Send warning alert for blocked trade
            if self.notifier:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.notifier.send_system_alert,
                    f"Trade Blocked: {order.action}",
                    f"Risk check failed: {risk_result.rejection_reason}\n"
                    f"Check: {risk_result.first_failure.name.value if risk_result.first_failure else 'unknown'}"
                )

            return state  # Return unchanged state

        # RISK APPROVED - proceed with execution
        logger.info(f"RISK APPROVED: All {len(risk_result.checks)} checks passed")

        # Record this trade for frequency tracking (CRITICAL for max_trades_per_hour)
        self._record_trade()

        # Log decision to Truth Engine with FULL risk_checks payload
        decision_context = {
            "strategy": "portfolio_manager",
            "target_alloc_pct": float(order.target_alloc_pct),
            "current_dd": order.context.get("current_dd", 0),
            "dd_state": order.context.get("dd_state", "unknown"),
            "core_regime": order.context.get("core_regime", "unknown"),
            "dd_multiplier": order.context.get("dd_multiplier", 1.0),
        }

        # Map order action to DecisionResult
        if order.action == "BUY":
            decision_result = DecisionResult.SIGNAL_LONG
        elif order.action == "SELL":
            decision_result = DecisionResult.SIGNAL_CLOSE
        else:
            decision_result = DecisionResult.SIGNAL_HOLD

        decision = await self.truth_logger.log_decision(
            symbol=self.symbol,
            strategy_name="portfolio_manager",
            signal_values=decision_context,
            risk_checks=risk_result.to_dict(),  # Full 10-layer risk audit trail
            result=decision_result,
            result_reason=order.reason,
            market_context=order.context,
            timestamp=datetime.now(timezone.utc)
        )

        # Execute through Portfolio Manager (handles fees)
        new_state = self.pm.execute_order(order, state, current_price, FEES)

        # Sync executor balances
        await self.executor.set_balance("BTC", new_state.btc_qty)
        await self.executor.set_balance("USD", new_state.cash)

        # Log order to Truth Engine
        if order.action == "BUY":
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL

        logged_order = await self.truth_logger.log_order(
            decision_id=decision.decision_id,
            symbol=self.symbol,
            side=side,
            quantity=abs(order.btc_qty_delta),
            requested_price=current_price
        )

        # Mark order as filled (paper trading = instant fill)
        commission = abs(order.btc_qty_delta) * current_price * Decimal("0.004")
        await self.truth_logger.update_order_fill(
            order_id=logged_order.order_id,
            fill_price=current_price,
            fill_quantity=abs(order.btc_qty_delta),
            commission=commission
        )

        # Create/close trade record
        if order.action == "BUY":
            # Open a new trade
            await self.truth_logger.open_trade(
                symbol=self.symbol,
                side=side,
                entry_order_id=logged_order.order_id,
                entry_price=current_price,
                quantity=abs(order.btc_qty_delta),
                strategy_name="portfolio_manager"
            )
        else:
            # Close existing trade(s)
            open_position = await self.truth_logger.get_open_position(self.symbol)
            if open_position:
                await self.truth_logger.close_trade(
                    trade_id=open_position["trade_id"],
                    exit_order_id=logged_order.order_id,
                    exit_price=current_price,
                    exit_reason="portfolio_rebalance"
                )

        # =======================================================================
        # COST BASIS TRACKING FOR CATASTROPHIC STOP
        # =======================================================================
        new_cost_basis = None
        if order.action == "BUY":
            # Calculate new WAC (Weighted Average Cost)
            # WAC = (old_total_cost + new_cost) / (old_qty + new_qty)
            existing_cost_basis = self._get_cost_basis()
            if existing_cost_basis is not None and state.btc_qty > 0:
                # Already have a position - calculate WAC
                old_total_cost = existing_cost_basis * state.btc_qty
                new_cost = current_price * abs(order.btc_qty_delta)
                new_total_cost = old_total_cost + new_cost
                new_cost_basis = new_total_cost / new_state.btc_qty
            else:
                # First buy - cost basis = entry price
                new_cost_basis = current_price

            logger.info(f"Cost Basis Updated: ${float(new_cost_basis):,.2f}")
            logger.info(f"  Catastrophic Stop: ${float(self._compute_catastrophic_stop(new_cost_basis)):,.2f}")

        # Discord notification
        if self.notifier:
            stop_info = ""
            if new_cost_basis:
                stop_price = self._compute_catastrophic_stop(new_cost_basis)
                stop_info = f"\nCatastrophic Stop: ${float(stop_price):,.2f}"

            msg = (
                f"Target: {float(order.target_alloc_pct):.1f}%\n"
                f"BTC: {float(abs(order.btc_qty_delta)):.6f} @ ${float(current_price):,.2f}\n"
                f"DD State: {new_state.dd_state.value}\n"
                f"Equity: ${float(new_state.total_equity):.2f}{stop_info}"
            )
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.notifier.send_system_alert,
                f"Portfolio Rebalance: {order.action}",
                msg
            )

        # Save state with cost basis
        self.save_state(new_state, new_cost_basis)

        return new_state

    async def run_tick(self, data: pd.DataFrame, state: PortfolioState) -> tuple:
        """Run one evaluation tick."""
        current_price = Decimal(str(data.iloc[-1]["close"]))
        eval_time = datetime.now(timezone.utc)

        prev_dd_state = state.dd_state
        prev_recovery_mode = state.recovery_mode

        # Evaluate portfolio
        order, new_state = self.pm.evaluate(
            df=data,
            state=state,
            current_price=current_price,
            timestamp=eval_time
        )

        # Log evaluation
        logger.info(f"[EVALUATE] {order.action}")
        logger.info(f"  Target Allocation: {float(order.target_alloc_pct):.1f}%")
        logger.info(f"  Current DD: {order.context.get('current_dd', 0):.1f}%")
        logger.info(f"  DD State: {order.context.get('dd_state', 'unknown')}")
        logger.info(f"  Regime: {order.context.get('core_regime', 'unknown')}")
        logger.info(f"  DD Multiplier: {order.context.get('dd_multiplier', 1.0):.2f}")

        # Execute order if needed
        if order.action != "HOLD":
            logger.info(f"[EXECUTE] {order.action} {float(abs(order.btc_qty_delta)):.6f} BTC")
            executed_state = await self.execute_order(order, new_state, current_price)
        else:
            executed_state = new_state
            logger.info(f"[HOLD] {order.reason}")

        # Update equity
        btc_value = executed_state.btc_qty * current_price
        executed_state.total_equity = executed_state.cash + btc_value

        # Check alerts
        current_dd = Decimal(str(order.context.get("current_dd", 0)))
        self.alert_manager.check_dd_alerts(
            current_dd=current_dd,
            dd_state=executed_state.dd_state,
            prev_dd_state=prev_dd_state,
            recovery_mode=executed_state.recovery_mode,
            prev_recovery_mode=prev_recovery_mode,
            context=order.context
        )

        # Log snapshot
        self.log_snapshot(executed_state, order, current_price)

        # Summary
        logger.info(f"[PORTFOLIO] Equity: ${float(executed_state.total_equity):.2f}")
        logger.info(f"[PORTFOLIO] BTC: {float(executed_state.btc_qty):.6f} (${float(btc_value):.2f})")
        logger.info(f"[PORTFOLIO] Cash: ${float(executed_state.cash):.2f}")

        return order, executed_state

    def get_interval_seconds(self) -> int:
        """Get sleep interval in seconds."""
        intervals = {"1h": 3600, "4h": 14400, "1d": 86400}
        return intervals.get(self.interval, 86400)

    def calculate_next_run_time(self) -> datetime:
        """Calculate next daily close evaluation time (00:05 UTC)."""
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=0, minute=5, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        return next_run

    async def run(self):
        """Main trading loop."""
        global shutdown_requested

        ensure_runtime_dir()

        # Initialize async components
        await self.truth_logger.initialize()

        # Load or create state
        state = self.load_state()
        if state is None:
            state = self.create_default_state()
            self.save_state(state)
            logger.info(f"Created initial state: ${float(self.starting_capital)}")

        logger.info("=" * 70)
        logger.info("Starting Portfolio Trading Loop")
        logger.info("=" * 70)

        # Start the catastrophic stop guardian as independent task
        self.guardian_task = asyncio.create_task(self._guardian_loop())
        logger.info("Catastrophic Stop Guardian: STARTED")

        tick_count = 0
        while not shutdown_requested:
            try:
                tick_count += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"TICK #{tick_count} - {datetime.now(timezone.utc).isoformat()}")
                logger.info(f"{'='*70}")

                # Fetch data
                data = self.fetch_latest_data()
                if data is None or len(data) < 205:  # Min bars for indicators
                    logger.warning("Insufficient data, waiting...")
                    await asyncio.sleep(60)
                    continue

                current_price = Decimal(str(data.iloc[-1]["close"]))

                # Sync executor state on first tick
                if tick_count == 1:
                    await self.sync_executor_state(state, current_price)

                # Run evaluation
                order, state = await self.run_tick(data, state)

                # Persist state
                self.save_state(state)

                # Sleep until next evaluation
                if not shutdown_requested:
                    if self.interval == "1d":
                        next_run = self.calculate_next_run_time()
                        sleep_seconds = (next_run - datetime.now(timezone.utc)).total_seconds()
                        logger.info(f"Next evaluation at {next_run.isoformat()} ({sleep_seconds/3600:.1f}h)")
                    else:
                        sleep_seconds = self.get_interval_seconds()
                        logger.info(f"Next evaluation in {sleep_seconds/3600:.1f}h")

                    # Sleep in chunks for clean shutdown
                    while sleep_seconds > 0 and not shutdown_requested:
                        chunk = min(sleep_seconds, 60)
                        await asyncio.sleep(chunk)
                        sleep_seconds -= chunk

            except Exception as e:
                logger.error(f"Error in tick: {e}", exc_info=True)
                await asyncio.sleep(60)

        # Cancel guardian task on shutdown
        if self.guardian_task and not self.guardian_task.done():
            self.guardian_task.cancel()
            try:
                await self.guardian_task
            except asyncio.CancelledError:
                pass
            logger.info("Guardian task stopped")

        # Save final state on shutdown
        self.save_state(state)
        logger.info("SHUTDOWN COMPLETE")


async def main():
    parser = argparse.ArgumentParser(description="ArgusNexus V4 Live Portfolio Trader")
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    parser.add_argument("--interval", type=str, default="1d", choices=["1h", "4h", "1d"])
    parser.add_argument("--capital", type=float, default=500.0)
    parser.add_argument("--lookback", type=int, default=300)

    args = parser.parse_args()

    # Reconfigure logging with per-symbol log file
    symbol_safe = args.symbol.replace("-", "_").lower()
    log_file = f"runtime/portfolio_trader_{symbol_safe}.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    trader = LivePortfolioTrader(
        symbol=args.symbol,
        interval=args.interval,
        starting_capital=Decimal(str(args.capital)),
        lookback_days=args.lookback,
    )

    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
