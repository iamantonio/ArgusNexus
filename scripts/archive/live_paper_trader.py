#!/usr/bin/env python3
"""
ArgusNexus V4 Live Paper Trader - TURTLE-4 "The Ratchet"

THE DEPLOYMENT: Moving from backtests to live paper trading.
Same engine, same strategy, same truth - just real-time data.

This script runs 24/7, fetching 4H candles and executing the TURTLE-4 strategy.
All decisions logged to the Glass Box (Truth Engine).

Usage:
    python scripts/live_paper_trader.py
    python scripts/live_paper_trader.py --interval 4h --capital 10000

Environment:
    - Execution Mode: PAPER (no real money at risk)
    - Strategy: TURTLE-4 (The Ratchet) with Chandelier Exit
    - Data: Yahoo Finance 4H candles (updated every 4 hours)

The Glass Box Promise:
    - Every candle close triggers a decision log
    - Chandelier stop values update and log as price moves
    - All trades recorded in v4_truth.db
"""

import sys
import time
import signal
import logging
import argparse
import asyncio
import json
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from engine import TradingEngine, TradingMode
from strategy.donchian import DonchianBreakout
from strategy.donchian import Signal as DonchianSignal
from risk import RiskManager, RiskConfig
from execution import PaperExecutor, ExecutionMode
from execution.websocket import WebSocketManager
from truth.logger import TruthLogger
from truth.schema import DecisionResult, OrderSide, ExitReason
from data.loader import fetch_yahoo_data, fetch_coinbase_data, resample_ohlcv
from notifier import DiscordNotifier

# Twitter Integration
from twitter import TwitterHook, TwitterHookConfig, create_twitter_hook

# Session Management imports (24/7 Perpetual Trading)
from session import (
    SessionManager,
    SessionConfig,
    SessionState,
    MarketSession,
    LiquidityMonitor,
)

# Learning System imports
from learning import (
    RegimeDetector,
    ConfidenceScorer,
    PPOAgent,
    PPOConfig,
    MarketRegime,
    SignalQuality,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Global shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info("Shutdown signal received. Finishing current tick...")
    shutdown_requested = True


class LivePaperTrader:
    """
    Live Paper Trading Engine for TURTLE-4 (Async).

    Runs the same TradingEngine as backtests, just with real-time data.
    The Glass Box logs everything - decisions, orders, trades.
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        interval: str = "4h",
        starting_capital: Decimal = Decimal("10000"),
        lookback_days: int = 120,  # Need 200 bars for SMA200
        data_source: str = "coinbase"  # "coinbase" or "yahoo"
    ):
        self.symbol = symbol
        self.interval = interval
        self.starting_capital = starting_capital
        self.lookback_days = lookback_days
        self.data_source = data_source.lower()

        # State tracking
        self.open_trade_id: Optional[str] = None
        self.entry_commission: Decimal = Decimal("0")
        self.open_quantity: Decimal = Decimal("0")

        # Paths
        self.db_path = Path(__file__).parent.parent / "data" / "v4_live_paper.db"
        self.runtime_dir = Path(__file__).parent.parent / "runtime"
        self.state_file = self.runtime_dir / "paper_trader_state.json"

        self._setup_components()

    def _setup_components(self):
        """Initialize all V4 components."""
        logger.info("=" * 70)
        logger.info("ArgusNexus V4 - LIVE PAPER TRADER (ASYNC)")
        logger.info("=" * 70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Interval: {self.interval}")
        logger.info(f"Starting Capital: ${self.starting_capital:,.2f}")
        logger.info(f"Data Source: {self.data_source.upper()}")
        logger.info(f"Database: {self.db_path}")
        logger.info("")

        # Strategy: TURTLE-4 "The Ratchet"
        logger.info("Strategy: DONCHIAN TURTLE-4 v5.0 (The Ratchet)")

        self.strategy = DonchianBreakout()

        # Risk Manager - load from YAML
        self.risk_config = RiskConfig.load_from_yaml()
        self.risk_manager = RiskManager(self.risk_config)

        # Paper Executor with realistic slippage
        self.executor = PaperExecutor(
            starting_balance=self.starting_capital,
            base_slippage_pct=Decimal("0.02"),
            noise_slippage_pct=Decimal("0.03"),
            size_impact_per_10k=Decimal("0.01"),
            fee_rate=Decimal("0.004")
        )

        # Truth Logger - async database
        self.truth_logger = TruthLogger(str(self.db_path))

        # Discord Notifier
        self.notifier = DiscordNotifier()
        if not self.notifier.url:
            self.notifier = None

        # Twitter Hook (Grok-powered organic tweets)
        self.twitter_hook = TwitterHook(TwitterHookConfig(
            enabled=True,
            simulation_mode=False,  # LIVE - Real posts to Twitter
            post_entries=True,
            post_exits=True,
            post_rejections=True,
            post_lessons=True,
            generate_threads=True,
            decision_base_url="http://localhost:8000/decision",
        ))
        logger.info("Twitter Hook: ENABLED (LIVE MODE - Real tweets)")

        # Trading Engine
        self.engine = TradingEngine(
            strategy=self.strategy,
            risk_manager=self.risk_manager,
            executor=self.executor,
            truth_logger=self.truth_logger,
            symbol=self.symbol,
            capital=self.starting_capital,
            notifier=self.notifier
        )

        # WebSocket Manager (Instant Stop-Loss Sidecar)
        self.ws_manager = WebSocketManager(symbols=[self.symbol])

        async def on_price_update(sym, price):
            if sym == self.symbol:
                await self.engine.update_price_and_check_stops(price)

        self.ws_manager.register_callback(on_price_update)

        # Session Manager (24/7 Perpetual Trading)
        logger.info("")
        logger.info("Session Management: ENABLED")
        self.session_manager = SessionManager()
        self.liquidity_monitor = LiquidityMonitor()
        session_info = self.session_manager.get_session_info()
        logger.info(f"  Current Session: {session_info['current_session'].upper()}")
        logger.info(f"  Session State: {session_info['session_state']}")
        logger.info(f"  Dead Zone: {'YES - TRADING PAUSED' if session_info['is_dead_zone'] else 'No'}")
        logger.info(f"  Position Multiplier: {session_info['position_multiplier']:.0%}")
        logger.info(f"  Next Session: {session_info['next_session']} in {session_info['time_to_next']}")

        # Learning System Components
        logger.info("")
        logger.info("Learning System: ENABLED")
        self.regime_detector = RegimeDetector()
        self.confidence_scorer = ConfidenceScorer()

        # PPO Agent with persistence
        ppo_config = PPOConfig(
            min_experiences=20,      # Burn-in after 20 trades
            update_frequency=10,     # Update every 10 trades
            learning_rate=0.0003,
        )
        self.ppo_agent = PPOAgent(
            db_path=str(self.db_path),
            config=ppo_config,
        )
        self.ppo_agent.load_weights()  # Load any saved weights

        # Track recent performance for PPO state
        self.recent_trades: list = []
        self.max_recent_trades = 50

        logger.info(f"  Regime Detector: 14 market conditions")
        logger.info(f"  Confidence Scorer: Multi-factor signal quality")
        logger.info(f"  PPO Agent: Adaptive position sizing (burned_in={self.ppo_agent.is_burned_in})")
        logger.info("")

    async def hydrate_and_reconcile(self) -> bool:
        """
        Hydrate position state from DB and reconcile on startup (Async).
        """
        logger.info("=" * 70)
        logger.info("POSITION STATE HYDRATION & RECONCILIATION")
        logger.info("=" * 70)

        # Initialize async logger
        await self.truth_logger.initialize()

        # Fetch historical data for highest_high computation
        data = self.fetch_latest_data()

        # Hydrate position state
        hydrate_result = await self.engine.hydrate_position_state(data)

        if hydrate_result.requires_fail_closed:
            await self.engine.enter_fail_closed(hydrate_result.error_message)
            return False

        if hydrate_result.position_found:
            # Sync our local state with engine
            self.open_trade_id = self.engine.open_trade_id
            self.open_quantity = hydrate_result.position_state.quantity if hydrate_result.position_state else Decimal("0")
            logger.info(f"[HYDRATE] Local state synced with engine")

        # Reconcile against paper executor
        reconcile_result = await self.engine.reconcile_position()

        if reconcile_result.requires_fail_closed:
            return False

        logger.info("[RECONCILE] State reconciliation PASSED")
        logger.info("=" * 70)
        logger.info("")

        return True

    def fetch_latest_data(self) -> Optional[pd.DataFrame]:
        """Fetch latest candle data (Sync call but could be wrapped in thread)."""
        try:
            if self.data_source == "coinbase":
                df, report = fetch_coinbase_data(
                    symbol=self.symbol,
                    lookback_days=self.lookback_days,
                    interval=self.interval,
                    verbose=False
                )
            else:
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=self.lookback_days)
                df, report = fetch_yahoo_data(
                    symbol=self.symbol,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1h",
                    verbose=False
                )
                if self.interval != "1h":
                    df = resample_ohlcv(df, self.interval, verbose=False)

            logger.info(f"Data loaded: {len(df)} bars, latest: {df['timestamp'].iloc[-1]}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return None

    def _get_recent_performance(self) -> dict:
        """Calculate recent performance metrics for PPO state."""
        if not self.recent_trades:
            return {"win_rate": 0.5, "avg_pnl_percent": 0.0, "current_drawdown": 0.0}

        wins = sum(1 for t in self.recent_trades if t.get("pnl_pct", 0) > 0)
        total = len(self.recent_trades)

        return {
            "win_rate": wins / total if total > 0 else 0.5,
            "avg_pnl_percent": sum(t.get("pnl_pct", 0) for t in self.recent_trades) / total if total > 0 else 0.0,
            "current_drawdown": 0.0,  # Could track actual DD
        }

    def _detect_regime(self, data: pd.DataFrame):
        """Detect current market regime from data."""
        if len(data) < 20:
            return None

        # Get latest values
        close = float(data["close"].iloc[-1])
        high = data["high"].iloc[-20:].max()
        low = data["low"].iloc[-20:].min()
        atr = float((high - low) / 20)  # Simple ATR approximation

        # Calculate ADX approximation (simplified)
        returns = data["close"].pct_change().iloc[-14:]
        adx = abs(returns.mean()) / (returns.std() + 1e-8) * 100
        adx = min(50, max(10, adx * 5))  # Scale to reasonable range

        # RSI
        gains = returns.clip(lower=0).sum()
        losses = (-returns.clip(upper=0)).sum()
        rsi = 100 - (100 / (1 + gains / (losses + 1e-8))) if losses > 0 else 50

        # SMAs
        sma_20 = float(data["close"].iloc[-20:].mean()) if len(data) >= 20 else close
        sma_50 = float(data["close"].iloc[-50:].mean()) if len(data) >= 50 else close

        return self.regime_detector.detect(
            current_price=close,
            atr=atr,
            adx=adx,
            rsi=rsi,
            sma_20=sma_20,
            sma_50=sma_50,
        )

    async def run_tick(self, data: pd.DataFrame) -> dict:
        """Run one tick through the trading engine with learning system (Async)."""

        # 0. Check session state (24/7 perpetual trading awareness)
        session_context = self.session_manager.get_session_context()
        session_event = self.session_manager.check_session_transition()

        if session_event:
            logger.info(f"[SESSION] Transition: {session_event.previous_session.value if session_event.previous_session else 'None'} -> {session_event.session.value}")
            if self.notifier:
                if session_event.session == MarketSession.DEAD_ZONE:
                    await self.notifier.send_async(
                        f"🌙 **DEAD ZONE ENTERED** - Trading paused until 00:00 UTC\n"
                        f"Session: {session_event.session.value}"
                    )
                elif session_event.previous_session == MarketSession.DEAD_ZONE:
                    await self.notifier.send_async(
                        f"☀️ **DEAD ZONE EXITED** - Trading resumed\n"
                        f"Session: {session_event.session.value}"
                    )

        # Check if we should pause trading
        should_pause, pause_reason = self.session_manager.should_pause_trading()
        if should_pause:
            logger.info(f"[SESSION] {pause_reason} - Skipping tick (exits still allowed)")
            # Still allow exits during dead zone, just no new entries
            if self.engine.has_open_position:
                # Run tick but engine should respect exit-only mode
                pass  # Continue to allow stop-loss/exit checks
            else:
                return {
                    "signal": "none",
                    "action": "SESSION_PAUSED",
                    "decision_id": None,
                    "trade_id": None,
                    "regime": None,
                    "confidence": None,
                    "ppo_size": None,
                    "session": session_context.session.value,
                    "session_state": session_context.state.value,
                }

        # Log session context
        logger.info(f"[SESSION] {session_context.session.value.upper()} | "
                   f"Multiplier: {session_context.position_size_multiplier:.0%} | "
                   f"Next: {session_context.time_to_next_session_minutes}m")

        # 1. Detect market regime
        regime_state = self._detect_regime(data)
        if regime_state:
            logger.info(f"[REGIME] {regime_state.current_regime.value} | "
                       f"Vol: {regime_state.volatility_regime.value} | "
                       f"Conf: {regime_state.confidence:.0%}")

        # 2. Run base strategy tick
        result = await self.engine.run_tick(data)

        # 3. Score signal confidence if we have a signal
        confidence_score = None
        ppo_recommendation = None

        if result.signal and result.signal.signal != DonchianSignal.HOLD:
            signal = result.signal
            context = signal.context

            # Build signal values for confidence scoring
            signal_values = {
                "adx": getattr(context, "adx", 25),
                "atr": float(getattr(context, "atr", 0) or 0),
                "current_price": float(data["close"].iloc[-1]),
                "rsi": getattr(context, "rsi", 50),
                "crossover_confirmed": signal.signal in [DonchianSignal.LONG, DonchianSignal.EXIT],
            }

            if regime_state:
                # Score the signal
                confidence_score = self.confidence_scorer.score_signal(
                    signal_type="long" if signal.signal == DonchianSignal.LONG else "exit",
                    signal_values=signal_values,
                    regime_state=regime_state,
                )

                logger.info(f"[CONFIDENCE] Score: {confidence_score.final_score:.0%} | "
                           f"Quality: {confidence_score.quality.value} | "
                           f"Action: {confidence_score.recommended_action}")

                # Get PPO position size recommendation
                ppo_recommendation = self.ppo_agent.recommend_position_size(
                    regime_state=regime_state,
                    signal_strength=confidence_score.signal_strength,
                    recent_performance=self._get_recent_performance(),
                    confidence_score=confidence_score.position_size_factor,
                )

                logger.info(f"[PPO] Size: {ppo_recommendation['recommended_size']:.0%} | "
                           f"Burned-in: {ppo_recommendation['is_burned_in']} | "
                           f"Experiences: {ppo_recommendation['experiences']}")

                # Log factors if any
                if ppo_recommendation.get("factors"):
                    for factor in ppo_recommendation["factors"]:
                        logger.info(f"[PPO] Factor: {factor}")

            # Log signal
            logger.info(f"[SIGNAL] {signal.signal.value.upper()}: {signal.reason[:80]}...")

            if self.engine.has_open_position and context.chandelier_stop:
                logger.info(f"[CHANDELIER] Stop: ${float(context.chandelier_stop):,.2f}")

        # 4. Track trade outcomes for learning
        if result.trade_id and result.action_taken in ["CLOSED", "STOPPED"]:
            # A trade just closed - record for learning
            trade_pnl = 0.0  # Would need to fetch from DB
            self.recent_trades.append({
                "trade_id": result.trade_id,
                "pnl_pct": trade_pnl,
                "regime": regime_state.current_regime.value if regime_state else "unknown",
            })
            # Keep only recent trades
            if len(self.recent_trades) > self.max_recent_trades:
                self.recent_trades = self.recent_trades[-self.max_recent_trades:]

        return {
            "signal": result.signal.signal.value if result.signal else "none",
            "action": result.action_taken,
            "decision_id": result.decision_id,
            "trade_id": result.trade_id,
            "regime": regime_state.current_regime.value if regime_state else None,
            "confidence": confidence_score.final_score if confidence_score else None,
            "ppo_size": ppo_recommendation["recommended_size"] if ppo_recommendation else None,
            "session": session_context.session.value,
            "session_state": session_context.state.value,
            "position_multiplier": session_context.position_size_multiplier,
        }

    def get_interval_seconds(self) -> int:
        intervals = {"1h": 3600, "4h": 14400, "1d": 86400}
        return intervals.get(self.interval, 14400)

    async def write_state(self, current_price: float = 0.0):
        """Write current state to JSON file for dashboard API."""
        try:
            balance = await self.executor.get_balance("USD")

            # Get position info
            has_position = self.engine.open_trade_id is not None
            qty = float(self.open_quantity) if has_position else 0.0
            cost_basis = float(self.engine._entry_price) if has_position and hasattr(self.engine, '_entry_price') and self.engine._entry_price else 0.0
            position_value = qty * current_price if has_position else 0.0
            total_equity = float(balance) + position_value

            # Track high water mark
            hwm = float(self.starting_capital)
            if hasattr(self, '_high_water_mark'):
                hwm = max(self._high_water_mark, total_equity)
            else:
                hwm = max(float(self.starting_capital), total_equity)
            self._high_water_mark = hwm

            # Get session info for state
            session_info = self.session_manager.get_session_info()

            state = {
                "total_equity": round(total_equity, 2),
                "cash": round(float(balance), 2),
                "high_water_mark": round(hwm, 2),
                "starting_capital": float(self.starting_capital),
                "positions": {},
                "dd_state": "normal",
                "current_dd_pct": round((hwm - total_equity) / hwm * 100, 2) if hwm > 0 else 0,
                "last_update": datetime.now(timezone.utc).isoformat(),
                "engine": "paper_learning",
                "symbol": self.symbol,
                "interval": self.interval,
                # Session awareness fields
                "session": {
                    "current": session_info["current_session"],
                    "state": session_info["session_state"],
                    "is_dead_zone": session_info["is_dead_zone"],
                    "position_multiplier": session_info["position_multiplier"],
                    "next_session": session_info["next_session"],
                    "time_to_next": session_info["time_to_next"],
                    "trading_allowed": session_info["trading_allowed"],
                },
            }

            if has_position and qty > 0:
                unrealized_pnl = (current_price - cost_basis) * qty if cost_basis > 0 else 0
                state["positions"][self.symbol] = {
                    "symbol": self.symbol,
                    "qty": qty,
                    "cost_basis": cost_basis,
                    "current_price": current_price,
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "regime": "unknown"
                }

            self.runtime_dir.mkdir(exist_ok=True)
            self.state_file.write_text(json.dumps(state, indent=2))

        except Exception as e:
            logger.warning(f"Failed to write state file: {e}")

    async def run(self):
        """Main trading loop (Async)."""
        global shutdown_requested

        logger.info("Starting async live paper trading loop...")

        if not await self.hydrate_and_reconcile():
            logger.critical("FAIL-CLOSED: Initial reconciliation failed")

        # Start WebSocket Sidecar
        await self.ws_manager.start()

        # Start Twitter Hook
        await self.twitter_hook.start()

        tick_count = 0
        while not shutdown_requested:
            try:
                tick_count += 1
                logger.info(f"{'='*70}\nTICK #{tick_count} - {datetime.now(timezone.utc).isoformat()}\n{'='*70}")

                data = self.fetch_latest_data()
                if data is None or len(data) < self.strategy.min_bars:
                    await asyncio.sleep(60)
                    continue

                reconcile_result = await self.engine.reconcile_position()
                result = await self.run_tick(data)

                # Twitter Hook: Post decisions and trade updates
                if result.get("decision_id"):
                    try:
                        # Fetch full decision from database
                        decision = await self.truth_logger.get_decision(result["decision_id"])
                        if decision:
                            await self.twitter_hook.on_decision(decision)
                    except Exception as e:
                        logger.warning(f"Twitter hook error (decision): {e}")

                if result.get("action") in ["CLOSED", "STOPPED"] and result.get("trade_id"):
                    try:
                        # Fetch trade data for exit tweet
                        trade = await self.truth_logger.get_trade(result["trade_id"])
                        if trade:
                            await self.twitter_hook.on_trade_closed(trade)
                    except Exception as e:
                        logger.warning(f"Twitter hook error (trade): {e}")

                # Summary
                balance = await self.executor.get_balance("USD")
                logger.info(f"[PORTFOLIO] USD Balance: ${float(balance):,.2f}")

                # Write state for dashboard
                current_price = float(data["close"].iloc[-1]) if len(data) > 0 else 0.0
                await self.write_state(current_price)

                if not shutdown_requested:
                    # Adjust sleep based on session state
                    if self.session_manager.is_dead_zone():
                        # During dead zone, check less frequently (every 5 min)
                        sleep_seconds = 300
                        logger.info(f"[SESSION] Dead zone - sleeping {sleep_seconds}s until next check")
                    else:
                        sleep_seconds = self.get_interval_seconds()

                    # Sleep in small increments to allow clean shutdown
                    for _ in range(sleep_seconds // 10):
                        if shutdown_requested: break
                        await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in tick: {e}", exc_info=True)
                await asyncio.sleep(60)

        # Shutdown WebSocket and Twitter Hook
        await self.ws_manager.stop()
        await self.twitter_hook.stop()
        logger.info("SHUTDOWN COMPLETE")


async def main():
    parser = argparse.ArgumentParser(description="ArgusNexus V4 Live Paper Trader (Async)")
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    parser.add_argument("--interval", type=str, default="4h", choices=["1h", "4h", "1d"])
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--data-source", type=str, default="coinbase")

    args = parser.parse_args()

    # Register signal handler for clean shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: signal_handler(None, None))

    trader = LivePaperTrader(
        symbol=args.symbol,
        interval=args.interval,
        starting_capital=Decimal(str(args.capital)),
        lookback_days=args.lookback,
        data_source=args.data_source
    )

    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())


def main():
    parser = argparse.ArgumentParser(
        description="ArgusNexus V4 Live Paper Trader - TURTLE-4"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC-USD",
        help="Trading symbol (default: BTC-USD)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="4h",
        choices=["1h", "4h", "1d"],
        help="Candle interval (default: 4h)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Starting capital (default: 10000)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Lookback days for historical data (default: 120)"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="coinbase",
        choices=["coinbase", "yahoo"],
        help="Data source: coinbase (live) or yahoo (backtest) (default: coinbase)"
    )

    args = parser.parse_args()

    trader = LivePaperTrader(
        symbol=args.symbol,
        interval=args.interval,
        starting_capital=Decimal(str(args.capital)),
        lookback_days=args.lookback,
        data_source=args.data_source
    )

    trader.run()


if __name__ == "__main__":
    main()
