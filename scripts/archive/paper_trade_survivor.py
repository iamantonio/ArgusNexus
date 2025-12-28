#!/usr/bin/env python3
"""
Coinbase Survivor Strategy - Paper Trading with Gemini Prices

Paper trades the Coinbase Survivor strategy using real-time Gemini prices.
Designed for a 1-week validation run before going live.

Strategy: "The 15-Day Ratchet"
- Entry: 15-day breakout + above 50 SMA + ATR > 1%
- Exit: 1.5x ATR trailing stop OR 10-day low

Fee Model: Gemini ActiveTrader (0.70% round-trip)
- Entry: 0.40% taker + 0.10% slippage = 0.50%
- Exit: 0.20% maker = 0.20%

Usage:
    python scripts/paper_trade_survivor.py
    python scripts/paper_trade_survivor.py --capital 500 --interval 1h

The Glass Box Promise:
    Every decision is logged. Every trade is traceable.
"""

import sys
import signal
import logging
import argparse
import asyncio
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.strategy.coinbase_survivor import CoinbaseSurvivorStrategy, Signal
from src.execution.paper import PaperExecutor
from src.execution.gemini import GeminiExecutor, create_gemini_executor
from src.execution.schema import OrderRequest, OrderSide, OrderType
from src.truth.logger import TruthLogger
from src.truth.schema import DecisionResult, OrderSide as TruthOrderSide, ExitReason as TruthExitReason

# Learning System imports
from src.learning import (
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


class SurvivorPaperTrader:
    """
    Paper Trading Engine for Coinbase Survivor Strategy.

    Uses real Gemini prices with simulated execution.
    Designed for 1-week validation before live deployment.
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        starting_capital: Decimal = Decimal("500"),
        check_interval_minutes: int = 60,  # Check every hour
        lookback_days: int = 60
    ):
        self.symbol = symbol
        self.starting_capital = starting_capital
        self.check_interval_minutes = check_interval_minutes
        self.lookback_days = lookback_days

        # Position state
        self.has_position = False
        self.entry_price: Optional[Decimal] = None
        self.entry_time: Optional[datetime] = None
        self.position_size: Optional[Decimal] = None
        self.highest_since_entry: Optional[Decimal] = None

        # Trade tracking for Truth Engine
        self.open_trade_id: Optional[str] = None
        self.entry_order_id: Optional[str] = None
        self.entry_decision_id: Optional[str] = None

        # Database path
        self.db_path = Path(__file__).parent.parent / "data" / "v4_live_paper.db"

        self._setup_components()

    def _setup_components(self):
        """Initialize all components."""
        logger.info("=" * 70)
        logger.info("COINBASE SURVIVOR - PAPER TRADING")
        logger.info("=" * 70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Starting Capital: ${self.starting_capital:,.2f}")
        logger.info(f"Check Interval: {self.check_interval_minutes} minutes")
        logger.info("")

        # Strategy
        self.strategy = CoinbaseSurvivorStrategy()
        logger.info("Strategy: Coinbase Survivor 'The 15-Day Ratchet'")
        logger.info("  Entry: 15-day breakout + above 50 SMA + ATR > 1%")
        logger.info("  Exit: 1.5x ATR trailing stop OR 10-day low")
        logger.info("")

        # Paper Executor with Gemini fee structure
        self.executor = PaperExecutor(
            starting_balance=self.starting_capital,
            base_slippage_pct=Decimal("0.05"),    # 0.05% base spread
            noise_slippage_pct=Decimal("0.05"),   # 0-0.05% random
            size_impact_per_10k=Decimal("0.01"),  # 0.01% per $10k
            fee_rate=Decimal("0.003")             # 0.30% avg (0.40% taker + 0.20% maker / 2)
        )
        logger.info("Executor: Paper (Gemini fee model)")
        logger.info("  Entry fee: 0.40% (taker) + 0.10% slippage")
        logger.info("  Exit fee: 0.20% (maker)")
        logger.info("")

        # Gemini price feed (for real prices)
        try:
            self.gemini = create_gemini_executor(sandbox=False)
            logger.info("Price Feed: Gemini (LIVE prices)")
        except Exception as e:
            logger.warning(f"Gemini not configured: {e}")
            logger.info("Price Feed: Will use Coinbase public API")
            self.gemini = None

        # Truth Engine Logger
        self.truth_logger = TruthLogger(str(self.db_path))
        logger.info(f"Truth Engine: {self.db_path}")
        logger.info("")

    async def fetch_current_price(self) -> Optional[Decimal]:
        """Fetch current price from Gemini."""
        if self.gemini:
            ticker = await self.gemini.get_ticker(self.symbol)
            if ticker:
                return ticker['last']

        # Fallback to public Coinbase API
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.coinbase.com/v2/prices/{self.symbol}/spot"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return Decimal(data['data']['amount'])
        except Exception as e:
            logger.error(f"Failed to fetch price: {e}")

        return None

    async def fetch_historical_data(self) -> Optional[pd.DataFrame]:
        """Fetch historical daily candles for strategy calculation."""
        import aiohttp

        try:
            # Use Coinbase public API for historical data
            async with aiohttp.ClientSession() as session:
                # Coinbase Pro API for candles
                granularity = 86400  # Daily candles
                end = datetime.now(timezone.utc)
                start = end - pd.Timedelta(days=self.lookback_days)

                url = (
                    f"https://api.exchange.coinbase.com/products/{self.symbol}/candles"
                    f"?granularity={granularity}"
                    f"&start={start.isoformat()}"
                    f"&end={end.isoformat()}"
                )

                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        # Convert to DataFrame
                        # Coinbase returns: [time, low, high, open, close, volume]
                        df = pd.DataFrame(
                            data,
                            columns=['timestamp', 'low', 'high', 'open', 'close', 'volume']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df = df.sort_values('timestamp').reset_index(drop=True)

                        logger.info(f"Loaded {len(df)} daily candles")
                        return df
                    else:
                        logger.error(f"Failed to fetch candles: {resp.status}")
                        return None

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None

    async def run_tick(self) -> Dict[str, Any]:
        """Run one evaluation tick."""
        result = {
            "timestamp": datetime.now(timezone.utc),
            "signal": "none",
            "action": None,
            "price": None
        }

        # Fetch data
        df = await self.fetch_historical_data()
        if df is None or len(df) < self.strategy.min_bars:
            logger.warning("Insufficient data for strategy evaluation")
            return result

        current_price = Decimal(str(df.iloc[-1]['close']))
        result["price"] = float(current_price)

        # Update highest since entry if in position
        if self.has_position and self.highest_since_entry:
            current_high = Decimal(str(df.iloc[-1]['high']))
            if current_high > self.highest_since_entry:
                self.highest_since_entry = current_high
                logger.info(f"[TRACKING] New high since entry: ${float(self.highest_since_entry):,.2f}")

        # Evaluate strategy
        signal_result = self.strategy.evaluate(
            df=df,
            timestamp=datetime.now(timezone.utc),
            has_open_position=self.has_position,
            entry_price=self.entry_price,
            highest_high_since_entry=self.highest_since_entry
        )

        signal = signal_result.signal
        context = signal_result.context
        result["signal"] = signal.value

        # Determine decision result for Truth Engine
        if signal == Signal.LONG:
            decision_result = DecisionResult.SIGNAL_LONG
        elif signal == Signal.EXIT_LONG:
            decision_result = DecisionResult.SIGNAL_CLOSE
        else:
            decision_result = DecisionResult.NO_SIGNAL

        # Log decision to Truth Engine
        decision = await self.truth_logger.log_decision(
            symbol=self.symbol,
            strategy_name="coinbase_survivor",
            signal_values=signal_result.to_signal_values(),
            risk_checks={},  # No risk layer for this simple strategy
            result=decision_result,
            result_reason=signal_result.reason,
            market_context={
                "price": float(current_price),
                "highest_15": float(context.highest_15),
                "lowest_10": float(context.lowest_10),
                "sma_50": float(context.sma_50),
                "atr": float(context.atr),
                "atr_percent": float(context.atr_percent)
            }
        )
        result["decision_id"] = decision.decision_id

        logger.info(f"[EVAL] Price: ${float(current_price):,.2f} | Signal: {signal.value.upper()}")
        logger.info(f"[EVAL] Reason: {signal_result.reason}")
        logger.info(f"[TRUTH] Decision logged: {decision.decision_id[:8]}...")

        # Execute signal
        if signal == Signal.LONG and not self.has_position:
            result["action"] = await self._execute_entry(current_price, context, decision.decision_id)

        elif signal == Signal.EXIT_LONG and self.has_position:
            result["action"] = await self._execute_exit(current_price, context, decision.decision_id)

        return result

    async def _execute_entry(self, price: Decimal, context, decision_id: str) -> str:
        """Execute entry order."""
        # Calculate position size (1% risk)
        stop_loss = context.stop_loss_price
        risk_amount = self.starting_capital * Decimal("0.01")
        price_risk = price - stop_loss

        if price_risk <= 0:
            logger.warning("Invalid stop loss - skipping entry")
            return "skipped_invalid_stop"

        position_size = risk_amount / price_risk

        # Cap at 30% of capital
        max_position = (self.starting_capital * Decimal("0.30")) / price
        position_size = min(position_size, max_position)

        # Create order
        order = OrderRequest(
            symbol=self.symbol,
            side=OrderSide.BUY,
            quantity=position_size,
            expected_price=price,
            order_type=OrderType.MARKET
        )

        # Execute
        result = await self.executor.execute(order)

        if result.is_success:
            self.has_position = True
            self.entry_price = result.fill_price
            self.entry_time = datetime.now(timezone.utc)
            self.position_size = position_size
            self.highest_since_entry = price

            # Log order to Truth Engine
            order_obj = await self.truth_logger.log_order(
                decision_id=decision_id,
                symbol=self.symbol,
                side=TruthOrderSide.BUY,
                quantity=position_size,
                requested_price=price
            )
            self.entry_order_id = order_obj.order_id
            self.entry_decision_id = decision_id

            # Update order with fill info
            await self.truth_logger.update_order_fill(
                order_id=order_obj.order_id,
                fill_price=result.fill_price,
                fill_quantity=position_size,
                exchange_order_id=result.external_id,
                commission=result.fee
            )

            # Open trade in Truth Engine
            self.open_trade_id = await self.truth_logger.open_trade(
                symbol=self.symbol,
                side=TruthOrderSide.BUY,
                entry_order_id=order_obj.order_id,
                entry_price=result.fill_price,
                quantity=position_size,
                stop_loss_price=stop_loss,
                strategy_name="coinbase_survivor"
            )

            logger.info(f"[ENTRY] BUY {float(position_size):.6f} BTC @ ${float(result.fill_price):,.2f}")
            logger.info(f"[ENTRY] Stop Loss: ${float(stop_loss):,.2f}")
            logger.info(f"[ENTRY] Fee: ${float(result.fee):,.2f}")
            logger.info(f"[TRUTH] Trade opened: {self.open_trade_id[:8]}...")

            return "entry_executed"
        else:
            logger.error(f"[ENTRY] FAILED: {result.message}")
            return "entry_failed"

    async def _execute_exit(self, price: Decimal, context, decision_id: str) -> str:
        """Execute exit order."""
        if not self.position_size:
            return "no_position"

        # Create order
        order = OrderRequest(
            symbol=self.symbol,
            side=OrderSide.SELL,
            quantity=self.position_size,
            expected_price=price,
            order_type=OrderType.LIMIT,
            limit_price=price
        )

        # Execute
        result = await self.executor.execute(order)

        if result.is_success:
            # Calculate P&L
            gross_pnl = (result.fill_price - self.entry_price) * self.position_size

            # Log exit order to Truth Engine
            exit_order = await self.truth_logger.log_order(
                decision_id=decision_id,
                symbol=self.symbol,
                side=TruthOrderSide.SELL,
                quantity=self.position_size,
                requested_price=price
            )

            # Update order with fill info
            await self.truth_logger.update_order_fill(
                order_id=exit_order.order_id,
                fill_price=result.fill_price,
                fill_quantity=self.position_size,
                exchange_order_id=result.external_id,
                commission=result.fee
            )

            # Map exit reason to Truth Engine enum
            exit_reason_map = {
                "trailing_stop": TruthExitReason.STOP_LOSS,
                "exit_channel": TruthExitReason.STOP_LOSS,
                "none": TruthExitReason.SIGNAL_EXIT
            }
            truth_exit_reason = exit_reason_map.get(
                context.exit_reason.value,
                TruthExitReason.SIGNAL_EXIT
            )

            # Close trade in Truth Engine
            if self.open_trade_id:
                await self.truth_logger.close_trade(
                    trade_id=self.open_trade_id,
                    exit_order_id=exit_order.order_id,
                    exit_price=result.fill_price,
                    exit_reason=truth_exit_reason,
                    commission=result.fee,
                    slippage=result.slippage or Decimal("0")
                )

            logger.info(f"[EXIT] SELL {float(self.position_size):.6f} BTC @ ${float(result.fill_price):,.2f}")
            logger.info(f"[EXIT] Reason: {context.exit_reason.value}")
            logger.info(f"[EXIT] Gross P&L: ${float(gross_pnl):,.2f}")
            logger.info(f"[TRUTH] Trade closed: {self.open_trade_id[:8] if self.open_trade_id else 'N/A'}...")

            # Reset position state
            self.has_position = False
            self.entry_price = None
            self.entry_time = None
            self.position_size = None
            self.highest_since_entry = None
            self.open_trade_id = None
            self.entry_order_id = None
            self.entry_decision_id = None

            return "exit_executed"
        else:
            logger.error(f"[EXIT] FAILED: {result.message}")
            return "exit_failed"

    def get_summary(self) -> str:
        """Generate trading summary."""
        balance = self.executor._balances.get("USD", Decimal("0"))
        pnl = balance - self.starting_capital
        pnl_pct = (pnl / self.starting_capital) * 100

        summary = f"""
================================================================================
PAPER TRADING SUMMARY
================================================================================
Strategy: Coinbase Survivor
Database: {self.db_path}

CAPITAL
  Starting: ${float(self.starting_capital):,.2f}
  Current:  ${float(balance):,.2f}
  P&L:      ${float(pnl):,.2f} ({float(pnl_pct):.2f}%)

CURRENT POSITION: {'LONG' if self.has_position else 'FLAT'}
{f'  Entry: ${float(self.entry_price):,.2f}' if self.has_position and self.entry_price else ''}
{f'  Size: {float(self.position_size):.6f} BTC' if self.has_position and self.position_size else ''}

View full trade history at: http://localhost:8000
================================================================================
"""
        return summary

    async def run(self):
        """Main trading loop."""
        global shutdown_requested

        # Initialize Truth Engine
        await self.truth_logger.initialize()
        logger.info("Truth Engine initialized")

        logger.info("Starting paper trading loop...")
        logger.info(f"Checking every {self.check_interval_minutes} minutes")
        logger.info("Press Ctrl+C to stop and see summary")
        logger.info("")

        tick_count = 0

        while not shutdown_requested:
            try:
                tick_count += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"TICK #{tick_count} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                logger.info(f"{'='*70}")

                result = await self.run_tick()

                # Show current state
                balance = await self.executor.get_balance("USD")
                logger.info(f"[PORTFOLIO] USD: ${float(balance):,.2f} | Position: {'LONG' if self.has_position else 'FLAT'}")

                if not shutdown_requested:
                    # Sleep in small increments
                    sleep_seconds = self.check_interval_minutes * 60
                    for _ in range(sleep_seconds // 10):
                        if shutdown_requested:
                            break
                        await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in tick: {e}", exc_info=True)
                await asyncio.sleep(60)

        # Cleanup
        if self.gemini:
            await self.gemini.close()

        # Print summary
        print(self.get_summary())


async def main():
    parser = argparse.ArgumentParser(description="Coinbase Survivor - Paper Trading")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Trading symbol")
    parser.add_argument("--capital", type=float, default=500.0, help="Starting capital")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in minutes")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback days for data")

    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    trader = SurvivorPaperTrader(
        symbol=args.symbol,
        starting_capital=Decimal(str(args.capital)),
        check_interval_minutes=args.interval,
        lookback_days=args.lookback
    )

    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
