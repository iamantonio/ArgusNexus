#!/usr/bin/env python3
"""
V4 Async Backtester - Protocol: CHRONOS

THE PHILOSOPHY: This backtest uses the EXACT same TradingEngine as live.
No special backtest logic. Row-by-row through history.

What we learn here, we learn for real.

Usage:
    python scripts/run_backtest.py
"""

import sys
import argparse
import logging
import asyncio
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Optional
import aiosqlite
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from engine import TradingEngine, TradingMode
from strategy.donchian import DonchianBreakout, Signal as DonchianSignal, create_strategy
from risk import RiskManager, RiskConfig
from execution import PaperExecutor
from truth.logger import TruthLogger
from truth.schema import DecisionResult, OrderSide, ExitReason
from data.loader import load_ohlcv, load_or_fetch_btc, resample_ohlcv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(days: int = 90, start_price: float = 100000.0, volatility: float = 0.02) -> pd.DataFrame:
    """Generate synthetic BTC price data for backtesting."""
    hours = days * 24
    np.random.seed(42)
    returns = np.random.normal(0, volatility / np.sqrt(24), hours)
    trend = np.zeros(hours); trend_strength = 0
    for i in range(1, hours):
        trend_strength = 0.95 * trend_strength + 0.05 * returns[i-1]
        trend[i] = trend_strength * 0.5
    returns = returns + trend
    prices = [start_price]
    for r in returns: prices.append(prices[-1] * (1 + r))
    prices = prices[1:]
    data = []
    base_time = datetime.now(timezone.utc) - timedelta(days=days)
    for i, close in enumerate(prices):
        timestamp = base_time + timedelta(hours=i)
        spread = abs(close * volatility / np.sqrt(24) * 2)
        high = close + np.random.uniform(0, spread)
        low = close - np.random.uniform(0, spread)
        open_p = low + np.random.uniform(0, high - low)
        high = max(high, open_p, close); low = min(low, open_p, close)
        data.append({
            "timestamp": timestamp, "open": open_p, "high": high, "low": low, "close": close, "volume": 1000000
        })
    return pd.DataFrame(data)


class BacktestEngine:
    """The V4 Backtester - Async reality."""

    def __init__(
        self,
        db_path: str,
        starting_capital: Decimal = Decimal("10000"),
        min_history: int = 50,
        strategy_name: str = "donchian"
    ):
        self.db_path = Path(db_path)
        self.starting_capital = starting_capital
        self.min_history = min_history
        self.strategy_name = strategy_name
        self.open_trade_id: Optional[str] = None
        self.entry_commission: Decimal = Decimal("0")
        self.open_quantity: Decimal = Decimal("0")

    async def _setup_components(self):
        """Initialize all components (Async)."""
        # Load from YAML if possible for strategy params
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                full_cfg = yaml.safe_load(f)
                strat_cfg = full_cfg.get('strategy', {}).get('donchian', {})
                self.strategy = create_strategy(**strat_cfg)
        else:
            self.strategy = DonchianBreakout()
        
        self.Signal = DonchianSignal
        self.min_history = max(self.min_history, self.strategy.min_bars)

        # DEVIL'S ADVOCATE: Should we use production risk settings in backtests?
        # PUSHBACK: Only if we want to test PORTFOLIO performance. For STRATEGY alpha,
        # we must loosen the concentration/frequency constraints so we actually see trades.
        self.risk_config = RiskConfig.load_from_yaml()
        self.risk_config.max_asset_concentration_pct = 100.0  # Allow full capital
        self.risk_config.max_correlated_exposure_pct = 100.0  # No correlation limits
        self.risk_config.max_trades_per_hour = 1000           # No frequency limits
        self.risk_manager = RiskManager(self.risk_config)

        self.executor = PaperExecutor(
            starting_balance=self.starting_capital,
            base_slippage_pct=Decimal("0.02"),
            noise_slippage_pct=Decimal("0.03"),
            size_impact_per_10k=Decimal("0.01"),
            fee_rate=Decimal("0.004")
        )

        self.truth_logger = TruthLogger(str(self.db_path))
        await self.truth_logger.initialize()
        await self.truth_logger.reset()

        self.engine = TradingEngine(
            strategy=self.strategy,
            risk_manager=self.risk_manager,
            executor=self.executor,
            truth_logger=self.truth_logger,
            symbol="BTC-USD",
            capital=self.starting_capital
        )

    async def run(self, data: pd.DataFrame) -> dict:
        """Run backtest row-by-row (Async)."""
        await self._setup_components()
        
        print(f"\nRunning backtest on {len(data)} bars...")
        ticks_processed = 0
        trades_executed = 0
        lookback = 250
        
        for i in range(self.min_history, len(data)):
            # Slice history up to current row
            history = data.iloc[max(0, i - lookback):i + 1].copy()
            
            # Run one tick through the async engine
            result = await self.engine.run_tick(history)
            ticks_processed += 1

            if result.action_taken == "executed":
                trades_executed += 1

            if ticks_processed % 500 == 0:
                print(f"  Processed {ticks_processed} ticks...")

        return await self.generate_report()

    async def generate_report(self) -> dict:
        """Generate performance report from Truth Engine (Async)."""
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute("""
                SELECT COUNT(*) as total_trades,
                       SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed_trades,
                       SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_trades,
                       SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN is_winner = 0 AND status = 'closed' THEN 1 ELSE 0 END) as losses,
                       COALESCE(SUM(CAST(net_pnl AS REAL)), 0) as total_pnl,
                       COALESCE(AVG(CAST(net_pnl AS REAL)), 0) as avg_pnl
                FROM trades
            """)
            pnl_row = await cursor.fetchone()
            
            cursor = await db.execute("SELECT net_pnl FROM trades WHERE status = 'closed' ORDER BY exit_timestamp")
            drawdown_rows = await cursor.fetchall()
            
            max_drawdown = 0; peak = 0; running_pnl = 0
            for row in drawdown_rows:
                running_pnl += float(row["net_pnl"])
                if running_pnl > peak: peak = running_pnl
                dd = peak - running_pnl
                if dd > max_drawdown: max_drawdown = dd

            cursor = await db.execute("SELECT AVG(CAST(slippage_percent AS REAL)) as avg_slippage_pct FROM orders WHERE slippage_percent IS NOT NULL")
            slippage_row = await cursor.fetchone()

            cursor = await db.execute("SELECT COUNT(*) as total_decisions FROM decisions")
            decision_row = await cursor.fetchone()

        return {
            "total_trades": pnl_row["total_trades"],
            "closed_trades": pnl_row["closed_trades"] or 0,
            "open_trades": pnl_row["open_trades"] or 0,
            "wins": pnl_row["wins"] or 0,
            "losses": pnl_row["losses"] or 0,
            "win_rate": (pnl_row["wins"] / pnl_row["closed_trades"] * 100) if pnl_row["closed_trades"] else 0,
            "total_pnl": pnl_row["total_pnl"],
            "avg_pnl_per_trade": pnl_row["avg_pnl"],
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": (max_drawdown / float(self.starting_capital) * 100) if max_drawdown > 0 else 0,
            "avg_slippage_pct": slippage_row["avg_slippage_pct"] or 0,
            "total_decisions": decision_row["total_decisions"]
        }


def print_report(report: dict, starting_capital: Decimal):
    """Print the final performance summary."""
    print(f"\n{'='*70}\nV4 ASYNC BACKTEST PERFORMANCE REPORT\n{'='*70}")
    pnl_pct = (report["total_pnl"] / float(starting_capital)) * 100
    print(f"  Total P&L:          ${report['total_pnl']:,.2f} ({pnl_pct:+.2f}%)")
    print(f"  Trades:             {report['total_trades']} ({report['closed_trades']} closed, {report['open_trades']} open)")
    print(f"  Win Rate:           {report['win_rate']:.1f}% ({report['wins']} wins, {report['losses']} losses)")
    print(f"  Max Drawdown:       {report['max_drawdown_pct']:.2f}%")
    print(f"  Avg Slippage:       {report['avg_slippage_pct']:.4f}%")
    print(f"  Total Decisions:    {report['total_decisions']}")
    print("-" * 70)
    
    if report['total_pnl'] > 0:
        print("VERDICT: PROFITABLE ✅")
    else:
        print("VERDICT: UNPROFITABLE ❌")
    print("=" * 70)


async def main_async():
    parser = argparse.ArgumentParser(description="V4 Backtester - Async Version")
    parser.add_argument("--data", type=str, default=None, help="Path to historical OHLCV CSV")
    parser.add_argument("--days", type=int, default=90, help="Days of synthetic data")
    parser.add_argument("--capital", type=float, default=10000.0, help="Starting capital")
    parser.add_argument("--strategy", type=str, default="donchian", help="Strategy choice")
    parser.add_argument("--real", action="store_true", help="Use real BTC-USD data from Yahoo Finance")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-15")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--resample", type=str, default=None)
    args = parser.parse_args()

    # Data loading/generation
    if args.real:
        print(f"Loading REAL BTC-USD data: {args.start} to {args.end} ({args.interval})")
        data = load_or_fetch_btc(start=args.start, end=args.end, interval=args.interval)
    elif args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Error: Data file not found: {args.data}")
            sys.exit(1)
        print(f"Loading historical data from {args.data}...")
        data, _ = load_ohlcv(args.data)
    else:
        print(f"Generating {args.days} days of synthetic data...")
        data = generate_synthetic_data(days=args.days)

    if args.resample:
        print(f"\nTHE ZOOM OUT: Resampling to {args.resample}...")
        data = resample_ohlcv(data, args.resample)

    db_path = Path(__file__).parent.parent / "data" / "v4_backtest.db"
    db_path.parent.mkdir(exist_ok=True)
    
    # Run Chronos
    backtester = BacktestEngine(
        db_path=str(db_path), 
        starting_capital=Decimal(str(args.capital)), 
        strategy_name=args.strategy
    )
    
    report = await backtester.run(data)
    print_report(report, Decimal(str(args.capital)))


if __name__ == "__main__":
    asyncio.run(main_async())