#!/usr/bin/env python3
"""
Price Zone Filter Backtest Analysis

Analyzes how the price zone filter would have affected the 94 historical trades.
Projects win rate lift and P&L improvement.

Usage:
    python scripts/price_zone_backtest_analysis.py
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiosqlite
from strategy.price_zone import PriceZoneFilter, PriceZone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class PriceZoneBacktestAnalyzer:
    """Analyzes backtest trades through the price zone filter lens."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.filter = PriceZoneFilter()

    async def get_backtest_trades(self) -> List[Dict[str, Any]]:
        """Get all trades from the backtest database."""
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT
                    t.trade_id,
                    t.symbol,
                    t.entry_timestamp,
                    CAST(t.entry_price AS REAL) as entry_price,
                    t.exit_timestamp,
                    CAST(t.exit_price AS REAL) as exit_price,
                    CAST(t.net_pnl AS REAL) as pnl,
                    CAST(t.realized_pnl_percent AS REAL) as pnl_pct,
                    t.is_winner,
                    t.is_valid
                FROM trades t
                WHERE t.status = 'closed' AND t.is_valid = 1
                ORDER BY t.entry_timestamp ASC
            """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    def analyze_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single trade through the price zone filter."""
        symbol = trade["symbol"]
        entry_price = trade["entry_price"]

        # Get price zone at entry
        zone = self.filter.get_zone(
            symbol=symbol,
            current_price=entry_price,
            btc_price=entry_price if "BTC" in symbol else None
        )

        return {
            **trade,
            "zone": zone.zone,
            "zone_reason": zone.reason,
            "ath_ratio": zone.ath_ratio,
            "is_distribution": zone.is_distribution,
            "is_caution": zone.is_caution,
            "would_be_blocked": zone.is_distribution,
        }

    async def run_analysis(self) -> Dict[str, Any]:
        """Run the full analysis."""
        logger.info("Loading backtest trades...")
        trades = await self.get_backtest_trades()
        logger.info(f"Found {len(trades)} trades to analyze")

        if not trades:
            return {"error": "No trades found"}

        # Analyze each trade
        analyzed = [self.analyze_trade(t) for t in trades]

        # Separate by zone
        distribution_trades = [t for t in analyzed if t["is_distribution"]]
        caution_trades = [t for t in analyzed if t["is_caution"] and not t["is_distribution"]]
        accumulation_trades = [t for t in analyzed if not t["is_caution"]]

        # Calculate stats for each zone
        def calc_stats(trades_list: List[Dict], label: str) -> Dict[str, Any]:
            if not trades_list:
                return {
                    "label": label,
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                }

            wins = sum(1 for t in trades_list if t["pnl"] > 0)
            losses = len(trades_list) - wins
            total_pnl = sum(t["pnl"] for t in trades_list)
            avg_pnl = total_pnl / len(trades_list)
            win_rate = wins / len(trades_list) * 100

            return {
                "label": label,
                "count": len(trades_list),
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
            }

        all_stats = calc_stats(analyzed, "All Trades")
        dist_stats = calc_stats(distribution_trades, "Distribution (>90% ATH)")
        caution_stats = calc_stats(caution_trades, "Caution (60-90% ATH)")
        accum_stats = calc_stats(accumulation_trades, "Accumulation (<60% ATH)")

        # Calculate what happens if we filter out distribution zone
        filtered_trades = [t for t in analyzed if not t["is_distribution"]]
        filtered_stats = calc_stats(filtered_trades, "After Filtering")

        # Calculate improvement
        improvement = {
            "trades_filtered": len(distribution_trades),
            "win_rate_before": all_stats["win_rate"],
            "win_rate_after": filtered_stats["win_rate"],
            "win_rate_delta": filtered_stats["win_rate"] - all_stats["win_rate"],
            "pnl_before": all_stats["total_pnl"],
            "pnl_after": filtered_stats["total_pnl"],
            "pnl_saved": all_stats["total_pnl"] - filtered_stats["total_pnl"],
            "losses_avoided": dist_stats["losses"],
        }

        return {
            "total_trades": len(analyzed),
            "all_trades": all_stats,
            "by_zone": {
                "distribution": dist_stats,
                "caution": caution_stats,
                "accumulation": accum_stats,
            },
            "filtered_result": filtered_stats,
            "improvement": improvement,
            "distribution_trades": [
                {
                    "entry_timestamp": t["entry_timestamp"],
                    "symbol": t["symbol"],
                    "entry_price": t["entry_price"],
                    "pnl": t["pnl"],
                    "zone_reason": t["zone_reason"],
                }
                for t in distribution_trades
            ],
        }


async def main():
    # Check both possible database locations
    db_paths = [
        Path(__file__).parent.parent / "data" / "v4_backtest.db",
        Path(__file__).parent.parent / "v4_backtest.db",
    ]

    db_path = None
    for path in db_paths:
        if path.exists():
            db_path = path
            break

    if not db_path:
        print("ERROR: No backtest database found")
        print("Run: python scripts/run_backtest.py first")
        return

    analyzer = PriceZoneBacktestAnalyzer(str(db_path))
    results = await analyzer.run_analysis()

    print("\n" + "=" * 70)
    print("PRICE ZONE FILTER BACKTEST ANALYSIS")
    print("=" * 70)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    # All trades stats
    all_stats = results["all_trades"]
    print(f"\n📊 ALL TRADES ({all_stats['count']})")
    print(f"   Win Rate: {all_stats['win_rate']:.1f}%")
    print(f"   Total P&L: ${all_stats['total_pnl']:,.2f}")
    print(f"   Avg P&L: ${all_stats['avg_pnl']:.2f}")

    # By zone breakdown
    print(f"\n📈 BREAKDOWN BY PRICE ZONE:")
    print("-" * 50)

    for zone_name, stats in results["by_zone"].items():
        if stats["count"] > 0:
            print(f"\n   {stats['label']}:")
            print(f"      Trades: {stats['count']}")
            print(f"      Win Rate: {stats['win_rate']:.1f}%")
            print(f"      Total P&L: ${stats['total_pnl']:,.2f}")
            print(f"      Avg P&L: ${stats['avg_pnl']:.2f}")

    # Filtered results
    filtered = results["filtered_result"]
    improvement = results["improvement"]

    print(f"\n✅ AFTER APPLYING ZONE FILTER (removing distribution zone):")
    print("-" * 50)
    print(f"   Trades Filtered: {improvement['trades_filtered']}")
    print(f"   Remaining Trades: {filtered['count']}")
    print(f"   New Win Rate: {filtered['win_rate']:.1f}% (was {improvement['win_rate_before']:.1f}%)")
    print(f"   Win Rate LIFT: +{improvement['win_rate_delta']:.1f}%")
    print(f"   New P&L: ${filtered['total_pnl']:,.2f}")

    # List distribution trades
    dist_trades = results.get("distribution_trades", [])
    if dist_trades:
        print(f"\n⚠️ DISTRIBUTION ZONE TRADES THAT WOULD BE BLOCKED:")
        print("-" * 50)
        for t in dist_trades:
            pnl_str = f"${t['pnl']:+,.2f}"
            outcome = "WIN ✓" if t['pnl'] > 0 else "LOSS ✗"
            entry_date = t['entry_timestamp'][:10] if t['entry_timestamp'] else "unknown"
            print(f"   {entry_date} | {t['symbol']} @ ${t['entry_price']:,.2f} | "
                  f"{pnl_str:>12} | {outcome}")

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if improvement["trades_filtered"] > 0:
        avg_loss_prevented = results["by_zone"]["distribution"]["total_pnl"] / improvement["trades_filtered"]
        print(f"   🎯 Would block {improvement['trades_filtered']} trades in distribution zone")
        print(f"   📈 Win rate improvement: +{improvement['win_rate_delta']:.1f}%")
        print(f"   💰 Avg loss per blocked trade: ${avg_loss_prevented:.2f}")

        if improvement["pnl_saved"] > 0:
            print(f"   ✅ Money saved by filter: ${improvement['pnl_saved']:,.2f}")
        else:
            print(f"   ⚠️ Opportunity cost (wins blocked): ${-improvement['pnl_saved']:,.2f}")
    else:
        print("   No trades would be blocked by the distribution zone filter")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
