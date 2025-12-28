#!/usr/bin/env python3
"""
MEXC Fee Simulation with Zone Filter

Simulates the backtest with:
1. MEXC fees (0.05% maker/taker) instead of Coinbase (0.4%)
2. Price zone filter applied (no trades in distribution zone)
3. Reduced slippage estimate (50% of original due to higher liquidity)

Projects net P&L improvement.
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiosqlite
from strategy.price_zone import PriceZoneFilter

# Fee structures
COINBASE_FEES = {
    "entry": 0.004,      # 0.4%
    "exit": 0.004,       # 0.4%
    "slippage": 0.002,   # 0.2% estimated
}

MEXC_FEES = {
    "entry": 0.0005,     # 0.05%
    "exit": 0.0005,      # 0.05%
    "slippage": 0.001,   # 0.1% (better liquidity)
}


async def load_trades(db_path: str) -> List[Dict[str, Any]]:
    """Load all closed trades from backtest."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT
                trade_id,
                symbol,
                entry_timestamp,
                CAST(entry_price AS REAL) as entry_price,
                CAST(exit_price AS REAL) as exit_price,
                CAST(quantity AS REAL) as quantity,
                CAST(realized_pnl AS REAL) as gross_pnl,
                CAST(total_commission AS REAL) as commission,
                CAST(total_slippage AS REAL) as slippage,
                CAST(net_pnl AS REAL) as net_pnl,
                is_winner
            FROM trades
            WHERE status = 'closed' AND is_valid = 1
            ORDER BY entry_timestamp
        """)
        return [dict(row) for row in await cursor.fetchall()]


def apply_zone_filter(trades: List[Dict], filter: PriceZoneFilter) -> List[Dict]:
    """Filter out trades that would be blocked by zone filter."""
    filtered = []
    blocked = []

    for trade in trades:
        zone = filter.get_zone(trade["symbol"], trade["entry_price"])
        trade["zone"] = zone.zone
        trade["would_block"] = zone.is_distribution

        if zone.is_distribution:
            blocked.append(trade)
        else:
            filtered.append(trade)

    return filtered, blocked


def recalculate_with_fees(trades: List[Dict], fee_structure: Dict) -> Dict[str, Any]:
    """Recalculate P&L with different fee structure."""
    total_gross = 0
    total_commission = 0
    total_slippage = 0
    total_net = 0
    wins = 0
    losses = 0

    for trade in trades:
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]
        qty = trade["quantity"]

        # Gross P&L (price difference only)
        gross_pnl = (exit_price - entry_price) * qty

        # New fees
        entry_value = entry_price * qty
        exit_value = exit_price * qty

        commission = entry_value * fee_structure["entry"] + exit_value * fee_structure["exit"]
        slippage = entry_value * fee_structure["slippage"] + exit_value * fee_structure["slippage"]

        net_pnl = gross_pnl - commission - slippage

        total_gross += gross_pnl
        total_commission += commission
        total_slippage += slippage
        total_net += net_pnl

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100 if trades else 0,
        "gross_pnl": total_gross,
        "commission": total_commission,
        "slippage": total_slippage,
        "net_pnl": total_net,
        "avg_pnl": total_net / len(trades) if trades else 0,
    }


async def main():
    db_path = Path(__file__).parent.parent / "data" / "v4_backtest.db"

    if not db_path.exists():
        print("ERROR: Backtest database not found")
        return

    # Load trades
    trades = await load_trades(str(db_path))
    print(f"\nLoaded {len(trades)} trades from backtest\n")

    # Apply zone filter
    zone_filter = PriceZoneFilter()
    filtered_trades, blocked_trades = apply_zone_filter(trades, zone_filter)

    print("=" * 70)
    print("MEXC SIMULATION WITH ZONE FILTER")
    print("=" * 70)

    # Scenario 1: Original (Coinbase fees, no filter)
    original = recalculate_with_fees(trades, COINBASE_FEES)

    # Scenario 2: Coinbase with zone filter
    coinbase_filtered = recalculate_with_fees(filtered_trades, COINBASE_FEES)

    # Scenario 3: MEXC fees, no filter
    mexc_all = recalculate_with_fees(trades, MEXC_FEES)

    # Scenario 4: MEXC fees + zone filter (THE TARGET)
    mexc_filtered = recalculate_with_fees(filtered_trades, MEXC_FEES)

    # Display results
    print(f"\n📊 SCENARIO COMPARISON:")
    print("-" * 70)
    print(f"{'Scenario':<35} {'Trades':<8} {'Win%':<8} {'Net P&L':<12} {'Δ vs Orig':<12}")
    print("-" * 70)

    scenarios = [
        ("1. Coinbase (Original)", original, 0),
        ("2. Coinbase + Zone Filter", coinbase_filtered, coinbase_filtered["net_pnl"] - original["net_pnl"]),
        ("3. MEXC (No Filter)", mexc_all, mexc_all["net_pnl"] - original["net_pnl"]),
        ("4. MEXC + Zone Filter ⭐", mexc_filtered, mexc_filtered["net_pnl"] - original["net_pnl"]),
    ]

    for name, stats, delta in scenarios:
        delta_str = f"+${delta:,.2f}" if delta > 0 else f"${delta:,.2f}"
        print(f"{name:<35} {stats['trades']:<8} {stats['win_rate']:<7.1f}% ${stats['net_pnl']:>10,.2f} {delta_str:>12}")

    print("-" * 70)

    # Detailed breakdown for target scenario
    print(f"\n⭐ TARGET SCENARIO: MEXC + Zone Filter")
    print("-" * 50)
    print(f"   Trades Taken:    {mexc_filtered['trades']} (blocked {len(blocked_trades)})")
    print(f"   Win Rate:        {mexc_filtered['win_rate']:.1f}%")
    print(f"   Gross P&L:       ${mexc_filtered['gross_pnl']:,.2f}")
    print(f"   Commission:      ${mexc_filtered['commission']:,.2f} (was ${original['commission']:,.2f})")
    print(f"   Slippage:        ${mexc_filtered['slippage']:,.2f} (was ${original['slippage']:,.2f})")
    print(f"   Net P&L:         ${mexc_filtered['net_pnl']:,.2f}")
    print(f"   Avg P&L/Trade:   ${mexc_filtered['avg_pnl']:.2f}")

    # Total savings
    fee_savings = original['commission'] - mexc_filtered['commission']
    slip_savings = original['slippage'] - mexc_filtered['slippage']
    zone_savings = sum(t['net_pnl'] for t in blocked_trades)
    total_improvement = mexc_filtered['net_pnl'] - original['net_pnl']

    print(f"\n💰 IMPROVEMENT BREAKDOWN:")
    print("-" * 50)
    print(f"   Fee Reduction:     +${fee_savings:,.2f}")
    print(f"   Slippage Reduction:+${slip_savings:,.2f}")
    print(f"   Zone Filter Saves: +${-zone_savings:,.2f}")
    print(f"   ─────────────────────────────")
    print(f"   TOTAL IMPROVEMENT: +${total_improvement:,.2f}")

    if mexc_filtered['net_pnl'] > 0:
        print(f"\n   🎯 NET P&L FLIPS GREEN: ${mexc_filtered['net_pnl']:,.2f}")
    else:
        print(f"\n   ⚠️ Still negative but improved by ${total_improvement:,.2f}")
        breakeven_slippage = mexc_filtered['slippage'] + mexc_filtered['net_pnl']
        print(f"   Need slippage < ${breakeven_slippage:,.2f} to break even")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
