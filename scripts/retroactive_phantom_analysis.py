#!/usr/bin/env python3
"""
Retroactive Phantom Trade Analysis

Analyzes historical HOLD decisions to determine what WOULD have happened
if we'd taken those trades. Populates the phantom_trades table with
regret/relief verdicts.

Usage:
    python scripts/retroactive_phantom_analysis.py [--limit 100] [--grade B]
"""

import sys
import argparse
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiosqlite
from data.loader import load_or_fetch_btc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class PhantomAnalyzer:
    """Analyzes HOLD decisions to create phantom trade outcomes."""

    # Verdict thresholds
    REGRET_THRESHOLD = 1.0    # >1% gain = regret (should have traded)
    RELIEF_THRESHOLD = -1.0   # <-1% loss = relief (glad we skipped)
    # Between -1% and +1% = neutral

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.price_cache: Dict[str, Any] = {}

    async def get_hold_decisions(
        self,
        limit: int = 100,
        grade_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get HOLD decisions that haven't been phantom-analyzed yet."""
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row

            # Get decisions not already in phantom_trades
            query = """
                SELECT d.decision_id, d.timestamp, d.symbol, d.signal_values
                FROM decisions d
                LEFT JOIN phantom_trades p ON d.decision_id = p.decision_id
                WHERE d.result = 'signal_hold'
                AND p.phantom_id IS NULL
            """
            params = []

            if grade_filter:
                query += " AND json_extract(d.signal_values, '$.grade') = ?"
                params.append(grade_filter)

            query += " ORDER BY d.timestamp ASC LIMIT ?"
            params.append(limit)

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                data = dict(row)
                if data.get('signal_values'):
                    data['signal_values'] = json.loads(data['signal_values'])
                results.append(data)
            return results

    async def load_price_data(self, symbol: str, start_date: str, end_date: str) -> None:
        """Load price data for analysis."""
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.price_cache:
            return

        try:
            # For now, focus on BTC (most liquid, most data)
            if "BTC" in symbol:
                df = load_or_fetch_btc(start=start_date, end=end_date, interval="1h")
                self.price_cache[cache_key] = df
                logger.info(f"Loaded {len(df)} bars for {symbol}")
            else:
                logger.warning(f"Price data not available for {symbol}, skipping")
        except Exception as e:
            logger.error(f"Failed to load price data for {symbol}: {e}")

    def calculate_phantom_outcome(
        self,
        entry_price: float,
        entry_time: datetime,
        signal_values: Dict[str, Any],
        price_data: Any
    ) -> Dict[str, Any]:
        """
        Calculate what would have happened if we'd taken this trade.

        Returns outcome dict with prices, stop hits, and verdict.
        """
        result = {
            "price_after_1h": None,
            "price_after_4h": None,
            "price_after_24h": None,
            "price_after_48h": None,
            "would_hit_chandelier": False,
            "would_hit_hard_stop": False,
            "chandelier_hit_time": None,
            "hard_stop_hit_time": None,
            "max_favorable_excursion": 0.0,
            "max_adverse_excursion": 0.0,
            "phantom_exit_price": None,
            "phantom_pnl_percent": None,
            "duration_to_exit_hours": None,
            "verdict": "neutral",
            "verdict_reason": "Insufficient data"
        }

        if price_data is None or len(price_data) == 0:
            return result

        # Get ATR for chandelier calculation (use 3% as fallback)
        atr_pct = signal_values.get('daily_atr', 0) or 3.0
        chandelier_stop = entry_price * (1 - atr_pct * 3 / 100)  # 3x ATR
        hard_stop = entry_price * 0.95  # 5% hard stop

        result["chandelier_stop_price"] = chandelier_stop
        result["hard_stop_price"] = hard_stop

        # Track through price data
        highest_high = entry_price
        lowest_low = entry_price
        exit_price = None
        exit_reason = None
        exit_time = None

        try:
            # Find rows after entry_time
            if hasattr(price_data, 'index'):
                # It's a DataFrame with datetime index
                future_data = price_data[price_data.index >= entry_time]
            else:
                future_data = price_data

            if len(future_data) == 0:
                return result

            for idx, row in enumerate(future_data.itertuples()):
                bar_time = row.Index if hasattr(row, 'Index') else entry_time + timedelta(hours=idx)
                high = float(row.high)
                low = float(row.low)
                close = float(row.close)

                hours_elapsed = (bar_time - entry_time).total_seconds() / 3600

                # Track MFE/MAE
                if high > highest_high:
                    highest_high = high
                    # Update chandelier stop (ratchet up)
                    chandelier_stop = highest_high * (1 - atr_pct * 3 / 100)
                if low < lowest_low:
                    lowest_low = low

                # Record milestone prices
                if hours_elapsed >= 1 and result["price_after_1h"] is None:
                    result["price_after_1h"] = close
                if hours_elapsed >= 4 and result["price_after_4h"] is None:
                    result["price_after_4h"] = close
                if hours_elapsed >= 24 and result["price_after_24h"] is None:
                    result["price_after_24h"] = close
                if hours_elapsed >= 48 and result["price_after_48h"] is None:
                    result["price_after_48h"] = close

                # Check stop hits (if not already exited)
                if exit_price is None:
                    if low <= hard_stop:
                        exit_price = hard_stop
                        exit_reason = "hard_stop"
                        exit_time = bar_time
                        result["would_hit_hard_stop"] = True
                        result["hard_stop_hit_time"] = bar_time.isoformat()
                    elif low <= chandelier_stop:
                        exit_price = chandelier_stop
                        exit_reason = "chandelier"
                        exit_time = bar_time
                        result["would_hit_chandelier"] = True
                        result["chandelier_hit_time"] = bar_time.isoformat()

                # Stop after 48h of data
                if hours_elapsed >= 48:
                    break

            # Calculate outcomes
            result["max_favorable_excursion"] = (highest_high - entry_price) / entry_price * 100
            result["max_adverse_excursion"] = (entry_price - lowest_low) / entry_price * 100

            # Determine exit price and P&L
            if exit_price is not None:
                result["phantom_exit_price"] = exit_price
                result["phantom_pnl_percent"] = (exit_price - entry_price) / entry_price * 100
                result["duration_to_exit_hours"] = (exit_time - entry_time).total_seconds() / 3600
            elif result["price_after_48h"] is not None:
                # Still holding after 48h - use 48h price
                result["phantom_exit_price"] = result["price_after_48h"]
                result["phantom_pnl_percent"] = (result["price_after_48h"] - entry_price) / entry_price * 100
                result["duration_to_exit_hours"] = 48.0
            elif result["price_after_24h"] is not None:
                # Use 24h if 48h not available
                result["phantom_exit_price"] = result["price_after_24h"]
                result["phantom_pnl_percent"] = (result["price_after_24h"] - entry_price) / entry_price * 100
                result["duration_to_exit_hours"] = 24.0

            # Determine verdict
            if result["phantom_pnl_percent"] is not None:
                pnl = result["phantom_pnl_percent"]
                if pnl > self.REGRET_THRESHOLD:
                    result["verdict"] = "regret"
                    result["verdict_reason"] = f"Would have gained {pnl:.2f}%"
                elif pnl < self.RELIEF_THRESHOLD:
                    result["verdict"] = "relief"
                    result["verdict_reason"] = f"Would have lost {abs(pnl):.2f}%"
                else:
                    result["verdict"] = "neutral"
                    result["verdict_reason"] = f"Marginal outcome: {pnl:+.2f}%"

        except Exception as e:
            logger.error(f"Error calculating phantom outcome: {e}")
            result["verdict_reason"] = f"Calculation error: {e}"

        return result

    async def create_phantom_trade(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> str:
        """Insert phantom trade into database."""
        import uuid
        phantom_id = str(uuid.uuid4())

        signal_values = decision.get('signal_values', {})
        entry_price = signal_values.get('current_price', 0)
        grade = signal_values.get('grade', 'C')

        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("""
                INSERT INTO phantom_trades (
                    phantom_id, decision_id, symbol, timestamp,
                    setup_grade, signal_type, hypothetical_entry,
                    price_after_1h, price_after_4h, price_after_24h, price_after_48h,
                    chandelier_stop_price, hard_stop_price,
                    would_hit_chandelier, would_hit_hard_stop,
                    chandelier_hit_time, hard_stop_hit_time,
                    max_favorable_excursion, max_adverse_excursion,
                    phantom_exit_price, phantom_pnl_percent, duration_to_exit_hours,
                    verdict, verdict_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                phantom_id,
                decision['decision_id'],
                decision['symbol'],
                decision['timestamp'],
                grade,
                'long',  # Currently only long signals
                entry_price,
                outcome.get('price_after_1h'),
                outcome.get('price_after_4h'),
                outcome.get('price_after_24h'),
                outcome.get('price_after_48h'),
                outcome.get('chandelier_stop_price'),
                outcome.get('hard_stop_price'),
                1 if outcome.get('would_hit_chandelier') else 0,
                1 if outcome.get('would_hit_hard_stop') else 0,
                outcome.get('chandelier_hit_time'),
                outcome.get('hard_stop_hit_time'),
                outcome.get('max_favorable_excursion'),
                outcome.get('max_adverse_excursion'),
                outcome.get('phantom_exit_price'),
                outcome.get('phantom_pnl_percent'),
                outcome.get('duration_to_exit_hours'),
                outcome.get('verdict', 'pending'),
                outcome.get('verdict_reason'),
                datetime.utcnow().isoformat()
            ))
            await db.commit()

        return phantom_id

    async def run_analysis(
        self,
        limit: int = 100,
        grade_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run retroactive analysis on HOLD decisions."""
        logger.info(f"Starting retroactive phantom analysis (limit={limit}, grade={grade_filter})")

        # Get decisions to analyze
        decisions = await self.get_hold_decisions(limit, grade_filter)
        logger.info(f"Found {len(decisions)} decisions to analyze")

        if not decisions:
            return {"analyzed": 0, "regrets": 0, "reliefs": 0, "neutral": 0}

        # Group by symbol and date range
        symbols = set(d['symbol'] for d in decisions)

        # Load price data
        min_date = min(d['timestamp'][:10] for d in decisions)
        max_date = datetime.utcnow().strftime("%Y-%m-%d")

        for symbol in symbols:
            if "BTC" in symbol:
                await self.load_price_data(symbol, min_date, max_date)

        # Process each decision
        results = {"analyzed": 0, "regrets": 0, "reliefs": 0, "neutral": 0, "skipped": 0}

        for i, decision in enumerate(decisions):
            try:
                signal_values = decision.get('signal_values', {})
                entry_price = signal_values.get('current_price', 0)

                if not entry_price:
                    results["skipped"] += 1
                    continue

                # Parse timestamp
                ts_str = decision['timestamp']
                entry_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00').replace('+00:00', ''))

                # Get price data for this symbol
                symbol = decision['symbol']
                cache_key = None
                price_data = None

                for key in self.price_cache:
                    if symbol.replace("-USD", "") in key:
                        cache_key = key
                        price_data = self.price_cache[key]
                        break

                # Calculate outcome
                outcome = self.calculate_phantom_outcome(
                    entry_price=entry_price,
                    entry_time=entry_time,
                    signal_values=signal_values,
                    price_data=price_data
                )

                # Create phantom trade
                phantom_id = await self.create_phantom_trade(decision, outcome)

                # Track results
                results["analyzed"] += 1
                verdict = outcome.get('verdict', 'neutral')
                if verdict == 'regret':
                    results["regrets"] += 1
                elif verdict == 'relief':
                    results["reliefs"] += 1
                else:
                    results["neutral"] += 1

                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(decisions)} decisions...")

            except Exception as e:
                logger.error(f"Error processing decision {decision.get('decision_id')}: {e}")
                results["skipped"] += 1

        return results


async def main():
    parser = argparse.ArgumentParser(description="Retroactive Phantom Trade Analysis")
    parser.add_argument("--limit", type=int, default=100, help="Max decisions to analyze")
    parser.add_argument("--grade", type=str, default=None, help="Filter by grade (A+, A, B, C)")
    parser.add_argument("--db", type=str, default="data/v4_live_paper.db", help="Database path")
    args = parser.parse_args()

    db_path = Path(__file__).parent.parent / args.db

    analyzer = PhantomAnalyzer(str(db_path))
    results = await analyzer.run_analysis(limit=args.limit, grade_filter=args.grade)

    print(f"\n{'='*60}")
    print("RETROACTIVE PHANTOM ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Analyzed:  {results['analyzed']}")
    print(f"  Regrets:   {results['regrets']} (should have traded)")
    print(f"  Reliefs:   {results['reliefs']} (glad we skipped)")
    print(f"  Neutral:   {results['neutral']} (marginal)")
    print(f"  Skipped:   {results.get('skipped', 0)}")

    if results['analyzed'] > 0:
        regret_rate = results['regrets'] / results['analyzed'] * 100
        relief_rate = results['reliefs'] / results['analyzed'] * 100
        print(f"\n  Regret Rate: {regret_rate:.1f}%")
        print(f"  Relief Rate: {relief_rate:.1f}%")

        if regret_rate > 30:
            print(f"\n  ⚠️  HIGH REGRET RATE - Consider loosening filters")
        elif relief_rate > 70:
            print(f"\n  ✅ HIGH RELIEF RATE - Filters are protecting capital")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
