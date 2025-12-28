#!/usr/bin/env python3
"""
Backfill Reflections Script

Generates reflections for past trades that don't have them.
This bootstraps Argus's memory from historical trading data.

Usage:
    python scripts/backfill_reflections.py [--days 30] [--limit 100] [--db PATH]
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    # Load env vars manually if dotenv not available
    import os
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Backfill reflections from past trades")
    parser.add_argument("--days", type=int, default=90, help="Look back N days (default: 90)")
    parser.add_argument("--limit", type=int, default=200, help="Max trades to process (default: 200)")
    parser.add_argument("--db", type=str, default=None, help="Database path (default: data/v4_live_paper.db)")
    args = parser.parse_args()

    # Determine database path
    if args.db:
        db_path = Path(args.db)
    else:
        db_path = PROJECT_ROOT / "data" / "v4_live_paper.db"

    if not db_path.exists():
        # Try other databases
        for alt in ["v4_backtest.db"]:
            alt_path = PROJECT_ROOT / "data" / alt
            if alt_path.exists():
                db_path = alt_path
                break

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.info("Available databases:")
        for f in (PROJECT_ROOT / "data").glob("*.db"):
            logger.info(f"  - {f.name}")
        return

    logger.info("=" * 60)
    logger.info("ARGUS REFLEXION BACKFILL")
    logger.info("=" * 60)
    logger.info(f"Database: {db_path}")
    logger.info(f"Look back: {args.days} days")
    logger.info(f"Max trades: {args.limit}")
    logger.info("")

    # Check how many closed trades exist
    import aiosqlite
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row

        # Count total closed trades
        cursor = await db.execute("""
            SELECT COUNT(*) as count FROM trades WHERE status = 'closed'
        """)
        row = await cursor.fetchone()
        total_closed = row["count"] if row else 0

        # Count trades without reflections
        cursor = await db.execute("""
            SELECT COUNT(*) as count
            FROM trades t
            LEFT JOIN reflections r ON t.trade_id = r.trade_id
            WHERE t.status = 'closed' AND r.reflection_id IS NULL
        """)
        row = await cursor.fetchone()
        without_reflections = row["count"] if row else 0

        # Count existing reflections
        try:
            cursor = await db.execute("SELECT COUNT(*) as count FROM reflections")
            row = await cursor.fetchone()
            existing_reflections = row["count"] if row else 0
        except:
            existing_reflections = 0

    logger.info(f"Total closed trades: {total_closed}")
    logger.info(f"Existing reflections: {existing_reflections}")
    logger.info(f"Trades needing reflection: {without_reflections}")
    logger.info("")

    if without_reflections == 0:
        logger.info("All trades already have reflections!")
        return

    # Initialize the reflexion engine
    from src.learning.reflexion import ReflexionEngine

    engine = ReflexionEngine(str(db_path))
    await engine.initialize()

    logger.info("Starting backfill...")
    logger.info("")

    # Run backfill
    count = await engine.backfill_reflections(days=args.days, limit=args.limit)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"BACKFILL COMPLETE: {count} reflections generated")
    logger.info("=" * 60)

    # Show summary
    if count > 0:
        summary = await engine.get_lessons_summary(days=args.days)

        logger.info("")
        logger.info("LEARNING SUMMARY:")
        logger.info(f"  Total reflections: {summary['stats']['total_reflections']}")
        logger.info(f"  From wins: {summary['stats']['from_wins']}")
        logger.info(f"  From losses: {summary['stats']['from_losses']}")
        logger.info(f"  Avg confidence: {summary['stats']['avg_confidence']:.0%}")

        if summary.get("top_lessons"):
            logger.info("")
            logger.info("TOP LESSONS LEARNED:")
            for i, lesson in enumerate(summary["top_lessons"][:5], 1):
                conf = lesson["confidence"]
                text = lesson["lesson_learned"][:60]
                logger.info(f"  {i}. [{conf:.0%}] {text}...")


if __name__ == "__main__":
    asyncio.run(main())
