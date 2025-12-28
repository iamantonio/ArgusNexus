#!/usr/bin/env python3
"""
Initialize the V4 Truth Engine Database

Run this script to create the database with all tables, indexes, and views.
Safe to run multiple times - uses IF NOT EXISTS.

Usage:
    python scripts/init_truth_db.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from truth.logger import TruthLogger


def main():
    # Database location
    db_path = Path(__file__).parent.parent / "data" / "v4_live_paper.db"

    print(f"Initializing Truth Engine at: {db_path}")

    # Create and initialize
    logger = TruthLogger(str(db_path))
    logger.initialize()

    print("✅ Truth Engine initialized successfully!")
    print()
    print("Tables created:")
    print("  - decisions  (WHY we considered trades)")
    print("  - orders     (WHAT orders were placed)")
    print("  - trades     (P&L OUTCOMES)")
    print()
    print("Views created:")
    print("  - v_trade_audit     (Full trade audit trail)")
    print("  - v_daily_pnl       (Daily P&L summary)")
    print("  - v_risk_rejections (Risk rejection analysis)")
    print()
    print(f"Database ready at: {db_path}")


if __name__ == "__main__":
    main()
