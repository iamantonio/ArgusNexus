#!/usr/bin/env python3
"""
Backfill Reflections Script (Synchronous Version)

Generates reflections for past trades using sqlite3 (no async dependencies).
This bootstraps Argus's memory from historical trading data.

Usage:
    python scripts/backfill_reflections_sync.py [--days 90] [--limit 100]
"""

import sqlite3
import json
import os
import sys
import argparse
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load env manually
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


class ReflectionType(Enum):
    WINNING_TRADE = "winning_trade"
    LOSING_TRADE = "losing_trade"
    MISSED_OPPORTUNITY = "missed_opportunity"
    FALSE_SIGNAL = "false_signal"
    TIMING_ERROR = "timing_error"
    SIZE_ERROR = "size_error"
    EXIT_ERROR = "exit_error"
    MARKET_REGIME = "market_regime"


class MarketRegime(Enum):
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    STRONG_UPTREND = "strong_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    UNKNOWN = "unknown"


# Schema for reflections table
REFLECTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS reflections (
    reflection_id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_winner INTEGER NOT NULL,
    pnl_percent REAL NOT NULL,
    duration_hours REAL NOT NULL,
    market_regime TEXT NOT NULL,
    entry_atr REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    reflection_type TEXT NOT NULL,
    what_happened TEXT NOT NULL,
    what_expected TEXT NOT NULL,
    lesson_learned TEXT NOT NULL,
    action_items TEXT NOT NULL,
    confidence REAL NOT NULL,
    applies_to_regimes TEXT NOT NULL,
    applies_to_symbols TEXT NOT NULL,
    context_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reflections_symbol ON reflections(symbol);
CREATE INDEX IF NOT EXISTS idx_reflections_regime ON reflections(market_regime);
CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(reflection_type);
CREATE INDEX IF NOT EXISTS idx_reflections_winner ON reflections(is_winner);
CREATE INDEX IF NOT EXISTS idx_reflections_confidence ON reflections(confidence);
"""


def _safe_float(val, default=0) -> float:
    """Safely convert a value to float"""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except:
            return default
    return default


def detect_regime(signals: Dict[str, Any], market: Dict[str, Any]) -> MarketRegime:
    """Detect market regime from signals and market context"""
    atr = _safe_float(signals.get("atr")) or _safe_float(market.get("atr")) or 0
    price = _safe_float(signals.get("current_price")) or _safe_float(market.get("price")) or 1
    adx = _safe_float(signals.get("adx")) or _safe_float(market.get("adx")) or 0
    trend = str(signals.get("trend", "") or market.get("trend", "")).lower()

    atr_pct = (atr / price * 100) if price > 0 else 0

    if atr_pct > 3:
        return MarketRegime.HIGH_VOLATILITY
    if adx > 25:
        if "up" in trend or "bull" in trend:
            return MarketRegime.STRONG_UPTREND
        elif "down" in trend or "bear" in trend:
            return MarketRegime.STRONG_DOWNTREND
    if atr_pct < 1:
        return MarketRegime.LOW_VOLATILITY
    if adx < 20:
        return MarketRegime.RANGING

    return MarketRegime.UNKNOWN


def compute_context_hash(regime: str, atr: float, outcome: str, exit_reason: str, duration: float) -> str:
    """Compute hash for similarity matching"""
    vol_bucket = "high" if atr > 0.02 else "normal" if atr > 0.01 else "low"
    dur_bucket = "quick" if duration < 4 else "medium" if duration < 24 else "long"
    context = {
        "regime": regime,
        "volatility_bucket": vol_bucket,
        "outcome": outcome,
        "exit": exit_reason,
        "duration": dur_bucket
    }
    return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:8]


def generate_fallback_reflection(trade: Dict[str, Any]) -> Dict[str, Any]:
    """Generate basic reflection without LLM"""
    is_winner = trade.get("is_winner", 0) == 1
    pnl_pct = float(trade.get("realized_pnl_percent") or 0)
    exit_reason = trade.get("exit_reason", "unknown")
    symbol = trade.get("symbol", "UNKNOWN")

    if is_winner:
        reflection_type = "winning_trade"
        what_happened = f"Trade on {symbol} closed profitably with {pnl_pct:.2f}% gain via {exit_reason}."
        lesson = "Maintain discipline and let winning trades run when conditions are favorable."
        action_items = [
            "Review entry signals for this successful setup",
            "Document market conditions that led to this win",
            "Consider if holding longer would have been beneficial"
        ]
        confidence = 0.7
    else:
        reflection_type = "losing_trade"
        what_happened = f"Trade on {symbol} closed with {pnl_pct:.2f}% loss via {exit_reason}."

        if exit_reason == "stop_loss":
            lesson = "Stop loss protected capital. Review if stop placement was appropriate for volatility."
            action_items = [
                "Analyze if stop was too tight for current volatility",
                "Check if entry timing could be improved",
                "Verify ATR multiplier was appropriate"
            ]
            confidence = 0.75
        elif exit_reason == "take_profit":
            # Losing trade with take profit exit is unusual
            lesson = "Trade closed at take profit but still lost - review commission/slippage impact."
            action_items = [
                "Review fee structure impact on small gains",
                "Consider larger profit targets"
            ]
            confidence = 0.6
        else:
            lesson = "Loss occurred - review entry conditions and risk management."
            action_items = [
                "Review signal quality at entry",
                "Verify position sizing was appropriate",
                "Check if market regime was favorable"
            ]
            confidence = 0.65

    return {
        "reflection_type": reflection_type,
        "what_happened": what_happened,
        "what_expected": "Strategy expected favorable price movement in trade direction.",
        "lesson_learned": lesson,
        "action_items": action_items,
        "confidence": confidence,
        "applies_to_regimes": ["all"],
        "applies_to_symbols": [symbol]
    }


def backfill_reflections(db_path: str, days: int = 90, limit: int = 100) -> int:
    """Generate reflections for past trades"""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Initialize reflections table
    conn.executescript(REFLECTIONS_SCHEMA)
    conn.commit()

    # Get cutoff date
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Find trades without reflections
    cursor = conn.execute("""
        SELECT t.*,
               d.signal_values,
               d.market_context
        FROM trades t
        LEFT JOIN decisions d ON t.entry_decision_id = d.decision_id
        LEFT JOIN reflections r ON t.trade_id = r.trade_id
        WHERE t.status = 'closed'
          AND t.exit_timestamp >= ?
          AND r.reflection_id IS NULL
        ORDER BY t.exit_timestamp DESC
        LIMIT ?
    """, (cutoff, limit))

    trades = cursor.fetchall()

    if not trades:
        print("No trades needing reflections found.")
        return 0

    print(f"Processing {len(trades)} trades...")
    count = 0

    for trade in trades:
        trade_dict = dict(trade)
        trade_id = trade_dict["trade_id"]

        try:
            # Parse signals (may already be dict or JSON string)
            signals = {}
            sv = trade_dict.get("signal_values")
            if sv:
                if isinstance(sv, dict):
                    signals = sv
                elif isinstance(sv, str):
                    try:
                        signals = json.loads(sv)
                    except:
                        pass

            market = {}
            mc = trade_dict.get("market_context")
            if mc:
                if isinstance(mc, dict):
                    market = mc
                elif isinstance(mc, str):
                    try:
                        market = json.loads(mc)
                    except:
                        pass

            # Calculate metrics
            entry_price = float(trade_dict.get("entry_price") or 0)
            exit_price = float(trade_dict.get("exit_price") or entry_price)
            pnl_percent = float(trade_dict.get("realized_pnl_percent") or 0)
            is_winner = trade_dict.get("is_winner", 0) == 1

            # Calculate duration
            entry_ts = datetime.fromisoformat(trade_dict["entry_timestamp"].replace("Z", "+00:00"))
            exit_ts_str = trade_dict.get("exit_timestamp")
            if exit_ts_str:
                exit_ts = datetime.fromisoformat(exit_ts_str.replace("Z", "+00:00"))
            else:
                exit_ts = datetime.now(timezone.utc)
            duration_hours = (exit_ts - entry_ts).total_seconds() / 3600

            # Detect regime
            regime = detect_regime(signals, market)

            # Get ATR
            atr = _safe_float(signals.get("atr")) or _safe_float(market.get("atr")) or 0
            entry_atr = (atr / entry_price) if entry_price > 0 else 0.01

            # Generate reflection
            reflection_data = generate_fallback_reflection(trade_dict)

            # Compute context hash
            outcome = "win" if is_winner else "loss"
            exit_reason = trade_dict.get("exit_reason", "unknown")
            context_hash = compute_context_hash(
                regime.value, entry_atr, outcome, exit_reason, duration_hours
            )

            # Create reflection record
            reflection_id = str(uuid.uuid4())
            created_at = datetime.now(timezone.utc).isoformat()

            conn.execute("""
                INSERT INTO reflections (
                    reflection_id, trade_id, symbol, created_at,
                    is_winner, pnl_percent, duration_hours,
                    market_regime, entry_atr, entry_price, exit_price,
                    exit_reason, reflection_type, what_happened,
                    what_expected, lesson_learned, action_items,
                    confidence, applies_to_regimes, applies_to_symbols,
                    context_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reflection_id,
                trade_id,
                trade_dict.get("symbol", "UNKNOWN"),
                created_at,
                1 if is_winner else 0,
                pnl_percent,
                duration_hours,
                regime.value,
                entry_atr,
                entry_price,
                exit_price,
                exit_reason,
                reflection_data["reflection_type"],
                reflection_data["what_happened"],
                reflection_data["what_expected"],
                reflection_data["lesson_learned"],
                json.dumps(reflection_data["action_items"]),
                reflection_data["confidence"],
                json.dumps(reflection_data["applies_to_regimes"]),
                json.dumps(reflection_data["applies_to_symbols"]),
                context_hash
            ))

            count += 1
            status = "WIN" if is_winner else "LOSS"
            print(f"  [{count}] {trade_dict.get('symbol')}: {status} {pnl_percent:+.2f}% -> {reflection_data['reflection_type']}")

        except Exception as e:
            print(f"  Error processing trade {trade_id}: {e}")
            continue

    conn.commit()
    conn.close()

    return count


def get_learning_summary(db_path: str) -> Dict[str, Any]:
    """Get summary of lessons learned"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Get stats
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(confidence) as avg_conf,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses,
                AVG(pnl_percent) as avg_pnl
            FROM reflections
        """)
        row = cursor.fetchone()

        stats = {
            "total_reflections": row["total"] or 0,
            "avg_confidence": round(row["avg_conf"] or 0, 2),
            "from_wins": row["wins"] or 0,
            "from_losses": row["losses"] or 0,
            "avg_pnl": round(row["avg_pnl"] or 0, 2)
        }

        # Get type distribution
        cursor = conn.execute("""
            SELECT reflection_type, COUNT(*) as count
            FROM reflections
            GROUP BY reflection_type
            ORDER BY count DESC
        """)
        types = {row["reflection_type"]: row["count"] for row in cursor.fetchall()}

        # Get top lessons
        cursor = conn.execute("""
            SELECT lesson_learned, confidence, reflection_type
            FROM reflections
            ORDER BY confidence DESC
            LIMIT 5
        """)
        top_lessons = [dict(row) for row in cursor.fetchall()]

        return {
            "stats": stats,
            "type_distribution": types,
            "top_lessons": top_lessons
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill reflections from past trades")
    parser.add_argument("--days", type=int, default=90, help="Look back N days (default: 90)")
    parser.add_argument("--limit", type=int, default=100, help="Max trades to process (default: 100)")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    args = parser.parse_args()

    # Find database
    if args.db:
        db_path = args.db
    else:
        for name in ["v4_live_paper.db", "v4_backtest.db"]:
            path = PROJECT_ROOT / "data" / name
            if path.exists():
                db_path = str(path)
                break
        else:
            print("No database found in data/")
            return

    print("=" * 60)
    print("ARGUS REFLEXION BACKFILL")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Look back: {args.days} days")
    print(f"Max trades: {args.limit}")
    print()

    # Check current state
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
    total_closed = cursor.fetchone()[0]

    try:
        cursor = conn.execute("SELECT COUNT(*) FROM reflections")
        existing = cursor.fetchone()[0]
    except:
        existing = 0

    conn.close()

    print(f"Total closed trades: {total_closed}")
    print(f"Existing reflections: {existing}")
    print()

    # Run backfill
    count = backfill_reflections(db_path, args.days, args.limit)

    print()
    print("=" * 60)
    print(f"BACKFILL COMPLETE: {count} reflections generated")
    print("=" * 60)

    if count > 0:
        summary = get_learning_summary(db_path)

        print()
        print("LEARNING SUMMARY:")
        print(f"  Total reflections: {summary['stats']['total_reflections']}")
        print(f"  From wins: {summary['stats']['from_wins']}")
        print(f"  From losses: {summary['stats']['from_losses']}")
        print(f"  Avg confidence: {summary['stats']['avg_confidence']:.0%}")

        if summary.get("type_distribution"):
            print()
            print("BY TYPE:")
            for rtype, cnt in summary["type_distribution"].items():
                print(f"  {rtype}: {cnt}")

        if summary.get("top_lessons"):
            print()
            print("TOP LESSONS:")
            for i, lesson in enumerate(summary["top_lessons"], 1):
                conf = lesson["confidence"]
                text = lesson["lesson_learned"][:55]
                print(f"  {i}. [{conf:.0%}] {text}...")


if __name__ == "__main__":
    main()
