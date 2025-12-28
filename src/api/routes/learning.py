"""
Learning & Reflexion Endpoints

Access Argus's learning system - reflections, lessons, and patterns.
These endpoints expose the Reflexion Layer that enables Argus to
learn from its trading mistakes.

"Those who cannot remember the past are condemned to repeat it."
"""

import aiosqlite
import json
from fastapi import APIRouter, Query, HTTPException
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "v4_live_paper.db"

# Reflexion Engine (lazy loaded)
_reflexion_engine = None


def get_reflexion_engine():
    """Get or create the Reflexion Engine."""
    global _reflexion_engine
    if _reflexion_engine is None:
        try:
            from src.learning.reflexion import ReflexionEngine
            _reflexion_engine = ReflexionEngine(str(DB_PATH))
        except ImportError as e:
            pass
    return _reflexion_engine


class LessonQuery(BaseModel):
    """Query parameters for lesson lookup"""
    symbol: str
    signals: dict = {}
    market_context: dict = {}


# =============================================================================
# REFLECTION ENDPOINTS
# =============================================================================

@router.get("/reflections")
async def get_reflections(
    symbol: Optional[str] = None,
    reflection_type: Optional[str] = None,
    market_regime: Optional[str] = None,
    is_winner: Optional[bool] = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Get reflections (lessons learned) from past trades.

    Filter by symbol, type, regime, outcome, or confidence level.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        query = "SELECT * FROM reflections WHERE confidence >= ?"
        params = [min_confidence]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if reflection_type:
            query += " AND reflection_type = ?"
            params.append(reflection_type)

        if market_regime:
            query += " AND market_regime = ?"
            params.append(market_regime)

        if is_winner is not None:
            query += " AND is_winner = ?"
            params.append(1 if is_winner else 0)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        try:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
        except Exception as e:
            # Table might not exist yet
            return []

        reflections = []
        for row in rows:
            r = dict(row)
            # Parse JSON fields
            try:
                r["action_items"] = json.loads(r["action_items"]) if r["action_items"] else []
            except:
                r["action_items"] = []
            try:
                r["applies_to_regimes"] = json.loads(r["applies_to_regimes"]) if r["applies_to_regimes"] else []
            except:
                r["applies_to_regimes"] = []
            try:
                r["applies_to_symbols"] = json.loads(r["applies_to_symbols"]) if r["applies_to_symbols"] else []
            except:
                r["applies_to_symbols"] = []
            reflections.append(r)

        return reflections


@router.get("/reflections/summary")
async def get_learning_summary(
    symbol: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=365)
):
    """
    Get a summary of lessons learned over a period.

    Returns:
    - Total reflections analyzed
    - Breakdown by reflection type
    - Top lessons by confidence
    - Learning progress metrics
    """
    engine = get_reflexion_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Reflexion Engine not available. Learning module may not be installed."
        )

    try:
        await engine.initialize()
        summary = await engine.get_lessons_summary(
            symbol=symbol,
            days=days,
            min_confidence=0.5
        )
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reflections/types")
async def get_reflection_types():
    """Get breakdown of reflections by type."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        try:
            cursor = await db.execute("""
                SELECT * FROM v_lessons_by_type
            """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            # View might not exist
            return []


@router.get("/reflections/top")
async def get_top_lessons(limit: int = Query(default=10, le=50)):
    """
    Get the highest-confidence lessons learned.

    These are the most reliable insights from past trading.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        try:
            cursor = await db.execute("""
                SELECT * FROM v_top_lessons LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            # View might not exist
            return []


@router.get("/reflections/progress")
async def get_learning_progress(days: int = Query(default=30, ge=1, le=365)):
    """
    Get learning progress over time.

    Shows how many reflections were generated per day.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        try:
            cursor = await db.execute("""
                SELECT * FROM v_learning_progress
                WHERE learn_date >= date('now', '-' || ? || ' days')
            """, (days,))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            # View might not exist
            return []


@router.get("/reflections/{reflection_id}")
async def get_reflection(reflection_id: str):
    """Get a single reflection by ID."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
            "SELECT * FROM reflections WHERE reflection_id = ?",
            (reflection_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Reflection not found")

        r = dict(row)
        # Parse JSON fields
        try:
            r["action_items"] = json.loads(r["action_items"]) if r["action_items"] else []
        except:
            r["action_items"] = []
        try:
            r["applies_to_regimes"] = json.loads(r["applies_to_regimes"]) if r["applies_to_regimes"] else []
        except:
            r["applies_to_regimes"] = []
        try:
            r["applies_to_symbols"] = json.loads(r["applies_to_symbols"]) if r["applies_to_symbols"] else []
        except:
            r["applies_to_symbols"] = []

        return r


# =============================================================================
# LESSON QUERY ENDPOINTS
# =============================================================================

@router.post("/lessons/query")
async def query_relevant_lessons(query: LessonQuery):
    """
    Query lessons relevant to a current trading context.

    This is the main interface for the trading system to incorporate
    past lessons into new decisions.

    Returns:
    - adjusted_confidence: Confidence adjustment factor
    - size_adjustment: Suggested position size multiplier
    - warnings: Lessons that warn against this trade
    - supporting_lessons: Lessons that support this trade
    - cautionary_lessons: Lessons that caution against it
    """
    engine = get_reflexion_engine()
    if engine is None:
        return {
            "adjusted_confidence": 1.0,
            "size_adjustment": 1.0,
            "warnings": [],
            "supporting_lessons": [],
            "cautionary_lessons": [],
            "lesson_count": 0,
            "message": "Reflexion Engine not available"
        }

    try:
        await engine.initialize()
        result = await engine.apply_lessons_to_decision(
            symbol=query.symbol,
            current_signals=query.signals,
            current_market=query.market_context,
            proposed_action="long"
        )
        return result
    except Exception as e:
        return {
            "adjusted_confidence": 1.0,
            "size_adjustment": 1.0,
            "warnings": [str(e)],
            "supporting_lessons": [],
            "cautionary_lessons": [],
            "lesson_count": 0,
            "error": str(e)
        }


# =============================================================================
# TRADE REFLECTION ENDPOINTS
# =============================================================================

@router.post("/trades/{trade_id}/reflect")
async def generate_reflection(
    trade_id: str,
    force: bool = Query(default=False, description="Force regenerate reflection")
):
    """
    Generate a reflection for a completed trade.

    This analyzes the trade outcome and extracts actionable lessons.
    Called automatically when trades close, but can be triggered manually.
    """
    engine = get_reflexion_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Reflexion Engine not available"
        )

    try:
        await engine.initialize()
        reflection = await engine.reflect_on_trade(trade_id, force=force)

        if reflection is None:
            raise HTTPException(
                status_code=404,
                detail="Trade not found or reflection generation failed"
            )

        return {
            "reflection_id": reflection.reflection_id,
            "trade_id": trade_id,
            "lesson_learned": reflection.lesson_learned,
            "reflection_type": reflection.reflection_type.value,
            "confidence": reflection.confidence,
            "action_items": reflection.action_items
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reflections/backfill")
async def backfill_reflections(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=100, le=500)
):
    """
    Generate reflections for past trades that don't have them.

    Useful for bootstrapping Argus's memory from historical trades.
    """
    engine = get_reflexion_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Reflexion Engine not available"
        )

    try:
        await engine.initialize()
        count = await engine.backfill_reflections(days=days, limit=limit)
        return {
            "reflections_generated": count,
            "period_days": days,
            "message": f"Generated {count} reflections from trades in the last {days} days"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PATTERN ENDPOINTS
# =============================================================================

@router.get("/patterns")
async def get_patterns(
    min_occurrences: int = Query(default=2, ge=1),
    min_confidence: float = Query(default=0.5, ge=0.0, le=1.0),
    limit: int = Query(default=20, le=100)
):
    """
    Get aggregated patterns from multiple reflections.

    Patterns are higher-level insights derived from clusters
    of similar lessons.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        try:
            cursor = await db.execute("""
                SELECT * FROM reflection_patterns
                WHERE occurrence_count >= ?
                  AND confidence >= ?
                ORDER BY confidence DESC, occurrence_count DESC
                LIMIT ?
            """, (min_occurrences, min_confidence, limit))
            rows = await cursor.fetchall()

            patterns = []
            for row in rows:
                p = dict(row)
                try:
                    p["market_regimes"] = json.loads(p["market_regimes"]) if p["market_regimes"] else []
                except:
                    p["market_regimes"] = []
                try:
                    p["source_reflection_ids"] = json.loads(p["source_reflection_ids"]) if p["source_reflection_ids"] else []
                except:
                    p["source_reflection_ids"] = []
                patterns.append(p)

            return patterns
        except Exception as e:
            # Table might not exist
            return []


# =============================================================================
# LEARNING STATS
# =============================================================================

@router.get("/stats")
async def get_learning_stats():
    """
    Get overall learning system statistics.

    Returns counts, averages, and health metrics for the learning system.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        stats = {
            "total_reflections": 0,
            "avg_confidence": 0,
            "wins_analyzed": 0,
            "losses_analyzed": 0,
            "patterns_detected": 0,
            "high_confidence_lessons": 0,
            "learning_active": False
        }

        try:
            # Reflection stats
            cursor = await db.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(confidence) as avg_conf,
                    SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN confidence >= 0.7 THEN 1 ELSE 0 END) as high_conf
                FROM reflections
            """)
            row = await cursor.fetchone()

            if row and row["total"] > 0:
                stats["total_reflections"] = row["total"]
                stats["avg_confidence"] = round(row["avg_conf"] or 0, 2)
                stats["wins_analyzed"] = row["wins"]
                stats["losses_analyzed"] = row["losses"]
                stats["high_confidence_lessons"] = row["high_conf"]
                stats["learning_active"] = True

            # Pattern count
            cursor = await db.execute("SELECT COUNT(*) as count FROM reflection_patterns")
            row = await cursor.fetchone()
            if row:
                stats["patterns_detected"] = row["count"]

        except Exception:
            # Tables might not exist yet
            pass

        return stats
