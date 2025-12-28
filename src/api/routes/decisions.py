"""
Decision Log Endpoints
Strategy evaluation history from Truth Engine.
"""

import aiosqlite
import json
from fastapi import APIRouter, Query, HTTPException
from pathlib import Path
from typing import Optional
import os

router = APIRouter()
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "v4_live_paper.db"

# LLM Insight generator (lazy loaded)
_insight_generator = None


def get_insight_generator():
    """Get or create the LLM insight generator."""
    global _insight_generator
    if _insight_generator is None:
        try:
            from src.truth.llm_insights import LLMInsightGenerator
            _insight_generator = LLMInsightGenerator()
        except ImportError:
            pass
    return _insight_generator


@router.get("/decisions")
async def get_decisions(
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
    result: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Get recent strategy decisions.

    Filter by strategy name, symbol, or decision result.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        query = "SELECT * FROM decisions WHERE 1=1"
        params = []

        if strategy and strategy != "all":
            query += " AND strategy_name = ?"
            params.append(strategy)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if result:
            query += " AND result = ?"
            params.append(result)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        decisions = []
        for row in rows:
            d = dict(row)
            # Parse JSON fields
            try:
                d["signal_values"] = json.loads(d["signal_values"]) if d["signal_values"] else {}
            except:
                d["signal_values"] = {}
            try:
                d["risk_checks"] = json.loads(d["risk_checks"]) if d["risk_checks"] else {}
            except:
                d["risk_checks"] = {}
            try:
                d["market_context"] = json.loads(d["market_context"]) if d["market_context"] else {}
            except:
                d["market_context"] = {}
            decisions.append(d)

        return decisions


@router.get("/decisions/stats")
async def get_decision_stats(strategy: Optional[str] = None, days: int = 7):
    """Get decision statistics by result type."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        query = """
            SELECT
                result,
                COUNT(*) as count
            FROM decisions
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
        """
        params = [days]

        if strategy and strategy != "all":
            query += " AND strategy_name = ?"
            params.append(strategy)

        query += " GROUP BY result ORDER BY count DESC"

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        return [dict(row) for row in rows]


@router.get("/decisions/{decision_id}")
async def get_decision(decision_id: str):
    """
    Get a single decision by ID with all parsed fields.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
            "SELECT * FROM decisions WHERE decision_id = ?",
            (decision_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Decision not found")

        d = dict(row)
        # Parse JSON fields
        try:
            d["signal_values"] = json.loads(d["signal_values"]) if d["signal_values"] else {}
        except:
            d["signal_values"] = {}
        try:
            d["risk_checks"] = json.loads(d["risk_checks"]) if d["risk_checks"] else {}
        except:
            d["risk_checks"] = {}
        try:
            d["market_context"] = json.loads(d["market_context"]) if d["market_context"] else {}
        except:
            d["market_context"] = {}
        try:
            d["llm_insight"] = json.loads(d["llm_insight"]) if d.get("llm_insight") else None
        except:
            d["llm_insight"] = None

        return d


@router.get("/decisions/{decision_id}/insight")
async def get_decision_insight(
    decision_id: str,
    regenerate: bool = Query(default=False, description="Force regenerate insight"),
    verify_market: bool = Query(default=True, description="Include real-time market verification")
):
    """
    Get or generate LLM insight for a decision with real-time market verification.

    Uses GPT-5.2 with LIVE MARKET DATA to provide:
    - Summary: Plain English explanation
    - Assessment: Honest critique verified against current market state
    - Market Verification: Price change since decision, validation status
    - Social Sentiment: LunarCrush Galaxy Score and sentiment (if LUNARCRUSH_API_KEY set)
    - Breaking News: Recent headlines (if TAVILY_API_KEY set)
    - Risk Flags: Potential concerns
    - Recommendation: What to do RIGHT NOW

    Required: OPENAI_API_KEY
    Optional: LUNARCRUSH_API_KEY, TAVILY_API_KEY (for enhanced insights)
    """
    # First, get the decision
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
            "SELECT * FROM decisions WHERE decision_id = ?",
            (decision_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Decision not found")

        decision = dict(row)

        # Check for existing insight
        if decision.get("llm_insight") and not regenerate:
            try:
                return {
                    "decision_id": decision_id,
                    "insight": json.loads(decision["llm_insight"]),
                    "cached": True
                }
            except:
                pass

        # Parse JSON fields for LLM
        try:
            decision["signal_values"] = json.loads(decision["signal_values"]) if decision["signal_values"] else {}
        except:
            decision["signal_values"] = {}
        try:
            decision["risk_checks"] = json.loads(decision["risk_checks"]) if decision["risk_checks"] else {}
        except:
            decision["risk_checks"] = {}
        try:
            decision["market_context"] = json.loads(decision["market_context"]) if decision["market_context"] else {}
        except:
            decision["market_context"] = {}

    # Generate insight
    generator = get_insight_generator()
    if not generator or not generator.is_available:
        # Return fallback insight
        from src.truth.llm_insights import LLMInsightGenerator
        fallback_gen = LLMInsightGenerator()
        insight = fallback_gen._generate_fallback_insight(decision)
        return {
            "decision_id": decision_id,
            "insight": insight.to_dict(),
            "cached": False,
            "fallback": True,
            "message": "LLM not available. Set OPENAI_API_KEY for AI-powered insights."
        }

    try:
        insight = await generator.analyze_decision(
            decision,
            verify_with_market=verify_market
        )
        insight_dict = insight.to_dict()

        # Always cache the insight for future review
        async with aiosqlite.connect(str(DB_PATH)) as db:
            await db.execute(
                "UPDATE decisions SET llm_insight = ? WHERE decision_id = ?",
                (json.dumps(insight_dict), decision_id)
            )
            await db.commit()

        return {
            "decision_id": decision_id,
            "insight": insight_dict,
            "cached": False,
            "market_verified": verify_market
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate insight: {str(e)}"
        )
