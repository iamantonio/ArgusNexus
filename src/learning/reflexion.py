"""
Reflexion Engine - Learning from Trading Mistakes

Implements the Reflexion framework for trading:
- Post-trade analysis using LLM
- Verbal self-reflection without model fine-tuning
- Queryable memory of past lessons
- Context-aware lesson retrieval for future decisions

Based on research into:
- Reflexion: Language Agents with Verbal Reinforcement Learning
- Quant firm approaches (Renaissance, Two Sigma)
- Trading bot failure patterns (Knight Capital, overfitting)

The key insight: Simple models that LEARN beat complex models that don't.
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ReflectionType(Enum):
    """Types of reflections/lessons learned"""
    WINNING_TRADE = "winning_trade"         # What we did right
    LOSING_TRADE = "losing_trade"           # What we did wrong
    MISSED_OPPORTUNITY = "missed_opportunity"  # Signal we should have taken
    FALSE_SIGNAL = "false_signal"           # Signal we shouldn't have taken
    TIMING_ERROR = "timing_error"           # Right direction, wrong timing
    SIZE_ERROR = "size_error"               # Right trade, wrong size
    EXIT_ERROR = "exit_error"               # Good entry, bad exit
    MARKET_REGIME = "market_regime"         # Regime-specific lesson


class MarketRegime(Enum):
    """Market regime classification"""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    STRONG_UPTREND = "strong_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    UNKNOWN = "unknown"


@dataclass
class Reflection:
    """
    A learned lesson from a completed trade.

    This is the atomic unit of Argus's memory - each reflection
    captures what happened, what was expected, and what to do
    differently next time.
    """
    reflection_id: str                      # UUID
    trade_id: str                           # Link to completed trade
    symbol: str                             # Trading pair
    created_at: datetime                    # When reflection was generated

    # Trade outcome
    is_winner: bool
    pnl_percent: float                      # Percentage P&L
    duration_hours: float                   # How long was position held

    # Context at time of trade
    market_regime: MarketRegime
    entry_atr: float                        # Volatility at entry
    entry_price: float
    exit_price: float
    exit_reason: str                        # stop_loss, take_profit, signal_exit, etc.

    # The learning
    reflection_type: ReflectionType
    what_happened: str                      # Factual description
    what_expected: str                      # What we thought would happen
    lesson_learned: str                     # Key takeaway
    action_items: List[str]                 # Specific changes to make

    # Confidence and applicability
    confidence: float                       # 0-1, how confident in this lesson
    applies_to_regimes: List[str]           # Which regimes this lesson applies to
    applies_to_symbols: List[str]           # Which symbols (or ["*"] for all)

    # For similarity matching
    context_hash: str = ""                  # Hash of key context factors

    def __post_init__(self):
        if not self.context_hash:
            self.context_hash = self._compute_context_hash()

    def _compute_context_hash(self) -> str:
        """Compute a hash for similarity matching"""
        context = {
            "regime": self.market_regime.value,
            "volatility_bucket": "high" if self.entry_atr > 0.02 else "normal" if self.entry_atr > 0.01 else "low",
            "outcome": "win" if self.is_winner else "loss",
            "exit": self.exit_reason,
            "duration": "quick" if self.duration_hours < 4 else "medium" if self.duration_hours < 24 else "long"
        }
        return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:8]

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "reflection_id": self.reflection_id,
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "created_at": self.created_at.isoformat(),
            "is_winner": self.is_winner,
            "pnl_percent": self.pnl_percent,
            "duration_hours": self.duration_hours,
            "market_regime": self.market_regime.value,
            "entry_atr": self.entry_atr,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "reflection_type": self.reflection_type.value,
            "what_happened": self.what_happened,
            "what_expected": self.what_expected,
            "lesson_learned": self.lesson_learned,
            "action_items": json.dumps(self.action_items),
            "confidence": self.confidence,
            "applies_to_regimes": json.dumps(self.applies_to_regimes),
            "applies_to_symbols": json.dumps(self.applies_to_symbols),
            "context_hash": self.context_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reflection":
        """Create from dictionary"""
        return cls(
            reflection_id=data["reflection_id"],
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            created_at=datetime.fromisoformat(data["created_at"]),
            is_winner=data["is_winner"],
            pnl_percent=data["pnl_percent"],
            duration_hours=data["duration_hours"],
            market_regime=MarketRegime(data["market_regime"]),
            entry_atr=data["entry_atr"],
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            exit_reason=data["exit_reason"],
            reflection_type=ReflectionType(data["reflection_type"]),
            what_happened=data["what_happened"],
            what_expected=data["what_expected"],
            lesson_learned=data["lesson_learned"],
            action_items=json.loads(data["action_items"]) if isinstance(data["action_items"], str) else data["action_items"],
            confidence=data["confidence"],
            applies_to_regimes=json.loads(data["applies_to_regimes"]) if isinstance(data["applies_to_regimes"], str) else data["applies_to_regimes"],
            applies_to_symbols=json.loads(data["applies_to_symbols"]) if isinstance(data["applies_to_symbols"], str) else data["applies_to_symbols"],
            context_hash=data.get("context_hash", "")
        )


# SQL Schema for reflections table
REFLECTIONS_SCHEMA = """
-- =============================================================================
-- REFLECTIONS TABLE: Lessons learned from past trades
-- The memory of Argus - where learning is stored
-- =============================================================================
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
    context_hash TEXT NOT NULL,
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_reflections_symbol ON reflections(symbol);
CREATE INDEX IF NOT EXISTS idx_reflections_regime ON reflections(market_regime);
CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(reflection_type);
CREATE INDEX IF NOT EXISTS idx_reflections_winner ON reflections(is_winner);
CREATE INDEX IF NOT EXISTS idx_reflections_hash ON reflections(context_hash);
CREATE INDEX IF NOT EXISTS idx_reflections_created ON reflections(created_at);

-- =============================================================================
-- REFLECTION PATTERNS TABLE: Aggregated patterns from reflections
-- Higher-level insights derived from multiple reflections
-- =============================================================================
CREATE TABLE IF NOT EXISTS reflection_patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_name TEXT NOT NULL,
    description TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    avg_pnl_impact REAL,
    market_regimes TEXT,       -- JSON array
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_reflection_ids TEXT -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_patterns_name ON reflection_patterns(pattern_name);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON reflection_patterns(confidence);
"""


class ReflexionEngine:
    """
    The Learning Core of Argus.

    Implements the Reflexion framework:
    1. After trade closes -> Reflect on outcome
    2. Generate lesson using LLM
    3. Store lesson for future reference
    4. Query relevant lessons for new decisions

    "The definition of insanity is doing the same thing over and over
    and expecting different results." - Argus doesn't do that.
    """

    REFLECTION_PROMPT = """You are a trading coach analyzing a completed trade.
Your job is to extract ACTIONABLE LESSONS from this trade outcome.

CRITICAL: Focus on what can be LEARNED and APPLIED to future trades.
This is not about blame - it's about improvement.

Trade Data:
{trade_data}

Market Context at Entry:
{entry_context}

Market Context at Exit:
{exit_context}

Previous Decision Analysis (if available):
{decision_insight}

Analyze this trade and provide:

1. REFLECTION_TYPE: One of:
   - winning_trade: What did we do right?
   - losing_trade: What went wrong?
   - missed_opportunity: Should we have held longer?
   - false_signal: Should we have skipped this trade?
   - timing_error: Right direction, wrong timing
   - size_error: Right trade, wrong size
   - exit_error: Good entry, bad exit

2. WHAT_HAPPENED: Factual description of the trade (2-3 sentences)

3. WHAT_EXPECTED: What did the strategy expect to happen? (1-2 sentences)

4. LESSON_LEARNED: The KEY takeaway from this trade (1 clear sentence)

5. ACTION_ITEMS: 2-4 specific, actionable changes for future trades
   - Be specific: "Reduce position size by 50% when ATR > 3%" not "Be more careful"
   - Be measurable: Include thresholds where possible
   - Be implementable: Something the trading system can actually do

6. CONFIDENCE: 0.0 to 1.0 - How confident are you in this lesson?
   - High confidence (0.8+): Clear cause-effect relationship
   - Medium (0.5-0.8): Likely pattern but not certain
   - Low (<0.5): Too many variables to be sure

7. APPLIES_TO_REGIMES: Which market regimes does this lesson apply to?
   - Options: high_volatility, low_volatility, strong_uptrend, strong_downtrend, ranging, breakout, breakdown, all

8. APPLIES_TO_SYMBOLS: Does this lesson apply to all symbols or specific ones?
   - Use ["*"] for all symbols, or list specific ones like ["BTC-USD", "ETH-USD"]

Output as JSON with these exact keys:
- reflection_type (string)
- what_happened (string)
- what_expected (string)
- lesson_learned (string)
- action_items (array of strings)
- confidence (float)
- applies_to_regimes (array of strings)
- applies_to_symbols (array of strings)

Remember: The goal is to make Argus BETTER, not to criticize past decisions.
Every trade is a learning opportunity."""

    QUERY_PROMPT = """You are helping a trading system recall relevant lessons from past trades.

Current Trading Context:
{current_context}

Past Reflections (lessons learned):
{reflections}

Based on the current context, identify which past lessons are MOST RELEVANT.

For each relevant lesson, explain:
1. Why it applies to the current situation
2. What specific action should be taken based on this lesson
3. How confident we should be in applying this lesson (considering context similarity)

Output as JSON with key "applicable_lessons" containing an array of:
- reflection_id (string)
- relevance_score (float 0-1)
- why_relevant (string)
- recommended_action (string)
- adjusted_confidence (float 0-1)

Only include lessons with relevance_score > 0.5.
If no lessons are relevant, return {"applicable_lessons": []}."""

    def __init__(
        self,
        db_path: str,
        api_key: Optional[str] = None,
        model: str = "gpt-4o"  # Using gpt-4o for reflection
    ):
        """
        Initialize the Reflexion Engine.

        Args:
            db_path: Path to the Truth Engine database
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for reflection generation
        """
        self.db_path = db_path
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Optional[AsyncOpenAI] = None

        if OPENAI_AVAILABLE and self.api_key:
            self._client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"ReflexionEngine initialized with {model}")
        else:
            logger.warning("OpenAI not available - reflections will use fallback mode")

    @property
    def is_available(self) -> bool:
        """Check if LLM reflection is available"""
        return self._client is not None

    async def initialize(self) -> None:
        """Initialize the reflections table in the database"""
        import aiosqlite
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(REFLECTIONS_SCHEMA)
            logger.info("Reflections table initialized")

    async def reflect_on_trade(
        self,
        trade_id: str,
        force: bool = False
    ) -> Optional[Reflection]:
        """
        Generate a reflection for a completed trade.

        This is the core learning function - called after each trade closes.

        Args:
            trade_id: The trade to reflect on
            force: If True, regenerate even if reflection exists

        Returns:
            The generated Reflection, or None if generation failed
        """
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Check if reflection already exists
            if not force:
                cursor = await db.execute(
                    "SELECT reflection_id FROM reflections WHERE trade_id = ?",
                    (trade_id,)
                )
                existing = await cursor.fetchone()
                if existing:
                    logger.debug(f"Reflection already exists for trade {trade_id}")
                    return await self.get_reflection(existing["reflection_id"])

            # Fetch trade data
            cursor = await db.execute("""
                SELECT t.*,
                       ed.signal_values as entry_signals,
                       ed.market_context as entry_market,
                       ed.risk_checks as entry_risk,
                       ed.llm_insight as entry_insight,
                       xd.signal_values as exit_signals,
                       xd.market_context as exit_market,
                       xd.llm_insight as exit_insight
                FROM trades t
                LEFT JOIN decisions ed ON t.entry_decision_id = ed.decision_id
                LEFT JOIN decisions xd ON t.exit_decision_id = xd.decision_id
                WHERE t.trade_id = ?
            """, (trade_id,))
            trade_row = await cursor.fetchone()

            if not trade_row:
                logger.error(f"Trade not found: {trade_id}")
                return None

            trade = dict(trade_row)

        # Parse JSON fields
        entry_signals = json.loads(trade.get("entry_signals") or "{}")
        entry_market = json.loads(trade.get("entry_market") or "{}")
        exit_signals = json.loads(trade.get("exit_signals") or "{}")
        exit_market = json.loads(trade.get("exit_market") or "{}")
        entry_insight = json.loads(trade.get("entry_insight") or "null")
        exit_insight = json.loads(trade.get("exit_insight") or "null")

        # Calculate metrics
        entry_price = float(trade["entry_price"])
        exit_price = float(trade["exit_price"]) if trade.get("exit_price") else entry_price
        pnl_percent = float(trade.get("realized_pnl_percent") or 0)
        is_winner = trade.get("is_winner", 0) == 1

        # Calculate duration
        entry_ts = datetime.fromisoformat(trade["entry_timestamp"])
        exit_ts = datetime.fromisoformat(trade["exit_timestamp"]) if trade.get("exit_timestamp") else datetime.now(timezone.utc)
        duration_hours = (exit_ts - entry_ts).total_seconds() / 3600

        # Determine market regime
        regime = self._detect_regime(entry_signals, entry_market)

        # Get ATR at entry
        entry_atr = entry_signals.get("atr", 0) or entry_market.get("atr", 0) or 0.01
        if entry_price > 0:
            entry_atr_pct = float(entry_atr) / entry_price
        else:
            entry_atr_pct = 0.01

        # Generate reflection using LLM
        reflection_data = await self._generate_reflection(
            trade=trade,
            entry_signals=entry_signals,
            entry_market=entry_market,
            exit_signals=exit_signals,
            exit_market=exit_market,
            entry_insight=entry_insight,
            exit_insight=exit_insight
        )

        if not reflection_data:
            logger.error(f"Failed to generate reflection for trade {trade_id}")
            return None

        # Create Reflection object
        reflection = Reflection(
            reflection_id=Reflection.generate_id(),
            trade_id=trade_id,
            symbol=trade["symbol"],
            created_at=datetime.now(timezone.utc),
            is_winner=is_winner,
            pnl_percent=pnl_percent,
            duration_hours=duration_hours,
            market_regime=regime,
            entry_atr=entry_atr_pct,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=trade.get("exit_reason") or "unknown",
            reflection_type=ReflectionType(reflection_data["reflection_type"]),
            what_happened=reflection_data["what_happened"],
            what_expected=reflection_data["what_expected"],
            lesson_learned=reflection_data["lesson_learned"],
            action_items=reflection_data["action_items"],
            confidence=reflection_data["confidence"],
            applies_to_regimes=reflection_data["applies_to_regimes"],
            applies_to_symbols=reflection_data["applies_to_symbols"]
        )

        # Store reflection
        await self._store_reflection(reflection)

        logger.info(f"Reflection generated for trade {trade_id}: {reflection.lesson_learned}")
        return reflection

    def _detect_regime(
        self,
        signals: Dict[str, Any],
        market: Dict[str, Any]
    ) -> MarketRegime:
        """Detect market regime from signals and market context"""
        atr = signals.get("atr", 0) or market.get("atr", 0)
        price = signals.get("current_price", 0) or market.get("price", 0)
        adx = signals.get("adx", 0) or market.get("adx", 0)
        trend = signals.get("trend", "") or market.get("trend", "")

        # Calculate ATR percentage if we have price
        atr_pct = (atr / price * 100) if price > 0 else 0

        # High volatility check
        if atr_pct > 3:  # ATR > 3% of price
            return MarketRegime.HIGH_VOLATILITY

        # Trend strength check
        if adx > 25:
            if "up" in str(trend).lower() or "bull" in str(trend).lower():
                return MarketRegime.STRONG_UPTREND
            elif "down" in str(trend).lower() or "bear" in str(trend).lower():
                return MarketRegime.STRONG_DOWNTREND

        # Low volatility
        if atr_pct < 1:
            return MarketRegime.LOW_VOLATILITY

        # Ranging market
        if adx < 20:
            return MarketRegime.RANGING

        return MarketRegime.UNKNOWN

    async def _generate_reflection(
        self,
        trade: Dict[str, Any],
        entry_signals: Dict[str, Any],
        entry_market: Dict[str, Any],
        exit_signals: Dict[str, Any],
        exit_market: Dict[str, Any],
        entry_insight: Optional[Dict[str, Any]],
        exit_insight: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate reflection using LLM"""

        # Format trade data
        trade_data = f"""
Symbol: {trade['symbol']}
Side: {trade['side']}
Entry Price: ${float(trade['entry_price']):,.2f}
Exit Price: ${float(trade['exit_price'] or 0):,.2f}
Quantity: {trade['quantity']}
P&L: {float(trade.get('realized_pnl_percent') or 0):.2f}%
Net P&L: ${float(trade.get('net_pnl') or 0):,.2f}
Duration: {trade.get('duration_seconds', 0) / 3600:.1f} hours
Exit Reason: {trade.get('exit_reason', 'unknown')}
Result: {'WIN' if trade.get('is_winner') else 'LOSS'}
"""

        entry_context = self._format_context(entry_signals, entry_market)
        exit_context = self._format_context(exit_signals, exit_market)

        # Format decision insight if available
        decision_insight = ""
        if entry_insight:
            decision_insight += f"Entry Analysis: {entry_insight.get('assessment', 'N/A')}\n"
            decision_insight += f"Entry Verdict: {entry_insight.get('verdict', 'N/A')}\n"
        if exit_insight:
            decision_insight += f"Exit Analysis: {exit_insight.get('assessment', 'N/A')}\n"
            decision_insight += f"Exit Verdict: {exit_insight.get('verdict', 'N/A')}\n"

        if not decision_insight:
            decision_insight = "No previous LLM analysis available"

        prompt = self.REFLECTION_PROMPT.format(
            trade_data=trade_data,
            entry_context=entry_context,
            exit_context=exit_context,
            decision_insight=decision_insight
        )

        if not self.is_available:
            return self._generate_fallback_reflection(trade)

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a trading coach analyzing completed trades to extract actionable lessons."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1500,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            logger.error(f"LLM reflection generation failed: {e}")
            return self._generate_fallback_reflection(trade)

    def _format_context(
        self,
        signals: Dict[str, Any],
        market: Dict[str, Any]
    ) -> str:
        """Format signals and market context for the prompt"""
        lines = []

        combined = {**market, **signals}
        for key, value in combined.items():
            if value is not None and value != "":
                readable_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    if 0 < abs(value) < 0.01:
                        lines.append(f"  {readable_key}: {value:.6f}")
                    elif abs(value) < 1:
                        lines.append(f"  {readable_key}: {value:.4f}")
                    else:
                        lines.append(f"  {readable_key}: {value:,.2f}")
                else:
                    lines.append(f"  {readable_key}: {value}")

        return "\n".join(lines) if lines else "  No context data available"

    def _generate_fallback_reflection(
        self,
        trade: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate basic reflection when LLM is not available"""
        is_winner = trade.get("is_winner", 0) == 1
        pnl_pct = float(trade.get("realized_pnl_percent") or 0)
        exit_reason = trade.get("exit_reason", "unknown")

        if is_winner:
            reflection_type = "winning_trade"
            what_happened = f"Trade closed profitably with {pnl_pct:.2f}% gain via {exit_reason}."
            lesson = "Maintain discipline and let winning trades run when conditions are favorable."
            action_items = [
                "Review entry signals for this successful setup",
                "Document market conditions that led to this win"
            ]
        else:
            reflection_type = "losing_trade"
            what_happened = f"Trade closed with {pnl_pct:.2f}% loss via {exit_reason}."

            if exit_reason == "stop_loss":
                lesson = "Stop loss protected capital - verify stop placement was appropriate."
                action_items = [
                    "Analyze if stop was too tight for current volatility",
                    "Check if entry timing could be improved"
                ]
            else:
                lesson = "Loss occurred - review entry conditions and risk management."
                action_items = [
                    "Review signal quality at entry",
                    "Verify position sizing was appropriate"
                ]

        return {
            "reflection_type": reflection_type,
            "what_happened": what_happened,
            "what_expected": "Strategy expected favorable price movement in trade direction.",
            "lesson_learned": lesson,
            "action_items": action_items,
            "confidence": 0.5,
            "applies_to_regimes": ["all"],
            "applies_to_symbols": ["*"]
        }

    async def _store_reflection(self, reflection: Reflection) -> None:
        """Store reflection in database"""
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            data = reflection.to_dict()
            await db.execute("""
                INSERT OR REPLACE INTO reflections (
                    reflection_id, trade_id, symbol, created_at,
                    is_winner, pnl_percent, duration_hours,
                    market_regime, entry_atr, entry_price, exit_price,
                    exit_reason, reflection_type, what_happened,
                    what_expected, lesson_learned, action_items,
                    confidence, applies_to_regimes, applies_to_symbols,
                    context_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["reflection_id"],
                data["trade_id"],
                data["symbol"],
                data["created_at"],
                1 if data["is_winner"] else 0,
                data["pnl_percent"],
                data["duration_hours"],
                data["market_regime"],
                data["entry_atr"],
                data["entry_price"],
                data["exit_price"],
                data["exit_reason"],
                data["reflection_type"],
                data["what_happened"],
                data["what_expected"],
                data["lesson_learned"],
                data["action_items"],
                data["confidence"],
                data["applies_to_regimes"],
                data["applies_to_symbols"],
                data["context_hash"]
            ))
            await db.commit()

    async def get_reflection(self, reflection_id: str) -> Optional[Reflection]:
        """Get a reflection by ID"""
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM reflections WHERE reflection_id = ?",
                (reflection_id,)
            )
            row = await cursor.fetchone()

            if row:
                return Reflection.from_dict(dict(row))
            return None

    async def query_relevant_lessons(
        self,
        symbol: str,
        regime: MarketRegime,
        atr_pct: float,
        limit: int = 10
    ) -> List[Reflection]:
        """
        Query past reflections relevant to current trading context.

        This is called BEFORE making a new trade decision to incorporate
        lessons from similar past situations.

        Args:
            symbol: Current symbol being traded
            regime: Current market regime
            atr_pct: Current ATR as percentage of price
            limit: Maximum number of reflections to return

        Returns:
            List of relevant Reflections, sorted by relevance
        """
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Query reflections that might apply
            # Priority: Same symbol + regime > Same regime > High confidence
            cursor = await db.execute("""
                SELECT *,
                    CASE
                        WHEN symbol = ? AND market_regime = ? THEN 3
                        WHEN market_regime = ? THEN 2
                        WHEN applies_to_symbols LIKE '%"*"%' THEN 1
                        ELSE 0
                    END as relevance_score
                FROM reflections
                WHERE confidence >= 0.5
                  AND (
                      symbol = ?
                      OR applies_to_symbols LIKE '%"*"%'
                      OR applies_to_symbols LIKE ?
                  )
                  AND (
                      market_regime = ?
                      OR applies_to_regimes LIKE '%"all"%'
                      OR applies_to_regimes LIKE ?
                  )
                ORDER BY relevance_score DESC, confidence DESC, created_at DESC
                LIMIT ?
            """, (
                symbol, regime.value,
                regime.value,
                symbol,
                f'%"{symbol}"%',
                regime.value,
                f'%"{regime.value}"%',
                limit
            ))

            rows = await cursor.fetchall()
            return [Reflection.from_dict(dict(row)) for row in rows]

    async def get_lessons_summary(
        self,
        symbol: Optional[str] = None,
        days: int = 30,
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """
        Get a summary of lessons learned over a period.

        Useful for periodic review of what Argus has learned.
        """
        import aiosqlite

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Base query
            symbol_filter = "AND symbol = ?" if symbol else ""
            params = [cutoff, min_confidence]
            if symbol:
                params.append(symbol)

            # Get aggregate stats
            cursor = await db.execute(f"""
                SELECT
                    COUNT(*) as total_reflections,
                    SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as from_wins,
                    SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as from_losses,
                    AVG(confidence) as avg_confidence,
                    AVG(pnl_percent) as avg_pnl
                FROM reflections
                WHERE created_at >= ?
                  AND confidence >= ?
                  {symbol_filter}
            """, params)
            stats = dict(await cursor.fetchone())

            # Get most common lesson types
            cursor = await db.execute(f"""
                SELECT reflection_type, COUNT(*) as count
                FROM reflections
                WHERE created_at >= ?
                  AND confidence >= ?
                  {symbol_filter}
                GROUP BY reflection_type
                ORDER BY count DESC
            """, params)
            type_counts = {row["reflection_type"]: row["count"] for row in await cursor.fetchall()}

            # Get top lessons by confidence
            cursor = await db.execute(f"""
                SELECT lesson_learned, confidence, reflection_type
                FROM reflections
                WHERE created_at >= ?
                  AND confidence >= ?
                  {symbol_filter}
                ORDER BY confidence DESC
                LIMIT 5
            """, params)
            top_lessons = [dict(row) for row in await cursor.fetchall()]

            return {
                "period_days": days,
                "symbol": symbol or "all",
                "stats": stats,
                "lesson_type_distribution": type_counts,
                "top_lessons": top_lessons
            }

    async def apply_lessons_to_decision(
        self,
        symbol: str,
        current_signals: Dict[str, Any],
        current_market: Dict[str, Any],
        proposed_action: str  # "long", "short", "hold"
    ) -> Dict[str, Any]:
        """
        Apply relevant lessons to a proposed trading decision.

        This is the KEY FUNCTION that makes Argus learn from its mistakes.
        Called before executing a trade to incorporate past lessons.

        Returns:
            - adjusted_confidence: Modified confidence based on lessons
            - size_adjustment: Suggested position size multiplier
            - warnings: Any warnings from past lessons
            - supporting_lessons: Lessons that support this trade
            - cautionary_lessons: Lessons that caution against this trade
        """
        # Detect current regime
        regime = self._detect_regime(current_signals, current_market)

        # Get current ATR percentage
        price = current_signals.get("current_price") or current_market.get("price") or 1
        atr = current_signals.get("atr") or current_market.get("atr") or 0
        atr_pct = (atr / price) if price > 0 else 0.01

        # Query relevant lessons
        lessons = await self.query_relevant_lessons(
            symbol=symbol,
            regime=regime,
            atr_pct=atr_pct,
            limit=15
        )

        if not lessons:
            return {
                "adjusted_confidence": 1.0,
                "size_adjustment": 1.0,
                "warnings": [],
                "supporting_lessons": [],
                "cautionary_lessons": [],
                "regime": regime.value,
                "lesson_count": 0
            }

        # Categorize lessons
        supporting = []
        cautionary = []
        warnings = []

        for lesson in lessons:
            # Check if this lesson is relevant to current action
            is_similar_context = (
                lesson.market_regime == regime or
                "all" in lesson.applies_to_regimes
            )

            if not is_similar_context:
                continue

            # Winning trades in similar context = supporting
            # Losing trades in similar context = cautionary
            if lesson.is_winner:
                supporting.append({
                    "lesson": lesson.lesson_learned,
                    "confidence": lesson.confidence,
                    "type": lesson.reflection_type.value
                })
            else:
                cautionary.append({
                    "lesson": lesson.lesson_learned,
                    "confidence": lesson.confidence,
                    "type": lesson.reflection_type.value,
                    "pnl": lesson.pnl_percent
                })

                # Add warnings for high-confidence losing lessons
                if lesson.confidence >= 0.7:
                    warnings.append(f"[{lesson.reflection_type.value}] {lesson.lesson_learned}")

        # Calculate adjustments
        # More cautionary lessons = lower confidence, smaller size
        num_supporting = len(supporting)
        num_cautionary = len(cautionary)

        if num_supporting + num_cautionary > 0:
            support_ratio = num_supporting / (num_supporting + num_cautionary)
        else:
            support_ratio = 0.5

        # Weighted by confidence
        avg_caution_conf = (
            sum(c["confidence"] for c in cautionary) / len(cautionary)
            if cautionary else 0
        )

        # Adjust confidence: neutral at 0.5, boost for support, reduce for caution
        confidence_adjustment = 0.8 + (support_ratio * 0.4)  # Range: 0.8 to 1.2

        # Size adjustment: reduce if high-confidence cautions exist
        if avg_caution_conf > 0.7 and num_cautionary > 2:
            size_adjustment = 0.5  # Cut size in half
        elif avg_caution_conf > 0.5 and num_cautionary > 1:
            size_adjustment = 0.75
        else:
            size_adjustment = 1.0

        return {
            "adjusted_confidence": round(confidence_adjustment, 2),
            "size_adjustment": round(size_adjustment, 2),
            "warnings": warnings[:3],  # Top 3 warnings
            "supporting_lessons": supporting[:3],
            "cautionary_lessons": cautionary[:3],
            "regime": regime.value,
            "lesson_count": len(lessons),
            "support_ratio": round(support_ratio, 2)
        }

    async def backfill_reflections(
        self,
        days: int = 30,
        limit: int = 100
    ) -> int:
        """
        Generate reflections for past trades that don't have them.

        Useful for bootstrapping Argus's memory from historical trades.

        Returns:
            Number of reflections generated
        """
        import aiosqlite

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        count = 0

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Find trades without reflections
            cursor = await db.execute("""
                SELECT t.trade_id
                FROM trades t
                LEFT JOIN reflections r ON t.trade_id = r.trade_id
                WHERE t.status = 'closed'
                  AND t.exit_timestamp >= ?
                  AND r.reflection_id IS NULL
                ORDER BY t.exit_timestamp DESC
                LIMIT ?
            """, (cutoff, limit))

            trade_ids = [row["trade_id"] for row in await cursor.fetchall()]

        logger.info(f"Backfilling reflections for {len(trade_ids)} trades...")

        for trade_id in trade_ids:
            try:
                reflection = await self.reflect_on_trade(trade_id)
                if reflection:
                    count += 1
                    logger.debug(f"Generated reflection {count}/{len(trade_ids)}")
            except Exception as e:
                logger.error(f"Failed to reflect on trade {trade_id}: {e}")

        logger.info(f"Backfill complete: {count} reflections generated")
        return count
