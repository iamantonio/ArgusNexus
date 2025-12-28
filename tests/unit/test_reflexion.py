"""
Tests for the Reflexion Layer - Argus's Learning System

Verifies that:
1. Reflections are generated correctly from trades
2. Lessons are queryable by context
3. Learning adjusts confidence and position sizing
4. The learning loop integrates with the trading pipeline
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import json
import tempfile
import os

# Set up test environment
os.environ["OPENAI_API_KEY"] = ""  # Disable LLM for unit tests

from src.learning.reflexion import (
    ReflexionEngine,
    Reflection,
    ReflectionType,
    MarketRegime,
    REFLECTIONS_SCHEMA
)


class TestReflectionDataclass:
    """Test the Reflection dataclass"""

    def test_reflection_creation(self):
        """Reflection should be created with all required fields"""
        reflection = Reflection(
            reflection_id="test-123",
            trade_id="trade-456",
            symbol="BTC-USD",
            created_at=datetime.now(timezone.utc),
            is_winner=True,
            pnl_percent=5.5,
            duration_hours=12.0,
            market_regime=MarketRegime.STRONG_UPTREND,
            entry_atr=0.02,
            entry_price=50000.0,
            exit_price=52750.0,
            exit_reason="take_profit",
            reflection_type=ReflectionType.WINNING_TRADE,
            what_happened="Trade closed at take profit after strong momentum",
            what_expected="Expected 3-5% gain based on trend strength",
            lesson_learned="Let winners run in strong trends",
            action_items=["Consider trailing stop", "Watch for reversal signals"],
            confidence=0.85,
            applies_to_regimes=["strong_uptrend", "breakout"],
            applies_to_symbols=["BTC-USD", "ETH-USD"]
        )

        assert reflection.reflection_id == "test-123"
        assert reflection.is_winner is True
        assert reflection.pnl_percent == 5.5
        assert reflection.market_regime == MarketRegime.STRONG_UPTREND
        assert reflection.confidence == 0.85
        assert len(reflection.action_items) == 2

    def test_reflection_context_hash(self):
        """Context hash should be computed automatically"""
        reflection = Reflection(
            reflection_id=Reflection.generate_id(),
            trade_id="trade-1",
            symbol="BTC-USD",
            created_at=datetime.now(timezone.utc),
            is_winner=False,
            pnl_percent=-2.5,
            duration_hours=6.0,
            market_regime=MarketRegime.HIGH_VOLATILITY,
            entry_atr=0.035,
            entry_price=48000.0,
            exit_price=46800.0,
            exit_reason="stop_loss",
            reflection_type=ReflectionType.LOSING_TRADE,
            what_happened="Stop loss triggered in volatile market",
            what_expected="Expected continuation of trend",
            lesson_learned="Widen stops in high volatility",
            action_items=["Increase ATR multiplier for stops"],
            confidence=0.7,
            applies_to_regimes=["high_volatility"],
            applies_to_symbols=["*"]
        )

        # Hash should be computed
        assert reflection.context_hash != ""
        assert len(reflection.context_hash) == 8

    def test_reflection_to_dict(self):
        """Reflection should serialize to dict correctly"""
        reflection = Reflection(
            reflection_id="test-dict",
            trade_id="trade-dict",
            symbol="ETH-USD",
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            is_winner=True,
            pnl_percent=3.2,
            duration_hours=8.5,
            market_regime=MarketRegime.BREAKOUT,
            entry_atr=0.018,
            entry_price=2500.0,
            exit_price=2580.0,
            exit_reason="signal_exit",
            reflection_type=ReflectionType.WINNING_TRADE,
            what_happened="Exited on signal",
            what_expected="Expected breakout continuation",
            lesson_learned="Trust the signals",
            action_items=["Keep following signals"],
            confidence=0.9,
            applies_to_regimes=["breakout"],
            applies_to_symbols=["ETH-USD"]
        )

        data = reflection.to_dict()

        assert data["reflection_id"] == "test-dict"
        assert data["symbol"] == "ETH-USD"
        assert data["pnl_percent"] == 3.2
        assert data["market_regime"] == "breakout"
        assert data["reflection_type"] == "winning_trade"
        # JSON fields should be serialized
        assert isinstance(data["action_items"], str)
        assert json.loads(data["action_items"]) == ["Keep following signals"]

    def test_reflection_from_dict(self):
        """Reflection should deserialize from dict correctly"""
        data = {
            "reflection_id": "from-dict-test",
            "trade_id": "trade-from-dict",
            "symbol": "SOL-USD",
            "created_at": "2024-02-20T15:45:00+00:00",
            "is_winner": False,
            "pnl_percent": -1.8,
            "duration_hours": 4.0,
            "market_regime": "ranging",
            "entry_atr": 0.025,
            "entry_price": 100.0,
            "exit_price": 98.2,
            "exit_reason": "stop_loss",
            "reflection_type": "false_signal",
            "what_happened": "False breakout",
            "what_expected": "Expected breakout continuation",
            "lesson_learned": "Avoid trading in ranging markets",
            "action_items": json.dumps(["Check ADX before entry"]),
            "confidence": 0.75,
            "applies_to_regimes": json.dumps(["ranging"]),
            "applies_to_symbols": json.dumps(["*"]),
            "context_hash": "abc12345"
        }

        reflection = Reflection.from_dict(data)

        assert reflection.reflection_id == "from-dict-test"
        assert reflection.is_winner is False
        assert reflection.market_regime == MarketRegime.RANGING
        assert reflection.reflection_type == ReflectionType.FALSE_SIGNAL
        assert reflection.action_items == ["Check ADX before entry"]


class TestMarketRegimeDetection:
    """Test market regime detection logic"""

    @pytest.fixture
    def engine(self):
        """Create a ReflexionEngine for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        engine = ReflexionEngine(db_path)
        yield engine
        os.unlink(db_path)

    def test_high_volatility_detection(self, engine):
        """High ATR should trigger high volatility regime"""
        signals = {
            "atr": 3500,  # 3.5% of price
            "current_price": 100000,
            "adx": 30
        }
        regime = engine._detect_regime(signals, {})
        assert regime == MarketRegime.HIGH_VOLATILITY

    def test_strong_uptrend_detection(self, engine):
        """Strong ADX with bullish trend should detect uptrend"""
        signals = {
            "atr": 1500,  # 1.5%
            "current_price": 100000,
            "adx": 35,
            "trend": "bullish"
        }
        regime = engine._detect_regime(signals, {})
        assert regime == MarketRegime.STRONG_UPTREND

    def test_strong_downtrend_detection(self, engine):
        """Strong ADX with bearish trend should detect downtrend"""
        signals = {
            "atr": 2000,
            "current_price": 100000,
            "adx": 28,
            "trend": "bearish"
        }
        regime = engine._detect_regime(signals, {})
        assert regime == MarketRegime.STRONG_DOWNTREND

    def test_ranging_market_detection(self, engine):
        """Low ADX should detect ranging market"""
        signals = {
            "atr": 1500,
            "current_price": 100000,
            "adx": 15,
            "trend": "neutral"
        }
        regime = engine._detect_regime(signals, {})
        assert regime == MarketRegime.RANGING

    def test_low_volatility_detection(self, engine):
        """Very low ATR should detect low volatility"""
        signals = {
            "atr": 800,  # 0.8%
            "current_price": 100000,
            "adx": 22
        }
        regime = engine._detect_regime(signals, {})
        assert regime == MarketRegime.LOW_VOLATILITY


class TestReflexionEngineDatabase:
    """Test ReflexionEngine database operations"""

    @pytest.fixture
    async def engine_with_db(self):
        """Create engine with initialized database"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        engine = ReflexionEngine(db_path)
        await engine.initialize()

        yield engine

        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_database_initialization(self, engine_with_db):
        """Database should initialize with correct schema"""
        import aiosqlite

        async with aiosqlite.connect(engine_with_db.db_path) as db:
            # Check reflections table exists
            cursor = await db.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='reflections'
            """)
            row = await cursor.fetchone()
            assert row is not None

            # Check indexes exist
            cursor = await db.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name LIKE 'idx_reflections_%'
            """)
            indexes = await cursor.fetchall()
            assert len(indexes) >= 5

    @pytest.mark.asyncio
    async def test_store_and_retrieve_reflection(self, engine_with_db):
        """Should store and retrieve reflections"""
        reflection = Reflection(
            reflection_id=Reflection.generate_id(),
            trade_id="test-trade-123",
            symbol="BTC-USD",
            created_at=datetime.now(timezone.utc),
            is_winner=True,
            pnl_percent=4.5,
            duration_hours=10.0,
            market_regime=MarketRegime.STRONG_UPTREND,
            entry_atr=0.02,
            entry_price=50000.0,
            exit_price=52250.0,
            exit_reason="take_profit",
            reflection_type=ReflectionType.WINNING_TRADE,
            what_happened="Good trade",
            what_expected="Expected profit",
            lesson_learned="Patience pays",
            action_items=["Keep doing this"],
            confidence=0.8,
            applies_to_regimes=["strong_uptrend"],
            applies_to_symbols=["BTC-USD"]
        )

        await engine_with_db._store_reflection(reflection)

        # Retrieve it
        retrieved = await engine_with_db.get_reflection(reflection.reflection_id)

        assert retrieved is not None
        assert retrieved.reflection_id == reflection.reflection_id
        assert retrieved.lesson_learned == "Patience pays"
        assert retrieved.confidence == 0.8


class TestLessonApplication:
    """Test applying lessons to new decisions"""

    @pytest.fixture
    async def engine_with_lessons(self):
        """Create engine with some pre-populated lessons"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        engine = ReflexionEngine(db_path)
        await engine.initialize()

        # Add some lessons
        lessons = [
            # Winning lesson in uptrend
            Reflection(
                reflection_id=Reflection.generate_id(),
                trade_id="trade-1",
                symbol="BTC-USD",
                created_at=datetime.now(timezone.utc),
                is_winner=True,
                pnl_percent=5.0,
                duration_hours=8.0,
                market_regime=MarketRegime.STRONG_UPTREND,
                entry_atr=0.02,
                entry_price=50000.0,
                exit_price=52500.0,
                exit_reason="take_profit",
                reflection_type=ReflectionType.WINNING_TRADE,
                what_happened="Trend continuation worked",
                what_expected="Expected 5% gain",
                lesson_learned="Trust the trend in uptrends",
                action_items=["Let winners run"],
                confidence=0.85,
                applies_to_regimes=["strong_uptrend"],
                applies_to_symbols=["BTC-USD"]
            ),
            # Losing lesson in ranging market
            Reflection(
                reflection_id=Reflection.generate_id(),
                trade_id="trade-2",
                symbol="BTC-USD",
                created_at=datetime.now(timezone.utc),
                is_winner=False,
                pnl_percent=-3.0,
                duration_hours=4.0,
                market_regime=MarketRegime.RANGING,
                entry_atr=0.015,
                entry_price=48000.0,
                exit_price=46560.0,
                exit_reason="stop_loss",
                reflection_type=ReflectionType.FALSE_SIGNAL,
                what_happened="False breakout in range",
                what_expected="Expected breakout",
                lesson_learned="Avoid breakouts in ranging markets",
                action_items=["Check ADX before entry", "Wait for confirmation"],
                confidence=0.8,
                applies_to_regimes=["ranging"],
                applies_to_symbols=["*"]
            ),
            # Another losing lesson in ranging
            Reflection(
                reflection_id=Reflection.generate_id(),
                trade_id="trade-3",
                symbol="ETH-USD",
                created_at=datetime.now(timezone.utc),
                is_winner=False,
                pnl_percent=-2.5,
                duration_hours=3.0,
                market_regime=MarketRegime.RANGING,
                entry_atr=0.018,
                entry_price=2000.0,
                exit_price=1950.0,
                exit_reason="stop_loss",
                reflection_type=ReflectionType.FALSE_SIGNAL,
                what_happened="Another failed range trade",
                what_expected="Expected trend start",
                lesson_learned="Ranging markets are choppy",
                action_items=["Reduce size in ranges"],
                confidence=0.75,
                applies_to_regimes=["ranging", "all"],
                applies_to_symbols=["*"]
            ),
        ]

        for lesson in lessons:
            await engine._store_reflection(lesson)

        yield engine

        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_apply_lessons_in_uptrend(self, engine_with_lessons):
        """In uptrend, supporting lessons should boost confidence"""
        result = await engine_with_lessons.apply_lessons_to_decision(
            symbol="BTC-USD",
            current_signals={
                "atr": 2000,
                "current_price": 100000,
                "adx": 30,
                "trend": "bullish"
            },
            current_market={},
            proposed_action="long"
        )

        # Should have lessons
        assert result["lesson_count"] > 0

        # Should have supporting lesson from uptrend win
        assert len(result["supporting_lessons"]) > 0

        # Confidence should be boosted (> 1.0 or near 1.0)
        assert result["adjusted_confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_apply_lessons_in_ranging(self, engine_with_lessons):
        """In ranging market, cautionary lessons should reduce confidence"""
        result = await engine_with_lessons.apply_lessons_to_decision(
            symbol="BTC-USD",
            current_signals={
                "atr": 1500,
                "current_price": 100000,
                "adx": 18,
                "trend": "neutral"
            },
            current_market={},
            proposed_action="long"
        )

        # Should have lessons
        assert result["lesson_count"] > 0

        # Should have cautionary lessons from ranging losses
        assert len(result["cautionary_lessons"]) > 0

        # Should have warnings
        assert len(result["warnings"]) > 0

        # Size should be reduced due to multiple cautions
        assert result["size_adjustment"] < 1.0

    @pytest.mark.asyncio
    async def test_no_lessons_returns_defaults(self, engine_with_lessons):
        """When no lessons apply, should return default values"""
        result = await engine_with_lessons.apply_lessons_to_decision(
            symbol="DOGE-USD",  # No lessons for this symbol
            current_signals={
                "atr": 5,  # Very low volatility
                "current_price": 0.1,
                "adx": 50
            },
            current_market={},
            proposed_action="long"
        )

        # May or may not have lessons (depends on "*" symbol matches)
        # But should return valid structure
        assert "adjusted_confidence" in result
        assert "size_adjustment" in result
        assert isinstance(result["warnings"], list)


class TestFallbackReflection:
    """Test fallback reflection generation when LLM unavailable"""

    @pytest.fixture
    def engine(self):
        """Create engine without LLM"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        engine = ReflexionEngine(db_path)
        yield engine
        os.unlink(db_path)

    def test_fallback_for_winning_trade(self, engine):
        """Should generate sensible fallback for winning trade"""
        trade = {
            "symbol": "BTC-USD",
            "side": "buy",
            "is_winner": 1,
            "realized_pnl_percent": 4.5,
            "exit_reason": "take_profit"
        }

        result = engine._generate_fallback_reflection(trade)

        assert result["reflection_type"] == "winning_trade"
        assert "discipline" in result["lesson_learned"].lower()
        assert result["confidence"] == 0.5  # Default for fallback

    def test_fallback_for_losing_trade_stop_loss(self, engine):
        """Should generate stop loss specific fallback for losing trade"""
        trade = {
            "symbol": "ETH-USD",
            "side": "buy",
            "is_winner": 0,
            "realized_pnl_percent": -2.0,
            "exit_reason": "stop_loss"
        }

        result = engine._generate_fallback_reflection(trade)

        assert result["reflection_type"] == "losing_trade"
        assert "stop" in result["lesson_learned"].lower()
        assert any("stop" in item.lower() for item in result["action_items"])

    def test_fallback_for_losing_trade_other(self, engine):
        """Should generate generic fallback for other losing trades"""
        trade = {
            "symbol": "SOL-USD",
            "side": "buy",
            "is_winner": 0,
            "realized_pnl_percent": -3.5,
            "exit_reason": "signal_exit"
        }

        result = engine._generate_fallback_reflection(trade)

        assert result["reflection_type"] == "losing_trade"
        assert result["applies_to_regimes"] == ["all"]
        assert result["applies_to_symbols"] == ["*"]


class TestReflectionTypes:
    """Test ReflectionType enum values"""

    def test_all_reflection_types_exist(self):
        """All expected reflection types should exist"""
        expected = [
            "winning_trade",
            "losing_trade",
            "missed_opportunity",
            "false_signal",
            "timing_error",
            "size_error",
            "exit_error",
            "market_regime"
        ]

        for name in expected:
            assert ReflectionType(name) is not None

    def test_reflection_type_values(self):
        """Enum values should match names"""
        assert ReflectionType.WINNING_TRADE.value == "winning_trade"
        assert ReflectionType.LOSING_TRADE.value == "losing_trade"
        assert ReflectionType.FALSE_SIGNAL.value == "false_signal"


class TestMarketRegimes:
    """Test MarketRegime enum values"""

    def test_all_market_regimes_exist(self):
        """All expected market regimes should exist"""
        expected = [
            "high_volatility",
            "low_volatility",
            "strong_uptrend",
            "strong_downtrend",
            "ranging",
            "breakout",
            "breakdown",
            "unknown"
        ]

        for name in expected:
            assert MarketRegime(name) is not None


class TestLessonsSummary:
    """Test learning summary generation"""

    @pytest.fixture
    async def engine_with_varied_lessons(self):
        """Create engine with varied lessons for summary testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        engine = ReflexionEngine(db_path)
        await engine.initialize()

        # Add varied lessons
        lesson_configs = [
            (True, 5.0, "winning_trade", 0.9),
            (True, 3.0, "winning_trade", 0.8),
            (False, -2.0, "losing_trade", 0.85),
            (False, -1.5, "timing_error", 0.7),
            (False, -3.0, "false_signal", 0.75),
        ]

        for is_win, pnl, rtype, conf in lesson_configs:
            reflection = Reflection(
                reflection_id=Reflection.generate_id(),
                trade_id=f"trade-{Reflection.generate_id()[:8]}",
                symbol="BTC-USD",
                created_at=datetime.now(timezone.utc),
                is_winner=is_win,
                pnl_percent=pnl,
                duration_hours=5.0,
                market_regime=MarketRegime.UNKNOWN,
                entry_atr=0.02,
                entry_price=50000.0,
                exit_price=50000.0 * (1 + pnl/100),
                exit_reason="signal_exit",
                reflection_type=ReflectionType(rtype),
                what_happened="Test happened",
                what_expected="Test expected",
                lesson_learned=f"Lesson from {rtype}",
                action_items=["Action 1"],
                confidence=conf,
                applies_to_regimes=["all"],
                applies_to_symbols=["*"]
            )
            await engine._store_reflection(reflection)

        yield engine

        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_lessons_summary_stats(self, engine_with_varied_lessons):
        """Summary should include correct statistics"""
        summary = await engine_with_varied_lessons.get_lessons_summary(
            symbol="BTC-USD",
            days=30,
            min_confidence=0.5
        )

        assert summary["stats"]["total_reflections"] == 5
        assert summary["stats"]["from_wins"] == 2
        assert summary["stats"]["from_losses"] == 3
        assert summary["stats"]["avg_confidence"] > 0.7

    @pytest.mark.asyncio
    async def test_lessons_summary_top_lessons(self, engine_with_varied_lessons):
        """Summary should include top lessons"""
        summary = await engine_with_varied_lessons.get_lessons_summary(
            days=30,
            min_confidence=0.5
        )

        assert "top_lessons" in summary
        assert len(summary["top_lessons"]) > 0

        # Top lesson should have highest confidence
        top = summary["top_lessons"][0]
        assert top["confidence"] >= 0.8


# Integration test with Truth Logger
class TestTruthLoggerIntegration:
    """Test integration with TruthLogger"""

    @pytest.mark.asyncio
    async def test_reflection_query_method(self):
        """TruthLogger should have reflection query method"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            from src.truth.logger import TruthLogger
            logger = TruthLogger(db_path)
            await logger.initialize()

            # Should have the method
            assert hasattr(logger, "get_relevant_lessons")
            assert hasattr(logger, "get_learning_summary")
            assert hasattr(logger, "backfill_reflections")

            # Query should return valid structure
            result = await logger.get_relevant_lessons(
                symbol="BTC-USD",
                signals={"atr": 2000, "current_price": 100000},
                market_context={}
            )

            assert "adjusted_confidence" in result
            assert "size_adjustment" in result
            assert "warnings" in result

        finally:
            os.unlink(db_path)
