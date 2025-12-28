"""
Integration Tests for ArgusNexus Learning System

Tests the complete learning pipeline:
1. Reflexion Layer - Post-trade analysis
2. Regime Detection - Market condition classification
3. Confidence Scoring - Signal quality filtering
4. PPO Agent - Adaptive position sizing

These tests verify that all components work together.
"""

import pytest
import tempfile
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Import learning modules
from src.learning import (
    # Reflexion
    ReflexionEngine,
    Reflection,
    ReflectionType,
    # Regime
    RegimeDetector,
    RegimeState,
    MarketRegime,
    detect_regime,
    REGIME_PARAMETERS,
    RegimeParameters,
    # Confidence
    ConfidenceScorer,
    ConfidenceScore,
    SignalQuality,
    HistoricalPerformance,
    score_signal,
    # PPO
    PPOAgent,
    PPOConfig,
    create_ppo_agent,
)


def make_regime_state(
    regime: MarketRegime = MarketRegime.RANGING,
    volatility: MarketRegime = MarketRegime.NORMAL_VOLATILITY,
    confidence: float = 0.5,
    atr_percent: float = 0.02,
    adx: float = 25.0,
) -> RegimeState:
    """Helper to create valid RegimeState objects for testing"""
    return RegimeState(
        current_regime=regime,
        volatility_regime=volatility,
        trend_regime=regime,
        confidence=confidence,
        detected_at=datetime.now(timezone.utc),
        parameters=REGIME_PARAMETERS.get(regime, RegimeParameters()),
        atr_percent=atr_percent,
        adx=adx,
        rsi=50.0,
        price_vs_sma20=0.0,
        price_vs_sma50=0.0,
        volume_ratio=1.0,
    )


class TestRegimeDetection:
    """Test regime detection module"""

    def test_detector_initialization(self):
        """Test RegimeDetector initializes correctly"""
        detector = RegimeDetector()
        assert detector is not None
        assert detector.volatility_lookback == 20

    def test_strong_uptrend_detection(self):
        """Test detection of strong uptrend"""
        detector = RegimeDetector()

        # Strong uptrend: high ADX, price above MAs
        state = detector.detect(
            current_price=55000,
            atr=1000,       # 1.8% volatility
            adx=35,         # Strong trend
            rsi=55,
            sma_20=52000,   # Price above SMA20
            sma_50=50000,   # Price above SMA50
        )

        assert state is not None
        assert isinstance(state, RegimeState)
        assert state.current_regime in [
            MarketRegime.STRONG_UPTREND,
            MarketRegime.WEAK_UPTREND,
            MarketRegime.BREAKOUT,
        ]

    def test_high_volatility_detection(self):
        """Test detection of high volatility regime"""
        detector = RegimeDetector()

        # High volatility: large ATR relative to price
        state = detector.detect(
            current_price=50000,
            atr=2500,       # 5% volatility - extreme
            adx=15,
            rsi=50,
        )

        assert state is not None
        # Should detect elevated volatility
        assert state.volatility_regime in [
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.EXTREME_VOLATILITY,
        ]

    def test_regime_parameters_exist(self):
        """Test that all regimes have parameters"""
        for regime in MarketRegime:
            assert regime in REGIME_PARAMETERS, f"Missing params for {regime}"
            params = REGIME_PARAMETERS[regime]
            assert hasattr(params, "position_size_multiplier")
            assert hasattr(params, "stop_atr_multiplier")
            assert hasattr(params, "allow_new_positions")

    def test_convenience_function(self):
        """Test detect_regime convenience function"""
        state = detect_regime(
            current_price=50000,
            atr=1000,
            adx=25,
        )
        assert isinstance(state, RegimeState)


class TestConfidenceScoring:
    """Test confidence scoring system"""

    def test_scorer_initialization(self):
        """Test ConfidenceScorer initializes correctly"""
        scorer = ConfidenceScorer()
        assert scorer is not None
        assert scorer.WEIGHT_SIGNAL == 0.40
        assert scorer.WEIGHT_REGIME == 0.25
        assert scorer.WEIGHT_HISTORY == 0.20
        assert scorer.WEIGHT_LESSONS == 0.15

    def test_score_strong_signal(self):
        """Test scoring of strong signal"""
        scorer = ConfidenceScorer()

        # Create a strong signal scenario
        signal_values = {
            "adx": 35,
            "atr": 1500,
            "current_price": 50000,
            "crossover_confirmed": True,
            "rsi": 45,
        }

        # Create favorable regime
        regime_state = make_regime_state(
            regime=MarketRegime.STRONG_UPTREND,
            volatility=MarketRegime.LOW_VOLATILITY,
            confidence=0.8,
        )

        score = scorer.score_signal(
            signal_type="long",
            signal_values=signal_values,
            regime_state=regime_state,
        )

        assert isinstance(score, ConfidenceScore)
        assert score.final_score > 0.5
        assert score.quality in [SignalQuality.GOOD, SignalQuality.EXCELLENT]
        assert score.recommended_action in ["execute", "reduce_size"]

    def test_score_weak_signal(self):
        """Test scoring of weak signal"""
        scorer = ConfidenceScorer()

        # Weak signal
        signal_values = {
            "adx": 12,
            "atr": 5000,
            "current_price": 50000,
            "rsi": 75,  # Overbought for long
        }

        # Unfavorable regime
        regime_state = make_regime_state(
            regime=MarketRegime.STRONG_DOWNTREND,
            volatility=MarketRegime.EXTREME_VOLATILITY,
            confidence=0.9,
        )

        score = scorer.score_signal(
            signal_type="long",
            signal_values=signal_values,
            regime_state=regime_state,
        )

        assert score.final_score < 0.5
        assert score.quality in [SignalQuality.POOR, SignalQuality.SKIP, SignalQuality.MARGINAL]

    def test_historical_performance_scoring(self):
        """Test historical performance affects score"""
        scorer = ConfidenceScorer()

        signal = {"adx": 25, "current_price": 50000}
        regime = make_regime_state()

        # Good historical performance
        good_history = HistoricalPerformance(
            total_trades=20,
            winning_trades=14,
            losing_trades=6,
            win_rate=0.7,
            avg_win_pct=3.5,
            avg_loss_pct=-1.5,
            profit_factor=2.33,
            expectancy=0.02,
        )

        score_with_history = scorer.score_signal(
            "long", signal, regime, historical_performance=good_history
        )

        # Poor historical performance
        bad_history = HistoricalPerformance(
            total_trades=20,
            winning_trades=6,
            losing_trades=14,
            win_rate=0.3,
            avg_win_pct=1.5,
            avg_loss_pct=-2.5,
            profit_factor=0.6,
            expectancy=-0.01,
        )

        score_without = scorer.score_signal(
            "long", signal, regime, historical_performance=bad_history
        )

        # Good history should score higher
        assert score_with_history.historical_success > score_without.historical_success

    def test_position_size_factor(self):
        """Test position size factor is calculated"""
        scorer = ConfidenceScorer()

        signal = {"adx": 30, "current_price": 50000}
        regime = make_regime_state(
            regime=MarketRegime.STRONG_UPTREND,
            volatility=MarketRegime.LOW_VOLATILITY,
            confidence=0.8,
        )

        score = scorer.score_signal("long", signal, regime)

        assert 0 <= score.position_size_factor <= 2.0

    def test_convenience_function(self):
        """Test score_signal convenience function"""
        regime = make_regime_state()

        result = score_signal(
            signal_type="long",
            signal_values={"adx": 20},
            regime_state=regime,
        )

        assert isinstance(result, ConfidenceScore)


class TestPPOAgent:
    """Test PPO agent for adaptive position sizing"""

    def test_agent_initialization(self):
        """Test PPO agent initializes correctly"""
        agent = PPOAgent()
        assert agent is not None
        assert not agent.is_burned_in
        assert len(agent.experience_buffer) == 0

    def test_state_features(self):
        """Test state feature extraction"""
        agent = PPOAgent()

        regime = make_regime_state(
            regime=MarketRegime.STRONG_UPTREND,
            volatility=MarketRegime.LOW_VOLATILITY,
            confidence=0.85,
            atr_percent=0.02,
        )

        state = agent.get_state_features(
            regime_state=regime,
            signal_strength=0.7,
            recent_performance={"win_rate": 0.6, "avg_pnl_percent": 1.5},
        )

        assert len(state) == 8
        assert all(isinstance(s, float) for s in state)
        # Check regime confidence is captured
        assert state[0] == 0.85

    def test_get_action(self):
        """Test action selection"""
        agent = PPOAgent()
        state = [0.5] * 8

        action, log_prob, value = agent.get_action(state, deterministic=True)

        assert 0 <= action <= 1
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_experience_storage(self):
        """Test experience storage in buffer"""
        agent = PPOAgent()

        for i in range(10):
            agent.store_experience(
                state=[0.5] * 8,
                action=0.5,
                reward=0.1 if i % 2 == 0 else -0.1,
                next_state=[0.5] * 8,
                done=False,
                log_prob=-0.5,
                value=0.1,
                trade_id=f"trade_{i}",
            )

        assert len(agent.experience_buffer) == 10
        assert agent.stats.total_experiences == 10

    def test_reward_calculation(self):
        """Test reward calculation logic"""
        agent = PPOAgent()

        # Profitable quick trade
        reward1 = agent.calculate_reward(
            pnl_percent=5.0,
            holding_period_hours=4,
            position_size=0.5,
        )

        # Loss on slow trade
        reward2 = agent.calculate_reward(
            pnl_percent=-3.0,
            holding_period_hours=48,
            position_size=0.8,
        )

        # Profit should be positive, loss negative
        assert reward1 > 0
        assert reward2 < 0
        # Large size on loss should amplify penalty
        assert abs(reward2) > abs(agent.calculate_reward(-3.0, 48, 0.3))

    def test_position_recommendation(self):
        """Test position size recommendation"""
        agent = PPOAgent()

        regime = make_regime_state(
            regime=MarketRegime.STRONG_UPTREND,
            volatility=MarketRegime.LOW_VOLATILITY,
            confidence=0.8,
            atr_percent=0.02,
        )

        result = agent.recommend_position_size(
            regime_state=regime,
            signal_strength=0.7,
            confidence_score=0.65,
        )

        assert "recommended_size" in result
        assert 0 < result["recommended_size"] <= 1.0
        assert "factors" in result
        assert "state_summary" in result

    def test_burnin_flag(self):
        """Test burn-in tracking"""
        config = PPOConfig(min_experiences=5)
        agent = PPOAgent(config=config)

        assert not agent.is_burned_in

        # Add experiences until burn-in
        for i in range(5):
            agent.store_experience(
                state=[0.5] * 8,
                action=0.5,
                reward=0.1,
                next_state=[0.5] * 8,
                done=False,
                log_prob=-0.5,
                value=0.1,
            )

        assert agent.is_burned_in

    def test_create_convenience_function(self):
        """Test create_ppo_agent convenience function"""
        agent = create_ppo_agent(learning_rate=0.001)
        assert agent is not None
        assert agent.config.learning_rate == 0.001


class TestLearningIntegration:
    """Integration tests for the complete learning pipeline"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Initialize schema
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                entry_timestamp TEXT NOT NULL,
                exit_timestamp TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL,
                status TEXT DEFAULT 'open',
                realized_pnl_percent REAL,
                is_winner INTEGER,
                exit_reason TEXT,
                entry_decision_id TEXT
            );

            CREATE TABLE decisions (
                decision_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                signal_values TEXT,
                market_context TEXT,
                created_at TEXT
            );
        """)
        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_regime_to_confidence_flow(self):
        """Test regime detection feeding into confidence scoring"""
        detector = RegimeDetector()

        # Detect uptrend regime
        regime_state = detector.detect(
            current_price=55000,
            atr=1000,
            adx=30,
            rsi=55,
            sma_20=52000,
            sma_50=50000,
        )

        # Use regime in confidence scoring
        scorer = ConfidenceScorer()
        signal = {
            "adx": 28,
            "current_price": 55000,
            "atr": 1000,
        }

        score = scorer.score_signal("long", signal, regime_state)

        # Uptrend regime should boost long signal confidence
        assert score.regime_alignment >= 0.4
        assert score.regime == regime_state.current_regime

    def test_confidence_to_ppo_flow(self):
        """Test confidence score feeding into PPO recommendations"""
        # Get confidence score
        regime = make_regime_state(
            regime=MarketRegime.STRONG_UPTREND,
            volatility=MarketRegime.LOW_VOLATILITY,
            confidence=0.8,
            atr_percent=0.02,
        )

        scorer = ConfidenceScorer()
        conf_score = scorer.score_signal(
            "long",
            {"adx": 30, "current_price": 50000},
            regime,
        )

        # Feed into PPO
        agent = PPOAgent()
        recommendation = agent.recommend_position_size(
            regime_state=regime,
            signal_strength=conf_score.signal_strength,
            confidence_score=conf_score.position_size_factor,
        )

        # Should produce reasonable recommendation
        assert 0 < recommendation["recommended_size"] <= 1.0
        # Confidence component should be included
        assert recommendation["confidence_component"] == conf_score.position_size_factor

    def test_full_pipeline_simulation(self):
        """Test complete learning pipeline with simulated trades"""
        # Initialize all components
        detector = RegimeDetector()
        scorer = ConfidenceScorer()
        config = PPOConfig(min_experiences=5, update_frequency=10)
        agent = PPOAgent(config=config)

        # Simulate 10 trades with varying conditions
        for i in range(10):
            price = 50000 + i * 100

            # 1. Detect regime (simulate changing conditions)
            regime_state = detector.detect(
                current_price=price,
                atr=1000 + i * 50,
                adx=25 + i,
                rsi=50 + i % 10,
            )

            # 2. Score signal
            signal = {"adx": 25 + i, "current_price": price, "atr": 1000}
            conf_score = scorer.score_signal("long", signal, regime_state)

            # 3. Get PPO recommendation
            state = agent.get_state_features(
                regime_state=regime_state,
                signal_strength=conf_score.signal_strength,
            )

            action, log_prob, value = agent.get_action(state)

            # 4. Simulate trade outcome
            pnl = 2.0 if i % 3 != 0 else -1.5  # 67% win rate
            reward = agent.calculate_reward(pnl, 8.0, action)

            # 5. Store experience
            next_state = agent.get_state_features(
                regime_state=regime_state,
                signal_strength=conf_score.signal_strength,
                recent_performance={"win_rate": 0.6, "avg_pnl_percent": pnl},
            )

            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=False,
                log_prob=log_prob,
                value=value,
            )

        # Verify system state
        assert agent.is_burned_in
        assert agent.stats.total_experiences == 10
        assert len(agent.recent_rewards) == 10

    def test_regime_adaptation(self):
        """Test that system adapts recommendations based on regime"""
        scorer = ConfidenceScorer()
        agent = PPOAgent()

        signal = {"adx": 25, "current_price": 50000}

        # Bull market regime
        bull_regime = make_regime_state(
            regime=MarketRegime.STRONG_UPTREND,
            volatility=MarketRegime.LOW_VOLATILITY,
            confidence=0.9,
        )

        bull_score = scorer.score_signal("long", signal, bull_regime)
        bull_rec = agent.recommend_position_size(
            regime_state=bull_regime,
            signal_strength=bull_score.signal_strength,
            confidence_score=bull_score.position_size_factor,
        )

        # Bear market regime
        bear_regime = make_regime_state(
            regime=MarketRegime.STRONG_DOWNTREND,
            volatility=MarketRegime.HIGH_VOLATILITY,
            confidence=0.9,
        )

        bear_score = scorer.score_signal("long", signal, bear_regime)
        bear_rec = agent.recommend_position_size(
            regime_state=bear_regime,
            signal_strength=bear_score.signal_strength,
            confidence_score=bear_score.position_size_factor,
        )

        # Bull market should have higher long score
        assert bull_score.regime_alignment > bear_score.regime_alignment
        # Position recommendations should differ
        assert bull_rec["recommended_size"] != bear_rec["recommended_size"]


class TestQualityThresholds:
    """Test signal quality classification"""

    def test_quality_thresholds(self):
        """Verify quality threshold classifications"""
        scorer = ConfidenceScorer()

        assert scorer.EXCELLENT_THRESHOLD == 0.80
        assert scorer.GOOD_THRESHOLD == 0.60
        assert scorer.MARGINAL_THRESHOLD == 0.40
        assert scorer.POOR_THRESHOLD == 0.25

    def test_quality_classification(self):
        """Test quality classification logic"""
        scorer = ConfidenceScorer()

        assert scorer._classify_quality(0.85) == SignalQuality.EXCELLENT
        assert scorer._classify_quality(0.70) == SignalQuality.GOOD
        assert scorer._classify_quality(0.50) == SignalQuality.MARGINAL
        assert scorer._classify_quality(0.30) == SignalQuality.POOR
        assert scorer._classify_quality(0.20) == SignalQuality.SKIP


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_minimal_inputs(self):
        """Test regime detection with minimal inputs"""
        detector = RegimeDetector()

        # Just required parameters
        state = detector.detect(
            current_price=50000,
            atr=1000,
            adx=25,
        )

        # Should return a state, not crash
        assert isinstance(state, RegimeState)

    def test_zero_atr(self):
        """Test confidence scoring with zero ATR"""
        scorer = ConfidenceScorer()

        signal = {"adx": 25, "atr": 0, "current_price": 50000}
        regime = make_regime_state()

        score = scorer.score_signal("long", signal, regime)
        assert isinstance(score, ConfidenceScore)

    def test_ppo_empty_buffer_recommendation(self):
        """Test PPO recommendation with empty buffer"""
        agent = PPOAgent()

        result = agent.recommend_position_size(confidence_score=0.5)

        # Should still work, using confidence more heavily
        assert result["recommended_size"] > 0
        assert not result["is_burned_in"]
        assert result["ppo_weight"] == 0.2  # Lower weight before burn-in
