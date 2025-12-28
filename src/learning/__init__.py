"""
ArgusNexus V4 Learning Module

The Reflexion Layer - Where Argus learns from its mistakes.

This module implements a self-improving trading system based on the Reflexion
framework: LLM-powered verbal reinforcement learning without fine-tuning.

Key Components:
- ReflexionEngine: Post-trade analysis and lesson learning
- Reflection: A learned lesson from a completed trade
- ReflectionMemory: Queryable storage for past lessons

The Learning Loop:
1. Trade closes -> Trigger reflection
2. LLM analyzes: What happened? What did we expect? What went wrong/right?
3. Store the lesson with context vectors
4. Future decisions query similar past lessons
5. Lessons inform position sizing, entry timing, exit decisions

"Those who cannot remember the past are condemned to repeat it."
- George Santayana

In Argus: "Those trades we don't learn from are mistakes we'll make again."
"""

from .reflexion import (
    ReflexionEngine,
    Reflection,
    ReflectionType,
    MarketRegime as ReflectionMarketRegime,
)

from .regime import (
    RegimeDetector,
    RegimeState,
    RegimeParameters,
    MarketRegime,
    detect_regime,
    REGIME_PARAMETERS,
)

from .confidence import (
    ConfidenceScorer,
    ConfidenceScore,
    SignalQuality,
    HistoricalPerformance,
    score_signal,
)

from .ppo_agent import (
    PPOAgent,
    PPOConfig,
    Experience,
    TrainingStats,
    create_ppo_agent,
)

__all__ = [
    # Reflexion Layer
    "ReflexionEngine",
    "Reflection",
    "ReflectionType",
    # Regime Detection
    "RegimeDetector",
    "RegimeState",
    "RegimeParameters",
    "MarketRegime",
    "detect_regime",
    "REGIME_PARAMETERS",
    # Confidence Scoring
    "ConfidenceScorer",
    "ConfidenceScore",
    "SignalQuality",
    "HistoricalPerformance",
    "score_signal",
    # Online Learning (PPO)
    "PPOAgent",
    "PPOConfig",
    "Experience",
    "TrainingStats",
    "create_ppo_agent",
]
