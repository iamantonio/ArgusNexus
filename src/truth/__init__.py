# Truth Engine - THE CORE OF V4
#
# "If we can't explain it, we don't trade it"
# "If we can't log it, it didn't happen"
#
# Three Tables:
# - decisions: Why did we consider this trade?
# - orders: What orders were placed?
# - trades: What was the P&L outcome?
#
# Every trade traceable: trades -> orders -> decisions

from .schema import Decision, Order, Trade, DecisionResult, OrderSide, OrderStatus, ExitReason
from .logger import TruthLogger
from .narrative import generate_narrative, NarrativeOutput

# LLM insights (optional - requires openai package)
try:
    from .llm_insights import LLMInsightGenerator, DecisionInsight, MarketVerification, analyze_decision
except ImportError:
    LLMInsightGenerator = None
    DecisionInsight = None
    MarketVerification = None
    analyze_decision = None

# Market data fetcher for real-time verification
try:
    from .market_data import MarketDataFetcher, MarketSnapshot, SocialMetrics, TechnicalIndicators
except ImportError:
    MarketDataFetcher = None
    MarketSnapshot = None
    SocialMetrics = None
    TechnicalIndicators = None

__all__ = [
    # Schema
    "Decision",
    "Order",
    "Trade",
    "DecisionResult",
    "OrderSide",
    "OrderStatus",
    "ExitReason",
    # Logger
    "TruthLogger",
    # Narrative
    "generate_narrative",
    "NarrativeOutput",
    # LLM Insights
    "LLMInsightGenerator",
    "DecisionInsight",
    "MarketVerification",
    "analyze_decision",
    # Market Data
    "MarketDataFetcher",
    "MarketSnapshot",
    "SocialMetrics",
    "TechnicalIndicators",
]
