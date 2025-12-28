"""
LLM-Powered Decision Insights - Real-time AI Analysis with Market Verification

Uses OpenAI GPT-5.2 to provide intelligent, honest assessments of trading decisions.
Verifies decisions against REAL-TIME market data including:
- Current price vs decision price
- Technical indicators (EMA, RSI, ATR, Bollinger)
- Social sentiment from LunarCrush
- Breaking news from web search

Usage:
    from src.truth.llm_insights import LLMInsightGenerator

    generator = LLMInsightGenerator()
    insight = await generator.analyze_decision(decision_dict, verify_with_market=True)
    # Returns: {
    #     "summary": "Human-readable explanation",
    #     "assessment": "Honest critique with real-time verification",
    #     "market_verification": {...},
    #     "risk_flags": [...],
    #     "confidence": 0.85
    # }
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import OpenAI - graceful fallback if not installed
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. LLM insights will use fallback mode.")


@dataclass
class MarketVerification:
    """Real-time market verification of the decision"""
    decision_price: float           # Price when decision was made
    current_price: float            # Current market price
    price_change_pct: float         # Percent change since decision
    validation_status: str          # "VALIDATED", "INVALIDATED", "NEUTRAL"
    validation_grade: str           # "good", "poor", "neutral"
    time_since_decision: str        # How long ago
    current_trend: str              # Current market trend
    social_sentiment: Optional[str] # LunarCrush sentiment if available
    galaxy_score: Optional[float]   # LunarCrush galaxy score if available
    breaking_news: List[str]        # Recent news headlines

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionInsight:
    """Structured insight output from LLM analysis"""
    summary: str                    # Plain English explanation of what happened
    assessment: str                 # Honest critique - is this a good decision?
    verdict: str                    # GOOD, BAD, or NEUTRAL - clear judgment
    verdict_reason: str             # One-line explanation of the verdict
    key_drivers: List[str]          # What factors drove this decision
    risk_flags: List[str]           # Potential concerns or warnings
    opportunities: List[str]        # Missed opportunities or better alternatives
    confidence_score: float         # LLM's confidence in the assessment (0-1)
    recommendation: str             # What should happen next
    generated_at: str               # Timestamp of insight generation
    market_verification: Optional[MarketVerification] = None  # Real-time verification

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.market_verification:
            result["market_verification"] = self.market_verification.to_dict()
        return result


class LLMInsightGenerator:
    """
    Generates AI-powered insights for trading decisions with real-time market verification.

    Uses GPT-5.2 to analyze decision data and provide honest assessments
    about the quality and reasoning behind each trading decision.

    Integrates:
    - Coinbase: Real-time prices and technical indicators
    - LunarCrush: Social sentiment and Galaxy Score
    - Tavily: Breaking news and web search
    """

    SYSTEM_PROMPT = """You are a senior trading analyst reviewing automated trading decisions.
Your job is to provide HONEST, CRITICAL assessments - not cheerleading.

IMPORTANT: You have access to REAL-TIME MARKET DATA including:
- Current price and how it compares to the decision price
- Technical indicators (EMA, RSI, ATR, Bollinger Bands)
- Social sentiment from LunarCrush (Galaxy Score, social volume)
- Breaking news headlines

Use this real-time data to VERIFY whether the decision was correct:
- If price moved in the expected direction: acknowledge the good call
- If price moved against the decision: flag this as a concern
- Consider social sentiment when evaluating timing
- Factor in breaking news that might affect the asset

CRITICAL: You MUST provide a clear VERDICT on whether this was a GOOD or BAD decision.

VERDICT CRITERIA:
- GOOD: Decision was correct given the information available, AND current market data confirms it was the right call
- BAD: Decision was wrong - price moved against it, or the logic was flawed, or better options existed
- NEUTRAL: Too early to judge, or the decision was reasonable but outcome is unclear

For each decision, analyze:
1. VERDICT: State clearly GOOD, BAD, or NEUTRAL
2. VERDICT_REASON: One sentence explaining why (e.g., "Price dropped 5% after long signal" or "Correctly identified trend reversal")
3. SUMMARY: Explain what happened in plain English (2-3 sentences)
4. ASSESSMENT: Give your honest opinion based on CURRENT market state
5. KEY DRIVERS: What factors drove this decision?
6. RISK FLAGS: What concerns exist, especially given current market conditions?
7. OPPORTUNITIES: Were there better alternatives? What does current data suggest?
8. RECOMMENDATION: What should the trader do RIGHT NOW based on current market state?

Be direct and critical. Use the real-time data to validate or challenge the decision.
Traders need truth, not validation.

Output your analysis as JSON with these exact keys:
- verdict (string: "GOOD", "BAD", or "NEUTRAL")
- verdict_reason (string: one sentence explanation)
- summary (string)
- assessment (string)
- key_drivers (array of strings)
- risk_flags (array of strings)
- opportunities (array of strings)
- recommendation (string)
- confidence_score (float 0-1, your confidence in this assessment)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5.2"):
        """
        Initialize the LLM Insight Generator.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use (default: gpt-5.2)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Optional[AsyncOpenAI] = None
        self._market_fetcher = None

        if OPENAI_AVAILABLE and self.api_key:
            self._client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"LLM Insight Generator initialized with {model}")
        else:
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI not available - using fallback insights")
            elif not self.api_key:
                logger.warning("No OPENAI_API_KEY - using fallback insights")

    async def _get_market_fetcher(self):
        """Lazy-load the market data fetcher"""
        if self._market_fetcher is None:
            from .market_data import MarketDataFetcher
            self._market_fetcher = MarketDataFetcher()
        return self._market_fetcher

    async def _fetch_market_verification(
        self,
        symbol: str,
        decision_price: float,
        decision_type: str,
        decision_timestamp: str
    ) -> tuple:
        """
        Fetch real-time market data and verify the decision.

        Returns:
            Tuple of (MarketVerification, market_context_string)
        """
        try:
            fetcher = await self._get_market_fetcher()
            snapshot = await fetcher.get_market_snapshot(symbol)

            current_price = snapshot.current_price

            # Calculate price change if we have a decision price
            if decision_price > 0:
                price_change_pct = ((current_price - decision_price) / decision_price * 100)

                # Determine validation status
                if decision_type in ["long", "signal_long", "buy"]:
                    if price_change_pct > 1:
                        validation_status = "VALIDATED - price rose since long signal"
                        validation_grade = "good"
                    elif price_change_pct < -2:
                        validation_status = "INVALIDATED - price fell since long signal"
                        validation_grade = "poor"
                    else:
                        validation_status = "NEUTRAL - insufficient movement"
                        validation_grade = "neutral"
                elif decision_type in ["short", "signal_short", "sell", "close", "signal_close"]:
                    if price_change_pct < -1:
                        validation_status = "VALIDATED - price fell since close/short"
                        validation_grade = "good"
                    elif price_change_pct > 2:
                        validation_status = "INVALIDATED - missed upside"
                        validation_grade = "poor"
                    else:
                        validation_status = "NEUTRAL - insufficient movement"
                        validation_grade = "neutral"
                else:
                    validation_status = "NEUTRAL - hold/no action"
                    validation_grade = "neutral"
            else:
                # No decision price available - can't validate, but still provide market data
                price_change_pct = 0
                validation_status = "UNABLE TO VALIDATE - no decision price recorded"
                validation_grade = "unknown"

            # Extract social sentiment
            social_sentiment = None
            galaxy_score = None
            if snapshot.social_metrics:
                social_sentiment = snapshot.social_metrics.sentiment_label
                galaxy_score = snapshot.social_metrics.galaxy_score

            # Extract news headlines
            breaking_news = [n.title for n in snapshot.recent_news[:5]]

            verification = MarketVerification(
                decision_price=decision_price,
                current_price=current_price,
                price_change_pct=round(price_change_pct, 2),
                validation_status=validation_status,
                validation_grade=validation_grade,
                time_since_decision=decision_timestamp,
                current_trend=snapshot.indicators.trend_direction,
                social_sentiment=social_sentiment,
                galaxy_score=galaxy_score,
                breaking_news=breaking_news
            )

            # Get the LLM-formatted context
            market_context = snapshot.to_llm_context()

            return verification, market_context

        except Exception as e:
            logger.error(f"Failed to fetch market verification: {e}")
            return None, None

    @property
    def is_available(self) -> bool:
        """Check if LLM insights are available"""
        return self._client is not None

    async def analyze_decision(
        self,
        decision: Dict[str, Any],
        include_history: Optional[List[Dict[str, Any]]] = None,
        verify_with_market: bool = True
    ) -> DecisionInsight:
        """
        Analyze a trading decision and generate insights with real-time verification.

        Args:
            decision: Decision data dictionary with keys like:
                - symbol, result, result_reason
                - signal_values (dict or JSON string)
                - market_context (dict or JSON string)
                - risk_checks (dict or JSON string)
            include_history: Optional list of recent decisions for context
            verify_with_market: If True, fetch real-time market data for verification

        Returns:
            DecisionInsight with analysis results and market verification
        """
        if not self.is_available:
            return self._generate_fallback_insight(decision)

        market_verification = None
        market_context_str = ""

        try:
            # Fetch real-time market data if requested
            if verify_with_market:
                symbol = decision.get("symbol", "")
                result = decision.get("result", "")
                decision_timestamp = decision.get("timestamp", "unknown")

                # Try to extract decision price from BOTH market_context and signal_values
                market_context = decision.get("market_context") or {}
                signal_values = decision.get("signal_values") or {}

                if isinstance(market_context, str):
                    try:
                        market_context = json.loads(market_context)
                    except:
                        market_context = {}

                if isinstance(signal_values, str):
                    try:
                        signal_values = json.loads(signal_values)
                    except:
                        signal_values = {}

                # Extract decision price - check signal_values first (most common), then market_context
                decision_price = 0
                # Check signal_values first - this is where current_price usually lives
                if signal_values.get("current_price"):
                    decision_price = signal_values.get("current_price")
                elif signal_values.get("price"):
                    decision_price = signal_values.get("price")
                elif signal_values.get("entry_price"):
                    decision_price = signal_values.get("entry_price")
                # Fallback to market_context
                elif market_context.get("current_price"):
                    decision_price = market_context.get("current_price")
                elif isinstance(market_context.get("price"), dict):
                    # Nested format: {"price": {"current": 98000, ...}}
                    decision_price = market_context.get("price", {}).get("current", 0)
                elif market_context.get("price"):
                    # Flat format: {"price": 98000}
                    decision_price = market_context.get("price")
                elif market_context.get("entry_price"):
                    decision_price = market_context.get("entry_price")

                # Always try to fetch market data if we have a symbol
                if symbol:
                    market_verification, market_context_str = await self._fetch_market_verification(
                        symbol=symbol,
                        decision_price=float(decision_price) if decision_price else 0,
                        decision_type=result,
                        decision_timestamp=decision_timestamp
                    )

            # Format decision data for the LLM
            formatted_decision = self._format_decision_for_llm(decision)

            # Build the user message with real-time data
            user_message = f"""Analyze this trading decision:

{formatted_decision}"""

            # Add real-time market data if available
            if market_context_str and market_verification:
                user_message += f"""

{market_context_str}

DECISION VERIFICATION:"""
                if market_verification.decision_price > 0:
                    user_message += f"""
- Decision Price: ${market_verification.decision_price:,.2f}
- Current Price: ${market_verification.current_price:,.2f}
- Price Change: {market_verification.price_change_pct:+.2f}%
- Status: {market_verification.validation_status}"""
                else:
                    user_message += f"""
- Current Price: ${market_verification.current_price:,.2f}
- Decision Price: Not recorded (cannot calculate price change)
- Status: {market_verification.validation_status}"""

                user_message += """

Use this real-time data to validate or challenge the original decision."""

            user_message += "\n\nProvide your honest assessment based on CURRENT market conditions."

            if include_history:
                history_summary = self._summarize_history(include_history)
                user_message += f"\n\nRecent trading history:\n{history_summary}"

            # Call OpenAI GPT-5.2
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_completion_tokens=2000
            )

            # Parse the response
            content = response.choices[0].message.content

            # Handle potential markdown code blocks in response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content.strip())

            return DecisionInsight(
                summary=parsed.get("summary", "Unable to generate summary"),
                assessment=parsed.get("assessment", "Unable to generate assessment"),
                verdict=parsed.get("verdict", "NEUTRAL").upper(),
                verdict_reason=parsed.get("verdict_reason", "Unable to determine"),
                key_drivers=parsed.get("key_drivers", []),
                risk_flags=parsed.get("risk_flags", []),
                opportunities=parsed.get("opportunities", []),
                confidence_score=float(parsed.get("confidence_score", 0.5)),
                recommendation=parsed.get("recommendation", ""),
                generated_at=datetime.utcnow().isoformat(),
                market_verification=market_verification
            )

        except Exception as e:
            logger.error(f"LLM insight generation failed: {e}")
            return self._generate_fallback_insight(decision, error=str(e))

    def _format_decision_for_llm(self, decision: Dict[str, Any]) -> str:
        """Format decision data into a readable string for the LLM"""
        lines = []

        # Basic info
        lines.append(f"Symbol: {decision.get('symbol', 'UNKNOWN')}")
        lines.append(f"Result: {decision.get('result', 'N/A')}")
        lines.append(f"Timestamp: {decision.get('timestamp', 'N/A')}")
        lines.append(f"Strategy: {decision.get('strategy_name', 'N/A')}")
        lines.append(f"Reason: {decision.get('result_reason', 'N/A')}")

        # Market context / signal values
        context = decision.get("market_context") or decision.get("signal_values") or {}
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except:
                context = {}

        if context:
            lines.append("\nMarket Context:")
            for key, value in context.items():
                readable_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    if 0 < abs(value) < 1:
                        lines.append(f"  {readable_key}: {value:.1%}")
                    else:
                        lines.append(f"  {readable_key}: {value:.4f}")
                elif isinstance(value, bool):
                    lines.append(f"  {readable_key}: {'Yes' if value else 'No'}")
                else:
                    lines.append(f"  {readable_key}: {value}")

        # Risk checks
        risk_checks = decision.get("risk_checks", {})
        if isinstance(risk_checks, str):
            try:
                risk_checks = json.loads(risk_checks)
            except:
                risk_checks = {}

        if risk_checks:
            lines.append("\nRisk Checks:")
            for check, result in risk_checks.items():
                if isinstance(result, dict):
                    passed = "PASSED" if result.get("passed", True) else "FAILED"
                    lines.append(f"  {check}: {passed}")

        return "\n".join(lines)

    def _summarize_history(self, history: List[Dict[str, Any]]) -> str:
        """Summarize recent trading history for context"""
        if not history:
            return "No recent history available."

        lines = []
        for i, decision in enumerate(history[-5:]):  # Last 5 decisions
            result = decision.get("result", "unknown")
            symbol = decision.get("symbol", "?")
            reason = decision.get("result_reason", "")[:50]
            lines.append(f"{i+1}. {symbol} - {result}: {reason}...")

        return "\n".join(lines)

    def _generate_fallback_insight(
        self,
        decision: Dict[str, Any],
        error: Optional[str] = None
    ) -> DecisionInsight:
        """Generate a basic insight when LLM is not available"""

        result = decision.get("result", "unknown")
        result_reason = decision.get("result_reason", "No reason provided")
        symbol = decision.get("symbol", "UNKNOWN")

        # Parse context for basic analysis
        context = decision.get("market_context") or decision.get("signal_values") or {}
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except:
                context = {}

        # Generate basic summary
        summary = f"Decision: {result} for {symbol}. {result_reason}"

        # Basic assessment based on result type
        if "rejected" in result.lower():
            assessment = "Trade was blocked by risk management. This is protective behavior."
        elif "signal_long" in result.lower() or "signal_short" in result.lower():
            assessment = "Entry signal generated and approved. Monitor position closely."
        elif "hold" in result.lower() or "no_signal" in result.lower():
            assessment = "No action taken. Market conditions don't warrant a trade."
        else:
            assessment = "Standard decision logged. Review context for details."

        # Extract key metrics
        key_drivers = []
        if context.get("alloc_drift"):
            key_drivers.append(f"Allocation drift: {context['alloc_drift']:.1%}")
        if context.get("dd_state") and context.get("dd_state") != "normal":
            key_drivers.append(f"Drawdown state: {context['dd_state']}")
        if context.get("core_regime"):
            key_drivers.append(f"Market regime: {context['core_regime']}")

        if not key_drivers:
            key_drivers = ["See market context for details"]

        # Risk flags
        risk_flags = []
        dd_multiplier = context.get("dd_multiplier", 1.0)
        if isinstance(dd_multiplier, (int, float)) and dd_multiplier < 0.5:
            risk_flags.append("Drawdown protection significantly reducing exposure")
        if context.get("recovery_mode"):
            risk_flags.append("In recovery mode - extra caution warranted")

        if error:
            risk_flags.append(f"LLM analysis failed: {error}")

        # Determine fallback verdict based on result type
        if "rejected" in result.lower():
            verdict = "NEUTRAL"
            verdict_reason = "Trade blocked by risk management - protective behavior"
        elif error:
            verdict = "NEUTRAL"
            verdict_reason = "Unable to analyze - LLM unavailable"
        else:
            verdict = "NEUTRAL"
            verdict_reason = "Insufficient data for judgment"

        return DecisionInsight(
            summary=summary,
            assessment=assessment if not error else f"[Fallback mode] {assessment}",
            verdict=verdict,
            verdict_reason=verdict_reason,
            key_drivers=key_drivers,
            risk_flags=risk_flags if risk_flags else ["No immediate concerns"],
            opportunities=[],
            confidence_score=0.3 if error else 0.5,
            recommendation="Enable LLM insights for deeper analysis" if error else "Continue monitoring",
            generated_at=datetime.utcnow().isoformat()
        )


# Convenience function for one-off analysis
async def analyze_decision(decision: Dict[str, Any]) -> DecisionInsight:
    """
    Convenience function to analyze a single decision.

    Creates a temporary generator instance for one-off use.
    For repeated use, instantiate LLMInsightGenerator directly.
    """
    generator = LLMInsightGenerator()
    return await generator.analyze_decision(decision)
