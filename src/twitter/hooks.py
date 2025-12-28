"""
Twitter Integration Hooks for Truth Engine

Automatically posts to Twitter when trading events occur.
Connects the Truth Engine's decision logging to Twitter content.

Events handled:
- Decision made (entry signal)
- Trade opened
- Trade closed (win/loss)
- Trade rejected (discipline showcase)
- Weekly summary
- Lessons learned
"""

# Load environment variables
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import logging

from .client import TwitterClient, MockTwitterClient
from .formatter import TweetFormatter, TweetContent
from .scheduler import TweetScheduler, ContentScheduleManager, Priority

logger = logging.getLogger(__name__)


class TradingEvent(Enum):
    """Trading events that trigger tweets"""
    DECISION_ENTRY = "decision_entry"      # Strategy signaled entry
    DECISION_REJECTED = "decision_rejected"  # Signal rejected by risk
    TRADE_OPENED = "trade_opened"          # Order filled
    TRADE_CLOSED_WIN = "trade_closed_win"  # Profitable exit
    TRADE_CLOSED_LOSS = "trade_closed_loss"  # Loss exit
    DAILY_SUMMARY = "daily_summary"        # End of day report
    WEEKLY_SUMMARY = "weekly_summary"      # End of week report
    LESSON_GENERATED = "lesson_generated"  # Learning system output


@dataclass
class TwitterHookConfig:
    """Configuration for Twitter hooks"""
    enabled: bool = True
    post_entries: bool = True
    post_exits: bool = True
    post_rejections: bool = True
    post_lessons: bool = True
    post_daily: bool = True
    post_weekly: bool = True
    generate_threads: bool = True
    simulation_mode: bool = False
    decision_base_url: str = "http://localhost:8000/decision"
    xai_api_key: Optional[str] = None


class TwitterHook:
    """
    Main hook class for Twitter integration.

    Registers with the trading engine to receive events
    and automatically generates/posts tweets.
    """

    def __init__(self, config: TwitterHookConfig = None):
        """
        Initialize Twitter hook.

        Args:
            config: TwitterHookConfig with settings
        """
        self.config = config or TwitterHookConfig()

        # Initialize components
        if self.config.simulation_mode:
            self.client = MockTwitterClient()
        else:
            self.client = TwitterClient()

        self.formatter = TweetFormatter(
            xai_api_key=self.config.xai_api_key,
            decision_base_url=self.config.decision_base_url,
        )

        self.scheduler = TweetScheduler(self.client)
        self.content_manager = ContentScheduleManager(self.scheduler, self.formatter)

        # Event handlers
        self._handlers: Dict[TradingEvent, List[Callable]] = {
            event: [] for event in TradingEvent
        }

        # State tracking
        self._pending_entries: Dict[str, Dict] = {}  # decision_id -> decision
        self._running = False

    async def start(self):
        """Start the Twitter hook (and scheduler)"""
        if not self.config.enabled:
            logger.info("Twitter hook disabled in config")
            return

        await self.scheduler.start()
        self._running = True
        mode = "SIMULATION" if self.config.simulation_mode else "LIVE"
        logger.info(f"Twitter hook started ({mode} mode)")

    async def stop(self):
        """Stop the Twitter hook"""
        await self.scheduler.stop()
        self._running = False
        logger.info("Twitter hook stopped")

    def register_handler(self, event: TradingEvent, handler: Callable):
        """Register a custom handler for an event"""
        self._handlers[event].append(handler)

    async def _notify_handlers(self, event: TradingEvent, data: Dict[str, Any]):
        """Notify all registered handlers for an event"""
        for handler in self._handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Handler error for {event}: {e}")

    # ==================== EVENT HANDLERS ====================

    async def on_decision(self, decision: Dict[str, Any]):
        """
        Handle a new decision from the strategy.

        Called when the strategy evaluates and produces a decision.
        """
        if not self.config.enabled or not self._running:
            return

        result = decision.get("result", "")

        # Entry signal
        if result in ["signal_long", "signal_short"] and self.config.post_entries:
            await self._handle_entry_decision(decision)

        # Rejection
        elif result == "risk_rejected" and self.config.post_rejections:
            await self._handle_rejection(decision)

        await self._notify_handlers(TradingEvent.DECISION_ENTRY, decision)

    async def _handle_entry_decision(self, decision: Dict[str, Any]):
        """Handle a trade entry decision"""
        decision_id = decision.get("decision_id", "")

        # Store for later matching with trade
        self._pending_entries[decision_id] = decision

        # Schedule entry tweet
        await self.content_manager.schedule_trade_entry(
            decision,
            generate_thread=self.config.generate_threads,
        )

        logger.info(f"[TWITTER] Scheduled entry tweet for decision {decision_id[:8]}...")

    async def _handle_rejection(self, decision: Dict[str, Any]):
        """Handle a trade rejection (discipline showcase)"""
        await self.content_manager.schedule_rejection(decision)

        decision_id = decision.get("decision_id", "")
        logger.info(f"[TWITTER] Scheduled rejection tweet for decision {decision_id[:8]}...")

        await self._notify_handlers(TradingEvent.DECISION_REJECTED, decision)

    async def on_trade_opened(self, trade: Dict[str, Any]):
        """
        Handle a trade being opened (order filled).

        This is called after the entry order fills.
        The entry tweet may already be posted.
        """
        if not self.config.enabled or not self._running:
            return

        # We've already posted entry tweet in on_decision
        # This is just for tracking
        await self._notify_handlers(TradingEvent.TRADE_OPENED, trade)

    async def on_trade_closed(self, trade: Dict[str, Any]):
        """
        Handle a trade being closed.

        Generates exit tweet with P&L and lessons.
        """
        if not self.config.enabled or not self._running:
            return

        if not self.config.post_exits:
            return

        decision_id = trade.get("decision_id", "")
        is_winner = trade.get("is_winner", trade.get("net_pnl", 0) > 0)

        # Get original entry decision
        entry_decision = self._pending_entries.pop(decision_id, None)

        if entry_decision is None:
            # Try to construct minimal decision from trade
            entry_decision = {
                "decision_id": decision_id,
                "symbol": trade.get("symbol", "BTC-USD"),
            }

        # Schedule exit tweet
        await self.content_manager.schedule_trade_exit(
            entry_decision,
            trade,
            generate_thread=self.config.generate_threads,
        )

        event = TradingEvent.TRADE_CLOSED_WIN if is_winner else TradingEvent.TRADE_CLOSED_LOSS
        await self._notify_handlers(event, trade)

        pnl = trade.get("net_pnl", 0)
        logger.info(f"[TWITTER] Scheduled exit tweet: {'WIN' if is_winner else 'LOSS'} ${pnl:.2f}")

    async def on_daily_summary(self, stats: Dict[str, Any], report_link: str):
        """
        Handle daily summary generation.

        Called at end of day to post performance summary.
        """
        if not self.config.enabled or not self._running:
            return

        if not self.config.post_daily:
            return

        await self.content_manager.schedule_daily_summary(stats, report_link)
        await self._notify_handlers(TradingEvent.DAILY_SUMMARY, stats)

        logger.info("[TWITTER] Scheduled daily summary tweet")

    async def on_weekly_summary(self, stats: Dict[str, Any], report_link: str):
        """
        Handle weekly summary generation.

        Called at end of week to post performance scorecard.
        """
        if not self.config.enabled or not self._running:
            return

        if not self.config.post_weekly:
            return

        await self.content_manager.schedule_weekly_scorecard(stats, report_link)
        await self._notify_handlers(TradingEvent.WEEKLY_SUMMARY, stats)

        logger.info("[TWITTER] Scheduled weekly scorecard tweet")

    async def on_lesson_generated(self, lesson: Dict[str, Any], learning_link: str):
        """
        Handle a new lesson from the learning system.

        Called when Reflexion engine generates insights.
        """
        if not self.config.enabled or not self._running:
            return

        if not self.config.post_lessons:
            return

        await self.content_manager.schedule_lesson(lesson, learning_link)
        await self._notify_handlers(TradingEvent.LESSON_GENERATED, lesson)

        logger.info("[TWITTER] Scheduled lesson learned tweet")

    # ==================== MANUAL TRIGGERS ====================

    async def post_now(self, content: TweetContent) -> Dict[str, Any]:
        """
        Immediately post a tweet (bypass scheduler).

        Use for urgent updates or manual posts.
        """
        return await self.client.post_content(content)

    async def post_custom(
        self,
        text: str,
        generate_organic: bool = True,
    ) -> Dict[str, Any]:
        """
        Post a custom tweet.

        Args:
            text: Base text (will be enhanced if organic=True)
            generate_organic: Use xAI to make it sound natural

        Returns:
            Posting result
        """
        if generate_organic:
            # Use xAI to enhance the text
            enhanced = await self.formatter._call_xai(
                f"""Rewrite this tweet in Argus's voice (nerdy but approachable):

{text}

Keep under 280 characters. Sound natural, not robotic."""
            )
            if enhanced:
                text = self.formatter._validate_tweet(enhanced)

        content = TweetContent(main_tweet=text)
        return await self.client.post_content(content, include_thread=False)

    # ==================== STATUS ====================

    def get_status(self) -> Dict[str, Any]:
        """Get current hook status"""
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "simulation_mode": self.config.simulation_mode,
            "pending_entries": len(self._pending_entries),
            "scheduler": self.scheduler.get_queue_status(),
            "recent_posts": self.scheduler.get_posting_history(10),
        }


# ==================== CONVENIENCE FUNCTIONS ====================

def create_twitter_hook(
    simulation: bool = True,
    decision_base_url: str = "http://localhost:8000/decision",
    xai_api_key: Optional[str] = None,
) -> TwitterHook:
    """
    Create a Twitter hook with sensible defaults.

    Args:
        simulation: Run in simulation mode (no real posts)
        decision_base_url: Base URL for decision links
        xai_api_key: xAI API key for organic content

    Returns:
        Configured TwitterHook instance
    """
    config = TwitterHookConfig(
        enabled=True,
        simulation_mode=simulation,
        decision_base_url=decision_base_url,
        xai_api_key=xai_api_key,
    )
    return TwitterHook(config)


async def demo_twitter_hook():
    """
    Demo the Twitter hook with simulated trades.

    Run this to see how tweets would look.
    """
    hook = create_twitter_hook(simulation=True)
    await hook.start()

    # Simulate a trade entry
    decision = {
        "decision_id": "abc123-test-decision",
        "symbol": "BTC-USD",
        "result": "signal_long",
        "signal_values": {
            "current_price": 88500.00,
            "fast_ema": 88450.00,
            "slow_ema": 88200.00,
            "atr": 1250.00,
            "stop_loss": 86000.00,
            "take_profit": 92000.00,
        },
        "risk_checks": {
            "trading_halted": {"passed": True},
            "frequency_limit": {"passed": True, "trades_today": 2, "limit": 10},
            "daily_loss_limit": {"passed": True},
            "drawdown_limit": {"passed": True},
            "concentration": {"passed": True, "exposure": 15, "limit": 30},
        },
        "result_reason": "Bullish EMA crossover with strong momentum",
    }

    await hook.on_decision(decision)

    # Wait for scheduler to process
    await asyncio.sleep(5)

    # Simulate trade close (win)
    trade = {
        "decision_id": "abc123-test-decision",
        "symbol": "BTC-USD",
        "entry_price": 88500.00,
        "exit_price": 91800.00,
        "net_pnl": 3300.00,
        "pnl_percent": 3.73,
        "duration_hours": 18.5,
        "exit_reason": "Take profit hit",
        "is_winner": True,
        "lesson": "Trend trades in low volatility work well",
    }

    await hook.on_trade_closed(trade)

    # Wait and stop
    await asyncio.sleep(5)
    await hook.stop()

    print("\nDemo complete! Check output above for simulated tweets.")


if __name__ == "__main__":
    asyncio.run(demo_twitter_hook())
