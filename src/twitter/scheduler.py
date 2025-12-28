"""
Tweet Scheduler for Optimal Posting Times

Schedules tweets for maximum engagement based on:
- Crypto Twitter peak hours
- User timezone analysis
- Content type priorities
- Rate limit management

Research-backed optimal times for crypto content (Dec 2025):
- Weekdays: 8-10 AM EST, 12-2 PM EST, 6-8 PM EST
- Weekends: 10 AM - 2 PM EST
- Avoid: Late night EST (2-6 AM)
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
from zoneinfo import ZoneInfo

from .formatter import TweetContent, TweetType
from .client import TwitterClient


class Priority(Enum):
    """Tweet priority levels"""
    CRITICAL = 1      # Trade entries/exits - post immediately
    HIGH = 2          # Rejections, lessons - post soon
    NORMAL = 3        # Educational content - schedule optimally
    LOW = 4           # Reposts, engagement - fill gaps


@dataclass(order=True)
class ScheduledTweet:
    """A tweet scheduled for posting"""
    scheduled_time: datetime
    priority: Priority = field(compare=False)
    content: TweetContent = field(compare=False)
    include_thread: bool = field(compare=False, default=True)
    callback: Optional[Callable] = field(compare=False, default=None)


class TweetScheduler:
    """
    Manages tweet scheduling and posting.

    Features:
    - Priority-based queue
    - Optimal time scheduling
    - Rate limit awareness
    - Async execution
    """

    # Optimal posting hours (in EST/New York)
    PEAK_HOURS_WEEKDAY = [8, 9, 10, 12, 13, 14, 18, 19, 20]
    PEAK_HOURS_WEEKEND = [10, 11, 12, 13, 14]

    # Avoid these hours (EST)
    DEAD_HOURS = [2, 3, 4, 5]

    def __init__(
        self,
        client: TwitterClient,
        timezone: str = "America/New_York",
    ):
        """
        Initialize scheduler.

        Args:
            client: TwitterClient instance for posting
            timezone: Timezone for scheduling (default: EST)
        """
        self.client = client
        self.timezone = ZoneInfo(timezone)
        self._queue: List[ScheduledTweet] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._posted_history: List[Dict] = []

    def schedule(
        self,
        content: TweetContent,
        priority: Priority = Priority.NORMAL,
        scheduled_time: Optional[datetime] = None,
        include_thread: bool = True,
        callback: Optional[Callable] = None,
    ) -> ScheduledTweet:
        """
        Schedule a tweet for posting.

        Args:
            content: TweetContent to post
            priority: Priority level
            scheduled_time: When to post (None = optimal time)
            include_thread: Whether to include thread
            callback: Function to call after posting

        Returns:
            ScheduledTweet object
        """
        if scheduled_time is None:
            scheduled_time = self._find_optimal_time(priority)

        tweet = ScheduledTweet(
            scheduled_time=scheduled_time,
            priority=priority,
            content=content,
            include_thread=include_thread,
            callback=callback,
        )

        heapq.heappush(self._queue, tweet)
        return tweet

    def schedule_immediate(
        self,
        content: TweetContent,
        include_thread: bool = True,
        callback: Optional[Callable] = None,
    ) -> ScheduledTweet:
        """Schedule a tweet for immediate posting (trade alerts)"""
        return self.schedule(
            content=content,
            priority=Priority.CRITICAL,
            scheduled_time=datetime.now(self.timezone),
            include_thread=include_thread,
            callback=callback,
        )

    def _find_optimal_time(self, priority: Priority) -> datetime:
        """
        Find the next optimal posting time.

        Higher priority = sooner posting.
        Lower priority = wait for peak hours.
        """
        now = datetime.now(self.timezone)

        if priority == Priority.CRITICAL:
            # Post immediately
            return now

        if priority == Priority.HIGH:
            # Post within the hour
            return now + timedelta(minutes=15)

        # For normal/low priority, find next peak hour
        current_hour = now.hour
        is_weekend = now.weekday() >= 5

        peak_hours = self.PEAK_HOURS_WEEKEND if is_weekend else self.PEAK_HOURS_WEEKDAY

        # Find next peak hour
        for hour in peak_hours:
            if hour > current_hour:
                target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                # Add small random offset (0-15 min) to avoid exact hour posting
                import random
                target += timedelta(minutes=random.randint(0, 15))
                return target

        # No peak hours left today, schedule for tomorrow
        tomorrow = now + timedelta(days=1)
        next_peak = peak_hours[0]
        target = tomorrow.replace(hour=next_peak, minute=0, second=0, microsecond=0)
        import random
        target += timedelta(minutes=random.randint(0, 15))
        return target

    def _is_dead_zone(self, time: datetime) -> bool:
        """Check if time is in a dead zone (low engagement)"""
        return time.hour in self.DEAD_HOURS

    async def start(self):
        """Start the scheduler background task"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        print("Tweet scheduler started")

    async def stop(self):
        """Stop the scheduler"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("Tweet scheduler stopped")

    async def _run_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                await self._process_queue()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(60)  # Wait on error

    async def _process_queue(self):
        """Process due tweets in the queue"""
        now = datetime.now(self.timezone)

        while self._queue and self._queue[0].scheduled_time <= now:
            tweet = heapq.heappop(self._queue)

            # Skip if in dead zone (reschedule normal/low priority)
            if self._is_dead_zone(now) and tweet.priority.value > Priority.HIGH.value:
                # Reschedule for next peak hour
                new_time = self._find_optimal_time(tweet.priority)
                tweet.scheduled_time = new_time
                heapq.heappush(self._queue, tweet)
                continue

            # Post the tweet
            try:
                result = await self.client.post_content(
                    tweet.content,
                    include_thread=tweet.include_thread,
                )

                # Log result
                self._posted_history.append({
                    "timestamp": now.isoformat(),
                    "tweet_type": tweet.content.tweet_type.value,
                    "decision_id": tweet.content.decision_id,
                    "result": result,
                })

                # Execute callback if provided
                if tweet.callback:
                    try:
                        if asyncio.iscoroutinefunction(tweet.callback):
                            await tweet.callback(result)
                        else:
                            tweet.callback(result)
                    except Exception as e:
                        print(f"Callback error: {e}")

                # Small delay between posts
                await asyncio.sleep(2)

            except Exception as e:
                print(f"Failed to post tweet: {e}")
                # Retry in 5 minutes
                tweet.scheduled_time = now + timedelta(minutes=5)
                heapq.heappush(self._queue, tweet)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queue_length": len(self._queue),
            "running": self._running,
            "next_tweet": (
                self._queue[0].scheduled_time.isoformat()
                if self._queue
                else None
            ),
            "posted_today": len([
                h for h in self._posted_history
                if datetime.fromisoformat(h["timestamp"]).date() == datetime.now(self.timezone).date()
            ]),
        }

    def get_posting_history(self, limit: int = 50) -> List[Dict]:
        """Get recent posting history"""
        return self._posted_history[-limit:]

    def clear_queue(self):
        """Clear all scheduled tweets"""
        self._queue = []


class ContentScheduleManager:
    """
    High-level manager for scheduling different content types.

    Handles the logic of what to post and when based on
    trading activity and engagement patterns.
    """

    def __init__(
        self,
        scheduler: TweetScheduler,
        formatter,  # TweetFormatter
    ):
        self.scheduler = scheduler
        self.formatter = formatter

        # Content quotas (per day)
        self.daily_limits = {
            TweetType.TRADE_ENTRY: 10,
            TweetType.TRADE_EXIT_WIN: 10,
            TweetType.TRADE_EXIT_LOSS: 10,
            TweetType.TRADE_REJECTION: 5,
            TweetType.DAILY_SUMMARY: 1,
            TweetType.WEEKLY_SCORECARD: 1,
            TweetType.LESSON_LEARNED: 2,
            TweetType.STRATEGY_EXPLAINER: 1,
        }

        self._daily_counts: Dict[TweetType, int] = {}
        self._last_reset: Optional[datetime] = None

    def _check_daily_reset(self):
        """Reset daily counts at midnight"""
        now = datetime.now()
        if self._last_reset is None or self._last_reset.date() != now.date():
            self._daily_counts = {t: 0 for t in TweetType}
            self._last_reset = now

    def can_post(self, tweet_type: TweetType) -> bool:
        """Check if we can post this type of content today"""
        self._check_daily_reset()
        limit = self.daily_limits.get(tweet_type, 10)
        current = self._daily_counts.get(tweet_type, 0)
        return current < limit

    async def schedule_trade_entry(
        self,
        decision: Dict[str, Any],
        generate_thread: bool = True,
    ) -> Optional[ScheduledTweet]:
        """Schedule a trade entry tweet"""
        if not self.can_post(TweetType.TRADE_ENTRY):
            return None

        content = await self.formatter.format_trade_entry(decision, generate_thread)
        scheduled = self.scheduler.schedule_immediate(content)

        self._daily_counts[TweetType.TRADE_ENTRY] = (
            self._daily_counts.get(TweetType.TRADE_ENTRY, 0) + 1
        )

        return scheduled

    async def schedule_trade_exit(
        self,
        entry_decision: Dict[str, Any],
        trade_record: Dict[str, Any],
        generate_thread: bool = True,
    ) -> Optional[ScheduledTweet]:
        """Schedule a trade exit tweet"""
        is_win = trade_record.get("is_winner", False)
        tweet_type = TweetType.TRADE_EXIT_WIN if is_win else TweetType.TRADE_EXIT_LOSS

        if not self.can_post(tweet_type):
            return None

        content = await self.formatter.format_trade_exit(
            entry_decision, trade_record, generate_thread
        )
        scheduled = self.scheduler.schedule_immediate(content)

        self._daily_counts[tweet_type] = (
            self._daily_counts.get(tweet_type, 0) + 1
        )

        return scheduled

    async def schedule_rejection(
        self,
        decision: Dict[str, Any],
    ) -> Optional[ScheduledTweet]:
        """Schedule a trade rejection tweet"""
        if not self.can_post(TweetType.TRADE_REJECTION):
            return None

        content = await self.formatter.format_rejection(decision)

        # Rejections are high priority but not immediate
        scheduled = self.scheduler.schedule(
            content,
            priority=Priority.HIGH,
        )

        self._daily_counts[TweetType.TRADE_REJECTION] = (
            self._daily_counts.get(TweetType.TRADE_REJECTION, 0) + 1
        )

        return scheduled

    async def schedule_daily_summary(
        self,
        stats: Dict[str, Any],
        report_link: str,
    ) -> Optional[ScheduledTweet]:
        """Schedule daily summary (end of day)"""
        if not self.can_post(TweetType.DAILY_SUMMARY):
            return None

        content = await self.formatter.format_daily_summary(stats, report_link)

        scheduled = self.scheduler.schedule(
            content,
            priority=Priority.NORMAL,
        )

        self._daily_counts[TweetType.DAILY_SUMMARY] = 1

        return scheduled

    async def schedule_weekly_scorecard(
        self,
        stats: Dict[str, Any],
        report_link: str,
    ) -> Optional[ScheduledTweet]:
        """Schedule weekly scorecard (typically Sunday evening)"""
        if not self.can_post(TweetType.WEEKLY_SCORECARD):
            return None

        content = await self.formatter.format_weekly_scorecard(stats, report_link)

        scheduled = self.scheduler.schedule(
            content,
            priority=Priority.NORMAL,
        )

        self._daily_counts[TweetType.WEEKLY_SCORECARD] = 1

        return scheduled

    async def schedule_lesson(
        self,
        lesson: Dict[str, Any],
        learning_link: str,
    ) -> Optional[ScheduledTweet]:
        """Schedule a lesson learned tweet"""
        if not self.can_post(TweetType.LESSON_LEARNED):
            return None

        content = await self.formatter.format_lesson_learned(lesson, learning_link)

        scheduled = self.scheduler.schedule(
            content,
            priority=Priority.NORMAL,
        )

        self._daily_counts[TweetType.LESSON_LEARNED] = (
            self._daily_counts.get(TweetType.LESSON_LEARNED, 0) + 1
        )

        return scheduled
