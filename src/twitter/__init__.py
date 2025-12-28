"""
Argus Twitter Bot Module

Automated Twitter posting for transparent trading updates.
Converts Truth Engine decisions into engaging, educational content.

Personality: Nerdy but Approachable
- Data-obsessed but friendly
- Explains concepts simply
- Admits mistakes openly
- Shows the work, always

Components:
- client.py: Twitter API v2 integration
- formatter.py: Decision -> Tweet conversion (with xAI Grok)
- templates.py: Voice/personality templates
- scheduler.py: Optimal posting time scheduler
- hooks.py: Truth Engine integration
"""

from .client import TwitterClient, MockTwitterClient
from .formatter import TweetFormatter, TweetContent
from .templates import TweetTemplates, Voice, TweetType
from .scheduler import TweetScheduler, ContentScheduleManager, Priority
from .hooks import TwitterHook, TwitterHookConfig, TradingEvent, create_twitter_hook

__all__ = [
    # Client
    "TwitterClient",
    "MockTwitterClient",
    # Formatter
    "TweetFormatter",
    "TweetContent",
    # Templates
    "TweetTemplates",
    "Voice",
    "TweetType",
    # Scheduler
    "TweetScheduler",
    "ContentScheduleManager",
    "Priority",
    # Hooks
    "TwitterHook",
    "TwitterHookConfig",
    "TradingEvent",
    "create_twitter_hook",
]
