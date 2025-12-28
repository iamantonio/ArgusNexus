"""
Twitter/X API Client for Argus

Handles posting tweets, threads, and managing engagement.
Uses Twitter API v2 with OAuth 2.0.

Features:
- Single tweet posting
- Thread posting with proper reply chains
- Rate limit handling
- Error recovery
"""

import os
import asyncio
import httpx
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from .formatter import TweetContent


@dataclass
class TweetResult:
    """Result of posting a tweet"""
    success: bool
    tweet_id: Optional[str] = None
    error: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ThreadResult:
    """Result of posting a thread"""
    success: bool
    tweet_ids: List[str] = None
    errors: List[str] = None
    urls: List[str] = None


class TwitterClient:
    """
    Twitter API v2 client for posting Argus updates.

    Uses OAuth 2.0 Bearer Token for app-only auth,
    or OAuth 1.0a for user-context operations (posting).
    """

    BASE_URL = "https://api.twitter.com/2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None,
        bearer_token: Optional[str] = None,
    ):
        """
        Initialize Twitter client.

        For posting, requires OAuth 1.0a credentials.
        For reading, can use Bearer token.
        """
        self.api_key = api_key or os.getenv("TWITTER_API_KEY")
        self.api_secret = api_secret or os.getenv("TWITTER_API_SECRET")
        self.access_token = access_token or os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_token_secret = access_token_secret or os.getenv("TWITTER_ACCESS_TOKEN_SECRET") or os.getenv("TWITTER_ACCESS_SECRET")
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")

        # Rate limiting
        self._last_tweet_time: Optional[datetime] = None
        self._tweets_in_window: int = 0
        self._window_start: Optional[datetime] = None

        # Validate credentials
        self._validate_credentials()

    def _validate_credentials(self):
        """Check if we have the necessary credentials"""
        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            print("Warning: Twitter OAuth 1.0a credentials incomplete. Posting will be simulated.")
            self._simulation_mode = True
        else:
            self._simulation_mode = False

    def _get_oauth1_header(self, method: str, url: str) -> str:
        """
        Generate OAuth 1.0a header for request signing.

        Uses the requests-oauthlib library for proper signing.
        Note: For JSON body requests (API v2), don't include body in signature.
        """
        try:
            from requests_oauthlib import OAuth1
            import requests

            auth = OAuth1(
                self.api_key,
                self.api_secret,
                self.access_token,
                self.access_token_secret,
            )

            # Create a prepared request to get the Authorization header
            # For JSON body requests, don't include body in OAuth signature
            req = requests.Request(method, url)
            prepared = req.prepare()
            auth(prepared)

            return prepared.headers.get("Authorization", "")
        except ImportError:
            print("Warning: requests-oauthlib not installed. Using simulation mode.")
            self._simulation_mode = True
            return ""

    async def _check_rate_limit(self):
        """
        Check and enforce rate limits.

        Twitter limits:
        - 200 tweets per 15-minute window (app)
        - 300 tweets per 3-hour window (user)
        - Minimum 1 second between tweets (safety)
        """
        now = datetime.utcnow()

        # Reset window if needed
        if self._window_start is None or (now - self._window_start) > timedelta(minutes=15):
            self._window_start = now
            self._tweets_in_window = 0

        # Check window limit
        if self._tweets_in_window >= 180:  # Conservative limit
            wait_seconds = (self._window_start + timedelta(minutes=15) - now).total_seconds()
            if wait_seconds > 0:
                print(f"Rate limit approaching. Waiting {wait_seconds:.0f}s...")
                await asyncio.sleep(wait_seconds)
                self._window_start = datetime.utcnow()
                self._tweets_in_window = 0

        # Minimum delay between tweets
        if self._last_tweet_time:
            elapsed = (now - self._last_tweet_time).total_seconds()
            if elapsed < 2:  # 2 second minimum gap
                await asyncio.sleep(2 - elapsed)

    async def post_tweet(
        self,
        content: str,
        reply_to: Optional[str] = None,
    ) -> TweetResult:
        """
        Post a single tweet.

        Args:
            content: Tweet text (max 280 chars)
            reply_to: Tweet ID to reply to (for threads)

        Returns:
            TweetResult with success status and tweet ID
        """
        await self._check_rate_limit()

        if self._simulation_mode:
            print(f"\n[SIMULATED TWEET]\n{content}\n")
            if reply_to:
                print(f"(Reply to: {reply_to})")
            return TweetResult(
                success=True,
                tweet_id=f"sim_{datetime.utcnow().timestamp()}",
                url=None,
            )

        url = f"{self.BASE_URL}/tweets"

        payload = {"text": content}
        if reply_to:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to}

        try:
            async with httpx.AsyncClient() as client:
                # Get OAuth header (don't include JSON body in signature)
                auth_header = self._get_oauth1_header("POST", url)

                response = await client.post(
                    url,
                    headers={
                        "Authorization": auth_header,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 201:
                    data = response.json()
                    tweet_id = data["data"]["id"]
                    self._last_tweet_time = datetime.utcnow()
                    self._tweets_in_window += 1

                    return TweetResult(
                        success=True,
                        tweet_id=tweet_id,
                        url=f"https://twitter.com/ArgusNexus/status/{tweet_id}",
                    )
                elif response.status_code == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("retry-after", 60))
                    return TweetResult(
                        success=False,
                        error=f"Rate limited. Retry after {retry_after}s",
                    )
                else:
                    return TweetResult(
                        success=False,
                        error=f"HTTP {response.status_code}: {response.text}",
                    )

        except Exception as e:
            return TweetResult(
                success=False,
                error=str(e),
            )

    async def post_thread(
        self,
        tweets: List[str],
    ) -> ThreadResult:
        """
        Post a thread of tweets.

        Each tweet is posted as a reply to the previous one.

        Args:
            tweets: List of tweet contents

        Returns:
            ThreadResult with all tweet IDs
        """
        if not tweets:
            return ThreadResult(success=False, errors=["Empty thread"])

        tweet_ids = []
        urls = []
        errors = []

        previous_id = None

        for i, content in enumerate(tweets):
            result = await self.post_tweet(content, reply_to=previous_id)

            if result.success:
                tweet_ids.append(result.tweet_id)
                urls.append(result.url)
                previous_id = result.tweet_id
                # Small delay between thread tweets
                await asyncio.sleep(1)
            else:
                errors.append(f"Tweet {i+1}: {result.error}")
                # Continue with thread even if one fails
                # (though reply chain will be broken)

        return ThreadResult(
            success=len(errors) == 0,
            tweet_ids=tweet_ids,
            urls=urls,
            errors=errors if errors else None,
        )

    async def post_content(
        self,
        content: TweetContent,
        include_thread: bool = True,
    ) -> Dict[str, Any]:
        """
        Post a TweetContent object (main tweet + optional thread).

        Args:
            content: TweetContent from formatter
            include_thread: Whether to post the full thread

        Returns:
            Dict with results for main tweet and thread
        """
        results = {"main_tweet": None, "thread": None}

        # Post main tweet
        main_result = await self.post_tweet(content.main_tweet)
        results["main_tweet"] = {
            "success": main_result.success,
            "tweet_id": main_result.tweet_id,
            "url": main_result.url,
            "error": main_result.error,
        }

        # Post thread if available and requested
        if include_thread and content.thread and main_result.success:
            # First tweet of thread is a reply to main tweet
            thread_result = await self.post_thread(content.thread)
            results["thread"] = {
                "success": thread_result.success,
                "tweet_ids": thread_result.tweet_ids,
                "urls": thread_result.urls,
                "errors": thread_result.errors,
            }

        return results

    async def delete_tweet(self, tweet_id: str) -> bool:
        """Delete a tweet by ID"""
        if self._simulation_mode:
            print(f"[SIMULATED DELETE] Tweet {tweet_id}")
            return True

        url = f"{self.BASE_URL}/tweets/{tweet_id}"

        try:
            async with httpx.AsyncClient() as client:
                auth_header = self._get_oauth1_header("DELETE", url)

                response = await client.delete(
                    url,
                    headers={"Authorization": auth_header},
                    timeout=30.0,
                )

                return response.status_code == 200
        except Exception as e:
            print(f"Delete error: {e}")
            return False

    async def get_mentions(self, since_id: Optional[str] = None) -> List[Dict]:
        """
        Get recent mentions (for engagement).

        Returns list of tweets mentioning @ArgusNexus.
        """
        if not self.bearer_token:
            return []

        # This would require additional API calls
        # Placeholder for future engagement features
        return []


class MockTwitterClient(TwitterClient):
    """
    Mock client for testing without API credentials.

    Logs all actions to console and returns simulated results.
    """

    def __init__(self):
        self._simulation_mode = True
        self._posted_tweets = []

    async def post_tweet(self, content: str, reply_to: Optional[str] = None) -> TweetResult:
        tweet_id = f"mock_{len(self._posted_tweets)}"
        self._posted_tweets.append({
            "id": tweet_id,
            "content": content,
            "reply_to": reply_to,
            "timestamp": datetime.utcnow().isoformat(),
        })

        print(f"\n{'='*50}")
        print(f"[MOCK TWEET] ID: {tweet_id}")
        if reply_to:
            print(f"Reply to: {reply_to}")
        print(f"{'='*50}")
        print(content)
        print(f"{'='*50}\n")

        return TweetResult(
            success=True,
            tweet_id=tweet_id,
            url=f"https://twitter.com/ArgusNexus/status/{tweet_id}",
        )

    def get_posted_tweets(self) -> List[Dict]:
        """Get all tweets posted in this session"""
        return self._posted_tweets
