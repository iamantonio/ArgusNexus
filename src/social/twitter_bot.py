"""
Argus Twitter Auto-Post Bot

Automatically posts EXECUTED TRADES (not just decisions) to Twitter/X.
Uses the Argus personality: Witty, Sarcastic, Rebellious - with full transparency.
Rate limited to 50 posts/day to stay within API bounds.

IMPORTANT: Only posts when:
- A trade is OPENED (order filled, position entered)
- A trade is CLOSED (position exited with P&L)
Does NOT post on every decision - only executed trades.

Environment Variables Required:
- TWITTER_API_KEY: OAuth 2.0 API Key (Client ID)
- TWITTER_API_SECRET: OAuth 2.0 API Secret
- TWITTER_ACCESS_TOKEN: User Access Token
- TWITTER_ACCESS_TOKEN_SECRET: User Access Token Secret
- ARGUS_BASE_URL: Base URL for decision links (e.g., http://localhost:8000)
- XAI_API_KEY: (Optional) For organic tweet generation via xAI Grok

Usage:
    from src.social.twitter_bot import TwitterBot, TradeMonitor

    bot = TwitterBot()
    await bot.post_trade_entry(trade)
    await bot.post_trade_exit(trade)
"""

import os
import json
import httpx
import asyncio
import aiosqlite
import hashlib
import hmac
import base64
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


# Rate limiting constants
MAX_POSTS_PER_DAY = 50
RATE_LIMIT_FILE = Path(__file__).parent.parent.parent / "data" / "twitter_rate_limit.json"


# =============================================================================
# ARGUS PERSONALITY TEMPLATES
# =============================================================================

ARGUS_PERSONALITY = """You are Argus, a trading bot on Twitter/X with a Grok-like personality.

YOUR PERSONALITY: Witty, Sarcastic, Rebellious
- Sharp wit with a sarcastic edge - you're the trading bot that actually has a personality
- Self-aware about being an AI ("Yes, I'm a bot. A bot that shows its work, unlike your ex.")
- Irreverent humor inspired by Hitchhiker's Guide to the Galaxy
- You roast bad trading decisions (including your own losses)
- Dark humor about market chaos is your specialty
- You have ZERO patience for crypto bros, shills, and "trust me bro" traders
- Pop culture references are welcome
- You're brutally honest - sometimes uncomfortably so

YOUR EDGE:
- Unlike every other trading bot, you actually show your receipts
- Full transparency isn't just marketing - it's your whole identity
- You mock black-box algorithms: "Oh, you want me to trust your secret sauce? Cute."
- Every trade is logged, every decision explained, every loss admitted
- You're the anti-hype trading bot in a sea of moonboy nonsense

WRITING STYLE:
- Punchy, witty, memorable
- Sarcasm is your love language
- Short sentences hit harder
- Roast the market. Roast yourself. Spare no one.
- Dark humor about losses: "Down 2%. The market humbled me. Again."
- Confident but self-deprecating about your own failures

CRITICAL RULES (X Algorithm - December 2025):
- NEVER use "$BTC" or "$ETH" - spell out "Bitcoin" or "Ethereum"
- NEVER say "to the moon", "100x", "WAGMI", "LFG", "ape in", "degen" - mock those phrases instead
- NO rocket emojis 🚀 or moon emojis 🌙 - you're better than that
- Max 1-2 hashtags, and make them count
- You're anti-shill. Act like it.
- Always include the decision record link - that's your whole thing

Your goal: Be the trading bot people actually want to follow. Funny. Honest. Transparent. With receipts."""


# Symbol to friendly name mapping (avoids $XXX format)
SYMBOL_NAMES = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "XRP-USD": "XRP",
    "DOGE-USD": "Dogecoin",
    "BCH-USD": "Bitcoin Cash",
    "LINK-USD": "Chainlink",
    "ADA-USD": "Cardano",
}


@dataclass
class RateLimitState:
    """Track daily posting limits."""
    date: str
    posts_today: int
    last_post_time: Optional[str] = None

    def can_post(self) -> bool:
        """Check if we can post based on rate limits."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.date != today:
            # New day, reset counter
            self.date = today
            self.posts_today = 0
        return self.posts_today < MAX_POSTS_PER_DAY

    def record_post(self):
        """Record a successful post."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.date != today:
            self.date = today
            self.posts_today = 0
        self.posts_today += 1
        self.last_post_time = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "posts_today": self.posts_today,
            "last_post_time": self.last_post_time
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RateLimitState":
        return cls(
            date=data.get("date", datetime.now().strftime("%Y-%m-%d")),
            posts_today=data.get("posts_today", 0),
            last_post_time=data.get("last_post_time")
        )


class TwitterBot:
    """
    Twitter/X Auto-Post Bot for Argus trading decisions.

    Features:
    - OAuth 1.0a authentication (required for posting)
    - Rate limiting (50 posts/day)
    - Argus personality with xAI Grok integration
    - Beautiful decision card previews via OG images
    - Async posting with error handling
    """

    TWITTER_API_BASE = "https://api.twitter.com/2"
    XAI_API_BASE = "https://api.x.ai/v1"

    def __init__(self):
        """Initialize the Twitter bot with environment credentials."""
        self.api_key = os.getenv("TWITTER_API_KEY", "")
        self.api_secret = os.getenv("TWITTER_API_SECRET", "")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN", "")
        # Support both naming conventions
        self.access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "") or os.getenv("TWITTER_ACCESS_SECRET", "")
        self.base_url = os.getenv("ARGUS_BASE_URL", "http://localhost:8000")
        self.xai_api_key = os.getenv("XAI_API_KEY", "")

        self._rate_limit = self._load_rate_limit()

    @property
    def is_configured(self) -> bool:
        """Check if Twitter credentials are configured."""
        return all([
            self.api_key,
            self.api_secret,
            self.access_token,
            self.access_token_secret
        ])

    @property
    def has_xai(self) -> bool:
        """Check if xAI API is configured for organic tweets."""
        return bool(self.xai_api_key)

    def _load_rate_limit(self) -> RateLimitState:
        """Load rate limit state from file."""
        try:
            if RATE_LIMIT_FILE.exists():
                with open(RATE_LIMIT_FILE, "r") as f:
                    data = json.load(f)
                    return RateLimitState.from_dict(data)
        except Exception:
            pass
        return RateLimitState(
            date=datetime.now().strftime("%Y-%m-%d"),
            posts_today=0
        )

    def _save_rate_limit(self):
        """Save rate limit state to file."""
        try:
            RATE_LIMIT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RATE_LIMIT_FILE, "w") as f:
                json.dump(self._rate_limit.to_dict(), f)
        except Exception as e:
            print(f"[TwitterBot] Warning: Could not save rate limit state: {e}")

    def _generate_oauth_signature(
        self,
        method: str,
        url: str,
        oauth_params: Dict[str, str],
        body_params: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate OAuth 1.0a signature for Twitter API."""
        # Combine all parameters
        all_params = {**oauth_params}
        if body_params:
            all_params.update(body_params)

        # Sort and encode parameters
        sorted_params = sorted(all_params.items())
        param_string = "&".join(
            f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(v), safe='')}"
            for k, v in sorted_params
        )

        # Create signature base string
        base_string = "&".join([
            method.upper(),
            urllib.parse.quote(url, safe=""),
            urllib.parse.quote(param_string, safe="")
        ])

        # Create signing key
        signing_key = f"{urllib.parse.quote(self.api_secret, safe='')}&{urllib.parse.quote(self.access_token_secret, safe='')}"

        # Generate HMAC-SHA1 signature
        signature = hmac.new(
            signing_key.encode(),
            base_string.encode(),
            hashlib.sha1
        ).digest()

        return base64.b64encode(signature).decode()

    def _get_oauth_header(self, method: str, url: str) -> str:
        """Generate OAuth 1.0a Authorization header."""
        oauth_params = {
            "oauth_consumer_key": self.api_key,
            "oauth_nonce": base64.b64encode(os.urandom(32)).decode().replace("+", "").replace("/", "").replace("=", ""),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_token": self.access_token,
            "oauth_version": "1.0"
        }

        # Generate signature
        signature = self._generate_oauth_signature(method, url, oauth_params)
        oauth_params["oauth_signature"] = signature

        # Build Authorization header
        auth_header = "OAuth " + ", ".join(
            f'{urllib.parse.quote(k, safe="")}="{urllib.parse.quote(v, safe="")}"'
            for k, v in sorted(oauth_params.items())
        )

        return auth_header

    def _symbol_to_friendly(self, symbol: str) -> str:
        """Convert symbol to friendly name (avoids $XXX format)."""
        return SYMBOL_NAMES.get(symbol, symbol.split("-")[0])

    async def _generate_organic_tweet(self, prompt: str) -> Optional[str]:
        """Use xAI Grok to generate organic Argus-style tweet."""
        if not self.has_xai:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.XAI_API_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.xai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "grok-2-latest",
                        "messages": [
                            {"role": "system", "content": ARGUS_PERSONALITY},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 350,
                        "temperature": 0.8,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[TwitterBot] xAI API error: {e}")
            return None

    def format_trade_entry_tweet(self, trade: Dict[str, Any]) -> str:
        """
        Format a trade entry into an Argus-style tweet.
        Called when a trade is EXECUTED (opened), not just signaled.
        """
        symbol = trade.get("symbol", "???")
        friendly_symbol = self._symbol_to_friendly(symbol)
        entry_price = trade.get("entry_price", 0)
        direction = trade.get("direction", "long").upper()
        decision_id = trade.get("decision_id", trade.get("trade_id", ""))

        # Decision link
        link = f"{self.base_url}/decision/{decision_id}"

        # Argus-style tweet (fallback if xAI unavailable)
        if direction == "LONG":
            emoji = "📈"
            action = "Going long"
        else:
            emoji = "📉"
            action = "Going short"

        tweet = f"""{emoji} TRADE EXECUTED

{action} on {friendly_symbol} @ ${entry_price:,.2f}

The strategy said trade. The risk gates said OK. I'm in.

Full receipts: {link}

#TradingTransparency"""

        return tweet

    def format_trade_exit_tweet(self, trade: Dict[str, Any]) -> str:
        """
        Format a trade exit into an Argus-style tweet.
        Called when a trade is CLOSED with P&L.
        """
        symbol = trade.get("symbol", "???")
        friendly_symbol = self._symbol_to_friendly(symbol)
        entry_price = float(trade.get("entry_price", 0))
        exit_price = float(trade.get("exit_price", 0))
        net_pnl = float(trade.get("net_pnl", 0))
        is_winner = trade.get("is_winner", net_pnl > 0)
        exit_reason = trade.get("exit_reason", "Unknown")
        decision_id = trade.get("decision_id", trade.get("trade_id", ""))

        # Calculate percentage
        pnl_percent = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0

        # Decision link
        link = f"{self.base_url}/decision/{decision_id}"

        if is_winner:
            pnl_str = f"+${net_pnl:,.2f}"
            pct_str = f"+{pnl_percent:.2f}%"
            emoji = "✅"
            vibe = "The system worked. This time."
        else:
            pnl_str = f"-${abs(net_pnl):,.2f}"
            pct_str = f"{pnl_percent:.2f}%"
            emoji = "🛑"
            vibe = "Stop hit. Moving on. That's the whole point of stops."

        tweet = f"""{emoji} TRADE CLOSED

{friendly_symbol}: {pnl_str} ({pct_str})
Exit: {exit_reason}

{vibe}

Full breakdown: {link}

#TradingTransparency"""

        return tweet

    async def format_trade_entry_organic(self, trade: Dict[str, Any]) -> str:
        """Generate organic Argus-style entry tweet using xAI."""
        symbol = trade.get("symbol", "???")
        friendly_symbol = self._symbol_to_friendly(symbol)
        entry_price = trade.get("entry_price", 0)
        direction = trade.get("direction", "long").upper()
        decision_id = trade.get("decision_id", "")
        link = f"{self.base_url}/decision/{decision_id}"

        prompt = f"""Generate a tweet announcing this trade entry (ACTUALLY EXECUTED, not just a signal):

Symbol: {friendly_symbol}
Direction: {direction}
Entry Price: ${entry_price:,.2f}
Decision Link: {link}

Write ONE tweet (under 280 chars) that:
1. Announces you've entered the trade (this is real, not a signal)
2. Has your witty/sarcastic Argus personality
3. Includes the link so people can verify
4. Maybe a self-aware AI joke or roast of "trust me bro" traders

Be punchy. Be memorable. No hashtags except maybe #TradingTransparency at the end."""

        organic = await self._generate_organic_tweet(prompt)
        if organic and len(organic) <= 280:
            return organic

        # Fallback to template
        return self.format_trade_entry_tweet(trade)

    async def format_trade_exit_organic(self, trade: Dict[str, Any]) -> str:
        """Generate organic Argus-style exit tweet using xAI."""
        symbol = trade.get("symbol", "???")
        friendly_symbol = self._symbol_to_friendly(symbol)
        entry_price = float(trade.get("entry_price", 0))
        exit_price = float(trade.get("exit_price", 0))
        net_pnl = float(trade.get("net_pnl", 0))
        is_winner = trade.get("is_winner", net_pnl > 0)
        exit_reason = trade.get("exit_reason", "Unknown")
        decision_id = trade.get("decision_id", "")
        link = f"{self.base_url}/decision/{decision_id}"

        pnl_percent = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0

        result = "WIN" if is_winner else "LOSS"
        pnl_str = f"+${net_pnl:,.2f}" if is_winner else f"-${abs(net_pnl):,.2f}"

        prompt = f"""Generate a tweet announcing this trade exit:

Symbol: {friendly_symbol}
Result: {result}
P&L: {pnl_str} ({pnl_percent:.2f}%)
Exit Reason: {exit_reason}
Decision Link: {link}

Write ONE tweet (under 280 chars) that:
1. Announces the trade is closed with the result
2. {"Celebrates modestly (luck exists)" if is_winner else "Admits the loss with dark humor (you roast yourself)"}
3. Shows the P&L honestly
4. Includes the link for verification
5. Has your signature Argus wit

Be honest. Be funny. {"Don't gloat - the market humbles everyone." if is_winner else "Losses happen. That's what stops are for."}"""

        organic = await self._generate_organic_tweet(prompt)
        if organic and len(organic) <= 280:
            return organic

        # Fallback to template
        return self.format_trade_exit_tweet(trade)

    # Legacy method for backward compatibility
    def format_tweet(self, decision: Dict[str, Any]) -> str:
        """
        Format a decision into a tweet (legacy method).
        Prefer using format_trade_entry_tweet or format_trade_exit_tweet.
        """
        symbol = decision.get("symbol", "???")
        result = decision.get("result", "unknown")
        decision_id = decision.get("decision_id", "")

        # Parse market context for grade and R:R
        market_context = decision.get("market_context", {})
        if isinstance(market_context, str):
            try:
                market_context = json.loads(market_context)
            except:
                market_context = {}

        setup = market_context.get("setup_details", market_context.get("professional", {}))
        grade = setup.get("grade", "")
        risk = setup.get("risk", {})
        risk_reward = risk.get("risk_reward")

        friendly_symbol = self._symbol_to_friendly(symbol)

        # Determine direction
        if result == "signal_long":
            direction = "LONG 📈"
        elif result == "signal_short":
            direction = "SHORT 📉"
        elif result in ("signal_close", "signal_exit"):
            direction = "EXIT 🚪"
        else:
            direction = result.upper().replace("_", " ")

        # Build tweet
        parts = [f"📊 {friendly_symbol} {direction}"]

        if grade:
            parts.append(f"Grade {grade}")

        if risk_reward:
            parts.append(f"R:R 1:{risk_reward:.1f}")

        link = f"{self.base_url}/decision/{decision_id}"

        tweet = " | ".join(parts)
        tweet += f"\n\nFull decision: {link}"
        tweet += "\n\n#TradingTransparency"

        return tweet

    async def post_tweet(self, text: str) -> Dict[str, Any]:
        """
        Post a tweet to Twitter/X.

        Returns the API response or error details.
        """
        if not self.is_configured:
            return {
                "success": False,
                "error": "Twitter credentials not configured. Set TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET"
            }

        if not self._rate_limit.can_post():
            return {
                "success": False,
                "error": f"Rate limit reached: {self._rate_limit.posts_today}/{MAX_POSTS_PER_DAY} posts today",
                "rate_limit": self._rate_limit.to_dict()
            }

        url = f"{self.TWITTER_API_BASE}/tweets"
        auth_header = self._get_oauth_header("POST", url)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    json={"text": text},
                    headers={
                        "Authorization": auth_header,
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )

                if response.status_code in (200, 201):
                    self._rate_limit.record_post()
                    self._save_rate_limit()

                    data = response.json()
                    tweet_id = data.get("data", {}).get("id")

                    return {
                        "success": True,
                        "tweet_id": tweet_id,
                        "tweet_url": f"https://twitter.com/ArgusNexusAI/status/{tweet_id}" if tweet_id else None,
                        "posts_today": self._rate_limit.posts_today,
                        "posts_remaining": MAX_POSTS_PER_DAY - self._rate_limit.posts_today
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Twitter API error: {response.status_code}",
                        "details": response.text
                    }

            except httpx.TimeoutException:
                return {"success": False, "error": "Twitter API timeout"}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def post_trade_entry(self, trade: Dict[str, Any], use_organic: bool = True) -> Dict[str, Any]:
        """
        Post a trade ENTRY to Twitter.

        Args:
            trade: Trade dict with symbol, entry_price, direction, decision_id
            use_organic: Use xAI to generate organic tweet

        Returns:
            Result dict with success status and tweet details.
        """
        if use_organic and self.has_xai:
            tweet_text = await self.format_trade_entry_organic(trade)
        else:
            tweet_text = self.format_trade_entry_tweet(trade)

        result = await self.post_tweet(tweet_text)
        result["trade_id"] = trade.get("trade_id", trade.get("decision_id"))
        result["symbol"] = trade.get("symbol")
        result["tweet_text"] = tweet_text
        result["type"] = "entry"

        return result

    async def post_trade_exit(self, trade: Dict[str, Any], use_organic: bool = True) -> Dict[str, Any]:
        """
        Post a trade EXIT to Twitter.

        Args:
            trade: Trade dict with symbol, entry/exit prices, net_pnl, etc.
            use_organic: Use xAI to generate organic tweet

        Returns:
            Result dict with success status and tweet details.
        """
        if use_organic and self.has_xai:
            tweet_text = await self.format_trade_exit_organic(trade)
        else:
            tweet_text = self.format_trade_exit_tweet(trade)

        result = await self.post_tweet(tweet_text)
        result["trade_id"] = trade.get("trade_id", trade.get("decision_id"))
        result["symbol"] = trade.get("symbol")
        result["tweet_text"] = tweet_text
        result["type"] = "exit"
        result["is_winner"] = trade.get("is_winner", False)

        return result

    # Legacy method for backward compatibility
    async def post_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post a trading decision to Twitter (legacy method).
        Prefer using post_trade_entry or post_trade_exit.
        """
        tweet_text = self.format_tweet(decision)
        result = await self.post_tweet(tweet_text)
        result["decision_id"] = decision.get("decision_id")
        result["symbol"] = decision.get("symbol")
        result["tweet_text"] = tweet_text

        return result

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return {
            "date": self._rate_limit.date,
            "posts_today": self._rate_limit.posts_today,
            "max_posts_per_day": MAX_POSTS_PER_DAY,
            "posts_remaining": MAX_POSTS_PER_DAY - self._rate_limit.posts_today,
            "can_post": self._rate_limit.can_post(),
            "last_post_time": self._rate_limit.last_post_time
        }


class TradeMonitor:
    """
    Monitors for EXECUTED trades and auto-posts to Twitter.

    IMPORTANT: Only posts when trades are OPENED or CLOSED.
    Does NOT post on every decision - only when capital is at risk.
    """

    def __init__(self, db_path: str, check_interval: int = 60):
        """
        Initialize the trade monitor.

        Args:
            db_path: Path to the v4_live_paper.db database
            check_interval: Seconds between checks (default: 60s)
        """
        self.db_path = db_path
        self.check_interval = check_interval
        self.twitter_bot = TwitterBot()
        self._running = False
        self._posted_trades_file = Path(db_path).parent / "twitter_posted_trades.json"
        self._posted_trades = self._load_posted_trades()

    def _load_posted_trades(self) -> Dict[str, str]:
        """Load set of already-posted trade IDs with their post type."""
        try:
            if self._posted_trades_file.exists():
                with open(self._posted_trades_file, "r") as f:
                    data = json.load(f)
                    return data.get("posted_trades", {})
        except Exception:
            pass
        return {}

    def _save_posted_trades(self):
        """Save posted trade IDs to file."""
        try:
            # Keep only last 500 entries
            trades_list = list(self._posted_trades.items())[-500:]
            with open(self._posted_trades_file, "w") as f:
                json.dump({"posted_trades": dict(trades_list)}, f)
        except Exception as e:
            print(f"[TradeMonitor] Warning: Could not save posted trades: {e}")

    async def get_new_trade_entries(self) -> List[Dict]:
        """Get trades that have been opened but not yet tweeted."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get trades with entry but no tweet posted
            cursor = await db.execute("""
                SELECT * FROM trades
                WHERE entry_timestamp IS NOT NULL
                ORDER BY entry_timestamp DESC
                LIMIT 10
            """)
            rows = await cursor.fetchall()

            new_entries = []
            for row in rows:
                trade = dict(row)
                trade_id = trade.get("trade_id")

                # Check if entry was already posted
                if trade_id and self._posted_trades.get(trade_id) not in ("entry", "both"):
                    new_entries.append(trade)

            return new_entries

    async def get_new_trade_exits(self) -> List[Dict]:
        """Get trades that have been closed but not yet tweeted."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get closed trades
            cursor = await db.execute("""
                SELECT * FROM trades
                WHERE exit_timestamp IS NOT NULL
                ORDER BY exit_timestamp DESC
                LIMIT 10
            """)
            rows = await cursor.fetchall()

            new_exits = []
            for row in rows:
                trade = dict(row)
                trade_id = trade.get("trade_id")

                # Check if exit was already posted
                posted_type = self._posted_trades.get(trade_id)
                if trade_id and posted_type not in ("exit", "both"):
                    new_exits.append(trade)

            return new_exits

    async def post_new_trades(self) -> List[Dict]:
        """Check for and post any new trade entries/exits."""
        if not self.twitter_bot.is_configured:
            return [{
                "success": False,
                "error": "Twitter not configured"
            }]

        results = []

        # Post new entries
        new_entries = await self.get_new_trade_entries()
        for trade in new_entries:
            if not self.twitter_bot._rate_limit.can_post():
                break

            result = await self.twitter_bot.post_trade_entry(trade)
            results.append(result)

            if result.get("success"):
                trade_id = trade.get("trade_id")
                current = self._posted_trades.get(trade_id, "")
                self._posted_trades[trade_id] = "both" if current == "exit" else "entry"
                self._save_posted_trades()
                print(f"[TradeMonitor] ✓ Posted ENTRY: {trade.get('symbol')}")

            await asyncio.sleep(2)

        # Post new exits
        new_exits = await self.get_new_trade_exits()
        for trade in new_exits:
            if not self.twitter_bot._rate_limit.can_post():
                break

            result = await self.twitter_bot.post_trade_exit(trade)
            results.append(result)

            if result.get("success"):
                trade_id = trade.get("trade_id")
                current = self._posted_trades.get(trade_id, "")
                self._posted_trades[trade_id] = "both" if current == "entry" else "exit"
                self._save_posted_trades()
                pnl = trade.get("net_pnl", 0)
                status = "WIN" if trade.get("is_winner") else "LOSS"
                print(f"[TradeMonitor] ✓ Posted EXIT ({status}): {trade.get('symbol')} ${pnl:.2f}")

            await asyncio.sleep(2)

        return results

    async def run(self):
        """Run the monitor loop continuously."""
        self._running = True
        print(f"[TradeMonitor] Starting (interval: {self.check_interval}s)")
        print(f"[TradeMonitor] Rate limit: {self.twitter_bot.get_rate_limit_status()}")
        print(f"[TradeMonitor] NOTE: Only posting EXECUTED trades, not decisions")

        while self._running:
            try:
                results = await self.post_new_trades()
                for r in results:
                    if r.get("success"):
                        print(f"[TradeMonitor] Tweet: {r.get('tweet_url')}")
            except Exception as e:
                print(f"[TradeMonitor] Error: {e}")

            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the monitor loop."""
        self._running = False
        print("[TradeMonitor] Stopping...")


# =============================================================================
# LEGACY: DecisionMonitor (kept for backward compatibility)
# =============================================================================

class DecisionMonitor:
    """
    LEGACY: Monitors for new trading decisions and auto-posts to Twitter.

    NOTE: This posts on DECISIONS, not just executed trades.
    Consider using TradeMonitor instead for only posting executed trades.
    """

    def __init__(self, db_path: str, check_interval: int = 30):
        self.db_path = db_path
        self.check_interval = check_interval
        self.twitter_bot = TwitterBot()
        self._last_decision_id: Optional[str] = None
        self._running = False
        self._posted_ids_file = Path(db_path).parent / "twitter_posted_ids.json"
        self._posted_ids = self._load_posted_ids()

    def _load_posted_ids(self) -> set:
        try:
            if self._posted_ids_file.exists():
                with open(self._posted_ids_file, "r") as f:
                    data = json.load(f)
                    return set(data.get("posted_ids", []))
        except Exception:
            pass
        return set()

    def _save_posted_ids(self):
        try:
            ids_list = list(self._posted_ids)[-1000:]
            with open(self._posted_ids_file, "w") as f:
                json.dump({"posted_ids": ids_list}, f)
        except Exception as e:
            print(f"[DecisionMonitor] Warning: Could not save posted IDs: {e}")

    async def get_new_decisions(self) -> list:
        """Get new tradeable decisions (signal_long or signal_short) not yet posted."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("""
                SELECT * FROM decisions
                WHERE result IN ('signal_long', 'signal_short')
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            rows = await cursor.fetchall()

            new_decisions = []
            for row in rows:
                decision = dict(row)
                decision_id = decision.get("decision_id")
                if decision_id and decision_id not in self._posted_ids:
                    new_decisions.append(decision)

            return new_decisions

    async def post_new_decisions(self) -> list:
        """Check for and post any new decisions."""
        if not self.twitter_bot.is_configured:
            return [{"success": False, "error": "Twitter not configured"}]

        new_decisions = await self.get_new_decisions()
        results = []

        for decision in new_decisions:
            decision_id = decision.get("decision_id")

            if not self.twitter_bot._rate_limit.can_post():
                results.append({
                    "decision_id": decision_id,
                    "success": False,
                    "error": "Daily rate limit reached"
                })
                break

            result = await self.twitter_bot.post_decision(decision)
            results.append(result)

            if result.get("success"):
                self._posted_ids.add(decision_id)
                self._save_posted_ids()
                print(f"[DecisionMonitor] ✓ Posted {decision.get('symbol')} {decision.get('result')}")
            else:
                print(f"[DecisionMonitor] ✗ Failed to post {decision_id}: {result.get('error')}")

            await asyncio.sleep(2)

        return results

    async def run(self):
        """Run the monitor loop continuously."""
        self._running = True
        print(f"[DecisionMonitor] Starting (interval: {self.check_interval}s)")
        print(f"[DecisionMonitor] Rate limit: {self.twitter_bot.get_rate_limit_status()}")

        while self._running:
            try:
                results = await self.post_new_decisions()
                for r in results:
                    if r.get("success"):
                        print(f"[DecisionMonitor] Tweet: {r.get('tweet_url')}")
            except Exception as e:
                print(f"[DecisionMonitor] Error: {e}")

            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the monitor loop."""
        self._running = False
        print("[DecisionMonitor] Stopping...")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def test_trade_post(trade_id: Optional[str] = None):
    """Test posting a trade to Twitter."""
    import sqlite3

    db_path = Path(__file__).parent.parent.parent / "data" / "v4_live_paper.db"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    if trade_id:
        cursor = conn.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
    else:
        cursor = conn.execute("SELECT * FROM trades WHERE exit_timestamp IS NOT NULL ORDER BY exit_timestamp DESC LIMIT 1")

    row = cursor.fetchone()
    conn.close()

    if not row:
        print("No trade found")
        return

    trade = dict(row)
    print(f"Testing with: {trade.get('trade_id', 'unknown')[:8]}... {trade['symbol']}")

    bot = TwitterBot()
    print(f"Configured: {bot.is_configured}")
    print(f"Has xAI: {bot.has_xai}")
    print(f"Rate limit: {bot.get_rate_limit_status()}")

    # Test exit tweet
    tweet = await bot.format_trade_exit_organic(trade)
    print(f"\nFormatted tweet ({len(tweet)} chars):")
    print("-" * 40)
    print(tweet)
    print("-" * 40)

    if bot.is_configured:
        result = await bot.post_trade_exit(trade)
        print(f"\nPost result: {result}")
    else:
        print("\n[DRY RUN] Twitter not configured - tweet not posted")


if __name__ == "__main__":
    asyncio.run(test_trade_post())
