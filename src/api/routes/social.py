"""
Social Media API Routes

Endpoints for Twitter/Discord auto-posting and management.

IMPORTANT: Twitter bot now only posts EXECUTED TRADES, not every decision.
This means it posts when:
- A trade is OPENED (order filled)
- A trade is CLOSED (position exited with P&L)

Uses Argus personality: Witty, Sarcastic, Rebellious - with full transparency.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pathlib import Path
import aiosqlite
import json

router = APIRouter(prefix="/social", tags=["social"])

# Database path
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "v4_live_paper.db"


@router.get("/twitter/status")
async def get_twitter_status():
    """
    Get Twitter bot configuration and rate limit status.
    Now includes xAI status for organic Argus-personality tweets.
    """
    from src.social.twitter_bot import TwitterBot

    bot = TwitterBot()

    return {
        "configured": bot.is_configured,
        "has_xai": bot.has_xai,
        "base_url": bot.base_url,
        "rate_limit": bot.get_rate_limit_status(),
        "mode": "trades_only",  # Only posts executed trades, not decisions
        "personality": "Argus - Witty, Sarcastic, Rebellious"
    }


@router.post("/twitter/test-trade")
async def test_twitter_trade_post(trade_id: str = Query(None, description="Trade ID to tweet (uses latest closed trade if not provided)")):
    """
    Test posting a TRADE (not decision) to Twitter.

    Only posts executed trades - this is the new behavior.
    If no trade_id is provided, uses the most recent closed trade.
    """
    from src.social.twitter_bot import TwitterBot

    bot = TwitterBot()

    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        if trade_id:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE exit_timestamp IS NOT NULL ORDER BY exit_timestamp DESC LIMIT 1"
            )

        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="No trade found")

        trade = dict(row)

    # Format the tweet with Argus personality
    if trade.get("exit_timestamp"):
        tweet_text = await bot.format_trade_exit_organic(trade)
        tweet_type = "exit"
    else:
        tweet_text = await bot.format_trade_entry_organic(trade)
        tweet_type = "entry"

    response = {
        "trade_id": trade.get("trade_id"),
        "symbol": trade.get("symbol"),
        "type": tweet_type,
        "is_winner": trade.get("is_winner"),
        "net_pnl": trade.get("net_pnl"),
        "tweet_text": tweet_text,
        "tweet_length": len(tweet_text),
        "configured": bot.is_configured,
        "has_xai": bot.has_xai,
        "rate_limit": bot.get_rate_limit_status()
    }

    if bot.is_configured:
        if tweet_type == "exit":
            result = await bot.post_trade_exit(trade)
        else:
            result = await bot.post_trade_entry(trade)
        response["post_result"] = result
    else:
        response["post_result"] = {
            "success": False,
            "dry_run": True,
            "message": "Twitter not configured. Set TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET to enable posting."
        }

    return response


@router.post("/twitter/test")
async def test_twitter_post(decision_id: str = Query(None, description="Decision ID to tweet (LEGACY - prefer /twitter/test-trade)")):
    """
    LEGACY: Test posting a decision to Twitter.

    NOTE: The new behavior is to post TRADES, not decisions.
    Use /twitter/test-trade for the new trade-based posting.
    This endpoint is kept for backward compatibility.
    """
    from src.social.twitter_bot import TwitterBot

    bot = TwitterBot()

    # Get the decision
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        if decision_id:
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE decision_id = ?",
                (decision_id,)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE result IN ('signal_long', 'signal_short') ORDER BY timestamp DESC LIMIT 1"
            )

        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="No decision found")

        decision = dict(row)

    # Format the tweet
    tweet_text = bot.format_tweet(decision)

    response = {
        "decision_id": decision.get("decision_id"),
        "symbol": decision.get("symbol"),
        "result": decision.get("result"),
        "tweet_text": tweet_text,
        "tweet_length": len(tweet_text),
        "configured": bot.is_configured,
        "rate_limit": bot.get_rate_limit_status(),
        "warning": "LEGACY ENDPOINT: Use /twitter/test-trade for new trade-based posting"
    }

    if bot.is_configured:
        # Actually post
        result = await bot.post_decision(decision)
        response["post_result"] = result
    else:
        response["post_result"] = {
            "success": False,
            "dry_run": True,
            "message": "Twitter not configured. Set TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET to enable posting."
        }

    return response


@router.get("/twitter/preview/{decision_id}")
async def preview_twitter_post(decision_id: str):
    """
    Preview what a tweet would look like for a given decision.

    Does not post - just shows the formatted tweet.
    """
    from src.social.twitter_bot import TwitterBot

    bot = TwitterBot()

    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
            "SELECT * FROM decisions WHERE decision_id = ?",
            (decision_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Decision not found")

        decision = dict(row)

    tweet_text = bot.format_tweet(decision)

    return {
        "decision_id": decision.get("decision_id"),
        "symbol": decision.get("symbol"),
        "result": decision.get("result"),
        "tweet_text": tweet_text,
        "tweet_length": len(tweet_text),
        "decision_url": f"{bot.base_url}/decision/{decision_id}",
        "card_image_url": f"{bot.base_url}/api/decision-card/{decision_id}"
    }


@router.get("/twitter/pending-trades")
async def get_pending_trades():
    """
    Get TRADES that haven't been tweeted yet.

    Returns trades that have been opened or closed but not yet posted.
    This is the new behavior - only posting executed trades, not decisions.
    """
    from src.social.twitter_bot import TradeMonitor

    monitor = TradeMonitor(str(DB_PATH))
    pending_entries = await monitor.get_new_trade_entries()
    pending_exits = await monitor.get_new_trade_exits()

    return {
        "pending_entries": len(pending_entries),
        "pending_exits": len(pending_exits),
        "total_pending": len(pending_entries) + len(pending_exits),
        "entries": [
            {
                "trade_id": t.get("trade_id"),
                "symbol": t.get("symbol"),
                "direction": t.get("direction"),
                "entry_price": t.get("entry_price"),
                "entry_timestamp": t.get("entry_timestamp")
            }
            for t in pending_entries
        ],
        "exits": [
            {
                "trade_id": t.get("trade_id"),
                "symbol": t.get("symbol"),
                "is_winner": t.get("is_winner"),
                "net_pnl": t.get("net_pnl"),
                "exit_timestamp": t.get("exit_timestamp")
            }
            for t in pending_exits
        ],
        "rate_limit": monitor.twitter_bot.get_rate_limit_status()
    }


@router.post("/twitter/post-pending-trades")
async def post_pending_trades():
    """
    Post all pending TRADES to Twitter.

    Only posts executed trades (entries and exits), not decisions.
    Respects rate limits (50/day max).
    """
    from src.social.twitter_bot import TradeMonitor

    monitor = TradeMonitor(str(DB_PATH))

    if not monitor.twitter_bot.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Twitter not configured. Set environment variables first."
        )

    results = await monitor.post_new_trades()

    entries_posted = sum(1 for r in results if r.get("success") and r.get("type") == "entry")
    exits_posted = sum(1 for r in results if r.get("success") and r.get("type") == "exit")

    return {
        "entries_posted": entries_posted,
        "exits_posted": exits_posted,
        "total_posted": entries_posted + exits_posted,
        "failed_count": sum(1 for r in results if not r.get("success")),
        "results": results,
        "rate_limit": monitor.twitter_bot.get_rate_limit_status()
    }


@router.get("/twitter/pending")
async def get_pending_tweets():
    """
    LEGACY: Get decisions that haven't been tweeted yet.

    NOTE: New behavior uses /twitter/pending-trades for executed trades only.
    This endpoint is kept for backward compatibility.
    """
    from src.social.twitter_bot import DecisionMonitor

    monitor = DecisionMonitor(str(DB_PATH))
    pending = await monitor.get_new_decisions()

    return {
        "pending_count": len(pending),
        "decisions": [
            {
                "decision_id": d.get("decision_id"),
                "symbol": d.get("symbol"),
                "result": d.get("result"),
                "timestamp": d.get("timestamp")
            }
            for d in pending
        ],
        "rate_limit": monitor.twitter_bot.get_rate_limit_status(),
        "warning": "LEGACY: Use /twitter/pending-trades for new trade-based behavior"
    }


@router.post("/twitter/post-pending")
async def post_pending_tweets():
    """
    LEGACY: Post all pending decisions to Twitter.

    NOTE: New behavior uses /twitter/post-pending-trades for executed trades only.
    This endpoint is kept for backward compatibility.
    """
    from src.social.twitter_bot import DecisionMonitor

    monitor = DecisionMonitor(str(DB_PATH))

    if not monitor.twitter_bot.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Twitter not configured. Set environment variables first."
        )

    results = await monitor.post_new_decisions()

    return {
        "posted_count": sum(1 for r in results if r.get("success")),
        "failed_count": sum(1 for r in results if not r.get("success")),
        "results": results,
        "rate_limit": monitor.twitter_bot.get_rate_limit_status(),
        "warning": "LEGACY: Use /twitter/post-pending-trades for new trade-based behavior"
    }


@router.get("/twitter/recent")
async def get_recent_tweets(limit: int = Query(default=5, le=20)):
    """
    Get recent tweets/activity for the Twitter feed widget.

    Returns formatted tweet-like objects from recent decisions.
    Falls back to recent trading activity if no posted tweets are tracked.
    """
    from src.social.twitter_bot import TwitterBot

    bot = TwitterBot()

    # Get recent decisions that would be tweetable
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("""
            SELECT decision_id, symbol, result, result_reason, timestamp, market_context
            FROM decisions
            WHERE result IN ('signal_long', 'signal_short', 'signal_close')
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        decisions = [dict(row) for row in await cursor.fetchall()]

    # Format as tweet-like objects
    tweets = []
    for d in decisions:
        result = d.get("result", "").replace("signal_", "").upper()

        # Parse market context for more details
        try:
            mc = json.loads(d.get("market_context", "{}"))
            setup = mc.get("setup_details", mc.get("professional", {}))
            grade = setup.get("grade", "")
            grade_text = f" [Grade {grade}]" if grade else ""
        except:
            grade_text = ""

        text = f"{result} signal on ${d['symbol']}{grade_text}. {d.get('result_reason', 'Analyzing market conditions...')[:100]}"

        tweets.append({
            "id": d["decision_id"],
            "text": text,
            "created_at": d["timestamp"],
            "url": f"{bot.base_url}/decision/{d['decision_id']}",
            "likes": 0,
            "replies": 0,
            "retweets": 0
        })

    return {
        "tweets": tweets,
        "count": len(tweets),
        "source": "decisions"
    }


# =============================================================================
# DISCORD ENDPOINTS
# =============================================================================

@router.get("/discord/status")
async def get_discord_status():
    """
    Get Discord webhook configuration status.
    """
    from src.social.discord_bot import DiscordWebhook

    webhook = DiscordWebhook()

    return {
        "configured": webhook.is_configured,
        "base_url": webhook.base_url,
        "bot_name": webhook.bot_name
    }


@router.post("/discord/test")
async def test_discord_webhook(decision_id: str = Query(None, description="Decision ID to send (uses latest if not provided)")):
    """
    Test sending a decision alert to Discord.

    If no decision_id is provided, uses the most recent signal decision.
    """
    from src.social.discord_bot import DiscordWebhook

    webhook = DiscordWebhook()

    if not webhook.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Discord webhook not configured. Set DISCORD_WEBHOOK_URL."
        )

    # Get the decision
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        if decision_id:
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE decision_id = ?",
                (decision_id,)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM decisions WHERE result IN ('signal_long', 'signal_short') ORDER BY timestamp DESC LIMIT 1"
            )

        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="No decision found")

        decision = dict(row)

    # Send to Discord
    result = await webhook.send_decision_alert(decision)

    return {
        "decision_id": decision.get("decision_id"),
        "symbol": decision.get("symbol"),
        "result": decision.get("result"),
        "discord_result": result
    }


@router.get("/discord/pending")
async def get_discord_pending():
    """
    Get decisions that haven't been posted to Discord yet.
    """
    from src.social.discord_bot import DiscordDecisionMonitor

    monitor = DiscordDecisionMonitor(str(DB_PATH))
    pending = await monitor.get_new_decisions()

    return {
        "pending_count": len(pending),
        "decisions": [
            {
                "decision_id": d.get("decision_id"),
                "symbol": d.get("symbol"),
                "result": d.get("result"),
                "timestamp": d.get("timestamp")
            }
            for d in pending
        ]
    }


@router.post("/discord/post-pending")
async def post_discord_pending():
    """
    Post all pending decisions to Discord.
    """
    from src.social.discord_bot import DiscordDecisionMonitor

    monitor = DiscordDecisionMonitor(str(DB_PATH))

    if not monitor.webhook.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Discord webhook not configured. Set DISCORD_WEBHOOK_URL."
        )

    results = await monitor.post_new_decisions()

    return {
        "posted_count": sum(1 for r in results if r.get("success")),
        "failed_count": sum(1 for r in results if not r.get("success")),
        "results": results
    }


@router.post("/discord/stats")
async def post_discord_stats():
    """
    Post current trading stats to Discord.
    """
    from src.social.discord_bot import DiscordWebhook

    webhook = DiscordWebhook()

    if not webhook.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Discord webhook not configured. Set DISCORD_WEBHOOK_URL."
        )

    # Get stats
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("SELECT COUNT(*) as count FROM decisions")
        decision_count = (await cursor.fetchone())["count"]

        cursor = await db.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CAST(net_pnl AS REAL)) as pnl
            FROM trades
            WHERE exit_timestamp IS NOT NULL
        """)
        row = await cursor.fetchone()

    total = row["total"] or 0
    wins = row["wins"] or 0
    pnl = row["pnl"] or 0
    win_rate = (wins / total * 100) if total > 0 else 0

    stats = {
        "win_rate": win_rate,
        "avg_rr": 1.8,  # TODO: Calculate from trades
        "total_decisions": decision_count,
        "total_pnl": pnl
    }

    result = await webhook.send_stats(stats)

    return {
        "stats": stats,
        "discord_result": result
    }
