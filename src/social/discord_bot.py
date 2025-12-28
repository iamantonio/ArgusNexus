"""
Argus Discord Bot

Real-time trading decision alerts and slash commands for the Discord community.

Features:
- Slash commands: /status, /last, /stats
- Real-time decision alerts via webhook
- Beautiful embeds with Argus branding
- Rate-aware design

Environment Variables Required:
- DISCORD_BOT_TOKEN: Bot token for slash commands
- DISCORD_WEBHOOK_URL: Webhook URL for alerts channel
- ARGUS_BASE_URL: Base URL for decision links

Usage:
    # For webhook alerts only (no bot required):
    from src.social.discord_bot import DiscordWebhook
    webhook = DiscordWebhook()
    await webhook.send_decision_alert(decision)

    # For full bot with slash commands:
    from src.social.discord_bot import run_bot
    run_bot()
"""

import os
import json
import asyncio
import aiosqlite
import httpx
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Try to import discord.py (optional for webhook-only mode)
try:
    import discord
    from discord import app_commands
    from discord.ext import commands, tasks
    DISCORD_PY_AVAILABLE = True
except ImportError:
    DISCORD_PY_AVAILABLE = False
    discord = None

# Paths
DB_PATH = Path(__file__).parent.parent.parent / "data" / "v4_live_paper.db"

# Colors matching Argus theme
COLORS = {
    "accent": 0x00D4AA,      # Teal accent
    "success": 0x00C853,     # Green for LONG
    "danger": 0xFF5252,      # Red for SHORT
    "warning": 0xFFC107,     # Yellow for warnings
    "neutral": 0x888888,     # Gray for info
}


@dataclass
class EmbedConfig:
    """Configuration for Discord embeds."""
    title: str
    description: str
    color: int
    fields: List[Dict[str, Any]]
    footer: str = "Argus | Transparency Triumphs"
    thumbnail_url: Optional[str] = None
    image_url: Optional[str] = None


class DiscordWebhook:
    """
    Discord Webhook handler for sending alerts.

    Works without discord.py - uses raw HTTP requests.
    Perfect for simple alert integration.
    """

    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.base_url = os.getenv("ARGUS_BASE_URL", "http://localhost:8000")
        self.bot_name = "Argus Trading Bot"
        self.avatar_url = None  # Can set to Argus logo URL

    @property
    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def _build_decision_embed(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Build a Discord embed for a trading decision."""
        symbol = decision.get("symbol", "???")
        result = decision.get("result", "unknown")
        decision_id = decision.get("decision_id", "")
        timestamp = decision.get("timestamp", "")
        reason = decision.get("result_reason", "No reason provided")

        # Parse market context
        market_context = decision.get("market_context", {})
        if isinstance(market_context, str):
            try:
                market_context = json.loads(market_context)
            except:
                market_context = {}

        setup = market_context.get("setup_details", market_context.get("professional", {}))
        grade = setup.get("grade", "")
        risk = setup.get("risk", {})
        stop = risk.get("stop")
        target = risk.get("target")
        risk_reward = risk.get("risk_reward")
        current_price = market_context.get("current_price") or setup.get("current_price")

        # Determine styling
        if result == "signal_long":
            color = COLORS["success"]
            direction = "📈 LONG"
            emoji = "🟢"
        elif result == "signal_short":
            color = COLORS["danger"]
            direction = "📉 SHORT"
            emoji = "🔴"
        elif result in ("signal_close", "signal_exit"):
            color = COLORS["warning"]
            direction = "🚪 EXIT"
            emoji = "🟡"
        else:
            color = COLORS["neutral"]
            direction = result.upper().replace("_", " ")
            emoji = "⚪"

        # Build fields
        fields = []

        if current_price:
            fields.append({
                "name": "💰 Entry Price",
                "value": f"${current_price:,.2f}" if isinstance(current_price, (int, float)) else str(current_price),
                "inline": True
            })

        if stop:
            fields.append({
                "name": "🛑 Stop Loss",
                "value": f"${stop:,.2f}" if isinstance(stop, (int, float)) else str(stop),
                "inline": True
            })

        if target:
            fields.append({
                "name": "🎯 Target",
                "value": f"${target:,.2f}" if isinstance(target, (int, float)) else str(target),
                "inline": True
            })

        if risk_reward:
            fields.append({
                "name": "⚖️ Risk:Reward",
                "value": f"1:{risk_reward:.1f}",
                "inline": True
            })

        if grade:
            grade_emoji = "🏆" if grade in ("A+", "A") else "📊" if grade == "B" else "📉"
            fields.append({
                "name": f"{grade_emoji} Grade",
                "value": grade,
                "inline": True
            })

        # Add reasoning
        if reason and len(reason) > 200:
            reason = reason[:197] + "..."
        fields.append({
            "name": "📝 Reasoning",
            "value": reason or "No specific reason recorded",
            "inline": False
        })

        # Decision link
        decision_url = f"{self.base_url}/decision/{decision_id}"

        embed = {
            "title": f"{emoji} {direction} | {symbol}",
            "description": f"**New trading decision detected**\n\n[View Full Decision Record]({decision_url})",
            "color": color,
            "fields": fields,
            "footer": {
                "text": "Argus | Every trade is traceable"
            },
            "timestamp": timestamp if "T" in str(timestamp) else None,
            "thumbnail": {
                "url": f"{self.base_url}/api/decision-card/{decision_id}"
            }
        }

        return embed

    def _build_stats_embed(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Build a Discord embed for trading stats."""
        win_rate = stats.get("win_rate", 0)
        total_decisions = stats.get("total_decisions", 0)
        total_pnl = stats.get("total_pnl", 0)
        avg_rr = stats.get("avg_rr", 0)

        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_text = f"+${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}"

        embed = {
            "title": "📊 Argus Trading Stats",
            "description": "Current performance metrics",
            "color": COLORS["accent"],
            "fields": [
                {"name": "🎯 Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
                {"name": "⚖️ Avg R:R", "value": f"1:{avg_rr:.1f}", "inline": True},
                {"name": "📋 Decisions", "value": str(total_decisions), "inline": True},
                {"name": f"{pnl_emoji} P&L", "value": pnl_text, "inline": True},
            ],
            "footer": {"text": "Argus | Transparency Triumphs"},
            "timestamp": datetime.utcnow().isoformat()
        }

        return embed

    async def send_embed(self, embed: Dict[str, Any], content: str = None) -> Dict[str, Any]:
        """Send an embed to the Discord webhook."""
        if not self.is_configured:
            return {
                "success": False,
                "error": "Discord webhook not configured. Set DISCORD_WEBHOOK_URL."
            }

        payload = {
            "username": self.bot_name,
            "embeds": [embed]
        }

        if content:
            payload["content"] = content

        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=30.0
                )

                if response.status_code in (200, 204):
                    return {"success": True}
                else:
                    return {
                        "success": False,
                        "error": f"Discord API error: {response.status_code}",
                        "details": response.text
                    }
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def send_decision_alert(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Send a decision alert to Discord."""
        embed = self._build_decision_embed(decision)

        # Add ping for important signals
        result = decision.get("result", "")
        content = None
        if result in ("signal_long", "signal_short"):
            market_context = decision.get("market_context", {})
            if isinstance(market_context, str):
                try:
                    market_context = json.loads(market_context)
                except:
                    market_context = {}
            grade = market_context.get("setup_details", {}).get("grade", "")
            if grade in ("A+", "A"):
                content = "🚨 **High-Grade Signal Detected!**"

        return await self.send_embed(embed, content)

    async def send_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Send stats summary to Discord."""
        embed = self._build_stats_embed(stats)
        return await self.send_embed(embed)


class DiscordDecisionMonitor:
    """
    Monitors for new decisions and sends Discord alerts.

    Similar to Twitter's DecisionMonitor but for Discord webhooks.
    """

    def __init__(self, db_path: str = None, check_interval: int = 30):
        self.db_path = db_path or str(DB_PATH)
        self.check_interval = check_interval
        self.webhook = DiscordWebhook()
        self._running = False
        self._posted_ids_file = Path(self.db_path).parent / "discord_posted_ids.json"
        self._posted_ids = self._load_posted_ids()

    def _load_posted_ids(self) -> set:
        try:
            if self._posted_ids_file.exists():
                with open(self._posted_ids_file, "r") as f:
                    data = json.load(f)
                    return set(data.get("posted_ids", []))
        except:
            pass
        return set()

    def _save_posted_ids(self):
        try:
            ids_list = list(self._posted_ids)[-1000:]  # Keep last 1000
            with open(self._posted_ids_file, "w") as f:
                json.dump({"posted_ids": ids_list}, f)
        except Exception as e:
            print(f"[DiscordMonitor] Warning: Could not save posted IDs: {e}")

    async def get_new_decisions(self) -> list:
        """Get new signal decisions not yet posted to Discord."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM decisions
                WHERE result IN ('signal_long', 'signal_short')
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            rows = await cursor.fetchall()

            return [
                dict(row) for row in rows
                if row["decision_id"] not in self._posted_ids
            ]

    async def post_new_decisions(self) -> list:
        """Post any new decisions to Discord."""
        if not self.webhook.is_configured:
            return [{"success": False, "error": "Discord not configured"}]

        new_decisions = await self.get_new_decisions()
        results = []

        for decision in new_decisions:
            decision_id = decision.get("decision_id")
            result = await self.webhook.send_decision_alert(decision)
            result["decision_id"] = decision_id
            results.append(result)

            if result.get("success"):
                self._posted_ids.add(decision_id)
                self._save_posted_ids()
                print(f"[DiscordMonitor] ✓ Posted {decision.get('symbol')} {decision.get('result')}")
            else:
                print(f"[DiscordMonitor] ✗ Failed: {result.get('error')}")

            await asyncio.sleep(1)  # Rate limit courtesy

        return results

    async def run(self):
        """Run the monitor loop continuously."""
        self._running = True
        print(f"[DiscordMonitor] Starting (interval: {self.check_interval}s)")

        while self._running:
            try:
                await self.post_new_decisions()
            except Exception as e:
                print(f"[DiscordMonitor] Error: {e}")
            await asyncio.sleep(self.check_interval)

    def stop(self):
        self._running = False


# =============================================================================
# DISCORD BOT WITH SLASH COMMANDS (requires discord.py)
# =============================================================================

if DISCORD_PY_AVAILABLE:

    class ArgusBot(commands.Bot):
        """
        Argus Discord Bot with slash commands.

        Commands:
        - /status - Current trading status
        - /last [count] - Last N decisions
        - /stats - Trading performance stats
        """

        def __init__(self):
            intents = discord.Intents.default()
            intents.message_content = True

            super().__init__(
                command_prefix="!",
                intents=intents,
                description="Argus - The Transparent Trading Bot"
            )

            self.db_path = str(DB_PATH)
            self.base_url = os.getenv("ARGUS_BASE_URL", "http://localhost:8000")

        async def setup_hook(self):
            """Setup slash commands."""
            await self.add_cog(ArgusCog(self))
            await self.tree.sync()
            print("[ArgusBot] Slash commands synced")

        async def on_ready(self):
            print(f"[ArgusBot] Logged in as {self.user}")
            print(f"[ArgusBot] Connected to {len(self.guilds)} guild(s)")


    class ArgusCog(commands.Cog):
        """Cog containing all Argus slash commands."""

        def __init__(self, bot: ArgusBot):
            self.bot = bot

        @app_commands.command(name="status", description="Get current Argus trading status")
        async def status(self, interaction: discord.Interaction):
            """Show current trading status."""
            await interaction.response.defer()

            async with aiosqlite.connect(self.bot.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Get latest decision
                cursor = await db.execute(
                    "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 1"
                )
                latest = await cursor.fetchone()

                # Get active position count
                cursor = await db.execute(
                    "SELECT COUNT(*) as count FROM trades WHERE exit_timestamp IS NULL"
                )
                positions = await cursor.fetchone()

            embed = discord.Embed(
                title="🔍 Argus Trading Status",
                color=COLORS["accent"]
            )

            if latest:
                latest = dict(latest)
                embed.add_field(
                    name="📊 Latest Decision",
                    value=f"{latest['symbol']} - {latest['result'].upper().replace('_', ' ')}",
                    inline=False
                )
                embed.add_field(
                    name="⏰ Timestamp",
                    value=latest['timestamp'][:19].replace('T', ' '),
                    inline=True
                )

            embed.add_field(
                name="📈 Open Positions",
                value=str(positions['count'] if positions else 0),
                inline=True
            )

            embed.set_footer(text="Argus | Transparency Triumphs")

            await interaction.followup.send(embed=embed)

        @app_commands.command(name="last", description="Get last N trading decisions")
        @app_commands.describe(count="Number of decisions to show (1-10)")
        async def last(self, interaction: discord.Interaction, count: int = 5):
            """Show last N decisions."""
            await interaction.response.defer()

            count = max(1, min(10, count))  # Clamp to 1-10

            async with aiosqlite.connect(self.bot.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?",
                    (count,)
                )
                rows = await cursor.fetchall()

            embed = discord.Embed(
                title=f"📋 Last {count} Decisions",
                color=COLORS["accent"]
            )

            for row in rows:
                d = dict(row)
                result = d['result']

                # Emoji based on result
                if result == "signal_long":
                    emoji = "🟢"
                elif result == "signal_short":
                    emoji = "🔴"
                elif result in ("signal_close", "signal_exit"):
                    emoji = "🟡"
                else:
                    emoji = "⚪"

                timestamp = d['timestamp'][:16].replace('T', ' ')
                decision_url = f"{self.bot.base_url}/decision/{d['decision_id']}"

                embed.add_field(
                    name=f"{emoji} {d['symbol']}",
                    value=f"{result.replace('_', ' ').title()}\n{timestamp}\n[View]({decision_url})",
                    inline=True
                )

            embed.set_footer(text="Argus | Every trade is traceable")

            await interaction.followup.send(embed=embed)

        @app_commands.command(name="stats", description="Get trading performance statistics")
        async def stats(self, interaction: discord.Interaction):
            """Show trading stats."""
            await interaction.response.defer()

            async with aiosqlite.connect(self.bot.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Decision count
                cursor = await db.execute("SELECT COUNT(*) as count FROM decisions")
                decision_count = (await cursor.fetchone())['count']

                # Trade stats
                cursor = await db.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                        SUM(CAST(net_pnl AS REAL)) as pnl
                    FROM trades
                    WHERE exit_timestamp IS NOT NULL
                """)
                trade_stats = await cursor.fetchone()

            total_trades = trade_stats['total'] or 0
            wins = trade_stats['wins'] or 0
            pnl = trade_stats['pnl'] or 0

            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_color = COLORS["success"] if pnl >= 0 else COLORS["danger"]
            pnl_text = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

            embed = discord.Embed(
                title="📊 Argus Performance Stats",
                color=pnl_color
            )

            embed.add_field(name="🎯 Win Rate", value=f"{win_rate:.1f}%", inline=True)
            embed.add_field(name="📈 Total Trades", value=str(total_trades), inline=True)
            embed.add_field(name="📋 Total Decisions", value=str(decision_count), inline=True)
            embed.add_field(name=f"{pnl_emoji} Net P&L", value=pnl_text, inline=True)
            embed.add_field(name="✅ Wins", value=str(wins), inline=True)
            embed.add_field(name="❌ Losses", value=str(total_trades - wins), inline=True)

            embed.set_footer(text="Argus | Transparency Triumphs")

            await interaction.followup.send(embed=embed)


    def run_bot():
        """Run the Discord bot."""
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            print("[ArgusBot] Error: DISCORD_BOT_TOKEN not set")
            return

        bot = ArgusBot()
        bot.run(token)


else:
    # Stub for when discord.py is not available
    def run_bot():
        print("[ArgusBot] discord.py not installed. Install with: pip install discord.py")
        print("[ArgusBot] Webhook-only mode is still available via DiscordWebhook class")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def test_webhook():
    """Test Discord webhook with a mock decision."""
    webhook = DiscordWebhook()
    print(f"Configured: {webhook.is_configured}")

    if not webhook.is_configured:
        print("Set DISCORD_WEBHOOK_URL to test")
        return

    # Mock decision
    mock = {
        "decision_id": "test-discord-1234",
        "symbol": "BTC-USD",
        "result": "signal_long",
        "result_reason": "Upper breakout with strong volume, all risk gates passed",
        "timestamp": datetime.utcnow().isoformat(),
        "market_context": json.dumps({
            "setup_details": {
                "grade": "A+",
                "risk": {"stop": 95000, "target": 105000, "risk_reward": 2.5}
            },
            "current_price": 98500
        })
    }

    result = await webhook.send_decision_alert(mock)
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(test_webhook())
