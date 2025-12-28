"""
ArgusNexus V4 - Discord Command Bot
Protocol: SIGNAL FLARE (Full Upgrade)

Two-way Discord integration:
- Sends trade alerts, scanner hits, risk warnings
- Receives commands: /status, /positions, /kill, /balance

The Fleet reports to Command. Command controls the Fleet.
"""

import os
import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable
from dataclasses import dataclass, field

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class FleetStatus:
    """Current status of the trading fleet."""
    active_pairs: list = field(default_factory=list)
    open_positions: dict = field(default_factory=dict)
    total_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    trades_today: int = 0
    is_running: bool = False
    last_signal: Optional[str] = None
    last_signal_time: Optional[datetime] = None


class ArgusNexusBot(commands.Bot):
    """
    ArgusNexus Discord Command Bot.
    """

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            description="ArgusNexus V4 Trading Fleet Command Interface"
        )

        self.authorized_users = self._parse_authorized_users()
        self.guild_id = int(os.getenv("DISCORD_GUILD_ID", "0"))
        self.fleet_status = FleetStatus()

        # Callbacks
        self.on_kill_command: Optional[Callable] = None
        self.on_pause_command: Optional[Callable] = None
        self.on_resume_command: Optional[Callable] = None
        self.get_balance_callback: Optional[Callable] = None

        # Alert channel
        self.alert_channel: Optional[discord.TextChannel] = None
        self.alert_channel_name = "argus-alerts"

    def _parse_authorized_users(self) -> set:
        users_str = os.getenv("DISCORD_AUTHORIZED_USERS", "")
        if not users_str:
            return set()
        return {int(uid.strip()) for uid in users_str.split(",") if uid.strip()}

    def is_authorized(self, user_id: int) -> bool:
        if not self.authorized_users:
            return True
        return user_id in self.authorized_users

    async def setup_hook(self):
        """Register commands on startup."""
        await self.add_cog(FleetCommands(self))

        if self.guild_id:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info(f"Synced commands to guild {self.guild_id}")
        else:
            await self.tree.sync()
            logger.info("Synced commands globally")

    async def on_ready(self):
        logger.info(f"ArgusNexus Bot online as {self.user}")
        logger.info(f"Authorized users: {self.authorized_users}")

        if self.guild_id:
            guild = self.get_guild(self.guild_id)
            if guild:
                for channel in guild.text_channels:
                    if channel.name == self.alert_channel_name:
                        self.alert_channel = channel
                        break

                if self.alert_channel:
                    logger.info(f"Alert channel: #{self.alert_channel.name}")
                    await self.send_system_alert(
                        "Bot Online",
                        "ArgusNexus V4 Command Interface ready. Use `/status` to check fleet."
                    )

    # =========================================================================
    # ALERT METHODS
    # =========================================================================

    async def send_trade_alert(self, symbol: str, side: str, entry_price: float,
                                size: float, stop_loss: float, take_profit: Optional[float] = None):
        if not self.alert_channel:
            return

        color = discord.Color.green() if side.lower() == 'buy' else discord.Color.red()
        embed = discord.Embed(
            title=f"🚨 NEW POSITION: {symbol}",
            description="**The Fleet has engaged a target.**",
            color=color,
            timestamp=datetime.utcnow()
        )
        embed.add_field(name="Action", value=side.upper(), inline=True)
        embed.add_field(name="Entry Price", value=f"${entry_price:,.4f}", inline=True)
        embed.add_field(name="Size", value=f"{size:.4f}", inline=True)
        embed.add_field(name="🛑 Stop", value=f"${stop_loss:,.4f}", inline=True)
        if take_profit:
            embed.add_field(name="🎯 Target", value=f"${take_profit:,.4f}", inline=True)
        embed.set_footer(text="ArgusNexus V4")
        await self.alert_channel.send(embed=embed)

    async def send_exit_alert(self, symbol: str, side: str, entry_price: float,
                               exit_price: float, pnl: float, pnl_percent: float):
        if not self.alert_channel:
            return

        color = discord.Color.green() if pnl >= 0 else discord.Color.red()
        emoji = "💰" if pnl >= 0 else "💸"
        embed = discord.Embed(
            title=f"{emoji} POSITION CLOSED: {symbol}",
            description="**The Fleet has exited a position.**",
            color=color,
            timestamp=datetime.utcnow()
        )
        embed.add_field(name="Direction", value=side.upper(), inline=True)
        embed.add_field(name="Entry", value=f"${entry_price:,.4f}", inline=True)
        embed.add_field(name="Exit", value=f"${exit_price:,.4f}", inline=True)
        embed.add_field(name="P&L", value=f"${pnl:,.2f}", inline=True)
        embed.add_field(name="Return", value=f"{pnl_percent:+.2f}%", inline=True)
        embed.set_footer(text="ArgusNexus V4")
        await self.alert_channel.send(embed=embed)

    async def send_system_alert(self, title: str, message: str):
        if not self.alert_channel:
            return

        embed = discord.Embed(
            title=f"🖥️ {title}",
            description=message,
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text="ArgusNexus V4")
        await self.alert_channel.send(embed=embed)


class FleetCommands(commands.Cog):
    """Slash commands for controlling the fleet."""

    def __init__(self, bot: ArgusNexusBot):
        self.bot = bot

    @app_commands.command(name="ping", description="Check bot latency")
    async def ping(self, interaction: discord.Interaction):
        latency = round(self.bot.latency * 1000)
        embed = discord.Embed(
            title="🏓 PONG",
            description=f"Latency: **{latency}ms**",
            color=discord.Color.green(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text="ArgusNexus V4")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="status", description="Get Fleet status overview")
    async def status(self, interaction: discord.Interaction):
        if not self.bot.is_authorized(interaction.user.id):
            await interaction.response.send_message("❌ Unauthorized", ephemeral=True)
            return

        status = self.bot.fleet_status
        embed = discord.Embed(
            title="📊 FLEET STATUS",
            color=discord.Color.green() if status.is_running else discord.Color.orange(),
            timestamp=datetime.utcnow()
        )
        embed.add_field(name="Status", value="🟢 ACTIVE" if status.is_running else "🟡 STANDBY", inline=True)
        embed.add_field(name="Active Pairs", value=str(len(status.active_pairs)) or "0", inline=True)
        embed.add_field(name="Open Positions", value=str(len(status.open_positions)), inline=True)
        embed.add_field(name="Daily P&L", value=f"${float(status.daily_pnl):,.2f}", inline=True)
        embed.add_field(name="Total P&L", value=f"${float(status.total_pnl):,.2f}", inline=True)
        embed.add_field(name="Trades Today", value=str(status.trades_today), inline=True)
        embed.set_footer(text="ArgusNexus V4")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="positions", description="List all open positions")
    async def positions(self, interaction: discord.Interaction):
        if not self.bot.is_authorized(interaction.user.id):
            await interaction.response.send_message("❌ Unauthorized", ephemeral=True)
            return

        positions = self.bot.fleet_status.open_positions
        if not positions:
            embed = discord.Embed(
                title="📋 OPEN POSITIONS",
                description="No open positions.",
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
        else:
            embed = discord.Embed(
                title="📋 OPEN POSITIONS",
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
            for symbol, pos in positions.items():
                pnl = pos.get('unrealized_pnl', 0)
                emoji = "🟢" if pnl >= 0 else "🔴"
                embed.add_field(
                    name=f"{emoji} {symbol}",
                    value=f"Entry: ${pos.get('entry_price', 0):,.2f}\nP&L: ${pnl:,.2f}",
                    inline=True
                )
        embed.set_footer(text="ArgusNexus V4")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="balance", description="Show account balances")
    async def balance(self, interaction: discord.Interaction):
        if not self.bot.is_authorized(interaction.user.id):
            await interaction.response.send_message("❌ Unauthorized", ephemeral=True)
            return

        embed = discord.Embed(
            title="💰 ACCOUNT BALANCE",
            description="Balance data from paper trading engine.",
            color=discord.Color.gold(),
            timestamp=datetime.utcnow()
        )

        if self.bot.get_balance_callback:
            balances = self.bot.get_balance_callback()
            for currency, amount in balances.items():
                embed.add_field(name=currency, value=f"{amount:,.4f}", inline=True)
        else:
            embed.add_field(name="USD", value="$10,000.00 (default)", inline=True)

        embed.set_footer(text="ArgusNexus V4")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="kill", description="Emergency stop all trading")
    async def kill(self, interaction: discord.Interaction):
        if not self.bot.is_authorized(interaction.user.id):
            await interaction.response.send_message("❌ Unauthorized", ephemeral=True)
            return

        if self.bot.on_kill_command:
            self.bot.on_kill_command()

        self.bot.fleet_status.is_running = False

        embed = discord.Embed(
            title="🛑 EMERGENCY STOP",
            description="**Fleet has been shut down.**\nAll trading halted.",
            color=discord.Color.red(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text="ArgusNexus V4")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="pause", description="Pause trading")
    async def pause(self, interaction: discord.Interaction):
        if not self.bot.is_authorized(interaction.user.id):
            await interaction.response.send_message("❌ Unauthorized", ephemeral=True)
            return

        if self.bot.on_pause_command:
            self.bot.on_pause_command()

        embed = discord.Embed(
            title="⏸️ FLEET PAUSED",
            description="Trading paused. Monitoring continues.",
            color=discord.Color.orange(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text="ArgusNexus V4")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="resume", description="Resume trading")
    async def resume(self, interaction: discord.Interaction):
        if not self.bot.is_authorized(interaction.user.id):
            await interaction.response.send_message("❌ Unauthorized", ephemeral=True)
            return

        if self.bot.on_resume_command:
            self.bot.on_resume_command()

        self.bot.fleet_status.is_running = True

        embed = discord.Embed(
            title="▶️ FLEET RESUMED",
            description="Trading resumed. Fleet is active.",
            color=discord.Color.green(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text="ArgusNexus V4")
        await interaction.response.send_message(embed=embed)


def run_bot():
    """Run the bot."""
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN not set")

    bot = ArgusNexusBot()
    bot.run(token)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting ArgusNexus Discord Bot...")
    print("Commands: /ping, /status, /positions, /balance, /kill, /pause, /resume")
    run_bot()
