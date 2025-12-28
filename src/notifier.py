"""
ArgusNexus V4 - Discord Notification Module
Protocol: SIGNAL FLARE

Sends formatted trade alerts to Discord via webhook.
The Fleet never trades in silence.
"""

import os
import requests
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")


class DiscordNotifier:
    """Sends beautiful embed cards to Discord for trade alerts."""

    def __init__(self, webhook_url: str = WEBHOOK_URL):
        self.url = webhook_url

    def send_trade_alert(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Sends a 'New Trade' alert with Green/Red styling.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: 'BUY' or 'SELL'
            entry_price: Entry price
            size: Position size
            stop_loss: Stop loss price
            take_profit: Optional take profit price

        Returns:
            True if sent successfully, False otherwise
        """
        # Color: Green for Buy (0x00FF00), Red for Sell (0xFF0000)
        color = 0x00FF00 if side.lower() == 'buy' else 0xFF0000

        fields = [
            {"name": "Action", "value": side.upper(), "inline": True},
            {"name": "Entry Price", "value": f"${entry_price:,.4f}", "inline": True},
            {"name": "Size", "value": f"{size:.4f}", "inline": True},
            {"name": "🛑 Hard Deck (Stop)", "value": f"${stop_loss:,.4f}", "inline": True},
        ]

        if take_profit:
            fields.append({"name": "🎯 Target (TP)", "value": f"${take_profit:,.4f}", "inline": True})

        fields.append({"name": "Strategy", "value": "TURTLE-4 Breakout", "inline": True})

        embed = {
            "title": f"🚨 NEW POSITION: {symbol}",
            "description": "**The Fleet has engaged a target.**",
            "color": color,
            "fields": fields,
            "footer": {
                "text": f"ArgusNexus V4 | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            }
        }

        return self._post(embed)

    def send_exit_alert(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float
    ) -> bool:
        """Sends a 'Position Closed' alert."""
        # Green for profit, Red for loss
        color = 0x00FF00 if pnl >= 0 else 0xFF0000
        emoji = "💰" if pnl >= 0 else "💸"

        embed = {
            "title": f"{emoji} POSITION CLOSED: {symbol}",
            "description": "**The Fleet has exited a position.**",
            "color": color,
            "fields": [
                {"name": "Direction", "value": side.upper(), "inline": True},
                {"name": "Entry", "value": f"${entry_price:,.4f}", "inline": True},
                {"name": "Exit", "value": f"${exit_price:,.4f}", "inline": True},
                {"name": "P&L", "value": f"${pnl:,.2f}", "inline": True},
                {"name": "Return", "value": f"{pnl_percent:+.2f}%", "inline": True},
            ],
            "footer": {
                "text": f"ArgusNexus V4 | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            }
        }

        return self._post(embed)

    def send_scanner_alert(
        self,
        symbol: str,
        price: float,
        breakout_level: float
    ) -> bool:
        """Sends a 'Scanner Found Target' alert (Yellow)."""
        embed = {
            "title": f"🔭 SCANNER TARGET: {symbol}",
            "description": "Potential breakout detected. Monitoring for entry.",
            "color": 0xFFFF00,  # Yellow
            "fields": [
                {"name": "Current Price", "value": f"${price:,.4f}", "inline": True},
                {"name": "Breakout Level", "value": f"${breakout_level:,.4f}", "inline": True},
                {"name": "Status", "value": "WAITING FOR CONFIRMATION", "inline": False}
            ],
            "footer": {
                "text": f"ArgusNexus V4 | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            }
        }
        return self._post(embed)

    def send_risk_alert(self, message: str, severity: str = "warning") -> bool:
        """Sends a risk management alert (Orange/Red)."""
        colors = {
            "info": 0x3498DB,     # Blue
            "warning": 0xFFA500,  # Orange
            "critical": 0xFF0000  # Red
        }

        embed = {
            "title": f"⚠️ RISK ALERT",
            "description": message,
            "color": colors.get(severity, 0xFFA500),
            "footer": {
                "text": f"ArgusNexus V4 | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            }
        }
        return self._post(embed)

    def send_system_alert(self, title: str, message: str) -> bool:
        """Sends a general system status alert (Blue)."""
        embed = {
            "title": f"🖥️ {title}",
            "description": message,
            "color": 0x3498DB,  # Blue
            "footer": {
                "text": f"ArgusNexus V4 | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            }
        }
        return self._post(embed)

    def _post(self, embed_data: dict) -> bool:
        """Posts embed to Discord webhook."""
        payload = {
            "username": "ArgusNexus Command",
            "avatar_url": "https://i.imgur.com/4M34hi2.png",
            "embeds": [embed_data]
        }

        try:
            response = requests.post(self.url, json=payload, timeout=10)
            return response.status_code in (200, 204)
        except Exception as e:
            print(f"Failed to send Discord alert: {e}")
            return False


# --- TEST IT ---
if __name__ == "__main__":
    if not WEBHOOK_URL:
        print("❌ DISCORD_WEBHOOK_URL not set in .env file")
        print("   Create a .env file with: DISCORD_WEBHOOK_URL=your_url_here")
        exit(1)

    notifier = DiscordNotifier()

    # Test trade entry
    print("📡 Sending test trade alert...")
    success = notifier.send_trade_alert(
        symbol="PYR-USD",
        side="BUY",
        entry_price=0.6200,
        size=1234.49,
        stop_loss=0.5380,
        take_profit=0.7440
    )

    if success:
        print("✅ Test alert sent! Check Discord.")
    else:
        print("❌ Failed to send alert. Check your webhook URL.")
