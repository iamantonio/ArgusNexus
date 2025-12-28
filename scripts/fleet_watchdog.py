#!/usr/bin/env python3
"""
ArgusNexus V4 Fleet Watchdog - Protocol: OVERSEER

Monitors the health of the trading fleet by checking the heartbeat file.
If a unit hasn't ticked in too long, or if the manager is dead, alerts Discord.

Usage:
    python scripts/fleet_watchdog.py --threshold 300
"""

import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from notifier import DiscordNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
HEARTBEAT_FILE = PROJECT_ROOT / "logs" / "fleet_heartbeat.json"

class FleetWatchdog:
    """Overseer that monitors fleet health."""

    def __init__(self, threshold_seconds: int = 300):
        self.threshold_seconds = threshold_seconds
        self.notifier = DiscordNotifier()
        self.last_alert_time = 0
        self.alert_cooldown = 1800  # 30 mins between duplicate alerts

    def check_health(self):
        """Check fleet health and alert if necessary."""
        if not HEARTBEAT_FILE.exists():
            self._alert("Heartbeat file MISSING! Fleet manager may be dead.")
            return

        try:
            with open(HEARTBEAT_FILE, 'r') as f:
                data = json.load(f)
            
            hb_ts = datetime.fromisoformat(data.get("timestamp", ""))
            now = datetime.now(timezone.utc)
            age = (now - hb_ts).total_seconds()

            # Check if manager is ticking
            if age > self.threshold_seconds:
                self._alert(f"Fleet Manager is STALE ({int(age)}s old). Watchdog suspecting crash.")
                return

            # Check individual units
            units = data.get("units", {})
            for symbol, info in units.items():
                if info.get("status") == "DOWN":
                    self._alert(f"Unit {symbol} is DOWN. Manager should be restarting it...")
                
        except Exception as e:
            logger.error(f"Watchdog error: {e}")

    def _alert(self, message: str):
        """Send Discord alert with cooldown."""
        logger.error(message)
        now = time.time()
        if now - self.last_alert_time > self.alert_cooldown:
            if self.notifier.url:
                try:
                    self.notifier.send_risk_alert(
                        f"🚨 **FLEET WATCHDOG ALERT**\n\n{message}",
                        severity="critical"
                    )
                    self.last_alert_time = now
                    logger.info("Watchdog alert sent to Discord.")
                except Exception as e:
                    logger.error(f"Failed to send alert: {e}")
            else:
                logger.warning("Discord notifier not configured.")

    def run(self):
        """Main watchdog loop."""
        logger.info(f"Fleet Watchdog started. Threshold: {self.threshold_seconds}s")
        while True:
            self.check_health()
            time.sleep(60)

def main():
    parser = argparse.ArgumentParser(description="ArgusNexus V4 Fleet Watchdog")
    parser.add_argument("--threshold", type=int, default=300, help="Stale threshold in seconds")
    args = parser.parse_args()

    watchdog = FleetWatchdog(threshold_seconds=args.threshold)
    watchdog.run()

if __name__ == "__main__":
    main()
