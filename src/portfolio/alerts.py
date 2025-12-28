"""
Portfolio Alert System

Handles DD threshold alerts with pluggable backends:
- Console/Logger (default)
- File (JSONL audit trail)
- Discord webhook (optional)
- Slack webhook (optional)

ALERT EVENTS:
- DD >= 10%  -> INFO: drawdown building
- DD >= 12%  -> WARNING: de-risk engaged
- DD >= 20%  -> CRITICAL: flattened (confirm target_alloc=0, sleeve forced out)
- Recovery mode entered/exited
- Order failure / partial fill / position mismatch
- DD state transitions

Usage:
    from src.portfolio.alerts import AlertManager, AlertLevel

    alerter = AlertManager()

    # Check DD alerts
    alerter.check_dd_alerts(
        current_dd=Decimal("15.0"),
        dd_state=DDState.WARNING,
        prev_dd_state=DDState.NORMAL,
        recovery_mode=False,
        prev_recovery_mode=False
    )

    # Send custom alert
    alerter.send(AlertLevel.WARNING, "CUSTOM", "Something happened")
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# =============================================================================
# Alert Types
# =============================================================================

class AlertLevel(Enum):
    """Alert severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """A single alert event."""
    level: AlertLevel
    event: str
    message: str
    timestamp: datetime
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "event": self.event,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


# =============================================================================
# Alert Backends
# =============================================================================

class AlertBackend(ABC):
    """Abstract base for alert backends."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send alert. Returns True if successful."""
        pass


class ConsoleAlertBackend(AlertBackend):
    """Log alerts to console/logger."""

    def send(self, alert: Alert) -> bool:
        msg = f"[{alert.event}] {alert.message}"

        if alert.level == AlertLevel.CRITICAL:
            logger.critical(msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(msg)
        elif alert.level == AlertLevel.INFO:
            logger.info(msg)
        else:
            logger.debug(msg)

        return True


class FileAlertBackend(AlertBackend):
    """Append alerts to JSONL file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> bool:
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(alert.to_dict(), default=str) + "\n")
            return True
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
            return False


class DiscordAlertBackend(AlertBackend):
    """
    Send alerts to Discord webhook.

    Set DISCORD_WEBHOOK_URL environment variable.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    def send(self, alert: Alert) -> bool:
        if not self.webhook_url:
            return False

        try:
            import requests

            # Color by level
            colors = {
                AlertLevel.CRITICAL: 0xFF0000,  # Red
                AlertLevel.WARNING: 0xFFA500,   # Orange
                AlertLevel.INFO: 0x00FF00,      # Green
                AlertLevel.DEBUG: 0x808080,     # Gray
            }

            payload = {
                "embeds": [{
                    "title": f"🚨 {alert.event}",
                    "description": alert.message,
                    "color": colors.get(alert.level, 0x808080),
                    "timestamp": alert.timestamp.isoformat(),
                    "footer": {"text": f"Level: {alert.level.value}"}
                }]
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            return response.status_code == 204

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False


class SlackAlertBackend(AlertBackend):
    """
    Send alerts to Slack webhook.

    Set SLACK_WEBHOOK_URL environment variable.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")

    def send(self, alert: Alert) -> bool:
        if not self.webhook_url:
            return False

        try:
            import requests

            # Emoji by level
            emojis = {
                AlertLevel.CRITICAL: "🔴",
                AlertLevel.WARNING: "🟠",
                AlertLevel.INFO: "🟢",
                AlertLevel.DEBUG: "⚪",
            }

            payload = {
                "text": f"{emojis.get(alert.level, '⚪')} *{alert.event}*\n{alert.message}"
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


# =============================================================================
# Alert Manager
# =============================================================================

# DD thresholds
DD_THRESHOLD_INFO = Decimal("10.0")
DD_THRESHOLD_WARNING = Decimal("12.0")
DD_THRESHOLD_CRITICAL = Decimal("20.0")


class AlertManager:
    """
    Central alert manager with pluggable backends.

    Default backends:
    - Console (always enabled)
    - File (runtime/alerts.jsonl)

    Optional backends (enabled by env vars):
    - Discord (DISCORD_WEBHOOK_URL)
    - Slack (SLACK_WEBHOOK_URL)
    """

    def __init__(
        self,
        alert_log_path: Path = Path("runtime/alerts.jsonl"),
        enable_discord: bool = True,
        enable_slack: bool = True
    ):
        self.backends: List[AlertBackend] = [
            ConsoleAlertBackend(),
            FileAlertBackend(alert_log_path),
        ]

        # Add optional backends if configured
        if enable_discord:
            discord = DiscordAlertBackend()
            if discord.webhook_url:
                self.backends.append(discord)
                logger.info("Discord alerts enabled")

        if enable_slack:
            slack = SlackAlertBackend()
            if slack.webhook_url:
                self.backends.append(slack)
                logger.info("Slack alerts enabled")

    def send(
        self,
        level: AlertLevel,
        event: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[bool]:
        """
        Send alert to all backends.

        Returns list of success/failure for each backend.
        """
        alert = Alert(
            level=level,
            event=event,
            message=message,
            timestamp=datetime.now(timezone.utc),
            context=context or {}
        )

        results = []
        for backend in self.backends:
            try:
                results.append(backend.send(alert))
            except Exception as e:
                logger.error(f"Backend {type(backend).__name__} failed: {e}")
                results.append(False)

        return results

    def check_dd_alerts(
        self,
        current_dd: Decimal,
        dd_state: "DDState",
        prev_dd_state: Optional["DDState"],
        recovery_mode: bool,
        prev_recovery_mode: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """
        Check DD thresholds and emit appropriate alerts.

        Returns list of alerts that were sent.
        """
        from src.portfolio.portfolio_manager import DDState

        alerts_sent = []
        ctx = context or {}

        # DD threshold alerts (only alert on crossing, not every tick)
        if current_dd >= DD_THRESHOLD_CRITICAL:
            self.send(
                AlertLevel.CRITICAL,
                "DD_CRITICAL",
                f"Drawdown {float(current_dd):.1f}% >= {float(DD_THRESHOLD_CRITICAL)}% - FLATTENED",
                {**ctx, "dd": float(current_dd)}
            )
            alerts_sent.append("DD_CRITICAL")

        elif current_dd >= DD_THRESHOLD_WARNING:
            self.send(
                AlertLevel.WARNING,
                "DD_WARNING",
                f"Drawdown {float(current_dd):.1f}% >= {float(DD_THRESHOLD_WARNING)}% - de-risk engaged",
                {**ctx, "dd": float(current_dd)}
            )
            alerts_sent.append("DD_WARNING")

        elif current_dd >= DD_THRESHOLD_INFO:
            self.send(
                AlertLevel.INFO,
                "DD_INFO",
                f"Drawdown {float(current_dd):.1f}% >= {float(DD_THRESHOLD_INFO)}% - building",
                {**ctx, "dd": float(current_dd)}
            )
            alerts_sent.append("DD_INFO")

        # DD state transition alerts
        if prev_dd_state and dd_state != prev_dd_state:
            level = AlertLevel.WARNING if dd_state in (DDState.CRITICAL, DDState.WARNING) else AlertLevel.INFO
            self.send(
                level,
                "DD_STATE_CHANGE",
                f"DD state changed: {prev_dd_state.value} -> {dd_state.value}",
                {**ctx, "from": prev_dd_state.value, "to": dd_state.value}
            )
            alerts_sent.append("DD_STATE_CHANGE")

        # Recovery mode alerts
        if recovery_mode and not prev_recovery_mode:
            self.send(
                AlertLevel.WARNING,
                "RECOVERY_ENTERED",
                "Entered recovery mode - exposure restricted until DD recovers + bull regime",
                ctx
            )
            alerts_sent.append("RECOVERY_ENTERED")

        elif not recovery_mode and prev_recovery_mode:
            self.send(
                AlertLevel.INFO,
                "RECOVERY_EXITED",
                "Exited recovery mode - normal operations resumed",
                ctx
            )
            alerts_sent.append("RECOVERY_EXITED")

        return alerts_sent

    def alert_order_failure(
        self,
        order_type: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Alert on order execution failure."""
        self.send(
            AlertLevel.CRITICAL,
            "ORDER_FAILURE",
            f"Order {order_type} failed: {reason}",
            context
        )

    def alert_position_mismatch(
        self,
        expected_qty: Decimal,
        actual_qty: Decimal,
        context: Optional[Dict[str, Any]] = None
    ):
        """Alert on position mismatch (reconciliation failure)."""
        self.send(
            AlertLevel.CRITICAL,
            "POSITION_MISMATCH",
            f"Position mismatch: expected {float(expected_qty):.6f}, actual {float(actual_qty):.6f}",
            {**(context or {}), "expected": float(expected_qty), "actual": float(actual_qty)}
        )

    def alert_partial_fill(
        self,
        requested_qty: Decimal,
        filled_qty: Decimal,
        context: Optional[Dict[str, Any]] = None
    ):
        """Alert on partial order fill."""
        fill_pct = (filled_qty / requested_qty * 100) if requested_qty > 0 else 0
        self.send(
            AlertLevel.WARNING,
            "PARTIAL_FILL",
            f"Order partially filled: {float(fill_pct):.1f}% ({float(filled_qty):.6f} of {float(requested_qty):.6f})",
            {**(context or {}), "requested": float(requested_qty), "filled": float(filled_qty)}
        )

    def alert_market_data_unavailable(self, symbol: str, reason: str):
        """Alert when market data is unavailable."""
        self.send(
            AlertLevel.WARNING,
            "MARKET_DATA_UNAVAILABLE",
            f"Market data unavailable for {symbol}: {reason}",
            {"symbol": symbol, "reason": reason}
        )


# =============================================================================
# Convenience singleton
# =============================================================================

_default_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create default alert manager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = AlertManager()
    return _default_manager
