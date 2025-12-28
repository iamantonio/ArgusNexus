#!/usr/bin/env python3
"""
ArgusNexus V4 - Automated Scanner Runner
Protocol: THE ALL-SEEING EYE

Runs the market scanner on a schedule and sends Discord alerts.
Designed to be called by cron every 4 hours.

Usage:
    python scripts/run_scanner.py              # Normal scan (Top 100)
    python scripts/run_scanner.py --top 50     # Scan top 50
    python scripts/run_scanner.py --full       # Full market scan

Cron Example (every 4 hours):
    0 */4 * * * /path/to/ArgusNexus/venv/bin/python /path/to/ArgusNexus/scripts/run_scanner.py >> /path/to/ArgusNexus/logs/scanner.log 2>&1
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from scanner import CoinbaseMarketScanner, Granularity, MarketScanResult, ScanResult
from notifier import DiscordNotifier

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


def log(message: str):
    """Print timestamped log message."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}")


def format_scan_summary(result: MarketScanResult) -> str:
    """Format scan results for logging."""
    lines = [
        "=" * 60,
        "SCANNER REPORT - THE ALL-SEEING EYE",
        "=" * 60,
        f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Duration: {result.scan_duration_seconds:.1f}s",
        f"Products: {result.successful}/{result.total_products} scanned",
        f"Failed: {result.failed}",
        f"Breakouts: {len(result.breakouts)}",
        ""
    ]

    if result.breakouts:
        lines.append("BREAKOUTS DETECTED:")
        for b in result.breakouts:
            lines.append(f"  - {b.product_id}: ${float(b.price):,.4f} (breakout at ${float(b.upper_channel):,.4f})")
    else:
        lines.append("No breakouts detected.")

    # Near breakouts (within 3%)
    near = []
    for sr in result.results:
        if sr.price > 0 and sr.upper_channel > 0 and sr.error is None:
            distance_pct = ((sr.upper_channel - sr.price) / sr.price) * 100
            if 0 < distance_pct <= 3.0:
                near.append((sr.product_id, float(sr.price), float(sr.upper_channel), distance_pct))

    if near:
        near.sort(key=lambda x: x[3])
        lines.append("")
        lines.append("NEAR BREAKOUTS (within 3%):")
        for pid, price, upper, dist in near[:5]:
            lines.append(f"  - {pid}: ${price:,.4f} ({dist:.1f}% to breakout)")

    lines.append("=" * 60)
    return "\n".join(lines)


async def run_scanner(top_n: int = 100, full_scan: bool = False) -> MarketScanResult:
    """Run the market scanner."""
    scanner = CoinbaseMarketScanner(
        requests_per_second=8,
        use_public_api=True,
        verbose=False  # Quiet for cron
    )

    result = await scanner.scan_market(
        granularity=Granularity.FOUR_HOUR,
        candle_limit=300,
        top_n=None if full_scan else top_n
    )

    return result


def send_discord_alerts(result: MarketScanResult, notifier: DiscordNotifier):
    """Send Discord alerts for any breakouts found."""
    if not result.breakouts:
        return 0

    sent = 0
    for b in result.breakouts:
        success = notifier.send_scanner_alert(
            symbol=b.product_id,
            price=float(b.price),
            breakout_level=float(b.upper_channel)
        )
        if success:
            sent += 1
            log(f"Discord alert sent for {b.product_id}")
        else:
            log(f"Failed to send Discord alert for {b.product_id}")

    return sent


def send_summary_alert(result: MarketScanResult, notifier: DiscordNotifier):
    """Send a summary alert to Discord (even when no breakouts)."""
    # Build near-breakout list
    near = []
    for sr in result.results:
        if sr.price > 0 and sr.upper_channel > 0 and sr.error is None:
            distance_pct = ((sr.upper_channel - sr.price) / sr.price) * 100
            if 0 < distance_pct <= 5.0:
                near.append((sr.product_id, distance_pct))
    near.sort(key=lambda x: x[1])

    # Build message
    if result.breakouts:
        title = f"BREAKOUT DETECTED: {len(result.breakouts)} signals"
        color = 0x00FF00  # Green
    elif near:
        title = f"Scanner Complete: {len(near)} near breakout"
        color = 0xFFFF00  # Yellow
    else:
        title = "Scanner Complete: No signals"
        color = 0x3498DB  # Blue

    fields = [
        {"name": "Products Scanned", "value": f"{result.successful}/{result.total_products}", "inline": True},
        {"name": "Duration", "value": f"{result.scan_duration_seconds:.1f}s", "inline": True},
        {"name": "Breakouts", "value": str(len(result.breakouts)), "inline": True},
    ]

    if near:
        near_str = "\n".join([f"{p}: {d:.1f}%" for p, d in near[:5]])
        fields.append({"name": "Near Breakout (within 5%)", "value": near_str, "inline": False})

    embed = {
        "title": f"🔭 {title}",
        "description": "The All-Seeing Eye has completed its scan.",
        "color": color,
        "fields": fields,
        "footer": {
            "text": f"ArgusNexus V4 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        }
    }

    return notifier._post(embed)


async def main():
    parser = argparse.ArgumentParser(description="ArgusNexus Automated Scanner")
    parser.add_argument("--top", type=int, default=100, help="Scan top N by volume (default: 100)")
    parser.add_argument("--full", action="store_true", help="Full market scan (slow)")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord alerts")
    parser.add_argument("--summary", action="store_true", help="Send summary alert even with no breakouts")
    args = parser.parse_args()

    log("=" * 60)
    log("THE ALL-SEEING EYE - Automated Scanner")
    log("=" * 60)

    mode = "FULL SCAN" if args.full else f"Top {args.top}"
    log(f"Mode: {mode}")

    # Run scanner
    log("Starting scan...")
    result = await run_scanner(top_n=args.top, full_scan=args.full)

    # Log results
    summary = format_scan_summary(result)
    print(summary)

    # Send Discord alerts
    if not args.no_discord:
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        if webhook_url:
            notifier = DiscordNotifier(webhook_url)

            # Send breakout alerts
            if result.breakouts:
                alerts_sent = send_discord_alerts(result, notifier)
                log(f"Sent {alerts_sent} breakout alerts to Discord")

            # Send summary if requested or if breakouts found
            if args.summary or result.breakouts:
                if send_summary_alert(result, notifier):
                    log("Summary alert sent to Discord")
        else:
            log("DISCORD_WEBHOOK_URL not set - skipping alerts")

    log("Scan complete.")
    return result


if __name__ == "__main__":
    result = asyncio.run(main())

    # Exit with code 0 if successful, 1 if errors
    sys.exit(0 if result.failed == 0 else 1)
