#!/usr/bin/env python3
"""
ArgusNexus V4 Fleet Manager - Protocol: ADMIRAL

Manages multiple TURTLE-4 trading units simultaneously.
Each unit trades a different asset, all writing to the shared Truth Engine.

Architecture:
- WAL Mode: SQLite Write-Ahead Logging for concurrent writes
- Auto-Restart: Dead units are automatically revived
- Shared Database: All units write to v4_live_paper.db
- Dashboard Integration: All positions appear in the Glass Cockpit

Usage:
    python src/fleet_manager.py

Fleet Roster:
    - BTC-USD: The King
    - ETH-USD: The Prince
    - SOL-USD: The Speed Demon
    - PYR-USD: The First Prey
"""

import subprocess
import time
import sys
import signal
import os
import json
import yaml
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from notifier import DiscordNotifier

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

def load_config():
    """Load configuration from YAML."""
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
fleet_config = config.get('fleet', {})
FLEET_ROSTER = fleet_config.get('roster', ["BTC-USD", "ETH-USD", "SOL-USD", "PYR-USD"])
CAPITAL_PER_UNIT = fleet_config.get('capital_per_unit', 10000)
INTERVAL = config.get('system', {}).get('interval', "4h")
DATA_SOURCE = config.get('system', {}).get('data_source', "coinbase")
HEARTBEAT_FILE = PROJECT_ROOT / "logs" / "fleet_heartbeat.json"

# --- DEPLOY LOCK ---
PROJECT_ROOT = Path(__file__).parent.parent
TRADER_SCRIPT = PROJECT_ROOT / "scripts" / "live_paper_trader.py"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

# Deploy lock file paths (CRITICAL: These were undefined causing crashes)
BYPASS_COOLDOWN_FILE = PROJECT_ROOT / "runtime" / "deploy_lock_bypass.json"
ARTIFACT_FILE = PROJECT_ROOT / "artifacts" / "rollout_artifact.json"
ARTIFACT_MAX_AGE_MINUTES = 60
BYPASS_COOLDOWN_HOURS = 24

processes = {}
start_times = {}

# Discord Notifier - The Voice of the Fleet
notifier = DiscordNotifier()


def check_bypass_cooldown() -> tuple[bool, str]:
    """Check if bypass is in cooldown period."""
    if not BYPASS_COOLDOWN_FILE.exists():
        return False, ""

    try:
        with open(BYPASS_COOLDOWN_FILE, 'r') as f:
            cooldown_data = json.load(f)

        cooldown_ts = datetime.fromisoformat(cooldown_data.get("timestamp", ""))
        now = datetime.now(timezone.utc)
        hours_since = (now - cooldown_ts).total_seconds() / 3600

        if hours_since < BYPASS_COOLDOWN_HOURS:
            remaining = BYPASS_COOLDOWN_HOURS - hours_since
            return True, f"{remaining:.1f} hours remaining (last bypass: {cooldown_data.get('reason', 'unknown')})"
        return False, ""
    except Exception:
        return False, ""


def record_bypass(reason: str) -> None:
    """Record bypass usage for cooldown and audit."""
    BYPASS_COOLDOWN_FILE.parent.mkdir(exist_ok=True)

    cooldown_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "user": os.environ.get("USER", "unknown")
    }

    with open(BYPASS_COOLDOWN_FILE, 'w') as f:
        json.dump(cooldown_data, f, indent=2)


def log_bypass_to_truth(reason: str) -> None:
    """Log bypass as CRITICAL event to Truth Engine (Async helper)."""
    async def _log():
        try:
            from truth.logger import TruthLogger
            from truth.schema import DecisionResult

            db_path = PROJECT_ROOT / "data" / "v4_live_paper.db"
            truth_logger = TruthLogger(str(db_path))
            await truth_logger.initialize()

            await truth_logger.log_decision(
                symbol="FLEET",
                strategy_name="deploy_lock_bypass",
                signal_values={
                    "event_type": "DEPLOY_LOCK_BYPASSED",
                    "severity": "CRITICAL",
                    "reason": reason,
                    "user": os.environ.get("USER", "unknown"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                risk_checks={},
                result=DecisionResult.NO_SIGNAL,
                result_reason=f"[CRITICAL] Deploy lock bypassed: {reason}"
            )
        except Exception as e:
            log(f"⚠️  Failed to log bypass to Truth Engine: {e}")
    
    asyncio.run(_log())


def check_deploy_lock() -> bool:
    """
    DEPLOY LOCK: Verify rollout artifact before allowing fleet launch.

    Validates:
    1. Artifact file exists
    2. exit_code == 0
    3. git_commit matches current HEAD
    4. timestamp is within ARTIFACT_MAX_AGE_MINUTES

    Bypass (emergency only):
    - Set DEPLOY_LOCK_BYPASS=1
    - Set DEPLOY_LOCK_BYPASS_REASON="your reason here"
    - Bypass is logged as CRITICAL to Truth Engine + Discord
    - 24-hour cooldown after bypass (requires DEPLOY_LOCK_BYPASS_ACK_RISK=1 to override)

    Returns True if deployment is allowed, False otherwise.
    """
    log("🔐 Checking deploy lock...")

    # --- EMERGENCY BYPASS PATH ---
    bypass_requested = os.environ.get("DEPLOY_LOCK_BYPASS") == "1"
    bypass_reason = os.environ.get("DEPLOY_LOCK_BYPASS_REASON", "").strip()

    if bypass_requested:
        # Require a reason
        if not bypass_reason:
            log("❌ BYPASS REJECTED: DEPLOY_LOCK_BYPASS_REASON not set")
            log("   To bypass, you must provide a reason:")
            log("   DEPLOY_LOCK_BYPASS=1 DEPLOY_LOCK_BYPASS_REASON=\"your reason\" python src/fleet_manager.py")
            return False

        # Check cooldown
        in_cooldown, cooldown_msg = check_bypass_cooldown()
        if in_cooldown:
            ack_risk = os.environ.get("DEPLOY_LOCK_BYPASS_ACK_RISK") == "1"
            if not ack_risk:
                log(f"❌ BYPASS REJECTED: In cooldown period ({cooldown_msg})")
                log("   A bypass was used recently. To bypass again:")
                log("   DEPLOY_LOCK_BYPASS=1 DEPLOY_LOCK_BYPASS_REASON=\"...\" DEPLOY_LOCK_BYPASS_ACK_RISK=1 python src/fleet_manager.py")
                return False
            else:
                log(f"⚠️  BYPASS COOLDOWN OVERRIDDEN (ACK_RISK set)")

        # Bypass approved - log everything
        log("")
        log("🚨 " + "=" * 56 + " 🚨")
        log("🚨 DEPLOY LOCK BYPASSED - EMERGENCY MODE")
        log("🚨 " + "=" * 56 + " 🚨")
        log(f"   Reason: {bypass_reason}")
        log(f"   User: {os.environ.get('USER', 'unknown')}")
        log("")

        # Record for cooldown
        record_bypass(bypass_reason)

        # Log to Truth Engine
        log_bypass_to_truth(bypass_reason)

        # Alert Discord
        if notifier:
            try:
                notifier.send_risk_alert(
                    f"🚨 **DEPLOY LOCK BYPASSED**\n\n"
                    f"**Reason:** {bypass_reason}\n"
                    f"**User:** {os.environ.get('USER', 'unknown')}\n\n"
                    f"This is logged as CRITICAL. Review required.",
                    severity="critical"
                )
                log("   Discord CRITICAL alert sent")
            except Exception as e:
                log(f"   ⚠️  Discord alert failed: {e}")

        log("")
        log("⚠️  Proceeding WITHOUT validation. You own this decision.")
        log("")
        return True

    # --- NORMAL VALIDATION PATH ---

    # Check 1: Artifact exists
    if not ARTIFACT_FILE.exists():
        log(f"❌ DEPLOY BLOCKED: Rollout artifact not found")
        log(f"   Expected: {ARTIFACT_FILE}")
        log(f"   Run: python scripts/rollout_pyr_controlled.py")
        return False

    # Load artifact
    try:
        with open(ARTIFACT_FILE, 'r') as f:
            artifact = json.load(f)
    except Exception as e:
        log(f"❌ DEPLOY BLOCKED: Cannot read artifact: {e}")
        return False

    # Check 2: exit_code == 0
    if artifact.get("exit_code") != 0:
        log(f"❌ DEPLOY BLOCKED: Artifact shows exit_code={artifact.get('exit_code')}")
        log(f"   Rollout did not pass. Re-run rollout_pyr_controlled.py")
        return False

    # Check 3: git_commit matches current HEAD
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        current_head = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        current_head = "unknown"

    artifact_commit = artifact.get("git_commit", "")
    if artifact_commit != current_head:
        log(f"❌ DEPLOY BLOCKED: Artifact is from different commit")
        log(f"   Artifact commit: {artifact_commit[:12]}")
        log(f"   Current HEAD:    {current_head[:12]}")
        log(f"   Re-run rollout_pyr_controlled.py after code changes")
        return False

    # Check 4: timestamp is fresh
    try:
        artifact_ts = datetime.fromisoformat(artifact.get("timestamp", ""))
        now = datetime.now(timezone.utc)
        age_minutes = (now - artifact_ts).total_seconds() / 60

        if age_minutes > ARTIFACT_MAX_AGE_MINUTES:
            log(f"❌ DEPLOY BLOCKED: Artifact is stale ({age_minutes:.1f} minutes old)")
            log(f"   Max age: {ARTIFACT_MAX_AGE_MINUTES} minutes")
            log(f"   Re-run rollout_pyr_controlled.py to generate fresh artifact")
            return False
    except Exception as e:
        log(f"❌ DEPLOY BLOCKED: Cannot parse artifact timestamp: {e}")
        return False

    # All checks passed
    log(f"✓ Deploy lock PASSED")
    log(f"   Commit: {artifact_commit[:12]}")
    log(f"   Age: {age_minutes:.1f} minutes")
    log(f"   Checks: {', '.join(artifact.get('checks_passed', []))}")
    return True


if not notifier.url:
    notifier = None


def log(message: str):
    """Timestamped logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def signal_handler(sig, frame):
    """Graceful shutdown of all fleet units."""
    log("🛑 ADMIRAL: Docking the Fleet...")

    # Send Discord alert
    if notifier:
        notifier.send_system_alert(
            "Fleet Shutdown",
            f"Admiral docking all {len(FLEET_ROSTER)} units. Trading halted."
        )

    for symbol, p in processes.items():
        if p and p.poll() is None:
            log(f"   Terminating {symbol} (PID: {p.pid})")
            p.terminate()

    # Give processes time to clean up
    time.sleep(2)

    # Force kill any stragglers
    for symbol, p in processes.items():
        if p and p.poll() is None:
            log(f"   Force killing {symbol}")
            p.kill()

    log("⚓ Fleet docked. All units accounted for.")
    sys.exit(0)

from data.loader import fetch_coinbase_data, resample_ohlcv

def calculate_dynamic_capital(symbol: str, base_capital: float) -> float:
    """
    Equalize dollar risk using 30-day volatility (ATR).
    High vol assets get less capital; low vol assets get more.
    """
    try:
        # Fetch data for volatility calculation
        df, _ = fetch_coinbase_data(symbol, lookback_days=30, interval="1d", verbose=False)
        if df.empty: return base_capital
        
        # Simple volatility metric: ATR as % of price
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
        atr_pct = (tr.rolling(14).mean() / close).iloc[-1] * 100
        
        # Target 2% daily volatility per unit
        # capital = target_risk_dollars / atr_dollars
        # For simplicity, we scale based on 2% benchmark
        # If ATR% is 4%, give 50% capital. If ATR% is 1%, give 200% capital.
        multiplier = 2.0 / float(atr_pct)
        # Cap multiplier between 0.5 and 2.0 to prevent extreme sizing
        multiplier = max(0.5, min(2.0, multiplier))
        
        dyn_capital = base_capital * multiplier
        log(f"💰 DYNAMIC ALLOC [{symbol}]: ATR%={float(atr_pct):.2f}, Multiplier={multiplier:.2f}x, Capital=${dyn_capital:,.0f}")
        return dyn_capital
    except Exception as e:
        log(f"⚠️  Dynamic capital failed for {symbol}: {e}")
        return base_capital

def launch_unit(symbol: str) -> subprocess.Popen:
    """Launch a single trading unit with dynamic capital."""
    # DEVIL'S ADVOCATE: Should we recalculate capital on every restart?
    # PUSHBACK: Recalculating on restart ensures we adapt to market shifts.
    # We use base capital from config as the anchor.
    unit_capital = calculate_dynamic_capital(symbol, CAPITAL_PER_UNIT)
    
    cmd = [
        str(VENV_PYTHON),
        str(TRADER_SCRIPT),
        "--symbol", symbol,
        "--interval", INTERVAL,
        "--capital", str(unit_capital),
        "--data-source", DATA_SOURCE
    ]

    # Launch with output to log files
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"{symbol.replace('-', '_').lower()}.log"

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Unit started at {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n")

    proc = subprocess.Popen(
        cmd,
        stdout=open(log_file, 'a'),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT)
    )

    return proc

def update_heartbeat():
    """Write current fleet status to heartbeat file."""
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "units": {}
    }
    for symbol in FLEET_ROSTER:
        p = processes.get(symbol)
        is_active = p and p.poll() is None
        status["units"][symbol] = {
            "status": "ACTIVE" if is_active else "DOWN",
            "pid": p.pid if is_active else None,
            "uptime": str(datetime.now() - start_times.get(symbol, datetime.now())) if is_active else "0"
        }
    
    try:
        HEARTBEAT_FILE.parent.mkdir(exist_ok=True)
        with open(HEARTBEAT_FILE, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        log(f"⚠️  Failed to update heartbeat: {e}")

def check_and_restart_units():
    """Check if any units have died and restart them."""
    for symbol in FLEET_ROSTER:
        p = processes.get(symbol)

        if p is None or p.poll() is not None:
            # Unit is dead or never started
            exit_code = p.returncode if p else "N/A"

            if p is not None:
                log(f"⚠️  Unit {symbol} died (exit code: {exit_code}). Restarting...")

            # Launch new instance
            new_proc = launch_unit(symbol)
            processes[symbol] = new_proc
            start_times[symbol] = datetime.now()

            if p is None:
                log(f"   ✅ Deployed: {symbol} (PID: {new_proc.pid})")
            else:
                log(f"   🔄 Restarted: {symbol} (PID: {new_proc.pid})")
                # Alert on restart
                if notifier:
                    notifier.send_risk_alert(
                        f"Unit **{symbol}** crashed (exit code: {exit_code}) and was auto-restarted.",
                        severity="warning"
                    )

def print_fleet_status():
    """Print current fleet status."""
    log("📊 FLEET STATUS:")
    for symbol in FLEET_ROSTER:
        p = processes.get(symbol)
        if p and p.poll() is None:
            uptime = datetime.now() - start_times.get(symbol, datetime.now())
            log(f"   {symbol}: ACTIVE (PID: {p.pid}, Uptime: {uptime})")
        else:
            log(f"   {symbol}: DOWN")


def print_morning_report():
    """
    Print comprehensive hourly status report.

    Includes:
    - Unit status (ACTIVE/PAUSED/FAIL_CLOSED)
    - Last tick time
    - Last reconcile outcome
    - Open positions (broker + DB)
    - Last CRITICAL (if any)
    """
    import sqlite3

    log("")
    log("=" * 70)
    log("📋 HOURLY FLEET REPORT - " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    log("=" * 70)

    db_path = PROJECT_ROOT / "data" / "v4_live_paper.db"

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        for symbol in FLEET_ROSTER:
            p = processes.get(symbol)

            # Process status
            if p and p.poll() is None:
                status = "ACTIVE"
                uptime = datetime.now() - start_times.get(symbol, datetime.now())
                uptime_str = str(uptime).split('.')[0]  # Remove microseconds
            else:
                status = "DOWN"
                uptime_str = "N/A"

            # Query last tick from decisions
            last_tick = conn.execute("""
                SELECT timestamp, result, result_reason
                FROM decisions
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,)).fetchone()

            if last_tick:
                # Extract time from ISO format (e.g., "2025-12-17T06:03:38.123")
                ts = last_tick['timestamp']
                if 'T' in ts:
                    tick_time = ts.split('T')[1][:8]  # HH:MM:SS
                else:
                    tick_time = ts[-8:]
                tick_result = last_tick['result']
            else:
                tick_time = "N/A"
                tick_result = "N/A"

            # Check for fail_closed status
            is_fail_closed = conn.execute("""
                SELECT 1 FROM decisions
                WHERE symbol = ?
                  AND result_reason LIKE '%FAIL_CLOSED%'
                  AND timestamp > datetime('now', '-10 minutes')
                LIMIT 1
            """, (symbol,)).fetchone()

            if is_fail_closed:
                status = "FAIL_CLOSED"

            # Open positions (from trades table)
            open_trade = conn.execute("""
                SELECT trade_id, entry_price, quantity
                FROM trades
                WHERE symbol = ? AND status = 'open'
            """, (symbol,)).fetchone()

            if open_trade:
                pos_info = f"LONG {open_trade['quantity']} @ ${open_trade['entry_price']}"
            else:
                pos_info = "FLAT"

            # Last CRITICAL in 24h
            last_critical = conn.execute("""
                SELECT timestamp, result_reason
                FROM decisions
                WHERE symbol = ?
                  AND result_reason LIKE '%CRITICAL%'
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,)).fetchone()

            if last_critical:
                critical_time = last_critical['timestamp'][:16]  # Date + time
                critical_reason = last_critical['result_reason'][:40]
            else:
                critical_time = None
                critical_reason = None

            # Print unit report
            log(f"")
            log(f"  {symbol}")
            log(f"    Status:      {status} (uptime: {uptime_str})")
            log(f"    Last Tick:   {tick_time} | Result: {tick_result}")
            log(f"    Position:    {pos_info}")
            if critical_time:
                log(f"    ⚠️  CRITICAL: {critical_time} - {critical_reason}")

        conn.close()

    except Exception as e:
        log(f"  ⚠️  Error generating report: {e}")

    log("")
    log("=" * 70)

    # Send to Discord as well
    if notifier:
        try:
            report_lines = []
            for symbol in FLEET_ROSTER:
                p = processes.get(symbol)
                status = "ACTIVE" if (p and p.poll() is None) else "DOWN"
                report_lines.append(f"• **{symbol}**: {status}")

            notifier.send_system_alert(
                "Hourly Fleet Report",
                "\n".join(report_lines)
            )
        except Exception:
            pass  # Don't fail on Discord error

def launch_fleet():
    """Main fleet launch sequence."""
    log("")
    log("=" * 60)
    log("🚀 ARGUSNEXUS V4 FLEET MANAGER - PROTOCOL: ADMIRAL")
    log("=" * 60)
    log(f"Fleet Roster: {FLEET_ROSTER}")
    log(f"Interval: {INTERVAL}")
    log(f"Capital per unit: ${CAPITAL_PER_UNIT:,}")
    log(f"Data Source: {DATA_SOURCE.upper()}")
    log("")

    # Initial deployment
    log("📡 Deploying all units...")
    for symbol in FLEET_ROSTER:
        proc = launch_unit(symbol)
        processes[symbol] = proc
        start_times[symbol] = datetime.now()
        log(f"   ✅ Deployed: {symbol} (PID: {proc.pid})")
        time.sleep(2)  # Stagger launches to avoid rate limit bursts

    log("")
    log("⚓ The Fleet is sailing.")
    log("   Dashboard: http://localhost:8000")
    log("   Press Ctrl+C to dock the fleet.")
    log("")

    # Send Discord launch alert
    if notifier:
        roster_str = ", ".join(FLEET_ROSTER)
        notifier.send_system_alert(
            "Fleet Deployed",
            f"**{len(FLEET_ROSTER)} units now trading 24/7**\n\n"
            f"**Roster:** {roster_str}\n"
            f"**Interval:** {INTERVAL}\n"
            f"**Capital/Unit:** ${CAPITAL_PER_UNIT:,}"
        )
        log("📡 Discord: Fleet launch alert sent")

    # Main monitoring loop
    status_interval = 300  # Print brief status every 5 minutes
    report_interval = 3600  # Print full morning report every hour
    last_status_time = time.time()
    last_report_time = time.time()

    try:
        while True:
            # Check for dead units and update heartbeat every 10 seconds
            time.sleep(10)
            check_and_restart_units()
            update_heartbeat()

            current_time = time.time()

            # Print brief status every 5 minutes
            if current_time - last_status_time > status_interval:
                print_fleet_status()
                last_status_time = current_time

            # Print comprehensive report every hour
            if current_time - last_report_time > report_interval:
                print_morning_report()
                last_report_time = current_time

    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Verify trader script exists
    if not TRADER_SCRIPT.exists():
        log(f"❌ ERROR: Trader script not found at {TRADER_SCRIPT}")
        sys.exit(1)

    # Verify venv python exists
    if not VENV_PYTHON.exists():
        log(f"❌ ERROR: Python not found at {VENV_PYTHON}")
        sys.exit(1)

    # --- DEPLOY LOCK CHECK ---
    # After PYR-2025-12-16-001, we require passing rollout validation before deploy.
    # This turns procedure into system - humans cannot accidentally bypass.
    if not check_deploy_lock():
        log("")
        log("=" * 60)
        log("FLEET LAUNCH BLOCKED BY DEPLOY LOCK")
        log("=" * 60)
        log("")
        log("The deploy lock ensures you've run the full validation sequence:")
        log("  1. pytest tests/test_position_reconciliation.py")
        log("  2. python scripts/validate_reconciliation.py")
        log("  3. python scripts/close_postmortem_PYR_2025_12_16.py")
        log("  4. python scripts/rollout_pyr_controlled.py")
        log("")
        log("Only step 4 generates the artifact that unlocks deployment.")
        log("")
        sys.exit(1)

    launch_fleet()
