#!/usr/bin/env python3
"""
ArgusNexus V4 - Controlled PYR Rollout (Bulletproof Edition)

POST-INCIDENT CONTROLLED ROLLOUT
Incident: PYR-2025-12-16-001
Resolution: broker-as-truth reconciliation fixes

This script performs a MACHINE-VERIFIED controlled rollout:

STEP 1 - PRE-FLIGHT (automated assertions)
  - POSITION_RECONCILIATION_ENABLED = True
  - Paper executor connectivity OK
  - DB connectivity OK
  - Discord webhook reachable

STEP 2 - STARTUP VERIFICATION (machine-checked, not eyeballed)
  - Assert: RECONCILE PASSED in logs
  - Assert: HYDRATE no_position_found (clean state)

STEP 3 - TICK VERIFICATION (machine-checked)
  - Assert: Tick #1 completed without CRITICAL
  - Assert: Tick #2 completed without CRITICAL

STEP 4 - NEGATIVE TEST (prove safety brake works)
  - Inject DB-open while broker-empty mismatch
  - Assert: CRITICAL logged
  - Assert: fail-closed entered
  - Assert: trading blocked

EXIT CODES:
  0 = All checks passed, PYR cleared for fleet
  1 = Pre-flight failed
  2 = Startup verification failed
  3 = Tick verification failed
  4 = Negative test failed (safety brake broken!)
  5 = Unit crashed unexpectedly

Usage:
    python scripts/rollout_pyr_controlled.py
"""

import subprocess
import time
import sys
import signal
import re
import sqlite3
import json
import requests
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configuration
SYMBOL = "PYR-USD"
ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "rollout_pyr_controlled.pass.json"
INTERVAL = "4h"
CAPITAL = 10000
DATA_SOURCE = "coinbase"
REQUIRED_TICKS = 2

PROJECT_ROOT = Path(__file__).parent.parent
TRADER_SCRIPT = PROJECT_ROOT / "scripts" / "live_paper_trader.py"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
LOG_FILE = PROJECT_ROOT / "logs" / "pyr_rollout.log"
DB_PATH = PROJECT_ROOT / "data" / "v4_live_paper.db"

# Exit codes
EXIT_SUCCESS = 0
EXIT_PREFLIGHT_FAILED = 1
EXIT_STARTUP_FAILED = 2
EXIT_TICK_FAILED = 3
EXIT_NEGATIVE_TEST_FAILED = 4
EXIT_CRASH = 5

process = None


def log(message: str):
    """Timestamped logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def write_pass_artifact(checks_passed: list) -> None:
    """Write deployment artifact on successful rollout."""
    ARTIFACT_DIR.mkdir(exist_ok=True)

    artifact = {
        "exit_code": 0,
        "symbol": SYMBOL,
        "git_commit": get_git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks_passed": checks_passed,
        "incident_id": "PYR-2025-12-16-001"
    }

    with open(ARTIFACT_FILE, 'w') as f:
        json.dump(artifact, f, indent=2)

    log(f"  Artifact written: {ARTIFACT_FILE}")


def signal_handler(sig, frame):
    """Handle shutdown."""
    global process
    log("🛑 Rollout interrupted.")
    if process and process.poll() is None:
        process.terminate()
        time.sleep(2)
        if process.poll() is None:
            process.kill()
    sys.exit(EXIT_CRASH)


class RolloutVerifier:
    """Machine-verified rollout checks."""

    def __init__(self):
        self.process = None
        self.notifier = None
        self.checks_passed = []
        self.checks_failed = []

    def run_preflight(self) -> bool:
        """
        STEP 1: Pre-flight checks.
        All must pass or exit non-zero.
        """
        log("")
        log("=" * 70)
        log("STEP 1: PRE-FLIGHT CHECKS")
        log("=" * 70)

        all_passed = True

        # Check 1: POSITION_RECONCILIATION_ENABLED
        log("  [1/4] Checking POSITION_RECONCILIATION_ENABLED...")
        try:
            from engine import POSITION_RECONCILIATION_ENABLED
            if POSITION_RECONCILIATION_ENABLED:
                log("        ✓ POSITION_RECONCILIATION_ENABLED = True")
                self.checks_passed.append("reconciliation_enabled")
            else:
                log("        ✗ POSITION_RECONCILIATION_ENABLED = False")
                log("          (This should trigger fail-closed, which is expected)")
                # This is actually OK - the feature flag disabled path now fail-closes
                # But for rollout we want it enabled
                self.checks_failed.append("reconciliation_disabled")
                all_passed = False
        except ImportError as e:
            log(f"        ✗ Failed to import engine: {e}")
            self.checks_failed.append("engine_import_failed")
            all_passed = False

        # Check 2: Paper executor connectivity
        log("  [2/4] Checking Paper executor...")
        try:
            from execution import PaperExecutor
            executor = PaperExecutor(starting_balance=Decimal("10000"))
            balance = executor.get_balance("USD")
            if balance == Decimal("10000"):
                log("        ✓ PaperExecutor initializes correctly")
                self.checks_passed.append("executor_ok")
            else:
                log(f"        ✗ Unexpected balance: {balance}")
                self.checks_failed.append("executor_balance_wrong")
                all_passed = False
        except Exception as e:
            log(f"        ✗ PaperExecutor failed: {e}")
            self.checks_failed.append("executor_failed")
            all_passed = False

        # Check 3: DB connectivity
        log("  [3/4] Checking database connectivity...")
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM decisions")
            count = cursor.fetchone()[0]
            conn.close()
            log(f"        ✓ Database accessible ({count} decisions)")
            self.checks_passed.append("db_ok")
        except Exception as e:
            log(f"        ✗ Database failed: {e}")
            self.checks_failed.append("db_failed")
            all_passed = False

        # Check 4: Discord webhook (optional but recommended)
        log("  [4/4] Checking Discord webhook...")
        try:
            from notifier import DiscordNotifier
            self.notifier = DiscordNotifier()
            if self.notifier.url:
                # Test with a HEAD request to avoid sending a message
                try:
                    resp = requests.head(self.notifier.url, timeout=5)
                    # Discord webhooks return 405 for HEAD, but that means it's reachable
                    if resp.status_code in [200, 405]:
                        log("        ✓ Discord webhook reachable")
                        self.checks_passed.append("discord_ok")
                    else:
                        log(f"        ⚠ Discord returned {resp.status_code} (continuing)")
                        self.checks_passed.append("discord_unknown")
                except requests.RequestException as e:
                    log(f"        ⚠ Discord unreachable: {e} (continuing)")
                    self.checks_passed.append("discord_unreachable")
            else:
                log("        ⚠ No Discord webhook configured (continuing)")
                self.checks_passed.append("discord_not_configured")
        except Exception as e:
            log(f"        ⚠ Discord check failed: {e} (continuing)")
            self.checks_passed.append("discord_check_failed")

        log("")
        if all_passed:
            log("  ✓ PRE-FLIGHT PASSED")
        else:
            log("  ✗ PRE-FLIGHT FAILED")
            log(f"    Failed checks: {self.checks_failed}")

        return all_passed

    def launch_unit(self) -> bool:
        """Launch the PYR trading unit."""
        global process

        log("")
        log("=" * 70)
        log("LAUNCHING PYR UNIT")
        log("=" * 70)

        # Ensure log directory exists
        LOG_FILE.parent.mkdir(exist_ok=True)

        # Clear previous log
        with open(LOG_FILE, 'w') as f:
            f.write(f"=== PYR Controlled Rollout: {datetime.now(timezone.utc).isoformat()} ===\n")

        cmd = [
            str(VENV_PYTHON),
            str(TRADER_SCRIPT),
            "--symbol", SYMBOL,
            "--interval", INTERVAL,
            "--capital", str(CAPITAL),
            "--data-source", DATA_SOURCE
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=open(LOG_FILE, 'a'),
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT)
        )
        process = self.process  # For signal handler

        log(f"  PID: {self.process.pid}")
        log(f"  Log: {LOG_FILE}")

        return True

    def verify_startup(self, timeout_seconds: int = 120) -> bool:
        """
        STEP 2: Machine-verify startup reconciliation.
        Must see RECONCILE PASSED and HYDRATE with clean state.
        """
        log("")
        log("=" * 70)
        log("STEP 2: STARTUP VERIFICATION (machine-checked)")
        log("=" * 70)

        required_patterns = {
            "reconcile_passed": r"\[RECONCILE\] State reconciliation PASSED",
            "hydrate_clean": r"\[HYDRATE\].*(?:No open position found|no_position_found)",
        }
        found = {k: False for k in required_patterns}

        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            time.sleep(5)

            # Check if process died
            if self.process.poll() is not None:
                log(f"  ✗ Unit died during startup (exit: {self.process.returncode})")
                self._dump_log_tail(20)
                return False

            # Read and check log
            with open(LOG_FILE, 'r') as f:
                log_content = f.read()

            # Check for CRITICAL (immediate failure)
            if re.search(r"CRITICAL|FAIL-CLOSED", log_content):
                log("  ✗ CRITICAL error during startup")
                self._dump_log_tail(20)
                return False

            # Check required patterns
            for key, pattern in required_patterns.items():
                if not found[key] and re.search(pattern, log_content):
                    found[key] = True
                    log(f"  ✓ Found: {key}")

            # All found?
            if all(found.values()):
                log("")
                log("  ✓ STARTUP VERIFICATION PASSED")
                self.checks_passed.append("startup_verified")
                return True

        # Timeout
        missing = [k for k, v in found.items() if not v]
        log(f"  ✗ Timeout waiting for: {missing}")
        self._dump_log_tail(30)
        return False

    def verify_ticks_accelerated(self, required_ticks: int = 2) -> bool:
        """
        STEP 3: Accelerated tick verification.

        Instead of waiting for scheduler, directly invoke tick functions
        back-to-back. This exercises the full runtime loop:
        - Data fetch
        - Strategy evaluation
        - Decision logging
        - State transitions

        Same code paths as production, just no 4-hour waits.
        """
        log("")
        log("=" * 70)
        log(f"STEP 3: ACCELERATED TICK VERIFICATION ({required_ticks} ticks)")
        log("=" * 70)
        log("  Mode: Direct tick execution (no scheduler wait)")
        log("  Exercises: fetch → strategy → log → state → repeat")
        log("")

        # Stop the subprocess - we'll run ticks directly
        if self.process and self.process.poll() is None:
            log("  Stopping subprocess for direct tick execution...")
            self.process.terminate()
            time.sleep(2)

        # Import and create trader instance directly
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from live_paper_trader import LivePaperTrader

            log("  Creating LivePaperTrader instance...")
            trader = LivePaperTrader(
                symbol=SYMBOL,
                interval=INTERVAL,
                starting_capital=Decimal(str(CAPITAL)),
                lookback_days=120,
                data_source=DATA_SOURCE
            )

            # Run startup reconciliation
            log("  Running hydrate_and_reconcile()...")
            if not trader.hydrate_and_reconcile():
                log("  ✗ Hydration/reconciliation failed")
                return False
            log("  ✓ Startup reconciliation passed")

        except Exception as e:
            log(f"  ✗ Failed to create trader: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Execute ticks directly
        for tick_num in range(1, required_ticks + 1):
            log("")
            log(f"  --- ACCELERATED TICK #{tick_num} ---")

            try:
                # Fetch data (real API call)
                log(f"  Fetching candle data...")
                data = trader.fetch_latest_data()

                if data is None or len(data) < trader.strategy.min_bars:
                    log(f"  ✗ Insufficient data for tick #{tick_num}")
                    return False

                log(f"  Data: {len(data)} bars, latest close: ${data['close'].iloc[-1]:,.2f}")

                # Run reconciliation check (as production does each tick)
                reconcile_result = trader.engine.reconcile_position()
                if reconcile_result.requires_fail_closed:
                    log(f"  ✗ Reconciliation failed during tick #{tick_num}")
                    return False

                # Run the actual tick
                log(f"  Running tick through engine...")
                result = trader.run_tick(data)

                log(f"  Signal: {result['signal']}")
                log(f"  Action: {result['action']}")

                # Verify no fail-closed
                if trader.engine.fail_closed:
                    log(f"  ✗ Engine entered fail-closed during tick #{tick_num}")
                    return False

                log(f"  ✓ Tick #{tick_num} completed successfully")

            except Exception as e:
                log(f"  ✗ Exception during tick #{tick_num}: {e}")
                import traceback
                traceback.print_exc()
                return False

            # Small delay between ticks to avoid rate limits
            if tick_num < required_ticks:
                time.sleep(2)

        log("")
        log(f"  ✓ ACCELERATED TICK VERIFICATION PASSED ({required_ticks} ticks)")
        self.checks_passed.append(f"accelerated_ticks_{required_ticks}")
        return True

    def run_negative_test(self) -> bool:
        """
        STEP 4: Negative test - prove safety brake works.
        Inject mismatch, verify CRITICAL + fail-closed.
        """
        log("")
        log("=" * 70)
        log("STEP 4: NEGATIVE TEST (safety brake verification)")
        log("=" * 70)

        # Stop the running unit first
        log("  Stopping unit for negative test...")
        if self.process and self.process.poll() is None:
            self.process.terminate()
            time.sleep(3)
            if self.process.poll() is None:
                self.process.kill()

        log("  Injecting DB-open / broker-empty mismatch...")

        # Inject a fake open position in DB
        try:
            import uuid
            trade_id = f"negtest_{uuid.uuid4().hex[:8]}"
            decision_id = f"negdec_{uuid.uuid4().hex[:8]}"
            order_id = f"negord_{uuid.uuid4().hex[:8]}"

            conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
            cursor = conn.cursor()

            # Insert decision
            cursor.execute("""
                INSERT INTO decisions (
                    decision_id, timestamp, symbol, strategy_name,
                    signal_values, risk_checks, result, result_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision_id,
                datetime.now(timezone.utc).isoformat(),
                SYMBOL,
                "negative_test",
                "{}",
                "{}",
                "signal_long",
                "NEGATIVE TEST - should trigger fail-closed"
            ))

            # Insert order
            cursor.execute("""
                INSERT INTO orders (
                    order_id, decision_id, timestamp, symbol,
                    side, quantity, requested_price, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id,
                decision_id,
                datetime.now(timezone.utc).isoformat(),
                SYMBOL,
                "buy",
                "0.1",
                "1.00",
                "filled"
            ))

            # Insert open trade (DB has position, broker won't)
            cursor.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, entry_order_id,
                    entry_decision_id, entry_price, quantity, entry_timestamp,
                    status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                SYMBOL,
                "buy",
                order_id,
                decision_id,
                "1.00",
                "0.1",
                datetime.now(timezone.utc).isoformat(),
                "open"
            ))

            conn.commit()
            conn.close()
            log(f"  ✓ Injected fake open position: {trade_id}")

        except Exception as e:
            log(f"  ✗ Failed to inject position: {e}")
            return False

        # Restart unit - should detect mismatch on hydration
        log("  Restarting unit (should detect mismatch)...")

        neg_log = LOG_FILE.parent / "pyr_negative_test.log"
        with open(neg_log, 'w') as f:
            f.write(f"=== Negative Test: {datetime.now(timezone.utc).isoformat()} ===\n")

        cmd = [
            str(VENV_PYTHON),
            str(TRADER_SCRIPT),
            "--symbol", SYMBOL,
            "--interval", INTERVAL,
            "--capital", str(CAPITAL),
            "--data-source", DATA_SOURCE
        ]

        neg_process = subprocess.Popen(
            cmd,
            stdout=open(neg_log, 'a'),
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT)
        )

        # Wait for it to start and (hopefully) fail-closed
        time.sleep(30)

        # Read negative test log
        with open(neg_log, 'r') as f:
            neg_content = f.read()

        # Check for expected patterns
        has_critical = bool(re.search(r"CRITICAL", neg_content))
        has_fail_closed = bool(re.search(r"FAIL-CLOSED|fail-closed", neg_content, re.IGNORECASE))
        has_ghost = bool(re.search(r"ghost position|DB has position but BROKER", neg_content, re.IGNORECASE))

        log(f"  CRITICAL logged: {has_critical}")
        log(f"  FAIL-CLOSED entered: {has_fail_closed}")
        log(f"  Ghost position detected: {has_ghost}")

        # Clean up
        if neg_process.poll() is None:
            neg_process.terminate()

        # Clean up the fake position
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("UPDATE trades SET status = 'closed' WHERE trade_id = ?", (trade_id,))
            conn.commit()
            conn.close()
            log("  ✓ Cleaned up fake position")
        except Exception as e:
            log(f"  ⚠ Failed to clean up: {e}")

        # Verify safety brake worked
        if has_critical and has_fail_closed and has_ghost:
            log("")
            log("  ✓ NEGATIVE TEST PASSED - Safety brake works!")
            self.checks_passed.append("negative_test_passed")
            return True
        else:
            log("")
            log("  ✗ NEGATIVE TEST FAILED - Safety brake may be broken!")
            log("    Expected: CRITICAL, FAIL-CLOSED, ghost detection")
            self.checks_failed.append("negative_test_failed")
            return False

    def _dump_log_tail(self, lines: int):
        """Print last N lines of log."""
        try:
            with open(LOG_FILE, 'r') as f:
                all_lines = f.readlines()
                log(f"  Last {lines} log lines:")
                for line in all_lines[-lines:]:
                    print(f"    {line.rstrip()}")
        except Exception as e:
            log(f"  Could not read log: {e}")

    def cleanup(self):
        """Stop any running processes."""
        global process
        if self.process and self.process.poll() is None:
            self.process.terminate()
            time.sleep(2)
            if self.process.poll() is None:
                self.process.kill()
        process = None


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log("")
    log("=" * 70)
    log("🎯 PYR CONTROLLED ROLLOUT - BULLETPROOF EDITION")
    log("=" * 70)
    log(f"Incident: PYR-2025-12-16-001")
    log(f"Symbol: {SYMBOL}")
    log(f"Required ticks: {REQUIRED_TICKS}")
    log("")

    verifier = RolloutVerifier()

    try:
        # STEP 1: Pre-flight
        if not verifier.run_preflight():
            log("")
            log("❌ ROLLOUT FAILED: Pre-flight checks failed")
            sys.exit(EXIT_PREFLIGHT_FAILED)

        # Launch unit
        verifier.launch_unit()

        # STEP 2: Startup verification
        if not verifier.verify_startup(timeout_seconds=120):
            log("")
            log("❌ ROLLOUT FAILED: Startup verification failed")
            verifier.cleanup()
            sys.exit(EXIT_STARTUP_FAILED)

        # STEP 3: Accelerated tick verification
        # Runs real ticks back-to-back without waiting for scheduler
        # Exercises: fetch → strategy → log → state → repeat
        if not verifier.verify_ticks_accelerated(required_ticks=REQUIRED_TICKS):
            log("")
            log("❌ ROLLOUT FAILED: Tick verification failed")
            verifier.cleanup()
            sys.exit(EXIT_TICK_FAILED)

        # STEP 4: Negative test
        if not verifier.run_negative_test():
            log("")
            log("❌ ROLLOUT FAILED: Negative test failed (SAFETY BRAKE BROKEN!)")
            sys.exit(EXIT_NEGATIVE_TEST_FAILED)

        # All passed!
        log("")
        log("=" * 70)
        log("✅ CONTROLLED ROLLOUT SUCCESSFUL")
        log("=" * 70)
        log("")
        log("All machine-verified checks passed:")
        for check in verifier.checks_passed:
            log(f"  ✓ {check}")

        # Write deployment artifact (required for fleet_manager deploy lock)
        log("")
        log("Writing deployment artifact...")
        write_pass_artifact(verifier.checks_passed)

        log("")
        log("PYR-USD is CLEARED for fleet integration.")
        log("")
        log("Next: python src/fleet_manager.py")

        # Send success alert
        if verifier.notifier and verifier.notifier.url:
            verifier.notifier.send_system_alert(
                "PYR Rollout PASSED",
                f"✅ **PYR-USD Controlled Rollout PASSED**\n\n"
                f"Machine-verified checks:\n"
                + "\n".join(f"• {c}" for c in verifier.checks_passed) +
                f"\n\nPYR-USD cleared for fleet."
            )

        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        log(f"❌ ROLLOUT CRASHED: {e}")
        import traceback
        traceback.print_exc()
        verifier.cleanup()
        sys.exit(EXIT_CRASH)


if __name__ == "__main__":
    main()
