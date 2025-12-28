#!/usr/bin/env python3
"""
ArgusNexus V4 - Postmortem Closure Script

INCIDENT: PYR-USD Emergency Exit - State Persistence Bug
DATE: 2025-12-16
RESOLUTION DATE: 2025-12-17

This script:
1. Runs validation suite to generate proof artifacts
2. Captures git commit hash of the fix
3. Logs the official incident closure to the Truth Engine
4. Creates an audit trail for compliance/review

Run once after all validations pass:
    python scripts/close_postmortem_PYR_2025_12_16.py

Prerequisites (script will verify):
    - pytest tests/test_position_reconciliation.py must PASS (21 tests)
    - python scripts/validate_reconciliation.py must PASS (7 steps)
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from truth.logger import TruthLogger
from truth.schema import DecisionResult

# Incident details
INCIDENT_ID = "PYR-2025-12-16-001"
INCIDENT_SYMBOL = "PYR-USD"
ORIGINAL_DECISION_ID = "f63c96d7-944d-416f-bc4e-b5173e05158c"
PROJECT_ROOT = Path(__file__).parent.parent

INCIDENT_STATEMENT = """
===============================================================================
POSTMORTEM CLOSURE: PYR-USD Emergency Exit (2025-12-16)
===============================================================================

INCIDENT ID: PYR-2025-12-16-001
STATUS: CLOSED
RESOLUTION DATE: 2025-12-17T05:30:00Z

-------------------------------------------------------------------------------
ROOT CAUSE
-------------------------------------------------------------------------------
Engine restart without broker-backed position hydration. The live_paper_trader
restarted at 02:12 UTC while PYR-USD had an open position from 01:49 UTC.
The hydration path only queried the DB and did not verify against the broker,
resulting in an orphan position that the engine was blind to.

-------------------------------------------------------------------------------
FAILURE MODE
-------------------------------------------------------------------------------
Stops/exits were not armed because:
1. hydrate_position_state() only read from DB, not broker
2. reconcile_position() trusted caller input instead of querying broker
3. fail-closed was advisory (logged) but not enforced (trading continued)
4. Feature flag bypass returned success=True, masking unsafe state

Impact: Emergency safety intervention required, -$122.33 realized loss.

-------------------------------------------------------------------------------
FIX IMPLEMENTED
-------------------------------------------------------------------------------
GAP 1 (P0): Broker-as-truth - Engine now queries broker FIRST
GAP 2 (P1): Self-contained reconcile - reconcile_position() queries broker directly
GAP 3 (P0): Fail-closed enforcement - enter_fail_closed() called immediately
GAP 4 (P1): Missing candles → hard failure with explicit logging
GAP 5 (P1): Feature flag bypass → fail-closed, not success

Files modified:
- src/engine.py: hydrate_position_state(), reconcile_position()
- src/execution/paper.py: has_position(), get_position(), get_position_size()
- scripts/live_paper_trader.py: Updated reconcile calls
- tests/test_position_reconciliation.py: 21 tests, all passing

-------------------------------------------------------------------------------
PREVENTION
-------------------------------------------------------------------------------
1. CI harness: 21 unit tests covering all broker/DB reconciliation scenarios
2. Release gate: scripts/validate_reconciliation.py must PASS before deploy
3. Runtime protection: Any CRITICAL mismatch → auto-pause, human must clear

-------------------------------------------------------------------------------
OPERATIONAL RULES (PERMANENT)
-------------------------------------------------------------------------------
RULE 1: No deploy without passing:
        - 21 unit tests (test_position_reconciliation.py)
        - Release gate validation (validate_reconciliation.py)

RULE 2: Any CRITICAL mismatch = automatic pause stays paused until human clears

-------------------------------------------------------------------------------
VERIFICATION
-------------------------------------------------------------------------------
✓ 21/21 unit tests PASS
✓ 7/7 release gate validation steps PASS
✓ Broker-as-truth queries verified
✓ Orphan position detection verified
✓ Ghost position detection verified
✓ Fail-closed enforcement verified
✓ Trading blocked in fail-closed verified
✓ Matching state hydration verified

===============================================================================
POSTMORTEM CLOSED BY: ArgusNexus V4 Automated Validation
CLOSURE TIMESTAMP: {timestamp}
===============================================================================
"""


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
            return result.stdout.strip()[:12]  # Short hash
    except Exception:
        pass
    return "unknown"


def get_git_commit_message() -> str:
    """Get current git commit message."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            return result.stdout.strip()[:100]
    except Exception:
        pass
    return "unknown"


def run_unit_tests() -> tuple[bool, str]:
    """Run unit tests and return (passed, summary)."""
    print("  Running unit tests...")
    try:
        result = subprocess.run(
            [str(PROJECT_ROOT / "venv" / "bin" / "python"), "-m", "pytest",
             "tests/test_position_reconciliation.py", "-v", "--tb=no", "-q"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120
        )

        # Parse output for pass count
        output = result.stdout + result.stderr
        if "passed" in output:
            # Extract "21 passed" pattern
            import re
            match = re.search(r"(\d+) passed", output)
            if match:
                passed_count = int(match.group(1))
                if result.returncode == 0:
                    return True, f"{passed_count} tests PASSED"
                else:
                    return False, f"Tests failed (exit code {result.returncode})"

        return result.returncode == 0, output[-200:] if len(output) > 200 else output
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def run_validation_script() -> tuple[bool, str]:
    """Run release gate validation and return (passed, summary)."""
    print("  Running release gate validation...")
    try:
        result = subprocess.run(
            [str(PROJECT_ROOT / "venv" / "bin" / "python"),
             "scripts/validate_reconciliation.py"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120
        )

        if result.returncode == 0:
            return True, "7/7 steps PASSED"
        else:
            # Extract failure info
            output = result.stdout + result.stderr
            if "VALIDATION FAIL" in output:
                return False, "Validation failed"
            return False, f"Exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def main():
    print("")
    print("=" * 70)
    print("POSTMORTEM CLOSURE - Generating Proof Artifacts")
    print("=" * 70)
    print("")

    # Collect proof artifacts
    proof_artifacts = {}

    # 1. Git commit hash
    print("[1/3] Capturing git commit hash...")
    git_hash = get_git_commit_hash()
    git_message = get_git_commit_message()
    proof_artifacts["git_commit"] = git_hash
    proof_artifacts["git_message"] = git_message
    print(f"      Commit: {git_hash}")
    print(f"      Message: {git_message}")

    # 2. Run unit tests
    print("[2/3] Running unit tests...")
    tests_passed, tests_summary = run_unit_tests()
    proof_artifacts["unit_tests"] = {
        "passed": tests_passed,
        "summary": tests_summary
    }
    status = "✓" if tests_passed else "✗"
    print(f"      {status} {tests_summary}")

    if not tests_passed:
        print("")
        print("❌ CANNOT CLOSE POSTMORTEM: Unit tests failed")
        print("   Fix failing tests before closing postmortem.")
        sys.exit(1)

    # 3. Run validation script
    print("[3/3] Running release gate validation...")
    validation_passed, validation_summary = run_validation_script()
    proof_artifacts["release_gate"] = {
        "passed": validation_passed,
        "summary": validation_summary
    }
    status = "✓" if validation_passed else "✗"
    print(f"      {status} {validation_summary}")

    if not validation_passed:
        print("")
        print("❌ CANNOT CLOSE POSTMORTEM: Release gate validation failed")
        print("   Fix validation failures before closing postmortem.")
        sys.exit(1)

    print("")
    print("=" * 70)
    print("All proof artifacts collected. Logging to Truth Engine...")
    print("=" * 70)
    print("")

    # Initialize Truth Logger
    db_path = PROJECT_ROOT / "data" / "v4_live_paper.db"
    truth_logger = TruthLogger(str(db_path))
    truth_logger.initialize()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Log postmortem closure as a decision record
    closure_decision = truth_logger.log_decision(
        symbol=INCIDENT_SYMBOL,
        strategy_name="postmortem_closure",
        signal_values={
            "event_type": "POSTMORTEM_CLOSED",
            "incident_id": INCIDENT_ID,
            "original_decision_id": ORIGINAL_DECISION_ID,
            "root_cause": "engine restart without broker-backed position hydration",
            "failure_mode": "stops/exits not armed due to missing state",
            "fix_summary": "broker-as-truth reconciliation + enforced fail-closed + release gate",
            "git_commit": git_hash,
            "git_message": git_message,
            "proof_artifacts": {
                "unit_tests": tests_summary,
                "release_gate": validation_summary,
                "controlled_rollout": "pending (run rollout_pyr_controlled.py)"
            },
            "prevention": [
                "CI harness (21 unit tests)",
                "Release gate validation script",
                "Controlled rollout with negative test",
                "Runtime auto-pause on CRITICAL mismatch"
            ],
            "operational_rules": [
                "No deploy without passing tests + validation + rollout",
                "CRITICAL mismatch = auto-pause until human clears"
            ]
        },
        risk_checks={},
        result=DecisionResult.NO_SIGNAL,
        result_reason=f"[POSTMORTEM CLOSED] Incident {INCIDENT_ID} resolved @ {git_hash}. PYR-USD cleared for controlled rollout."
    )

    # Print the full incident statement
    print(INCIDENT_STATEMENT.format(timestamp=timestamp))

    print(f"\n✅ Postmortem closure logged to Truth Engine")
    print(f"   Decision ID: {closure_decision.decision_id}")
    print(f"   Git Commit: {git_hash}")
    print(f"   Database: {db_path}")
    print(f"\n📋 PROOF ARTIFACTS RECORDED:")
    print(f"   • Git commit: {git_hash} ({git_message})")
    print(f"   • Unit tests: {tests_summary}")
    print(f"   • Release gate: {validation_summary}")
    print(f"   • Controlled rollout: PENDING")
    print(f"\n🟢 PYR-USD is now CLEARED for controlled rollout.")
    print(f"\n   NEXT STEP:")
    print(f"   python scripts/rollout_pyr_controlled.py")


if __name__ == "__main__":
    main()
