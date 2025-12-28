#!/usr/bin/env python3
"""
Paper Trading Burn-In Script

Runs the hardened Portfolio Manager with:
- Full state persistence (survives restarts)
- Daily snapshot logging
- DD threshold alerting
- Atomic state writes

CRITICAL: This script maintains stateful behavior across restarts.
- PortfolioState is persisted to JSON after every evaluation
- HWM, recovery_mode, bars_in_critical, last_rebalance_at all preserved
- On restart, state is hydrated from disk - no loss of DD tracking

Usage:
    python scripts/paper_trade_burnin.py

    # Or as a systemd service:
    # [Unit]
    # Description=ArgusNexus Portfolio Burn-In
    # [Service]
    # ExecStart=/path/to/venv/bin/python /path/to/scripts/paper_trade_burnin.py
    # Restart=always
"""

import json
import os
import sys
import time
import logging
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio.portfolio_manager import PortfolioManager, PortfolioState, DDState
from src.portfolio.alerts import AlertManager, AlertLevel
from src.data.loader import fetch_coinbase_data

# =============================================================================
# Configuration
# =============================================================================

STATE_PATH = Path("runtime/paper_state.json")
SNAPSHOT_LOG_PATH = Path("runtime/daily_snapshots.jsonl")
ALERT_LOG_PATH = Path("runtime/alerts.jsonl")
INITIAL_CAPITAL = Decimal("500")
EVALUATION_INTERVAL_HOURS = 24  # Once per day
SYMBOL = "BTC-USD"

# Fee structure (Gemini paper trading)
FEES = {
    "entry": 0.004,   # 0.4% entry fee
    "exit": 0.002,    # 0.2% exit fee
    "slippage": 0.001 # 0.1% slippage
}

# Alert thresholds
DD_ALERT_INFO = Decimal("10.0")      # DD >= 10% - heads up
DD_ALERT_WARNING = Decimal("12.0")   # DD >= 12% - warning engaged
DD_ALERT_CRITICAL = Decimal("20.0")  # DD >= 20% - flattened

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("runtime/burnin.log")
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# State Persistence
# =============================================================================

def ensure_runtime_dir():
    """Ensure runtime directory exists."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: dict):
    """
    Atomically write JSON to disk.

    Uses tmp file + rename to prevent corruption on crash.
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def load_state() -> Optional[PortfolioState]:
    """
    Load persisted state from disk.

    Returns None if no state file exists (first run).
    """
    if not STATE_PATH.exists():
        logger.info("No persisted state found - starting fresh")
        return None

    try:
        data = json.loads(STATE_PATH.read_text())
        state = PortfolioState.from_dict(data)
        logger.info(f"Loaded persisted state: equity=${float(state.total_equity):.2f}, "
                   f"DD_state={state.dd_state.value}, bars_in_critical={state.bars_in_critical}")
        return state
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return None


def save_state(state: PortfolioState):
    """Persist state to disk atomically."""
    atomic_write_json(STATE_PATH, state.to_dict())
    logger.debug(f"State persisted: equity=${float(state.total_equity):.2f}")


def create_default_state() -> PortfolioState:
    """Create default initial state."""
    return PortfolioState(
        total_equity=INITIAL_CAPITAL,
        btc_qty=Decimal("0"),
        cash=INITIAL_CAPITAL,
        high_water_mark=INITIAL_CAPITAL,
        dd_state=DDState.NORMAL,
        recovery_mode=False,
        bars_in_critical=0,
        sleeve_in_position=False,
        sleeve_entry_price=None,
        last_rebalance_time=None,
        last_update=None,
    )

# =============================================================================
# Snapshot Logging
# =============================================================================

def log_daily_snapshot(
    state: PortfolioState,
    order_context: Dict[str, Any],
    btc_price: Decimal,
    action: str
):
    """
    Append daily snapshot to JSONL log.

    This log is append-only and provides a complete audit trail.
    """
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "equity": float(state.total_equity),
        "btc_qty": float(state.btc_qty),
        "cash": float(state.cash),
        "high_water_mark": float(state.high_water_mark),
        "dd_state": state.dd_state.value,
        "recovery_mode": state.recovery_mode,
        "bars_in_critical": state.bars_in_critical,
        "sleeve_in_position": state.sleeve_in_position,
        "btc_price": float(btc_price),
        "action": action,
        "context": order_context
    }

    with SNAPSHOT_LOG_PATH.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")


# =============================================================================
# Alerting (uses AlertManager)
# =============================================================================

# Global alert manager instance
alert_manager = AlertManager(alert_log_path=ALERT_LOG_PATH)

# =============================================================================
# Main Evaluation Loop
# =============================================================================

def get_current_price(df) -> Decimal:
    """Get current BTC price from latest bar."""
    return Decimal(str(df.iloc[-1]["close"]))


def run_evaluation():
    """
    Run a single portfolio evaluation cycle.

    This is the main loop body - called once per day.
    """
    logger.info("=" * 60)
    logger.info("Starting portfolio evaluation")

    # 1) Load or create state
    state = load_state()
    if state is None:
        state = create_default_state()
        save_state(state)
        logger.info(f"Created initial state: ${float(INITIAL_CAPITAL)}")

    prev_dd_state = state.dd_state
    prev_recovery_mode = state.recovery_mode

    # 2) Fetch latest market data (need ~250 bars for indicators)
    try:
        df, report = fetch_coinbase_data(
            symbol=SYMBOL,
            lookback_days=300,  # Extra buffer for indicators
            interval="1d",
            verbose=False
        )
        logger.info(f"Fetched {len(df)} daily bars, latest: {df.index[-1] if hasattr(df, 'index') else 'N/A'}")
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        return

    current_price = get_current_price(df)
    logger.info(f"Current BTC price: ${float(current_price):,.2f}")

    # 3) Initialize Portfolio Manager
    pm = PortfolioManager()

    # 4) Evaluate portfolio
    eval_time = datetime.now(timezone.utc)
    order, new_state = pm.evaluate(
        df=df,
        state=state,
        current_price=current_price,
        timestamp=eval_time
    )

    logger.info(f"Evaluation result: {order.action}")
    logger.info(f"  Target allocation: {float(order.target_alloc_pct):.1f}%")
    logger.info(f"  Current DD: {order.context.get('current_dd', 0):.1f}%")
    logger.info(f"  DD state: {order.context.get('dd_state', 'unknown')}")
    logger.info(f"  Regime: {order.context.get('core_regime', 'unknown')}")

    # 5) Execute order (paper)
    if order.action != "HOLD":
        executed_state = pm.execute_order(order, new_state, current_price, FEES)
        logger.info(f"Executed {order.action}: BTC delta={float(order.btc_qty_delta):.6f}")
        logger.info(f"  New BTC qty: {float(executed_state.btc_qty):.6f}")
        logger.info(f"  New cash: ${float(executed_state.cash):.2f}")
    else:
        executed_state = new_state
        logger.info(f"HOLD: {order.reason}")

    # 6) Update equity
    btc_value = executed_state.btc_qty * current_price
    executed_state.total_equity = executed_state.cash + btc_value
    logger.info(f"Portfolio value: ${float(executed_state.total_equity):.2f}")

    # 7) Check alerts
    current_dd = Decimal(str(order.context.get("current_dd", 0)))
    alert_manager.check_dd_alerts(
        current_dd=current_dd,
        dd_state=executed_state.dd_state,
        prev_dd_state=prev_dd_state,
        recovery_mode=executed_state.recovery_mode,
        prev_recovery_mode=prev_recovery_mode,
        context=order.context
    )

    # 8) Persist state
    save_state(executed_state)

    # 9) Log daily snapshot
    log_daily_snapshot(
        state=executed_state,
        order_context=order.context,
        btc_price=current_price,
        action=order.action
    )

    logger.info("Evaluation complete")
    logger.info("=" * 60)


def calculate_next_run_time() -> datetime:
    """
    Calculate next daily close evaluation time.

    Runs at 00:05 UTC (5 minutes after daily close).
    """
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=0, minute=5, second=0, microsecond=0)

    if next_run <= now:
        next_run += timedelta(days=1)

    return next_run


def main():
    """
    Main entry point for paper trading burn-in.

    Runs continuously, evaluating once per day after daily close.
    """
    ensure_runtime_dir()

    logger.info("=" * 60)
    logger.info("ArgusNexus Portfolio Burn-In Starting")
    logger.info(f"Initial capital: ${float(INITIAL_CAPITAL)}")
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"State path: {STATE_PATH}")
    logger.info("=" * 60)

    # Run once immediately on startup
    run_evaluation()

    # Then run on schedule
    while True:
        next_run = calculate_next_run_time()
        sleep_seconds = (next_run - datetime.now(timezone.utc)).total_seconds()

        logger.info(f"Next evaluation at {next_run.isoformat()} ({sleep_seconds/3600:.1f}h from now)")

        # Sleep in chunks to allow graceful shutdown
        while sleep_seconds > 0:
            chunk = min(sleep_seconds, 300)  # Wake every 5 minutes to check
            time.sleep(chunk)
            sleep_seconds -= chunk

        run_evaluation()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutdown requested - saving state and exiting")
        sys.exit(0)
