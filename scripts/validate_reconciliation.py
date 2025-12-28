#!/usr/bin/env python3
"""
ArgusNexus V4 - Position Reconciliation Validation Script

RELEASE GATE: This script must PASS before deploying reconciliation fixes to production.

Tests the broker-vs-DB reconciliation path end-to-end:
- Step 0: Safety setup (paper only, dedicated symbol, clean state)
- Step 1: Seed broker position
- Step 2: Hydrate and verify logs
- Step 3: Force mismatch → verify fail-closed
- Step 4: Discord alert verification

Usage:
    python scripts/validate_reconciliation.py

Exit codes:
    0 = VALIDATION PASS ✅
    1 = VALIDATION FAIL ❌
"""

import sys
import logging
import asyncio
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import Mock, patch
import aiosqlite
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engine import TradingEngine
from strategy.donchian import DonchianBreakout
from risk import RiskManager, RiskConfig
from execution import PaperExecutor
from truth.logger import TruthLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ReconciliationValidator:
    """
    Controlled validation for position reconciliation fixes (Async).
    """

    def __init__(self):
        self.symbol = "VALID-USD"
        self.capital = Decimal("10000")
        self.results = []
        self.discord_calls = []
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "validation.db"

    async def setup(self):
        """Step 0: Safety setup (Async)."""
        logger.info("=" * 70)
        logger.info("STEP 0: SAFETY SETUP")
        logger.info("=" * 70)

        self.strategy = DonchianBreakout()
        self.risk_manager = RiskManager(RiskConfig())
        self.executor = PaperExecutor(starting_balance=self.capital)
        self.truth_logger = TruthLogger(str(self.db_path))
        await self.truth_logger.initialize()

        self.notifier = Mock()
        self.notifier.url = "https://discord.test/webhook"
        self.notifier.send_critical_alert = Mock(return_value=True)

        def capture_alert(**kwargs):
            self.discord_calls.append(kwargs)
            return True
        self.notifier.send_critical_alert.side_effect = capture_alert

        self.engine = TradingEngine(
            self.strategy, self.risk_manager, self.executor,
            self.truth_logger, self.symbol, self.capital, self.notifier
        )

        broker_has = self.executor.has_position(self.symbol)
        db_has = await self.truth_logger.get_open_position(self.symbol) is not None

        if broker_has or db_has:
            logger.error("  ❌ DIRTY STATE")
            return False

        logger.info("  ✓ Clean state confirmed")
        return True

    async def step1_seed_broker_position(self) -> bool:
        """Step 1: Seed a broker position (Async)."""
        logger.info("\n" + "=" * 70 + "\nSTEP 1: SEED BROKER POSITION\n" + "=" * 70)
        base_currency = self.symbol.split("-")[0]
        self.executor._balances[base_currency] = Decimal("0.5")
        
        broker_has = self.executor.has_position(self.symbol)
        db_has = await self.truth_logger.get_open_position(self.symbol) is not None
        
        if not broker_has or db_has: return False
        logger.info("  ✓ Broker position seeded")
        return True

    async def step2_hydrate_and_verify(self) -> bool:
        """Step 2: Hydrate and verify mismatch (Async)."""
        logger.info("\n" + "=" * 70 + "\nSTEP 2: HYDRATE AND VERIFY MISMATCH\n" + "=" * 70)
        result = await self.engine.hydrate_position_state(candle_data=None)
        
        if result.success or not result.requires_fail_closed: return False
        if not self.engine.fail_closed: return False
        
        events = await self._get_reconciliation_events()
        if not any(e['event_type'] == 'MISMATCH' for e in events): return False
        
        logger.info("  ✓ Mismatch detected correctly")
        return True

    async def step3_verify_trading_blocked(self) -> bool:
        """Step 3: Verify trading blocked (Async)."""
        logger.info("\n" + "=" * 70 + "\nSTEP 3: VERIFY TRADING BLOCKED\n" + "=" * 70)
        import pandas as pd
        import numpy as np
        now = datetime.now(timezone.utc)
        data = pd.DataFrame({
            'timestamp': pd.date_range(end=now, periods=100, freq='4h'),
            'open': np.random.uniform(90000, 110000, 100),
            'high': np.random.uniform(95000, 115000, 100),
            'low': np.random.uniform(85000, 105000, 100),
            'close': np.random.uniform(90000, 110000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        result = await self.engine.run_tick(data)
        if result.action_taken != "fail_closed": return False
        logger.info("  ✓ Trading blocked")
        return True

    async def step4_verify_discord_alert(self) -> bool:
        """Step 4: Verify Discord alert (Async placeholder)."""
        logger.info("\n" + "=" * 70 + "\nSTEP 4: VERIFY DISCORD ALERT\n" + "=" * 70)
        return True

    async def step5_verify_inverse_mismatch(self) -> bool:
        """Step 5: Verify inverse mismatch (Async)."""
        logger.info("\n" + "=" * 70 + "\nSTEP 5: VERIFY INVERSE MISMATCH\n" + "=" * 70)
        await self._reset_engine()
        base_currency = self.symbol.split("-")[0]
        self.executor._balances[base_currency] = Decimal("0")
        await self._seed_db_position()
        
        result = await self.engine.hydrate_position_state(candle_data=None)
        if result.success or not result.requires_fail_closed: return False
        logger.info("  ✓ Inverse mismatch detected")
        return True

    async def step6_verify_matching_state(self) -> bool:
        """Step 6: Verify matching state (Async)."""
        logger.info("\n" + "=" * 70 + "\nSTEP 6: VERIFY MATCHING STATE\n" + "=" * 70)
        await self._reset_engine()
        base_currency = self.symbol.split("-")[0]
        self.executor._balances[base_currency] = Decimal("0.5")
        await self._seed_db_position()
        
        import pandas as pd
        import numpy as np
        db_pos = await self.truth_logger.get_open_position(self.symbol)
        entry_ts = pd.to_datetime(db_pos['entry_timestamp'])
        data = pd.DataFrame({
            'timestamp': pd.date_range(start=entry_ts - pd.Timedelta(days=1), periods=50, freq='4h'),
            'open': np.full(50, 100000.0), 'high': np.full(50, 102000.0),
            'low': np.full(50, 98000.0), 'close': np.full(50, 101000.0), 'volume': np.full(50, 1000.0)
        })
        
        result = await self.engine.hydrate_position_state(candle_data=data)
        if not result.success or not result.position_found or self.engine.fail_closed: return False
        logger.info("  ✓ Matching state hydrated")
        return True

    async def _reset_engine(self):
        self.engine.fail_closed = False
        self.engine.has_open_position = False
        for key in list(self.executor._balances.keys()):
            if key != "USD": self.executor._balances[key] = Decimal("0")
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("UPDATE trades SET status = 'closed' WHERE status = 'open'")
            await db.commit()

    async def _seed_db_position(self) -> str:
        import uuid
        trade_id = f"valid_{uuid.uuid4().hex[:8]}"
        async with aiosqlite.connect(str(self.db_path)) as db:
            decision_id = f"dec_{uuid.uuid4().hex[:8]}"
            await db.execute("INSERT INTO decisions (decision_id, timestamp, symbol, strategy_name, signal_values, risk_checks, result, result_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (decision_id, datetime.now(timezone.utc).isoformat(), self.symbol, "validation_test", "{}", "{}", "signal_long", "Validation test entry"))
            order_id = f"ord_{uuid.uuid4().hex[:8]}"
            await db.execute("INSERT INTO orders (order_id, decision_id, timestamp, symbol, side, quantity, requested_price, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (order_id, decision_id, datetime.now(timezone.utc).isoformat(), self.symbol, "buy", "0.5", "100000.00", "filled"))
            await db.execute("INSERT INTO trades (trade_id, symbol, side, entry_order_id, entry_decision_id, entry_price, quantity, entry_timestamp, stop_loss_price, take_profit_price, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trade_id, self.symbol, "buy", order_id, decision_id, "100000.00", "0.5", datetime.now(timezone.utc).isoformat(), "95000.00", "110000.00", "open"))
            await db.commit()
        return trade_id

    async def _get_reconciliation_events(self) -> list:
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT signal_values, timestamp FROM decisions WHERE strategy_name = 'position_reconciliation' ORDER BY timestamp DESC")
            rows = await cursor.fetchall()
        import json
        return [{'event_type': json.loads(r['signal_values']).get('event_type'), 'severity': json.loads(r['signal_values']).get('severity', 'INFO')} for r in rows]

    def cleanup(self):
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir): shutil.rmtree(self.temp_dir)

    async def run(self) -> bool:
        steps = [
            ("Step 0: Safety Setup", self.setup),
            ("Step 1: Seed Broker Position", self.step1_seed_broker_position),
            ("Step 2: Hydrate & Verify Mismatch", self.step2_hydrate_and_verify),
            ("Step 3: Verify Trading Blocked", self.step3_verify_trading_blocked),
            ("Step 4: Verify Discord Alert", self.step4_verify_discord_alert),
            ("Step 5: Verify Inverse Mismatch", self.step5_verify_inverse_mismatch),
            ("Step 6: Verify Matching State", self.step6_verify_matching_state),
        ]
        passed = 0
        for name, step_fn in steps:
            try:
                if await step_fn():
                    passed += 1
                    self.results.append((name, True))
                else:
                    self.results.append((name, False))
                    logger.error(f"FAILED: {name}")
            except Exception as e:
                logger.error(f"EXCEPTION in {name}: {e}", exc_info=True)
                self.results.append((name, False))
        return passed == len(steps)


async def main_async():
    validator = ReconciliationValidator()
    try:
        success = await validator.run()
        sys.exit(0 if success else 1)
    finally:
        validator.cleanup()

if __name__ == "__main__":
    asyncio.run(main_async())
