# Validation Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build replay runner for 100+ validated trades (Gate A) and bracket order infrastructure (safety foundation)

**Architecture:** Two parallel workstreams - (1) Replay runner that feeds historical candles through production engine path with run metadata tracking, (2) Bracket executor for atomic TP/SL orders on Coinbase. Both share schema changes.

**Tech Stack:** Python 3.12, SQLite, pandas, Coinbase Advanced Trade API, pytest

**Design Reference:** See `/path/to/ArgusNexus/docs/plans/2025-12-17-validation-infrastructure-design.md` for full design decisions.

---

## Phase 0: Schema Foundation (Shared)

### Task 0.1: Add `runs` Table Migration

**Files:**
- Modify: `src/truth/schema.py:296-462` (SQL_SCHEMA)

**Step 1: Write the failing test**

```python
# tests/unit/test_runs_schema.py
import sqlite3
import tempfile
import pytest
from truth.schema import SQL_SCHEMA

def test_runs_table_exists():
    """runs table should exist in schema."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        conn = sqlite3.connect(f.name)
        conn.executescript(SQL_SCHEMA)

        # Check runs table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        assert cursor.fetchone() is not None, "runs table should exist"

        # Check required columns
        cursor = conn.execute("PRAGMA table_info(runs)")
        columns = {row[1] for row in cursor.fetchall()}
        required = {'run_id', 'run_mode', 'capital_mode', 'market_time_basis',
                    'data_integrity', 'created_by', 'git_sha', 'config_hash',
                    'symbols', 'start_date', 'end_date', 'started_at', 'completed_at',
                    'trades_opened', 'trades_closed', 'notes'}
        assert required.issubset(columns), f"Missing columns: {required - columns}"
        conn.close()
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit/test_runs_schema.py -v`
Expected: FAIL with "runs table should exist"

**Step 3: Write minimal implementation**

Add to `src/truth/schema.py` SQL_SCHEMA (after existing CREATE TABLE statements):

```sql
-- =============================================================================
-- RUNS TABLE: Metadata for replay/paper/live runs
-- =============================================================================
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_mode TEXT NOT NULL,           -- replay_backtest | paper_live
    capital_mode TEXT,                -- independent | portfolio
    market_time_basis TEXT,           -- candle_timestamp | wall_clock
    data_integrity TEXT,              -- ok | degraded_skip_volume
    created_by TEXT,                  -- replay_runner | live_paper_trader
    git_sha TEXT,
    config_hash TEXT,
    symbols TEXT,
    start_date TEXT,
    end_date TEXT,
    started_at TEXT,
    completed_at TEXT,
    trades_opened INTEGER DEFAULT 0,
    trades_closed INTEGER DEFAULT 0,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_mode ON runs(run_mode);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit/test_runs_schema.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_runs_schema.py src/truth/schema.py
git commit -m "feat(schema): add runs table for run metadata tracking"
```

---

### Task 0.2: Add `run_id` FK to Existing Tables

**Files:**
- Modify: `src/truth/schema.py:296-462` (SQL_SCHEMA)

**Step 1: Write the failing test**

```python
# tests/unit/test_run_id_columns.py
import sqlite3
import tempfile
import pytest
from truth.schema import SQL_SCHEMA

def test_run_id_column_on_decisions():
    """decisions table should have nullable run_id column."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        conn = sqlite3.connect(f.name)
        conn.executescript(SQL_SCHEMA)

        cursor = conn.execute("PRAGMA table_info(decisions)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}  # name: type
        assert 'run_id' in columns, "decisions should have run_id column"
        conn.close()

def test_run_id_column_on_orders():
    """orders table should have nullable run_id column."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        conn = sqlite3.connect(f.name)
        conn.executescript(SQL_SCHEMA)

        cursor = conn.execute("PRAGMA table_info(orders)")
        columns = {row[1] for row in cursor.fetchall()}
        assert 'run_id' in columns, "orders should have run_id column"
        conn.close()

def test_run_id_column_on_trades():
    """trades table should have nullable run_id column."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        conn = sqlite3.connect(f.name)
        conn.executescript(SQL_SCHEMA)

        cursor = conn.execute("PRAGMA table_info(trades)")
        columns = {row[1] for row in cursor.fetchall()}
        assert 'run_id' in columns, "trades should have run_id column"
        conn.close()

def test_run_id_indexes_exist():
    """Indexes on run_id should exist for query performance."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        conn = sqlite3.connect(f.name)
        conn.executescript(SQL_SCHEMA)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE '%run_id%'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        assert len(indexes) >= 3, f"Expected 3+ run_id indexes, got: {indexes}"
        conn.close()
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit/test_run_id_columns.py -v`
Expected: FAIL with "decisions should have run_id column"

**Step 3: Write minimal implementation**

Modify `src/truth/schema.py` - Add `run_id TEXT` column to each table:

In decisions table (after `order_id TEXT`):
```sql
    order_id TEXT,                    -- FK to orders if order was placed
    run_id TEXT                       -- FK to runs (nullable for backward compat)
);
```

In orders table (after `error_message TEXT`):
```sql
    error_message TEXT,
    run_id TEXT,                      -- FK to runs (nullable)
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);
```

In trades table (after `is_winner INTEGER`):
```sql
    is_winner INTEGER,                -- NULL for open trades (0 or 1)
    run_id TEXT,                      -- FK to runs (nullable)
    FOREIGN KEY (entry_order_id) REFERENCES orders(order_id),
    ...
);
```

Add indexes at end of index section:
```sql
-- Run tracking indexes
CREATE INDEX IF NOT EXISTS idx_decisions_run_id ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_orders_run_id ON orders(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id);
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit/test_run_id_columns.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_run_id_columns.py src/truth/schema.py
git commit -m "feat(schema): add run_id FK to decisions, orders, trades tables"
```

---

## Phase 1: Replay Runner - Data Layer

### Task 1.1: Create Coinbase Data Loader

**Files:**
- Create: `src/replay/__init__.py`
- Create: `src/replay/data_loader.py`
- Test: `tests/unit/test_replay_data_loader.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_replay_data_loader.py
import pytest
from decimal import Decimal
from datetime import datetime
import pandas as pd

def test_data_loader_import():
    """DataLoader should be importable."""
    from replay.data_loader import DataLoader
    assert DataLoader is not None

def test_data_loader_returns_dataframe():
    """DataLoader.load should return pandas DataFrame."""
    from replay.data_loader import DataLoader

    # Use cache if available, otherwise this is a slow integration test
    loader = DataLoader(cache_dir="data/cache")
    df = loader.load(
        symbol="BTC-USD",
        start="2024-01-01",
        end="2024-01-07",
        granularity="4h"
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume', 'timestamp'])

def test_data_loader_validates_volume():
    """DataLoader should flag rows with volume=0."""
    from replay.data_loader import DataLoader

    loader = DataLoader(cache_dir="data/cache")
    df = loader.load(
        symbol="BTC-USD",
        start="2024-01-01",
        end="2024-01-07",
        granularity="4h"
    )

    # Real Coinbase data should have volume > 0
    zero_volume_rows = (df['volume'] == 0).sum()
    assert zero_volume_rows == 0, f"Found {zero_volume_rows} rows with volume=0"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit/test_replay_data_loader.py::test_data_loader_import -v`
Expected: FAIL with "No module named 'replay'"

**Step 3: Write minimal implementation**

```python
# src/replay/__init__.py
"""Replay Runner - Fast path to 100 validated trades."""
from .data_loader import DataLoader

__all__ = ['DataLoader']
```

```python
# src/replay/data_loader.py
"""
Coinbase Data Loader for Replay Runner

Fetches OHLCV data from Coinbase API with local caching.
CSV files abandoned due to volume=0 data quality issues.
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load historical OHLCV data from Coinbase with caching.

    Primary source: Coinbase API (canonical)
    Cache: data/cache/{symbol}_{start}_{end}_{granularity}.parquet
    """

    # Coinbase granularity mapping (seconds)
    GRANULARITY_MAP = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }

    # Coinbase rate limit: 300 candles per request
    MAX_CANDLES_PER_REQUEST = 300

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data loader.

        Args:
            cache_dir: Directory for cached data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.exchange.coinbase.com"

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        granularity: str = "4h",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load OHLCV data for symbol and date range.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            granularity: Candle size (1m, 5m, 15m, 1h, 4h, 1d)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        cache_file = self._get_cache_path(symbol, start, end, granularity)

        if use_cache and cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            self._validate_data(df, symbol)
            return df

        # Fetch from API
        logger.info(f"Fetching from Coinbase API: {symbol} {start} to {end}")
        df = self._fetch_from_api(symbol, start, end, granularity)

        # Cache for future use
        if use_cache:
            df.to_parquet(cache_file)
            logger.info(f"Cached to: {cache_file}")

        self._validate_data(df, symbol)
        return df

    def _get_cache_path(
        self, symbol: str, start: str, end: str, granularity: str
    ) -> Path:
        """Generate cache file path."""
        safe_symbol = symbol.replace("-", "_").lower()
        filename = f"{safe_symbol}_{start}_{end}_{granularity}.parquet"
        return self.cache_dir / filename

    def _fetch_from_api(
        self, symbol: str, start: str, end: str, granularity: str
    ) -> pd.DataFrame:
        """Fetch data from Coinbase public API."""
        granularity_seconds = self.GRANULARITY_MAP.get(granularity)
        if not granularity_seconds:
            raise ValueError(f"Invalid granularity: {granularity}")

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)  # Include end date

        all_candles = []
        current_start = start_dt

        while current_start < end_dt:
            # Calculate chunk end (max 300 candles)
            chunk_seconds = granularity_seconds * self.MAX_CANDLES_PER_REQUEST
            chunk_end = min(
                current_start + timedelta(seconds=chunk_seconds),
                end_dt
            )

            candles = self._fetch_chunk(
                symbol, current_start, chunk_end, granularity_seconds
            )
            all_candles.extend(candles)

            current_start = chunk_end

        # Convert to DataFrame
        df = pd.DataFrame(all_candles)
        if len(df) == 0:
            raise ValueError(f"No data returned for {symbol} from {start} to {end}")

        df.columns = ['timestamp', 'low', 'high', 'open', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Reorder columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        return df

    def _fetch_chunk(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        granularity: int
    ) -> list:
        """Fetch a single chunk of candles."""
        url = f"{self.base_url}/products/{symbol}/candles"
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "granularity": granularity
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Validate data quality."""
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            pct = zero_volume / len(df) * 100
            logger.warning(
                f"[DATA QUALITY] {symbol}: {zero_volume} rows ({pct:.1f}%) have volume=0"
            )
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit/test_replay_data_loader.py::test_data_loader_import -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/replay/__init__.py src/replay/data_loader.py tests/unit/test_replay_data_loader.py
git commit -m "feat(replay): add Coinbase data loader with caching"
```

---

### Task 1.2: Add `as_of` Timestamp Parameter to Engine

**Files:**
- Modify: `src/engine.py:571-598` (run_tick method signature and timestamp handling)
- Test: `tests/unit/test_engine_as_of.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_engine_as_of.py
import pytest
from datetime import datetime
from decimal import Decimal
import pandas as pd

def test_engine_run_tick_accepts_as_of():
    """Engine.run_tick should accept optional as_of timestamp."""
    from engine import TradingEngine
    from strategy.donchian import DonchianBreakout
    from risk import RiskManager, RiskConfig
    from execution import PaperExecutor
    from truth.logger import TruthLogger
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        strategy = DonchianBreakout()
        risk = RiskManager(RiskConfig())
        executor = PaperExecutor(starting_balance=Decimal("500"))
        logger = TruthLogger(f.name)

        engine = TradingEngine(
            strategy=strategy,
            risk_manager=risk,
            executor=executor,
            truth_logger=logger,
            symbol="BTC-USD",
            capital=Decimal("500")
        )

        # Create minimal test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='4h'),
            'open': [100000.0] * 100,
            'high': [101000.0] * 100,
            'low': [99000.0] * 100,
            'close': [100500.0] * 100,
            'volume': [1000.0] * 100
        })

        historical_time = datetime(2024, 1, 15, 12, 0, 0)

        # Should not raise - as_of is optional
        result = engine.run_tick(df, as_of=historical_time)

        # Verify the timestamp used is the as_of, not wall clock
        assert result.timestamp == historical_time

def test_engine_run_tick_defaults_to_utcnow():
    """Engine.run_tick should default to utcnow if as_of not provided."""
    from engine import TradingEngine
    from strategy.donchian import DonchianBreakout
    from risk import RiskManager, RiskConfig
    from execution import PaperExecutor
    from truth.logger import TruthLogger
    import tempfile
    from datetime import datetime, timedelta

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        strategy = DonchianBreakout()
        risk = RiskManager(RiskConfig())
        executor = PaperExecutor(starting_balance=Decimal("500"))
        logger = TruthLogger(f.name)

        engine = TradingEngine(
            strategy=strategy,
            risk_manager=risk,
            executor=executor,
            truth_logger=logger,
            symbol="BTC-USD",
            capital=Decimal("500")
        )

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='4h'),
            'open': [100000.0] * 100,
            'high': [101000.0] * 100,
            'low': [99000.0] * 100,
            'close': [100500.0] * 100,
            'volume': [1000.0] * 100
        })

        before = datetime.utcnow()
        result = engine.run_tick(df)  # No as_of
        after = datetime.utcnow()

        # Timestamp should be within the call window
        assert before <= result.timestamp <= after
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit/test_engine_as_of.py::test_engine_run_tick_accepts_as_of -v`
Expected: FAIL with "TypeError: run_tick() got an unexpected keyword argument 'as_of'"

**Step 3: Write minimal implementation**

Modify `src/engine.py:571` - Update run_tick signature:

```python
def run_tick(
    self,
    data: pd.DataFrame,
    as_of: Optional[datetime] = None
) -> TickResult:
    """
    Run one tick through the engine.

    THE SEQUENCE (do not change order):
    0. Check fail-closed state - BLOCK if set
    1. Ask the Brain - strategy.evaluate(data)
    2. Check the Conscience - risk.evaluate(trade_request)
    3. Log the Decision - truth.log_decision() ALWAYS
    4. Move the Hands - executor.execute() IF APPROVED
    5. Log the Action - truth.log_order(), truth.log_trade()

    Args:
        data: DataFrame with OHLCV data for analysis
        as_of: Optional timestamp for replay mode (defaults to utcnow)

    Returns:
        TickResult with all outcomes
    """
    timestamp = as_of or datetime.utcnow()
    result = TickResult(
        timestamp=timestamp,
        # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit/test_engine_as_of.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engine.py tests/unit/test_engine_as_of.py
git commit -m "feat(engine): add as_of timestamp parameter for replay mode"
```

---

### Task 1.3: Create Replay Runner Core

**Files:**
- Create: `src/replay/runner.py`
- Test: `tests/unit/test_replay_runner.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_replay_runner.py
import pytest
from decimal import Decimal
import tempfile

def test_replay_runner_import():
    """ReplayRunner should be importable."""
    from replay.runner import ReplayRunner
    assert ReplayRunner is not None

def test_replay_runner_creates_run_record():
    """ReplayRunner should create a run record in the database."""
    from replay.runner import ReplayRunner
    from truth.logger import TruthLogger
    import sqlite3

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        logger = TruthLogger(f.name)
        logger.initialize()

        runner = ReplayRunner(
            db_path=f.name,
            symbols=["BTC-USD"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            capital=Decimal("500"),
            capital_mode="independent"
        )

        run_id = runner.run_id
        assert run_id is not None
        assert run_id.startswith("replay_")

        # Check run record exists in DB
        conn = sqlite3.connect(f.name)
        cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None, "Run record should exist in database"

def test_replay_runner_increments_trade_counts():
    """ReplayRunner should track trades opened/closed counts."""
    from replay.runner import ReplayRunner

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        runner = ReplayRunner(
            db_path=f.name,
            symbols=["BTC-USD"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            capital=Decimal("500")
        )

        # These should be callable
        runner.record_trade_opened()
        runner.record_trade_closed()

        assert runner.trades_opened == 1
        assert runner.trades_closed == 1
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit/test_replay_runner.py::test_replay_runner_import -v`
Expected: FAIL with "cannot import name 'ReplayRunner'"

**Step 3: Write minimal implementation**

```python
# src/replay/runner.py
"""
Replay Runner - Fast path to 100 validated trades.

Key principle: Uses the SAME engine.run_tick() path as live trading.
The only difference is where the candles come from (cache vs websocket).
"""

import hashlib
import logging
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from truth.logger import TruthLogger
from truth.schema import SQL_SCHEMA


logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a replay run."""
    symbols: List[str]
    start_date: str
    end_date: str
    capital: Decimal
    capital_mode: str = "independent"  # independent | portfolio
    granularity: str = "4h"
    skip_volume_filter: bool = False


class ReplayRunner:
    """
    Orchestrates batch replay through the production engine.

    Creates run metadata, manages multiple symbol engines,
    and generates validation reports.
    """

    def __init__(
        self,
        db_path: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        capital: Decimal,
        capital_mode: str = "independent",
        granularity: str = "4h",
        skip_volume_filter: bool = False
    ):
        """
        Initialize replay runner.

        Args:
            db_path: Path to Truth Engine database
            symbols: List of trading pairs to replay
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            capital: Starting capital per symbol (independent mode)
            capital_mode: "independent" or "portfolio"
            granularity: Candle granularity
            skip_volume_filter: If True, tag as degraded_skip_volume
        """
        self.db_path = Path(db_path)
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.capital = capital
        self.capital_mode = capital_mode
        self.granularity = granularity
        self.skip_volume_filter = skip_volume_filter

        # Generate run ID
        self.run_id = f"replay_{uuid.uuid4().hex[:8]}"

        # Track metrics
        self.trades_opened = 0
        self.trades_closed = 0
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # Initialize database and create run record
        self._init_db()
        self._create_run_record()

    def _init_db(self) -> None:
        """Initialize database schema."""
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(SQL_SCHEMA)
        conn.close()

    def _get_git_sha(self) -> Optional[str]:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else None
        except Exception:
            return None

    def _get_config_hash(self) -> str:
        """Generate hash of run configuration."""
        config_str = f"{self.symbols}{self.start_date}{self.end_date}{self.capital}{self.capital_mode}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

    def _create_run_record(self) -> None:
        """Create run record in database."""
        import sqlite3

        self.started_at = datetime.utcnow()
        data_integrity = "degraded_skip_volume" if self.skip_volume_filter else "ok"

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT INTO runs (
                run_id, run_mode, capital_mode, market_time_basis,
                data_integrity, created_by, git_sha, config_hash,
                symbols, start_date, end_date, started_at,
                trades_opened, trades_closed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.run_id,
            "replay_backtest",
            self.capital_mode,
            "candle_timestamp",
            data_integrity,
            "replay_runner",
            self._get_git_sha(),
            self._get_config_hash(),
            ",".join(self.symbols),
            self.start_date,
            self.end_date,
            self.started_at.isoformat(),
            0,
            0
        ))
        conn.commit()
        conn.close()

        logger.info(f"Created run record: {self.run_id}")

    def record_trade_opened(self) -> None:
        """Increment trades opened counter."""
        self.trades_opened += 1
        self._update_run_counts()

    def record_trade_closed(self) -> None:
        """Increment trades closed counter."""
        self.trades_closed += 1
        self._update_run_counts()

    def _update_run_counts(self) -> None:
        """Update trade counts in database."""
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            UPDATE runs SET trades_opened = ?, trades_closed = ?
            WHERE run_id = ?
        """, (self.trades_opened, self.trades_closed, self.run_id))
        conn.commit()
        conn.close()

    def complete(self, notes: Optional[str] = None) -> None:
        """Mark run as completed."""
        import sqlite3

        self.completed_at = datetime.utcnow()

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            UPDATE runs SET completed_at = ?, notes = ?
            WHERE run_id = ?
        """, (self.completed_at.isoformat(), notes, self.run_id))
        conn.commit()
        conn.close()

        logger.info(f"Run completed: {self.run_id} | Opened: {self.trades_opened} | Closed: {self.trades_closed}")
```

Update `src/replay/__init__.py`:

```python
"""Replay Runner - Fast path to 100 validated trades."""
from .data_loader import DataLoader
from .runner import ReplayRunner

__all__ = ['DataLoader', 'ReplayRunner']
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit/test_replay_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/replay/runner.py src/replay/__init__.py tests/unit/test_replay_runner.py
git commit -m "feat(replay): add ReplayRunner with run metadata tracking"
```

---

### Task 1.4: Create Replay Report Generator

**Files:**
- Create: `src/replay/report.py`
- Test: `tests/unit/test_replay_report.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_replay_report.py
import pytest
import json
import tempfile

def test_report_generator_import():
    """ReportGenerator should be importable."""
    from replay.report import ReportGenerator
    assert ReportGenerator is not None

def test_report_generator_produces_json():
    """ReportGenerator should produce valid JSON report."""
    from replay.report import ReportGenerator
    from truth.logger import TruthLogger

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        logger = TruthLogger(f.name)
        logger.initialize()

        generator = ReportGenerator(db_path=f.name, run_id="test_run_123")
        report = generator.generate()

        # Should be valid JSON
        json_str = json.dumps(report)
        assert json_str is not None

        # Should have required sections
        assert "trades" in report
        assert "risk" in report
        assert "data_quality" in report

def test_report_calculates_expectancy():
    """ReportGenerator should calculate expectancy correctly."""
    from replay.report import ReportGenerator

    # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
    # With 50% win rate, $10 avg win, $5 avg loss:
    # Expectancy = (0.5 * 10) - (0.5 * 5) = 5 - 2.5 = 2.5

    expectancy = ReportGenerator._calculate_expectancy(
        win_rate=0.5,
        avg_win=10.0,
        avg_loss=5.0
    )
    assert expectancy == 2.5
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit/test_replay_report.py::test_report_generator_import -v`
Expected: FAIL with "cannot import name 'ReportGenerator'"

**Step 3: Write minimal implementation**

```python
# src/replay/report.py
"""
Replay Report Generator

Generates validation reports from replay runs with metrics:
- Trades: opened, closed, win rate, R:R, expectancy
- Risk: max drawdown, consecutive losses
- Data Quality: gaps, volume anomalies
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Trade-level metrics."""
    trades_opened: int
    trades_closed: int
    wins: int
    losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    rr_actual: float
    expectancy: float
    profit_factor: float


@dataclass
class RiskMetrics:
    """Risk-level metrics."""
    max_drawdown_dollars: float
    max_drawdown_percent: float
    peak_equity: float
    final_equity: float
    max_consecutive_losses: int
    max_consecutive_loss_dollars: float


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    candles_processed: int
    gaps_detected: int
    volume_anomalies: int


class ReportGenerator:
    """
    Generate validation reports from replay runs.
    """

    def __init__(self, db_path: str, run_id: str):
        """
        Initialize report generator.

        Args:
            db_path: Path to Truth Engine database
            run_id: Run ID to generate report for
        """
        self.db_path = Path(db_path)
        self.run_id = run_id

    def generate(self) -> Dict[str, Any]:
        """
        Generate full validation report.

        Returns:
            Dict with trades, risk, data_quality sections
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            trades = self._get_trades(conn)
            run_info = self._get_run_info(conn)

            return {
                "run_id": self.run_id,
                "run_info": run_info,
                "trades": self._calculate_trade_metrics(trades),
                "risk": self._calculate_risk_metrics(trades),
                "data_quality": self._calculate_data_quality(conn),
                "generated_at": datetime.utcnow().isoformat()
            }
        finally:
            conn.close()

    def _get_run_info(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Get run metadata."""
        cursor = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (self.run_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    def _get_trades(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Get all closed trades for this run."""
        cursor = conn.execute("""
            SELECT * FROM trades
            WHERE run_id = ? AND status = 'closed'
            ORDER BY exit_timestamp
        """, (self.run_id,))
        return [dict(row) for row in cursor.fetchall()]

    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate trade-level metrics."""
        if not trades:
            return {
                "trades_closed": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "rr_actual": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0
            }

        wins = [t for t in trades if t.get("is_winner")]
        losses = [t for t in trades if not t.get("is_winner")]

        win_pnls = [float(t["net_pnl"]) for t in wins if t.get("net_pnl")]
        loss_pnls = [abs(float(t["net_pnl"])) for t in losses if t.get("net_pnl")]

        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
        win_rate = len(wins) / len(trades) if trades else 0.0

        rr_actual = avg_win / avg_loss if avg_loss > 0 else 0.0
        expectancy = self._calculate_expectancy(win_rate, avg_win, avg_loss)

        total_wins = sum(win_pnls)
        total_losses = sum(loss_pnls)
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        return {
            "trades_closed": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "rr_actual": round(rr_actual, 2),
            "expectancy": round(expectancy, 2),
            "profit_factor": round(profit_factor, 2)
        }

    @staticmethod
    def _calculate_expectancy(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate expectancy per trade.

        Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        """
        loss_rate = 1 - win_rate
        return (win_rate * avg_win) - (loss_rate * avg_loss)

    def _calculate_risk_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate risk metrics including drawdown."""
        if not trades:
            return {
                "max_drawdown_dollars": 0.0,
                "max_drawdown_percent": 0.0,
                "peak_equity": 0.0,
                "final_equity": 0.0,
                "max_consecutive_losses": 0,
                "max_consecutive_loss_dollars": 0.0
            }

        # Build equity curve
        equity = 0.0
        peak = 0.0
        max_dd_dollars = 0.0
        max_dd_percent = 0.0

        consecutive_losses = 0
        max_consecutive_losses = 0
        consecutive_loss_dollars = 0.0
        max_consecutive_loss_dollars = 0.0

        for trade in trades:
            pnl = float(trade.get("net_pnl", 0))
            equity += pnl

            if equity > peak:
                peak = equity

            drawdown = peak - equity
            if drawdown > max_dd_dollars:
                max_dd_dollars = drawdown
                if peak > 0:
                    max_dd_percent = (drawdown / peak) * 100

            # Track consecutive losses
            if not trade.get("is_winner"):
                consecutive_losses += 1
                consecutive_loss_dollars += abs(pnl)
                if consecutive_losses > max_consecutive_losses:
                    max_consecutive_losses = consecutive_losses
                    max_consecutive_loss_dollars = consecutive_loss_dollars
            else:
                consecutive_losses = 0
                consecutive_loss_dollars = 0.0

        return {
            "max_drawdown_dollars": round(max_dd_dollars, 2),
            "max_drawdown_percent": round(max_dd_percent, 2),
            "peak_equity": round(peak, 2),
            "final_equity": round(equity, 2),
            "max_consecutive_losses": max_consecutive_losses,
            "max_consecutive_loss_dollars": round(max_consecutive_loss_dollars, 2)
        }

    def _calculate_data_quality(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        # Placeholder - would need candle-level tracking
        return {
            "candles_processed": 0,
            "gaps_detected": 0,
            "volume_anomalies": 0
        }

    def to_json(self, indent: int = 2) -> str:
        """Generate report as JSON string."""
        return json.dumps(self.generate(), indent=indent)

    def print_console(self) -> None:
        """Print report to console."""
        report = self.generate()

        print("\n" + "=" * 60)
        print(f"REPLAY VALIDATION REPORT - {self.run_id}")
        print("=" * 60)

        trades = report["trades"]
        print(f"\nTRADES")
        print(f"  Closed: {trades['trades_closed']}")
        print(f"  Win Rate: {trades['win_rate']}%")
        print(f"  Avg Win: ${trades['avg_win']}")
        print(f"  Avg Loss: ${trades['avg_loss']}")
        print(f"  R:R Actual: {trades['rr_actual']}")
        print(f"  Expectancy: ${trades['expectancy']}/trade")
        print(f"  Profit Factor: {trades['profit_factor']}")

        risk = report["risk"]
        print(f"\nRISK")
        print(f"  Max Drawdown: ${risk['max_drawdown_dollars']} ({risk['max_drawdown_percent']}%)")
        print(f"  Peak Equity: ${risk['peak_equity']}")
        print(f"  Final Equity: ${risk['final_equity']}")
        print(f"  Max Consecutive Losses: {risk['max_consecutive_losses']} (${risk['max_consecutive_loss_dollars']})")

        print("\n" + "=" * 60)
```

Update `src/replay/__init__.py`:

```python
"""Replay Runner - Fast path to 100 validated trades."""
from .data_loader import DataLoader
from .runner import ReplayRunner
from .report import ReportGenerator

__all__ = ['DataLoader', 'ReplayRunner', 'ReportGenerator']
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit/test_replay_report.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/replay/report.py src/replay/__init__.py tests/unit/test_replay_report.py
git commit -m "feat(replay): add ReportGenerator with validation metrics"
```

---

### Task 1.5: Create CLI Entry Point

**Files:**
- Create: `scripts/replay_runner.py`
- Test: Manual CLI test

**Step 1: Write the implementation**

```python
#!/usr/bin/env python3
"""
Replay Runner CLI - Fast path to 100 validated trades.

Usage:
    python scripts/replay_runner.py \\
        --symbols BTC-USD,ETH-USD,SOL-USD \\
        --start 2024-01-01 \\
        --end 2025-12-17 \\
        --capital 500 \\
        --data-source cache \\
        --report-format json

Debug flags:
    --skip-volume-filter   # Tags run as data_integrity=degraded_skip_volume
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from replay import DataLoader, ReplayRunner, ReportGenerator
from engine import TradingEngine
from strategy.donchian import DonchianBreakout
from risk import RiskManager, RiskConfig
from execution import PaperExecutor
from truth.logger import TruthLogger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Replay Runner - Fast path to 100 validated trades"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC-USD",
        help="Comma-separated list of symbols (default: BTC-USD)"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=500.0,
        help="Starting capital per symbol (default: 500)"
    )
    parser.add_argument(
        "--capital-mode",
        type=str,
        choices=["independent", "portfolio"],
        default="independent",
        help="Capital mode (default: independent)"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["api", "cache"],
        default="cache",
        help="Data source (default: cache)"
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="4h",
        help="Candle granularity (default: 4h)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/v4_replay.db",
        help="Database path (default: data/v4_replay.db)"
    )
    parser.add_argument(
        "--report-format",
        type=str,
        choices=["json", "console", "both"],
        default="both",
        help="Report format (default: both)"
    )
    parser.add_argument(
        "--skip-volume-filter",
        action="store_true",
        help="Skip volume validation (tags run as degraded)"
    )

    return parser.parse_args()


def run_replay(
    symbol: str,
    data: "pd.DataFrame",
    runner: ReplayRunner,
    db_path: str,
    capital: Decimal
) -> int:
    """
    Run replay for a single symbol.

    Returns:
        Number of trades closed
    """
    logger.info(f"Starting replay for {symbol} ({len(data)} candles)")

    # Initialize components
    strategy = DonchianBreakout()
    risk_manager = RiskManager(RiskConfig())
    executor = PaperExecutor(starting_balance=capital)
    truth_logger = TruthLogger(db_path)

    engine = TradingEngine(
        strategy=strategy,
        risk_manager=risk_manager,
        executor=executor,
        truth_logger=truth_logger,
        symbol=symbol,
        capital=capital
    )

    # Rolling window size (same as live)
    WINDOW_SIZE = 300
    trades_closed = 0

    # Iterate through candles
    for i in range(WINDOW_SIZE, len(data)):
        window = data.iloc[i - WINDOW_SIZE:i + 1].copy()
        candle_time = window.iloc[-1]['timestamp']

        # Convert to datetime if needed
        if hasattr(candle_time, 'to_pydatetime'):
            candle_time = candle_time.to_pydatetime()

        # Run tick with historical timestamp
        result = engine.run_tick(window, as_of=candle_time)

        if result.trade_id:
            runner.record_trade_opened()
            logger.info(f"  [{candle_time}] Trade OPENED: {result.trade_id}")

        # Check for position exit
        # (simplified - full implementation would handle exit signals)

    return trades_closed


def main():
    """Main entry point."""
    args = parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    capital = Decimal(str(args.capital))

    logger.info("=" * 60)
    logger.info("REPLAY RUNNER - Gate A Validation")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Capital: ${capital} ({args.capital_mode} mode)")
    logger.info(f"Data source: {args.data_source}")

    # Initialize runner
    runner = ReplayRunner(
        db_path=args.db_path,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        capital=capital,
        capital_mode=args.capital_mode,
        granularity=args.granularity,
        skip_volume_filter=args.skip_volume_filter
    )

    logger.info(f"Run ID: {runner.run_id}")

    # Load data
    data_loader = DataLoader(cache_dir="data/cache")

    for symbol in symbols:
        logger.info(f"\n--- Processing {symbol} ---")

        use_cache = args.data_source == "cache"
        data = data_loader.load(
            symbol=symbol,
            start=args.start,
            end=args.end,
            granularity=args.granularity,
            use_cache=use_cache
        )

        logger.info(f"Loaded {len(data)} candles")

        # Run replay
        run_replay(
            symbol=symbol,
            data=data,
            runner=runner,
            db_path=args.db_path,
            capital=capital
        )

    # Complete run
    runner.complete()

    # Generate report
    report_gen = ReportGenerator(db_path=args.db_path, run_id=runner.run_id)

    if args.report_format in ["console", "both"]:
        report_gen.print_console()

    if args.report_format in ["json", "both"]:
        report_path = Path(f"data/reports/{runner.run_id}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_gen.to_json())
        logger.info(f"\nJSON report saved to: {report_path}")

    logger.info("\nReplay complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Make executable and test**

```bash
chmod +x scripts/replay_runner.py
source venv/bin/activate && python scripts/replay_runner.py --symbols BTC-USD --start 2024-01-01 --end 2024-01-07 --help
```

Expected: Help message displayed

**Step 3: Commit**

```bash
git add scripts/replay_runner.py
git commit -m "feat(replay): add CLI entry point for replay runner"
```

---

## Phase 2: Bracket Orders (Safety Foundation)

### Task 2.1: Create Product Specs Module

**Files:**
- Create: `src/execution/product_specs.py`
- Test: `tests/unit/test_product_specs.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_product_specs.py
import pytest
from decimal import Decimal

def test_product_specs_import():
    """ProductSpecs should be importable."""
    from execution.product_specs import ProductSpecs
    assert ProductSpecs is not None

def test_btc_usd_specs():
    """BTC-USD should have correct increments."""
    from execution.product_specs import ProductSpecs

    specs = ProductSpecs.get("BTC-USD")

    assert specs.base_increment == Decimal("0.00000001")  # 8 decimals
    assert specs.quote_increment == Decimal("0.01")       # 2 decimals
    assert specs.min_market_funds > 0

def test_round_price_floor():
    """round_price_floor should round DOWN to increment."""
    from execution.product_specs import ProductSpecs

    specs = ProductSpecs.get("BTC-USD")

    # $100,000.456 should become $100,000.45 (floor)
    result = specs.round_price_floor(Decimal("100000.456"))
    assert result == Decimal("100000.45")

def test_round_price_ceil():
    """round_price_ceil should round UP to increment."""
    from execution.product_specs import ProductSpecs

    specs = ProductSpecs.get("BTC-USD")

    # $100,000.451 should become $100,000.46 (ceil)
    result = specs.round_price_ceil(Decimal("100000.451"))
    assert result == Decimal("100000.46")

def test_round_quantity():
    """round_quantity should round to base increment."""
    from execution.product_specs import ProductSpecs

    specs = ProductSpecs.get("BTC-USD")

    # 0.123456789 should become 0.12345678 (8 decimals)
    result = specs.round_quantity(Decimal("0.123456789"))
    assert result == Decimal("0.12345678")
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit/test_product_specs.py::test_product_specs_import -v`
Expected: FAIL with "cannot import name 'ProductSpecs'"

**Step 3: Write minimal implementation**

```python
# src/execution/product_specs.py
"""
Product Specifications for Coinbase Trading Pairs

Contains increment/minimum requirements for order building.
Values sourced from Coinbase product info endpoint.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, Optional


@dataclass
class ProductSpec:
    """Specification for a trading product."""

    symbol: str
    base_currency: str
    quote_currency: str
    base_increment: Decimal      # Smallest quantity step
    quote_increment: Decimal     # Smallest price step
    min_market_funds: Decimal    # Minimum order value
    max_market_funds: Decimal    # Maximum order value

    def round_price_floor(self, price: Decimal) -> Decimal:
        """Round price DOWN to quote increment (for entry/TP)."""
        return (price / self.quote_increment).quantize(
            Decimal("1"), rounding=ROUND_DOWN
        ) * self.quote_increment

    def round_price_ceil(self, price: Decimal) -> Decimal:
        """Round price UP to quote increment (for stop trigger)."""
        return (price / self.quote_increment).quantize(
            Decimal("1"), rounding=ROUND_UP
        ) * self.quote_increment

    def round_quantity(self, quantity: Decimal) -> Decimal:
        """Round quantity DOWN to base increment."""
        return (quantity / self.base_increment).quantize(
            Decimal("1"), rounding=ROUND_DOWN
        ) * self.base_increment

    def validate_order_value(self, quantity: Decimal, price: Decimal) -> bool:
        """Check if order meets minimum/maximum value requirements."""
        value = quantity * price
        return self.min_market_funds <= value <= self.max_market_funds


class ProductSpecs:
    """
    Registry of product specifications.

    Usage:
        specs = ProductSpecs.get("BTC-USD")
        rounded_price = specs.round_price_floor(Decimal("100000.456"))
    """

    # Core trio specifications (from Coinbase product info)
    _SPECS: Dict[str, ProductSpec] = {
        "BTC-USD": ProductSpec(
            symbol="BTC-USD",
            base_currency="BTC",
            quote_currency="USD",
            base_increment=Decimal("0.00000001"),  # 8 decimals
            quote_increment=Decimal("0.01"),        # 2 decimals
            min_market_funds=Decimal("1.00"),       # $1 minimum
            max_market_funds=Decimal("1000000.00")  # $1M maximum
        ),
        "ETH-USD": ProductSpec(
            symbol="ETH-USD",
            base_currency="ETH",
            quote_currency="USD",
            base_increment=Decimal("0.00000001"),  # 8 decimals
            quote_increment=Decimal("0.01"),        # 2 decimals
            min_market_funds=Decimal("1.00"),
            max_market_funds=Decimal("1000000.00")
        ),
        "SOL-USD": ProductSpec(
            symbol="SOL-USD",
            base_currency="SOL",
            quote_currency="USD",
            base_increment=Decimal("0.00000001"),  # 8 decimals
            quote_increment=Decimal("0.01"),        # 2 decimals
            min_market_funds=Decimal("1.00"),
            max_market_funds=Decimal("1000000.00")
        ),
    }

    @classmethod
    def get(cls, symbol: str) -> ProductSpec:
        """
        Get product specification for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")

        Returns:
            ProductSpec for the symbol

        Raises:
            KeyError: If symbol not in registry
        """
        if symbol not in cls._SPECS:
            raise KeyError(
                f"No specs for {symbol}. Available: {list(cls._SPECS.keys())}"
            )
        return cls._SPECS[symbol]

    @classmethod
    def has(cls, symbol: str) -> bool:
        """Check if symbol has specs registered."""
        return symbol in cls._SPECS
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit/test_product_specs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/execution/product_specs.py tests/unit/test_product_specs.py
git commit -m "feat(execution): add ProductSpecs for price/quantity rounding"
```

---

### Task 2.2: Create Bracket Order Builder

**Files:**
- Create: `src/execution/bracket.py`
- Test: `tests/unit/test_bracket_builder.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_bracket_builder.py
import pytest
from decimal import Decimal

def test_bracket_builder_import():
    """BracketOrderBuilder should be importable."""
    from execution.bracket import BracketOrderBuilder
    assert BracketOrderBuilder is not None

def test_build_bracket_payload():
    """Should build correct Coinbase bracket payload structure."""
    from execution.bracket import BracketOrderBuilder

    builder = BracketOrderBuilder()
    payload = builder.build(
        symbol="BTC-USD",
        side="BUY",
        quantity=Decimal("0.001"),
        entry_price=Decimal("104500.00"),
        take_profit_price=Decimal("112860.00"),
        stop_trigger_price=Decimal("102361.00")
    )

    # Check required fields
    assert payload["product_id"] == "BTC-USD"
    assert payload["side"] == "BUY"
    assert "client_order_id" in payload
    assert payload["client_order_id"].startswith("ANX4_")

    # Check order configuration
    assert "order_configuration" in payload
    order_config = payload["order_configuration"]
    assert "limit_limit_gtc" in order_config

    # Check attached bracket
    assert "attached_order_configuration" in payload
    attached = payload["attached_order_configuration"]
    assert "trigger_bracket_gtc" in attached
    bracket = attached["trigger_bracket_gtc"]
    assert bracket["limit_price"] == "112860.00"
    assert bracket["stop_trigger_price"] == "102361.00"

def test_client_order_id_format():
    """client_order_id should follow ANX4_{run8}_{symbol}_{ts}_{hash8} format."""
    from execution.bracket import BracketOrderBuilder
    import re

    builder = BracketOrderBuilder(run_id="abc12345")
    payload = builder.build(
        symbol="BTC-USD",
        side="BUY",
        quantity=Decimal("0.001"),
        entry_price=Decimal("104500.00"),
        take_profit_price=Decimal("112860.00"),
        stop_trigger_price=Decimal("102361.00")
    )

    client_id = payload["client_order_id"]
    # ANX4_abc12345_BTCUSD_20251217T120000_7f3a2b1c
    pattern = r"ANX4_[a-f0-9]{8}_[A-Z]+_\d{8}T\d{6}_[a-f0-9]{8}"
    assert re.match(pattern, client_id), f"Invalid format: {client_id}"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit/test_bracket_builder.py::test_bracket_builder_import -v`
Expected: FAIL with "cannot import name 'BracketOrderBuilder'"

**Step 3: Write minimal implementation**

```python
# src/execution/bracket.py
"""
Bracket Order Builder for Coinbase Advanced Trade API

Builds atomic bracket orders with TP/SL attached at entry time.
Uses trigger_bracket_gtc for guaranteed protection.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from .product_specs import ProductSpecs


logger = logging.getLogger(__name__)


@dataclass
class BracketConfig:
    """Configuration for bracket orders."""

    # Coinbase applies 5% buffer on stop-limit (hypothesis - confirm with live test)
    stop_buffer_percent: Decimal = Decimal("0.05")

    # Post-only for maker fees
    post_only: bool = False


class BracketOrderBuilder:
    """
    Build Coinbase bracket order payloads.

    Generates properly formatted payloads with:
    - Atomic entry + TP + SL via attached_order_configuration
    - Deterministic client_order_id for idempotency
    - Directional price rounding (floor for entry/TP, ceil for stop)
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        config: Optional[BracketConfig] = None
    ):
        """
        Initialize bracket builder.

        Args:
            run_id: Run identifier for client_order_id (8 chars)
            config: Bracket configuration
        """
        self.run_id = run_id or "00000000"
        self.config = config or BracketConfig()

    def build(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        take_profit_price: Decimal,
        stop_trigger_price: Decimal,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Build bracket order payload.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: "BUY" or "SELL"
            quantity: Position size
            entry_price: Limit entry price
            take_profit_price: Take profit limit price
            stop_trigger_price: Stop loss trigger price
            timestamp: Order timestamp (for client_order_id)

        Returns:
            Dict payload for Coinbase create_order API
        """
        specs = ProductSpecs.get(symbol)
        ts = timestamp or datetime.utcnow()

        # Round prices directionally
        entry_rounded = specs.round_price_floor(entry_price)
        tp_rounded = specs.round_price_floor(take_profit_price)
        stop_rounded = specs.round_price_ceil(stop_trigger_price)
        qty_rounded = specs.round_quantity(quantity)

        # Generate idempotency key
        client_order_id = self._generate_client_order_id(
            symbol=symbol,
            timestamp=ts,
            entry_price=entry_rounded,
            quantity=qty_rounded
        )

        payload = {
            "client_order_id": client_order_id,
            "product_id": symbol,
            "side": side.upper(),
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": str(qty_rounded),
                    "limit_price": str(entry_rounded),
                    "post_only": self.config.post_only
                }
            },
            "attached_order_configuration": {
                "trigger_bracket_gtc": {
                    "limit_price": str(tp_rounded),
                    "stop_trigger_price": str(stop_rounded)
                }
            }
        }

        logger.info(
            f"Built bracket order: {symbol} {side} {qty_rounded} @ {entry_rounded} | "
            f"TP: {tp_rounded} | SL trigger: {stop_rounded}"
        )

        return payload

    def _generate_client_order_id(
        self,
        symbol: str,
        timestamp: datetime,
        entry_price: Decimal,
        quantity: Decimal
    ) -> str:
        """
        Generate deterministic client_order_id.

        Format: ANX4_{run8}_{symbol}_{ts}_{hash8}

        The hash ensures uniqueness even with same symbol/time.
        """
        # Clean symbol (remove hyphen)
        clean_symbol = symbol.replace("-", "")

        # Format timestamp
        ts_str = timestamp.strftime("%Y%m%dT%H%M%S")

        # Generate hash from order details
        hash_input = f"{symbol}{timestamp.isoformat()}{entry_price}{quantity}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]

        return f"ANX4_{self.run_id[:8]}_{clean_symbol}_{ts_str}_{hash_value}"

    def calculate_worst_case_risk(
        self,
        entry_price: Decimal,
        stop_trigger_price: Decimal,
        quantity: Decimal
    ) -> Decimal:
        """
        Calculate worst-case risk accounting for stop buffer.

        Coinbase trigger_bracket_gtc may fill up to 5% below trigger.

        Args:
            entry_price: Entry price
            stop_trigger_price: Stop trigger price
            quantity: Position size

        Returns:
            Worst-case dollar loss
        """
        worst_fill = stop_trigger_price * (1 - self.config.stop_buffer_percent)
        worst_risk_per_unit = entry_price - worst_fill
        return worst_risk_per_unit * quantity
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit/test_bracket_builder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/execution/bracket.py tests/unit/test_bracket_builder.py
git commit -m "feat(execution): add BracketOrderBuilder for atomic TP/SL orders"
```

---

## Remaining Tasks Summary

The remaining implementation tasks follow the same pattern:

### Phase 2 Continued:
- **Task 2.3:** Bracket-aware position sizing
- **Task 2.4:** Bracket executor with verification
- **Task 2.5:** Engine state machine integration (IDLE → ENTRY_PENDING → POSITION_OPEN)
- **Task 2.6:** Single finalize_exit() path for all exit types

### Phase 3: Integration
- **Task 3.1:** Add run_id to TruthLogger methods
- **Task 3.2:** Wire bracket executor into engine
- **Task 3.3:** End-to-end smoke test

---

## Verification Checklist

Before marking implementation complete:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Replay runner produces 100+ closed trades
- [ ] Report metrics match manual calculation
- [ ] Bracket orders verified on Coinbase sandbox
- [ ] No `datetime.utcnow()` calls in replay path
- [ ] All trades tagged with run_id
- [ ] Git commits atomic and well-described

---

## How to Execute This Plan

**Option 1: Subagent-Driven (this session)**
- Use superpowers:subagent-driven-development
- Fresh subagent per task with code review

**Option 2: Parallel Session**
- Open new session in worktree: `cd .worktrees/validation-infra`
- Use superpowers:executing-plans
- Batch execution with checkpoints

---

*Plan generated: 2025-12-17*
*Design reference: docs/plans/2025-12-17-validation-infrastructure-design.md*
