"""
Truth Engine Database Schema - V4 Core

THE THREE TABLES OF TRUTH:
1. decisions - Why did we consider this trade? (CONTEXT)
2. orders - What orders were placed? (ACTIONS)
3. trades - What was the P&L outcome? (RESULTS)

Every trade is traceable: trades -> orders -> decisions
Query time target: <5 seconds to explain any trade

Design Principles:
- Log DECISIONS (state), not just ACTIONS (events)
- Every field must answer "why did this happen?"
- Decimal stored as TEXT to preserve precision
- JSON for complex nested data (signal values, risk checks)
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import uuid


class DecisionResult(Enum):
    """Outcome of a trading decision evaluation"""
    SIGNAL_LONG = "signal_long"          # Strategy says go long
    SIGNAL_SHORT = "signal_short"        # Strategy says go short
    SIGNAL_CLOSE = "signal_close"        # Strategy says close position
    SIGNAL_HOLD = "signal_hold"          # No action needed
    RISK_REJECTED = "risk_rejected"      # Signal generated but risk check failed
    SESSION_PAUSED = "session_paused"    # Paused due to dead zone or session state
    PORTFOLIO_BLOCKED = "portfolio_blocked"  # Blocked by portfolio-level risk
    MTF_CONFLICT = "mtf_conflict"        # Multi-timeframe signals conflicted
    NO_SIGNAL = "no_signal"              # Strategy evaluated, no signal
    ERROR = "error"                      # Evaluation failed


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExitReason(Enum):
    """Why a position was closed"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    SIGNAL_EXIT = "signal_exit"          # Strategy signaled exit
    MANUAL = "manual"                    # User intervention
    RISK_LIMIT = "risk_limit"            # Risk system forced close
    END_OF_DAY = "end_of_day"            # Time-based exit
    ERROR = "error"                      # Forced close due to error


@dataclass
class Decision:
    """
    Records WHY a trade was considered.

    This is the CONTEXT table - the most important for debugging.
    Every time the strategy evaluates, a decision is logged.
    Even if no trade happens, we record why.

    Query: "Why did trade X happen?"
    Answer: Join to decisions table, read signal_values and risk_checks
    """
    decision_id: str                     # UUID - primary key
    timestamp: datetime                  # When evaluation occurred
    symbol: str                          # Trading pair (e.g., "BTC-USD")
    strategy_name: str                   # "dual_ema_crossover"

    # What the strategy saw (THE GOLD)
    signal_values: Dict[str, Any] = field(default_factory=dict)
    # Example: {
    #   "fast_ema": 50123.45,
    #   "slow_ema": 50100.00,
    #   "ema_diff": 23.45,
    #   "ema_diff_prev": -15.20,
    #   "crossover": "bullish",
    #   "atr": 234.56,
    #   "current_price": 50150.00
    # }

    # Risk check results - ALL checks logged
    risk_checks: Dict[str, Any] = field(default_factory=dict)
    # Example: {
    #   "trading_halted": {"passed": true, "value": false},
    #   "frequency_limit": {"passed": true, "trades_today": 2, "limit": 10},
    #   "daily_loss_limit": {"passed": true, "current_loss": 0.5, "limit": 2.0},
    #   "drawdown_limit": {"passed": true, "current_dd": 1.2, "limit": 5.0},
    #   "circuit_breaker": {"passed": true, "price_move": 2.1, "threshold": 8.0},
    #   "risk_reward": {"passed": true, "ratio": 1.8, "minimum": 1.5},
    #   "concentration": {"passed": true, "symbol_exposure": 15, "limit": 30},
    #   "correlation": {"passed": true, "correlated_exposure": 25, "limit": 50}
    # }

    # Market context at decision time
    market_context: Dict[str, Any] = field(default_factory=dict)
    # Example: {
    #   "bid": 50145.00,
    #   "ask": 50155.00,
    #   "spread_pct": 0.02,
    #   "volume_24h": 1234567890,
    #   "volatility": "normal"
    # }

    result: DecisionResult = DecisionResult.NO_SIGNAL
    result_reason: str = ""              # Human-readable explanation

    # Linkage
    order_id: Optional[str] = None       # If order was placed, link to it

    # Session context for 24/7 perpetual trading
    session_context: Optional[Dict[str, Any]] = None
    # Example: {
    #   "session": "london_ny",
    #   "state": "active",
    #   "is_dead_zone": false,
    #   "liquidity_score": 0.85,
    #   "position_size_multiplier": 1.0,
    #   "time_to_next_session_minutes": 45
    # }

    # Multi-timeframe aggregation result
    mtf_aggregation: Optional[Dict[str, Any]] = None
    # Example: {
    #   "signal": "long",
    #   "is_aligned": true,
    #   "contributing_timeframes": ["1h", "4h"],
    #   "individual_signals": {...}
    # }

    # Portfolio-level risk check result
    portfolio_risk: Optional[Dict[str, Any]] = None
    # Example: {
    #   "approved": true,
    #   "total_exposure_pct": 45.2,
    #   "correlation_group": "large_cap",
    #   "group_exposure_pct": 32.1
    # }

    # LLM-generated insight (GPT-5.2)
    llm_insight: Optional[Dict[str, Any]] = None
    # Example: {
    #   "summary": "Reducing XRP exposure from 100% to 78.8%...",
    #   "assessment": "Good decision - drift was significant...",
    #   "risk_flags": ["High volatility period"],
    #   "confidence_score": 0.85
    # }

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "signal_values": json.dumps(self.signal_values),
            "risk_checks": json.dumps(self.risk_checks),
            "market_context": json.dumps(self.market_context),
            "result": self.result.value,
            "result_reason": self.result_reason,
            "order_id": self.order_id,
            "session_context": json.dumps(self.session_context) if self.session_context else None,
            "mtf_aggregation": json.dumps(self.mtf_aggregation) if self.mtf_aggregation else None,
            "portfolio_risk": json.dumps(self.portfolio_risk) if self.portfolio_risk else None,
            "llm_insight": json.dumps(self.llm_insight) if self.llm_insight else None
        }


@dataclass
class Order:
    """
    Records WHAT orders were placed.

    This is the ACTIONS table.
    Links back to the decision that caused it.
    Tracks execution details and slippage.
    """
    order_id: str                        # UUID - primary key
    decision_id: str                     # FK to decisions - WHY this order exists
    timestamp: datetime                  # When order was created
    symbol: str                          # Trading pair
    side: OrderSide                      # buy or sell
    quantity: Decimal                    # Size of order

    # Pricing
    requested_price: Optional[Decimal] = None    # Limit price or expected price
    fill_price: Optional[Decimal] = None         # Actual execution price
    fill_quantity: Optional[Decimal] = None      # Actual filled quantity
    fill_timestamp: Optional[datetime] = None    # When fill occurred

    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None      # ID from exchange

    # Slippage tracking (calculated on fill)
    slippage_amount: Optional[Decimal] = None    # fill_price - requested_price
    slippage_percent: Optional[Decimal] = None   # slippage as percentage

    # Fees
    commission: Optional[Decimal] = None
    commission_asset: Optional[str] = None

    # Error tracking
    error_message: Optional[str] = None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    def calculate_slippage(self) -> None:
        """Calculate slippage after fill"""
        if self.fill_price and self.requested_price:
            self.slippage_amount = self.fill_price - self.requested_price
            if self.requested_price != 0:
                self.slippage_percent = (self.slippage_amount / self.requested_price) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "order_id": self.order_id,
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "requested_price": str(self.requested_price) if self.requested_price else None,
            "fill_price": str(self.fill_price) if self.fill_price else None,
            "fill_quantity": str(self.fill_quantity) if self.fill_quantity else None,
            "fill_timestamp": self.fill_timestamp.isoformat() if self.fill_timestamp else None,
            "status": self.status.value,
            "exchange_order_id": self.exchange_order_id,
            "slippage_amount": str(self.slippage_amount) if self.slippage_amount else None,
            "slippage_percent": str(self.slippage_percent) if self.slippage_percent else None,
            "commission": str(self.commission) if self.commission else None,
            "commission_asset": self.commission_asset,
            "error_message": self.error_message
        }


@dataclass
class Trade:
    """
    Records the P&L OUTCOME of a closed position.

    This is the RESULTS table.
    Links back through orders to decisions.
    This is THE source of truth for profitability.

    Query: "Was I profitable this week?"
    Answer: SELECT SUM(realized_pnl) FROM trades WHERE exit_timestamp > ?
    """
    trade_id: str                        # UUID - primary key
    symbol: str                          # Trading pair
    side: OrderSide                      # Entry side (buy = long, sell = short)

    # Order linkage
    entry_order_id: str                  # FK to orders
    exit_order_id: str                   # FK to orders
    entry_decision_id: str               # FK to decisions (for fast queries)
    exit_decision_id: str                # FK to decisions

    # Timestamps
    entry_timestamp: datetime
    exit_timestamp: datetime
    duration_seconds: int                # Calculated: exit - entry

    # Prices and size
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal

    # THE NUMBERS THAT MATTER
    realized_pnl: Decimal                # In quote currency (USD)
    realized_pnl_percent: Decimal        # As percentage of entry value

    # Cost tracking
    total_commission: Decimal            # Entry + exit commissions
    total_slippage: Decimal              # Entry + exit slippage
    net_pnl: Decimal                     # realized_pnl - commission - slippage

    # Exit classification
    exit_reason: ExitReason

    # Win/Loss classification
    is_winner: bool                      # net_pnl > 0

    # Data quality tracking (PR-3)
    is_valid: bool = True                # False if required data missing
    invalid_reason: Optional[str] = None # Why trade is invalid (if is_valid=False)

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_order_id": self.entry_order_id,
            "exit_order_id": self.exit_order_id,
            "entry_decision_id": self.entry_decision_id,
            "exit_decision_id": self.exit_decision_id,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "exit_timestamp": self.exit_timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "entry_price": str(self.entry_price),
            "exit_price": str(self.exit_price),
            "quantity": str(self.quantity),
            "realized_pnl": str(self.realized_pnl),
            "realized_pnl_percent": str(self.realized_pnl_percent),
            "total_commission": str(self.total_commission),
            "total_slippage": str(self.total_slippage),
            "net_pnl": str(self.net_pnl),
            "exit_reason": self.exit_reason.value,
            "is_winner": self.is_winner,
            "is_valid": self.is_valid,
            "invalid_reason": self.invalid_reason
        }


@dataclass
class PhantomTrade:
    """
    Records a hypothetical trade we DIDN'T take.

    Used for validation: tracks what WOULD have happened if we'd
    traded B/C grade setups that were filtered out.

    Verdict Types:
    - regret: We should have taken this trade (profitable)
    - relief: Glad we skipped it (would have lost)
    - neutral: Marginal outcome (within noise threshold)
    - pending: Not yet evaluated

    Query: "Are we leaving money on the table with strict filtering?"
    Answer: SELECT * FROM v_phantom_by_grade ORDER BY setup_grade
    """
    phantom_id: str
    decision_id: str                      # Links to the HOLD decision
    symbol: str
    timestamp: datetime

    # Setup context
    setup_grade: str                      # A+, A, B, C
    signal_type: str                      # long, short
    hypothetical_entry: float

    # Stop levels at entry
    chandelier_stop_price: Optional[float] = None
    hard_stop_price: Optional[float] = None

    # Outcome tracking (filled in by background job)
    price_after_1h: Optional[float] = None
    price_after_4h: Optional[float] = None
    price_after_24h: Optional[float] = None
    price_after_48h: Optional[float] = None

    # Stop analysis
    would_hit_chandelier: Optional[bool] = None
    would_hit_hard_stop: Optional[bool] = None
    chandelier_hit_time: Optional[datetime] = None
    hard_stop_hit_time: Optional[datetime] = None

    # Calculated outcomes
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    phantom_exit_price: Optional[float] = None
    phantom_pnl_percent: Optional[float] = None
    duration_to_exit_hours: Optional[float] = None

    # Verdict
    verdict: str = "pending"
    verdict_reason: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            "phantom_id": self.phantom_id,
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "setup_grade": self.setup_grade,
            "signal_type": self.signal_type,
            "hypothetical_entry": self.hypothetical_entry,
            "chandelier_stop_price": self.chandelier_stop_price,
            "hard_stop_price": self.hard_stop_price,
            "price_after_1h": self.price_after_1h,
            "price_after_4h": self.price_after_4h,
            "price_after_24h": self.price_after_24h,
            "price_after_48h": self.price_after_48h,
            "would_hit_chandelier": 1 if self.would_hit_chandelier else 0 if self.would_hit_chandelier is not None else None,
            "would_hit_hard_stop": 1 if self.would_hit_hard_stop else 0 if self.would_hit_hard_stop is not None else None,
            "chandelier_hit_time": self.chandelier_hit_time.isoformat() if self.chandelier_hit_time else None,
            "hard_stop_hit_time": self.hard_stop_hit_time.isoformat() if self.hard_stop_hit_time else None,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "phantom_exit_price": self.phantom_exit_price,
            "phantom_pnl_percent": self.phantom_pnl_percent,
            "duration_to_exit_hours": self.duration_to_exit_hours,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# =============================================================================
# SQL Schema for SQLite
# =============================================================================

SQL_SCHEMA = """
-- Truth Engine Schema V4
-- The Glass Box: Every trade traceable in <5 seconds

-- =============================================================================
-- DECISIONS TABLE: WHY did we consider this trade?
-- =============================================================================
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    signal_values TEXT NOT NULL,      -- JSON: what the strategy saw
    risk_checks TEXT NOT NULL,        -- JSON: all risk check results
    market_context TEXT,              -- JSON: market state at decision time
    result TEXT NOT NULL,             -- DecisionResult enum value
    result_reason TEXT,               -- Human-readable explanation
    order_id TEXT,                    -- FK to orders if order was placed
    session_context TEXT,             -- JSON: session/liquidity context (24/7 trading)
    mtf_aggregation TEXT,             -- JSON: multi-timeframe signal aggregation
    portfolio_risk TEXT,              -- JSON: portfolio-level risk check result
    llm_insight TEXT                  -- JSON: AI-generated insight (GPT-5.2)
);

-- =============================================================================
-- ORDERS TABLE: WHAT orders were placed?
-- =============================================================================
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,               -- 'buy' or 'sell'
    quantity TEXT NOT NULL,           -- Decimal as string
    requested_price TEXT,
    fill_price TEXT,
    fill_quantity TEXT,
    fill_timestamp TEXT,
    status TEXT NOT NULL,             -- OrderStatus enum value
    exchange_order_id TEXT,
    slippage_amount TEXT,
    slippage_percent TEXT,
    commission TEXT,
    commission_asset TEXT,
    error_message TEXT,
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

-- =============================================================================
-- TRADES TABLE: WHAT was the P&L outcome?
-- Now supports OPEN and CLOSED trades with status field
-- =============================================================================
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',  -- 'open' or 'closed'
    entry_order_id TEXT NOT NULL,
    exit_order_id TEXT,                   -- NULL for open trades
    entry_decision_id TEXT NOT NULL,
    exit_decision_id TEXT,                -- NULL for open trades
    entry_timestamp TEXT NOT NULL,
    exit_timestamp TEXT,                  -- NULL for open trades
    duration_seconds INTEGER,             -- NULL for open trades
    entry_price TEXT NOT NULL,
    exit_price TEXT,                      -- NULL for open trades
    quantity TEXT NOT NULL,
    stop_loss_price TEXT,                 -- For open trade management
    take_profit_price TEXT,               -- For open trade management
    strategy_name TEXT,                   -- Track which strategy opened this
    realized_pnl TEXT,                    -- NULL for open trades
    realized_pnl_percent TEXT,            -- NULL for open trades
    total_commission TEXT,
    total_slippage TEXT,
    net_pnl TEXT,                         -- NULL for open trades
    exit_reason TEXT,                     -- NULL for open trades
    is_winner INTEGER,                    -- NULL for open trades (0 or 1)
    is_valid INTEGER DEFAULT 1,           -- 0 if required data missing (PR-3)
    invalid_reason TEXT,                  -- Why trade is invalid (PR-3)
    FOREIGN KEY (entry_order_id) REFERENCES orders(order_id),
    FOREIGN KEY (exit_order_id) REFERENCES orders(order_id),
    FOREIGN KEY (entry_decision_id) REFERENCES decisions(decision_id),
    FOREIGN KEY (exit_decision_id) REFERENCES decisions(decision_id)
);

-- =============================================================================
-- INDEXES: Fast queries for the Glass Box promise
-- =============================================================================

-- Find decisions by time range
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
-- Find decisions by symbol
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
-- Find decisions by result (e.g., all RISK_REJECTED)
CREATE INDEX IF NOT EXISTS idx_decisions_result ON decisions(result);

-- Find orders by decision (trace back to why)
CREATE INDEX IF NOT EXISTS idx_orders_decision ON orders(decision_id);
-- Find orders by status
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
-- Find orders by time
CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(timestamp);

-- Find trades by symbol
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
-- Find trades by entry time (for P&L queries)
CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_timestamp);
-- Find trades by exit time
CREATE INDEX IF NOT EXISTS idx_trades_exit ON trades(exit_timestamp);
-- Find winning/losing trades
CREATE INDEX IF NOT EXISTS idx_trades_winner ON trades(is_winner);
-- Find trades by exit reason
CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades(exit_reason);
-- Find open trades
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

-- =============================================================================
-- REFLECTIONS TABLE: Lessons learned from past trades (Reflexion Layer)
-- The memory of Argus - where learning is stored
-- =============================================================================
CREATE TABLE IF NOT EXISTS reflections (
    reflection_id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_winner INTEGER NOT NULL,
    pnl_percent REAL NOT NULL,
    duration_hours REAL NOT NULL,
    market_regime TEXT NOT NULL,
    entry_atr REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    reflection_type TEXT NOT NULL,
    what_happened TEXT NOT NULL,
    what_expected TEXT NOT NULL,
    lesson_learned TEXT NOT NULL,
    action_items TEXT NOT NULL,          -- JSON array
    confidence REAL NOT NULL,
    applies_to_regimes TEXT NOT NULL,    -- JSON array
    applies_to_symbols TEXT NOT NULL,    -- JSON array
    context_hash TEXT NOT NULL,
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
);

-- Indexes for fast lesson queries
CREATE INDEX IF NOT EXISTS idx_reflections_symbol ON reflections(symbol);
CREATE INDEX IF NOT EXISTS idx_reflections_regime ON reflections(market_regime);
CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(reflection_type);
CREATE INDEX IF NOT EXISTS idx_reflections_winner ON reflections(is_winner);
CREATE INDEX IF NOT EXISTS idx_reflections_hash ON reflections(context_hash);
CREATE INDEX IF NOT EXISTS idx_reflections_created ON reflections(created_at);
CREATE INDEX IF NOT EXISTS idx_reflections_confidence ON reflections(confidence);

-- =============================================================================
-- REFLECTION PATTERNS TABLE: Aggregated patterns from multiple reflections
-- Higher-level insights derived from clusters of similar lessons
-- =============================================================================
CREATE TABLE IF NOT EXISTS reflection_patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_name TEXT NOT NULL,
    description TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    avg_pnl_impact REAL,
    market_regimes TEXT,                 -- JSON array
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_reflection_ids TEXT           -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_patterns_name ON reflection_patterns(pattern_name);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON reflection_patterns(confidence);

-- =============================================================================
-- VIEWS: Pre-built queries for common analysis
-- =============================================================================

-- Full trade audit: Everything about a trade in one query
CREATE VIEW IF NOT EXISTS v_trade_audit AS
SELECT
    t.trade_id,
    t.symbol,
    t.side,
    t.entry_timestamp,
    t.exit_timestamp,
    t.duration_seconds,
    t.entry_price,
    t.exit_price,
    t.quantity,
    t.realized_pnl,
    t.net_pnl,
    t.exit_reason,
    t.is_winner,
    ed.signal_values as entry_signal,
    ed.risk_checks as entry_risk_checks,
    ed.result_reason as entry_reason,
    xd.signal_values as exit_signal,
    xd.result_reason as exit_reason_detail
FROM trades t
JOIN decisions ed ON t.entry_decision_id = ed.decision_id
JOIN decisions xd ON t.exit_decision_id = xd.decision_id;

-- Daily P&L summary
CREATE VIEW IF NOT EXISTS v_daily_pnl AS
SELECT
    date(exit_timestamp) as trade_date,
    COUNT(*) as trade_count,
    SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses,
    ROUND(100.0 * SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate,
    SUM(CAST(realized_pnl AS REAL)) as gross_pnl,
    SUM(CAST(net_pnl AS REAL)) as net_pnl,
    SUM(CAST(total_commission AS REAL)) as total_commission,
    SUM(CAST(total_slippage AS REAL)) as total_slippage
FROM trades
GROUP BY date(exit_timestamp)
ORDER BY trade_date DESC;

-- Risk rejection analysis
CREATE VIEW IF NOT EXISTS v_risk_rejections AS
SELECT
    date(timestamp) as rejection_date,
    symbol,
    COUNT(*) as rejection_count,
    result_reason
FROM decisions
WHERE result = 'risk_rejected'
GROUP BY date(timestamp), symbol, result_reason
ORDER BY rejection_date DESC, rejection_count DESC;

-- =============================================================================
-- REFLEXION VIEWS: Learning analytics
-- =============================================================================

-- Lessons learned summary by type
CREATE VIEW IF NOT EXISTS v_lessons_by_type AS
SELECT
    reflection_type,
    COUNT(*) as lesson_count,
    AVG(confidence) as avg_confidence,
    SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as from_wins,
    SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as from_losses,
    AVG(pnl_percent) as avg_trade_pnl
FROM reflections
GROUP BY reflection_type
ORDER BY lesson_count DESC;

-- High-confidence lessons for quick reference
CREATE VIEW IF NOT EXISTS v_top_lessons AS
SELECT
    symbol,
    market_regime,
    reflection_type,
    lesson_learned,
    confidence,
    pnl_percent,
    created_at
FROM reflections
WHERE confidence >= 0.7
ORDER BY confidence DESC, created_at DESC
LIMIT 50;

-- Learning progress over time
CREATE VIEW IF NOT EXISTS v_learning_progress AS
SELECT
    date(created_at) as learn_date,
    COUNT(*) as reflections_generated,
    AVG(confidence) as avg_confidence,
    SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as analyzed_wins,
    SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as analyzed_losses
FROM reflections
GROUP BY date(created_at)
ORDER BY learn_date DESC;

-- =============================================================================
-- PHANTOM TRADES TABLE: Hypothetical trades we DIDN'T take (Validation Layer)
-- Tracks what WOULD have happened if we'd traded B/C grade setups
-- =============================================================================
CREATE TABLE IF NOT EXISTS phantom_trades (
    phantom_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,            -- Links to the HOLD decision
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,              -- When we would have entered

    -- Setup context at decision time
    setup_grade TEXT NOT NULL,            -- A+, A, B, C
    signal_type TEXT NOT NULL,            -- long, short
    hypothetical_entry REAL NOT NULL,     -- Price we would have entered at

    -- Outcome tracking (filled in by background job)
    price_after_1h REAL,
    price_after_4h REAL,
    price_after_24h REAL,
    price_after_48h REAL,

    -- Stop analysis
    chandelier_stop_price REAL,           -- Where Chandelier would have been
    hard_stop_price REAL,                 -- Where hard stop would have been
    would_hit_chandelier INTEGER,         -- 0/1: Did price hit chandelier?
    would_hit_hard_stop INTEGER,          -- 0/1: Did price hit hard stop?
    chandelier_hit_time TEXT,             -- When chandelier would have triggered
    hard_stop_hit_time TEXT,              -- When hard stop would have triggered

    -- Calculated outcomes
    max_favorable_excursion REAL,         -- Best price reached (MFE)
    max_adverse_excursion REAL,           -- Worst price reached (MAE)
    phantom_exit_price REAL,              -- Where we would have exited
    phantom_pnl_percent REAL,             -- Hypothetical P&L %
    duration_to_exit_hours REAL,          -- How long until exit

    -- Verdict
    verdict TEXT NOT NULL DEFAULT 'pending',  -- pending, regret, relief, neutral
    verdict_reason TEXT,

    -- Metadata
    created_at TEXT NOT NULL,
    updated_at TEXT,

    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

-- Indexes for phantom trade analysis
CREATE INDEX IF NOT EXISTS idx_phantom_symbol ON phantom_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_phantom_grade ON phantom_trades(setup_grade);
CREATE INDEX IF NOT EXISTS idx_phantom_verdict ON phantom_trades(verdict);
CREATE INDEX IF NOT EXISTS idx_phantom_timestamp ON phantom_trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_phantom_decision ON phantom_trades(decision_id);

-- View: Phantom trade summary by grade
CREATE VIEW IF NOT EXISTS v_phantom_by_grade AS
SELECT
    setup_grade,
    COUNT(*) as total_phantoms,
    SUM(CASE WHEN verdict = 'regret' THEN 1 ELSE 0 END) as regrets,
    SUM(CASE WHEN verdict = 'relief' THEN 1 ELSE 0 END) as reliefs,
    ROUND(100.0 * SUM(CASE WHEN verdict = 'regret' THEN 1 ELSE 0 END) / COUNT(*), 1) as regret_rate,
    ROUND(AVG(phantom_pnl_percent), 2) as avg_phantom_pnl,
    ROUND(AVG(duration_to_exit_hours), 1) as avg_duration_hours
FROM phantom_trades
WHERE verdict != 'pending'
GROUP BY setup_grade
ORDER BY setup_grade;
"""


# =============================================================================
# Utility Functions
# =============================================================================

def decimal_to_str(d: Optional[Decimal]) -> Optional[str]:
    """Convert Decimal to string for storage, preserving None"""
    return str(d) if d is not None else None


def str_to_decimal(s: Optional[str]) -> Optional[Decimal]:
    """Convert string back to Decimal, preserving None"""
    return Decimal(s) if s is not None else None
