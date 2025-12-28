#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
ArgusNexus V4 - THE UNIFIED TRADER (Professional Mode)
═══════════════════════════════════════════════════════════════════════════════

THIS IS THE ONLY TRADING SCRIPT. Use for PAPER and LIVE.

"Trade like a professional human, but 24/7 with faster decisions."

═══════════════════════════════════════════════════════════════════════════════
PROFESSIONAL MULTI-TIMEFRAME ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│  CONTEXT LAYER (Daily)     - Overall trend bias, key support/resistance    │
│  SETUP LAYER (4h)          - Donchian breakout formation, trend alignment  │
│  TIMING LAYER (1h)         - Entry trigger confirmation, momentum check    │
│  EXECUTION                 - Only A+/A/B setups with all layers aligned    │
└─────────────────────────────────────────────────────────────────────────────┘

Setup Grades (Position Sizing):
- A+ : All 3 TFs aligned, strong trend, volume > 2x → 1.5x base position
- A  : 2 TFs aligned, decent trend, volume confirms → 1.0x base position
- B  : 1 TF signal, some alignment → 0.5x base position
- C  : Conflicting signals → NO TRADE

═══════════════════════════════════════════════════════════════════════════════
KEY FEATURES
═══════════════════════════════════════════════════════════════════════════════

1. PROFESSIONAL TRADING (NEW):
   - Multi-timeframe analysis: Daily → 4h → 1h
   - Only trades high-quality setups (grades A+, A, B)
   - Conviction-based position sizing
   - Waits for timing layer confirmation

2. LEARNING SYSTEM:
   - RegimeDetector: 14 market classifications
   - ConfidenceScorer: Multi-factor signal quality
   - PPOAgent: Adaptive position sizing

3. SESSION MANAGEMENT:
   - Dead zone detection (22:00-00:00 UTC)
   - Session transitions (Asia/London/NY)
   - Position multipliers per session

4. PORTFOLIO CONTROLS:
   - 8 symbols: BTC, ETH, XRP, SOL, DOGE, ADA, BCH, LINK
   - Max 25% per symbol (grade-adjusted)
   - Unified drawdown tracking

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════

    # Paper trading (default)
    python scripts/live_unified_trader.py --capital 4000

    # Custom symbols
    python scripts/live_unified_trader.py --symbols BTC-USD,ETH-USD,SOL-USD

Environment Variables:
    DISCORD_WEBHOOK_URL - Discord notifications

State: runtime/paper_trader_state.json
Logs:  runtime/unified_trader.log
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import signal
import logging
import argparse
import asyncio
import json
import yaml
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import aiohttp

from src.portfolio.portfolio_manager import PortfolioManager, PortfolioState, DDState, RebalanceOrder
from src.execution import PaperExecutor
from src.truth.logger import TruthLogger
from src.truth.schema import DecisionResult, OrderSide, ExitReason
from src.data.loader import fetch_coinbase_data
from src.notifier import DiscordNotifier
from src.risk.manager import RiskManager, RiskConfig

# Twitter Integration
from src.twitter.hooks import TwitterHook, TwitterHookConfig, create_twitter_hook

# Learning System imports
from src.learning import (
    RegimeDetector,
    ConfidenceScorer,
    PPOAgent,
    PPOConfig,
    MarketRegime,
    SignalQuality,
)

# Session Management (24/7 Perpetual Trading)
from src.session import (
    SessionManager,
    SessionConfig,
    SessionState,
    MarketSession,
    LiquidityMonitor,
)

# Professional Multi-Timeframe Trading System
from src.strategy.professional import (
    MultiTimeframeAnalyzer,
    ProfessionalSetup,
    SetupGrade,
    TrendBias,
)

# Price Zone Filter - Distribution Zone Detection
from src.strategy.price_zone import get_price_zone_filter, PriceZone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("runtime/unified_trader.log")
    ]
)
logger = logging.getLogger(__name__)

# Paths - use paper_trader_state.json for dashboard compatibility
STATE_PATH = Path("runtime/paper_trader_state.json")
SNAPSHOT_PATH = Path("runtime/unified_snapshots.jsonl")

# Fee structure
FEES = {
    "entry": 0.004,   # 0.4% entry fee
    "exit": 0.002,    # 0.2% exit fee
    "slippage": 0.001 # 0.1% slippage
}

# Default symbols
DEFAULT_SYMBOLS = [
    "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD",
    "DOGE-USD", "ADA-USD", "BCH-USD", "LINK-USD"
]

# Global shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info("Shutdown signal received. Finishing current tick...")
    shutdown_requested = True


@dataclass
class SymbolPosition:
    """Position for a single symbol."""
    symbol: str
    qty: Decimal = Decimal("0")
    cost_basis: Decimal = Decimal("0")
    entry_time: Optional[str] = None
    current_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    regime: str = "unknown"
    # Safety net stops (P0 protection)
    hard_stop_price: Optional[Decimal] = None  # Emergency exit if breached
    hard_stop_pct: Decimal = Decimal("0.05")   # Default 5% from entry
    chandelier_stop: Optional[Decimal] = None   # Dynamic trailing stop
    highest_high_since_entry: Optional[Decimal] = None  # For chandelier calc (LONG)
    # SHORT position support (v6.1)
    is_short_position: bool = False  # True for shorts, False for longs
    lowest_low_since_entry: Optional[Decimal] = None  # For chandelier calc (SHORT)

    def calculate_hard_stop(self) -> Decimal:
        """Calculate hard stop at X% from entry (below for LONG, above for SHORT)."""
        if self.cost_basis > 0:
            if self.is_short_position:
                # SHORT: stop ABOVE entry (price going up is bad)
                return self.cost_basis * (1 + self.hard_stop_pct)
            else:
                # LONG: stop BELOW entry (price going down is bad)
                return self.cost_basis * (1 - self.hard_stop_pct)
        return Decimal("0")

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "qty": float(self.qty),
            "cost_basis": float(self.cost_basis),
            "entry_time": self.entry_time,
            "current_price": float(self.current_price),
            "unrealized_pnl": float(self.unrealized_pnl),
            "regime": self.regime,
            "hard_stop_price": float(self.hard_stop_price) if self.hard_stop_price else None,
            "hard_stop_pct": float(self.hard_stop_pct),
            "chandelier_stop": float(self.chandelier_stop) if self.chandelier_stop else None,
            "highest_high_since_entry": float(self.highest_high_since_entry) if self.highest_high_since_entry else None,
            "is_short_position": self.is_short_position,
            "lowest_low_since_entry": float(self.lowest_low_since_entry) if self.lowest_low_since_entry else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolPosition":
        pos = cls(
            symbol=data["symbol"],
            qty=Decimal(str(data.get("qty", 0))),
            cost_basis=Decimal(str(data.get("cost_basis", 0))),
            entry_time=data.get("entry_time"),
            current_price=Decimal(str(data.get("current_price", 0))),
            unrealized_pnl=Decimal(str(data.get("unrealized_pnl", 0))),
            regime=data.get("regime", "unknown"),
            hard_stop_price=Decimal(str(data["hard_stop_price"])) if data.get("hard_stop_price") else None,
            hard_stop_pct=Decimal(str(data.get("hard_stop_pct", 0.05))),
            chandelier_stop=Decimal(str(data["chandelier_stop"])) if data.get("chandelier_stop") else None,
            highest_high_since_entry=Decimal(str(data["highest_high_since_entry"])) if data.get("highest_high_since_entry") else None,
            is_short_position=data.get("is_short_position", False),
            lowest_low_since_entry=Decimal(str(data["lowest_low_since_entry"])) if data.get("lowest_low_since_entry") else None
        )
        # Auto-calculate hard stop if not set
        if pos.hard_stop_price is None and pos.cost_basis > 0:
            pos.hard_stop_price = pos.calculate_hard_stop()
        return pos


@dataclass
class UnifiedPortfolioState:
    """Unified state for entire multi-symbol portfolio."""
    total_equity: Decimal
    cash: Decimal
    high_water_mark: Decimal
    starting_capital: Decimal = Decimal("10000")
    positions: Dict[str, SymbolPosition] = field(default_factory=dict)
    dd_state: str = "normal"
    current_dd_pct: Decimal = Decimal("0")
    last_update: str = ""

    def to_dict(self) -> dict:
        return {
            "total_equity": float(self.total_equity),
            "cash": float(self.cash),
            "high_water_mark": float(self.high_water_mark),
            "starting_capital": float(self.starting_capital),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "dd_state": self.dd_state,
            "current_dd_pct": float(self.current_dd_pct),
            "last_update": self.last_update,
            "engine": "unified_mtf",
            "symbols": list(self.positions.keys()) if self.positions else []
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedPortfolioState":
        positions = {}
        for symbol, pos_data in data.get("positions", {}).items():
            positions[symbol] = SymbolPosition.from_dict(pos_data)

        return cls(
            total_equity=Decimal(str(data.get("total_equity", 0))),
            cash=Decimal(str(data.get("cash", 0))),
            high_water_mark=Decimal(str(data.get("high_water_mark", 0))),
            starting_capital=Decimal(str(data.get("starting_capital", data.get("high_water_mark", 10000)))),
            positions=positions,
            dd_state=data.get("dd_state", "normal"),
            current_dd_pct=Decimal(str(data.get("current_dd_pct", 0))),
            last_update=data.get("last_update", "")
        )

    def get_position_value(self) -> Decimal:
        """Total value of all positions."""
        return sum(p.qty * p.current_price for p in self.positions.values())

    def get_position_count(self) -> int:
        """Number of open positions."""
        return sum(1 for p in self.positions.values() if p.qty > 0)

    def recalculate_equity(self):
        """Recalculate total equity from cash + positions."""
        self.total_equity = self.cash + self.get_position_value()
        if self.total_equity > self.high_water_mark:
            self.high_water_mark = self.total_equity
        if self.high_water_mark > 0:
            self.current_dd_pct = (self.high_water_mark - self.total_equity) / self.high_water_mark * 100


@dataclass
class SymbolSignal:
    """Trading signal for a symbol."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    regime: str  # bull, bear, sideways
    target_alloc_pct: Decimal
    current_price: Decimal
    reason: str
    context: Dict[str, Any] = field(default_factory=dict)
    signal_type: Optional[str] = None  # long, short, exit_long, exit_short (v6.1)

    @property
    def priority(self) -> int:
        """Signal priority for allocation. Higher = more priority."""
        regime_priority = {"bull": 3, "sideways": 2, "bear": 1, "unknown": 0}
        return regime_priority.get(self.regime, 0)


# =============================================================================
# ADAPTIVE SCANNING SYSTEM
# =============================================================================
# Mimics professional trader behavior:
# - NORMAL: Check every 60 min (no setups forming)
# - ALERT:  Check every 15 min (setup forming on 4h)
# - TRIGGER: Check every 5 min (price near breakout level)
# =============================================================================

class ScanMode(Enum):
    """Adaptive scanning modes based on market conditions."""
    NORMAL = "normal"     # 60 min - routine scanning
    ALERT = "alert"       # 15 min - setup forming
    TRIGGER = "trigger"   # 5 min  - imminent breakout

# Scan intervals in seconds
SCAN_INTERVALS = {
    ScanMode.NORMAL: 30,     # 30 seconds (was 60 minutes)
    ScanMode.ALERT: 30,      # 30 seconds (was 15 minutes)
    ScanMode.TRIGGER: 30,    # 30 seconds (was 5 minutes)
}


@dataclass
class SetupProximity:
    """Tracks how close a symbol is to a breakout."""
    symbol: str
    current_price: Decimal
    upper_channel: Optional[Decimal]  # Resistance (breakout level)
    lower_channel: Optional[Decimal]  # Support (breakdown level)
    distance_to_breakout_pct: float   # % distance to upper channel
    distance_to_breakdown_pct: float  # % distance to lower channel
    grade: str                         # Current setup grade
    trend_aligned: bool
    momentum_aligned: bool
    volume_confirmed: bool

    @property
    def is_near_breakout(self) -> bool:
        """Within 1.5% of breakout level."""
        return 0 < self.distance_to_breakout_pct <= 1.5

    @property
    def is_forming_setup(self) -> bool:
        """Setup is forming (within 3% and aligned)."""
        return (0 < self.distance_to_breakout_pct <= 3.0 and
                self.trend_aligned and
                self.grade in ("A+", "A", "B"))

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "current_price": float(self.current_price),
            "upper_channel": float(self.upper_channel) if self.upper_channel else None,
            "lower_channel": float(self.lower_channel) if self.lower_channel else None,
            "distance_to_breakout_pct": self.distance_to_breakout_pct,
            "distance_to_breakdown_pct": self.distance_to_breakdown_pct,
            "grade": self.grade,
            "is_near_breakout": self.is_near_breakout,
            "is_forming_setup": self.is_forming_setup,
        }


class UnifiedPortfolioTrader:
    """
    Unified multi-symbol portfolio trader.

    Manages a single capital pool across multiple symbols with:
    - Portfolio-level allocation decisions
    - Unified drawdown tracking
    - Signal prioritization
    - Position sizing based on available capital
    """

    def __init__(
        self,
        symbols: List[str],
        starting_capital: Decimal,
        interval: str = "1h",
        max_position_pct: float = 0.25,  # Max 25% per symbol
        lookback_days: int = 365
    ):
        self.symbols = symbols
        self.starting_capital = starting_capital
        self.interval = interval
        self.max_position_pct = Decimal(str(max_position_pct))
        self.lookback_days = lookback_days

        # Portfolio managers per symbol (for signal generation)
        self.symbol_managers: Dict[str, PortfolioManager] = {}
        for symbol in symbols:
            self.symbol_managers[symbol] = PortfolioManager(
                rebalance_cooldown_days=0  # We handle cooldown at portfolio level
            )

        # Single paper executor for the unified portfolio
        self.executor = PaperExecutor(
            starting_balance=starting_capital,
            base_slippage_pct=Decimal("0.001"),
            fee_rate=Decimal("0.004")
        )

        # Truth logger
        self.truth_logger: Optional[TruthLogger] = None

        # Discord notifier
        self.notifier: Optional[DiscordNotifier] = None
        try:
            self.notifier = DiscordNotifier()
            logger.info("Discord alerts enabled")
        except Exception as e:
            logger.warning(f"Discord notifications disabled: {e}")

        # Twitter Integration (always enabled)
        self.twitter_hook: Optional[TwitterHook] = None
        try:
            import os
            xai_api_key = os.getenv("XAI_API_KEY")
            self.twitter_hook = create_twitter_hook(
                simulation=False,  # LIVE mode - actually post tweets
                decision_base_url="http://localhost:8000/decision",
                xai_api_key=xai_api_key,
            )
            logger.info("Twitter integration initialized (LIVE mode)")
        except Exception as e:
            logger.warning(f"Twitter integration disabled: {e}")

        # Learning System Components
        logger.info("")
        logger.info("=" * 50)
        logger.info("LEARNING SYSTEM: ENABLED")
        logger.info("=" * 50)

        # Regime Detector - identifies 14 market conditions
        self.regime_detector = RegimeDetector()
        logger.info("  [1/3] Regime Detector: 14 market classifications")

        # Confidence Scorer - multi-factor signal quality
        self.confidence_scorer = ConfidenceScorer(db_path="data/v4_live_paper.db")
        logger.info("  [2/3] Confidence Scorer: Multi-factor quality assessment")

        # PPO Agent - adaptive position sizing
        ppo_config = PPOConfig(
            learning_rate=0.0003,
            gamma=0.99,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )
        self.ppo_agent = PPOAgent(config=ppo_config)
        logger.info("  [3/3] PPO Agent: Adaptive position sizing (burn-in: 20 trades)")
        logger.info("")

        # Session Manager (24/7 Perpetual Trading)
        # Load session config from config.yaml
        session_config = SessionConfig()
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
                if full_config and "session" in full_config:
                    session_config = SessionConfig.from_dict(full_config["session"])
                    logger.info(f"  Session config loaded from config.yaml (dead_zone_start_hour={session_config.dead_zone_start_hour})")
        self.session_manager = SessionManager(config=session_config)
        self.liquidity_monitor = LiquidityMonitor()
        session_info = self.session_manager.get_session_info()
        logger.info("SESSION MANAGEMENT: ENABLED")
        logger.info(f"  Current Session: {session_info['current_session'].upper()}")
        logger.info(f"  Dead Zone: {'YES' if session_info['is_dead_zone'] else 'No'}")
        logger.info("")

        # Professional Multi-Timeframe System
        self.mtf_analyzer = MultiTimeframeAnalyzer(
            entry_period=55,
            exit_period=55,
            atr_period=14,
            chandelier_multiplier=3.0,
            trend_sma_period=200,
            volume_avg_period=20,
            volume_multiplier=1.5,
            adx_period=14,
            adx_threshold=25.0,
            min_volume_for_a_plus=2.0,
            min_adx_for_a_plus=30.0,
        )
        logger.info("PROFESSIONAL TRADING: ENABLED")
        logger.info("  Context Layer: Daily (trend bias, key levels)")
        logger.info("  Setup Layer: 4h (Donchian breakout formation)")
        logger.info("  Timing Layer: 1h (entry trigger)")
        logger.info("  Grades: A+ (1.5x) | A (1.0x) | B (0.5x) | C (no trade)")
        logger.info("")

        # State
        self.state: Optional[UnifiedPortfolioState] = None

        # Cooldown tracking (portfolio-level)
        self.last_trade_time: Dict[str, datetime] = {}
        self.trade_cooldown = timedelta(hours=1)  # 1 hour cooldown per symbol

        # Adaptive Scanning System
        self.current_scan_mode = ScanMode.NORMAL
        self.previous_scan_mode = ScanMode.NORMAL
        self.scan_mode_since: datetime = datetime.now(timezone.utc)
        self.setup_proximities: Dict[str, SetupProximity] = {}
        logger.info("ADAPTIVE SCANNING: ENABLED")
        logger.info("  NORMAL:  60 min (routine)")
        logger.info("  ALERT:   15 min (setup forming)")
        logger.info("  TRIGGER:  5 min (near breakout)")
        logger.info("")

    async def initialize(self):
        """Initialize the trader."""
        # Ensure runtime directory
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Initialize truth logger
        db_path = Path("data/v4_live_paper.db")
        self.truth_logger = TruthLogger(str(db_path))
        await self.truth_logger.initialize()
        logger.info(f"Truth Engine initialized at {db_path}")

        # Start Twitter integration
        if self.twitter_hook:
            await self.twitter_hook.start()
            logger.info("Twitter hook started - tweets will be posted on trades")

        # Load or create state
        self.state = self.load_state()
        if self.state is None:
            self.state = UnifiedPortfolioState(
                total_equity=self.starting_capital,
                cash=self.starting_capital,
                high_water_mark=self.starting_capital,
                starting_capital=self.starting_capital,
                positions={},
                dd_state="normal",
                current_dd_pct=Decimal("0"),
                last_update=datetime.now(timezone.utc).isoformat()
            )
            self.save_state()
            logger.info(f"Created initial state: ${float(self.starting_capital):,.2f}")
        else:
            logger.info(f"Loaded existing state: equity=${float(self.state.total_equity):,.2f}, "
                       f"positions={self.state.get_position_count()}")

    def load_state(self) -> Optional[UnifiedPortfolioState]:
        """Load state from disk."""
        if STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text())
                return UnifiedPortfolioState.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return None

    def save_state(self):
        """Save state to disk atomically."""
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state.to_dict(), indent=2))
        tmp.replace(STATE_PATH)

    async def fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch candle data for a symbol."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: fetch_coinbase_data(
                    symbol=symbol,
                    interval=self.interval,
                    lookback_days=self.lookback_days
                )
            )
            # fetch_coinbase_data returns (DataFrame, DataQualityReport) tuple
            df, quality_report = result
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    async def fetch_multi_timeframe_data(
        self,
        symbol: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for all timeframes needed for professional analysis.

        Returns:
            Dict with keys "1d", "4h", "1h" and DataFrame values
        """
        timeframes = {
            "1d": {"interval": "1d", "lookback": 365},   # 1 year of daily
            "4h": {"interval": "4h", "lookback": 90},    # 90 days of 4h
            "1h": {"interval": "1h", "lookback": 30},    # 30 days of 1h
        }

        results = {}

        # Fetch all timeframes in parallel
        async def fetch_tf(tf_key: str, tf_config: dict) -> tuple:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: fetch_coinbase_data(
                        symbol=symbol,
                        interval=tf_config["interval"],
                        lookback_days=tf_config["lookback"]
                    )
                )
                df, _ = result
                return tf_key, df
            except Exception as e:
                logger.warning(f"Failed to fetch {tf_key} data for {symbol}: {e}")
                return tf_key, None

        tasks = [fetch_tf(k, v) for k, v in timeframes.items()]
        fetched = await asyncio.gather(*tasks)

        for tf_key, df in fetched:
            results[tf_key] = df

        return results

    async def fetch_spot_prices(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch real-time spot prices from Coinbase API for all symbols.

        This provides independent verification of prices at decision time,
        separate from the candle close prices used for strategy calculations.

        Returns:
            Dict mapping symbol to {price, timestamp, bid, ask}
        """
        spot_prices = {}
        fetch_time = datetime.now(timezone.utc)

        async with aiohttp.ClientSession() as session:
            for symbol in self.symbols:
                try:
                    # Use Coinbase public ticker endpoint
                    url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            spot_prices[symbol] = {
                                "price": float(data.get("price", 0)),
                                "bid": float(data.get("bid", 0)),
                                "ask": float(data.get("ask", 0)),
                                "time": data.get("time", fetch_time.isoformat()),
                                "fetch_time": fetch_time.isoformat(),
                            }
                        else:
                            logger.warning(f"Failed to fetch spot price for {symbol}: HTTP {response.status}")
                except Exception as e:
                    logger.warning(f"Error fetching spot price for {symbol}: {e}")

        return spot_prices

    async def evaluate_symbol(
        self,
        symbol: str,
        mtf_data: Dict[str, pd.DataFrame],
        available_capital: Decimal,
        btc_price: Optional[float] = None
    ) -> SymbolSignal:
        """
        Evaluate a single symbol using Professional Multi-Timeframe analysis.

        Args:
            symbol: Trading pair
            mtf_data: Dict with "1d", "4h", "1h" DataFrames
            available_capital: Available cash for allocation

        Returns:
            SymbolSignal with grade-based action and position sizing
        """
        pm = self.symbol_managers[symbol]

        # Get the 1h data for current price (most recent)
        df_1h = mtf_data.get("1h")
        df_4h = mtf_data.get("4h")
        df_1d = mtf_data.get("1d")

        # Fallback to available data (prefer 1h for timing precision)
        primary_df = None
        if df_1h is not None and len(df_1h) > 0:
            primary_df = df_1h
        elif df_4h is not None and len(df_4h) > 0:
            primary_df = df_4h
        elif df_1d is not None and len(df_1d) > 0:
            primary_df = df_1d

        if primary_df is None or len(primary_df) == 0:
            return SymbolSignal(
                symbol=symbol,
                action="HOLD",
                regime="unknown",
                target_alloc_pct=Decimal("0"),
                current_price=Decimal("0"),
                reason="No data available",
                context={}
            )

        current_price = Decimal(str(primary_df.iloc[-1]["close"]))

        # Get existing position for this symbol
        position = self.state.positions.get(symbol)
        has_position = position is not None and position.qty > 0
        entry_price = position.cost_basis if has_position else None

        if position:
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.cost_basis) * position.qty

        # ========================================
        # PROFESSIONAL MULTI-TIMEFRAME ANALYSIS
        # ========================================
        professional_setup = None
        grade = SetupGrade.C
        grade_reason = "MTF analysis not available"
        position_multiplier = 0.0
        zone_info: Optional[Dict[str, Any]] = None  # Price zone filter data

        try:
            if df_1d is not None and df_4h is not None and df_1h is not None:
                if len(df_1d) >= 55 and len(df_4h) >= 55 and len(df_1h) >= 55:
                    # Get SHORT position data if applicable (v6.1)
                    is_short = position.is_short_position if position else False
                    highest_high = position.highest_high_since_entry if position and not is_short else None
                    lowest_low = position.lowest_low_since_entry if position and is_short else None

                    professional_setup = self.mtf_analyzer.analyze(
                        symbol=symbol,
                        df_daily=df_1d,
                        df_4h=df_4h,
                        df_1h=df_1h,
                        has_position=has_position,
                        entry_price=entry_price,
                        highest_high_since_entry=highest_high,
                        is_short_position=is_short,
                        lowest_low_since_entry=lowest_low,
                    )
                    grade = professional_setup.grade
                    grade_reason = professional_setup.grade_reason
                    position_multiplier = professional_setup.position_multiplier

                    # Log the professional analysis
                    logger.info(f"  [{symbol}] MTF Grade: {grade.value} | "
                               f"Trend Align: {'✓' if professional_setup.trend_alignment else '✗'} | "
                               f"Momentum: {'✓' if professional_setup.momentum_alignment else '✗'} | "
                               f"Volume: {'✓' if professional_setup.volume_confirmation else '✗'}")

                    # ========================================
                    # PRICE ZONE FILTER (Distribution Zone Protection)
                    # ========================================
                    # Override grade to C if in distribution zone (>90% ATH)
                    # Based on backtest: Distribution zone = 8% win rate
                    price_zone_filter = get_price_zone_filter()
                    zone = price_zone_filter.get_zone(
                        symbol=symbol,
                        current_price=float(current_price),
                        btc_price=btc_price
                    )

                    # Capture zone info for decision logging
                    zone_info = zone.to_dict()

                    if zone.is_distribution and grade != SetupGrade.C:
                        original_grade = grade.value
                        grade = SetupGrade.C
                        grade_reason = f"ZONE OVERRIDE ({original_grade}→C): {zone.reason}"
                        position_multiplier = 0.0
                        zone_info["grade_override"] = {"from": original_grade, "to": "C"}
                        logger.warning(f"  [{symbol}] ⚠️ DISTRIBUTION ZONE: {zone.reason}")
                        logger.warning(f"  [{symbol}] Grade downgraded {original_grade} → C (NO TRADE)")
                    elif zone.is_caution:
                        logger.info(f"  [{symbol}] ⚡ CAUTION ZONE: {zone.reason}")
                        if zone.btc_context:
                            logger.info(f"  [{symbol}] BTC Context: {zone.btc_context}")
                else:
                    grade_reason = f"Insufficient bars: 1d={len(df_1d)}, 4h={len(df_4h)}, 1h={len(df_1h)}"
            else:
                missing = []
                if df_1d is None:
                    missing.append("1d")
                if df_4h is None:
                    missing.append("4h")
                if df_1h is None:
                    missing.append("1h")
                grade_reason = f"Missing timeframes: {', '.join(missing)}"

        except Exception as e:
            logger.warning(f"MTF analysis failed for {symbol}: {e}")
            grade_reason = f"MTF error: {str(e)}"

        # ========================================
        # REGIME DETECTION (for context)
        # ========================================
        market_regime = "unknown"
        regime_confidence = 0.5

        try:
            close = float(primary_df.iloc[-1]["close"])
            atr = float(primary_df["close"].rolling(14).apply(
                lambda x: (x.max() - x.min()) / x.mean() * 100 if x.mean() > 0 else 0
            ).iloc[-1]) if len(primary_df) >= 14 else 2.0

            regime_result = self.regime_detector.detect(
                current_price=close,
                atr=atr,
                adx=25.0,
                rsi=50.0,
            )
            market_regime = regime_result.current_regime.value if hasattr(regime_result.current_regime, 'value') else str(regime_result.current_regime)
            regime_confidence = regime_result.confidence
        except Exception as e:
            logger.warning(f"Regime detection failed for {symbol}: {e}")

        # ========================================
        # DETERMINE ACTION BASED ON GRADE
        # ========================================
        action = "HOLD"
        signal_type: Optional[str] = None  # long, short, exit_long, exit_short (v6.1)
        reason = grade_reason

        if grade == SetupGrade.C:
            # No trade - not enough alignment
            action = "HOLD"
            reason = f"Grade C: {grade_reason}"

        elif grade in (SetupGrade.A_PLUS, SetupGrade.A, SetupGrade.B):
            # Check the professional setup's signal
            if professional_setup:
                from src.strategy.donchian import Signal
                if professional_setup.signal == Signal.LONG:
                    action = "BUY"
                    signal_type = "long"
                    reason = f"Grade {grade.value}: {grade_reason}"
                elif professional_setup.signal == Signal.SHORT:
                    # SHORT entry (v6.1) - use SELL action with signal_type to differentiate
                    action = "SELL"
                    signal_type = "short"
                    reason = f"Grade {grade.value} SHORT: {grade_reason}"
                elif professional_setup.signal == Signal.EXIT_LONG:
                    action = "SELL"
                    signal_type = "exit_long"
                    reason = f"Exit long signal: {grade_reason}"
                elif professional_setup.signal == Signal.EXIT_SHORT:
                    # Exit short = BUY to cover
                    action = "BUY"
                    signal_type = "exit_short"
                    reason = f"Exit short signal: {grade_reason}"
                else:
                    action = "HOLD"
                    signal_type = None
                    reason = f"Grade {grade.value} but no entry trigger yet"

        # For exits, also check traditional portfolio manager
        if has_position:
            symbol_equity = available_capital / len(self.symbols)
            symbol_state = PortfolioState(
                total_equity=symbol_equity,
                btc_qty=position.qty if position else Decimal("0"),
                cash=available_capital / len(self.symbols),
                high_water_mark=symbol_equity,
                dd_state=DDState.NORMAL,
                recovery_mode=False,
                bars_in_critical=0,
                sleeve_in_position=has_position,
                sleeve_entry_price=position.cost_basis if position else Decimal("0"),
                last_rebalance_time=datetime.now(timezone.utc),
                last_update=datetime.now(timezone.utc)
            )

            # Check for exit signals
            order, _ = pm.evaluate(
                df=primary_df,
                state=symbol_state,
                current_price=current_price,
                timestamp=datetime.now(timezone.utc)
            )

            if order.action == "SELL":
                action = "SELL"
                reason = order.reason

        # ========================================
        # P0 SAFETY NET: HARD STOP CHECK
        # ========================================
        # Emergency exit if price breaches hard stop (default 5% from entry)
        # LONG: triggers when price falls BELOW stop
        # SHORT: triggers when price rises ABOVE stop
        if has_position and position:
            # Ensure hard stop is set
            if position.hard_stop_price is None and position.cost_basis > 0:
                position.hard_stop_price = position.calculate_hard_stop()
                stop_direction = "+" if position.is_short_position else "-"
                logger.info(f"[SAFETY] {symbol}: Set hard stop at ${float(position.hard_stop_price):,.2f} "
                           f"({stop_direction}{float(position.hard_stop_pct)*100:.0f}% from ${float(position.cost_basis):,.2f})")

            # Check if price breached hard stop (inverse for SHORT)
            hard_stop_triggered = False
            if position.hard_stop_price:
                if position.is_short_position:
                    # SHORT: stop when price goes UP
                    hard_stop_triggered = current_price >= position.hard_stop_price
                else:
                    # LONG: stop when price goes DOWN
                    hard_stop_triggered = current_price <= position.hard_stop_price

            if hard_stop_triggered:
                action = "BUY" if position.is_short_position else "SELL"
                if position.is_short_position:
                    loss_pct = ((position.cost_basis - current_price) / position.cost_basis) * 100
                    compare_op = ">="
                else:
                    loss_pct = ((current_price - position.cost_basis) / position.cost_basis) * 100
                    compare_op = "<="
                reason = (f"🚨 HARD STOP TRIGGERED: Price ${float(current_price):,.2f} {compare_op} "
                         f"stop ${float(position.hard_stop_price):,.2f} "
                         f"(loss: {float(loss_pct):.1f}%)")
                logger.warning(f"[SAFETY] {symbol}: {reason}")

            # Check if price breached chandelier stop (primary exit, inverse for SHORT)
            elif position.chandelier_stop:
                chandelier_triggered = False
                if position.is_short_position:
                    # SHORT: exit when price goes UP above chandelier
                    chandelier_triggered = current_price >= position.chandelier_stop
                else:
                    # LONG: exit when price goes DOWN below chandelier
                    chandelier_triggered = current_price <= position.chandelier_stop

                if chandelier_triggered:
                    action = "BUY" if position.is_short_position else "SELL"
                    if position.is_short_position:
                        pnl_pct = ((position.cost_basis - current_price) / position.cost_basis) * 100
                        ref_price = position.lowest_low_since_entry
                        ref_label = "LL"
                        compare_op = ">="
                    else:
                        pnl_pct = ((current_price - position.cost_basis) / position.cost_basis) * 100
                        ref_price = position.highest_high_since_entry
                        ref_label = "HH"
                        compare_op = "<="
                    reason = (f"⚡ CHANDELIER EXIT: Price ${float(current_price):,.2f} {compare_op} "
                             f"stop ${float(position.chandelier_stop):,.2f} "
                             f"({ref_label}: ${float(ref_price):,.2f}, P&L: {float(pnl_pct):+.1f}%)")
                    logger.warning(f"[CHANDELIER] {symbol}: {reason}")

        # ========================================
        # BUILD CONTEXT FOR LOGGING
        # ========================================
        # Build reasoning array for dashboard visibility (P0 fix)
        reasoning = []
        if professional_setup:
            # Trend alignment
            if professional_setup.trend_alignment:
                reasoning.append(f"Trend aligned across timeframes ({professional_setup.context_layer.trend_bias.value})")
            else:
                reasoning.append("Trend NOT aligned - conflicting TF signals")
            # Momentum
            if professional_setup.momentum_alignment:
                reasoning.append("Momentum confirms direction")
            else:
                reasoning.append("Momentum NOT confirming")
            # Volume
            if professional_setup.volume_confirmation:
                vol_ratio = professional_setup.timing_layer.volume_ratio or 0
                reasoning.append(f"Volume confirmed ({vol_ratio:.1f}x avg)")
            else:
                reasoning.append("Volume NOT confirmed")
            # Add grade reason
            reasoning.append(f"Grade {grade.value}: {grade_reason}")
        else:
            reasoning.append(grade_reason)

        # Extract daily ATR from the context layer's donchian result
        daily_atr = 0.0
        if professional_setup and professional_setup.context_layer.donchian_result:
            daily_atr = float(professional_setup.context_layer.donchian_result.context.atr)

        context = {
            "professional": {
                "grade": grade.value,
                "grade_reason": grade_reason,
                "reasoning": reasoning,  # P0: Include reasoning array
                "position_multiplier": position_multiplier,
                "trend_alignment": professional_setup.trend_alignment if professional_setup else False,
                "momentum_alignment": professional_setup.momentum_alignment if professional_setup else False,
                "volume_confirmation": professional_setup.volume_confirmation if professional_setup else False,
                "zone_filter": zone_info,  # Price zone filter data
                "daily_atr": daily_atr,  # ATR from daily timeframe
            } if professional_setup else {"grade": "C", "reason": grade_reason, "reasoning": reasoning, "daily_atr": 0.0},
            "learning": {
                "regime": market_regime,
                "regime_confidence": regime_confidence,
            },
            "timeframes": {
                "daily_trend": professional_setup.context_layer.trend_bias.value if professional_setup else "unknown",
                "h4_signal": professional_setup.setup_layer.signal.value if professional_setup else "hold",
                "h1_signal": professional_setup.timing_layer.signal.value if professional_setup else "hold",
            } if professional_setup else {},
        }

        # Add full professional setup to context for dashboard
        if professional_setup:
            context["setup_details"] = professional_setup.to_dict()

        # Add safety stop levels to context (P0 visibility)
        if has_position and position:
            context["safety_stops"] = {
                "hard_stop_price": float(position.hard_stop_price) if position.hard_stop_price else None,
                "hard_stop_pct": float(position.hard_stop_pct),
                "chandelier_stop": float(position.chandelier_stop) if position.chandelier_stop else None,
                "highest_high_since_entry": float(position.highest_high_since_entry) if position.highest_high_since_entry else None,
                "entry_price": float(position.cost_basis),
                "current_price": float(current_price),
                "distance_to_hard_stop_pct": float((current_price - position.hard_stop_price) / current_price * 100) if position.hard_stop_price else None
            }

        return SymbolSignal(
            symbol=symbol,
            action=action,
            regime=market_regime,
            target_alloc_pct=Decimal(str(position_multiplier)),
            current_price=current_price,
            reason=reason,
            context=context,
            signal_type=signal_type  # v6.1: long, short, exit_long, exit_short
        )

    def allocate_capital(
        self,
        signals: List[SymbolSignal],
        available_cash: Decimal
    ) -> Dict[str, Decimal]:
        """
        Allocate capital across symbols based on professional grades.

        Grade-Based Position Sizing:
        - A+ : 1.5x base allocation (high conviction)
        - A  : 1.0x base allocation (standard)
        - B  : 0.5x base allocation (reduced)
        - C  : 0.0x (no trade)

        Priority: A+ signals > A signals > B signals

        Returns dict of symbol -> allocation amount
        """
        allocations: Dict[str, Decimal] = {}

        # Filter to BUY signals with valid grades (A+, A, B)
        buy_signals = [s for s in signals if s.action == "BUY"]

        # Sort by grade quality: A+ > A > B
        def grade_priority(signal: SymbolSignal) -> int:
            grade = signal.context.get("professional", {}).get("grade", "C")
            return {"A+": 3, "A": 2, "B": 1, "C": 0}.get(grade, 0)

        buy_signals.sort(key=grade_priority, reverse=True)

        if not buy_signals:
            return allocations

        # Base allocation per symbol (equal split)
        base_per_symbol = self.state.high_water_mark * self.max_position_pct

        remaining_cash = available_cash

        for signal in buy_signals:
            if remaining_cash <= 0:
                break

            # Get grade-based position multiplier
            position_multiplier = float(signal.target_alloc_pct)  # This now holds the grade multiplier
            if position_multiplier <= 0:
                logger.debug(f"{signal.symbol}: Grade C, skipping")
                continue

            grade = signal.context.get("professional", {}).get("grade", "?")

            # Check cooldown
            last_trade = self.last_trade_time.get(signal.symbol)
            if last_trade and datetime.now(timezone.utc) - last_trade < self.trade_cooldown:
                logger.debug(f"{signal.symbol}: Cooldown active, skipping")
                continue

            # Check if already at max position
            existing_position = self.state.positions.get(signal.symbol)
            existing_value = Decimal("0")
            if existing_position and existing_position.qty > 0:
                existing_value = existing_position.qty * existing_position.current_price

            # Calculate grade-adjusted max allocation
            grade_adjusted_max = base_per_symbol * Decimal(str(position_multiplier))

            # How much more can we allocate to this symbol?
            remaining_allocation = grade_adjusted_max - existing_value
            if remaining_allocation <= 0:
                logger.debug(f"{signal.symbol}: Already at max position for grade {grade}")
                continue

            # Allocate up to remaining allocation or available cash
            allocation = min(remaining_allocation, remaining_cash)

            # Minimum trade size check ($50)
            if allocation < 50:
                continue

            logger.info(f"  [ALLOC] {signal.symbol}: ${float(allocation):,.0f} "
                       f"(Grade {grade} = {position_multiplier:.1f}x)")

            allocations[signal.symbol] = allocation
            remaining_cash -= allocation

        return allocations

    async def execute_trade(
        self,
        symbol: str,
        action: str,
        amount: Decimal,
        current_price: Decimal,
        signal: SymbolSignal
    ) -> bool:
        """Execute a trade for a symbol. Supports LONG and SHORT positions (v6.1)."""
        try:
            qty = amount / (current_price * (1 + Decimal(str(FEES["slippage"])) + Decimal(str(FEES["entry"]))))

            # Check for existing position state
            existing_pos = self.state.positions.get(symbol)
            has_long = existing_pos is not None and existing_pos.qty > 0 and not existing_pos.is_short_position
            has_short = existing_pos is not None and existing_pos.qty > 0 and existing_pos.is_short_position

            if action == "BUY":
                # ========================================
                # BUY: Either close SHORT or open/add LONG
                # ========================================
                if has_short:
                    # Close SHORT position (buy to cover) - v6.1
                    return await self._close_short_position(symbol, current_price, signal)

                # Open or add to LONG position
                fill_price = current_price * (1 + Decimal(str(FEES["slippage"])))
                fee = qty * fill_price * Decimal(str(FEES["entry"]))
                total_cost = qty * fill_price + fee

                if total_cost > self.state.cash:
                    # Reduce qty to fit
                    qty = self.state.cash / (fill_price * (1 + Decimal(str(FEES["entry"]))))
                    total_cost = qty * fill_price * (1 + Decimal(str(FEES["entry"])))

                if qty <= 0:
                    return False

                # Update state
                self.state.cash -= total_cost

                # Update or create position
                if has_long:
                    # Add to existing LONG position (weighted average cost basis)
                    pos = self.state.positions[symbol]
                    old_cost = pos.qty * pos.cost_basis
                    new_cost = qty * fill_price
                    pos.qty += qty
                    pos.cost_basis = (old_cost + new_cost) / pos.qty
                    pos.current_price = current_price
                else:
                    # New LONG position
                    new_pos = SymbolPosition(
                        symbol=symbol,
                        qty=qty,
                        cost_basis=fill_price,
                        entry_time=datetime.now(timezone.utc).isoformat(),
                        current_price=current_price,
                        unrealized_pnl=Decimal("0"),
                        regime=signal.regime,
                        is_short_position=False  # Explicitly LONG
                    )
                    # Calculate hard stop immediately (P0 safety net)
                    new_pos.hard_stop_price = new_pos.calculate_hard_stop()
                    new_pos.highest_high_since_entry = current_price  # Initialize for chandelier
                    self.state.positions[symbol] = new_pos
                    logger.info(f"[SAFETY] {symbol}: Hard stop set at ${float(new_pos.hard_stop_price):,.2f} "
                               f"(-{float(new_pos.hard_stop_pct)*100:.0f}% from ${float(fill_price):,.2f})")

                # Log to truth engine (convert Decimals in context to floats)
                safe_context = {k: float(v) if isinstance(v, Decimal) else v
                               for k, v in signal.context.items()}
                decision = await self.truth_logger.log_decision(
                    symbol=symbol,
                    strategy_name="unified_portfolio",
                    signal_values={
                        "action": action,
                        "regime": signal.regime,
                        "target_alloc_pct": float(signal.target_alloc_pct)
                    },
                    risk_checks={"portfolio_approved": True},
                    result=DecisionResult.SIGNAL_LONG,
                    result_reason=signal.reason,
                    market_context=safe_context,
                    timestamp=datetime.now(timezone.utc)
                )

                order = await self.truth_logger.log_order(
                    decision_id=decision.decision_id,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=qty,
                    requested_price=current_price
                )

                await self.truth_logger.update_order_fill(
                    order_id=order.order_id,
                    fill_price=fill_price,
                    fill_quantity=qty,
                    commission=fee
                )

                # Get the position's hard stop for DB logging
                pos = self.state.positions.get(symbol)
                hard_stop = pos.hard_stop_price if pos else None

                await self.truth_logger.open_trade(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    entry_order_id=order.order_id,
                    entry_price=fill_price,
                    quantity=qty,
                    stop_loss_price=hard_stop,  # P0: Pass hard stop to DB
                    strategy_name="unified_portfolio"
                )

                self.last_trade_time[symbol] = datetime.now(timezone.utc)

                logger.info(f"BUY {symbol}: {float(qty):.6f} @ ${float(fill_price):,.2f} "
                           f"(${float(total_cost):,.2f} total)")

                # Tweet the trade entry
                if self.twitter_hook:
                    try:
                        professional = signal.context.get("professional", {})
                        await self.twitter_hook.on_decision({
                            "decision_id": decision.decision_id,
                            "symbol": symbol,
                            "result": "signal_long",
                            "signal_values": {
                                "current_price": float(fill_price),
                                "grade": professional.get("grade", "?"),
                                "position_size": float(qty),
                                "cost_usd": float(total_cost),
                            },
                            "risk_checks": {
                                "portfolio_approved": {"passed": True},
                            },
                            "result_reason": signal.reason,
                        })
                        logger.info(f"[TWITTER] Entry tweet scheduled for {symbol}")
                    except Exception as e:
                        logger.warning(f"[TWITTER] Failed to tweet entry: {e}")

                return True

            elif action == "SELL":
                # ========================================
                # SELL: Either open SHORT or close LONG
                # ========================================
                if signal.signal_type == "short" and not has_long:
                    # Open SHORT position (v6.1)
                    return await self._open_short_position(symbol, qty, current_price, signal)

                # Close LONG position (original behavior)
                position = self.state.positions.get(symbol)
                if not position or position.qty <= 0:
                    return False

                sell_qty = position.qty
                fill_price = current_price * (1 - Decimal(str(FEES["slippage"])))
                fee = sell_qty * fill_price * Decimal(str(FEES["exit"]))
                proceeds = sell_qty * fill_price - fee

                # Update state
                self.state.cash += proceeds

                # Calculate P&L
                pnl = (fill_price - position.cost_basis) * sell_qty - fee

                # Log to truth engine (convert Decimals in context to floats)
                safe_context = {k: float(v) if isinstance(v, Decimal) else v
                               for k, v in signal.context.items()}
                decision = await self.truth_logger.log_decision(
                    symbol=symbol,
                    strategy_name="unified_portfolio",
                    signal_values={
                        "action": action,
                        "regime": signal.regime
                    },
                    risk_checks={"portfolio_approved": True},
                    result=DecisionResult.SIGNAL_CLOSE,
                    result_reason=signal.reason,
                    market_context=safe_context,
                    timestamp=datetime.now(timezone.utc)
                )

                order = await self.truth_logger.log_order(
                    decision_id=decision.decision_id,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=sell_qty,
                    requested_price=current_price
                )

                await self.truth_logger.update_order_fill(
                    order_id=order.order_id,
                    fill_price=fill_price,
                    fill_quantity=sell_qty,
                    commission=fee
                )

                # Close trade
                open_trade = await self.truth_logger.get_open_position(symbol)
                if open_trade:
                    await self.truth_logger.close_trade(
                        trade_id=open_trade["trade_id"],
                        exit_order_id=order.order_id,
                        exit_price=fill_price,
                        exit_reason=ExitReason.SIGNAL_EXIT,
                        commission=fee
                    )

                # Clear position
                entry_price = position.cost_basis  # Save before clearing
                position.qty = Decimal("0")

                self.last_trade_time[symbol] = datetime.now(timezone.utc)

                logger.info(f"SELL {symbol}: {float(sell_qty):.6f} @ ${float(fill_price):,.2f} "
                           f"(P&L: ${float(pnl):,.2f})")

                # Tweet the trade exit
                if self.twitter_hook:
                    try:
                        is_winner = pnl > 0
                        pnl_pct = float((fill_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                        await self.twitter_hook.on_trade_closed({
                            "decision_id": decision.decision_id,
                            "symbol": symbol,
                            "entry_price": float(entry_price),
                            "exit_price": float(fill_price),
                            "net_pnl": float(pnl),
                            "pnl_percent": pnl_pct,
                            "exit_reason": signal.reason,
                            "is_winner": is_winner,
                        })
                        result_emoji = "✅" if is_winner else "❌"
                        logger.info(f"[TWITTER] {result_emoji} Exit tweet scheduled for {symbol} (P&L: ${float(pnl):,.2f})")
                    except Exception as e:
                        logger.warning(f"[TWITTER] Failed to tweet exit: {e}")

                return True

        except Exception as e:
            logger.error(f"Trade execution failed for {symbol}: {e}")
            return False

        return False

    # =========================================================================
    # SHORT POSITION HELPERS (v6.1)
    # =========================================================================

    async def _open_short_position(
        self,
        symbol: str,
        qty: Decimal,
        current_price: Decimal,
        signal: SymbolSignal
    ) -> bool:
        """Open a simulated SHORT position with collateral locking."""
        try:
            fill_price = current_price * (1 - Decimal(str(FEES["slippage"])))
            fee = qty * fill_price * Decimal(str(FEES["entry"]))

            # For simulated shorts, lock collateral (100% of position value)
            collateral = qty * fill_price + fee

            if collateral > self.state.cash:
                # Reduce qty to fit available collateral
                qty = self.state.cash / (fill_price * (1 + Decimal(str(FEES["entry"]))))
                collateral = qty * fill_price * (1 + Decimal(str(FEES["entry"])))

            if qty <= 0:
                return False

            # Lock collateral
            self.state.cash -= collateral

            # Create SHORT position
            new_pos = SymbolPosition(
                symbol=symbol,
                qty=qty,
                cost_basis=fill_price,
                entry_time=datetime.now(timezone.utc).isoformat(),
                current_price=current_price,
                unrealized_pnl=Decimal("0"),
                regime=signal.regime,
                is_short_position=True  # Mark as SHORT
            )
            # Calculate hard stop (ABOVE entry for shorts)
            new_pos.hard_stop_price = new_pos.calculate_hard_stop()
            new_pos.lowest_low_since_entry = current_price  # Initialize for chandelier
            self.state.positions[symbol] = new_pos

            logger.info(f"[SAFETY] {symbol}: SHORT hard stop set at ${float(new_pos.hard_stop_price):,.2f} "
                       f"(+{float(new_pos.hard_stop_pct)*100:.0f}% from ${float(fill_price):,.2f})")

            # Log to truth engine
            safe_context = {k: float(v) if isinstance(v, Decimal) else v
                           for k, v in signal.context.items()}
            decision = await self.truth_logger.log_decision(
                symbol=symbol,
                strategy_name="unified_portfolio",
                signal_values={
                    "action": "SELL",
                    "signal_type": "short",
                    "regime": signal.regime,
                    "target_alloc_pct": float(signal.target_alloc_pct)
                },
                risk_checks={"portfolio_approved": True},
                result=DecisionResult.SIGNAL_SHORT,
                result_reason=signal.reason,
                market_context=safe_context,
                timestamp=datetime.now(timezone.utc)
            )

            order = await self.truth_logger.log_order(
                decision_id=decision.decision_id,
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=qty,
                requested_price=current_price
            )

            await self.truth_logger.update_order_fill(
                order_id=order.order_id,
                fill_price=fill_price,
                fill_quantity=qty,
                commission=fee
            )

            await self.truth_logger.open_trade(
                symbol=symbol,
                side=OrderSide.SELL,  # SHORT entry
                entry_order_id=order.order_id,
                entry_price=fill_price,
                quantity=qty,
                stop_loss_price=new_pos.hard_stop_price,
                strategy_name="unified_portfolio"
            )

            self.last_trade_time[symbol] = datetime.now(timezone.utc)

            logger.info(f"SHORT {symbol}: {float(qty):.6f} @ ${float(fill_price):,.2f} "
                       f"(collateral: ${float(collateral):,.2f})")

            return True

        except Exception as e:
            logger.error(f"SHORT entry failed for {symbol}: {e}")
            return False

    async def _close_short_position(
        self,
        symbol: str,
        current_price: Decimal,
        signal: SymbolSignal
    ) -> bool:
        """Close a SHORT position (buy to cover)."""
        try:
            position = self.state.positions.get(symbol)
            if not position or position.qty <= 0 or not position.is_short_position:
                return False

            cover_qty = position.qty
            fill_price = current_price * (1 + Decimal(str(FEES["slippage"])))  # Buy at slightly higher price
            fee = cover_qty * fill_price * Decimal(str(FEES["exit"]))

            # Calculate P&L for SHORT: (entry - exit) * qty
            pnl = (position.cost_basis - fill_price) * cover_qty - fee

            # Return collateral + P&L (collateral was entry_price * qty)
            collateral_return = position.cost_basis * cover_qty
            self.state.cash += collateral_return + pnl

            # Log to truth engine
            safe_context = {k: float(v) if isinstance(v, Decimal) else v
                           for k, v in signal.context.items()}
            decision = await self.truth_logger.log_decision(
                symbol=symbol,
                strategy_name="unified_portfolio",
                signal_values={
                    "action": "BUY",
                    "signal_type": "exit_short",
                    "regime": signal.regime
                },
                risk_checks={"portfolio_approved": True},
                result=DecisionResult.SIGNAL_CLOSE,
                result_reason=signal.reason,
                market_context=safe_context,
                timestamp=datetime.now(timezone.utc)
            )

            order = await self.truth_logger.log_order(
                decision_id=decision.decision_id,
                symbol=symbol,
                side=OrderSide.BUY,  # Buy to cover
                quantity=cover_qty,
                requested_price=current_price
            )

            await self.truth_logger.update_order_fill(
                order_id=order.order_id,
                fill_price=fill_price,
                fill_quantity=cover_qty,
                commission=fee
            )

            # Close trade
            open_trade = await self.truth_logger.get_open_position(symbol)
            if open_trade:
                await self.truth_logger.close_trade(
                    trade_id=open_trade["trade_id"],
                    exit_order_id=order.order_id,
                    exit_price=fill_price,
                    exit_reason=ExitReason.SIGNAL_EXIT,
                    commission=fee
                )

            # Clear position
            entry_price = position.cost_basis
            position.qty = Decimal("0")
            position.is_short_position = False

            self.last_trade_time[symbol] = datetime.now(timezone.utc)

            pnl_pct = float((entry_price - fill_price) / entry_price * 100) if entry_price > 0 else 0
            logger.info(f"COVER SHORT {symbol}: {float(cover_qty):.6f} @ ${float(fill_price):,.2f} "
                       f"(P&L: ${float(pnl):,.2f}, {pnl_pct:+.1f}%)")

            return True

        except Exception as e:
            logger.error(f"SHORT exit failed for {symbol}: {e}")
            return False

    # =========================================================================
    # ADAPTIVE SCANNING METHODS
    # =========================================================================

    def calculate_setup_proximity(
        self,
        symbol: str,
        setup: "ProfessionalSetup"
    ) -> SetupProximity:
        """
        Calculate how close a symbol is to a breakout opportunity.

        Used to determine if we should scan more frequently.
        """
        current_price = setup.timing_layer.current_price
        upper_channel = setup.timing_layer.upper_channel
        lower_channel = setup.timing_layer.lower_channel

        # Calculate distances
        if upper_channel and upper_channel > 0:
            dist_breakout = float((upper_channel - current_price) / current_price * 100)
        else:
            dist_breakout = 999.0  # No channel = far from breakout

        if lower_channel and lower_channel > 0:
            dist_breakdown = float((current_price - lower_channel) / current_price * 100)
        else:
            dist_breakdown = 999.0

        return SetupProximity(
            symbol=symbol,
            current_price=current_price,
            upper_channel=upper_channel,
            lower_channel=lower_channel,
            distance_to_breakout_pct=max(0, dist_breakout),
            distance_to_breakdown_pct=max(0, dist_breakdown),
            grade=setup.grade.value,
            trend_aligned=setup.trend_alignment,
            momentum_aligned=setup.momentum_alignment,
            volume_confirmed=setup.volume_confirmation,
        )

    def determine_scan_mode(self) -> Tuple[ScanMode, str, List[str]]:
        """
        Determine the appropriate scan mode based on all symbol proximities.

        Returns:
            Tuple of (mode, reason, hot_symbols)
        """
        if not self.setup_proximities:
            return ScanMode.NORMAL, "No proximity data yet", []

        # Find symbols near breakout (TRIGGER mode)
        trigger_symbols = [
            sym for sym, prox in self.setup_proximities.items()
            if prox.is_near_breakout and prox.trend_aligned
        ]
        if trigger_symbols:
            return (
                ScanMode.TRIGGER,
                f"Price within 1.5% of breakout: {', '.join(trigger_symbols)}",
                trigger_symbols
            )

        # Find symbols with forming setups (ALERT mode)
        alert_symbols = [
            sym for sym, prox in self.setup_proximities.items()
            if prox.is_forming_setup
        ]
        if alert_symbols:
            return (
                ScanMode.ALERT,
                f"Setup forming (within 3%): {', '.join(alert_symbols)}",
                alert_symbols
            )

        # Check for any grade A/B setups even if not close to breakout
        graded_symbols = [
            sym for sym, prox in self.setup_proximities.items()
            if prox.grade in ("A+", "A") and prox.trend_aligned
        ]
        if graded_symbols:
            return (
                ScanMode.ALERT,
                f"High-grade setups detected: {', '.join(graded_symbols)}",
                graded_symbols
            )

        return ScanMode.NORMAL, "No active setups", []

    def update_scan_mode(self) -> bool:
        """
        Update the current scan mode and log transitions.

        Returns:
            True if mode changed
        """
        new_mode, reason, hot_symbols = self.determine_scan_mode()

        if new_mode != self.current_scan_mode:
            self.previous_scan_mode = self.current_scan_mode
            self.current_scan_mode = new_mode
            self.scan_mode_since = datetime.now(timezone.utc)

            # Log the transition
            old_interval = SCAN_INTERVALS[self.previous_scan_mode] // 60
            new_interval = SCAN_INTERVALS[new_mode] // 60

            if new_mode.value > self.previous_scan_mode.value:
                # Escalating - more urgent
                logger.warning(
                    f"[ADAPTIVE] ⚡ ESCALATE: {self.previous_scan_mode.value} → {new_mode.value} "
                    f"({old_interval}min → {new_interval}min) | {reason}"
                )
                # Discord alert for escalation (sync call, wrapped)
                if self.notifier and new_mode == ScanMode.TRIGGER:
                    try:
                        self.notifier.send_system_alert(
                            title="⚡ TRIGGER MODE",
                            message=f"Near breakout: {', '.join(hot_symbols)}\nScanning every 5 minutes"
                        )
                    except Exception as e:
                        logger.debug(f"Discord alert failed: {e}")
            else:
                # De-escalating - calming down
                logger.info(
                    f"[ADAPTIVE] 📉 DE-ESCALATE: {self.previous_scan_mode.value} → {new_mode.value} "
                    f"({old_interval}min → {new_interval}min) | {reason}"
                )

            return True

        return False

    def get_adaptive_interval(self) -> int:
        """Get the current scan interval in seconds based on mode."""
        return SCAN_INTERVALS[self.current_scan_mode]

    def get_scan_status(self) -> Dict[str, Any]:
        """Get current adaptive scanning status for API/dashboard."""
        mode_duration = (datetime.now(timezone.utc) - self.scan_mode_since).total_seconds()

        hot_symbols = []
        for sym, prox in self.setup_proximities.items():
            if prox.is_near_breakout or prox.is_forming_setup:
                hot_symbols.append({
                    "symbol": sym,
                    "distance_to_breakout_pct": prox.distance_to_breakout_pct,
                    "grade": prox.grade,
                    "near_breakout": prox.is_near_breakout,
                    "forming_setup": prox.is_forming_setup,
                })

        return {
            "mode": self.current_scan_mode.value,
            "interval_seconds": self.get_adaptive_interval(),
            "interval_minutes": self.get_adaptive_interval() // 60,
            "mode_since": self.scan_mode_since.isoformat(),
            "mode_duration_seconds": mode_duration,
            "hot_symbols": hot_symbols,
            "proximities": {k: v.to_dict() for k, v in self.setup_proximities.items()},
        }

    async def run_tick(self) -> Dict[str, Any]:
        """Run one evaluation tick for all symbols."""
        tick_time = datetime.now(timezone.utc)
        logger.info("=" * 70)
        logger.info(f"TICK - {tick_time.isoformat()}")
        logger.info("=" * 70)

        # ========================================
        # 24/7 SESSION MANAGEMENT
        # ========================================
        session_event = self.session_manager.check_session_transition()
        if session_event:
            logger.info(f"[SESSION] Transition: {session_event.previous_session.value if session_event.previous_session else 'None'} -> {session_event.session.value}")
            if self.notifier:
                if session_event.session == MarketSession.DEAD_ZONE:
                    await self.notifier.send_async(
                        f"🌙 **DEAD ZONE ENTERED** - New entries paused until 00:00 UTC\n"
                        f"Active positions will still be monitored for exits."
                    )
                elif session_event.previous_session == MarketSession.DEAD_ZONE:
                    await self.notifier.send_async(
                        f"☀️ **DEAD ZONE EXITED** - Trading resumed\n"
                        f"Session: {session_event.session.value}"
                    )

        # Check if we should pause new entries (exits still allowed)
        should_pause, pause_reason = self.session_manager.should_pause_trading()
        session_info = self.session_manager.get_session_info()
        logger.info(f"[SESSION] {session_info['current_session'].upper()} | "
                   f"Dead Zone: {'YES' if session_info['is_dead_zone'] else 'No'} | "
                   f"Position Mult: {session_info['position_multiplier']:.0%}")

        # ========================================
        # PROFESSIONAL MULTI-TIMEFRAME DATA FETCH
        # ========================================
        logger.info("[MTF] Fetching Daily/4h/1h data for all symbols...")

        # Fetch multi-timeframe data for all symbols in parallel
        mtf_tasks = {
            symbol: self.fetch_multi_timeframe_data(symbol)
            for symbol in self.symbols
        }

        symbol_mtf_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        for symbol, task in mtf_tasks.items():
            mtf_data = await task
            if mtf_data and any(df is not None for df in mtf_data.values()):
                symbol_mtf_data[symbol] = mtf_data
                bars_info = ", ".join(f"{tf}={len(df) if df is not None else 0}"
                                     for tf, df in mtf_data.items())
                logger.debug(f"{symbol}: {bars_info}")
            else:
                logger.warning(f"{symbol}: No data available")

        if not symbol_mtf_data:
            logger.error("No data available for any symbol")
            return {"error": "no_data"}

        logger.info(f"[MTF] Loaded data for {len(symbol_mtf_data)} symbols")

        # Fetch real-time spot prices for verification
        spot_prices = await self.fetch_spot_prices()
        logger.info(f"[SPOT] Fetched live prices for {len(spot_prices)} symbols")

        # Update current prices for existing positions (use 1h for most recent)
        for symbol, mtf_data in symbol_mtf_data.items():
            # Get best available timeframe data (prefer 1h for most recent price)
            df = None
            for tf in ["1h", "4h", "1d"]:
                tf_df = mtf_data.get(tf)
                if tf_df is not None and len(tf_df) > 0:
                    df = tf_df
                    break
            if df is not None and len(df) > 0:
                current_price = Decimal(str(df.iloc[-1]["close"]))
                current_high = Decimal(str(df.iloc[-1]["high"]))
                current_low = Decimal(str(df.iloc[-1]["low"]))
                if symbol in self.state.positions:
                    pos = self.state.positions[symbol]
                    pos.current_price = current_price
                    if pos.qty > 0:
                        # P&L calculation (inverse for SHORT)
                        if pos.is_short_position:
                            pos.unrealized_pnl = (pos.cost_basis - current_price) * pos.qty
                        else:
                            pos.unrealized_pnl = (current_price - pos.cost_basis) * pos.qty

                        # P0: Update chandelier stop (The Ratchet)
                        if pos.is_short_position:
                            # SHORT: Track lowest low since entry
                            if pos.lowest_low_since_entry is None:
                                pos.lowest_low_since_entry = current_low
                            elif current_low < pos.lowest_low_since_entry:
                                pos.lowest_low_since_entry = current_low
                                logger.debug(f"[CHANDELIER] {symbol}: New LL ${float(current_low):,.2f}")
                        else:
                            # LONG: Track highest high since entry
                            if pos.highest_high_since_entry is None:
                                pos.highest_high_since_entry = current_high
                            elif current_high > pos.highest_high_since_entry:
                                pos.highest_high_since_entry = current_high
                                logger.debug(f"[CHANDELIER] {symbol}: New HH ${float(current_high):,.2f}")

                        # Calculate ATR for chandelier
                        if len(df) >= 14:
                            high_col = df['high'].astype(float)
                            low_col = df['low'].astype(float)
                            close_col = df['close'].astype(float)
                            tr1 = high_col - low_col
                            tr2 = abs(high_col - close_col.shift(1))
                            tr3 = abs(low_col - close_col.shift(1))
                            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                            atr = Decimal(str(tr.ewm(span=14, adjust=False).mean().iloc[-1]))

                            if pos.is_short_position:
                                # SHORT: Chandelier Stop = Lowest Low + (3 * ATR)
                                new_chandelier = pos.lowest_low_since_entry + (atr * Decimal("3"))
                                # Ratchet only moves DOWN for shorts
                                if pos.chandelier_stop is None or new_chandelier < pos.chandelier_stop:
                                    pos.chandelier_stop = new_chandelier
                                    logger.debug(f"[CHANDELIER] {symbol}: Stop ratcheted DOWN to ${float(new_chandelier):,.2f}")
                            else:
                                # LONG: Chandelier Stop = Highest High - (3 * ATR)
                                new_chandelier = pos.highest_high_since_entry - (atr * Decimal("3"))
                                # Ratchet only moves UP for longs
                                if pos.chandelier_stop is None or new_chandelier > pos.chandelier_stop:
                                    pos.chandelier_stop = new_chandelier
                                    logger.debug(f"[CHANDELIER] {symbol}: Stop ratcheted UP to ${float(new_chandelier):,.2f}")

        # Recalculate portfolio equity
        self.state.recalculate_equity()

        # Check portfolio-level drawdown
        if self.state.current_dd_pct > 20:
            self.state.dd_state = "critical"
            logger.warning(f"PORTFOLIO DD CRITICAL: {float(self.state.current_dd_pct):.1f}%")
        elif self.state.current_dd_pct > 12:
            self.state.dd_state = "warning"
            logger.warning(f"PORTFOLIO DD WARNING: {float(self.state.current_dd_pct):.1f}%")
        else:
            self.state.dd_state = "normal"

        # ========================================
        # PROFESSIONAL SYMBOL EVALUATION
        # ========================================
        logger.info("[MTF] Evaluating setups across all timeframes...")

        # Get BTC price for market regime context (used by price zone filter)
        btc_spot = spot_prices.get("BTC-USD", {}).get("price")
        if btc_spot:
            logger.info(f"[ZONE] BTC market context: ${btc_spot:,.0f}")

        signals: List[SymbolSignal] = []
        for symbol, mtf_data in symbol_mtf_data.items():
            signal = await self.evaluate_symbol(symbol, mtf_data, self.state.cash, btc_price=btc_spot)
            signals.append(signal)

            # Enhanced logging with grade
            grade = signal.context.get("professional", {}).get("grade", "?")
            action_str = f"[{signal.action}]" if signal.action != "HOLD" else "[HOLD]"
            logger.info(f"{symbol}: {action_str} Grade={grade} regime={signal.regime} "
                       f"price=${float(signal.current_price):,.2f}")

            # Log decision to Truth Engine (HOLD signals only - BUY/SELL logged in execute_trade)
            # This prevents duplicate decision entries for executed trades
            professional_data = signal.context.get("professional", {})
            timeframes_data = signal.context.get("timeframes", {})

            if signal.action == "HOLD":
                # Get spot price verification data for this symbol
                spot_data = spot_prices.get(symbol, {})

                # Get candle timestamp from the 1h data
                mtf_data = symbol_mtf_data.get(symbol, {})
                candle_ts = None
                for tf in ["1h", "4h", "1d"]:
                    tf_df = mtf_data.get(tf)
                    if tf_df is not None and len(tf_df) > 0:
                        candle_ts = tf_df.iloc[-1]["timestamp"].isoformat() if hasattr(tf_df.iloc[-1]["timestamp"], "isoformat") else str(tf_df.iloc[-1]["timestamp"])
                        break

                decision = await self.truth_logger.log_decision(
                    symbol=symbol,
                    strategy_name="professional_mtf",
                    signal_values={
                        "action": signal.action,
                        "current_price": float(signal.current_price),
                        "grade": professional_data.get("grade", "?"),
                        "trend_bias": timeframes_data.get("daily_trend", "unknown"),
                        "trend_alignment": professional_data.get("trend_alignment", 0),
                        "momentum_alignment": professional_data.get("momentum_alignment", 0),
                        "volume_confirmed": professional_data.get("volume_confirmation", False),
                        "daily_atr": professional_data.get("daily_atr", 0),
                        "reasoning": professional_data.get("reasoning", []),
                    },
                    risk_checks={
                        "regime": signal.regime,
                        "drawdown_state": self.state.dd_state,
                        "current_dd_pct": float(self.state.current_dd_pct),
                        "session_paused": should_pause,
                    },
                    result=DecisionResult.SIGNAL_HOLD,
                    result_reason=professional_data.get("reasoning", ["No reasoning"])[0] if professional_data.get("reasoning") else f"Grade {grade} setup",
                    market_context={
                        "regime": signal.regime,
                        "session": session_info.get("current_session", "unknown"),
                        "tick_time": tick_time.isoformat(),
                        # Price verification fields
                        "api_spot_price": spot_data.get("price"),
                        "api_spot_time": spot_data.get("time"),
                        "api_fetch_time": spot_data.get("fetch_time"),
                        "candle_timestamp": candle_ts,
                    },
                )

                # PHANTOM TRADE LOGGING: Track C-grade (and B-grade without triggers) for validation
                current_grade = professional_data.get("grade", "C")
                if current_grade in ("C", "B") and signal.action == "HOLD":
                    try:
                        entry_price = float(signal.current_price)
                        daily_atr = professional_data.get("daily_atr", 0) or (entry_price * 0.03)  # 3% fallback
                        chandelier_stop = entry_price - (3 * daily_atr)  # 3x ATR
                        hard_stop = entry_price * 0.95  # 5% hard stop

                        await self.truth_logger.log_phantom_trade(
                            decision_id=decision.decision_id,
                            symbol=symbol,
                            timestamp=tick_time,
                            setup_grade=current_grade,
                            signal_type="long",  # Currently only long signals
                            hypothetical_entry=entry_price,
                            chandelier_stop_price=chandelier_stop,
                            hard_stop_price=hard_stop,
                        )
                        logger.debug(f"  [{symbol}] Phantom trade logged (Grade {current_grade})")
                    except Exception as phantom_err:
                        logger.warning(f"Failed to log phantom for {symbol}: {phantom_err}")

            # Calculate setup proximity for adaptive scanning
            setup_details = signal.context.get("setup_details")
            if setup_details:
                timing = setup_details.get("layers", {}).get("timing_1h", {})
                upper_ch = timing.get("donchian", {}).get("upper_channel")
                lower_ch = timing.get("donchian", {}).get("lower_channel")
                price = float(signal.current_price)

                dist_breakout = ((upper_ch - price) / price * 100) if upper_ch and price > 0 else 999.0
                dist_breakdown = ((price - lower_ch) / price * 100) if lower_ch and price > 0 else 999.0

                self.setup_proximities[symbol] = SetupProximity(
                    symbol=symbol,
                    current_price=signal.current_price,
                    upper_channel=Decimal(str(upper_ch)) if upper_ch else None,
                    lower_channel=Decimal(str(lower_ch)) if lower_ch else None,
                    distance_to_breakout_pct=max(0, dist_breakout),
                    distance_to_breakdown_pct=max(0, dist_breakdown),
                    grade=professional_data.get("grade", "C"),
                    trend_aligned=professional_data.get("trend_alignment", False),
                    momentum_aligned=professional_data.get("momentum_alignment", False),
                    volume_confirmed=professional_data.get("volume_confirmed", False),
                )

        # ========================================
        # ADAPTIVE SCANNING - UPDATE MODE
        # ========================================
        mode_changed = self.update_scan_mode()
        next_interval = self.get_adaptive_interval()
        logger.info(f"[ADAPTIVE] Mode: {self.current_scan_mode.value.upper()} | "
                   f"Next scan in {next_interval // 60} min")

        # Log hot symbols if any
        hot_count = sum(1 for p in self.setup_proximities.values()
                       if p.is_near_breakout or p.is_forming_setup)
        if hot_count > 0:
            hot_list = [f"{p.symbol}({p.distance_to_breakout_pct:.1f}%)"
                       for p in self.setup_proximities.values()
                       if p.is_near_breakout or p.is_forming_setup]
            logger.info(f"[ADAPTIVE] 🔥 Hot symbols: {', '.join(hot_list)}")

        # Get buy signals and allocate capital
        allocations = self.allocate_capital(signals, self.state.cash)

        # Execute trades (skip new entries during dead zone)
        trades_executed = 0
        if should_pause:
            logger.info(f"[SESSION] {pause_reason} - Skipping new entries (exits still allowed)")
        else:
            for symbol, allocation in allocations.items():
                signal = next(s for s in signals if s.symbol == symbol)
                success = await self.execute_trade(
                    symbol=symbol,
                    action="BUY",
                    amount=allocation,
                    current_price=signal.current_price,
                    signal=signal
                )
                if success:
                    trades_executed += 1

        # Check for sell signals (ALWAYS execute exits, even in dead zone)
        for signal in signals:
            if signal.action == "SELL":
                position = self.state.positions.get(signal.symbol)
                if position and position.qty > 0:
                    success = await self.execute_trade(
                        symbol=signal.symbol,
                        action="SELL",
                        amount=position.qty * position.current_price,
                        current_price=signal.current_price,
                        signal=signal
                    )
                    if success:
                        trades_executed += 1

        # Recalculate and save state
        self.state.recalculate_equity()
        self.state.last_update = tick_time.isoformat()
        self.save_state()

        # Log portfolio summary
        logger.info("-" * 70)
        logger.info(f"PORTFOLIO SUMMARY")
        logger.info(f"  Equity: ${float(self.state.total_equity):,.2f}")
        logger.info(f"  Cash: ${float(self.state.cash):,.2f}")
        logger.info(f"  Positions: {self.state.get_position_count()}")
        logger.info(f"  DD: {float(self.state.current_dd_pct):.1f}% ({self.state.dd_state})")
        logger.info(f"  Trades this tick: {trades_executed}")

        for symbol, pos in self.state.positions.items():
            if pos.qty > 0:
                value = pos.qty * pos.current_price
                pnl_pct = (pos.current_price - pos.cost_basis) / pos.cost_basis * 100 if pos.cost_basis > 0 else 0
                logger.info(f"  {symbol}: {float(pos.qty):.6f} @ ${float(pos.cost_basis):,.2f} "
                           f"-> ${float(pos.current_price):,.2f} ({float(pnl_pct):+.2f}%)")

        return {
            "tick_time": tick_time.isoformat(),
            "equity": float(self.state.total_equity),
            "cash": float(self.state.cash),
            "positions": self.state.get_position_count(),
            "trades": trades_executed,
            "dd_pct": float(self.state.current_dd_pct),
            "scan": self.get_scan_status(),
        }

    def get_interval_seconds(self) -> int:
        """Get sleep interval in seconds."""
        intervals = {"1h": 3600, "4h": 14400, "1d": 86400}
        return intervals.get(self.interval, 3600)

    async def run(self):
        """Main trading loop."""
        global shutdown_requested

        logger.info("=" * 70)
        logger.info("ArgusNexus V4 - UNIFIED PORTFOLIO TRADER (Adaptive)")
        logger.info("=" * 70)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Base Interval: {self.interval}")
        logger.info(f"Starting Capital: ${float(self.starting_capital):,.2f}")
        logger.info(f"Max Position %: {float(self.max_position_pct) * 100:.0f}%")
        logger.info("=" * 70)
        logger.info("ADAPTIVE SCANNING INTERVALS:")
        logger.info(f"  NORMAL:  {SCAN_INTERVALS[ScanMode.NORMAL] // 60} min (no setups)")
        logger.info(f"  ALERT:   {SCAN_INTERVALS[ScanMode.ALERT] // 60} min (setup forming)")
        logger.info(f"  TRIGGER: {SCAN_INTERVALS[ScanMode.TRIGGER] // 60} min (near breakout)")
        logger.info("=" * 70)

        await self.initialize()

        while not shutdown_requested:
            try:
                result = await self.run_tick()

                if shutdown_requested:
                    break

                # Get ADAPTIVE interval based on current market conditions
                interval_seconds = self.get_adaptive_interval()

                # Calculate next run time
                next_run = datetime.now(timezone.utc) + timedelta(seconds=interval_seconds)
                mode_str = self.current_scan_mode.value.upper()
                logger.info(f"Next tick at {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC "
                           f"({mode_str} mode: {interval_seconds // 60} min)")

                # Sleep in small increments to allow graceful shutdown
                # Also check for mode changes during sleep (escalate faster if needed)
                sleep_until = next_run
                while datetime.now(timezone.utc) < sleep_until and not shutdown_requested:
                    remaining = (sleep_until - datetime.now(timezone.utc)).total_seconds()
                    await asyncio.sleep(min(30, max(1, remaining)))  # Check every 30s max

            except Exception as e:
                logger.error(f"Error in tick: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)  # Wait before retry

        # Stop Twitter integration
        if self.twitter_hook:
            await self.twitter_hook.stop()
            logger.info("Twitter hook stopped")

        logger.info("Unified trader stopped")


async def main():
    parser = argparse.ArgumentParser(description="ArgusNexus V4 Unified Portfolio Trader")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                       help="Comma-separated list of symbols")
    parser.add_argument("--capital", type=float, default=4000,
                       help="Starting capital in USD")
    parser.add_argument("--interval", choices=["1h", "4h", "1d"], default="1h",
                       help="Trading interval")
    parser.add_argument("--max-position-pct", type=float, default=0.25,
                       help="Maximum position size as fraction of portfolio (0.25 = 25%%)")
    parser.add_argument("--lookback", type=int, default=365,
                       help="Lookback period in days")

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    trader = UnifiedPortfolioTrader(
        symbols=symbols,
        starting_capital=Decimal(str(args.capital)),
        interval=args.interval,
        max_position_pct=args.max_position_pct,
        lookback_days=args.lookback
    )

    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
