"""
Engine Module - Trading Engine Components

Re-exports the original TradingEngine plus multi-timeframe extensions.

Original (from engine.py):
    from src.engine import TradingEngine, TradingMode

Multi-timeframe (new):
    from src.engine import MultiTimeframeEngine, TimeframeUnit, SignalAggregator
"""

# Re-export original TradingEngine from the sibling module
# The original engine.py is at src/engine.py, this package is src/engine/
# Python resolves this as a relative import from parent
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from the original engine.py module directly
import importlib.util
_engine_path = os.path.join(_parent_dir, "engine.py")
_spec = importlib.util.spec_from_file_location("_original_engine", _engine_path)
_original_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original_engine)

# Re-export original components
TradingEngine = _original_engine.TradingEngine
TradingMode = _original_engine.TradingMode
PositionState = _original_engine.PositionState
ReconciliationResult = _original_engine.ReconciliationResult
TickResult = _original_engine.TickResult
POSITION_RECONCILIATION_ENABLED = _original_engine.POSITION_RECONCILIATION_ENABLED

# Multi-timeframe components
from .multi_timeframe import (
    # Enums
    SignalType,
    ConflictResolution,

    # Dataclasses
    TimeframeSignal,
    AggregatedSignal,
    TimeframeConfig,
    MultiTimeframeConfig,

    # Classes
    SignalAggregator,
    TimeframeUnit,
    MultiTimeframeEngine,

    # Utilities
    create_mtf_signal_id
)


__all__ = [
    # Original engine components
    "TradingEngine",
    "TradingMode",
    "PositionState",
    "ReconciliationResult",
    "TickResult",
    "POSITION_RECONCILIATION_ENABLED",

    # Multi-timeframe enums
    "SignalType",
    "ConflictResolution",

    # Multi-timeframe dataclasses
    "TimeframeSignal",
    "AggregatedSignal",
    "TimeframeConfig",
    "MultiTimeframeConfig",

    # Multi-timeframe classes
    "SignalAggregator",
    "TimeframeUnit",
    "MultiTimeframeEngine",

    # Utilities
    "create_mtf_signal_id"
]
