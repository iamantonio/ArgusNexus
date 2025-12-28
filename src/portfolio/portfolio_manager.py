"""
Portfolio Manager - Single Source of Truth

Combines two strategy components into ONE consolidated BTC position:
- Vol-Regime Core (85% of portfolio)
- MostlyLong BTC Sleeve (15% of portfolio)

Applies portfolio-level drawdown circuit breaker that overrides
individual strategy signals when triggered.

CRITICAL: All trades go through this manager. Individual strategies
do NOT execute trades directly.

Recovery Logic (HARDENED):
- NO HWM reset on regime-based recovery (prevents wiping DD history)
- Require BOTH (DD < threshold) AND (bull regime) for re-entry
- Track bars_in_critical to prevent premature re-entry
- Rebalance cooldown to reduce trade frequency
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from src.strategy.vol_regime_core import VolRegimeCoreStrategy, VolRegimeResult
from src.strategy.mostly_long_btc import MostlyLongBTCStrategy, SignalResult, Signal


class DDState(Enum):
    """Drawdown circuit breaker state"""
    NORMAL = "normal"        # DD < 15%
    WARNING = "warning"      # 15% <= DD < 22%
    CRITICAL = "critical"    # DD >= 22%
    RECOVERY = "recovery"    # After critical, waiting for DD < 10% AND bull regime


@dataclass
class PortfolioState:
    """Persistent portfolio state"""
    # Capital tracking
    total_equity: Decimal
    btc_qty: Decimal
    cash: Decimal
    high_water_mark: Decimal

    # DD circuit breaker
    dd_state: DDState = DDState.NORMAL
    recovery_mode: bool = False
    bars_in_critical: int = 0  # Track consecutive bars in critical state

    # Sleeve tracking
    sleeve_in_position: bool = False
    sleeve_entry_price: Optional[Decimal] = None

    # Rebalance cooldown
    last_rebalance_time: Optional[datetime] = None

    # Timestamps
    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to JSON-safe dict (floats, strings, ISO timestamps)."""
        return {
            "total_equity": str(self.total_equity),  # Use str to preserve precision
            "btc_qty": str(self.btc_qty),
            "cash": str(self.cash),
            "high_water_mark": str(self.high_water_mark),
            "dd_state": self.dd_state.value,
            "recovery_mode": self.recovery_mode,
            "bars_in_critical": self.bars_in_critical,
            "sleeve_in_position": self.sleeve_in_position,
            "sleeve_entry_price": str(self.sleeve_entry_price) if self.sleeve_entry_price else None,
            "last_rebalance_time": self.last_rebalance_time.isoformat() if self.last_rebalance_time else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioState":
        """
        Deserialize state from dict (JSON).

        Handles:
        - Decimal fields from str (preserves precision) or float
        - DDState enum from string value
        - datetime from ISO string
        - Optional fields that may be None
        """
        def to_decimal(val) -> Decimal:
            """Convert str/float/int to Decimal."""
            if val is None:
                return Decimal("0")
            return Decimal(str(val))

        def to_optional_decimal(val) -> Optional[Decimal]:
            """Convert to Decimal or None."""
            if val is None:
                return None
            return Decimal(str(val))

        def to_datetime(val) -> Optional[datetime]:
            """Parse ISO datetime string or return None."""
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        return cls(
            total_equity=to_decimal(data.get("total_equity", 0)),
            btc_qty=to_decimal(data.get("btc_qty", 0)),
            cash=to_decimal(data.get("cash", 0)),
            high_water_mark=to_decimal(data.get("high_water_mark", 0)),
            dd_state=DDState(data.get("dd_state", "normal")),
            recovery_mode=data.get("recovery_mode", False),
            bars_in_critical=data.get("bars_in_critical", 0),
            sleeve_in_position=data.get("sleeve_in_position", False),
            sleeve_entry_price=to_optional_decimal(data.get("sleeve_entry_price")),
            last_rebalance_time=to_datetime(data.get("last_rebalance_time")),
            last_update=to_datetime(data.get("last_update")),
        )


@dataclass
class RebalanceOrder:
    """Single consolidated order to execute"""
    action: str  # "BUY", "SELL", "HOLD"
    btc_qty_delta: Decimal  # Positive = buy, negative = sell
    target_btc_qty: Decimal
    target_alloc_pct: Decimal
    reason: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "btc_qty_delta": float(self.btc_qty_delta),
            "target_btc_qty": float(self.target_btc_qty),
            "target_alloc_pct": float(self.target_alloc_pct),
            "reason": self.reason,
            "context": self.context,
        }


class PortfolioManager:
    """
    Portfolio Manager - combines strategies into single position.

    Architecture:
    1. Evaluate Vol-Regime Core -> core_target_alloc
    2. Evaluate MostlyLong Sleeve -> sleeve_in_position
    3. Compute portfolio drawdown
    4. Apply DD circuit breaker (overrides strategies if critical)
    5. Combine: target = (0.85 * core_alloc + 0.15 * sleeve_alloc) * dd_mult
    6. Apply rebalance cooldown (min 5 days between rebalances unless DD state changes)
    7. Rebalance only if drift > threshold
    8. Output ONE order

    Parameters:
        core_weight: Weight for Vol-Regime Core (default: 0.85)
        sleeve_weight: Weight for MostlyLong Sleeve (default: 0.15)
        rebalance_threshold: Rebalance if drift exceeds this (default: 0.15)
        rebalance_cooldown_days: Min days between rebalances (default: 5)
        dd_warning: DD level that triggers warning (default: 15%)
        dd_critical: DD level that triggers critical (default: 22%)
        dd_recovery_full: DD level to restore full exposure (default: 10%)
        dd_recovery_half: DD level to restore half exposure (default: 15%)
        min_bars_in_critical: Min bars in critical before allowing recovery (default: 7)
    """

    def __init__(
        self,
        core_weight: float = 0.85,
        sleeve_weight: float = 0.15,
        rebalance_threshold: float = 0.15,
        rebalance_cooldown_days: int = 5,
        dd_warning: float = 12.0,  # Earlier warning to start de-risking
        dd_critical: float = 20.0,  # Tighter critical to enforce 25% max DD
        dd_recovery_full: float = 10.0,  # FIXED: was 8, now 10 per spec
        dd_recovery_half: float = 15.0,   # FIXED: was 12, now 15 per spec
        min_bars_in_critical: int = 14,  # Wait 2 weeks before time-based recovery
        recovery_alloc: float = 0.30     # Conservative 30% on time-based recovery
    ):
        self.core_weight = Decimal(str(core_weight))
        self.sleeve_weight = Decimal(str(sleeve_weight))
        self.rebalance_threshold = Decimal(str(rebalance_threshold))
        self.rebalance_cooldown_days = rebalance_cooldown_days
        self.dd_warning = Decimal(str(dd_warning))
        self.dd_critical = Decimal(str(dd_critical))
        self.dd_recovery_full = Decimal(str(dd_recovery_full))
        self.dd_recovery_half = Decimal(str(dd_recovery_half))
        self.min_bars_in_critical = min_bars_in_critical
        self.recovery_alloc = Decimal(str(recovery_alloc))

        # Strategy instances
        self.core_strategy = VolRegimeCoreStrategy()
        self.sleeve_strategy = MostlyLongBTCStrategy()

    def _compute_drawdown(
        self,
        current_equity: Decimal,
        high_water_mark: Decimal
    ) -> Decimal:
        """Compute current drawdown percentage"""
        if high_water_mark <= 0:
            return Decimal("0")
        dd = (1 - current_equity / high_water_mark) * 100
        return max(Decimal("0"), dd)

    def _compute_dd_multiplier(
        self,
        current_dd: Decimal,
        current_state: DDState,
        recovery_mode: bool,
        bars_in_critical: int,
        current_regime: str
    ) -> Tuple[Decimal, DDState, bool, int]:
        """
        Compute DD multiplier and update state.

        HARDENED Recovery Logic:
        - NO HWM reset (preserves drawdown history)
        - TIME-BASED RECOVERY checked FIRST to break death spiral
        - Require min_bars_in_critical AND bull regime for recovery
        - Gradual re-entry: half exposure first, then full

        Death Spiral Fix:
        - At 0% allocation, equity is frozen, so DD never improves
        - Time-based recovery MUST override DD thresholds after waiting period
        - This allows gradual re-entry even with elevated DD

        Returns:
            (dd_multiplier, new_state, new_recovery_mode, new_bars_in_critical)
        """
        new_state = current_state
        new_recovery_mode = recovery_mode
        new_bars_critical = bars_in_critical

        # =================================================================
        # TIME-BASED RECOVERY OVERRIDE (checked FIRST to break death spiral)
        # =================================================================
        # If we've been in recovery mode long enough AND bull regime,
        # allow gradual re-entry REGARDLESS of current DD level.
        # This breaks the death spiral where frozen equity prevents DD improvement.
        if recovery_mode and bars_in_critical >= self.min_bars_in_critical and current_regime == "bull":
            if current_dd < self.dd_recovery_full:
                # Full recovery: DD has finally dropped to safe level
                new_state = DDState.NORMAL
                new_recovery_mode = False
                new_bars_critical = 0
                dd_mult = Decimal("1.0")
            elif current_dd < self.dd_recovery_half:
                # Partial recovery: DD in moderate zone
                new_state = DDState.RECOVERY
                dd_mult = Decimal("0.5")
            else:
                # Time-based recovery: DD still elevated but we've waited long enough
                # Allow conservative exposure to start participating in recovery
                new_state = DDState.RECOVERY
                dd_mult = self.recovery_alloc  # Conservative re-entry (default 30%)
            return dd_mult, new_state, new_recovery_mode, new_bars_critical

        # =================================================================
        # STANDARD DD-BASED STATE TRANSITIONS
        # =================================================================
        if current_dd >= self.dd_critical:
            new_state = DDState.CRITICAL
            new_recovery_mode = True
            new_bars_critical = bars_in_critical + 1
            dd_mult = Decimal("0.0")

        elif current_dd >= self.dd_warning:
            if recovery_mode:
                # In recovery with DD still elevated, waiting for time-based trigger
                new_state = DDState.RECOVERY
                new_bars_critical = bars_in_critical + 1  # Keep counting
                dd_mult = Decimal("0.0")
            else:
                new_state = DDState.WARNING
                dd_mult = Decimal("0.5")

        else:
            # DD is below warning level
            if recovery_mode:
                # Check if we can recover (DD thresholds met)
                if current_dd < self.dd_recovery_full and current_regime == "bull":
                    new_state = DDState.NORMAL
                    new_recovery_mode = False
                    new_bars_critical = 0
                    dd_mult = Decimal("1.0")
                elif current_dd < self.dd_recovery_half and current_regime in ("bull", "sideways"):
                    new_state = DDState.RECOVERY
                    dd_mult = Decimal("0.5")
                else:
                    # Below warning but not fully recovered yet
                    new_state = DDState.RECOVERY
                    new_bars_critical = bars_in_critical + 1
                    dd_mult = Decimal("0.0")
            else:
                new_state = DDState.NORMAL
                dd_mult = Decimal("1.0")

        return dd_mult, new_state, new_recovery_mode, new_bars_critical

    def _evaluate_sleeve(
        self,
        df: pd.DataFrame,
        state: PortfolioState,
        timestamp: datetime
    ) -> Tuple[bool, Optional[Decimal]]:
        """
        Evaluate MostlyLong sleeve and return position state.

        Returns:
            (sleeve_should_be_in, new_entry_price_if_entering)
        """
        result = self.sleeve_strategy.evaluate(
            df,
            timestamp=timestamp,
            has_open_position=state.sleeve_in_position,
            entry_price=state.sleeve_entry_price
        )

        if result.signal == Signal.LONG and not state.sleeve_in_position:
            # Signal to enter
            return True, Decimal(str(df.iloc[-1]['close']))
        elif result.signal == Signal.EXIT_LONG and state.sleeve_in_position:
            # Signal to exit
            return False, None
        else:
            # Hold current state
            return state.sleeve_in_position, state.sleeve_entry_price

    def _can_rebalance(
        self,
        current_time: datetime,
        last_rebalance: Optional[datetime],
        dd_state_changed: bool
    ) -> bool:
        """
        Check if rebalance is allowed based on cooldown.

        Rebalance is allowed if:
        - DD state changed (always allow for risk management)
        - OR cooldown period has passed since last rebalance
        - OR this is the first rebalance
        """
        if dd_state_changed:
            return True  # Always allow rebalance on DD state change

        if last_rebalance is None:
            return True  # First rebalance

        days_since = (current_time - last_rebalance).days
        return days_since >= self.rebalance_cooldown_days

    def evaluate(
        self,
        df: pd.DataFrame,
        state: PortfolioState,
        current_price: Decimal,
        timestamp: Optional[datetime] = None
    ) -> Tuple[RebalanceOrder, PortfolioState]:
        """
        Evaluate portfolio and return rebalance order.

        Args:
            df: DataFrame with OHLCV columns (must have >= 205 bars)
            state: Current portfolio state
            current_price: Current BTC price
            timestamp: Evaluation timestamp

        Returns:
            (RebalanceOrder, updated PortfolioState)
        """
        eval_time = timestamp or datetime.utcnow()

        # Update equity with current price
        btc_value = state.btc_qty * current_price
        total_equity = state.cash + btc_value

        # Update high water mark (NO reset on recovery - HARDENED)
        new_hwm = max(state.high_water_mark, total_equity)

        # Compute current drawdown
        current_dd = self._compute_drawdown(total_equity, new_hwm)

        # Evaluate core strategy FIRST (we need regime for recovery logic)
        core_result = self.core_strategy.evaluate(df, timestamp=eval_time)
        core_target_alloc = Decimal(str(core_result.target_alloc))
        current_regime = core_result.context.regime

        # Compute DD multiplier and update state (HARDENED - no HWM reset)
        dd_mult, new_dd_state, new_recovery_mode, new_bars_critical = self._compute_dd_multiplier(
            current_dd,
            state.dd_state,
            state.recovery_mode,
            state.bars_in_critical,
            current_regime
        )

        # Check if DD state changed (for cooldown bypass)
        dd_state_changed = new_dd_state != state.dd_state

        # Evaluate sleeve strategy
        sleeve_in, sleeve_entry = self._evaluate_sleeve(df, state, eval_time)

        # CRITICAL: If DD multiplier is 0, force sleeve exit (override MostlyLong)
        if dd_mult == 0:
            sleeve_in = False
            sleeve_entry = None

        # Compute combined target allocation
        core_contrib = self.core_weight * core_target_alloc
        sleeve_contrib = self.sleeve_weight * (Decimal("1.0") if sleeve_in else Decimal("0.0"))
        raw_target = core_contrib + sleeve_contrib
        target_alloc = raw_target * dd_mult
        target_alloc = max(Decimal("0.0"), min(Decimal("1.0"), target_alloc))

        # Current allocation
        current_alloc = btc_value / total_equity if total_equity > 0 else Decimal("0")

        # Check if rebalance needed
        alloc_drift = abs(target_alloc - current_alloc)

        # Check cooldown
        can_rebalance = self._can_rebalance(
            eval_time,
            state.last_rebalance_time,
            dd_state_changed
        )

        # Build context
        context = {
            "current_dd": float(current_dd),
            "dd_state": new_dd_state.value,
            "dd_multiplier": float(dd_mult),
            "recovery_mode": new_recovery_mode,
            "bars_in_critical": new_bars_critical,
            "core_target": float(core_target_alloc),
            "core_regime": current_regime,
            "sleeve_in": sleeve_in,
            "raw_target": float(raw_target),
            "final_target": float(target_alloc),
            "current_alloc": float(current_alloc),
            "alloc_drift": float(alloc_drift),
            "can_rebalance": can_rebalance,
            "dd_state_changed": dd_state_changed,
        }

        # Determine new rebalance time (will be updated if we actually rebalance)
        new_rebalance_time = state.last_rebalance_time

        # Create base state (will update rebalance time if we trade)
        new_state = PortfolioState(
            total_equity=total_equity,
            btc_qty=state.btc_qty,
            cash=state.cash,
            high_water_mark=new_hwm,
            dd_state=new_dd_state,
            recovery_mode=new_recovery_mode,
            bars_in_critical=new_bars_critical,
            sleeve_in_position=sleeve_in,
            sleeve_entry_price=sleeve_entry,
            last_rebalance_time=new_rebalance_time,
            last_update=eval_time
        )

        # Check if we should hold
        should_hold = (
            alloc_drift <= self.rebalance_threshold or
            not can_rebalance
        )

        if should_hold:
            # Build narrative hold reason
            if alloc_drift <= self.rebalance_threshold and not can_rebalance:
                reason = f"Holding position. Drift ({alloc_drift:.1%}) within threshold and rebalance cooldown active."
            elif alloc_drift <= self.rebalance_threshold:
                reason = f"Holding position. Allocation drift ({alloc_drift:.1%}) is within acceptable threshold."
            else:
                reason = f"Holding position. Rebalance cooldown active, next rebalance allowed soon."

            return RebalanceOrder(
                action="HOLD",
                btc_qty_delta=Decimal("0"),
                target_btc_qty=state.btc_qty,
                target_alloc_pct=target_alloc * 100,
                reason=reason,
                context=context
            ), new_state

        # Calculate target BTC quantity
        target_btc_value = total_equity * target_alloc
        target_btc_qty = target_btc_value / current_price if current_price > 0 else Decimal("0")
        btc_delta = target_btc_qty - state.btc_qty

        # Update rebalance time since we're trading
        new_state.last_rebalance_time = eval_time

        # Build narrative reason based on action
        dd_note = f" DD protection at {dd_mult:.0%}." if dd_mult < 1 else ""
        regime_note = f" {current_regime.title()} regime." if current_regime != "sideways" else ""

        if btc_delta > 0:
            action = "BUY"
            reason = (
                f"Increasing exposure {current_alloc:.0%} → {target_alloc:.0%}. "
                f"Portfolio {alloc_drift:.1%} below target.{regime_note}{dd_note}"
            )
        elif btc_delta < 0:
            action = "SELL"
            reason = (
                f"Reducing exposure {current_alloc:.0%} → {target_alloc:.0%}. "
                f"Portfolio {alloc_drift:.1%} above target.{regime_note}{dd_note}"
            )
        else:
            action = "HOLD"
            reason = "Position at target. No adjustment needed."

        return RebalanceOrder(
            action=action,
            btc_qty_delta=btc_delta,
            target_btc_qty=target_btc_qty,
            target_alloc_pct=target_alloc * 100,
            reason=reason,
            context=context
        ), new_state

    def execute_order(
        self,
        order: RebalanceOrder,
        state: PortfolioState,
        current_price: Decimal,
        fees: Dict[str, float]
    ) -> PortfolioState:
        """
        Execute order and return updated state.

        Args:
            order: RebalanceOrder to execute
            state: Current portfolio state
            current_price: Current BTC price
            fees: Fee structure {"entry": 0.004, "exit": 0.002, "slippage": 0.001}

        Returns:
            Updated PortfolioState
        """
        if order.action == "HOLD":
            return state

        new_state = PortfolioState(
            total_equity=state.total_equity,
            btc_qty=state.btc_qty,
            cash=state.cash,
            high_water_mark=state.high_water_mark,
            dd_state=state.dd_state,
            recovery_mode=state.recovery_mode,
            bars_in_critical=state.bars_in_critical,
            sleeve_in_position=state.sleeve_in_position,
            sleeve_entry_price=state.sleeve_entry_price,
            last_rebalance_time=state.last_rebalance_time,
            last_update=state.last_update
        )

        if order.action == "BUY":
            # Buy BTC
            buy_qty = order.btc_qty_delta
            slippage = Decimal(str(fees.get("slippage", 0.001)))
            entry_fee = Decimal(str(fees.get("entry", 0.004)))

            fill_price = current_price * (1 + slippage)
            buy_value = buy_qty * fill_price
            fee = buy_value * entry_fee
            total_cost = buy_value + fee

            # If insufficient cash, reduce qty to what we can afford
            if total_cost > new_state.cash:
                # Solve for max_qty: qty * fill_price * (1 + entry_fee) = cash
                max_qty = new_state.cash / (fill_price * (1 + entry_fee))
                buy_qty = max_qty
                buy_value = buy_qty * fill_price
                fee = buy_value * entry_fee
                total_cost = buy_value + fee

            if buy_qty > 0 and total_cost <= new_state.cash:
                new_state.btc_qty += buy_qty
                new_state.cash -= total_cost

        elif order.action == "SELL":
            # Sell BTC
            sell_qty = abs(order.btc_qty_delta)
            slippage = Decimal(str(fees.get("slippage", 0.001)))
            exit_fee = Decimal(str(fees.get("exit", 0.002)))

            if sell_qty <= new_state.btc_qty:
                fill_price = current_price * (1 - slippage)
                sell_value = sell_qty * fill_price
                fee = sell_value * exit_fee
                net_proceeds = sell_value - fee

                new_state.btc_qty -= sell_qty
                new_state.cash += net_proceeds

        # Update total equity
        btc_value = new_state.btc_qty * current_price
        new_state.total_equity = new_state.cash + btc_value

        return new_state


# Factory function
def create_portfolio_manager(
    core_weight: float = 0.85,
    sleeve_weight: float = 0.15,
    rebalance_threshold: float = 0.15,
    rebalance_cooldown_days: int = 5,
    dd_warning: float = 15.0,
    dd_critical: float = 22.0
) -> PortfolioManager:
    """Create Portfolio Manager instance"""
    return PortfolioManager(
        core_weight=core_weight,
        sleeve_weight=sleeve_weight,
        rebalance_threshold=rebalance_threshold,
        rebalance_cooldown_days=rebalance_cooldown_days,
        dd_warning=dd_warning,
        dd_critical=dd_critical
    )


# Default instance
default_manager = PortfolioManager()
