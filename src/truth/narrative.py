"""
Decision Narrative Generator - Human-Readable Trade Explanations

Transforms raw decision data (signal_values, market_context, result_reason)
into clear, insightful narratives that answer: "Why did this happen?"

Usage:
    from src.truth.narrative import generate_narrative

    narrative = generate_narrative(decision_dict)
    # Returns: {
    #     "headline": "Reducing XRP exposure from 100% → 78.8%",
    #     "summary": "Portfolio drifted above target...",
    #     "key_factors": [...],
    #     "context_explained": {...}
    # }
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class NarrativeOutput:
    """Structured narrative output for a decision"""
    headline: str
    summary: str
    key_factors: List[str]
    context_explained: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "key_factors": self.key_factors,
            "context_explained": self.context_explained
        }


def generate_narrative(decision: Dict[str, Any]) -> NarrativeOutput:
    """
    Generate a human-readable narrative from decision data.

    Supports multiple decision types:
    - Portfolio rebalancing (SELL/BUY with allocation context)
    - Strategy signals (EMA crossover, etc.)
    - Risk rejections
    - Halted/blocked states
    """
    result = decision.get("result", "")
    result_reason = decision.get("result_reason", "")
    symbol = decision.get("symbol", "UNKNOWN")
    strategy = decision.get("strategy_name", "unknown")

    # Parse context - could be JSON string or dict
    context = decision.get("market_context") or decision.get("signal_values") or {}
    if isinstance(context, str):
        import json
        try:
            context = json.loads(context)
        except:
            context = {}

    # Route to appropriate generator
    if "portfolio" in strategy.lower() or "rebalance" in strategy.lower():
        return _generate_portfolio_narrative(symbol, result, result_reason, context)
    elif "ema" in strategy.lower() or "crossover" in strategy.lower():
        return _generate_ema_narrative(symbol, result, result_reason, context)
    elif result == "risk_rejected":
        return _generate_risk_rejection_narrative(symbol, result_reason, context)
    elif "HALTED" in result_reason or "FAIL_CLOSED" in result_reason:
        return _generate_halt_narrative(symbol, result_reason, context)
    else:
        return _generate_generic_narrative(symbol, result, result_reason, context)


def _generate_portfolio_narrative(
    symbol: str,
    result: str,
    result_reason: str,
    context: Dict[str, Any]
) -> NarrativeOutput:
    """Generate narrative for portfolio rebalancing decisions"""

    # Extract key metrics
    current_alloc = context.get("current_alloc", 0)
    final_target = context.get("final_target", 0)
    alloc_drift = context.get("alloc_drift", 0)
    current_dd = context.get("current_dd", 0)
    dd_state = context.get("dd_state", "normal")
    dd_multiplier = context.get("dd_multiplier", 1.0)
    core_target = context.get("core_target", 0)
    core_regime = context.get("core_regime", "unknown")
    sleeve_in = context.get("sleeve_in", False)
    recovery_mode = context.get("recovery_mode", False)
    can_rebalance = context.get("can_rebalance", True)

    # Build headline
    action = "SELL" if "SELL" in result_reason else "BUY" if "BUY" in result_reason else "HOLD"

    if action == "SELL":
        headline = f"Reducing {symbol} exposure: {current_alloc:.0%} → {final_target:.0%}"
    elif action == "BUY":
        headline = f"Increasing {symbol} exposure: {current_alloc:.0%} → {final_target:.0%}"
    else:
        headline = f"Holding {symbol} position at {current_alloc:.0%}"

    # Build summary
    summary_parts = []

    if action != "HOLD":
        if alloc_drift > 0:
            direction = "above" if current_alloc > final_target else "below"
            summary_parts.append(
                f"Portfolio drifted {alloc_drift:.1%} {direction} target allocation."
            )

    # Explain core strategy contribution
    regime_desc = {
        "bull": "bullish conditions favor high exposure",
        "sideways": "ranging market suggests moderate exposure",
        "bear": "bearish conditions call for reduced exposure"
    }.get(core_regime, f"{core_regime} regime")
    summary_parts.append(f"Core strategy targets {core_target:.0%} ({regime_desc}).")

    # Explain sleeve
    if sleeve_in:
        summary_parts.append("Sleeve strategy is active, adding exposure.")

    # Explain DD impact
    if dd_multiplier < 1.0:
        summary_parts.append(
            f"Drawdown protection active: {dd_state} state reduces allocation by {(1-dd_multiplier):.0%}."
        )

    if recovery_mode:
        summary_parts.append("Recovery mode: cautiously rebuilding after drawdown.")

    if not can_rebalance and action == "HOLD":
        summary_parts.append("Rebalance cooldown is active.")

    summary = " ".join(summary_parts)

    # Key factors that drove this decision
    key_factors = []

    if alloc_drift > 0.10:
        key_factors.append(f"Significant drift: {alloc_drift:.1%} from target")

    if dd_multiplier < 1.0:
        key_factors.append(f"DD protection: {dd_state} ({dd_multiplier:.0%} multiplier)")

    if core_regime in ["bull", "bear"]:
        key_factors.append(f"Market regime: {core_regime}")

    if sleeve_in:
        key_factors.append("Sleeve strategy active")

    if not key_factors:
        key_factors.append("Routine rebalancing check")

    # Context explanations
    context_explained = {
        "Current Allocation": f"{current_alloc:.1%} of portfolio in {symbol}",
        "Target Allocation": f"{final_target:.1%} (core {core_target:.0%} + sleeve adjustments)",
        "Allocation Drift": f"{alloc_drift:.1%} deviation from target",
        "Drawdown": f"{current_dd:.1f}% from high water mark" if current_dd > 0 else "No drawdown (at or near highs)",
        "DD State": _explain_dd_state(dd_state, dd_multiplier),
        "Market Regime": _explain_regime(core_regime),
        "Sleeve Position": "Active (adding momentum exposure)" if sleeve_in else "Inactive",
        "Can Rebalance": "Yes" if can_rebalance else "No (cooldown active)"
    }

    return NarrativeOutput(
        headline=headline,
        summary=summary,
        key_factors=key_factors,
        context_explained=context_explained
    )


def _generate_ema_narrative(
    symbol: str,
    result: str,
    result_reason: str,
    context: Dict[str, Any]
) -> NarrativeOutput:
    """Generate narrative for EMA crossover signals"""

    fast_ema = context.get("fast_ema", 0)
    slow_ema = context.get("slow_ema", 0)
    ema_diff = context.get("ema_diff", 0)
    crossover = context.get("crossover", "none")
    atr = context.get("atr", 0)
    current_price = context.get("current_price", 0)
    confidence = context.get("confidence", 0)

    # Build headline
    if "signal_long" in result.lower():
        headline = f"LONG signal on {symbol}: Bullish EMA crossover"
    elif "signal_short" in result.lower():
        headline = f"SHORT signal on {symbol}: Bearish EMA crossover"
    elif "signal_close" in result.lower():
        headline = f"EXIT signal on {symbol}: Closing position"
    else:
        headline = f"No action on {symbol}: EMAs not crossed"

    # Summary
    summary_parts = []

    if crossover == "bullish":
        summary_parts.append(f"Fast EMA ({fast_ema:.2f}) crossed above Slow EMA ({slow_ema:.2f}).")
    elif crossover == "bearish":
        summary_parts.append(f"Fast EMA ({fast_ema:.2f}) crossed below Slow EMA ({slow_ema:.2f}).")
    else:
        diff_dir = "above" if ema_diff > 0 else "below"
        summary_parts.append(f"Fast EMA is {abs(ema_diff):.2f} {diff_dir} Slow EMA. No crossover.")

    if atr > 0:
        summary_parts.append(f"ATR volatility measure: {atr:.2f}.")

    if confidence > 0:
        summary_parts.append(f"Signal confidence: {confidence:.0%}.")

    summary = " ".join(summary_parts)

    key_factors = []
    if crossover in ["bullish", "bearish"]:
        key_factors.append(f"{crossover.title()} EMA crossover detected")
    if confidence > 0.7:
        key_factors.append(f"High confidence signal ({confidence:.0%})")
    elif confidence > 0:
        key_factors.append(f"Moderate confidence ({confidence:.0%})")

    context_explained = {
        "Fast EMA": f"{fast_ema:.2f}" if fast_ema else "N/A",
        "Slow EMA": f"{slow_ema:.2f}" if slow_ema else "N/A",
        "EMA Difference": f"{ema_diff:.2f}" if ema_diff else "N/A",
        "Current Price": f"${current_price:,.2f}" if current_price else "N/A",
        "ATR (Volatility)": f"{atr:.2f}" if atr else "N/A",
        "Crossover": crossover.title() if crossover else "None"
    }

    return NarrativeOutput(
        headline=headline,
        summary=summary,
        key_factors=key_factors if key_factors else ["Routine market evaluation"],
        context_explained=context_explained
    )


def _generate_risk_rejection_narrative(
    symbol: str,
    result_reason: str,
    context: Dict[str, Any]
) -> NarrativeOutput:
    """Generate narrative for risk-rejected decisions"""

    headline = f"Trade BLOCKED on {symbol}: Risk check failed"

    # Try to extract which check failed
    risk_checks = context.get("risk_checks", {})

    failed_checks = []
    for check_name, check_result in risk_checks.items():
        if isinstance(check_result, dict) and not check_result.get("passed", True):
            failed_checks.append(check_name)

    if failed_checks:
        summary = f"Signal was generated but blocked by risk management. Failed checks: {', '.join(failed_checks)}."
    else:
        summary = f"Signal was generated but blocked by risk management. Reason: {result_reason}"

    key_factors = [f"Risk check failed: {check}" for check in failed_checks] or [result_reason]

    context_explained = {"Rejection Reason": result_reason}
    for check_name, check_result in risk_checks.items():
        if isinstance(check_result, dict):
            passed = "Passed" if check_result.get("passed", True) else "FAILED"
            context_explained[check_name.replace("_", " ").title()] = passed

    return NarrativeOutput(
        headline=headline,
        summary=summary,
        key_factors=key_factors,
        context_explained=context_explained
    )


def _generate_halt_narrative(
    symbol: str,
    result_reason: str,
    context: Dict[str, Any]
) -> NarrativeOutput:
    """Generate narrative for halted/fail-closed states"""

    if "FAIL_CLOSED" in result_reason:
        headline = f"FAIL CLOSED on {symbol}: Safety shutdown"
        summary = "Trading engine encountered an error and closed positions for safety. Manual review required."
    else:
        headline = f"HALTED on {symbol}: Trading suspended"
        summary = f"Trading is suspended. Reason: {result_reason}"

    key_factors = ["Trading halted", result_reason.split(":")[-1].strip() if ":" in result_reason else result_reason]

    return NarrativeOutput(
        headline=headline,
        summary=summary,
        key_factors=key_factors,
        context_explained={"Status": result_reason, "Action Required": "Review and manually resume if appropriate"}
    )


def _generate_generic_narrative(
    symbol: str,
    result: str,
    result_reason: str,
    context: Dict[str, Any]
) -> NarrativeOutput:
    """Fallback generic narrative generator"""

    result_display = result.replace("_", " ").title() if result else "Unknown"

    headline = f"{result_display} on {symbol}"
    summary = result_reason or "No additional details available."

    # Convert context to human-readable
    context_explained = {}
    for key, value in context.items():
        readable_key = key.replace("_", " ").title()
        if isinstance(value, float):
            if abs(value) < 1 and value != 0:
                context_explained[readable_key] = f"{value:.1%}"
            else:
                context_explained[readable_key] = f"{value:.2f}"
        elif isinstance(value, bool):
            context_explained[readable_key] = "Yes" if value else "No"
        else:
            context_explained[readable_key] = str(value)

    return NarrativeOutput(
        headline=headline,
        summary=summary,
        key_factors=[result_reason] if result_reason else ["Decision logged"],
        context_explained=context_explained
    )


def _explain_dd_state(dd_state: str, dd_multiplier: float) -> str:
    """Generate human-readable DD state explanation"""
    explanations = {
        "normal": "No drawdown concern. Full allocation permitted.",
        "warning": f"Mild drawdown. Allocation reduced to {dd_multiplier:.0%} of target.",
        "critical": f"Significant drawdown. Allocation reduced to {dd_multiplier:.0%} of target as protection.",
        "recovery": "Recovering from drawdown. Cautiously rebuilding positions."
    }
    return explanations.get(dd_state, f"{dd_state} ({dd_multiplier:.0%} multiplier)")


def _explain_regime(regime: str) -> str:
    """Generate human-readable regime explanation"""
    explanations = {
        "bull": "Bullish trend detected. Higher allocations favored.",
        "sideways": "Ranging/sideways market. Moderate allocations.",
        "bear": "Bearish trend detected. Lower allocations for protection."
    }
    return explanations.get(regime, regime.title())
