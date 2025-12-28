#!/usr/bin/env python3
"""
ArgusNexus V4 - Monthly Performance Report

Compares live trading results against backtest expectations.
Run monthly to detect edge degradation early.

Usage:
    python scripts/monthly_performance_report.py
    python scripts/monthly_performance_report.py --discord  # Send to Discord
    python scripts/monthly_performance_report.py --months 3  # Last 3 months

Cron (1st of each month at 8am UTC):
    0 8 1 * * /path/to/venv/bin/python /path/to/scripts/monthly_performance_report.py --discord
"""

import sys
import json
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional
import requests

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
DB_PATH = Path(__file__).parent.parent / "data" / "v4_live_paper.db"
STATE_PATH = Path(__file__).parent.parent / "runtime" / "portfolio_state.json"
REPORT_DIR = Path(__file__).parent.parent / "runtime" / "reports"

# Backtest expectations (from your validated backtest)
BACKTEST_EXPECTATIONS = {
    "annual_return_pct": 32.0,
    "max_drawdown_pct": 22.4,
    "trades_per_year": 33,
    "win_rate_pct": 48.0,  # Approximate
    "sharpe_ratio": 1.2,   # Approximate
    "profit_factor": 1.5,  # Approximate
}

# Warning thresholds (% deviation from backtest)
THRESHOLDS = {
    "win_rate_drop": 0.20,      # 20% relative drop is warning
    "drawdown_exceed": 0.50,    # 50% higher DD is warning
    "return_underperform": 0.30, # 30% underperformance is warning
}


def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_portfolio_state() -> Dict[str, Any]:
    """Load current portfolio state."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def fetch_current_price() -> float:
    """Fetch current BTC price."""
    try:
        url = "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return float(response.json().get("price", 0))
    except Exception:
        pass
    return 0.0


def calculate_live_metrics(months: int = 1) -> Dict[str, Any]:
    """Calculate live trading metrics for the period."""
    conn = get_db_connection()

    start_date = datetime.now(timezone.utc) - timedelta(days=months * 30)
    start_str = start_date.isoformat()

    # Get closed trades
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses,
            COALESCE(SUM(CAST(realized_pnl AS REAL)), 0) as realized_pnl,
            COALESCE(SUM(CAST(net_pnl AS REAL)), 0) as net_pnl,
            COALESCE(AVG(CASE WHEN is_winner = 1 THEN CAST(realized_pnl AS REAL) END), 0) as avg_win,
            COALESCE(AVG(CASE WHEN is_winner = 0 THEN CAST(realized_pnl AS REAL) END), 0) as avg_loss
        FROM trades
        WHERE status = 'closed'
          AND exit_timestamp >= ?
          AND strategy_name = 'portfolio_manager'
    """, (start_str,))

    closed = cursor.fetchone()

    # Get open positions
    cursor = conn.execute("""
        SELECT COUNT(*) as open_count,
               COALESCE(SUM(CAST(quantity AS REAL) * CAST(entry_price AS REAL)), 0) as open_value
        FROM trades
        WHERE status = 'open'
          AND strategy_name = 'portfolio_manager'
    """)
    open_pos = cursor.fetchone()

    # Get decisions count
    cursor = conn.execute("""
        SELECT COUNT(*) as decision_count
        FROM decisions
        WHERE timestamp >= ?
          AND strategy_name = 'portfolio_manager'
    """, (start_str,))
    decisions = cursor.fetchone()

    conn.close()

    # Calculate metrics
    total_trades = closed["total_trades"] or 0
    wins = closed["wins"] or 0
    losses = closed["losses"] or 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    avg_win = abs(closed["avg_win"] or 0)
    avg_loss = abs(closed["avg_loss"] or 0)
    profit_factor = (avg_win / avg_loss) if avg_loss > 0 else 0

    # Get portfolio state for equity/DD
    state = get_portfolio_state()
    btc_price = fetch_current_price()

    btc_qty = float(state.get("btc_qty", 0))
    cash = float(state.get("cash", 0))
    hwm = float(state.get("high_water_mark", 500))

    current_equity = cash + (btc_qty * btc_price)
    current_dd = ((hwm - current_equity) / hwm * 100) if hwm > 0 else 0

    # Annualized return (rough estimate)
    starting_capital = 500  # From config
    total_return_pct = ((current_equity - starting_capital) / starting_capital * 100)

    # Annualize based on days running
    days_running = max(1, (datetime.now(timezone.utc) - start_date).days)
    annualized_return = total_return_pct * (365 / days_running)

    return {
        "period_months": months,
        "start_date": start_str,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 1),
        "realized_pnl": round(closed["realized_pnl"] or 0, 2),
        "net_pnl": round(closed["net_pnl"] or 0, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "open_positions": open_pos["open_count"] or 0,
        "decision_count": decisions["decision_count"] or 0,
        "current_equity": round(current_equity, 2),
        "high_water_mark": round(hwm, 2),
        "current_dd_pct": round(current_dd, 1),
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized_return, 1),
        "dd_state": state.get("dd_state", "unknown"),
        "recovery_mode": state.get("recovery_mode", False),
        "btc_price": round(btc_price, 2),
    }


def compare_to_backtest(live: Dict[str, Any]) -> Dict[str, Any]:
    """Compare live metrics to backtest expectations."""
    expected = BACKTEST_EXPECTATIONS

    comparisons = {}
    warnings = []

    # Win Rate comparison
    if live["total_trades"] > 0:
        wr_expected = expected["win_rate_pct"]
        wr_actual = live["win_rate_pct"]
        wr_diff = (wr_expected - wr_actual) / wr_expected if wr_expected > 0 else 0

        comparisons["win_rate"] = {
            "expected": wr_expected,
            "actual": wr_actual,
            "diff_pct": round(wr_diff * 100, 1),
            "status": "OK" if wr_diff < THRESHOLDS["win_rate_drop"] else "WARNING"
        }

        if wr_diff >= THRESHOLDS["win_rate_drop"]:
            warnings.append(f"Win rate {wr_actual:.1f}% is {wr_diff*100:.0f}% below expected {wr_expected:.1f}%")

    # Drawdown comparison
    dd_expected = expected["max_drawdown_pct"]
    dd_actual = live["current_dd_pct"]
    dd_ratio = dd_actual / dd_expected if dd_expected > 0 else 0

    comparisons["drawdown"] = {
        "expected_max": dd_expected,
        "actual": dd_actual,
        "ratio": round(dd_ratio, 2),
        "status": "OK" if dd_ratio < (1 + THRESHOLDS["drawdown_exceed"]) else "WARNING"
    }

    if dd_ratio >= (1 + THRESHOLDS["drawdown_exceed"]):
        warnings.append(f"Drawdown {dd_actual:.1f}% exceeds backtest max {dd_expected:.1f}% by {(dd_ratio-1)*100:.0f}%")

    # Return comparison (if enough data)
    if live["annualized_return_pct"] != 0:
        ret_expected = expected["annual_return_pct"]
        ret_actual = live["annualized_return_pct"]
        ret_diff = (ret_expected - ret_actual) / ret_expected if ret_expected > 0 else 0

        comparisons["return"] = {
            "expected_annual": ret_expected,
            "actual_annual": ret_actual,
            "diff_pct": round(ret_diff * 100, 1),
            "status": "OK" if ret_diff < THRESHOLDS["return_underperform"] else "WARNING"
        }

        if ret_diff >= THRESHOLDS["return_underperform"]:
            warnings.append(f"Return {ret_actual:.1f}% is {ret_diff*100:.0f}% below expected {ret_expected:.1f}%")

    # Trade frequency
    trades_per_month_expected = expected["trades_per_year"] / 12
    trades_per_month_actual = live["total_trades"] / max(1, live["period_months"])

    comparisons["trade_frequency"] = {
        "expected_monthly": round(trades_per_month_expected, 1),
        "actual_monthly": round(trades_per_month_actual, 1),
    }

    return {
        "comparisons": comparisons,
        "warnings": warnings,
        "overall_status": "WARNING" if warnings else "HEALTHY"
    }


def generate_report(months: int = 1) -> str:
    """Generate the full performance report."""
    live = calculate_live_metrics(months)
    comparison = compare_to_backtest(live)

    status_emoji = "🟢" if comparison["overall_status"] == "HEALTHY" else "🟡"

    report = f"""
{'='*60}
📊 ARGUSNEXUS MONTHLY PERFORMANCE REPORT
{'='*60}

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
Period: Last {months} month(s)
Status: {status_emoji} {comparison['overall_status']}

{'─'*60}
💰 PORTFOLIO STATUS
{'─'*60}
  Current Equity:    ${live['current_equity']:,.2f}
  High Water Mark:   ${live['high_water_mark']:,.2f}
  Current Drawdown:  {live['current_dd_pct']:.1f}%
  DD State:          {live['dd_state'].upper()}
  Recovery Mode:     {'YES' if live['recovery_mode'] else 'NO'}
  BTC Price:         ${live['btc_price']:,.2f}

{'─'*60}
📈 PERFORMANCE METRICS
{'─'*60}
  Total Return:      {live['total_return_pct']:+.2f}%
  Annualized:        {live['annualized_return_pct']:+.1f}%  (backtest: {BACKTEST_EXPECTATIONS['annual_return_pct']}%)
  Realized P&L:      ${live['realized_pnl']:+,.2f}

{'─'*60}
🎯 TRADING ACTIVITY
{'─'*60}
  Total Trades:      {live['total_trades']}
  Wins / Losses:     {live['wins']}W / {live['losses']}L
  Win Rate:          {live['win_rate_pct']:.1f}%  (backtest: {BACKTEST_EXPECTATIONS['win_rate_pct']}%)
  Avg Win:           ${live['avg_win']:.2f}
  Avg Loss:          ${live['avg_loss']:.2f}
  Profit Factor:     {live['profit_factor']:.2f}
  Open Positions:    {live['open_positions']}
  Decisions Made:    {live['decision_count']}

{'─'*60}
⚖️ BACKTEST COMPARISON
{'─'*60}"""

    for metric, data in comparison["comparisons"].items():
        status_icon = "✓" if data.get("status", "OK") == "OK" else "⚠"
        report += f"\n  {status_icon} {metric.replace('_', ' ').title()}: "
        if "expected" in data and "actual" in data:
            report += f"{data['actual']} (expected: {data['expected']})"
        elif "expected_max" in data:
            report += f"{data['actual']}% of {data['expected_max']}% max"
        else:
            report += f"{data.get('actual_monthly', 'N/A')}/mo (expected: {data.get('expected_monthly', 'N/A')}/mo)"

    if comparison["warnings"]:
        report += f"\n\n{'─'*60}\n⚠️  WARNINGS\n{'─'*60}"
        for warning in comparison["warnings"]:
            report += f"\n  • {warning}"
        report += "\n\n  ACTION: Review strategy parameters and recent market conditions."
    else:
        report += f"\n\n{'─'*60}\n✅ NO WARNINGS - Strategy performing within expectations\n{'─'*60}"

    report += f"""

{'─'*60}
📋 RECOMMENDATIONS
{'─'*60}
  • Next backtest due: {(datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')}
  • Monitor DD if > {BACKTEST_EXPECTATIONS['max_drawdown_pct'] * 1.2:.1f}%
  • Re-optimize if win rate < {BACKTEST_EXPECTATIONS['win_rate_pct'] * 0.7:.0f}%

{'='*60}
"""
    return report


def send_to_discord(report: str, webhook_url: Optional[str] = None) -> bool:
    """Send report to Discord webhook."""
    import os

    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        print("No Discord webhook URL configured")
        return False

    # Truncate for Discord (2000 char limit)
    if len(report) > 1900:
        report = report[:1900] + "\n... (truncated)"

    payload = {
        "content": f"```\n{report}\n```"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 204
    except Exception as e:
        print(f"Failed to send to Discord: {e}")
        return False


def save_report(report: str, metrics: Dict[str, Any]) -> Path:
    """Save report to file."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Save text report
    report_path = REPORT_DIR / f"performance_{timestamp}.txt"
    report_path.write_text(report)

    # Save JSON metrics
    json_path = REPORT_DIR / f"metrics_{timestamp}.json"
    json_path.write_text(json.dumps(metrics, indent=2, default=str))

    return report_path


def main():
    parser = argparse.ArgumentParser(description="ArgusNexus Monthly Performance Report")
    parser.add_argument("--months", type=int, default=1, help="Number of months to analyze")
    parser.add_argument("--discord", action="store_true", help="Send to Discord")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")

    args = parser.parse_args()

    # Generate report
    report = generate_report(args.months)
    metrics = calculate_live_metrics(args.months)

    # Save report
    report_path = save_report(report, metrics)

    # Output
    if not args.quiet:
        print(report)
        print(f"\nReport saved to: {report_path}")

    # Send to Discord if requested
    if args.discord:
        if send_to_discord(report):
            print("✓ Sent to Discord")
        else:
            print("✗ Failed to send to Discord")


if __name__ == "__main__":
    main()
