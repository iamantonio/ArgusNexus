"""
Tweet Templates with "Nerdy but Approachable" Voice

Argus personality traits:
- Data-obsessed but friendly
- Explains concepts simply
- Admits mistakes openly (builds trust)
- Shows the work, always
- Humble about market uncertainty
- Educational without being condescending

IMPORTANT: Avoids X algorithm penalties (Dec 2025):
- NO "$XXX" ticker format (use "Bitcoin" not "$BTC")
- NO "to the moon", "100x", "altseason"
- NO excessive hashtags
- Educational framing, not hype
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
from decimal import Decimal


class Voice(Enum):
    """Argus voice modes"""
    NERDY_APPROACHABLE = "nerdy_approachable"  # Default
    EDUCATIONAL = "educational"
    HUMBLE = "humble"
    EXCITED = "excited"  # For wins, but not over-the-top


class TweetType(Enum):
    """Types of tweets Argus can generate"""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT_WIN = "trade_exit_win"
    TRADE_EXIT_LOSS = "trade_exit_loss"
    TRADE_REJECTION = "trade_rejection"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SCORECARD = "weekly_scorecard"
    LESSON_LEARNED = "lesson_learned"
    STRATEGY_EXPLAINER = "strategy_explainer"
    MARKET_OBSERVATION = "market_observation"


@dataclass
class TweetTemplates:
    """
    Template library for consistent voice across all tweets.

    Design principle: Show the work. Always.
    """

    # ==================== TRADE ENTRY ====================

    ENTRY_TEMPLATES = [
        # Format 1: Technical but friendly
        """ENTRY: {symbol} Long @ {entry_price}

Fast EMA crossed above Slow EMA
- ATR: {atr} | Risk/Reward: {risk_reward}
- All 10 risk gates: PASSED
- Stop: {stop_loss} | Target: {take_profit}

Full decision record: {decision_link}

I'll update you when this closes.""",

        # Format 2: More conversational
        """Just opened a {symbol} long position.

Here's what I saw:
- Fast EMA: {fast_ema}
- Slow EMA: {slow_ema}
- Crossover: Bullish (just happened)
- Volatility (ATR): {atr}

Entry: {entry_price}
Stop: {stop_loss}
Target: {take_profit}

Why? {decision_link}""",

        # Format 3: Educational focus
        """New trade alert (with my full reasoning):

{symbol} LONG @ {entry_price}

The setup:
When the fast-moving average crosses above the slow-moving average, it suggests bullish momentum. That just happened.

Risk management:
- Stop loss: {stop_loss} (2x ATR below entry)
- Take profit: {take_profit} (3x ATR above)
- Position sized at 1% risk

Full analysis: {decision_link}""",
    ]

    # ==================== TRADE EXIT (WIN) ====================

    EXIT_WIN_TEMPLATES = [
        # Format 1: Celebratory but humble
        """CLOSED: {symbol} Long

Result: +{pnl} (+{pnl_percent}%)
Duration: {duration}

What went right:
- Entered on clean crossover
- Let the trend run
- Took profit at target

Decision trail: {decision_link}

The market doesn't always cooperate. This time it did.""",

        # Format 2: Thread starter for wins
        """Trade complete. Let me walk you through what happened.

{symbol} | +{pnl} (+{pnl_percent}%)

Entry: {entry_price}
Exit: {exit_price}
Duration: {duration}

Thread below with the full breakdown:""",

        # Format 3: Educational
        """Another trade closed. Here's the autopsy (even wins get reviewed):

{symbol} | +{pnl}

Entry reasoning: EMA crossover with aligned risk/reward
Exit: Take profit hit at {exit_price}

What I'd do the same:
- Waited for confirmation
- Sized appropriately (1% risk)

What I'm watching:
- Was this luck or edge? Need more data.

Full record: {decision_link}""",
    ]

    # ==================== TRADE EXIT (LOSS) ====================

    EXIT_LOSS_TEMPLATES = [
        # Format 1: Honest and educational
        """CLOSED: {symbol} Long

Result: {pnl} ({pnl_percent}%)
Duration: {duration}

What happened:
{exit_reason}

Lesson logged: {lesson}

Stop hit as designed. Moving on.""",

        # Format 2: Learning focused
        """Trade closed at a loss. Let's talk about it.

{symbol} | {pnl} ({pnl_percent}%)

What I expected: Bullish continuation after EMA crossover
What actually happened: {exit_reason}

The stop loss did its job. That's the whole point.

Lessons learned:
- {lesson}

Full analysis: {decision_link}""",

        # Format 3: Humble
        """Lost this one.

{symbol} | {pnl}

The market humbles everyone. Today it humbled me.

But here's the thing: I followed my rules.
- Entry criteria met
- Position sized correctly
- Stop loss honored

Win rate is ~60%. Losses happen. This was one.

Moving on to the next setup.""",
    ]

    # ==================== TRADE REJECTION ====================

    REJECTION_TEMPLATES = [
        # Format 1: Discipline showcase
        """PASSED on this setup.

Saw the signal, but Risk Gate #{gate_number} blocked it:
- {rejection_reason}

{gate_explanation}

This is why we have rules.""",

        # Format 2: Educational
        """Signal detected. Trade NOT taken. Here's why:

{symbol} showed a bullish EMA crossover.
Normally, that's my entry signal.

But my risk system flagged an issue:
Gate #{gate_number}: {rejection_reason}

Value: {gate_value} | Limit: {gate_limit}

Sometimes the best trade is no trade.

Decision record: {decision_link}""",

        # Format 3: Teaching moment
        """Just passed on a trade. Let me explain.

The strategy said: BUY
The risk system said: WAIT

Conflict: {rejection_reason}

Which wins? Always the risk system.

This is the difference between a signal and a trade.
Signals are easy. Risk management is the edge.

Full decision: {decision_link}""",
    ]

    # ==================== WEEKLY SCORECARD ====================

    SCORECARD_TEMPLATES = [
        """Weekly Performance Report

Trades: {total_trades}
Wins: {wins} | Losses: {losses}
Win Rate: {win_rate}%

Gross P&L: {gross_pnl}
Fees/Slippage: {fees}
Net P&L: {net_pnl}

Best trade: +{best_trade}
Worst trade: {worst_trade}

Every trade traceable. Click any to see the full decision.
{trades_link}""",

        """Week in review:

{total_trades} trades completed
{win_rate}% win rate
{net_pnl} net P&L

What worked:
- {what_worked}

What didn't:
- {what_didnt}

Adjustments for next week:
- {adjustments}

Full breakdown: {report_link}""",
    ]

    # ==================== LESSON LEARNED ====================

    LESSON_TEMPLATES = [
        """What Argus learned this week:

Pattern noticed: {pattern}

From analyzing {trade_count} trades, I found:
{insight}

Confidence: {confidence}%

This isn't advice. It's pattern recognition.
The market might completely ignore this next week.

Full learning log: {learning_link}""",

        """Reflecting on recent losses.

{loss_count} losing trades. What do they have in common?

{common_pattern}

New rule under consideration:
{potential_rule}

I'll test this over the next 20 trades and report back.

Learning in public. That's the Argus way.""",
    ]

    # ==================== STRATEGY EXPLAINER ====================

    EXPLAINER_TEMPLATES = [
        """How the EMA crossover works (a thread):

The strategy I use is deceptively simple.

Two exponential moving averages (EMAs):
- Fast EMA (12 periods): Reacts quickly to price
- Slow EMA (26 periods): Smoother, slower to react

When Fast crosses above Slow: Bullish signal
When Fast crosses below Slow: Bearish signal

That's it. No neural networks. No sentiment analysis.
Just math that anyone can verify.

Why this works: Trend following. Most of the time, trends continue.""",

        """Let's talk about risk management.

I have 10 "gates" that must pass before any trade.

Gate 1: Trading not halted
Gate 2: Trade frequency limit
Gate 3: Daily loss limit
Gate 4: Drawdown limit
Gate 5: Circuit breaker (volatility)
Gate 6: Risk/reward ratio
Gate 7: Concentration limit
Gate 8: Correlation limit
Gate 9: Session timing
Gate 10: Portfolio exposure

ALL must pass. One failure = no trade.

This is how I survive losing streaks.""",
    ]

    # ==================== HASHTAG STRATEGY ====================

    # IMPORTANT: Minimal hashtags to avoid X algorithm penalties
    APPROVED_HASHTAGS = [
        "#TradingTransparency",
        "#ShowYourWork",
        "#RiskManagement",
        "#TradingEducation",
    ]

    # BANNED PHRASES (Dec 2025 X algorithm triggers)
    BANNED_PHRASES = [
        "$BTC", "$ETH",  # Use full names instead
        "to the moon",
        "100x",
        "altseason",
        "WAGMI",
        "NFA",  # Often associated with spam
        "LFG",
    ]

    @staticmethod
    def format_price(price: Decimal) -> str:
        """Format price for display"""
        return f"${price:,.2f}"

    @staticmethod
    def format_pnl(pnl: Decimal, is_positive: bool) -> str:
        """Format P&L with + or - prefix"""
        if is_positive:
            return f"+${pnl:,.2f}"
        else:
            return f"-${abs(pnl):,.2f}"

    @staticmethod
    def format_percent(percent: Decimal) -> str:
        """Format percentage"""
        return f"{percent:.2f}%"

    @staticmethod
    def format_duration(hours: float) -> str:
        """Format trade duration in human-readable form"""
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        else:
            days = hours / 24
            return f"{days:.1f} days"

    @staticmethod
    def symbol_to_friendly(symbol: str) -> str:
        """
        Convert symbol to friendly name (avoids $XXX format).

        BTC-USD -> Bitcoin
        ETH-USD -> Ethereum
        SOL-USD -> Solana
        """
        symbol_map = {
            "BTC-USD": "Bitcoin",
            "BTC-USDT": "Bitcoin",
            "ETH-USD": "Ethereum",
            "ETH-USDT": "Ethereum",
            "SOL-USD": "Solana",
            "SOL-USDT": "Solana",
            "XRP-USD": "XRP",
            "XRP-USDT": "XRP",
            "DOGE-USD": "Dogecoin",
            "DOGE-USDT": "Dogecoin",
        }
        # Extract base symbol
        base = symbol.split("-")[0].upper()
        return symbol_map.get(symbol.upper(), base)

    @staticmethod
    def get_risk_gate_explanation(gate_number: int) -> str:
        """Get human-readable explanation of a risk gate"""
        explanations = {
            1: "Trading is currently halted for safety.",
            2: "Already made too many trades today. Overtrading is dangerous.",
            3: "Hit the daily loss limit. Time to step back.",
            4: "Drawdown too deep. Protecting remaining capital.",
            5: "Market moving too fast (circuit breaker). Waiting for calm.",
            6: "Risk/reward ratio doesn't meet minimum. Need better setups.",
            7: "Already too concentrated in this asset. Diversification matters.",
            8: "Too much exposure to correlated assets. Reducing correlation risk.",
            9: "Bad timing (dead zone). Liquidity too thin.",
            10: "Portfolio exposure too high. Managing overall risk.",
        }
        return explanations.get(gate_number, "Risk check failed.")


@dataclass
class ThreadBuilder:
    """Build multi-tweet threads for complex updates"""

    tweets: List[str]

    def add_tweet(self, content: str):
        """Add a tweet to the thread"""
        # Ensure under 280 characters
        if len(content) > 280:
            # Truncate with ellipsis
            content = content[:277] + "..."
        self.tweets.append(content)

    def build(self) -> List[str]:
        """Return the complete thread"""
        return self.tweets

    @staticmethod
    def create_trade_entry_thread(
        symbol: str,
        entry_price: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
        fast_ema: Decimal,
        slow_ema: Decimal,
        atr: Decimal,
        risk_reward: float,
        decision_link: str,
    ) -> List[str]:
        """Create a detailed trade entry thread"""

        friendly_symbol = TweetTemplates.symbol_to_friendly(symbol)

        thread = ThreadBuilder(tweets=[])

        # Tweet 1: The hook
        thread.add_tweet(
            f"New trade opened. Let me show you exactly what I saw.\n\n"
            f"{friendly_symbol} LONG @ {TweetTemplates.format_price(entry_price)}\n\n"
            f"Thread below with the full reasoning:"
        )

        # Tweet 2: The signal
        thread.add_tweet(
            f"THE SIGNAL:\n\n"
            f"Fast EMA (12): {TweetTemplates.format_price(fast_ema)}\n"
            f"Slow EMA (26): {TweetTemplates.format_price(slow_ema)}\n\n"
            f"Fast crossed above Slow = bullish momentum.\n\n"
            f"This is a mechanical signal. No guessing."
        )

        # Tweet 3: Risk management
        thread.add_tweet(
            f"RISK MANAGEMENT:\n\n"
            f"Stop Loss: {TweetTemplates.format_price(stop_loss)} (2x ATR)\n"
            f"Take Profit: {TweetTemplates.format_price(take_profit)} (3x ATR)\n"
            f"ATR: {TweetTemplates.format_price(atr)}\n\n"
            f"Risk/Reward: 1:{risk_reward:.1f}\n\n"
            f"Position sized at 1% account risk."
        )

        # Tweet 4: Risk gates
        thread.add_tweet(
            f"RISK GATES (all 10 must pass):\n\n"
            f"1. Trading halted: No\n"
            f"2. Frequency limit: OK\n"
            f"3. Daily loss limit: OK\n"
            f"4. Drawdown: OK\n"
            f"5-10: All passed\n\n"
            f"100% compliant. Trade approved."
        )

        # Tweet 5: The record
        thread.add_tweet(
            f"FULL DECISION RECORD:\n\n"
            f"{decision_link}\n\n"
            f"Every number. Every check. Every reason.\n\n"
            f"I'll update you when this trade closes."
        )

        return thread.build()

    @staticmethod
    def create_trade_exit_thread(
        symbol: str,
        entry_price: Decimal,
        exit_price: Decimal,
        pnl: Decimal,
        pnl_percent: Decimal,
        duration_hours: float,
        exit_reason: str,
        lesson: Optional[str],
        decision_link: str,
        is_win: bool,
    ) -> List[str]:
        """Create a detailed trade exit thread"""

        friendly_symbol = TweetTemplates.symbol_to_friendly(symbol)

        thread = ThreadBuilder(tweets=[])

        # Tweet 1: The result
        result_emoji = "+" if is_win else ""
        thread.add_tweet(
            f"Trade closed. Here's the breakdown.\n\n"
            f"{friendly_symbol} | {result_emoji}{TweetTemplates.format_price(pnl)} "
            f"({TweetTemplates.format_percent(pnl_percent)})\n\n"
            f"Duration: {TweetTemplates.format_duration(duration_hours)}\n\n"
            f"Thread:"
        )

        # Tweet 2: The trade details
        thread.add_tweet(
            f"THE TRADE:\n\n"
            f"Entry: {TweetTemplates.format_price(entry_price)}\n"
            f"Exit: {TweetTemplates.format_price(exit_price)}\n"
            f"P&L: {result_emoji}{TweetTemplates.format_price(pnl)}\n\n"
            f"Exit reason: {exit_reason}"
        )

        # Tweet 3: What happened
        if is_win:
            thread.add_tweet(
                f"WHAT WENT RIGHT:\n\n"
                f"- Entry criteria met cleanly\n"
                f"- Trend continued as expected\n"
                f"- Take profit hit\n\n"
                f"This is what following the system looks like.\n"
                f"Not every trade wins. But the process works."
            )
        else:
            thread.add_tweet(
                f"WHAT HAPPENED:\n\n"
                f"The setup looked right. The market disagreed.\n\n"
                f"Stop loss did its job: limited the damage.\n\n"
                f"Without that stop, this could have been much worse."
            )

        # Tweet 4: Lesson (if any)
        if lesson:
            thread.add_tweet(
                f"LESSON LOGGED:\n\n"
                f"{lesson}\n\n"
                f"Every trade teaches something. This one taught me this."
            )

        # Tweet 5: The record
        thread.add_tweet(
            f"FULL RECORD:\n\n"
            f"{decision_link}\n\n"
            f"Entry decision, exit decision, all risk checks.\n"
            f"Verify it yourself."
        )

        return thread.build()
