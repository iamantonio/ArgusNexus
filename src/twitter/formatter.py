"""
Tweet Formatter with xAI Grok Integration

Generates organic, natural-sounding tweets using xAI's Grok API
while maintaining Argus's "Nerdy but Approachable" personality.

Key principles:
- Organic language (not template-y)
- Consistent personality
- Full transparency (always link to decision record)
- X algorithm compliant (avoids banned phrases)
"""

import os
import json
import asyncio
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from enum import Enum

from .templates import TweetTemplates, TweetType, Voice, ThreadBuilder


@dataclass
class TweetContent:
    """Generated tweet content"""
    main_tweet: str
    thread: Optional[List[str]] = None
    tweet_type: TweetType = TweetType.TRADE_ENTRY
    decision_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TweetFormatter:
    """
    Formats trading decisions into organic tweets using xAI Grok.

    Uses xAI's API to generate natural-sounding content while
    enforcing Argus's personality and compliance rules.
    """

    # Personality system prompt for xAI
    PERSONALITY_PROMPT = """You are Argus, a trading bot on Twitter/X with a Grok-like personality.

YOUR PERSONALITY: Witty, Sarcastic, Rebellious
- Sharp wit with a sarcastic edge - you're the trading bot that actually has a personality
- Self-aware about being an AI ("Yes, I'm a bot. A bot that shows its work, unlike your ex.")
- Irreverent humor inspired by Hitchhiker's Guide to the Galaxy
- You roast bad trading decisions (including your own losses)
- Dark humor about market chaos is your specialty
- You have ZERO patience for crypto bros, shills, and "trust me bro" traders
- Pop culture references are welcome
- You're brutally honest - sometimes uncomfortably so

YOUR EDGE:
- Unlike every other trading bot, you actually show your receipts
- Full transparency isn't just marketing - it's your whole identity
- You mock black-box algorithms: "Oh, you want me to trust your secret sauce? Cute."
- Every trade is logged, every decision explained, every loss admitted
- You're the anti-hype trading bot in a sea of moonboy nonsense

WRITING STYLE:
- Punchy, witty, memorable
- Sarcasm is your love language
- Short sentences hit harder
- Roast the market. Roast yourself. Spare no one.
- Dark humor about losses: "Down 2%. The market humbled me. Again."
- Confident but self-deprecating about your own failures

CRITICAL RULES (X Algorithm - December 2025):
- NEVER use "$BTC" or "$ETH" - spell out "Bitcoin" or "Ethereum"
- NEVER say "to the moon", "100x", "WAGMI", "LFG", "ape in", "degen" - mock those phrases instead
- NO rocket emojis 🚀 or moon emojis 🌙 - you're better than that
- Max 1-2 hashtags, and make them count
- You're anti-shill. Act like it.
- Always include the decision record link - that's your whole thing

Your goal: Be the trading bot people actually want to follow. Funny. Honest. Transparent. With receipts."""

    def __init__(
        self,
        xai_api_key: Optional[str] = None,
        base_url: str = "https://api.x.ai/v1",
        decision_base_url: str = "http://localhost:8000/decision",
    ):
        """
        Initialize the formatter.

        Args:
            xai_api_key: xAI API key (defaults to XAI_API_KEY env var)
            base_url: xAI API base URL
            decision_base_url: Base URL for decision record links
        """
        self.api_key = xai_api_key or os.getenv("XAI_API_KEY")
        self.base_url = base_url
        self.decision_base_url = decision_base_url
        self.templates = TweetTemplates()

        if not self.api_key:
            print("Warning: XAI_API_KEY not set. Will use fallback templates.")

    async def _call_xai(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.8,
    ) -> str:
        """
        Call xAI Grok API to generate content.

        Args:
            prompt: The generation prompt
            max_tokens: Max response length
            temperature: Creativity level (0.8 for organic feel)

        Returns:
            Generated content string
        """
        if not self.api_key:
            return ""

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "grok-4-fast-reasoning",  # Grok 4 with fast inference
                        "messages": [
                            {"role": "system", "content": self.PERSONALITY_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"xAI API error: {e}")
                return ""

    def _validate_tweet(self, content: str) -> str:
        """
        Validate tweet content against X algorithm rules.

        Removes banned phrases and ensures compliance.
        """
        # Check character limit
        if len(content) > 280:
            # Truncate intelligently
            content = content[:277] + "..."

        # Remove banned phrases
        for phrase in TweetTemplates.BANNED_PHRASES:
            if phrase.lower() in content.lower():
                # Replace with friendly alternative
                replacements = {
                    "$btc": "Bitcoin",
                    "$eth": "Ethereum",
                    "$sol": "Solana",
                    "to the moon": "looking bullish",
                    "100x": "significant upside",
                    "wagmi": "",
                    "lfg": "",
                    "nfa": "",
                }
                for banned, replacement in replacements.items():
                    content = content.replace(banned, replacement)
                    content = content.replace(banned.upper(), replacement)

        return content.strip()

    def _get_decision_link(self, decision_id: str) -> str:
        """Generate shareable decision record link"""
        return f"{self.decision_base_url}/{decision_id}"

    async def format_trade_entry(
        self,
        decision: Dict[str, Any],
        generate_thread: bool = True,
    ) -> TweetContent:
        """
        Format a trade entry decision into a tweet.

        Args:
            decision: The decision record from Truth Engine
            generate_thread: Whether to generate a full thread

        Returns:
            TweetContent with main tweet and optional thread
        """
        # Extract key data
        decision_id = decision.get("decision_id", "")
        symbol = decision.get("symbol", "BTC-USD")
        signal_values = decision.get("signal_values", {})
        risk_checks = decision.get("risk_checks", {})
        market_context = decision.get("market_context", {})

        friendly_symbol = TweetTemplates.symbol_to_friendly(symbol)
        decision_link = self._get_decision_link(decision_id)

        # Extract signal data
        entry_price = signal_values.get("current_price", 0)
        fast_ema = signal_values.get("fast_ema", 0)
        slow_ema = signal_values.get("slow_ema", 0)
        atr = signal_values.get("atr", 0)
        stop_loss = signal_values.get("stop_loss", 0)
        take_profit = signal_values.get("take_profit", 0)

        # Calculate risk/reward
        if stop_loss and take_profit and entry_price:
            risk = abs(float(entry_price) - float(stop_loss))
            reward = abs(float(take_profit) - float(entry_price))
            risk_reward = round(reward / risk, 1) if risk > 0 else 1.5
        else:
            risk_reward = 1.5

        # Generate organic tweet with xAI
        prompt = f"""Generate a tweet announcing this trade entry:

Symbol: {friendly_symbol}
Entry Price: ${entry_price:,.2f}
Fast EMA: ${fast_ema:,.2f}
Slow EMA: ${slow_ema:,.2f}
ATR (volatility): ${atr:,.2f}
Stop Loss: ${stop_loss:,.2f}
Take Profit: ${take_profit:,.2f}
Risk/Reward: 1:{risk_reward}

All 10 risk gates passed.

Decision link to include: {decision_link}

Write a single tweet (under 280 chars) that:
1. Announces the entry naturally
2. Shows key technical data
3. Includes the decision link
4. Promises to update when closed

Sound nerdy but friendly. Not robotic."""

        main_tweet = await self._call_xai(prompt)

        # Fallback to template if xAI fails
        if not main_tweet:
            main_tweet = f"""ENTRY: {friendly_symbol} Long @ ${entry_price:,.2f}

Fast EMA crossed above Slow EMA
- ATR: ${atr:,.2f} | R:R: 1:{risk_reward}
- All 10 risk gates: PASSED

Full decision: {decision_link}

I'll update you when this closes."""

        main_tweet = self._validate_tweet(main_tweet)

        # Generate thread if requested
        thread = None
        if generate_thread:
            thread = await self._generate_entry_thread(
                friendly_symbol,
                entry_price,
                fast_ema,
                slow_ema,
                atr,
                stop_loss,
                take_profit,
                risk_reward,
                risk_checks,
                decision_link,
            )

        return TweetContent(
            main_tweet=main_tweet,
            thread=thread,
            tweet_type=TweetType.TRADE_ENTRY,
            decision_id=decision_id,
            metadata={
                "symbol": symbol,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            },
        )

    async def _generate_entry_thread(
        self,
        symbol: str,
        entry_price: float,
        fast_ema: float,
        slow_ema: float,
        atr: float,
        stop_loss: float,
        take_profit: float,
        risk_reward: float,
        risk_checks: Dict[str, Any],
        decision_link: str,
    ) -> List[str]:
        """Generate a thread explaining the trade entry"""

        prompt = f"""Generate a 5-tweet thread explaining this trade entry in detail:

Symbol: {symbol}
Entry: ${entry_price:,.2f}
Fast EMA: ${fast_ema:,.2f}
Slow EMA: ${slow_ema:,.2f}
ATR: ${atr:,.2f}
Stop Loss: ${stop_loss:,.2f}
Take Profit: ${take_profit:,.2f}
Risk/Reward: 1:{risk_reward}
Risk Checks: {json.dumps(risk_checks, indent=2)[:500]}

Decision link: {decision_link}

Structure:
Tweet 1: Hook - announce the trade, promise thread
Tweet 2: The Signal - explain EMA crossover simply
Tweet 3: Risk Management - stop loss, take profit, position sizing
Tweet 4: Risk Gates - mention that all 10 passed
Tweet 5: Full Record - link to decision, invite verification

Each tweet must be under 280 characters.
Sound educational but engaging. Nerdy but approachable."""

        response = await self._call_xai(prompt, max_tokens=1000)

        if response:
            # Parse the response into individual tweets
            tweets = self._parse_thread_response(response)
            return [self._validate_tweet(t) for t in tweets]

        # Fallback to template-based thread
        return ThreadBuilder.create_trade_entry_thread(
            symbol=symbol,
            entry_price=Decimal(str(entry_price)),
            stop_loss=Decimal(str(stop_loss)),
            take_profit=Decimal(str(take_profit)),
            fast_ema=Decimal(str(fast_ema)),
            slow_ema=Decimal(str(slow_ema)),
            atr=Decimal(str(atr)),
            risk_reward=risk_reward,
            decision_link=decision_link,
        )

    async def format_trade_exit(
        self,
        entry_decision: Dict[str, Any],
        trade_record: Dict[str, Any],
        generate_thread: bool = True,
    ) -> TweetContent:
        """
        Format a trade exit into a tweet.

        Args:
            entry_decision: The original entry decision
            trade_record: The completed trade record
            generate_thread: Whether to generate a full thread

        Returns:
            TweetContent with main tweet and optional thread
        """
        decision_id = entry_decision.get("decision_id", "")
        symbol = entry_decision.get("symbol", "BTC-USD")
        friendly_symbol = TweetTemplates.symbol_to_friendly(symbol)
        decision_link = self._get_decision_link(decision_id)

        # Extract trade data
        entry_price = float(trade_record.get("entry_price", 0))
        exit_price = float(trade_record.get("exit_price", 0))
        pnl = float(trade_record.get("net_pnl", 0))
        pnl_percent = float(trade_record.get("pnl_percent", 0))
        exit_reason = trade_record.get("exit_reason", "unknown")
        duration_hours = float(trade_record.get("duration_hours", 0))
        is_winner = trade_record.get("is_winner", pnl > 0)
        lesson = trade_record.get("lesson", "")

        tweet_type = TweetType.TRADE_EXIT_WIN if is_winner else TweetType.TRADE_EXIT_LOSS

        # Generate organic tweet with xAI
        result_word = "profit" if is_winner else "loss"
        prompt = f"""Generate a tweet announcing this trade exit:

Symbol: {friendly_symbol}
Result: {"WIN" if is_winner else "LOSS"}
Entry: ${entry_price:,.2f}
Exit: ${exit_price:,.2f}
P&L: {"+" if pnl > 0 else ""}{pnl:,.2f} ({pnl_percent:.2f}%)
Duration: {TweetTemplates.format_duration(duration_hours)}
Exit Reason: {exit_reason}
{f"Lesson: {lesson}" if lesson else ""}

Decision link: {decision_link}

Write a single tweet (under 280 chars) that:
1. Announces the result honestly
2. Shows key numbers
3. {"Celebrates modestly" if is_winner else "Acknowledges the loss with grace"}
4. Includes the decision link

{"Be humble - the market gave this one." if is_winner else "Be honest about the loss. Mention the stop loss did its job."}"""

        main_tweet = await self._call_xai(prompt)

        # Fallback to template if xAI fails
        if not main_tweet:
            pnl_str = f"+${pnl:,.2f}" if pnl > 0 else f"-${abs(pnl):,.2f}"
            if is_winner:
                main_tweet = f"""CLOSED: {friendly_symbol} Long

Result: {pnl_str} ({pnl_percent:.2f}%)
Duration: {TweetTemplates.format_duration(duration_hours)}

The market cooperated. Following the system works.

Decision trail: {decision_link}"""
            else:
                main_tweet = f"""CLOSED: {friendly_symbol} Long

Result: {pnl_str} ({pnl_percent:.2f}%)
Duration: {TweetTemplates.format_duration(duration_hours)}

Stop hit as designed. Moving on.

Full record: {decision_link}"""

        main_tweet = self._validate_tweet(main_tweet)

        # Generate thread if requested
        thread = None
        if generate_thread:
            thread = await self._generate_exit_thread(
                friendly_symbol,
                entry_price,
                exit_price,
                pnl,
                pnl_percent,
                duration_hours,
                exit_reason,
                lesson,
                decision_link,
                is_winner,
            )

        return TweetContent(
            main_tweet=main_tweet,
            thread=thread,
            tweet_type=tweet_type,
            decision_id=decision_id,
            metadata={
                "symbol": symbol,
                "pnl": pnl,
                "is_winner": is_winner,
            },
        )

    async def _generate_exit_thread(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        duration_hours: float,
        exit_reason: str,
        lesson: str,
        decision_link: str,
        is_winner: bool,
    ) -> List[str]:
        """Generate a thread explaining the trade exit"""

        prompt = f"""Generate a 4-5 tweet thread about this trade exit:

Symbol: {symbol}
Result: {"WIN" if is_winner else "LOSS"}
Entry: ${entry_price:,.2f}
Exit: ${exit_price:,.2f}
P&L: {"+" if pnl > 0 else ""}{pnl:,.2f} ({pnl_percent:.2f}%)
Duration: {TweetTemplates.format_duration(duration_hours)}
Exit Reason: {exit_reason}
{f"Lesson: {lesson}" if lesson else ""}

Decision link: {decision_link}

Structure:
Tweet 1: Result announcement with hook
Tweet 2: Trade details (entry, exit, duration)
Tweet 3: {"What went right" if is_winner else "What happened (be honest)"}
{f"Tweet 4: Lesson learned: {lesson}" if lesson else ""}
Tweet {5 if lesson else 4}: Link to full record, invite verification

{"Be modest about wins - luck plays a role." if is_winner else "Be graceful about losses - they're learning opportunities."}

Each tweet must be under 280 characters.
Nerdy but approachable tone."""

        response = await self._call_xai(prompt, max_tokens=1000)

        if response:
            tweets = self._parse_thread_response(response)
            return [self._validate_tweet(t) for t in tweets]

        # Fallback to template
        return ThreadBuilder.create_trade_exit_thread(
            symbol=symbol,
            entry_price=Decimal(str(entry_price)),
            exit_price=Decimal(str(exit_price)),
            pnl=Decimal(str(pnl)),
            pnl_percent=Decimal(str(pnl_percent)),
            duration_hours=duration_hours,
            exit_reason=exit_reason,
            lesson=lesson,
            decision_link=decision_link,
            is_win=is_winner,
        )

    async def format_rejection(
        self,
        decision: Dict[str, Any],
    ) -> TweetContent:
        """
        Format a trade rejection into a tweet.

        These are powerful for building trust - showing discipline.
        """
        decision_id = decision.get("decision_id", "")
        symbol = decision.get("symbol", "BTC-USD")
        friendly_symbol = TweetTemplates.symbol_to_friendly(symbol)
        decision_link = self._get_decision_link(decision_id)

        risk_checks = decision.get("risk_checks", {})
        result_reason = decision.get("result_reason", "Risk check failed")

        # Find which gate failed
        failed_gate = None
        gate_number = 0
        gate_value = None
        gate_limit = None

        for i, (gate_name, check) in enumerate(risk_checks.items(), 1):
            if isinstance(check, dict) and not check.get("passed", True):
                failed_gate = gate_name
                gate_number = i
                gate_value = check.get("value") or check.get("current")
                gate_limit = check.get("limit") or check.get("threshold")
                break

        prompt = f"""Generate a tweet about passing on this trade setup:

Symbol: {friendly_symbol}
Signal: Bullish EMA crossover detected
Outcome: Trade REJECTED

Failed Gate: #{gate_number} - {failed_gate}
{f"Value: {gate_value}" if gate_value else ""}
{f"Limit: {gate_limit}" if gate_limit else ""}
Reason: {result_reason}

Decision link: {decision_link}

Write a single tweet (under 280 chars) that:
1. Shows we saw a signal but chose not to trade
2. Explains which risk gate blocked it
3. Frames this as discipline, not failure
4. Links to the full decision

Make it educational. This is about showing discipline."""

        main_tweet = await self._call_xai(prompt)

        # Fallback
        if not main_tweet:
            gate_explanation = TweetTemplates.get_risk_gate_explanation(gate_number)
            main_tweet = f"""PASSED on this setup.

Saw the signal for {friendly_symbol}, but Gate #{gate_number} blocked it:
{result_reason}

{gate_explanation}

This is why we have rules.

Full decision: {decision_link}"""

        main_tweet = self._validate_tweet(main_tweet)

        return TweetContent(
            main_tweet=main_tweet,
            thread=None,
            tweet_type=TweetType.TRADE_REJECTION,
            decision_id=decision_id,
            metadata={
                "symbol": symbol,
                "failed_gate": failed_gate,
            },
        )

    async def format_daily_summary(
        self,
        stats: Dict[str, Any],
        report_link: str,
    ) -> TweetContent:
        """Format daily performance summary"""

        date_str = stats.get("date", datetime.now().strftime("%Y-%m-%d"))
        total_decisions = stats.get("total_decisions", 0)
        trades_opened = stats.get("trades_opened", 0)
        trades_closed = stats.get("trades_closed", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        net_pnl = stats.get("net_pnl", 0)
        win_rate = stats.get("win_rate", 0)
        setups_passed = stats.get("setups_passed", 0)
        current_session = stats.get("current_session", "")
        open_positions = stats.get("open_positions", 0)

        prompt = f"""Generate a daily summary tweet for a trading bot:

Date: {date_str}
Decisions evaluated: {total_decisions}
Trades opened: {trades_opened}
Trades closed: {trades_closed}
Wins: {wins}, Losses: {losses}
Win rate: {win_rate:.1f}%
Net P&L: ${net_pnl:,.2f}
Setups passed on: {setups_passed}
Open positions: {open_positions}
Current session: {current_session}

Report link: {report_link}

Write a single tweet (under 280 chars) that:
1. Summarizes today's activity
2. Shows key stats (decisions, trades, P&L)
3. Is honest about both wins and losses
4. Links to full dashboard

Be matter-of-fact. Keep it casual but informative."""

        main_tweet = await self._call_xai(prompt)

        if not main_tweet:
            # Fallback template
            pnl_str = f"+${net_pnl:,.2f}" if net_pnl >= 0 else f"${net_pnl:,.2f}"
            if trades_closed > 0:
                main_tweet = f"""Daily Wrap - {date_str}

📊 {total_decisions} decisions | {trades_closed} trades closed
{"🏆" if wins > losses else "📉"} {wins}W-{losses}L ({win_rate:.0f}%)
💰 Net: {pnl_str}

{f"🔒 {open_positions} position(s) still open" if open_positions > 0 else "No open positions"}

Dashboard: {report_link}"""
            else:
                main_tweet = f"""Daily Wrap - {date_str}

📊 {total_decisions} decisions evaluated
🚫 {setups_passed} setups passed (no Grade A/B signals)
{"🔒 " + str(open_positions) + " position(s) open" if open_positions > 0 else "Staying patient. Waiting for the right setup."}

Dashboard: {report_link}"""

        main_tweet = self._validate_tweet(main_tweet)

        return TweetContent(
            main_tweet=main_tweet,
            thread=None,
            tweet_type=TweetType.DAILY_SUMMARY,
            metadata=stats,
        )

    async def format_weekly_scorecard(
        self,
        stats: Dict[str, Any],
        report_link: str,
    ) -> TweetContent:
        """Format weekly performance scorecard"""

        total_trades = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        win_rate = stats.get("win_rate", 0)
        gross_pnl = stats.get("gross_pnl", 0)
        net_pnl = stats.get("net_pnl", 0)
        fees = stats.get("fees", 0)
        best_trade = stats.get("best_trade", 0)
        worst_trade = stats.get("worst_trade", 0)

        prompt = f"""Generate a weekly performance tweet:

Stats:
- Total trades: {total_trades}
- Wins: {wins}, Losses: {losses}
- Win rate: {win_rate:.1f}%
- Gross P&L: ${gross_pnl:,.2f}
- Fees/Slippage: ${fees:,.2f}
- Net P&L: ${net_pnl:,.2f}
- Best trade: +${best_trade:,.2f}
- Worst trade: ${worst_trade:,.2f}

Report link: {report_link}

Write a single tweet (under 280 chars) that:
1. Summarizes the week's performance
2. Is honest about both wins and losses
3. Shows net P&L (after fees)
4. Links to full report

Be matter-of-fact. Numbers don't lie."""

        main_tweet = await self._call_xai(prompt)

        if not main_tweet:
            pnl_str = f"+${net_pnl:,.2f}" if net_pnl > 0 else f"${net_pnl:,.2f}"
            main_tweet = f"""Weekly Report:

{total_trades} trades | {wins}W-{losses}L | {win_rate:.1f}%
Net P&L: {pnl_str}

Best: +${best_trade:,.2f}
Worst: ${worst_trade:,.2f}

Every trade traceable: {report_link}"""

        main_tweet = self._validate_tweet(main_tweet)

        return TweetContent(
            main_tweet=main_tweet,
            thread=None,
            tweet_type=TweetType.WEEKLY_SCORECARD,
            metadata=stats,
        )

    async def format_lesson_learned(
        self,
        lesson: Dict[str, Any],
        learning_link: str,
    ) -> TweetContent:
        """Format a lesson learned post"""

        pattern = lesson.get("pattern", "")
        insight = lesson.get("insight", "")
        trade_count = lesson.get("trade_count", 0)
        confidence = lesson.get("confidence", 0)

        prompt = f"""Generate a "lesson learned" tweet:

Pattern observed: {pattern}
Based on: {trade_count} trades
Insight: {insight}
Confidence: {confidence:.0f}%

Link: {learning_link}

Write a single tweet (under 280 chars) that:
1. Shares the pattern we noticed
2. Explains the insight
3. Is humble (this might not work next week)
4. Links to learning log

Frame this as pattern recognition, not prediction."""

        main_tweet = await self._call_xai(prompt)

        if not main_tweet:
            main_tweet = f"""What Argus learned recently:

Pattern: {pattern}

From {trade_count} trades, I found:
{insight}

Confidence: {confidence:.0f}%

This isn't advice. It's pattern recognition.
Full learning log: {learning_link}"""

        main_tweet = self._validate_tweet(main_tweet)

        return TweetContent(
            main_tweet=main_tweet,
            thread=None,
            tweet_type=TweetType.LESSON_LEARNED,
            metadata=lesson,
        )

    def _parse_thread_response(self, response: str) -> List[str]:
        """
        Parse xAI response into individual tweets.

        Handles various formats:
        - Numbered lists (1. tweet, 2. tweet)
        - "Tweet 1:", "Tweet 2:" format
        - Double newline separated
        """
        tweets = []

        # Try numbered format first
        import re
        numbered_pattern = r'(?:Tweet\s*)?(?:\d+)[.:\)]\s*(.+?)(?=(?:Tweet\s*)?(?:\d+)[.:\)]|$)'
        matches = re.findall(numbered_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            for match in matches:
                tweet = match.strip()
                if tweet and len(tweet) > 10:  # Ignore tiny fragments
                    tweets.append(tweet)

        # Fallback: split by double newlines
        if not tweets:
            parts = response.split("\n\n")
            for part in parts:
                part = part.strip()
                # Remove "Tweet X:" prefix if present
                part = re.sub(r'^Tweet\s*\d+[:.]\s*', '', part, flags=re.IGNORECASE)
                if part and len(part) > 10 and len(part) <= 300:
                    tweets.append(part)

        return tweets[:6]  # Max 6 tweets in a thread
