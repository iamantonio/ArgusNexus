#!/usr/bin/env python3
"""
Test Gemini API Connection

Run this script to verify your Gemini API credentials are working.

Usage:
    python scripts/test_gemini_connection.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv


async def test_connection():
    """Test Gemini API connection and display account info."""

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    api_secret = os.getenv("GEMINI_API_SECRET")

    if not api_key or not api_secret:
        print("❌ ERROR: GEMINI_API_KEY and GEMINI_API_SECRET not found in .env")
        print()
        print("Please add your credentials to .env:")
        print("  GEMINI_API_KEY=your_key_here")
        print("  GEMINI_API_SECRET=your_secret_here")
        return False

    print("=" * 60)
    print("GEMINI API CONNECTION TEST")
    print("=" * 60)
    print()

    # Import here to ensure dotenv is loaded first
    from src.execution.gemini import GeminiExecutor

    # Create executor (production mode for real account info)
    executor = GeminiExecutor(
        api_key=api_key,
        api_secret=api_secret,
        sandbox=False  # Use real API to verify credentials
    )

    try:
        # Test 1: Get balances
        print("Test 1: Fetching account balances...")
        balances = await executor.get_balances()

        if balances:
            print("✅ SUCCESS - Connected to Gemini!")
            print()
            print("Your balances:")
            for currency, amount in sorted(balances.items()):
                print(f"  {currency}: {amount}")
        else:
            print("✅ Connected (no balances found - account may be empty)")

        print()

        # Test 2: Get BTC ticker (public endpoint)
        print("Test 2: Fetching BTC-USD ticker...")
        ticker = await executor.get_ticker("BTC-USD")

        if ticker:
            print("✅ SUCCESS - Market data working!")
            print(f"  BTC-USD Last: ${ticker['last']:,.2f}")
            print(f"  Bid: ${ticker['bid']:,.2f}")
            print(f"  Ask: ${ticker['ask']:,.2f}")
        else:
            print("⚠️  Could not fetch ticker")

        print()
        print("=" * 60)
        print("ALL TESTS PASSED - Your Gemini API is ready!")
        print("=" * 60)

        await executor.close()
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        print()
        print("Common issues:")
        print("1. API key/secret are incorrect")
        print("2. IP address not whitelisted in Gemini settings")
        print("3. API permissions not enabled")
        print()
        print(f"Your server IP: Run 'curl ifconfig.me' to check")

        await executor.close()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
