# Strategy Module
# V4 Strategies: Multiple validated strategies for portfolio sleeves
#
# Available Strategies:
# - Dual EMA Crossover with ATR Stops (dual_ema)
# - Coinbase Survivor - 15-Day Ratchet (coinbase_survivor)
# - Donchian Channel Breakout (donchian)
# - ADX Trend Strength (adx_trend)
# - MostlyLong BTC with Emergency Exit (mostly_long_btc) - 15% sleeve
# - Vol-Regime Core (vol_regime_core) - 85% core allocation

from .mostly_long_btc import MostlyLongBTCStrategy, create_strategy as create_mostly_long
from .coinbase_survivor import CoinbaseSurvivorStrategy, create_strategy as create_coinbase_survivor
from .vol_regime_core import VolRegimeCoreStrategy, create_strategy as create_vol_regime

__all__ = [
    "MostlyLongBTCStrategy",
    "create_mostly_long",
    "CoinbaseSurvivorStrategy",
    "create_coinbase_survivor",
    "VolRegimeCoreStrategy",
    "create_vol_regime",
]
