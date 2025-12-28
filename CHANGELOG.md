# Changelog

All notable changes to ArgusNexus will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] - 2025-12-28 - Initial Open Source Release

### Features

#### TURTLE-4 Strategy (v6.1) - Bidirectional Trading
- **LONG entries**: Breakout above 55-period Donchian Channel high
- **SHORT entries**: Breakdown below 55-period Donchian Channel low
- **Chandelier Exit**: Trailing stop that only moves in profitable direction
- **Inverse Chandelier**: For SHORT positions, stop ratchets DOWN
- **Multi-timeframe analysis**: 1h/4h signal alignment with weighted scoring
- **Setup grading**: A+/A/B/C grades determine position sizing

#### Truth Engine - Decision Logging
- SQLite-based logging of every trading decision
- Three-table architecture: `decisions` → `orders` → `trades`
- Full traceability from trade outcome back to original signal
- Reflection system for post-trade analysis

#### 10-Layer Risk Gate
- Per-trade risk limits (default: 1% of capital)
- Daily loss limits with automatic halt
- Max drawdown protection
- Circuit breaker with cooldown period
- Asset concentration limits
- Correlated exposure limits
- Portfolio-level risk coordination

#### Paper Trading with SHORT Support
- Full simulation of margin trading for SHORT positions
- 100% collateral requirement (conservative)
- Inverse P&L calculation for shorts
- Position tracking with unrealized P&L

#### Dashboard API
- FastAPI backend for monitoring
- Real-time health check endpoint
- Decision and trade query endpoints
- Social card generation for sharing

### Architecture
```
src/
├── strategy/      # Trading strategies (Donchian, etc.)
├── risk/          # 10-layer risk management
├── execution/     # Paper/Live execution router
├── truth/         # Decision logging (Truth Engine)
├── api/           # FastAPI dashboard backend
└── data/          # Data loaders (Coinbase)
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.
