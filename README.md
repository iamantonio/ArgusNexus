# ArgusNexus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     █████╗ ██████╗  ██████╗ ██╗   ██╗███████╗                   ║
║    ██╔══██╗██╔══██╗██╔════╝ ██║   ██║██╔════╝                   ║
║    ███████║██████╔╝██║  ███╗██║   ██║███████╗                   ║
║    ██╔══██║██╔══██╗██║   ██║██║   ██║╚════██║                   ║
║    ██║  ██║██║  ██║╚██████╔╝╚██████╔╝███████║                   ║
║    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝                   ║
║                                                                  ║
║         Professional Cryptocurrency Trading Framework           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

**The Truth Engine** - A trading system where every decision is explainable and every outcome is logged.

> **Disclaimer**: This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## Features

| Component | Description |
|-----------|-------------|
| **Truth Engine** | SQLite-based decision logging - every trade is traceable |
| **TURTLE-4 Strategy** | Donchian Channel breakout with Chandelier Exit |
| **10-Layer Risk Gate** | Comprehensive risk management system |
| **Multi-Timeframe Analysis** | Align signals across 1h, 4h timeframes |
| **Paper Trading** | Full simulation with SHORT support |
| **Dashboard API** | FastAPI backend for monitoring |

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ArgusNexus/ArgusNexus.git
cd ArgusNexus

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy example configurations
cp config.yaml.example config.yaml
cp .env.example .env

# 5. Edit .env with your API keys
# See .env.example for required keys

# 6. Run the paper trader
python scripts/live_unified_trader.py

# 7. Start the dashboard API (optional)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 8. Access the dashboard
open http://localhost:8000
```

## The Strategy: TURTLE-4 "The Ratchet"

A Donchian Channel breakout system with trailing Chandelier Exit. Supports both LONG and SHORT positions.

| Parameter | LONG | SHORT |
|-----------|------|-------|
| Entry | Break above 55-period High | Break below 55-period Low |
| Trend Filter | Price > 200 SMA | Price < 200 SMA |
| Volume | > 1.5x 20-period avg | > 1.5x 20-period avg |
| Exit (Primary) | Chandelier: HH - 3×ATR | Inverse Chandelier: LL + 3×ATR |
| Exit (Backup) | 55-period Low breakdown | 55-period High breakout |
| Position Size | 100% base | 50% base (conservative) |

**The Ratchet**: LONG stops only move UP (lock gains on rallies). SHORT stops only move DOWN (lock gains on drops).

## Architecture

```
ArgusNexus/
├── src/
│   ├── strategy/          # Trading strategies (Donchian, etc.)
│   ├── risk/              # 10-layer risk management
│   ├── execution/         # Paper/Live execution router
│   ├── truth/             # Decision logging (The Truth Engine)
│   ├── api/               # FastAPI dashboard backend
│   └── data/              # Data loaders (Coinbase)
├── scripts/
│   ├── live_unified_trader.py    # Multi-asset paper trader
│   ├── run_backtest.py           # Backtesting runner
│   └── watchdog.sh               # Health monitoring
├── data/
│   ├── demo.db            # Demo database with sample trades
│   └── golden_test.db     # Test fixture
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## The Truth Engine

Three tables track everything:

| Table | Purpose |
|-------|---------|
| `decisions` | Why did we consider this trade? |
| `orders` | What orders were placed? |
| `trades` | What was the P&L outcome? |

**Every trade is traceable:** `trades` → `orders` → `decisions`

## Configuration

See `config.yaml.example` for all available options:

- **Risk Settings**: Per-trade risk, daily limits, circuit breakers
- **Strategy Parameters**: Donchian periods, ATR multipliers
- **Session Management**: Dead zones, liquidity filters
- **Multi-Timeframe**: Timeframe weights, conflict resolution

## Monitoring

| Component | Description |
|-----------|-------------|
| `/api/system/status` | Health check endpoint |
| `scripts/watchdog.sh` | Health monitoring script |
| Dashboard | Real-time trader status |

## Principles

1. **If we can't explain it, we don't trade it**
2. **If we can't log it, it didn't happen**
3. **Simple > Complex until simple is profitable**
4. **ONE strategy only until that strategy is validated**

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_strategy.py -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Build your own trading system. Log everything. Trust the math.*
