# Runbook — Combined Portfolio Strategy

## Overview

This runbook covers operational procedures for the ArgusNexus Combined Portfolio Strategy:
- **Vol-Regime Core (85%)** - Dynamic allocation based on regime + volatility
- **MostlyLong Sleeve (15%)** - Trend-following with emergency exits
- **Portfolio-level DD circuit breaker** - Enforces 25% max drawdown

## Quick Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dd_warning` | 12% | Start de-risking (50% exposure) |
| `dd_critical` | 20% | Full de-risk (0% exposure) |
| `dd_recovery_full` | 10% | Full recovery (requires bull + DD<10%) |
| `dd_recovery_half` | 15% | Partial recovery (50% exposure) |
| `min_bars_in_critical` | 14 | Min days before time-based recovery |
| `recovery_alloc` | 30% | Conservative allocation on time-based recovery |
| `rebalance_cooldown` | 5 days | Min days between rebalances |

---

## Manual Flatten

### When to Use
- Emergency de-risk outside normal circuit breaker
- System malfunction detected
- Exchange issues preventing normal operation

### Command
```bash
# Set state to 0% allocation
python3 -c "
import json
from pathlib import Path

STATE_PATH = Path('runtime/paper_state.json')
state = json.loads(STATE_PATH.read_text())

# Force critical state
state['dd_state'] = 'critical'
state['recovery_mode'] = True
state['bars_in_critical'] = 0

# Write back
STATE_PATH.write_text(json.dumps(state, indent=2))
print('State forced to CRITICAL - next evaluation will flatten')
"
```

### Alternative: Direct Position Close
```bash
# If exchange CLI available
# gemini-cli sell BTC-USD --all

# Verify position is flat
# gemini-cli positions
```

### Verify
- [ ] BTC position = 0
- [ ] Cash = full portfolio value
- [ ] `runtime/paper_state.json` shows `dd_state: critical`
- [ ] Alert log shows manual flatten event

---

## Restart Behavior

### On Boot
1. Script loads `runtime/paper_state.json`
2. Reconciles stored state vs actual balances (if live)
3. If mismatch detected: **fail-closed** (HOLD, no trades)
4. Alerts on any reconciliation failure

### State File Structure
```json
{
  "total_equity": "500.00",
  "btc_qty": "0.005",
  "cash": "300.00",
  "high_water_mark": "550.00",
  "dd_state": "normal",
  "recovery_mode": false,
  "bars_in_critical": 0,
  "sleeve_in_position": true,
  "sleeve_entry_price": "95000.00",
  "last_rebalance_time": "2024-06-15T12:30:45",
  "last_update": "2024-06-16T08:00:00"
}
```

### Recovery from Corrupt State
```bash
# Backup current state
cp runtime/paper_state.json runtime/paper_state.json.bak

# Option 1: Reset to conservative defaults
python3 -c "
import json
from pathlib import Path

default_state = {
    'total_equity': '500',  # <-- ADJUST TO ACTUAL
    'btc_qty': '0',
    'cash': '500',
    'high_water_mark': '500',
    'dd_state': 'critical',
    'recovery_mode': True,
    'bars_in_critical': 0,
    'sleeve_in_position': False,
    'sleeve_entry_price': None,
    'last_rebalance_time': None,
    'last_update': None
}
Path('runtime/paper_state.json').write_text(json.dumps(default_state, indent=2))
print('State reset to conservative defaults')
"

# Option 2: Reconstruct from exchange
# Query actual balances and rebuild state
```

### Mismatch Handling
If `state.btc_qty` doesn't match exchange position:

1. **Log alert** (CRITICAL: POSITION_MISMATCH)
2. **HOLD** - No new trades until resolved
3. Operator must investigate:
   - Was there a partial fill?
   - Did exchange reject order?
   - Is this a stale state file?

---

## Exchange Downtime

### If Market Data Unavailable
- Do NOT trade
- Preserve last known state
- Alert **once per hour** until restored
- Continue polling for reconnection

### Behavior
```
[No data] -> HOLD
[No data for 1 hour] -> Alert: MARKET_DATA_UNAVAILABLE
[Data restored] -> Resume normal evaluation
```

### Manual Override During Downtime
If you need to act during downtime:
1. Manually update state file if needed
2. Execute trades via exchange CLI/UI
3. Update state file to match new position
4. Restart script when market data returns

---

## Order Failures / Partial Fills

### Retry Policy
| Error Type | Retry? | Max Retries | Backoff |
|------------|--------|-------------|---------|
| Network timeout | Yes | 3 | Exponential |
| Rate limit | Yes | 5 | 60s fixed |
| Insufficient funds | No | 0 | N/A |
| Invalid order | No | 0 | N/A |
| Exchange error | Yes | 2 | 30s |

### When to Stop Trading
- **3 consecutive failures** → Alert + HOLD
- **Insufficient funds** → Alert + manual intervention needed
- **Position mismatch > 1%** → Alert + HOLD until reconciled

### Reconciliation Process
1. Query actual exchange balance
2. Compare to `state.btc_qty`
3. If mismatch:
   - Log discrepancy with timestamp
   - Alert: POSITION_MISMATCH
   - Options:
     a. Update state to match exchange (trust exchange)
     b. Execute corrective trade (trust state)
     c. Manual review (trust neither)

---

## Daily Checklist (Operator)

### Morning (After Daily Close)
- [ ] Confirm latest daily bar timestamp in logs
- [ ] Check current DD and dd_state
- [ ] Review any overnight alerts
- [ ] Verify state file was updated

### Commands
```bash
# Check latest state
cat runtime/paper_state.json | jq '.'

# Check latest snapshot
tail -1 runtime/daily_snapshots.jsonl | jq '.'

# Check recent alerts
tail -5 runtime/alerts.jsonl | jq '.'

# Check service status
systemctl status argusnexus-burnin

# Check logs
journalctl -u argusnexus-burnin --since "1 hour ago"
```

### Weekly
- [ ] Review trade log for the week
- [ ] Compare performance vs BTC B&H
- [ ] Verify DD stayed within bounds
- [ ] Check for any recurring alerts

---

## Alert Reference

| Alert | Level | Meaning | Action |
|-------|-------|---------|--------|
| `DD_INFO` | INFO | DD >= 10% | Monitor |
| `DD_WARNING` | WARNING | DD >= 12% | Expect reduced exposure |
| `DD_CRITICAL` | CRITICAL | DD >= 20% | Position should be flat |
| `DD_STATE_CHANGE` | WARNING/INFO | State transition | Review context |
| `RECOVERY_ENTERED` | WARNING | Entered recovery mode | Limited exposure |
| `RECOVERY_EXITED` | INFO | Back to normal | Full exposure available |
| `ORDER_FAILURE` | CRITICAL | Trade failed | Investigate immediately |
| `POSITION_MISMATCH` | CRITICAL | State != exchange | Reconcile before trading |
| `PARTIAL_FILL` | WARNING | Order partially filled | May need follow-up |
| `MARKET_DATA_UNAVAILABLE` | WARNING | No price data | Holding until restored |

---

## Systemd Service

### Service File
```ini
# /etc/systemd/system/argusnexus-burnin.service
[Unit]
Description=ArgusNexus Portfolio Burn-In
After=network.target

[Service]
Type=simple
User=tony
WorkingDirectory=/home/tony/ArgusNexus-V4-Core
ExecStart=/home/tony/ArgusNexus-V4-Core/venv/bin/python3 scripts/paper_trade_burnin.py
Restart=always
RestartSec=60
StandardOutput=journal
StandardError=journal

# Environment for alerts
Environment="DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/..."

[Install]
WantedBy=multi-user.target
```

### Commands
```bash
# Install service
sudo cp scripts/argusnexus-burnin.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable argusnexus-burnin
sudo systemctl start argusnexus-burnin

# Status
sudo systemctl status argusnexus-burnin

# Logs
journalctl -u argusnexus-burnin -f

# Restart
sudo systemctl restart argusnexus-burnin

# Stop
sudo systemctl stop argusnexus-burnin
```

---

## Emergency Contacts

| Role | Contact | When to Alert |
|------|---------|---------------|
| Primary Operator | [Your contact] | Any CRITICAL alert |
| Exchange Support | [Exchange contact] | Order failures, API issues |

---

## Recovery Procedures

### After Prolonged Downtime (>24h)
1. Check current BTC price vs last known
2. Review what happened during downtime
3. Consider if state needs adjustment
4. Restart with conservative settings if unsure

### After System Crash
1. Check state file integrity
2. Verify last successful snapshot
3. Compare state to exchange position
4. Restart only after reconciliation

### After Major Market Event
1. Monitor DD levels closely
2. Expect possible CRITICAL state
3. Allow circuit breaker to work
4. Don't manually intervene unless malfunction

---

---

## Paper Trader Monitoring (TURTLE-4 v6.1)

### Automated Watchdog

The paper trader is monitored by `scripts/watchdog.sh` running via cron every 2 minutes.

| Check | Threshold | Alert |
|-------|-----------|-------|
| Process running | `pgrep -f "live_(unified|paper)_trader"` | DOWN if not found |
| Log freshness | Modified within 120 seconds | STALE if older |

### Discord Alerts

| Alert | Color | Meaning |
|-------|-------|---------|
| DOWN | Red | Paper trader process not running |
| STALE | Yellow | Process running but log not updating |
| RECOVERED | Green | Trader back online after outage |

### Health Check Endpoint

```bash
# Check trader status
curl http://localhost:8000/api/system/status | jq '.'

# Response:
{
  "status": "online",      # online | stale | offline
  "healthy": true,
  "message": "Paper trader running normally",
  "process": { "running": true, "pid": "1402035" },
  "log": { "fresh": true, "age_seconds": 15 }
}
```

### Dashboard Indicator

The dashboard header shows a "Trader" status indicator that polls every 30 seconds:
- **Green dot + "Trader"** - Running normally
- **Yellow dot + "Stale"** - Process running but may be stuck
- **Red dot + "Down"** - Not running

### Manual Restart

```bash
# Check if running
pgrep -af "live_unified_trader"

# If not running, restart:
cd /home/tony/ArgusNexus-V4-Core
PYTHONPATH=/home/tony/ArgusNexus-V4-Core nohup ./venv/bin/python scripts/live_unified_trader.py --capital 10000 >> runtime/paper_trader.log 2>&1 &

# Verify
tail -f runtime/paper_trader.log
```

### Cron Configuration

```bash
# View current cron
crontab -l | grep watchdog

# Should show:
# */2 * * * * /home/tony/ArgusNexus-V4-Core/scripts/watchdog.sh >> /home/tony/ArgusNexus-V4-Core/runtime/watchdog.log 2>&1
```

### Watchdog Logs

```bash
# View watchdog history
tail -50 runtime/watchdog.log

# Check for recent alerts
grep -E "(ALERT|WARNING|RECOVERED)" runtime/watchdog.log | tail -20
```

---

## Version History

| Date | Change |
|------|--------|
| 2024-12-18 | Initial runbook created |
| 2025-12-27 | Added Paper Trader Monitoring section (watchdog + Discord) |
