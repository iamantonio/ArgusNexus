#!/usr/bin/env bash
#
# ArgusNexus Paper Trader Watchdog
# Monitors paper trader health and sends Discord alerts when down.
#
# Usage: ./scripts/watchdog.sh
#
# Configuration:
#   Set DISCORD_WEBHOOK_URL environment variable or in .env file
#   Optionally set MAX_LOG_AGE_SECONDS (default: 120)
#
# Cron example (every 2 minutes):
#   */2 * * * * /path/to/ArgusNexus-V4-Core/scripts/watchdog.sh >> /path/to/runtime/watchdog.log 2>&1

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables from .env
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Configuration
LOG_PATH="$PROJECT_ROOT/runtime/paper_trader.log"
STATE_FILE="$PROJECT_ROOT/runtime/.watchdog_state"
MAX_LOG_AGE_SECONDS="${MAX_LOG_AGE_SECONDS:-120}"
DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"

# Timestamps
NOW=$(date +%s)
NOW_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Status tracking
PROCESS_RUNNING=false
LOG_FRESH=false
LOG_AGE=0

# ============================================================================
# FUNCTIONS
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

send_discord_alert() {
    local title="$1"
    local description="$2"
    local color="$3"  # Decimal color: 16711680=red, 65280=green, 16776960=yellow

    if [[ -z "$DISCORD_WEBHOOK_URL" ]]; then
        log "WARNING: DISCORD_WEBHOOK_URL not set, skipping alert"
        return 0
    fi

    local payload=$(cat <<EOF
{
  "embeds": [{
    "title": "$title",
    "description": "$description",
    "color": $color,
    "footer": {"text": "ArgusNexus Watchdog"},
    "timestamp": "$NOW_ISO"
  }]
}
EOF
)

    curl -s -H "Content-Type: application/json" \
        -d "$payload" \
        "$DISCORD_WEBHOOK_URL" >/dev/null 2>&1 || log "ERROR: Failed to send Discord alert"
}

check_process() {
    # Check for both live_unified_trader.py and live_paper_trader.py
    if pgrep -f "live_(unified|paper)_trader" >/dev/null 2>&1; then
        PROCESS_RUNNING=true
    fi
}

check_log_freshness() {
    if [[ -f "$LOG_PATH" ]]; then
        local mtime
        mtime=$(stat -c %Y "$LOG_PATH" 2>/dev/null || stat -f %m "$LOG_PATH" 2>/dev/null)
        LOG_AGE=$((NOW - mtime))
        if [[ $LOG_AGE -lt $MAX_LOG_AGE_SECONDS ]]; then
            LOG_FRESH=true
        fi
    fi
}

get_last_state() {
    if [[ -f "$STATE_FILE" ]]; then
        cat "$STATE_FILE"
    else
        echo "unknown"
    fi
}

set_state() {
    echo "$1" > "$STATE_FILE"
}

# ============================================================================
# MAIN
# ============================================================================

log "Watchdog check started"

# Run checks
check_process
check_log_freshness

# Determine current status
if $PROCESS_RUNNING && $LOG_FRESH; then
    CURRENT_STATE="healthy"
    STATUS_MSG="Paper trader online (log ${LOG_AGE}s old)"
elif $PROCESS_RUNNING && ! $LOG_FRESH; then
    CURRENT_STATE="stale"
    STATUS_MSG="Process running but log stale (${LOG_AGE}s old)"
else
    CURRENT_STATE="down"
    STATUS_MSG="Paper trader NOT running"
fi

# Get previous state
LAST_STATE=$(get_last_state)

log "Status: $CURRENT_STATE ($STATUS_MSG) | Previous: $LAST_STATE"

# Alert on state changes
if [[ "$CURRENT_STATE" != "$LAST_STATE" ]]; then
    case "$CURRENT_STATE" in
        down)
            log "ALERT: Paper trader went DOWN"
            send_discord_alert \
                "ALERT: Paper Trader DOWN" \
                "The paper trader process has stopped running.\n\n**Action Required:** Check logs and restart.\n\`\`\`\ncd $PROJECT_ROOT\nnohup ./venv/bin/python -m src.paper_trader >> runtime/paper_trader.log 2>&1 &\n\`\`\`" \
                16711680  # Red
            ;;
        stale)
            log "WARNING: Paper trader appears stale"
            send_discord_alert \
                "WARNING: Paper Trader Stale" \
                "Process is running but log hasn't updated in ${LOG_AGE} seconds.\n\nMay be stuck or hanging." \
                16776960  # Yellow
            ;;
        healthy)
            if [[ "$LAST_STATE" == "down" ]] || [[ "$LAST_STATE" == "stale" ]]; then
                log "RECOVERED: Paper trader is back online"
                send_discord_alert \
                    "RECOVERED: Paper Trader Online" \
                    "The paper trader has recovered and is running normally." \
                    65280  # Green
            fi
            ;;
    esac
fi

# Save current state
set_state "$CURRENT_STATE"

log "Watchdog check complete"
