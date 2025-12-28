#!/bin/bash
# Deploy ArgusNexus Bloomberg Terminal Dashboard
# Run with: sudo bash scripts/deploy_bloomberg.sh

set -e

echo "=============================================="
echo "DEPLOYING ARGUSNEXUS BLOOMBERG TERMINAL"
echo "=============================================="
echo ""

# Install new API service
echo "[1/4] Installing Bloomberg Terminal API service..."
cp scripts/argusnexus-dashboard-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable argusnexus-dashboard-api
echo "      Done"

# Stop old Streamlit dashboard
echo "[2/4] Stopping old Streamlit dashboard..."
systemctl stop argus-dashboard 2>/dev/null || true
systemctl disable argus-dashboard 2>/dev/null || true
echo "      Done"

# Start new API service
echo "[3/4] Starting Bloomberg Terminal..."
systemctl start argusnexus-dashboard-api
echo "      Done"

# Update Caddy to proxy port 8000
echo "[4/4] Updating reverse proxy..."
# Check if Caddy is running and update config
if command -v caddy &> /dev/null; then
    # Get current Caddyfile location
    if [ -f /etc/caddy/Caddyfile ]; then
        # Backup and update
        cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak
        sed -i 's/localhost:8501/localhost:8000/g' /etc/caddy/Caddyfile
        systemctl reload caddy 2>/dev/null || caddy reload --config /etc/caddy/Caddyfile
        echo "      Caddy updated to proxy port 8000"
    else
        echo "      Caddyfile not found at /etc/caddy/Caddyfile"
    fi
else
    echo "      Caddy not installed, skipping reverse proxy update"
fi

# Verify
echo ""
echo "=== Service Status ==="
systemctl status argusnexus-dashboard-api --no-pager -l | head -15

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE"
echo "=============================================="
echo ""
echo "View dashboard at: http://localhost:8000"
echo ""
echo "Useful commands:"
echo "  - API logs: journalctl -u argusnexus-dashboard-api -f"
echo "  - Restart: sudo systemctl restart argusnexus-dashboard-api"
echo "  - Stop: sudo systemctl stop argusnexus-dashboard-api"
echo ""
