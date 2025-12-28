#!/bin/bash
# Deploy Coinbase Survivor Paper Trading Service
# Run with: sudo bash scripts/deploy_survivor.sh

set -e

echo "=============================================="
echo "DEPLOYING COINBASE SURVIVOR PAPER TRADING"
echo "=============================================="
echo ""

# Install Survivor service
echo "[1/4] Installing Survivor paper trading service..."
cp scripts/argusnexus-survivor-paper.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable argusnexus-survivor-paper
echo "      ✓ Service installed"

# Start Survivor service
echo "[2/4] Starting Survivor paper trading..."
systemctl start argusnexus-survivor-paper
echo "      ✓ Paper trading started"

# Restart dashboard
echo "[3/4] Restarting dashboard..."
systemctl restart argus-dashboard
echo "      ✓ Dashboard restarted"

# Verify
echo "[4/4] Verifying services..."
echo ""
echo "=== Survivor Paper Trading ==="
systemctl status argusnexus-survivor-paper --no-pager -l | head -15
echo ""
echo "=== Dashboard ==="
systemctl status argus-dashboard --no-pager -l | head -10

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE"
echo "=============================================="
echo ""
echo "View dashboard at: http://localhost:8000"
echo ""
echo "Useful commands:"
echo "  - Survivor logs: journalctl -u argusnexus-survivor-paper -f"
echo "  - Dashboard logs: journalctl -u argus-dashboard -f"
echo "  - Stop Survivor: sudo systemctl stop argusnexus-survivor-paper"
echo ""
