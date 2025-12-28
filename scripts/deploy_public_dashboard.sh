#!/bin/bash
# ============================================================
# Protocol: PUBLIC ACCESS - ArgusNexus Dashboard Deployment
# ============================================================
# Run as: sudo bash scripts/deploy_public_dashboard.sh
#
# This script:
# 1. Installs Caddy (reverse proxy with auto-SSL)
# 2. Configures HTTPS for your-domain.com
# 3. Creates systemd service for persistent dashboard
# ============================================================

set -e  # Exit on any error

echo ""
echo "============================================================"
echo "  🐢 ArgusNexus V4 | Protocol: PUBLIC ACCESS"
echo "============================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)"
    exit 1
fi

# Configuration
DOMAIN="your-domain.com"
USER="tony"
WORKDIR="/home/tony/ArgusNexus-V4-Core"
VENV="${WORKDIR}/venv"
PORT=8501

echo "=== PHASE 1: Installing Caddy Server ==="
apt install -y debian-keyring debian-archive-keyring apt-transport-https curl

# Add Caddy GPG key and repo
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg 2>/dev/null || true
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list

apt update
apt install -y caddy

echo "✓ Caddy installed"

echo ""
echo "=== PHASE 2: Configuring Reverse Proxy ==="

# Backup existing Caddyfile
if [ -f /etc/caddy/Caddyfile ]; then
    cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.backup.$(date +%Y%m%d_%H%M%S)
fi

# Write new Caddyfile
cat > /etc/caddy/Caddyfile << EOF
# ArgusNexus V4 Dashboard - Auto-SSL Configuration
${DOMAIN} {
    reverse_proxy localhost:${PORT}

    # Security headers
    header {
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
    }
}
EOF

echo "✓ Caddyfile configured for ${DOMAIN}"

echo ""
echo "=== PHASE 3: Creating Systemd Service ==="

cat > /etc/systemd/system/argus-dashboard.service << EOF
[Unit]
Description=ArgusNexus V4 Dashboard - TURTLE-4 Command Center
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${WORKDIR}
ExecStart=${VENV}/bin/streamlit run src/dashboard.py --server.port ${PORT} --server.headless true
Restart=always
RestartSec=10
Environment="PATH=${VENV}/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
EOF

echo "✓ Systemd service created"

echo ""
echo "=== PHASE 4: Enabling & Starting Services ==="

# Kill any existing streamlit processes
pkill -f "streamlit run src/dashboard.py" 2>/dev/null || true

# Reload and start services
systemctl daemon-reload
systemctl enable argus-dashboard
systemctl restart argus-dashboard

# Give dashboard time to start
sleep 3

# Reload Caddy (will auto-provision SSL cert)
systemctl reload caddy

echo "✓ Services started"

echo ""
echo "============================================================"
echo "  🎉 DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "  Dashboard URL: https://${DOMAIN}"
echo "  Local URL:     http://localhost:${PORT}"
echo ""
echo "  Service Status:"
systemctl status argus-dashboard --no-pager -l | head -15
echo ""
echo "============================================================"
echo "  DNS Note: If this is a new domain, wait 5-10 minutes"
echo "  for DNS propagation before testing HTTPS."
echo "============================================================"
