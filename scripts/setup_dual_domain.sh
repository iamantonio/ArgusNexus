#!/bin/bash
# Setup dual domain: your-domain.com (dashboard) + your-brain-domain.com (brain)

set -e

echo "🧠 Setting up dual domain configuration..."
echo ""

# Backup current config
echo "[1/3] Backing up current Caddyfile..."
sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak.$(date +%Y%m%d_%H%M%S)

# Write new config
echo "[2/3] Writing new Caddyfile..."
sudo tee /etc/caddy/Caddyfile > /dev/null << 'EOF'
# Main Dashboard
your-domain.com {
    reverse_proxy localhost:8000

    header {
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
    }
}

# Neural Core (Brain) - your-brain-domain.com
your-brain-domain.com {
    rewrite / /brain
    reverse_proxy localhost:8000

    header {
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
    }
}
EOF

# Reload Caddy
echo "[3/3] Reloading Caddy..."
sudo systemctl reload caddy

echo ""
echo "✅ Done!"
echo ""
echo "   your-domain.com  →  Main Dashboard"
echo "   your-brain-domain.com   →  Neural Core 🧠"
echo ""
echo "Note: Make sure your-brain-domain.com DNS points to this server!"
