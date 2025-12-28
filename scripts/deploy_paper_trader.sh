#!/bin/bash
#
# ArgusNexus V4 Paper Trader Deployment Script
#
# Usage:
#   ./scripts/deploy_paper_trader.sh install   # Install systemd service
#   ./scripts/deploy_paper_trader.sh start     # Start the service
#   ./scripts/deploy_paper_trader.sh stop      # Stop the service
#   ./scripts/deploy_paper_trader.sh status    # Check service status
#   ./scripts/deploy_paper_trader.sh logs      # View live logs
#   ./scripts/deploy_paper_trader.sh test      # Run a quick test (1 tick)
#

set -e

SERVICE_NAME="argusnexus-v4-paper"
SERVICE_FILE="/path/to/ArgusNexus/scripts/argusnexus-v4-paper.service"
INSTALL_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

print_banner() {
    echo "======================================================================"
    echo "  ArgusNexus V4 - TURTLE-4 Paper Trader"
    echo "======================================================================"
    echo ""
}

install_service() {
    print_banner
    echo "Installing systemd service..."

    # Copy service file
    sudo cp "$SERVICE_FILE" "$INSTALL_PATH"

    # Reload systemd
    sudo systemctl daemon-reload

    # Enable service
    sudo systemctl enable "$SERVICE_NAME"

    echo ""
    echo "Service installed and enabled."
    echo "Run './scripts/deploy_paper_trader.sh start' to start the trader."
}

start_service() {
    print_banner
    echo "Starting paper trader..."

    sudo systemctl start "$SERVICE_NAME"

    sleep 2
    sudo systemctl status "$SERVICE_NAME" --no-pager

    echo ""
    echo "Paper trader is running."
    echo "Run './scripts/deploy_paper_trader.sh logs' to view live logs."
}

stop_service() {
    print_banner
    echo "Stopping paper trader..."

    sudo systemctl stop "$SERVICE_NAME"

    echo "Paper trader stopped."
}

show_status() {
    print_banner

    sudo systemctl status "$SERVICE_NAME" --no-pager || true

    echo ""
    echo "Recent activity:"
    sudo journalctl -u "$SERVICE_NAME" -n 20 --no-pager || true
}

show_logs() {
    print_banner
    echo "Live logs (Ctrl+C to exit):"
    echo ""

    sudo journalctl -u "$SERVICE_NAME" -f
}

run_test() {
    print_banner
    echo "Running quick test (1 tick)..."
    echo ""

    cd /path/to/ArgusNexus
    timeout 60 ./venv/bin/python scripts/live_paper_trader.py --interval 4h || true

    echo ""
    echo "Test complete. Check the output above."
}

# Main
case "$1" in
    install)
        install_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    test)
        run_test
        ;;
    *)
        echo "Usage: $0 {install|start|stop|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  install  - Install systemd service"
        echo "  start    - Start the paper trader"
        echo "  stop     - Stop the paper trader"
        echo "  status   - Check service status"
        echo "  logs     - View live logs"
        echo "  test     - Run a quick test (1 tick)"
        exit 1
        ;;
esac
