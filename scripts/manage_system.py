#!/usr/bin/env python3
import os
import subprocess
import sys

def check_root():
    """Ensure the script is run as root (sudo)."""
    if os.geteuid() != 0:
        print("❌ Error: This script must be run as root!")
        print("👉 Try running: sudo python3 manage_system.py")
        sys.exit(1)

def restart_service(service_name):
    """Restart a systemd service (requires root)."""
    print(f"🔄 Restarting {service_name}...")
    try:
        subprocess.run(["systemctl", "restart", service_name], check=True)
        print(f"✅ {service_name} restarted successfully.")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to restart {service_name}. Is it installed?")

def flush_dns():
    """Flush DNS cache (requires root)."""
    print("🧹 Flushing DNS cache...")
    try:
        # For systemd-resolved systems (Ubuntu/Debian/Fedora)
        subprocess.run(["resolvectl", "flush-caches"], check=True)
        print("✅ DNS cache flushed.")
    except FileNotFoundError:
        # Fallback for older systems
        print("⚠️ 'resolvectl' not found. Trying 'systemd-resolve'...")
        try:
            subprocess.run(["systemd-resolve", "--flush-caches"], check=True)
            print("✅ DNS cache flushed.")
        except Exception as e:
            print(f"❌ Could not flush DNS: {e}")

def main():
    check_root()
    
    print("--- 🛠️  System Manager Tool 🛠️  ---")
    print("1. Restart Nginx Web Server")
    print("2. Restart SSH Service")
    print("3. Flush DNS Cache")
    print("4. Exit")
    
    choice = input("\nSelect an option (1-4): ")

    if choice == "1":
        restart_service("nginx")
    elif choice == "2":
        restart_service("ssh")
    elif choice == "3":
        flush_dns()
    elif choice == "4":
        print("Exiting.")
        sys.exit(0)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
