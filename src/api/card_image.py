"""
Enhanced OG Image Card Generator for Social Sharing

Generates beautiful, Twitter-optimized PNG images for decision cards.
Design spec by Amelia - optimized for X/Twitter rich previews (1200x630).
"""

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional
import json


# Colors (matching Argus dark theme)
COLORS = {
    "bg_primary": "#0a0a0f",
    "bg_secondary": "#12121a",
    "bg_tertiary": "#1a1a25",
    "text_primary": "#e0e0e0",
    "text_secondary": "#888888",
    "accent": "#00d4aa",
    "success": "#00c853",
    "danger": "#ff5252",
    "warning": "#ffc107",
    "border": "#2a2a35",
}

# Card dimensions (Twitter optimized)
CARD_WIDTH = 1200
CARD_HEIGHT = 630


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a monospace font, with fallback to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/ubuntu/UbuntuMono-Regular.ttf",
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue

    # Fallback to default
    return ImageFont.load_default()


def format_price(price: Any) -> str:
    """Format price with proper decimal places."""
    if price is None:
        return "--"
    try:
        p = float(price)
        if p >= 1000:
            return f"${p:,.0f}"
        elif p >= 100:
            return f"${p:,.2f}"
        else:
            return f"${p:,.4f}"
    except:
        return "--"


def generate_decision_card(decision: Dict[str, Any]) -> bytes:
    """
    Generate a PNG image card for a trading decision.

    Args:
        decision: Decision data with signal_values, market_context, etc.

    Returns:
        PNG image as bytes
    """
    # Create image with gradient background
    img = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), hex_to_rgb(COLORS["bg_primary"]))
    draw = ImageDraw.Draw(img)

    # Add subtle gradient effect (darker at bottom)
    for y in range(CARD_HEIGHT):
        darkness = int(5 * (y / CARD_HEIGHT))
        color = tuple(max(0, c - darkness) for c in hex_to_rgb(COLORS["bg_primary"]))
        draw.line([(0, y), (CARD_WIDTH, y)], fill=color)

    # Fonts
    font_logo = get_font(42, bold=True)
    font_tagline = get_font(18)
    font_badge = get_font(28, bold=True)
    font_symbol = get_font(36, bold=True)
    font_grade = get_font(24, bold=True)
    font_label = get_font(14)
    font_value = get_font(32, bold=True)
    font_small_value = get_font(24)
    font_footer = get_font(16)
    font_cta = get_font(18, bold=True)

    # Extract data
    symbol = decision.get("symbol", "???")
    result = decision.get("result", "unknown")
    decision_id = decision.get("decision_id", "")[:8]
    timestamp = decision.get("timestamp", "")

    # Parse market context
    market_context = decision.get("market_context", {})
    if isinstance(market_context, str):
        try:
            market_context = json.loads(market_context)
        except:
            market_context = {}

    setup = market_context.get("setup_details", market_context.get("professional", {}))
    risk = setup.get("risk", {})
    layers = setup.get("layers", {})
    daily = layers.get("context_daily", {})

    # Get values
    grade = setup.get("grade", market_context.get("professional", {}).get("grade", ""))
    current_price = market_context.get("current_price") or setup.get("current_price") or daily.get("current_price")
    stop_price = risk.get("stop")
    target_price = risk.get("target")
    risk_reward = risk.get("risk_reward")

    # Determine styling based on result
    if result == "signal_long":
        result_color = COLORS["success"]
        result_text = "LONG"
        badge_icon = "▲"
    elif result == "signal_short":
        result_color = COLORS["danger"]
        result_text = "SHORT"
        badge_icon = "▼"
    elif result == "signal_close" or result == "signal_exit":
        result_color = COLORS["warning"]
        result_text = "EXIT"
        badge_icon = "✕"
    elif result == "risk_rejected":
        result_color = COLORS["warning"]
        result_text = "BLOCKED"
        badge_icon = "⊘"
    else:
        result_color = COLORS["text_secondary"]
        result_text = result.upper().replace("_", " ")[:10]
        badge_icon = "◆"

    # Grade colors
    if grade in ("A+", "A"):
        grade_color = COLORS["success"]
    elif grade == "B":
        grade_color = COLORS["warning"]
    else:
        grade_color = COLORS["text_secondary"]

    # ===== DRAW THE CARD =====

    # Border with accent color
    draw.rectangle(
        [(20, 20), (CARD_WIDTH - 20, CARD_HEIGHT - 20)],
        outline=hex_to_rgb(COLORS["accent"]),
        width=2
    )

    # Logo area (top left)
    draw.text((50, 40), "ARGUS", font=font_logo, fill=hex_to_rgb(COLORS["accent"]))
    draw.text((50, 90), "The Only Trading Bot That Shows Its Work", font=font_tagline, fill=hex_to_rgb(COLORS["text_secondary"]))

    # Decision indicator (top right corner)
    draw.text((CARD_WIDTH - 200, 50), "DECISION", font=font_label, fill=hex_to_rgb(COLORS["text_secondary"]))

    # Horizontal separator
    draw.line([(50, 130), (CARD_WIDTH - 50, 130)], fill=hex_to_rgb(COLORS["border"]), width=2)

    # Main content area
    y_start = 160

    # Result badge (pill shape)
    badge_width = 160
    badge_height = 50
    badge_x = 50
    badge_y = y_start

    # Draw pill background
    draw.rounded_rectangle(
        [(badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height)],
        radius=25,
        fill=hex_to_rgb(result_color + "33"),  # Transparent version
        outline=hex_to_rgb(result_color),
        width=2
    )
    draw.text(
        (badge_x + badge_width // 2, badge_y + badge_height // 2),
        f"{badge_icon} {result_text}",
        font=font_badge,
        fill=hex_to_rgb(result_color),
        anchor="mm"
    )

    # Symbol
    draw.text((badge_x + badge_width + 30, badge_y + 8), symbol, font=font_symbol, fill=hex_to_rgb(COLORS["text_primary"]))

    # Grade badge
    if grade:
        grade_x = badge_x + badge_width + 250
        draw.text((grade_x, badge_y + 12), f"[Grade {grade}]", font=font_grade, fill=hex_to_rgb(grade_color))

    # ===== PRICE GRID =====
    grid_y = y_start + 80
    col_width = 280

    # Entry Price
    draw.text((50, grid_y), "ENTRY PRICE", font=font_label, fill=hex_to_rgb(COLORS["accent"]))
    draw.text((50, grid_y + 25), format_price(current_price), font=font_value, fill=hex_to_rgb(COLORS["text_primary"]))

    # Stop Loss
    draw.text((50 + col_width, grid_y), "STOP LOSS", font=font_label, fill=hex_to_rgb(COLORS["danger"]))
    draw.text((50 + col_width, grid_y + 25), format_price(stop_price), font=font_value, fill=hex_to_rgb(COLORS["danger"]))

    # Target
    draw.text((50 + col_width * 2, grid_y), "TARGET", font=font_label, fill=hex_to_rgb(COLORS["success"]))
    draw.text((50 + col_width * 2, grid_y + 25), format_price(target_price), font=font_value, fill=hex_to_rgb(COLORS["success"]))

    # Risk:Reward
    draw.text((50 + col_width * 3, grid_y), "RISK:REWARD", font=font_label, fill=hex_to_rgb(COLORS["accent"]))
    rr_text = f"1:{risk_reward:.1f}" if risk_reward else "--"
    draw.text((50 + col_width * 3, grid_y + 25), rr_text, font=font_value, fill=hex_to_rgb(COLORS["accent"]))

    # ===== RISK GATES SECTION =====
    gates_y = grid_y + 100
    draw.line([(50, gates_y), (CARD_WIDTH - 50, gates_y)], fill=hex_to_rgb(COLORS["border"]), width=1)

    # Parse risk checks
    risk_checks = decision.get("risk_checks", {})
    if isinstance(risk_checks, str):
        try:
            risk_checks = json.loads(risk_checks)
        except:
            risk_checks = {}

    # Count passed/failed gates
    total_gates = len(risk_checks)
    passed_gates = sum(1 for v in risk_checks.values() if (v.get("passed") if isinstance(v, dict) else v))

    draw.text((50, gates_y + 15), "RISK GATES", font=font_label, fill=hex_to_rgb(COLORS["accent"]))

    if total_gates > 0:
        gate_status = f"✓ {passed_gates}/{total_gates} Passed"
        gate_color = COLORS["success"] if passed_gates == total_gates else COLORS["warning"]
        draw.text((200, gates_y + 15), gate_status, font=font_small_value, fill=hex_to_rgb(gate_color))

    # ===== REASONING SECTION =====
    reason_y = gates_y + 60
    draw.rectangle(
        [(50, reason_y), (CARD_WIDTH - 50, reason_y + 80)],
        fill=hex_to_rgb(COLORS["bg_tertiary"])
    )
    # Accent bar on left
    draw.rectangle(
        [(50, reason_y), (55, reason_y + 80)],
        fill=hex_to_rgb(COLORS["accent"])
    )

    reason_text = decision.get("result_reason", "No specific reason recorded")
    if reason_text and len(reason_text) > 100:
        reason_text = reason_text[:97] + "..."

    draw.text((70, reason_y + 10), "DECISION REASONING", font=font_label, fill=hex_to_rgb(COLORS["accent"]))
    draw.text((70, reason_y + 35), reason_text or "No reason provided", font=font_footer, fill=hex_to_rgb(COLORS["text_primary"]))

    # ===== FOOTER =====
    footer_y = CARD_HEIGHT - 70
    draw.line([(50, footer_y - 10), (CARD_WIDTH - 50, footer_y - 10)], fill=hex_to_rgb(COLORS["border"]), width=1)

    # Decision ID and timestamp
    draw.text((50, footer_y), f"ID: {decision_id}...", font=font_footer, fill=hex_to_rgb(COLORS["text_secondary"]))
    if timestamp:
        ts_display = timestamp[:19].replace("T", " ") if "T" in timestamp else timestamp[:19]
        draw.text((250, footer_y), ts_display, font=font_footer, fill=hex_to_rgb(COLORS["text_secondary"]))

    # CTA
    draw.text(
        (CARD_WIDTH - 50, footer_y),
        "Every trade is traceable. See the full record →",
        font=font_cta,
        fill=hex_to_rgb(COLORS["accent"]),
        anchor="ra"
    )

    # ===== WATERMARK =====
    draw.text(
        (CARD_WIDTH - 50, 50),
        "your-domain.com",
        font=font_tagline,
        fill=hex_to_rgb(COLORS["text_secondary"]),
        anchor="ra"
    )

    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return buffer.getvalue()


def generate_stats_card(stats: Dict[str, Any]) -> bytes:
    """
    Generate a PNG image card for overall trading stats.

    Args:
        stats: Stats data with win_rate, total_pnl, etc.

    Returns:
        PNG image as bytes
    """
    img = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), hex_to_rgb(COLORS["bg_primary"]))
    draw = ImageDraw.Draw(img)

    # Add gradient
    for y in range(CARD_HEIGHT):
        darkness = int(5 * (y / CARD_HEIGHT))
        color = tuple(max(0, c - darkness) for c in hex_to_rgb(COLORS["bg_primary"]))
        draw.line([(0, y), (CARD_WIDTH, y)], fill=color)

    # Fonts
    font_logo = get_font(48, bold=True)
    font_tagline = get_font(20)
    font_stat_value = get_font(64, bold=True)
    font_stat_label = get_font(18)
    font_footer = get_font(18, bold=True)

    # Border
    draw.rectangle(
        [(20, 20), (CARD_WIDTH - 20, CARD_HEIGHT - 20)],
        outline=hex_to_rgb(COLORS["accent"]),
        width=3
    )

    # Logo
    draw.text((CARD_WIDTH // 2, 60), "ARGUS LIVE STATS", font=font_logo, fill=hex_to_rgb(COLORS["accent"]), anchor="mm")
    draw.text((CARD_WIDTH // 2, 110), "Transparency First · Every Decision Logged", font=font_tagline, fill=hex_to_rgb(COLORS["text_secondary"]), anchor="mm")

    # Stats grid
    y_center = CARD_HEIGHT // 2 + 20
    col_width = CARD_WIDTH // 4

    # Win Rate
    win_rate = stats.get("win_rate", 0)
    wr_color = COLORS["success"] if win_rate >= 50 else COLORS["warning"] if win_rate >= 40 else COLORS["danger"]
    draw.text((col_width * 0.5, y_center - 30), f"{win_rate:.1f}%", font=font_stat_value, fill=hex_to_rgb(wr_color), anchor="mm")
    draw.text((col_width * 0.5, y_center + 30), "Win Rate", font=font_stat_label, fill=hex_to_rgb(COLORS["text_secondary"]), anchor="mm")

    # R:R
    avg_rr = stats.get("avg_rr", 0)
    draw.text((col_width * 1.5, y_center - 30), f"1:{avg_rr:.1f}", font=font_stat_value, fill=hex_to_rgb(COLORS["accent"]), anchor="mm")
    draw.text((col_width * 1.5, y_center + 30), "Avg R:R", font=font_stat_label, fill=hex_to_rgb(COLORS["text_secondary"]), anchor="mm")

    # Total Decisions
    total_decisions = stats.get("total_decisions", 0)
    draw.text((col_width * 2.5, y_center - 30), str(total_decisions), font=font_stat_value, fill=hex_to_rgb(COLORS["text_primary"]), anchor="mm")
    draw.text((col_width * 2.5, y_center + 30), "Decisions", font=font_stat_label, fill=hex_to_rgb(COLORS["text_secondary"]), anchor="mm")

    # P&L
    pnl = stats.get("total_pnl", 0)
    pnl_color = COLORS["success"] if pnl >= 0 else COLORS["danger"]
    pnl_text = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
    draw.text((col_width * 3.5, y_center - 30), pnl_text, font=font_stat_value, fill=hex_to_rgb(pnl_color), anchor="mm")
    draw.text((col_width * 3.5, y_center + 30), "P&L", font=font_stat_label, fill=hex_to_rgb(COLORS["text_secondary"]), anchor="mm")

    # Footer
    draw.text((CARD_WIDTH // 2, CARD_HEIGHT - 50), "your-domain.com · Transparency Triumphs", font=font_footer, fill=hex_to_rgb(COLORS["accent"]), anchor="mm")

    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return buffer.getvalue()
