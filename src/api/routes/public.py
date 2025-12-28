"""
Public API Routes for Twitter/Social Sharing

These routes serve:
- Shareable decision pages with OpenGraph meta tags
- Performance scoreboards
- Decision card images for social previews
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import aiosqlite
import json

router = APIRouter()

# Paths
STATIC_DIR = Path(__file__).parent.parent.parent.parent / "static"
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "v4_live_paper.db"


@router.get("/decision/{decision_id}", response_class=HTMLResponse)
async def get_decision_page(decision_id: str, request: Request):
    """
    Serve the decision page with proper OpenGraph meta tags.

    This enables rich previews when shared on Twitter/X.
    """
    # Verify decision exists
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT symbol, result, result_reason FROM decisions WHERE decision_id = ?",
            (decision_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Decision not found")

        decision = dict(row)

    # Read the template
    template_path = STATIC_DIR / "decision.html"
    if not template_path.exists():
        raise HTTPException(status_code=500, detail="Template not found")

    with open(template_path, "r") as f:
        html = f.read()

    # Inject OpenGraph meta tags for this specific decision
    symbol = decision["symbol"]
    result = decision["result"].upper().replace("_", " ")
    reason = decision["result_reason"] or "Full transparency trading decision"

    # Truncate reason for meta description
    if len(reason) > 150:
        reason = reason[:147] + "..."

    title = f"{symbol} {result} | Argus Decision"
    base_url = str(request.base_url).rstrip("/")
    page_url = f"{base_url}/decision/{decision_id}"
    image_url = f"{base_url}/api/decision-card/{decision_id}"

    # Replace placeholder meta tags
    html = html.replace(
        '<meta property="og:title" content="Argus Decision Record" id="og-title">',
        f'<meta property="og:title" content="{title}" id="og-title">'
    )
    html = html.replace(
        '<meta property="og:description" content="Full transparency trading decision" id="og-description">',
        f'<meta property="og:description" content="{reason}" id="og-description">'
    )
    html = html.replace(
        '<meta property="og:url" content="" id="og-url">',
        f'<meta property="og:url" content="{page_url}" id="og-url">'
    )
    html = html.replace(
        '<meta property="og:image" content="/api/decision-card-image" id="og-image">',
        f'<meta property="og:image" content="{image_url}" id="og-image">'
    )
    html = html.replace(
        '<meta name="twitter:title" content="Argus Decision Record" id="tw-title">',
        f'<meta name="twitter:title" content="{title}" id="tw-title">'
    )
    html = html.replace(
        '<meta name="twitter:description" content="Every trade is traceable. See the full decision." id="tw-description">',
        f'<meta name="twitter:description" content="{reason}" id="tw-description">'
    )
    html = html.replace(
        '<meta name="twitter:image" content="/api/decision-card-image" id="tw-image">',
        f'<meta name="twitter:image" content="{image_url}" id="tw-image">'
    )
    html = html.replace(
        '<title>Decision Record | Argus</title>',
        f'<title>{title}</title>'
    )

    return HTMLResponse(content=html)


@router.get("/scoreboard", response_class=HTMLResponse)
async def get_scoreboard_page():
    """Serve the performance scoreboard page."""
    template_path = STATIC_DIR / "scoreboard.html"
    if not template_path.exists():
        raise HTTPException(status_code=500, detail="Template not found")

    with open(template_path, "r") as f:
        html = f.read()

    return HTMLResponse(content=html)


@router.get("/api/decision-card/{decision_id}")
async def get_decision_card_image(decision_id: str):
    """
    Generate a beautiful PNG social card image for a decision.

    Returns a 1200x630 PNG optimized for Twitter/X rich previews.
    Features: Grade badge, Entry/Stop/Target prices, R:R ratio, Risk gates status.
    """
    from fastapi.responses import Response
    from src.api.card_image import generate_decision_card

    # Get decision data
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM decisions WHERE decision_id = ?",
            (decision_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Decision not found")

        decision = dict(row)

    # Generate PNG image
    try:
        png_bytes = generate_decision_card(decision)
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")


@router.get("/api/stats-card")
async def get_stats_card_image():
    """
    Generate a PNG image card showing overall trading stats.

    Returns a 1200x630 PNG optimized for Twitter/X with:
    - Win Rate
    - Average R:R
    - Total Decisions
    - P&L
    """
    from fastapi.responses import Response
    from src.api.card_image import generate_stats_card

    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        # Get decision stats
        cursor = await db.execute("SELECT COUNT(*) as total FROM decisions")
        row = await cursor.fetchone()
        total_decisions = row["total"] if row else 0

        # Get trade stats
        cursor = await db.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CAST(net_pnl AS REAL)) as net_pnl
            FROM trades
            WHERE exit_timestamp IS NOT NULL
        """)
        row = await cursor.fetchone()

        if row and row["total_trades"] and row["total_trades"] > 0:
            win_rate = (row["wins"] / row["total_trades"]) * 100
            total_pnl = row["net_pnl"] or 0
        else:
            win_rate = 0
            total_pnl = 0

    stats = {
        "win_rate": win_rate,
        "avg_rr": 1.8,  # TODO: Calculate from actual trades
        "total_decisions": total_decisions,
        "total_pnl": total_pnl
    }

    try:
        png_bytes = generate_stats_card(stats)
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=300",  # Cache for 5 minutes
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")


@router.get("/api/metrics")
async def get_metrics(days: int = 7):
    """
    Get trading performance metrics.

    Returns aggregate stats for the scoreboard.
    """
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

        # Get trade stats
        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as losses,
                SUM(CAST(net_pnl AS REAL)) as net_pnl,
                SUM(CAST(realized_pnl AS REAL)) as gross_pnl,
                SUM(CAST(total_commission AS REAL)) + SUM(CAST(total_slippage AS REAL)) as fees,
                MAX(CAST(net_pnl AS REAL)) as best_trade,
                MIN(CAST(net_pnl AS REAL)) as worst_trade
            FROM trades
            WHERE exit_timestamp >= datetime('now', '-' || ? || ' days')
              AND exit_timestamp IS NOT NULL
        """

        cursor = await db.execute(query, (days,))
        row = await cursor.fetchone()

        if row:
            result = dict(row)
            # Handle None values
            for key in result:
                if result[key] is None:
                    result[key] = 0
            return result

        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0,
            "gross_pnl": 0,
            "fees": 0,
            "best_trade": 0,
            "worst_trade": 0,
        }
