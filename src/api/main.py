"""
ArgusNexus V4 | Bloomberg Terminal API
Professional trading dashboard backend with REST endpoints.

Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Security:
    - Public read access: All GET endpoints are publicly accessible (view-only)
    - Owner authentication: POST/write endpoints require session login
    - Set ARGUS_OWNER_PASSWORD env var for owner login
    - CORS restricted to localhost by default (set ARGUS_CORS_ORIGINS for others)
"""

# Load environment variables from .env FIRST
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import os
import secrets
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .routes import trades, decisions, metrics, prices, stream, learning, session, public, auth, system, social, stats

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = BASE_DIR / "data" / "v4_live_paper.db"

# Configure logging
logger = logging.getLogger(__name__)

# ==============================================================================
# SECURITY CONFIGURATION
# ==============================================================================

# Owner password for write access (required for POST endpoints)
# Set ARGUS_OWNER_PASSWORD environment variable
OWNER_PASSWORD = os.getenv("ARGUS_OWNER_PASSWORD")
if not OWNER_PASSWORD:
    # Generate a random password for this session and warn
    OWNER_PASSWORD = secrets.token_urlsafe(16)
    logger.warning("=" * 70)
    logger.warning("SECURITY: No ARGUS_OWNER_PASSWORD set!")
    logger.warning(f"Generated session password: {OWNER_PASSWORD}")
    logger.warning("Set ARGUS_OWNER_PASSWORD in .env for persistent login")
    logger.warning("=" * 70)

# Session storage (in-memory, clears on restart)
# Maps session_token -> {"created": timestamp, "expires": timestamp}
active_sessions: dict[str, dict] = {}
SESSION_DURATION = 60 * 60 * 24  # 24 hours

# CORS origins - defaults to localhost only
# Set ARGUS_CORS_ORIGINS as comma-separated list for additional origins
DEFAULT_CORS_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",  # Common React dev port
]
CORS_ORIGINS = os.getenv("ARGUS_CORS_ORIGINS", "").split(",")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS if o.strip()] or DEFAULT_CORS_ORIGINS

# ==============================================================================
# APP SETUP
# ==============================================================================

app = FastAPI(
    title="ArgusNexus V4 API",
    description="Truth Engine API - Bloomberg Terminal Dashboard",
    version="4.0.0",
    docs_url="/api/docs",
    redoc_url=None
)


# ==============================================================================
# SESSION-BASED AUTHENTICATION MIDDLEWARE
# ==============================================================================

import time


def is_valid_session(token: str) -> bool:
    """Check if a session token is valid and not expired."""
    if not token or token not in active_sessions:
        return False
    session = active_sessions[token]
    if time.time() > session["expires"]:
        # Clean up expired session
        del active_sessions[token]
        return False
    return True


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    Session-based authentication middleware.

    - GET requests: Public (read-only access for visitors)
    - POST requests: Require valid session (owner must be logged in)
    - Auth endpoints (/api/auth/*): Always accessible
    """
    path = request.url.path
    method = request.method

    # Auth endpoints are always accessible
    if path.startswith("/api/auth/"):
        response = await call_next(request)
        return response

    # POST/PUT/DELETE requests to /api/* require session authentication
    if method in ("POST", "PUT", "DELETE") and path.startswith("/api/"):
        session_token = request.cookies.get("argus_session")

        if not is_valid_session(session_token):
            logger.warning(f"Unauthorized {method} to {path} from {request.client.host}")
            raise HTTPException(
                status_code=401,
                detail="Login required for this action"
            )

    response = await call_next(request)
    return response


# ==============================================================================
# CORS MIDDLEWARE - RESTRICTED TO CONFIGURED ORIGINS
# ==============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    allow_credentials=True,  # Required for session cookies
)

# Include routers
app.include_router(auth.router, prefix="/api", tags=["Auth"])  # Must be first for auth endpoints
app.include_router(metrics.router, prefix="/api", tags=["Metrics"])
app.include_router(trades.router, prefix="/api", tags=["Trades"])
app.include_router(decisions.router, prefix="/api", tags=["Decisions"])
app.include_router(prices.router, prefix="/api", tags=["Prices"])
app.include_router(stream.router, prefix="/api", tags=["Stream"])
app.include_router(learning.router, prefix="/api/learning", tags=["Learning"])
app.include_router(session.router, prefix="/api", tags=["Session"])
app.include_router(public.router, tags=["Public"])  # Public pages for Twitter sharing
app.include_router(system.router, prefix="/api", tags=["System"])  # System health monitoring
app.include_router(social.router, prefix="/api", tags=["Social"])  # Twitter/Discord auto-posting
app.include_router(stats.router, prefix="/api", tags=["Stats"])  # Public stats dashboard

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the Bloomberg Terminal dashboard."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/brain", include_in_schema=False)
async def brain():
    """Serve the Argus Neural Core visualization."""
    return FileResponse(str(STATIC_DIR / "brain.html"))


@app.get("/stats", include_in_schema=False)
async def stats_page():
    """Serve the public stats dashboard."""
    return FileResponse(str(STATIC_DIR / "stats.html"))


@app.get("/strategy", include_in_schema=False)
async def strategy_page():
    """Serve the TURTLE-4 Strategy Explainer page."""
    return FileResponse(str(STATIC_DIR / "strategy.html"))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "operational", "service": "argusnexus-v4"}
