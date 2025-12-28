"""
Authentication API Routes

Session-based authentication for ArgusNexus dashboard.
- Visitors can view (GET endpoints are public)
- Owner must login for write access (POST endpoints require session)
"""

import secrets
import time
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

router = APIRouter()


class LoginRequest(BaseModel):
    password: str


@router.post("/auth/login")
async def login(request: Request, response: Response, body: LoginRequest):
    """
    Authenticate as the owner to gain write access.

    Sets a session cookie on success.
    """
    # Import from main module to access shared state
    from ..main import OWNER_PASSWORD, active_sessions, SESSION_DURATION

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(body.password, OWNER_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid password")

    # Create new session
    session_token = secrets.token_urlsafe(32)
    now = time.time()
    active_sessions[session_token] = {
        "created": now,
        "expires": now + SESSION_DURATION,
    }

    # Set secure cookie
    # SECURITY: secure=True in production (HTTPS), False for local dev
    import os
    is_production = os.getenv("PRODUCTION", "false").lower() == "true"

    response.set_cookie(
        key="argus_session",
        value=session_token,
        httponly=True,
        secure=is_production,  # True in production with HTTPS
        samesite="lax",
        max_age=SESSION_DURATION,
    )

    return {"success": True, "message": "Logged in successfully"}


@router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """
    End the current session and clear the cookie.
    """
    from ..main import active_sessions

    session_token = request.cookies.get("argus_session")
    if session_token and session_token in active_sessions:
        del active_sessions[session_token]

    response.delete_cookie(key="argus_session")
    return {"success": True, "message": "Logged out successfully"}


@router.get("/auth/status")
async def auth_status(request: Request):
    """
    Check current authentication status.

    Returns whether the user is authenticated (owner) or a visitor.
    """
    from ..main import active_sessions, is_valid_session

    session_token = request.cookies.get("argus_session")
    is_authenticated = is_valid_session(session_token)

    return {
        "authenticated": is_authenticated,
        "role": "owner" if is_authenticated else "visitor",
    }
