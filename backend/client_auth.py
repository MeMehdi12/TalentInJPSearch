"""
Client Authentication & Tenant Isolation
=========================================

Extracts the `client_id` from an incoming FastAPI Request using the
following priority chain:

  1. JWT Bearer token  (Authorization: Bearer <token>)
     → reads the "client_id" claim from the payload
     → requires JWT_SECRET env var to be set

  2. X-Client-ID header (set directly by the current Talentin frontend)

  3. API key → client mapping  (X-API-Key matched against API_KEY_CLIENT_MAP)
     → e.g. '{"my-key": "talentin", "client-key": "acme"}' in .env

  4. DEFAULT_CLIENT_ID env var  (fallback, default = "talentin")

Usage in endpoints:
    from client_auth import get_client_id

    @app.get("/api/v2/search")
    async def my_endpoint(request: Request):
        client_id = get_client_id(request)
        ...
"""

import json
import logging
from typing import Optional

from fastapi import Request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded config helpers (avoids circular imports at module load time)
# ---------------------------------------------------------------------------

def _get_jwt_secret() -> Optional[str]:
    from config import get_config
    return get_config().jwt_secret


def _get_jwt_algorithm() -> str:
    from config import get_config
    return get_config().jwt_algorithm


def _get_default_client_id() -> str:
    from config import get_config
    return get_config().default_client_id


def _get_api_key_map() -> dict:
    from config import get_config
    raw = get_config().api_key_client_map_json or "{}"
    try:
        return json.loads(raw)
    except Exception:
        logger.warning("API_KEY_CLIENT_MAP is not valid JSON — ignoring")
        return {}


# ---------------------------------------------------------------------------
# JWT decoding  (uses PyJWT if installed, optional dependency)
# ---------------------------------------------------------------------------

def _decode_jwt(token: str) -> Optional[str]:
    """
    Decode a JWT and return its "client_id" claim.
    Returns None on any failure so we can fall through to the next method.
    """
    secret = _get_jwt_secret()
    if not secret:
        return None  # JWT validation not configured

    try:
        import jwt as pyjwt  # PyJWT library
        payload = pyjwt.decode(
            token,
            secret,
            algorithms=[_get_jwt_algorithm()],
        )
        return payload.get("client_id")
    except Exception as exc:
        logger.warning(f"JWT decode failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_client_id(request: Request) -> Optional[str]:
    """
    Extract the client_id from the request.
    Returns None if no client identity is present (no JWT, no header, no API-key match).
    The caller decides whether to fall back to a default or reject the request.
    """

    # 1. JWT Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
        client_id = _decode_jwt(token)
        if client_id:
            logger.debug(f"client_id '{client_id}' from JWT")
            return client_id

    # 2. API key → client mapping (must come before X-Client-ID)
    api_key = request.headers.get("X-API-Key", "").strip()
    api_key_valid = False
    if api_key:
        key_map = _get_api_key_map()
        mapped = key_map.get(api_key)
        if mapped:
            api_key_valid = True
            # If X-Client-ID is also set and the caller has a valid API key,
            # allow the override (admin/mediator use case).
            client_id_header = request.headers.get("X-Client-ID", "").strip()
            if client_id_header:
                logger.debug(f"client_id '{client_id_header}' from X-Client-ID (API-key authenticated)")
                return client_id_header
            logger.debug(f"client_id '{mapped}' from API key map")
            return mapped

    # 3. X-Client-ID header — ONLY accepted from same-origin requests
    #    (frontend sets this from sessionStorage, CORS prevents cross-origin).
    #    We accept it here for the frontend flow but note it is NOT
    #    authenticated on its own — it relies on frontend-only auth.
    client_id_header = request.headers.get("X-Client-ID", "").strip()
    if client_id_header:
        logger.debug(f"client_id '{client_id_header}' from X-Client-ID header (frontend)")
        return client_id_header

    # No identity found — return None (caller must decide to reject or use default)
    logger.debug("No client identity found in request")
    return None


def require_client_id(request: Request) -> str:
    """
    Resolves client_id from the request and raises HTTP 403 if none is found.
    All data-serving endpoints must use this — never get_client_id() directly.
    """
    from fastapi import HTTPException
    client_id = get_client_id(request)
    if not client_id:
        raise HTTPException(
            status_code=403,
            detail="Cannot determine client context. Pass X-Client-ID header or a valid Bearer token."
        )
    return client_id.strip()
