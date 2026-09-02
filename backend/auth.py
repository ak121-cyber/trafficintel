"""Registration, login and JWT sessions.

Two deliberate choices worth knowing before changing anything here.

**The JWT is delivered in an HttpOnly cookie**, not handed to JavaScript. The
results page plays the annotated video with `<video src="/api/result/{id}/video">`,
and a video element cannot attach an Authorization header. Header-only auth would
therefore have left playback either broken or unprotected. A cookie is sent by the
browser on that request automatically. It also means the React code never holds a
token, so there is nothing for a cross-site script to steal out of localStorage.
An `Authorization: Bearer` header is still accepted, for scripts and tests.

**Passwords are hashed with scrypt from the standard library.** bcrypt via passlib
is the more common choice, but it needs a compiled wheel and the passlib/bcrypt
version pairing breaks regularly on Windows. scrypt is memory-hard, ships with
CPython, and cannot fail to install.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from backend import db

log = logging.getLogger("trafficintel.auth")

ROOT = Path(__file__).resolve().parent.parent

COOKIE_NAME = "trafficintel_auth"
TOKEN_TTL = timedelta(days=7)
ALGORITHM = "HS256"

# scrypt cost. n=2**14 with r=8 is the widely used interactive-login setting:
# roughly 50 ms and 16 MB per verification, slow enough to make offline guessing
# expensive without making a login feel sluggish.
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1

MIN_PASSWORD = 8
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")


def _load_secret() -> str:
    """Return the JWT signing secret.

    Prefers JWT_SECRET from the environment. If it is not set, a random secret is
    generated once and written to .jwt_secret (gitignored) rather than being
    regenerated per start - a fresh secret on every restart would silently log
    every user out. Nothing is ever hardcoded, and the fallback is reported so it
    does not go unnoticed.
    """
    env = (os.environ.get("JWT_SECRET") or "").strip()
    if env:
        return env

    path = ROOT / ".jwt_secret"
    if path.exists():
        stored = path.read_text(encoding="utf-8").strip()
        if stored:
            return stored

    generated = secrets.token_urlsafe(48)
    try:
        path.write_text(generated, encoding="utf-8")
        log.warning("JWT_SECRET was not set. Generated one and stored it in "
                    ".jwt_secret - set JWT_SECRET in .env for a real deployment.")
    except OSError:
        log.warning("JWT_SECRET was not set and .jwt_secret could not be written. "
                    "Sessions will not survive a restart.")
    return generated


SECRET = _load_secret()


def configured() -> bool:
    """True when user accounts can gate this server.

    Used by app.py to decide whether binding to a public address is acceptable:
    the original rule required a shared access token, and per-user login now
    satisfies the same requirement.
    """
    return bool(SECRET)


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N,
                            r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Any malformed stored value is a failed login, never
    an exception that would leak which accounts have unusual records."""
    try:
        scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        candidate = hashlib.scrypt(password.encode("utf-8"),
                                   salt=bytes.fromhex(salt_hex),
                                   n=int(n), r=int(r), p=int(p), dklen=32)
    except (ValueError, AttributeError, TypeError):
        return False
    return hmac.compare_digest(candidate.hex(), digest_hex)


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #

def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(user_id: str) -> str:
    """Sign an HS256 JWT.

    PyJWT is used when installed. The fallback is a real HS256 JWT too - same
    header, same claims, same signature construction - so a missing optional
    dependency cannot stop people logging in the day before a deadline.
    """
    now = datetime.now(timezone.utc)
    claims = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + TOKEN_TTL).timestamp()),
    }
    try:
        import jwt

        token = jwt.encode(claims, SECRET, algorithm=ALGORITHM)
        # PyJWT 1.x returned bytes, 2.x returns str. set_cookie needs str, and a
        # bytes value would raise deep inside the response instead of here.
        return token.decode("utf-8") if isinstance(token, bytes) else token
    except ImportError:
        import json

        header = _b64(json.dumps({"alg": ALGORITHM, "typ": "JWT"},
                                 separators=(",", ":")).encode())
        body = _b64(json.dumps(claims, separators=(",", ":")).encode())
        signing_input = f"{header}.{body}".encode()
        sig = hmac.new(SECRET.encode(), signing_input, hashlib.sha256).digest()
        return f"{header}.{body}.{_b64(sig)}"


def read_token(token: str) -> Optional[str]:
    """Return the user id from a valid, unexpired token, else None."""
    if not token:
        return None
    try:
        import jwt

        try:
            claims = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        except Exception:                                             # noqa: BLE001
            return None
        return claims.get("sub")
    except ImportError:
        pass

    import json

    try:
        header_b64, body_b64, sig_b64 = token.split(".")
        expected = hmac.new(SECRET.encode(), f"{header_b64}.{body_b64}".encode(),
                            hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(sig_b64), expected):
            return None
        claims = json.loads(_unb64(body_b64))
    except Exception:                                                 # noqa: BLE001
        return None
    if int(claims.get("exp", 0)) < datetime.now(timezone.utc).timestamp():
        return None
    return claims.get("sub")


def _set_cookie(response: Response, request: Request, token: str) -> None:
    # Secure is set only on HTTPS. Hardcoding it would break the app behind a
    # plain-http tunnel or on localhost, where the cookie would be silently
    # dropped and login would appear to succeed while never taking effect.
    https = (request.url.scheme == "https"
             or request.headers.get("x-forwarded-proto") == "https")
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=https,
        max_age=int(TOKEN_TTL.total_seconds()),
        path="/",
    )


# --------------------------------------------------------------------------- #
# Dependency
# --------------------------------------------------------------------------- #

def current_user(request: Request,
                 trafficintel_auth: Optional[str] = Cookie(default=None)) -> dict:
    """FastAPI dependency: the logged-in user, or HTTP 401.

    401 rather than 403 throughout, so the frontend has one unambiguous signal
    meaning "your session is gone, show the login page".
    """
    token = trafficintel_auth
    if not token:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()

    if not token:
        raise HTTPException(status_code=401, detail="Sign in to continue.")

    user_id = read_token(token)
    if not user_id:
        raise HTTPException(status_code=401,
                            detail="Your session has expired. Please sign in again.")

    try:
        user = db.get_user(user_id)
    except db.DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if user is None:
        raise HTTPException(status_code=401, detail="This account no longer exists.")
    return user


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterIn(BaseModel):
    name: str = ""
    email: str = ""
    password: str = ""
    confirm_password: str = ""


class LoginIn(BaseModel):
    email: str = ""
    password: str = ""


def _validate_registration(payload: RegisterIn) -> tuple[str, str]:
    name = (payload.name or "").strip()
    email = (payload.email or "").strip().lower()

    if not name or not email or not payload.password:
        raise HTTPException(status_code=400,
                            detail="Name, email and password are all required.")
    if len(name) < 2:
        raise HTTPException(status_code=400,
                            detail="Please enter your name.")
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400,
                            detail="That does not look like a valid email address.")
    if len(payload.password) < MIN_PASSWORD:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {MIN_PASSWORD} characters.")
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="The passwords do not match.")
    return name, email


@router.post("/register")
def register(payload: RegisterIn, request: Request, response: Response):
    name, email = _validate_registration(payload)
    try:
        user = db.create_user(name, email, hash_password(payload.password))
    except ValueError as exc:
        # 409, not 400: the request was well formed, the email is just taken.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except db.DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    log.info("Registered %s", email)
    _set_cookie(response, request, make_token(user["_id"]))
    return {"user": db.public_user(user), **db.credit_state(user["_id"])}


@router.post("/login")
def login(payload: LoginIn, request: Request, response: Response):
    email = (payload.email or "").strip().lower()
    if not email or not payload.password:
        raise HTTPException(status_code=400,
                            detail="Enter your email and password.")
    try:
        user = db.find_by_email(email)
    except db.DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # One message for both "no such account" and "wrong password", so the
    # response cannot be used to enumerate which emails are registered.
    if user is None or not verify_password(payload.password,
                                           user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    _set_cookie(response, request, make_token(user["_id"]))
    log.info("Login %s", email)
    return {"user": db.public_user(user), **db.credit_state(user["_id"])}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: dict = Depends(current_user)):
    """Identity plus the live credit state, which is what the nav bar renders."""
    return {"user": db.public_user(user), **db.credit_state(user["_id"])}
