"""Access control for exposing TrafficIntel beyond localhost.

This app was written as a single-user local prototype: no auth, permissive CORS,
2 GB uploads, and a queue that hands every accepted job to the GPU. All of that is
reasonable on 127.0.0.1 and actively dangerous on a public URL, because every
request spends someone else's GPU time and disk.

The rule enforced here: **if the server binds to anything other than loopback, it
must have a shared access token, or it refuses to start.** Failing closed is
deliberate. The alternative is a tunnel quietly publishing an open endpoint that
runs arbitrary uploaded video on the owner's machine, and the owner finding out
from their electricity bill or a full disk.

Auth is a single shared token, not a user system. That matches the actual use
case - handing a URL and a password to a few people - and anything more would be
security theatre without a real user store behind it.

Two ways to present the token:

* ``X-Access-Token`` header - for scripts and ``tools/test_api.py``.
* a cookie set by ``POST /login`` - for browsers.

The cookie matters more than it looks. The annotated result is played by a plain
``<video src="/api/result/{id}/video">`` tag, and the browser will not attach a
custom header to that request. Header-only auth would therefore gate the API but
leave video playback either broken or open, depending on which way it failed.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
import threading
import time
from typing import Optional

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

log = logging.getLogger("trafficintel.security")

ENV_VAR = "TRAFFICINTEL_ACCESS_TOKEN"
COOKIE_NAME = "trafficintel_session"
HEADER_NAME = "x-access-token"

# Reachable without a token. Deliberately tiny.
#   /login  - obviously, or nobody could ever authenticate
#   /api/ping - so a tunnel or uptime check can confirm the process is alive
#     without revealing the GPU, model paths or job list that /api/health does.
PUBLIC_PATHS = frozenset({"/login", "/logout", "/api/ping"})

# Brute-force throttle. A shared password is short enough to guess if an attacker
# is allowed unlimited attempts, and this app has no other rate limiting.
MAX_FAILURES = 8
LOCKOUT_SECONDS = 300.0

_failures: dict[str, list] = {}
_failures_lock = threading.Lock()


def load_token() -> Optional[str]:
    """Read the shared token from the environment, or None if unset."""
    token = (os.environ.get(ENV_VAR) or "").strip()
    return token or None


def is_loopback(host: str) -> bool:
    """Whether binding to `host` keeps the server off the network.

    An empty host or "0.0.0.0"/"::" means "every interface", which is the case
    that needs a token most, so anything unparseable is treated as public.
    """
    host = (host or "").strip()
    if not host:
        return False
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A hostname we cannot resolve to a loopback address. Assume public.
        return False


def enforce_bind_policy(host: str, token: Optional[str]) -> None:
    """Refuse to start an unauthenticated server on a public interface.

    Raises SystemExit rather than warning, because a warning scrolls past in a
    terminal and the process keeps serving. There is no safe way to continue.
    """
    if token or is_loopback(host):
        return
    raise SystemExit(
        f"\nRefusing to start: --host {host} is reachable from the network and no\n"
        f"access token is set, so anyone who finds the URL could upload video and\n"
        f"run jobs on your GPU.\n\n"
        f"Set a token first:\n\n"
        f'    PowerShell:  $env:{ENV_VAR} = "some-long-random-string"\n'
        f'    bash:        export {ENV_VAR}="some-long-random-string"\n\n'
        f"Or bind to localhost only (--host 127.0.0.1), which needs no token.\n"
        f"See docs/SHARING.md.\n"
    )


def token_is_valid(supplied: Optional[str], token: str) -> bool:
    """Constant-time comparison, so response timing does not leak the token."""
    if not supplied:
        return False
    return secrets.compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _locked_out(key: str) -> float:
    """Seconds remaining in the lockout, or 0.0 if the client may try again."""
    now = time.time()
    with _failures_lock:
        stamps = [t for t in _failures.get(key, []) if now - t < LOCKOUT_SECONDS]
        _failures[key] = stamps
        if len(stamps) >= MAX_FAILURES:
            return LOCKOUT_SECONDS - (now - stamps[0])
    return 0.0


def _record_failure(key: str) -> None:
    with _failures_lock:
        _failures.setdefault(key, []).append(time.time())


def _clear_failures(key: str) -> None:
    with _failures_lock:
        _failures.pop(key, None)


def install(app, token: Optional[str]) -> None:
    """Attach the auth gate and the /login and /logout routes.

    With no token the gate is not installed at all, so local development on
    127.0.0.1 behaves exactly as before. enforce_bind_policy() is what guarantees
    that this only happens on loopback.
    """

    @app.get("/api/ping")
    def ping():
        """Liveness only - deliberately reveals nothing about the machine."""
        return {"ok": True}

    if not token:
        log.info("No %s set - running without authentication (localhost only)", ENV_VAR)
        return

    log.info("Access token enabled; browsers must sign in at /login")

    @app.middleware("http")
    async def auth_gate(request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        supplied = request.headers.get(HEADER_NAME) or request.cookies.get(COOKIE_NAME)
        if token_is_valid(supplied, token):
            return await call_next(request)

        # A browser navigating to a page should be offered the login form; an API
        # client should get a clean 401 it can act on rather than a page of HTML.
        wants_html = "text/html" in (request.headers.get("accept") or "")
        if wants_html and request.method == "GET":
            return RedirectResponse("/login", status_code=303)
        return JSONResponse(
            status_code=401,
            content={"detail": f"Authentication required. Send the {HEADER_NAME} "
                               f"header, or sign in at /login."},
        )

    @app.get("/login")
    def login_form(request: Request):
        if token_is_valid(request.cookies.get(COOKIE_NAME), token):
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(_login_page())

    @app.post("/login")
    async def login_submit(request: Request):
        key = _client_key(request)
        remaining = _locked_out(key)
        if remaining > 0:
            log.warning("Login locked out for %s (%.0fs remaining)", key, remaining)
            return HTMLResponse(
                _login_page(f"Too many attempts. Try again in {remaining / 60:.0f} min."),
                status_code=429,
            )

        form = await request.form()
        supplied = str(form.get("token") or "")
        if not token_is_valid(supplied, token):
            _record_failure(key)
            log.warning("Failed login from %s", key)
            return HTMLResponse(_login_page("That access token is not correct."),
                                status_code=401)

        _clear_failures(key)
        log.info("Successful login from %s", key)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            COOKIE_NAME, token,
            httponly=True,          # not readable from JS, so XSS cannot steal it
            samesite="lax",
            # Only mark Secure when the request actually arrived over HTTPS. A
            # tunnel terminates TLS and forwards http, so hardcoding Secure=True
            # would set a cookie the browser then refuses to send back.
            secure=request.url.scheme == "https"
                   or request.headers.get("x-forwarded-proto") == "https",
            max_age=7 * 24 * 3600,
            path="/",
        )
        return response

    @app.get("/logout")
    def logout():
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME, path="/")
        return response


def _login_page(error: str = "") -> str:
    """The sign-in page, styled to match frontend/style.css.

    Inlined rather than kept in frontend/ because the frontend directory is
    mounted as StaticFiles at "/", and a file there would be served without
    passing through the auth gate.
    """
    banner = (
        f'<p class="err">{error}</p>' if error else
        '<p class="hint">Ask whoever runs this server for the access token.</p>'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in - TrafficIntel</title>
<style>
  :root {{
    --bg:#0a1117; --surface:#101a22; --line:#22323f; --text:#e8eef2;
    --text-dim:#93a4b1; --muted:#6b7f8c; --accent:#37cd9b; --danger:#eb4d5c;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; min-height:100vh; display:flex; align-items:center;
    justify-content:center; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }}
  .card {{
    width:100%; max-width:380px; padding:32px;
    background:var(--surface); border:1px solid var(--line); border-radius:10px;
  }}
  h1 {{ margin:0 0 4px; font-size:20px; letter-spacing:-0.01em; }}
  h1 .tick {{ color:var(--accent); }}
  .sub {{ margin:0 0 24px; color:var(--muted); font-size:13px; }}
  label {{ display:block; margin-bottom:8px; font-size:13px; color:var(--text-dim); }}
  input {{
    width:100%; padding:10px 12px; border-radius:8px; font-size:14px;
    background:#0a1117; border:1px solid var(--line); color:var(--text);
  }}
  input:focus {{ outline:none; border-color:var(--accent); }}
  button {{
    width:100%; margin-top:16px; padding:11px; border:0; border-radius:8px;
    background:var(--accent); color:#06231a; font-size:14px; font-weight:600;
    cursor:pointer;
  }}
  button:hover {{ filter:brightness(1.08); }}
  .err {{
    margin:16px 0 0; padding:10px 12px; border-radius:8px; font-size:13px;
    background:#2a1216; border:1px solid #5c2028; color:#f3b7bd;
  }}
  .hint {{ margin:16px 0 0; color:var(--muted); font-size:12px; line-height:1.5; }}
</style>
</head>
<body>
  <form class="card" method="post" action="/login">
    <h1>TrafficIntel<span class="tick">.</span></h1>
    <p class="sub">This server is password protected.</p>
    <label for="token">Access token</label>
    <input id="token" name="token" type="password" autocomplete="current-password"
           autofocus required>
    <button type="submit">Sign in</button>
    {banner}
  </form>
</body>
</html>"""
