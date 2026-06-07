"""
Automated Kite Connect access-token refresh via headless web login.

Uses unofficial Kite web endpoints to perform the daily login without a browser:
  1. POST /api/login         → exchange user_id + password for request_id
  2. POST /api/twofa         → exchange request_id + TOTP for session cookies
  3. GET  /connect/login     → follow OAuth redirect, capture request_token

The caller must then exchange request_token for an access_token via the
official SDK (KiteIngestor.refresh_access_token).

These endpoints are undocumented and may change without notice.  All errors
are raised — retries and alerting are handled at the scheduler level.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pyotp
import requests
import structlog

log = structlog.get_logger(__name__)

_LOGIN_URL = "https://kite.zerodha.com/api/login"
_TWOFA_URL = "https://kite.zerodha.com/api/twofa"
_CONNECT_LOGIN = "https://kite.zerodha.com/connect/login"
_REQUEST_TIMEOUT = 30


def fetch_request_token(
    api_key: str,
    user_id: str,
    password: str,
    totp_secret: str,
) -> str:
    """Perform headless Kite login and return the request_token.

    Raises RuntimeError on any step failure.
    """
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    request_id = _step_login(session, user_id, password)
    _step_twofa(session, user_id, request_id, totp_secret)
    return _step_capture_token(session, api_key, user_id)


def _step_login(session: requests.Session, user_id: str, password: str) -> str:
    resp = session.post(
        _LOGIN_URL,
        data={"user_id": user_id, "password": password},
        timeout=_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    payload: dict = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Kite login failed: {payload.get('message', payload)}")
    request_id: str = payload["data"]["request_id"]
    log.info("kite_auth_login_ok", user_id=user_id)
    return request_id


def _step_twofa(session: requests.Session, user_id: str, request_id: str, totp_secret: str) -> None:
    totp_value = pyotp.TOTP(totp_secret).now()
    resp = session.post(
        _TWOFA_URL,
        data={
            "user_id": user_id,
            "request_id": request_id,
            "twofa_value": totp_value,
            "twofa_type": "totp",
        },
        timeout=_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    payload: dict = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Kite 2FA failed: {payload.get('message', payload)}")
    log.info("kite_auth_2fa_ok", user_id=user_id)


def _step_capture_token(session: requests.Session, api_key: str, user_id: str) -> str:
    connect_url = f"{_CONNECT_LOGIN}?api_key={api_key}&v=3"
    resp = session.get(connect_url, timeout=_REQUEST_TIMEOUT, allow_redirects=False)

    location = resp.headers.get("Location", "")
    if not location:
        # Some Kite versions return 200 with a JS redirect — follow once more
        resp = session.get(connect_url, timeout=_REQUEST_TIMEOUT, allow_redirects=True)
        location = str(resp.url)

    parsed = urlparse(location)
    params = parse_qs(parsed.query)
    token = params.get("request_token", [None])[0]
    if not token:
        raise RuntimeError(f"request_token not found in redirect location: {location!r}")

    log.info("kite_auth_request_token_captured", user_id=user_id)
    return token
