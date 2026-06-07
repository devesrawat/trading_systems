"""Tests for data/kite_auth.py — mocked at the requests.Session boundary."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data.kite_auth import (
    _step_capture_token,
    _step_login,
    _step_twofa,
    fetch_request_token,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_login_response(request_id: str = "req_abc123") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"status": "success", "data": {"request_id": request_id}}
    resp.raise_for_status.return_value = None
    return resp


def _ok_twofa_response() -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"status": "success", "data": {}}
    resp.raise_for_status.return_value = None
    return resp


def _redirect_response(request_token: str = "tok_xyz") -> MagicMock:
    resp = MagicMock()
    resp.headers = {
        "Location": f"https://example.com/callback?request_token={request_token}&status=success"
    }
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# _step_login
# ---------------------------------------------------------------------------


def test_step_login_returns_request_id():
    session = MagicMock()
    session.post.return_value = _ok_login_response("req_111")
    assert _step_login(session, "ZX1234", "secret") == "req_111"


def test_step_login_raises_on_failure():
    session = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"status": "error", "message": "Invalid credentials"}
    resp.raise_for_status.return_value = None
    session.post.return_value = resp
    with pytest.raises(RuntimeError, match="Kite login failed"):
        _step_login(session, "ZX1234", "wrong")


def test_step_login_propagates_http_errors():
    session = MagicMock()
    resp = MagicMock()
    resp.raise_for_status.side_effect = Exception("HTTP 403")
    session.post.return_value = resp
    with pytest.raises(Exception, match="HTTP 403"):
        _step_login(session, "ZX1234", "secret")


# ---------------------------------------------------------------------------
# _step_twofa
# ---------------------------------------------------------------------------


def test_step_twofa_ok():
    session = MagicMock()
    session.post.return_value = _ok_twofa_response()
    with patch("pyotp.TOTP") as mock_totp:
        mock_totp.return_value.now.return_value = "123456"
        _step_twofa(session, "ZX1234", "req_111", "JBSWY3DPEHPK3PXP")
    session.post.assert_called_once()
    call_data = session.post.call_args.kwargs["data"]
    assert call_data["twofa_value"] == "123456"
    assert call_data["twofa_type"] == "totp"


def test_step_twofa_raises_on_failure():
    session = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"status": "error", "message": "Invalid TOTP"}
    resp.raise_for_status.return_value = None
    session.post.return_value = resp
    with patch("pyotp.TOTP") as mock_totp:
        mock_totp.return_value.now.return_value = "000000"
        with pytest.raises(RuntimeError, match="Kite 2FA failed"):
            _step_twofa(session, "ZX1234", "req_111", "JBSWY3DPEHPK3PXP")


# ---------------------------------------------------------------------------
# _step_capture_token
# ---------------------------------------------------------------------------


def test_step_capture_token_from_redirect_location():
    session = MagicMock()
    session.get.return_value = _redirect_response("tok_abc")
    token = _step_capture_token(session, "api_key_123", "ZX1234")
    assert token == "tok_abc"


def test_step_capture_token_follows_when_no_location():
    """When Location header is absent, fall back to following the redirect."""
    session = MagicMock()

    first_resp = MagicMock()
    first_resp.headers = {}  # no Location

    second_resp = MagicMock()
    second_resp.url = "https://example.com/callback?request_token=tok_follow&status=success"

    session.get.side_effect = [first_resp, second_resp]
    token = _step_capture_token(session, "api_key_123", "ZX1234")
    assert token == "tok_follow"


def test_step_capture_token_raises_when_missing():
    session = MagicMock()
    resp = MagicMock()
    resp.headers = {"Location": "https://example.com/callback?status=success"}
    session.get.return_value = resp
    with pytest.raises(RuntimeError, match="request_token not found"):
        _step_capture_token(session, "api_key_123", "ZX1234")


# ---------------------------------------------------------------------------
# fetch_request_token — end-to-end with all steps mocked
# ---------------------------------------------------------------------------


def test_fetch_request_token_happy_path():
    with (
        patch("data.kite_auth.requests.Session") as MockSession,
        patch("pyotp.TOTP") as MockTOTP,
    ):
        MockTOTP.return_value.now.return_value = "654321"

        mock_session = MagicMock()
        MockSession.return_value = mock_session

        mock_session.post.side_effect = [
            _ok_login_response("req_happy"),
            _ok_twofa_response(),
        ]
        mock_session.get.return_value = _redirect_response("tok_happy")

        token = fetch_request_token(
            api_key="key123",
            user_id="ZX1234",
            password="pass",
            totp_secret="JBSWY3DPEHPK3PXP",
        )

    assert token == "tok_happy"
    # Login called with correct credentials
    login_call = mock_session.post.call_args_list[0]
    assert login_call.kwargs["data"]["user_id"] == "ZX1234"
    assert login_call.kwargs["data"]["password"] == "pass"
    # 2FA called with the request_id from login step
    twofa_call = mock_session.post.call_args_list[1]
    assert twofa_call.kwargs["data"]["request_id"] == "req_happy"


def test_fetch_request_token_propagates_login_error():
    with (
        patch("data.kite_auth.requests.Session") as MockSession,
        patch("pyotp.TOTP"),
    ):
        mock_session = MagicMock()
        MockSession.return_value = mock_session
        resp = MagicMock()
        resp.json.return_value = {"status": "error", "message": "bad password"}
        resp.raise_for_status.return_value = None
        mock_session.post.return_value = resp

        with pytest.raises(RuntimeError, match="Kite login failed"):
            fetch_request_token("key", "ZX1234", "wrong", "SECRET")
