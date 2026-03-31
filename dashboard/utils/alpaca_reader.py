"""Read-only Alpaca account snapshot for dashboard display.

Uses synchronous httpx (compatible with Streamlit's execution model).
Never places orders or modifies account state.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


def get_alpaca_credentials() -> tuple[str, str, str] | None:
    """Return (api_key, secret_key, base_url) from env, or None if not configured."""
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or api_key == "your_paper_api_key_here":
        return None
    if not secret_key or secret_key == "your_paper_secret_key_here":
        return None
    return api_key, secret_key, base_url


def fetch_account() -> dict[str, Any] | None:
    """Fetch current account info from Alpaca. Returns None on failure."""
    creds = get_alpaca_credentials()
    if creds is None:
        return None

    api_key, secret_key, base_url = creds
    try:
        resp = httpx.get(
            f"{base_url}/v2/account",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_positions() -> list[dict[str, Any]]:
    """Fetch open positions from Alpaca. Returns empty list on failure."""
    creds = get_alpaca_credentials()
    if creds is None:
        return []

    api_key, secret_key, base_url = creds
    try:
        resp = httpx.get(
            f"{base_url}/v2/positions",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def fetch_orders(status: str = "open") -> list[dict[str, Any]]:
    """Fetch orders from Alpaca. Returns empty list on failure."""
    creds = get_alpaca_credentials()
    if creds is None:
        return []

    api_key, secret_key, base_url = creds
    try:
        resp = httpx.get(
            f"{base_url}/v2/orders",
            params={"status": status, "limit": 50},
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []
