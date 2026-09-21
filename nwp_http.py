"""Small JSON GET helper used by the NWP fetchers.

The framework Python on this machine fails SSL verification with urllib's
default context. requests+certifi (or a curl fallback) is reliable.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any
from urllib.parse import urlencode

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "BjorkoForecast/1.0 (Chalmers Bjorko research turbine)",
}


def get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 30) -> Any:
    """GET JSON from url. Raises urllib.error.HTTPError-like exceptions via requests."""
    try:
        import requests

        response = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as requests_err:
        # Last resort: curl uses the system certificate store.
        full = url if not params else f"{url}?{urlencode(params, doseq=True)}"
        try:
            completed = subprocess.run(
                ["curl", "-sS", "-f", "-H", "Accept: application/json", "-A", DEFAULT_HEADERS["User-Agent"], full],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
            return json.loads(completed.stdout)
        except Exception as curl_err:
            raise RuntimeError(f"GET {url} failed via requests ({requests_err}) and curl ({curl_err})") from requests_err
