"""Shared HTTP helper for Baseball-Reference scrapers.

robots.txt specifies `Crawl-delay: 3` for `User-agent: *` -- every live request made
through `get()` sleeps 3s afterward to respect that. Shared by bref_players.py and
bref_slate.py so the rate-limiting logic lives in exactly one place.
"""

from __future__ import annotations

import time

import requests

BASE_URL = "https://www.baseball-reference.com"
CRAWL_DELAY_SECONDS = 3
HEADERS = {"User-Agent": "sba-research-tool/0.1 (personal project; contact: erikgoeke@gmail.com)"}


def get(path: str, params: dict | None = None) -> requests.Response:
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15, allow_redirects=True)
    resp.encoding = "utf-8"  # BR serves UTF-8 without always declaring it; avoids mangled accented names
    time.sleep(CRAWL_DELAY_SECONDS)
    return resp
