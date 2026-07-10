"""Shared HTTP helper for Baseball-Reference scrapers.

robots.txt specifies `Crawl-delay: 3` for `User-agent: *` -- every live request made
through `get()` sleeps 3s afterward to respect that. Shared by bref_players.py and
bref_slate.py so the rate-limiting logic lives in exactly one place.

`sba train`/`sba report` now make ~1,900+ of these calls in one run (one per
historical starting pitcher-season, via starter_features.py) -- a transient
DNS/connection blip on any single one of those would otherwise crash the whole
run. Retrying here, in the one shared low-level function, fixes that for every
caller (bref_players.py, bref_slate.py) instead of needing it repeated at each
call site.
"""

from __future__ import annotations

import time

import requests

BASE_URL = "https://www.baseball-reference.com"
CRAWL_DELAY_SECONDS = 3
HEADERS = {"User-Agent": "sba-research-tool/0.1 (personal project; contact: erikgoeke@gmail.com)"}
MAX_ATTEMPTS = 3


def get(path: str, params: dict | None = None) -> requests.Response:
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15, allow_redirects=True)
            break
        except requests.exceptions.RequestException:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(5 * (attempt + 1))
    resp.encoding = "utf-8"  # BR serves UTF-8 without always declaring it; avoids mangled accented names
    time.sleep(CRAWL_DELAY_SECONDS)
    return resp
