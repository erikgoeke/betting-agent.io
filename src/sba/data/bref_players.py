"""From-scratch Baseball-Reference scraper for player-level game logs.

Only three endpoints are used, all allowed by robots.txt as of this writing:
  - /search/search.fcgi  (name -> canonical player page, via redirect)
  - /players/gl.fcgi?t=b  (batting game log)
  - /players/gl.fcgi?t=p  (pitching game log)

robots.txt specifies `Crawl-delay: 3` for `User-agent: *` -- every live request
here sleeps 3s afterward to respect that, and results are cached locally so a
given player/season is only ever fetched once.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from io import StringIO

import pandas as pd

from sba.config import PLAYER_BATTING_DIR, PLAYER_ID_CACHE_PATH, PLAYER_PITCHING_DIR
from sba.data import bref_http

ID_PATTERN = re.compile(r"^[a-z\-']+\d{2}$")
CACHE_TTL_HOURS = 20  # roughly one game-day; keeps a daily scan from serving stale logs

BATTING_TABLE_ID = "players_standard_batting"
BATTING_COLUMNS = [
    "date", "team", "is_home", "opponent", "PA", "AB", "H", "2B", "3B", "HR", "TB", "BB", "SO",
    "HBP", "OBP", "SLG", "OPS", "BAbip",
]

PITCHING_TABLE_ID = "players_standard_pitching"
PITCHING_COLUMNS = [
    "date", "team", "is_home", "opponent", "IP", "BF", "SO", "ER", "R", "H", "BB", "HR", "HBP",
    "FIP", "ERA", "Pit", "GB", "FB", "LD", "PU",
]


class PlayerLookupError(RuntimeError):
    pass


def looks_like_player_id(value: str) -> bool:
    return bool(ID_PATTERN.match(value.strip().lower()))


def _load_id_cache() -> dict:
    if PLAYER_ID_CACHE_PATH.exists():
        return json.loads(PLAYER_ID_CACHE_PATH.read_text())
    return {}


def _save_id_cache(cache: dict) -> None:
    PLAYER_ID_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def resolve_player_id(name_or_id: str) -> str:
    """Resolve a player name to their Baseball-Reference ID (e.g. 'colege01').

    If `name_or_id` already looks like a BR ID, it's returned as-is. Names are
    resolved via BR's own search redirect and cached locally so repeat lookups
    don't hit the network.
    """
    query = name_or_id.strip()
    if looks_like_player_id(query):
        return query.lower()

    cache = _load_id_cache()
    cache_key = query.lower()
    if cache_key in cache:
        return cache[cache_key]

    resp = bref_http.get("/search/search.fcgi", params={"search": query})
    match = re.search(r"/players/\w/(\w+)\.shtml", resp.url)
    if not match:
        raise PlayerLookupError(
            f"Couldn't resolve a unique Baseball-Reference player for '{name_or_id}'. "
            "Search https://www.baseball-reference.com/search/ manually and pass the "
            "player ID from the URL directly (e.g. 'colege01') instead."
        )

    player_id = match.group(1)
    cache[cache_key] = player_id
    _save_id_cache(cache)
    return player_id


def _clean_game_log(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["date"])  # drops repeated header rows and any summary rows
    df["is_home"] = df["Unnamed: 5"].isna()
    df = df.rename(columns={"Team": "team", "Opp": "opponent"})
    return df


def _is_cache_fresh(cache_path, season: int) -> bool:
    """A past, completed season's game log never changes -- cache it forever.
    Only the current season's cache expires (a live scan needs today's game),
    which is what the 20h TTL was originally designed for.

    Without this, historical bulk fetches (starter_features.py calls this for
    ~1,900 mostly-past-season pitcher logs) start silently re-scraping
    Baseball-Reference from scratch the moment the cache turns 20 hours old,
    even though nothing about a 2023 game log could possibly have changed.
    """
    if season < datetime.now().year:
        return True
    age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


def _fetch_game_log(player_id: str, season: int, *, stat_type: str, table_id: str, columns: list[str]) -> pd.DataFrame:
    cache_dir = PLAYER_BATTING_DIR if stat_type == "b" else PLAYER_PITCHING_DIR
    cache_path = cache_dir / f"{player_id}_{season}.parquet"
    if cache_path.exists() and _is_cache_fresh(cache_path, season):
        cached = pd.read_parquet(cache_path)
        # A restored CI cache can carry parquet files written under an older
        # code version (e.g. before FIP/HR/Pit were added to PITCHING_COLUMNS),
        # with a fresh-looking mtime that survives the TTL check above. Treat a
        # schema mismatch as a cache miss rather than silently handing back a
        # frame that's missing columns callers now expect.
        if set(columns).issubset(cached.columns):
            return cached

    resp = bref_http.get("/players/gl.fcgi", params={"id": player_id, "t": stat_type, "year": season})
    try:
        tables = pd.read_html(StringIO(resp.text), attrs={"id": table_id})
    except ValueError:
        # pandas raises ValueError("No tables found") directly rather than
        # returning an empty list -- e.g. a position player with no pitching
        # log that season, or a bad crosswalk id. Same "give up on this
        # player" outcome as an empty result, just a different pandas API shape.
        tables = []
    if not tables:
        raise PlayerLookupError(f"No {stat_type} game log table found for '{player_id}' in {season}.")

    cleaned = _clean_game_log(tables[0])
    non_numeric = {"date", "team", "is_home", "opponent"}
    for col in columns:
        if col not in cleaned.columns:
            cleaned[col] = pd.NA
        elif col not in non_numeric:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    result = cleaned[columns].sort_values("date").reset_index(drop=True)

    result.to_parquet(cache_path, index=False)
    return result


def fetch_batting_game_log(player_id: str, season: int) -> pd.DataFrame:
    return _fetch_game_log(player_id, season, stat_type="b", table_id=BATTING_TABLE_ID, columns=BATTING_COLUMNS)


def fetch_pitching_game_log(player_id: str, season: int) -> pd.DataFrame:
    return _fetch_game_log(player_id, season, stat_type="p", table_id=PITCHING_TABLE_ID, columns=PITCHING_COLUMNS)
