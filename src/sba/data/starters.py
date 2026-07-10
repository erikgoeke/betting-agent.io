"""Starting-pitcher identity per historical game, via Retrosheet game logs.

games.parquet (mlb_stats.py) only has team-level runs -- it doesn't say who
started. Retrosheet's public gamelogs (one file per season, no auth required)
carry `home_starting_pitcher_id` / `visiting_starting_pitcher_id`, but those are
Retrosheet IDs, not the Baseball-Reference IDs the rest of this project uses
(bref_players.py). `pybaseball.chadwick_register()` provides the crosswalk.
"""

from __future__ import annotations

import time
from functools import lru_cache
from io import StringIO
from typing import Callable

import pandas as pd
import pybaseball as pb
import requests
from pybaseball.retrosheet import gamelog_columns

from sba.config import STARTS_CACHE_PATH
from sba.data.bref_players import PlayerLookupError, fetch_pitching_game_log

GAMELOG_URL = "https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/seasons/{season}/{filename}"
# Retrosheet's own filename casing is inconsistent across seasons (e.g. GL2023.TXT vs gl2024.txt).
GAMELOG_FILENAMES = ["GL{season}.TXT", "gl{season}.txt"]

# Retrosheet's 3-letter team codes differ from Baseball-Reference's in several cases.
RETRO_TO_BBREF_TEAM = {
    "ANA": "LAA", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CHW", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KCA": "KCR", "LAN": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYA": "NYY", "NYN": "NYM", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDN": "SDP", "SEA": "SEA", "SFN": "SFG",
    "SLN": "STL", "TBA": "TBR", "TEX": "TEX", "TOR": "TOR", "WAS": "WSN",
}

STARTS_COLUMNS = [
    "season", "date", "home_team", "away_team",
    "home_starter_id", "home_starter_name", "away_starter_id", "away_starter_name",
]


def retro_to_bbref_team(retro_code: str, season: int) -> str:
    team = RETRO_TO_BBREF_TEAM.get(retro_code, retro_code)
    return "ATH" if team == "OAK" and season >= 2025 else team


class GamelogNotAvailable(RuntimeError):
    pass


@lru_cache(maxsize=None)
def fetch_season_gamelog(season: int) -> pd.DataFrame:
    """Raw Retrosheet gamelog for one season (all 161 columns).

    Retrosheet publishes the current season's gamelog with a lag, so the most
    recent season may not have a file yet -- callers should treat that as
    "no starter data for this season" rather than a hard failure.

    Memoized (in-process only) since starts.py, team_game_stats.py, and
    umpire_features.py all fetch the same per-season file independently.
    """
    for filename in GAMELOG_FILENAMES:
        full_url = GAMELOG_URL.format(season=season, filename=filename.format(season=season))
        for attempt in range(3):
            try:
                resp = requests.get(full_url, timeout=30)
                break
            except requests.exceptions.RequestException:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        if resp.status_code == 200:
            return pd.read_csv(StringIO(resp.text), header=None, names=gamelog_columns)
    raise GamelogNotAvailable(f"No Retrosheet gamelog published yet for season {season}.")


def _crosswalk(retro_ids: pd.Series) -> pd.Series:
    """Map Retrosheet player IDs to Baseball-Reference IDs."""
    register = pb.chadwick_register()
    lookup = register.dropna(subset=["key_retro", "key_bbref"]).set_index("key_retro")["key_bbref"]
    return retro_ids.map(lookup)


def build_starts_table(seasons: list[int]) -> pd.DataFrame:
    """One row per historical game: both starters' Retrosheet IDs resolved to bbref IDs.

    Seasons without a published gamelog yet (see fetch_season_gamelog) are
    silently skipped rather than failing the whole batch.
    """
    frames = []
    for s in seasons:
        try:
            frames.append(fetch_season_gamelog(s).assign(season=s))
        except GamelogNotAvailable as e:
            print(f"skipping starter data for {s}: {e}")
    if not frames:
        return pd.DataFrame(columns=STARTS_COLUMNS)
    logs = pd.concat(frames, ignore_index=True)

    starts = pd.DataFrame(
        {
            "season": logs["season"],
            "date": pd.to_datetime(logs["date"], format="%Y%m%d"),
            "home_team": [retro_to_bbref_team(t, s) for t, s in zip(logs["home_team"], logs["season"])],
            "away_team": [retro_to_bbref_team(t, s) for t, s in zip(logs["visiting_team"], logs["season"])],
            "home_starter_retro_id": logs["home_starting_pitcher_id"],
            "home_starter_name": logs["home_starting_pitcher_name"],
            "away_starter_retro_id": logs["visiting_starting_pitcher_id"],
            "away_starter_name": logs["visiting_starting_pitcher_name"],
        }
    )
    starts["home_starter_id"] = _crosswalk(starts["home_starter_retro_id"])
    starts["away_starter_id"] = _crosswalk(starts["away_starter_retro_id"])
    # games.parquet (mlb_stats.py) keeps only one row per (season, date, home, away)
    # even for doubleheaders -- match that convention so merges by that key don't
    # fan out (Retrosheet's own gamelog has one row per doubleheader *leg*).
    starts = starts.drop_duplicates(subset=["season", "date", "home_team", "away_team"], keep="first")
    return starts[STARTS_COLUMNS].reset_index(drop=True)


def fetch_starts(seasons: list[int], *, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch (and cache) starter identity for a list of seasons."""
    cached = pd.read_parquet(STARTS_CACHE_PATH) if STARTS_CACHE_PATH.exists() else None
    cached_seasons = set(cached["season"].unique()) if cached is not None else set()
    needs_fetch = {s for s in seasons if force_refresh or s not in cached_seasons}

    if needs_fetch:
        new_rows = build_starts_table(sorted(needs_fetch))
        if cached is not None and not new_rows.empty:
            fresh = pd.concat([cached[~cached["season"].isin(needs_fetch)], new_rows], ignore_index=True)
        elif cached is not None:
            fresh = cached  # nothing new (e.g. an in-progress season with no gamelog yet)
        else:
            fresh = new_rows
        fresh = fresh.sort_values("date")
        fresh.to_parquet(STARTS_CACHE_PATH, index=False)
    else:
        fresh = cached

    return fresh[fresh["season"].isin(seasons)].reset_index(drop=True)


def unique_starter_seasons(starts: pd.DataFrame) -> list[tuple[str, int]]:
    """Every distinct (bbref_id, season) pair that needs its own game log fetched."""
    home = starts[["home_starter_id", "season"]].rename(columns={"home_starter_id": "id"})
    away = starts[["away_starter_id", "season"]].rename(columns={"away_starter_id": "id"})
    pairs = pd.concat([home, away], ignore_index=True).dropna().drop_duplicates()
    return list(pairs.itertuples(index=False, name=None))


def backfill_starter_logs(starts: pd.DataFrame, *, on_progress: Callable[[str], None] | None = None) -> dict:
    """Scrape (and cache, via bref_players' own parquet cache) every starter's own
    game log for every season they appear in `starts`.

    Respects the existing 3s Baseball-Reference crawl delay in bref_http -- with
    ~1,900 distinct (pitcher, season) pairs across 2021-2025 this takes roughly
    an hour and a half. Safe to interrupt and rerun: already-cached player-seasons
    are skipped for free. Transient network errors (DNS blips, timeouts) are
    retried a couple of times with backoff rather than aborting the whole run.
    """
    pairs = unique_starter_seasons(starts)
    n_ok, n_errors = 0, 0
    for i, (player_id, season) in enumerate(pairs, start=1):
        for attempt in range(3):
            try:
                fetch_pitching_game_log(player_id, int(season))
                n_ok += 1
                break
            except PlayerLookupError:
                n_errors += 1
                break
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    n_errors += 1
                    if on_progress is not None:
                        on_progress(f"  giving up on {player_id} {season} after 3 attempts: {e}")
                else:
                    time.sleep(5 * (attempt + 1))
        if on_progress is not None:
            on_progress(f"[{i}/{len(pairs)}] {player_id} {season} (ok={n_ok}, errors={n_errors})")
    return {"n_total": len(pairs), "n_ok": n_ok, "n_errors": n_errors}
