"""Team-level hard-hit% per game, from Statcast (Baseball Savant) batted-ball data.

Statcast's own team codes differ from Baseball-Reference's in a few cases (KC
not KCR, SD not SDP, WSH not WSN) and -- unlike Retrosheet -- it applies the
*current* franchise code retroactively to historical seasons (Oakland shows as
"ATH" even in 2021-2024 data, not "OAK"). Normalize to Retrosheet-style codes
first, then reuse starters.py's season-aware OAK/ATH split so this lines up
with games.parquet's convention.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import pybaseball as pb
import requests

from sba.config import HARD_HIT_CACHE_PATH
from sba.data.starters import retro_to_bbref_team

HARD_HIT_THRESHOLD_MPH = 95.0

STATCAST_TO_RETRO_TEAM = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHN", "CWS": "CHA",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KCA",
    "LAA": "ANA", "LAD": "LAN", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYN",
    "NYY": "NYA", "OAK": "OAK", "ATH": "OAK", "PHI": "PHI", "PIT": "PIT", "SD": "SDN",
    "SEA": "SEA", "SF": "SFN", "STL": "SLN", "TB": "TBA", "TEX": "TEX", "TOR": "TOR", "WSH": "WAS",
}


def retro_to_bbref_team_from_statcast(code: str, season: int) -> str:
    return retro_to_bbref_team(STATCAST_TO_RETRO_TEAM.get(code, code), season)


def _statcast_with_retry(start_dt: str, end_dt: str, *, attempts: int = 3) -> pd.DataFrame:
    """pybaseball's statcast() internally chunks into many day-by-day requests
    over a multi-month range -- a single transient timeout among dozens of
    those shouldn't crash the whole `sba train` run."""
    for attempt in range(attempts):
        try:
            return pb.statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
        except (requests.exceptions.RequestException, ConnectionError) as e:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")  # loop always returns or raises


def fetch_season_hard_hit(season: int) -> pd.DataFrame:
    """One row per (team, date): that team's hard-hit% among batted balls that game."""
    raw = _statcast_with_retry(f"{season}-03-01", f"{season}-11-15")
    if raw.empty:
        return pd.DataFrame(columns=["season", "date", "team", "hard_hit_rate", "n_batted_balls"])

    batted = raw[(raw["type"] == "X") & raw["launch_speed"].notna()].copy()
    batted["team"] = [
        retro_to_bbref_team_from_statcast(away if topbot == "Top" else home, season)
        for away, home, topbot in zip(batted["away_team"], batted["home_team"], batted["inning_topbot"])
    ]
    batted["hard_hit"] = batted["launch_speed"] >= HARD_HIT_THRESHOLD_MPH

    grouped = batted.groupby(["team", "game_date"]).agg(
        hard_hit_rate=("hard_hit", "mean"), n_batted_balls=("hard_hit", "size")
    ).reset_index()
    grouped["season"] = season
    grouped = grouped.rename(columns={"game_date": "date"})
    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped[["season", "date", "team", "hard_hit_rate", "n_batted_balls"]]


HARD_HIT_WINDOW = 15
MIN_PRIOR_GAMES = 3


def add_hard_hit_rolling(hard_hit: pd.DataFrame, window: int = HARD_HIT_WINDOW) -> pd.DataFrame:
    """Leak-free rolling hard-hit% -- each row's OWN hard_hit_rate is that game's
    result, so it must be shifted out before being used as a predictive feature."""
    hard_hit = hard_hit.sort_values(["team", "date"]).reset_index(drop=True)
    hard_hit["rolling_hard_hit_rate"] = hard_hit.groupby("team")["hard_hit_rate"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PRIOR_GAMES).mean()
    )
    return hard_hit


def team_hard_hit_form_asof(hard_hit: pd.DataFrame, team: str, *, as_of: pd.Timestamp, window: int = HARD_HIT_WINDOW) -> float:
    """A team's rolling hard-hit% entering an upcoming (not-yet-played) game."""
    recent = hard_hit[(hard_hit["team"] == team) & (hard_hit["date"] < as_of)].sort_values("date").tail(window)
    if len(recent) < MIN_PRIOR_GAMES:
        return float("nan")
    return recent["hard_hit_rate"].mean()


def fetch_hard_hit(seasons: list[int], *, force_refresh: bool = False) -> pd.DataFrame:
    # Unlike Retrosheet (which embargoes the current season's gamelog until it's
    # over, so it never successfully caches mid-season -- see starters.py),
    # Statcast publishes with only a day or two of lag. Without forcing a refetch,
    # the current season would cache after its first partial fetch and then never
    # pick up newer games for the rest of the year.
    current_year = datetime.now().year
    cached = pd.read_parquet(HARD_HIT_CACHE_PATH) if HARD_HIT_CACHE_PATH.exists() else None
    cached_seasons = set(cached["season"].unique()) if cached is not None else set()
    needs_fetch = {s for s in seasons if force_refresh or s == current_year or s not in cached_seasons}

    if needs_fetch:
        new_rows = pd.concat([fetch_season_hard_hit(s) for s in sorted(needs_fetch)], ignore_index=True)
        if cached is not None and not new_rows.empty:
            fresh = pd.concat([cached[~cached["season"].isin(needs_fetch)], new_rows], ignore_index=True)
        elif cached is not None:
            fresh = cached
        else:
            fresh = new_rows
        fresh = fresh.sort_values(["team", "date"])
        fresh.to_parquet(HARD_HIT_CACHE_PATH, index=False)
    else:
        fresh = cached

    return fresh[fresh["season"].isin(seasons)].reset_index(drop=True)
