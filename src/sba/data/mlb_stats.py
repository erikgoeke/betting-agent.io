"""Historical MLB game results via pybaseball, cached locally as parquet."""

from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd
import pybaseball as pb

from sba.config import GAMES_CACHE_PATH

pb.cache.enable()

_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

GAME_COLUMNS = [
    "season", "date", "home_team", "away_team", "home_runs", "away_runs", "home_win",
]


def teams_for_season(season: int) -> list[str]:
    """Baseball-Reference swapped the Athletics' code from OAK to ATH starting in 2025."""
    teams = list(_TEAMS)
    if season >= 2025:
        teams[teams.index("OAK")] = "ATH"
    return teams


def _fetch_team_season(season: int, team: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df = pb.schedule_and_record(season, team)
    df = df.dropna(subset=["R", "RA"]).copy()  # drop unplayed/future/postponed games
    df["season"] = season
    df["team"] = team
    return df


def _parse_dates(df: pd.DataFrame) -> pd.Series:
    date_str = df["Date"].str.replace(r"\s*\(\d+\)$", "", regex=True).str.split(", ").str[1]
    return pd.to_datetime(date_str + " " + df["season"].astype(str), format="%b %d %Y", errors="coerce")


def fetch_season(season: int) -> pd.DataFrame:
    """Fetch every team's schedule for a season and collapse to one row per game."""
    frames = [_fetch_team_season(season, team) for team in teams_for_season(season)]
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["date"] = _parse_dates(all_rows)
    all_rows = all_rows.dropna(subset=["date"])
    all_rows["game_num"] = all_rows.groupby(["team", "date"]).cumcount()

    home = all_rows[all_rows["Home_Away"] == "Home"].rename(
        columns={"team": "home_team", "Opp": "away_team", "R": "home_runs", "RA": "away_runs"}
    )
    away = all_rows[all_rows["Home_Away"] == "@"].rename(columns={"team": "away_team", "Opp": "home_team"})

    games = home.merge(
        away[["season", "date", "game_num", "home_team", "away_team"]],
        on=["season", "date", "game_num", "home_team", "away_team"],
        how="inner",
    )
    games["home_win"] = (games["home_runs"] > games["away_runs"]).astype(int)
    return games[GAME_COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_seasons(seasons: list[int], *, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch (and cache) game results for a list of seasons.

    The current calendar year's season is always refetched, even if already cached --
    it's still in progress, so treating it as permanently satisfied would mean daily
    automation (e.g. a scheduled CI run) silently uses stale current-season team form
    forever. Only fully-completed past seasons are treated as immutable/cacheable.
    """
    current_year = datetime.now().year
    cached = pd.read_parquet(GAMES_CACHE_PATH) if GAMES_CACHE_PATH.exists() else None
    cached_seasons = set(cached["season"].unique()) if cached is not None else set()

    needs_fetch = {s for s in seasons if force_refresh or s == current_year or s not in cached_seasons}

    if needs_fetch:
        fresh = pd.concat([fetch_season(s) for s in needs_fetch], ignore_index=True)
        if cached is not None:
            fresh = pd.concat([cached[~cached["season"].isin(needs_fetch)], fresh], ignore_index=True)
        fresh = fresh.drop_duplicates(subset=["season", "date", "home_team", "away_team"]).sort_values("date")
        fresh.to_parquet(GAMES_CACHE_PATH, index=False)
    else:
        fresh = cached

    return fresh[fresh["season"].isin(seasons)].reset_index(drop=True)


def team_recent_form(games: pd.DataFrame, team: str, *, as_of: pd.Timestamp, window: int) -> pd.DataFrame:
    """Return a team's most recent `window` games strictly before `as_of` (used for live picks)."""
    mask = ((games["home_team"] == team) | (games["away_team"] == team)) & (games["date"] < as_of)
    return games[mask].sort_values("date").tail(window)
