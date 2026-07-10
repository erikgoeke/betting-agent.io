"""Empirical, leak-free run-scoring park factors.

Rather than hardcoding published park-factor numbers (which drift year to
year and would be an unverifiable magic constant in this codebase), this
computes each team's home park factor directly from games.parquet: total
runs per game at their home park vs. total runs per game in their own road
games, using only seasons strictly before the one being predicted.
"""

from __future__ import annotations

import pandas as pd

MIN_GAMES_FOR_FACTOR = 10
DEFAULT_FACTOR = 1.0


def compute_park_factors(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team): that team's home park factor, using only
    strictly prior seasons. The first season(s) a team has no history default
    to a neutral 1.0 factor.
    """
    games = games.assign(total_runs=games["home_runs"] + games["away_runs"])
    all_teams = pd.concat([games["home_team"], games["away_team"]]).unique()
    seasons = sorted(games["season"].unique())

    rows = []
    for season in seasons:
        prior = games[games["season"] < season]
        for team in all_teams:
            home_games = prior[prior["home_team"] == team]
            road_games = prior[prior["away_team"] == team]
            if len(home_games) < MIN_GAMES_FOR_FACTOR or len(road_games) < MIN_GAMES_FOR_FACTOR:
                factor = DEFAULT_FACTOR
            else:
                road_rpg = road_games["total_runs"].mean()
                factor = home_games["total_runs"].mean() / road_rpg if road_rpg > 0 else DEFAULT_FACTOR
            rows.append({"season": season, "team": team, "park_factor": factor})

    return pd.DataFrame(rows)


def latest_park_factor(games: pd.DataFrame, team: str, *, as_of_season: int) -> float:
    """A team's home park factor as of `as_of_season`, for live picks (no need
    to recompute the full table for every team)."""
    prior = games[games["season"] < as_of_season]
    home_games = prior[prior["home_team"] == team]
    road_games = prior[prior["away_team"] == team]
    if len(home_games) < MIN_GAMES_FOR_FACTOR or len(road_games) < MIN_GAMES_FOR_FACTOR:
        return DEFAULT_FACTOR
    road_rpg = road_games["away_runs"].add(road_games["home_runs"]).mean()
    home_rpg = home_games["home_runs"].add(home_games["away_runs"]).mean()
    return home_rpg / road_rpg if road_rpg > 0 else DEFAULT_FACTOR
