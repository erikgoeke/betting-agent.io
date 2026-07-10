"""Home-plate umpire run-environment feature.

Reuses the same Retrosheet gamelogs as starters.py/team_game_stats.py (no new
scraping) -- best-effort, since it's a single-value context feature rather
than a home/away diff: an empirical, leak-free "how many total runs tend to
get scored in games this umpire calls behind the plate," the same one-pass
empirical-factor idea as park_factors.py, using only that umpire's strictly
prior games.

This is a coarser proxy than a real strike-zone-size metric (which would need
Statcast pitch-by-pitch data joined to umpire identity, a bigger undertaking)
but needs nothing beyond data already being fetched for other features.
"""

from __future__ import annotations

import pandas as pd

from sba.data.starters import GamelogNotAvailable, fetch_season_gamelog, retro_to_bbref_team

MIN_GAMES_FOR_FACTOR = 20
DEFAULT_RUN_FACTOR = 1.0


def build_umpire_game_log(seasons: list[int]) -> pd.DataFrame:
    """One row per game: home-plate umpire id, matchup, and total runs scored."""
    frames = []
    for s in seasons:
        try:
            frames.append(fetch_season_gamelog(s).assign(season=s))
        except GamelogNotAvailable:
            continue
    if not frames:
        return pd.DataFrame(columns=["season", "date", "home_team", "away_team", "ump_home_id", "total_runs"])

    logs = pd.concat(frames, ignore_index=True)
    table = pd.DataFrame(
        {
            "season": logs["season"],
            "date": pd.to_datetime(logs["date"], format="%Y%m%d"),
            "home_team": [retro_to_bbref_team(t, s) for t, s in zip(logs["home_team"], logs["season"])],
            "away_team": [retro_to_bbref_team(t, s) for t, s in zip(logs["visiting_team"], logs["season"])],
            "ump_home_id": logs["ump_home_id"],
            "total_runs": logs["home_score"] + logs["visiting_score"],
        }
    )
    # Same doubleheader-leg convention as starters.py/team_game_stats.py.
    table = table.drop_duplicates(subset=["season", "date", "home_team", "away_team"], keep="first")
    return table.sort_values(["ump_home_id", "date"]).reset_index(drop=True)


def add_umpire_rolling_factor(ump_log: pd.DataFrame, league_avg_runs: float) -> pd.DataFrame:
    """Leak-free rolling run factor per umpire (their own prior games only)."""
    ump_log = ump_log.copy()
    ump_log["rolling_avg_runs"] = ump_log.groupby("ump_home_id")["total_runs"].transform(
        lambda s: s.shift(1).expanding(min_periods=MIN_GAMES_FOR_FACTOR).mean()
    )
    ump_log["ump_run_factor"] = (ump_log["rolling_avg_runs"] / league_avg_runs).fillna(DEFAULT_RUN_FACTOR)
    return ump_log


def umpire_factor_table(seasons: list[int], league_avg_runs: float) -> pd.DataFrame:
    """One row per game: home_team/away_team/date + that game's leak-free umpire run factor."""
    ump_log = build_umpire_game_log(seasons)
    if ump_log.empty:
        return ump_log.assign(ump_run_factor=pd.Series(dtype=float))
    return add_umpire_rolling_factor(ump_log, league_avg_runs)
