"""Feature engineering: turn game results into a leakage-free feature table.

All rolling stats are shifted by one game before the window is computed, so a
game's features only ever use information strictly prior to that game.
"""

from __future__ import annotations

import pandas as pd

ROLLING_WINDOW = 15
MIN_PRIOR_GAMES = 3

FEATURE_COLUMNS = ["win_pct_diff", "run_diff_diff", "rest_days_diff"]
LABEL_COLUMN = "home_win"


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per team per game, from that team's perspective."""
    home = games.rename(
        columns={"home_team": "team", "away_team": "opponent", "home_runs": "runs_for", "away_runs": "runs_against"}
    )[["season", "date", "team", "opponent", "runs_for", "runs_against"]]
    home["win"] = games["home_win"]

    away = games.rename(
        columns={"away_team": "team", "home_team": "opponent", "away_runs": "runs_for", "home_runs": "runs_against"}
    )[["season", "date", "team", "opponent", "runs_for", "runs_against"]]
    away["win"] = 1 - games["home_win"]

    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["team", "date"]).reset_index(drop=True)


def add_rolling_form(log: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    log = log.copy()
    log["run_diff"] = log["runs_for"] - log["runs_against"]
    grouped = log.groupby("team")
    log["rolling_win_pct"] = grouped["win"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PRIOR_GAMES).mean()
    )
    log["rolling_run_diff"] = grouped["run_diff"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PRIOR_GAMES).mean()
    )
    log["rest_days"] = grouped["date"].diff().dt.days
    return log


def build_features(games: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Build a features + label table, one row per game, dropping games without
    enough team history yet (early season / new team stretch)."""
    log = add_rolling_form(_team_game_log(games), window=window)

    form_cols = ["season", "date", "team", "opponent", "rolling_win_pct", "rolling_run_diff", "rest_days"]
    home_form = log[form_cols].rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "rolling_win_pct": "home_rolling_win_pct",
            "rolling_run_diff": "home_rolling_run_diff",
            "rest_days": "home_rest_days",
        }
    )
    away_form = log[form_cols].rename(
        columns={
            "team": "away_team",
            "opponent": "home_team",
            "rolling_win_pct": "away_rolling_win_pct",
            "rolling_run_diff": "away_rolling_run_diff",
            "rest_days": "away_rest_days",
        }
    )

    features = games.merge(home_form, on=["season", "date", "home_team", "away_team"], how="left")
    features = features.merge(away_form, on=["season", "date", "home_team", "away_team"], how="left")

    features["win_pct_diff"] = features["home_rolling_win_pct"] - features["away_rolling_win_pct"]
    features["run_diff_diff"] = features["home_rolling_run_diff"] - features["away_rolling_run_diff"]
    features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]

    return features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def build_live_features(home_recent: pd.DataFrame, away_recent: pd.DataFrame, home_team: str, away_team: str) -> dict:
    """Build a single feature row for an upcoming game from each team's recent game logs."""

    def _team_stats(recent: pd.DataFrame, team: str) -> tuple[float, float]:
        runs_for = recent.apply(lambda r: r["home_runs"] if r["home_team"] == team else r["away_runs"], axis=1)
        runs_against = recent.apply(lambda r: r["away_runs"] if r["home_team"] == team else r["home_runs"], axis=1)
        wins = recent.apply(
            lambda r: (r["home_win"] == 1) == (r["home_team"] == team), axis=1
        )
        return wins.mean(), (runs_for - runs_against).mean()

    home_win_pct, home_run_diff = _team_stats(home_recent, home_team)
    away_win_pct, away_run_diff = _team_stats(away_recent, away_team)

    return {
        "win_pct_diff": home_win_pct - away_win_pct,
        "run_diff_diff": home_run_diff - away_run_diff,
        "rest_days_diff": 0.0,
    }
