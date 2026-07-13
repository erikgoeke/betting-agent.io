"""Continuous Elo ratings computed from games.parquet -- zero extra scraping.

Complements the rolling-window form features: a 15-game window forgets
everything older, while Elo integrates a team's whole history with newer
results weighted by the K-factor, and carries across season boundaries
(regressed partway to the mean, since rosters change but don't fully reset).
Constants follow the well-known FiveThirtyEight MLB Elo conventions: low K
(baseball outcomes are noisy, single games say little) and ~24 points of
home-field advantage (roughly the historical 54% home win rate).

Leak-free by construction: each game's recorded ratings are the ratings
*entering* that game; the update happens after.
"""

from __future__ import annotations

import pandas as pd

ELO_START = 1500.0
ELO_K = 4.0
ELO_HOME_ADV = 24.0  # added to the home team's rating inside the expectation only
SEASON_CARRYOVER = 2 / 3  # regress 1/3 of the way back to the mean each new season


def _expected_home_win(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + ELO_HOME_ADV)) / 400.0))


def _replay(games: pd.DataFrame, record_rows: bool) -> tuple[dict[str, float], list[dict]]:
    """Chronological Elo pass. Returns final ratings and (optionally) one row
    per game holding both teams' PRE-game ratings."""
    ratings: dict[str, float] = {}
    current_season: int | None = None
    rows: list[dict] = []

    for row in games.sort_values("date").itertuples():
        if current_season is not None and row.season != current_season:
            for team in ratings:
                ratings[team] = ELO_START + SEASON_CARRYOVER * (ratings[team] - ELO_START)
        current_season = row.season

        home_elo = ratings.get(row.home_team, ELO_START)
        away_elo = ratings.get(row.away_team, ELO_START)
        if record_rows:
            rows.append(
                {
                    "season": row.season, "date": row.date,
                    "home_team": row.home_team, "away_team": row.away_team,
                    "home_elo": home_elo, "away_elo": away_elo,
                }
            )

        delta = ELO_K * (row.home_win - _expected_home_win(home_elo, away_elo))
        ratings[row.home_team] = home_elo + delta
        ratings[row.away_team] = away_elo - delta

    return ratings, rows


def compute_elo_table(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game: both teams' pre-game Elo ratings, chronological order."""
    _, rows = _replay(games, record_rows=True)
    return pd.DataFrame(rows)


def elo_ratings_asof(games: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """Current ratings from all games strictly before `as_of` (live-picks path)."""
    ratings, _ = _replay(games[games["date"] < as_of], record_rows=False)
    return ratings
