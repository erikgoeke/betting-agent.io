import pandas as pd

from sba.elo import ELO_START, compute_elo_table, elo_ratings_asof


def _streak_games(n: int = 10) -> pd.DataFrame:
    """NYY beats BOS n times in a row."""
    dates = pd.date_range("2024-04-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "season": 2024,
            "date": dates,
            "home_team": "NYY",
            "away_team": "BOS",
            "home_runs": 5.0,
            "away_runs": 2.0,
            "home_win": 1,
        }
    )


def test_first_game_starts_at_baseline_and_winner_gains():
    table = compute_elo_table(_streak_games())

    assert table.iloc[0]["home_elo"] == ELO_START
    assert table.iloc[0]["away_elo"] == ELO_START
    # Ratings recorded per game are PRE-game: by game 2 the streak shows.
    assert table.iloc[1]["home_elo"] > ELO_START > table.iloc[1]["away_elo"]
    assert table["home_elo"].is_monotonic_increasing


def test_elo_is_leak_free_asof():
    games = _streak_games()
    # Ratings as of game 5's date must equal the table's PRE-game ratings for game 5.
    table = compute_elo_table(games)
    ratings = elo_ratings_asof(games, games.iloc[4]["date"])
    assert ratings["NYY"] == table.iloc[4]["home_elo"]
    assert ratings["BOS"] == table.iloc[4]["away_elo"]


def test_ratings_regress_toward_mean_between_seasons():
    season1 = _streak_games()
    next_season = season1.copy()
    next_season["season"] = 2025
    next_season["date"] = pd.date_range("2025-04-01", periods=len(next_season), freq="D")
    games = pd.concat([season1, next_season], ignore_index=True)

    table = compute_elo_table(games)
    end_of_2024_nyy = elo_ratings_asof(games, pd.Timestamp("2024-12-31"))["NYY"]
    start_of_2025_nyy = table[table["season"] == 2025].iloc[0]["home_elo"]

    assert ELO_START < start_of_2025_nyy < end_of_2024_nyy
