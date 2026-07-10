import pandas as pd

from sba.features import CORE_FEATURE_COLUMNS, FEATURE_COLUMNS, LABEL_COLUMN, MIN_PRIOR_GAMES, build_features


def _make_games() -> pd.DataFrame:
    """NYY and BOS alternate home/away over 8 games; home team always wins."""
    dates = pd.date_range("2024-04-01", periods=8, freq="D")
    home_teams = ["NYY", "BOS", "NYY", "BOS", "NYY", "BOS", "NYY", "BOS"]
    away_teams = ["BOS", "NYY", "BOS", "NYY", "BOS", "NYY", "BOS", "NYY"]
    rows = [
        {
            "season": 2024,
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_runs": 5,
            "away_runs": 2,
            "home_win": 1,
        }
        for date, home, away in zip(dates, home_teams, away_teams)
    ]
    return pd.DataFrame(rows)


def test_build_features_drops_games_without_enough_history():
    games = _make_games()
    features = build_features(games)

    assert len(features) < len(games)
    assert set(FEATURE_COLUMNS + [LABEL_COLUMN]).issubset(features.columns)
    # Core features are always present; starter-pitcher features are allowed to be
    # NaN (e.g. this synthetic test's fake games have no real resolved starter) --
    # LightGBM/XGBoost handle that natively, see features.py's CORE_FEATURE_COLUMNS.
    assert features[CORE_FEATURE_COLUMNS].isna().sum().sum() == 0


def test_build_features_uses_only_prior_games_no_leakage():
    games = _make_games()
    features = build_features(games)

    a_home_rows = features[features["home_team"] == "NYY"].sort_values("date")
    assert not a_home_rows.empty
    first_row = a_home_rows.iloc[0]

    prior_a_games = games[
        ((games["home_team"] == "NYY") | (games["away_team"] == "NYY")) & (games["date"] < first_row["date"])
    ]
    assert prior_a_games.shape[0] >= MIN_PRIOR_GAMES

    expected_wins = sum(
        (r["home_team"] == "NYY" and r["home_win"] == 1) or (r["away_team"] == "NYY" and r["home_win"] == 0)
        for _, r in prior_a_games.iterrows()
    )
    expected_win_pct = expected_wins / len(prior_a_games)

    assert first_row["home_rolling_win_pct"] == expected_win_pct
