from datetime import datetime

import pandas as pd
import pytest

from sba.data import mlb_stats

CURRENT_YEAR = datetime.now().year
PAST_YEAR = CURRENT_YEAR - 1


def _fake_season_games(season: int, n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [season] * n,
            "date": pd.date_range(f"{season}-04-01", periods=n),
            "home_team": ["AAA"] * n,
            "away_team": ["BBB"] * n,
            "home_runs": [5] * n,
            "away_runs": [2] * n,
            "home_win": [1] * n,
        }
    )


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    path = tmp_path / "games.parquet"
    monkeypatch.setattr(mlb_stats, "GAMES_CACHE_PATH", path)
    return path


def test_fetch_seasons_always_refetches_current_year(monkeypatch, cache_path):
    calls = []

    def fake_fetch_season(season):
        calls.append(season)
        return _fake_season_games(season)

    monkeypatch.setattr(mlb_stats, "fetch_season", fake_fetch_season)

    mlb_stats.fetch_seasons([PAST_YEAR, CURRENT_YEAR])
    assert sorted(calls) == sorted([PAST_YEAR, CURRENT_YEAR])

    calls.clear()
    mlb_stats.fetch_seasons([PAST_YEAR, CURRENT_YEAR])
    # Past year already cached and shouldn't be refetched; current year always is.
    assert calls == [CURRENT_YEAR]


def test_fetch_seasons_serves_past_seasons_from_cache_without_network(monkeypatch, cache_path):
    monkeypatch.setattr(mlb_stats, "fetch_season", lambda season: _fake_season_games(season))
    mlb_stats.fetch_seasons([PAST_YEAR])

    def _boom(season):
        raise AssertionError("should not refetch an already-cached past season")

    monkeypatch.setattr(mlb_stats, "fetch_season", _boom)
    result = mlb_stats.fetch_seasons([PAST_YEAR])
    assert len(result) == 2


def test_fetch_seasons_force_refresh_refetches_everything(monkeypatch, cache_path):
    calls = []

    def fake_fetch_season(season):
        calls.append(season)
        return _fake_season_games(season)

    monkeypatch.setattr(mlb_stats, "fetch_season", fake_fetch_season)
    mlb_stats.fetch_seasons([PAST_YEAR])
    calls.clear()

    mlb_stats.fetch_seasons([PAST_YEAR], force_refresh=True)
    assert calls == [PAST_YEAR]
