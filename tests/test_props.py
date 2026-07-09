import pandas as pd

from sba import props


def _batting_log(n: int) -> pd.DataFrame:
    dates = pd.date_range("2024-04-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "team": ["NYY"] * n,
            "is_home": [True, False] * (n // 2) + [True] * (n % 2),
            "opponent": ["BOS"] * n,
            "PA": [4] * n,
            "AB": [4] * n,
            "H": [1] * n,  # a hit every game
            "HR": [0] * n,  # never a home run
            "TB": [2] * n,
            "BB": [0] * n,
            "SO": [1] * n,
        }
    )


def _pitching_log(n: int, strikeouts: list[int]) -> pd.DataFrame:
    dates = pd.date_range("2024-04-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "team": ["NYY"] * n,
            "is_home": [True] * n,
            "opponent": ["BOS"] * n,
            "IP": [6.0] * n,
            "BF": [24] * n,
            "SO": strikeouts,
            "ER": [2] * n,
            "H": [5] * n,
            "BB": [2] * n,
        }
    )


def test_project_batter_uses_cached_recent_games(monkeypatch):
    log = _batting_log(20)
    monkeypatch.setattr(props, "resolve_player_id", lambda name: "fakeid01")
    monkeypatch.setattr(props, "fetch_batting_game_log", lambda player_id, season: log)

    proj = props.project_batter("fakeid01", season=2024, games=20)

    assert proj.n_games == 20
    # Every game had exactly one hit and zero home runs -> should be near-certain/near-zero.
    assert proj.hit_prob > 0.99
    assert proj.hr_prob < 0.01
    assert proj.projected_total_bases == 2.0


def test_project_batter_falls_back_to_prior_season_when_too_few_games(monkeypatch):
    thin_current_season = _batting_log(2)
    full_prior_season = _batting_log(20)

    def fake_fetch(player_id, season):
        return thin_current_season if season == 2024 else full_prior_season

    monkeypatch.setattr(props, "resolve_player_id", lambda name: "fakeid01")
    monkeypatch.setattr(props, "fetch_batting_game_log", fake_fetch)

    proj = props.project_batter("fakeid01", season=2024, games=20)
    # 2 games this season + 18 pulled from the prior season fallback, capped at `games`.
    assert proj.n_games == 20


def test_project_pitcher_weights_recent_starts_more_heavily(monkeypatch):
    # Strikeouts trending up over the sampled starts -- EWM projection should sit
    # above the simple average, since it weights the more recent (higher) starts more.
    strikeouts = [3, 4, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11]
    log = _pitching_log(len(strikeouts), strikeouts)

    monkeypatch.setattr(props, "resolve_player_id", lambda name: "fakeid01")
    monkeypatch.setattr(props, "fetch_pitching_game_log", lambda player_id, season: log)

    proj = props.project_pitcher("fakeid01", season=2024, starts=12)

    simple_average = sum(strikeouts) / len(strikeouts)
    assert proj.projected_strikeouts > simple_average
    assert proj.n_appearances == 12
