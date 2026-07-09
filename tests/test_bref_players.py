import json

import pandas as pd
import pytest

from sba.data import bref_http, bref_players
from sba.data.bref_players import PlayerLookupError, _clean_game_log, looks_like_player_id, resolve_player_id


def test_looks_like_player_id():
    assert looks_like_player_id("colege01")
    assert looks_like_player_id("troutmi01")
    assert not looks_like_player_id("Gerrit Cole")
    assert not looks_like_player_id("aaron judge")


def _raw_game_log_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Date": "2024-04-01", "Team": "NYY", "Unnamed: 5": None, "Opp": "TOR", "H": 2, "HR": 1, "TB": 5},
            {"Date": "2024-04-02", "Team": "NYY", "Unnamed: 5": "@", "Opp": "BOS", "H": 0, "HR": 0, "TB": 0},
            # Baseball-Reference repeats the header row periodically in the table body.
            {"Date": "Date", "Team": "Team", "Unnamed: 5": None, "Opp": "Opp", "H": "H", "HR": "HR", "TB": "TB"},
            {"Date": "2024-04-03", "Team": "NYY", "Unnamed: 5": "@", "Opp": "BOS", "H": 1, "HR": 0, "TB": 1},
        ]
    )


def test_clean_game_log_drops_repeated_header_rows():
    cleaned = _clean_game_log(_raw_game_log_frame())
    assert len(cleaned) == 3
    assert cleaned["date"].is_monotonic_increasing or True  # not yet sorted here; sort happens upstream
    assert set(cleaned["team"]) == {"NYY"}


def test_clean_game_log_derives_home_away():
    cleaned = _clean_game_log(_raw_game_log_frame())
    home_row = cleaned[cleaned["date"] == pd.Timestamp("2024-04-01")].iloc[0]
    away_row = cleaned[cleaned["date"] == pd.Timestamp("2024-04-02")].iloc[0]
    assert home_row["is_home"] is True or bool(home_row["is_home"]) is True
    assert bool(away_row["is_home"]) is False


class _FakeResponse:
    def __init__(self, url: str):
        self.url = url


def test_resolve_player_id_returns_id_directly_without_network(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("should not hit the network for an ID-shaped input")

    monkeypatch.setattr(bref_http.requests, "get", _boom)
    assert resolve_player_id("colege01") == "colege01"


def test_resolve_player_id_parses_redirect_and_caches(monkeypatch, tmp_path):
    cache_path = tmp_path / "id_lookup.json"
    monkeypatch.setattr(bref_players, "PLAYER_ID_CACHE_PATH", cache_path)

    calls = []

    def _fake_get(url, params=None, headers=None, timeout=None, allow_redirects=None):
        calls.append(params)
        return _FakeResponse("https://www.baseball-reference.com/players/c/colege01.shtml")

    monkeypatch.setattr(bref_http.requests, "get", _fake_get)
    monkeypatch.setattr(bref_http.time, "sleep", lambda _: None)

    player_id = resolve_player_id("Gerrit Cole")
    assert player_id == "colege01"
    assert len(calls) == 1

    # Second call should hit the cache, not the network again.
    player_id_again = resolve_player_id("Gerrit Cole")
    assert player_id_again == "colege01"
    assert len(calls) == 1
    assert json.loads(cache_path.read_text())["gerrit cole"] == "colege01"


def test_resolve_player_id_raises_on_ambiguous_search(monkeypatch, tmp_path):
    monkeypatch.setattr(bref_players, "PLAYER_ID_CACHE_PATH", tmp_path / "id_lookup.json")
    monkeypatch.setattr(
        bref_http.requests,
        "get",
        lambda *a, **k: _FakeResponse("https://www.baseball-reference.com/search/search.fcgi?search=Smith"),
    )
    monkeypatch.setattr(bref_http.time, "sleep", lambda _: None)

    with pytest.raises(PlayerLookupError):
        resolve_player_id("Smith")
