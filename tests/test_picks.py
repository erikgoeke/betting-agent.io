from datetime import datetime, timezone

from sba.picks import filter_todays_games


def _game(commence_time: str) -> dict:
    return {"commence_time": commence_time, "home_team": "NYY", "away_team": "BOS"}


def test_filter_todays_games_keeps_todays_and_drops_tomorrows():
    # "Now" is July 9, 5:47 PM ET (21:47 UTC).
    now = datetime(2026, 7, 9, 21, 47, tzinfo=timezone.utc)
    games = [
        _game("2026-07-09T22:46:00Z"),  # July 9, 6:46 PM ET -- today
        _game("2026-07-10T02:10:00Z"),  # July 9, 10:10 PM ET -- still today in ET
        _game("2026-07-10T17:05:00Z"),  # July 10, 1:05 PM ET -- tomorrow
        _game("2026-07-10T23:05:00Z"),  # July 10, 7:05 PM ET -- tomorrow
    ]

    kept = filter_todays_games(games, now=now)

    assert [g["commence_time"] for g in kept] == ["2026-07-09T22:46:00Z", "2026-07-10T02:10:00Z"]


def test_filter_todays_games_uses_eastern_date_not_utc():
    # 10 PM ET on July 9 is already July 10 in UTC -- a UTC-date comparison would
    # wrongly drop the late game.
    now = datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc)  # July 9, 10 PM ET
    late_game = _game("2026-07-10T02:10:00Z")  # July 9, 10:10 PM ET

    assert filter_todays_games([late_game], now=now) == [late_game]


def test_filter_todays_games_drops_in_play_games():
    # A game that started 30 minutes ago is still "today" but its odds are live
    # in-game prices, not the pregame line -- it must be excluded.
    now = datetime(2026, 7, 9, 23, 30, tzinfo=timezone.utc)
    started = _game("2026-07-09T23:00:00Z")
    upcoming = _game("2026-07-10T02:10:00Z")

    assert filter_todays_games([started, upcoming], now=now) == [upcoming]
