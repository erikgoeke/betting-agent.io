import math

import pandas as pd
import pytest

from sba import tracking
from sba.picks import Pick


def _pick(commence_time="2026-07-09T23:00:00Z", price=-120, edge=0.04) -> Pick:
    return Pick(
        commence_time=commence_time,
        home_team="ATL",
        away_team="PIT",
        model_home_win_prob=0.58,
        market_home_win_prob=0.54,
        side="home",
        side_price=price,
        side_model_prob=0.58,
        side_market_prob=0.54,
        edge=edge,
        suggested_stake_pct=0.02,
        n_books=8,
    )


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "picks.csv"
    monkeypatch.setattr(tracking, "PICKS_LOG_PATH", path)
    return path


def test_log_picks_replaces_ungraded_duplicate_instead_of_appending(log_path):
    tracking.log_picks([_pick(price=-120)])
    tracking.log_picks([_pick(price=-115)])  # same game re-logged with fresh odds

    log = pd.read_csv(log_path)
    assert len(log) == 1
    assert log.iloc[0]["side_price"] == -115


def test_log_picks_never_touches_graded_rows(log_path):
    tracking.log_picks([_pick()])
    log = pd.read_csv(log_path)
    log["won"] = True
    log["result"] = "home_win"
    log.to_csv(log_path, index=False)

    tracking.log_picks([_pick(price=-110)])  # same game key, but history is graded

    log = pd.read_csv(log_path)
    assert len(log) == 2  # graded row preserved, fresh row appended


def test_read_log_normalizes_won_column_through_csv_round_trip(log_path):
    tracking.log_picks([_pick(), _pick(commence_time="2026-07-09T23:30:00Z")])
    log = pd.read_csv(log_path)
    log.loc[0, "won"] = True
    log.to_csv(log_path, index=False)  # bool + NaN round-trips as object/strings

    reread = tracking._read_log()
    assert reread.loc[0, "won"] is True or reread.loc[0, "won"] == True  # noqa: E712
    assert pd.isna(reread.loc[1, "won"])


def test_grade_picks_purges_in_play_captures_instead_of_grading_them(log_path, monkeypatch):
    import sba.tracking as tr

    # Row logged 2h AFTER first pitch -> in-play odds capture, must be purged.
    in_play = _pick(commence_time="2026-07-08T23:00:00Z")
    pregame = _pick(commence_time="2026-07-08T23:30:00Z")
    tracking.log_picks([in_play, pregame])
    log = pd.read_csv(log_path)
    log["logged_at"] = ["2026-07-09T01:00:00+00:00", "2026-07-08T15:00:00+00:00"]
    log.to_csv(log_path, index=False)

    games = pd.DataFrame(
        {
            "season": [2026, 2026], "date": pd.to_datetime(["2026-07-08", "2026-07-08"]),
            "home_team": ["ATL", "ATL"], "away_team": ["PIT", "PIT"],
            "home_runs": [5, 5], "away_runs": [2, 2], "home_win": [1, 1],
        }
    )
    monkeypatch.setattr(tr, "fetch_seasons", lambda seasons: games.copy())

    graded = tracking.grade_picks(2026)

    remaining = pd.read_csv(log_path)
    assert len(remaining) == 1  # in-play row purged from the log entirely
    assert len(graded) == 1  # only the pregame pick was graded
    assert remaining.iloc[0]["commence_time"] == "2026-07-08T23:30:00Z"


def test_summarize_record_units_and_hit_rate():
    graded = pd.DataFrame(
        [
            {"side_price": -120, "won": True},   # +0.8333u
            {"side_price": 150, "won": True},    # +1.5u
            {"side_price": -105, "won": False},  # -1u
        ]
    )
    record = tracking.summarize_record(graded)

    assert record.n == 3
    assert record.wins == 2
    assert record.losses == 1
    assert math.isclose(record.hit_rate, 2 / 3)
    assert math.isclose(record.units, 0.8333, rel_tol=1e-3) or math.isclose(record.units, 100 / 120 + 1.5 - 1, rel_tol=1e-9)
