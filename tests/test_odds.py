import math

from sba.data.odds import (
    american_to_decimal,
    american_to_prob,
    devig_two_way,
    games_with_devigged_odds,
)


def test_american_to_prob_favorite_and_underdog():
    assert math.isclose(american_to_prob(-110), 110 / 210)
    assert math.isclose(american_to_prob(150), 100 / 250)


def test_american_to_decimal():
    assert math.isclose(american_to_decimal(150), 2.5)
    assert math.isclose(american_to_decimal(-200), 1.5)


def test_devig_two_way_removes_vig():
    # -110/-110 has ~4.76% vig on each side; devigged should be exactly 50/50.
    raw_home = american_to_prob(-110)
    raw_away = american_to_prob(-110)
    fair_home, fair_away = devig_two_way(raw_home, raw_away)
    assert math.isclose(fair_home, 0.5, abs_tol=1e-9)
    assert math.isclose(fair_away, 0.5, abs_tol=1e-9)
    assert math.isclose(fair_home + fair_away, 1.0)


def test_devig_two_way_favors_the_favorite():
    raw_home = american_to_prob(-150)  # favorite
    raw_away = american_to_prob(130)  # underdog
    fair_home, fair_away = devig_two_way(raw_home, raw_away)
    assert fair_home > fair_away
    assert math.isclose(fair_home + fair_away, 1.0)


def test_games_with_devigged_odds_parses_and_averages_bookmakers():
    raw = [
        {
            "id": "abc123",
            "commence_time": "2026-07-09T23:00:00Z",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "New York Yankees", "price": -120},
                                {"name": "Boston Red Sox", "price": 110},
                            ],
                        }
                    ],
                },
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "New York Yankees", "price": -115},
                                {"name": "Boston Red Sox", "price": 105},
                            ],
                        }
                    ],
                },
            ],
        }
    ]

    parsed = games_with_devigged_odds(raw)
    assert len(parsed) == 1
    game = parsed[0]
    assert game["home_team"] == "NYY"
    assert game["away_team"] == "BOS"
    assert game["n_bookmakers"] == 2
    assert 0.5 < game["market_home_win_prob"] < 0.6
    assert game["best_home_price"] == -115  # less negative = better payout for the favorite
    assert game["best_away_price"] == 110  # more positive = better payout for the underdog


def test_games_with_devigged_odds_skips_unmapped_teams():
    raw = [
        {
            "id": "xyz",
            "commence_time": "2026-07-09T23:00:00Z",
            "home_team": "Some Minor League Team",
            "away_team": "Boston Red Sox",
            "bookmakers": [],
        }
    ]
    assert games_with_devigged_odds(raw) == []
