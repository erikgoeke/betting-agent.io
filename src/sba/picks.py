"""Generate today's ranked +EV picks: model probability vs devigged market probability."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sba.data.mlb_stats import fetch_seasons, team_recent_form
from sba.data.odds import american_to_decimal, fetch_mlb_odds, games_with_devigged_odds
from sba.features import ROLLING_WINDOW, FEATURE_COLUMNS, build_live_features
from sba.model import load, predict_proba

KELLY_FRACTION = 0.25  # quarter-Kelly: conservative stake sizing


@dataclass
class Pick:
    commence_time: str
    home_team: str
    away_team: str
    model_home_win_prob: float
    market_home_win_prob: float
    side: str  # "home" or "away"
    side_price: float
    side_model_prob: float
    side_market_prob: float
    edge: float
    suggested_stake_pct: float


def _kelly_stake(model_prob: float, decimal_odds: float) -> float:
    """Fractional-Kelly stake as a percent of bankroll. Clipped to 0 when there's no edge."""
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    f_star = (b * model_prob - (1 - model_prob)) / b
    return max(0.0, f_star * KELLY_FRACTION)


def generate_picks(*, history_seasons: list[int]) -> list[Pick]:
    pipeline = load()
    games = fetch_seasons(history_seasons)
    raw_odds = fetch_mlb_odds()
    market_games = games_with_devigged_odds(raw_odds)

    picks = []
    for mg in market_games:
        as_of = pd.Timestamp(mg["commence_time"]).tz_localize(None)
        home_recent = team_recent_form(games, mg["home_team"], as_of=as_of, window=ROLLING_WINDOW)
        away_recent = team_recent_form(games, mg["away_team"], as_of=as_of, window=ROLLING_WINDOW)
        if len(home_recent) < 3 or len(away_recent) < 3:
            continue  # not enough recent history to trust the model for this team

        live_features = build_live_features(home_recent, away_recent, mg["home_team"], mg["away_team"])
        feature_row = pd.DataFrame([live_features])[FEATURE_COLUMNS]
        model_home_prob = float(predict_proba(pipeline, feature_row).iloc[0])

        market_home_prob = mg["market_home_win_prob"]
        home_edge = model_home_prob - market_home_prob
        away_edge = (1 - model_home_prob) - (1 - market_home_prob)

        if home_edge >= away_edge:
            side, side_price, edge = "home", mg["best_home_price"], home_edge
            side_model_prob, side_market_prob = model_home_prob, market_home_prob
        else:
            side, side_price, edge = "away", mg["best_away_price"], away_edge
            side_model_prob, side_market_prob = 1 - model_home_prob, 1 - market_home_prob

        stake_pct = _kelly_stake(side_model_prob, american_to_decimal(side_price))

        picks.append(
            Pick(
                commence_time=mg["commence_time"],
                home_team=mg["home_team"],
                away_team=mg["away_team"],
                model_home_win_prob=model_home_prob,
                market_home_win_prob=market_home_prob,
                side=side,
                side_price=side_price,
                side_model_prob=side_model_prob,
                side_market_prob=side_market_prob,
                edge=edge,
                suggested_stake_pct=stake_pct,
            )
        )

    return sorted(picks, key=lambda p: p.edge, reverse=True)
