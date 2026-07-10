"""Generate today's ranked +EV picks: model probability vs devigged market probability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from sba.data.mlb_stats import fetch_seasons, team_recent_form
from sba.data.odds import american_to_decimal, fetch_mlb_odds, games_with_devigged_odds
from sba.features import ROLLING_WINDOW, FEATURE_COLUMNS, build_live_features
from sba.model import load, predict_proba

KELLY_FRACTION = 0.25  # quarter-Kelly: conservative stake sizing
EASTERN = ZoneInfo("America/New_York")


def filter_todays_games(market_games: list[dict], now: datetime | None = None, days_ahead: int = 0) -> list[dict]:
    """Keep only games on the target US-Eastern date (today + days_ahead) that
    haven't started yet.

    The odds feed returns several days of upcoming games ("today" is defined in
    Eastern time, matching the props scan's Baseball-Reference previews page), and
    it also carries in-play games with live mid-game odds -- a trailing team shows
    +2000-style prices that have nothing to do with the pregame line the model
    reasons about. Only pregame captures are usable for picks and for the graded
    record.
    """
    now = now or datetime.now(timezone.utc)
    target_date = (now.astimezone(EASTERN) + timedelta(days=days_ahead)).date()

    kept = []
    for mg in market_games:
        commence = datetime.fromisoformat(mg["commence_time"].replace("Z", "+00:00"))
        if commence.astimezone(EASTERN).date() == target_date and commence > now:
            kept.append(mg)
    return kept


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
    n_books: int = 0
    home_price: float = 0.0  # best posted line for each side, regardless of pick
    away_price: float = 0.0


def _kelly_stake(model_prob: float, decimal_odds: float) -> float:
    """Fractional-Kelly stake as a percent of bankroll. Clipped to 0 when there's no edge."""
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    f_star = (b * model_prob - (1 - model_prob)) / b
    return max(0.0, f_star * KELLY_FRACTION)


def generate_picks(*, history_seasons: list[int], days_ahead: int = 0) -> list[Pick]:
    pipeline = load()
    games = fetch_seasons(history_seasons)
    raw_odds = fetch_mlb_odds()
    market_games = filter_todays_games(games_with_devigged_odds(raw_odds), days_ahead=days_ahead)

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
                n_books=mg["n_bookmakers"],
                home_price=mg["best_home_price"],
                away_price=mg["best_away_price"],
            )
        )

    return sorted(picks, key=lambda p: p.edge, reverse=True)
