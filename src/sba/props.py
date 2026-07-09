"""Recency-weighted player prop projections from Baseball-Reference game logs.

Not a fitted model -- an exponentially-weighted moving average (EWM) over each
player's own recent games, which weights recent form more heavily than older
games. Simple and transparent, not dressed up as more rigorous than it is.
No live odds are compared here (player props aren't on the free odds tier);
these are projections to compare manually against whatever line you're looking at.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from sba.data.bref_players import fetch_batting_game_log, fetch_pitching_game_log, resolve_player_id

MIN_GAMES_BEFORE_SEASON_FALLBACK = 5
BATTER_EWM_SPAN = 10
PITCHER_EWM_SPAN = 6


@dataclass
class BatterProjection:
    player_id: str
    season: int
    n_games: int
    hit_prob: float
    projected_total_bases: float
    hr_prob: float
    recent_log: pd.DataFrame


@dataclass
class PitcherProjection:
    player_id: str
    season: int
    n_appearances: int
    projected_strikeouts: float
    recent_log: pd.DataFrame


def _recent_games(fetch_fn, player_id: str, season: int, n: int) -> pd.DataFrame:
    log = fetch_fn(player_id, season)
    if len(log) < MIN_GAMES_BEFORE_SEASON_FALLBACK:
        prior = fetch_fn(player_id, season - 1)
        log = pd.concat([prior, log], ignore_index=True).sort_values("date")
    return log.tail(n).reset_index(drop=True)


def _ewm_last(series: pd.Series, span: int) -> float:
    return float(series.astype(float).ewm(span=span).mean().iloc[-1])


def project_batter(name_or_id: str, *, season: int | None = None, games: int = 20) -> BatterProjection:
    player_id = resolve_player_id(name_or_id)
    season = season or datetime.now().year
    log = _recent_games(fetch_batting_game_log, player_id, season, games)
    if log.empty:
        raise ValueError(f"No batting game log found for '{name_or_id}' in {season} (or {season - 1}).")

    return BatterProjection(
        player_id=player_id,
        season=season,
        n_games=len(log),
        hit_prob=_ewm_last((log["H"] >= 1), BATTER_EWM_SPAN),
        projected_total_bases=_ewm_last(log["TB"], BATTER_EWM_SPAN),
        hr_prob=_ewm_last((log["HR"] >= 1), BATTER_EWM_SPAN),
        recent_log=log,
    )


def project_pitcher(name_or_id: str, *, season: int | None = None, starts: int = 12) -> PitcherProjection:
    player_id = resolve_player_id(name_or_id)
    season = season or datetime.now().year
    log = _recent_games(fetch_pitching_game_log, player_id, season, starts)
    if log.empty:
        raise ValueError(f"No pitching game log found for '{name_or_id}' in {season} (or {season - 1}).")

    return PitcherProjection(
        player_id=player_id,
        season=season,
        n_appearances=len(log),
        projected_strikeouts=_ewm_last(log["SO"], PITCHER_EWM_SPAN),
        recent_log=log,
    )
