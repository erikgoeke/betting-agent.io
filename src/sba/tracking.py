"""Forward-track generated picks and grade them once results are known.

This exists because our free odds data has no historical archive: instead of
backtesting against paid historical odds, we log every live pick and grade it
against real results as games finish, building our own track record over time.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from sba.config import PICKS_LOG_PATH
from sba.data.mlb_stats import fetch_seasons
from sba.picks import Pick

LOG_COLUMNS = [
    "logged_at", "commence_time", "home_team", "away_team",
    "side", "side_price", "model_home_win_prob", "market_home_win_prob",
    "edge", "suggested_stake_pct", "result", "won",
]


def log_picks(picks: list[Pick]) -> pd.DataFrame:
    rows = [
        {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "commence_time": p.commence_time,
            "home_team": p.home_team,
            "away_team": p.away_team,
            "side": p.side,
            "side_price": p.side_price,
            "model_home_win_prob": p.model_home_win_prob,
            "market_home_win_prob": p.market_home_win_prob,
            "edge": p.edge,
            "suggested_stake_pct": p.suggested_stake_pct,
            "result": None,
            "won": None,
        }
        for p in picks
    ]
    new_rows = pd.DataFrame(rows, columns=LOG_COLUMNS)

    if PICKS_LOG_PATH.exists():
        existing = pd.read_csv(PICKS_LOG_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(PICKS_LOG_PATH, index=False)
    return new_rows


def _game_date_eastern(commence_time: str) -> pd.Timestamp:
    """MLB schedule dates are US local dates; approximate with America/New_York
    so evening games don't shift to the wrong calendar day after UTC conversion."""
    ts = pd.Timestamp(commence_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York").normalize().tz_localize(None)


def grade_picks(season: int) -> pd.DataFrame:
    """Fill in actual outcomes for previously logged picks and return the graded rows."""
    if not PICKS_LOG_PATH.exists():
        raise FileNotFoundError(f"No picks log found at {PICKS_LOG_PATH} -- run `sba picks` first.")

    log = pd.read_csv(PICKS_LOG_PATH)
    games = fetch_seasons([season], force_refresh=True)
    games["date"] = pd.to_datetime(games["date"])

    ungraded = log[log["won"].isna()]
    for idx in ungraded.index:
        row = log.loc[idx]
        game_date = _game_date_eastern(row["commence_time"])
        match = games[
            (games["home_team"] == row["home_team"])
            & (games["away_team"] == row["away_team"])
            & (games["date"] == game_date)
        ]
        if match.empty:
            continue  # game hasn't been played yet (or falls in a different season)
        home_won = bool(match.iloc[0]["home_win"])
        picked_home = row["side"] == "home"
        won = picked_home == home_won
        log.loc[idx, "result"] = "home_win" if home_won else "away_win"
        log.loc[idx, "won"] = won

    log.to_csv(PICKS_LOG_PATH, index=False)
    return log.dropna(subset=["won"])
