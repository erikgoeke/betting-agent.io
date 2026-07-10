"""Forward-track generated picks and grade them once results are known.

This exists because our free odds data has no historical archive: instead of
backtesting against paid historical odds, we log every live pick and grade it
against real results as games finish, building our own track record over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from sba.config import PICKS_LOG_PATH
from sba.data.mlb_stats import fetch_seasons
from sba.data.odds import american_to_decimal
from sba.picks import Pick

LOG_COLUMNS = [
    "logged_at", "commence_time", "home_team", "away_team",
    "side", "side_price", "model_home_win_prob", "market_home_win_prob",
    "edge", "suggested_stake_pct", "n_books", "home_price", "away_price",
    "result", "won",
]

GAME_KEY = ["commence_time", "home_team", "away_team"]


def _read_log() -> pd.DataFrame:
    """Read the picks log with the `won` column normalized to real booleans.

    CSV round-trips a mixed bool/NaN column as "True"/"False" strings in an
    object column; downstream logic (isna checks, means) needs actual booleans.
    Older logs may predate newer columns; reindexing fills those with NaN.
    """
    log = pd.read_csv(PICKS_LOG_PATH)
    log = log.reindex(columns=LOG_COLUMNS)
    log["won"] = log["won"].map({True: True, False: False, "True": True, "False": False})
    return log


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
            "n_books": p.n_books,
            "home_price": p.home_price,
            "away_price": p.away_price,
            "result": None,
            "won": None,
        }
        for p in picks
    ]
    new_rows = pd.DataFrame(rows, columns=LOG_COLUMNS)

    if PICKS_LOG_PATH.exists():
        existing = _read_log()
        # Re-logging the same game (e.g. multiple runs in one day) replaces the
        # stale ungraded row with the fresh odds instead of duplicating it.
        # Graded rows are history and are never touched.
        new_keys = set(map(tuple, new_rows[GAME_KEY].itertuples(index=False)))
        is_replaced = existing["won"].isna() & existing[GAME_KEY].apply(tuple, axis=1).isin(new_keys)
        combined = pd.concat([existing[~is_replaced], new_rows], ignore_index=True)
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

    log = _read_log()

    # Purge ungraded rows that were logged at/after first pitch: those captured live
    # in-game odds (the feed carries in-play games), not the pregame line the model
    # reasons about -- grading them would corrupt the P/L record with mid-game prices.
    logged_at = pd.to_datetime(log["logged_at"], utc=True, format="ISO8601")
    commence = pd.to_datetime(log["commence_time"], utc=True, format="ISO8601")
    in_play_capture = log["won"].isna() & (logged_at >= commence)
    log = log[~in_play_capture].reset_index(drop=True)

    games = fetch_seasons([season])
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


def todays_picks_from_log() -> pd.DataFrame:
    """Every pick logged for today's US-Eastern date -- upcoming, started, and
    finished games alike (the odds feed only carries upcoming games, but the log
    keeps what the morning run captured before games began)."""
    if not PICKS_LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = _read_log()
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    game_dates = log["commence_time"].map(_game_date_eastern)
    return log[game_dates == today].reset_index(drop=True)


@dataclass
class Record:
    n: int
    wins: int
    losses: int
    hit_rate: float
    units: float  # cumulative P/L betting one unit per pick at the logged price


def summarize_record(graded: pd.DataFrame) -> Record:
    """Running record over graded picks, flat one unit per pick at the logged price."""
    wins = int(graded["won"].sum())
    losses = len(graded) - wins
    units = sum(
        (american_to_decimal(row["side_price"]) - 1) if row["won"] else -1.0
        for _, row in graded.iterrows()
    )
    return Record(
        n=len(graded),
        wins=wins,
        losses=losses,
        hit_rate=wins / len(graded) if len(graded) else 0.0,
        units=units,
    )
