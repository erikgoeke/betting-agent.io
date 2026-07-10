"""Starting-pitcher rolling features: WHIP, K%, BB%, FIP, rest days, home/away split.

Built from each starter's own Baseball-Reference game log (bref_players.py),
identified per historical game via starters.py's Retrosheet crosswalk. All
rolling stats are shifted by one start before the window is computed, same
leak-free convention as features.py's team-level rolling stats.
"""

from __future__ import annotations

import pandas as pd

from sba.data.bref_players import PlayerLookupError, fetch_pitching_game_log

STARTER_WINDOW = 10
MIN_PRIOR_STARTS = 3

STARTER_STAT_COLUMNS = ["whip", "k_rate", "bb_rate", "fip", "rest_days"]


def true_innings(ip_raw: pd.Series) -> pd.Series:
    """Baseball-Reference reports IP as e.g. 6.1/6.2 meaning 6-and-a-third/6-and-two-thirds
    innings, not 6.1/6.2 decimal -- the fractional digit is a *third* of an inning (an out
    count), not a tenth. Converts to true decimal innings; NaN IP (occasional bad row on
    Baseball-Reference's page) stays NaN rather than crashing the int cast."""
    whole = ip_raw.apply(lambda x: int(x) if pd.notna(x) else float("nan"))
    extra_outs = ((ip_raw - whole) * 10).round()
    return whole + extra_outs / 3


def _load_pitcher_log(player_id: str, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        try:
            frames.append(fetch_pitching_game_log(player_id, season))
        except PlayerLookupError:
            continue
    if not frames:
        return pd.DataFrame(columns=["date", "is_home", "IP", "BF", "SO", "BB", "H", "FIP"])
    log = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    log["ip_true"] = true_innings(log["IP"])
    # float("nan"), not pd.NA -- replacing with pd.NA on a float64 column upcasts
    # it to object dtype, which breaks the rolling().mean() calls downstream.
    log["whip"] = (log["H"] + log["BB"]) / log["ip_true"].replace(0, float("nan"))
    log["k_rate"] = log["SO"] / log["BF"].replace(0, float("nan"))
    log["bb_rate"] = log["BB"] / log["BF"].replace(0, float("nan"))
    return log


def add_starter_rolling_stats(log: pd.DataFrame, window: int = STARTER_WINDOW) -> pd.DataFrame:
    """Leak-free rolling WHIP/K%/BB%/FIP + rest days + a home/away-context FIP split,
    computed over a single pitcher's own chronological log (across seasons)."""
    log = log.sort_values("date").reset_index(drop=True)
    for col, source in [("whip", "whip"), ("k_rate", "k_rate"), ("bb_rate", "bb_rate"), ("fip", "FIP")]:
        log[f"rolling_{col}"] = log[source].shift(1).rolling(window, min_periods=MIN_PRIOR_STARTS).mean()
    log["rest_days"] = log["date"].diff().dt.days

    # Home/away split: this start's FIP rolling average using only the pitcher's
    # own prior starts in the same home/away context, falling back to the
    # context-blind rolling FIP when there isn't enough same-context history yet.
    log["split_rolling_fip"] = log.groupby("is_home")["FIP"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PRIOR_STARTS).mean()
    )
    log["split_rolling_fip"] = log["split_rolling_fip"].fillna(log["rolling_fip"])
    return log


def starter_form_asof(player_id: str, as_of: pd.Timestamp, *, window: int = STARTER_WINDOW) -> dict:
    """A single starter's rolling form entering a not-yet-played game (live picks path).

    Loads the pitcher's own cached game log for the season of `as_of` plus the
    prior season (in case they're early in a new season), same fallback pattern
    as props.py's player projections.
    """
    as_of = pd.Timestamp(as_of)
    log = _load_pitcher_log(player_id, [as_of.year - 1, as_of.year])
    prior_starts = log[log["date"] < as_of].tail(window)
    if len(prior_starts) < MIN_PRIOR_STARTS:
        return {stat: float("nan") for stat in STARTER_STAT_COLUMNS}

    rest_days = (as_of - prior_starts["date"].iloc[-1]).days
    return {
        "whip": prior_starts["whip"].mean(),
        "k_rate": prior_starts["k_rate"].mean(),
        "bb_rate": prior_starts["bb_rate"].mean(),
        "fip": prior_starts["FIP"].mean(),
        "rest_days": rest_days,
    }


def build_starter_rolling_table(starts: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, date) a pitcher started: their rolling form
    entering that start. Built once per unique starter (not once per game)."""
    home = starts[["home_starter_id", "season"]].rename(columns={"home_starter_id": "id"})
    away = starts[["away_starter_id", "season"]].rename(columns={"away_starter_id": "id"})
    pitcher_seasons = pd.concat([home, away], ignore_index=True).dropna().groupby("id")["season"].apply(
        lambda s: sorted(set(int(x) for x in s))
    )

    rows = []
    for player_id, seasons in pitcher_seasons.items():
        pitcher_log = _load_pitcher_log(player_id, seasons)
        if pitcher_log.empty:
            continue
        rolled = add_starter_rolling_stats(pitcher_log)
        rolled["player_id"] = player_id
        rows.append(rolled[["player_id", "date", "rolling_whip", "rolling_k_rate", "rolling_bb_rate", "split_rolling_fip", "rest_days"]])

    if not rows:
        return pd.DataFrame(columns=["player_id", "date", "whip", "k_rate", "bb_rate", "fip", "rest_days"])

    table = pd.concat(rows, ignore_index=True)
    return table.rename(
        columns={
            "rolling_whip": "whip", "rolling_k_rate": "k_rate", "rolling_bb_rate": "bb_rate", "split_rolling_fip": "fip",
        }
    )
