"""Team-level offense (OPS/ISO/BABIP/K%/BB%) and bullpen-usage rolling features,
built from team_game_stats.py's Retrosheet-derived per-game box scores.

Same leak-free convention as features.py: every rolling stat is shifted by one
game before the window is computed.

Bullpen note: this is a fatigue *proxy* (rolling count of pitchers used per
game), not a per-reliever ERA/xFIP -- that would need every reliever's own
game log (a much bigger scrape, in the same spirit as starter_features.py but
for the whole bullpen). Pitchers-used is a reasonable stand-in for recent
bullpen strain without that additional scrape.
"""

from __future__ import annotations

import pandas as pd

OFFENSE_WINDOW = 15
MIN_PRIOR_GAMES = 3
BULLPEN_WINDOW = 3

OFFENSE_RATE_COLUMNS = ["ops", "iso", "babip", "k_rate", "bb_rate"]
OFFENSE_DIFF_COLUMNS = [f"off_{c}_diff" for c in OFFENSE_RATE_COLUMNS] + ["bullpen_usage_diff"]


def add_offense_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pa = (df["ab"] + df["bb"] + df["hbp"] + df["sf"]).replace(0, float("nan"))
    ab = df["ab"].replace(0, float("nan"))
    obp = (df["h"] + df["bb"] + df["hbp"]) / pa
    avg = df["h"] / ab
    singles = df["h"] - df["doubles"] - df["triples"] - df["hr"]
    slg = (singles + 2 * df["doubles"] + 3 * df["triples"] + 4 * df["hr"]) / ab
    babip_denom = (df["ab"] - df["so"] - df["hr"] + df["sf"]).replace(0, float("nan"))

    df["ops"] = obp + slg
    df["iso"] = slg - avg
    df["babip"] = (df["h"] - df["hr"]) / babip_denom
    df["k_rate"] = df["so"] / pa
    df["bb_rate"] = df["bb"] / pa
    return df


def add_offense_rolling(df: pd.DataFrame, window: int = OFFENSE_WINDOW) -> pd.DataFrame:
    df = add_offense_rates(df).sort_values(["team", "date"]).reset_index(drop=True)
    grouped = df.groupby("team")
    for col in OFFENSE_RATE_COLUMNS:
        df[f"rolling_{col}"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=MIN_PRIOR_GAMES).mean()
        )
    df["rolling_pitchers_used"] = grouped["pitchers_used"].transform(
        lambda s: s.shift(1).rolling(BULLPEN_WINDOW, min_periods=1).mean()
    )
    return df


def team_offense_form_asof(team_stats: pd.DataFrame, team: str, *, as_of: pd.Timestamp, window: int = OFFENSE_WINDOW) -> dict:
    """A team's offense/bullpen rolling form as of an upcoming (not-yet-played) game."""
    recent = team_stats[(team_stats["team"] == team) & (team_stats["date"] < as_of)].sort_values("date").tail(window)
    if len(recent) < MIN_PRIOR_GAMES:
        return {**{c: float("nan") for c in OFFENSE_RATE_COLUMNS}, "pitchers_used": float("nan")}
    rated = add_offense_rates(recent)
    result = {c: rated[c].mean() for c in OFFENSE_RATE_COLUMNS}
    result["pitchers_used"] = recent["pitchers_used"].tail(BULLPEN_WINDOW).mean()
    return result
