"""Team-level offense + bullpen-usage box score, per game, from Retrosheet gamelogs.

Reuses starters.py's Retrosheet gamelog fetch (same source, same season files) --
no new scraping needed, since Retrosheet's per-game linescore already carries
full team batting totals (AB, H, 2B, 3B, HR, BB, K, HBP, SF) and how many
pitchers each team used that game.
"""

from __future__ import annotations

import pandas as pd

from sba.config import TEAM_GAME_STATS_CACHE_PATH
from sba.data.starters import GamelogNotAvailable, retro_to_bbref_team, fetch_season_gamelog

TEAM_GAME_STATS_COLUMNS = [
    "season", "date", "team", "opponent",
    "ab", "h", "doubles", "triples", "hr", "bb", "so", "hbp", "sf", "pitchers_used",
]


def _side_frame(logs: pd.DataFrame, side: str) -> pd.DataFrame:
    prefix = "home" if side == "home" else "visiting"
    team_col, opp_col = ("home_team", "visiting_team") if side == "home" else ("visiting_team", "home_team")
    return pd.DataFrame(
        {
            "season": logs["season"],
            "date": logs["date"],
            "team": [retro_to_bbref_team(t, s) for t, s in zip(logs[team_col], logs["season"])],
            "opponent": [retro_to_bbref_team(t, s) for t, s in zip(logs[opp_col], logs["season"])],
            "ab": logs[f"{prefix}_abs"],
            "h": logs[f"{prefix}_hits"],
            "doubles": logs[f"{prefix}_doubles"],
            "triples": logs[f"{prefix}_triples"],
            "hr": logs[f"{prefix}_homeruns"],
            "bb": logs[f"{prefix}_bb"],
            "so": logs[f"{prefix}_k"],
            "hbp": logs[f"{prefix}_hbp"],
            "sf": logs[f"{prefix}_sac_flies"],
            "pitchers_used": logs[f"{prefix}_pitchers_used"],
        }
    )


def build_team_game_stats(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        try:
            frames.append(fetch_season_gamelog(s).assign(season=s))
        except GamelogNotAvailable as e:
            print(f"skipping team game stats for {s}: {e}")
    if not frames:
        return pd.DataFrame(columns=TEAM_GAME_STATS_COLUMNS)

    logs = pd.concat(frames, ignore_index=True)
    logs["date"] = pd.to_datetime(logs["date"], format="%Y%m%d")

    table = pd.concat([_side_frame(logs, "home"), _side_frame(logs, "visiting")], ignore_index=True)
    # games.parquet (mlb_stats.py) keeps only one row per team per date even for
    # doubleheaders -- match that convention so merges keyed on (season, date,
    # team) don't fan out (Retrosheet's own gamelog has one row per DH *leg*).
    table = table.drop_duplicates(subset=["season", "date", "team"], keep="first")
    return table.sort_values(["team", "date"]).reset_index(drop=True)


def fetch_team_game_stats(seasons: list[int], *, force_refresh: bool = False) -> pd.DataFrame:
    cached = pd.read_parquet(TEAM_GAME_STATS_CACHE_PATH) if TEAM_GAME_STATS_CACHE_PATH.exists() else None
    cached_seasons = set(cached["season"].unique()) if cached is not None else set()
    needs_fetch = {s for s in seasons if force_refresh or s not in cached_seasons}

    if needs_fetch:
        new_rows = build_team_game_stats(sorted(needs_fetch))
        if cached is not None and not new_rows.empty:
            fresh = pd.concat([cached[~cached["season"].isin(needs_fetch)], new_rows], ignore_index=True)
        elif cached is not None:
            fresh = cached
        else:
            fresh = new_rows
        fresh = fresh.sort_values(["team", "date"])
        fresh.to_parquet(TEAM_GAME_STATS_CACHE_PATH, index=False)
    else:
        fresh = cached

    return fresh[fresh["season"].isin(seasons)].reset_index(drop=True)
