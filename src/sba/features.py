"""Feature engineering: turn game results into a leakage-free feature table.

All rolling stats are shifted by one game before the window is computed, so a
game's features only ever use information strictly prior to that game.

Strength-of-schedule: raw rolling win_pct/run_diff treat a team that beat five
last-place teams the same as one that beat five first-place teams. `sos_adj_*`
corrects for this with a one-pass "SRS-lite" adjustment (the same idea as
basketball-reference's Simple Rating System, without the iterative solve):
a team's adjusted rating is its own rolling form *plus* the average rolling
form of the opponents it just faced, so beating good teams counts for more
than beating bad ones.
"""

from __future__ import annotations

import pandas as pd

from sba.data.mlb_stats import team_recent_form
from sba.data.stadiums import haversine_km, timezone_shift
from sba.data.starters import fetch_starts
from sba.data.statcast_data import add_hard_hit_rolling, fetch_hard_hit, team_hard_hit_form_asof
from sba.data.team_game_stats import fetch_team_game_stats
from sba.data.weather import WEATHER_COLUMNS, fetch_weather_history, game_day_weather
from sba.park_factors import compute_park_factors, latest_park_factor
from sba.starter_features import STARTER_WINDOW, build_starter_rolling_table, starter_form_asof
from sba.team_offense_features import OFFENSE_DIFF_COLUMNS, OFFENSE_RATE_COLUMNS, add_offense_rolling, team_offense_form_asof
from sba.umpire_features import umpire_factor_table

ROLLING_WINDOW = 15
MIN_PRIOR_GAMES = 3

# Always present -- rows missing any of these get dropped (see build_features).
CORE_FEATURE_COLUMNS = [
    "win_pct_diff", "run_diff_diff", "rest_days_diff",
    "sos_adj_win_pct_diff", "sos_adj_run_diff_diff",
    "home_park_factor", "away_travel_km", "away_timezone_shift",
    *WEATHER_COLUMNS,
]
# Starter-pitcher and team-offense/bullpen features are allowed to be NaN (some
# historical games have no resolved starter, a starter/team with too few prior
# games, or a Retrosheet<->schedule join miss) -- LightGBM/XGBoost handle missing
# values natively, unlike the old logistic-regression model, so there's no need
# to drop those rows just to keep the feature.
STARTER_DIFF_COLUMNS = ["starter_whip_diff", "starter_k_rate_diff", "starter_bb_rate_diff", "starter_fip_diff", "starter_rest_days_diff"]
# ump_run_factor has no live-picks source (no probable-umpire feed) -- always NaN
# for build_live_features; harmless for tree models, just never informative live.
OTHER_DIFF_COLUMNS = ["hard_hit_rate_diff", "ump_run_factor"]

FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + STARTER_DIFF_COLUMNS + OFFENSE_DIFF_COLUMNS + OTHER_DIFF_COLUMNS
LABEL_COLUMN = "home_win"


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per team per game, from that team's perspective."""
    home = games.rename(
        columns={"home_team": "team", "away_team": "opponent", "home_runs": "runs_for", "away_runs": "runs_against"}
    )[["season", "date", "team", "opponent", "runs_for", "runs_against"]]
    home["win"] = games["home_win"]

    away = games.rename(
        columns={"away_team": "team", "home_team": "opponent", "away_runs": "runs_for", "home_runs": "runs_against"}
    )[["season", "date", "team", "opponent", "runs_for", "runs_against"]]
    away["win"] = 1 - games["home_win"]

    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["team", "date"]).reset_index(drop=True)


def add_rolling_form(log: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    log = log.copy()
    log["run_diff"] = log["runs_for"] - log["runs_against"]
    grouped = log.groupby("team")
    log["rolling_win_pct"] = grouped["win"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PRIOR_GAMES).mean()
    )
    log["rolling_run_diff"] = grouped["run_diff"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PRIOR_GAMES).mean()
    )
    log["rest_days"] = grouped["date"].diff().dt.days

    # Each row's opponent's own rolling form entering that same game (leak-free,
    # since it's the opponent's pre-game rolling stat too) -- the raw ingredient
    # for a strength-of-schedule adjustment.
    opponent_form = log[["team", "date", "rolling_win_pct", "rolling_run_diff"]].rename(
        columns={"team": "opponent", "rolling_win_pct": "opp_rolling_win_pct", "rolling_run_diff": "opp_rolling_run_diff"}
    )
    log = log.merge(opponent_form, on=["opponent", "date"], how="left")

    grouped = log.groupby("team")  # re-group: merge reset the frame
    log["sos_rolling_win_pct"] = grouped["opp_rolling_win_pct"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PRIOR_GAMES).mean()
    )
    log["sos_rolling_run_diff"] = grouped["opp_rolling_run_diff"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PRIOR_GAMES).mean()
    )
    log["sos_adj_win_pct"] = log["rolling_win_pct"] + (log["sos_rolling_win_pct"] - 0.5)
    log["sos_adj_run_diff"] = log["rolling_run_diff"] + log["sos_rolling_run_diff"]
    return log


def build_features(games: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Build a features + label table, one row per game, dropping games without
    enough team history yet (early season / new team stretch)."""
    log = add_rolling_form(_team_game_log(games), window=window)

    form_cols = [
        "season", "date", "team", "opponent",
        "rolling_win_pct", "rolling_run_diff", "rest_days", "sos_adj_win_pct", "sos_adj_run_diff",
    ]
    home_form = log[form_cols].rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "rolling_win_pct": "home_rolling_win_pct",
            "rolling_run_diff": "home_rolling_run_diff",
            "rest_days": "home_rest_days",
            "sos_adj_win_pct": "home_sos_adj_win_pct",
            "sos_adj_run_diff": "home_sos_adj_run_diff",
        }
    )
    away_form = log[form_cols].rename(
        columns={
            "team": "away_team",
            "opponent": "home_team",
            "rolling_win_pct": "away_rolling_win_pct",
            "rolling_run_diff": "away_rolling_run_diff",
            "rest_days": "away_rest_days",
            "sos_adj_win_pct": "away_sos_adj_win_pct",
            "sos_adj_run_diff": "away_sos_adj_run_diff",
        }
    )

    features = games.merge(home_form, on=["season", "date", "home_team", "away_team"], how="left")
    features = features.merge(away_form, on=["season", "date", "home_team", "away_team"], how="left")

    features["win_pct_diff"] = features["home_rolling_win_pct"] - features["away_rolling_win_pct"]
    features["run_diff_diff"] = features["home_rolling_run_diff"] - features["away_rolling_run_diff"]
    features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]
    features["sos_adj_win_pct_diff"] = features["home_sos_adj_win_pct"] - features["away_sos_adj_win_pct"]
    features["sos_adj_run_diff_diff"] = features["home_sos_adj_run_diff"] - features["away_sos_adj_run_diff"]

    park_factors = compute_park_factors(games).rename(columns={"team": "home_team", "park_factor": "home_park_factor"})
    features = features.merge(park_factors, on=["season", "home_team"], how="left")

    features["away_travel_km"] = [
        haversine_km(away, home) for away, home in zip(features["away_team"], features["home_team"])
    ]
    features["away_timezone_shift"] = [
        timezone_shift(away, home) for away, home in zip(features["away_team"], features["home_team"])
    ]

    # The archive API only serves data with a few days' lag -- cap the end date
    # so a still-in-progress season's most recent games don't 400 the request.
    weather_end = min(games["date"].max(), pd.Timestamp.now().normalize() - pd.Timedelta(days=3))
    weather = fetch_weather_history(str(games["date"].min().date()), str(weather_end.date()))
    weather = weather.rename(columns={"team": "home_team"})
    features = features.merge(weather, on=["home_team", "date"], how="left")

    starter_stats = ("whip", "k_rate", "bb_rate", "fip", "rest_days")
    starts = fetch_starts(sorted(games["season"].unique().tolist()))
    if not starts.empty:
        rolling = build_starter_rolling_table(starts)
        home_rolling = rolling.rename(columns={"player_id": "home_starter_id", **{s: f"home_starter_{s}" for s in starter_stats}})
        away_rolling = rolling.rename(columns={"player_id": "away_starter_id", **{s: f"away_starter_{s}" for s in starter_stats}})
        starts = starts.merge(home_rolling, on=["home_starter_id", "date"], how="left")
        starts = starts.merge(away_rolling, on=["away_starter_id", "date"], how="left")

        starter_cols = [f"{side}_starter_{s}" for side in ("home", "away") for s in starter_stats]
        features = features.merge(
            starts[["season", "date", "home_team", "away_team", *starter_cols]],
            on=["season", "date", "home_team", "away_team"], how="left",
        )
        for stat in starter_stats:
            features[f"starter_{stat}_diff"] = features[f"home_starter_{stat}"] - features[f"away_starter_{stat}"]
    else:
        for stat in starter_stats:
            features[f"starter_{stat}_diff"] = float("nan")

    team_stats = fetch_team_game_stats(sorted(games["season"].unique().tolist()))
    if not team_stats.empty:
        offense_rolling = add_offense_rolling(team_stats)
        offense_cols = ["season", "date", "team", *[f"rolling_{c}" for c in OFFENSE_RATE_COLUMNS], "rolling_pitchers_used"]
        offense_home = offense_rolling[offense_cols].rename(
            columns={"team": "home_team", **{f"rolling_{c}": f"home_off_{c}" for c in OFFENSE_RATE_COLUMNS}, "rolling_pitchers_used": "home_bullpen_usage"}
        )
        offense_away = offense_rolling[offense_cols].rename(
            columns={"team": "away_team", **{f"rolling_{c}": f"away_off_{c}" for c in OFFENSE_RATE_COLUMNS}, "rolling_pitchers_used": "away_bullpen_usage"}
        )
        features = features.merge(offense_home, on=["season", "date", "home_team"], how="left")
        features = features.merge(offense_away, on=["season", "date", "away_team"], how="left")
        for stat in OFFENSE_RATE_COLUMNS:
            features[f"off_{stat}_diff"] = features[f"home_off_{stat}"] - features[f"away_off_{stat}"]
        features["bullpen_usage_diff"] = features["home_bullpen_usage"] - features["away_bullpen_usage"]
    else:
        for col in OFFENSE_DIFF_COLUMNS:
            features[col] = float("nan")

    hard_hit = fetch_hard_hit(sorted(games["season"].unique().tolist()))
    if not hard_hit.empty:
        hard_hit = add_hard_hit_rolling(hard_hit)
        home_hh = hard_hit.rename(columns={"team": "home_team", "rolling_hard_hit_rate": "home_hard_hit_rate"})
        away_hh = hard_hit.rename(columns={"team": "away_team", "rolling_hard_hit_rate": "away_hard_hit_rate"})
        features = features.merge(
            home_hh[["season", "date", "home_team", "home_hard_hit_rate"]], on=["season", "date", "home_team"], how="left"
        )
        features = features.merge(
            away_hh[["season", "date", "away_team", "away_hard_hit_rate"]], on=["season", "date", "away_team"], how="left"
        )
        features["hard_hit_rate_diff"] = features["home_hard_hit_rate"] - features["away_hard_hit_rate"]
    else:
        features["hard_hit_rate_diff"] = float("nan")

    league_avg_runs = (games["home_runs"] + games["away_runs"]).mean()
    ump_table = umpire_factor_table(sorted(games["season"].unique().tolist()), league_avg_runs)
    if not ump_table.empty:
        features = features.merge(
            ump_table[["season", "date", "home_team", "away_team", "ump_run_factor"]],
            on=["season", "date", "home_team", "away_team"], how="left",
        )
    else:
        features["ump_run_factor"] = float("nan")

    return features.dropna(subset=CORE_FEATURE_COLUMNS).reset_index(drop=True)


def _team_form_stats(recent: pd.DataFrame, team: str) -> dict:
    """Win pct + run diff from a team's own recent-games slice (see mlb_stats.team_recent_form)."""
    is_home = recent["home_team"] == team
    runs_for = recent["home_runs"].where(is_home, recent["away_runs"])
    runs_against = recent["away_runs"].where(is_home, recent["home_runs"])
    wins = (recent["home_win"] == 1) == is_home
    return {"win_pct": wins.mean(), "run_diff": (runs_for - runs_against).mean()}


def _sos_adjusted_stats(games: pd.DataFrame, team: str, recent: pd.DataFrame, *, window: int) -> dict:
    """Strength-of-schedule adjustment for a live (not-yet-played) game.

    Same SRS-lite idea as add_rolling_form's vectorized version, just computed
    by looking up each recent opponent's own rolling form directly (only ~window
    lookups against the in-memory games history, cheap for a single upcoming game).
    """
    own = _team_form_stats(recent, team)
    opponent_qualities = []
    for row in recent.itertuples():
        opponent = row.away_team if row.home_team == team else row.home_team
        opp_recent = team_recent_form(games, opponent, as_of=row.date, window=window)
        if len(opp_recent) < MIN_PRIOR_GAMES:
            continue
        opponent_qualities.append(_team_form_stats(opp_recent, opponent))

    if not opponent_qualities:
        return {"sos_adj_win_pct": own["win_pct"], "sos_adj_run_diff": own["run_diff"]}

    avg_opp_win_pct = sum(q["win_pct"] for q in opponent_qualities) / len(opponent_qualities)
    avg_opp_run_diff = sum(q["run_diff"] for q in opponent_qualities) / len(opponent_qualities)
    return {
        "sos_adj_win_pct": own["win_pct"] + (avg_opp_win_pct - 0.5),
        "sos_adj_run_diff": own["run_diff"] + avg_opp_run_diff,
    }


def build_live_features(
    games: pd.DataFrame,
    home_recent: pd.DataFrame,
    away_recent: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
    *,
    window: int = ROLLING_WINDOW,
    home_starter_id: str | None = None,
    away_starter_id: str | None = None,
) -> dict:
    """Build a single feature row for an upcoming game.

    `games` is the full multi-season history (needed to look up each recent
    opponent's own form for the strength-of-schedule adjustment); `home_recent`/
    `away_recent` are each team's own last `window` games (see team_recent_form);
    `game_date` is used to look up that day's weather forecast at the home park.
    `home_starter_id`/`away_starter_id` are Baseball-Reference player IDs for the
    probable starters (see bref_slate.py); omitted (unannounced) starters just
    leave the starter-diff features as NaN, which the model handles natively.
    """
    home = _team_form_stats(home_recent, home_team)
    away = _team_form_stats(away_recent, away_team)
    home_sos = _sos_adjusted_stats(games, home_team, home_recent, window=window)
    away_sos = _sos_adjusted_stats(games, away_team, away_recent, window=window)
    current_season = int(games["season"].max())

    features = {
        "win_pct_diff": home["win_pct"] - away["win_pct"],
        "run_diff_diff": home["run_diff"] - away["run_diff"],
        "rest_days_diff": 0.0,
        "sos_adj_win_pct_diff": home_sos["sos_adj_win_pct"] - away_sos["sos_adj_win_pct"],
        "sos_adj_run_diff_diff": home_sos["sos_adj_run_diff"] - away_sos["sos_adj_run_diff"],
        "home_park_factor": latest_park_factor(games, home_team, as_of_season=current_season),
        "away_travel_km": haversine_km(away_team, home_team),
        "away_timezone_shift": timezone_shift(away_team, home_team),
        **game_day_weather(home_team, game_date),
    }

    home_starter = starter_form_asof(home_starter_id, game_date, window=STARTER_WINDOW) if home_starter_id else None
    away_starter = starter_form_asof(away_starter_id, game_date, window=STARTER_WINDOW) if away_starter_id else None
    for stat in ("whip", "k_rate", "bb_rate", "fip", "rest_days"):
        home_val = home_starter[stat] if home_starter else float("nan")
        away_val = away_starter[stat] if away_starter else float("nan")
        features[f"starter_{stat}_diff"] = home_val - away_val

    team_stats = fetch_team_game_stats(sorted(games["season"].unique().tolist()))
    home_offense = team_offense_form_asof(team_stats, home_team, as_of=game_date) if not team_stats.empty else None
    away_offense = team_offense_form_asof(team_stats, away_team, as_of=game_date) if not team_stats.empty else None
    for stat in OFFENSE_RATE_COLUMNS:
        home_val = home_offense[stat] if home_offense else float("nan")
        away_val = away_offense[stat] if away_offense else float("nan")
        features[f"off_{stat}_diff"] = home_val - away_val
    home_bullpen = home_offense["pitchers_used"] if home_offense else float("nan")
    away_bullpen = away_offense["pitchers_used"] if away_offense else float("nan")
    features["bullpen_usage_diff"] = home_bullpen - away_bullpen

    hard_hit = fetch_hard_hit(sorted(games["season"].unique().tolist()))
    if not hard_hit.empty:
        home_hh = team_hard_hit_form_asof(hard_hit, home_team, as_of=game_date)
        away_hh = team_hard_hit_form_asof(hard_hit, away_team, as_of=game_date)
        features["hard_hit_rate_diff"] = home_hh - away_hh
    else:
        features["hard_hit_rate_diff"] = float("nan")

    # No probable-umpire feed for live picks -- always neutral/NaN here, only
    # populated for historical training rows (see build_features).
    features["ump_run_factor"] = float("nan")

    return features
