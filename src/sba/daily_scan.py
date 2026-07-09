"""Scan today's MLB slate and rank player prop projections.

No live odds are used here -- player props aren't on the free Odds API tier (see
props.py). "Best" means strongest projection, ranked separately per prop type; it is
not a market edge. Batters come from each team's active hitting pool filtered by
recent playing time, not a confirmed lineup -- Baseball-Reference has no pregame
source for that (see bref_slate.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sba.data.bref_slate import fetch_game_preview_html, fetch_todays_games, parse_lineup_pool
from sba.props import BatterProjection, PitcherProjection, project_batter, project_pitcher

DEFAULT_MIN_BATTER_PA_LAST_28D = 20


@dataclass
class PitcherEntry:
    team: str
    opponent: str
    name: str
    projection: PitcherProjection


@dataclass
class BatterEntry:
    team: str
    opponent: str
    name: str
    projection: BatterProjection


@dataclass
class DailyScanResult:
    pitchers: list[PitcherEntry] = field(default_factory=list)
    batters: list[BatterEntry] = field(default_factory=list)
    n_games: int = 0
    n_players_considered: int = 0
    n_errors: int = 0
    n_games_skipped: int = 0


def scan_today(
    min_batter_pa_last_28d: int = DEFAULT_MIN_BATTER_PA_LAST_28D,
    on_progress: Callable[[str], None] | None = None,
) -> DailyScanResult:
    def _report(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    _report("Fetching today's slate...")
    games = fetch_todays_games()
    result = DailyScanResult(n_games=len(games))
    _report(f"Found {len(games)} games today.")

    seen_pitchers: set[str] = set()
    seen_batters: set[str] = set()

    for game_num, game in enumerate(games, start=1):
        matchup = f"{game['away_team']} @ {game['home_team']}"
        sides = ((game["away_team"], game["home_team"], "away_pitcher"), (game["home_team"], game["away_team"], "home_pitcher"))
        for team, opponent, pitcher_key in sides:
            pitcher = game.get(pitcher_key)
            if not pitcher or pitcher["id"] in seen_pitchers:
                continue
            seen_pitchers.add(pitcher["id"])
            result.n_players_considered += 1
            _report(f"[{game_num}/{len(games)}] {matchup} -- pitcher: {pitcher['name']}")
            try:
                projection = project_pitcher(pitcher["id"])
                result.pitchers.append(PitcherEntry(team=team, opponent=opponent, name=pitcher["name"], projection=projection))
            except Exception:
                result.n_errors += 1

        try:
            preview_html = fetch_game_preview_html(game["preview_url"])
        except Exception:
            # A transient network/parse failure on one game's preview page shouldn't
            # abort a 15-30 minute scan -- skip this game's batters and keep going.
            result.n_games_skipped += 1
            _report(f"[{game_num}/{len(games)}] {matchup} -- couldn't fetch preview page, skipping batters.")
            continue
        for team, opponent in ((game["away_team"], game["home_team"]), (game["home_team"], game["away_team"])):
            pool = parse_lineup_pool(preview_html, team)
            for batter in pool:
                if batter["pa_last_28d"] < min_batter_pa_last_28d or batter["player_id"] in seen_batters:
                    continue
                seen_batters.add(batter["player_id"])
                result.n_players_considered += 1
                _report(f"[{game_num}/{len(games)}] {matchup} -- batter: {batter['name']}")
                try:
                    projection = project_batter(batter["player_id"])
                    result.batters.append(BatterEntry(team=team, opponent=opponent, name=batter["name"], projection=projection))
                except Exception:
                    result.n_errors += 1

    _report(
        f"Done: {len(result.pitchers)} pitchers, {len(result.batters)} batters projected "
        f"({result.n_errors} players skipped, {result.n_games_skipped} games skipped)."
    )

    return result


def top_pitchers_by_strikeouts(result: DailyScanResult, n: int = 10) -> list[PitcherEntry]:
    return sorted(result.pitchers, key=lambda e: e.projection.projected_strikeouts, reverse=True)[:n]


def top_batters_by_hit_prob(result: DailyScanResult, n: int = 10) -> list[BatterEntry]:
    return sorted(result.batters, key=lambda e: e.projection.hit_prob, reverse=True)[:n]


def top_batters_by_hr_prob(result: DailyScanResult, n: int = 10) -> list[BatterEntry]:
    return sorted(result.batters, key=lambda e: e.projection.hr_prob, reverse=True)[:n]


def top_batters_by_total_bases(result: DailyScanResult, n: int = 10) -> list[BatterEntry]:
    return sorted(result.batters, key=lambda e: e.projection.projected_total_bases, reverse=True)[:n]
