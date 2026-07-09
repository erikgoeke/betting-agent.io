"""Scrape today's MLB slate (games, probable pitchers, active batters) from
Baseball-Reference's daily preview pages.

  - /previews/index.shtml         today's games + probable starting pitchers
  - /previews/{year}/{game}.shtml  per-game preview, including each team's
                                   active hitters and their recent usage

Neither path is in BR's robots.txt disallow list; both go through bref_http.get(),
which respects the site's `Crawl-delay: 3`.

BR doesn't publish a confirmed pregame batting lineup -- `fetch_lineup_pool` returns
the team's active hitting pool (everyone with a recent game log entry), not a
specific 9-man order. Callers should filter by recent playing time (`pa_last_28d`)
to focus on likely starters.
"""

from __future__ import annotations

import re
from io import StringIO

import pandas as pd

from sba.data import bref_http

GAME_BLOCK_PATTERN = re.compile(r'<div class="game_summary.*?</div>', re.DOTALL)
TEAM_CODE_PATTERN = re.compile(r"/teams/([A-Z]+)/\d+\.shtml")
PREVIEW_URL_PATTERN = re.compile(r'href="(/previews/\d+/[^"]+\.shtml)"')
GAME_TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}(?:AM|PM))\b")
# Ties each probable pitcher to its own team code (rather than assuming a fixed
# away/home order), so a TBD pitcher on one side doesn't shift the other team's
# real pitcher into the wrong slot.
PITCHER_ROW_PATTERN = re.compile(r"<strong>([A-Z]+)</strong></td>\s*<td>(.*?)</td>\s*</tr>", re.DOTALL)
PITCHER_LINK_PATTERN = re.compile(r'/players/[a-z]/([a-z0-9]+)\.shtml">([^<]+)</a>')
BATTER_LINK_PATTERN = re.compile(r'/players/[a-z]/([a-z0-9]+)\.shtml"[^>]*tip="([^"]+)"')


def _extract_pitchers_by_team(block: str) -> dict[str, dict | None]:
    pitchers_by_team = {}
    for team, cell in PITCHER_ROW_PATTERN.findall(block):
        link = PITCHER_LINK_PATTERN.search(cell)
        pitchers_by_team[team] = {"id": link.group(1), "name": link.group(2)} if link else None
    return pitchers_by_team


def _parse_game_block(block: str) -> dict | None:
    teams = TEAM_CODE_PATTERN.findall(block)
    preview_match = PREVIEW_URL_PATTERN.search(block)
    if len(teams) < 2 or preview_match is None:
        return None

    away_team, home_team = teams[0], teams[1]
    time_match = GAME_TIME_PATTERN.search(block)
    pitchers_by_team = _extract_pitchers_by_team(block)

    return {
        "away_team": away_team,
        "home_team": home_team,
        "game_time": time_match.group(1) if time_match else None,
        "preview_url": preview_match.group(1),
        "away_pitcher": pitchers_by_team.get(away_team),
        "home_pitcher": pitchers_by_team.get(home_team),
    }


def fetch_todays_games() -> list[dict]:
    """Today's MLB slate: matchups, game times, and probable starting pitchers."""
    resp = bref_http.get("/previews/index.shtml")
    blocks = GAME_BLOCK_PATTERN.findall(resp.text)
    games = [_parse_game_block(block) for block in blocks]
    return [g for g in games if g is not None]


def fetch_game_preview_html(preview_url: str) -> str:
    """Fetch a game preview page's raw HTML once, for parsing both teams' hitting pools."""
    return bref_http.get(preview_url).text


def parse_lineup_pool(html: str, team: str) -> list[dict]:
    """A team's active hitters from an already-fetched preview page, with recent usage.

    Returns [{player_id, name, pa_last_28d}, ...] -- the team's available hitting
    pool (not a confirmed batting order; BR has no pregame source for that).
    """
    table_id = f"batters_{team}"

    try:
        tables = pd.read_html(StringIO(html), attrs={"id": table_id})
    except (ValueError, ImportError):
        return []  # no table with this id on the page
    if not tables:
        return []
    stats = tables[0]
    stats = stats[stats["Batter"] != "TOTAL"]

    start = html.find(f'id="{table_id}"')
    end = html.find("</table>", start)
    links = BATTER_LINK_PATTERN.findall(html[start:end]) if start != -1 else []

    if len(links) != len(stats):
        return []  # structure didn't match what we expected -- skip rather than guess

    pa_last_28d = pd.to_numeric(stats["PA Last 28d"], errors="coerce").fillna(0).tolist()
    return [
        {"player_id": player_id, "name": name, "pa_last_28d": pa}
        for (player_id, name), pa in zip(links, pa_last_28d)
    ]


def fetch_lineup_pool(preview_url: str, team: str) -> list[dict]:
    """Convenience wrapper: fetch + parse a single team's pool (fetches the page)."""
    return parse_lineup_pool(fetch_game_preview_html(preview_url), team)
