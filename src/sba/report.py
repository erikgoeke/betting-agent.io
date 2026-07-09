"""Render a self-contained static HTML report: moneyline picks + today's prop scan.

Meant to be regenerated on a schedule (see .github/workflows/publish.yml) and
published as a static site -- GitHub Pages can't run the Python backend live, so this
page always reflects the results of the last time it was generated, not a live query.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from sba.config import DEFAULT_SEASONS
from sba.daily_scan import (
    DailyScanResult,
    scan_today,
    top_batters_by_hit_prob,
    top_batters_by_hr_prob,
    top_batters_by_total_bases,
    top_pitchers_by_strikeouts,
)
from sba.data.odds import OddsAPIError
from sba.picks import Pick, generate_picks

TOP_N = 10

PAGE_STYLE = """
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width: 960px; margin: 2rem auto; padding: 0 1rem;
  line-height: 1.5; color: #1a1a1a; background: #fff;
}
@media (prefers-color-scheme: dark) {
  body { color: #e8e8e8; background: #121212; }
  table { border-color: #333 !important; }
  th { background: #1e1e1e !important; }
  tr:nth-child(even) td { background: #191919 !important; }
  .disclaimer, .unavailable { background: #241f10 !important; color: #d8c98a !important; }
  a { color: #7ab8ff; }
}
h1 { margin-bottom: 0.2rem; }
.meta { color: #777; font-size: 0.9rem; margin-bottom: 2rem; }
h2 { margin-top: 2.5rem; border-bottom: 2px solid #ccc; padding-bottom: 0.3rem; }
.table-wrap { overflow-x: auto; margin: 1rem 0; }
table { border-collapse: collapse; width: 100%; border: 1px solid #ddd; }
th, td { text-align: left; padding: 0.4rem 0.7rem; border-bottom: 1px solid #ddd; white-space: nowrap; }
th { background: #f5f5f5; }
tr:nth-child(even) td { background: #fafafa; }
.disclaimer, .unavailable {
  background: #fff8e1; color: #6b5900; border-radius: 6px;
  padding: 0.8rem 1rem; font-size: 0.9rem; margin: 1rem 0;
}
footer { margin-top: 3rem; color: #888; font-size: 0.85rem; }
"""

DISCLAIMER = (
    "Informational only, not financial advice. Confirm sports betting is legal in your "
    "jurisdiction and gamble responsibly. Prop projections are a recency-weighted average "
    "of each player's own recent games, not compared to any live line (player props aren't "
    "on the free Odds API tier). Batters come from each team's recently active hitting "
    "pool, not a confirmed lineup."
)


def _esc(value) -> str:
    return html.escape(str(value))


def _render_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "<p><em>No data for today.</em></p>"
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _render_picks_section(picks: list[Pick] | None, error: str | None) -> str:
    if error is not None:
        return f'<h2>Moneyline picks</h2><p class="unavailable">Picks unavailable: {_esc(error)}</p>'

    rows = [
        [
            p.commence_time,
            f"{p.away_team} @ {p.home_team}",
            p.home_team if p.side == "home" else p.away_team,
            f"{p.side_price:+.0f}",
            f"{p.side_model_prob:.1%}",
            f"{p.side_market_prob:.1%}",
            f"{p.edge:+.1%}",
            f"{p.suggested_stake_pct:.1%}",
        ]
        for p in picks
    ]
    headers = ["Time (UTC)", "Matchup", "Pick", "Price", "Model %", "Market %", "Edge", "Kelly stake %"]
    return f"<h2>Moneyline picks</h2>{_render_table(headers, rows)}"


def _render_props_section(result: DailyScanResult) -> str:
    pitcher_rows = [
        [e.name, e.team, e.opponent, f"{e.projection.projected_strikeouts:.2f}", e.projection.n_appearances]
        for e in top_pitchers_by_strikeouts(result, TOP_N)
    ]
    hit_rows = [
        [e.name, e.team, e.opponent, f"{e.projection.hit_prob:.1%}", e.projection.n_games]
        for e in top_batters_by_hit_prob(result, TOP_N)
    ]
    hr_rows = [
        [e.name, e.team, e.opponent, f"{e.projection.hr_prob:.1%}", e.projection.n_games]
        for e in top_batters_by_hr_prob(result, TOP_N)
    ]
    tb_rows = [
        [e.name, e.team, e.opponent, f"{e.projection.projected_total_bases:.2f}", e.projection.n_games]
        for e in top_batters_by_total_bases(result, TOP_N)
    ]

    return f"""
<h2>Today's prop projections</h2>
<p class="meta">{result.n_games} games, {result.n_players_considered} players considered,
{result.n_errors} skipped.</p>
<h3>Top pitcher strikeout projections</h3>
{_render_table(["Pitcher", "Team", "Opp", "Proj. Ks", "Appearances"], pitcher_rows)}
<h3>Top batter hit-probability projections</h3>
{_render_table(["Batter", "Team", "Opp", "Hit %", "Games"], hit_rows)}
<h3>Top batter HR-probability projections</h3>
{_render_table(["Batter", "Team", "Opp", "HR %", "Games"], hr_rows)}
<h3>Top batter total-bases projections</h3>
{_render_table(["Batter", "Team", "Opp", "Proj. TB", "Games"], tb_rows)}
"""


def generate_report(output_path: Path) -> None:
    scan_result = scan_today()

    picks: list[Pick] | None = None
    picks_error: str | None = None
    try:
        picks = generate_picks(history_seasons=DEFAULT_SEASONS)
    except (OddsAPIError, FileNotFoundError) as e:
        picks_error = str(e)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sba -- MLB betting analysis</title>
<style>{PAGE_STYLE}</style>
</head>
<body>
<h1>sba -- MLB betting analysis</h1>
<p class="meta">Generated at {generated_at}. Regenerated on a schedule -- this is a snapshot, not live data.</p>
<div class="disclaimer">{DISCLAIMER}</div>
{_render_picks_section(picks, picks_error)}
{_render_props_section(scan_result)}
<footer>Built with <a href="https://github.com">sba</a>, a research tool -- not an autonomous bot, does not place bets.</footer>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
