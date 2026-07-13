"""Render a self-contained static HTML dashboard: moneyline model vs. market + prop scan.

Meant to be regenerated on a schedule (see .github/workflows/publish.yml) and
published as a static site -- GitHub Pages can't run the Python backend live, so this
page always reflects the results of the last time it was generated, not a live query.

Formulas are rendered as native MathML (no CDN/JS math library needed), and all
styling is inline CSS driven by role-named custom properties with deliberate light
and dark values.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from sba.config import CURRENT_YEAR, DEFAULT_SEASONS
from sba.daily_scan import (
    DailyScanResult,
    scan_today,
    top_batters_by_hit_prob,
    top_batters_by_hr_prob,
    top_batters_by_total_bases,
    top_pitchers_by_strikeouts,
)
from sba.data.mlb_stats import fetch_seasons
from sba.data.odds import OddsAPIError, american_to_decimal, prob_to_american
from sba.features import build_features
from sba.model import load as load_model
from sba.model import predict_proba
from sba.picks import KELLY_FRACTION, Pick, generate_picks
from sba.tracking import _game_date_eastern, grade_picks, log_picks, summarize_record, todays_picks_from_log

TOP_N = 10
PAGE_PASSWORD_ENV = "PAGE_PASSWORD"
EASTERN = ZoneInfo("America/New_York")

PAGE_STYLE = """
:root {
  /* Always dark -- deliberately not tied to the viewer's system preference. */
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --inset: #141413;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --hairline: #2c2c2a; --ring: rgba(255,255,255,0.09);
  --accent: #3987e5; --accent-2: #9085e9;
  --good: #0ca30c; --bad: #e66767;
  --shadow: none;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5; -webkit-font-smoothing: antialiased;
}
body::before {
  content: ""; display: block; height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
.masthead {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap; margin-bottom: 2.25rem;
}
.brand { display: flex; align-items: center; gap: 0.65rem; }
.mark {
  width: 34px; height: 34px; border-radius: 9px; flex: none;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color: #fff; font-weight: 750; font-size: 0.82rem; letter-spacing: 0.02em;
  display: flex; align-items: center; justify-content: center;
}
.brand .name { font-weight: 700; letter-spacing: -0.01em; }
.brand .tag { display: block; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.run-pill {
  display: inline-flex; align-items: center; gap: 0.45rem;
  font-size: 0.76rem; font-weight: 550; color: var(--ink-2);
  background: var(--surface); border: 1px solid var(--ring);
  border-radius: 999px; padding: 0.3rem 0.8rem;
  font-variant-numeric: tabular-nums;
}
.run-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--good); flex: none; }
h1 { font-size: 2.1rem; margin: 0 0 0.2rem; letter-spacing: -0.03em; font-weight: 750; }
.dateline { color: var(--muted); font-size: 0.92rem; margin: 0 0 2rem; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.8rem; margin-bottom: 2.5rem; }
.tile {
  background: var(--surface); border: 1px solid var(--ring); border-radius: 12px;
  padding: 0.95rem 1.05rem 0.85rem; box-shadow: var(--shadow);
}
.tile .k { font-size: 0.68rem; font-weight: 650; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.tile .v { font-size: 1.85rem; font-weight: 700; letter-spacing: -0.03em; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }
.tile .v small { font-size: 0.9rem; font-weight: 500; color: var(--ink-2); letter-spacing: 0; }
.tile.hero { border-color: color-mix(in srgb, var(--accent) 35%, var(--ring)); }
.tile.hero .v { color: var(--accent); }
.card {
  background: var(--surface); border: 1px solid var(--ring); border-radius: 14px;
  padding: 1.35rem 1.5rem 1.15rem; margin-bottom: 1.5rem; box-shadow: var(--shadow);
}
.card > h2, .card h2 {
  display: flex; align-items: center; gap: 0.55rem;
  font-size: 1.02rem; margin: 0 0 0.2rem; letter-spacing: -0.01em; font-weight: 700;
}
.card h2::before {
  content: ""; width: 8px; height: 8px; border-radius: 2.5px; flex: none;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
}
.card > .sub { color: var(--muted); font-size: 0.82rem; margin: 0 0 1rem; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem; }
@media (max-width: 520px) { .grid2 { grid-template-columns: 1fr; } }
.grid2 .card { margin-bottom: 0; }
.table-wrap { overflow-x: auto; margin: 0 -0.4rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th {
  text-align: left; color: var(--muted); font-size: 0.67rem; font-weight: 650;
  letter-spacing: 0.1em; text-transform: uppercase;
  padding: 0.35rem 0.6rem 0.5rem; border-bottom: 1px solid var(--hairline);
  white-space: nowrap;
}
td {
  padding: 0.55rem 0.6rem; border-bottom: 1px solid var(--hairline);
  white-space: nowrap; vertical-align: middle;
  transition: background 0.12s ease;
}
tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: color-mix(in srgb, var(--accent) 6%, transparent); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.strong { font-weight: 650; }
td.tc { color: var(--ink-2); font-size: 0.8rem; letter-spacing: 0.03em; }
.pos { color: var(--good); font-weight: 650; }
.neg { color: var(--bad); font-weight: 650; }
tr.won td { background: color-mix(in srgb, var(--good) 9%, transparent); }
tr.lost td { background: color-mix(in srgb, var(--bad) 7%, transparent); }
tbody tr.won:hover td { background: color-mix(in srgb, var(--good) 15%, transparent); }
tbody tr.lost:hover td { background: color-mix(in srgb, var(--bad) 12%, transparent); }
.result { font-weight: 650; }
.record { display: inline-flex; gap: 0.9rem; flex-wrap: wrap; font-variant-numeric: tabular-nums; }
.chip {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.12rem 0.6rem 0.12rem 0.5rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 650;
  border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
  background: color-mix(in srgb, var(--accent) 10%, transparent); color: var(--ink);
}
.chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); flex: none; }
.chip small { font-weight: 500; color: var(--ink-2); }
.bar {
  display: inline-block; vertical-align: middle; height: 4px; border-radius: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2)); margin-left: 0.55rem;
}
.unavailable {
  border: 1px dashed var(--hairline); border-radius: 10px; color: var(--ink-2);
  background: var(--inset); padding: 0.85rem 1.1rem; font-size: 0.88rem;
}
.daytabs {
  display: inline-flex; gap: 2px; background: var(--inset);
  border: 1px solid var(--ring); border-radius: 10px; padding: 3px; margin-bottom: 1.25rem;
}
.daytabs button {
  border: none; background: transparent; color: var(--ink-2); cursor: pointer;
  font: inherit; font-size: 0.85rem; font-weight: 600;
  padding: 0.38rem 1.1rem; border-radius: 8px;
}
.daytabs button.active { background: var(--surface); color: var(--ink); box-shadow: 0 0 0 1px var(--ring); }
.daytabs button small { display: block; font-size: 0.66rem; font-weight: 500; color: var(--muted); letter-spacing: 0.02em; }
.day-panel[hidden] { display: none; }
.math-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
.fx { background: var(--inset); border: 1px solid var(--ring); border-radius: 12px; padding: 1rem 1.1rem; }
.fx h3 {
  font-size: 0.7rem; margin: 0 0 0.6rem; color: var(--muted); font-weight: 650;
  letter-spacing: 0.1em; text-transform: uppercase;
}
.fx math { display: block; font-size: 1.08rem; margin: 0.25rem 0 0.65rem; }
.fx p { font-size: 0.8rem; color: var(--muted); margin: 0; line-height: 1.55; }
footer {
  margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--hairline);
  color: var(--muted); font-size: 0.8rem; display: flex; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap;
}
#gate { max-width: 360px; margin: 20vh auto 0; padding: 0 1.25rem; text-align: center; }
#gate .panel {
  background: var(--surface); border: 1px solid var(--ring); border-radius: 16px;
  box-shadow: var(--shadow); padding: 2rem 1.75rem 1.75rem;
}
#gate .mark { margin: 0 auto 1rem; width: 42px; height: 42px; font-size: 0.95rem; }
#gate h2 { margin: 0 0 0.25rem; letter-spacing: -0.01em; }
#gate .hint { color: var(--muted); font-size: 0.84rem; margin: 0 0 1.1rem; }
#gate input {
  display: block; width: 100%; padding: 0.65rem 0.8rem; margin: 0 0 0.8rem;
  font-size: 1rem; border: 1px solid var(--hairline); border-radius: 10px;
  background: var(--page); color: var(--ink); outline: none; text-align: center;
}
#gate input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent); }
#gate button {
  width: 100%; padding: 0.6rem 1.6rem; font-size: 0.95rem; font-weight: 650; cursor: pointer;
  border: none; border-radius: 10px; color: #fff;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
}
#gate-error { color: var(--bad); display: none; font-size: 0.85rem; margin: 0.75rem 0 0; }
#history-date {
  padding: 0.5rem 0.7rem; font-size: 0.95rem; border: 1px solid var(--hairline); border-radius: 8px;
  background: var(--page); color: var(--ink); margin: 0 0 1rem;
}
#history-date:focus { border-color: var(--accent); outline: none; }
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _fmt_line(american: float) -> str:
    return f"{american:+.0f}"


def _break_even_line(prob: float) -> str:
    """The model probability as a fair American line -- 'bet yes only at a better price'.

    An EWM of a 0/1 indicator can legitimately hit exactly 0 or 1 (e.g. a hit in
    every one of the last 20 games), where no finite line exists.
    """
    if prob <= 0 or prob >= 1:
        return "&mdash;"
    return _fmt_line(prob_to_american(prob))


def _fmt_time_et(commence_time: str) -> str:
    try:
        ts = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return ts.astimezone(EASTERN).strftime("%-I:%M %p ET")
    except ValueError:
        return commence_time


def _bar(value: float, max_value: float, max_px: int = 44) -> str:
    """Tiny inline data bar, length proportional to value within its table."""
    if max_value <= 0:
        return ""
    width = max(2, round(max_px * value / max_value))
    return f'<span class="bar" style="width:{width}px"></span>'


def _table(headers: list[tuple[str, bool]], rows: list[list[str]]) -> str:
    """headers: (label, is_numeric). Cell strings are pre-rendered HTML."""
    if not rows:
        return '<p class="unavailable">No qualifying entries today.</p>'
    head = "".join(f'<th{" class=\"num\"" if num else ""}>{_esc(label)}</th>' for label, num in headers)
    body = "".join("<tr>" + "".join(cells) + "</tr>" for cells in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _picks_to_frame(picks: list[Pick]) -> pd.DataFrame:
    """Fallback frame when there is no picks log to read today's slate from."""
    return pd.DataFrame(
        [
            {
                "commence_time": p.commence_time, "home_team": p.home_team, "away_team": p.away_team,
                "side": p.side, "side_price": p.side_price,
                "model_home_win_prob": p.model_home_win_prob, "market_home_win_prob": p.market_home_win_prob,
                "edge": p.edge, "suggested_stake_pct": p.suggested_stake_pct,
                "n_books": p.n_books, "home_price": p.home_price, "away_price": p.away_price,
                "result": None, "won": None,
            }
            for p in picks
        ]
    )


def _row_status(row, now: datetime) -> tuple[str, str]:
    """(status html, tr class) for a game row: graded final, started, or upcoming."""
    if row["won"] is True:
        return '<span class="result pos">&#10003; Won</span>', "won"
    if row["won"] is False:
        return '<span class="result neg">&#10007; Lost</span>', "lost"
    commence = datetime.fromisoformat(str(row["commence_time"]).replace("Z", "+00:00"))
    if commence <= now:
        return '<span class="tc">In progress</span>', ""
    return '<span class="tc">Upcoming</span>', ""


def _render_tiles(result: DailyScanResult, picks: list[Pick] | None) -> str:
    tiles = [
        ("Games today", f"{result.n_games}", ""),
        ("Players modeled", f"{result.n_players_considered - result.n_errors}", ""),
        ("Starting pitchers", f"{len(result.pitchers)}", ""),
        ("Batters in pool", f"{len(result.batters)}", ""),
    ]
    if picks:
        best = max(picks, key=lambda p: p.edge)
        tiles.append(
            ("Best model edge", f"{best.edge:+.1%} <small>{_esc(best.away_team)}@{_esc(best.home_team)}</small>", " hero")
        )
    cells = "".join(
        f'<div class="tile{cls}"><div class="k">{_esc(k)}</div><div class="v">{v}</div></div>' for k, v, cls in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _render_picks_section(
    today: pd.DataFrame, error: str | None, now: datetime,
    empty_note: str = "No games logged for today yet.",
) -> str:
    header = (
        '<h2>Moneyline &mdash; model vs. market</h2>'
        '<p class="sub">Fair line is the no-vig price implied by the model '
        "probability; book line is the best posted price across surveyed sportsbooks &mdash; for games "
        "already started or finished, the last price captured before first pitch. Edge is the "
        "model&ndash;market probability gap on the picked side.</p>"
    )
    if today.empty:
        reason = f"Picks unavailable: {_esc(error)}" if error else _esc(empty_note)
        return f'<div class="card">{header}<p class="unavailable">{reason}</p></div>'

    headers = [
        ("Time", False), ("Matchup", False), ("Pick", False), ("Status", False),
        ("Book line", True), ("Fair line", True), ("Line gap", True),
        ("Model", True), ("Market", True), ("Edge", True),
        ("¼-Kelly", True), ("Books", True),
    ]
    max_edge = max(today["edge"].max(), 0)
    body = []
    for _, row in today.sort_values("edge", ascending=False).iterrows():
        pick_home = row["side"] == "home"
        pick_team = row["home_team"] if pick_home else row["away_team"]
        model_prob = row["model_home_win_prob"] if pick_home else 1 - row["model_home_win_prob"]
        market_prob = row["market_home_win_prob"] if pick_home else 1 - row["market_home_win_prob"]
        fair = prob_to_american(model_prob)
        gap = row["side_price"] - fair
        status, tr_cls = _row_status(row, now)
        n_books = "&mdash;" if pd.isna(row["n_books"]) else f"{int(row['n_books'])}"
        body.append(
            f'<tr class="{tr_cls}">'
            f'<td class="tc">{_esc(_fmt_time_et(row["commence_time"]))}</td>'
            f'<td>{_esc(row["away_team"])} @ {_esc(row["home_team"])}</td>'
            f'<td class="strong"><span class="chip">{_esc(pick_team)}&nbsp;<small>{_esc(row["side"])}</small></span></td>'
            f"<td>{status}</td>"
            f'<td class="num strong">{_fmt_line(row["side_price"])}</td>'
            f'<td class="num">{_fmt_line(fair)}</td>'
            f'<td class="num {"pos" if gap >= 0 else "neg"}">{gap:+.0f}</td>'
            f'<td class="num">{model_prob:.1%}</td>'
            f'<td class="num">{market_prob:.1%}</td>'
            f'<td class="num"><span class="{"pos" if row["edge"] >= 0 else "neg"}">{row["edge"]:+.1%}</span>{_bar(max(row["edge"], 0), max_edge)}</td>'
            f'<td class="num">{row["suggested_stake_pct"]:.1%}</td>'
            f'<td class="num">{n_books}</td>'
            "</tr>"
        )
    head = "".join(f'<th{" class=\"num\"" if num else ""}>{_esc(label)}</th>' for label, num in headers)
    note = f'<p class="sub">Live odds refresh unavailable this run: {_esc(error)}</p>' if error else ""
    return (
        f'<div class="card">{header}{note}'
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>'
    )


def _render_winners_section(today: pd.DataFrame, now: datetime) -> str:
    """Every game ranked by the model's win confidence -- who's most likely to win,
    which is a different question from where the betting value is."""
    if today.empty:
        return ""

    frame = today.copy()
    frame["win_prob"] = frame["model_home_win_prob"].map(lambda p: max(p, 1 - p))
    max_prob = frame["win_prob"].max()

    body = []
    for _, row in frame.sort_values("win_prob", ascending=False).iterrows():
        home_favored = row["model_home_win_prob"] >= 0.5
        winner = row["home_team"] if home_favored else row["away_team"]
        market = row["market_home_win_prob"] if home_favored else 1 - row["market_home_win_prob"]
        price = row["home_price"] if home_favored else row["away_price"]
        price_str = "&mdash;" if pd.isna(price) else _fmt_line(price)
        # Green/red only when the model's predicted WINNER was right/wrong -- this can
        # differ from the edge pick's result when they were different sides.
        winner_won = None
        if row["won"] is not None and not pd.isna(row["won"]):
            edge_pick_home = row["side"] == "home"
            home_won = bool(row["won"]) == edge_pick_home
            winner_won = home_won == home_favored
        status = (
            '<span class="result pos">&#10003; Won</span>' if winner_won is True
            else '<span class="result neg">&#10007; Lost</span>' if winner_won is False
            else _row_status(row, now)[0]
        )
        tr_cls = "won" if winner_won is True else "lost" if winner_won is False else ""
        body.append(
            f'<tr class="{tr_cls}">'
            f'<td class="tc">{_esc(_fmt_time_et(row["commence_time"]))}</td>'
            f'<td>{_esc(row["away_team"])} @ {_esc(row["home_team"])}</td>'
            f'<td class="strong"><span class="chip">{_esc(winner)}&nbsp;<small>{"home" if home_favored else "away"}</small></span></td>'
            f"<td>{status}</td>"
            f'<td class="num strong">{row["win_prob"]:.1%}{_bar(row["win_prob"], max_prob)}</td>'
            f'<td class="num">{_break_even_line(row["win_prob"])}</td>'
            f'<td class="num">{market:.1%}</td>'
            f'<td class="num">{price_str}</td>'
            "</tr>"
        )

    headers = [
        ("Time", False), ("Matchup", False), ("Predicted winner", False), ("Status", False),
        ("Win prob", True), ("Fair line", True), ("Market", True), ("Book line", True),
    ]
    head = "".join(f'<th{" class=\"num\"" if num else ""}>{_esc(label)}</th>' for label, num in headers)
    return (
        '<div class="card"><h2>Most likely winners</h2>'
        '<p class="sub">Every game ranked by the model\'s win confidence. Confidence is not value: '
        "a heavy favorite can be a bad bet at its price, and the best-value plays are usually in the "
        "edge table above, not here.</p>"
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>'
    )


def _model_retrospective(days_offset: int, now: datetime) -> pd.DataFrame:
    """The model's pregame opinion, reconstructed after the fact, for one day's
    completed games. Needs no odds: features are pure team form, and the feature
    builder only uses games strictly before each game's date -- so this is the
    same probability the model would have produced before first pitch."""
    games = fetch_seasons(DEFAULT_SEASONS)
    features = build_features(games)
    target = (now.astimezone(EASTERN) + timedelta(days=days_offset)).date()
    day = features[features["date"].dt.date == target].copy()
    if day.empty:
        return day
    day["prob_home"] = predict_proba(load_model(), day)
    return day


def _render_retrospective_card(day: pd.DataFrame) -> str:
    """Finished games vs. the model's reconstructed pregame call -- no odds involved."""
    if day.empty:
        return ""

    n_correct = int((day["home_win"] == (day["prob_home"] >= 0.5).astype(int)).sum())
    body = []
    for _, row in day.assign(conf=day["prob_home"].map(lambda p: max(p, 1 - p))).sort_values("conf", ascending=False).iterrows():
        home_favored = row["prob_home"] >= 0.5
        winner = row["home_team"] if home_favored else row["away_team"]
        prob = row["prob_home"] if home_favored else 1 - row["prob_home"]
        correct = bool(row["home_win"]) == home_favored
        body.append(
            f'<tr class="{"won" if correct else "lost"}">'
            f'<td>{_esc(row["away_team"])} @ {_esc(row["home_team"])}</td>'
            f'<td class="num">{row["away_runs"]:.0f}&ndash;{row["home_runs"]:.0f}</td>'
            f'<td class="strong"><span class="chip">{_esc(winner)}&nbsp;<small>{"home" if home_favored else "away"}</small></span></td>'
            f'<td class="num strong">{prob:.1%}</td>'
            f'<td class="num">{_break_even_line(prob)}</td>'
            f'<td><span class="result {"pos" if correct else "neg"}">{"&#10003; Correct" if correct else "&#10007; Wrong"}</span></td>'
            "</tr>"
        )
    headers = [("Matchup", False), ("Final", True), ("Model pick", False), ("Win prob", True), ("Fair line", True), ("Verdict", False)]
    head = "".join(f'<th{" class=\"num\"" if num else ""}>{_esc(label)}</th>' for label, num in headers)
    return (
        '<div class="card"><h2>Model retrospective</h2>'
        f'<p class="sub">No pregame odds were logged for this day, but the model\'s call is reconstructable '
        f"from team form alone &mdash; it picked the winner in <strong>{n_correct} of {len(day)}</strong> finished games. "
        "Fair line is what the model would have considered a fair price; no comparison to actual books was possible.</p>"
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>'
    )


def _render_model_update_card(actual: pd.DataFrame, now: datetime, days_offset: int = -1) -> str:
    """For a past day's *actually logged and graded* picks that lost, show what
    today's freshly retrained model would have called for the same games.

    This does not rewrite the historical record -- the logged pick and its
    result stay exactly what they were graded as, since that's a real track
    record. It's an honest, additive diff against the model that was actually
    live at the time, using the same odds-free reconstruction as the
    retrospective card above.
    """
    graded = actual.dropna(subset=["won"])
    losses = graded[graded["won"] == False]  # noqa: E712 -- pandas bool column, not a Python bool
    if losses.empty:
        return ""

    day = _model_retrospective(days_offset, now)
    if day.empty:
        return ""

    merged = losses.merge(
        day[["home_team", "away_team", "prob_home"]], on=["home_team", "away_team"], how="inner"
    )
    if merged.empty:
        return ""

    body = []
    n_now_correct = 0
    for _, row in merged.iterrows():
        home_won = row["result"] == "home_win"
        old_side = row["side"]
        new_home_favored = row["prob_home"] >= 0.5
        new_side = "home" if new_home_favored else "away"
        new_prob = row["prob_home"] if new_home_favored else 1 - row["prob_home"]
        new_winner = row["home_team"] if new_home_favored else row["away_team"]
        new_correct = new_home_favored == home_won
        if new_correct:
            n_now_correct += 1
        body.append(
            f'<tr class="{"won" if new_correct else "lost"}">'
            f'<td>{_esc(row["away_team"])} @ {_esc(row["home_team"])}</td>'
            f'<td>{_esc(row["home_team"] if old_side == "home" else row["away_team"])}</td>'
            f'<td class="strong"><span class="chip">{_esc(new_winner)}&nbsp;<small>{new_side}</small></span></td>'
            f'<td class="num strong">{new_prob:.1%}</td>'
            f'<td><span class="result {"pos" if new_correct else "neg"}">'
            f'{"&#10003; Would now be right" if new_correct else "&#10007; Still wrong"}</span></td>'
            "</tr>"
        )

    headers = [("Matchup", False), ("Actual pick", False), ("Current model's pick", False), ("Win prob", True), ("Verdict", False)]
    head = "".join(f'<th{" class=\"num\"" if num else ""}>{_esc(label)}</th>' for label, num in headers)
    return (
        '<div class="card"><h2>Retrained model vs. yesterday\'s losses</h2>'
        f'<p class="sub">Of yesterday\'s <strong>{len(merged)}</strong> losing pick(s), the model as retrained '
        f"today would have picked the winner in <strong>{n_now_correct}</strong>. This doesn't change the "
        "logged record above -- that's what was actually bet -- it's a transparent look at whether recent "
        "model changes would have helped.</p>"
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>'
    )


def _build_history_index() -> dict[str, list[dict]]:
    """Every day this season's reconstructed pregame model call, keyed by ISO
    date -- powers the client-side "browse any date" picker below. Same
    odds-free reconstruction as _model_retrospective (features only ever use
    information strictly before each game, so this is the same probability
    the model would have produced before first pitch), just precomputed for
    every date at once instead of one day at a time.

    Scoped to the current season only, not full multi-season history -- a few
    hundred games embeds fine in a static page; six seasons' worth (~13k
    games) would bloat it for little practical benefit (nobody's placing a
    bet on a 2022 game).
    """
    games = fetch_seasons(DEFAULT_SEASONS)
    features = build_features(games)
    season_features = features[features["season"] == CURRENT_YEAR]
    if season_features.empty:
        return {}

    probs = predict_proba(load_model(), season_features)
    index: dict[str, list[dict]] = {}
    for (_, row), prob in zip(season_features.iterrows(), probs):
        date_key = row["date"].strftime("%Y-%m-%d")
        index.setdefault(date_key, []).append(
            {
                "away_team": row["away_team"],
                "home_team": row["home_team"],
                "away_runs": float(row["away_runs"]),
                "home_runs": float(row["home_runs"]),
                "home_win": int(row["home_win"]),
                "prob_home": round(float(prob), 4),
            }
        )
    return index


def _build_edge_history_index(graded: pd.DataFrame) -> dict[str, list[dict]]:
    """Graded edge picks (real logged odds) keyed by ET game date -- the value-side
    history the picks log has been accumulating since logging began. Unlike the
    model retrospective this CANNOT be reconstructed for earlier dates: an edge
    pick is model-vs-market, and no historical odds source exists (free tier)."""
    if graded is None or graded.empty:
        return {}
    index: dict[str, list[dict]] = {}
    for _, row in graded.dropna(subset=["won"]).iterrows():
        date_key = _game_date_eastern(row["commence_time"]).strftime("%Y-%m-%d")
        picked_home = row["side"] == "home"
        index.setdefault(date_key, []).append(
            {
                "away_team": row["away_team"],
                "home_team": row["home_team"],
                "pick": row["home_team"] if picked_home else row["away_team"],
                "side": row["side"],
                "price": float(row["side_price"]),
                "model": round(float(row["model_home_win_prob"] if picked_home else 1 - row["model_home_win_prob"]), 4),
                "edge": round(float(row["edge"]), 4),
                "won": bool(row["won"]),
            }
        )
    return index


def _render_history_browser(history_index: dict[str, list[dict]], edge_index: dict[str, list[dict]] | None = None) -> str:
    """A date picker over the season's history, rendered entirely client-side via
    embedded JSON -- this is a static page with no backend to query on demand.

    Two layers per date: the actual edge picks graded at their logged real odds
    (only for dates since pick-logging began -- edge picks are model-vs-market and
    historical odds don't exist before then), and the current model's winner calls
    reapplied in hindsight (available for every date, no odds involved).
    """
    edge_index = edge_index or {}
    if not history_index and not edge_index:
        return ""

    dates = sorted(set(history_index) | set(edge_index))
    data_json = json.dumps(history_index).replace("</script>", "<\\/script>")
    edge_json = json.dumps(edge_index).replace("</script>", "<\\/script>")
    return f"""
<div class="card">
<h2>Browse any date this season</h2>
<p class="sub"><strong>Edge picks</strong> are the value plays graded at their real logged odds -- the
same thing the Results table tracks. Only dates since pick-logging began have them: an edge is
model-vs-market, and no historical odds source exists for earlier dates. <strong>Predicted winners</strong>
is <strong>today's</strong> model reapplied to every game as a straight who-wins call (no odds involved) --
recomputed each time the model retrains, so it can disagree with what was actually bet at the time.
That disagreement is the model improving, not a bug.</p>
<input type="date" id="history-date" min="{dates[0]}" max="{dates[-1]}">
<div class="daytabs" id="history-tabs">
  <button type="button" data-view="edge" class="active">Edge picks<small>real odds</small></button>
  <button type="button" data-view="winners">Predicted winners<small>no odds</small></button>
</div>
<div id="history-edge-panel"></div>
<div id="history-winners-panel" hidden></div>
</div>
<script type="application/json" id="history-data">{data_json}</script>
<script type="application/json" id="edge-history-data">{edge_json}</script>
<script>
(function () {{
  var data = JSON.parse(document.getElementById("history-data").textContent);
  var edgeData = JSON.parse(document.getElementById("edge-history-data").textContent);
  var input = document.getElementById("history-date");
  var edgePanel = document.getElementById("history-edge-panel");
  var winnersPanel = document.getElementById("history-winners-panel");
  var tabs = document.querySelectorAll("#history-tabs button");

  function americanToDecimal(o) {{ return o > 0 ? 1 + o / 100 : 1 + 100 / -o; }}
  function recordSummary(wins, losses, units, note) {{
    var unitsStr = (units >= 0 ? "+" : "") + units.toFixed(2) + "u";
    var hitRate = (wins + losses) ? (wins / (wins + losses) * 100).toFixed(1) : "0.0";
    return '<p class="sub"><span class="record"><span>Record <strong>' + wins + '&ndash;' + losses + '</strong></span>' +
      '<span>Hit rate <strong>' + hitRate + '%</strong></span>' +
      '<span>P/L <strong class="' + (units >= 0 ? "pos" : "neg") + '">' + unitsStr + '</strong> ' + note + '</span></span></p>';
  }}

  function edgeSection(picks) {{
    if (!picks || !picks.length) {{
      return '<p class="unavailable">No picks were logged with odds for this date &mdash; edge history only exists ' +
        'from the day pick-logging began. Earlier dates have no odds to grade against; see the Predicted winners tab.</p>';
    }}
    var wins = 0, units = 0;
    var rows = picks.map(function (g) {{
      var pl = g.won ? americanToDecimal(g.price) - 1 : -1;
      if (g.won) wins++;
      units += pl;
      return '<tr class="' + (g.won ? "won" : "lost") + '">' +
        '<td>' + g.away_team + ' @ ' + g.home_team + '</td>' +
        '<td class="strong"><span class="chip">' + g.pick + '&nbsp;<small>' + g.side + '</small></span></td>' +
        '<td class="num">' + (g.price > 0 ? "+" : "") + g.price.toFixed(0) + '</td>' +
        '<td class="num">' + (g.model * 100).toFixed(1) + '%</td>' +
        '<td class="num">' + (g.edge >= 0 ? "+" : "") + (g.edge * 100).toFixed(1) + '%</td>' +
        '<td><span class="result ' + (g.won ? "pos" : "neg") + '">' + (g.won ? "&#10003; Won" : "&#10007; Lost") + '</span></td>' +
        '<td class="num ' + (pl >= 0 ? "pos" : "neg") + '">' + (pl >= 0 ? "+" : "") + pl.toFixed(2) + 'u</td></tr>';
    }}).join("");
    return recordSummary(wins, picks.length - wins, units, "(flat 1u at the actual captured price)") +
      '<div class="table-wrap"><table><thead><tr><th>Matchup</th><th>Pick</th><th class="num">Price</th>' +
      '<th class="num">Model</th><th class="num">Edge</th><th>Result</th><th class="num">P/L</th></tr></thead><tbody>' +
      rows + '</tbody></table></div>';
  }}

  function winnerSection(games) {{
    if (!games || !games.length) {{
      return '<p class="unavailable">No games this season have enough prior team history for a model call on this date.</p>';
    }}
    var STANDARD_DECIMAL_ODDS = 1 + 100 / 110;  // no real odds for this view -- see card note
    var wins = 0, units = 0;
    var rows = games.map(function (g) {{
      var homeFavored = g.prob_home >= 0.5;
      var winner = homeFavored ? g.home_team : g.away_team;
      var prob = homeFavored ? g.prob_home : 1 - g.prob_home;
      var correct = (g.home_win === 1) === homeFavored;
      if (correct) {{ wins++; units += STANDARD_DECIMAL_ODDS - 1; }} else {{ units -= 1; }}
      return '<tr class="' + (correct ? "won" : "lost") + '">' +
        '<td>' + g.away_team + ' @ ' + g.home_team + '</td>' +
        '<td class="num">' + g.away_runs.toFixed(0) + '&ndash;' + g.home_runs.toFixed(0) + '</td>' +
        '<td class="strong"><span class="chip">' + winner + '&nbsp;<small>' + (homeFavored ? "home" : "away") + '</small></span></td>' +
        '<td class="num strong">' + (prob * 100).toFixed(1) + '%</td>' +
        '<td><span class="result ' + (correct ? "pos" : "neg") + '">' + (correct ? "&#10003; Correct" : "&#10007; Wrong") + '</span></td></tr>';
    }}).join("");
    return recordSummary(wins, games.length - wins, units, "(flat 1u at a standard -110 line, not a real price)") +
      '<div class="table-wrap"><table><thead><tr><th>Matchup</th><th class="num">Final</th><th>Model pick</th>' +
      '<th class="num">Win prob</th><th>Verdict</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }}

  function render(dateKey) {{
    edgePanel.innerHTML = edgeSection(edgeData[dateKey]);
    winnersPanel.innerHTML = winnerSection(data[dateKey]);
  }}

  tabs.forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      tabs.forEach(function (b) {{ b.classList.remove("active"); }});
      btn.classList.add("active");
      edgePanel.hidden = btn.dataset.view !== "edge";
      winnersPanel.hidden = btn.dataset.view !== "winners";
    }});
  }});

  input.addEventListener("change", function () {{ render(input.value); }});
  var allDates = Object.keys(data).concat(Object.keys(edgeData));
  var last = allDates.sort().slice(-1)[0];
  if (last) {{ input.value = last; render(last); }}
}})();
</script>
"""


def _render_day_tabs(
    yesterday: pd.DataFrame, today: pd.DataFrame, tomorrow: pd.DataFrame,
    error: str | None, now: datetime,
) -> str:
    """Yesterday / Today / Tomorrow switcher over the game tables.

    Yesterday comes from the picks log (graded results at pregame-captured odds);
    tomorrow from the early posted lines, which move before first pitch and are
    refreshed by tomorrow's own runs.
    """
    et_now = now.astimezone(EASTERN)
    days = [
        ("yesterday", "Yesterday", (et_now - timedelta(days=1)).strftime("%b %-d"), yesterday, -1,
         "No pregame-logged games for yesterday.", ""),
        ("today", "Today", et_now.strftime("%b %-d"), today, 0,
         "No games logged for today yet.", ""),
        ("tomorrow", "Tomorrow", (et_now + timedelta(days=1)).strftime("%b %-d"), tomorrow, None,
         "No lines posted for tomorrow yet.",
         '<p class="sub">Early lines &mdash; prices move before first pitch; tomorrow\'s own runs refresh them.</p>'),
    ]
    tabs = "".join(
        f'<button type="button" data-day="{key}"{" class=\"active\"" if key == "today" else ""}>'
        f"{label}<small>{stamp}</small></button>"
        for key, label, stamp, *_ in days
    )
    panels = []
    for key, _, _, frame, retro_offset, empty, note in days:
        # When a past day has no logged odds, fall back to the model's reconstructed
        # pregame call graded against the final scores (no odds needed for that).
        content = ""
        if frame.empty and retro_offset is not None:
            try:
                content = _render_retrospective_card(_model_retrospective(retro_offset, now))
            except Exception:
                content = ""
        if not content:
            content = (
                _render_picks_section(frame, error if key == "today" else None, now, empty_note=empty)
                + _render_winners_section(frame, now)
            )
            if key == "yesterday" and retro_offset is not None:
                try:
                    content += _render_model_update_card(frame, now, days_offset=retro_offset)
                except Exception:
                    pass
        panels.append(f'<div class="day-panel" id="day-{key}"{"" if key == "today" else " hidden"}>{note}{content}</div>')
    panels = "".join(panels)
    script = """
<script>
document.querySelectorAll(".daytabs button").forEach(function (btn) {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".daytabs button").forEach(function (b) { b.classList.remove("active"); });
    btn.classList.add("active");
    document.querySelectorAll(".day-panel").forEach(function (p) { p.hidden = true; });
    document.getElementById("day-" + btn.dataset.day).hidden = false;
  });
});
</script>
"""
    return f'<div class="daytabs">{tabs}</div>{panels}{script}'


def _render_results_section(graded: pd.DataFrame, max_rows: int = 20) -> str:
    """Graded past picks, most recent first -- green rows won, red rows lost."""
    if graded.empty:
        return ""
    record = summarize_record(graded)
    recent = graded.sort_values("commence_time", ascending=False).head(max_rows)

    headers = [
        ("Date", False), ("Matchup", False), ("Pick", False),
        ("Price", True), ("Model", True), ("Edge", True), ("Result", False), ("P/L", True),
    ]
    head = "".join(f'<th{" class=\"num\"" if num else ""}>{_esc(label)}</th>' for label, num in headers)

    body = []
    for _, row in recent.iterrows():
        won = bool(row["won"])
        pick_team = row["home_team"] if row["side"] == "home" else row["away_team"]
        model_prob = row["model_home_win_prob"] if row["side"] == "home" else 1 - row["model_home_win_prob"]
        pl = (american_to_decimal(row["side_price"]) - 1) if won else -1.0
        date = datetime.fromisoformat(str(row["commence_time"]).replace("Z", "+00:00")).astimezone(EASTERN)
        body.append(
            f'<tr class="{"won" if won else "lost"}">'
            f'<td class="tc">{date.strftime("%b %-d")}</td>'
            f"<td>{_esc(row['away_team'])} @ {_esc(row['home_team'])}</td>"
            f'<td class="strong"><span class="chip">{_esc(pick_team)}&nbsp;<small>{_esc(row["side"])}</small></span></td>'
            f'<td class="num">{_fmt_line(row["side_price"])}</td>'
            f'<td class="num">{model_prob:.1%}</td>'
            f'<td class="num">{row["edge"]:+.1%}</td>'
            f'<td><span class="result {"pos" if won else "neg"}">{"&#10003; Won" if won else "&#10007; Lost"}</span></td>'
            f'<td class="num {"pos" if won else "neg"}">{pl:+.2f}u</td>'
            "</tr>"
        )

    return (
        '<div class="card"><h2>Results &mdash; graded picks</h2>'
        f'<p class="sub"><span class="record"><span>Record <strong>{record.wins}&ndash;{record.losses}</strong></span>'
        f"<span>Hit rate <strong>{record.hit_rate:.1%}</strong></span>"
        f'<span>P/L <strong class="{"pos" if record.units >= 0 else "neg"}">{record.units:+.2f}u</strong> '
        f"(flat 1u per pick)</span></span></p>"
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
        "</div>"
    )


def _props_card(title: str, sub: str, headers: list[tuple[str, bool]], rows: list[list[str]]) -> str:
    return f'<div class="card"><h2>{title}</h2><p class="sub">{sub}</p>{_table(headers, rows)}</div>'


def _render_props_section(result: DailyScanResult) -> str:
    def batter_rows(entries, fmt, key, with_break_even=False):
        values = [key(e) for e in entries]
        max_v = max(values, default=0)
        rows = []
        for e in entries:
            # The break-even column replaces the data bar: probability values cluster
            # in a narrow band where a bar reads as noise, and the extra column needs
            # the width.
            metric = fmt(key(e)) if with_break_even else f"{fmt(key(e))}{_bar(key(e), max_v)}"
            cells = [
                f"<td>{_esc(e.name)}</td>",
                f'<td class="tc">{_esc(e.team)}</td>',
                f'<td class="tc">{_esc(e.opponent)}</td>',
                f'<td class="num strong">{metric}</td>',
            ]
            if with_break_even:
                cells.append(f'<td class="num">{_break_even_line(key(e))}</td>')
            cells.append(f'<td class="num">{e.projection.n_games}</td>')
            rows.append(cells)
        return rows

    pitchers = top_pitchers_by_strikeouts(result, TOP_N)
    max_k = max((e.projection.projected_strikeouts for e in pitchers), default=0)
    pitcher_rows = [
        [
            f"<td>{_esc(e.name)}</td>",
            f'<td class="tc">{_esc(e.team)}</td>',
            f'<td class="tc">{_esc(e.opponent)}</td>',
            f'<td class="num strong">{e.projection.projected_strikeouts:.2f}{_bar(e.projection.projected_strikeouts, max_k)}</td>',
            f'<td class="num">{e.projection.n_appearances}</td>',
        ]
        for e in pitchers
    ]

    bcols = [("Batter", False), ("Team", False), ("Opp", False)]
    pcols = [("Pitcher", False), ("Team", False), ("Opp", False)]
    n = [("N", True)]

    return (
        '<div class="grid2">'
        + _props_card(
            "Pitcher strikeouts", "EWM projection over recent appearances",
            pcols + [("Proj. K", True)] + n, pitcher_rows,
        )
        + _props_card(
            "Batter hit probability",
            "EWM of the 1+ hit indicator &mdash; bet &ldquo;yes&rdquo; only at a better price than break-even",
            bcols + [("P(hit)", True), ("Break-even", True)] + n,
            batter_rows(top_batters_by_hit_prob(result, TOP_N), lambda v: f"{v:.1%}", lambda e: e.projection.hit_prob, with_break_even=True),
        )
        + _props_card(
            "Batter home-run probability",
            "EWM of the 1+ HR indicator &mdash; bet &ldquo;yes&rdquo; only at a better price than break-even",
            bcols + [("P(HR)", True), ("Break-even", True)] + n,
            batter_rows(top_batters_by_hr_prob(result, TOP_N), lambda v: f"{v:.1%}", lambda e: e.projection.hr_prob, with_break_even=True),
        )
        + _props_card(
            "Batter total bases", "EWM of per-game total bases",
            bcols + [("Proj. TB", True)] + n,
            batter_rows(top_batters_by_total_bases(result, TOP_N), lambda v: f"{v:.2f}", lambda e: e.projection.projected_total_bases),
        )
        + "</div>"
    )


def _render_methodology() -> str:
    """The model's actual math, rendered as native MathML."""
    kelly_denom = int(round(1 / KELLY_FRACTION))
    blocks = [
        (
            "Win-probability model",
            """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <mi>P</mi><mo>(</mo><mtext>home</mtext><mo>)</mo><mo>=</mo>
            <mi>g</mi><mo>(</mo><msub><mi>F</mi><mtext>GBM</mtext></msub>
            <mo>(</mo><mi mathvariant="bold">x</mi><mo>)</mo><mo>)</mo><mo>,</mo>
            <mspace width="0.5em"/>
            <msub><mi>F</mi><mtext>GBM</mtext></msub><mo>(</mo><mi mathvariant="bold">x</mi><mo>)</mo>
            <mo>=</mo><munderover><mo>&sum;</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>T</mi></munderover>
            <mi>&eta;</mi><mo>&sdot;</mo><msub><mi>f</mi><mi>t</mi></msub><mo>(</mo><mi mathvariant="bold">x</mi><mo>)</mo>
            </mrow></math>""",
            "Gradient-boosted trees (LightGBM, Optuna-tuned) over ~25 leak-free features -- "
            "Elo ratings, rolling form at 5/15/30-game windows, starting-pitcher FIP/WHIP/K%/BB%, "
            "team OPS/ISO/BABIP, Statcast xwOBA-on-contact/barrel/hard-hit rates, park factor, "
            "weather, umpire tendency -- with isotonic calibration g(&middot;) so the output "
            "probabilities are honest, and recent seasons weighted more heavily in training.",
        ),
        (
            "Implied probability of an American line",
            """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <msub><mi>p</mi><mtext>imp</mtext></msub><mo>=</mo>
            <mo>{</mo><mtable columnalign="left">
            <mtr><mtd><mfrac><mn>100</mn><mrow><mi>o</mi><mo>+</mo><mn>100</mn></mrow></mfrac></mtd>
            <mtd><mi>o</mi><mo>&gt;</mo><mn>0</mn></mtd></mtr>
            <mtr><mtd><mfrac><mrow><mo>|</mo><mi>o</mi><mo>|</mo></mrow>
            <mrow><mo>|</mo><mi>o</mi><mo>|</mo><mo>+</mo><mn>100</mn></mrow></mfrac></mtd>
            <mtd><mi>o</mi><mo>&lt;</mo><mn>0</mn></mtd></mtr>
            </mtable></mrow></math>""",
            "Raw implied probabilities include the bookmaker's margin (vig), so the two sides sum past 1.",
        ),
        (
            "Proportional devig",
            """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <msub><mi>p</mi><mtext>fair</mtext></msub><mo>=</mo>
            <mfrac><msub><mi>p</mi><mi>H</mi></msub>
            <mrow><msub><mi>p</mi><mi>H</mi></msub><mo>+</mo><msub><mi>p</mi><mi>A</mi></msub></mrow></mfrac>
            </mrow></math>""",
            "Normalizes both sides' implied probabilities to sum to 1, then averages across surveyed books "
            "for a consensus market probability.",
        ),
        (
            "Edge",
            """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <mi>&epsilon;</mi><mo>=</mo>
            <msub><mi>p</mi><mtext>model</mtext></msub><mo>&minus;</mo>
            <msub><mi>p</mi><mtext>fair</mtext></msub>
            </mrow></math>""",
            "The model&ndash;market disagreement on the picked side; picks are ranked by &epsilon; descending.",
        ),
        (
            f"Fractional Kelly stake (1/{kelly_denom})",
            f"""<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <msup><mi>f</mi><mo>*</mo></msup><mo>=</mo>
            <mfrac><mrow><mi>b</mi><mi>p</mi><mo>&minus;</mo><mi>q</mi></mrow><mi>b</mi></mfrac>
            <mo>,</mo><mspace width="0.6em"></mspace>
            <mi>f</mi><mo>=</mo><mfrac><msup><mi>f</mi><mo>*</mo></msup><mn>{kelly_denom}</mn></mfrac>
            </mrow></math>""",
            "b = decimal odds &minus; 1, p = model win probability, q = 1 &minus; p. "
            f"Staking f*/{kelly_denom} trades growth for drawdown control.",
        ),
        (
            "Prop projection (EWM)",
            """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <msub><mover><mi>y</mi><mo>^</mo></mover><mi>t</mi></msub><mo>=</mo>
            <mfrac>
            <mrow><munderover><mo>&sum;</mo><mrow><mi>i</mi><mo>=</mo><mn>0</mn></mrow><mi>n</mi></munderover>
            <msup><mrow><mo>(</mo><mn>1</mn><mo>&minus;</mo><mi>&alpha;</mi><mo>)</mo></mrow><mi>i</mi></msup>
            <msub><mi>y</mi><mrow><mi>t</mi><mo>&minus;</mo><mi>i</mi></mrow></msub></mrow>
            <mrow><munderover><mo>&sum;</mo><mrow><mi>i</mi><mo>=</mo><mn>0</mn></mrow><mi>n</mi></munderover>
            <msup><mrow><mo>(</mo><mn>1</mn><mo>&minus;</mo><mi>&alpha;</mi><mo>)</mo></mrow><mi>i</mi></msup></mrow>
            </mfrac>
            <mo>,</mo><mspace width="0.6em"></mspace>
            <mi>&alpha;</mi><mo>=</mo><mfrac><mn>2</mn><mrow><mi>s</mi><mo>+</mo><mn>1</mn></mrow></mfrac>
            </mrow></math>""",
            "Exponentially-weighted mean of each player's own recent game log "
            "(span s = 10 for batters, 6 for pitchers) &mdash; recent form dominates.",
        ),
    ]
    cards = "".join(
        f'<div class="fx"><h3>{title}</h3>{mathml}<p>{desc}</p></div>' for title, mathml, desc in blocks
    )
    return (
        '<div class="card"><h2>Methodology</h2>'
        '<p class="sub">Every number above is produced by these six expressions.</p>'
        f'<div class="math-grid">{cards}</div></div>'
    )


def _wrap_with_password_gate(body_html: str, password_hash: str) -> str:
    """Hide `body_html` behind a client-side password prompt.

    NOT real security: the protected content is still present verbatim in the
    page source (just CSS-hidden), so anyone who views source bypasses this
    entirely. It only keeps casual visitors from landing on the content directly.
    Unlock state is remembered per-browser-tab via sessionStorage.
    """
    return f"""
<div id="gate">
  <div class="panel">
    <div class="mark">SBA</div>
    <h2>Private report</h2>
    <p class="hint">Enter the password to view today's model output.</p>
    <form id="gate-form">
      <input type="password" id="gate-input" autocomplete="off" autofocus>
      <button type="submit">Unlock</button>
      <p id="gate-error">Incorrect password.</p>
    </form>
  </div>
</div>
<div id="protected" style="display:none">
{body_html}
</div>
<script>
(function () {{
  var HASH = {password_hash!r};
  function toHex(buf) {{
    return Array.from(new Uint8Array(buf)).map(function (b) {{ return b.toString(16).padStart(2, "0"); }}).join("");
  }}
  function unlock() {{
    document.getElementById("gate").style.display = "none";
    document.getElementById("protected").style.display = "block";
  }}
  if (sessionStorage.getItem("sba_unlocked") === "1") {{ unlock(); }}
  document.getElementById("gate-form").addEventListener("submit", function (e) {{
    e.preventDefault();
    var value = document.getElementById("gate-input").value;
    crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)).then(function (buf) {{
      if (toHex(buf) === HASH) {{
        sessionStorage.setItem("sba_unlocked", "1");
        unlock();
      }} else {{
        document.getElementById("gate-error").style.display = "block";
      }}
    }});
  }});
}})();
</script>
"""


def generate_report(output_path: Path) -> None:
    scan_result = scan_today()

    picks: list[Pick] | None = None
    picks_error: str | None = None
    try:
        picks = generate_picks(history_seasons=DEFAULT_SEASONS)
        log_picks(picks)  # so a later run can grade today's picks against results
    except (OddsAPIError, FileNotFoundError) as e:
        picks_error = str(e)

    # Grade previously logged picks against final scores. Degrades to omitting the
    # results section (no log yet, no finished games) rather than failing the report.
    try:
        graded = grade_picks(CURRENT_YEAR)
    except Exception:
        graded = pd.DataFrame()

    # Today's full slate comes from the log (started/finished games drop out of the
    # live odds feed); fall back to the fresh picks when there's no log to read.
    try:
        today_df = todays_picks_from_log()
    except Exception:
        today_df = pd.DataFrame()
    if today_df.empty and picks:
        today_df = _picks_to_frame(picks)

    try:
        yesterday_df = todays_picks_from_log(day_offset=-1)
    except Exception:
        yesterday_df = pd.DataFrame()

    try:
        tomorrow_df = _picks_to_frame(generate_picks(history_seasons=DEFAULT_SEASONS, days_ahead=1))
    except Exception:
        tomorrow_df = pd.DataFrame()

    try:
        history_index = _build_history_index()
    except Exception:
        history_index = {}

    now_for_status = datetime.now(timezone.utc)

    now_utc = datetime.now(timezone.utc)
    generated_at = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    dateline = now_utc.astimezone(EASTERN).strftime("%A, %B %-d, %Y")

    body_content = f"""
<div class="wrap">
<div class="masthead">
  <div class="brand">
    <div class="mark">SBA</div>
    <div><span class="name">Sports Betting Analytics</span><span class="tag">Quantitative MLB Model</span></div>
  </div>
  <div class="run-pill"><span class="dot"></span>Model run {generated_at}</div>
</div>
<h1>Daily edge report</h1>
<p class="dateline">{dateline} &middot; snapshot of the last scheduled run</p>
{_render_tiles(scan_result, picks)}
{_render_day_tabs(yesterday_df, today_df, tomorrow_df, picks_error, now_for_status)}
{_render_results_section(graded)}
{_render_history_browser(history_index, _build_edge_history_index(graded))}
{_render_props_section(scan_result)}
{_render_methodology()}
<footer>
  <span>Calibrated gradient-boosted moneyline model &middot; EWM prop projections &middot; consensus devigged across surveyed books</span>
  <span>SBA &middot; {now_utc.year}</span>
</footer>
</div>
"""

    password = os.environ.get(PAGE_PASSWORD_ENV)
    if password:
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        body_content = _wrap_with_password_gate(body_content, password_hash)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SBA &middot; Daily edge report</title>
<style>{PAGE_STYLE}</style>
</head>
<body>
{body_content}
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
