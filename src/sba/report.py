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
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sba.config import DEFAULT_SEASONS
from sba.daily_scan import (
    DailyScanResult,
    scan_today,
    top_batters_by_hit_prob,
    top_batters_by_hr_prob,
    top_batters_by_total_bases,
    top_pitchers_by_strikeouts,
)
from sba.data.odds import OddsAPIError, prob_to_american
from sba.picks import KELLY_FRACTION, Pick, generate_picks

TOP_N = 10
PAGE_PASSWORD_ENV = "PAGE_PASSWORD"
EASTERN = ZoneInfo("America/New_York")

PAGE_STYLE = """
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --hairline: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --accent: #2a78d6; --accent-soft: #cde2fb;
  --good: #006300; --bad: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --hairline: #2c2c2a; --ring: rgba(255,255,255,0.10);
    --accent: #3987e5; --accent-soft: #104281;
    --good: #0ca30c; --bad: #d03b3b;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.5; -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
.eyebrow {
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--accent); margin: 0 0 0.4rem;
}
h1 { font-size: 1.9rem; margin: 0 0 0.25rem; letter-spacing: -0.02em; }
.stamp { color: var(--muted); font-size: 0.85rem; margin: 0 0 2rem; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin-bottom: 2.25rem; }
.tile {
  background: var(--surface); border: 1px solid var(--ring); border-radius: 10px;
  padding: 0.9rem 1rem 0.8rem;
}
.tile .k { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.tile .v { font-size: 1.75rem; font-weight: 650; letter-spacing: -0.02em; margin-top: 0.1rem; }
.tile .v small { font-size: 0.95rem; font-weight: 500; color: var(--ink-2); }
.card {
  background: var(--surface); border: 1px solid var(--ring); border-radius: 12px;
  padding: 1.25rem 1.4rem 1.1rem; margin-bottom: 1.5rem;
}
.card > h2 { font-size: 1.05rem; margin: 0 0 0.15rem; letter-spacing: -0.01em; }
.card > .sub { color: var(--muted); font-size: 0.82rem; margin: 0 0 0.9rem; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem; }
@media (max-width: 520px) { .grid2 { grid-template-columns: 1fr; } }
.grid2 .card { margin-bottom: 0; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th {
  text-align: left; color: var(--muted); font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 0.35rem 0.75rem 0.45rem; border-bottom: 1px solid var(--hairline);
  white-space: nowrap;
}
td { padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--hairline); white-space: nowrap; vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: color-mix(in srgb, var(--accent) 5%, transparent); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.strong { font-weight: 600; }
.pos { color: var(--good); font-weight: 600; }
.neg { color: var(--bad); font-weight: 600; }
.chip {
  display: inline-block; padding: 0.05rem 0.5rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 600; border: 1px solid var(--ring);
  background: color-mix(in srgb, var(--accent) 12%, transparent); color: var(--ink);
}
.bar { display: inline-block; vertical-align: middle; height: 4px; border-radius: 2px; background: var(--accent); margin-left: 0.55rem; }
.unavailable {
  border: 1px dashed var(--hairline); border-radius: 8px; color: var(--ink-2);
  padding: 0.8rem 1rem; font-size: 0.88rem;
}
.math-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
.fx {
  border: 1px solid var(--ring); border-radius: 10px; padding: 0.9rem 1rem;
}
.fx h3 { font-size: 0.82rem; margin: 0 0 0.5rem; color: var(--ink-2); font-weight: 600; }
.fx math { display: block; font-size: 1.05rem; margin: 0.25rem 0 0.55rem; }
.fx p { font-size: 0.8rem; color: var(--muted); margin: 0; }
footer { margin-top: 2.5rem; color: var(--muted); font-size: 0.8rem; }
#gate { max-width: 340px; margin: 22vh auto 0; text-align: center; padding: 0 1.25rem; }
#gate h2 { letter-spacing: -0.01em; }
#gate input {
  display: block; width: 100%; padding: 0.6rem 0.75rem; margin: 0.9rem 0;
  font-size: 1rem; border: 1px solid var(--ring); border-radius: 8px;
  background: var(--surface); color: var(--ink);
}
#gate button {
  padding: 0.55rem 1.6rem; font-size: 0.95rem; font-weight: 600; cursor: pointer;
  border: none; border-radius: 8px; background: var(--accent); color: #fff;
}
#gate-error { color: var(--bad); display: none; font-size: 0.85rem; }
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


def _render_tiles(result: DailyScanResult, picks: list[Pick] | None) -> str:
    tiles = [
        ("Games today", f"{result.n_games}"),
        ("Players modeled", f"{result.n_players_considered - result.n_errors}"),
        ("Starting pitchers", f"{len(result.pitchers)}"),
        ("Batters in pool", f"{len(result.batters)}"),
    ]
    if picks:
        best = max(picks, key=lambda p: p.edge)
        tiles.append(("Best model edge", f"{best.edge:+.1%} <small>{_esc(best.away_team)}@{_esc(best.home_team)}</small>"))
    cells = "".join(f'<div class="tile"><div class="k">{_esc(k)}</div><div class="v">{v}</div></div>' for k, v in tiles)
    return f'<div class="tiles">{cells}</div>'


def _render_picks_section(picks: list[Pick] | None, error: str | None) -> str:
    header = (
        '<h2>Moneyline &mdash; model vs. market</h2>'
        '<p class="sub">Fair line is the no-vig price implied by the model probability; '
        'book line is the best price currently posted across surveyed sportsbooks. '
        'Edge is the model&ndash;market probability gap on the picked side.</p>'
    )
    if error is not None:
        return f'<div class="card">{header}<p class="unavailable">Picks unavailable: {_esc(error)}</p></div>'

    headers = [
        ("Time", False), ("Matchup", False), ("Pick", False),
        ("Book line", True), ("Fair line", True), ("Line gap", True),
        ("Model", True), ("Market", True), ("Edge", True),
        ("¼-Kelly", True), ("Books", True),
    ]
    max_edge = max((p.edge for p in picks), default=0)
    rows = []
    for p in sorted(picks, key=lambda x: x.edge, reverse=True):
        pick_team = p.home_team if p.side == "home" else p.away_team
        fair = prob_to_american(p.side_model_prob)
        gap = p.side_price - fair  # positive: book pays better than the model's fair price
        gap_cls = "pos" if gap >= 0 else "neg"
        rows.append([
            f'<td>{_esc(_fmt_time_et(p.commence_time))}</td>',
            f'<td>{_esc(p.away_team)} @ {_esc(p.home_team)}</td>',
            f'<td class="strong"><span class="chip">{_esc(pick_team)} {_esc(p.side)}</span></td>',
            f'<td class="num strong">{_fmt_line(p.side_price)}</td>',
            f'<td class="num">{_fmt_line(fair)}</td>',
            f'<td class="num {gap_cls}">{gap:+.0f}</td>',
            f'<td class="num">{p.side_model_prob:.1%}</td>',
            f'<td class="num">{p.side_market_prob:.1%}</td>',
            f'<td class="num"><span class="{"pos" if p.edge >= 0 else "neg"}">{p.edge:+.1%}</span>{_bar(max(p.edge, 0), max_edge)}</td>',
            f'<td class="num">{p.suggested_stake_pct:.1%}</td>',
            f'<td class="num">{p.n_books}</td>',
        ])
    return f'<div class="card">{header}{_table(headers, rows)}</div>'


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
                f"<td>{_esc(e.team)}</td>",
                f"<td>{_esc(e.opponent)}</td>",
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
            f"<td>{_esc(e.team)}</td>",
            f"<td>{_esc(e.opponent)}</td>",
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
            <mi>&sigma;</mi><mo>(</mo><msup><mi mathvariant="bold">&beta;</mi><mi>T</mi></msup>
            <mi mathvariant="bold">x</mi><mo>)</mo><mo>=</mo>
            <mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo>
            <msup><mi>e</mi><mrow><mo>&minus;</mo><msup><mi mathvariant="bold">&beta;</mi><mi>T</mi></msup>
            <mi mathvariant="bold">x</mi></mrow></msup></mrow></mfrac>
            </mrow></math>""",
            "Logistic regression on standardized form differentials "
            "x = [&Delta;win%, &Delta;run-diff, &Delta;rest], fit on six seasons of games.",
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
  <h2>Enter password</h2>
  <form id="gate-form">
    <input type="password" id="gate-input" autocomplete="off" autofocus>
    <button type="submit">Enter</button>
    <p id="gate-error">Incorrect password.</p>
  </form>
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
    except (OddsAPIError, FileNotFoundError) as e:
        picks_error = str(e)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body_content = f"""
<div class="wrap">
<p class="eyebrow">SBA &middot; Quantitative MLB Model</p>
<h1>Daily edge report</h1>
<p class="stamp">Generated {generated_at} &middot; snapshot of the last scheduled run</p>
{_render_tiles(scan_result, picks)}
{_render_picks_section(picks, picks_error)}
{_render_props_section(scan_result)}
{_render_methodology()}
<footer>Logistic moneyline model &middot; EWM prop projections &middot; consensus devigged across surveyed books</footer>
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
