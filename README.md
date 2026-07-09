# sba — MLB Betting Analysis Agent

A research tool, not an autonomous bot: it trains a win-probability model on historical
MLB results, compares its predictions to live sportsbook odds, and surfaces ranked
+EV picks for you to act on manually. It does not place bets.

> **Disclaimer**: This is an informational/statistical tool, not financial advice.
> Sports betting is illegal or restricted in many jurisdictions — confirm it's legal
> where you are before using this for real wagers. Gamble responsibly; never bet more
> than you can afford to lose.

## How it works

1. **Historical data** comes from [pybaseball](https://github.com/jldbc/pybaseball)
   (free, no key) — team-level game results scraped from Baseball-Reference.
2. **Features** are simple rolling team-form stats computed strictly from games
   *before* the one being predicted (win %, run differential, rest days) — no leakage.
3. **Model** is a logistic regression predicting home-team win probability, trained on
   past seasons and evaluated on a held-out season.
4. **Live odds** come from [The Odds API](https://the-odds-api.com/) (free tier: 25
   requests/day, MLB moneylines). Odds are "devigged" (vig removed) to get the market's
   fair implied probability.
5. **Picks** are ranked by edge = model probability − market probability, with a
   conservative quarter-Kelly suggested stake size (informational only).

## Player props

Player prop projections (pitcher strikeouts, batter hit/HR probability, total bases)
work differently from the moneyline picks above:

- **No live odds comparison.** Player props aren't on The Odds API's free tier (that's
  a paid Business-tier feature). These commands print a projection you compare manually
  against whatever line you're looking at — they don't fetch or devig prop odds.
- **Not a fitted model.** Each command scrapes the specific player's own recent game log
  directly from Baseball-Reference (`data/bref_players.py`, a small from-scratch scraper —
  separate from the pybaseball-based team data above) and projects via an
  exponentially-weighted moving average over their recent games/starts. It's a transparent
  recency-weighted average, not a trained model with opponent adjustment.
- Requests to Baseball-Reference sleep 3s between calls (per their `robots.txt`
  `Crawl-delay`) and are cached locally per player/season, so repeat lookups are instant.

```bash
uv run sba prop-batter "Aaron Judge"
uv run sba prop-batter colege01 --games 15      # a Baseball-Reference ID also works
uv run sba prop-pitcher "Gerrit Cole" --starts 10
```

If a player name is ambiguous or not found, the error message tells you to look up their
Baseball-Reference ID manually (from the URL on their player page) and pass that instead.

### Daily scan across the full slate

`sba props-today` scans every game on today's MLB slate and ranks the strongest prop
projections, instead of looking up one player at a time:

```bash
uv run sba props-today
uv run sba props-today --top 15 --min-pa 30   # show more players, raise the activity bar
```

How it finds "today's players", entirely from Baseball-Reference (no odds API call):
- `https://www.baseball-reference.com/previews/index.shtml` lists today's games and
  probable starting pitchers, with direct links to their player pages.
- Each game's own preview page lists each team's active hitters with recent playing
  time (`PA Last 28d`). `--min-pa` filters this down to regulars.

Two things worth knowing:
- **This is a hitting pool, not a confirmed lineup.** Baseball-Reference doesn't
  publish confirmed pregame batting orders — the `--min-pa` filter is the closest
  available proxy for "who's likely to play," and can include a bench player or miss
  a last-minute lineup change.
- **It's slow on a cold cache.** A full slate is on the order of ~15 preview pages +
  ~30 pitchers + up to a few hundred batters (before filtering), each request 3
  seconds apart per Baseball-Reference's `robots.txt`. Expect 15-30+ minutes the first
  time; player game logs are cached for `CACHE_TTL_HOURS` (20h), so same-day re-runs
  and future days are much faster. It's meant to be run once a day, not interactively.
- Same as the single-player commands: rankings are by projection strength only, not
  compared to any live line (player props aren't on the free Odds API tier).

### A note on backtesting

The free odds tier has no historical archive (that's a paid feature), so there's no
way to backtest *betting ROI* against real historical lines here. `sba backtest`
instead validates the *model itself* — accuracy, log loss, and calibration against
real game outcomes on a held-out season. To build a real track record, `sba picks`
logs every live pick it generates to `logs/picks.csv`, and `sba grade` fills in actual
outcomes once games finish — so you accumulate genuine forward performance over time
instead of relying on paid historical data.

## Setup

```bash
uv sync
cp .env.example .env
# Get a free key at https://the-odds-api.com/ and put it in .env as ODDS_API_KEY=...
```

## Usage

```bash
# Fetch & cache historical game results (defaults to the last 6 seasons)
uv run sba fetch-data

# Train the model on cached data and save it to models/
uv run sba train

# Evaluate model accuracy/calibration on a held-out season
uv run sba backtest --test-season 2025

# Generate today's ranked picks from live odds (requires ODDS_API_KEY)
uv run sba picks

# Fill in real outcomes for previously logged picks and see your hit rate
uv run sba grade
```

Run `uv run sba --help` or `uv run sba <command> --help` for all options.

## Publish to GitHub Pages

CLI output only goes so far if you'd rather check a web page than run commands. There's
a `sba report` command plus a GitHub Actions workflow (`.github/workflows/publish.yml`)
that runs it on a schedule and publishes the result as a static site.

**Important**: GitHub Pages only serves static files — it can't run this project's
Python backend live. The published page shows the results of the *last scheduled run*,
regenerated fresh each time (default: daily at 11:00 UTC), not a live/on-demand query.

Setup (one-time):
1. Push this repo to GitHub (must be a **public** repo for free GitHub Pages).
2. Add your Odds API key as a secret: repo **Settings → Secrets and variables →
   Actions → New repository secret**, name `ODDS_API_KEY`. Without it, the moneyline
   picks section just shows "unavailable" — the props section doesn't need it.
3. (Optional) Add a `PAGE_PASSWORD` secret the same way to put a password prompt in
   front of the page — see "Password-gating the page" below before relying on this.
4. Trigger the workflow once: **Actions** tab → "Publish MLB report to GitHub Pages" →
   **Run workflow**. Its first successful run creates a `gh-pages` branch.
5. Enable Pages: repo **Settings → Pages → Build and deployment → Source: "Deploy from
   a branch"** → branch `gh-pages`, folder `/ (root)`. (The `gh-pages` branch won't be
   selectable here until step 4 has completed successfully at least once.)
6. After that, it's automatic — the workflow re-runs daily (default: 11:00 UTC) and
   re-publishes to `gh-pages` each time, or trigger it manually anytime from the
   Actions tab.

### Password-gating the page

Setting a `PAGE_PASSWORD` secret puts a password prompt in front of the page's
content. Be clear about what this is and isn't: it's a **client-side visibility gate,
not real security**. The protected content is still present verbatim in the page's
HTML source (just hidden with CSS until unlocked) — anyone who views source, or reads
the raw file on the `gh-pages` branch, sees everything regardless of the password. It
only keeps casual visitors from landing directly on the content. If you need real
access control, that means either a private repo (GitHub Pro+, restricted to invited
GitHub accounts, not a shared password) or hosting elsewhere with actual server-side
auth — both are bigger changes than this project takes on.

Once it's run successfully, the page is live at `https://{your-username}.github.io/{repo-name}/`.

Expect each run to take roughly 20-40 minutes — it's doing the same full slate scan as
`sba props-today` (see the timing note above), just on a schedule instead of by hand.

## Project layout

```
src/sba/
  config.py          env/paths
  data/
    mlb_stats.py       pybaseball fetch + local parquet cache (team-level)
    odds.py             The Odds API client + devig math
    bref_http.py         shared rate-limited HTTP helper for the BR scrapers below
    bref_players.py       from-scratch Baseball-Reference scraper (player game logs)
    bref_slate.py           from-scratch scraper for today's games/pitchers/lineups
  features.py           rolling team-form feature engineering (leakage-free)
  model.py               train/save/load the win-probability model
  backtest.py             season-holdout accuracy/calibration evaluation
  picks.py                 model vs. market -> ranked +EV picks
  props.py                  EWM-based player prop projections (single player)
  daily_scan.py               scan today's full slate + rank prop projections
  report.py                     render the props scan + picks as a static HTML page
  tracking.py                     log picks, grade them against real outcomes
  cli.py                            `sba` command entrypoint
.github/workflows/publish.yml  scheduled CI job: report.py -> GitHub Pages
```

## Extending to other sports

The pattern is: swap `data/mlb_stats.py` for a sport-specific stats source, adjust
`features.py` for that sport's meaningful stats, and point `data/odds.py` at the
matching `sport_key` on The Odds API (e.g. `basketball_nba`, `americanfootball_nfl`,
`soccer_epl`). Everything downstream (model, backtest, picks, tracking, CLI) is
sport-agnostic and should work unchanged.
