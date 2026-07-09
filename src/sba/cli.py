"""CLI entrypoint: `sba fetch-data|train|backtest|picks|grade`."""

from __future__ import annotations

from pathlib import Path

import requests
import typer
from rich.console import Console
from rich.table import Table

from sba.backtest import run_backtest
from sba.config import CURRENT_YEAR, DEFAULT_SEASONS
from sba.daily_scan import (
    DEFAULT_MIN_BATTER_PA_LAST_28D,
    scan_today,
    top_batters_by_hit_prob,
    top_batters_by_hr_prob,
    top_batters_by_total_bases,
    top_pitchers_by_strikeouts,
)
from sba.data.bref_players import PlayerLookupError
from sba.data.mlb_stats import fetch_seasons
from sba.data.odds import OddsAPIError
from sba.features import build_features
from sba.model import save
from sba.model import train as train_model
from sba.picks import generate_picks
from sba.props import project_batter, project_pitcher
from sba.report import generate_report
from sba.tracking import grade_picks, log_picks

app = typer.Typer(help="MLB betting analysis agent -- informational picks only, not financial advice.")
console = Console()


@app.command("fetch-data")
def fetch_data_cmd(
    seasons: list[int] = typer.Option(DEFAULT_SEASONS, help="Seasons to fetch/cache."),
    force_refresh: bool = typer.Option(False, help="Re-download even if already cached."),
) -> None:
    """Fetch and cache historical MLB game results."""
    games = fetch_seasons(seasons, force_refresh=force_refresh)
    console.print(f"Cached {len(games)} games across seasons {sorted(games['season'].unique().tolist())}")


@app.command()
def train(seasons: list[int] = typer.Option(DEFAULT_SEASONS, help="Seasons to train on.")) -> None:
    """Train the win-probability model on all cached seasons and save it."""
    games = fetch_seasons(seasons)
    features = build_features(games)
    pipeline = train_model(features)
    save(pipeline)
    console.print(f"Trained on {len(features)} games ({sorted(features['season'].unique().tolist())}). Model saved.")


@app.command()
def backtest(
    test_season: int = typer.Option(CURRENT_YEAR - 1, help="Season to hold out for evaluation."),
    seasons: list[int] = typer.Option(DEFAULT_SEASONS, help="Seasons to pull data from."),
) -> None:
    """Evaluate model accuracy/calibration on a held-out season (not a betting ROI backtest)."""
    games = fetch_seasons(seasons)
    result = run_backtest(games, test_season)
    console.print(f"Train seasons: {result.train_seasons} ({result.n_train} games)")
    console.print(f"Test season: {result.test_season} ({result.n_test} games)")
    console.print(f"Accuracy: {result.accuracy:.3f}  Log loss: {result.log_loss:.3f}  Brier: {result.brier_score:.3f}")

    table = Table(title="Calibration (predicted vs. actual home-win rate, by decile)")
    for col in result.calibration.columns:
        table.add_column(str(col))
    for _, row in result.calibration.iterrows():
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)


@app.command()
def picks(
    seasons: list[int] = typer.Option(DEFAULT_SEASONS, help="Seasons of history to use for team form."),
    log: bool = typer.Option(True, help="Log picks to logs/picks.csv for later grading."),
) -> None:
    """Generate today's ranked +EV picks by comparing the model to live odds."""
    try:
        generated = generate_picks(history_seasons=seasons)
    except OddsAPIError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]\nRun `sba train` first.")
        raise typer.Exit(1) from e
    if not generated:
        console.print("No picks generated (no games today, or odds/team data unavailable).")
        raise typer.Exit()

    table = Table(title="Today's MLB picks -- informational only, not financial advice")
    for col in ["Time (UTC)", "Matchup", "Pick", "Price", "Model %", "Market %", "Edge", "Kelly stake %"]:
        table.add_column(col)
    for p in generated:
        matchup = f"{p.away_team} @ {p.home_team}"
        pick_team = p.home_team if p.side == "home" else p.away_team
        table.add_row(
            p.commence_time,
            matchup,
            pick_team,
            f"{p.side_price:+.0f}",
            f"{p.side_model_prob:.1%}",
            f"{p.side_market_prob:.1%}",
            f"{p.edge:+.1%}",
            f"{p.suggested_stake_pct:.1%}",
        )
    console.print(table)

    if log:
        log_picks(generated)
        console.print(f"Logged {len(generated)} picks to logs/picks.csv")


@app.command()
def grade(season: int = typer.Option(CURRENT_YEAR, help="Season to grade picks against.")) -> None:
    """Fill in real outcomes for previously logged picks and report a running record."""
    try:
        graded = grade_picks(season)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    if graded.empty:
        console.print("No graded picks yet (either nothing logged, or none of the logged games have finished).")
        raise typer.Exit()

    hit_rate = graded["won"].mean()
    console.print(f"{len(graded)} graded picks -- hit rate: {hit_rate:.1%}")


@app.command("prop-batter")
def prop_batter_cmd(
    player: str = typer.Argument(..., help="Player name (e.g. 'Aaron Judge') or Baseball-Reference ID."),
    season: int | None = typer.Option(None, help="Season to pull game logs from (defaults to current year)."),
    games: int = typer.Option(20, help="How many recent games to fetch (falls back to prior season if too few)."),
) -> None:
    """Project a batter's hit/HR probability and total bases from recent form (no live odds)."""
    try:
        proj = project_batter(player, season=season, games=games)
    except (PlayerLookupError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[bold]{player}[/bold] ({proj.player_id}) -- last {proj.n_games} games, season {proj.season}")
    console.print(f"Hit probability: {proj.hit_prob:.1%}")
    console.print(f"Projected total bases: {proj.projected_total_bases:.2f}")
    console.print(f"HR probability: {proj.hr_prob:.1%}")
    console.print(
        "[dim]Recency-weighted average of the player's own recent games -- "
        "not a fitted model, not compared to live odds.[/dim]"
    )


@app.command("prop-pitcher")
def prop_pitcher_cmd(
    player: str = typer.Argument(..., help="Player name (e.g. 'Gerrit Cole') or Baseball-Reference ID."),
    season: int | None = typer.Option(None, help="Season to pull game logs from (defaults to current year)."),
    starts: int = typer.Option(12, help="How many recent appearances to fetch (falls back to prior season if too few)."),
) -> None:
    """Project a pitcher's strikeouts from recent form (no live odds)."""
    try:
        proj = project_pitcher(player, season=season, starts=starts)
    except (PlayerLookupError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    console.print(
        f"[bold]{player}[/bold] ({proj.player_id}) -- last {proj.n_appearances} appearances, season {proj.season}"
    )
    console.print(f"Projected strikeouts: {proj.projected_strikeouts:.2f}")
    console.print(
        "[dim]Recency-weighted average of the pitcher's own recent appearances "
        "(starts and relief outings alike) -- not a fitted model, not compared to live odds.[/dim]"
    )


@app.command("props-today")
def props_today_cmd(
    top: int = typer.Option(10, help="How many players to show per ranking."),
    min_pa: int = typer.Option(DEFAULT_MIN_BATTER_PA_LAST_28D, help="Minimum PA in the last 28 days for a batter to be considered."),
) -> None:
    """Scan today's full MLB slate and rank the strongest prop projections.

    No live odds are used (player props aren't on the free Odds API tier) -- rankings
    are by projection strength, not market edge. Batters come from each team's active
    hitting pool filtered by recent playing time, not a confirmed lineup (Baseball-
    Reference has no pregame source for that). This scrapes every player on the slate
    and can take a while on a cold cache -- see the README.
    """
    try:
        with console.status("Starting scan...") as status:
            result = scan_today(min_batter_pa_last_28d=min_pa, on_progress=status.update)
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Couldn't reach Baseball-Reference: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(
        f"\n{result.n_games} games, {result.n_players_considered} players considered, "
        f"{result.n_errors} players skipped (insufficient history or a parse error), "
        f"{result.n_games_skipped} games skipped (couldn't fetch preview page).\n"
    )

    pitcher_table = Table(title=f"Top {top} pitcher strikeout projections")
    for col in ["Pitcher", "Team", "Opp", "Proj. Ks", "Appearances"]:
        pitcher_table.add_column(col)
    for e in top_pitchers_by_strikeouts(result, top):
        pitcher_table.add_row(e.name, e.team, e.opponent, f"{e.projection.projected_strikeouts:.2f}", str(e.projection.n_appearances))
    console.print(pitcher_table)

    hit_table = Table(title=f"Top {top} batter hit-probability projections")
    for col in ["Batter", "Team", "Opp", "Hit %", "Games"]:
        hit_table.add_column(col)
    for e in top_batters_by_hit_prob(result, top):
        hit_table.add_row(e.name, e.team, e.opponent, f"{e.projection.hit_prob:.1%}", str(e.projection.n_games))
    console.print(hit_table)

    hr_table = Table(title=f"Top {top} batter HR-probability projections")
    for col in ["Batter", "Team", "Opp", "HR %", "Games"]:
        hr_table.add_column(col)
    for e in top_batters_by_hr_prob(result, top):
        hr_table.add_row(e.name, e.team, e.opponent, f"{e.projection.hr_prob:.1%}", str(e.projection.n_games))
    console.print(hr_table)

    tb_table = Table(title=f"Top {top} batter total-bases projections")
    for col in ["Batter", "Team", "Opp", "Proj. TB", "Games"]:
        tb_table.add_column(col)
    for e in top_batters_by_total_bases(result, top):
        tb_table.add_row(e.name, e.team, e.opponent, f"{e.projection.projected_total_bases:.2f}", str(e.projection.n_games))
    console.print(tb_table)

    console.print(
        "[dim]Recency-weighted projections, not compared to any live line -- player props "
        "aren't on the free Odds API tier. Batters are each team's recently active hitting "
        "pool, not a confirmed lineup.[/dim]"
    )


@app.command()
def report(
    output: Path = typer.Option(Path("public/index.html"), help="Where to write the static HTML report."),
) -> None:
    """Generate a static HTML report (moneyline picks + today's prop scan) for publishing.

    Runs the same scan as `props-today` plus `picks`, and renders both to a single
    self-contained HTML file -- meant for a scheduled CI job that publishes it to
    GitHub Pages (see .github/workflows/publish.yml and the README). A missing
    ODDS_API_KEY or untrained model degrades the picks section instead of failing
    the whole report, since this is meant to run unattended.
    """
    with console.status("Generating report..."):
        generate_report(output)
    console.print(f"Wrote report to {output}")


if __name__ == "__main__":
    app()
