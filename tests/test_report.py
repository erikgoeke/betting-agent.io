import hashlib

import pandas as pd
import pytest

from sba import daily_scan, report
from sba.data.odds import OddsAPIError
from sba.picks import Pick
from sba.props import BatterProjection, PitcherProjection


@pytest.fixture(autouse=True)
def _stub_tracking(monkeypatch):
    """Keep report tests from writing to / reading the real picks log."""
    monkeypatch.setattr(report, "log_picks", lambda picks: None)
    monkeypatch.setattr(report, "grade_picks", lambda season: (_ for _ in ()).throw(FileNotFoundError("no log")))
    monkeypatch.setattr(report, "todays_picks_from_log", lambda: pd.DataFrame())
    monkeypatch.setattr(report, "_model_retrospective", lambda offset, now: pd.DataFrame())


def _fake_scan_result() -> daily_scan.DailyScanResult:
    result = daily_scan.DailyScanResult(n_games=1, n_players_considered=2, n_errors=0)
    result.pitchers = [
        daily_scan.PitcherEntry(
            team="PHI",
            opponent="CIN",
            name="Test Pitcher",
            projection=PitcherProjection(player_id="p1", season=2026, n_appearances=10, projected_strikeouts=7.5, recent_log=None),
        )
    ]
    result.batters = [
        daily_scan.BatterEntry(
            team="ATL",
            opponent="PIT",
            name="Test Batter",
            projection=BatterProjection(
                player_id="b1", season=2026, n_games=15, hit_prob=0.75, projected_total_bases=2.1, hr_prob=0.12, recent_log=None
            ),
        )
    ]
    return result


def _fake_pick() -> Pick:
    return Pick(
        commence_time="2026-07-09T23:00:00Z",
        home_team="NYY",
        away_team="BOS",
        model_home_win_prob=0.58,
        market_home_win_prob=0.52,
        side="home",
        side_price=-130,
        side_model_prob=0.58,
        side_market_prob=0.52,
        edge=0.06,
        suggested_stake_pct=0.021,
    )


def test_generate_report_includes_picks_and_props(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "scan_today", lambda: _fake_scan_result())
    monkeypatch.setattr(report, "generate_picks", lambda history_seasons: [_fake_pick()])

    output_path = tmp_path / "index.html"
    report.generate_report(output_path)

    text = output_path.read_text()
    assert "Test Pitcher" in text
    assert "Test Batter" in text
    assert "NYY" in text and "BOS" in text
    assert "-130" in text
    assert "Moneyline" in text
    assert "Methodology" in text
    # Fair line for the 58% model probability: -100*0.58/0.42 = -138.
    assert "-138" in text
    # Break-even lines for the prop probabilities: 75% hit prob -> -300, 12% HR prob -> +733.
    assert "Break-even" in text
    assert "-300" in text
    assert "+733" in text
    # MathML methodology section renders natively, no external scripts.
    assert "<math" in text
    assert "cdn" not in text.lower()


def test_winners_section_ranks_by_confidence_and_can_disagree_with_edge_pick():
    from datetime import datetime, timezone

    # Model says home wins 44% -> most likely winner is the AWAY team, even though
    # the edge pick (side="home") is the home dog. The section must use the away
    # side's price, which is why Pick carries both prices.
    dog_pick = Pick(
        commence_time="2026-07-09T23:00:00Z", home_team="CIN", away_team="PHI",
        model_home_win_prob=0.44, market_home_win_prob=0.38, side="home",
        side_price=148, side_model_prob=0.44, side_market_prob=0.38,
        edge=0.06, suggested_stake_pct=0.02, home_price=148, away_price=-160,
    )
    confident_pick = Pick(
        commence_time="2026-07-09T23:00:00Z", home_team="DET", away_team="ATH",
        model_home_win_prob=0.63, market_home_win_prob=0.54, side="home",
        side_price=-121, side_model_prob=0.63, side_market_prob=0.54,
        edge=0.09, suggested_stake_pct=0.03, home_price=-121, away_price=112,
    )

    frame = report._picks_to_frame([dog_pick, confident_pick])
    html = report._render_winners_section(frame, now=datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc))

    # PHI (away, 56%) is the predicted winner of the CIN game at the away price.
    assert "PHI" in html and "-160" in html
    # DET (63%) ranks above PHI (56%).
    assert html.index("DET") < html.index("PHI")
    assert "Most likely winners" in html
    assert "Upcoming" in html
    # Fair line for DET's 63% win probability: -100*0.63/0.37 = -170.
    assert "Fair line" in html
    assert "-170" in html


def test_picks_section_shows_finished_games_with_results():
    from datetime import datetime, timezone

    frame = report._picks_to_frame([_fake_pick()])
    frame.loc[0, "won"] = True
    # A second, still-upcoming game.
    frame = pd.concat([frame, report._picks_to_frame([_fake_pick()])], ignore_index=True)
    frame.loc[1, "commence_time"] = "2026-07-09T23:59:00Z"

    html = report._render_picks_section(frame, error=None, now=datetime(2026, 7, 9, 23, 30, tzinfo=timezone.utc))

    assert '<tr class="won">' in html
    assert "&#10003; Won" in html
    assert "Upcoming" in html


def test_retrospective_card_grades_model_calls_against_finals():
    day = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09", "2026-07-09"]),
            "home_team": ["DET", "CIN"],
            "away_team": ["ATH", "PHI"],
            "home_runs": [5.0, 2.0],
            "away_runs": [3.0, 7.0],
            "home_win": [1, 0],
            "prob_home": [0.63, 0.56],  # DET call correct; CIN call wrong
        }
    )

    html = report._render_retrospective_card(day)

    assert "Model retrospective" in html
    assert "1 of 2" in html
    assert '<tr class="won">' in html and '<tr class="lost">' in html
    assert "&#10003; Correct" in html and "&#10007; Wrong" in html
    # Fair line for the 63% call: -170.
    assert "-170" in html
    # Final scores shown away-home to match the matchup order.
    assert "3&ndash;5" in html and "7&ndash;2" in html


def test_day_tabs_use_retrospective_when_day_has_no_log(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "scan_today", lambda: _fake_scan_result())
    monkeypatch.setattr(report, "generate_picks", lambda history_seasons, days_ahead=0: [_fake_pick()] if days_ahead == 0 else [])

    retro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-08"]),
            "home_team": ["DET"], "away_team": ["ATH"],
            "home_runs": [5.0], "away_runs": [3.0], "home_win": [1], "prob_home": [0.63],
        }
    )
    monkeypatch.setattr(report, "_model_retrospective", lambda offset, now: retro if offset == -1 else pd.DataFrame())

    output_path = tmp_path / "index.html"
    report.generate_report(output_path)

    text = output_path.read_text()
    yesterday_panel = text.split('id="day-yesterday"')[1].split('id="day-today"')[0]
    assert "Model retrospective" in yesterday_panel
    assert "DET" in yesterday_panel


def test_day_tabs_render_yesterday_today_tomorrow(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "scan_today", lambda: _fake_scan_result())
    monkeypatch.setattr(report, "generate_picks", lambda history_seasons, days_ahead=0: [_fake_pick()])

    output_path = tmp_path / "index.html"
    report.generate_report(output_path)

    text = output_path.read_text()
    assert 'data-day="yesterday"' in text and 'data-day="today"' in text and 'data-day="tomorrow"' in text
    assert 'id="day-yesterday" hidden' in text  # non-default panels start hidden
    assert 'id="day-today">' in text  # today visible by default
    assert "Early lines" in text  # tomorrow's caveat note
    # Tomorrow's panel is populated from the same stubbed picks.
    tomorrow_panel = text.split('id="day-tomorrow"')[1]
    assert "NYY" in tomorrow_panel
    # Yesterday (no log in tests) shows its empty note rather than nothing.
    yesterday_panel = text.split('id="day-yesterday"')[1].split('id="day-today"')[0]
    assert "No pregame-logged games for yesterday." in yesterday_panel


def _graded_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "logged_at": "2026-07-08T11:00:00+00:00", "commence_time": "2026-07-08T23:05:00Z",
                "home_team": "ATL", "away_team": "PIT", "side": "home", "side_price": -120,
                "model_home_win_prob": 0.60, "market_home_win_prob": 0.55,
                "edge": 0.05, "suggested_stake_pct": 0.02, "result": "home_win", "won": True,
            },
            {
                "logged_at": "2026-07-08T11:00:00+00:00", "commence_time": "2026-07-08T23:10:00Z",
                "home_team": "NYY", "away_team": "BOS", "side": "away", "side_price": 150,
                "model_home_win_prob": 0.45, "market_home_win_prob": 0.50,
                "edge": 0.05, "suggested_stake_pct": 0.02, "result": "home_win", "won": False,
            },
        ]
    )


def test_generate_report_renders_graded_results_green_and_red(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "scan_today", lambda: _fake_scan_result())
    monkeypatch.setattr(report, "generate_picks", lambda history_seasons: [_fake_pick()])
    monkeypatch.setattr(report, "grade_picks", lambda season: _graded_frame())

    output_path = tmp_path / "index.html"
    report.generate_report(output_path)

    text = output_path.read_text()
    assert "Results &mdash; graded picks" in text
    assert '<tr class="won">' in text
    assert '<tr class="lost">' in text
    assert "&#10003; Won" in text
    assert "&#10007; Lost" in text
    # Record 1-1; won at -120 pays +0.83u, loss is -1u -> -0.17u.
    assert "Record <strong>1&ndash;1</strong>" in text
    assert "-0.17u" in text


def test_break_even_line_handles_degenerate_probabilities():
    from sba.report import _break_even_line

    assert _break_even_line(1.0) == "&mdash;"
    assert _break_even_line(0.0) == "&mdash;"
    assert _break_even_line(0.5) == "+100"


def test_generate_report_degrades_gracefully_when_picks_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "scan_today", lambda: _fake_scan_result())

    def _boom(history_seasons):
        raise OddsAPIError("No ODDS_API_KEY set.")

    monkeypatch.setattr(report, "generate_picks", _boom)

    output_path = tmp_path / "index.html"
    report.generate_report(output_path)

    text = output_path.read_text()
    assert "Picks unavailable" in text
    assert "No ODDS_API_KEY set." in text
    # Props section should still render fine even though picks failed.
    assert "Test Pitcher" in text


def test_generate_report_escapes_html_in_player_names(tmp_path, monkeypatch):
    result = daily_scan.DailyScanResult(n_games=1, n_players_considered=1, n_errors=0)
    result.pitchers = [
        daily_scan.PitcherEntry(
            team="PHI",
            opponent="CIN",
            name="<script>alert(1)</script>",
            projection=PitcherProjection(player_id="p1", season=2026, n_appearances=5, projected_strikeouts=5.0, recent_log=None),
        )
    ]
    monkeypatch.setattr(report, "scan_today", lambda: result)
    monkeypatch.setattr(report, "generate_picks", lambda history_seasons: [])

    output_path = tmp_path / "index.html"
    report.generate_report(output_path)

    text = output_path.read_text()
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_generate_report_without_password_has_no_gate(tmp_path, monkeypatch):
    monkeypatch.delenv(report.PAGE_PASSWORD_ENV, raising=False)
    monkeypatch.setattr(report, "scan_today", lambda: _fake_scan_result())
    monkeypatch.setattr(report, "generate_picks", lambda history_seasons: [_fake_pick()])

    output_path = tmp_path / "index.html"
    report.generate_report(output_path)

    text = output_path.read_text()
    assert 'id="gate"' not in text
    assert "Test Pitcher" in text


def test_generate_report_with_password_embeds_hash_not_plaintext(tmp_path, monkeypatch):
    monkeypatch.setenv(report.PAGE_PASSWORD_ENV, "hunter2")
    monkeypatch.setattr(report, "scan_today", lambda: _fake_scan_result())
    monkeypatch.setattr(report, "generate_picks", lambda history_seasons: [_fake_pick()])

    output_path = tmp_path / "index.html"
    report.generate_report(output_path)

    text = output_path.read_text()
    assert 'id="gate"' in text
    assert "hunter2" not in text
    assert hashlib.sha256(b"hunter2").hexdigest() in text
    # Content is still present in the source (this is a visibility gate, not real
    # security) -- just wrapped so it's CSS-hidden until unlocked client-side.
    assert "Test Pitcher" in text
    assert 'id="protected" style="display:none"' in text
