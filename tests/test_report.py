from sba import daily_scan, report
from sba.data.odds import OddsAPIError
from sba.picks import Pick
from sba.props import BatterProjection, PitcherProjection


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
    assert "Moneyline picks" in text
    assert "Today's prop projections" in text
    assert "not financial advice" in text.lower()


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
