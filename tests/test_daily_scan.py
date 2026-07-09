from sba import daily_scan
from sba.props import BatterProjection, PitcherProjection


def _fake_pitcher_projection(k: float) -> PitcherProjection:
    return PitcherProjection(player_id="x", season=2026, n_appearances=10, projected_strikeouts=k, recent_log=None)


def _fake_batter_projection(hit_prob: float, hr_prob: float = 0.05, tb: float = 1.5) -> BatterProjection:
    return BatterProjection(
        player_id="x", season=2026, n_games=15, hit_prob=hit_prob, projected_total_bases=tb, hr_prob=hr_prob, recent_log=None
    )


def _game(away, home, away_pid="p_away", home_pid="p_home", preview_url="/previews/x.shtml"):
    return {
        "away_team": away,
        "home_team": home,
        "game_time": "7:05PM",
        "preview_url": preview_url,
        "away_pitcher": {"id": away_pid, "name": f"{away_pid} name"},
        "home_pitcher": {"id": home_pid, "name": f"{home_pid} name"},
    }


def test_scan_today_dedupes_pitchers_seen_in_multiple_games(monkeypatch):
    # Same pitcher id shows up as both a "home" and "away" starter across two fake
    # games (shouldn't happen in real data, but the dedup should hold regardless).
    games = [_game("ATL", "PIT", away_pid="dupe01", home_pid="kellemi03"), _game("NYM", "KCR", away_pid="dupe01", home_pid="wachami01")]
    monkeypatch.setattr(daily_scan, "fetch_todays_games", lambda: games)
    monkeypatch.setattr(daily_scan, "fetch_game_preview_html", lambda url: "")
    monkeypatch.setattr(daily_scan, "parse_lineup_pool", lambda html, team: [])

    calls = []

    def fake_project_pitcher(player_id):
        calls.append(player_id)
        return _fake_pitcher_projection(k=5.0)

    monkeypatch.setattr(daily_scan, "project_pitcher", fake_project_pitcher)

    result = daily_scan.scan_today()

    assert calls.count("dupe01") == 1
    assert len(result.pitchers) == 3  # dupe01, kellemi03, wachami01
    assert result.n_errors == 0


def test_scan_today_tolerates_a_single_player_error(monkeypatch):
    games = [_game("ATL", "PIT", away_pid="bad01", home_pid="good01")]
    monkeypatch.setattr(daily_scan, "fetch_todays_games", lambda: games)
    monkeypatch.setattr(daily_scan, "fetch_game_preview_html", lambda url: "")
    monkeypatch.setattr(daily_scan, "parse_lineup_pool", lambda html, team: [])

    def fake_project_pitcher(player_id):
        if player_id == "bad01":
            raise ValueError("no game log found")
        return _fake_pitcher_projection(k=7.5)

    monkeypatch.setattr(daily_scan, "project_pitcher", fake_project_pitcher)

    result = daily_scan.scan_today()

    assert result.n_errors == 1
    assert len(result.pitchers) == 1
    assert result.pitchers[0].name == "good01 name"


def test_scan_today_tolerates_a_preview_page_fetch_failure(monkeypatch):
    # A network hiccup fetching one game's preview page shouldn't abort a long scan --
    # that game's batters are skipped, but its pitchers (already fetched separately)
    # and the other game's batters should still come through.
    games = [
        _game("ATL", "PIT", away_pid="p1", home_pid="p2", preview_url="/previews/bad.shtml"),
        _game("NYM", "KCR", away_pid="p3", home_pid="p4", preview_url="/previews/good.shtml"),
    ]
    monkeypatch.setattr(daily_scan, "fetch_todays_games", lambda: games)
    monkeypatch.setattr(daily_scan, "project_pitcher", lambda player_id: _fake_pitcher_projection(k=5.0))

    def fake_fetch_preview(url):
        if url == "/previews/bad.shtml":
            raise ConnectionError("DNS resolution failed")
        return "html"

    monkeypatch.setattr(daily_scan, "fetch_game_preview_html", fake_fetch_preview)
    monkeypatch.setattr(
        daily_scan, "parse_lineup_pool", lambda html, team: [{"player_id": f"b_{team}", "name": team, "pa_last_28d": 50}]
    )
    monkeypatch.setattr(daily_scan, "project_batter", lambda player_id: _fake_batter_projection(hit_prob=0.6))

    result = daily_scan.scan_today()

    assert result.n_games_skipped == 1
    assert len(result.pitchers) == 4  # all 4 pitchers still projected
    assert len(result.batters) == 2  # only the second game's 2 batters


def test_scan_today_filters_batters_by_min_pa_and_dedupes(monkeypatch):
    games = [_game("ATL", "PIT", preview_url="/previews/g1.shtml")]
    monkeypatch.setattr(daily_scan, "fetch_todays_games", lambda: games)
    monkeypatch.setattr(daily_scan, "fetch_game_preview_html", lambda url: "html-for-" + url)

    pools = {
        "ATL": [
            {"player_id": "regular01", "name": "Regular Starter", "pa_last_28d": 80},
            {"player_id": "bench01", "name": "Bench Guy", "pa_last_28d": 3},
        ],
        "PIT": [{"player_id": "regular02", "name": "Other Starter", "pa_last_28d": 60}],
    }
    monkeypatch.setattr(daily_scan, "parse_lineup_pool", lambda html, team: pools[team])
    monkeypatch.setattr(daily_scan, "project_pitcher", lambda player_id: _fake_pitcher_projection(k=5.0))

    projected_batters = []

    def fake_project_batter(player_id):
        projected_batters.append(player_id)
        return _fake_batter_projection(hit_prob=0.6)

    monkeypatch.setattr(daily_scan, "project_batter", fake_project_batter)

    result = daily_scan.scan_today(min_batter_pa_last_28d=20)

    assert "bench01" not in projected_batters
    assert set(projected_batters) == {"regular01", "regular02"}
    assert len(result.batters) == 2


def test_top_pitchers_by_strikeouts_ranks_descending():
    result = daily_scan.DailyScanResult()
    result.pitchers = [
        daily_scan.PitcherEntry(team="A", opponent="B", name="Low K", projection=_fake_pitcher_projection(4.0)),
        daily_scan.PitcherEntry(team="A", opponent="B", name="High K", projection=_fake_pitcher_projection(9.0)),
        daily_scan.PitcherEntry(team="A", opponent="B", name="Mid K", projection=_fake_pitcher_projection(6.0)),
    ]

    top = daily_scan.top_pitchers_by_strikeouts(result, n=2)

    assert [e.name for e in top] == ["High K", "Mid K"]


def test_top_batters_rankings_use_correct_metric():
    result = daily_scan.DailyScanResult()
    result.batters = [
        daily_scan.BatterEntry(team="A", opponent="B", name="Hit King", projection=_fake_batter_projection(hit_prob=0.9, hr_prob=0.01, tb=1.0)),
        daily_scan.BatterEntry(team="A", opponent="B", name="Power Hitter", projection=_fake_batter_projection(hit_prob=0.4, hr_prob=0.3, tb=3.0)),
    ]

    assert daily_scan.top_batters_by_hit_prob(result, n=1)[0].name == "Hit King"
    assert daily_scan.top_batters_by_hr_prob(result, n=1)[0].name == "Power Hitter"
    assert daily_scan.top_batters_by_total_bases(result, n=1)[0].name == "Power Hitter"
