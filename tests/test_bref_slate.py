from sba.data import bref_slate


def _game_block(
    away_code="ATL",
    home_code="PIT",
    preview_url="/previews/2026/PIT202607090.shtml",
    game_time="12:35PM",
    away_pitcher='<a href="https://www.baseball-reference.com/players/e/elderbr01.shtml">Bryce Elder</a><br />(#55)',
    home_pitcher='<a href="https://www.baseball-reference.com/players/k/kellemi03.shtml">Mitch Keller</a><br />(#23)',
) -> str:
    return f"""
<div class="game_summary nohover ">
    <table class="teams">
        <tbody>
        <tr class="">
            <td><a href="/teams/{away_code}/2026.shtml">Away</a></td>
            <td class="right"></td>
            <td class="right gamelink">
                <a href="{preview_url}">Preview</a>
            </td>
        </tr>
        <tr class="">
            <td><a href="/teams/{home_code}/2026.shtml">Home</a></td>
            <td class="right"></td>
            <td class="right">{game_time}
            </td>
        </tr>
        </tbody>
    </table>
    <table>
        <tbody>
            <tr><td><strong>{away_code}</strong></td><td>{away_pitcher}</td></tr>
            <tr><td><strong>{home_code}</strong></td><td>{home_pitcher}</td></tr>
        </tbody>
    </table>
</div>
"""


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


def test_fetch_todays_games_parses_multiple_games(monkeypatch):
    html = _game_block() + _game_block(
        away_code="KCR", home_code="NYM", preview_url="/previews/2026/NYN202607090.shtml", game_time="1:10PM"
    )
    monkeypatch.setattr(bref_slate.bref_http, "get", lambda *a, **k: _FakeResponse(html))

    games = bref_slate.fetch_todays_games()

    assert len(games) == 2
    first = games[0]
    assert first["away_team"] == "ATL"
    assert first["home_team"] == "PIT"
    assert first["game_time"] == "12:35PM"
    assert first["preview_url"] == "/previews/2026/PIT202607090.shtml"
    assert first["away_pitcher"] == {"id": "elderbr01", "name": "Bryce Elder"}
    assert first["home_pitcher"] == {"id": "kellemi03", "name": "Mitch Keller"}
    assert games[1]["away_team"] == "KCR"


def test_fetch_todays_games_handles_missing_pitcher(monkeypatch):
    html = _game_block(away_pitcher="TBD", home_pitcher='<a href="https://www.baseball-reference.com/players/k/kellemi03.shtml">Mitch Keller</a>')
    monkeypatch.setattr(bref_slate.bref_http, "get", lambda *a, **k: _FakeResponse(html))

    games = bref_slate.fetch_todays_games()

    assert len(games) == 1
    assert games[0]["away_pitcher"] is None
    assert games[0]["home_pitcher"] == {"id": "kellemi03", "name": "Mitch Keller"}


def _batters_table_html(team: str, rows: list[tuple[str, str, int]]) -> str:
    """rows: list of (player_id, name, pa_last_28d)."""
    body_rows = "".join(
        f'<tr><td><a href="/players/x/{pid}.shtml" class="poptip" tip="{name}">{name[:4]}</a></td>'
        f"<td>300</td><td>{pa}</td></tr>"
        for pid, name, pa in rows
    )
    total_row = "<tr><td>TOTAL</td><td>999</td><td>999</td></tr>"
    return f"""
<table id="batters_{team}">
<thead><tr><th>Batter</th><th>PA</th><th>PA Last 28d</th></tr></thead>
<tbody>{body_rows}{total_row}</tbody>
</table>
"""


def test_parse_lineup_pool_extracts_players_and_excludes_total():
    html = _batters_table_html(
        "ATL",
        [("aaaaaa01", "Player One", 50), ("bbbbbb01", "Player Two", 2)],
    )
    pool = bref_slate.parse_lineup_pool(html, "ATL")

    assert len(pool) == 2
    assert pool[0] == {"player_id": "aaaaaa01", "name": "Player One", "pa_last_28d": 50.0}
    assert pool[1]["name"] == "Player Two"


def test_parse_lineup_pool_returns_empty_for_missing_table():
    assert bref_slate.parse_lineup_pool("<html><body>no table here</body></html>", "ATL") == []
