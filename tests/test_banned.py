from jevlab import vault
from jevlab.live import best_saved
from jevlab.modes import HIGH_SCORES, SHORTEST_YES, STRICT
from jevlab.publish import ban_fall_through, sweep_banned
from jevlab.rules import banned


def _reset_bans(tmp_path, monkeypatch):
    monkeypatch.setattr(banned, "PATH", tmp_path / "banned_phrases.json")
    banned._phrases = {}
    banned._mtime = None
    banned._checked = 0.0


def _save(slug, phrase, p, board=HIGH_SCORES, status="candidate"):
    vault.save(
        slug,
        STRICT,
        phrase,
        p_mean=p,
        p_lcb=p,
        spread=0.01,
        n=5,
        units=len(phrase.split()),
        leader=None,
        beats=True,
        title=slug,
        board=board,
    )
    if status != "candidate":
        vault.set_status(slug, STRICT, phrase, status, board)


def test_banned_phrase_records_only_the_named_span(tmp_path, monkeypatch):
    _reset_bans(tmp_path, monkeypatch)

    assert (
        banned.quoted_ban('"this logic game" is banned. Take one of its words out or try something else.')
        == "this logic game"
    )
    assert banned.quoted_ban('\\"in this logic\\" is banned. Take one of its words out or try something else.') == (
        "in this logic"
    )
    assert banned.quoted_ban("banned_phrase") is None

    named = banned.note(
        "banned_phrase",
        '"in this logic" is banned. Take one of its words out or try something else.',
    )
    assert named == "in this logic"
    assert banned.contains("in this logic game other words are not banned")
    assert banned.contains("in-this-logic")
    assert not banned.contains("logic game")
    assert not banned.contains("in this")
    assert not banned.contains("logic")
    assert not banned.contains("this")

    assert banned.note("banned_phrase", "rejected") is None
    assert not banned.contains("rejected")
    assert list(banned.phrases()) == ["in this logic"]
    banned._phrases = {}
    banned._mtime = None
    banned._checked = 0.0


def test_drop_banned_clears_every_vault_then_the_next_line_can_publish(tmp_path, monkeypatch):
    _reset_bans(tmp_path, monkeypatch)
    monkeypatch.setattr(vault, "VAULT", tmp_path / "vault")
    _save("alpha", "in this logic game", 0.99, status="queued")
    _save("alpha", "a clean winner", 0.80)
    _save("beta", "in-this-logic elsewhere", 0.97, board=SHORTEST_YES)
    _save("gamma", "logic game", 0.96)

    question = {"slug": "alpha", "title": "alpha"}
    entries = vault.all_entries()
    assert best_saved(question, entries)["phrase"] == "in this logic game"

    banned.note(
        "banned_phrase",
        '"in this logic" is banned. Take one of its words out or try something else.',
    )
    assert best_saved(question, vault.all_entries())["phrase"] == "a clean winner"

    logs = []
    removed = sweep_banned(logs.append)
    assert {e["phrase"] for e in removed} == {"in this logic game", "in-this-logic elsewhere"}
    assert {e["phrase"] for e in vault.all_entries()} == {"a clean winner", "logic game"}
    assert logs == [
        "removed 2 vault line(s) containing a banned phrase: in this logic game; in-this-logic elsewhere"
    ]

    nxt = ban_fall_through(
        {"slug": "alpha", "mode": STRICT, "phrase": "in this logic game", "board": HIGH_SCORES, "target": ""},
        0,
        logs.append,
    )
    assert nxt["phrase"] == "a clean winner"
    assert nxt["status"] == "queued"
    assert sweep_banned(logs.append) == []
    banned._phrases = {}
    banned._mtime = None
    banned._checked = 0.0
