"""Unit tests for the notify package: HTML→Markdown, message splitting,
Discord webhook posting, and the platform dispatcher.

All HTTP calls are monkeypatched — no network, no real webhooks.
"""
from __future__ import annotations

import importlib
import logging
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture
def clean_env(monkeypatch):
    """Remove all notification env vars so each test starts from a known state."""
    for k in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "DISCORD_WEBHOOK_URL"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def discord_bot(clean_env):
    """Re-import discord_bot under a clean env so module-level WEBHOOK_URL is empty."""
    import notify.discord_bot as dc
    return importlib.reload(dc)


class _FakeResp:
    def __init__(self, status: int = 204):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# ──────────────────────────────────────────────────────────────────────────
# _html_to_markdown
# ──────────────────────────────────────────────────────────────────────────
def test_html_to_markdown_bold_italic(discord_bot):
    assert discord_bot._html_to_markdown("<b>X</b>") == "**X**"
    assert discord_bot._html_to_markdown("<i>Y</i>") == "*Y*"
    assert discord_bot._html_to_markdown("<b>A</b> <i>B</i>") == "**A** *B*"


def test_html_to_markdown_strips_unknown_tags(discord_bot):
    """Unknown tags (e.g. <code>, <pre>) get stripped, content preserved."""
    assert discord_bot._html_to_markdown("hi <code>x</code> bye") == "hi x bye"


def test_html_to_markdown_preserves_emoji_and_newlines(discord_bot):
    text = "📊 <b>台股選股報告</b>\n主力訊號：3 支"
    out = discord_bot._html_to_markdown(text)
    assert "📊" in out
    assert "**台股選股報告**" in out
    assert "\n" in out


# ──────────────────────────────────────────────────────────────────────────
# _split_for_discord
# ──────────────────────────────────────────────────────────────────────────
def test_split_short_message_returns_single_chunk(discord_bot):
    assert discord_bot._split_for_discord("hello") == ["hello"]


def test_split_respects_2000_char_limit(discord_bot):
    text = "\n".join(["x" * 100] * 30)  # ~3000 chars
    chunks = discord_bot._split_for_discord(text)
    assert len(chunks) >= 2
    assert all(len(c) <= discord_bot.DISCORD_MAX_LEN for c in chunks)


def test_split_preserves_all_lines(discord_bot):
    lines = [f"line-{i}" for i in range(500)]
    text = "\n".join(lines)
    chunks = discord_bot._split_for_discord(text)
    rejoined = "\n".join(chunks)
    for line in lines:
        assert line in rejoined


def test_split_breaks_on_line_boundaries(discord_bot):
    """Each chunk should start cleanly on a line, not mid-word."""
    text = "\n".join(["abcdefghij"] * 300)  # 11 chars × 300 ≈ 3300
    chunks = discord_bot._split_for_discord(text, limit=100)
    for c in chunks:
        # Every line in a chunk must be one of the original lines (no half-lines).
        for sub in c.split("\n"):
            assert sub == "abcdefghij" or sub == ""


# ──────────────────────────────────────────────────────────────────────────
# discord_bot.send_message
# ──────────────────────────────────────────────────────────────────────────
def test_send_message_no_webhook_returns_false(discord_bot, caplog):
    with caplog.at_level(logging.WARNING):
        assert discord_bot.send_message("hi") is False
    assert "Discord not configured" in caplog.text


def test_send_message_posts_converted_markdown(monkeypatch, clean_env):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/webhook/abc")
    import notify.discord_bot as dc
    importlib.reload(dc)

    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _FakeResp(204)

    monkeypatch.setattr(dc.requests, "post", fake_post)
    monkeypatch.setattr(dc.time, "sleep", lambda _s: None)

    assert dc.send_message("<b>hello</b> <i>world</i>") is True
    assert captured["url"] == "https://discord.test/webhook/abc"
    assert captured["json"] == {"content": "**hello** *world*"}


def test_send_message_splits_long_payload_into_multiple_posts(monkeypatch, clean_env):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/webhook/abc")
    import notify.discord_bot as dc
    importlib.reload(dc)

    posts: list[dict] = []
    monkeypatch.setattr(dc.requests, "post",
                        lambda url, json, timeout: posts.append(json) or _FakeResp(204))
    monkeypatch.setattr(dc.time, "sleep", lambda _s: None)

    long_text = "\n".join(["y" * 100] * 30)  # ~3000 chars
    assert dc.send_message(long_text) is True
    assert len(posts) >= 2
    assert all(len(p["content"]) <= dc.DISCORD_MAX_LEN for p in posts)


def test_send_message_returns_false_on_http_error(monkeypatch, clean_env, caplog):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/webhook/abc")
    import notify.discord_bot as dc
    importlib.reload(dc)

    monkeypatch.setattr(dc.requests, "post",
                        lambda url, json, timeout: _FakeResp(500))
    monkeypatch.setattr(dc.time, "sleep", lambda _s: None)

    with caplog.at_level(logging.ERROR):
        assert dc.send_message("hi") is False
    assert "Discord send failed" in caplog.text


# ──────────────────────────────────────────────────────────────────────────
# discord_bot.notify (end-to-end with stubbed format_signals)
# ──────────────────────────────────────────────────────────────────────────
def test_notify_sends_each_formatted_message(monkeypatch, clean_env):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/webhook/abc")
    import notify.discord_bot as dc
    importlib.reload(dc)

    monkeypatch.setattr(dc, "format_signals",
                        lambda signals, date: ["<b>m1</b>", "<i>m2</i>", "m3"])
    sent: list[str] = []
    monkeypatch.setattr(dc, "send_message", lambda t: sent.append(t) or True)
    monkeypatch.setattr(dc.time, "sleep", lambda _s: None)

    dc.notify({"long": pd.DataFrame()})
    assert sent == ["<b>m1</b>", "<i>m2</i>", "m3"]


# ──────────────────────────────────────────────────────────────────────────
# Dispatcher (notify.__init__)
# ──────────────────────────────────────────────────────────────────────────
def test_dispatcher_warns_when_nothing_configured(clean_env, caplog):
    import notify
    importlib.reload(notify)
    with caplog.at_level(logging.WARNING):
        notify.notify({})
    assert "No notification platform configured" in caplog.text


def test_dispatcher_routes_to_telegram_only(monkeypatch, clean_env):
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")
    import notify
    importlib.reload(notify)

    calls = {"tg": 0, "dc": 0}
    import notify.telegram_bot as tg
    import notify.discord_bot as dc
    monkeypatch.setattr(tg, "notify", lambda s: calls.__setitem__("tg", calls["tg"] + 1))
    monkeypatch.setattr(dc, "notify", lambda s: calls.__setitem__("dc", calls["dc"] + 1))

    notify.notify({})
    assert calls == {"tg": 1, "dc": 0}


def test_dispatcher_routes_to_discord_only(monkeypatch, clean_env):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/x")
    import notify
    importlib.reload(notify)

    calls = {"tg": 0, "dc": 0}
    import notify.telegram_bot as tg
    import notify.discord_bot as dc
    monkeypatch.setattr(tg, "notify", lambda s: calls.__setitem__("tg", calls["tg"] + 1))
    monkeypatch.setattr(dc, "notify", lambda s: calls.__setitem__("dc", calls["dc"] + 1))

    notify.notify({})
    assert calls == {"tg": 0, "dc": 1}


def test_dispatcher_routes_to_both_when_configured(monkeypatch, clean_env):
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/x")
    import notify
    importlib.reload(notify)

    calls = {"tg": 0, "dc": 0}
    import notify.telegram_bot as tg
    import notify.discord_bot as dc
    monkeypatch.setattr(tg, "notify", lambda s: calls.__setitem__("tg", calls["tg"] + 1))
    monkeypatch.setattr(dc, "notify", lambda s: calls.__setitem__("dc", calls["dc"] + 1))

    notify.notify({})
    assert calls == {"tg": 1, "dc": 1}


def test_dispatcher_skips_telegram_when_only_token_set(monkeypatch, clean_env):
    """Both TELEGRAM_TOKEN AND TELEGRAM_CHAT_ID required — token alone shouldn't fire."""
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")  # no CHAT_ID
    import notify
    importlib.reload(notify)

    calls = {"tg": 0}
    import notify.telegram_bot as tg
    monkeypatch.setattr(tg, "notify", lambda s: calls.__setitem__("tg", calls["tg"] + 1))

    notify.notify({})
    assert calls["tg"] == 0
