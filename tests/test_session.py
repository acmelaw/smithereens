"""Tests for Session — messages, persistence, cost tracking, compaction."""

import json
from pathlib import Path

import pytest

from smithereens.config import Config
from smithereens.session import Session


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def config(tmp_path):
    """Config with a temp sessions dir so tests never touch ~/.smithereens."""
    c = Config()
    c.sessions_dir = tmp_path / "sessions"
    return c


@pytest.fixture
def session(config):
    return Session(config=config)


# ── Message handling ──────────────────────────────────────────


class TestMessages:
    def test_add_user_message_appends(self, session):
        session.add_user_message("hello")
        assert session.messages[-1] == {"role": "user", "content": "hello"}

    def test_add_user_message_increments_turn_count(self, session):
        assert session.turn_count == 0
        session.add_user_message("a")
        session.add_user_message("b")
        assert session.turn_count == 2

    def test_initial_messages_empty(self, session):
        assert session.messages == []


# ── Usage tracking ────────────────────────────────────────────


class TestUsage:
    def test_update_usage_accumulates(self, session):
        session.update_usage(100, 50)
        session.update_usage(200, 100)
        assert session.total_input_tokens == 300
        assert session.total_output_tokens == 150

    def test_cost_is_zero_initially(self, session):
        assert session.cost == 0.0

    def test_cost_uses_pricing(self, session):
        session.config.price_input = 3.0
        session.config.price_output = 15.0
        session.update_usage(1_000_000, 1_000_000)
        assert session.cost == pytest.approx(18.0)

    def test_cost_scales_linearly(self, session):
        session.config.price_input = 2.0
        session.config.price_output = 10.0
        session.update_usage(500_000, 100_000)
        expected = (500_000 * 2.0 + 100_000 * 10.0) / 1_000_000
        assert session.cost == pytest.approx(expected)


# ── Compaction ────────────────────────────────────────────────


class TestCompaction:
    def test_does_nothing_when_under_limit(self, session):
        for i in range(5):
            session.messages.append({"role": "user", "content": f"msg {i}"})
        session.maybe_compact()
        assert len(session.messages) == 5

    def test_trims_to_max_messages(self, session):
        session.config.max_messages = 4
        for i in range(10):
            session.messages.append({"role": "user", "content": f"msg {i}"})
        session.maybe_compact()
        assert len(session.messages) == 4
        # Keeps the most recent
        assert session.messages[-1]["content"] == "msg 9"


# ── Persistence roundtrip ────────────────────────────────────


class TestPersistence:
    def test_save_creates_file(self, session):
        path = session.save()
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_file_has_expected_keys(self, session):
        session.add_user_message("test")
        path = session.save()
        data = json.loads(path.read_text())
        for key in ("id", "messages", "model", "input_tokens", "turns"):
            assert key in data

    def test_roundtrip_messages(self, config, session):
        session.add_user_message("hello")
        session.add_user_message("world")
        session.update_usage(100, 50)
        path = session.save()

        restored = Session.resume(session.session_id, config)
        assert len(restored.messages) == 2
        assert restored.messages[0]["content"] == "hello"
        assert restored.messages[1]["content"] == "world"

    def test_roundtrip_usage(self, config, session):
        session.update_usage(100, 50)
        session.save()
        restored = Session.resume(session.session_id, config)
        assert restored.total_input_tokens == 100
        assert restored.total_output_tokens == 50

    def test_roundtrip_turn_count(self, config, session):
        session.add_user_message("a")
        session.add_user_message("b")
        session.save()
        restored = Session.resume(session.session_id, config)
        assert restored.turn_count == 2

    def test_resume_by_number(self, config):
        """Resume by numeric index (1 = most recent)."""
        s1 = Session(config=config)
        s1.add_user_message("first")
        s1.save()

        s2 = Session(config=config)
        s2.add_user_message("second")
        s2.save()

        restored = Session.resume("1", config)
        # "1" should be the most recent session
        assert restored.session_id == s2.session_id

    def test_resume_nonexistent_raises(self, config):
        with pytest.raises(FileNotFoundError):
            Session.resume("nonexistent-id-xyz", config)

    def test_save_creates_sessions_dir(self, tmp_path):
        c = Config()
        c.sessions_dir = tmp_path / "deep" / "nested" / "sessions"
        s = Session(config=c)
        path = s.save()
        assert path.exists()
