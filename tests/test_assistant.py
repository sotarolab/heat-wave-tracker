"""
tests/test_assistant.py
========================
Pure-logic tests for src/heat/assistant.py: rate limiting, render
capture, and configuration. No network and no SDK, so they run under
requirements-dev.txt; the model-facing behaviour is covered by the
offline eval suite in scripts/assistant_evals.py instead.
"""
import pytest

from src.heat import assistant


class TestRateLimit:
    def setup_method(self):
        assistant._hits.clear()

    def test_allows_then_blocks_per_minute(self):
        for _ in range(assistant.ASK_PER_MINUTE):
            assert assistant.check_rate_limit("a", now=100.0) is None
        assert "short time" in assistant.check_rate_limit("a", now=100.0)

    def test_minute_window_slides(self):
        for _ in range(assistant.ASK_PER_MINUTE):
            assistant.check_rate_limit("b", now=0.0)
        assert assistant.check_rate_limit("b", now=61.0) is None

    def test_daily_cap(self):
        t = 0.0
        for i in range(assistant.ASK_PER_DAY):
            assistant.check_rate_limit("c", now=t)
            t += 61.0
        assert "Daily question limit" in assistant.check_rate_limit("c", now=t)

    def test_clients_are_independent(self):
        for _ in range(assistant.ASK_PER_MINUTE):
            assistant.check_rate_limit("d", now=5.0)
        assert assistant.check_rate_limit("e", now=5.0) is None

    def test_global_cap(self, monkeypatch):
        monkeypatch.setattr(assistant, "ASK_GLOBAL_PER_DAY", 2)
        assert assistant.check_rate_limit("f", now=0.0) is None
        assert assistant.check_rate_limit("g", now=0.0) is None
        assert "daily limit" in assistant.check_rate_limit("h", now=0.0)


class TestRenderCapture:
    def test_capture_records_into_active_context(self):
        token = assistant._renders.set([])
        try:
            assert assistant._capture("table", {"title": "t", "columns": ["a"], "rows": [["1"]]}) == "rendered"
            got = assistant._renders.get()
        finally:
            assistant._renders.reset(token)
        assert got == [{"kind": "table", "title": "t", "columns": ["a"], "rows": [["1"]]}]

    def test_capture_outside_context_is_noop(self):
        # A tool called with no active ask() must not raise or leak state.
        assert assistant._capture("chart", {"title": "x"}) == "rendered"

    def test_capture_kind_is_not_overwritten_by_spec(self):
        token = assistant._renders.set([])
        try:
            assistant._capture("chart", {"chart_type": "bar", "title": "x"})
            got = assistant._renders.get()[0]
        finally:
            assistant._renders.reset(token)
        assert got["kind"] == "chart" and got["chart_type"] == "bar"


class TestConfiguration:
    def test_unavailable_without_provider(self, monkeypatch):
        monkeypatch.setattr(assistant, "_provider", None)
        assert assistant.available() is False

    def test_provider_access_raises_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr(assistant, "_provider", None)
        with pytest.raises(RuntimeError):
            assistant._p()
