"""The writing-rules check: what it flags and what it leaves alone."""
from scripts.check_phrases import checked, violations


def test_flags_each_rule_with_its_line():
    text = "fine\nan em dash " + chr(0x2014) + " here\nto be " + "hon" + "est\n"
    assert violations("a.md", text) == ["a.md:2: em dash", "a.md:3: candour word"]


def test_clean_text_passes():
    assert violations("a.md", "plain prose, with a hyphen - and nothing else\n") == []


def test_legacy_and_binary_paths_are_skipped():
    assert checked("docs/PLAN.md") and checked("service/app.py")
    assert not checked("notebooks/common.py") and not checked("app.py")
    assert not checked("data/conus_heat_tracker.nc")
