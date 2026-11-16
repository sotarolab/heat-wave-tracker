"""The header badge: live only while the cycle is recent, and never on unreadable input."""
import pandas as pd

from src.heat.freshness import age_hours, badge

NOW = pd.Timestamp("2026-11-16 18:00", tz="UTC")


def test_recent_cycle_is_live():
    assert badge("2026-11-16 12:00:00", NOW) == ("LIVE", True)
    assert age_hours("2026-11-16 12:00:00", NOW) == 6.0


def test_old_cycle_is_stale_in_hours_then_days():
    assert badge("2026-11-15 12:00:00", NOW) == ("STALE 30 h", False)
    assert badge("2026-09-17 12:00:00", NOW) == ("STALE 60 d", False)


def test_unreadable_init_is_never_live():
    assert badge("data not loaded", NOW) == ("NO DATA", False)
    assert badge("unknown init", NOW) == ("NO DATA", False)
