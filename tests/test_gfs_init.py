"""Which GFS cycle the fetch asks for. A cycle is taken to be published four hours after its init."""
import pandas as pd
import pytest

from src.heat.gfs_conus import _latest_gfs_init


@pytest.mark.parametrize(
    "now, expect",
    [
        ("2026-11-17 00:05", "2026-11-16 18:00"),  # before 04Z: yesterday's 18Z, not today's unpublished 00Z
        ("2026-11-17 03:59", "2026-11-16 18:00"),
        ("2026-11-17 04:00", "2026-11-17 00:00"),
        ("2026-11-17 12:30", "2026-11-17 06:00"),
        ("2026-11-17 23:59", "2026-11-17 18:00"),
    ],
)
def test_latest_init_is_the_newest_cycle_at_least_four_hours_old(now, expect):
    assert _latest_gfs_init(pd.Timestamp(now, tz="UTC")) == pd.Timestamp(expect, tz="UTC")
