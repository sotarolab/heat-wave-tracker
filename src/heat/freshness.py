"""How old the forecast on the page is, and whether it may be called live.

The header is the only place a reader learns whether the refresh is running. A stalled
refresh once left a two-month-old cycle under a LIVE badge, so the age is computed when
the page loads rather than when the process starts.
"""
from __future__ import annotations

import pandas as pd

# GFS cycles are six hours apart and publish four to five hours late, so a healthy page
# is at most about eleven hours behind its init time.
LIVE_MAX_AGE_H = 12.0


def age_hours(gfs_init: str, now: pd.Timestamp) -> float | None:
    """Hours between the cycle's init time (UTC, as stored in the dataset) and `now`."""
    try:
        init = pd.Timestamp(gfs_init)
    except (ValueError, TypeError):
        return None
    if pd.isna(init):
        return None
    if init.tzinfo is None:
        init = init.tz_localize("UTC")
    return (now.tz_convert("UTC") - init).total_seconds() / 3600.0


def badge(gfs_init: str, now: pd.Timestamp) -> tuple[str, bool]:
    """The badge text and whether the forecast is fresh enough to be called live."""
    age = age_hours(gfs_init, now)
    if age is None:
        return "NO DATA", False
    if age <= LIVE_MAX_AGE_H:
        return "LIVE", True
    return (f"STALE {age:.0f} h" if age < 48 else f"STALE {age / 24:.0f} d"), False
