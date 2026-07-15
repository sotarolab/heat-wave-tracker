"""
src/heat/climate.py
====================
Fetch NWS Daily Climate Report (CLI) text products via IEM's AFOS
archive, for actual-vs-normal temperature comparison at ASOS stations
that have one.

Not every station has a CLI report, it is issued per NWS office policy,
not for every airport. Callers should treat a None return as "not
available for this station," not an error.

Gotchas:

1. CLI product ID is "CLI" plus the station's ICAO code with the
   leading "K" removed (e.g. KDCA -> CLIDCA). Confirmed against several
   major and minor CONUS stations before relying on it.
2. Different NWS offices format the MAXIMUM/MINIMUM temperature row
   slightly differently. Some omit the trailing "LAST YEAR" column
   entirely, which silently shifts every field after it by one position
   under naive whitespace-split parsing. Confirmed live against a real
   report where this produced a 4-digit year in the "normal" slot
   instead of a temperature. The record year is the one unambiguous
   anchor in that row: it is always a 4-digit 18xx/19xx/20xx token,
   distinct from any plausible temperature value, and "normal" is
   always the field immediately after it. _row_values anchors on that
   token instead of counting positions from the end. Departure is
   computed as observed minus normal rather than parsed from the row,
   sidestepping the same ambiguity entirely, since departure is that by
   definition.
"""
from __future__ import annotations

import re
import requests

IEM_AFOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
USER_AGENT = "heat-wave-tracker/0.1 (portfolio/research use)"

_DATE_RE = re.compile(r"CLIMATE SUMMARY FOR ([A-Z]+ +\d{1,2} +\d{4})")
_YEAR_RE = re.compile(r"^(?:18|19|20)\d{2}$")


def _to_int(tok: str) -> int | None:
    """Parse one CLI report token to int, treating NWS's missing/trace
    markers (M, MM, T) as None instead of raising."""
    tok = tok.upper()
    if tok in ("M", "MM", "T"):
        return None
    try:
        return int(tok)
    except ValueError:
        return None


def _row_values(temp_section: str, label: str) -> dict | None:
    """Extract observed/normal from the MAXIMUM or MINIMUM row of the TEMPERATURE section.

    See module Gotcha 2 for why this anchors on the record-year token
    rather than counting fields from either end of the row.

    Parameters
    ----------
    temp_section : str
        The TEMPERATURE (F) block of a CLI report, as extracted by
        fetch_climate_summary.
    label : str
        "MAXIMUM" or "MINIMUM", matching the row's leading label.

    Returns
    -------
    dict or None
        None if the row is missing or cannot be parsed. Otherwise a
        dict with observed, normal, and departure (all int, degrees F).
    """
    for line in temp_section.splitlines():
        stripped = line.strip()
        if not stripped.startswith(label):
            continue
        tokens = stripped.split()
        if len(tokens) < 3:
            return None
        observed = _to_int(tokens[1].rstrip("R"))
        if observed is None:
            return None
        year_idx = next((i for i, t in enumerate(tokens) if _YEAR_RE.match(t)), None)
        if year_idx is None or year_idx + 1 >= len(tokens):
            return None
        normal = _to_int(tokens[year_idx + 1])
        if normal is None:
            return None
        return {"observed": observed, "normal": normal, "departure": observed - normal}
    return None


def fetch_climate_summary(station_id: str) -> dict | None:
    """Fetch the latest CLI (Daily Climate Report) for one station.

    Uses the MAXIMUM row specifically, not the daily average, since a
    heat tracker cares about how unusual the day's peak heat is.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code, e.g. "KDCA".

    Returns
    -------
    dict or None
        None if the station has no CLI product, the report cannot be
        parsed, or the key values are missing or flagged. Otherwise a
        dict with:

        period_label : str
            "Yesterday", "Today", or "Latest" if neither marker is
            present in the report.
        date : str or None
            Report date, e.g. "July 2 2026".
        high_actual_f : int
            Today's (or yesterday's) actual high, degrees F.
        high_normal_f : int
            The 1991-2020 normal high for this date, degrees F.
        high_departure_f : int
            high_actual_f - high_normal_f.
    """
    code = station_id[1:] if station_id.startswith("K") else station_id  # see module Gotcha 1
    try:
        resp = requests.get(
            IEM_AFOS_URL,
            params={"pil": f"CLI{code}", "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[climate] IEM fetch failed for CLI{code}: {exc}")
        return None

    text = resp.text
    if not text.strip():
        return None

    temp_match = re.search(
        r"TEMPERATURE\s*\(F\)(.*?)(?:PRECIPITATION|SNOWFALL|DEGREE DAYS|WIND \(|$)",
        text, re.DOTALL,
    )
    if not temp_match:
        return None
    temp_section = temp_match.group(1)

    period_label = "Yesterday" if "YESTERDAY" in temp_section else (
        "Today" if "TODAY" in temp_section else None)

    high = _row_values(temp_section, "MAXIMUM")
    if not high or None in (high["observed"], high["normal"], high["departure"]):
        return None

    date_match = _DATE_RE.search(text)
    date_str = date_match.group(1).title() if date_match else None

    return {
        "period_label":    period_label or "Latest",
        "date":            date_str,
        "high_actual_f":   high["observed"],
        "high_normal_f":   high["normal"],
        "high_departure_f": high["departure"],
    }
