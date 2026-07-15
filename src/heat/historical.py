"""
src/heat/historical.py
=======================
Resolve each station's ICAO code to its IEM ASOS network/station id, and
fetch daily max temperature going back to 1972, for building the annual
block-maxima archive used by the GEV return-period estimate.

Gotchas:

1. Not all ASOS stations use their 4-letter ICAO id in IEM's system.
   Several first-order NWS stations kept their older 3-letter id even
   after joining the ASOS network. For example KDCA has no "*_ASOS"
   record at all, the real one is "DCA" under network "VA_ASOS",
   confirmed live against IEM's station metadata API. resolve_iem_station
   goes through that API instead of guessing (e.g. "just strip the K"),
   to avoid silently mismatching stations like that one.
2. KPBI (West Palm Beach Intl) is a real, standard ICAO with no IEM
   record under "KPBI" or "PBI" at all. It is archived under the
   unrelated legacy id "DJT" instead, confirmed against IEM's station
   metadata API. Also aliased in src/heat/asos.py for the live-obs
   fetch, since string-transform guesses like "strip the K" cannot
   derive this one.
3. IEM's daily-summary feed can contain corrupted values. One station's
   max_temp_f once read 614.0F for a single day; several others had a
   single year 15-40F above every other year on record. A single
   corrupted year is enough to distort a GEV fit into an implausible,
   unbounded-looking heavy tail. Confirmed live on KYUM: two roughly
   140F "years" turned an otherwise sane fit into one that predicted
   180F+ at a 500-year return period. filter_annual_max_outliers exists
   specifically to catch this before it reaches fit_gev.
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

STATION_LOOKUP_URL = "https://mesonet.agron.iastate.edu/api/1/station/{}.json"
DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
USER_AGENT = "heat-wave-tracker/0.1 (portfolio/research use)"

_MANUAL_OVERRIDES = {"KPBI": "DJT"}  # see module Gotcha 2


def resolve_iem_station(icao: str) -> dict | None:
    """Find the ASOS network and station id IEM actually uses for an ICAO code.

    Tries the ICAO as-is, then with a leading "K" stripped (the common
    first-order-station quirk, see module Gotcha 1), then a small
    manual override table for the handful of stations that quirk does
    not cover (see module Gotcha 2).

    Parameters
    ----------
    icao : str
        4-letter ICAO station code, e.g. "KDCA".

    Returns
    -------
    dict or None
        None if no "*_ASOS" network record exists under any candidate
        id. Otherwise a dict with:

        iem_id : str
            The id IEM actually indexes this station under.
        network : str
            The ASOS network name, e.g. "VA_ASOS".
        archive_begin : str or None
            ISO timestamp of the earliest available record, if IEM
            reports one.
    """
    candidates = [icao]
    if icao.startswith("K") and len(icao) == 4:
        candidates.append(icao[1:])
    if icao in _MANUAL_OVERRIDES:
        candidates.append(_MANUAL_OVERRIDES[icao])

    for candidate in candidates:
        try:
            resp = requests.get(STATION_LOOKUP_URL.format(candidate),
                                headers={"User-Agent": USER_AGENT}, timeout=20)
            resp.raise_for_status()
        except Exception:
            continue
        rows = resp.json().get("data", [])
        asos_rows = [r for r in rows if r.get("network", "").endswith("_ASOS")]
        if asos_rows:
            # a station can have more than one *_ASOS row (rare network
            # reassignments) - prefer the one with the longest record
            best = min(asos_rows, key=lambda r: r.get("archive_begin") or "9999")
            return {"iem_id": candidate, "network": best["network"],
                    "archive_begin": best.get("archive_begin")}
    return None


def fetch_daily_max_temp(iem_id: str, network: str, start_year: int, end_year: int,
                         max_retries: int = 4) -> pd.DataFrame:
    """Daily max temperature for one station over a range of years.

    Retries with exponential backoff on transient failures, since IEM
    rate-limits under load (hit live during earlier work on this
    project).

    Parameters
    ----------
    iem_id : str
        IEM's station id, from resolve_iem_station, not necessarily the
        ICAO code.
    network : str
        IEM ASOS network name, from resolve_iem_station.
    start_year, end_year : int
        Inclusive year range to fetch.
    max_retries : int, optional
        Retry attempts before giving up. Default 4.

    Returns
    -------
    pd.DataFrame
        Empty on repeated failure, the caller should treat that as
        "skip this station," not fatal. Otherwise has columns:

        day : datetime64
            Calendar date.
        max_temp_f : float
            Daily maximum temperature, degrees F.
    """
    params = {
        "station": iem_id, "network": network,
        "year1": start_year, "month1": 1, "day1": 1,
        "year2": end_year,   "month2": 12, "day2": 31,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(DAILY_URL, params=params,
                                headers={"User-Agent": USER_AGENT}, timeout=60)
            resp.raise_for_status()
            text = resp.text.strip()
            if text.startswith("ERROR"):
                print(f"[historical] {iem_id}/{network}: {text}")
                return pd.DataFrame()
            df = pd.read_csv(io.StringIO(text))
            df.columns = [c.strip() for c in df.columns]
            df["day"] = pd.to_datetime(df["day"], errors="coerce")
            df["max_temp_f"] = pd.to_numeric(df["max_temp_f"], errors="coerce")
            return df.dropna(subset=["day", "max_temp_f"])
        except Exception as exc:
            wait = 2 ** attempt
            print(f"[historical] {iem_id} attempt {attempt + 1} failed ({exc}), retrying in {wait}s")
            time.sleep(wait)
    return pd.DataFrame()


def annual_max_temp(daily_df: pd.DataFrame) -> pd.Series:
    """Reduce a daily temperature record to one annual maximum per calendar year.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of fetch_daily_max_temp.

    Returns
    -------
    pd.Series
        Index is the year (int), values are the annual max temperature,
        degrees F.
    """
    if daily_df.empty:
        return pd.Series(dtype=float)
    s = daily_df.set_index("day")["max_temp_f"]
    return s.groupby(s.index.year).max()


# just above the verified all-time US temperature record (129.9F, Death
# Valley, Aug 2020) - anything above this at any CONUS airport station is
# not a real observation
_PHYSICAL_BOUND_F = 130.0
_ROBUST_Z_THRESHOLD = 6.0


def filter_annual_max_outliers(annual_max: pd.Series) -> pd.Series:
    """Drop years whose annual max temperature is not physically plausible.

    See module Gotcha 3 for why this exists. Two checks, either one
    enough to drop a year:

    - an absolute physical bound (130F), catches gross errors outright
    - a robust z-score against the station's own other years (median
      and MAD, not mean and std, so a single bad year cannot inflate
      the very spread used to judge itself), catches subtler errors
      that are still far below 130F but clearly impossible for that
      specific station's climate, e.g. a 118F "summer max" in Bangor,
      Maine

    Parameters
    ----------
    annual_max : pd.Series
        Output of annual_max_temp, one value per year.

    Returns
    -------
    pd.Series
        Same as annual_max with implausible years removed. Returned
        unchanged if fewer than 3 years are present, since a
        meaningful median/MAD needs at least that many.
    """
    if len(annual_max) < 3:
        return annual_max
    median = annual_max.median()
    mad = (annual_max - median).abs().median()
    robust_z = (annual_max - median).abs() / (1.4826 * mad + 1e-9)
    keep = (annual_max <= _PHYSICAL_BOUND_F) & (robust_z <= _ROBUST_Z_THRESHOLD)
    return annual_max[keep]
