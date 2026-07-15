"""
src/heat/asos.py
=================
Fetch recent ASOS observations from Iowa Environmental Mesonet (IEM) for
a single station. Used by the station panel in app.py, live fetch on
click, cached in-process.

Returns a tidy pandas DataFrame with temp_c and dewpoint_c columns.

Gotchas:

1. report_type=3 (routine hourly METAR, reported around :53 past the
   hour) is used deliberately, not report_type=1 or 2. report_type=1
   looks like it should mean "everything" but actually returns a
   separate 5-minute AWOS feed with temp/dewpoint blank on at least one
   station tested here, not useful. report_type=2 (SPECI, irregular
   reports issued between routine ones to catch fast-changing
   conditions) was included at one point to catch conditions sooner,
   but that mixed irregular timestamps into what should be a clean
   hourly series. Routine-only (3) gives exact, evenly spaced hourly
   points, which is what the bias correction pairs against and what
   the time series chart plots.
2. KPBI (West Palm Beach Intl) is a real, standard ICAO code, but IEM's
   ASOS network has no record under "KPBI" or "PBI" at all. It is
   archived under the legacy 3-letter id "DJT" instead, confirmed
   against IEM's station metadata API while building the historical
   climate archive. fetch_station_obs("KPBI") silently returned 0 obs
   before _STATION_ALIASES was added. Kept as a fetch-only alias here,
   not a stations.py rename, since "KPBI" is the correct code to show
   users, unlike KHTW/KFGZ, which were themselves wrong codes and got
   corrected directly in src/heat/stations.py instead.
"""
from __future__ import annotations

import io
import requests
import pandas as pd

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
USER_AGENT = "heat-wave-tracker/0.1 (portfolio/research use)"

_STATION_ALIASES = {"KPBI": "DJT"}  # see module Gotcha 2


def _f_to_c(series: pd.Series) -> pd.Series:
    """Fahrenheit to Celsius, coercing non-numeric values (e.g. IEM's "M"
    missing-value marker) to None instead of raising."""
    return pd.to_numeric(series, errors="coerce").apply(
        lambda x: (x - 32.0) * 5.0 / 9.0 if pd.notna(x) else None
    )


def fetch_station_obs(station_id: str, hours: int = 72) -> pd.DataFrame:
    """Fetch the last `hours` of ASOS observations for one station from IEM.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code, e.g. "KDCA". Resolved through
        _STATION_ALIASES first (see module Gotcha 2).
    hours : int, optional
        How many hours back from now to fetch. Default 72.

    Returns
    -------
    pd.DataFrame
        Empty on any fetch error or if IEM returns no usable rows.
        Otherwise has columns:

        valid_utc : Timestamp
            Tz-aware UTC observation time.
        temp_c : float
            2m temperature, degrees C.
        dewpoint_c : float
            2m dewpoint, degrees C.
        rh : float
            Relative humidity, percent.
        wind_spd_kt : float
            Wind speed, knots.
    """
    end   = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(hours=hours)
    iem_station = _STATION_ALIASES.get(station_id, station_id)

    params = {
        "station": iem_station,
        "data":    "tmpf,dwpf,relh,sknt",
        "year1":   start.year,  "month1": start.month,  "day1": start.day,
        "hour1":   start.hour,
        "year2":   end.year,    "month2": end.month,    "day2": end.day,
        "hour2":   end.hour,
        "tz":      "UTC",
        "format":  "onlycomma",
        "latlon":  "no",
        "missing": "M",
        "trace":   "T",
        "report_type": "3",  # routine hourly METAR only, see module Gotcha 1
    }

    try:
        resp = requests.get(
            IEM_URL, params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[asos] IEM fetch failed for {station_id}: {exc}")
        return pd.DataFrame()

    lines = [ln for ln in resp.text.strip().split("\n")
             if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df.columns = [c.strip() for c in df.columns]

    if "valid" not in df.columns:
        return pd.DataFrame()

    df["valid_utc"]   = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df["temp_c"]      = _f_to_c(df.get("tmpf", pd.Series(dtype=float)))
    df["dewpoint_c"]  = _f_to_c(df.get("dwpf", pd.Series(dtype=float)))
    df["rh"]          = pd.to_numeric(df.get("relh", pd.Series(dtype=float)), errors="coerce")
    df["wind_spd_kt"] = pd.to_numeric(df.get("sknt", pd.Series(dtype=float)), errors="coerce")

    df = df[df["valid_utc"].notna() & df["temp_c"].notna()].copy()
    return df.reset_index(drop=True)
