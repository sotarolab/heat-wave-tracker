"""
src/heat/asos.py
================
Fetch recent ASOS observations from Iowa Environmental Mesonet (IEM)
for a single station. Used by the station panel in app.py (live fetch
on click, cached in-process).

Returns a tidy pandas DataFrame with temp_c and dewpoint_c columns.
"""
from __future__ import annotations

import io
import requests
import pandas as pd

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
USER_AGENT = "heat-wave-tracker/0.1 (portfolio/research use)"


def _f_to_c(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").apply(
        lambda x: (x - 32.0) * 5.0 / 9.0 if pd.notna(x) else None
    )


def _coerce(val) -> float | None:
    s = str(val).strip()
    if s in ("", "M", "T", "None", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_station_obs(station_id: str, hours: int = 72) -> pd.DataFrame:
    """
    Fetch the last `hours` of ASOS observations for one station from IEM.

    Returns DataFrame with columns:
        valid_utc   - tz-aware UTC timestamp
        temp_c      - 2m temperature (°C)
        dewpoint_c  - 2m dewpoint (°C)
        rh          - relative humidity (%)
        wind_spd_kt - wind speed (knots)
    Returns an empty DataFrame on error.
    """
    end   = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(hours=hours)

    params = {
        "station": station_id,
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
        # 3 = routine hourly METAR, 2 = SPECI (issued between routine
        # reports when conditions change quickly - e.g. a front moving
        # through). Routine-only meant a fast-changing event could be
        # invisible for up to an hour; requesting both catches it as soon
        # as the station itself reports it. (report_type=1 looked like it
        # should mean "everything" but actually returns a separate 5-minute
        # AWOS feed with temp/dewpoint blank on this station - not useful.)
        "report_type": ["2", "3"],
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
