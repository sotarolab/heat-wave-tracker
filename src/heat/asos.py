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


# Fields requested from IEM. The first four feed the existing columns; the
# rest are the observed regime variables (wind, gusts, sky cover, ceiling,
# pressure, visibility, present weather) that the forecast-error analysis
# needs: calm + clear nights are where the model's nocturnal warm bias
# lives, and sky cover is invisible to the forecast pairing otherwise.
IEM_FIELDS = ["tmpf", "dwpf", "relh", "sknt",
              "gust", "drct", "mslp", "alti", "vsby",
              "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4",
              "wxcodes"]

# METAR sky-cover codes to fractional cover. VV (vertical visibility,
# obscured sky) counts as overcast. The observation's sky cover is the
# MAXIMUM across reported layers.
_SKY_FRACTION = {"CLR": 0.0, "SKC": 0.0, "NSC": 0.0, "NCD": 0.0,
                 "FEW": 0.25, "SCT": 0.5, "BKN": 0.75, "OVC": 1.0, "VV": 1.0}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce")


def parse_iem_csv(text: str) -> pd.DataFrame:
    """Parse IEM's onlycomma CSV into the tidy observation frame.

    Split out from the fetch so the parsing is unit-testable without the
    network. Returns an empty frame if there is no usable data.
    """
    lines = [ln for ln in text.strip().split("\n") if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df.columns = [c.strip() for c in df.columns]
    if "valid" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["valid_utc"]    = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    out["temp_c"]       = _f_to_c(_num(df, "tmpf"))
    out["dewpoint_c"]   = _f_to_c(_num(df, "dwpf"))
    out["rh"]           = _num(df, "relh")
    out["wind_spd_kt"]  = _num(df, "sknt")
    out["wind_gust_kt"] = _num(df, "gust")
    out["wind_dir_deg"] = _num(df, "drct")
    # Sea-level pressure when reported, else altimeter (inHg) converted.
    mslp = _num(df, "mslp"); alti = _num(df, "alti") * 33.8639
    out["pressure_hpa"] = mslp.where(mslp.notna(), alti)
    out["visibility_mi"] = _num(df, "vsby")

    layers = [df.get(f"skyc{i}", pd.Series(index=df.index, dtype=object)).astype(str).str.strip().str.upper()
              for i in range(1, 5)]
    frac = pd.concat([lyr.map(_SKY_FRACTION) for lyr in layers], axis=1)
    out["sky_cover"] = frac.max(axis=1, skipna=True)
    out.loc[frac.isna().all(axis=1), "sky_cover"] = float("nan")

    # Ceiling: lowest layer that is broken, overcast, or obscured.
    ceil = pd.Series(float("nan"), index=df.index)
    for i in range(4, 0, -1):
        code = layers[i - 1]
        lvl = _num(df, f"skyl{i}")
        hit = code.isin(["BKN", "OVC", "VV"]) & lvl.notna()
        ceil = ceil.where(~hit, lvl)
    out["ceiling_ft"] = ceil

    wx = df.get("wxcodes", pd.Series(index=df.index, dtype=object)).astype(str).str.strip()
    out["wx_codes"] = wx.where(~wx.isin(["M", "nan", ""]), None)

    out = out[out["valid_utc"].notna() & out["temp_c"].notna()].copy()
    return out.reset_index(drop=True)


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
        wind_gust_kt, wind_dir_deg, pressure_hpa, visibility_mi,
        sky_cover (0 to 1, max over layers), ceiling_ft, wx_codes
            Observed regime variables; NaN/None where not reported.
    """
    end   = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(hours=hours)
    iem_station = _STATION_ALIASES.get(station_id, station_id)

    params = {
        "station": iem_station,
        "data":    ",".join(IEM_FIELDS),
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

    return parse_iem_csv(resp.text)
