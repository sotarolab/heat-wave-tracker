"""
src/heat/da/asos_network.py
===========================
CONUS-wide ASOS observation ingest for the surface analysis: build the
network station table from IEM metadata, fetch a short observation window
for all of it around an analysis time, QC, and reduce to one observation
per station.

This deliberately does NOT reuse src/heat/asos.py's fetch_station_obs():
that function is shaped for the station panel (one station, 72 h, live on
click). The analysis needs the transpose — thousands of stations, one
narrow time window — which the same IEM CGI serves efficiently when
stations are batched into one request. What IS reused: report_type=3
(routine METARs only, see asos.py Gotcha 1 for why), the CGI parameter
conventions, and °C units.

Gotchas:

1. IEM organizes ASOS metadata as one network per state ("IA_ASOS", ...),
   each served as GeoJSON. There is no single all-CONUS endpoint, so
   fetch_station_table() loops the 48 CONUS states + DC and concatenates.
   That is 49 small metadata requests — fine for a build-time table that
   is then cached to data/da/asos_stations.json and refreshed rarely, not
   something to do per cycle.
2. The obs CGI accepts repeated station= parameters, but URLs have length
   limits and IEM's worker time is bounded, so fetch_obs_window() batches
   (default 100 stations/request). Batch failures degrade gracefully: a
   failed batch loses those stations for this cycle and is reported in the
   returned diagnostics, never raised — one flaky request must not kill an
   operational analysis cycle that has 95% of its observations in hand.
3. QC rejects are counted by rule, not silently dropped, and the counts
   travel with the result. If a station network suddenly loses half its
   obs to QC, that fact should surface in cycle logs, not be discovered
   months later in a verification regression.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from src.heat.da.config import (
    OBS_MATCH_TOLERANCE_MIN,
    STATION_TABLE_PATH,
    TEMP_QC_BOUNDS_C,
)

IEM_OBS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_NETWORK_GEOJSON_URL = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
USER_AGENT = "heat-wave-tracker/0.1 (research use)"

CONUS_STATES = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
]


def _f_to_c(series: pd.Series) -> pd.Series:
    """Fahrenheit to Celsius, coercing IEM's "M" missing marker to NaN."""
    return (pd.to_numeric(series, errors="coerce") - 32.0) * 5.0 / 9.0


# -------------------------------------------------------- station table ---


def fetch_station_table(states=None, timeout: int = 30) -> pd.DataFrame:
    """Build the CONUS ASOS station table from IEM per-state network metadata.

    Returns a DataFrame with columns: station_id, name, lat, lon, network.
    States whose metadata request fails are skipped with a printed notice
    (see Gotcha 2's degrade-gracefully rationale). Duplicate station ids
    across networks keep the first occurrence.
    """
    frames = []
    for state in states or CONUS_STATES:
        network = f"{state}_ASOS"
        try:
            resp = requests.get(
                IEM_NETWORK_GEOJSON_URL.format(network=network),
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            resp.raise_for_status()
            features = resp.json().get("features", [])
        except Exception as exc:
            print(f"[da.asos] station metadata failed for {network}: {exc}")
            continue
        rows = []
        for feat in features:
            props = feat.get("properties", {})
            lon, lat = feat.get("geometry", {}).get("coordinates", (None, None))
            rows.append(
                {
                    "station_id": props.get("sid"),
                    "name": props.get("sname"),
                    "lat": lat,
                    "lon": lon,
                    "network": network,
                }
            )
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["station_id", "name", "lat", "lon", "network"])
    table = pd.concat(frames, ignore_index=True).dropna(subset=["station_id", "lat", "lon"])
    return table.drop_duplicates(subset="station_id", keep="first").reset_index(drop=True)


def load_station_table(path=STATION_TABLE_PATH) -> pd.DataFrame:
    """Load the cached station table written by scripts/fetch_da_obs.py."""
    return pd.read_json(path, orient="records")


def save_station_table(table: pd.DataFrame, path=STATION_TABLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_json(path, orient="records", indent=1)


# ---------------------------------------------------------- obs fetching ---


def fetch_obs_window(
    station_ids,
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int = 100,
    timeout: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Fetch temp/dewpoint obs for many stations over [start, end] UTC.

    Returns (obs, diagnostics): obs has columns station_id, valid_utc,
    temp_c, dewpoint_c; diagnostics counts requested/returned stations and
    failed batches (Gotcha 2).
    """
    ids = list(station_ids)
    frames = []
    failed_batches = 0
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        params = [
            ("data", "tmpf"),
            ("data", "dwpf"),
            ("year1", start.year), ("month1", start.month),
            ("day1", start.day), ("hour1", start.hour), ("minute1", start.minute),
            ("year2", end.year), ("month2", end.month),
            ("day2", end.day), ("hour2", end.hour), ("minute2", end.minute),
            ("tz", "UTC"),
            ("format", "onlycomma"),
            ("latlon", "no"),
            ("missing", "M"),
            ("trace", "T"),
            ("report_type", "3"),  # routine METARs only, see asos.py Gotcha 1
        ] + [("station", s) for s in batch]
        try:
            resp = requests.get(
                IEM_OBS_URL, params=params,
                headers={"User-Agent": USER_AGENT}, timeout=timeout,
            )
            resp.raise_for_status()
            frame = pd.read_csv(io.StringIO(resp.text), comment="#")
        except Exception as exc:
            print(f"[da.asos] obs batch {i // batch_size} failed: {exc}")
            failed_batches += 1
            continue
        if not frame.empty:
            frames.append(frame)
    diagnostics = {
        "stations_requested": len(ids),
        "failed_batches": failed_batches,
        "batches": -(-len(ids) // batch_size) if ids else 0,
    }
    if not frames:
        diagnostics["stations_returned"] = 0
        return pd.DataFrame(columns=["station_id", "valid_utc", "temp_c", "dewpoint_c"]), diagnostics
    raw = pd.concat(frames, ignore_index=True)
    obs = pd.DataFrame(
        {
            "station_id": raw["station"],
            "valid_utc": pd.to_datetime(raw["valid"], utc=True),
            "temp_c": _f_to_c(raw["tmpf"]),
            "dewpoint_c": _f_to_c(raw["dwpf"]),
        }
    )
    diagnostics["stations_returned"] = obs["station_id"].nunique()
    return obs, diagnostics


# ------------------------------------------------------------------- QC ---


def qc_obs(obs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Physical-plausibility QC. Returns (passed, reject_counts_by_rule).

    Rules (each counted separately, Gotcha 3):
      missing_temp     temp_c is NaN (dewpoint may be legitimately missing;
                       a temp-only ob is still assimilable for T)
      temp_bounds      temp_c outside TEMP_QC_BOUNDS_C
      dewpoint_bounds  dewpoint_c present but outside TEMP_QC_BOUNDS_C
                       (dewpoint is nulled, row kept)
      supersaturated   dewpoint_c > temp_c + 0.5 (dewpoint nulled, row
                       kept; 0.5 °C slack because independently rounded
                       METAR T/Td can legitimately cross by rounding)
      duplicate        same (station_id, valid_utc) reported twice
    """
    counts = {}
    out = obs.copy()

    dupes = out.duplicated(subset=["station_id", "valid_utc"], keep="first")
    counts["duplicate"] = int(dupes.sum())
    out = out[~dupes]

    missing = out["temp_c"].isna()
    counts["missing_temp"] = int(missing.sum())
    out = out[~missing]

    lo, hi = TEMP_QC_BOUNDS_C
    bad_t = (out["temp_c"] < lo) | (out["temp_c"] > hi)
    counts["temp_bounds"] = int(bad_t.sum())
    out = out[~bad_t]

    bad_td = out["dewpoint_c"].notna() & ((out["dewpoint_c"] < lo) | (out["dewpoint_c"] > hi))
    counts["dewpoint_bounds"] = int(bad_td.sum())
    out.loc[bad_td, "dewpoint_c"] = float("nan")

    supersat = out["dewpoint_c"].notna() & (out["dewpoint_c"] > out["temp_c"] + 0.5)
    counts["supersaturated"] = int(supersat.sum())
    out.loc[supersat, "dewpoint_c"] = float("nan")

    return out.reset_index(drop=True), counts


def select_analysis_obs(
    obs: pd.DataFrame,
    analysis_time: pd.Timestamp,
    tolerance_min: int = OBS_MATCH_TOLERANCE_MIN,
) -> pd.DataFrame:
    """One observation per station: nearest analysis_time within tolerance.

    Ties (equidistant before/after) keep the earlier observation, purely so
    the choice is deterministic.
    """
    if obs.empty:
        return obs.copy()
    out = obs.copy()
    out["offset_s"] = (out["valid_utc"] - analysis_time).dt.total_seconds()
    out["abs_offset_s"] = out["offset_s"].abs()
    out = out[out["abs_offset_s"] <= tolerance_min * 60]
    out = out.sort_values(["station_id", "abs_offset_s", "offset_s"])
    out = out.groupby("station_id", as_index=False).first()
    return out.drop(columns=["offset_s", "abs_offset_s"])
