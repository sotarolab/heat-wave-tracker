"""
scripts/log_forecast_obs.py
============================
Appends today's newly-available (forecast, observed) pairs to a Neon
Postgres table (forecast_obs_pairs), so forecast-error history accumulates
across days instead of being recomputed fresh every 6 hours and thrown
away, as the live app's same-day bias correction does. Meant to run as an
added step in the existing GFS refresh workflow, right after a fresh
forecast is fetched.

Idempotent: inserts are ON CONFLICT (station_id, metric, forecast_valid_time)
DO NOTHING, so re-running against overlapping ASOS history (each run fetches
the last 72h) never creates duplicates - only genuinely new paired points
get added each time.

Deliberately a separate pairing pass from src.heat.bias._paired_today,
which is scoped to a single calendar day (built for "how's today doing"
in the live app's interactive bias-correction control) - this script wants
every paired point across the last 72h of ASOS history each run, not just
today's, so the accumulated table doesn't have gaps depending on which
day/hour a given scheduled run happens to land on.

Requires the NEON_DATABASE_URL environment variable (a Postgres connection
string) - never hardcoded, never logged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.heat.asos import fetch_station_obs
from src.heat.compute import heat_index_array
from src.heat.gfs_conus import DEFAULT_OUT, load_or_fetch
from src.heat.stations import MAJOR_CONUS_STATIONS

MATCH_TOLERANCE = pd.Timedelta(minutes=45)  # same tolerance as bias._paired_today
ASOS_HOURS = 72

# metric -> (GFS variable name, observed-value column in fetch_station_obs's output)
# "hi" observed values aren't in the raw ASOS feed - computed below from
# observed temp_c + dewpoint_c with the same formula the forecast side uses,
# same approach app.py's station chart already takes for the live comparison.
METRICS = {"t2m": "temp_c", "td2m": "dewpoint_c", "hi": "hi_c"}


def _paired_recent(forecast_series: pd.Series, obs_df: pd.DataFrame,
                   obs_col: str) -> pd.DataFrame:
    """All (forecast, observed) pairs across obs_df's full time range,
    nearest-matched within MATCH_TOLERANCE - see module docstring for why
    this isn't just src.heat.bias._paired_today.
    """
    if obs_df.empty:
        return pd.DataFrame(columns=["time", "forecast", "observed"])
    odf = (obs_df.dropna(subset=[obs_col])[["valid_utc", obs_col]]
           .rename(columns={"valid_utc": "time", obs_col: "observed"})
           .sort_values("time"))
    if odf.empty:
        return pd.DataFrame(columns=["time", "forecast", "observed"])
    past = forecast_series[forecast_series.index <= odf["time"].max()]
    if past.empty:
        return pd.DataFrame(columns=["time", "forecast", "observed"])
    fdf = past.rename("forecast").reset_index().rename(columns={"index": "time"}).sort_values("time")
    # same pandas merge_asof unit-normalization gotcha as bias._paired_today
    fdf["time"] = fdf["time"].dt.as_unit("us")
    odf["time"] = odf["time"].dt.as_unit("us")
    return pd.merge_asof(fdf, odf, on="time", direction="nearest",
                         tolerance=MATCH_TOLERANCE).dropna(subset=["observed"])


def _collect_pairs(ds, gfs_init: pd.Timestamp) -> list[tuple]:
    """One row per (station, metric, matched timestep) across every major
    station, ready for a bulk insert."""
    rows: list[tuple] = []
    for station in MAJOR_CONUS_STATIONS:
        station_id = station["id"]
        obs_df = fetch_station_obs(station_id, hours=ASOS_HOURS)
        if obs_df.empty:
            continue

        # Real observed Feels Like, computed from actual observed temp+dewpoint
        # with the same NWS formula the forecast side uses - a validated fact
        # about what already happened, not a model output. Same approach as
        # the live app's station chart.
        has_td = obs_df["dewpoint_c"].notna()
        obs_df = obs_df.copy()
        obs_df["hi_c"] = float("nan")
        if has_td.any():
            obs_df.loc[has_td, "hi_c"] = heat_index_array(
                obs_df.loc[has_td, "temp_c"].values,
                obs_df.loc[has_td, "dewpoint_c"].values,
            )

        sel = dict(latitude=station["lat"], longitude=station["lon"], method="nearest")
        for metric, obs_col in METRICS.items():
            series = ds[metric].sel(**sel).to_series()
            # GFS times are tz-naive but represent UTC; obs_df's valid_utc is
            # tz-aware - merge_asof requires both sides to agree.
            series.index = pd.DatetimeIndex(series.index).tz_localize("UTC")
            paired = _paired_recent(series, obs_df, obs_col)
            for row in paired.itertuples(index=False):
                rows.append((
                    station_id, metric, row.time.to_pydatetime(),
                    float(row.forecast), float(row.observed),
                    gfs_init.to_pydatetime(),
                ))
    return rows


def main() -> None:
    db_url = os.environ.get("NEON_DATABASE_URL")
    if not db_url:
        print("[log_forecast_obs] NEON_DATABASE_URL not set - skipping")
        return

    ds = load_or_fetch(DEFAULT_OUT)
    if ds is None:
        print("[log_forecast_obs] No GFS data loaded - skipping")
        return
    gfs_init = pd.Timestamp(ds.attrs.get("gfs_init"))

    print("[log_forecast_obs] Collecting pairs across all stations ...")
    rows = _collect_pairs(ds, gfs_init)
    print(f"[log_forecast_obs] {len(rows)} candidate pairs collected")
    if not rows:
        return

    conn = psycopg2.connect(db_url)
    try:
        with conn, conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO forecast_obs_pairs
                    (station_id, metric, forecast_valid_time,
                     forecast_value_c, observed_value_c, gfs_init_time)
                VALUES %s
                ON CONFLICT (station_id, metric, forecast_valid_time) DO NOTHING
                """,
                rows,
            )
            print(f"[log_forecast_obs] Upserted (new rows: {cur.rowcount})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
