"""
scripts/log_corrections.py
===========================
Applies the trained t2m correction to the freshly fetched GFS forecast
and logs one row per (station, validated step) to ml_corrections
(issue #16). Runs in the refresh workflow directly after
log_forecast_obs.py; the app's verification panel scores these rows
against observations as they arrive, so the skill record accumulates
out of sample by construction.

Only leads at or below correction.VALIDATED_LEAD_H are logged; the
model's measured scope ends there (docs/ml/EXPERIMENTS.md). Each row
also stores a trailing per-station offset baseline computed at the
same moment, so the panel compares three quantities scored on
identical rows: raw forecast, offset baseline, model.

Idempotent: the primary key includes gfs_init_time and model_version,
and inserts are ON CONFLICT DO NOTHING, so re-runs never duplicate and
a future model version writes new rows instead of overwriting the old
version's record.

Requires NEON_DATABASE_URL; exits quietly without it, like the pairs
logger.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.heat import correction
from src.heat.gfs_conus import DEFAULT_OUT, load_or_fetch
from src.heat.stations import MAJOR_CONUS_STATIONS


def main() -> None:
    if not os.environ.get("NEON_DATABASE_URL"):
        print("[log_corrections] NEON_DATABASE_URL not set - skipping")
        return
    ds = load_or_fetch(DEFAULT_OUT)
    if ds is None:
        print("[log_corrections] no GFS data - skipping")
        return
    model = correction.load_model()
    if model is None:
        print("[log_corrections] no model - skipping")
        return

    init = pd.Timestamp(ds.attrs["gfs_init"]).tz_localize("UTC")
    hist7, offset = correction.error_history(init)
    print(f"[log_corrections] init {init}  history: {len(hist7)} stations")

    valid = pd.DatetimeIndex(ds.time.values).tz_localize("UTC")
    mask = correction.validated_mask(valid, init)
    if not mask.any():
        print("[log_corrections] no steps within validated lead - skipping")
        return

    rows = []
    for stn in MAJOR_CONUS_STATIONS:
        sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")
        t = ds["t2m"].sel(**sel).values[mask]
        td = ds["td2m"].sel(**sel).values[mask]
        X = correction.build_features(stn, valid[mask], t, td, init,
                                      hist7.get(stn["id"], 0.0))
        corr = model.predict(X)
        base = offset.get(stn["id"])
        for vt, lead, raw, c in zip(valid[mask], X.lead_h, t, corr):
            if np.isnan(raw):
                continue
            rows.append((stn["id"], vt.to_pydatetime(), init.to_pydatetime(),
                         correction.MODEL_VERSION, float(lead), float(raw),
                         float(raw + base) if base is not None else None,
                         float(raw + c)))
    print(f"[log_corrections] {len(rows)} rows for {mask.sum()} steps")
    if not rows:
        return

    import psycopg2
    import psycopg2.extras
    conn = psycopg2.connect(os.environ["NEON_DATABASE_URL"])
    try:
        with conn, conn.cursor() as cur:
            # fetch=True with RETURNING gives the true insert count;
            # cur.rowcount after execute_values reflects only the final
            # page (default page_size=100), which under-reports.
            inserted = psycopg2.extras.execute_values(cur, """
                INSERT INTO ml_corrections
                    (station_id, forecast_valid_time, gfs_init_time,
                     model_version, lead_h, raw_value_c,
                     offset_baseline_c, corrected_value_c)
                VALUES %s
                ON CONFLICT (station_id, forecast_valid_time,
                             gfs_init_time, model_version) DO NOTHING
                RETURNING 1
            """, rows, fetch=True)
            print(f"[log_corrections] inserted (new rows: {len(inserted)})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
