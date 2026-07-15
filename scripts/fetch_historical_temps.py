"""
scripts/fetch_historical_temps.py
==================================
Bulk-fetch daily max temperature (1972-2025) for all 165 stations from IEM
and reduce to one annual-maximum-temperature series per station, saved to
data/climate_extremes.parquet. This is the block-maxima archive the GEV
return-period estimate is fit against.

Requires data/iem_station_map.json (run scripts/build_iem_station_map.py
first). Station record length varies a lot in practice (see that map's
"archive_begin" field) - some stations only go back to the 1980s/90s, not
the full 54 years. That's recorded per station (n_years) rather than
silently padded or hidden, since a GEV fit on a short record deserves a
visibly different confidence level than one on 54 years.

One-time/offline script - not run by the live app. Takes several minutes
(165 stations, one request each, with courtesy rate-limiting and real
retry/backoff - IEM rate-limits under load, hit live during earlier work
on this project).

Usage
-----
    python scripts/fetch_historical_temps.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.heat.stations import MAJOR_CONUS_STATIONS
from src.heat.historical import fetch_daily_max_temp, annual_max_temp, filter_annual_max_outliers

STATION_MAP_PATH = Path("data") / "iem_station_map.json"
OUT_PATH = Path("data") / "climate_extremes.parquet"
START_YEAR, END_YEAR = 1972, 2025


def main():
    if not STATION_MAP_PATH.exists():
        print(f"[historical] {STATION_MAP_PATH} not found - run "
              f"scripts/build_iem_station_map.py first.")
        sys.exit(1)
    station_map = json.loads(STATION_MAP_PATH.read_text())

    rows = []
    misses = []
    for i, stn in enumerate(MAJOR_CONUS_STATIONS, start=1):
        sid = stn["id"]
        resolved = station_map.get(sid)
        print(f"  [{i}/{len(MAJOR_CONUS_STATIONS)}] {sid} ...", end=" ", flush=True)
        if resolved is None:
            print("no IEM mapping - skipped")
            misses.append(sid)
            continue

        daily = fetch_daily_max_temp(resolved["iem_id"], resolved["network"],
                                     START_YEAR, END_YEAR)
        annual = annual_max_temp(daily)
        if annual.empty:
            print("no data")
            misses.append(sid)
            time.sleep(1.5)
            continue

        n_before = len(annual)
        annual = filter_annual_max_outliers(annual)
        n_dropped = n_before - len(annual)
        if annual.empty:
            print("all years dropped as implausible")
            misses.append(sid)
            time.sleep(1.5)
            continue

        for year, temp_f in annual.items():
            rows.append({"station": sid, "year": int(year), "annual_max_temp_f": float(temp_f)})
        dropped_note = f", {n_dropped} dropped as implausible" if n_dropped else ""
        print(f"{len(annual)} years ({int(annual.index.min())}-{int(annual.index.max())}){dropped_note}")
        time.sleep(1.5)  # courtesy rate-limiting - IEM has 429'd this project before

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    n_stations = df["station"].nunique() if not df.empty else 0
    print(f"\n[historical] {n_stations}/{len(MAJOR_CONUS_STATIONS)} stations, "
          f"{len(df)} station-years -> {OUT_PATH}")
    if misses:
        print(f"[historical] no data for: {', '.join(misses)}")


if __name__ == "__main__":
    main()
