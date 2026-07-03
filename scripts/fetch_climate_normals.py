"""
scripts/fetch_climate_normals.py
==================================
Bulk-fetch NWS Daily Climate Report (CLI) data for all 165 stations and
save to data/climate_normals.json.

Not every station has a CLI product (it's issued per NWS office policy),
so this logs a coverage summary rather than treating misses as failures.
Run on the same schedule as the GFS refresh - a CLI report only updates
once a day, so there's no benefit to fetching it live per station click.

Usage
-----
    python scripts/fetch_climate_normals.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.heat.stations import MAJOR_CONUS_STATIONS
from src.heat.climate import fetch_climate_summary

OUT_PATH = Path("data") / "climate_normals.json"


def main():
    results = {}
    misses = []

    for i, stn in enumerate(MAJOR_CONUS_STATIONS, start=1):
        sid = stn["id"]
        print(f"  [{i}/{len(MAJOR_CONUS_STATIONS)}] {sid} ...", end=" ", flush=True)
        summary = fetch_climate_summary(sid)
        if summary is None:
            print("no data")
            misses.append(sid)
        else:
            print(f"{summary['high_actual_f']}F ({summary['high_departure_f']:+d} vs normal)")
            results[sid] = summary
        time.sleep(0.2)  # light rate-limiting courtesy to IEM

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=1))

    print(f"\n[climate] {len(results)}/{len(MAJOR_CONUS_STATIONS)} stations covered "
          f"-> {OUT_PATH}")
    if misses:
        print(f"[climate] no CLI report for: {', '.join(misses)}")


if __name__ == "__main__":
    main()
