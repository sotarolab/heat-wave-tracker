"""
scripts/build_iem_station_map.py
=================================
Resolve all 165 stations' ICAO codes to their IEM ASOS network/station id
and save to data/iem_station_map.json. IEM's network assignments don't
change, so this is a one-time/rarely-rerun step - kept separate from the
actual historical data fetch so a resolution failure doesn't waste a slow
bulk download.

Usage
-----
    python scripts/build_iem_station_map.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.heat.stations import MAJOR_CONUS_STATIONS
from src.heat.historical import resolve_iem_station

OUT_PATH = Path("data") / "iem_station_map.json"


def main():
    results = {}
    misses = []
    for i, stn in enumerate(MAJOR_CONUS_STATIONS, start=1):
        sid = stn["id"]
        print(f"  [{i}/{len(MAJOR_CONUS_STATIONS)}] {sid} ...", end=" ", flush=True)
        resolved = resolve_iem_station(sid)
        if resolved is None:
            print("NOT FOUND")
            misses.append(sid)
        else:
            print(f"{resolved['iem_id']} / {resolved['network']} "
                  f"(since {resolved['archive_begin']})")
            results[sid] = resolved
        time.sleep(0.15)  # light rate-limiting courtesy to IEM

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=1))
    print(f"\n[historical] resolved {len(results)}/{len(MAJOR_CONUS_STATIONS)} "
          f"stations -> {OUT_PATH}")
    if misses:
        print(f"[historical] could not resolve: {', '.join(misses)}")


if __name__ == "__main__":
    main()
