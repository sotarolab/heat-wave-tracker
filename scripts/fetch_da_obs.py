"""Fetch the CONUS ASOS observation snapshot for a surface-analysis cycle.

Driver only — logic lives in src/heat/da/asos_network.py.

Usage:
    python scripts/fetch_da_obs.py --build-station-table   # once / rarely
    python scripts/fetch_da_obs.py --analysis-time 2026-08-24T12
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heat.da import asos_network, split  # noqa: E402
from src.heat.da.config import DA_DATA_DIR, OBS_MATCH_TOLERANCE_MIN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-station-table", action="store_true",
                        help="(Re)build and cache the IEM CONUS station table, "
                             "then exit.")
    parser.add_argument("--analysis-time", default=None,
                        help="UTC synoptic hour, e.g. 2026-08-24T12. "
                             "Default: most recent synoptic hour.")
    parser.add_argument("--out", default=None,
                        help="Output parquet path (default under data/da/).")
    args = parser.parse_args()

    if args.build_station_table:
        table = asos_network.fetch_station_table()
        asos_network.save_station_table(table)
        assim, withheld = split.partition(table["station_id"])
        print(f"stations: {len(table)}  "
              f"(assimilated {len(assim)}, withheld {len(withheld)})")
        return

    analysis_time = (
        pd.Timestamp(args.analysis_time)
        if args.analysis_time
        else pd.Timestamp.utcnow().tz_localize(None).floor("6h")
    ).tz_localize("UTC")

    table = asos_network.load_station_table()
    tol = pd.Timedelta(minutes=OBS_MATCH_TOLERANCE_MIN)
    obs, diag = asos_network.fetch_obs_window(
        table["station_id"], analysis_time - tol, analysis_time + tol
    )
    print(f"fetch: {diag}")
    obs, rejects = asos_network.qc_obs(obs)
    print(f"qc rejects: {rejects}")
    selected = asos_network.select_analysis_obs(obs, analysis_time)
    print(f"selected: {len(selected)} stations at {analysis_time}")

    out = Path(args.out) if args.out else (
        DA_DATA_DIR / f"obs_{analysis_time:%Y%m%dT%H}Z.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
