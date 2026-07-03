"""
scripts/fetch_gfs_conus.py
===========================
Pre-fetch GFS CONUS data for the heat wave tracker.

Downloads T2m + Td2m from NOAA AWS via Herbie, computes heat index, and
saves to data/conus_heat_tracker.nc. Defaults to every 2 hours (61 steps,
~40 MB) — a middle ground between the original 4 slots/day and full hourly
(121 steps, ~80 MB, over GitHub's 50 MB soft file-size limit).

This script runs locally. Commit the output file so Render serves from it
without a cold-start download.

Usage
-----
    # Latest available GFS cycle (default: 61 steps, F000–F120 every 2 h)
    python scripts/fetch_gfs_conus.py

    # Specific init time
    python scripts/fetch_gfs_conus.py --init "2026-07-02 00:00"

    # Full hourly (121 steps, ~80 MB — exceeds GitHub's 50 MB soft limit)
    python scripts/fetch_gfs_conus.py --step 1

    # Coarser/faster fetch (21 steps, F000–F120 every 6 h) — smaller file
    python scripts/fetch_gfs_conus.py --step 6

    # Force re-download even if file already exists
    python scripts/fetch_gfs_conus.py --overwrite
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.heat.gfs_conus import fetch_gfs_conus, DEFAULT_OUT


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GFS CONUS T2m + Td2m and compute heat stress variables."
    )
    parser.add_argument(
        "--init", default=None, metavar="DATETIME",
        help="GFS init time, e.g. '2026-07-02 00:00'. Default: latest available.",
    )
    parser.add_argument(
        "--hours", type=int, default=120,
        help="Forecast horizon in hours (default: 120).",
    )
    parser.add_argument(
        "--step", type=int, default=2,
        help="Interval between lead times in hours (default: 2).",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), metavar="PATH",
        help=f"Output NetCDF path (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download even if the output file already exists.",
    )
    args = parser.parse_args()

    import pandas as pd
    init_dt = pd.Timestamp(args.init) if args.init else None
    fxx_range = range(0, args.hours + 1, args.step)

    fetch_gfs_conus(
        init_dt   = init_dt,
        fxx_range = fxx_range,
        out_path  = Path(args.out),
        overwrite = args.overwrite,
    )


if __name__ == "__main__":
    main()
