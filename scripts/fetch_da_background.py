"""Fetch the GEFS ensemble background for a surface-analysis cycle.

Driver only — logic lives in src/heat/da/gefs_members.py.

Usage:
    python scripts/fetch_da_background.py                 # latest cycle
    python scripts/fetch_da_background.py --analysis-time 2026-08-24T12
    python scripts/fetch_da_background.py --members c00 p01 p02   # smoke test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heat.da.config import MEMBER_LABELS  # noqa: E402
from src.heat.da.gefs_members import fetch_background  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-time", default=None,
                        help="UTC synoptic hour, e.g. 2026-08-24T12. "
                             "Default: latest available.")
    parser.add_argument("--lead-hours", type=int, default=None,
                        help="Background lead (default: config value).")
    parser.add_argument("--members", nargs="+", default=None,
                        metavar="LABEL", choices=MEMBER_LABELS,
                        help="Subset of member labels, e.g. c00 p01. "
                             "Default: all 31.")
    parser.add_argument("--min-members", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    kwargs = {"overwrite": args.overwrite}
    if args.analysis_time is not None:
        kwargs["analysis_time"] = args.analysis_time
    if args.lead_hours is not None:
        kwargs["lead_hours"] = args.lead_hours
    if args.members is not None:
        kwargs["members"] = args.members
        kwargs["min_members"] = args.min_members or len(args.members)
    elif args.min_members is not None:
        kwargs["min_members"] = args.min_members

    ds = fetch_background(**kwargs)
    print(f"members: {list(ds.member.values)}")
    print(f"grid: {ds.sizes['latitude']} x {ds.sizes['longitude']}")
    print(f"t2m ens-mean range: {float(ds.t2m.mean('member').min()):.1f} "
          f"to {float(ds.t2m.mean('member').max()):.1f} C")


if __name__ == "__main__":
    main()
