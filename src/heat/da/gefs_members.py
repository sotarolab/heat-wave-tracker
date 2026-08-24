"""
src/heat/da/gefs_members.py
===========================
Fetch the GEFS ensemble (control + 30 perturbed members) 2 m temperature
and dewpoint at one short lead time and stack it into a single
(member, latitude, longitude) Dataset: the analysis background for the
surface-analysis subsystem (docs/da/DESIGN.md).

Follows src/heat/gfs_conus.py conventions exactly where they overlap:
units °C, longitudes converted 0-360 -> [-180, 180] on save, Herbie
subset-by-search-string so full GRIB files are never downloaded, and lazy
herbie/xarray imports so this module is importable by the
requirements-dev-only CI (the GRIB stack is deliberately absent there).

Gotchas:

1. GEFS 0.25° data lives in the pgrb2s ("atmos.25" in Herbie's product
   naming) file set, which carries a reduced field list — but 2 m TMP and
   DPT are in it, for every member, which is exactly what we need and at
   twice the resolution of the pgrb2a 0.5° set.
2. Individual member fetches are allowed to fail (a late-arriving member
   on AWS must not kill an operational cycle: an EnSRF with 28 members is
   fine, a crashed job is not). Missing members are reported in the
   Dataset's `missing_members` attr and printed, never silent — a quietly
   shrinking ensemble would bias the spread-based covariance downward
   cycle after cycle. fetch_background() raises only below `min_members`.
3. The background convention is "the freshest cycle whose `lead` forecast
   is valid now-ish": for a 12Z analysis with 6 h lead, we want the 06Z
   cycle's F006. latest_background_init() therefore steps back from the
   *analysis* time, reusing gfs_conus's ~4 h publication-lag assumption
   (GEFS publishes on roughly the same schedule as GFS).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.heat.da.config import (
    BACKGROUND_LEAD_HOURS,
    CONUS_BBOX,
    DA_DATA_DIR,
    MEMBER_LABELS,
)

_PUBLICATION_LAG_HOURS = 4  # see Gotcha 3


def _member_number(label: str) -> int:
    """'c00' -> 0, 'p07' -> 7: the integer form Herbie's GEFS template takes."""
    return 0 if label == "c00" else int(label[1:])


def background_path(analysis_time: pd.Timestamp, out_dir: Path = DA_DATA_DIR) -> Path:
    return Path(out_dir) / f"background_{pd.Timestamp(analysis_time):%Y%m%dT%H}Z.nc"


def latest_background_init(
    analysis_time: pd.Timestamp | None = None,
    lead_hours: int = BACKGROUND_LEAD_HOURS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pick (init, analysis_time) for the freshest usable background.

    analysis_time defaults to the most recent synoptic hour (00/06/12/18Z)
    whose background cycle (analysis_time - lead_hours) should already be
    published per the ~4 h lag assumption (Gotcha 3).
    """
    now = pd.Timestamp.utcnow().tz_localize(None)
    candidate = now.floor("6h") if analysis_time is None else pd.Timestamp(analysis_time)
    if candidate.tzinfo is not None:
        candidate = candidate.tz_localize(None)
    while analysis_time is None:
        init = candidate - pd.Timedelta(hours=lead_hours)
        if (now - init).total_seconds() >= _PUBLICATION_LAG_HOURS * 3600:
            break
        candidate -= pd.Timedelta(hours=6)
    return candidate - pd.Timedelta(hours=lead_hours), candidate


def fetch_background(
    analysis_time: pd.Timestamp | None = None,
    lead_hours: int = BACKGROUND_LEAD_HOURS,
    members: list[str] = MEMBER_LABELS,
    bbox: list = CONUS_BBOX,
    out_dir: Path = DA_DATA_DIR,
    overwrite: bool = False,
    min_members: int = 20,
):
    """Fetch the GEFS member stack valid at analysis_time; save and return it.

    Returns an xr.Dataset with variables t2m, td2m (°C) on dims
    (member, latitude, longitude), attrs gefs_init, analysis_time,
    lead_hours, missing_members.
    """
    import xarray as xr
    from herbie import Herbie

    init_dt, analysis_time = latest_background_init(analysis_time, lead_hours)
    out_path = background_path(analysis_time, out_dir)
    if not overwrite and out_path.exists():
        print(f"[da.gefs] Loading existing background: {out_path}")
        return xr.open_dataset(out_path)

    west, south, east, north = bbox
    lon_min_360, lon_max_360 = west % 360, east % 360

    print(f"[da.gefs] Background init {init_dt} F{lead_hours:03d} "
          f"(valid {analysis_time}), {len(members)} members")

    slices, missing = [], []
    for label in members:
        try:
            H = Herbie(init_dt, model="gefs", product="atmos.25",
                       member=_member_number(label), fxx=lead_hours, verbose=False)
            t_raw = H.xarray(":TMP:2 m above ground:", remove_grib=True)
            d_raw = H.xarray(":DPT:2 m above ground:", remove_grib=True)
        except Exception as exc:
            print(f"  {label} skipped ({exc})")
            missing.append(label)
            continue

        def _extract(raw) -> "xr.DataArray":
            dv = [v for v in raw.data_vars
                  if v not in ("step", "time", "valid_time")][0]
            da = raw[dv] - 273.15  # K to C
            lat_mask = (da.latitude >= south) & (da.latitude <= north)
            lon_mask = (da.longitude >= lon_min_360) & (da.longitude <= lon_max_360)
            da = da.isel(latitude=lat_mask, longitude=lon_mask)
            for c in ("step", "valid_time", "heightAboveGround", "number",
                      "surface", "meanSea", "nominalTop"):
                da = da.drop_vars(c, errors="ignore")
            return da

        member_ds = xr.Dataset({"t2m": _extract(t_raw), "td2m": _extract(d_raw)})
        slices.append(member_ds.expand_dims("member").assign_coords(member=[label]))
        print(f"  {label} ok")

    if len(slices) < min_members:
        raise RuntimeError(
            f"Only {len(slices)}/{len(members)} members fetched "
            f"(min_members={min_members}) - check cycle availability."
        )
    if missing:
        print(f"[da.gefs] WARNING: missing members this cycle: {missing}")

    ds = xr.concat(slices, dim="member")
    lons = ds.longitude.values.copy()
    lons[lons > 180] -= 360
    ds = ds.assign_coords(longitude=lons).sortby("longitude")
    ds.attrs.update(
        gefs_init=str(init_dt),
        analysis_time=str(analysis_time),
        lead_hours=lead_hours,
        missing_members=",".join(missing) if missing else "",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"[da.gefs] Saved {out_path} ({len(slices)} members)")
    return ds
