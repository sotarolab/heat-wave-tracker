"""
src/heat/gfs_conus.py
=====================
Download GFS 2m temperature and dewpoint for CONUS and compute derived
heat stress variables. Saves to a single NetCDF file for fast app startup.

The file is designed to be pre-fetched once locally and either:
  a) committed to the git repo (~20–30 MB, within GitHub limits), or
  b) re-fetched by Render on startup if the file is not present.

Variables stored in the output file:
  t2m  [°C]  2m temperature (K→°C converted)
  td2m [°C]  2m dewpoint    (K→°C converted)
  hi   [°C]  Heat index    (NWS Rothfusz)
  wbt  [°C]  Wet bulb temp  (Stull 2011)

Longitude convention: −180 to +180 (converted from GFS 0–360).
Latitude order: north-to-south (as Herbie/cfgrib returns for GFS).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# CONUS bounding box [west, south, east, north]
CONUS_BBOX = [-127.0, 23.0, -65.0, 51.0]

# Default output filename
DEFAULT_OUT = Path("data") / "conus_heat_tracker.nc"


def _latest_gfs_init() -> pd.Timestamp:
    """
    Return the most recent GFS init time likely available on NOAA AWS.
    GFS publishes approximately 4 h after initialization.
    """
    now = pd.Timestamp.utcnow()
    for cycle_h in (18, 12, 6, 0):
        candidate = now.floor("D") + pd.Timedelta(hours=cycle_h)
        if (now - candidate).total_seconds() >= 4 * 3600:
            return candidate
    return now.floor("D")


def fetch_gfs_conus(
    init_dt:   pd.Timestamp | None = None,
    fxx_range: range = range(0, 121, 6),
    out_path:  Path  = DEFAULT_OUT,
    overwrite: bool  = False,
    bbox:      list  = CONUS_BBOX,
    label:     str   = "CONUS",
) -> xr.Dataset:
    """
    Fetch GFS T2m + Td2m for a bounding box and compute HI + WBT.
    Saves the result to `out_path` and returns the xarray Dataset.

    Parameters
    ----------
    init_dt   : GFS initialization time (UTC). Default: latest available cycle.
    fxx_range : Lead hours to fetch. Default: F000–F120 every 6 h (21 steps).
    out_path  : Where to save the NetCDF.
    overwrite : If False and the file exists, load and return it immediately.
    bbox      : [west, south, east, north] in -180..180 longitude. Default CONUS.
    label     : Just for log messages (e.g. "Alaska", "Hawaii").
    """
    from herbie import Herbie
    from .compute import heat_index_array, wet_bulb_array

    if not overwrite and Path(out_path).exists():
        print(f"[gfs] Loading existing file: {out_path}")
        return xr.open_dataset(out_path)

    if init_dt is None:
        init_dt = _latest_gfs_init()
    init_dt = pd.Timestamp(init_dt)
    if init_dt.tzinfo is not None:
        init_dt = init_dt.tz_localize(None)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    west, south, east, north = bbox
    # GFS uses 0–360 longitude; convert bbox for subsetting
    lon_min_360 = west % 360
    lon_max_360 = east % 360

    print(f"[gfs] Fetching {label} T2m + Td2m - init {init_dt}  "
          f"({len(list(fxx_range))} steps)")

    slices = []
    for fxx in fxx_range:
        print(f"  F{fxx:03d} ...", end=" ", flush=True)
        try:
            H = Herbie(init_dt, model="gfs", product="pgrb2.0p25",
                       fxx=fxx, verbose=False)
            t_raw = H.xarray(":TMP:2 m above ground:", remove_grib=True)
            d_raw = H.xarray(":DPT:2 m above ground:", remove_grib=True)
        except Exception as exc:
            print(f"skipped ({exc})")
            continue

        def _extract(raw) -> xr.DataArray:
            dv = [v for v in raw.data_vars
                  if v not in ("step", "time", "valid_time")][0]
            da = raw[dv] - 273.15  # K → °C
            # Subset to CONUS (GFS lons are 0–360)
            lat_mask = (da.latitude >= south) & (da.latitude <= north)
            lon_mask = (da.longitude >= lon_min_360) & (da.longitude <= lon_max_360)
            da = da.isel(latitude=lat_mask, longitude=lon_mask)
            # Drop cfgrib auxiliary coords that block concat
            for c in ("step", "valid_time", "heightAboveGround",
                      "surface", "meanSea", "nominalTop"):
                da = da.drop_vars(c, errors="ignore")
            return da

        t2m  = _extract(t_raw)
        td2m = _extract(d_raw)

        valid_ts = init_dt + pd.Timedelta(hours=fxx)
        t2m  = t2m.expand_dims("time").assign_coords(time=[valid_ts])
        td2m = td2m.expand_dims("time").assign_coords(time=[valid_ts])

        slices.append(xr.Dataset({"t2m": t2m, "td2m": td2m}))
        print("ok")

    if not slices:
        raise RuntimeError("No GFS data fetched - check init time availability.")

    ds = xr.concat(slices, dim="time")

    # Convert longitudes from 0–360 to −180–180 for mapbox compatibility
    lons = ds.longitude.values.copy()
    lons[lons > 180] -= 360
    ds = ds.assign_coords(longitude=lons).sortby("longitude")

    # Compute derived heat stress variables
    print("[gfs] Computing heat index + wet bulb ...", end=" ")
    t_arr  = ds["t2m"].values
    td_arr = ds["td2m"].values

    hi_arr  = heat_index_array(t_arr, td_arr)
    wbt_arr = wet_bulb_array(t_arr, td_arr)

    ds["hi"]  = xr.DataArray(hi_arr,  dims=ds["t2m"].dims, coords=ds["t2m"].coords,
                              attrs={"units": "°C", "long_name": "Heat Index (NWS Rothfusz)"})
    ds["wbt"] = xr.DataArray(wbt_arr, dims=ds["t2m"].dims, coords=ds["t2m"].coords,
                              attrs={"units": "°C", "long_name": "Wet Bulb Temp (Stull 2011)"})
    ds.attrs["gfs_init"] = str(init_dt)
    print("ok")

    print(f"[gfs] Saving → {out_path} ...", end=" ")
    ds.to_netcdf(out_path)
    print("ok")

    return ds


def load_or_fetch(out_path: Path = DEFAULT_OUT, **kwargs) -> xr.Dataset | None:
    """
    Load the pre-fetched GFS file if it exists, otherwise return None.
    Does NOT attempt to download - call fetch_gfs_conus() for that.
    """
    if Path(out_path).exists():
        return xr.open_dataset(out_path)
    return None
