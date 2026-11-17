"""What each route returns, built from the forecast dataset. Pure functions: no I/O.

The dataset is the committed GFS file: variables `t2m` and `hi` in Celsius on
(time, latitude, longitude), valid times in UTC, `attrs["gfs_init"]` the cycle.
Every number the page shows is final here: unit conversion, rounding and the colour
scale are not the page's work.
"""

from __future__ import annotations

from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr

from service import fields
from src.heat.freshness import age_hours, badge
from src.heat.stations import MAJOR_CONUS_STATIONS

Unit = Literal["F", "C"]
# The station list predates the type gate; this names its shape for the code that reads it.
STATIONS: list[dict[str, Any]] = MAJOR_CONUS_STATIONS
# One clock for a map that spans four time zones, as the national products use.
DISPLAY_TZ = ZoneInfo("America/New_York")


def _out(values_c: Any, unit: Unit) -> Any:
    return values_c * 9.0 / 5.0 + 32.0 if unit == "F" else values_c


def _times(ds: xr.Dataset) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ds.time.values).tz_localize("UTC")


def _days(ds: xr.Dataset) -> list[tuple[pd.Timestamp, list[int]]]:
    """The forecast steps grouped by local day, in order: (the day, the step indices in it)."""
    local = _times(ds).tz_convert(DISPLAY_TZ)
    out: dict[pd.Timestamp, list[int]] = {}
    for i, t in enumerate(local):
        out.setdefault(t.normalize(), []).append(i)
    return list(out.items())


def frames(ds: xr.Dataset, var: fields.Variable, now: pd.Timestamp) -> tuple[list[dict[str, Any]], int]:
    """The frames of a variable and the one to open on: the step nearest now, or today."""
    times = _times(ds)
    if var.per == "day":
        days = _days(ds)
        today = now.tz_convert(DISPLAY_TZ).normalize()
        start = next((k for k, (d, _) in enumerate(days) if d >= today), len(days) - 1)
        return [{"i": k, "time": d.strftime("%Y-%m-%d"), "label": d.strftime("%a %b %-d")} for k, (d, _) in enumerate(days)], start
    local = times.tz_convert(DISPLAY_TZ)
    out = [
        {"i": i, "time": t.strftime("%Y-%m-%dT%H:%MZ"), "label": f"{loc.strftime('%a %b %-d, %-I %p')} ET ({t.strftime('%H')}Z)"}
        for i, (t, loc) in enumerate(zip(times, local, strict=True))
    ]
    start = int(np.abs((times - now).total_seconds()).argmin())
    return out, start


def field(ds: xr.Dataset, var: fields.Variable, i: int) -> fields.Floats:
    """Frame i of a variable as a 2D array in Celsius. Raises IndexError past the last frame."""
    da = ds[var.source]
    if var.per == "day":
        days = _days(ds)
        if not 0 <= i < len(days):
            raise IndexError(i)
        return np.asarray(da.isel(time=days[i][1]).max("time").values)
    if not 0 <= i < da.sizes["time"]:
        raise IndexError(i)
    return np.asarray(da.isel(time=i).values)


def cycle(ds: xr.Dataset) -> str:
    """A token that changes with the forecast cycle, for image URLs the browser may cache."""
    return "".join(ch for ch in str(ds.attrs.get("gfs_init", "")) if ch.isdigit())[:10] or "none"


def legend(var: fields.Variable, unit: Unit) -> dict[str, Any]:
    if var.kind == "risk":
        items: list[dict[str, Any]] = [{"label": fields.NO_RISK[0], "color": fields.NO_RISK[1], "from": None}]
        items += [
            {"label": label, "color": color, "from": round(lower_f if unit == "F" else fields.f_to_c(lower_f), 1)}
            for lower_f, label, color in fields.RISK_CATEGORIES_F
        ]
        return {"kind": "categories", "items": items}
    return {"kind": "ramp", "stops": fields.ramp_stops(var), "min": round(_out(var.vmin_c, unit)), "max": round(_out(var.vmax_c, unit))}


def field_meta(ds: xr.Dataset, now: pd.Timestamp, unit: Unit) -> dict[str, Any]:
    south, west, north, east = fields.edges(ds.latitude.values, ds.longitude.values)
    variables = []
    for var in fields.VARIABLES.values():
        fr, start = frames(ds, var, now)
        variables.append({"id": var.id, "label": var.label, "note": var.note, "frames": fr, "start": start, "legend": legend(var, unit)})
    return {"cycle": cycle(ds), "unit": unit, "bounds": [[south, west], [north, east]], "opacity": fields.OPACITY, "variables": variables}


def stations(ds: xr.Dataset, unit: Unit) -> dict[str, Any]:
    """Every station with the forecast at its nearest grid point, for each variable and frame."""
    lat = xr.DataArray([s["lat"] for s in MAJOR_CONUS_STATIONS], dims="station")
    lon = xr.DataArray([s["lon"] for s in MAJOR_CONUS_STATIONS], dims="station")
    values: dict[str, list[list[float | None]]] = {}
    for var in fields.VARIABLES.values():
        at = ds[var.source].sel(latitude=lat, longitude=lon, method="nearest")  # (time, station)
        if var.per == "day":
            at = xr.concat([at.isel(time=idx).max("time") for _, idx in _days(ds)], dim="time")
        table = np.round(_out(at.values.T.astype(float), unit), 1)  # (station, frame)
        values[var.id] = [[None if np.isnan(v) else float(v) for v in row] for row in table]
    rows = [{"id": s["id"], "name": s["name"], "state": s["state"], "lat": s["lat"], "lon": s["lon"]} for s in MAJOR_CONUS_STATIONS]
    return {"unit": unit, "stations": rows, "values": values}


def manifest(ds: xr.Dataset, now: pd.Timestamp, commit: str | None) -> dict[str, Any]:
    init = str(ds.attrs.get("gfs_init", ""))
    text, live = badge(init, now)
    age = age_hours(init, now)
    try:
        label = pd.Timestamp(init).strftime("GFS %HZ %b %-d")
    except (ValueError, TypeError):
        label = "No forecast"
    return {
        "init": init,
        "label": label if init else "No forecast",
        "age_h": None if age is None else round(age, 1),
        "badge": text,
        "live": live,
        "steps": int(ds.sizes["time"]),
        "stations": len(MAJOR_CONUS_STATIONS),
        "commit": commit,
    }


def network(ds: xr.Dataset, latest: dict[str, tuple[float, pd.Timestamp]] | None, unit: Unit, now: pd.Timestamp, window_h: int = 2) -> dict[str, Any]:
    """Every station in one table: what it reads now, what the forecast said for now, and the next two highs.

    `latest` is each station's most recent temperature in Celsius and its time, or None when the
    network feed is unavailable; the forecast columns are filled either way.
    """
    lat = xr.DataArray([s["lat"] for s in STATIONS], dims="station")
    lon = xr.DataArray([s["lon"] for s in STATIONS], dims="station")
    at = ds["t2m"].sel(latitude=lat, longitude=lon, method="nearest").values.astype(float)  # (time, station)
    times = _times(ds)
    current = now <= times[-1]
    step = int(np.abs((times - now).total_seconds()).argmin())

    def num(c: float | None, delta: bool = False) -> float | None:
        if c is None or np.isnan(c):
            return None
        return round(float(c * 9.0 / 5.0 if delta and unit == "F" else c if delta else _out(c, unit)), 1)

    rows = []
    for k, s in enumerate(STATIONS):
        local = times.tz_convert(ZoneInfo(s["tz"]))
        today = now.tz_convert(ZoneInfo(s["tz"])).normalize()
        days = local.normalize()
        high = [float(np.nanmax(at[days == d, k])) if (days == d).any() else None for d in (today, today + pd.Timedelta(days=1))]
        obs = latest.get(s["id"]) if latest and current else None
        fcst = float(at[step, k]) if current else None
        rows.append({
            "id": s["id"], "name": s["name"], "state": s["state"], "lat": s["lat"], "lon": s["lon"],
            "observed": num(obs[0]) if obs else None,
            "observed_min_ago": None if not obs else int((now - obs[1]).total_seconds() // 60),
            "forecast": num(fcst),
            "error": num(obs[0] - fcst, delta=True) if obs and fcst is not None else None,
            "high_today": num(high[0]), "high_tomorrow": num(high[1]),
        })
    errors: list[float] = [abs(float(r["error"])) for r in rows if r["error"] is not None]
    return {
        "unit": unit, "current": current, "window_h": window_h, "reporting": sum(r["observed"] is not None for r in rows), "stations": len(rows),
        "mean_abs_error": round(float(np.mean(errors)), 1) if errors else None, "rows": rows,
    }
