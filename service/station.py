"""One station's page: observations, the raw forecast, the corrections, and how they verify.

A pure function of the forecast dataset, the station's observations and the logged model
output. Two corrections are shown and kept apart, because they use different information:

- the same-day correction (`src/heat/bias.py`): today's mean error of this forecast at this
  station, applied to the next 24 hours, with a prediction interval from today's residuals.
  It uses observations that arrive between the six-hourly jobs, so it is computed here.
- the learned correction: the gradient-boosted model's output for this cycle, logged by the
  refresh workflow and read from the database. It is drawn as logged and never recomputed.

Both correct GFS. HRRR and NBM are drawn raw beside them for the hours ahead, for comparison.
The other variables (dew point, humidity, wind, cloud, precipitation, pressure) follow as
observed and as each model has them, so the temperature can be read in its weather.

Every number is converted and rounded here; the page draws the table as it comes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr

from service.payloads import Unit
from service.sources import MODELS
from src.heat.bias import BIAS_MIN_PAIRS, today_forecast_bias
from src.heat.compute import heat_index_array

SAME_DAY_HOURS = 24  # how far ahead today's mean error is taken to say anything
LOOK_BACK_HOURS = 36
ML_BEHIND_HOURS, ML_AHEAD_HOURS = 24, 48

# Colour tokens the page resolves; one meaning each, on every chart.
OBS, RAW, SAME_DAY, ML = "--obs", "--raw", "--series-4", "--ml"


def _val(c: float, unit: Unit) -> float:
    return c * 9.0 / 5.0 + 32.0 if unit == "F" else c


def _delta(c: float, unit: Unit) -> float:
    return c * 9.0 / 5.0 if unit == "F" else c


def _deg(c: float | None, unit: Unit, delta: bool = False, signed: bool = False) -> str:
    if c is None or np.isnan(c):
        return "no value"
    if delta:  # a difference keeps its tenth: half a degree of error is the point
        return f"{_delta(c, unit):+.1f}°{unit}" if signed else f"{_delta(c, unit):.1f}°{unit}"
    return f"{_val(c, unit):.0f}°{unit}"


def _raw_series(ds: xr.Dataset, stn: dict[str, Any], var: str) -> pd.Series:
    s = ds[var].sel(latitude=stn["lat"], longitude=stn["lon"], method="nearest").to_series()
    s.index = pd.DatetimeIndex(s.index).tz_localize("UTC")
    return s.astype(float)


def _observed(obs: pd.DataFrame | None, tz: ZoneInfo) -> pd.DataFrame:
    """Observations with a local timestamp and the heat index computed as the forecast's is."""
    if obs is None or obs.empty:
        return pd.DataFrame(columns=["valid_utc", "valid_local", "temp_c", "hi_c"])
    out = obs.dropna(subset=["temp_c"]).sort_values("valid_utc").copy()
    out["valid_local"] = out["valid_utc"].dt.tz_convert(tz)
    t, td = out["temp_c"].to_numpy(dtype=float), out["dewpoint_c"].to_numpy(dtype=float)
    hi = np.full(len(out), np.nan)
    ok = ~np.isnan(td)
    if ok.any():
        hi[ok] = heat_index_array(t[ok], td[ok])  # type: ignore[no-untyped-call]
    out["hi_c"] = hi
    return out


Convert = Callable[[Any], Any]


def _column(index: pd.DatetimeIndex, s: pd.Series, convert: Convert, digits: int = 1) -> list[float | None]:
    """A series laid on the chart's time axis, converted for display: null where it has no value."""
    v = s[~s.index.duplicated()].reindex(index)
    return [None if np.isnan(x) else round(float(convert(x)), digits) for x in v.values]


def _picture(
    axis: pd.DatetimeIndex,
    o: pd.DataFrame,
    obs_t: pd.DatetimeIndex,
    ahead: dict[str, pd.DataFrame],
    raw_td: pd.Series | None,
    unit: Unit,
) -> list[dict[str, Any]]:
    """The other variables, each as observed and as GFS, HRRR and NBM have it for the hours ahead.

    Metric in, display units out: degrees as asked, wind in mph or km/h, precipitation in inches
    or millimetres with them. Precipitation is a running total: what the station measured over the
    window, and each model's amount added on from now, so the curves continue where the gauge stopped.
    """
    us = unit == "F"
    deg, wind_unit, rain_unit = f"°{unit}", " mph" if us else " km/h", " in" if us else " mm"
    temp: Convert = lambda c: _val(c, unit)  # noqa: E731
    wind: Convert = lambda kmh: kmh * 0.621371 if us else kmh  # noqa: E731
    rain: Convert = lambda mm: mm / 25.4 if us else mm  # noqa: E731
    same: Convert = lambda v: v  # noqa: E731

    def observed(col: str, scale: float = 1.0) -> pd.Series | None:
        if col not in o or not o[col].notna().any():
            return None
        return pd.Series(o[col].to_numpy(dtype=float) * scale, index=obs_t)

    def chart(cid: str, label: str, unit_text: str, model_col: str, convert: Convert, digits: int,
              obs: pd.Series | None, extra: list[dict[str, Any]] | None = None, **limits: float) -> dict[str, Any] | None:
        lines: list[dict[str, Any]] = [
            {"label": MODELS[m][0], "color": MODELS[m][1], "values": _column(axis, df[model_col], convert, digits), **({"dash": True} if m == "gfs_global" else {})}
            for m, df in ahead.items() if df[model_col].notna().any()
        ]
        lines += extra or []
        if obs is not None:
            lines.append({"label": "Observed", "color": OBS, "values": _column(axis, obs, convert, digits), "width": 3})
        return {"id": cid, "label": label, "unit": unit_text, "lines": lines, **limits} if lines else None

    # Dew point: GFS is in the forecast file as issued, so it is drawn from there, from the cycle's start.
    td_file: list[dict[str, Any]] = [] if raw_td is None else [{"label": "GFS, raw", "color": RAW, "values": _column(axis, raw_td, temp), "dash": True}]
    td_models = {m: df for m, df in ahead.items() if m != "gfs_global" or raw_td is None}
    td_lines: list[dict[str, Any]] = [{"label": MODELS[m][0], "color": MODELS[m][1], "values": _column(axis, df["dewpoint_c"], temp)} for m, df in td_models.items() if df["dewpoint_c"].notna().any()]
    td_obs = observed("dewpoint_c")
    if td_obs is not None:
        td_lines.append({"label": "Observed", "color": OBS, "values": _column(axis, td_obs, temp), "width": 3})
    dew = {"id": "td", "label": "Dew point", "unit": deg, "lines": td_file + td_lines} if td_file or td_lines else None

    gust = observed("wind_gust_kt", 1.852)
    gusts: list[dict[str, Any]] = [] if gust is None else [{"label": "Observed gusts", "color": OBS, "values": _column(axis, gust, wind, 0), "dots": True}]

    # Running totals. The hourly reports are summed over the window; the models continue from that total.
    measured = observed("precip_in", 25.4)
    total = measured.fillna(0.0).cumsum() if measured is not None else None
    base = float(total.iloc[-1]) if total is not None and len(total) else 0.0
    rain_ahead = {m: df.assign(precip_mm=base + df["precip_mm"].cumsum()) for m, df in ahead.items() if df["precip_mm"].notna().any()}
    precip_lines: list[dict[str, Any]] = [
        {"label": MODELS[m][0], "color": MODELS[m][1], "values": _column(axis, df["precip_mm"], rain, 2 if us else 1), **({"dash": True} if m == "gfs_global" else {})}
        for m, df in rain_ahead.items()
    ]
    if total is not None:
        precip_lines.append({"label": "Observed", "color": OBS, "values": _column(axis, total, rain, 2 if us else 1), "width": 3})
    precip = {"id": "precip", "label": "Precipitation, running total", "unit": rain_unit, "lines": precip_lines, "floor": 0} if precip_lines else None

    cloud_obs = observed("sky_cover", 100.0)
    out = [
        dew,
        chart("rh", "Humidity", "%", "rh", same, 0, observed("rh"), floor=0, ceil=100),
        chart("wind", "Wind", wind_unit, "wind_kmh", wind, 0, observed("wind_spd_kt", 1.852), gusts, floor=0),
        chart("cloud", "Cloud cover", "%", "cloud_pct", same, 0, cloud_obs, floor=0, ceil=100),
        precip,
        chart("pressure", "Sea-level pressure", " hPa", "pressure_hpa", same, 1, observed("pressure_hpa")),
    ]
    return [c for c in out if c is not None]


def build(
    ds: xr.Dataset,
    stn: dict[str, Any],
    obs: pd.DataFrame | None,
    ml: pd.DataFrame | None,
    model: dict[str, Any] | None,
    unit: Unit,
    now: pd.Timestamp,
    others: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    tz = ZoneInfo(stn.get("tz", "America/New_York"))
    raw = {k: _raw_series(ds, stn, k) for k in ("t2m", "hi")}
    end = raw["t2m"].index[-1]
    stale = end < now
    o = _observed(None if stale else obs, tz)  # a forecast that has ended has nothing current to verify it
    now_obs = o["valid_local"].iloc[-1] if len(o) else None
    today = now.tz_convert(tz).date()

    # Same-day correction: today's mean error of this forecast, from the steps that have happened.
    bias: dict[str, dict[str, Any] | None] = {}
    for key, col in (("t2m", "temp_c"), ("hi", "hi_c")):
        local = raw[key].copy()
        local.index = pd.DatetimeIndex(local.index).tz_convert(tz)
        bias[key] = None if now_obs is None else today_forecast_bias(local, o, col, now_obs.date(), now_obs)

    ahead = (raw["t2m"].index > now) & (raw["t2m"].index <= now + pd.Timedelta(hours=SAME_DAY_HOURS))
    same_day = {k: (raw[k][ahead] + b["bias"]) if (b := bias[k]) else None for k in raw}
    pi = bias["t2m"]["pi95_halfwidth"] if bias["t2m"] else 0.0  # an interval is claimed for temperature only

    if ml is not None:
        ml = ml[(ml.index >= now - pd.Timedelta(hours=ML_BEHIND_HOURS)) & (ml.index <= now + pd.Timedelta(hours=ML_AHEAD_HOURS))]
        if ml.empty:
            ml = None

    # One time axis for every line: the forecast steps and the observation times in the window.
    # The observations lead in from before the forecast begins, so the day so far is on the chart.
    start = raw["t2m"].index[0] if stale else now - pd.Timedelta(hours=LOOK_BACK_HOURS)
    o_win = o[o["valid_utc"] >= start]
    obs_t = pd.DatetimeIndex(o_win["valid_utc"]).floor("min")
    axis = raw["t2m"].index[raw["t2m"].index >= start].union(obs_t)
    if ml is not None:
        axis = axis.union(ml.index[ml.index >= start])
    # Other models: the hours ahead only, and nothing over a forecast that has ended.
    ahead_of_now = {} if stale or not others else {m: df[(df.index > now) & (df.index <= end)] for m, df in others.items() if m in MODELS}
    ahead_of_now = {m: df for m, df in ahead_of_now.items() if len(df)}
    for df in ahead_of_now.values():
        axis = axis.union(df.index)

    axis = pd.DatetimeIndex(axis)
    temp: Convert = lambda c: _val(c, unit)  # noqa: E731
    deg = f"°{unit}"

    def line(label: str, color: str, s: pd.Series, **extra: Any) -> dict[str, Any]:
        return {"label": label, "color": color, "values": _column(axis, s, temp), **extra}

    charts: list[dict[str, Any]] = []
    for key, col, title in (("t2m", "temp_c", "Temperature"), ("hi", "hi_c", "Feels like")):
        lines = [line("GFS, raw", RAW, raw[key], dash=True)]
        if key == "t2m":  # the file's GFS is the one drawn here, as issued
            lines += [line(f"{MODELS[m][0]}, raw", MODELS[m][1], df["temp_c"]) for m, df in ahead_of_now.items() if m != "gfs_global" and df["temp_c"].notna().any()]
        sd = same_day[key]
        if sd is not None and len(sd):
            band = {"band": {"lo": _column(axis, sd - pi, temp), "up": _column(axis, sd + pi, temp)}} if key == "t2m" and pi > 0 else {}
            lines.append(line("Corrected with today's observations", SAME_DAY, sd, **band))
        if key == "t2m" and ml is not None:
            band = {"band": {"lo": _column(axis, ml["lo_c"], temp), "up": _column(axis, ml["up_c"], temp)}} if ml[["lo_c", "up_c"]].notna().all(axis=None) else {}
            lines.append(line(f"Learned correction ({ml.attrs.get('version', 'model')})", ML, ml["ml_c"], **band))
        if len(o_win):
            lines.append(line("Observed", OBS, pd.Series(o_win[col].values, index=obs_t), width=3))
        charts.append({"id": key, "label": title, "unit": deg, "lines": lines})
    raw_td = _raw_series(ds, stn, "td2m") if "td2m" in ds else None
    charts += _picture(axis, o_win, obs_t, ahead_of_now, raw_td, unit)

    # Today's high: what has been observed, and the best line for the hours of today still ahead.
    local_idx = pd.DatetimeIndex(raw["t2m"].index).tz_convert(tz)
    is_today = np.array([t.date() == today for t in local_idx])
    raw_high = float(raw["t2m"][is_today].max()) if is_today.any() else None
    obs_today = o[[t.date() == today for t in o["valid_local"]]] if len(o) else o
    obs_high = float(obs_today["temp_c"].max()) if len(obs_today) else None
    left_today = is_today & (raw["t2m"].index > now)
    source, rest = "the raw forecast", raw["t2m"][left_today]
    if ml is not None and (ml_rest := ml["ml_c"][[t.tz_convert(tz).date() == today and t > now for t in ml.index]]).size:
        source, rest = "the learned correction", ml_rest
    elif same_day["t2m"] is not None and (sd_rest := same_day["t2m"].reindex(raw["t2m"].index[left_today]).dropna()).size:
        source, rest = "today's observations", sd_rest
    # Without observations the hours of today already past are the raw forecast's to speak for.
    past_today = raw["t2m"][is_today & (raw["t2m"].index <= now)]
    so_far = obs_high if obs_high is not None else (float(past_today.max()) if len(past_today) else None)
    candidates = [v for v in (so_far, float(rest.max()) if len(rest) else None) if v is not None]
    best_high = max(candidates) if candidates and not stale else None

    name = stn["name"]
    b = bias["t2m"]
    if stale:
        headline = f"{name}: this forecast ended {end.tz_convert(tz).strftime('%b %-d')}. It is shown as issued; there is nothing current to verify it against."
    elif now_obs is None:
        headline = f"{name}: GFS has today's high at {_deg(raw_high, unit)}. Observations are unavailable right now."
    else:
        parts = [f"{name} is {_deg(float(o['temp_c'].iloc[-1]), unit)}."]
        if b:
            lean = "too cold" if b["bias"] > 0 else "too warm"
            parts.append(f"GFS has run {_deg(abs(b['bias']), unit, delta=True)} {lean} today over {b['n_used']} forecast steps.")
        if best_high is not None and raw_high is not None:
            parts.append(f"Today's high, from {source}: {_deg(best_high, unit)}, against {_deg(raw_high, unit)} raw.")
        headline = " ".join(parts)

    tiles = []
    if now_obs is not None:
        tiles.append({"label": "Now", "value": _deg(float(o["temp_c"].iloc[-1]), unit), "note": f"observed {now_obs.strftime('%-I:%M %p %Z')}"})
    tiles.append({"label": "Today's high, best estimate", "value": _deg(best_high, unit), "note": "forecast has ended" if stale else f"from {source}"})
    tiles.append({"label": "Today's high, raw GFS", "value": _deg(raw_high, unit), "note": "nearest grid point"})
    if obs_high is not None:
        tiles.append({"label": "Observed high so far", "value": _deg(obs_high, unit), "note": today.strftime("%a %b %-d")})
    if b:
        tiles.append({"label": "GFS error today", "value": _deg(b["bias"], unit, delta=True, signed=True), "note": f"observed minus forecast, {b['n_used']} steps"})

    today_rows = [
        {"metric": label, "n": r["n_used"], "bias": round(_delta(r["bias"], unit), 1), "rmse": round(_delta(r["rmse"], unit), 1),
         "interval": round(_delta(r["pi95_halfwidth"], unit), 1) if key == "t2m" and r["pi95_halfwidth"] > 0 else None}
        for key, label in (("t2m", "Temperature"), ("hi", "Feels like"))
        if (r := bias[key])
    ]
    model_row = None
    if model:
        model_row = {
            "version": model["version"], "days": model["days"], "n": model["n"],
            "rmse_raw": round(_delta(model["rmse_raw_c"], unit), 2), "rmse_offset": round(_delta(model["rmse_offset_c"], unit), 2),
            "rmse_ml": round(_delta(model["rmse_ml_c"], unit), 2),
            "band_coverage_pct": None if model["band_coverage"] is None else round(100 * model["band_coverage"]),
        }

    return {
        "station": {k: stn[k] for k in ("id", "name", "state", "lat", "lon")} | {"tz": str(tz)},
        "unit": unit,
        "stale": stale,
        "headline": headline,
        "tiles": tiles,
        "time": [int(t.timestamp()) for t in axis],
        "now": int(now.timestamp()),
        "charts": charts,
        "verification": {"today": today_rows, "model": model_row},
        "has_learned": ml is not None,
        "same_day_hours": SAME_DAY_HOURS,
        "min_steps": BIAS_MIN_PAIRS,
    }
