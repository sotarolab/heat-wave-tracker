"""
heat-wave-tracker / app.py
===========================
CONUS heat wave dashboard — GFS surface field with station click panel.

Variables: 2m Temperature | Heat Index (NWS) | Risk Level (NWS heat index
categories, daily max)

Run locally:
    python app.py

On Render (Gunicorn):
    gunicorn app:server
"""
import base64
import io
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from dash import Dash, dcc, html, Input, Output, State, ctx

from src.heat.gfs_conus import load_or_fetch, DEFAULT_OUT
from src.heat.stations  import MAJOR_CONUS_STATIONS, get_station
from src.heat.asos      import fetch_station_obs


# ── constants ─────────────────────────────────────────────────────────────────

DATA_PATH  = DEFAULT_OUT
CONUS_BBOX = [-127.0, 23.0, -65.0, 51.0]

# Overall page content is capped to this width and centered — keeps the map's
# aspect ratio sane on wide monitors and matches the zoom heuristic below.
PAGE_MAX_WIDTH = 1400
MAP_HEIGHT     = 620

# Reference timezone for map-level labels (a single CONUS raster snapshot spans
# 4 zones at once, so there's no true "local" time for it — Eastern is used as
# the display convention, with UTC alongside for anyone who needs it).
_DISPLAY_TZ = ZoneInfo("America/New_York")

VARIABLE_META = {
    "t2m": {
        "label":  "2m Temperature",
        "cmap":   "RdYlBu_r",
        "vmin":   10.0,
        "vmax":   45.0,
        "plotly": "RdYlBu_r",
    },
    "hi": {
        "label":  "Heat Index",
        "cmap":   "YlOrRd",
        "vmin":   20.0,
        "vmax":   55.0,
        "plotly": "YlOrRd",
    },
    "risk": {
        "label":  "Risk Level (Heat Index)",
        "cmap":   None,
        "vmin":   None,
        "vmax":   None,
        "plotly": None,
    },
}

_MPL_CMAPS = {
    "RdYlBu_r": plt.cm.RdYlBu_r,
    "YlOrRd":   plt.cm.YlOrRd,
    "Blues":    plt.cm.Blues,
}

# NWS heat index risk categories — native NOAA thresholds (°F) plus their
# °C equivalents, so each display unit uses its own natural round numbers.
# A cell below the first threshold is left fully transparent (no elevated risk).
RISK_CATEGORIES_F = [
    (80.0,  "Caution",          "#f5c451"),
    (90.0,  "Extreme Caution",  "#f2994a"),
    (103.0, "Danger",           "#e8592e"),
    (125.0, "Extreme Danger",   "#d1382a"),
]
RISK_CATEGORIES_C = [
    (round((f - 32.0) * 5.0 / 9.0, 1), label, color)
    for f, label, color in RISK_CATEGORIES_F
]

# Plain-language versions of NOAA's official heat-index risk definitions —
# same source NWS/NYT-style heat maps cite, phrased for a non-meteorologist.
RISK_DESCRIPTIONS = {
    "No Elevated Risk": "Comfortable — no unusual heat risk.",
    "Caution":          "Fatigue possible with prolonged outdoor exposure or activity.",
    "Extreme Caution":  "Heat cramps or exhaustion possible with prolonged exposure or activity.",
    "Danger":           "Heat cramps or exhaustion likely; heat stroke possible if prolonged.",
    "Extreme Danger":   "Heat stroke highly likely.",
}

_PANEL_BG   = "#0f172a"
_PANEL_GRID = "#1e293b"
_PANEL_FONT = "#cbd5e1"

# ── startup: load GFS data once ───────────────────────────────────────────────

_GFS_DS = load_or_fetch(DATA_PATH)

if _GFS_DS is not None:
    print(f"[app] GFS data loaded: {DATA_PATH}")
    print(f"      times:     {_GFS_DS.time.values[0]} → {_GFS_DS.time.values[-1]}")
    print(f"      variables: {list(_GFS_DS.data_vars)}")
    _GFS_INIT = _GFS_DS.attrs.get("gfs_init", "unknown init")
else:
    print(f"[app] WARNING: {DATA_PATH} not found.")
    print("      Run: python scripts/fetch_gfs_conus.py")
    _GFS_INIT = "data not loaded"

# In-process cache for ASOS obs, keyed by station_id
_asos_cache: dict[str, pd.DataFrame] = {}


# ── unit helpers ──────────────────────────────────────────────────────────────

def _convert(value_c, unit: str):
    """Celsius -> display unit. NaN-safe (arithmetic on NaN stays NaN)."""
    return value_c * 9.0 / 5.0 + 32.0 if unit == "F" else value_c


def _convert_array(arr_c: np.ndarray, unit: str) -> np.ndarray:
    return arr_c * 9.0 / 5.0 + 32.0 if unit == "F" else arr_c


def _unit_label(unit: str) -> str:
    return "°F" if unit == "F" else "°C"


def _risk_categories(unit: str) -> list[tuple]:
    return RISK_CATEGORIES_F if unit == "F" else RISK_CATEGORIES_C


# ── time helpers ──────────────────────────────────────────────────────────────

def _to_et(np_time) -> pd.Timestamp:
    """GFS times are stored tz-naive but represent UTC. Convert to Eastern."""
    return pd.Timestamp(np_time).tz_localize("UTC").tz_convert(_DISPLAY_TZ)


def _et_utc_label(np_time) -> str:
    """Map header label: Eastern first (display convention), UTC alongside."""
    ts_et  = _to_et(np_time)
    ts_utc = pd.Timestamp(np_time)
    return f"{ts_et.strftime('%b %d · %I:%M %p')} {ts_et.strftime('%Z')}  ({ts_utc.strftime('%H')}Z)"


def _unique_forecast_days() -> list[tuple]:
    """
    Unique Eastern-local calendar dates spanned by the forecast, each paired
    with the index of its first GFS timestep (used as the day-dropdown value).
    """
    if _GFS_DS is None:
        return []
    seen: dict = {}
    for i, t in enumerate(_GFS_DS.time.values):
        d = _to_et(t).date()
        seen.setdefault(d, i)
    return sorted(seen.items())


def _day_time_indices(date_) -> list[int]:
    """All GFS time indices whose Eastern-local date equals `date_`."""
    return [i for i, t in enumerate(_GFS_DS.time.values) if _to_et(t).date() == date_]


def _hour_options_for_day(day_first_idx: int) -> list[dict]:
    date_ = _to_et(_GFS_DS.time.values[int(day_first_idx)]).date()
    idxs = _day_time_indices(date_)
    return [
        {"label": _to_et(_GFS_DS.time.values[i]).strftime("%I:%M %p ET"), "value": i}
        for i in idxs
    ]


# Precomputed once at startup — the day list is identical for every variable.
_DAY_OPTIONS = [
    {"label": _to_et(_GFS_DS.time.values[idx]).strftime("%a %b %d"), "value": idx}
    for _, idx in _unique_forecast_days()
] if _GFS_DS is not None else []
_DEFAULT_DAY_VALUE = _DAY_OPTIONS[0]["value"] if _DAY_OPTIONS else None
_INITIAL_HOUR_OPTIONS = (
    _hour_options_for_day(_DEFAULT_DAY_VALUE) if _DEFAULT_DAY_VALUE is not None else []
)
_DEFAULT_HOUR_VALUE = _INITIAL_HOUR_OPTIONS[0]["value"] if _INITIAL_HOUR_OPTIONS else None


# ── figure helpers ────────────────────────────────────────────────────────────

def _field_to_image(
    data:    np.ndarray,
    cmap_name: str,
    vmin:    float,
    vmax:    float,
    opacity: float = 0.72,
) -> str:
    """Render a 2D field (N→S, W→E) to a base64 PNG for mapbox image layers.
    NaN cells (e.g. ocean, masked out) render fully transparent."""
    cmap = _MPL_CMAPS.get(cmap_name, plt.cm.viridis)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    valid = ~np.isnan(data)
    rgba = cmap(norm(np.where(valid, data, vmin)))
    rgba[:, :, 3] = np.where(valid, opacity, 0.0)
    rgba = np.clip(rgba, 0.0, 1.0)
    img_bytes = (rgba * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_bytes, mode="RGBA").save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _field_to_risk_image(data: np.ndarray, categories: list[tuple],
                         opacity: float = 0.72) -> str:
    """
    Render a 2D Heat Index field (N→S, W→E) as discrete NWS risk categories.
    Cells below the Caution threshold (and NaNs) are left fully transparent.
    """
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    filled = np.nan_to_num(data, nan=-999.0)
    for lower, _, hexcolor in categories:
        mask = filled >= lower
        r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = int(255 * opacity)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


NO_DATA_COLOR    = "#94a3b8"  # missing/unavailable value
NO_RISK_COLOR    = "#0ca30c"  # below Caution — a real category, not missing data
NO_RISK_LABEL    = "No Elevated Risk"


def _risk_color(v, categories: list[tuple]) -> str:
    """Map a raw Heat Index value (in the same unit as `categories`) to its color."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return NO_DATA_COLOR
    for lower, _, color in reversed(categories):
        if v >= lower:
            return color
    return NO_RISK_COLOR


def _bbox_zoom_center(bbox: list, width_px: float = PAGE_MAX_WIDTH - 48,
                      height_px: float = MAP_HEIGHT) -> tuple:
    """
    Mercator-correct "fit bounds" zoom (same approach as Google/Mapbox GL's
    fitBounds): computes the zoom needed to fit the bbox in each dimension
    separately and takes the more restrictive one, so the bbox is guaranteed
    to fit without overflowing either axis. The dcc.Graph container is
    responsive width-wise, so `width_px`/`height_px` are tuned to this app's
    capped page width/map height rather than measured live.
    """
    west, south, east, north = bbox
    center = {"lat": (south + north) / 2, "lon": (west + east) / 2}

    def _lat_rad(lat: float) -> float:
        s = math.sin(math.radians(lat))
        rad_x2 = math.log((1 + s) / (1 - s)) / 2
        return max(min(rad_x2, math.pi), -math.pi) / 2

    lat_fraction = (_lat_rad(north) - _lat_rad(south)) / math.pi
    lng_fraction = (east - west) / 360.0

    lat_zoom = math.log2(height_px / 256.0 / lat_fraction) if lat_fraction > 0 else 21.0
    lng_zoom = math.log2(width_px  / 256.0 / lng_fraction) if lng_fraction > 0 else 21.0

    zoom = max(2.0, min(7.5, round(min(lat_zoom, lng_zoom), 1)))
    return zoom, center


def _station_placeholder(message: str, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color="#64748b"), align="center")
    fig.update_layout(
        paper_bgcolor=_PANEL_BG, plot_bgcolor=_PANEL_BG,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=8, r=8, t=8, b=8), height=height,
    )
    return fig


def _mapbox_figure(
    data:     np.ndarray,
    lats:     np.ndarray,
    lons:     np.ndarray,
    var_key:  str,
    title:    str,
    station_values: list | None = None,
    uirevision: str = "default",
    unit: str = "C",
) -> go.Figure:
    """
    GFS field as raster image layer on CartoDB Positron, with colored
    station markers overlaid. Station markers share the field's colorscale
    so hot stations look red and cool stations look blue — same as the field.

    For var_key == "risk", the field/markers use the discrete NWS heat-index
    risk categories instead, with a categorical legend in place of a colorbar.
    """
    is_risk = var_key == "risk"
    west, east   = float(lons.min()), float(lons.max())
    south, north = float(lats.min()), float(lats.max())

    # Row 0 must be the north edge for the mapbox image layer
    plot_data = data[::-1, :] if lats[0] < lats[-1] else data
    plot_data = _convert_array(plot_data, unit)

    if is_risk:
        categories = _risk_categories(unit)
        img_src = _field_to_risk_image(plot_data, categories)
    else:
        vm = VARIABLE_META[var_key]
        vmin, vmax = _convert(vm["vmin"], unit), _convert(vm["vmax"], unit)
        img_src = _field_to_image(plot_data, vm["cmap"], vmin, vmax)

    if station_values is not None:
        station_values = [_convert(v, unit) for v in station_values]

    image_corners = [
        [west, north], [east, north], [east, south], [west, south],
    ]

    fig = go.Figure()

    if is_risk:
        # One legend-only ghost trace per category (mapbox has no discrete colorbar).
        # "No Elevated Risk" is listed first — it's a real category (below Caution),
        # not missing data, and needs to read that way at a glance.
        legend_entries = [(NO_RISK_COLOR, NO_RISK_LABEL)] + [(c, l) for _, l, c in categories]
        for color, label in legend_entries:
            fig.add_trace(go.Scattermapbox(
                lat=[89.0], lon=[0.0], mode="markers",
                marker=dict(size=10, color=color),
                name=label, showlegend=True, hoverinfo="none",
            ))
    else:
        # Off-screen ghost point carries the continuous colorbar
        fig.add_trace(go.Scattermapbox(
            lat=[89.0], lon=[0.0], mode="markers",
            marker=dict(
                size=1, color=[(vmin + vmax) / 2],
                colorscale=vm["plotly"], cmin=vmin, cmax=vmax,
                showscale=True,
                colorbar=dict(
                    title=dict(text=_unit_label(unit), font=dict(color="#cbd5e1")),
                    thickness=14, len=0.75,
                    bgcolor="rgba(15,23,42,0.7)",
                    tickfont=dict(color="#cbd5e1"),
                    x=1.0,
                ),
            ),
            showlegend=False, hoverinfo="none", opacity=0,
        ))

    # Station markers — colored by current GFS value at each station
    stn_lats = [s["lat"]  for s in MAJOR_CONUS_STATIONS]
    stn_lons = [s["lon"]  for s in MAJOR_CONUS_STATIONS]
    stn_ids  = [s["id"]   for s in MAJOR_CONUS_STATIONS]
    stn_text = [f"{s['id']} — {s['name']} ({s['state']})" for s in MAJOR_CONUS_STATIONS]

    if station_values is not None and is_risk:
        marker_kwargs = dict(color=[_risk_color(v, categories) for v in station_values],
                             colorscale=None, cmin=None, cmax=None)
    elif station_values is not None:
        marker_kwargs = dict(color=station_values, colorscale=vm["plotly"],
                             cmin=vmin, cmax=vmax)
    else:
        marker_kwargs = dict(color=NO_DATA_COLOR, colorscale=None, cmin=None, cmax=None)

    fig.add_trace(go.Scattermapbox(
        lat=stn_lats, lon=stn_lons, mode="markers",
        marker=dict(size=9, showscale=False, opacity=0.90, **marker_kwargs),
        customdata=stn_ids,
        hovertext=stn_text,
        hoverinfo="text",
        showlegend=False,
    ))

    zoom, center = _bbox_zoom_center(CONUS_BBOX)

    fig.update_layout(
        uirevision=uirevision,
        showlegend=is_risk,
        legend=dict(
            bgcolor="rgba(15,23,42,0.75)", bordercolor="#334155", borderwidth=1,
            font=dict(size=11, color="#cbd5e1"), x=0.01, y=0.01,
            xanchor="left", yanchor="bottom",
        ) if is_risk else None,
        title=dict(text=title, font=dict(size=13, color="#e2e8f0"),
                   x=0.01, xanchor="left"),
        mapbox=dict(
            style="carto-positron",
            center=center,
            zoom=zoom,
            layers=[dict(
                sourcetype="image", source=img_src,
                coordinates=image_corners, opacity=0.72, below="traces",
            )],
        ),
        height=MAP_HEIGHT,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
    )
    return fig


def _get_station_values(var_key: str, time_idx: int) -> list[float] | None:
    """Extract GFS values (°C) at each station's nearest grid point for the current time step."""
    if _GFS_DS is None or var_key not in _GFS_DS:
        return None
    da = _GFS_DS[var_key].isel(time=min(time_idx, len(_GFS_DS.time) - 1))
    vals = []
    for s in MAJOR_CONUS_STATIONS:
        try:
            v = float(da.sel(latitude=s["lat"], longitude=s["lon"],
                             method="nearest").values)
        except Exception:
            v = float("nan")
        vals.append(round(v, 1))
    return vals


def _get_station_risk_values(time_idxs: list[int]) -> list[float] | None:
    """Max Heat Index (°C) at each station's nearest grid point over a set of time steps."""
    if _GFS_DS is None or "hi" not in _GFS_DS:
        return None
    da = _GFS_DS["hi"].isel(time=time_idxs)
    vals = []
    for s in MAJOR_CONUS_STATIONS:
        try:
            v = float(da.sel(latitude=s["lat"], longitude=s["lon"],
                             method="nearest").max().values)
        except Exception:
            v = float("nan")
        vals.append(round(v, 1))
    return vals


def _leaderboard_table(time_idx: int, unit: str, top_n: int = 15) -> html.Div:
    """
    Ranked table of the hottest stations right now, by forecasted Heat Index.
    Ranked by *current forecasted severity*, not historical records — this app
    has no climate-normals data source, so it can't verify record claims.
    """
    if _GFS_DS is None:
        return html.Div()

    vals = _get_station_values("hi", int(time_idx or 0))
    if vals is None:
        return html.Div()

    ranked = sorted(
        (
            (s, v) for s, v in zip(MAJOR_CONUS_STATIONS, vals)
            if v is not None and not math.isnan(v)
        ),
        key=lambda pair: pair[1], reverse=True,
    )[:top_n]

    unit_label = _unit_label(unit)
    rows = [html.Div([
        html.Span("#",          style={"width": "26px", "fontWeight": "700"}),
        html.Span("City",       style={"flex": "1", "fontWeight": "700"}),
        html.Span("Feels Like", style={"width": "84px", "textAlign": "right", "fontWeight": "700"}),
        html.Span("Risk",       style={"width": "130px", "textAlign": "right", "fontWeight": "700"}),
    ], style={"display": "flex", "padding": "6px 10px", "fontSize": "11px",
              "color": "#94a3b8", "borderBottom": "1px solid #334155"})]

    for i, (s, hi_c) in enumerate(ranked, start=1):
        num_fmt = f"{_convert(hi_c, unit):.0f}" if unit == "F" else f"{_convert(hi_c, unit):.1f}"
        cat_label, cat_color = NO_RISK_LABEL, NO_RISK_COLOR
        for lower, label, color in reversed(RISK_CATEGORIES_C):
            if hi_c >= lower:
                cat_label, cat_color = label, color
                break
        rows.append(html.Div([
            html.Span(str(i), style={"width": "26px", "color": "#64748b"}),
            html.Span(f"{s['name']} ({s['state']})", style={"flex": "1", "color": "#e2e8f0"}),
            html.Span(f"{num_fmt}{unit_label}",
                      style={"width": "84px", "textAlign": "right",
                             "color": "#f8fafc", "fontWeight": "600"}),
            html.Span(html.Span(cat_label, style={
                "backgroundColor": cat_color, "color": "#0f172a", "padding": "2px 8px",
                "borderRadius": "999px", "fontSize": "10px", "fontWeight": "700",
            }), style={"width": "130px", "textAlign": "right"}),
        ], style={"display": "flex", "alignItems": "center", "padding": "6px 10px",
                  "fontSize": "12px", "borderBottom": "1px solid #1e293b"}))

    return html.Div([
        html.H3("Hottest Cities Right Now",
                style={"fontSize": "14px", "color": "#f8fafc", "margin": "0 0 4px 0"}),
        html.Div("Ranked by forecasted Heat Index for the selected day/time.",
                 style={"fontSize": "11px", "color": "#64748b", "marginBottom": "8px"}),
        html.Div(rows),
    ], style={"backgroundColor": "#1e293b", "borderRadius": "8px", "padding": "12px 14px",
              "marginTop": "16px"})


# ── station panel figure ──────────────────────────────────────────────────────

DEFAULT_SERIES = ["hi", "t2m", "obs"]


def _build_station_figure(station_id: str, asos_df: pd.DataFrame,
                          time_idx: int, unit: str = "C",
                          series: list[str] | None = None) -> go.Figure:
    """GFS forecast (T2m, HI) + ASOS observations for one station, in the
    station's own local time zone and the selected display unit. `series`
    controls which of t2m/hi/obs/dew traces are drawn."""
    if series is None:
        series = DEFAULT_SERIES
    stn = get_station(station_id)
    if stn is None:
        return _station_placeholder(f"Station {station_id} not in catalog.")
    if _GFS_DS is None:
        return _station_placeholder("GFS data not loaded.")

    tz = ZoneInfo(stn.get("tz", "America/New_York"))
    unit_label = _unit_label(unit)

    # Extract GFS time series at station's nearest grid point
    sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")
    gfs_t2m = _GFS_DS["t2m"].sel(**sel).to_series()
    gfs_hi  = _GFS_DS["hi"].sel(**sel).to_series()

    def _to_local(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(idx)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return idx.tz_convert(tz)

    gfs_t2m.index = _to_local(gfs_t2m.index)
    gfs_hi.index  = _to_local(gfs_hi.index)

    # Animation cursor
    times     = _GFS_DS.time.values
    cursor_ts = _to_local(pd.DatetimeIndex([times[min(int(time_idx or 0), len(times) - 1)]]))[0]
    tz_abbr   = cursor_ts.strftime("%Z")

    # ASOS obs run up to whenever they were fetched (real "now"), which is
    # typically several hours after the GFS init time — without this, the
    # forecast lines and the obs dots both cover that overlap window,
    # showing two different answers for the same past hours. Trim the
    # forecast lines to start only where real observations leave off.
    gfs_t2m_line, gfs_hi_line = gfs_t2m, gfs_hi
    if "obs" in series and not asos_df.empty:
        obs_local = asos_df["valid_utc"].dropna().dt.tz_convert(tz)
        if not obs_local.empty:
            latest_obs = obs_local.max()
            gfs_t2m_line = gfs_t2m[gfs_t2m.index > latest_obs]
            gfs_hi_line  = gfs_hi[gfs_hi.index > latest_obs]

    fig = go.Figure()

    # GFS forecast lines. Heat Index is the "feels like" number most people
    # already recognize, so it's the bold primary line; T2m is a thin reference.
    if "t2m" in series:
        fig.add_trace(go.Scatter(
            x=gfs_t2m_line.index, y=_convert_array(gfs_t2m_line.values, unit), mode="lines",
            line=dict(color="#38bdf8", width=1.4, dash="dot"),
            name="Actual Temp",
            hovertemplate=f"T2m: %{{y:.1f}}{unit_label}  %{{x|%b %d %I:%M %p}}<extra></extra>",
        ))
    if "hi" in series:
        fig.add_trace(go.Scatter(
            x=gfs_hi_line.index, y=_convert_array(gfs_hi_line.values, unit), mode="lines",
            line=dict(color="#fb923c", width=2.8),
            name="Feels Like (Heat Index)",
            hovertemplate=f"Feels like: %{{y:.1f}}{unit_label}  %{{x|%b %d %I:%M %p}}<extra></extra>",
        ))
    # ASOS observations (valid_utc is already tz-aware UTC)
    if not asos_df.empty:
        if "obs" in series:
            obs = asos_df.dropna(subset=["temp_c"]).copy()
            if not obs.empty:
                obs["valid_local"] = obs["valid_utc"].dt.tz_convert(tz)
                fig.add_trace(go.Scatter(
                    x=obs["valid_local"], y=_convert_array(obs["temp_c"].values, unit), mode="markers",
                    marker=dict(color="#38bdf8", size=5, opacity=0.85),
                    name=f"{station_id} obs",
                    hovertemplate=f"Obs: %{{y:.1f}}{unit_label}  %{{x|%b %d %I:%M %p}}<extra></extra>",
                ))
        if "dew" in series:
            dew = asos_df.dropna(subset=["dewpoint_c"]).copy()
            if not dew.empty:
                dew["valid_local"] = dew["valid_utc"].dt.tz_convert(tz)
                fig.add_trace(go.Scatter(
                    x=dew["valid_local"], y=_convert_array(dew["dewpoint_c"].values, unit), mode="markers",
                    marker=dict(color="#22d3ee", size=4, opacity=0.70, symbol="square"),
                    name=f"{station_id} Td obs",
                    hovertemplate=f"Td obs: %{{y:.1f}}{unit_label}  %{{x|%b %d %I:%M %p}}<extra></extra>",
                ))

    # Selected-time cursor — not literally "now": marks whatever day/time is
    # picked in the Day/Time dropdowns, which can be a future forecast day.
    # Distinct white so it doesn't blend with the amber threshold line.
    fig.add_shape(
        type="line",
        x0=cursor_ts.isoformat(), x1=cursor_ts.isoformat(),
        y0=0, y1=1, yref="paper",
        line=dict(color="rgba(226,232,240,0.85)", width=1.5, dash="dot"),
    )
    fig.add_annotation(
        x=cursor_ts.isoformat(), y=1, yref="paper",
        text="Selected time", showarrow=False,
        font=dict(size=9, color="#e2e8f0"),
        xanchor="left", yanchor="bottom",
    )

    # Reference line: NWS "Extreme Caution" threshold (32°C/90°F HI) — the
    # boundary on the official heat index chart, not a fixed national
    # "Excessive Heat Warning" trigger (those are set regionally by local
    # NWS offices and are typically much higher, 100-115°F+).
    # Only makes sense alongside the Heat Index series, so it follows that toggle.
    # Spans the full visible range — including the ASOS obs history, not just
    # the forecast portion from "now" onward — and is bold enough to read as
    # a hard line, not a faint gridline.
    x0_dt, x1_dt = gfs_t2m.index[0], gfs_t2m.index[-1]
    if "obs" in series and not asos_df.empty:
        obs_local = asos_df["valid_utc"].dropna().dt.tz_convert(tz)
        if not obs_local.empty:
            x0_dt = min(x0_dt, obs_local.min())
    x0, x1 = x0_dt.isoformat(), x1_dt.isoformat()
    for thresh_c, desc, color, requires in [
        (32.0, "(Extreme Caution begins)", "rgba(241,245,249,0.75)", "hi"),
    ]:
        if requires not in series:
            continue
        y_disp = _convert(thresh_c, unit)
        num_fmt = f"{y_disp:.0f}" if unit == "F" else f"{y_disp:.1f}"
        label = f"{num_fmt}{unit_label} {desc}"
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y_disp, y1=y_disp,
                      xref="x", yref="y",
                      line=dict(color=color, width=1.6, dash="dash"))
        fig.add_annotation(x=x1, y=y_disp, xref="x", yref="y",
                           text=label, showarrow=False,
                           font=dict(size=8, color=color),
                           xanchor="right", yanchor="bottom")

    fig.update_layout(
        paper_bgcolor=_PANEL_BG, plot_bgcolor=_PANEL_BG,
        title=dict(
            text=f"{station_id} — {stn['name']} ({stn['state']})  ·  times in {tz_abbr}",
            font=dict(size=13, color=_PANEL_FONT), x=0.01, xanchor="left",
        ),
        uirevision=station_id,
        xaxis=dict(gridcolor=_PANEL_GRID, color=_PANEL_FONT,
                   tickformat="%b %d\n%I %p", nticks=8, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=_PANEL_GRID, color=_PANEL_FONT,
                   title=dict(text=unit_label, font=dict(size=11)),
                   tickfont=dict(size=10)),
        legend=dict(orientation="h", font=dict(size=10, color=_PANEL_FONT),
                    bgcolor="rgba(0,0,0,0)", x=0, y=-0.22,
                    xanchor="left", yanchor="top", tracegroupgap=0),
        margin=dict(l=46, r=10, t=40, b=76),
        height=340,
    )
    return fig


def _hero_tile(station_id: str, time_idx: int, unit: str) -> html.Div:
    """Big 'feels like' number + plain-language risk category for the
    selected station/time — the headline fact, ahead of the line chart."""
    stn = get_station(station_id)
    if stn is None or _GFS_DS is None:
        return html.Div()

    tz  = ZoneInfo(stn.get("tz", "America/New_York"))
    idx = min(int(time_idx or 0), len(_GFS_DS.time) - 1)
    sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")
    hi_c = float(_GFS_DS["hi"].sel(**sel).isel(time=idx).values)
    ts_local = pd.Timestamp(_GFS_DS.time.values[idx]).tz_localize("UTC").tz_convert(tz)

    unit_label = _unit_label(unit)
    num_fmt = f"{_convert(hi_c, unit):.0f}" if unit == "F" else f"{_convert(hi_c, unit):.1f}"

    cat_label, cat_color = NO_RISK_LABEL, NO_RISK_COLOR
    for lower, label, color in reversed(RISK_CATEGORIES_C):
        if hi_c >= lower:
            cat_label, cat_color = label, color
            break
    desc = RISK_DESCRIPTIONS.get(cat_label, "")

    return html.Div(
        style={"padding": "14px 18px", "backgroundColor": "#1e293b",
               "borderRadius": "8px", "marginBottom": "10px",
               "border": f"1px solid {cat_color}55"},
        children=[
            html.Div([
                html.Span(f"{num_fmt}{unit_label}",
                          style={"fontSize": "42px", "fontWeight": "700", "color": "#f8fafc"}),
                html.Span(f"  Feels like  ·  {ts_local.strftime('%a %I:%M %p').replace(' 0', ' ')}",
                          style={"fontSize": "13px", "color": "#94a3b8", "marginLeft": "8px"}),
            ]),
            html.Div([
                html.Span(cat_label, style={
                    "backgroundColor": cat_color, "color": "#0f172a", "padding": "3px 10px",
                    "borderRadius": "999px", "fontSize": "11px", "fontWeight": "700",
                }),
                html.Span(desc, style={"fontSize": "12px", "color": "#94a3b8"}),
            ], style={"marginTop": "8px", "display": "flex", "alignItems": "center",
                      "flexWrap": "wrap", "gap": "8px"}),
        ],
    )




# ── Dash app ──────────────────────────────────────────────────────────────────

app    = Dash(__name__)
server = app.server   # Gunicorn entry point
app.title = "US Heat Wave Tracker"

_SOCIAL_DESCRIPTION = (
    "Live GFS forecast — Heat Index, risk levels, and real ASOS observations "
    "for 165 US cities during the July 2026 heat wave."
)

# Open Graph / Twitter Card tags so a shared link renders a real preview card
# (title + description + image) instead of a blank/generic one. This is the
# single highest-leverage thing for click-through when this gets shared.
app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <meta property="og:title" content="US Heat Wave Tracker">
        <meta property="og:description" content="{_SOCIAL_DESCRIPTION}">
        <meta property="og:type" content="website">
        <meta property="og:image" content="/assets/social_preview.png">
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="US Heat Wave Tracker">
        <meta name="twitter:description" content="{_SOCIAL_DESCRIPTION}">
        <meta name="twitter:image" content="/assets/social_preview.png">
        <meta name="description" content="{_SOCIAL_DESCRIPTION}">
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""

_init_label = f"Init: {_GFS_INIT}" if _GFS_DS is not None else "No data"

_LABEL_STYLE = {"fontWeight": "600", "fontSize": "12px", "color": "#94a3b8",
                "display": "block", "marginBottom": "4px"}
_DROPDOWN_STYLE = {"width": "170px", "fontSize": "13px", "color": "#0f172a"}


def _page_section(bg, border, children):
    """Full-bleed colored bar with content capped to PAGE_MAX_WIDTH and centered."""
    return html.Div(
        style={"backgroundColor": bg, "borderBottom": f"1px solid {border}"} if bg else {},
        children=html.Div(
            style={"maxWidth": f"{PAGE_MAX_WIDTH}px", "margin": "0 auto"},
            children=children,
        ),
    )


app.layout = html.Div(
    style={"fontFamily": "sans-serif", "backgroundColor": "#0f172a",
           "minHeight": "100vh", "color": "#e2e8f0"},
    children=[

        # ── header ────────────────────────────────────────────────────────────
        _page_section("#1e293b", "#334155", html.Div(
            style={"padding": "10px 24px",
                   "display": "flex", "alignItems": "center",
                   "justifyContent": "space-between"},
            children=[
                html.H1("US Heat Wave Tracker",
                    style={"color": "#f8fafc", "margin": 0,
                           "fontSize": "20px", "fontWeight": "600"}),
                html.Span(_init_label,
                    style={"fontSize": "12px", "color": "#64748b"}),
            ],
        )),

        # ── controls ──────────────────────────────────────────────────────────
        _page_section("#1e293b", "#334155", html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "flex-end",
                   "padding": "10px 24px", "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Label("Variable", style=_LABEL_STYLE),
                    dcc.Dropdown(
                        id="variable-selector",
                        options=[
                            {"label": "2m Temperature",          "value": "t2m"},
                            {"label": "Heat Index",               "value": "hi"},
                            {"label": "Risk Level (Heat Index)",  "value": "risk"},
                        ],
                        value="t2m",
                        clearable=False,
                        style={"width": "220px", "fontSize": "13px", "color": "#0f172a"},
                    ),
                ]),
                html.Div([
                    html.Label("Units", style=_LABEL_STYLE),
                    dcc.RadioItems(
                        id="unit-selector",
                        options=[{"label": " °F", "value": "F"},
                                {"label": " °C", "value": "C"}],
                        value="F", inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "14px", "fontSize": "13px",
                                    "color": "#cbd5e1"},
                    ),
                ]),
                html.Div([
                    html.Label("Day", style=_LABEL_STYLE),
                    dcc.Dropdown(
                        id="day-dropdown", options=_DAY_OPTIONS,
                        value=_DEFAULT_DAY_VALUE, clearable=False,
                        style=_DROPDOWN_STYLE,
                    ),
                ]),
                html.Div(id="hour-dropdown-wrap", children=[
                    html.Label("Time", style=_LABEL_STYLE),
                    dcc.Dropdown(
                        id="hour-dropdown", options=_INITIAL_HOUR_OPTIONS,
                        value=_DEFAULT_HOUR_VALUE, clearable=False,
                        style=_DROPDOWN_STYLE,
                    ),
                ]),
            ],
        )),

        # ── CONUS map ─────────────────────────────────────────────────────────
        _page_section(None, None, html.Div(
            style={"padding": "16px 24px 0 24px"},
            children=[
                dcc.Graph(
                    id="field-map",
                    style={"height": f"{MAP_HEIGHT}px"},
                    config={"scrollZoom": True},
                ),
                html.Label(id="time-label",
                    style={"display": "block", "textAlign": "center",
                           "fontSize": "13px", "color": "#94a3b8",
                           "margin": "8px 0"}),
                html.Div(id="risk-caption"),
                html.Div(id="leaderboard-panel"),
            ],
        )),

        # ── station panel ─────────────────────────────────────────────────────
        _page_section(None, None, html.Div(
            style={"padding": "0 24px 24px 24px"},
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center",
                           "justifyContent": "space-between", "flexWrap": "wrap",
                           "gap": "8px", "paddingTop": "8px",
                           "borderTop": "1px solid #334155", "marginBottom": "8px"},
                    children=[
                        html.Div(
                            "Click a station on the map to see GFS forecast + ASOS observations",
                            id="station-hint",
                            style={"fontSize": "12px", "color": "#475569"},
                        ),
                        dcc.Checklist(
                            id="series-selector",
                            options=[
                                {"label": " Feels Like (Heat Index)", "value": "hi"},
                                {"label": " Actual Temp",             "value": "t2m"},
                                {"label": " ASOS Temp obs",           "value": "obs"},
                                {"label": " ASOS Dewpoint obs",       "value": "dew"},
                            ],
                            value=DEFAULT_SERIES, inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "14px", "fontSize": "12px",
                                        "color": "#cbd5e1"},
                        ),
                    ],
                ),
                html.Div([
                    "Heat Index = how hot it feels once humidity is factored in (NWS formula). ",
                    html.A("Learn more at weather.gov",
                          href="https://www.weather.gov/safety/heat-index",
                          target="_blank",
                          style={"color": "#64748b", "textDecoration": "underline"}),
                ], style={"fontSize": "11px", "color": "#64748b", "marginBottom": "10px",
                         "maxWidth": "820px", "lineHeight": "1.5"}),
                html.Div(id="station-panel"),
            ],
        )),

        # ── footer ────────────────────────────────────────────────────────────
        _page_section(None, None, html.Div(
            style={"padding": "16px 24px 24px 24px", "borderTop": "1px solid #1e293b",
                   "textAlign": "center", "fontSize": "11px", "color": "#475569"},
            children=[
                html.Div("Built by Sebastian Otarola-Bustos, PhD  ·  Rockville, MD"),
                html.Div(
                    "Forecast: NOAA GFS  ·  Observations: NOAA/Iowa Environmental Mesonet ASOS  ·  "
                    "Risk categories: National Weather Service",
                    style={"marginTop": "4px", "color": "#374151"},
                ),
            ],
        )),

        # ── hidden components ─────────────────────────────────────────────────
        dcc.Store(id="selected-station"),
        dcc.Store(id="current-time-idx"),
    ],
)


# ── callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("hour-dropdown", "options"),
    Output("hour-dropdown", "value"),
    Input("day-dropdown", "value"),
)
def update_hour_options(day_first_idx):
    if _GFS_DS is None or day_first_idx is None:
        return [], None
    options = _hour_options_for_day(day_first_idx)
    return options, options[0]["value"] if options else None


@app.callback(
    Output("hour-dropdown-wrap", "style"),
    Input("variable-selector", "value"),
)
def toggle_hour_dropdown(var_key):
    return {"display": "none"} if var_key == "risk" else {}


@app.callback(
    Output("current-time-idx", "data"),
    Input("day-dropdown", "value"),
    Input("hour-dropdown", "value"),
    Input("variable-selector", "value"),
)
def update_current_time_idx(day_idx, hour_idx, var_key):
    if var_key == "risk":
        return day_idx if day_idx is not None else 0
    return hour_idx if hour_idx is not None else (day_idx if day_idx is not None else 0)


@app.callback(
    Output("field-map",        "figure"),
    Output("time-label",       "children"),
    Output("leaderboard-panel", "children"),
    Input("variable-selector", "value"),
    Input("current-time-idx",  "data"),
    Input("unit-selector",     "value"),
)
def update_map(var_key, time_idx, unit):
    time_idx = int(time_idx or 0)

    if _GFS_DS is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Run: python scripts/fetch_gfs_conus.py",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#64748b"),
        )
        fig.update_layout(height=MAP_HEIGHT, paper_bgcolor="#0f172a",
                          plot_bgcolor="#0f172a",
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig, "No data", html.Div()

    lats = _GFS_DS.latitude.values
    lons = _GFS_DS.longitude.values
    leaderboard = _leaderboard_table(time_idx, unit)

    if var_key == "risk":
        local_date = _to_et(_GFS_DS.time.values[time_idx]).date()
        idxs = _day_time_indices(local_date)
        data = _GFS_DS["hi"].isel(time=idxs).max(dim="time").values
        day_label = _to_et(_GFS_DS.time.values[time_idx]).strftime("%A, %b %d")
        title = f"US Heat Wave  ·  Risk Level (daily max Heat Index)  |  {day_label}"
        stn_vals = _get_station_risk_values(idxs)
        fig = _mapbox_figure(
            data=data, lats=lats, lons=lons, var_key="risk", title=title,
            station_values=stn_vals, uirevision="conus_risk", unit=unit,
        )
        return fig, day_label, leaderboard

    vm     = VARIABLE_META.get(var_key, VARIABLE_META["t2m"])
    da     = _GFS_DS[var_key].isel(time=time_idx)
    data   = da.values

    ts_str = _et_utc_label(_GFS_DS.time.values[time_idx])
    title  = f"US Heat Wave  ·  {vm['label']}  |  {ts_str}"

    stn_vals = _get_station_values(var_key, time_idx)

    fig = _mapbox_figure(
        data           = data,
        lats           = lats,
        lons           = lons,
        var_key        = var_key,
        title          = title,
        station_values = stn_vals,
        uirevision     = f"conus_{var_key}",
        unit           = unit,
    )
    return fig, ts_str, leaderboard


@app.callback(
    Output("risk-caption", "children"),
    Input("variable-selector", "value"),
)
def update_risk_caption(var_key):
    """Plain-language legend under the map — what each risk color actually means."""
    if var_key != "risk":
        return []
    order = [(NO_RISK_LABEL, NO_RISK_COLOR)] + [(label, color) for _, label, color in RISK_CATEGORIES_F]
    chips = []
    for label, color in order:
        chips.append(html.Span([
            html.Span(label, style={
                "backgroundColor": color, "color": "#0f172a", "padding": "2px 8px",
                "borderRadius": "999px", "fontWeight": "700", "marginRight": "6px",
                "fontSize": "11px",
            }),
            html.Span(RISK_DESCRIPTIONS.get(label, ""), style={"color": "#94a3b8"}),
        ], style={"marginRight": "20px", "whiteSpace": "nowrap"}))
    return html.Div(chips, style={"display": "flex", "flexWrap": "wrap",
                                  "gap": "6px 0", "fontSize": "11px",
                                  "padding": "8px 0 0 0"})


@app.callback(
    Output("selected-station", "data"),
    Input("field-map", "clickData"),
    State("selected-station", "data"),
)
def select_station(clickData, current):
    if not clickData:
        return current
    pts = clickData.get("points", [{}])
    if not pts:
        return current
    cdata = pts[0].get("customdata")
    # Only accept clicks on station markers (customdata = ICAO code starting with K)
    if cdata and isinstance(cdata, str) and cdata.startswith("K"):
        return cdata
    return current


@app.callback(
    Output("station-panel", "children"),
    Output("station-hint",  "children"),
    Input("selected-station", "data"),
    Input("current-time-idx", "data"),
    Input("unit-selector",    "value"),
    Input("series-selector",  "value"),
)
def update_station_panel(station_id, time_idx, unit, series):
    time_idx = int(time_idx or 0)

    if not station_id:
        return (
            html.Div(style={"height": "60px"}),
            "Click a station on the map to see GFS forecast + ASOS observations",
        )

    stn = get_station(station_id)
    if stn is None:
        return (
            _station_placeholder(f"Station {station_id} not in catalog."),
            f"Station: {station_id}",
        )

    # Fetch ASOS from IEM on first click; serve from cache on subsequent updates.
    # Only cache non-empty results — a transient IEM failure/rate-limit (empty
    # df) shouldn't be remembered forever, or the station gets stuck at 0 obs.
    if station_id in _asos_cache:
        asos_df = _asos_cache[station_id]
    else:
        print(f"[app] Fetching ASOS for {station_id} from IEM ...", end=" ")
        asos_df = fetch_station_obs(station_id, hours=72)
        if not asos_df.empty:
            _asos_cache[station_id] = asos_df
        print(f"{len(asos_df)} obs")

    fig   = _build_station_figure(station_id, asos_df, time_idx, unit=unit, series=series)
    hint  = (f"Station: {station_id} — {stn['name']} ({stn['state']})  "
             f"·  {len(asos_df)} recent ASOS obs  ·  Click another station to switch")

    panel = html.Div([
        _hero_tile(station_id, time_idx, unit),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ])
    return panel, hint


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if _GFS_DS is None:
        print(
            "\nNo GFS data found. Pre-fetch it first:\n"
            "    python scripts/fetch_gfs_conus.py\n"
        )
    app.run(debug=True, host="0.0.0.0", port=8051)
