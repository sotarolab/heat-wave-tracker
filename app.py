"""
heat-wave-tracker / app.py
============================
CONUS heat wave dashboard. GFS surface field on a map, with a station
click panel showing the forecast, real ASOS observations, a same-day
bias correction, a Brier score / RMSE verification summary, and a GEV
return-period estimate against each station's own historical record.

Variables: 2m Temperature, Heat Index (NWS), Risk Level (NWS heat index
categories, daily max).

Run locally:
    python app.py

On Render (Gunicorn):
    gunicorn app:server

Gotchas:

1. GFS times are stored tz-naive but represent UTC. The map always
   displays Eastern Time (a single CONUS raster snapshot spans 4 time
   zones at once, so there is no true "local" time for it), while each
   station panel shows that station's own local time. These two clocks
   can look "desynced" even when both are correct, confirmed live with
   Yuma, AZ, which does not observe DST: the map's "4:00 AM EDT" and
   the station's own "1:00 AM" were the same instant, but nothing said
   so. Fixed by leading with Eastern Time everywhere station-local time
   is shown too, with the local time in parentheses only when it
   actually differs from Eastern.
2. Scattermapbox station highlighting does not reliably update the
   position of a newly created overlay trace across Plotly.react()
   calls without a paid Mapbox token, confirmed live after several
   failed attempts at the more obvious approach. The fix used here
   bakes highlight state into the per-point size/color arrays of
   already-existing, already-reliable traces instead of adding a
   separate overlay trace per selection.
3. A single Dash callback cannot both consume a component as an Input
   and structurally recreate that same component as part of its
   output's children. bias-window-dropdown and bias-display-mode are
   both Inputs to update_station_panel and are also driven by that same
   callback's Outputs, so both are declared once as static layout
   components (with explicit prop-level Outputs) rather than being
   rebuilt inside the dynamically-generated station panel.
4. The default time-slider position picks whichever forecast step is
   closest to now, even if that is a few minutes or hours in the past,
   rather than always rounding up to the next future step. Confirmed
   live: at 8:30 PM the only forward option under 2-hour GFS resolution
   was 10 PM, whose forecast read noticeably cooler than what conditions
   actually felt like at 8:30 PM.
5. The hero tile, leaderboard, and map always show the forecast value
   for the selected time, never a live-swapped real observation, even
   when the selected time is close to now. An earlier version swapped
   in a live observation near "now," which meant the same station at
   the same nominal moment could show two different numbers depending
   on which part of the UI you looked at. A fresh real observation is
   shown as a clearly labeled secondary line instead of replacing the
   forecast headline.
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
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL, no_update

from src.heat.gfs_conus import load_or_fetch, DEFAULT_OUT
from src.heat.stations  import MAJOR_CONUS_STATIONS, get_station
from src.heat.asos      import fetch_station_obs
from src.heat.compute   import heat_index_array
from src.heat.bias      import (today_forecast_bias, brier_score_exceedance,
                                BIAS_MIN_PAIRS, BIAS_MATCH_TOLERANCE)
from src.heat.extremes  import (fit_gev, return_level, return_period, support,
                                plotting_positions, GEV_MIN_YEARS)


# ── constants ─────────────────────────────────────────────────────────────────

DATA_PATH  = DEFAULT_OUT
CONUS_BBOX = [-127.0, 23.0, -65.0, 51.0]

# Overall page content is capped to this width and centered - keeps the map's
# aspect ratio sane on wide monitors and matches the zoom heuristic below.
PAGE_MAX_WIDTH = 1400
MAP_HEIGHT     = 620

# Mirrors assets/mobile.css's own #field-map height override, which only
# applies below a 600px *window* width. viewport-width (see the
# clientside callback near app.layout) already nets out the ~48px
# container padding the CSS breakpoint doesn't need to, so the threshold
# here is shifted by that same amount to stay in sync with the CSS.
MOBILE_WIDTH_BREAKPOINT = 600 - 48
MOBILE_MAP_HEIGHT = 380

# Reference timezone for map-level labels (a single CONUS raster snapshot spans
# 4 zones at once, so there's no true "local" time for it - Eastern is used as
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

# NWS heat index risk categories - native NOAA thresholds (°F) plus their
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

# "Extreme Caution begins" threshold (90F/32C), shared by the station
# chart's reference line and the Brier score exceedance event so the two
# can't silently drift apart into different numbers.
_EXTREME_CAUTION_HI_C = 32.0

# Plain-language versions of NOAA's official heat-index risk definitions,
# same source NWS/NYT-style heat maps cite, phrased for a non-meteorologist.
RISK_DESCRIPTIONS = {
    "No Elevated Risk": "Comfortable - no unusual heat risk.",
    "Caution":          "Fatigue possible with prolonged outdoor exposure or activity.",
    "Extreme Caution":  "Heat cramps or exhaustion possible with prolonged exposure or activity.",
    "Danger":           "Heat cramps or exhaustion likely; heat stroke possible if prolonged.",
    "Extreme Danger":   "Heat stroke highly likely.",
}

# Dewpoint comfort bands - NWS convention, dewpoint rather than relative
# humidity because RH is misleading for comfort (it's relative to
# temperature: 90% RH at 60F is a pleasant morning, 90% RH at 85F is
# dangerous), while dewpoint directly reflects moisture content regardless
# of temperature.
DEWPOINT_BANDS_F = [
    (70.0, "Muggy",      "#f2994a"),
    (75.0, "Oppressive", "#e8592e"),
]
DEWPOINT_BANDS_C = [
    (round((f - 32.0) * 5.0 / 9.0, 1), label, color)
    for f, label, color in DEWPOINT_BANDS_F
]


def _dewpoint_band(td_c: float) -> tuple[str, str]:
    """Plain-language dewpoint comfort band. Returns ('Comfortable', muted
    color) below the lowest threshold."""
    label, color = "Comfortable", "#64748b"
    for lower, lbl, c in DEWPOINT_BANDS_C:
        if td_c >= lower:
            label, color = lbl, c
    return label, color


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

# In-process cache for ASOS obs, keyed by station_id -> (fetched_at, df).
# TTL'd so the app doesn't keep serving an ever-staler snapshot for stations
# that were clicked once early in the process's lifetime.
_asos_cache: dict[str, tuple[pd.Timestamp, pd.DataFrame]] = {}
_ASOS_CACHE_TTL = pd.Timedelta(minutes=15)

# NWS Daily Climate Report data (actual vs 1991-2020 normal), bulk-fetched
# once via scripts/fetch_climate_normals.py - not every station has one, so
# a missing key here just means that station's climate context isn't shown.
_CLIMATE_NORMALS_PATH = Path("data") / "climate_normals.json"
if _CLIMATE_NORMALS_PATH.exists():
    import json as _json
    _CLIMATE_NORMALS = _json.loads(_CLIMATE_NORMALS_PATH.read_text())
    print(f"[app] Climate normals loaded: {len(_CLIMATE_NORMALS)} stations")
else:
    _CLIMATE_NORMALS = {}

# 1972-2025 annual max temperature per station (scripts/fetch_historical_temps.py),
# used for the GEV return-period popup. Record length varies by station
# (some ASOS records only start in the 1980s/90s) - each fit reports its
# own n_years rather than assuming the full range, and GEV fits are
# skipped below extremes.GEV_MIN_YEARS regardless of what's on disk.
_CLIMATE_EXTREMES_PATH = Path("data") / "climate_extremes.parquet"
_GEV_FITS: dict[str, dict] = {}
_GEV_ANNUAL_MAXIMA: dict[str, np.ndarray] = {}  # raw annual maxima (F), for the histogram overlay
if _CLIMATE_EXTREMES_PATH.exists():
    _extremes_df = pd.read_parquet(_CLIMATE_EXTREMES_PATH)
    for _sid, _grp in _extremes_df.groupby("station"):
        _fit = fit_gev(_grp["annual_max_temp_f"].values)
        if _fit is not None:
            _GEV_FITS[_sid] = _fit
            _GEV_ANNUAL_MAXIMA[_sid] = _grp["annual_max_temp_f"].values
    print(f"[app] Historical GEV fits: {len(_GEV_FITS)} stations "
          f"({len(_extremes_df)} station-years)")
else:
    print("[app] No data/climate_extremes.parquet - GEV popup disabled "
          "(run scripts/fetch_historical_temps.py)")


# ── unit helpers ──────────────────────────────────────────────────────────────

def _convert(value_c, unit: str):
    """Celsius -> display unit. NaN-safe (arithmetic on NaN stays NaN)."""
    return value_c * 9.0 / 5.0 + 32.0 if unit == "F" else value_c


def _convert_array(arr_c: np.ndarray, unit: str) -> np.ndarray:
    return arr_c * 9.0 / 5.0 + 32.0 if unit == "F" else arr_c


def _convert_delta(delta_c: float, unit: str) -> float:
    """Convert a temperature *difference* (not an absolute value) - no +32
    offset, since that only applies to absolute Celsius->Fahrenheit points."""
    return delta_c * 9.0 / 5.0 if unit == "F" else delta_c


def _unit_label(unit: str) -> str:
    return "°F" if unit == "F" else "°C"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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
    with the index of its first GFS timestep (used for the time-slider's
    day-boundary marks).
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


def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=_DISPLAY_TZ)


def _default_time_idx() -> int:
    """
    Index of the closest-to-now forecast step across the whole series - not
    day-scoped like the old per-day hour-dropdown default was, since the
    slider spans the entire forecast and "today" isn't a special case at
    this level anymore. Same don't-round-up-to-next-future reasoning as
    before: the nearest step is used even if it's a few minutes/hours in
    the past, rather than always the next future step, which with 2-hour-
    resolution GFS data could be misleadingly different from current actual
    conditions (caught in practice: at 8:30 PM the only forward option was
    10 PM, whose forecast read cooler than what it actually felt like).
    """
    if _GFS_DS is None:
        return 0
    now = _now_et()
    times = _GFS_DS.time.values
    return int(min(range(len(times)),
                   key=lambda i: abs((_to_et(times[i]) - now).total_seconds())))


# Precomputed once at startup.
_TIME_SLIDER_MAX = (len(_GFS_DS.time.values) - 1) if _GFS_DS is not None else 0
_TIME_SLIDER_MARKS = {
    idx: {"label": _to_et(_GFS_DS.time.values[idx]).strftime("%a %b %d"),
         "style": {"fontSize": "10px", "color": "#94a3b8"}}
    for _, idx in _unique_forecast_days()
} if _GFS_DS is not None else {}
_DEFAULT_TIME_IDX = _default_time_idx()


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
NO_RISK_COLOR    = "#0ca30c"  # below Caution - a real category, not missing data
NO_RISK_LABEL    = "No Elevated Risk"


def _risk_color(v, categories: list[tuple]) -> str:
    """Map a raw Heat Index value (in the same unit as `categories`) to its color."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return NO_DATA_COLOR
    for lower, _, color in reversed(categories):
        if v >= lower:
            return color
    return NO_RISK_COLOR


def _size_for_values(values, vmin: float, vmax: float,
                     min_size: float = 6.0, max_size: float = 18.0) -> list[float]:
    """Marker size scaled linearly with value, so hotter stations read as
    bigger dots too, not just a different color - a second, redundant
    channel that's easier to scan at a glance and doesn't rely on color
    perception alone."""
    span = vmax - vmin
    sizes = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            sizes.append(min_size)
            continue
        frac = (v - vmin) / span if span else 0.5
        frac = max(0.0, min(1.0, frac))
        sizes.append(min_size + frac * (max_size - min_size))
    return sizes


def _bbox_zoom_center(bbox: list, width_px: float = PAGE_MAX_WIDTH - 48,
                      height_px: float = MAP_HEIGHT, north_bias: float = 0.0) -> tuple:
    """
    Mercator-correct "fit bounds" zoom (same approach as Google/Mapbox GL's
    fitBounds): computes the zoom needed to fit the bbox in each dimension
    separately and takes the more restrictive one, so the bbox is guaranteed
    to fit without overflowing either axis. The dcc.Graph container is
    responsive width-wise, so `width_px`/`height_px` are tuned to this app's
    capped page width/map height rather than measured live.

    north_bias : float, optional
        0 (default) centers the extra margin symmetrically N/S, exactly as
        before. When the frame's aspect ratio is much taller/narrower than
        the bbox's own shape (e.g. CONUS's wide ~2.2:1 landscape squeezed
        into a near-square mobile map), fitting by width alone reveals far
        more vertical extent than the bbox needs, split evenly N/S - on a
        US-focused map that means showing as much of Mexico/Central
        America to the south as ocean/Canada to the north, which reads as
        broken rather than just "some margin." A positive north_bias
        shifts that same total extra margin northward instead of
        splitting it evenly - 1.0 pushes all of it north (no southward
        excess at all), 0 leaves it symmetric.
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

    if north_bias and lat_fraction > 0 and zoom < lat_zoom:
        shown_lat_fraction = height_px / (256.0 * 2 ** zoom)
        extra_fraction = shown_lat_fraction - lat_fraction
        if extra_fraction > 0:
            # Linear degrees-per-fraction approximation, fine at CONUS's
            # latitudes (not close enough to the poles for Mercator's
            # nonlinearity to matter for a UI framing decision like this).
            extra_degrees = extra_fraction / lat_fraction * (north - south)
            center["lat"] += north_bias * extra_degrees / 2

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


def _frame_traces(data, lats, lons, var_key, unit, station_values, selected_station,
                  categories=None, vmin=None, vmax=None):
    """The per-timestep pieces of the CONUS map: the raster image layer
    (as a mapbox-layer dict) and the two station-marker traces (halo +
    colored dots), split out from _mapbox_figure as a single source of
    truth for "what does timestep X look like."

    For var_key == "risk", pass categories (from _risk_categories);
    otherwise pass vmin/vmax. Both already unit-converted by the caller,
    same as station_values is expected raw (this function converts it).

    Returns
    -------
    tuple
        (image_layer: dict, halo_trace: go.Scattermapbox,
        marker_trace: go.Scattermapbox)
    """
    is_risk = var_key == "risk"
    west, east   = float(lons.min()), float(lons.max())
    south, north = float(lats.min()), float(lats.max())

    # Row 0 must be the north edge for the mapbox image layer
    plot_data = data[::-1, :] if lats[0] < lats[-1] else data
    plot_data = _convert_array(plot_data, unit)

    # Downsampled just for the image encode: the source grid (113x249)
    # already exceeds what's visually distinguishable for a smoothed color
    # field at map display size, and PNG encoding is real per-request CPU
    # cost, so this cuts payload size (and encode time) roughly 3x with no
    # visible quality loss. image_corners below maps physical lat/lon
    # bounds, not pixel count, so it's unaffected by this.
    img_data = plot_data[::2, ::2]

    if is_risk:
        img_src = _field_to_risk_image(img_data, categories)
    else:
        img_src = _field_to_image(img_data, VARIABLE_META[var_key]["cmap"], vmin, vmax)

    if station_values is not None:
        station_values = [_convert(v, unit) for v in station_values]

    image_corners = [
        [west, north], [east, north], [east, south], [west, south],
    ]
    image_layer = dict(sourcetype="image", source=img_src,
                       coordinates=image_corners, opacity=0.72, below="traces")

    # Station markers - colored by current GFS value at each station
    stn_lats = [s["lat"]  for s in MAJOR_CONUS_STATIONS]
    stn_lons = [s["lon"]  for s in MAJOR_CONUS_STATIONS]
    stn_ids  = [s["id"]   for s in MAJOR_CONUS_STATIONS]
    stn_text = [f"{s['id']} - {s['name']} ({s['state']})" for s in MAJOR_CONUS_STATIONS]

    # Size uses today's actual min/max, not the fixed color-scale range -
    # otherwise a typical day's spread (e.g. 75-105F within a 50-113F color
    # range) only occupies the middle of the size range and barely reads as
    # different sizes at all.
    valid_vals = [v for v in (station_values or [])
                 if v is not None and not (isinstance(v, float) and math.isnan(v))]
    size_lo, size_hi = (min(valid_vals), max(valid_vals)) if valid_vals else (0.0, 1.0)

    if station_values is not None and is_risk:
        marker_kwargs = dict(color=[_risk_color(v, categories) for v in station_values],
                             colorscale=None, cmin=None, cmax=None,
                             size=_size_for_values(station_values, size_lo, size_hi))
    elif station_values is not None:
        marker_kwargs = dict(color=station_values, colorscale=VARIABLE_META[var_key]["plotly"],
                             cmin=vmin, cmax=vmax,
                             size=_size_for_values(station_values, size_lo, size_hi))
    else:
        marker_kwargs = dict(color=NO_DATA_COLOR, colorscale=None, cmin=None, cmax=None,
                             size=9)

    # Scattermapbox markers have no border/line property (unlike regular
    # Scatter), so a dark "halo" trace underneath - same points, slightly
    # larger, solid dark navy - fakes a thin outline. Without it, light
    # station colors disappear into light basemap/field colors. Kept small
    # and fixed-width (not proportional to marker size) so it reads as a
    # subtle border even on the smallest dots, not a dominant ring that
    # drowns out the size-by-severity signal.
    sizes = marker_kwargs["size"]
    sizes_list = sizes if isinstance(sizes, list) else [sizes] * len(stn_ids)
    halo_sizes = [s + 1.5 for s in sizes_list]

    # Selection highlight: baked directly into the halo trace's per-point
    # size/color (bigger + blue instead of the uniform dark navy border)
    # rather than a separate overlay trace (see module Gotcha 2). Two
    # prior attempts at a separate ring trace (plain Scattermapbox
    # marker, and a duplicate-output callback) both showed the ring
    # correctly on the first selection ever made and then never moved
    # again on later clicks, even though the server-sent figure JSON
    # was verified correct every time via direct API calls. The
    # halo/color traces do not have that problem, their per-point
    # size/color arrays already update correctly on every render (that
    # is how station colors track the selected time/variable right
    # now), so the highlight rides along on a trace already proven to
    # work.
    sel_idx = stn_ids.index(selected_station) if selected_station in stn_ids else None
    halo_colors = ["#0f172a"] * len(stn_ids)
    if sel_idx is not None:
        halo_sizes[sel_idx] = sizes_list[sel_idx] + 10
        halo_colors[sel_idx] = "#38bdf8"

    halo_trace = go.Scattermapbox(
        lat=stn_lats, lon=stn_lons, mode="markers",
        marker=dict(size=halo_sizes, color=halo_colors, opacity=0.55),
        hoverinfo="none", showlegend=False,
    )

    # Name label for the selected station, folded into the color-marker
    # trace below via per-point text (empty string for everyone else) -
    # same reasoning as the halo: this trace's per-point arrays are
    # already proven to update reliably, a separate text trace was not.
    label_text = ["" for _ in stn_ids]
    if sel_idx is not None:
        label_text[sel_idx] = MAJOR_CONUS_STATIONS[sel_idx]["name"]

    marker_trace = go.Scattermapbox(
        lat=stn_lats, lon=stn_lons,
        mode="markers+text" if sel_idx is not None else "markers",
        marker=dict(showscale=False, opacity=0.90, **marker_kwargs),
        text=label_text,
        textposition="top center",
        textfont=dict(size=13, color="#38bdf8"),
        customdata=stn_ids,
        hovertext=stn_text,
        hoverinfo="text",
        showlegend=False,
    )

    return image_layer, halo_trace, marker_trace


def _mapbox_figure(
    data:     np.ndarray,
    lats:     np.ndarray,
    lons:     np.ndarray,
    var_key:  str,
    title:    str,
    station_values: list | None = None,
    uirevision: str = "default",
    unit: str = "C",
    selected_station: str | None = None,
    map_width_px: float | None = None,
) -> go.Figure:
    """
    GFS field as raster image layer on CartoDB Positron, with colored
    station markers overlaid. Station markers share the field's colorscale
    so hot stations look red and cool stations look blue - same as the field.

    For var_key == "risk", the field/markers use the discrete NWS heat-index
    risk categories instead, with a categorical legend in place of a colorbar.
    """
    is_risk = var_key == "risk"
    categories = _risk_categories(unit) if is_risk else None
    vm = VARIABLE_META[var_key] if not is_risk else None
    vmin, vmax = (_convert(vm["vmin"], unit), _convert(vm["vmax"], unit)) if vm else (None, None)

    fig = go.Figure()

    if is_risk:
        # One legend-only ghost trace per category (mapbox has no discrete colorbar).
        # "No Elevated Risk" is listed first - it's a real category (below Caution),
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

    image_layer, halo_trace, marker_trace = _frame_traces(
        data, lats, lons, var_key, unit, station_values, selected_station,
        categories=categories, vmin=vmin, vmax=vmax,
    )
    fig.add_trace(halo_trace)
    fig.add_trace(marker_trace)

    effective_width_px = map_width_px if map_width_px else PAGE_MAX_WIDTH - 48
    # Tracks the same breakpoint mobile.css uses for #field-map's own
    # height, so the zoom fit is computed against the actual rendered
    # aspect ratio rather than assuming the desktop frame's shape.
    is_narrow = effective_width_px <= MOBILE_WIDTH_BREAKPOINT
    effective_height_px = MOBILE_MAP_HEIGHT if is_narrow else MAP_HEIGHT
    # A narrow/near-square mobile frame is a much worse aspect-ratio match
    # for CONUS's wide landscape shape than desktop's frame is, so fitting
    # by width (the binding dimension either way) reveals a lot more N/S
    # margin than CONUS needs - split evenly, that meant showing as much
    # of Mexico/Central America south as ocean/Canada north, which is what
    # actually made the map look broken on phones. north_bias pushes most
    # of that unavoidable extra margin northward instead.
    zoom, center = _bbox_zoom_center(
        CONUS_BBOX, width_px=effective_width_px, height_px=effective_height_px,
        north_bias=0.85 if is_narrow else 0.0)

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
            layers=[image_layer],
            # Caps pan/zoom-out to roughly North America. Without this the
            # map has no minimum zoom, so zooming out wraps the world tiles
            # and shows the CONUS raster floating twice on a repeating map.
            bounds=dict(west=-145.0, east=-50.0, south=8.0, north=65.0),
        ),
        height=MAP_HEIGHT,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
    )
    return fig


def _get_station_values(var_key: str, time_idx: int) -> list[float] | None:
    """GFS values at each station's nearest grid point, for one time step.

    Parameters
    ----------
    var_key : str
        Data variable name in _GFS_DS, e.g. "t2m" or "hi".
    time_idx : int
        Index into _GFS_DS's time dimension. Clamped to the last valid
        index if out of range.

    Returns
    -------
    list of float or None
        None if _GFS_DS is not loaded or var_key is missing. Otherwise
        one value per station in MAJOR_CONUS_STATIONS order, degrees C,
        NaN where a station's lookup fails.
    """
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
    """Max Heat Index at each station's nearest grid point, over a set of time steps.

    Parameters
    ----------
    time_idxs : list of int
        Indices into _GFS_DS's time dimension to take the max over.

    Returns
    -------
    list of float or None
        None if _GFS_DS is not loaded or has no "hi" variable. Otherwise
        one value per station in MAJOR_CONUS_STATIONS order, degrees C,
        NaN where a station's lookup fails.
    """
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


def _get_station_daily_peaks(var_key: str = "hi") -> list[tuple]:
    """Peak value and time index per station, for each unique forecast day separately.

    Backs the day-by-day "Peak This Event" leaderboard. Deliberately not
    one global peak across the whole forecast window: a single flat
    ranking by each station's all-time peak mixed days that are not
    really comparable, a 112F Wednesday reading and a 108F Sunday
    reading are not competing for the same "worst" in any meaningful
    sense, they are two different moments in the heat wave's
    progression across the country as the ridge moves. Grouping by day
    makes that progression the organizing structure instead of a fact
    buried in a "when" column.

    Parameters
    ----------
    var_key : str, optional
        Data variable name in _GFS_DS. Default "hi".

    Returns
    -------
    list of tuple
        [(date, [(station, peak_val, peak_idx), ...]), ...], sorted by
        date, each day's station list sorted by peak_val descending.
        Empty list if _GFS_DS is not loaded or var_key is missing.
    """
    if _GFS_DS is None or var_key not in _GFS_DS:
        return []
    da = _GFS_DS[var_key]
    out = []
    for date_, _ in _unique_forecast_days():
        day_idxs = _day_time_indices(date_)
        if not day_idxs:
            continue
        day_da = da.isel(time=day_idxs)
        day_rows = []
        for s in MAJOR_CONUS_STATIONS:
            try:
                series = day_da.sel(latitude=s["lat"], longitude=s["lon"],
                                    method="nearest").values
                local_peak_pos = int(np.nanargmax(series))
                peak_val = float(series[local_peak_pos])
                peak_idx = day_idxs[local_peak_pos]   # map back to the global time index
            except Exception:
                continue
            if not math.isnan(peak_val):
                day_rows.append((s, round(peak_val, 1), peak_idx))
        day_rows.sort(key=lambda triple: triple[1], reverse=True)
        out.append((date_, day_rows))
    return out


def _peak_leaderboard_table(unit: str, top_n_per_day: int = 4,
                            selected_station: str | None = None) -> html.Div:
    """Each forecast day's hottest cities by peak forecasted Heat Index,
    broken out day by day - reads as the story of the heat wave moving
    across the country instead of one flat ranking that implies moments
    from different days are directly comparable."""
    if _GFS_DS is None:
        return html.Div()

    daily = _get_station_daily_peaks("hi")
    if not daily:
        return html.Div()

    unit_label = _unit_label(unit)
    day_sections = []
    for date_, day_rows in daily:
        top_rows = day_rows[:top_n_per_day]
        if not top_rows:
            continue
        row_divs = []
        for i, (s, hi_c, idx) in enumerate(top_rows, start=1):
            num_fmt = f"{_convert(hi_c, unit):.0f}" if unit == "F" else f"{_convert(hi_c, unit):.1f}"
            when = _to_et(_GFS_DS.time.values[idx]).strftime("%I:%M %p").replace(" 0", " ")
            row_divs.append(html.Div([
                html.Span(str(i), style={"width": "20px", "color": "#64748b"}),
                html.Button(
                    f"{s['name']} ({s['state']})",
                    id={"type": "leaderboard-station", "index": s["id"]},
                    n_clicks=0,
                    title="Click to view this station's forecast + observations",
                    style={"flex": "1", "textAlign": "left", "color": "#e2e8f0",
                           "background": "none", "border": "none", "padding": 0,
                           "font": "inherit", "cursor": "pointer",
                           "fontWeight": "700" if s["id"] == selected_station else "400",
                           "textDecoration": "underline", "textDecorationColor": "#334155"},
                ),
                html.Span(f"{num_fmt}{unit_label}",
                          style={"width": "70px", "textAlign": "right",
                                 "color": "#f8fafc", "fontWeight": "600"}),
                html.Span(when, style={"width": "90px", "textAlign": "right", "color": "#94a3b8"}),
            ], style={"display": "flex", "alignItems": "center", "padding": "5px 10px",
                      "fontSize": "12px", "borderBottom": "1px solid #1e293b"}))

        day_sections.append(html.Div([
            html.Div(date_.strftime("%A, %b %d"),
                     style={"fontSize": "12px", "fontWeight": "700", "color": "#fb923c",
                            "padding": "8px 10px 4px 10px"}),
            html.Div(row_divs),
        ]))

    return html.Div([
        html.H3("Hottest Cities - Peak This Event",
                style={"fontSize": "14px", "color": "#f8fafc", "margin": "0 0 4px 0"}),
        html.Div("Each day's hottest cities by peak forecasted Heat Index - "
                 "see how the heat moves across the country",
                 style={"fontSize": "11px", "color": "#64748b", "marginBottom": "4px"}),
        html.Div(day_sections),
    ], style={"backgroundColor": "#1e293b", "borderRadius": "8px", "padding": "12px 14px"})


def _risk_source_link() -> html.Span:
    """Citation for the risk categories, placed right next to them (not just
    buried in the page footer) so the source is obvious wherever they appear."""
    return html.Span([
        "Source: ",
        html.A("NWS heat index chart", href="https://www.weather.gov/safety/heat-index",
              target="_blank", style={"color": "#64748b", "textDecoration": "underline"}),
    ], style={"color": "#64748b", "whiteSpace": "nowrap"})


def _leaderboard_table(time_idx: int, unit: str, time_label: str = "",
                       top_n: int = 15, selected_station: str | None = None) -> html.Div:
    """Ranked table of the hottest stations for the currently-displayed day.

    Ranks by each station's own peak forecasted Heat Index for that day,
    not a single instant snapshot at time_idx. A single global timestamp
    is timezone-unfair: at any given moment, Eastern stations might be
    sitting near their local afternoon heat peak while Pacific/Mountain
    stations are still hours away from theirs, so the ranking would
    mostly reflect which time zone happens to be at its local peak right
    then, not genuine relative severity. Using each station's own peak
    across its full local calendar day (the same _day_time_indices and
    _get_station_risk_values aggregation already used for the risk map)
    sidesteps that: every station is evaluated at its own worst moment
    of that day, regardless of time zone. As a side effect this also
    makes the leaderboard far more stable while the animation plays, it
    only changes when the slider crosses into a new day, not on every
    intraday step.

    Parameters
    ----------
    time_idx : int
        Index into _GFS_DS's time dimension, used only to determine
        which Eastern-local calendar day is currently displayed.
    unit : str
        "F" or "C".
    time_label : str, optional
        Shown in the panel subtitle.
    top_n : int, optional
        How many stations to list. Default 15.

    Returns
    -------
    html.Div
        Empty if _GFS_DS is not loaded.
    """
    if _GFS_DS is None:
        return html.Div()

    time_idx = int(time_idx or 0)
    local_date = _to_et(_GFS_DS.time.values[time_idx]).date()
    vals = _get_station_risk_values(_day_time_indices(local_date))
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
            html.Button(
                f"{s['name']} ({s['state']})",
                id={"type": "leaderboard-station", "index": s["id"]},
                n_clicks=0,
                title="Click to view this station's forecast + observations",
                style={"flex": "1", "textAlign": "left", "color": "#e2e8f0",
                       "background": "none", "border": "none", "padding": 0,
                       "font": "inherit", "cursor": "pointer",
                       "fontWeight": "700" if s["id"] == selected_station else "400",
                       "textDecoration": "underline", "textDecorationColor": "#334155"},
            ),
            html.Span(f"{num_fmt}{unit_label}",
                      style={"width": "84px", "textAlign": "right",
                             "color": "#f8fafc", "fontWeight": "600"}),
            html.Span(html.Span(cat_label, title=RISK_DESCRIPTIONS.get(cat_label, ""), style={
                "backgroundColor": cat_color, "color": "#0f172a", "padding": "2px 8px",
                "borderRadius": "999px", "fontSize": "10px", "fontWeight": "700",
                "cursor": "help",
            }), style={"width": "130px", "textAlign": "right"}),
        ], style={"display": "flex", "alignItems": "center", "padding": "6px 10px",
                  "fontSize": "12px", "borderBottom": "1px solid #1e293b"}))

    subtitle = "Ranked by each city's own peak forecasted Heat Index this day"
    if time_label:
        subtitle += f"  ·  {time_label}"

    return html.Div([
        html.H3("Hottest Cities Today",
                style={"fontSize": "14px", "color": "#f8fafc", "margin": "0 0 4px 0"}),
        html.Div(subtitle,
                 style={"fontSize": "11px", "color": "#64748b", "marginBottom": "8px"}),
        html.Div(rows),
    ], style={"backgroundColor": "#1e293b", "borderRadius": "8px", "padding": "12px 14px"})


def _risk_legend() -> html.Div:
    """Standalone risk-category legend, placed right under the main map so
    the color coding is explained wherever it appears on screen. A row of
    cards (chip above its description, not side by side) rather than an
    inline-wrapping flow of text - the flow version wrapped mid-sentence
    at normal window widths, and a single-column stacked list looked too
    sparse spanning the map's full width."""
    legend_order = [(NO_RISK_LABEL, NO_RISK_COLOR)] + [(label, color) for _, label, color in RISK_CATEGORIES_F]
    legend_cards = [
        html.Div([
            html.Span(label, style={
                "backgroundColor": color, "color": "#0f172a", "padding": "3px 10px",
                "borderRadius": "999px", "fontSize": "11px", "fontWeight": "700",
                "display": "inline-block", "marginBottom": "6px",
            }),
            html.Div(RISK_DESCRIPTIONS.get(label, ""),
                     style={"color": "#94a3b8", "lineHeight": "1.4"}),
        ], style={"padding": "10px 12px", "backgroundColor": "#16202f",
                  "borderRadius": "6px"})
        for label, color in legend_order
    ]
    return html.Div([
        html.Div(legend_cards, style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
            "gap": "10px",
        }),
        html.Div(_risk_source_link(), style={"paddingTop": "8px"}),
    ], style={"fontSize": "11.5px", "margin": "10px 0 0 0", "padding": "14px 16px",
             "border": "1px solid #334155", "borderRadius": "8px",
             "backgroundColor": "#1e293b"})


# ── station panel figure ──────────────────────────────────────────────────────

DEFAULT_METRICS = ["hi", "t2m"]

# Color encodes the metric. Actual Temp gets both a past (observed) and
# future (forecast) segment, split at "now" so they never overlap. Feels
# Like only ever shows the forward-looking forecast - there's no need to
# compare it against a computed "observed feels like," which is a derived
# quantity most people don't intuitively reason about, and doubled the
# number of things on screen for no real benefit.
_METRIC_META = {
    "t2m":  {"label": "Actual Temp", "color": "#38bdf8", "width": 1.4, "dash": "dot"},
    "hi":   {"label": "Feels Like",  "color": "#fb923c", "width": 2.8, "dash": "solid"},
    "td2m": {"label": "Dewpoint",    "color": "#34d399", "width": 1.6, "dash": "solid"},
}


_BIAS_MIN_PAIRS = BIAS_MIN_PAIRS
_BIAS_MATCH_TOLERANCE = BIAS_MATCH_TOLERANCE
_today_forecast_bias = today_forecast_bias
_brier_score_exceedance = brier_score_exceedance

# How far past "now" the 95% PI band is drawn - a same-day bias/sigma
# estimated from a few hours of today's observations does not have
# anything meaningful to say about forecast error 5 days out, in a
# different weather regime entirely. Confirmed live: drawing it flat
# across the whole forecast horizon made an honestly-tight same-day
# estimate look alarmingly wide, since the same band gets visually
# stacked across many days at once.
_PI_MAX_LEAD_HOURS = 24

# Same-day bias trend: how the computed correction has moved *within
# today* as more obs arrive, not a multi-day history (that needs the
# forecast archive - separate, larger project). In-process only, so it
# resets on redeploy; that's fine, it's meant to answer "is the model
# getting worse or better as today goes on," not to be a durable record.
_bias_trend_log: dict[tuple[str, str], list[tuple[pd.Timestamp, float]]] = {}
_BIAS_TREND_MAX_POINTS = 20


def _log_bias_trend(station_id: str, metric: str, as_of: pd.Timestamp, bias_c: float) -> None:
    """Append one same-day bias reading to the in-process trend log.

    Deduplicates by observation timestamp and resets automatically at
    local midnight, so the log always reflects only today's readings
    for this station/metric.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code.
    metric : str
        "hi", "t2m", or "td2m".
    as_of : pd.Timestamp
        Timestamp of the observation this bias reading is based on.
    bias_c : float
        Bias value to log, degrees C.
    """
    key = (station_id, metric)
    log = _bias_trend_log.setdefault(key, [])
    if log and log[-1][0] == as_of:
        return  # no new obs since the last log entry - do not record a duplicate point
    if log and log[-1][0].date() != as_of.date():
        log.clear()  # new calendar day - yesterday's trend is not relevant to today's
    log.append((as_of, bias_c))
    if len(log) > _BIAS_TREND_MAX_POINTS:
        del log[0]


def _bias_trend_summary(station_id: str, metric: str, unit: str) -> str:
    """Human-readable summary of how today's bias has moved so far.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code.
    metric : str
        "hi", "t2m", or "td2m".
    unit : str
        "F" or "C".

    Returns
    -------
    str
        Something like "+0.4F -> +1.2F" with a direction arrow appended,
        or an empty string if fewer than 2 points have been logged today.
    """
    log = _bias_trend_log.get((station_id, metric), [])
    if len(log) < 2:
        return ""
    first_disp = _convert_delta(log[0][1], unit)
    last_disp  = _convert_delta(log[-1][1], unit)
    unit_label = _unit_label(unit)
    if last_disp > first_disp + 0.3:
        arrow = "↑"
    elif last_disp < first_disp - 0.3:
        arrow = "↓"
    else:
        arrow = "→"
    label = _METRIC_META.get(metric, {}).get("label", metric)
    return f"{label}: {first_disp:+.1f}{unit_label} → {last_disp:+.1f}{unit_label} {arrow}"


def _station_verification(station_id: str, asos_df: pd.DataFrame) -> dict | None:
    """Same-day forecast-skill stats for one station.

    N, Bias, and RMSE for Feels Like and Actual Temp, plus a Brier score
    for Feels Like against the Extreme Caution threshold, the only
    metric with a real NWS risk threshold to score exceedance against.
    Grows through the day as real observations arrive, same as the bias
    correction itself.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code.
    asos_df : pd.DataFrame
        Output of fetch_station_obs for this station.

    Returns
    -------
    dict or None
        None if there is not enough same-day data yet for either
        metric. Otherwise a dict keyed by "hi" and/or "t2m", each value
        the dict today_forecast_bias returns (plus "brier" for "hi").
    """
    stn = get_station(station_id)
    if stn is None or _GFS_DS is None or asos_df.empty:
        return None
    tz = ZoneInfo(stn.get("tz", "America/New_York"))
    sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")

    obs_local = asos_df.dropna(subset=["temp_c"]).copy()
    obs_local["valid_local"] = obs_local["valid_utc"].dt.tz_convert(tz)
    if obs_local.empty:
        return None
    has_td = obs_local["dewpoint_c"].notna()
    obs_local["hi_c"] = float("nan")
    if has_td.any():
        obs_local.loc[has_td, "hi_c"] = heat_index_array(
            obs_local.loc[has_td, "temp_c"].values,
            obs_local.loc[has_td, "dewpoint_c"].values,
        )
    now_ts = obs_local["valid_local"].max()

    stats = {}
    for key, obs_col in (("hi", "hi_c"), ("t2m", "temp_c")):
        series = _GFS_DS[key].sel(**sel).to_series()
        idx = pd.DatetimeIndex(series.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        series.index = idx.tz_convert(tz)
        result = _today_forecast_bias(series, obs_local, obs_col, now_ts.date(), now_ts)
        if result is None:
            continue
        row = dict(result)
        if key == "hi":
            brier = _brier_score_exceedance(series, obs_local, obs_col,
                                            now_ts.date(), now_ts, _EXTREME_CAUTION_HI_C)
            row["brier"] = brier["brier"] if brier else None
        stats[key] = row
    return stats or None


def _verification_box(station_id: str, unit: str, stats: dict) -> html.Details:
    """Compact same-day forecast-skill summary, in the spirit of a
    verification table but scoped to what a single day of one GFS run
    actually supports - no lead-time buckets (that needs a multi-run
    archive, a separate/larger project), just today's running N/Bias/RMSE/
    Brier so far.

    Collapsed by default, like the GEV popup - this is methodology detail
    for someone auditing the forecast, not something a casual reader needs
    to see just to check the weather. Keeping it opt-in avoids stacking
    every analytical block in front of the actual temperature chart.
    """
    unit_label = _unit_label(unit)
    rows = []
    for key, label in (("hi", "Feels Like"), ("t2m", "Actual Temp")):
        row = stats.get(key)
        if row is None:
            continue
        bias_disp = _convert_delta(row["bias"], unit)
        rmse_disp = _convert_delta(row["rmse"], unit)
        parts = [f"N={row['n_used']}", f"Bias: {bias_disp:+.1f}{unit_label}",
                 f"RMSE: {rmse_disp:.1f}{unit_label}"]
        if row.get("brier") is not None:
            parts.append(f"Brier: {row['brier']:.2f}")
        rows.append(html.Div([
            html.Span(f"{label}  ", style={"fontWeight": "700", "color": "#e2e8f0"}),
            html.Span("  ·  ".join(parts), style={"color": "#94a3b8"}),
        ], style={"fontSize": "12px", "padding": "2px 0"}))

    if not rows:
        return html.Details()

    return html.Details([
        html.Summary("Today's Forecast Skill",
                     style={"fontSize": "12px", "color": "#94a3b8", "cursor": "pointer"}),
        html.Div([
            *rows,
            html.Div("Same-day, in-sample - not an independent skill score. Grows as today's "
                     "real observations arrive. Brier scores Feels Like against the Extreme "
                     "Caution threshold.",
                     style={"fontSize": "10px", "color": "#64748b", "marginTop": "4px"}),
        ], style={"padding": "8px 4px 2px 4px"}),
    ], style={"backgroundColor": "#1e293b", "borderRadius": "8px", "padding": "10px 12px",
             "marginBottom": "8px", "border": "1px solid #334155"})


def _build_station_figure(station_id: str, asos_df: pd.DataFrame,
                          time_idx: int, unit: str = "C",
                          metrics: list[str] | None = None,
                          bias_window_hours: float | None = None,
                          show_raw: bool = False) -> go.Figure:
    """GFS forecast and ASOS observations for one station.

    Plotted in the station's own local time zone and chosen display
    unit. One "best estimate" line per metric, bias-corrected when
    enough same-day observations exist, the raw model output otherwise,
    never both lines at once (see show_raw).

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code.
    asos_df : pd.DataFrame
        Output of fetch_station_obs for this station. Empty is
        acceptable, the figure just omits the observed points and bias
        correction.
    time_idx : int
        Index into _GFS_DS's time dimension, used only to place the
        "viewing forecast for" cursor annotation.
    unit : str, optional
        "F" or "C". Default "C".
    metrics : list of str or None, optional
        Which of "hi", "t2m", "td2m" to plot. Default DEFAULT_METRICS.
    bias_window_hours : float or None, optional
        How far back the bias correction looks, forwarded to
        today_forecast_bias. None uses all of today (the interactive
        same-day window control).
    show_raw : bool, optional
        If True, plot the unmodified model forecast instead of the
        bias-corrected line. Full-transparency toggle, still exactly
        one line at a time.

    Returns
    -------
    tuple of (go.Figure, int or None)
        The figure, and n_available, the minimum same-day pair count
        across the plotted metrics (None if no metric could be
        bias-corrected at all). n_available is what the bias-window
        control uses to decide whether it has anything meaningful to
        offer yet.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS
    stn = get_station(station_id)
    if stn is None:
        return _station_placeholder(f"Station {station_id} not in catalog."), None
    if _GFS_DS is None:
        return _station_placeholder("GFS data not loaded."), None

    tz = ZoneInfo(stn.get("tz", "America/New_York"))
    unit_label = _unit_label(unit)

    # Extract GFS time series at station's nearest grid point
    sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")
    gfs_series = {
        "t2m":  _GFS_DS["t2m"].sel(**sel).to_series(),
        "hi":   _GFS_DS["hi"].sel(**sel).to_series(),
        "td2m": _GFS_DS["td2m"].sel(**sel).to_series(),
    }

    def _to_local(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(idx)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return idx.tz_convert(tz)

    for s in gfs_series.values():
        s.index = _to_local(s.index)

    # Animation cursor
    times     = _GFS_DS.time.values
    cursor_ts = _to_local(pd.DatetimeIndex([times[min(int(time_idx or 0), len(times) - 1)]]))[0]
    tz_abbr   = cursor_ts.strftime("%Z")

    # ASOS obs run up to whenever they were fetched (real "now"), which is
    # usually hours after the GFS init time. "Now" is the natural dividing
    # line between what actually happened (observed) and what's predicted
    # (forecast) - forecast lines only ever show the future side of it.
    obs_local = pd.DataFrame()
    now_ts = None
    if not asos_df.empty:
        obs_local = asos_df.dropna(subset=["temp_c"]).copy()
        obs_local["valid_local"] = obs_local["valid_utc"].dt.tz_convert(tz)
        if not obs_local.empty:
            now_ts = obs_local["valid_local"].max()
            # Real observed Feels Like isn't in the raw ASOS feed - compute
            # it from actual observed temp+dewpoint with the same NWS
            # formula used for the forecast. This is a validated fact about
            # what already happened, not a model output.
            has_td = obs_local["dewpoint_c"].notna()
            obs_local["hi_c"] = float("nan")
            if has_td.any():
                obs_local.loc[has_td, "hi_c"] = heat_index_array(
                    obs_local.loc[has_td, "temp_c"].values,
                    obs_local.loc[has_td, "dewpoint_c"].values,
                )

    fig = go.Figure()
    n_available = None   # min across plotted metrics with enough same-day pairs to correct at all

    for key in ("hi", "t2m", "td2m"):
        if key not in metrics:
            continue
        meta = _METRIC_META[key]
        line_series = gfs_series[key]
        if now_ts is not None:
            line_series = line_series[line_series.index > now_ts]

        obs_col = {"t2m": "temp_c", "hi": "hi_c", "td2m": "dewpoint_c"}[key]

        # Single "best estimate" forecast line, always present - never a
        # separate raw line plus a sometimes-there corrected line, which
        # duplicated the same curve when uncorrected and was easy to miss
        # when it wasn't. Bias-corrected when enough same-day paired obs
        # exist to trust a correction (_BIAS_MIN_PAIRS); otherwise this
        # *is* the raw forecast, unchanged - graceful fallback, not a
        # gap. Labeled honestly either way so "corrected" never appears
        # unless a correction was actually applied.
        bias_result = None
        if now_ts is not None and not line_series.empty:
            bias_result = _today_forecast_bias(gfs_series[key], obs_local, obs_col,
                                               now_ts.date(), now_ts, window_hours=bias_window_hours)
            # Trend logging always uses the full-day bias (window_hours=None),
            # independent of whatever window is currently selected for
            # display - otherwise switching windows mid-day would mix
            # differently-computed values into one trend series and the
            # "is it getting better or worse" read would be meaningless.
            trend_result = bias_result if bias_window_hours is None else _today_forecast_bias(
                gfs_series[key], obs_local, obs_col, now_ts.date(), now_ts, window_hours=None)
            if trend_result is not None:
                _log_bias_trend(station_id, key, now_ts, trend_result["bias"])

        # n_available tracks regardless of show_raw, so the bias-window
        # dropdown stays populated/visible even while viewing raw - the
        # toggle only changes which line is drawn, not whether the
        # correction machinery is running underneath.
        if bias_result is not None:
            n_available = (bias_result["n_available"] if n_available is None
                           else min(n_available, bias_result["n_available"]))

        if show_raw:
            best_series = line_series
            pi95_c = None
            trace_name = f"{meta['label']} (Forecast, raw)"
            hover = (f"{meta['label']} (Forecast, raw - not bias-corrected): "
                     f"%{{y:.1f}}{unit_label}  %{{x|%b %d %I:%M %p}}<extra></extra>")
        elif bias_result is not None:
            bias_c  = bias_result["bias"]
            pi95_c  = bias_result["pi95_halfwidth"]
            n_used  = bias_result["n_used"]
            best_series = line_series + bias_c
            bias_disp = _convert_delta(bias_c, unit)
            trace_name = f"{meta['label']} (Bias-Corrected)"
            hover = (f"{meta['label']} (Bias-Corrected, {bias_disp:+.1f}{unit_label} "
                     f"vs {n_used} obs today): %{{y:.1f}}{unit_label}  "
                     f"%{{x|%b %d %I:%M %p}}<extra></extra>")
        else:
            best_series = line_series
            pi95_c = None
            trace_name = f"{meta['label']} (Forecast)"
            hover = (f"{meta['label']} (Forecast, not enough obs yet to bias-correct): "
                     f"%{{y:.1f}}{unit_label}  %{{x|%b %d %I:%M %p}}<extra></extra>")

        # 95% predictive interval around the corrected line - "error
        # dressing" of a deterministic forecast (Roulston & Smith 2003),
        # not NGR/EMOS (Gneiting et al. 2005), which requires an ensemble
        # and CRPS-trained regression coefficients this app has neither
        # of. The residuals of the same n same-day points used for the
        # bias correction are assumed roughly Normal; the half-width
        # comes from src/heat/bias.py's pi95_halfwidth, which uses a
        # t-distribution (not a fixed 1.96) since sigma itself is
        # estimated from as few as BIAS_MIN_PAIRS=3 points, and widens
        # further for uncertainty in the estimated mean - both matter
        # most exactly when n is smallest. Shown as an explicit visual
        # band (so its width is honest about the uncertainty) rather than
        # baked silently into the line. Labeled "95% PI" (prediction
        # interval), not "95% CI" - a CI is uncertainty about the
        # estimated mean, a PI is where a new observation is expected to
        # fall, which is what this band is actually claiming, and a PI
        # is necessarily wider than the corresponding CI.
        #
        # Two further restrictions beyond "enough same-day points exist"
        # (PI_MIN_PAIRS, in bias.py):
        #
        # 1. Skipped entirely for Feels Like ("hi"). Heat Index is a
        #    nonlinear, piecewise function of T and Td (see
        #    src/heat/compute.py: an escalation threshold plus separate
        #    low/high-humidity adjustment branches), so its same-day
        #    residuals are less likely to actually be the roughly Normal,
        #    symmetric errors this interval assumes. Temperature and
        #    Dewpoint are the more fundamental, directly-observed,
        #    better-behaved quantities to claim a prediction interval on.
        # 2. Truncated to _PI_MAX_LEAD_HOURS past "now" for T/Td, not
        #    drawn flat across the whole multi-day forecast horizon - a
        #    same-day estimate does not have anything meaningful to say
        #    about forecast error days out, in what could be a completely
        #    different weather regime.
        if pi95_c and key != "hi" and now_ts is not None:
            band_series = best_series[best_series.index <= now_ts + pd.Timedelta(hours=_PI_MAX_LEAD_HOURS)]
            if not band_series.empty:
                upper = _convert_array((band_series + pi95_c).values, unit)
                lower = _convert_array((band_series - pi95_c).values, unit)
                fig.add_trace(go.Scatter(
                    x=list(band_series.index) + list(band_series.index[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill="toself", fillcolor=_hex_to_rgba(meta["color"], 0.12),
                    line=dict(width=0), hoverinfo="skip", showlegend=True,
                    name=f"{meta['label']} (95% PI)",
                ))

        fig.add_trace(go.Scatter(
            x=best_series.index, y=_convert_array(best_series.values, unit),
            mode="lines",
            line=dict(color=meta["color"], width=meta["width"], dash=meta["dash"]),
            name=trace_name,
            hovertemplate=hover,
        ))

        if not obs_local.empty:
            obs_points = obs_local.dropna(subset=[obs_col])
            if not obs_points.empty:
                fig.add_trace(go.Scatter(
                    x=obs_points["valid_local"], y=_convert_array(obs_points[obs_col].values, unit),
                    mode="markers",
                    marker=dict(color=meta["color"], size=5, opacity=0.85),
                    name=f"{meta['label']} (Observed)",
                    hovertemplate=f"{meta['label']} (Observed): %{{y:.1f}}{unit_label}  "
                                  f"%{{x|%b %d %I:%M %p}}<extra></extra>",
                ))

    # "Now" cursor - the real observed/forecast boundary, not the day/time
    # picked in the Day/Time dropdowns (that's shown in the title instead).
    # This is the one universally meaningful reference point on the chart:
    # everything left of it happened, everything right is predicted.
    if now_ts is not None:
        fig.add_shape(
            type="line",
            x0=now_ts.isoformat(), x1=now_ts.isoformat(),
            y0=0, y1=1, yref="paper",
            line=dict(color="rgba(226,232,240,0.85)", width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=now_ts.isoformat(), y=1, yref="paper",
            text="Now", showarrow=False,
            font=dict(size=9, color="#e2e8f0"),
            xanchor="left", yanchor="bottom",
        )

    # Reference line: NWS "Extreme Caution" threshold (32°C/90°F HI) - the
    # boundary on the official heat index chart, not a fixed national
    # "Excessive Heat Warning" trigger (those are set regionally by local
    # NWS offices and are typically much higher, 100-115°F+).
    # Only makes sense alongside the Heat Index metric, so it follows that toggle.
    # Spans the full visible range - including the ASOS obs history, not just
    # the forecast portion from "now" onward - and is bold enough to read as
    # a hard line, not a faint gridline.
    x0_dt = gfs_series["t2m"].index[0]
    x1_dt = gfs_series["t2m"].index[-1]
    if not obs_local.empty:
        x0_dt = min(x0_dt, obs_local["valid_local"].min())
    x0, x1 = x0_dt.isoformat(), x1_dt.isoformat()
    for thresh_c, desc, color, requires in [
        (_EXTREME_CAUTION_HI_C, "(Extreme Caution begins)", "rgba(241,245,249,0.75)", "hi"),
    ]:
        if requires not in metrics:
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

    selected_str = cursor_ts.strftime("%a %b %d, %I:%M %p").replace(" 0", " ")
    fig.update_layout(
        paper_bgcolor=_PANEL_BG, plot_bgcolor=_PANEL_BG,
        title=dict(
            text=f"{station_id} - {stn['name']} ({stn['state']})  ·  times in {tz_abbr}"
                 f"  ·  viewing forecast for {selected_str}",
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
    return fig, n_available


_HERO_OBS_FRESHNESS = pd.Timedelta(hours=2)


def _hero_card(number_text: str, sub_label: str, border_color: str,
               chip: tuple[str, str] | None, chip_desc: str,
               extra_lines: list) -> html.Div:
    """The headline-number card (Feels Like) at the top of the station
    panel - its own shell function mainly so the number/chip/extra-lines
    layout stays consistent if another headline card is ever added."""
    children = [
        html.Div([
            html.Span(number_text,
                      style={"fontSize": "34px", "fontWeight": "700", "color": "#f8fafc"}),
            html.Span(f"  {sub_label}",
                      style={"fontSize": "13px", "color": "#94a3b8", "marginLeft": "6px"}),
        ]),
    ]
    if chip is not None:
        chip_label, chip_color = chip
        children.append(html.Div([
            html.Span(chip_label, style={
                "backgroundColor": chip_color, "color": "#0f172a", "padding": "3px 10px",
                "borderRadius": "999px", "fontSize": "11px", "fontWeight": "700",
            }),
            html.Span(chip_desc, style={"fontSize": "12px", "color": "#94a3b8"}),
        ], style={"marginTop": "8px", "display": "flex", "alignItems": "center",
                  "flexWrap": "wrap", "gap": "8px"}))
    children.extend(extra_lines)
    return html.Div(children, style={
        "padding": "12px 18px", "backgroundColor": "#1e293b", "borderRadius": "8px",
        "border": f"1px solid {border_color}55",
    })


def _hero_tile(station_id: str, time_idx: int, unit: str, asos_df: pd.DataFrame) -> html.Div:
    """Single "Feels Like" headline card, ahead of the line charts.

    A separate "Temperature" card was tried and dropped. Feels Like and
    raw temperature usually sit within a degree or two of each other,
    so two similarly sized headline numbers read as redundant rather
    than informative. Raw temperature is not lost, it is still in the
    Temperature & Dewpoint chart below and in the GEV popup's callout.

    Always the forecast for the selected day/time, matching the
    leaderboard and map, see module Gotcha 5.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code.
    time_idx : int
        Index into _GFS_DS's time dimension.
    unit : str
        "F" or "C".
    asos_df : pd.DataFrame
        Output of fetch_station_obs for this station, used only for
        the optional "actual right now" secondary line.

    Returns
    -------
    html.Div
        Empty if the station is unknown or _GFS_DS is not loaded.
    """
    stn = get_station(station_id)
    if stn is None or _GFS_DS is None:
        return html.Div()

    tz  = ZoneInfo(stn.get("tz", "America/New_York"))
    idx = min(int(time_idx or 0), len(_GFS_DS.time) - 1)
    sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")
    hi_c = float(_GFS_DS["hi"].sel(**sel).isel(time=idx).values)
    np_time  = _GFS_DS.time.values[idx]
    ts_et    = _to_et(np_time)
    ts_local = pd.Timestamp(np_time).tz_localize("UTC").tz_convert(tz)
    # Eastern Time leads (matches the map header, which is always ET) so
    # the two never look desynced side by side - confirmed live they can
    # otherwise appear to disagree even when correct (Yuma, AZ doesn't
    # observe DST, so the map's "04:00 AM EDT" and the station's own
    # "1:00 AM" are the same instant, but nothing said so). Station-local
    # time follows in parentheses only when it actually differs from ET -
    # a DC-area station showing "(1:00 AM local)" next to its own already-
    # Eastern time would be redundant noise, not a second data point.
    et_str = ts_et.strftime("%a %I:%M %p %Z").replace(" 0", " ")
    if tz == _DISPLAY_TZ:
        time_label = f"{et_str} (forecast)"
    else:
        local_str = ts_local.strftime("%I:%M %p %Z").replace(" 0", " ")
        time_label = f"{et_str} ({local_str} local) (forecast)"

    unit_label = _unit_label(unit)
    hi_fmt = f"{_convert(hi_c, unit):.0f}" if unit == "F" else f"{_convert(hi_c, unit):.1f}"

    cat_label, cat_color = NO_RISK_LABEL, NO_RISK_COLOR
    for lower, label, color in reversed(RISK_CATEGORIES_C):
        if hi_c >= lower:
            cat_label, cat_color = label, color
            break
    desc = RISK_DESCRIPTIONS.get(cat_label, "")

    # Plain text, not a badge - "Comfortable" is the common case (most
    # dewpoints most of the time), and a badge that fires almost always
    # reads as noise, not signal. Only Muggy/Oppressive earn a colored
    # inline call-out.
    extra_lines = []
    if "td2m" in _GFS_DS:
        td_c = float(_GFS_DS["td2m"].sel(**sel).isel(time=idx).values)
        dp_label, dp_color = _dewpoint_band(td_c)
        dp_fmt = f"{_convert(td_c, unit):.0f}" if unit == "F" else f"{_convert(td_c, unit):.1f}"
        dp_spans = [html.Span(f"Dewpoint: {dp_fmt}{unit_label}", style={"color": "#cbd5e1"})]
        if dp_label != "Comfortable":
            dp_spans.append(html.Span(f"  {dp_label}", style={
                "color": dp_color, "marginLeft": "4px", "fontWeight": "600"}))
        extra_lines.append(html.Div(dp_spans, style={"fontSize": "12px", "marginTop": "6px"}))

    # A fresh real observation gets its own secondary line - never
    # replacing the forecast headline itself.
    if not asos_df.empty:
        obs = asos_df.dropna(subset=["temp_c", "dewpoint_c"])
        if not obs.empty:
            latest = obs.loc[obs["valid_utc"].idxmax()]
            if pd.Timestamp.now(tz="UTC") - latest["valid_utc"] <= _HERO_OBS_FRESHNESS:
                obs_hi_c = float(heat_index_array([latest["temp_c"]], [latest["dewpoint_c"]])[0])
                obs_hi_fmt = (f"{_convert(obs_hi_c, unit):.0f}" if unit == "F"
                              else f"{_convert(obs_hi_c, unit):.1f}")
                extra_lines.append(html.Div(f"Actual right now: {obs_hi_fmt}{unit_label}",
                                            style={"fontSize": "11px", "color": "#64748b",
                                                  "marginTop": "6px"}))

    feels_like_card = _hero_card(
        f"{hi_fmt}{unit_label}", "Feels like", cat_color,
        (cat_label, cat_color), desc, extra_lines)

    return html.Div([
        html.Div(time_label, style={"fontSize": "14px", "fontWeight": "700",
                                    "color": "#22d3ee", "marginBottom": "6px"}),
        html.Div(feels_like_card, style={"maxWidth": "420px"}),
    ], style={"marginBottom": "10px"})


def _climate_context(station_id: str, unit: str) -> html.Div:
    """
    How the day's actual high compares to the 1991-2020 NWS normal for
    this station, from a bulk-fetched NWS Daily Climate Report.

    Not every station has a CLI product (issued per NWS office policy),
    and coverage fluctuates day to day - a missing entry gets an explicit
    "unavailable" note rather than silently showing nothing, so it reads
    as "checked, no data" instead of looking broken.

    The report date is shown explicitly (not just "Today's"/"Yesterday's")
    because bulk-fetched reports don't all refresh in lockstep - some
    stations' most recent available report can be several days old if
    their office hasn't issued a new one, and a relative label would
    silently imply it's current when it isn't.
    """
    info = _CLIMATE_NORMALS.get(station_id)
    if info is None:
        return html.Div(
            "Climate comparison unavailable for this station "
            "(no NWS daily climate report issued for it).",
            style={"fontSize": "12px", "color": "#64748b", "marginBottom": "10px"},
        )

    unit_label   = _unit_label(unit)
    actual_disp  = _convert((info["high_actual_f"] - 32) * 5 / 9, unit)
    normal_disp  = _convert((info["high_normal_f"] - 32) * 5 / 9, unit)
    departure_f  = info["high_departure_f"]
    num_fmt = (lambda v: f"{v:.0f}") if unit == "F" else (lambda v: f"{v:.1f}")
    departure_disp = abs(departure_f) * (5 / 9 if unit == "C" else 1)

    if departure_f > 0:
        direction, color = "above", "#f2994a"
    elif departure_f < 0:
        direction, color = "below", "#38bdf8"
    else:
        direction, color = "at", "#94a3b8"

    report_date = pd.Timestamp(info["date"]).strftime("%b %d").replace(" 0", " ")
    in_progress = " so far" if info["period_label"] == "Today" else ""
    period = f"{report_date} actual-temp high{in_progress}"

    # Explicitly labeled "actual-temp," not just "high" - this comes from
    # the NWS CLI report, which only ever reports raw temperature, never
    # Heat Index. Sitting right under the "Feels Like" hero number without
    # that label, this read as directly contradicting it (e.g. "91F Feels
    # Like" above "89F, below normal" right below), when the two numbers
    # are actually different physical quantities, not a contradiction.
    text = (f"{period}: {num_fmt(actual_disp)}{unit_label} - "
            f"{num_fmt(departure_disp)}{unit_label} {direction} the 1991-2020 normal "
            f"({num_fmt(normal_disp)}{unit_label})")

    return html.Div(
        text,
        style={"fontSize": "12px", "color": color, "marginBottom": "10px"},
    )


def _gev_distribution_figure(fit: dict, annual_maxima_f: np.ndarray | None,
                             current_temp_c: float | None, unit: str) -> go.Figure:
    """The classic EVT "return level plot" for one station's GEV fit.

    Temperature (return level) on the y-axis against return period on
    a log-scaled x-axis, giving the familiar upward-curving shape these
    are usually drawn as, e.g. NOAA Atlas 14, USGS flood-frequency
    reports. Return period is the axis itself here, not a value needing
    its own reference lines or annotations, which sidesteps the label
    crowding a probability-axis version of this plot needed custom
    collision handling for in an earlier version.

    Actual annual maxima are overlaid as points at their empirical
    return period (1 / Gringorten plotting position), the standard way
    to visually check whether the fitted curve tracks the real data's
    rank-based probabilities.

    Parameters
    ----------
    fit : dict
        Output of src.heat.extremes.fit_gev.
    annual_maxima_f : np.ndarray or None
        Raw annual maxima, degrees F, for the empirical overlay points.
        None or empty skips that trace.
    current_temp_c : float or None
        Currently-viewed forecast temperature, degrees C, marked with a
        vertical line if its return period falls within the plotted
        range.
    unit : str
        "F" or "C" for axis display.

    Returns
    -------
    go.Figure
    """
    unit_label = _unit_label(unit)
    lo_support, hi_support = support(fit)

    periods = np.logspace(np.log10(1.01), np.log10(500), 400)
    levels_f = np.array([return_level(fit, t) for t in periods])
    valid = (levels_f >= lo_support) & (levels_f <= hi_support)
    periods, levels_f = periods[valid], levels_f[valid]
    levels_disp = _convert_array((levels_f - 32.0) * 5.0 / 9.0, unit)

    fig = go.Figure()

    if annual_maxima_f is not None and len(annual_maxima_f) > 0:
        emp_values_f, emp_probs = plotting_positions(annual_maxima_f)
        emp_periods = 1.0 / emp_probs
        emp_disp = _convert_array((emp_values_f - 32.0) * 5.0 / 9.0, unit)
        fig.add_trace(go.Scatter(
            x=emp_periods, y=emp_disp, mode="markers",
            marker=dict(color="#38bdf8", size=6),
            name=f"{len(annual_maxima_f)} yrs of actual data",
            hovertemplate=f"~1-in-%{{x:.0f}}-yr  ·  %{{y:.0f}}{unit_label}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=periods, y=levels_disp, mode="lines",
        line=dict(color="#fb923c", width=2.5),
        name="Fitted GEV",
        hovertemplate=f"~1-in-%{{x:.0f}}-yr  ·  %{{y:.0f}}{unit_label}<extra></extra>",
    ))

    if current_temp_c is not None:
        current_f = current_temp_c * 9.0 / 5.0 + 32.0
        current_period = return_period(fit, current_f)
        if current_period is not None and periods.min() <= current_period <= periods.max():
            fig.add_shape(type="line", x0=current_period, x1=current_period,
                         y0=0, y1=1, yref="paper",
                         line=dict(color="#f8fafc", width=2, dash="dot"))

    fig.update_layout(
        paper_bgcolor=_PANEL_BG, plot_bgcolor=_PANEL_BG,
        xaxis=dict(title=dict(text="Return period (years)", font=dict(size=10)),
                  type="log", gridcolor=_PANEL_GRID, color=_PANEL_FONT, tickfont=dict(size=9)),
        yaxis=dict(title=dict(text=f"Annual max temperature ({unit_label})", font=dict(size=10)),
                  gridcolor=_PANEL_GRID, color=_PANEL_FONT, tickfont=dict(size=9)),
        legend=dict(orientation="h", font=dict(size=9, color=_PANEL_FONT),
                   bgcolor="rgba(0,0,0,0)", x=0, y=-0.3, xanchor="left", yanchor="top"),
        margin=dict(l=44, r=10, t=24, b=64),
        height=260,
    )
    return fig


def _gev_popup(station_id: str, unit: str, current_temp_c: float | None) -> html.Details:
    """How rare the currently-viewed forecast temperature is, historically.

    Collapsed by default, a real expander, not shown up front. Reads
    the station's stationary GEV fit against its historical annual
    maxima, shown as the actual fitted distribution rather than just a
    number. Stationary only: treats the historical record as one
    unchanging distribution. Under a warming climate that is a real
    simplification, a non-stationary fit would let the location
    parameter trend with year, said explicitly in the caveat text this
    renders rather than left implicit.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code.
    unit : str
        "F" or "C".
    current_temp_c : float or None
        Currently-viewed forecast temperature, degrees C, used for the
        "roughly a 1-in-N-year event" callout sentence.

    Returns
    -------
    html.Details
        Empty (invisible) if no GEV fit exists for this station, either
        the archive was not built, or the station's record is shorter
        than extremes.GEV_MIN_YEARS.
    """
    fit = _GEV_FITS.get(station_id)
    if fit is None:
        return html.Details()

    unit_label = _unit_label(unit)
    num_fmt = (lambda v: f"{v:.0f}") if unit == "F" else (lambda v: f"{v:.1f}")

    children = []
    if current_temp_c is not None:
        current_f = current_temp_c * 9.0 / 5.0 + 32.0
        period = return_period(fit, current_f)
        if period is not None:
            rarity = "an ordinary summer day here" if period < 1.5 \
                else f"roughly a 1-in-{period:.0f}-year event"
            current_disp = _convert(current_temp_c, unit)
            children.append(html.Div(
                f"The forecast you're viewing ({num_fmt(current_disp)}{unit_label}) is "
                f"{rarity} at this station, historically.",
                style={"fontSize": "12px", "color": "#f8fafc", "fontWeight": "600",
                      "marginBottom": "4px"},
            ))

    fig = _gev_distribution_figure(fit, _GEV_ANNUAL_MAXIMA.get(station_id),
                                   current_temp_c, unit)
    children.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))
    children.append(html.Div(
        f"Stationary GEV fit to {fit['n_years']} years of annual max temperature "
        f"(station record back to 1972 where available). Doesn't account for a warming "
        f"trend within that record, so long return periods likely understate how often "
        f"today's extremes will recur going forward.",
        style={"fontSize": "10px", "color": "#64748b", "marginTop": "4px", "lineHeight": "1.4"},
    ))

    return html.Details([
        html.Summary("How rare is this heat, historically? (GEV analysis)",
                     style={"fontSize": "12px", "color": "#94a3b8", "cursor": "pointer"}),
        html.Div(children, style={"padding": "8px 4px 2px 4px"}),
    ], style={"backgroundColor": "#1e293b", "borderRadius": "8px", "padding": "10px 12px",
             "marginBottom": "8px", "border": "1px solid #334155"})


# Warm-night bands - actual overnight low temperature, not heat index. The
# body needs nighttime cooling to recover from daytime heat stress; a
# night that doesn't drop enough is a well-documented driver of heat-
# related mortality independent of how hot the day itself was (e.g. the
# 1995 Chicago and 2003 European heat waves).
WARM_NIGHT_BANDS_F = [
    (75.0, "Warm Night",      "#f2994a"),
    (80.0, "Very Warm Night", "#e8592e"),
]
WARM_NIGHT_BANDS_C = [
    (round((f - 32.0) * 5.0 / 9.0, 1), label, color)
    for f, label, color in WARM_NIGHT_BANDS_F
]


def _warm_night_band(low_c: float) -> tuple[str | None, str | None]:
    """Plain-language warm-night band for an overnight low temperature.

    Parameters
    ----------
    low_c : float
        Overnight low temperature, degrees C.

    Returns
    -------
    tuple of (str or None, str or None)
        (label, color), both None below the lowest threshold.
    """
    label, color = None, None
    for lower, lbl, c in WARM_NIGHT_BANDS_C:
        if low_c >= lower:
            label, color = lbl, c
    return label, color


def _overnight_lows(station_id: str) -> list[dict]:
    """Per-day overnight low for a station across the whole forecast window.

    Computed in the station's own local time zone, not the app's
    canonical Eastern day boundaries the leaderboard uses, since warm
    nights are inherently a local-time concept.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code.

    Returns
    -------
    list of dict
        [{"date": date, "low_c": float}, ...], one entry per forecast
        day. Empty if the station is unknown or _GFS_DS is not loaded.
    """
    stn = get_station(station_id)
    if stn is None or _GFS_DS is None:
        return []
    tz = ZoneInfo(stn.get("tz", "America/New_York"))
    sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")
    t2m_series = _GFS_DS["t2m"].sel(**sel).to_series()
    idx_local = pd.DatetimeIndex(t2m_series.index)
    if idx_local.tz is None:
        idx_local = idx_local.tz_localize("UTC")
    idx_local = idx_local.tz_convert(tz)
    t2m_series.index = idx_local

    out = []
    for date_ in sorted(set(idx_local.date)):
        day_vals = t2m_series[idx_local.date == date_]
        if day_vals.empty:
            continue
        out.append({"date": date_, "low_c": float(day_vals.min())})
    return out


_RISK_CATEGORY_ORDER = [NO_RISK_LABEL] + [label for _, label, _ in RISK_CATEGORIES_F]


def _heat_index_category(hi_c: float) -> str:
    """NWS risk category label for a Heat Index value.

    Parameters
    ----------
    hi_c : float
        Heat Index, degrees C.

    Returns
    -------
    str
        One of RISK_CATEGORIES_F's labels, or NO_RISK_LABEL below the
        lowest threshold.
    """
    cat = NO_RISK_LABEL
    for lower, label, _ in RISK_CATEGORIES_C:
        if hi_c >= lower:
            cat = label
    return cat


def _heat_streak(station_id: str) -> dict | None:
    """Consecutive forecast days at or above today's risk category.

    Starts counting from the first forecast day and continues while the
    daytime peak Heat Index stays at or above that first day's own risk
    category. Forward-looking only, by construction: this app has no
    persisted history of past days, that would need a separate, larger
    archive project, so it cannot say "day N of an M-day streak"
    counting backward from days that already happened, only "the next
    M days stay this bad," which is the honest version of the same
    warning given what data actually exists right now.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code.

    Returns
    -------
    dict or None
        {"category": str, "streak_days": int}, or None if the station
        is unknown, _GFS_DS is not loaded, or today is not elevated
        risk at all.
    """
    stn = get_station(station_id)
    if stn is None or _GFS_DS is None:
        return None
    tz = ZoneInfo(stn.get("tz", "America/New_York"))
    sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")
    hi_series = _GFS_DS["hi"].sel(**sel).to_series()
    idx_local = pd.DatetimeIndex(hi_series.index)
    if idx_local.tz is None:
        idx_local = idx_local.tz_localize("UTC")
    idx_local = idx_local.tz_convert(tz)
    hi_series.index = idx_local

    daily_peaks = []
    for date_ in sorted(set(idx_local.date)):
        day_vals = hi_series[idx_local.date == date_]
        if day_vals.empty:
            continue
        daily_peaks.append((date_, float(day_vals.max())))
    if not daily_peaks:
        return None

    today_cat = _heat_index_category(daily_peaks[0][1])
    if today_cat == NO_RISK_LABEL:
        return None
    today_rank = _RISK_CATEGORY_ORDER.index(today_cat)

    streak_days = 0
    for _, peak in daily_peaks:
        if _RISK_CATEGORY_ORDER.index(_heat_index_category(peak)) >= today_rank:
            streak_days += 1
        else:
            break

    return {"category": today_cat, "streak_days": streak_days}


def _overnight_and_streak_panel(station_id: str) -> html.Div:
    """
    Cumulative-danger streak banner, based on consecutive days' *daytime*
    peak Heat Index only - it says nothing about overnight lows, so the
    copy must not imply it does. Originally paired with
    a day-by-day chip row of each night's low, but that duplicated what
    the chart right below it already shows visually (the nightly troughs,
    against the same 90F reference line) - dropped to cut panel clutter;
    the chart is where the actual per-night detail lives now. _overnight_
    lows()/_warm_night_band() stay defined (unused here) since a lighter-
    weight surface for that data - a tooltip, a popup - is a plausible
    near-term addition.
    """
    streak = _heat_streak(station_id)
    if streak is None or streak["streak_days"] < 2:
        return html.Div()

    return html.Div([
        html.Span(f"{streak['streak_days']} consecutive days ", style={
            "fontWeight": "700", "color": "#f2994a"}),
        html.Span(f"forecast at or above {streak['category']}", style={
            "fontWeight": "700", "color": "#f2994a"}),
        html.Span(" - back-to-back high-risk days compound heat stress on the body; danger is cumulative.",
                 style={"color": "#94a3b8"}),
    ], style={"fontSize": "12px", "padding": "10px 0 4px 0",
             "borderTop": "1px solid #334155", "marginTop": "4px"})


# ── Dash app ──────────────────────────────────────────────────────────────────

app    = Dash(__name__)
server = app.server   # Gunicorn entry point
app.title = "2026 US Heat Wave Tracker"

_SOCIAL_DESCRIPTION = (
    "Live GFS forecast - Heat Index, risk levels, and real ASOS observations "
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
        <meta property="og:title" content="2026 US Heat Wave Tracker">
        <meta property="og:description" content="{_SOCIAL_DESCRIPTION}">
        <meta property="og:type" content="website">
        <meta property="og:image" content="/assets/social_preview.png">
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="2026 US Heat Wave Tracker">
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
_PLAY_BTN_STYLE = {
    "backgroundColor": "#334155", "color": "#e2e8f0",
    "border": "1px solid #475569", "borderRadius": "6px",
    "padding": "6px 14px", "fontSize": "13px", "cursor": "pointer",
    "flexShrink": "0",
}


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
                html.Div([
                    html.H1("2026 US Heat Wave Tracker",
                        style={"color": "#f8fafc", "margin": 0, "display": "inline-block",
                               "fontSize": "20px", "fontWeight": "600", "verticalAlign": "middle"}),
                    html.Span("LIVE", title="Forecast data refreshes automatically every 6 hours",
                        style={"backgroundColor": "#dc2626", "color": "#fef2f2",
                               "fontSize": "10px", "fontWeight": "700", "letterSpacing": "0.5px",
                               "padding": "2px 7px", "borderRadius": "999px",
                               "marginLeft": "10px", "verticalAlign": "middle"}),
                ]),
                html.Span(f"{_init_label}  ·  Auto-updates every 6h",
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
                    html.Label("Time", style=_LABEL_STYLE),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "12px"},
                        children=[
                            html.Button("▶  Play", id="play-btn", n_clicks=0,
                                style=_PLAY_BTN_STYLE),
                            dcc.Slider(
                                id="time-slider",
                                min=0, max=_TIME_SLIDER_MAX, step=1, value=_DEFAULT_TIME_IDX,
                                marks=_TIME_SLIDER_MARKS,
                                tooltip={"placement": "bottom", "always_visible": False},
                                updatemode="mouseup",
                            ),
                        ],
                    ),
                ], style={"flex": "1", "minWidth": "260px"}),
            ],
        )),

        # ── CONUS map ─────────────────────────────────────────────────────────
        _page_section(None, None, html.Div(
            style={"padding": "16px 24px 0 24px"},
            children=[
                dcc.Graph(
                    id="field-map",
                    style={"height": f"{MAP_HEIGHT}px"},
                    config={"scrollZoom": True, "displayModeBar": False},
                ),
                html.Label(id="time-label",
                    style={"display": "block", "textAlign": "center",
                           "fontSize": "13px", "color": "#94a3b8",
                           "margin": "8px 0"}),
                html.Div(id="risk-legend-panel"),
            ],
        )),

        # ── station headline ──────────────────────────────────────────────────
        # "People first, scientist second": the headline number, risk
        # category, and today's context sit right under the map, ahead of
        # the leaderboard. Everything more technical (verification, GEV
        # analysis, bias controls, the charts themselves) lives in its own
        # section below the leaderboard instead, for whoever wants to dig
        # in - see "station deep-dive" further down.
        _page_section(None, None, html.Div(
            style={"padding": "0 24px 24px 24px"},
            children=[
                html.Div(
                    "Click a station on the map to see GFS forecast + ASOS observations",
                    id="station-hint",
                    style={"fontSize": "13px", "fontWeight": "600", "color": "#cbd5e1",
                          "paddingTop": "8px", "borderTop": "1px solid #334155",
                          "marginBottom": "8px"},
                ),
                html.Div(
                    "Heat Index = how hot it feels once humidity is factored in (NWS formula).",
                    style={"fontSize": "12px", "color": "#94a3b8", "marginBottom": "10px",
                          "maxWidth": "820px", "lineHeight": "1.5"}),
                html.Div(id="station-panel-headline"),
            ],
        )),

        # ── leaderboard ───────────────────────────────────────────────────────
        _page_section(None, None, html.Div(
            style={"padding": "0 24px 24px 24px"},
            children=[
                dcc.RadioItems(
                    id="leaderboard-mode",
                    options=[
                        {"label": " Today's Peak",     "value": "now"},
                        {"label": " Peak This Event",  "value": "peak"},
                    ],
                    value="now", inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "14px", "fontSize": "12px", "color": "#cbd5e1"},
                    style={"marginBottom": "8px"},
                ),
                html.Div(id="leaderboard-panel"),
            ],
        )),

        # ── station deep-dive ─────────────────────────────────────────────────
        # The "scientist" half of the station panel - the bias-correction
        # controls, the actual time series charts, and the GEV analysis
        # (an opt-in expander) last, after the charts rather than before
        # them - GEV is a step more abstract/scientific than a time series
        # chart, so it reads better as "one more thing to dig into" once
        # you've already seen the concrete forecast. Bias controls sit
        # immediately above the charts they affect (Show toggle, then
        # window dropdown below it) rather than off on their own; both
        # stay static, outside the dynamically-rebuilt containers, since
        # each is simultaneously an Input and an indirect Output of
        # update_station_panel - a component can't be both consumed and
        # structurally recreated by the same callback.
        _page_section(None, None, html.Div(
            style={"padding": "0 24px 24px 24px"},
            children=[
                html.Div(
                    style={"display": "flex", "flexDirection": "column", "gap": "10px",
                           "paddingTop": "10px", "borderTop": "1px solid #334155"},
                    children=[
                        html.Div(id="series-selector-wrap",
                                 style={"display": "none", "gap": "20px", "flexWrap": "wrap"},
                                 children=[
                            html.Div([
                                html.Span("Show  ", style={"fontSize": "11px", "color": "#64748b"}),
                                # Full transparency toggle: switches which
                                # single line is plotted, never both at once
                                # (that's the exact pattern collapsed away
                                # earlier for being unreadable/misleading).
                                # Default stays on the corrected view - this
                                # is an opt-in "show me the raw model output"
                                # control, not a return to showing both.
                                dcc.RadioItems(
                                    id="bias-display-mode",
                                    options=[
                                        {"label": " Bias-Corrected", "value": "corrected"},
                                        {"label": " Raw Forecast",   "value": "raw"},
                                    ],
                                    value="corrected", inline=True,
                                    inputStyle={"marginRight": "4px"},
                                    labelStyle={"marginRight": "14px", "fontSize": "12px",
                                                "color": "#cbd5e1"},
                                ),
                            ]),
                        ]),
                        html.Div(id="bias-window-wrap", style={"display": "none"},
                                children=[
                            html.Span(id="bias-window-label",
                                      style={"fontSize": "11px", "color": "#94a3b8",
                                             "display": "block", "marginBottom": "2px"}),
                            dcc.Dropdown(
                                id="bias-window-dropdown",
                                options=[
                                    {"label": "Last 3 hours",  "value": 3},
                                    {"label": "Last 6 hours",  "value": 6},
                                    {"label": "Last 12 hours", "value": 12},
                                    {"label": "All of today",  "value": "all"},
                                ],
                                value=6, clearable=False,
                                style={"width": "220px", "fontSize": "12px", "color": "#0f172a"},
                            ),
                        ]),
                    ],
                ),
                html.Div(id="station-panel-charts"),
                html.Div(id="station-panel-analysis"),
            ],
        )),

        # ── forecast verification ────────────────────────────────────────────
        # The deepest-in-the-weeds content (same-day N/Bias/RMSE/Brier) gets
        # the very last content section on the page, below even the charts -
        # audit-trail material for whoever scrolls all the way down, not
        # something competing for attention with anything above it.
        _page_section(None, None, html.Div(
            style={"padding": "0 24px 24px 24px"},
            children=[html.Div(id="station-panel-verification")],
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
        dcc.Store(id="selected-station", data="KDCA"),
        dcc.Store(id="current-time-idx"),
        dcc.Store(id="viewport-width", data=PAGE_MAX_WIDTH - 48),
        # 1800ms, not 800: real measured margin above the deployed Render
        # instance's own response time for a single frame (~415ms average,
        # more under jitter/load) - 800ms left too little room and the
        # slider visibly fell behind the map on any device hitting that
        # same server, PC included. Less snappy, but stays in sync.
        dcc.Interval(id="animation-interval", interval=1800, n_intervals=0, disabled=True),
    ],
)


# viewport-width defaults to the desktop-sized assumption above (so the
# server-rendered first paint looks right before any JS has run); this
# clientside callback then corrects it to the real browser width once on
# load, so _bbox_zoom_center fits CONUS to the map's actual width instead
# of assuming a wide desktop frame - without this, a phone in portrait
# gets a zoom level tuned for ~1350px, cropping the coasts off-screen.
app.clientside_callback(
    """
    function(_) {
        return Math.max(280, window.innerWidth - 48);
    }
    """,
    Output("viewport-width", "data"),
    Input("field-map", "id"),
)


# ── callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("play-btn", "style"),
    Input("viewport-width", "data"),
)
def toggle_play_button(viewport_width):
    """Animation is one server round trip per frame (see advance_frame) -
    fine on desktop, but real, measured round-trip time against the
    deployed server can't reliably keep pace with it on narrow/mobile
    connections, visibly falling behind rather than animating smoothly.
    Rather than ship a feature that's known to misbehave there, Play is
    hidden entirely on narrow screens - manual slider scrubbing still
    works everywhere, since it's a single request, not a repeating tick.
    """
    style = dict(_PLAY_BTN_STYLE)
    if (viewport_width or 0) <= MOBILE_WIDTH_BREAKPOINT:
        style["display"] = "none"
    return style


@app.callback(
    Output("series-selector-wrap", "style"),
    Input("selected-station", "data"),
)
def toggle_series_selector(station_id):
    shown = {"display": "flex", "gap": "20px", "flexWrap": "wrap"}
    return shown if station_id else {"display": "none"}


@app.callback(
    Output("current-time-idx", "data"),
    Input("time-slider", "value"),
)
def update_current_time_idx(time_idx):
    # Plain passthrough - kept as its own Store (rather than pointing every
    # downstream callback at time-slider.value directly) so update_map,
    # update_leaderboard_and_legend, and update_station_panel didn't all
    # need to change when the Day/Time dropdowns became a single slider.
    # For "risk" mode this is still just a timestamp index - _build_field_map
    # already derives "which day" from whatever index it's given, so the
    # slider needs no separate day-only mode the way the old hour-dropdown
    # did.
    return time_idx if time_idx is not None else 0


@app.callback(
    Output("animation-interval", "disabled"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State("animation-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_animation(_n_clicks, currently_disabled):
    if currently_disabled:
        return False, "⏸  Pause"
    return True, "▶  Play"


@app.callback(
    Output("time-slider", "value"),
    Input("animation-interval", "n_intervals"),
    State("time-slider", "value"),
    State("time-slider", "max"),
    prevent_initial_call=True,
)
def advance_frame(_n_intervals, current_value, max_value):
    current = int(current_value or 0)
    maximum = int(max_value or 0)
    return (current + 1) % (maximum + 1)


def _build_field_map(var_key, time_idx, unit, selected_station, map_width_px=None):
    """Figure-construction logic for the map, used by update_map.
    Returns (fig, time_label)."""
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
        return fig, "No data"

    lats = _GFS_DS.latitude.values
    lons = _GFS_DS.longitude.values

    if var_key == "risk":
        local_date = _to_et(_GFS_DS.time.values[time_idx]).date()
        # Running peak through the day, not the whole day's max regardless
        # of position - only hours up to and including the current frame
        # count. Previously this always maxed over every hour of the day
        # no matter where time_idx pointed, so the map looked frozen while
        # scrubbing or playing within a single day, only changing once the
        # slider crossed into the next calendar day. Deliberately does not
        # drop back down overnight the way classifying the instantaneous
        # Heat Index would: the day's peak danger stays shown as the
        # running reference for how much heat exposure this place has
        # already had today, since that exposure does not undo itself
        # once the sun goes down.
        idxs = [i for i in _day_time_indices(local_date) if i <= time_idx]
        data = _GFS_DS["hi"].isel(time=idxs).max(dim="time").values
        ts_str = _et_utc_label(_GFS_DS.time.values[time_idx])
        title = f"2026 US Heat Wave  ·  Risk Level (today's peak so far)  |  {ts_str}"
        stn_vals = _get_station_risk_values(idxs)
        fig = _mapbox_figure(
            data=data, lats=lats, lons=lons, var_key="risk", title=title,
            station_values=stn_vals, uirevision="conus_risk", unit=unit,
            selected_station=selected_station, map_width_px=map_width_px,
        )
        return fig, ts_str

    vm     = VARIABLE_META.get(var_key, VARIABLE_META["t2m"])
    da     = _GFS_DS[var_key].isel(time=time_idx)
    data   = da.values

    ts_str = _et_utc_label(_GFS_DS.time.values[time_idx])
    title  = f"2026 US Heat Wave  ·  {vm['label']}  |  {ts_str}"

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
        selected_station = selected_station,
        map_width_px   = map_width_px,
    )
    return fig, ts_str


@app.callback(
    Output("field-map",  "figure"),
    Output("time-label", "children"),
    Input("variable-selector", "value"),
    Input("current-time-idx",  "data"),
    Input("unit-selector",     "value"),
    Input("selected-station",  "data"),
    Input("viewport-width",    "data"),
)
def update_map(var_key, time_idx, unit, selected_station, viewport_width):
    """
    Single callback, single owner of field-map.figure, selected_station
    as a normal Input - no allow_duplicate split. Two different attempts
    at splitting the highlight into its own allow_duplicate=True callback
    both failed in the real browser (fired once for the initial render,
    then silently stopped updating on later clicks) despite working when
    invoked directly via the Dash HTTP API - that dual-owner pattern is
    unreliable here, so this reverts to the plain single-callback form.

    viewport_width has to be an Input, not a State: it starts at the
    desktop-sized default and is corrected once by a clientside callback
    after the real browser width is known (see below). A State would
    read that correction too late to matter for first paint - exactly
    the case that matters most, a phone user opening the map without
    touching any other control - so this fires once more on load in
    exchange for the initial render actually being right. It
    deliberately does not react to later window resizes/rotation, same
    one-time-measure tradeoff as not adding a resize listener.
    """
    return _build_field_map(var_key, time_idx, unit, selected_station, viewport_width)


@app.callback(
    Output("leaderboard-panel", "children"),
    Output("risk-legend-panel", "children"),
    Input("variable-selector", "value"),
    Input("current-time-idx",  "data"),
    Input("unit-selector",     "value"),
    Input("leaderboard-mode",  "value"),
    Input("selected-station",  "data"),
)
def update_leaderboard_and_legend(var_key, time_idx, unit, leaderboard_mode, selected_station):
    """
    Split off from update_map: the leaderboard's pattern-matched row IDs
    and the legend don't need field-map.figure's own rebuild-cost concerns
    (that one's a single-owner callback, no allow_duplicate split, since a
    dual-owner split was suspected of being why its highlight ring wasn't
    reliably rendering in the browser). selected_station is still a plain
    Input here, same as update_map takes it - the leaderboard build itself
    is cheap (no image encoding), so bolding the selected city's row on
    every click doesn't carry that same risk.

    Two leaderboard modes: "now" is the original slider-position snapshot
    (reshuffles as you scrub/play the animation); "peak" is a stable
    ranking by each city's worst moment anywhere in the 5-day window, so
    there's still a "who has it worst overall" read while the animation
    is playing. "peak" doesn't depend on time_idx at all, but it's cheap
    (in-memory xarray max, no I/O) so recomputing it on every slider tick
    isn't worth a separate callback just to avoid.
    """
    if _GFS_DS is None:
        return html.Div(), html.Div()
    time_idx = int(time_idx or 0)
    if leaderboard_mode == "peak":
        leaderboard = _peak_leaderboard_table(unit, selected_station=selected_station)
    else:
        # Day-level label, not the precise instant _et_utc_label gives
        # elsewhere - the ranking below is now a per-day peak, not tied to
        # this exact timestamp, so labeling it down to the minute would
        # overstate the precision of what's actually being shown.
        day_label = _to_et(_GFS_DS.time.values[time_idx]).strftime("%A, %b %d")
        leaderboard = _leaderboard_table(time_idx, unit, day_label, selected_station=selected_station)
    legend = _risk_legend()
    return leaderboard, legend


@app.callback(
    Output("selected-station", "data", allow_duplicate=True),
    Input("field-map", "clickData"),
    State("selected-station", "data"),
    prevent_initial_call=True,
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
    Output("selected-station", "data", allow_duplicate=True),
    Input({"type": "leaderboard-station", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_station_from_leaderboard(_n_clicks):
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "leaderboard-station":
        return triggered["index"]
    return no_update


@app.callback(
    Output("station-panel-headline",     "children"),
    Output("station-panel-analysis",     "children"),
    Output("station-panel-charts",       "children"),
    Output("station-panel-verification", "children"),
    Output("station-hint",    "children"),
    Output("bias-window-wrap",  "style"),
    Output("bias-window-label", "children"),
    Input("selected-station", "data"),
    Input("current-time-idx", "data"),
    Input("unit-selector",    "value"),
    Input("bias-window-dropdown", "value"),
    Input("bias-display-mode",    "value"),
)
def update_station_panel(station_id, time_idx, unit, bias_window, bias_display_mode):
    time_idx = int(time_idx or 0)
    hidden_window = ({"display": "none"}, "")

    if not station_id:
        return (
            html.Div(style={"height": "60px"}),
            html.Div(),
            html.Div(),
            html.Div(),
            "Click a station on the map to see GFS forecast + ASOS observations",
            *hidden_window,
        )

    stn = get_station(station_id)
    if stn is None:
        return (
            _station_placeholder(f"Station {station_id} not in catalog."),
            html.Div(),
            html.Div(),
            html.Div(),
            f"Station: {station_id}",
            *hidden_window,
        )

    # Fetch ASOS from IEM on first click, or once the cached copy is older
    # than the TTL; serve from cache in between. Only cache non-empty results
    # - a transient IEM failure/rate-limit (empty df) shouldn't be remembered
    # forever, or the station gets stuck at 0 obs.
    cached = _asos_cache.get(station_id)
    if cached is not None and pd.Timestamp.utcnow() - cached[0] < _ASOS_CACHE_TTL:
        asos_df = cached[1]
    else:
        print(f"[app] Fetching ASOS for {station_id} from IEM ...", end=" ")
        asos_df = fetch_station_obs(station_id, hours=72)
        if not asos_df.empty:
            _asos_cache[station_id] = (pd.Timestamp.utcnow(), asos_df)
        elif cached is not None:
            asos_df = cached[1]   # fetch failed - fall back to stale cache rather than showing nothing
        print(f"{len(asos_df)} obs")

    # Interactive same-day bias-correction solver: the dropdown's own
    # options/value need no per-station clamping (unlike a slider's min/
    # max/value, which had to be driven by explicit Outputs to avoid
    # fighting with its own Input) - "Last 3h" means the same thing
    # regardless of station, so it can just be read directly.
    window_hours = None if bias_window in (None, "all") else float(bias_window)
    show_raw = bias_display_mode == "raw"
    all_metrics = ["hi", "t2m", "td2m"]
    _, n_available = _build_station_figure(station_id, asos_df, time_idx, unit=unit, metrics=all_metrics)

    if n_available is None or n_available <= _BIAS_MIN_PAIRS:
        window_outputs = hidden_window
    else:
        trend_lines = [_bias_trend_summary(station_id, m, unit) for m in all_metrics]
        trend_text  = "  ·  ".join(t for t in trend_lines if t)
        label = f"Bias correction window ({n_available} same-day obs available)"
        if trend_text:
            label += f"  -  Today's trend: {trend_text}"
        window_outputs = (
            {"display": "block", "marginBottom": "10px"},
            label,
        )

    # Two panels, not one combined chart with togglable lines: Feels Like
    # is the headline (its own chart), Actual Temp + Dewpoint share a
    # second chart specifically so the T-Td *spread* is readable - the
    # closer the two lines sit to each other, the higher the relative
    # humidity, which is a genuinely standard way meteorologists read
    # moisture conditions off a plot without needing a separate RH number.
    fig_feels_like, _ = _build_station_figure(
        station_id, asos_df, time_idx, unit=unit, metrics=["hi"],
        bias_window_hours=window_hours, show_raw=show_raw)
    fig_temp_dewpoint, _ = _build_station_figure(
        station_id, asos_df, time_idx, unit=unit, metrics=["t2m", "td2m"],
        bias_window_hours=window_hours, show_raw=show_raw)

    hint  = (f"Station: {station_id} - {stn['name']} ({stn['state']})  "
             f"·  {len(asos_df)} recent ASOS obs  ·  Click another station to switch")

    verification_stats = _station_verification(station_id, asos_df)
    verification_children = (
        [_verification_box(station_id, unit, verification_stats)]
        if verification_stats else []
    )

    current_temp_c = None
    if _GFS_DS is not None:
        _sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")
        _idx = min(time_idx, len(_GFS_DS.time) - 1)
        current_temp_c = float(_GFS_DS["t2m"].sel(**_sel).isel(time=_idx).values)

    panel_headline = html.Div([
        _hero_tile(station_id, time_idx, unit, asos_df),
        _climate_context(station_id, unit),
        _overnight_and_streak_panel(station_id),
    ])
    panel_analysis = html.Div([
        _gev_popup(station_id, unit, current_temp_c),
    ])
    panel_charts = html.Div([
        dcc.Graph(figure=fig_feels_like, config={"displayModeBar": False}),
        html.Div("Temperature & Dewpoint - closer lines mean higher humidity",
                 style={"fontSize": "11px", "color": "#64748b", "margin": "4px 0 0 4px"}),
        dcc.Graph(figure=fig_temp_dewpoint, config={"displayModeBar": False}),
    ])
    panel_verification = html.Div(verification_children)
    return (panel_headline, panel_analysis, panel_charts, panel_verification,
            hint, *window_outputs)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if _GFS_DS is None:
        print(
            "\nNo GFS data found. Pre-fetch it first:\n"
            "    python scripts/fetch_gfs_conus.py\n"
        )
    app.run(debug=True, host="0.0.0.0", port=8051)
