# US Heat Wave Tracker

## What this is

A Dash (Plotly) single-page dashboard tracking the July 2026 CONUS heat wave.
GFS forecast field (2m Temperature / Heat Index / NWS Risk Level) with 165
major ASOS station markers. Click any station for its GFS forecast overlaid
with live ASOS observations from IEM, shown in that station's local time
zone. Includes a "Hottest Cities Right Now" leaderboard ranked by forecasted
Heat Index. Built to host on Render as a LinkedIn showcase.

Built by Sebastian Otarola-Bustos, PhD.

---

## Architecture

**No database.** Two data sources:

| Source | What | How |
|--------|------|-----|
| GFS CONUS raster | T2m + Td2m + Heat Index, F000–F120 at 6h, CONUS | Pre-fetched once as `data/conus_heat_tracker.nc` (~14 MB). Render serves from file. |
| ASOS observations | Last 72h per station | Fetched live from IEM on station click. Cached in-process (only successful fetches — a rate-limited/failed fetch isn't cached, so it retries later). |

**Longitude convention:** The saved NetCDF uses −180 to +180 (converted from GFS 0–360 during fetch). Mapbox and station lookups use negative US longitudes directly — no normalization needed in app.py.

**Latitude order:** North-to-south, as cfgrib returns GFS data. `_field_to_image` checks this and does not flip (correct for mapbox image layers).

**Time zones:** GFS times are stored tz-naive in the NetCDF but represent UTC.
Each station in `stations.py` carries an IANA `tz` field (generated once via
`timezonefinder`, hardcoded — not a runtime dependency). The station panel
converts to that station's local time; the map header has no single "local"
time (one raster snapshot spans 4 US zones), so it displays Eastern Time
alongside UTC as a display convention.

---

## File structure

```
heat-wave-tracker/
├── app.py                      ← Dash app, Gunicorn entry point (app.server)
├── requirements.txt
├── Dockerfile
├── assets/
│   └── social_preview.png      ← Open Graph / Twitter Card preview image
├── data/
│   └── conus_heat_tracker.nc   ← Pre-fetched GFS (create with fetch script)
├── scripts/
│   └── fetch_gfs_conus.py      ← CLI to download GFS data
└── src/heat/
    ├── compute.py              ← heat_index_array() (NWS Rothfusz)
    ├── stations.py             ← 165 major CONUS ASOS stations (hardcoded, with tz)
    ├── asos.py                 ← IEM live fetch for one station
    └── gfs_conus.py            ← Herbie GFS fetch + NetCDF save
```

---

## How to run locally

```bash
git clone https://github.com/sotarolab/heat-wave-tracker.git
cd heat-wave-tracker
pip install -r requirements.txt

# Step 1: fetch the GFS data (~3–5 min, downloads from NOAA AWS via Herbie)
python scripts/fetch_gfs_conus.py

# Step 2: start the app
python app.py
# → http://localhost:8051
```

The app starts without data and shows an instruction message. Fetching
data after startup is not supported — run the script first.

---

## Variables

| Key | Label | Formula |
|-----|-------|---------|
| `t2m` | 2m Temperature | GFS T2m, K→°C |
| `hi`  | Heat Index | NWS Rothfusz (1990 SR 90-23). Input: T+RH. Output: °C. |
| `risk` | Risk Level (Heat Index) | Discrete NWS heat-index categories, daily max (see below). |

HI is computed in `src/heat/compute.py` and stored in the NetCDF. Switching
variables in the app requires no recomputation. `risk` is derived from `hi`
at render time (no new NetCDF variable) and can be shown in °F or °C.

**Risk Level categories** (`RISK_CATEGORIES_F`/`_C` in `app.py`, official NOAA
thresholds): Caution 80–90°F, Extreme Caution 90–103°F, Danger 103–125°F,
Extreme Danger 125°F+. Below Caution shows as "No Elevated Risk" (a real
category, not missing data — colored distinctly from the "no data" gray).
For a selected day, shown as that day's maximum HI per grid cell — mirrors
how NWS/media heat-risk maps present "highest level forecast each day"
rather than an instantaneous value.

---

## Key UI pieces

- **Day + Time dropdowns** replace a slider — no animation/autoplay; simpler
  and more compact than scrubbing a timeline.
- **Hero tile**: big "feels like" number + plain-language risk category for
  the selected station, ahead of the line chart.
- **Hottest Cities leaderboard**: top 15 stations ranked by forecasted Heat
  Index for the selected day/time. Ranked by *current forecasted severity*,
  not historical records — this app has no climate-normals data source.
- **°F/°C toggle** applies everywhere (map colorbar, station chart, leaderboard).

ASOS data is cached in `_asos_cache` after the first click, so subsequent
updates for that station are pure Python (no HTTP).

---

## Station markers

All 165 stations from `src/heat/stations.py` are shown on the map as dots
colored by the current GFS value at that grid point — same colorscale as the
field (or the same discrete risk category colors, for the Risk Level
variable). `customdata` on each marker holds the ICAO ID, used by the
`select_station` callback to open that station's panel.

---

## Deploying to Render

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. New Render web service → connect the repo → "Docker" environment.
3. Render build: `docker build`, start: `gunicorn app:server` (in the Dockerfile).
4. Single worker (`--workers=1`) because `_GFS_DS` and `_asos_cache` are
   process-global — multiple workers would each reload the ~14 MB file.
5. Once deployed, update the `og:image`/`og:url` meta tags in `app.py`'s
   `index_string` to the live absolute URL for reliable social-card previews.

---

## Fetch script options

```bash
# Latest GFS cycle, 21 steps at 6h (default)
python scripts/fetch_gfs_conus.py

# Specific init time
python scripts/fetch_gfs_conus.py --init "2026-07-02 00:00"

# Denser data (41 steps at 3h, larger file)
python scripts/fetch_gfs_conus.py --step 3 --hours 120

# Overwrite existing file
python scripts/fetch_gfs_conus.py --overwrite
```

---

## Known limitations / easy next steps

- **No GitHub Actions cron.** Data is a snapshot. To update: re-run fetch
  script locally and re-deploy (push new NetCDF).
- **Single Render worker.** Fine for a short-run showcase; not for
  multi-user production.
- **ASOS cache is process-lifetime only.** Restarts clear it; first click
  per station re-fetches from IEM (~1–2 s).
- **No small-multiples overview.** Considered a NYT-style 6-panel daily-risk
  thumbnail strip above the main map; deferred in favor of one interactive
  map + day/time dropdowns, since it keeps station click/hover working.
- **Timezone assignment is per-station, not per-instant.** `stations.py`'s
  `tz` field was generated once via `timezonefinder` on each station's
  lat/lon and hardcoded — correct for all 165 current stations but a new
  station added by hand needs its `tz` looked up the same way (don't guess
  from `state` alone; several states span multiple zones).
- **No historical-records comparison.** The leaderboard ranks by current
  forecast severity, not against climate normals — adding that would need a
  separate historical data source.
