# US nowcast

[![QC](https://github.com/sotarolab/heat-wave-tracker/actions/workflows/qc.yml/badge.svg)](https://github.com/sotarolab/heat-wave-tracker/actions/workflows/qc.yml)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

**Live: https://heat-wave-tracker.onrender.com**

Station temperature forecasts for 165 US cities, corrected and checked against
what happened. Every six hours a job fetches the GFS forecast, pairs it with the
observations that have arrived since, runs a learned correction, and logs both.
The page shows the forecast, the corrections and how each has verified.

It began as a tracker for the July 2026 heat wave and now runs year-round.

## The app

| Tab | What it shows | From |
|---|---|---|
| Forecast | One station: what it measured, GFS as issued, HRRR and NBM for the hours ahead, the same-day correction with its interval, the learned correction with its band, today's high, and today's verification. Under it, dew point, humidity, wind, cloud, precipitation and pressure, observed and forecast. | GFS (NOAA), ASOS by way of IEM, HRRR and NBM by way of Open-Meteo, the forecast archive |
| Map | The GFS field over the lower 48 for the next five days, every station in the same colours, radar or satellite over it. A station clicked on the map shows its forecast underneath. | GFS, IEM radar and GOES composites |
| Stations | Every station's latest reading beside GFS for now, largest misses first. | ASOS by way of IEM |
| Satellite | GOES loops by sector and band, and the national radar loop. | NOAA STAR, NWS |

## The two corrections

- **Same-day correction.** GFS shifted by its mean error at the station so far
  today, for the next 24 hours, with a 95% prediction interval from today's
  residuals (`src/heat/bias.py`). It needs no history and works from the third
  verified forecast step of the day.
- **Learned correction.** A gradient-boosted model trained on the archive of past
  forecasts and observations, with a calibrated band (`src/heat/correction.py`,
  notebooks 02 to 05). It runs in the refresh job before the observations arrive,
  so every row it is later scored on is out of sample. Results are in
  [docs/ml/EXPERIMENTS.md](docs/ml/EXPERIMENTS.md) and on the page.

## Architecture, plan and process

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the system, the model lifecycle
  and the layering, as diagrams, with what is built and what is planned.
- [docs/PLAN.md](docs/PLAN.md): the order of work, one issue per step.
- [docs/process/WORKFLOW.md](docs/process/WORKFLOW.md): the code is written with
  AI agents under gates. This page lists the gates, the skills and subagents kept
  in `.claude/`, and which parts a person reads line by line.
- [docs/decisions/](docs/decisions/): one page per decision, dated.

One rule holds the design together: jobs write, routes read. The scheduled job
computes and commits; the service reads what it wrote and draws nothing it had to
fit at request time, with one stated exception (the same-day correction).

## Run it locally

```bash
git clone https://github.com/sotarolab/heat-wave-tracker.git
cd heat-wave-tracker
pip install -r service/requirements.txt
cd client && npm ci && npm run build && cd ..
uvicorn service.app:app --port 8051      # http://localhost:8051
```

The forecast file in `data/` is the one the last refresh committed. For a newer
one, `pip install -r requirements.txt` and `python scripts/fetch_gfs_conus.py --overwrite`
(three to five minutes; needs eccodes).

While working on the client, `npm run dev` in `client/` serves it with `/api`
proxied to port 8051.

The learned correction and its scored record are read from the forecast archive.
Set `NEON_DATABASE_URL` to see them locally; without it the page shows the
observations, the models and the same-day correction.

## Checks

```bash
pip install -r requirements-dev.txt
ruff check . && mypy && pytest -q && python scripts/check_phrases.py
cd client && npm run build && npx vitest run
docker build -t us-nowcast . && docker run -p 8051:8051 us-nowcast
```

The same run in CI on every pull request, including the container build and a
call to each route.

## Deployment

One Docker container on Render (`Dockerfile`): the client is built in one stage,
the service runs in the next. The refresh job (`.github/workflows/refresh-gfs-data.yml`)
commits the new forecast file to `master` and calls Render's deploy hook, so each
cycle is a fresh container.

## Known simplifications

Left behind with the Dash page for now, with the code still in `src/heat/`:

- the return period of a forecast high against the station's 54-year record
  (`src/heat/extremes.py`);
- yesterday's high against the 1991 to 2020 normal (`src/heat/climate.py`);
- overnight lows and the heat streak count.

## Data sources

- **Forecast**: NOAA GFS, fetched with [Herbie](https://github.com/blaylockbk/Herbie); HRRR and NBM from [Open-Meteo](https://open-meteo.com/)
- **Observations**: the ASOS network, from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/)
- **Radar and satellite**: NWS, NOAA STAR, Iowa Environmental Mesonet
- **Risk categories**: NWS heat index thresholds
- **Basemap**: Esri World Dark Gray Canvas (Esri, HERE, Garmin, OpenStreetMap contributors)

Not a warning product. For heat or any other hazard, check [weather.gov](https://www.weather.gov/).

## License

[CC BY-NC 4.0](LICENSE): free to share and adapt with attribution,
non-commercial use only.
