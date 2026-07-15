# US Heat Wave Tracker

A live dashboard tracking the July 2026 CONUS heat wave: GFS forecast
(temperature, Heat Index, NWS risk levels) over 165 US cities, animated
over the full forecast window, with real observations pulled from
NOAA's ASOS network on click, and a "Hottest Cities" leaderboard.

Built with Python, Dash/Plotly, and NOAA's GFS + ASOS data.

## Beyond the raw forecast

Each station panel goes past a plain forecast line:

- **Same-day bias correction**: the raw GFS forecast is corrected
  against that station's own real observations from earlier today,
  with a genuine 95% prediction interval (not a placeholder band) that
  widens appropriately when only a handful of same-day observations
  are available, and is scoped to the near-term forecast rather than
  extrapolated across the whole multi-day window.
- **Live verification**: a running scorecard of bias, RMSE, and a
  Brier score comparing the day's forecast against what actually
  happened, updating as new observations arrive.
- **Historical extreme-value analysis**: each station's forecast is
  checked against a 54-year historical temperature record using a
  Generalized Extreme Value (GEV) fit, answering "how rare is this,
  historically, at this specific city."

## Run it locally

```bash
git clone https://github.com/sotarolab/heat-wave-tracker.git
cd heat-wave-tracker
pip install -r requirements.txt
python scripts/fetch_gfs_conus.py   # pulls current GFS forecast (~3-5 min)
python app.py                       # http://localhost:8051
```

The historical extreme-value archive (`data/climate_extremes.parquet`)
is pre-built and included; see `scripts/fetch_historical_temps.py` if
you want to regenerate it.

## Data sources

- **Forecast**: NOAA GFS, fetched via [Herbie](https://github.com/blaylockbk/Herbie)
- **Observations**: [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) ASOS network, current and historical
- **Climate normals**: NWS Daily Climate Report (1991-2020 normals)
- **Risk categories**: official NWS heat-index thresholds

## License

[CC BY-NC 4.0](LICENSE): free to share and adapt with attribution,
non-commercial use only.
