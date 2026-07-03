# US Heat Wave Tracker

A live dashboard tracking the July 2026 CONUS heat wave — GFS forecast
(temperature, Heat Index, NWS risk levels) over 165 US cities, with
real-time observations pulled from NOAA's ASOS network on click, and a
"Hottest Cities Right Now" leaderboard.

Built with Python, Dash/Plotly, and NOAA's GFS + ASOS data.

## Run it locally

```bash
git clone https://github.com/sotarolab/heat-wave-tracker.git
cd heat-wave-tracker
pip install -r requirements.txt
python scripts/fetch_gfs_conus.py   # pulls current GFS forecast (~3–5 min)
python app.py                       # → http://localhost:8051
```

## Data sources

- **Forecast**: NOAA GFS, fetched via [Herbie](https://github.com/blaylockbk/Herbie)
- **Observations**: [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) ASOS network
- **Risk categories**: official NWS heat-index thresholds

Built by Sebastian Otarola-Bustos, PhD.
