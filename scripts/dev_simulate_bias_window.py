"""
scripts/dev_simulate_bias_window.py
====================================
Dev-only harness: seeds fake ASOS observations for one station (default
KDCA) with a known bias and enough same-day pairs to make the interactive
bias-correction window dropdown immediately usable, then launches the
actual app so you can try it in a browser.

This does NOT modify app.py - it's a standalone script that monkey-patches
fetch_station_obs before the real Dash server starts, purely for local
manual testing. Not part of the deployed app.

Usage
-----
    python scripts/dev_simulate_bias_window.py [--station KDCA] [--n 12]

Then open http://localhost:8051, click the seeded station, and try the
"Bias correction window" dropdown (Last 3h / 6h / 12h / All of today).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def _seed_asos_df(station_id: str, n: int, heat_app) -> pd.DataFrame:
    """Synthetic ASOS observations with a known warm bias, for exercising
    the bias-correction UI without waiting on real IEM data.

    n hourly observations spanning as much of the forecast's first
    local day as possible, running a steady +2.5C warm bias against the
    real forecast plus a bit of scatter so the confidence band has
    visible width.

    Two things a naive "n hours ending at wall-clock now" version got
    wrong, found by checking actual state after seeding:

    1. "now" for the bias correction is the latest observation's
       timestamp, not real wall-clock time. Observations need to
       actually extend into the forecast's local "today," not just end
       near real now.
    2. n_available is capped by how many forecast steps (2-hourly here)
       fall in [start of local today, now], not by how many
       observations exist. Seeding more observations than there are
       forecast steps to pair against does not raise n_available at
       all.

    So this anchors directly on the forecast series' own local calendar
    day instead of wall-clock time.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code to seed, e.g. "KDCA".
    n : int
        Number of hourly observations to synthesize.
    heat_app : module
        The imported app module, used to read the real GFS forecast
        the synthetic observations should look plausible against.

    Returns
    -------
    pd.DataFrame
        Same shape fetch_station_obs would return.
    """
    from zoneinfo import ZoneInfo

    stn = heat_app.get_station(station_id)
    tz = ZoneInfo(stn.get("tz", "America/New_York")) if stn else ZoneInfo("America/New_York")

    sel = dict(latitude=stn["lat"], longitude=stn["lon"], method="nearest")
    fcst_index = heat_app._GFS_DS["t2m"].sel(**sel).to_series().index
    fcst_local = pd.DatetimeIndex(fcst_index).tz_localize("UTC").tz_convert(tz) \
        if pd.DatetimeIndex(fcst_index).tz is None else pd.DatetimeIndex(fcst_index).tz_convert(tz)

    today = fcst_local[0].date()
    today_steps = fcst_local[fcst_local.date == today]
    # Pretend "now" is the last forecast step still on the forecast's own
    # first local day, so every 2-hourly step that day counts as "past".
    now_local = today_steps.max()

    times = pd.date_range(end=now_local, periods=n, freq="1h", tz=tz).tz_convert("UTC")
    rng = np.random.default_rng(seed=42)

    base_temp_c = 32.0
    noise = rng.normal(loc=2.5, scale=0.8, size=n)   # mean +2.5C bias, real scatter
    temp_c = base_temp_c + noise
    dewpoint_c = temp_c - rng.uniform(8.0, 12.0, size=n)

    return pd.DataFrame({
        "valid_utc":   times,
        "temp_c":      temp_c,
        "dewpoint_c":  dewpoint_c,
        "rh":          60.0,
        "wind_spd_kt": 5.0,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="KDCA", metavar="ICAO")
    parser.add_argument("--n", type=int, default=12, metavar="N",
                        help="Number of seeded hourly observations (default 12).")
    args = parser.parse_args()

    import app as heat_app

    seeded_df = _seed_asos_df(args.station, args.n, heat_app)
    heat_app._asos_cache[args.station] = (pd.Timestamp.utcnow(), seeded_df)

    # Patch fetch_station_obs so a cache miss on the SEEDED station (e.g.
    # after the 15-min TTL expires) still gets fake data instead of a real
    # IEM call. Real bug, caught live: this used to patch unconditionally
    # for every station, so clicking any *other* station (e.g. KABQ) also
    # got fake, unrealistically humid synthetic observations compared
    # against its real forecast - producing a wildly inflated, nonsensical
    # bias correction that looked like a real bug in the correction logic
    # itself. Every other station must keep hitting the real IEM fetch.
    _real_fetch = heat_app.fetch_station_obs

    def _fake_fetch(station_id, hours=72):
        if station_id == args.station:
            return _seed_asos_df(station_id, args.n, heat_app)
        return _real_fetch(station_id, hours=hours)

    heat_app.fetch_station_obs = _fake_fetch

    print(f"[dev] Seeded {args.n} fake ASOS obs for {args.station} "
          f"(steady ~+2.5C warm bias, real scatter).")
    print(f"[dev] Open http://localhost:8051, click {args.station}, "
          f"and try the bias correction window dropdown.")
    print("[dev] This is fake data for UI testing only - not real observations.")

    heat_app.app.run(debug=False, host="0.0.0.0", port=8051)


if __name__ == "__main__":
    main()
