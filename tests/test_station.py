"""The station page on a synthetic day.

1. The same-day correction is `bias.py`'s number, applied to the next 24 hours and no further,
   and its interval is drawn only for temperature.
2. The learned correction is drawn as logged, inside its display window, with its band.
3. Today's high takes the observations so far and the best available line for the hours left.
4. A forecast that has ended, or a station with no observations, still gives a page.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from service import live, sources, station
from service.app import app, dataset
from src.heat.bias import today_forecast_bias
from src.heat.stations import get_station

STN = get_station("KDCA")
NOW = pd.Timestamp("2026-07-15 18:30", tz="UTC")  # 14:30 EDT
TIMES = pd.date_range("2026-07-15 06:00", periods=37, freq="2h")  # 02:00 EDT Jul 15 to Jul 18
LATS = np.arange(51.0, 22.9, -0.25)
LONS = np.arange(-127.0, -64.9, 0.25)


def forecast() -> xr.Dataset:
    """25 C everywhere, rising to 30 C at 20Z (16:00 EDT) on the first day; heat index one degree above."""
    t = np.full(len(TIMES), 25.0)
    t[TIMES == pd.Timestamp("2026-07-15 20:00")] = 30.0
    cube = np.broadcast_to(t[:, None, None], (len(TIMES), len(LATS), len(LONS))).astype("float32")
    ds = xr.Dataset({"t2m": (("time", "latitude", "longitude"), cube.copy()), "hi": (("time", "latitude", "longitude"), cube + 1.0)},
                    coords={"time": TIMES, "latitude": LATS, "longitude": LONS})
    ds.attrs["gfs_init"] = "2026-07-15 06:00:00"
    return ds


def observations() -> pd.DataFrame:
    """Hourly from 06Z to 18Z today: the forecast plus 2 C, give or take half a degree."""
    t = pd.date_range("2026-07-15 06:00", "2026-07-15 18:00", freq="1h", tz="UTC")
    wobble = np.where((np.arange(len(t)) // 2) % 2 == 0, 0.5, -0.5)  # alternates at the two-hourly forecast steps
    return pd.DataFrame({"valid_utc": t, "temp_c": 27.0 + wobble, "dewpoint_c": 10.0, "rh": 35.0, "wind_spd_kt": 5.0})


def learned() -> pd.DataFrame:
    t = pd.date_range("2026-07-15 06:00", periods=31, freq="2h", tz="UTC")  # logged to 60 h of lead
    df = pd.DataFrame({"ml_c": 26.5, "lo_c": 25.0, "up_c": 28.0, "raw_c": 25.0}, index=t)
    df.attrs["version"] = "xgb-v2"
    return df


def by_label(page: dict, chart: str = "t2m") -> dict:
    return {line["label"]: line for line in page["charts"][chart]}


def at(page: dict, line: dict, when: str) -> float | None:
    return line["values"][page["time"].index(int(pd.Timestamp(when, tz="UTC").timestamp()))]


def test_same_day_correction_is_bias_py_applied_for_24_hours():
    ds, obs = forecast(), observations()
    page = station.build(ds, STN, obs, None, None, "C", NOW)
    local = ds["t2m"].sel(latitude=STN["lat"], longitude=STN["lon"], method="nearest").to_series()
    local.index = pd.DatetimeIndex(local.index).tz_localize("UTC").tz_convert(STN["tz"])
    o = obs.assign(valid_local=obs["valid_utc"].dt.tz_convert(STN["tz"]))
    expect = today_forecast_bias(local, o, "temp_c", o["valid_local"].max().date(), o["valid_local"].max())
    row = page["verification"]["today"][0]
    assert (row["metric"], row["n"], row["bias"], row["rmse"]) == ("Temperature", expect["n_used"], round(expect["bias"], 1), round(expect["rmse"], 1))
    assert row["interval"] == round(expect["pi95_halfwidth"], 1) and row["interval"] > 0

    line = by_label(page)["Corrected with today's observations"]
    assert at(page, line, "2026-07-15 20:00") == round(30.0 + expect["bias"], 1)  # the next step
    assert at(page, line, "2026-07-15 18:00") is None  # not drawn over the past
    assert at(page, line, "2026-07-16 18:00") == round(25.0 + expect["bias"], 1)  # 23.5 h ahead: inside
    assert at(page, line, "2026-07-16 20:00") is None  # 25.5 h ahead: outside
    up, lo = line["band"]["up"], line["band"]["lo"]
    i = page["time"].index(int(pd.Timestamp("2026-07-15 20:00", tz="UTC").timestamp()))
    assert up[i] - lo[i] == pytest.approx(2 * expect["pi95_halfwidth"], abs=0.11)
    assert "band" not in by_label(page, "hi")["Corrected with today's observations"]  # no interval for feels like


def test_learned_correction_is_drawn_as_logged_in_its_window():
    page = station.build(forecast(), STN, observations(), learned(), None, "C", NOW)
    line = by_label(page)["Learned correction (xgb-v2)"]
    assert at(page, line, "2026-07-15 20:00") == 26.5
    i = page["time"].index(int(pd.Timestamp("2026-07-16 10:00", tz="UTC").timestamp()))
    assert (line["band"]["lo"][i], line["band"]["up"][i]) == (25.0, 28.0)
    assert at(page, line, "2026-07-17 18:00") == 26.5  # 47.5 h ahead: inside
    assert at(page, line, "2026-07-17 20:00") is None  # 49.5 h ahead: outside
    assert page["has_learned"] and "Learned correction (xgb-v2)" not in by_label(page, "hi")


def test_todays_high_uses_observations_and_the_best_line_left():
    ds, obs = forecast(), observations()
    plain = station.build(ds, STN, obs, None, None, "C", NOW)
    tiles = {t["label"]: t for t in plain["tiles"]}
    assert tiles["Today's high, raw GFS"]["value"] == "30°C"
    assert tiles["Today's high, best estimate"]["value"] == "32°C" and tiles["Today's high, best estimate"]["note"] == "from today's observations"
    assert tiles["Observed high so far"]["value"] == "28°C" and tiles["Now"]["value"] == "28°C"
    assert "too cold" in plain["headline"] and "32°C, against 30°C raw" in plain["headline"]
    # With the learned line the remaining hours come from it: 26.5 C, so the observed 27.5 C stands.
    with_ml = station.build(ds, STN, obs, learned(), None, "C", NOW)
    best = {t["label"]: t for t in with_ml["tiles"]}["Today's high, best estimate"]
    assert (best["value"], best["note"]) == ("28°C", "from the learned correction")
    f = {t["label"]: t for t in station.build(ds, STN, obs, None, None, "F", NOW)["tiles"]}
    bias_c = plain["verification"]["today"][0]["bias"]
    assert f["Today's high, raw GFS"]["value"] == "86°F"
    assert bias_c == 2.1 and f["GFS error today"]["value"] == "+3.7°F"  # a difference: no 32 added


def test_a_page_without_observations_or_with_an_ended_forecast():
    ds = forecast()
    bare = station.build(ds, STN, None, None, None, "F", NOW)
    assert [line["label"] for line in bare["charts"]["t2m"]] == ["GFS, raw"] and "unavailable" in bare["headline"]
    assert bare["verification"] == {"today": [], "model": None} and not bare["stale"]
    late = station.build(ds, STN, observations(), learned(), None, "F", pd.Timestamp("2026-09-01", tz="UTC"))
    assert late["stale"] and "ended Jul 18" in late["headline"]
    assert [line["label"] for line in late["charts"]["t2m"]] == ["GFS, raw"]  # nothing current is drawn over it
    assert len(late["time"]) == len(TIMES)


def test_other_models_are_drawn_for_the_hours_ahead_only():
    t = pd.date_range("2026-07-15 12:00", periods=48, freq="1h", tz="UTC")
    others = {"gfs_hrrr": pd.Series(24.0, index=t), "unknown_model": pd.Series(1.0, index=t)}
    page = station.build(forecast(), STN, observations(), None, None, "C", NOW, others)
    hrrr = by_label(page)["HRRR, raw"]
    assert at(page, hrrr, "2026-07-15 18:00") is None and at(page, hrrr, "2026-07-15 19:00") == 24.0
    assert "unknown_model, raw" not in by_label(page) and "HRRR, raw" not in by_label(page, "hi")
    late = station.build(forecast(), STN, observations(), None, None, "C", pd.Timestamp("2026-09-01", tz="UTC"), others)
    assert "HRRR, raw" not in by_label(late)


def test_model_scores_are_converted_as_differences():
    model = {"version": "xgb-v1", "days": 14, "n": 1200, "rmse_raw_c": 2.0, "rmse_offset_c": 1.8, "rmse_ml_c": 1.5, "band_coverage": 0.943}
    row = station.build(forecast(), STN, observations(), None, model, "F", NOW)["verification"]["model"]
    assert row == {"version": "xgb-v1", "days": 14, "n": 1200, "rmse_raw": 3.6, "rmse_offset": 3.24, "rmse_ml": 2.7, "band_coverage_pct": 94}


def test_routes_serve_the_page_and_the_network_table(monkeypatch):
    live.clear()
    ds = forecast()
    app.dependency_overrides[dataset] = lambda: ds
    monkeypatch.setattr("service.app._now", lambda: NOW)
    monkeypatch.setattr(sources, "observations", lambda sid: observations())
    monkeypatch.setattr(sources, "ml_lines", lambda init: {"KDCA": learned()})
    monkeypatch.setattr(sources, "ml_verification", lambda days=14: None)
    monkeypatch.setattr(sources, "models", lambda sid, lat, lon: None)
    monkeypatch.setattr(sources, "network_latest", lambda ids, now: {"KDCA": (27.0, NOW - pd.Timedelta(minutes=30))})
    try:
        c = TestClient(app)
        page = c.get("/api/station/kdca?unit=C").json()
        assert page["station"]["id"] == "KDCA" and page["has_learned"] and page["unit"] == "C"
        assert c.get("/api/station/NOPE").status_code == 404
        net = c.get("/api/network?unit=C").json()
        row = next(r for r in net["rows"] if r["id"] == "KDCA")
        # 18:30Z is nearest the 18Z step (25 C); the high today is the 30 C step, tomorrow 25 C.
        assert (row["observed"], row["observed_min_ago"], row["forecast"], row["error"]) == (27.0, 30, 25.0, 2.0)
        assert (row["high_today"], row["high_tomorrow"]) == (30.0, 25.0)
        assert net["reporting"] == 1 and net["mean_abs_error"] == 2.0 and net["current"]
    finally:
        app.dependency_overrides.clear()


def test_network_observations_come_back_under_icao_codes(monkeypatch):
    live.clear()
    asked = {}

    def fake(ids, start, end, batch_size=100):
        asked["ids"], asked["batch"] = list(ids), batch_size
        t = pd.Timestamp("2026-07-15 18:00", tz="UTC")
        rows = [("DCA", t - pd.Timedelta(minutes=50), 20.0), ("DCA", t, 21.0), ("DJT", t, 30.0), ("ZZZ", t, 1.0), ("LAX", t, float("nan"))]
        return pd.DataFrame(rows, columns=["station_id", "valid_utc", "temp_c"]), {}

    monkeypatch.setattr(sources, "fetch_obs_window", fake)
    got = sources.network_latest(["KDCA", "KPBI", "KLAX"], NOW)
    assert asked == {"ids": ["DCA", "DJT", "LAX"], "batch": 3}  # one request, IEM's own ids, the West Palm Beach alias
    assert got == {"KDCA": (21.0, pd.Timestamp("2026-07-15 18:00", tz="UTC")), "KPBI": (30.0, pd.Timestamp("2026-07-15 18:00", tz="UTC"))}
    live.clear()


def test_iem_requests_are_spaced(monkeypatch):
    clock, slept = [100.0], []
    monkeypatch.setattr(sources.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sources.time, "sleep", lambda s: (slept.append(round(s, 2)), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(sources, "_iem_last", [0.0])
    assert [sources._iem(lambda: "a"), sources._iem(lambda: "b")] == ["a", "b"]
    assert slept == [sources.IEM_GAP_S]  # the first goes at once, the second waits out the gap
    clock[0] += 10
    sources._iem(lambda: "c")
    assert slept == [sources.IEM_GAP_S]
