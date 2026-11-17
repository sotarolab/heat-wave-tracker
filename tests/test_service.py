"""The service on a small synthetic forecast: route shapes, units, and where a field lands.

1. A field image is north-up and its rows are evenly spaced in Mercator y, so a feature at a
   known latitude is drawn at the Mercator position of that latitude, not the linear one.
2. Station values are the nearest grid point's, converted and rounded in Python.
3. The risk variable has one frame per local day and takes the day's maximum heat index.
"""
import io

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from PIL import Image

from service import fields, live, payloads
from service.app import app, dataset
from src.heat.stations import MAJOR_CONUS_STATIONS

LATS = np.arange(51.0, 22.9, -0.25)  # north to south, as the GFS file is
LONS = np.arange(-127.0, -64.9, 0.25)
TIMES = pd.date_range("2026-07-15 12:00", periods=13, freq="2h")  # 08:00 EDT Jul 15 to 08:00 Jul 16


def forecast() -> xr.Dataset:
    """Temperature equal to latitude everywhere; heat index 30 C, with one 40 C step on the first day."""
    lat3 = np.broadcast_to(LATS[None, :, None], (len(TIMES), len(LATS), len(LONS))).astype("float32")
    hi = np.full_like(lat3, 30.0)
    hi[3] = 40.0  # 18Z, 14:00 EDT on Jul 15
    coords = {"time": TIMES, "latitude": LATS, "longitude": LONS}
    ds = xr.Dataset({"t2m": (("time", "latitude", "longitude"), lat3.copy()), "hi": (("time", "latitude", "longitude"), hi)}, coords=coords)
    ds.attrs["gfs_init"] = "2026-07-15 12:00:00"
    return ds


@pytest.fixture()
def client():
    live.clear()
    ds = forecast()
    app.dependency_overrides[dataset] = lambda: ds
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_mercator_rows_put_a_latitude_at_its_mercator_position():
    rows = fields.mercator_rows(LATS, 2 * len(LATS))
    south, _, north, _ = fields.edges(LATS, LONS)
    y = lambda lat: np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))  # noqa: E731
    for lat in (30.0, 37.0, 45.0):
        expect = (y(north) - y(lat)) / (y(north) - y(south)) * len(rows)  # image row of this latitude
        drawn = np.flatnonzero(LATS[rows] == lat)
        assert drawn.min() <= expect <= drawn.max() + 1
    # Linear placement would be wrong by many rows in the middle of the grid.
    linear = (north - 37.0) / (north - south) * len(rows)
    assert abs(linear - np.flatnonzero(LATS[rows] == 37.0).mean()) > 5
    assert rows[0] == 0 and rows[-1] == len(LATS) - 1 and (np.diff(rows) >= 0).all()


def test_field_png_decodes_north_up_with_transparent_gaps(client):
    r = client.get("/api/field/t2m/0.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    img = np.asarray(Image.open(io.BytesIO(r.content)))
    assert img.shape == (2 * len(LATS), len(LONS), 4)
    # North is the first row: 51 C there is past the top of the scale, 23 C in the south is not.
    top = (np.array(matplotlib.colormaps["RdYlBu_r"](1.0)) * 255).astype(np.uint8)
    assert (img[0, 0, :3] == top[:3]).all() and (img[-1, 0, :3] != top[:3]).any()
    assert (img[..., 3] == int(255 * fields.OPACITY)).all()
    risk = np.asarray(Image.open(io.BytesIO(client.get("/api/field/risk/0.png").content)))
    assert (risk[..., 3] > 0).all()  # 40 C is 104 F: Danger everywhere on the first day
    assert (np.asarray(Image.open(io.BytesIO(client.get("/api/field/risk/1.png").content)))[..., 3] > 0).all()  # 30 C is 86 F: Caution


def test_unknown_field_or_frame_is_404(client):
    assert client.get("/api/field/nope/0.png").status_code == 404
    assert client.get("/api/field/t2m/99.png").status_code == 404
    assert client.get("/api/field/risk/9.png").status_code == 404
    assert client.get("/api/field/t2m/-1.png").status_code == 404  # not the last frame by negative index
    assert client.get("/api/field/risk/-1.png").status_code == 404


def test_stations_are_nearest_grid_point_in_the_asked_unit(client):
    c = client.get("/api/stations?unit=C").json()
    f = client.get("/api/stations").json()
    assert f["unit"] == "F" and len(c["stations"]) == len(MAJOR_CONUS_STATIONS)
    k = next(i for i, s in enumerate(MAJOR_CONUS_STATIONS) if s["id"] == "KDCA")
    nearest = LATS[np.abs(LATS - MAJOR_CONUS_STATIONS[k]["lat"]).argmin()]
    assert nearest == 38.75
    assert c["values"]["t2m"][k] == [38.8] * len(TIMES)  # one decimal, rounded in Python
    assert f["values"]["t2m"][k][0] == 101.8  # 38.75 C
    # Risk: one value per local day, the day's maximum heat index.
    assert c["values"]["risk"][k] == [40.0, 30.0]


def test_meta_frames_legend_and_bounds(client):
    m = client.get("/api/field/meta").json()
    assert m["cycle"] == "2026071512" and m["bounds"] == [[22.875, -127.125], [51.125, -64.875]]
    by = {v["id"]: v for v in m["variables"]}
    assert len(by["t2m"]["frames"]) == len(TIMES) and by["t2m"]["frames"][0]["label"] == "Wed Jul 15, 8 AM ET (12Z)"
    assert [f["label"] for f in by["risk"]["frames"]] == ["Wed Jul 15", "Thu Jul 16"]
    assert by["t2m"]["legend"] == {"kind": "ramp", "stops": fields.ramp_stops(fields.VARIABLES["t2m"]), "min": -4, "max": 113}
    assert [i["from"] for i in by["risk"]["legend"]["items"]] == [None, 80.0, 90.0, 103.0, 125.0]
    assert client.get("/api/field/meta?unit=C").json()["variables"][2]["legend"]["items"][1]["from"] == 26.7


def test_start_frame_is_the_step_nearest_now():
    ds = forecast()
    _, start = payloads.frames(ds, fields.VARIABLES["t2m"], pd.Timestamp("2026-07-15 19:10", tz="UTC"))
    assert start == 4  # 20Z is nearer than 18Z
    _, day = payloads.frames(ds, fields.VARIABLES["risk"], pd.Timestamp("2026-07-16 05:00", tz="UTC"))
    assert day == 1  # 01:00 EDT on Jul 16
    _, late = payloads.frames(ds, fields.VARIABLES["risk"], pd.Timestamp("2026-09-01", tz="UTC"))
    assert late == 1  # past the forecast: the last day


def test_manifest_reports_age_and_staleness():
    ds = forecast()
    m = payloads.manifest(ds, pd.Timestamp("2026-07-15 18:00", tz="UTC"), "abc123")
    assert m == {"init": "2026-07-15 12:00:00", "label": "GFS 12Z Jul 15", "age_h": 6.0, "badge": "LIVE", "live": True, "steps": 13, "stations": len(MAJOR_CONUS_STATIONS), "commit": "abc123"}
    assert payloads.manifest(ds, pd.Timestamp("2026-07-18 12:00", tz="UTC"), None)["badge"] == "STALE 3 d"
    ds.attrs["gfs_init"] = ""
    none = payloads.manifest(ds, pd.Timestamp("2026-07-18 12:00", tz="UTC"), None)
    assert (none["label"], none["badge"], none["live"]) == ("No forecast", "NO DATA", False)


def test_missing_forecast_is_503_not_a_crash():
    live.clear()

    def gone():
        from fastapi import HTTPException
        raise HTTPException(503, "The forecast file has not been fetched yet.")

    app.dependency_overrides[dataset] = gone
    try:
        assert TestClient(app).get("/api/manifest").status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_cache_serves_the_last_good_value_and_does_not_hammer_a_failing_feed(monkeypatch):
    live.clear()
    clock = [1000.0]
    monkeypatch.setattr(live.time, "time", lambda: clock[0])
    calls = []

    def flaky():
        calls.append(clock[0])
        if len(calls) != 2:
            raise RuntimeError("429")
        return "good"

    with pytest.raises(RuntimeError):
        live.cached("k", 900, flaky)
    with pytest.raises(live.Unavailable):  # ten seconds later: not asked again
        clock[0] += 10
        live.cached("k", 900, flaky)
    assert len(calls) == 1
    clock[0] += 60
    assert live.cached("k", 900, flaky) == "good"
    clock[0] += 901  # stale, and the refetch fails: the last good value is served
    assert live.cached("k", 900, flaky) == "good" and len(calls) == 3
    live.clear()
