"""The web service: the JSON API, the field images, and the React client at the root.

    uvicorn service.app:app --port 8051

One service, one page. Jobs write and routes read (docs/ARCHITECTURE.md): nothing here
trains, scores or fits. The forecast dataset is the file the scheduled workflow commits;
it is opened once, on first use.
"""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import xarray as xr
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from service import fields, live, payloads, sources, station
from src.heat.gfs_conus import DEFAULT_OUT, load_or_fetch
from src.heat.stations import get_station

# A station page reads four things from outside (observations, other models, the logged
# correction, its record). Side by side they cost the slowest one, not the sum.
_READS = ThreadPoolExecutor(max_workers=8, thread_name_prefix="read")


def _warm() -> None:
    """Load what every first visitor would otherwise wait for: the forecast file, the station
    table, and the two database reads. Best effort: a failure here is found again on request."""
    try:
        ds = _open()
        if ds is not None:
            live.cached(f"stations:{payloads.cycle(ds)}:F", live.FOREVER, lambda: payloads.stations(ds, "F"))
            sources.ml_lines(str(ds.attrs.get("gfs_init", "")))
            sources.ml_verification()
    except Exception as exc:
        print(f"[service] warm-up skipped ({type(exc).__name__})")


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # In the background, so the container answers its health check at once.
    threading.Thread(target=_warm, name="warm", daemon=True).start()
    yield


app = FastAPI(title="US nowcast", version="0.1", lifespan=_lifespan)
# The stations table is 20,000 numbers; it compresses to a fifth.
app.add_middleware(GZipMiddleware, minimum_size=2000)


@lru_cache(maxsize=1)
def _open() -> xr.Dataset | None:
    """The committed forecast file, read into memory once: about 30 MB, and every route reads it."""
    # HEAT_FORECAST_FILE points a development run at a forecast fetched somewhere else.
    ds = load_or_fetch(Path(os.environ.get("HEAT_FORECAST_FILE", DEFAULT_OUT)))
    return None if ds is None else ds.load()


def dataset() -> xr.Dataset:
    """The forecast dataset, or 503 when the file is not there. Tests override this."""
    ds = _open()
    if ds is None:
        raise HTTPException(503, "The forecast file has not been fetched yet.")
    return ds


Forecast = Annotated[xr.Dataset, Depends(dataset)]


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    """Liveness. The page still serves without a forecast file, and says so in the manifest."""
    return {"ok": True, "forecast": _open() is not None}


@app.get("/api/manifest")
def manifest(ds: Forecast) -> dict[str, Any]:
    # Render sets RENDER_GIT_COMMIT on every deploy.
    return payloads.manifest(ds, _now(), os.environ.get("RENDER_GIT_COMMIT"))


@app.get("/api/stations")
def stations(ds: Forecast, unit: payloads.Unit = "F") -> Any:
    return live.cached(f"stations:{payloads.cycle(ds)}:{unit}", live.FOREVER, lambda: payloads.stations(ds, unit))


@app.get("/api/network")
def network(ds: Forecast, unit: payloads.Unit = "F") -> dict[str, Any]:
    now = _now()
    latest = sources.network_latest([str(s["id"]) for s in payloads.STATIONS], now)
    return payloads.network(ds, latest, unit, now, sources.NETWORK_WINDOW_H)


@app.get("/api/station/{station_id}")
def station_page(station_id: str, ds: Forecast, unit: payloads.Unit = "F") -> dict[str, Any]:
    stn = get_station(station_id.upper())
    if stn is None:
        raise HTTPException(404, f"No such station: {station_id}")
    obs = _READS.submit(sources.observations, stn["id"])
    others = _READS.submit(sources.models, stn["id"], stn["lat"], stn["lon"])
    lines = _READS.submit(sources.ml_lines, str(ds.attrs.get("gfs_init", "")))
    record = _READS.submit(sources.ml_verification)
    return station.build(ds, stn, obs.result(), (lines.result() or {}).get(stn["id"]), record.result(), unit, _now(), others.result())


@app.get("/api/field/meta")
def field_meta(ds: Forecast, unit: payloads.Unit = "F") -> dict[str, Any]:
    return payloads.field_meta(ds, _now(), unit)


@app.get("/api/field/{var_id}/{i}.png")
def field_png(var_id: str, i: int, ds: Forecast) -> Response:
    var = fields.VARIABLES.get(var_id)
    if var is None:
        raise HTTPException(404, f"No such field: {var_id}")
    def draw() -> bytes:
        return fields.render(payloads.field(ds, var, i), ds.latitude.values, var)

    try:
        png = live.cached(f"field:{payloads.cycle(ds)}:{var_id}:{i}", live.FOREVER, draw)
    except IndexError:
        raise HTTPException(404, f"{var.label} has no frame {i}") from None
    # The page puts the cycle in the URL, so an image never changes under its address.
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400, immutable"})


_CLIENT_DIST = Path(__file__).resolve().parent.parent / "client" / "dist"
if _CLIENT_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_CLIENT_DIST), html=True), name="client")
else:

    @app.get("/", include_in_schema=False)
    def client_not_built() -> HTMLResponse:
        """Says why there is no page, where a bare 404 would not."""
        return HTMLResponse("<p>The client is not built. Run <code>npm ci &amp;&amp; npm run build</code> in <code>client/</code>.</p>", status_code=503)
