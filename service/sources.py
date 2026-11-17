"""What the service reads from outside the forecast file: live observations and the database.

Each read goes through `live.cached`, so a slow feed costs one request and a failed one
serves the last good value. Without NEON_DATABASE_URL the database reads return None and
the page shows the forecast without the learned correction.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
import requests

from service import live
from src.heat.asos import _STATION_ALIASES, fetch_station_obs
from src.heat.da.asos_network import fetch_obs_window

OBS_TTL_S = 15 * 60
NETWORK_TTL_S = 10 * 60
NETWORK_WINDOW_H = 2
DB_TTL_S = 10 * 60
MODELS_TTL_S = 60 * 60
# Other models at the station, by way of Open-Meteo: the label on the chart and the colour token.
# GFS is listed for the variables the committed forecast file does not carry.
MODELS = {"gfs_global": ("GFS", "--raw"), "gfs_hrrr": ("HRRR", "--series-1"), "ncep_nbm_conus": ("NBM", "--series-3")}
# Open-Meteo's name for each variable and the column it becomes: Celsius, percent, km/h, mm, hPa.
MODEL_VARS = {
    "temperature_2m": "temp_c", "dew_point_2m": "dewpoint_c", "relative_humidity_2m": "rh", "wind_speed_10m": "wind_kmh",
    "precipitation": "precip_mm", "cloud_cover": "cloud_pct", "pressure_msl": "pressure_hpa",
}
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
# IEM answers requests sent in quick succession with 429. Calls to it go one at a time, spaced.
IEM_GAP_S = 1.5
_iem_lock = threading.Lock()
_iem_last = [0.0]
T = TypeVar("T")


def _iem(call: Callable[[], T]) -> T:
    """Run one IEM request, no sooner than IEM_GAP_S after the last one began."""
    with _iem_lock:
        wait = _iem_last[0] + IEM_GAP_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _iem_last[0] = time.monotonic()
        return call()


def observations(station_id: str) -> pd.DataFrame | None:
    """The station's last 72 hours from IEM, or None when IEM is down and nothing is held."""

    def fetch() -> pd.DataFrame:
        df = _iem(lambda: fetch_station_obs(station_id, hours=72))
        if df.empty:  # fetch_station_obs returns an empty frame on any error
            raise RuntimeError(f"no observations for {station_id}")
        return df

    try:
        return live.cached(f"obs:{station_id}", OBS_TTL_S, fetch)  # type: ignore[no-any-return]
    except Exception:
        return None


def iem_id(icao: str) -> str:
    """The id IEM files a station under: the ICAO code without its leading K, or a known alias."""
    return str(_STATION_ALIASES.get(icao, icao[1:] if len(icao) == 4 and icao.startswith("K") else icao))


def network_latest(station_ids: list[str], now: pd.Timestamp) -> dict[str, tuple[float, pd.Timestamp]] | None:
    """Each station's most recent temperature (Celsius) in the last NETWORK_WINDOW_H hours, and when, by ICAO code."""

    def fetch() -> dict[str, tuple[float, pd.Timestamp]]:
        icao = {iem_id(s): s for s in station_ids}  # IEM answers under its own ids
        # One request for the whole network: IEM refuses a second one sent straight after the first.
        obs, _ = _iem(lambda: fetch_obs_window(list(icao), now - pd.Timedelta(hours=NETWORK_WINDOW_H), now, batch_size=len(icao)))
        obs = obs.dropna(subset=["temp_c"]).sort_values("valid_utc")
        if obs.empty:
            raise RuntimeError("no network observations")
        last = obs.groupby("station_id").tail(1)
        return {
            icao[str(s)]: (float(t), pd.Timestamp(v))
            for s, t, v in zip(last["station_id"], last["temp_c"], last["valid_utc"], strict=True)
            if str(s) in icao
        }

    try:
        return live.cached("network", NETWORK_TTL_S, fetch)  # type: ignore[no-any-return]
    except Exception:
        return None


def models(station_id: str, lat: float, lon: float) -> dict[str, pd.DataFrame] | None:
    """The latest GFS, HRRR and NBM at the station for the hours ahead: one frame per model,
    columns as in MODEL_VARS, by UTC time.

    Only the hours ahead are asked for. Open-Meteo fills past hours with each model's most
    recent run, which would sit beside the observations looking better than any forecast
    that was actually issued.
    """

    def fetch() -> dict[str, pd.DataFrame]:
        r = requests.get(
            OPEN_METEO,
            params={"latitude": str(lat), "longitude": str(lon), "hourly": ",".join(MODEL_VARS), "models": ",".join(MODELS),
                    "past_hours": "0", "forecast_hours": "72", "timeformat": "unixtime", "timezone": "UTC"},
            timeout=15,
        )
        r.raise_for_status()
        hourly = r.json()["hourly"]
        index = pd.to_datetime(hourly["time"], unit="s", utc=True)
        out = {}
        for m in MODELS:
            df = pd.DataFrame({col: pd.Series(hourly.get(f"{var}_{m}"), index=index, dtype=float) for var, col in MODEL_VARS.items()})
            if df.notna().any(axis=None):
                out[m] = df
        return out

    try:
        return live.cached(f"models:{station_id}", MODELS_TTL_S, fetch)  # type: ignore[no-any-return]
    except Exception:
        return None


def _query(sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]] | None:
    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        return None
    try:
        import psycopg2  # type: ignore[import-untyped]

        conn = psycopg2.connect(url, connect_timeout=10)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())
        finally:
            conn.close()
    except Exception as exc:  # the page works without the database; say why it is missing
        print(f"[service] database read failed ({type(exc).__name__})")
        return None


_lines: dict[str, dict[str, pd.DataFrame]] = {}


def ml_lines(gfs_init: str) -> dict[str, pd.DataFrame] | None:
    """The logged correction for this forecast cycle, per station: ml_c, lo_c, up_c, raw_c by valid time.

    Read, not recomputed: inference runs in the refresh workflow, so the line drawn is the
    line that is later scored. The lead-aware candidate is drawn when it has rows for this
    cycle, otherwise the incumbent.
    """

    def fetch() -> dict[str, pd.DataFrame] | None:
        try:
            init = pd.Timestamp(gfs_init).tz_localize("UTC").to_pydatetime()
        except (ValueError, TypeError):
            return None
        for version in ("xgb-v2", "xgb-v1"):
            rows = _query(
                """SELECT station_id, forecast_valid_time, corrected_value_c, pi_lo_c, pi_hi_c, raw_value_c
                   FROM ml_corrections WHERE gfs_init_time = %s AND model_version = %s
                   ORDER BY station_id, forecast_valid_time""",
                (init, version),
            )
            if rows:
                df = pd.DataFrame(rows, columns=["station_id", "time", "ml_c", "lo_c", "up_c", "raw_c"])
                df["time"] = pd.to_datetime(df["time"], utc=True)
                out = {str(sid): g.set_index("time")[["ml_c", "lo_c", "up_c", "raw_c"]].astype(float) for sid, g in df.groupby("station_id")}
                for g in out.values():
                    g.attrs["version"] = version
                return out
        return None

    # A cycle's rows do not change once logged, so a reply with rows is held for the life of the
    # process. An empty reply is asked for again after DB_TTL_S: the job may not have run yet.
    if gfs_init in _lines:
        return _lines[gfs_init]
    got: dict[str, pd.DataFrame] | None = live.cached(f"ml:{gfs_init}", DB_TTL_S, fetch)
    if got is not None:
        _lines.clear()  # one cycle at a time
        _lines[gfs_init] = got
    return got


def ml_verification(days: int = 14) -> dict[str, Any] | None:
    """The incumbent model scored against observations that arrived after each forecast:
    raw forecast, per-station offset and model on identical rows, over the last `days`."""

    def fetch() -> dict[str, Any] | None:
        rows = _query(
            """SELECT c.model_version, COUNT(*),
                      SQRT(AVG(POWER(c.raw_value_c       - p.observed_value_c, 2))),
                      SQRT(AVG(POWER(c.offset_baseline_c - p.observed_value_c, 2))),
                      SQRT(AVG(POWER(c.corrected_value_c - p.observed_value_c, 2))),
                      AVG(CASE WHEN c.pi_lo_c IS NULL THEN NULL
                               WHEN p.observed_value_c BETWEEN c.pi_lo_c AND c.pi_hi_c THEN 1.0 ELSE 0.0 END)
               FROM ml_corrections c
               JOIN forecast_obs_pairs p
                 ON p.station_id = c.station_id AND p.metric = 't2m' AND p.forecast_valid_time = c.forecast_valid_time
               WHERE c.forecast_valid_time >= NOW() - INTERVAL '1 day' * %s
                 AND c.offset_baseline_c IS NOT NULL AND c.model_version = 'xgb-v1'
               GROUP BY c.model_version LIMIT 1""",
            (days,),
        )
        if not rows or not rows[0][1]:
            return None
        version, n, raw, offset, ml, cover = rows[0]
        return {"version": version, "days": days, "n": int(n), "rmse_raw_c": float(raw), "rmse_offset_c": float(offset),
                "rmse_ml_c": float(ml), "band_coverage": None if cover is None else float(cover)}

    return live.cached(f"mlverif:{days}", DB_TTL_S, fetch)  # type: ignore[no-any-return]
