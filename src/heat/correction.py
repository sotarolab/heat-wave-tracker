"""
src/heat/correction.py
=======================
Inference for the learned t2m forecast correction (issue #16). The model
and its evaluation are documented in docs/ml/EXPERIMENTS.md; this module
applies the trained artifact operationally.

Design constraints, in order of importance:

1. Causality. Every feature uses only information available at the
   forecast init time: forecast values, station constants, calendar and
   clock terms, and per-station error history restricted to COMPLETED
   UTC days strictly before the init date. tests/test_correction.py
   enforces this.
2. Scope. Corrections are produced only for leads at or below
   VALIDATED_LEAD_H (8 h), the range the training archive covers.
   Longer leads are not corrected rather than extrapolated; the
   measured failure of extrapolation is recorded in EXPERIMENTS.md.
3. Degradation. No database, no artifact, or no xgboost installs to a
   no-op (None return), never an exception that could break the refresh
   workflow or the app. Heavy imports (xgboost, psycopg2) are lazy so
   the requirements-dev CI can import and test the pure logic.
4. The artifact lives in the ml_models table, not in git. The archive
   it was trained on is private; the trained file stays with it.

Feature order must match the training definition exactly
(notebooks/common.build_features). A mismatch is a silent wrong answer,
which is why FEATURES is asserted against the artifact's stored feature
names at load time.
"""
from __future__ import annotations

import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MODEL_NAME = "t2m_correction"
MODEL_VERSION = "xgb-v1"
VALIDATED_LEAD_H = 8.0

FEATURES = ["fcst_t2m", "dewpoint_dep", "lat", "lon", "hour_sin", "hour_cos",
            "doy_sin", "doy_cos", "lead_h", "stn_err_7d"]

# Trailing window for the live per-station offset baseline logged next to
# each correction. 21 days: long enough to be stable, short enough to track
# regime drift; the choice only affects the comparison column, never the
# model input.
OFFSET_BASELINE_DAYS = 21


def _db_url() -> str | None:
    return os.environ.get("NEON_DATABASE_URL")


def load_model(db_url: str | None = None):
    """Fetch the artifact from ml_models and return a fitted regressor,
    or None if the database, row, or xgboost is unavailable."""
    db_url = db_url or _db_url()
    if not db_url:
        return None
    try:
        import tempfile
        import psycopg2
        import xgboost as xgb
        conn = psycopg2.connect(db_url, connect_timeout=10)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT artifact FROM ml_models WHERE name=%s AND version=%s",
                            (MODEL_NAME, MODEL_VERSION))
                row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            print(f"[correction] no artifact {MODEL_NAME}/{MODEL_VERSION} in ml_models")
            return None
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            f.write(bytes(row[0])); f.flush()
            model = xgb.XGBRegressor()
            model.load_model(f.name)
        stored = list(model.get_booster().feature_names or [])
        assert stored == FEATURES, f"feature mismatch: artifact {stored} vs code {FEATURES}"
        return model
    except Exception as exc:
        print(f"[correction] model unavailable ({exc})")
        return None


def error_history(init_time_utc: pd.Timestamp, db_url: str | None = None,
                  days: int = 7) -> tuple[dict[str, float], dict[str, float]]:
    """Per-station mean t2m error over (a) the trailing `days` completed
    UTC days before init_time_utc's date (the stn_err_7d feature), and
    (b) the trailing OFFSET_BASELINE_DAYS (the logged offset baseline).

    The cutoff is the start of init's UTC day, not init itself: the
    training feature was built from completed local days, and a
    same-day partial window would leak observations the deployed model
    cannot have at the moment users see the forecast.

    Returns ({}, {}) when the database is unavailable.
    """
    db_url = db_url or _db_url()
    if not db_url:
        return {}, {}
    day0 = pd.Timestamp(init_time_utc).tz_convert("UTC").normalize()
    try:
        import psycopg2
        conn = psycopg2.connect(db_url, connect_timeout=10)
        try:
            with conn.cursor() as cur:
                out = []
                for nd in (days, OFFSET_BASELINE_DAYS):
                    cur.execute(
                        """SELECT station_id,
                                  AVG(observed_value_c - forecast_value_c)
                           FROM forecast_obs_pairs
                           WHERE metric='t2m'
                             AND forecast_valid_time >= %s - INTERVAL '1 day' * %s
                             AND forecast_valid_time <  %s
                           GROUP BY station_id""",
                        (day0.to_pydatetime(), nd, day0.to_pydatetime()))
                    out.append({sid: float(v) for sid, v in cur.fetchall() if v is not None})
        finally:
            conn.close()
        return out[0], out[1]
    except Exception as exc:
        print(f"[correction] error history unavailable ({exc})")
        return {}, {}


def build_features(station: dict, valid_times_utc: pd.DatetimeIndex,
                   fcst_t2m_c: np.ndarray, fcst_td2m_c: np.ndarray,
                   init_time_utc: pd.Timestamp,
                   stn_err_7d: float) -> pd.DataFrame:
    """Feature matrix for one station's forecast steps. Pure function of
    its arguments: no I/O, so it is testable without network or model."""
    idx = pd.DatetimeIndex(valid_times_utc)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(ZoneInfo(station.get("tz", "America/New_York")))
    lead_h = (idx - init_time_utc).total_seconds() / 3600.0
    hour = local.hour + local.minute / 60.0
    doy = local.dayofyear
    dep = np.asarray(fcst_t2m_c, float) - np.asarray(fcst_td2m_c, float)
    return pd.DataFrame({
        "fcst_t2m": np.asarray(fcst_t2m_c, float),
        "dewpoint_dep": np.where(np.isnan(dep), 0.0, dep),
        "lat": station["lat"], "lon": station["lon"],
        "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
        "doy_sin": np.sin(2*np.pi*doy/365.25), "doy_cos": np.cos(2*np.pi*doy/365.25),
        "lead_h": lead_h,
        "stn_err_7d": stn_err_7d,
    })[FEATURES]


def validated_mask(valid_times_utc: pd.DatetimeIndex,
                   init_time_utc: pd.Timestamp) -> np.ndarray:
    idx = pd.DatetimeIndex(valid_times_utc)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    lead_h = (idx - init_time_utc).total_seconds() / 3600.0
    return (lead_h >= 0) & (lead_h <= VALIDATED_LEAD_H)


# ── calibrated prediction interval (issue #17) ───────────────────────────

INTERVAL_ALPHA = 0.05


def load_interval_models(db_url: str | None = None):
    """Both quantile artifacts plus the CQR margin calibrated in
    notebooks/05_interval_calibration.ipynb. Returns (q_lo, q_hi,
    margin_c) or None under the same degradation rules as load_model.

    The margin is stored in the artifacts' metadata and is tied to the
    point model version: a retrained model requires recalibration, so a
    version mismatch refuses to load rather than shipping a stale band.
    """
    db_url = db_url or _db_url()
    if not db_url:
        return None
    try:
        import json
        import tempfile
        import psycopg2
        import xgboost as xgb
        conn = psycopg2.connect(db_url, connect_timeout=10)
        try:
            with conn.cursor() as cur:
                cur.execute("""SELECT name, artifact, metadata FROM ml_models
                               WHERE name IN ('t2m_interval_q025','t2m_interval_q975')
                                 AND version=%s""", (MODEL_VERSION,))
                rows = {name: (art, meta) for name, art, meta in cur.fetchall()}
        finally:
            conn.close()
        if set(rows) != {"t2m_interval_q025", "t2m_interval_q975"}:
            print("[correction] interval artifacts incomplete - band disabled")
            return None
        models = {}
        margin = None
        for name, (art, meta) in rows.items():
            meta = meta if isinstance(meta, dict) else json.loads(meta)
            if meta.get("point_model_version") != MODEL_VERSION:
                print("[correction] interval calibrated for a different model version - band disabled")
                return None
            margin = float(meta["cqr_margin_c"])
            with tempfile.NamedTemporaryFile(suffix=".json") as f:
                f.write(bytes(art)); f.flush()
                m = xgb.XGBRegressor(); m.load_model(f.name)
            models[name] = m
        return models["t2m_interval_q025"], models["t2m_interval_q975"], margin
    except Exception as exc:
        print(f"[correction] interval models unavailable ({exc})")
        return None


def apply_margin(q_lo: np.ndarray, q_hi: np.ndarray, margin_c: float) -> tuple[np.ndarray, np.ndarray]:
    """CQR band assembly: widen both raw quantile predictions by the
    calibrated margin. Split out as a pure function so the ordering
    guarantee (lo <= hi after widening, given lo <= hi before) is
    testable without artifacts."""
    lo = np.asarray(q_lo, float) - margin_c
    hi = np.asarray(q_hi, float) + margin_c
    return np.minimum(lo, hi), np.maximum(lo, hi)
