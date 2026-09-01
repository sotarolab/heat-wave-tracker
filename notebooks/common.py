"""
notebooks/common.py
====================
Shared helpers for the ML-correction notebooks (01-03). Deliberately a
single flat module, not a package: these notebooks are the pre-production
proving ground, and anything that survives them graduates into src/heat/
with tests. Keeping the shared surface small and in one file makes that
extraction an unbiased move instead of a copy-paste.

Conventions match the rest of the repo: degrees C everywhere, UTC
timestamps tz-aware, station metadata comes from src.heat.stations (the
same source the app uses, so the notebooks can't drift from production).

Security note: the Neon connection string is read from the environment
(NEON_DATABASE_URL, with a fallback that parses the gitignored ../.env).
It must never be printed, echoed into a notebook output cell, or written
into an artifact - the notebooks are intended to be committed publicly.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACTS = HERE / "artifacts"
PAIRS_SNAPSHOT = ARTIFACTS / "forecast_obs_pairs.parquet"


# ── data access ──────────────────────────────────────────────────────────

def _database_url() -> str:
    url = os.environ.get("NEON_DATABASE_URL")
    if url:
        return url
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("NEON_DATABASE_URL="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("NEON_DATABASE_URL not set and no ../.env found")


def load_pairs(refresh: bool = False) -> pd.DataFrame:
    """The full forecast_obs_pairs archive, snapshotted locally.

    First call pulls from Neon and writes notebooks/artifacts/
    forecast_obs_pairs.parquet; later calls read the snapshot so every
    notebook in a session sees the same frozen data (and re-running a
    notebook doesn't silently pick up rows that arrived mid-analysis).
    refresh=True re-pulls.
    """
    if PAIRS_SNAPSHOT.exists() and not refresh:
        return pd.read_parquet(PAIRS_SNAPSHOT)
    import psycopg2
    conn = psycopg2.connect(_database_url())
    try:
        df = pd.read_sql(
            """SELECT station_id, metric, forecast_valid_time,
                      forecast_value_c, observed_value_c, gfs_init_time
               FROM forecast_obs_pairs
               ORDER BY station_id, metric, forecast_valid_time""",
            conn,
        )
    finally:
        conn.close()
    ARTIFACTS.mkdir(exist_ok=True)
    df.to_parquet(PAIRS_SNAPSHOT)
    return df


def station_table() -> pd.DataFrame:
    """Station metadata from the same module the live app uses."""
    import sys
    sys.path.insert(0, str(REPO))
    from src.heat.stations import MAJOR_CONUS_STATIONS
    return pd.DataFrame(MAJOR_CONUS_STATIONS).set_index("id")


def wide_pairs(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pivot the long (station, metric, time) archive to one row per
    (station, valid_time), with forecast/observed columns per metric,
    local-time fields, and the t2m error target.

    Local hour is the *station's* hour: the diurnal error cycle is a
    solar phenomenon, and 18 UTC is afternoon in DC but morning in
    Seattle - pooling on UTC hour would smear the strongest signal in
    the dataset across time zones.
    """
    if df is None:
        df = load_pairs()
    w = df.pivot_table(
        index=["station_id", "forecast_valid_time"],
        columns="metric",
        values=["forecast_value_c", "observed_value_c"],
        aggfunc="first",
    )
    w.columns = [f"{'fcst' if a=='forecast_value_c' else 'obs'}_{b}" for a, b in w.columns]
    w = w.reset_index()

    init = (df[df.metric == "t2m"]
            .set_index(["station_id", "forecast_valid_time"])["gfs_init_time"])
    w = w.join(init, on=["station_id", "forecast_valid_time"])

    stns = station_table()
    w["lat"] = w.station_id.map(stns.lat)
    w["lon"] = w.station_id.map(stns.lon)
    tzs = w.station_id.map(stns.tz.fillna("America/New_York"))
    local = [t.tz_convert(z) for t, z in zip(w.forecast_valid_time, tzs)]
    w["local_hour"] = [t.hour + t.minute / 60 for t in local]
    w["local_date"] = [t.date() for t in local]
    w["doy"] = [t.timetuple().tm_yday for t in local]
    w["lead_h"] = (w.forecast_valid_time - w.gfs_init_time).dt.total_seconds() / 3600.0
    w["err_t2m"] = w.obs_t2m - w.fcst_t2m           # target: observed minus forecast
    w["dewpoint_dep"] = w.fcst_t2m - w.fcst_td2m    # dryness proxy, known at forecast time
    return w.dropna(subset=["err_t2m"]).reset_index(drop=True)


# ── the frozen split ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class TemporalSplit:
    """Named, frozen train/validation/test contract.

    Contiguous blocks with the test block at the END, for the same three
    reasons caudal's splits document: (1) forecast errors are strongly
    autocorrelated day-to-day, so random held-out days leak through their
    neighbors; (2) fit-on-history, deploy-forward is how the model will
    actually be used, so validating on the most recent past makes
    validation skill an unbiased predictor of deployment skill; (3) the
    adjacent block shares the evaluation period's weather regime.

    Boundaries are UTC dates on forecast_valid_time. Frozen: results
    quoted anywhere must come from V1; a new split is a new name.
    """
    name: str
    train_end: str       # inclusive
    valid_end: str       # inclusive; test is everything after

    def label(self, ts: pd.Series) -> pd.Series:
        d = ts.dt.tz_convert("UTC").dt.date.astype(str)
        return pd.Series(
            np.where(d <= self.train_end, "train",
                     np.where(d <= self.valid_end, "valid", "test")),
            index=ts.index, name="split",
        )


# 48 days of archive (2026-07-15 .. 2026-08-31) -> 33 / 7 / 8.
SPLIT_V1 = TemporalSplit("V1", train_end="2026-08-16", valid_end="2026-08-23")

WITHHELD_FRACTION = 0.2
SPLIT_SALT = "hwt-ml-v1"


def is_withheld(station_id: str) -> bool:
    """Deterministic ~20% station holdout for spatial generalization
    checks - same hash-based discipline as the DA design, so the set is
    stable across machines and reruns, but salted independently so the
    two experiments' holdout sets are not correlated."""
    h = hashlib.sha256(f"{SPLIT_SALT}:{station_id}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < WITHHELD_FRACTION


# ── baselines (all causal) ───────────────────────────────────────────────

def rmse(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def baseline_predictions(w: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Predicted err_t2m under each baseline, causally.

    raw            : predict 0 correction.
    global_offset  : train-block mean error, one number.
    station_offset : train-block mean error per station; stations absent
                     from the train block fall back to the global offset.
    same_day       : replica of the live app's scheme (src/heat/bias.py)
                     - mean error of the SAME station's EARLIER pairs on
                     the SAME local day, needing >= 3 of them, else 0.
                     Uses only information available at each row's valid
                     time, like the app at "now".
    """
    out = pd.DataFrame(index=w.index)
    tr = w[split == "train"]
    out["raw"] = 0.0
    out["global_offset"] = tr.err_t2m.mean()
    stn_mean = tr.groupby("station_id").err_t2m.mean()
    out["station_offset"] = w.station_id.map(stn_mean).fillna(tr.err_t2m.mean())

    # groupby cum* on a sorted frame returns series in SORTED row order;
    # build the column in that order with its index intact, then reindex
    # back to w's order - np.where on the raw values would pair sorted
    # positions with unsorted rows (caught live: it produced NaNs and a
    # silently misaligned baseline).
    ws = w.sort_values(["station_id", "local_date", "forecast_valid_time"])
    grp = ws.groupby(["station_id", "local_date"])
    prior_n = grp.cumcount()
    prior_mean = (grp.err_t2m.cumsum() - ws.err_t2m) / prior_n.replace(0, np.nan)
    same_day = pd.Series(np.where(prior_n >= 3, prior_mean, 0.0), index=ws.index)
    out["same_day"] = same_day.reindex(w.index)
    assert not out.same_day.isna().any()
    return out


def score_table(w: pd.DataFrame, split: pd.Series, preds: pd.DataFrame,
                block: str = "test") -> pd.DataFrame:
    """RMSE / MAE / bias of the residual (err - predicted correction) on
    one split block, per prediction column, sorted best-last."""
    m = (split == block).to_numpy()
    rows = []
    for col in preds.columns:
        resid = w.err_t2m[m] - preds[col][m]
        rows.append({"method": col, "rmse": rmse(resid),
                     "mae": float(resid.abs().mean()),
                     "bias": float(resid.mean()), "n": int(m.sum())})
    return (pd.DataFrame(rows).set_index("method")
            .sort_values("rmse", ascending=False).round(3))


# ── model features (shared by notebooks 03 and 04) ───────────────────────

def build_features(w: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for the learned correction. Everything here is
    known at forecast time: forecast values, station location, clock and
    calendar harmonics, lead, and a LAGGED station error history
    (trailing 7 completed local days, shifted so a row never sees its
    own day). Moved here from notebook 03 once notebook 04 needed the
    identical definition - one source of truth, per the graduation path
    in the module docstring."""
    X = pd.DataFrame(index=w.index)
    X["fcst_t2m"] = w.fcst_t2m
    X["dewpoint_dep"] = w.dewpoint_dep.fillna(w.dewpoint_dep.median())
    X["lat"], X["lon"] = w.lat, w.lon
    X["hour_sin"] = np.sin(2*np.pi*w.local_hour/24); X["hour_cos"] = np.cos(2*np.pi*w.local_hour/24)
    X["doy_sin"]  = np.sin(2*np.pi*w.doy/365.25);    X["doy_cos"]  = np.cos(2*np.pi*w.doy/365.25)
    X["lead_h"] = w.lead_h

    daily = (w.groupby(["station_id","local_date"]).err_t2m.mean()
               .groupby(level=0, group_keys=False)
               .apply(lambda s: s.rolling(7, min_periods=2).mean().shift(1)))
    X["stn_err_7d"] = pd.MultiIndex.from_frame(w[["station_id","local_date"]]).map(daily)
    X["stn_err_7d"] = X.stn_err_7d.fillna(0.0)
    return X


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km; accepts arrays, broadcasts."""
    p = np.pi / 180.0
    dlat, dlon = (lat2-lat1)*p, (lon2-lon1)*p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
