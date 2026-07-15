"""
src/heat/bias.py
=================
Same-day, same-station bias correction for GFS forecasts against real ASOS
observations. Not a fitted statistical model, just a simple empirical
offset (observed minus forecast, averaged over recent same-day pairs),
only trustworthy once a handful of paired points exist for today.

Statistically this is closer to error dressing of a deterministic
forecast (Roulston and Smith 2003) than to NGR/EMOS (Gneiting et al.
2005). NGR needs an ensemble (variance regressed against ensemble
spread) and coefficients fit by minimizing CRPS over a training period.
There is only one deterministic GFS run here and no training period, so
the predictive interval below leans on the residual spread of recent
same-day pairs instead, with a small-sample correction (see
pi95_halfwidth below) rather than an ensemble-spread relationship.

Split out from app.py so it can be unit tested without importing the full
Dash/GFS-fetching stack (cfgrib/eccodes/herbie-data). See
requirements-dev.txt for that split.

Gotchas:

1. pandas merge_asof requires both sides' datetime64 columns to share the
   same storage unit (e.g. nanoseconds vs microseconds). The GFS series
   (from xarray/netCDF4) and the ASOS series (freshly parsed from a CSV)
   are not guaranteed to agree, so _paired_today normalizes both to
   microseconds explicitly before merging, rather than relying on them
   happening to match.
"""
from __future__ import annotations

import math

import pandas as pd
from scipy.stats import t as _t_dist

BIAS_MIN_PAIRS = 3
BIAS_MATCH_TOLERANCE = pd.Timedelta(minutes=45)

# Separate, higher floor for the *interval*, not the point estimate. At
# n=BIAS_MIN_PAIRS=3 (2 degrees of freedom), t.ppf(0.975, 2) is about
# 4.30, more than double the Normal-distribution 1.96, and the sigma
# estimate itself is highly unstable with only 3 points - one point
# rotating in or out of the window can swing the whole band a lot.
# Confirmed live: at n=3 the combined multiplier (t-value * sqrt(1+1/n))
# is about 5x sigma, versus about 3x at n=5 - a real, visible difference
# in how wide and how volatile the band looks, not just a theoretical
# one. A bias-corrected point estimate from 3 points is still
# directionally useful even though imprecise, so that keeps the lower
# BIAS_MIN_PAIRS floor; the interval only renders once there is enough
# data for the width to mean something instead of just looking like
# noise.
PI_MIN_PAIRS = 5


def _paired_today(forecast_series: pd.Series, obs_local: pd.DataFrame,
                  obs_col: str, today, now_ts: pd.Timestamp) -> pd.DataFrame:
    """All of today's (forecast, observed) pairs so far, nearest-matched
    within BIAS_MATCH_TOLERANCE.

    Shared by today_forecast_bias and brier_score_exceedance so the two
    can never disagree on which points count as today's data.

    Parameters
    ----------
    forecast_series : pd.Series
        Forecast values indexed by local timestamp.
    obs_local : pd.DataFrame
        Observations with a "valid_local" timestamp column and an
        obs_col value column.
    obs_col : str
        Name of the observed-value column in obs_local.
    today : date
        Calendar date to restrict both sides to.
    now_ts : pd.Timestamp
        Only forecast steps at or before this instant are eligible.

    Returns
    -------
    pd.DataFrame
        Columns time, forecast, observed. Empty if either side has no
        data for today.
    """
    past = forecast_series[(forecast_series.index.date == today) & (forecast_series.index <= now_ts)]
    if past.empty or obs_local.empty:
        return pd.DataFrame(columns=["time", "forecast", "observed"])
    obs_today = obs_local[obs_local["valid_local"].dt.date == today].dropna(subset=[obs_col])
    if obs_today.empty:
        return pd.DataFrame(columns=["time", "forecast", "observed"])

    fdf = past.rename("forecast").reset_index().rename(columns={"index": "time"}).sort_values("time")
    odf = (obs_today[["valid_local", obs_col]]
           .rename(columns={"valid_local": "time", obs_col: "observed"})
           .sort_values("time"))
    # see module Gotcha 1: normalize datetime64 storage units before merge_asof
    fdf["time"] = fdf["time"].dt.as_unit("us")
    odf["time"] = odf["time"].dt.as_unit("us")

    return pd.merge_asof(fdf, odf, on="time", direction="nearest",
                         tolerance=BIAS_MATCH_TOLERANCE).dropna(subset=["observed"])


def _windowed(paired: pd.DataFrame, now_ts: pd.Timestamp,
             window_hours: float | None) -> pd.DataFrame:
    """Restrict paired points to a trailing window, with a trust floor.

    "Last N hours" means every pair whose forecast time falls in
    [now_ts - window_hours, now_ts]. Never drops below the trust floor:
    if the chosen window is narrower than the data supports, falls back
    to the most recent BIAS_MIN_PAIRS pairs regardless of window.

    Parameters
    ----------
    paired : pd.DataFrame
        Output of _paired_today.
    now_ts : pd.Timestamp
        End of the trailing window.
    window_hours : float or None
        Window length in hours. None returns paired unchanged.

    Returns
    -------
    pd.DataFrame
        Subset of paired, at least BIAS_MIN_PAIRS rows if paired has
        that many to begin with.
    """
    if window_hours is None:
        return paired
    cutoff = now_ts - pd.Timedelta(hours=window_hours)
    used = paired[paired["time"] >= cutoff]
    if len(used) < BIAS_MIN_PAIRS:
        used = paired.tail(BIAS_MIN_PAIRS)
    return used


def today_forecast_bias(forecast_series: pd.Series, obs_local: pd.DataFrame,
                        obs_col: str, today, now_ts: pd.Timestamp,
                        window_hours: float | None = None) -> dict | None:
    """Mean (observed - forecast) bias for one station/metric today.

    Uses only forecast steps that have already happened, paired with the
    nearest real observation within BIAS_MATCH_TOLERANCE.

    Parameters
    ----------
    forecast_series : pd.Series
        Forecast values indexed by local timestamp.
    obs_local : pd.DataFrame
        Observations with a "valid_local" timestamp column and an
        obs_col value column.
    obs_col : str
        Name of the observed-value column in obs_local.
    today : date
        Calendar date to restrict both sides to.
    now_ts : pd.Timestamp
        Only forecast steps at or before this instant are eligible.
    window_hours : float or None, optional
        Restrict to pairs within the trailing window_hours of now_ts
        instead of all of today. Lets a caller explore how the
        correction and its uncertainty respond to how recent the signal
        is (the interactive same-day window control). None uses all of
        today's pairs. An hours-based window rather than a point count,
        because it means the same thing regardless of forecast fetch
        cadence, whereas a point count silently changes meaning if the
        fetch interval ever changes.

    Returns
    -------
    dict or None
        None if fewer than BIAS_MIN_PAIRS pairs exist in total,
        regardless of window_hours. Otherwise a dict with:

        bias : float
            Mean (observed - forecast) error over the points used.
        sigma : float
            Residual standard deviation after removing the mean.
            Assumes roughly normally distributed errors, a placeholder
            worth caveats at low n.
        pi95_halfwidth : float
            Half-width of a 95% prediction interval (PI) for a new
            observation, not a confidence interval (CI): a CI is
            uncertainty about the estimated mean, a PI is where a new
            observation is expected to fall, and is necessarily wider.
            0.0 if n_used < PI_MIN_PAIRS, a separate, higher floor than
            BIAS_MIN_PAIRS (see its definition above) - below that, the
            t-distribution correction this needs makes the interval so
            wide and so unstable run to run that it reads as noise
            instead of signal. Otherwise
            sigma * t.ppf(0.975, n-1) * sqrt(1+1/n). Uses the
            t-distribution rather than a fixed 1.96 because sigma
            itself is estimated from a small sample, where the correct
            95% critical value is well above 1.96. The sqrt(1+1/n)
            factor accounts for uncertainty in the estimated mean
            itself, not just the spread of past residuals around it.
            Converges to the familiar 1.96*sigma as n grows.
        rmse : float
            Root-mean-square error over the same points, raw (not
            de-meaned like sigma), so it reflects the uncorrected
            forecast's actual miss distance, not just its spread.
        n_used : int
            How many pairs actually went into bias/sigma/rmse.
        n_available : int
            How many same-day pairs exist in total, for deciding
            whether an interactive window control has anything
            meaningful to offer yet.
    """
    paired = _paired_today(forecast_series, obs_local, obs_col, today, now_ts)
    n_available = len(paired)
    if n_available < BIAS_MIN_PAIRS:
        return None

    used = _windowed(paired, now_ts, window_hours)
    errors = used["observed"] - used["forecast"]
    n_used = len(errors)
    sigma = float(errors.std(ddof=1)) if n_used > 1 else 0.0
    pi95_halfwidth = (sigma * float(_t_dist.ppf(0.975, n_used - 1)) * (1.0 + 1.0 / n_used) ** 0.5
                      if n_used >= PI_MIN_PAIRS else 0.0)
    return {
        "bias":        float(errors.mean()),
        "sigma":       sigma,
        "pi95_halfwidth": pi95_halfwidth,
        "rmse":        float((errors ** 2).mean() ** 0.5),
        "n_used":      n_used,
        "n_available": n_available,
    }


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf, to avoid a scipy dependency for
    one line of math elsewhere in this module."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def brier_score_exceedance(forecast_series: pd.Series, obs_local: pd.DataFrame,
                           obs_col: str, today, now_ts: pd.Timestamp,
                           threshold_c: float,
                           window_hours: float | None = None) -> dict | None:
    """Brier score for today's paired points against a threshold-exceedance event.

    Scores the probability the app is implicitly showing the user (the
    bias-corrected forecast plus a Normal residual spread) against
    whether each point actually crossed threshold_c. For each pair,
    p_exceed = P(observed > threshold_c) under a Normal centered on
    (raw forecast + bias) with the day's residual sigma, then
    brier = mean((p_exceed - actually_exceeded)^2). 0 is perfect, 0.25
    is no better than a coin flip on a 50/50 event, 1 is worst possible.

    Caveat worth keeping in mind: the bias and sigma used to build each
    point's probability come from the same same-day sample being scored
    against (in-sample, not held out), same spirit as
    today_forecast_bias itself, which is explicitly not a fitted
    statistical model. This answers "was my stated confidence
    self-consistent with today's outcomes," not "how skillful is this
    forecast out-of-sample."

    Parameters
    ----------
    forecast_series : pd.Series
        Forecast values indexed by local timestamp.
    obs_local : pd.DataFrame
        Observations with a "valid_local" timestamp column and an
        obs_col value column.
    obs_col : str
        Name of the observed-value column in obs_local.
    today : date
        Calendar date to restrict both sides to.
    now_ts : pd.Timestamp
        Only forecast steps at or before this instant are eligible.
    threshold_c : float
        Exceedance threshold, same units as forecast_series and
        obs_local[obs_col].
    window_hours : float or None, optional
        Same meaning as in today_forecast_bias.

    Returns
    -------
    dict or None
        None if fewer than BIAS_MIN_PAIRS pairs exist, or if sigma is
        exactly 0 (perfect agreement so far, no spread to build a
        probability from). Otherwise a dict with:

        brier : float
            The Brier score, in [0, 1].
        n_used : int
            How many pairs went into the score.
    """
    paired = _paired_today(forecast_series, obs_local, obs_col, today, now_ts)
    if len(paired) < BIAS_MIN_PAIRS:
        return None

    used = _windowed(paired, now_ts, window_hours)
    errors = used["observed"] - used["forecast"]
    bias_c = float(errors.mean())
    sigma_c = float(errors.std(ddof=1)) if len(used) > 1 else 0.0
    if sigma_c <= 0:
        return None

    corrected = used["forecast"] + bias_c
    p_exceed = (threshold_c - corrected).apply(lambda d: 1.0 - _normal_cdf(d / sigma_c))
    actually_exceeded = (used["observed"] > threshold_c).astype(float)

    return {
        "brier":  float(((p_exceed - actually_exceeded) ** 2).mean()),
        "n_used": len(used),
    }
