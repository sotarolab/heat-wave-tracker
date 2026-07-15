"""
src/heat/extremes.py
=====================
GEV (Generalized Extreme Value) fit to a station's annual maximum
temperature series, and return-period/return-level queries against it.
Pure math, no network calls, so it is testable without the historical
data archive. See requirements-dev.txt for why that split matters here.

Stationary fit only: this treats the full station record as one
unchanging distribution. A warming climate means that is a
simplification (a true non-stationary fit would let the location
parameter trend with year). That is a limitation worth surfacing to
users as a caveat, not hiding.

Gotchas:

1. scipy sign convention (the actual bug this file exists to get
   right). scipy.stats.genextreme's shape parameter c is the negative
   of the standard Coles (2001) xi convention used in the
   extreme-value-theory literature and by every return-period formula
   you will find outside scipy's own docs. xi > 0 means heavy tailed
   and unbounded above (Frechet type, genuinely unbounded extremes are
   possible). xi < 0 means bounded above (Weibull type, there is a
   hard ceiling). xi = 0 is Gumbel (exponential tail). This module
   reports xi (the literature convention) to callers, but always calls
   back into scipy's own cdf/ppf/sf with the original c, because those
   functions are defined in scipy's parameter space, not the
   literature's. Re-deriving the formulas by hand in the xi convention
   would risk reintroducing the same sign bug in a different place.
   Caught the hard way in a sibling project before this one existed,
   fixed here with tests that recover a known synthetic c and check
   its sign against xi directly (see tests/test_extremes.py).
2. A historical daily-max-temperature feed can contain corrupted values
   (one station's daily summary once reported 614F for a single day).
   A single bad year is enough to distort a GEV fit into an implausible
   unbounded-looking heavy tail. This module assumes its caller has
   already filtered obvious outliers before calling fit_gev (see
   src/heat/historical.py's filter_annual_max_outliers) rather than
   trying to guard against it here, since outlier rejection needs the
   full annual-maxima series, not a single point.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genextreme

GEV_MIN_YEARS = 20  # below this, MLE shape-parameter estimates are unreliable


def fit_gev(annual_maxima) -> dict | None:
    """Fit a stationary GEV to a station's annual maximum temperature series.

    Parameters
    ----------
    annual_maxima : array_like
        One value per year, same units throughout (this project always
        fits in degrees F). NaNs are dropped before fitting.

    Returns
    -------
    dict or None
        None if fewer than GEV_MIN_YEARS valid points are available.
        Otherwise a dict with:

        c : float
            scipy's shape parameter. Pass this back into
            genextreme.cdf/ppf/sf, not xi (see module Gotcha 1).
        xi : float
            The same shape parameter in the standard Coles (2001)
            convention, xi = -c, for human-readable reporting.
        loc : float
            Location parameter.
        scale : float
            Scale parameter.
        n_years : int
            How many annual maxima went into the fit.
    """
    data = np.asarray(annual_maxima, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < GEV_MIN_YEARS:
        return None

    c, loc, scale = genextreme.fit(data)
    return {"c": float(c), "xi": float(-c), "loc": float(loc),
            "scale": float(scale), "n_years": n}


def return_level(fit: dict, return_period_years: float) -> float:
    """The value expected to be equaled or exceeded once every return_period_years.

    Read off the fitted distribution's inverse CDF (PPF) at the
    corresponding annual exceedance probability. For example, the
    "100-year" high temperature.

    Parameters
    ----------
    fit : dict
        Output of fit_gev.
    return_period_years : float
        Return period in years. Must be greater than 1.

    Returns
    -------
    float
        The return level, same units fit_gev was called with.
    """
    p_annual_exceed = 1.0 / return_period_years
    return float(genextreme.ppf(1.0 - p_annual_exceed, fit["c"],
                                loc=fit["loc"], scale=fit["scale"]))


def return_period(fit: dict, value: float) -> float | None:
    """Roughly how rare a given annual-max value is, in years.

    For example, "about a 1-in-40-year high." Computed from the fitted
    distribution's survival function (1 - CDF).

    Parameters
    ----------
    fit : dict
        Output of fit_gev.
    value : float
        Value to evaluate, same units fit_gev was called with.

    Returns
    -------
    float or None
        None if value falls at or beyond the edge of the fitted
        distribution's support, where the estimate is undefined or
        numerically meaningless rather than just "very rare."
    """
    sf_val = float(genextreme.sf(value, fit["c"], loc=fit["loc"], scale=fit["scale"]))
    if sf_val <= 0:
        return None
    return 1.0 / sf_val


def support(fit: dict) -> tuple[float, float]:
    """The fitted distribution's valid range.

    May be +/-inf on one side, depending on the sign of the shape
    parameter (see module Gotcha 1). Callers evaluating the fit outside
    this range will get NaN or 0 back from scipy, matching scipy's own
    convention.

    Parameters
    ----------
    fit : dict
        Output of fit_gev.

    Returns
    -------
    tuple of float
        (lower, upper) bounds of the support.
    """
    return genextreme.support(fit["c"], loc=fit["loc"], scale=fit["scale"])


def sf(fit: dict, x) -> np.ndarray:
    """Annual exceedance probability P(X > x), the survival function (1 - CDF).

    Monotonically decreasing in x. Its reciprocal is exactly the return
    period in years, the same quantity return_period() computes for a
    single scalar, with the sf <= 0 guard applied there.

    Parameters
    ----------
    fit : dict
        Output of fit_gev.
    x : array_like
        Values to evaluate, same units fit_gev was called with.

    Returns
    -------
    array_like
        Exceedance probability at each x, same shape as x.
    """
    return genextreme.sf(x, fit["c"], loc=fit["loc"], scale=fit["scale"])


def plotting_positions(annual_maxima) -> tuple[np.ndarray, np.ndarray]:
    """Empirical (value, exceedance probability) points for the raw annual maxima.

    Uses the Gringorten plotting position, the standard choice for
    GEV-fitted data. Weibull's plotting position is more common
    generically, but Gringorten has less bias specifically for
    extreme-value distributions.

    Parameters
    ----------
    annual_maxima : array_like
        One value per year, same units throughout.

    Returns
    -------
    values : np.ndarray
        Annual maxima sorted descending.
    probs : np.ndarray
        Matching empirical exceedance probability for each value, same
        order. The largest observed year gets the smallest probability.
    """
    data = np.sort(np.asarray(annual_maxima, dtype=float))[::-1]
    n = len(data)
    ranks = np.arange(1, n + 1)
    p = (ranks - 0.44) / (n + 0.12)
    return data, p
