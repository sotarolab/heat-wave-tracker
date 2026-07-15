"""
tests/test_extremes.py
=======================
Tests for src/heat/extremes.py, the GEV fit and return-period/return-
level queries. The one bug this file exists to catch: scipy's shape
parameter sign convention (c = -xi), which bit this project once already
(in the sibling yaghan repo, unfixed there) and must not recur here. All
synthetic, no network involved anywhere in this module.
"""
import numpy as np
import pytest
from scipy.stats import genextreme

from src.heat.extremes import (fit_gev, return_level, return_period, sf,
                               plotting_positions, GEV_MIN_YEARS)


class TestMinimumSampleSize:
    def test_returns_none_below_min_years(self):
        data = np.full(GEV_MIN_YEARS - 1, 90.0)
        assert fit_gev(data) is None

    def test_fits_at_exactly_min_years(self):
        rng = np.random.default_rng(0)
        data = genextreme.rvs(0.1, loc=95.0, scale=5.0, size=GEV_MIN_YEARS,
                              random_state=rng)
        result = fit_gev(data)
        assert result is not None
        assert result["n_years"] == GEV_MIN_YEARS

    def test_nan_values_are_dropped_not_counted(self):
        """NaNs should not count toward the minimum-years threshold in
        either direction, they should just be excluded from n_years."""
        rng = np.random.default_rng(1)
        data = genextreme.rvs(0.1, loc=95.0, scale=5.0, size=GEV_MIN_YEARS,
                              random_state=rng)
        data_with_nans = np.concatenate([data, [np.nan, np.nan]])
        result = fit_gev(data_with_nans)
        assert result["n_years"] == GEV_MIN_YEARS


class TestSignConvention:
    """The actual bug-prevention tests. fit_gev must recover the known
    generating parameters of synthetic data, in both scipy's own c
    space and the literature's xi = -c space. A sign error would flip
    xi's sign relative to c on every non-symmetric case, which these
    would catch immediately."""

    def test_recovers_known_positive_shape_frechet_type(self):
        """c > 0 in scipy's convention should map to xi < 0 in the
        Coles convention."""
        rng = np.random.default_rng(42)
        true_c = 0.35
        data = genextreme.rvs(true_c, loc=100.0, scale=8.0, size=8000,
                              random_state=rng)
        result = fit_gev(data)
        assert result is not None
        assert result["c"] == pytest.approx(true_c, abs=0.05)
        assert result["xi"] == pytest.approx(-true_c, abs=0.05)
        assert result["xi"] < 0

    def test_recovers_known_negative_shape_weibull_type(self):
        """c < 0 in scipy's convention should map to xi > 0 in the
        Coles convention."""
        rng = np.random.default_rng(43)
        true_c = -0.25
        data = genextreme.rvs(true_c, loc=100.0, scale=8.0, size=8000,
                              random_state=rng)
        result = fit_gev(data)
        assert result is not None
        assert result["c"] == pytest.approx(true_c, abs=0.05)
        assert result["xi"] == pytest.approx(-true_c, abs=0.05)
        assert result["xi"] > 0

    def test_xi_is_always_exactly_negative_c(self):
        rng = np.random.default_rng(44)
        data = genextreme.rvs(0.15, loc=90.0, scale=6.0, size=100, random_state=rng)
        result = fit_gev(data)
        assert result["xi"] == pytest.approx(-result["c"], abs=1e-12)


class TestReturnLevelAndPeriod:
    def test_return_level_increases_with_return_period(self):
        rng = np.random.default_rng(7)
        data = genextreme.rvs(0.1, loc=100.0, scale=5.0, size=1000, random_state=rng)
        fit = fit_gev(data)
        level_10 = return_level(fit, 10)
        level_50 = return_level(fit, 50)
        level_100 = return_level(fit, 100)
        assert level_10 < level_50 < level_100

    def test_return_period_and_return_level_are_inverse(self):
        rng = np.random.default_rng(8)
        data = genextreme.rvs(0.1, loc=100.0, scale=5.0, size=1000, random_state=rng)
        fit = fit_gev(data)
        level_40 = return_level(fit, 40)
        recovered_period = return_period(fit, level_40)
        assert recovered_period == pytest.approx(40.0, rel=0.02)

    def test_more_extreme_value_has_longer_return_period(self):
        rng = np.random.default_rng(9)
        data = genextreme.rvs(0.1, loc=100.0, scale=5.0, size=1000, random_state=rng)
        fit = fit_gev(data)
        period_moderate = return_period(fit, 105.0)
        period_extreme = return_period(fit, 115.0)
        assert period_moderate is not None and period_extreme is not None
        assert period_extreme > period_moderate

    def test_return_period_none_beyond_distribution_support(self):
        """Bounded (Weibull type in the Coles sense, c > 0 in scipy's)
        has a hard upper limit. Querying well past it must return
        None, not a bogus finite number from an out-of-support
        survival function. scipy.genextreme.support() is used directly
        here rather than hand-deriving the boundary formula, to avoid
        the risk of a second, independent sign mistake in the test
        itself, which a first draft of this test made by expecting
        c < 0 to be the bounded-above case when it is actually c > 0.
        """
        rng = np.random.default_rng(10)
        data = genextreme.rvs(0.3, loc=100.0, scale=5.0, size=1000, random_state=rng)
        fit = fit_gev(data)
        assert fit["c"] > 0
        _, upper_bound = genextreme.support(fit["c"], loc=fit["loc"], scale=fit["scale"])
        assert upper_bound < np.inf
        assert return_period(fit, upper_bound + 50.0) is None


class TestSurvivalFunction:
    def test_sf_decreases_monotonically(self):
        rng = np.random.default_rng(11)
        data = genextreme.rvs(0.1, loc=100.0, scale=5.0, size=1000, random_state=rng)
        fit = fit_gev(data)
        x = np.array([90.0, 100.0, 110.0, 120.0])
        p = sf(fit, x)
        assert np.all(np.diff(p) < 0)

    def test_sf_reciprocal_matches_return_period(self):
        """sf and return_period should agree on the same quantity by
        construction, checked here explicitly rather than assumed."""
        rng = np.random.default_rng(12)
        data = genextreme.rvs(0.1, loc=100.0, scale=5.0, size=1000, random_state=rng)
        fit = fit_gev(data)
        value = 108.0
        p = float(sf(fit, value))
        period = return_period(fit, value)
        assert period == pytest.approx(1.0 / p, rel=1e-9)


class TestPlottingPositions:
    def test_sorted_descending_by_value(self):
        data = [90.0, 105.0, 95.0, 100.0]
        values, probs = plotting_positions(data)
        assert list(values) == sorted(data, reverse=True)

    def test_probabilities_between_zero_and_one_and_increasing_with_rank(self):
        """The largest value (rank 1) should get the smallest
        exceedance probability, and probability should rise
        monotonically as value falls."""
        data = [90.0, 105.0, 95.0, 100.0, 98.0]
        values, probs = plotting_positions(data)
        assert np.all((probs > 0) & (probs < 1))
        assert np.all(np.diff(probs) > 0)

    def test_length_matches_input(self):
        data = np.arange(90.0, 90.0 + 30)
        values, probs = plotting_positions(data)
        assert len(values) == 30
        assert len(probs) == 30
