"""
tests/test_historical.py
=========================
Tests for src/heat/historical.py's pure functions. The network-fetching
functions (resolve_iem_station, fetch_daily_max_temp) are not covered
here, same reasoning as requirements-dev.txt's split for the
GFS-fetching stack. All synthetic, no network involved anywhere in this
module.
"""
import pandas as pd
import pytest

from src.heat.historical import filter_annual_max_outliers


def _series(values, start_year=1990):
    return pd.Series(values, index=range(start_year, start_year + len(values)))


class TestFilterAnnualMaxOutliers:
    def test_too_few_years_returned_unchanged(self):
        """Fewer than 3 years is not enough to compute a meaningful
        median/MAD, so nothing should be dropped, even a value as
        implausible as 200F."""
        s = _series([95.0, 200.0])
        result = filter_annual_max_outliers(s)
        assert list(result) == [95.0, 200.0]

    def test_drops_value_above_physical_bound(self):
        s = _series([95.0, 96.0, 94.0, 97.0, 614.0])
        result = filter_annual_max_outliers(s)
        assert 614.0 not in result.values
        assert len(result) == 4

    def test_drops_robust_outlier_below_physical_bound(self):
        """Bangor, Maine style case: every other year sits in a tight
        band, one year is far outside it but still under the 130F
        absolute bound, so only the robust z-score check catches it.
        """
        s = _series([93.0, 94.0, 92.0, 93.0, 95.0, 94.0, 118.0])
        result = filter_annual_max_outliers(s)
        assert 118.0 not in result.values
        assert len(result) == 6

    def test_keeps_genuinely_hot_but_consistent_climate(self):
        """Yuma, Arizona style real data: tightly clustered mid-110s to
        low-120s. Nothing here should be flagged as an outlier, since a
        hot climate is not the same thing as a corrupted reading."""
        s = _series([112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
                     118.0, 120.0, 121.0, 123.0])
        result = filter_annual_max_outliers(s)
        assert len(result) == len(s)

    def test_keeps_all_when_no_outliers_present(self):
        s = _series([90.0, 91.0, 89.0, 92.0, 90.5])
        result = filter_annual_max_outliers(s)
        assert len(result) == len(s)

    def test_result_index_preserved_for_kept_years(self):
        """The dropped year's index should disappear along with its
        value, not just get its value blanked out, so downstream code
        can trust len(result) as the real sample size."""
        s = _series([90.0, 91.0, 89.0, 614.0, 92.0], start_year=2000)
        result = filter_annual_max_outliers(s)
        assert 2003 not in result.index
        assert list(result.index) == [2000, 2001, 2002, 2004]
