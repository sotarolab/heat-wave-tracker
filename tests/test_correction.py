"""
tests/test_correction.py
=========================
Pure-logic tests for src/heat/correction.py: feature construction,
causality boundaries, lead scoping, and degradation. No network, no
xgboost, no database - runnable under requirements-dev.txt, same as the
rest of the suite.
"""
import numpy as np
import pandas as pd
import pytest

from src.heat import correction

STN = {"id": "KTST", "lat": 38.85, "lon": -77.04, "tz": "America/New_York"}


def _times(init, hours):
    return pd.DatetimeIndex([init + pd.Timedelta(hours=h) for h in hours])


INIT = pd.Timestamp("2026-09-01 12:00", tz="UTC")


class TestBuildFeatures:
    def test_columns_match_training_contract(self):
        X = correction.build_features(STN, _times(INIT, [0, 2]), np.array([30.0, 31.0]),
                                      np.array([20.0, 20.0]), INIT, 0.5)
        assert list(X.columns) == correction.FEATURES

    def test_lead_hours(self):
        X = correction.build_features(STN, _times(INIT, [0, 2, 8]),
                                      np.zeros(3), np.zeros(3), INIT, 0.0)
        assert list(X.lead_h) == [0.0, 2.0, 8.0]

    def test_local_hour_uses_station_timezone(self):
        # 12 UTC on Sep 1 is 08:00 EDT: hour_cos must reflect 8, not 12.
        X = correction.build_features(STN, _times(INIT, [0]), np.array([30.0]),
                                      np.array([20.0]), INIT, 0.0)
        assert X.hour_cos.iloc[0] == pytest.approx(np.cos(2*np.pi*8/24))

    def test_missing_dewpoint_becomes_zero_depression(self):
        X = correction.build_features(STN, _times(INIT, [0]), np.array([30.0]),
                                      np.array([np.nan]), INIT, 0.0)
        assert X.dewpoint_dep.iloc[0] == 0.0

    def test_history_scalar_broadcasts(self):
        X = correction.build_features(STN, _times(INIT, [0, 2]), np.zeros(2),
                                      np.zeros(2), INIT, -1.25)
        assert (X.stn_err_7d == -1.25).all()

    def test_pure_no_io(self):
        # Must succeed with no database URL in the environment at all.
        import os
        old = os.environ.pop("NEON_DATABASE_URL", None)
        try:
            correction.build_features(STN, _times(INIT, [0]), np.array([1.0]),
                                      np.array([0.0]), INIT, 0.0)
        finally:
            if old is not None:
                os.environ["NEON_DATABASE_URL"] = old


class TestValidatedMask:
    def test_scope_boundary_inclusive(self):
        m = correction.validated_mask(_times(INIT, [0, 8, 8.001, 10]), INIT)
        assert list(m) == [True, True, False, False]

    def test_negative_leads_excluded(self):
        # A stale file whose steps predate init must not be corrected.
        m = correction.validated_mask(_times(INIT, [-2, 0]), INIT)
        assert list(m) == [False, True]

    def test_naive_times_treated_as_utc(self):
        naive = pd.DatetimeIndex([pd.Timestamp("2026-09-01 14:00")])
        assert correction.validated_mask(naive, INIT)[0]


class TestDegradation:
    def test_load_model_without_db_url(self, monkeypatch):
        monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
        assert correction.load_model() is None

    def test_error_history_without_db_url(self, monkeypatch):
        monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
        assert correction.error_history(INIT) == ({}, {})

    def test_error_history_bad_url_degrades(self, monkeypatch):
        # An unreachable database must degrade to empty, not raise.
        monkeypatch.setenv("NEON_DATABASE_URL",
                           "postgresql://u:p@127.0.0.1:59999/none?connect_timeout=1")
        h7, off = correction.error_history(INIT)
        assert h7 == {} and off == {}


class TestApplyMargin:
    def test_widens_symmetrically(self):
        lo, hi = correction.apply_margin(np.array([-1.0]), np.array([1.0]), 0.5)
        assert lo[0] == -1.5 and hi[0] == 1.5

    def test_ordering_preserved_even_if_quantiles_cross(self):
        # Quantile regressors can cross on rare inputs; the band must
        # still come back ordered rather than inverted.
        lo, hi = correction.apply_margin(np.array([0.4]), np.array([0.1]), 0.05)
        assert lo[0] <= hi[0]

    def test_zero_margin_identity(self):
        lo, hi = correction.apply_margin(np.array([-2.0, 0.0]), np.array([1.0, 3.0]), 0.0)
        assert list(lo) == [-2.0, 0.0] and list(hi) == [1.0, 3.0]


class TestIntervalDegradation:
    def test_load_interval_without_db_url(self, monkeypatch):
        monkeypatch.delenv("NEON_DATABASE_URL", raising=False)
        assert correction.load_interval_models() is None


class TestMarginForLead:
    def test_scalar_passthrough(self):
        assert correction.margin_for_lead(1.5, 30.0) == 1.5

    def test_bin_lookup(self):
        m = {"0-8": 0.2, "8-24": 0.5, "24-120": 1.0}
        assert correction.margin_for_lead(m, 0.0) == 0.2
        assert correction.margin_for_lead(m, 8.0) == 0.2
        assert correction.margin_for_lead(m, 8.1) == 0.5
        assert correction.margin_for_lead(m, 200.0) == 1.0  # beyond last bin
