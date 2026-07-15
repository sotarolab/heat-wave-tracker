"""
tests/test_bias.py
===================
Tests for src/heat/bias.py, the same-day forecast bias correction, its
interactive window solver, the Brier score, and the 95% predictive
interval. All synthetic, no network involved anywhere in this module.
"""
import pandas as pd
import pytest

from src.heat.bias import (today_forecast_bias, brier_score_exceedance,
                           BIAS_MIN_PAIRS, PI_MIN_PAIRS, BIAS_MATCH_TOLERANCE)

TZ = "America/New_York"


def _series(hours, values):
    """Hourly forecast series starting at local midnight + hours[0]."""
    base = pd.Timestamp("2026-07-14", tz=TZ)
    idx = pd.DatetimeIndex([base + pd.Timedelta(hours=h) for h in hours])
    return pd.Series(values, index=idx)


def _obs(hours, values, col="temp_c"):
    base = pd.Timestamp("2026-07-14", tz=TZ)
    return pd.DataFrame({
        "valid_local": [base + pd.Timedelta(hours=h) for h in hours],
        col: values,
    })


class TestTooFewPairs:
    def test_returns_none_below_min_pairs(self):
        """Only 2 forecast/obs pairs exist today, one short of
        BIAS_MIN_PAIRS."""
        forecast = _series([6, 7], [80.0, 81.0])
        obs = _obs([6, 7], [82.0, 83.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result is None

    def test_returns_none_with_no_observations_yet(self):
        """No observations at all today should behave the same as too
        few, not raise."""
        forecast = _series([6, 7, 8], [80.0, 81.0, 82.0])
        obs = pd.DataFrame(columns=["valid_local", "temp_c"])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result is None

    def test_returns_none_when_only_future_forecast_steps_exist(self):
        """now_ts is before every forecast step, so nothing has
        happened yet to pair against, regardless of how many
        observations exist."""
        forecast = _series([6, 7, 8], [80.0, 81.0, 82.0])
        obs = _obs([1, 2, 3, 4, 5], [70.0, 71.0, 72.0, 73.0, 74.0])
        now = pd.Timestamp("2026-07-14", tz=TZ) + pd.Timedelta(hours=5, minutes=30)
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result is None


class TestBiasAndSigmaMath:
    def test_bias_is_mean_observed_minus_forecast(self):
        """Observed consistently 2 degrees warmer than forecast at every
        point."""
        forecast = _series([6, 7, 8, 9], [80.0, 82.0, 84.0, 86.0])
        obs = _obs([6, 7, 8, 9], [82.0, 84.0, 86.0, 88.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result is not None
        assert result["bias"] == pytest.approx(2.0, abs=1e-9)

    def test_zero_bias_gives_zero_sigma_for_perfect_agreement(self):
        forecast = _series([6, 7, 8, 9], [80.0, 82.0, 84.0, 86.0])
        obs = _obs([6, 7, 8, 9], [80.0, 82.0, 84.0, 86.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result["bias"] == pytest.approx(0.0, abs=1e-9)
        assert result["sigma"] == pytest.approx(0.0, abs=1e-9)

    def test_sigma_matches_independent_stddev_of_residuals(self):
        """Errors +1, +3, +2, +4 give mean=2.5, sample std (ddof=1)
        computed independently here rather than trusting a copy of the
        same formula."""
        forecast = _series([6, 7, 8, 9], [80.0, 82.0, 84.0, 86.0])
        obs = _obs([6, 7, 8, 9], [81.0, 85.0, 86.0, 90.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        errors = pd.Series([1.0, 3.0, 2.0, 4.0])
        assert result["bias"] == pytest.approx(errors.mean(), abs=1e-9)
        assert result["sigma"] == pytest.approx(errors.std(ddof=1), abs=1e-9)

    def test_requesting_below_min_pairs_still_yields_finite_sigma(self):
        """n=1 gets floored to BIAS_MIN_PAIRS (see
        TestInteractiveWindowSolver), so this also exercises the ddof=1
        guard at the smallest reachable sample: std of a single-point
        sample is undefined (NaN) unless explicitly guarded, which
        would silently break the uncertainty band.
        """
        forecast = _series([6, 7, 8], [80.0, 81.0, 82.0])
        obs = _obs([6, 7, 8], [80.0, 81.0, 82.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now, window_hours=0.1)
        assert result["n_used"] == BIAS_MIN_PAIRS
        assert result["sigma"] == pytest.approx(0.0, abs=1e-9)


class TestInteractiveWindowSolver:
    def test_window_none_uses_all_available_pairs(self):
        forecast = _series(range(6, 12), [80.0] * 6)
        obs = _obs(range(6, 12), [82.0] * 6)
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now, window_hours=None)
        assert result["n_used"] == 6
        assert result["n_available"] == 6

    def test_window_selects_recent_pairs_not_earliest(self):
        """Early obs (hours 6-7, outside the trailing 2.5h window from
        now=hour 10, i.e. before cutoff=7.5) are +1 off. Recent obs
        (hours 8-10, inside the window) are +5 off. Bias should reflect
        only the windowed pairs, not a blend of all 5.
        """
        forecast = _series([6, 7, 8, 9, 10], [80.0] * 5)
        obs = _obs([6, 7, 8, 9, 10], [81.0, 81.0, 85.0, 85.0, 85.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now, window_hours=2.5)
        assert result["n_used"] == BIAS_MIN_PAIRS
        assert result["bias"] == pytest.approx(5.0, abs=1e-9)

    def test_n_available_reflects_total_regardless_of_window(self):
        forecast = _series(range(6, 14), [80.0] * 8)
        obs = _obs(range(6, 14), [81.0] * 8)
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now, window_hours=2)
        assert result["n_used"] == BIAS_MIN_PAIRS  # only ~2-3 pairs fall within 2h, floored to the trust min
        assert result["n_available"] == 8

    def test_window_narrower_than_trust_floor_falls_back_to_min_pairs(self):
        """A window so narrow it would otherwise select fewer than
        BIAS_MIN_PAIRS should not be able to push the sample below the
        trust floor. The interactive dropdown offers sane choices, but
        the function itself should not rely on that alone.
        """
        forecast = _series([6, 7, 8, 9, 10], [80.0] * 5)
        obs = _obs([6, 7, 8, 9, 10], [81.0] * 5)
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now, window_hours=0.1)
        assert result["n_used"] == BIAS_MIN_PAIRS

    def test_window_wider_than_available_uses_all_available(self):
        forecast = _series([6, 7, 8], [80.0, 81.0, 82.0])
        obs = _obs([6, 7, 8], [81.0, 82.0, 83.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now, window_hours=500)
        assert result["n_used"] == 3


class TestMatchTolerance:
    def test_obs_outside_tolerance_window_is_excluded(self):
        """A single forecast step at hour 8, but the only observation
        is well outside BIAS_MATCH_TOLERANCE (45 min), so it should not
        pair."""
        forecast = _series([8], [80.0])
        far_offset = BIAS_MATCH_TOLERANCE + pd.Timedelta(minutes=30)
        obs = pd.DataFrame({
            "valid_local": [forecast.index[0] + far_offset],
            "temp_c": [82.0],
        })
        now = forecast.index[0] + far_offset
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result is None

    def test_obs_just_inside_tolerance_window_is_included(self):
        forecast = _series([8], [80.0])
        near_offset = BIAS_MATCH_TOLERANCE - pd.Timedelta(minutes=1)
        # need BIAS_MIN_PAIRS total pairs, so add two more clean same-time pairs
        forecast = _series([6, 7, 8], [80.0, 80.0, 80.0])
        obs = pd.DataFrame({
            "valid_local": [
                forecast.index[0],
                forecast.index[1],
                forecast.index[2] + near_offset,
            ],
            "temp_c": [81.0, 81.0, 81.0],
        })
        now = forecast.index[-1] + near_offset
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result is not None
        assert result["n_available"] == 3


class TestOnlyTodaysObservationsCount:
    def test_yesterdays_observation_is_excluded_even_within_tolerance(self):
        """Same-day correction must not accidentally pair against a
        yesterday observation just because merge_asof finds it
        "nearest" - the obs_today filtering must exclude it before the
        merge happens, not rely on the merge itself to sort it out.
        """
        forecast = _series([0, 1, 6], [80.0, 80.0, 80.0])
        base = pd.Timestamp("2026-07-14", tz=TZ)
        obs = pd.DataFrame({
            "valid_local": [
                base - pd.Timedelta(minutes=10),  # yesterday 23:50
                base + pd.Timedelta(hours=1),
                base + pd.Timedelta(hours=6),
            ],
            "temp_c": [99.0, 81.0, 81.0],
        })
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result is None  # only 2 of the 3 obs are actually "today" - below BIAS_MIN_PAIRS


class TestRMSE:
    def test_rmse_matches_raw_root_mean_square_error(self):
        """Same fixture as test_sigma_matches_independent_stddev_of_residuals
        above, errors +1, +3, +2, +4."""
        forecast = _series([6, 7, 8, 9], [80.0, 82.0, 84.0, 86.0])
        obs = _obs([6, 7, 8, 9], [81.0, 85.0, 86.0, 90.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        errors = pd.Series([1.0, 3.0, 2.0, 4.0])
        expected_rmse = (errors ** 2).mean() ** 0.5
        assert result["rmse"] == pytest.approx(expected_rmse, abs=1e-9)

    def test_rmse_is_zero_for_perfect_agreement(self):
        forecast = _series([6, 7, 8, 9], [80.0, 82.0, 84.0, 86.0])
        obs = _obs([6, 7, 8, 9], [80.0, 82.0, 84.0, 86.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_rmse_reflects_raw_miss_distance_even_with_zero_mean_bias(self):
        """Errors +5/-5/0 average to zero bias, but the forecast is
        clearly missing by up to 5 degrees. RMSE, unlike bias, must
        reflect that, since it is not de-meaned like sigma is.
        """
        forecast = _series([6, 7, 8], [80.0, 80.0, 80.0])
        obs = _obs([6, 7, 8], [85.0, 75.0, 80.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now, window_hours=None)
        errors = pd.Series([5.0, -5.0, 0.0])
        assert result["bias"] == pytest.approx(0.0, abs=1e-9)
        assert result["rmse"] == pytest.approx((errors ** 2).mean() ** 0.5, abs=1e-9)


class TestBrierScoreExceedance:
    def test_none_below_min_pairs(self):
        forecast = _series([6, 7], [30.0, 30.0])
        obs = _obs([6, 7], [30.0, 30.0])
        now = forecast.index[-1]
        result = brier_score_exceedance(forecast, obs, "temp_c", now.date(), now, threshold_c=32.0)
        assert result is None

    def test_none_when_sigma_is_zero(self):
        """Perfect same-day agreement leaves no residual spread to
        build a probability from, so a Brier score cannot be formed."""
        forecast = _series([6, 7, 8], [30.0, 30.0, 30.0])
        obs = _obs([6, 7, 8], [30.0, 30.0, 30.0])
        now = forecast.index[-1]
        result = brier_score_exceedance(forecast, obs, "temp_c", now.date(), now, threshold_c=32.0)
        assert result is None

    def test_confident_correct_predictions_score_near_zero(self):
        """Forecast far below threshold with tiny scatter, so p_exceed
        should be near 0 for every point, matching
        actually_exceeded=0 everywhere."""
        forecast = _series([6, 7, 8, 9, 10], [20.0] * 5)
        obs = _obs([6, 7, 8, 9, 10], [20.1, 19.9, 20.2, 19.8, 20.0])
        now = forecast.index[-1]
        result = brier_score_exceedance(forecast, obs, "temp_c", now.date(), now, threshold_c=32.0)
        assert result is not None
        assert result["brier"] < 0.05

    def test_score_matches_hand_computed_value(self):
        """Reimplements the same normal-CDF-based scoring by hand here,
        independent of _normal_cdf in bias.py, so this is a real check
        against the formula, not a check that the code agrees with
        itself."""
        import math
        forecast = _series([6, 7, 8, 9], [30.0, 30.0, 30.0, 30.0])
        obs = _obs([6, 7, 8, 9], [30.0, 30.0, 30.0, 34.0])
        now = forecast.index[-1]
        result = brier_score_exceedance(forecast, obs, "temp_c", now.date(), now, threshold_c=32.0)
        assert result is not None

        errors = pd.Series([0.0, 0.0, 0.0, 4.0])
        bias, sigma = errors.mean(), errors.std(ddof=1)
        corrected = pd.Series([30.0, 30.0, 30.0, 30.0]) + bias

        def norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        p_exceed = (32.0 - corrected).apply(lambda d: 1.0 - norm_cdf(d / sigma))
        actually = pd.Series([0.0, 0.0, 0.0, 1.0])
        expected_brier = ((p_exceed - actually) ** 2).mean()
        assert result["brier"] == pytest.approx(expected_brier, abs=1e-9)
        assert result["n_used"] == 4


class TestPI95Halfwidth:
    def test_zero_when_sigma_is_zero(self):
        forecast = _series([6, 7, 8], [80.0, 80.0, 80.0])
        obs = _obs([6, 7, 8], [81.0, 81.0, 81.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result["sigma"] == pytest.approx(0.0, abs=1e-9)
        assert result["pi95_halfwidth"] == pytest.approx(0.0, abs=1e-9)

    def test_zero_below_pi_min_pairs_even_with_real_spread(self):
        """PI_MIN_PAIRS is a separate, higher floor than BIAS_MIN_PAIRS -
        below it, the interval is suppressed entirely (0.0) even though
        the bias/sigma point estimates are still computed and real,
        since at that few points the t-distribution correction makes
        the band both very wide and very unstable run to run.
        """
        forecast = _series([6, 7, 8], [80.0, 80.0, 80.0])
        obs = _obs([6, 7, 8], [79.0, 82.0, 81.0])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result["n_used"] == 3 < PI_MIN_PAIRS
        assert result["sigma"] > 0.0
        assert result["pi95_halfwidth"] == pytest.approx(0.0, abs=1e-9)

    def test_wider_than_naive_1_96_sigma_at_pi_min_pairs(self):
        """At n=PI_MIN_PAIRS, the correct 95% predictive multiplier is
        still well above the Normal-distribution 1.96. Using a fixed
        1.96 regardless of n understates uncertainty exactly when the
        sample is smallest.
        """
        hours = list(range(6, 6 + PI_MIN_PAIRS))
        values = [79.0, 82.0, 81.0, 83.0, 80.0][:PI_MIN_PAIRS]
        forecast = _series(hours, [80.0] * PI_MIN_PAIRS)
        obs = _obs(hours, values)
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result["n_used"] == PI_MIN_PAIRS
        naive_1_96 = 1.96 * result["sigma"]
        assert result["pi95_halfwidth"] > naive_1_96

    def test_matches_hand_computed_t_distribution_value(self):
        """Independent computation using scipy's t.ppf directly, rather
        than trusting a copy of the same formula inside bias.py."""
        import math
        from scipy.stats import t as t_dist
        hours = list(range(6, 6 + PI_MIN_PAIRS))
        values = [81.0, 83.0, 79.0, 82.0, 80.5][:PI_MIN_PAIRS]
        forecast = _series(hours, [80.0] * PI_MIN_PAIRS)
        obs = _obs(hours, values)
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        n = result["n_used"]
        expected = result["sigma"] * t_dist.ppf(0.975, n - 1) * math.sqrt(1.0 + 1.0 / n)
        assert result["pi95_halfwidth"] == pytest.approx(expected, rel=1e-9)

    def test_converges_toward_1_96_sigma_as_n_grows(self):
        """Not an exact equality, the t-distribution only converges to
        Normal in the limit, just a sanity check that the small-n
        correction shrinks toward the familiar constant as more points
        become available. Hours must stay within 0-23:
        today_forecast_bias only counts same-calendar-day pairs, so
        anything at or past hour 24 would roll into "tomorrow" and get
        silently excluded, undercounting n_used.
        """
        hours = range(0, 24)
        forecast = _series(hours, [80.0] * 24)
        obs = _obs(hours, [80.0 + (i % 3 - 1) for i in range(24)])
        now = forecast.index[-1]
        result = today_forecast_bias(forecast, obs, "temp_c", now.date(), now)
        assert result["n_used"] == 24
        naive_1_96 = 1.96 * result["sigma"]
        assert result["pi95_halfwidth"] == pytest.approx(naive_1_96, rel=0.15)
