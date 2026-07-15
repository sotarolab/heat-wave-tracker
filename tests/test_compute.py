"""
tests/test_compute.py
======================
Regression tests for src/heat/compute.py.

These exist because heat_index_array() has already had two real bugs
caught by manual QC against the NWS spec: it returned a below-spec value
for temperatures under 80F, and used an approximate (not the official)
condition for switching to the full Rothfusz regression. Both were fixed
by inspection, not by a test - these tests exist so a future change can't
silently reintroduce them. All synthetic, no network involved anywhere
in this module.
"""
import numpy as np
import pytest

from src.heat.compute import relative_humidity, heat_index_array, wet_bulb_array


def f_to_c(f):
    return (f - 32.0) * 5.0 / 9.0


def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


def rothfusz_f(tf, rh):
    """Independent re-implementation of the Rothfusz regression, used to
    cross-check the coefficients in compute.py rather than trust a
    copy-pasted duplicate of the same code."""
    return (-42.379
            + 2.04901523 * tf
            + 10.14333127 * rh
            - 0.22475541 * tf * rh
            - 0.00683783 * tf**2
            - 0.05481717 * rh**2
            + 0.00122874 * tf**2 * rh
            + 0.00085282 * tf * rh**2
            - 0.00000199 * tf**2 * rh**2)


class TestRelativeHumidity:
    def test_saturation_is_100_percent(self):
        """Dewpoint equal to temperature means the air is saturated, by
        definition. Basic sanity check on the Magnus formula."""
        t = np.array([10.0, 25.0, 35.0])
        rh = relative_humidity(t, t)
        np.testing.assert_allclose(rh, 100.0, atol=1e-9)

    def test_drier_air_has_lower_dewpoint_and_lower_rh(self):
        """Direction check, not a magnitude check - a wider T-Td spread
        should always mean lower RH."""
        t = np.array([30.0, 30.0])
        td = np.array([25.0, 15.0])  # second point much drier
        rh = relative_humidity(t, td)
        assert rh[1] < rh[0]


class TestHeatIndexBelow80F:
    def test_cool_temperature_returns_actual_temp_unchanged(self):
        """NWS: heat index is undefined below 80F. This was the first
        real bug found here - the Steadman formula used to be applied
        regardless of temperature, producing a "feels like" colder than
        the actual temperature.
        """
        t_c = np.array([f_to_c(56.0), f_to_c(70.0), f_to_c(79.9)])
        td_c = np.array([f_to_c(50.0), f_to_c(60.0), f_to_c(70.0)])
        hi_c = heat_index_array(t_c, td_c)
        np.testing.assert_allclose(hi_c, t_c, atol=1e-6)

    def test_freezing_and_subzero_also_pass_through(self):
        """Same passthrough rule should hold arbitrarily far below 80F,
        not just just-under-80F."""
        t_c = np.array([0.0, -10.0])
        td_c = np.array([-2.0, -15.0])
        hi_c = heat_index_array(t_c, td_c)
        np.testing.assert_allclose(hi_c, t_c, atol=1e-6)


class TestHeatIndexRothfuszRegion:
    def test_matches_independent_rothfusz_computation(self):
        """Hot and humid enough to guarantee the full regression applies
        (T >= 80F and RH >= 40%), away from the low/high-humidity
        adjustment bands, so this isolates just the core regression
        against an independently re-implemented version of the same
        formula (see rothfusz_f above).
        """
        t_c, td_c = f_to_c(96.0), f_to_c(75.0)
        rh = relative_humidity(np.array([t_c]), np.array([td_c]))[0]
        assert rh >= 40.0
        expected_f = rothfusz_f(96.0, rh)
        actual_f = c_to_f(heat_index_array(np.array([t_c]), np.array([td_c]))[0])
        assert actual_f == pytest.approx(expected_f, abs=1e-6)

    def test_heat_index_meets_or_exceeds_actual_temp_when_humid(self):
        """Physical sanity check: once in Rothfusz territory with real
        humidity, "feels like" should never read cooler than the actual
        temperature."""
        t_c, td_c = f_to_c(95.0), f_to_c(78.0)
        hi_c = heat_index_array(np.array([t_c]), np.array([td_c]))[0]
        assert hi_c >= t_c

    def test_more_humidity_at_fixed_temperature_increases_heat_index(self):
        """Direction check at fixed temperature: more moisture in the
        air should never lower the heat index."""
        t_c = np.array([f_to_c(92.0), f_to_c(92.0)])
        td_c = np.array([f_to_c(65.0), f_to_c(80.0)])  # second is more humid
        hi_c = heat_index_array(t_c, td_c)
        assert hi_c[1] > hi_c[0]


class TestHeatIndexAt80FBoundary:
    def test_near_80F_and_dry_uses_steadman_not_rothfusz(self):
        """NWS's real switching rule averages the simple Steadman
        estimate with actual temperature, and escalates to the full
        regression only once that average reaches 80F. Right at the
        80F floor with very low humidity, the simple estimate is low
        enough that the average stays under 80F, so this case should
        use Steadman, not Rothfusz.
        """
        tf, rh_target = 80.5, 0.0
        t_c = f_to_c(tf)
        for td_f in np.arange(-40.0, 40.0, 0.5):
            rh = relative_humidity(np.array([t_c]), np.array([f_to_c(td_f)]))[0]
            if abs(rh - rh_target) < 1.0:
                break
        td_c = f_to_c(td_f)
        rh = relative_humidity(np.array([t_c]), np.array([td_c]))[0]

        simple = 0.5 * (tf + 61.0 + (tf - 68.0) * 1.2 + rh * 0.094)
        assert (tf + simple) / 2.0 < 80.0  # confirms this case does not escalate

        actual_f = c_to_f(heat_index_array(np.array([t_c]), np.array([td_c]))[0])
        assert actual_f == pytest.approx(simple, abs=1e-6)

    def test_hot_temperature_escalates_even_under_40_percent_rh(self):
        """Regression test for the second real bug found here: a flat
        "RH >= 40%" switching condition, instead of the official
        average-based one, understated heat index at high temperature
        with moderate-but-under-40% humidity. Caught via a real ASOS
        reading (100F, 39.6% RH, 71F dewpoint) that returned 102F under
        the flat-RH rule instead of the officially published value near
        110F.
        """
        tf = 100.0
        t_c = f_to_c(tf)
        td_c = f_to_c(71.0)
        rh = relative_humidity(np.array([t_c]), np.array([td_c]))[0]
        assert rh < 40.0  # exactly the case the old flat threshold missed

        expected_f = rothfusz_f(tf, rh)
        actual_f = c_to_f(heat_index_array(np.array([t_c]), np.array([td_c]))[0])
        assert actual_f == pytest.approx(expected_f, abs=1e-6)
        assert actual_f > 105.0  # sanity floor, nowhere near the old 102F result


class TestHeatIndexShapeAndTypes:
    def test_preserves_input_shape(self):
        """heat_index_array is meant to work directly on the (time, lat,
        lon) grids the GFS pipeline hands it, not just 1D series."""
        t_c = np.full((3, 4), 30.0)
        td_c = np.full((3, 4), 22.0)
        hi_c = heat_index_array(t_c, td_c)
        assert hi_c.shape == (3, 4)

    def test_accepts_plain_lists(self):
        """Callers should not have to remember to wrap inputs in
        np.asarray themselves."""
        hi_c = heat_index_array([30.0, 35.0], [20.0, 26.0])
        assert len(hi_c) == 2

    def test_nan_input_propagates_as_nan(self):
        """Missing data should stay missing, not silently become some
        other number."""
        hi_c = heat_index_array(np.array([np.nan]), np.array([20.0]))
        assert np.isnan(hi_c[0])


class TestWetBulb:
    def test_saturation_wet_bulb_equals_dry_bulb(self):
        """At 100% RH, wet bulb temperature equals actual temperature,
        by definition."""
        t_c = np.array([20.0, 30.0])
        wbt_c = wet_bulb_array(t_c, t_c)
        np.testing.assert_allclose(wbt_c, t_c, atol=0.7)  # Stull's own stated accuracy

    def test_wet_bulb_never_exceeds_actual_temperature(self):
        """Physical sanity check: wet bulb temperature cannot be warmer
        than the actual air temperature."""
        t_c = np.array([30.0, 35.0, 25.0])
        td_c = np.array([15.0, 20.0, 10.0])
        wbt_c = wet_bulb_array(t_c, td_c)
        assert np.all(wbt_c <= t_c + 0.1)  # small tolerance for approximation error
