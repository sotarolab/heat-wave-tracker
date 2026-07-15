"""
src/heat/compute.py
====================
Heat stress derived variables from 2m temperature and dewpoint.

Heat Index (HI): NWS Rothfusz regression (1990 SR 90-23).
Wet Bulb Temp (WBT): Stull (2011) empirical approximation.

Both functions accept numpy arrays (or anything array_like) and return
arrays of the same shape. Pure math, no I/O, so these are safe to unit
test without any of the network or GRIB-decoding dependencies the rest of
the project needs.

Gotchas:

1. NWS's real Rothfusz switching rule is not "RH >= 40%". That 40% mark
   is just the axis range printed on the popular public reference chart,
   not the actual computation trigger. The real rule averages the simple
   Steadman estimate with the actual temperature and escalates to the
   full Rothfusz regression once that average reaches 80F, almost
   independent of RH. A flat 40% RH threshold badly understates heat
   index at high temperature with moderate-but-under-40% humidity, where
   the regression should still apply. Caught live against a real ASOS
   reading (100F, 39.6% RH), which returned 102F under the flat-RH rule
   instead of the officially published value near 110F.
2. Heat index is undefined below 80F per NWS convention.
   heat_index_array returns the actual temperature unchanged below that
   point, not the Steadman simple formula. Steadman's simple formula was
   only ever meant as an internal stepping stone toward deciding whether
   to escalate to the full regression, not a value fit to show on its
   own below 80F.

References
----------
Rothfusz (1990). NWS Technical Attachment SR 90-23.
Stull R. (2011). J. Appl. Meteor. Climatol., 50(11), 2267-2269.
    DOI: 10.1175/JAMC-D-11-0143.1
"""
import numpy as np


def relative_humidity(t_c, td_c):
    """Relative humidity from temperature and dewpoint, via the Magnus formula.

    Parameters
    ----------
    t_c : array_like
        Air temperature, degrees C.
    td_c : array_like
        Dewpoint temperature, degrees C, same shape as t_c.

    Returns
    -------
    array_like
        Relative humidity in percent, same shape as the inputs.
    """
    a, b = 17.625, 243.04
    return 100.0 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))


def heat_index_array(t_c, td_c):
    """NWS Rothfusz heat index, vectorized.

    See module docstring Gotchas 1 and 2 for the two non-obvious rules
    this encodes: the real 80F averaged-temperature escalation trigger
    (not a flat RH threshold), and the below-80F passthrough.

    Parameters
    ----------
    t_c : array_like
        Air temperature, degrees C, any shape.
    td_c : array_like
        Dewpoint temperature, degrees C, same shape as t_c.

    Returns
    -------
    array_like
        Heat index, degrees C, same shape as the inputs.
    """
    t  = np.asarray(t_c,  dtype=float)
    td = np.asarray(td_c, dtype=float)
    rh = relative_humidity(t, td)
    tf = t * 9.0 / 5.0 + 32.0

    # below 80F, heat index is undefined per NWS - report actual temperature
    hi_f = tf.copy()

    # 80F+: at minimum use the Steadman simple approximation
    warm = tf >= 80.0
    simple = 0.5 * (tf + 61.0 + (tf - 68.0) * 1.2 + rh * 0.094)
    hi_f[warm] = simple[warm]

    # escalate to the full Rothfusz regression once the average of actual
    # temp and the simple estimate reaches 80F (see Gotcha 1, not RH >= 40%)
    escalate = warm & (((tf + simple) / 2.0) >= 80.0)
    tm, rm = tf[escalate], rh[escalate]
    hi_f[escalate] = (-42.379
                  + 2.04901523 * tm
                  + 10.14333127 * rm
                  - 0.22475541 * tm * rm
                  - 0.00683783 * tm**2
                  - 0.05481717 * rm**2
                  + 0.00122874 * tm**2 * rm
                  + 0.00085282 * tm * rm**2
                  - 0.00000199 * tm**2 * rm**2)

    # low-humidity adjustment (dry hot desert air): subtract from HI
    lmask = escalate & (rh < 13.0) & (tf >= 80.0) & (tf <= 112.0)
    if lmask.any():
        hi_f[lmask] -= ((13.0 - rh[lmask]) / 4.0
                        * np.sqrt((17.0 - np.abs(tf[lmask] - 95.0)) / 17.0))

    # high-humidity adjustment (muggy warm air): add to HI
    hmask = escalate & (rh > 85.0) & (tf >= 80.0) & (tf <= 87.0)
    if hmask.any():
        hi_f[hmask] += (rh[hmask] - 85.0) / 10.0 * (87.0 - tf[hmask]) / 5.0

    return (hi_f - 32.0) * 5.0 / 9.0


def wet_bulb_array(t_c, td_c):
    """Stull (2011) empirical wet bulb temperature approximation.

    Valid for RH in [5, 99] percent and T in [-20, 50] degrees C.
    Accuracy about +/-0.65C across the valid range, per Stull (2011).

    Parameters
    ----------
    t_c : array_like
        Air temperature, degrees C, any shape.
    td_c : array_like
        Dewpoint temperature, degrees C, same shape as t_c.

    Returns
    -------
    array_like
        Wet bulb temperature, degrees C, same shape as the inputs.
    """
    t  = np.asarray(t_c,  dtype=float)
    td = np.asarray(td_c, dtype=float)
    rh = relative_humidity(t, td)
    return (t  * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
            + np.arctan(t + rh)
            - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)
