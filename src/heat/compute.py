"""
src/heat/compute.py
===================
Heat stress derived variables from 2m temperature and dewpoint.

Heat Index (HI): NWS Rothfusz regression (1990 SR 90-23).
Wet Bulb Temp (WBT): Stull (2011) empirical approximation.

Both functions accept numpy arrays and return arrays of the same shape.

References
----------
Rothfusz (1990). NWS Technical Attachment SR 90-23.
Stull R. (2011). J. Appl. Meteor. Climatol., 50(11), 2267-2269.
    DOI: 10.1175/JAMC-D-11-0143.1
"""
import numpy as np


def relative_humidity(t_c, td_c):
    """Magnus formula: RH [%] from T and Td (both °C)."""
    a, b = 17.625, 243.04
    return 100.0 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))


def heat_index_array(t_c, td_c):
    """
    NWS Rothfusz heat index, vectorized.
    Inputs: T and Td in °C (any shape). Output: heat index in °C.

    Below the Rothfusz threshold (T < 80°F or RH < 40%), the Steadman
    simple linear approximation is used instead.
    """
    t  = np.asarray(t_c,  dtype=float)
    td = np.asarray(td_c, dtype=float)
    rh = relative_humidity(t, td)
    tf = t * 9.0 / 5.0 + 32.0

    # Steadman simple approximation (default, below threshold)
    hi_f = 0.5 * (tf + 61.0 + (tf - 68.0) * 1.2 + rh * 0.094)

    # Rothfusz full regression where T ≥ 80°F and RH ≥ 40%
    mask = (tf >= 80.0) & (rh >= 40.0)
    tm, rm = tf[mask], rh[mask]
    hi_f[mask] = (-42.379
                  + 2.04901523 * tm
                  + 10.14333127 * rm
                  - 0.22475541 * tm * rm
                  - 0.00683783 * tm**2
                  - 0.05481717 * rm**2
                  + 0.00122874 * tm**2 * rm
                  + 0.00085282 * tm * rm**2
                  - 0.00000199 * tm**2 * rm**2)

    # Low-humidity adjustment (dry hot desert air): subtract from HI
    lmask = mask & (rh < 13.0) & (tf >= 80.0) & (tf <= 112.0)
    if lmask.any():
        hi_f[lmask] -= ((13.0 - rh[lmask]) / 4.0
                        * np.sqrt((17.0 - np.abs(tf[lmask] - 95.0)) / 17.0))

    # High-humidity adjustment (muggy warm air): add to HI
    hmask = mask & (rh > 85.0) & (tf >= 80.0) & (tf <= 87.0)
    if hmask.any():
        hi_f[hmask] += (rh[hmask] - 85.0) / 10.0 * (87.0 - tf[hmask]) / 5.0

    return (hi_f - 32.0) * 5.0 / 9.0


def wet_bulb_array(t_c, td_c):
    """
    Stull (2011) empirical wet bulb temperature approximation.
    Valid for RH ∈ [5, 99]% and T ∈ [-20, 50]°C.
    Accuracy ≈ ±0.65°C across the valid range.

    Returns wet bulb temperature in °C.
    """
    t  = np.asarray(t_c,  dtype=float)
    td = np.asarray(td_c, dtype=float)
    rh = relative_humidity(t, td)
    return (t  * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
            + np.arctan(t + rh)
            - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)
