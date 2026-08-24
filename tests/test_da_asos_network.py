"""Unit tests for src/heat/da/asos_network.py — QC and analysis-time selection.

Synthetic in-memory frames only; the network-touching fetchers are
exercised by their driver script, not here (repo convention: unit tests
never hit the network).
"""
import pandas as pd
import pytest

from src.heat.da.asos_network import qc_obs, select_analysis_obs

T0 = pd.Timestamp("2026-08-24T12:00:00Z")


def _obs(rows):
    return pd.DataFrame(rows, columns=["station_id", "valid_utc", "temp_c", "dewpoint_c"])


def test_qc_passes_clean_obs_untouched():
    obs = _obs([("KDCA", T0, 31.0, 22.0), ("KRDU", T0, 29.5, 21.0)])
    passed, rejects = qc_obs(obs)
    assert len(passed) == 2
    assert all(v == 0 for v in rejects.values())


def test_qc_drops_missing_and_out_of_bounds_temp():
    obs = _obs([
        ("KAAA", T0, None, 20.0),     # missing temp -> dropped
        ("KBBB", T0, 75.0, 20.0),     # impossible temp -> dropped
        ("KCCC", T0, -70.0, None),    # impossible temp -> dropped
        ("KDDD", T0, 30.0, 20.0),     # clean
    ])
    passed, rejects = qc_obs(obs)
    assert list(passed["station_id"]) == ["KDDD"]
    assert rejects["missing_temp"] == 1
    assert rejects["temp_bounds"] == 2


def test_qc_nulls_bad_dewpoint_but_keeps_temp():
    obs = _obs([
        ("KAAA", T0, 30.0, 90.0),   # dewpoint out of bounds -> nulled
        ("KBBB", T0, 20.0, 25.0),   # supersaturated -> nulled
        ("KCCC", T0, 20.0, 20.4),   # within rounding slack -> kept
    ])
    passed, rejects = qc_obs(obs)
    assert len(passed) == 3  # temp obs all survive
    assert rejects["dewpoint_bounds"] == 1
    assert rejects["supersaturated"] == 1
    by_id = passed.set_index("station_id")["dewpoint_c"]
    assert pd.isna(by_id["KAAA"]) and pd.isna(by_id["KBBB"])
    assert by_id["KCCC"] == pytest.approx(20.4)


def test_qc_counts_and_drops_duplicates():
    obs = _obs([("KAAA", T0, 30.0, 20.0), ("KAAA", T0, 30.5, 20.5)])
    passed, rejects = qc_obs(obs)
    assert rejects["duplicate"] == 1
    assert len(passed) == 1
    assert passed.iloc[0]["temp_c"] == pytest.approx(30.0)  # keep="first"


def test_select_picks_nearest_within_tolerance():
    obs = _obs([
        ("KAAA", T0 - pd.Timedelta("7min"), 30.0, 20.0),   # nearest for KAAA
        ("KAAA", T0 + pd.Timedelta("53min"), 31.0, 21.0),  # outside 45 min
        ("KBBB", T0 + pd.Timedelta("40min"), 25.0, 15.0),  # inside, only ob
        ("KCCC", T0 + pd.Timedelta("46min"), 28.0, 18.0),  # outside -> station absent
    ])
    sel = select_analysis_obs(obs, T0)
    assert sorted(sel["station_id"]) == ["KAAA", "KBBB"]
    assert sel.set_index("station_id").loc["KAAA", "temp_c"] == pytest.approx(30.0)


def test_select_tie_prefers_earlier_ob():
    obs = _obs([
        ("KAAA", T0 - pd.Timedelta("10min"), 30.0, 20.0),
        ("KAAA", T0 + pd.Timedelta("10min"), 31.0, 21.0),
    ])
    sel = select_analysis_obs(obs, T0)
    assert len(sel) == 1
    assert sel.iloc[0]["temp_c"] == pytest.approx(30.0)


def test_select_empty_input_returns_empty():
    sel = select_analysis_obs(_obs([]), T0)
    assert sel.empty
