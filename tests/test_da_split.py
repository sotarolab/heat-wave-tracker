"""Unit tests for src/heat/da/split.py — deterministic withheld-station split.

Pure functions, no network, importable with requirements-dev only.
"""
import pytest

from src.heat.da.split import is_withheld, partition, withheld_score

STATIONS = [f"K{a}{b}{c}"
            for a in "ABCDE" for b in "LMNOP" for c in "RSTUV"]  # 125 fake ids


def test_score_is_deterministic_and_in_range():
    for sid in STATIONS:
        s1, s2 = withheld_score(sid), withheld_score(sid)
        assert s1 == s2
        assert 0.0 <= s1 < 1.0


def test_split_is_stable_across_input_order():
    a1, w1 = partition(STATIONS)
    a2, w2 = partition(list(reversed(STATIONS)))
    assert set(a1) == set(a2)
    assert set(w1) == set(w2)


def test_withheld_fraction_is_approximately_requested():
    _, withheld = partition(STATIONS, fraction=0.2)
    # 125 hash draws at p=0.2: allow a generous binomial band, the point is
    # "roughly 20%", not exact stratification.
    assert 0.10 <= len(withheld) / len(STATIONS) <= 0.32


def test_partition_preserves_membership_and_covers_input():
    assim, withheld = partition(STATIONS)
    assert set(assim) | set(withheld) == set(STATIONS)
    assert set(assim) & set(withheld) == set()
    assert all(not is_withheld(s) for s in assim)
    assert all(is_withheld(s) for s in withheld)


def test_salt_changes_the_split():
    _, w_default = partition(STATIONS, salt="da-v1")
    _, w_other = partition(STATIONS, salt="da-v2")
    assert set(w_default) != set(w_other)


def test_duplicate_ids_are_an_error():
    with pytest.raises(ValueError, match="KDCA"):
        partition(["KDCA", "KRDU", "KDCA"])


def test_split_is_frozen_against_the_current_salt():
    # Pin a handful of concrete assignments under the shipped salt ("da-v1").
    # If this test fails, the split changed — which invalidates every
    # verification row produced so far and must be a deliberate,
    # documented decision (see config.SPLIT_SALT), never an accident.
    expected = {
        "KDCA": False, "KRDU": True, "KLAX": False, "KORD": False,
        "KPHX": True, "KSEA": False, "KMIA": False, "KDEN": False,
    }
    assert {sid: is_withheld(sid) for sid in expected} == expected
