"""
src/heat/da/split.py
====================
Deterministic withheld-station split for analysis verification.

~20% of stations are never assimilated; they exist only to answer, every
cycle, "does the analysis beat the raw background at locations it never
saw?" (docs/da/DESIGN.md, verification protocol).

The split is a stable hash of the station id, NOT a random draw, for one
load-bearing reason: the verification time series must stay comparable
across cycles, machines, library versions, and refactors. `random.seed()`
does not guarantee that (station table ordering changes, Python hash
randomization, numpy generator changes across versions all break it);
md5(station_id + salt) does. The salt versions the split — bumping it
reassigns stations and therefore invalidates comparability with every
previously produced verification row, so it changes only with a documented
reason in config.py.
"""
from __future__ import annotations

import hashlib

from src.heat.da.config import SPLIT_SALT, WITHHELD_FRACTION

# Denominator for mapping the hash to [0, 1). Any large constant works; a
# power of two keeps the mapping exact in float.
_HASH_BUCKETS = 2**32


def withheld_score(station_id: str, salt: str = SPLIT_SALT) -> float:
    """Stable score in [0, 1) for one station. Pure function of (id, salt)."""
    digest = hashlib.md5(f"{salt}:{station_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / _HASH_BUCKETS


def is_withheld(
    station_id: str,
    fraction: float = WITHHELD_FRACTION,
    salt: str = SPLIT_SALT,
) -> bool:
    """True if this station is verification-only (never assimilated)."""
    return withheld_score(station_id, salt) < fraction


def partition(station_ids, fraction: float = WITHHELD_FRACTION, salt: str = SPLIT_SALT):
    """Split ids into (assimilated, withheld) lists, input order preserved.

    Duplicate ids are an error rather than silently deduplicated: the same
    physical station appearing twice (e.g. under both a 3- and 4-letter id,
    see src/heat/asos.py Gotcha 2) could land on both sides of the split and
    quietly contaminate verification with assimilated information.
    """
    ids = list(station_ids)
    if len(set(ids)) != len(ids):
        seen, dupes = set(), set()
        for s in ids:
            (dupes if s in seen else seen).add(s)
        raise ValueError(f"duplicate station ids in split input: {sorted(dupes)}")
    assimilated = [s for s in ids if not is_withheld(s, fraction, salt)]
    withheld = [s for s in ids if is_withheld(s, fraction, salt)]
    return assimilated, withheld
