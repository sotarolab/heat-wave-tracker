"""
src/heat/da/config.py
=====================
Single source of truth for the surface-analysis (DA) subsystem's paths and
constants. Anything tunable lives here, not inline in the modules that use
it, so an experiment's configuration is readable in one place and a change
shows up in one diff.

Kept import-light on purpose: this module must be importable with only the
requirements-dev stack (numpy/pandas/scipy), because unit tests and CI
import the DA package without the GRIB/xarray stack installed. See
requirements-dev.txt for why those two dependency sets are deliberately
separate.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths ---

DA_DATA_DIR = Path("data") / "da"
STATION_TABLE_PATH = DA_DATA_DIR / "asos_stations.json"

# ----------------------------------------------------------- background ---

# Same CONUS bbox as src/heat/gfs_conus.py (west, south, east, north),
# duplicated rather than imported because gfs_conus imports the GRIB stack
# at module top and this module must stay light (see module docstring).
CONUS_BBOX = [-127.0, 23.0, -65.0, 51.0]

# Background lead time. 6 h is the classic cycle-to-cycle convention: short
# enough that ensemble spread is a meaningful error estimate at analysis
# time, long enough that the background has not seen the verifying obs.
BACKGROUND_LEAD_HOURS = 6

# GEFS membership: control (c00) plus 30 perturbed members. Member labels
# follow the NOMADS/AWS file naming (gec00, gep01..gep30).
N_PERTURBED_MEMBERS = 30
MEMBER_LABELS = ["c00"] + [f"p{m:02d}" for m in range(1, N_PERTURBED_MEMBERS + 1)]

# --------------------------------------------------------- observations ---

# One ob per station, nearest the analysis time, within this tolerance.
# Routine METARs report around :53 past the hour (see src/heat/asos.py
# Gotcha 1), so 45 min covers one full routine cycle around any synoptic
# analysis hour without ever spanning two of a station's routine reports.
OBS_MATCH_TOLERANCE_MIN = 45

# Physical QC bounds (°C). Wider than any plausible CONUS surface value so
# these only ever catch encoding/transmission faults, never real extremes
# (cf. the deliberately-below-the-record reasoning used for precipitation
# rate QC elsewhere in this project's family of tools).
TEMP_QC_BOUNDS_C = (-60.0, 60.0)

# ------------------------------------------------------- withheld split ---

# Fraction of stations withheld from assimilation for verification, chosen
# by a deterministic hash (src/heat/da/split.py) so the split never changes
# across cycles, machines, or refactors. The salt versions the split: bump
# it only with a documented reason, because doing so invalidates
# comparability of every verification series produced before the bump.
WITHHELD_FRACTION = 0.2
SPLIT_SALT = "da-v1"
