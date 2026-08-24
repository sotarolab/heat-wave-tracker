# Ensemble surface analysis (DA) — design

Tracking epic: [#2](https://github.com/sotarolab/heat-wave-tracker/issues/2).
This document is the source of truth for scope and method; issues track
execution. It describes the system as designed — where reality diverges
during implementation, this file gets updated in the same PR that diverges.

## Problem

The tracker shows two views of the surface state today: the raw GFS grid,
and per-station same-day bias corrections (`src/heat/bias.py`) that exist
only at the 165 leaderboard stations and only as point offsets. Nothing
produces a spatially consistent, observation-corrected best estimate of the
current 2 m temperature/dewpoint field. The gap is largest exactly where the
map is most interesting: structured regimes (fronts, marine layers, drylines)
where model error is coherent over hundreds of kilometers and a dense
observation network can see it.

## Approach

A cycled surface analysis: every 6 hours, combine

- **Background**: the freshest GEFS 6 h ensemble forecast (31 members,
  0.25°, 2 m T and Td over CONUS). Short lead keeps the background honest
  (it has not seen the verifying observations) while staying close enough
  to analysis time that its ensemble spread is a meaningful error estimate.
  We never cycle our own model — each analysis is an independent experiment,
  which removes filter-divergence risk while the system's behavior is being
  learned.
- **Observations**: a CONUS-wide ASOS/METAR snapshot from IEM (thousands of
  sites, not just the 165 leaderboard stations), QC'd, one routine ob per
  station nearest the analysis time.

via an ensemble square-root filter (EnSRF; Whitaker & Hamill 2002) with
Gaspari–Cohn localization and multiplicative inflation.

## Verification protocol

The design's central discipline: **~20% of stations are withheld by a
deterministic hash of the station id and never assimilated.** Every cycle
reports background vs analysis RMSE/bias at those stations. Hash-based
rather than random so the split is stable across cycles, machines, and
refactors — the verification series stays comparable forever.

Comparison tiers, all sharing one observation operator and one metrics
schema so differences isolate the covariance model:

| tier | covariance | question answered |
|------|-----------|-------------------|
| 0 | none (raw background) | how good is GEFS 6 h already? |
| 1 | static isotropic (OI) | what does *any* obs blending buy? |
| 2 | ensemble, localized (EnSRF) | what does flow dependence buy over static B? |
| 3 | EnSRF + learned debias | what does removing systematic background bias buy? |

Tier 3 trains a small probabilistic correction on the Neon
`forecast_obs_pairs` archive (short-lead GFS-vs-ASOS pairs, accumulating
since 2026-07-16 — the same lead regime as the background) and applies it
to the background at observation locations before innovations are computed.

Everything downstream of a cycle is a pure function of two artifacts: the
saved analysis dataset and one appended metrics row (fixed column schema,
append guarded against schema drift). No plot or table reads raw cycle data
directly.

## Package layout

```
src/heat/da/
  config.py        # paths + physical/tuning constants, the only place either lives
  gefs_members.py  # tier-1 ingest: GEFS member stack (lazy herbie/xarray imports)
  asos_network.py  # tier-2 ingest: station table, obs window fetch, QC
  split.py         # deterministic withheld-station split
  oi.py            # tier-1 analysis: static-B OI          (issue #5)
  ensrf.py         # tier-2 analysis: localized EnSRF      (issue #6)
  verify.py        # withheld-station metrics + schema     (issue #5/#6)
scripts/
  fetch_da_background.py   # argparse driver, logic lives in the package
  fetch_da_obs.py
```

Conventions carried over from the existing codebase: driver scripts are
argparse + printing only; heavy optional deps (herbie/cfgrib/xarray) import
lazily so the requirements-dev-only CI can import and unit-test the pure
logic; unit tests use synthetic in-memory frames, never the network; units
are °C and longitudes [-180, 180] everywhere, matching
`src/heat/gfs_conus.py`.

## Non-goals

Surface variables only (no upper air, no radiances); no cycling of our own
forecast model; no attempt to beat operational analyses (RTMA exists — the
point here is a transparent, verifiable system whose every component we can
explain and ablate, not a product replacement).
