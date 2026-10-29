# Plan review: open decisions

Review of `docs/PLAN_2026-10-28.md`, started 2026-10-28. Worked one item at a time with the
`review` skill. Each decision is applied to the plan when made; the ones marked ADR get a
record in `docs/decisions/`.

| # | Item | Status | ADR |
|---|---|---|---|
| 1 | Observation and settlement keys exclude `received_at`, which breaks append-only | decided | yes |
| 2 | Lead has no definition for a daily target | decided | yes |
| 3 | Scorecard page goes live in step 4 but the client arrives in step 6 | decided | no |
| 4 | Baselines: HRRR lead coverage, NDFD source, feeds confirmed before step 3 | decided | no |
| 5 | model_runs volume on Neon: store full horizons or only what is scored | decided | yes |
| 6 | Conformal gate on one block drifts with season; monitor on scorecard | decided | no |
| 7 | Backfilled model runs carry an assumed `received_at` | decided | no |
| 8 | "Kept as-is" list vs the empty-package rule; `da/` on the list | decided | no |
| 9 | Who writes which tests; "local day never stored" wording; `cli_site` per station | decided | no |
| 10 | Branch state: 20 unmerged commits on feat/obs-extras and two untracked scripts | open | no |

## Where we are

Updated 2026-10-29, end of the second session. Items 1 to 9 are decided and applied to the
plan. Item 10 was presented and is waiting for an answer. Say "next" or "/review" to resume.

**Item 10, pending.** The rebuild's starting point. Facts gathered: feat/obs-extras holds 20
commits (the Sol assistant with tests and evals, extra observed variables in asos.py, the
skills); master has only data auto-refresh commits since; fifteen stale branches from
August and September. Untracked `scripts/build_reforecast_archive.py` walks git history for
GFS by run and lead since 2026-07-03 with true receipt times, which beats the assumed
latency in item 7 for GFS. Recommendation: PR feat/obs-extras to master, run the code-review
skill, user merges; start the rebuild branch from master; commit the reforecast script as the
GFS backfill source; delete stale branches after a look.

**After item 10, in order.**
1. `/adr` for items 1, 2 and 5 (three records in `docs/decisions/`).
2. Commit `docs/PLAN_2026-10-28.md` and this file.
3. Step 1 of the plan: the skeleton commit.

## Decisions

**1. Keys (2026-10-28).** `received_at` joins the primary key of `observations` and
`settlements`; a correction is a new row and reads take the latest row at or before `as_of`.
`scores` is the one derived table rebuilt in place. Needs an ADR.

**2. Lead and issue slots (2026-10-28).** Lead is hours from the issue slot to the end of the
target's local day. Predictions issue at 00, 06, 12, 18 local; each target is scored at the six
slots before its typical time; ingest keeps the latest run per model per slot; one training
sample per station, day and slot. Needs an ADR. **Amended the same day** after reading the
product notes of a planning app (today and tomorrow, the evening-before alert, raw models to five
days): the 18:00 local slot also issues for days 2 through 5; the model trains to day 3; days 4
and 5 are scored raw first.

**Spatial extension (2026-10-28).** `station_id` is namespaced by network; `stations` gets
`network`, `settles`, `settlement_source` and `tier`; spatial features take a frame of
located observations and never know the network. This closes the `cli_site` part of item 9.

**3. Client order (2026-10-28).** The React skeleton and the Scorecard page move into step 4.
Station page parity and Dash retirement stay in step 6.

**5. model_runs layout and scale (2026-10-29).** One row per model, station, run and
variable with the hourly horizon as a float4 array. 165 stations, four slots, three variables.
Rows older than 18 months roll off to parquet. Needs an ADR.

**6. Coverage monitor (2026-10-29).** Training-time conformal gate kept; `covered_90` per
lead bucket over a trailing 30 days must stay in 85 to 95, and a breach fails the nightly job.

**7. Backfill (2026-10-29).** Backfilled model runs carry `source = backfill` and a documented
per-model latency constant; backtest and scorecard split backfilled from live; headline numbers
from the live period only.

**8. Porting (2026-10-29).** The package starts empty; modules are candidates ported one at a
time after reading; `da/` narrows to `gefs_members.py`; the nine test files are the acceptance
bar; `app.py` and the scripts are rewritten.

**9. Wording (2026-10-29).** Science tests are person-written first; local day is derived in
one module and nowhere else; `cli_site` was replaced by `settlement_source` under the spatial
decision.

**4, partial (2026-10-28).** Model set is GFS, ECMWF IFS, ICON (global, also the Chile set),
HRRR, NBM (CONUS) and the NWS forecast. **Closed (2026-10-28):** a one-day feed spike opens
step 2; one station, one slot, all six sources; the resulting table of source, variables,
horizon and cadence per model goes into `docs/ARCHITECTURE.md`.
NAM out (retirement path), AIFS out by choice. Spread is a feature (inter-model and GEFS
members), scored by tercile. The feed and lead-coverage check for item 4 is still open.
