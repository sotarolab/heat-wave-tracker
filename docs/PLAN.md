# Plan

The page is rebuilt in place as one FastAPI service and one React page, around the
correction model that already runs. The model gains a tracked lifecycle. Each step is one
issue and one pull request; the Dash page keeps serving until the cutover.

The target is drawn in [ARCHITECTURE.md](ARCHITECTURE.md). The way a change moves is in
[process/WORKFLOW.md](process/WORKFLOW.md).

## Order of work

| Step | Issue | Delivers |
|---|---|---|
| 0 | #33 | The scheduled refresh back on; the header says how old the forecast is. |
| 1 | #34 | This plan, the architecture document, templates, skills, subagents, gates. |
| 2 | #35 | `service/` and `client/`: the shell, the Map tab, the Satellite tab. |
| 3 | #36 | The Forecast and Stations tabs: observations, GFS, HRRR and NBM, the two corrections, today's verification. |
| 4 | #37 | Training as a script, MLflow tracking and registry, data contracts, the golden backtest, the model card. |
| 5 | #38 | The Lab tab: scorecard by lead, coverage, SHAP, feature importance. |
| 6 | #39 | Cutover: the service replaces Dash on the live URL; dependencies locked with uv. |
| 7 | #40 | Drift report, skill decay, scheduled retraining, promotion by pull request. |
| 8 | #41 | 500 hPa height and 850 hPa temperature overlays; deep stations with several models. |
| 9 | #42 | Analysis by optimal interpolation, scored at withheld stations. |
| 10 | #43 | The assistant restricted to tool results, its evaluation suite, an MCP server. |
| 11 | #48 | The correction trained on HRRR, scored beside the GFS-based one. |

## Decisions already taken

| Decision | Record |
|---|---|
| One service and one page replace Dash; jobs write, routes read. | [0001](decisions/0001-one-service-one-page.md) |
| The first model shown is the existing hourly correction. | this plan |
| Optimal interpolation replaces the ensemble analysis designed in `da/DESIGN.md`. | ADR with #42 |
| MLflow for tracking and the registry; artifacts stay in the database for now. | ADR with #37 |
| All 165 stations stay on the map; about 12 get the multi-model view. | #41 |

## Later

- Rain gauges on the map and in the Stations table (#49).

## Not planned

New model families, an ensemble analysis, personal weather station networks, a feature
store, and any orchestration beyond GitHub Actions.
