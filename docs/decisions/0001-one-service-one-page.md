# 0001. Serve one FastAPI service and one React page; retire Dash

Date: 2026-11-16
Status: accepted

## Context

The page is a single Dash module of about 3,900 lines. Layout, callbacks, database
queries, image rendering and statistics share one file and one process, so no part of it
can be tested without the rest, and the page computes at request time. Two sibling
projects use a smaller format: a FastAPI service that returns finished numbers, and a
React page that draws them.

## Decision

The page becomes `service/` (FastAPI) and `client/` (Vite, React, TypeScript, uPlot,
Leaflet) in this repo, on the same Render service and URL. Scheduled jobs compute and
write; routes read. The science in `src/heat/` is unchanged and imports nothing from the
service. Dash keeps serving from `master` until the Map, Forecast, Stations and Lab tabs
exist, then `app.py` is removed in one pull request.

## Consequences

Payload builders are plain functions with tests. The client needs a build step, so the
container gains a node stage. Field images move from Plotly layers to PNG routes drawn on
a Leaflet map. Two languages are now gated in CI. The Dash page receives no new features.

## Alternatives rejected

- Keep Dash and add panes: the single-module shape and request-time computation remain.
- A static page on a data branch: the corrected line and live observations need a process.
- A new repository: the archive, the model and the tests are here.
