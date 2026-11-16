---
name: verify
description: Boot the service, call every route, build the client and report what a reviewer needs to know: which routes answered, with what shape, and how old the data is. Use when the user says "/verify", "check the app", or after a change to service/ or client/. Pass a URL to check a deployed instance instead.
---

# Verify

## Local

1. Start the service on a free port: `uvicorn service.app:app --port 8051` (background).
   Until `service/` exists, say so and run `pytest -q` instead.
2. Read the route list from `service/app.py`. For each route call it with `curl -s -w`
   and record status, time and the top-level keys of the JSON (or the image size for a
   PNG route). Use one real station id from `/api/stations`.
3. `GET /api/manifest`: report the forecast cycle and its age. A stale cycle is a finding.
4. In `client/`: `npm run build`, then `npx vitest run`.
5. Stop the service.

## Deployed

With a URL argument, do steps 2 and 3 against it, plus `GET /healthz`.

## Report

A table: route, status, milliseconds, shape. Then findings, most serious first: any
non-200, any empty payload, any route slower than two seconds, stale data. No finding
means say so in one line.

## Rules

- Never print environment variables or the contents of `.env`.
- Read-only: this skill changes no file and deploys nothing.
