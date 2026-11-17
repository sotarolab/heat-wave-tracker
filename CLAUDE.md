# Rules for work in this repo

Read `docs/ARCHITECTURE.md` and `docs/PLAN.md` first. The way a change moves is in
`docs/process/WORKFLOW.md`.

## What must stay true

1. **Jobs write, routes read.** A route never trains, scores or fits. It reads a file a
   job wrote, or a public feed through `service/live.py:cached`.
2. **Point in time.** A forecast is scored only against observations that arrived after
   it, and a model is trained only on rows before its test block. Features use nothing
   later than the forecast's init time.
3. **No metric by hand.** A number in a document or a pull request comes from a file a
   job wrote, with the commit. Results live in `docs/ml/EXPERIMENTS.md` and `data/lab/`.
4. **The archive is private.** Forecast and observation pairs stay in the database. Only
   aggregates are committed. No model artifact is committed.
5. **Science stays readable.** Modules in `src/heat/` are small enough to read in one
   sitting and import nothing from `service/`. Their tests state the claim first.
6. **A feature ships with its gate.** Tests arrive in the same pull request as the code.
7. **The page computes nothing.** Units, rounding, colours of a field and the text of a
   headline are decided in Python.

## Process

- One issue, one branch from `master`, one pull request that starts with `Closes #N`.
- Plan before a change that crosses layers. Write the ADR in the session the decision is
  made (`/adr`).
- Run the gates before opening a pull request (`/pr` does it): `ruff check .`, `mypy`,
  `pytest`, `python scripts/check_phrases.py`, and in `client/` `npm run build` and
  `npx vitest run` once the client exists.
- The owner merges. Never merge, never rewrite a pushed branch, never push to `master`.
- State in the pull request which parts an agent wrote and which a person read.

## Writing

Plain scientific prose. One claim per sentence. No em dashes. Do not assert candour;
show the evidence. Comments say why, not what. Public text describes the software and
the science only.

## Environment

- Python 3.11. Tests: `pytest`. The GFS fetch needs eccodes; the tests do not.
- Secrets (`NEON_DATABASE_URL`, the deploy hook, API keys) are environment variables set
  in GitHub and Render. Never print, log or commit them. Never read `.env`.
- The page is `service/` (FastAPI) and `client/` (React). The container installs
  `service/requirements.txt`; the forecast jobs install `requirements.txt`.
