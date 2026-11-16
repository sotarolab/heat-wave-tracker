# How a change moves

The code in this repo is written with AI agents (Claude Code) under gates. This page says
what the gates are and which parts a person reads line by line, so that the history can
be checked rather than trusted.

## 1. The path of one change

1. **Issue.** Every change starts as an issue from the template: context, scope,
   acceptance criteria as checkboxes, verification commands, out of scope. The `issue`
   skill writes it from an item in [PLAN.md](../PLAN.md).
2. **Plan.** For a change that crosses layers, the agent plans first and the plan is
   approved before code is written. A decision taken on the way becomes an ADR in the
   same session (`adr` skill).
3. **Branch.** From `master`, one branch per issue. A pushed branch is never rewritten.
4. **Tests first for science.** In `src/heat/` and the training scripts, the test states
   the claim before the code exists.
5. **Gates.** `ruff`, `mypy` on the gated modules, `pytest` (including the golden
   backtest once it exists), the writing-rules check, and the client build and tests.
   The same commands run in `qc.yml` on every pull request.
6. **Agent review.** The code-review skill, then the two subagents where they apply:
   `leakage-auditor` on science and training changes, `contract-checker` on changes to
   payloads or client types.
7. **Pull request.** The `pr` skill fills the template: the issue it closes, what
   changed, the verification commands with their output, and which parts an agent wrote
   and which a person read.
8. **Human review and merge.** The repository owner reads and merges. Agents do not merge.
9. **Deploy and verify.** The `verify` skill boots the service, calls every route and
   builds the client; after the cutover it also checks the live URL.

## 2. Who reads what

| Area | Written by | Read line by line by a person |
|---|---|---|
| `src/heat/` science | agent or person | yes, every line |
| `scripts/train_correction.py`, the golden backtest | agent or person | yes |
| `service/` payload builders | agent | yes |
| `service/` routes and cache, `client/`, workflows | agent | no: covered by tests, types and the contract check |
| ADRs, the model card | agent drafts | yes |

## 3. Tools in the repo

| Tool | Where | Use |
|---|---|---|
| Skills | `.claude/skills/` | `issue`, `pr`, `verify`, `adr`, `explain`, `review` |
| Subagents | `.claude/agents/` | `leakage-auditor`, `contract-checker` |
| Hooks and permissions | `.claude/settings.json` | format new packages on edit; no edits to `.env` or `data/` |
| Rules | `CLAUDE.md` | read by the agent at the start of every session |
| Commit hooks | `.pre-commit-config.yaml` | ruff and the writing-rules check |
| CI | `.github/workflows/qc.yml` | the gates above on every pull request |

## 4. Numbers

No metric is typed by hand. Scores are written by jobs into `data/lab/` and by
`scripts/export_runs.py` into `docs/ml/EXPERIMENTS.md`, each with the commit that
produced it. A pull request that states a number points at the file.
