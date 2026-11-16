---
name: issue
description: Turn a plan item or a described change into a GitHub issue in the repo's template, with acceptance criteria that can be checked by running something. Use when the user says "/issue", "open an issue for", or before starting work that has no issue.
---

# Issue

Every change starts as an issue. The pull request closes it.

## Procedure

1. Read `docs/PLAN.md` and the code the change touches. Check `gh issue list` for an
   existing issue that covers it; if one exists, say so and stop.
2. Draft the body in the shape of `.github/ISSUE_TEMPLATE/change.md`:
   Context, Scope, Acceptance criteria, Verification, Out of scope.
3. Acceptance criteria are checkboxes, each one a statement a command can confirm.
   "The tab renders" is not a criterion; "`/verify` reports 200 on `/api/lab`" is.
4. Show the draft. On approval, create it:
   `gh issue create --title "<title>" --label "area:<process|service|client|ml|docs>" --milestone "<milestone>" --body-file <file>`
5. If the change is a new step, add its row to `docs/PLAN.md` in the pull request that
   delivers it.

## Rules

- One issue per pull request. If the scope needs two pull requests, write two issues.
- Under 250 words. Plain prose, no em dashes.
- State what is out of scope; it is what a reviewer checks against.
