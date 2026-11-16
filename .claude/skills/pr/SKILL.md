---
name: pr
description: Run the gates, fill the pull request template from the issue's acceptance criteria and the actual command output, and open the pull request. Use when the user says "/pr", "open the PR", or when the work for an issue is complete.
---

# Pull request

## Procedure

1. Find the issue the branch delivers (`gh issue view N`). If there is none, run the
   `issue` skill first.
2. Run every gate and keep the output:
   - `ruff check .`
   - `mypy`
   - `pytest -q`
   - `python scripts/check_phrases.py`
   - in `client/`, when it exists: `npm run build` and `npx vitest run`
   A failing gate stops the procedure. Fix it or report it; never open the pull request
   over a red gate.
3. Run the code-review skill on the diff. Where the diff touches `src/heat/` or a
   training script, run the `leakage-auditor` subagent. Where it touches
   `service/payloads.py` or `client/src/api/`, run `contract-checker`. Fix what they find
   or list what was left and why.
4. Check each acceptance criterion in the issue against the diff. An unmet criterion is
   stated in the pull request, not hidden.
5. Fill `.github/PULL_REQUEST_TEMPLATE.md`: `Closes #N` first; what changed; the commands
   from step 2 with their real output, trimmed; the checklist; which parts an agent wrote
   and which reviews ran.
6. Push the branch and `gh pr create --base master`. Report the URL.

## Rules

- Never merge. Never push to `master`. Never rewrite a pushed branch; add a commit.
- Commit subjects are short and imperative; bodies say why.
- A number in the description points at the file that holds it.
