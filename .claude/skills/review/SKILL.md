---
name: review
description: Walk the owner through the next piece of work one item at a time: the next open issue in the plan, or the open pull request for it. Use when the user says "/review", "next", "what's next", or asks where the rebuild stands.
---

# Review the plan, one item per turn

1. Read `docs/PLAN.md`. List open pull requests (`gh pr list`) and the milestone's open
   issues (`gh issue list --milestone "Rebuild in the family format"`).
2. Pick the single next item: an open pull request waiting for the owner before any
   unstarted issue.
3. For a pull request, give: what it closes, the three files worth reading first with
   `file:line` references, what the gates and reviews reported, and what is left
   unchecked in the issue's acceptance criteria.
4. For an unstarted issue, give: its scope in three lines, what it depends on, and the
   first decision the owner has to take, if any.
5. Stop. One item per turn. Wait for the owner's decision or "next".

## Rules

- Under 250 words per turn. Point at lines; do not paste code.
- Do not start work in this skill. It reports and waits.
