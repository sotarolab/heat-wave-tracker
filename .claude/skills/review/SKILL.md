---
name: review
description: Walk the user through open decisions one at a time from docs/PLAN_REVIEW_*.md. Use when the user says "next", "next item", "/review", or asks to continue the plan review. One item per turn, then stop and wait for the decision.
---

# Review walk-through

The user reviews decisions one at a time. Each turn covers exactly one item.

## Procedure

1. Open the newest `docs/PLAN_REVIEW_*.md`. Find the first item with status `open`.
2. If the user's message carries a decision for the previous item, record it first:
   set that item's status to `decided`, write the decision in one or two sentences
   under it, and apply it to `docs/PLAN_*.md` with an Edit. Do not create an ADR
   here; `/adr` does that when the user asks.
3. Present the next open item in this shape, nothing else:
   - **What is at stake**, two or three sentences, pointing at the exact lines in the plan.
   - **Options**, at most three, each one line with its cost.
   - **Recommendation**, one line, with the single strongest reason.
   - **Decide**: the one question the user has to answer.
4. Stop. Do not present the next item until the user answers.

## Rules

- Never bundle items. If two items depend on each other, say so and still present one.
- Keep the item under 200 words. The user reads it in the terminal.
- When all items are decided, say so and list which decisions need an ADR.
