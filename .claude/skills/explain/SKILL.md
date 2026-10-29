---
name: explain
description: Explain an existing module so the user can state in one sentence what it does and why, before it is ported into the new src/heat package. Use when the user says "/explain <path>", "walk me through <module>", or asks what a file does. Explains only; never writes or moves code.
---

# Explain a module before porting

The new package starts empty. A module moves only after the user has read it and can say
in one sentence what it does and why. This skill is the reading aid.

## Procedure

1. Read the whole module and its tests. Do not skim.
2. Reply in this shape, under 400 words:
   - **One sentence**: what it does and why it exists. The user will repeat this back.
   - **Inputs and outputs**: the frames or values in, the frames or values out, with shapes.
   - **The three lines that matter**: `file:line` references to the places where the
     science or the invariant lives. Everything else is plumbing.
   - **What the tests prove** and what they do not.
   - **Smell list**: anything that would break the plan's rules (computes at request time,
     reads without `as_of`, derives local day, holds state).
3. End with: "Say the sentence back and I will record it in the port log."
4. When the user says the sentence, append a row to `docs/PORT_LOG.md`
   (`| date | module | user's sentence | decision: port / rewrite / drop |`) and stop.

## Rules

- Never propose the port in the same turn. Reading first, writing later.
- If the module is over 300 lines, split the explanation by function and ask which to start with.
- Point at lines; do not paste code blocks longer than five lines.
