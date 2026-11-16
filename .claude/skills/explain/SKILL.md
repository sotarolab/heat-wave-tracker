---
name: explain
description: Explain an existing module so the reader can state in one sentence what it does and why, before reviewing or changing it. Use when the user says "/explain <path>", "walk me through <module>", or asks what a file does. Explains only; never writes or moves code.
---

# Explain a module

Science modules in `src/heat/` and the training scripts are read line by line by a
person. This skill is the reading aid.

## Procedure

1. Read the whole module and its tests. Do not skim.
2. Reply in this shape, under 400 words:
   - **One sentence**: what it does and why it exists.
   - **Inputs and outputs**: the frames or values in and out, with shapes and units.
   - **The three lines that matter**: `file:line` references to where the science or the
     invariant lives. Everything else is plumbing.
   - **What the tests prove** and what they do not.
   - **Smell list**: anything that breaks a rule in `CLAUDE.md` (computes at request
     time, uses information later than the init time, mixes units, holds state).

## Rules

- Never propose a change in the same turn. Reading first, writing later.
- If the module is over 300 lines, split the explanation by function and ask which to start with.
- Point at lines; do not paste code blocks longer than five lines.
