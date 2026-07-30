# Elastic RT Coding — Prompt 6 (M5: Beyond the Week-1 prototype)

## Goals

Extend the forward model past the L23-only prototype along the **phase-function** and
**full-BRDF** axes it could not exercise. This milestone is deliberately **coarse** in
the coding plan — its tasks are to be **detailed once the M4 results are in** (this doc
is the placeholder to fill then, modeled on how PAB grows a stage doc per stage).

## Claude

### Skills

Consider `.claude/skills/` (`critical-partner`, `code-review`).

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements*. Still **forward-model only**
and **differentiable** (inversion remains a separate design).

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M5.
- **Design** — `design/rt_elastic_model.md` §7–§8 (beyond week 1; phase-function
  parameterization generalization) and the synthesis roadmap in
  `context/RT/rt_elastic_model.md` §7.
- **Implementation record** — `design/rt_elastic_implementation.md` (M4 results drive
  the M5 scope).

## Prompts

1. Read this doc. Execute the 1st task in the "M5" section below. (First: turn the
   sketch below into a detailed, gated task list — the way earlier prompts were — using
   the M4 results.)

## M5 (sketch — to be detailed from M4 results)

### Tasks

1. **Detail this milestone.** Using the M4 outcomes, expand the sketch below into
   concrete, `pytest`-gated tasks (as in prompts 1–5), and record them here + in the
   implementation record. Ask any scoping questions in the Q&A section first.

Sketch of the intended scope:

- **New reference runs.** Commission **HydroLight** runs that explicitly vary the
  **particle phase function** and the **sensor zenith/azimuth** (the axes L23 fixes);
  add **PB24** (5000 IOPs × 1300 geometries) as a multi-angular cross-comparison.
- **Extend the emulator** on the richer reference; re-run the §6 protocol with
  **held-out phase-function shapes** and full geometry.
- **Promote the phase parameter** from `B_p` to the **ZTT backward-VSF** parameterization
  (§4.2 of the design) — without changing the `forward` signature.
- **Freeze the `forward` API** as the shared engine for training-data generation and the
  (separately designed) inversion.

### Q&A

## Logging

### <Date> (Short summary)

<Detailed description>

## Logs
