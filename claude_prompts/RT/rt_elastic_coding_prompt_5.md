# Elastic RT Coding — Prompt 5 (M4: Validation — Week-1 prototype complete)

## Goals

Implement **Milestone M4**: the **validation protocol** and the acceptance gate that
declares the Week-1 prototype **done**. Run the design §6 protocol on the held-out
splits, compare against Gordon / PR05 / O25, and ship a metrics table + figures.

## Claude

### Skills

Consider `.claude/skills/` (`code-review`, `verify`, `dataviz` for the figures).

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements*.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M4 (the acceptance gate)
  and the Testing strategy.
- **Design** — `design/rt_elastic_model.md` §6 (validation protocol: accuracy per
  λ/geometry/`B_p`; held-out generalization; speed; the gradient gate).
- **Split policy (CQ6)** — random 20% of scenes; **hold out 60° solar zenith**
  (train 0°/30°).
- **Comparison models** — the Gordon ladder and O25 form (reuse the synthesis-doc figure
  machinery in `context/RT/make_rt_elastic_figures.py`); PR05 where available.
- **Existing seams** — `robust/rt/{hybrid,ztt,emulator}.py`, the M1 loader/splits.

## Prompts

1. Read this doc. Execute the 1st task in the "M4" section below.
2. Read this doc. Execute the 2nd task in the "M4" section below.
3. Read this doc. Execute the 1st task in the "Pull Request" section below.

## M4

### Tasks

1. **Validation module.** `robust/rt/validation.py`: rRMS (rrs-space, relatively
   weighted) **per λ, per solar-zenith, per `B_p` bin**; **held-out** metrics on the CQ6
   splits; **throughput**; the **gradient-correctness** gate — all alongside **Gordon,
   PR05, O25** on the same splits. `design/py/run_validation.py` regenerates the metrics
   table + a couple of figures into `reports/figs/` (or `design/figs/`).

2. **Acceptance gate.** `robust/tests/test_validation.py`: the prototype passes iff the
   **hybrid beats standard Gordon on BOTH held-out splits** (random-20% scenes; unseen
   60° zenith) **and** passes the gradient gate. Commit the metrics table + figures.
   Update `design/rt_elastic_implementation.md` — mark M4 done, record the numbers and
   honest failure modes.

## Pull Request

1. The prototype is complete — prepare it for a PR (JXP will issue and run git):
   - a short summary of what M0–M4 delivered, the validation numbers, and known gaps
     (nadir-only / fixed-FF L23; full BRDF + phase-function-shape await M5);
   - confirm `pytest` is green and the branch is clean for review.

### Q&A

## Next

→ `rt_elastic_coding_prompt_6.md` (M5: beyond the prototype).

## Logging

### <Date> (Short summary)

<Detailed description>

## Logs
