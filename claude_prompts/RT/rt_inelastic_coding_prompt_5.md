# Inelastic RT Coding — Prompt 5 (M4: Validation — prototype complete)

## Goals

Implement **Milestone M4** of the coding plan
(`design/rt_inelastic_model_coding_plan.md`): the full design-§6 validation
protocol and acceptance gate, the review pass, the committed
metrics/figures, and the finished implementation record. The inelastic
prototype is **done** when the M4 gate passes.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `code-review` for the
review task, `critical-partner`) as helpful.

### Working agreements

As in `rt_inelastic_coding_prompt_1.md`: JXP runs git (branch
`rt-inelastic-prototype`); `ocean14` on this machine, CPU JAX; every task
`pytest`-gated; the **elastic hash-regression stays green**. Use Fable if
you can. Log your work.

## Context

Read before coding:

- **Coding plan** — M4 section + Definition of done + the speed-risk entry.
- **Design** — `design/rt_inelastic_model.md` §6 (the gate, verbatim) and
  the reported-not-gated diagnostics.
- **Elastic validation** — `robust/rt/validation.py`,
  `design/py/run_validation.py`, `design/validation/` (the artifact
  pattern to extend, not fork).

## Status entering M4

*(Filled by M3's final task.)*

## Prompts

1. Read this doc. Execute the 1st task in the "M4" section below. If you have
   any questions, ask me in the Q&A section below. Use Fable if you can. Log
   your work.
2. Read this doc. Execute the 2nd task. Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task — the review pass. Use Fable if you
   can. Log your work.
4. Read this doc. Execute the 4th task — the notebook and the implementation
   record wrap-up. Check my answers in Q&A. Use Fable if you can. Log your
   work.

## M4

### Tasks

1. **Validation protocol.** Extend `robust/rt/validation.py` with the design
   §6 protocol: held-out (elastic splits) rrs-space rRMS vs `Rrs_X4`, per λ /
   per zenith; the per-process delta metrics; throughput vs the elastic
   hybrid (same batch, same machine); the gradient gate incl. φ_C; the
   elastic hash-regression. Diagnostics (reported, not gated): performance
   by a_ph(440) decile; the φ_C-linearity check against the scaled-truth
   construction; `emission_shape='double'` behavior. If the ≤ 2× speed
   budget is threatened, apply the coding-plan fallback (precompute
   zenith-static quantities at trace time; fuse the excitation quadrature).

2. **Artifacts.** Extend `design/py/run_validation.py` so one command
   regenerates the inelastic metrics table + figures into
   `design/validation/` (alongside, not replacing, the elastic artifacts).
   Figures follow the recorded style conventions; every number quoted in
   the implementation record must be regenerable by this script.

   **Gate (acceptance = design §6).** `robust/tests/test_inelastic_validation.py`:
   (1) held-out total rRMS vs X4 ≤ **0.5 %** at each zenith; (2) Raman delta
   ≤ 5 % incl. 0°; (3) fluorescence delta ≤ 5 %; (4) `inelastic=None`
   bit-identical; (5) gradient checks for all inputs incl. φ_C; (6)
   full-batch forward ≤ **2×** elastic-hybrid runtime.

3. **Review pass (CQ6).** Run a code review over the full branch diff
   (e.g. the `code-review` skill at high effort); address the findings
   (fix or explicitly decline with a reason in the log) **before** the gate
   is declared. JXP may additionally supply an external (Cursor) review —
   address those comments too if provided in Q&A.

4. **Wrap-up.** Finish `design/rt_inelastic_implementation.md` (all
   milestones, environment, module index, results tables, the definition-
   of-done statement mirroring the elastic record §6.10). Notebook
   `notebooks/RT/rt_inelastic_coding_5.ipynb` — executed: the headline
   validation figures (total rRMS ladder incl. elastic-only baseline;
   per-process before/after; speed and gradient tables) with honest
   captions about what the prototype may and may not claim (mirror the
   elastic `prototype_summary.md` candor — e.g. the 3-zenith geometry
   caveat, φ_C-linearity untested against varied-φ truth). Note the branch
   state for JXP's PR.

### Q&A

## Next

The prototype is complete. Next steps live in
`claude_prompts/RT/rt_inelastic_prompts.md` (the Report prompts), and the
"Beyond v1" section of the coding plan (HydroLight wishlist) awaits JXP's
compute planning.

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
