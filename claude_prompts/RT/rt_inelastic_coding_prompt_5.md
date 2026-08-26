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

As in `rt_inelastic_coding_prompt_1.md`: JXP runs git — branch
**`inelastic-rt`** (prompt 1 Q&A Q2; *not* the coding plan's
`rt-inelastic-prototype`); `ocean14` on this machine, CPU JAX; reuse over
reinvention; every task `pytest`-gated; the **elastic hash-regression stays
green** — two-tier (record §2.8): strict SHA-256 pins on dev machines, the
ULP-closeness regression everywhere. Never `pip install -r requirements.txt`
wholesale (the `git+` lines clobber the editable `bing`/`ocpy` checkouts).
Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — M4 section + Definition of done + the speed-risk entry.
- **Design** — `design/rt_inelastic_model.md` §6 (the gate, verbatim) and
  the reported-not-gated diagnostics.
- **The implementation record** — `design/rt_inelastic_implementation.md`
  §§4–5 (v0.23): everything M2–M3 built, its API, and every measured
  number. Chronology: prompts 3–4 Logs.
- **The M3 tests** — `robust/tests/test_inelastic_corr.py`: the held-out
  gates, the weights-integrity regression, and the corrected-path FD
  protocol M4's gradient gate should reference rather than re-derive.
- **Elastic validation** — `robust/rt/validation.py`,
  `design/py/run_validation.py`, `design/validation/` (the artifact
  pattern to extend, not fork).

## Status entering M4

*(Filled by M3's final task, 2026-08-26. Details: record v0.23 §5;
chronology: prompt 4 Logs. Q&A state: **all prior questions answered and
closed** — corrections default-on approved by JXP, the stray `rob/`
directory kept.)*

**M3 is complete — all six tasks (incl. the JXP-inserted PR-review task);
PR #18 merged.** Suite: **416 passed, 1 skipped** (the skip is deliberate —
the missing-weights fallback test retired itself when the weights landed);
ruff + format clean; elastic hash pins green. What M4 inherits:

- **The complete forward model, corrected by default.**
  `forward(..., inelastic=Inelastic())` is now the full design-§2 law —
  `(Rrs_ZTT + ΔRrs) × f_R + φ_C·K_fl(1 + δ_F)` with
  `f_R = 1 + (f_phys − 1)(1 + δ_R)` — via `corrections=None` resolving to
  the packaged heads (`robust/rt/files/{raman,fl}_corr_l23.npz`, 129
  params each, tanh bounds 1.0/0.5). `corrections=False` is the analytic
  backbone (the M2 pins use it, 13 call sites); `inelastic=None` is bitwise
  elastic. The a_ph requirement raises pre-emulator.
- **The per-process M4 gate lines are already met and test-pinned**
  (`test_inelastic_corr.py`, full release, held-out scenes, committed
  weights only): Raman increment median **−0.14/−0.10/−0.21 %** at
  0/30/60° (550–700 nm; gate ≤ 5 % — the analytic backbone's −38.6 % @ 0°
  closed), 490 nm ≤ +1.0 %; fluorescence 685 nm **+0.08/+0.07/+0.10 %**
  (analytic was −13.7 % @ 60°). The a_ph(440)-decile diagnostic is flat
  (±0.6 %; the analytic drifts −11 → +11 % across deciles). **M4's
  genuinely new gate numbers are (1) total held-out rRMS vs Rrs_X4 and
  (6) speed ≤ 2× elastic** — everything else is standing tests.
- **Gradient gates exist through both paths** (analytic:
  `test_inelastic.py`; corrected: `test_inelastic_corr.py`) for
  `a, bb_p, a_ph, φ_C, θ_s`, float64, per-variable steps, **θ_s at 35°** —
  packaged `Ed` is piecewise-linear with anchors at 0/30/60°, so the
  θ_s-derivative is one-sided at the anchors (kink; record §4.4).
- **The extrapolation finding M4's candor must carry** (record §5.3,
  notebook 4 §3): a δ_R trained without 60° produces **−74 %** increment
  error there — worse than no correction. The heads are interpolators in
  cos θ_s; the 3-zenith caveat is now a *measured* cliff, not a
  hypothetical. (φ_C-linearity likewise has truth only at 0.02;
  `emission_shape='double'` is unvalidatable; below 400 nm the excitation
  clamps — the model's stated domain is λ ≥ 400 nm, three zeniths,
  L23-like water.)
- **Speed is unmeasured** — the M4 risk. The fluorescence kernel's
  `(n_batch, n_em, 65)` contraction and the Raman factor's excitation
  interpolation are the new cost; the coding-plan fallback (precompute
  zenith-static quantities; fuse the quadrature) waits on a measurement,
  not a guess.
- **Weight regeneration is safe**: `design/py/train_inelastic_corr.py`
  writes candidate-beside-destination, verifies the reloaded δ, then
  `os.replace`s (the PR #18 Bugbot finding, fixed at M3 task 5 with the
  elastic `write_weights` pattern). ~60 s/head; deterministic from seeds.
- **Review pattern**: JXP issues a PR per milestone and triggers
  `@cursor review`; expect Bugbot findings to address alongside the
  task-3 self-review.
- **Notebook tooling**: kernelspec `ocean14`; execute via the `os_313`
  env's `jupyter nbconvert --execute`; build programmatically (`nbformat`
  from os_313) in the house style of notebooks 1–4; commit with outputs.

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
   elastic hash-regression. **Reuse over re-derivation**: the deltas,
   gradients, and hash lines are already standing tests
   (`test_inelastic_corr.py`, `test_inelastic.py`,
   `test_inelastic_types.py`) — the protocol should *compute and report*
   them through `validation.py`'s metric machinery (`rrms` is the metric,
   already differentiable), not fork new definitions; the genuinely new
   measurements are the **total rRMS vs X4** and the **speed ratio**.
   Diagnostics (reported, not gated): performance by a_ph(440) decile
   (measured flat ±0.6 % at M3 — confirm through the protocol); the
   φ_C-linearity check against the scaled-truth construction;
   `emission_shape='double'` behavior. If the ≤ 2× speed budget is
   threatened, apply the coding-plan fallback (precompute zenith-static
   quantities at trace time; fuse the excitation quadrature) — measure
   first, the fluorescence contraction is the suspect.

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
   is declared. JXP's per-milestone pattern is a PR + `@cursor review`
   (M0's PR #14, M3's PR #18 — the Bugbot findings were real both times);
   address those comments too when the M4 PR appears, checking Q&A for
   pointers.

4. **Wrap-up.** Finish `design/rt_inelastic_implementation.md` (all
   milestones, environment, module index, results tables, the definition-
   of-done statement mirroring the elastic record §6.10). Notebook
   `notebooks/RT/rt_inelastic_coding_5.ipynb` — executed: the headline
   validation figures (total rRMS ladder incl. elastic-only baseline;
   per-process before/after; speed and gradient tables) with honest
   captions about what the prototype may and may not claim (mirror the
   elastic `prototype_summary.md` candor). The caveat list is now
   *measured*, not hypothetical — carry it verbatim from the record:
   the 3-zenith geometry cliff (−74 % at an unseen 60°, notebook 4 §3),
   φ_C-linearity with truth only at 0.02, `'double'` emission
   unvalidatable, λ ≥ 400 nm domain (excitation clamp), and the
   θ_s-anchor derivative kink. Note the branch state for JXP's PR.

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
