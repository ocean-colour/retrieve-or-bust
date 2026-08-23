# Inelastic RT Coding — Prompt 3 (M2: Analytic terms in JAX)

## Goals

Implement **Milestone M2** of the coding plan
(`design/rt_inelastic_model_coding_plan.md`): the analytic Raman factor and
the φ_C-linear fluorescence kernel as pure JAX functions — ports of the
**fixed** BING physics — composed into `forward()` per the design's
composition law, pinned against live BING at rtol ≤ 1e-6, and characterized
against the L23 truth (the known error table the M3 heads must then fix).

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `critical-partner`,
`code-review`) as helpful.

### Working agreements

As in `rt_inelastic_coding_prompt_1.md`: JXP runs git (branch
`rt-inelastic-prototype`); `ocean14` on this machine, CPU JAX; reuse over
reinvention; every task `pytest`-gated; the **elastic hash-regression stays
green**. Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — M2 section (incl. the BING-branch risk + sentinel test).
- **Design** — `design/rt_inelastic_model.md` §2 (composition), §4.3–4.4
  (the term-by-term physics, constants, and the *measured* backbone errors),
  §4.6 (batching/gradients).
- **The fixed BING source** — `bing/rt/rrs.py`
  (`calc_raman_correction_factor`, `calc_Rrs_fluorescence` — docstrings
  carry the 1/π derivation and L23 validation numbers), `bing/rt/raman.py`,
  `bing/rt/chl_fl.py`. This machine's `bing` checkout must contain the
  `inelastic-fixes` work (PR issued 2026-08-20).
- **The assessment error table** —
  `context/RT/rt_inelastic_bing_summary.md` §4 and
  `context/RT/rt_inelastic_metrics.csv`.

## Status entering M2

*(Filled by M1's final task.)*

## Prompts

1. Read this doc. Execute the 1st task in the "M2" section below. If you have
   any questions, ask me in the Q&A section below. Use Fable if you can. Log
   your work.
2. Read this doc. Execute the 2nd task. Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task. Check my answers in Q&A; if you have
   additional questions, ask in Q&A. Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can.
   Log your work.
5. Read this doc. Execute the 5th task — modifying the next prompt doc,
   `rt_inelastic_coding_prompt_4.md`. Use Fable if you can. Log your work.

## M2

### Tasks

1. **Raman factor.** `robust/rt/inelastic.py::raman_factor(iops, geometry,
   wave) → f_phys(λ)`: b_R(λ′) (Bartlett 1998; b_R(488) = 2.6e-4 m⁻¹,
   energy exponent 5.5); single-shift excitation grid + IOP interpolation
   (M1 helpers); S&P98 first- + second-order two-flow terms
   (μ_d = 0.9, μ_u = μ_R = 0.5); **true Ed ratio** from `robust.rt.ed`;
   assembled as `f_phys = (R_E + R_Raman)/R_E`. Wire into
   `hybrid.forward` as `× f_R` behind `Inelastic.raman` (with
   `f_R = f_phys` until M3 adds δ_R).

2. **Fluorescence kernel.** `robust/rt/inelastic.py::fluorescence_kernel
   (iops, geometry, wave) → K_fl(λ)` with φ_C factored out
   (`Rrs_fl = φ_C · K_fl`): source a_ph(λ′) over the 370–690 nm excitation
   quadrature; λ′/λ quanta→energy factor; per-λ_em κ_F; **L_u = E_u/π**;
   A·rrs/(1−B·rrs) with A = 0.52, B = 1.7. Emission line: single Gaussian
   (685 nm, σ = 10.6 nm) default; `emission_shape='double'` implemented but
   flagged un-validatable (design §4.4). Wire into `forward` as
   `+ φ_C·K_fl` behind `Inelastic.fluorescence` (requires `a_ph`; raise a
   clear error if absent).

3. **Cross-check + characterization + gradients.**
   - `robust/tests/test_inelastic_bing_xcheck.py` (module `skipif` when
     `bing` is unavailable — CI): **sentinel first** — evaluate BING's
     fluorescence on a known input and assert the post-fix normalization
     (fails loudly with "BING checkout predates inelastic-fixes" if the π
     fix is missing), then pin `raman_factor` vs
     `bing.rt.rrs.calc_raman_correction_factor` and `φ_C·K_fl` vs
     `bing.rt.rrs.calc_Rrs_fluorescence` at **rtol ≤ 1e-6** on the fixture
     scenes, all three zeniths.
   - `test_inelastic.py`: reproduce the assessment's error table against
     the fixture truth channels with banded tolerances (Raman increment
     error ≈ +1 %/−4 % at 30°/60°, ≈ −39 % at 0°; fluorescence 685 nm
     ratio ≈ 1.00/0.95/0.86) — these *characterize* the backbone; the test
     asserts the band, not zero error.
   - Gradients: `jax.grad` vs central finite differences (float64) for
     `a, bb_p, a_ph, φ_C, θ_s` through the full composed `forward`.

   **Gate.** All three test groups green (xcheck green *on this machine*);
   elastic hash-regression still green. Update the implementation record.

4. **Notebook.** `notebooks/RT/rt_inelastic_coding_3.ipynb` — executed.
   Show: f_phys vs the L23 Raman truth per zenith (the −39 % @ 0° gap that
   M3 must close); φ_C·K_fl vs the fluorescence truth (amplitude ~right,
   trophic drift visible); a ∂Rrs/∂φ_C spectrum (the new physiology
   handle). Explain the composition law and why Raman is a ratio while
   fluorescence is additive.

5. **Finally.** Modify the next prompt doc,
   `rt_inelastic_coding_prompt_4.md`, given what M2 actually established.
   Log your work.

### Q&A

## Next

→ `rt_inelastic_coding_prompt_4.md` (M3: correction heads).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
