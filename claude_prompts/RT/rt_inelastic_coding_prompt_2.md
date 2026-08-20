# Inelastic RT Coding — Prompt 2 (M1: Ed, excitation grid, X2/X4 data)

## Goals

Implement **Milestone M1** of the coding plan
(`design/rt_inelastic_model_coding_plan.md`): the solar-spectrum module, the
Raman excitation-grid infrastructure, the X2/X4 data pipeline with truth
channels on the **elastic splits**, and the sibling CI fixture. After M1,
everything the analytic terms (M2) and the correction heads (M3) consume
exists and is tested.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `critical-partner`,
`code-review`) as helpful.

### Working agreements

As in `rt_inelastic_coding_prompt_1.md`: JXP runs git (branch
`rt-inelastic-prototype`); `ocean14` on this machine, CPU JAX; reuse
elastic-`robust`/`bing`/`ocpy` machinery and follow the recorded conventions;
every task `pytest`-gated; the **elastic hash-regression stays green**.
Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — M1 section + Package layout.
- **Design** — `design/rt_inelastic_model.md` §4.1 (truth channels), §4.2
  (Ed module + the solar-model caveat), §3 (excitation-grid internals,
  λ ≥ 400 nm support).
- **Elastic data module** — `robust/rt/data/l23.py` (the loader and split
  machinery to extend, *not* replace) and `robust/tests/conftest.py`
  (`needs_l23`, fixtures dir).
- **BING reference** — `bing/tests/files/gen_l23_inelastic_fixture.py` (the
  fixture-generator pattern; scene-independence assert for Ed).

## Status entering M1

*(Filled by M0's final task.)*

## Prompts

1. Read this doc. Execute the 1st task in the "M1" section below. If you have
   any questions, ask me in the Q&A section below. Use Fable if you can. Log
   your work.
2. Read this doc. Execute the 2nd task. Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task. Check my answers in Q&A; if you have
   additional questions, ask in Q&A. Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can.
   Log your work.
5. Read this doc. Execute the 5th task — modifying the next prompt doc,
   `rt_inelastic_coding_prompt_3.md`. Use Fable if you can. Log your work.

## M1

### Tasks

1. **Ed module.** `design/py/gen_inelastic_fixture.py` (part 1): extract
   `Ed(0⁺)(λ)` from `Hydrolight2{00,30,60}.nc`, asserting scene-independence
   (relative scatter < 10⁻³) before collapsing to one spectrum per zenith;
   write `robust/rt/data/ed_l23.npz`. New `robust/rt/ed.py`: `Ed(θ_s, λ)`
   from the packaged spectra with linear θ_s interpolation over 0–60°, the
   `Geometry.Ed` override, and a ratio helper `Ed(λ′)/Ed(λ)`. Document the
   design §4.2 solar-model caveat in the module docstring.

   **Test:** `test_ed.py` — packaged vs raw netCDF (rtol 1e-5, `needs_l23`);
   interpolation exact at the three anchors; override plumbs through;
   ratio helper against a hand computation.

2. **Excitation-grid infrastructure.** In `robust/rt/conventions.py`: the
   Raman shift constant (3400 cm⁻¹), the excitation map λ′(λ) (wavenumber
   form), and a JAX-friendly linear interpolation of spectra onto λ′ that is
   **differentiable w.r.t. the spectrum values**. Enforce/document the
   λ ≥ 400 nm official support (excitation for 400 nm needs 352 nm — inside
   the L23 grid; below that, extrapolation with a documented caveat, no
   error).

   **Test:** λ′(488) ≈ 583.6 nm (cross-check
   `bing.rt.raman.excitation_to_emission_wavelength` round-trip);
   interpolation matches `numpy.interp`; `jax.grad` through the
   interpolation is finite and matches finite differences.

3. **X2/X4 data + sibling fixture.** Extend `robust/rt/data/l23.py`: load
   scenarios 2 and 4 for Y∈{0,30,60}; build the truth channels
   (`Rrs_X2/Rrs_X1`, `Rrs_X4−Rrs_X2`, `Rrs_X4`); extract `a_ph`; **reuse the
   elastic split objects verbatim** (same seed/indices — a test must prove
   it). `design/py/gen_inelastic_fixture.py` (part 2): the sibling CI
   fixture `robust/tests/files/l23_inelastic_fixture.npz` — the elastic
   fixture's 50 scene indices × {a, a_ph, bb, Rrs_X1, Rrs_X2, Rrs_X4} at all
   three zeniths (+ nothing else; Ed ships in the package data). Do **not**
   touch the elastic fixture's bytes.

   **Gate.** `test_l23_inelastic_data.py`: shapes; truth-channel identities
   on golden fixture rows cross-checked against the raw netCDF
   (`needs_l23` for the raw half); split-index equality with the elastic
   splits; fixture file small (≲ 200 kB) and loadable without `$OS_COLOR`.
   Update the implementation record; note the branch state for JXP.

4. **Notebook.** `notebooks/RT/rt_inelastic_coding_2.ipynb` — executed, with
   outputs. Show: the three Ed spectra and the Ed(λ′)/Ed(λ) ratio (why
   flat-Ed distorts the Raman correction); the truth channels for a few
   scenes across trophic states (how big each process is); the split reuse.
   Link back to the assessment report figures rather than re-deriving them.

5. **Finally.** Modify the next prompt doc,
   `rt_inelastic_coding_prompt_3.md`, given what M1 actually established.
   Log your work.

### Q&A

## Next

→ `rt_inelastic_coding_prompt_3.md` (M2: analytic terms in JAX).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
