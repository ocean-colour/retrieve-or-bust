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

As in `rt_inelastic_coding_prompt_1.md`: JXP runs git — branch
**`inelastic-rt`** (JXP kept the existing name over the plan's
`rt-inelastic-prototype`; prompt 1 Q&A Q2); `ocean14` on this machine, CPU
JAX; reuse elastic-`robust`/`bing`/`ocpy` machinery and follow the recorded
conventions; every task `pytest`-gated; the **elastic hash-regression stays
green** (the two pinned SHA-256s in `robust/tests/test_inelastic_types.py`).
Never `pip install -r requirements.txt` wholesale on this machine — its
`git+` lines would clobber the editable `bing`/`ocpy` checkouts (Q1,
confirmed). Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — M1 section + Package layout.
- **Design** — `design/rt_inelastic_model.md` §4.1 (truth channels), §4.2
  (Ed module + the solar-model caveat), §3 (excitation-grid internals,
  λ ≥ 400 nm support).
- **Elastic data module** — `robust/rt/data/l23.py` (the loader and split
  machinery to extend, *not* replace) and `robust/tests/conftest.py`
  (`needs_l23`, fixtures dir).
- **BING reference** —
  `/mnt/tank/Oceanography/python/bing/bing/tests/files/gen_l23_inelastic_fixture.py`
  (the fixture-generator pattern; scene-independence assert for Ed; verified
  present on the `inelastic-fixes` checkout, alongside its committed
  `l23_inelastic_fixture.npz`).

## Status entering M1

*(Filled by M0's final task, 2026-08-21. Details:
`design/rt_inelastic_implementation.md` v0.4 §2; chronology: prompt 1 Logs.)*

**M0 is complete — all five tasks, gate green.** What M1 can rely on:

- **Environment (tank server, `ocean14`).** jax/jaxlib **0.11.1** (CPU),
  flax 0.12.9, optax 0.2.8, jaxtyping 0.3.11, ipykernel 7.3.0 — all verified
  (backend, x64, grad, jit; pre-existing stack untouched). The machine has an
  NVIDIA GPU; jax's "falling back to cpu" notice is normal log noise. The
  full suite runs here: **309 passed** (279 elastic unmodified + 30 M0),
  `ruff check` + `ruff format --check` clean. `$OS_COLOR` is set
  (`/home/xavier/Oceanography/data/Color`), so `needs_l23` tests run rather
  than skip.
- **The API M1 extends is pinned.** `Inelastic(phi_C=0.02, raman=True,
  fluorescence=True, emission_shape='single', cdom_fl=None)` — `phi_C` a
  pytree leaf, the switches static; `IOPs.a_ph` and `Geometry.Ed` optional
  with `None` defaults (no leaves unset); `forward`/`rrs_forward` take
  **keyword-only** `inelastic=None` (the `mode` positional slot was taken).
  Passing an instance raises `NotImplementedError` until M2 — M1 does not
  change that.
- **The recurring gate is armed.** Two SHA-256 pins (Rrs and rrs on the
  50-scene fixture, `check_domain=False`) in `test_inelastic_types.py`,
  computed pre-change and platform-anchored to this machine — JXP has
  acknowledged the cross-platform caveat (Q2). Every M1 task ends with these
  green.
- **The M2 cross-check dependency is in place.** `bing` is an *editable*
  install from `/mnt/tank/Oceanography/python/bing`, checkout on
  **`inelastic-fixes`** — verified. `bing.rt.raman` exposes
  `excitation_to_emission_wavelength` / `emission_to_excitation_wavelength`
  and `WAVENUMBER_SHIFT_CENTER = 3400.0`.
- **The X2/X4 data is on disk here** (verified 2026-08-21):
  `Hydrolight{200,230,260}.nc` and `Hydrolight{400,430,460}.nc` (plus
  `_profile` siblings) in the L23 directory beside the elastic three.
- **Notebook tooling.** Kernelspec **`ocean14`** is registered (the elastic
  notebooks ran on the laptop; this machine had none). Execute with the
  `os_313` env's `jupyter nbconvert --execute` — it launches the `ocean14`
  kernel — and commit with that kernelspec, as notebook 1 does.
- **One known gap M1 must close** (task 3): `conftest.needs_l23` guards only
  the *elastic* X=1 files (`L23_ELASTIC_FILES`). Tests reading the raw X2/X4
  netCDFs need their own guard (e.g. a `needs_l23_inelastic` marker over the
  six files above) so a machine with partial data skips cleanly instead of
  failing.

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

   **Test:** the emission for 488 nm excitation is **585.08 nm** and the
   excitation for 488 nm emission is **418.55 nm** (computed from the exact
   wavenumber form; cross-check the round trip against
   `bing.rt.raman.excitation_to_emission_wavelength` /
   `emission_to_excitation_wavelength`). *Correction from M0's task 5: this
   line originally said "λ′(488) ≈ 583.6 nm" — wrong; even bing's own
   docstring example ("583.0 # approximately") is off. 1e7/488 − 3400 =
   17091.8 cm⁻¹ → 585.076 nm; pin the computed value, not a prose
   approximation.* Interpolation matches `numpy.interp`; `jax.grad` through
   the interpolation is finite and matches finite differences (float64 via
   the `jax_x64` fixture, dtypes pinned on the arrays — elastic record §2).

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
   on golden fixture rows cross-checked against the raw netCDF (guarded by
   the **new X2/X4 marker** — see Status: `needs_l23` covers only the
   elastic files, so add e.g. `needs_l23_inelastic` over the six inelastic
   netCDFs rather than overloading it); split-index equality with the
   elastic splits; fixture file small (**≲ 300 kB** — 18 float32 arrays of
   50×81; the elastic fixture's 15 arrays weigh 213 kB, so the plan's
   original ≲ 200 kB bound was arithmetically impossible) and loadable
   without `$OS_COLOR`. Update the implementation record; note the branch
   state for JXP.

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
