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

As in `rt_inelastic_coding_prompt_1.md`: JXP runs git — branch
**`inelastic-rt`** (Q&A Q2 of prompt 1); `ocean14` on this machine, CPU JAX;
reuse over reinvention; every task `pytest`-gated; the **elastic
hash-regression stays green** — now two-tier (record §2.8): the strict
SHA-256 pins run on dev machines and skip under `$CI`, the ULP-closeness
regression against `robust/tests/files/elastic_reference_outputs.npz` runs
everywhere. Never `pip install -r requirements.txt` wholesale here (Q1: the
`git+` lines would clobber the editable `bing`/`ocpy` checkouts). Use Fable
if you can. Log your work.

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

*(Filled by M1's final task, 2026-08-24. Details:
`design/rt_inelastic_implementation.md` v0.12 §3; chronology: prompt 2 Logs.)*

**M1 is complete — all five tasks, gate green** (355 passed; the two-tier
elastic hash gate green; ruff clean; notebooks 1–2 executed and committed).
What M2 can rely on, and what it must not break:

- **The consumables are in place, tested, differentiable.**
  - `robust.rt.ed`: `Ed(theta_s, wave, override=None)` and
    `ratio(theta_s, wave_num, wave_den, override=None)` — packaged L23
    spectra (2 kB `ed_l23.npz`), linear-in-θ_s, clamped at every edge;
    the ratio is guaranteed single-sky. Measured: `Ed(λ′)/Ed(λ)` spans
    0.44–1.59 (×3.6) over the official band and is nearly
    zenith-independent.
  - `robust.rt.conventions`: `RAMAN_SHIFT = 3400.0`,
    `raman_excitation`/`raman_emission` (488 em ← 418.553 ex; 488 ex →
    585.076 em; exact inverses; equal to bing's pair), and
    `interp_spectrum(wave_new, grid, spectra)` — clamped-linear, batched,
    differentiable in the spectrum values, dtype-promoting. This is *the*
    interpolation for excitation-grid IOPs; do not write another.
    `RAMAN_WAVE_MIN_OFFICIAL = 400.0` (352.11 nm excitation edge inside the
    grid; below 400 the machinery clamps — caveat, not error).
  - `robust.rt.data.l23`: `load_inelastic_batch` / `L23InelasticBatch` with
    `iops.a_ph` set, truth-channel properties (`truth_raman_factor =
    Rrs_X2/Rrs_X1`, `truth_fluorescence = Rrs_X4 − Rrs_X2`),
    `PHI_C_L23 = 0.02`, `select_inelastic`, and elastic-split reuse proved
    mask-for-mask. Fixture-fed everywhere via
    `inelastic_npz_reader(L23_INELASTIC_FIXTURE, L23_SMALL_FIXTURE)`
    (conftest exposes both paths and a session fixture
    `l23_small_inelastic_batch`).
- **Measured truths to code against** (all pinned by tests): IOPs and
  `aph` are bit-identical across X=1/2/4; the Raman factor is ≥ 1.0076
  everywhere, largest for the *clearest* water, max 1.36 on the 50-scene
  fixture vs 2.51 on the full release — so **recompute characterization
  bands on whatever set the test actually uses** rather than assuming the
  assessment's full-set numbers transfer; the 685 nm fluorescence delta is
  strictly positive in every sample.
- **The M0 guard M2 must consciously replace**: `forward(...,
  inelastic=<instance>)` currently raises `NotImplementedError` naming M2,
  and `test_inelastic_types.py::test_inelastic_instance_raises_until_m2`
  pins that. Task 1/2 replace the raise with the composition — update that
  test *deliberately* (e.g. to assert the composed path instead), and touch
  nothing on the `inelastic=None` route: the hash pins must stay green
  bit-for-bit.
- **The `bing` cross-check target is ready**: editable checkout on
  `inelastic-fixes`, `calc_raman_correction_factor` /
  `calc_Rrs_fluorescence` present in `bing/rt/rrs.py` (verified). Its
  wavenumber machinery matches ours exactly (rtol 1e-12 tests).
- **Review debt**: robust/-side cleared (record §3.2.1); still open in
  `context/RT` only (the figure script's env-var lookup and the CSV column
  mislabels — Cursor findings 4–5), untouched by M2 unless the assessment
  material is regenerated.
- **Notebook tooling**: kernelspec `ocean14`; execute via the `os_313`
  env's `jupyter nbconvert --execute`; commit with outputs.

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
   wave) → f_phys(λ)`: b_R(λ′) with **b_R(488) = 2.6e-4 m⁻¹ — the
   HydroLight value, `bing.rt.raman.B_RAMAN_488_HYDROLIGHT`, bing's
   default** *(correction from M1's task 5: this line originally attributed
   2.6e-4 to "Bartlett 1998", but Bartlett is 2.7e-4
   (`B_RAMAN_488_BARTLETT`); 2.6e-4 is what HydroLight — and therefore the
   L23 truth — used, which is why it must be the default here)*; energy
   exponent 5.5; single-shift excitation grid + IOP interpolation via the
   M1 helpers **by name**: `conventions.raman_excitation` +
   `conventions.interp_spectrum` (do not reimplement either); S&P98
   first- + second-order two-flow terms (μ_d = 0.9, μ_u = μ_R = 0.5);
   **true Ed ratio** from `robust.rt.ed.ratio` (pass `geometry.Ed` as the
   override so the seam works end to end); assembled as
   `f_phys = (R_E + R_Raman)/R_E`. Wire into `hybrid.forward` as `× f_R`
   behind `Inelastic.raman` (with `f_R = f_phys` until M3 adds δ_R) —
   replacing the M0 `NotImplementedError` guard; update its pinned test
   deliberately (see Status).

2. **Fluorescence kernel.** `robust/rt/inelastic.py::fluorescence_kernel
   (iops, geometry, wave) → K_fl(λ)` with φ_C factored out
   (`Rrs_fl = φ_C · K_fl`): source a_ph(λ′) over the 370–690 nm excitation
   quadrature; λ′/λ quanta→energy factor; per-λ_em κ_F; **L_u = E_u/π**;
   the surface transfer via `conventions.rrs_to_Rrs` (A = 0.52, B = 1.7 —
   already pinned equal to bing's; do not re-spell the formula). Emission
   line: single Gaussian
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
     asserts the band, not zero error. *Caveat from M1: those numbers were
     measured on the full release / bing's 40-scene fixture; our 50-scene
     fixture samples a narrower range (Raman factor max 1.36 vs 2.51), so
     compute the actual medians on the set the test uses first and set the
     bands around* **those**, *logging any material difference from the
     assessment's table rather than force-fitting it.*
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

**Q1 (Claude, 2026-08-24, task 1).** Housekeeping, not physics: your staging
area contains a new file **`rob/__init__.py`** (empty, in a stray `rob/`
directory at the repo root) — staged (`git status` shows `A`) but not created
by any prompt work. It looks like an accidental `mkdir rob` + add (perhaps a
truncated `robust`?). Please unstage/remove it (or tell me its purpose) so it
doesn't ride along into the next commit. Everything else in this task raised
no questions.

*Follow-up (Claude, 2026-08-24, task 2).* No answer was recorded, and commit
`2c1c259` meanwhile brought `rob/` in — the empty `__init__.py` **plus**
`rob/data/Dutkiewicz2015/optics_phyo_8_1nm.txt` (3631 lines of phytoplankton
optics). The data file suggests it may be deliberate after all; if so, ignore
this — if not, it now takes a `git rm` rather than an unstage. Either way a
one-word answer here closes the question. Task 2 itself raised no new
questions (the one open design point — how `K_fl` stays φ_C-linear while the
surface transfer is nonlinear — is answered by design §4.4 itself: the kernel
is defined at the truth's φ_C = 0.02, `K_fl = Rrs_fl(0.02)/0.02`; logged
below and in the record §4.3).

## Next

→ `rt_inelastic_coding_prompt_4.md` (M3: correction heads).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-24 (M2 task 1 — raman_factor ported, composed, characterized; 368 green)

The analytic Raman factor is in and wired; record §4.1–4.2 (v0.14). Model:
Fable 5. Q&A: doc 2's is empty; prompt 6 of doc 2 was read as this doc's
prompt 1 (doc 2 has only five prompts — same convention as the M0→M1
handoff). **One new Q&A above (Q1): a stray staged `rob/__init__.py` in your
index — please unstage.**

- `robust/rt/inelastic.py::raman_factor`: S&P98 Eqs. 5/11/18/23 term-for-term
  from fixed BING; `raman_bb = ½·2.6e-4·(488/λ′)^5.5` (HydroLight constant
  per the M1 correction; analytic ½ where bing integrates to 0.5±2e-7);
  excitation IOPs via `conventions.interp_spectrum`; **true Ed ratio** via
  `ed.ratio(..., override=geometry.Ed)` — the M1 seam works end to end
  (tested: a flat-sky override moves the red factor the documented
  direction).
- Wiring: `_apply_inelastic` composes in Rrs space per design §2;
  `inelastic=None` returns the same object (hash pins green throughout);
  all-off is bitwise elastic; fluorescence (the default!) still raises
  naming "M2 task 2"; `mode='emulator'` refuses composition (ValueError) —
  a term, not a model. The M0 guard test needed only a docstring update.
- **Port quality, measured**: max rel vs live bing **1.6e-7** (gate: 1e-6).
  Characterization vs fixture truth (median increment error, 550–700 nm):
  **+1.6 % / −4.0 % / −38.6 %** at 30/60/0° — the assessment's +1/−4/−39
  table reproduced on our own fixture, retiring the M1 band-transfer
  concern. Pinned as bands; the 0° failure is δ_R's target, not a bug.
- 13 tests (constants/xcheck spot/physicality/bands/seam/jax/wiring); suite
  **368 passed** warning-free (two tests first requested jnp.float64 outside
  the x64 fixture and were silently truncated — the dtype lesson's fourth
  appearance; reference arithmetic moved to NumPy); ruff clean.

Task 2 (fluorescence kernel) starts with the composition seam already in
place — `_apply_inelastic` grows one `+ φ_C·K_fl` line before the
down-conversion.

### 2026-08-24 (M2 task 2 — fluorescence_kernel ported, composed, characterized; 381 green)

The φ_C-linear fluorescence kernel is in and wired; record §4.3 (v0.15).
Model: Fable 5. Q&A: no answer to Q1 was found; meanwhile commit `2c1c259`
*committed* the stray `rob/` (with a Dutkiewicz-2015 optics table, so perhaps
deliberate) — **follow-up appended under Q1, one word closes it**. No new
physics questions. (This prompt arrived as "execute the 1st prompt" with
task 1 already done and logged — read as the next unexecuted one, the same
offset convention both prior hand-offs used.)

- `robust/rt/inelastic.py::fluorescence_kernel`: fixed BING's
  `calc_Rrs_fluorescence` per unit yield — `b_bF = ½·φ_C·a_ph(λ′)`, fixed
  65-node 370–690 nm trapezoid quadrature (`fl_excitation_grid`; fixed so
  shapes are static under `jit` and bing can be fed identical nodes), λ′/λ
  quanta→energy, true `Ed(λ′)/Ed(λ_em)` single-sky via the override seam,
  per-λ_em κ_F, **L_u = E_u/π**, `h_C` via `emission_line` (single default;
  `'double'` implemented, flagged unvalidatable), surface transfer via
  `conventions.rrs_to_Rrs` (not re-spelled).
- **The φ_C-linearity mechanics, made concrete** (design §4.4): the kernel
  is evaluated at `PHI_C_REF = 0.02` and divided by it, so `φ_C·K_fl` is
  *exactly* bing at the truth's yield and linear-by-construction elsewhere;
  the neglected `(1 − B·rrs)` piece is O(10⁻³). `PHI_C_REF == PHI_C_L23`
  pinned by test.
- Wiring: `_apply_inelastic` is now the whole §2 law (up-convert once,
  `× f_R`, `+ φ_C·K_fl` with `phi_C` leaf-aligned, down-convert once);
  `None`/all-off still return the same object — hash pins green. The
  task-1 `NotImplementedError` guard retired for the **a_ph requirement**:
  clear `ValueError` at both `rrs_forward` (fast, pre-emulator — where the
  old guard sat) and the kernel; the M0 guard test updated deliberately to
  `test_inelastic_fluorescence_without_aph_raises`.
- **Port quality, measured**: float32 agreement vs live bing ~3e-6 (65-node
  trapezoid accumulation); **float64 exact to ~7e-16** — task 3's xcheck
  pins the 1e-6 gate under x64. Characterization vs fixture truth (median
  model/truth at 685 nm): **0.991 / 0.937 / 0.853** at 0/30/60° — the
  assessment's 1.00/0.95/0.86 on our own fixture; pinned as bands
  (π-regression or flat-Ed fails them instantly); the zenith drift is δ_F's
  target (M3), not a bug.
- 13 tests (constants/emission/xcheck spot incl. double/physicality/bands/
  seam/jax/wiring/linearity/∂Rrs/∂φ_C/gradients incl. a_ph); suite
  **381 passed** warning-free (also `-W error`); ruff clean.

Task 3 (formal xcheck + error table + FD gradients) starts with both terms
composed and spot-checked; the sentinel (bing-predates-fixes) and the x64
rtol ≤ 1e-6 pins are its first moves, and the characterization bands above
are the fixture-measured numbers it should build on.
