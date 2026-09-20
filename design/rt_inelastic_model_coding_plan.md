# Coding Plan — Inelastic RT Forward Model

*Staged, milestone-gated implementation plan for the differentiable inelastic-RT
extension specified in [`design/rt_inelastic_model.md`](rt_inelastic_model.md).*

This is the build plan for the **~1-week inelastic prototype**: Raman scattering
and chlorophyll-a fluorescence added to the elastic hybrid as
`Rrs_total = (Rrs_ZTT + ΔRrs) × f_R + Rrs_fl`, with analytic (fixed-BING-ported)
backbones and two bounded learned correction heads, gated by the design §6
acceptance criteria. It turns the design into concrete milestones, each with
tasks, deliverables, and a **pytest acceptance gate**.

## Ground rules (from Q&A/Coding)

- **Structure (CQ1).** Five milestones **M0–M4**, one prompt doc each
  (`claude_prompts/RT/rt_inelastic_coding_prompt_1..5.md`); a coarse
  "beyond v1" section here only. ~1 week of prototype work.
- **Machine & environment (CQ2).** **This machine (tank server)**; M0 installs
  the JAX stack into `ocean14` here (it is currently absent — verified
  2026-08-20). The stack is already declared in `requirements.txt` from the
  elastic effort; M0 repeats the elastic install procedure (dry-run first,
  purely-additive verification).
- **BING cross-check (CQ3).** Option **(a)**: the M2 cross-check tests
  **import `bing` directly** and compare the JAX ports against the fixed BING
  implementations at runtime. JXP is issuing the PR for the BING
  `inelastic-fixes` work now; the cross-check requires that code on the
  active BING checkout (see Risks). The tests `skipif` when `bing` (or its
  fixed functions) is unavailable — e.g. on GitHub CI — so CI stays green
  while local runs enforce the pin.
- **CI fixture (CQ4).** A **sibling** fixture file with the *same 50 scene
  indices* as the elastic fixture, adding the X2/X4 channels + `a_ph` + Ed;
  the elastic fixture's bytes are untouched.
- **Layout (CQ5).** Training scripts in `design/py/`; validation artifacts in
  `design/validation/`; narrative in a new
  `design/rt_inelastic_implementation.md` (mirroring the elastic record);
  trained head weights committed under `robust/rt/files/`. `report/` waits
  for the end-of-effort report prompt.
- **Branch & review (CQ6).** JXP creates **`rt-inelastic-prototype`** and runs
  all git (per `CLAUDE.md`); milestones are reviewable commits/PRs; a code
  review pass is an **M4 task** before the gate is declared. No CI-workflow
  changes: new tests join the existing pytest tree.

## Package layout (extends the elastic layout)

```
robust/rt/
  types.py             # EXTEND: IOPs += a_ph; new Inelastic; Geometry += Ed
  conventions.py       # EXTEND: Raman shift const; excitation-grid helpers
  ed.py                # NEW: Ed(θ_s, λ) from packaged L23 spectra + override
  inelastic.py         # NEW: analytic f_phys (Raman) & K_fl (fluorescence);
                       #      composition into forward()
  inelastic_corr.py    # NEW: δ_R / δ_F Flax heads + training entry points
  hybrid.py            # EXTEND: forward(..., inelastic=None) wiring
  validation.py        # EXTEND: §6 inelastic protocol
  data/l23.py          # EXTEND: X2/X4 loaders, truth channels, elastic splits
  data/ed_l23.npz      # NEW: the three L23 Ed(0+) spectra
  files/raman_corr_l23.npz, fl_corr_l23.npz   # NEW: trained head weights
robust/tests/
  test_inelastic_types.py, test_ed.py, test_inelastic.py,
  test_inelastic_bing_xcheck.py, test_inelastic_corr.py,
  test_inelastic_validation.py
  files/l23_inelastic_fixture.npz             # NEW sibling fixture (CQ4)
design/py/
  gen_inelastic_fixture.py, train_inelastic_corr.py,
  run_validation.py (EXTEND)
```

---

## Milestones at a glance

| M | Goal | pytest acceptance gate |
|---|------|------------------------|
| **M0** | Env (this machine) + API extension | jax(CPU) imports in `ocean14`; `inelastic=None` **bit-identical** to elastic hybrid; all elastic tests still green |
| **M1** | Ed module + excitation grid + X2/X4 data | packaged Ed matches L23 (scene-independence asserted); loaders/truth channels round-trip; sibling fixture golden values; splits == elastic splits |
| **M2** | Analytic terms in JAX | cross-check vs fixed-BING at **rtol ≤ 1e-6** (live `bing` import, CQ3a); reproduces the assessment's error table vs truth; grad vs finite-diff incl. **∂/∂φ_C** |
| **M3** | Correction heads δ_R, δ_F | **per-process delta gates on held-out scenes**: Raman increment median \|err\| ≤ 5 % (550–700 nm) at *every* zenith incl. 0°; fluorescence 685 nm peak median \|err\| ≤ 5 % |
| **M4** | Validation (**prototype done**) | full design §6 gate: total X4 rRMS ≤ 0.5 %/zenith; deltas ≤ 5 %; elastic hash-regression; grads; speed ≤ 2× elastic; review pass done; metrics + figures committed |

---

## M0 — Environment & API extension

**Tasks.**
- Install the JAX stack into `ocean14` **on this machine** (CQ2):
  `pip install --dry-run -r requirements.txt` (verify purely additive — the
  elastic effort's procedure), then install; verify `jax.default_backend() ==
  "cpu"`, float64 via `jax_enable_x64`, `jax.grad` smoke, and that the
  pre-existing stack (`numpy, scipy, xarray, bing, ocpy, emcee`) still
  imports.
- `types.py`: `IOPs` gains optional `a_ph` (default `None`); new
  `Inelastic(phi_C=0.02, raman=True, fluorescence=True,
  emission_shape='single', cdom_fl=None)` pytree; `Geometry` gains optional
  `Ed=(wave_Ed, Ed)` override. All strictly backward compatible.
- `hybrid.forward(..., inelastic=None)`: `None` → the existing elastic path,
  untouched code route (no-op guarantee by construction, not by arithmetic).

**Deliverable.** Extended, importable API; elastic behavior provably unchanged.
**Gate.** `test_inelastic_types.py`: pytree flatten/unflatten, defaults;
**elastic regression**: `forward(..., inelastic=None)` output on the elastic
CI fixture is **bit-identical** (hash) to the pre-change hybrid; the full
existing elastic test suite passes unmodified; env checks green.

## M1 — Ed module, excitation grid, X2/X4 data

**Tasks.**
- `design/py/gen_inelastic_fixture.py` + `robust/rt/data/ed_l23.npz`: extract
  `Ed(0⁺)(λ)` from `Hydrolight2{00,30,60}.nc`, asserting scene-independence
  (< 10⁻³ relative scatter) before collapsing to one spectrum per zenith.
- `ed.py`: `Ed(θ_s, λ)` — packaged spectra, linear interpolation in θ_s over
  0–60°, `Geometry.Ed` override; ratio helper `Ed(λ′)/Ed(λ)`.
- `conventions.py`: Raman shift (3400 cm⁻¹) constants; excitation-grid map
  λ′(λ); JAX-friendly linear interpolation of IOP spectra onto λ′
  (differentiable w.r.t. the IOP values).
- `data/l23.py`: X2/X4 loaders (`load_ds(2|4, Y)`); truth-channel builders
  (`Rrs_X2/Rrs_X1`, `Rrs_X4−Rrs_X2`, `Rrs_X4`); `a_ph` extraction; **reuse
  the elastic splits objects verbatim** (same seed/indices).
- Sibling CI fixture `robust/tests/files/l23_inelastic_fixture.npz` (CQ4):
  the elastic fixture's 50 scene indices × {a, a_ph, bb, Rrs_X1, Rrs_X2,
  Rrs_X4} at all three zeniths + the three Ed spectra.

**Deliverable.** One-call inelastic data pipeline + packaged Ed.
**Gate.** `test_ed.py` (packaged vs raw netCDF at rtol 1e-5; interpolation
endpoints exact; override plumbs through); `test_l23_inelastic_data.py`
(shapes; truth-channel identities on fixture golden rows; split indices
identical to the elastic effort's).

## M2 — Analytic terms in JAX

**Tasks.**
- `inelastic.py::raman_factor`: b_R(λ′) (Bartlett, energy exponent 5.5,
  b_R(488) = 2.6e-4), single-shift excitation grid, S&P98 first- +
  second-order two-flow terms (μ_d = 0.9, μ_u = μ_R = 0.5), **true Ed
  ratio**, assembled as the ratio `f_phys = (R_E + R_Raman)/R_E`.
- `inelastic.py::fluorescence_kernel`: `K_fl` with b_F = a_ph(λ′) source
  (φ_C factored out), 370–690 nm excitation quadrature, λ′/λ, per-λ_em κ_F,
  **L_u = E_u/π**, A·rrs/(1−B·rrs); single-Gaussian emission (`'double'`
  switchable, untested against L23); `Rrs_fl = φ_C · K_fl`.
- Composition in `hybrid.forward` per design §2.

**Deliverable.** Differentiable analytic inelastic terms, composed.
**Gate.** `test_inelastic_bing_xcheck.py` (CQ3a): imports `bing` and pins
`raman_factor` vs `bing.rt.rrs.calc_raman_correction_factor` and
`φ_C·K_fl` vs `bing.rt.rrs.calc_Rrs_fluorescence` at **rtol ≤ 1e-6** on the
fixture scenes (module `skipif` when `bing`/fixed functions are absent — CI);
`test_inelastic.py`: reproduces the assessment's error table vs fixture truth
(Raman increment error ≈ +1 %/−4 % at 30°/60°, ≈ −39 % at 0°; fluorescence
685 nm ratio ≈ 1.00/0.95/0.86 — banded tolerances, these *characterize* the
backbone, they don't gate to zero); `jax.grad` vs central differences for
`a, bb_p, a_ph, φ_C, θ_s`.

## M3 — Correction heads δ_R and δ_F

**Tasks.**
- `inelastic_corr.py`: two small Flax MLPs (design §4.5 features; bounded
  tanh outputs; O(10²–10³) params each — size chosen by held-out validation,
  start minimal): `f_R = 1 + (f_phys − 1)(1 + δ_R)`;
  `Rrs_fl = φ_C · K_fl · (1 + δ_F)` (δ_F independent of φ_C).
- `design/py/train_inelastic_corr.py`: Optax training on the train split,
  all three zeniths; targets = the X2/X1 and X4−X2 truth channels;
  **relatively weighted** losses (the elastic/BING lesson); fixed seeds.
- Commit trained weights `robust/rt/files/{raman,fl}_corr_l23.npz`.

**Deliverable.** Trained heads; `forward()` complete.
**Gate.** `test_inelastic_corr.py` on **held-out scenes**: Raman increment
median |error| ≤ **5 %** over 550–700 nm at each zenith **including 0°** (the
analytic backbone fails this at 0° by −39 % — this line is what δ_R earns);
fluorescence 685 nm peak median |error| ≤ **5 %** per zenith; loading
committed weights reproduces the metrics (no train-at-test-time).

## M4 — Validation — *prototype complete*

**Tasks.**
- `validation.py`: the design §6 protocol — per-λ/per-zenith rRMS vs
  `Rrs_X4`; per-process delta metrics; elastic hash-regression; gradient
  gate incl. φ_C; throughput vs the elastic hybrid. Diagnostics (reported,
  not gated): a_ph(440)-decile performance; φ_C-linearity check against the
  scaled-truth construction; behavior with `emission_shape='double'`.
- Extend `design/py/run_validation.py` → metrics table + figures into
  `design/validation/`.
- **Review pass (CQ6)** over the branch diff; findings fixed before the gate
  is declared.
- Write `design/rt_inelastic_implementation.md` (the narrative record,
  mirroring the elastic one).

**Deliverable + Definition of Done.** Prototype done when the M4 gate passes.
**Gate (acceptance = design §6).** (1) held-out total rRMS vs X4 ≤ **0.5 %**
at each zenith; (2) Raman delta ≤ 5 % incl. 0°; (3) fluorescence delta ≤ 5 %;
(4) `inelastic=None` bit-identical; (5) gradient checks pass for all inputs
incl. φ_C; (6) full-batch forward ≤ **2×** elastic-hybrid runtime; metrics +
figures committed and regenerable by `run_validation.py`.

## Beyond v1 (coarse — no prompt docs yet)

Per design §8: the HydroLight wishlist (denser zeniths → real
zenith-interpolation gates; varied φ_C → test the linearity bet; CDOM-fl
on/off pairs → populate the `cdom_fl` hook; off-nadir views; alternative
solar spectra → the DQ5 concern made testable; sub-350 nm). Plus real-sky Ed
through the `Geometry.Ed` seam and PS I double-Gaussian validation.
Milestones to be detailed once M4 results are in.

*Update (2026-08-29):* CDOM fluorescence now has its own design doc and
M5/M6 milestone plan in
[`design/rt_cdom_fluorescence_model.md`](rt_cdom_fluorescence_model.md),
executed by `claude_prompts/RT/rt_cdom_coding_prompt_1.md` (M5 now; M6
deferred until HydroLight CDOM-fl truth exists).

---

## Testing strategy

- **Layout/conventions:** as elastic — `pytest` in `robust/tests/`,
  `conftest.py`, fixtures under `files/`, ruff, light `jaxtyping`,
  CPU-deterministic (seeds; float64 for FD checks).
- **Recurring first-class gates:** elastic **hash-regression** (every
  milestone from M0 on), **gradient-correctness** (M2, M4), **golden values**
  against raw L23 netCDF (M1) and against live fixed-BING (M2, CQ3a).
- **Gating philosophy:** unlike the elastic plan's relative-only gates, the
  per-process deltas gate to **absolute** bars (≤ 5 %) because DQ6 fixed them
  explicitly; total-Rrs gates to ≤ 0.5 %. Diagnostics that lack truth data
  (φ_C-linearity, double-Gaussian) are reported, never gated.
- **bing-dependent tests** (`test_inelastic_bing_xcheck.py`) skip cleanly
  when `bing` is not importable (GitHub CI); they are mandatory-green on this
  machine before M2 closes.

## Requirements / dependency changes

None beyond the elastic effort's declarations: `requirements.txt` already
lists `jax`, `flax`, `optax`, `jaxtyping`. M0's job is *installing* that
stack into this machine's `ocean14` (absent as of 2026-08-20), not declaring
it. `bing` remains undeclared (dev-machine-only test dependency, CQ3a).

## Risks & de-risking

- **BING branch dependency (M2, CQ3a).** The cross-check needs the *fixed*
  BING (`inelastic-fixes`; PR in progress). If the active BING checkout lacks
  the fixes, the xcheck would pin against wrong physics — the test therefore
  first asserts a sentinel (e.g. the 1/π behavior on a known input) and
  fails loudly with "BING checkout predates inelastic-fixes" rather than
  comparing. If the PR merge slips, M2 proceeds against the branch checkout.
- **JAX install perturbing `ocean14` (M0).** Same mitigation as elastic:
  dry-run first; fall back to a dedicated env if not purely additive (the
  elastic install on the laptop was additive; expect the same here).
- **δ_R at zenith 0° (M3).** The −39 % backbone error is the largest lift.
  If a compact head can't reach 5 % at 0° with three zeniths of training
  data, options in order: add θ_s-interaction features, allow a modestly
  larger head, or (last) gate 0° at a relaxed bar with JXP's sign-off —
  flagged, not assumed.
- **Fluorescence at the eutrophic tail (M3).** ×2–3 amplitude drift where
  scenes are sparse; monitor the decile diagnostic during training, consider
  loss upweighting of the tail.
- **Speed (M4).** The excitation quadrature triples the per-λ work; if the
  2× budget is threatened, precompute zenith-static quantities (Ed ratios,
  bb_R) at trace time and keep the quadrature fused in one einsum.

## Definition of done (inelastic prototype)

**M4 gate passed:** `forward(iops, phase_params, geometry, wave, inelastic)`
reproduces the all-processes-on L23 ocean to ≤ 0.5 % held-out rRMS at every
zenith with per-process deltas ≤ 5 % (including Raman at high sun), leaves
elastic-only behavior bit-identical, is differentiable in everything
including φ_C, runs within 2× of the elastic hybrid, and ships with
committed weights, a validation table + figures, and an implementation
record — all `pytest`-green on branch `rt-inelastic-prototype` for JXP to
review and merge.
