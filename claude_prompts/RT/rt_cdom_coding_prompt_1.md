# CDOM Fluorescence Coding — Prompt 1 (M5: analytic term & interface)

## Goals

Implement **Milestone M5** of the CDOM-fluorescence design
(`design/rt_cdom_fluorescence_model.md`): the analytic CDOM-fluorescence
emission term on the Hawes et al. (1992) quantum-efficiency basis, the full
interface (`IOPs.a_cdom`, `CDOMFl(scale=1.0)`, the `Inelastic.cdom_fl` slot
retyped from reserved to live), and the correction head δ_C **defined but
untrained** — gated by the truth-less v1 acceptance criteria (design §5:
off-state bit-identity, correctness pins, literature-plausibility band,
gradients, speed). No head training in M5: no CDOM-fl truth exists anywhere in
hand (M6 is deferred until the HydroLight runs of design §7 land).

Work happens on a **fresh branch off `main`** (JXP creates it once
`inelastic-rt` is merged; the branch name is JXP's call — do not hardcode
one). The default model must remain provably CDOM-fl-free:
`Inelastic(..., cdom_fl=None)` (the default) stays **bit-identical** to the
shipped inelastic output, because the X4 truth and the reported 0.34 % gate
omit CDOM fluorescence.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `critical-partner`,
`code-review`) as helpful. Note: the `inelastic-rrs` skill documents **BING's**
Raman/Chl-fl wiring — BING has no CDOM-fluorescence implementation at all, so
that skill is background only here (there is no BING reference to cross-check
this term against).

### Working agreements (hold for the whole M5 prompt doc)

- **Git is handled by JXP** (per `CLAUDE.md`). Work on the fresh branch off
  `main` that JXP creates; the milestone is a reviewable commit/PR. Do **not**
  run state-changing git commands; read-only inspection is fine.
- **Python only**, in the `ocean14` conda env; **CPU-only JAX**.
- **Reuse, don't reinvent.** Build on the shipped `robust/rt` modules — the
  Ed module, the two-flow emission transport and quanta/energy bookkeeping in
  the Chl-fl kernel, the excitation-grid helpers, the pytree/validator
  patterns in `types.py`, and the two-tier (strict-hash / ULP-closeness)
  regression pattern. Do **not** install `requirements.txt` wholesale on a dev
  machine (its `git+` lines clobber editable `bing`/`ocpy` checkouts — prompt
  1 Q&A Q1 of the inelastic effort).
- **Every task is `pytest`-gated**, and from task 1 onward **both** existing
  hash-regressions — elastic (`inelastic=None`) *and* inelastic
  (`cdom_fl=None`, pinned in task 5) — must stay green.
  Use Fable if you can. Log your work.

## Context

Read before coding:

- **Design** — `design/rt_cdom_fluorescence_model.md` (all of it — it is
  short; §2 architecture, §3 interface, §5 the v1 gate, §8 risks).
- **Companion design** — `design/rt_inelastic_model.md` (§2 composition, §4.2
  Ed, §4.4 the Chl-fl kernel this term's transport mirrors, §4.5 the bounded
  head pattern).
- **Implementation record** — `design/rt_inelastic_implementation.md` (the
  conventions, environment, and gotchas the inelastic milestones established).

Where this sits: the inelastic prototype (Raman + Chl-a fluorescence) is
gate-passed and reported (`reports/report_rt_inelastic_model.md`, v1.0,
2026-08-27 — held-out 0.34 % rRMS vs X4, per-process ≤ 1.03 %, bit-identical
elastic off-state, 1.59× elastic runtime). This milestone adds the **third**
inelastic term, CDOM fluorescence, per the companion design: analytic-only,
default-off, head defined but untrained — because L23 omits CDOM-fl and BING
never implemented it, there is no truth channel and no cross-check reference;
correctness pins and a literature-plausibility band stand in (design §5) until
the design-§7 HydroLight runs unblock M6.

## Status entering M5

- The inelastic milestones M0–M4 are complete on `inelastic-rt`; the report
  refers to `main` (JXP merges before/as this effort starts). Suite state at
  the end of M4: **431 passed, 1 skipped** from the repo root in `ocean14`.
- `robust/rt/types.py` today: `IOPs` has `a_ph` (optional, validated ≤ a) but
  **no `a_cdom`**; `Inelastic.cdom_fl` is typed `Scalar | None = None` and its
  validator **rejects** any non-None value ("reserved hook") — task 1 retypes
  it to `CDOMFl | None`.
- `robust/rt/hybrid.py` composes `(Rrs_ZTT + ΔRrs) × f_R + Rrs_fl`; there is
  no CDOM branch. The two-tier regression pattern (strict SHA-256 pins on the
  dev machine, `skipif $CI`; ULP-closeness vs committed reference arrays
  everywhere) is established for the elastic off-state.
- The Ed module (`robust/rt/ed.py`), excitation-grid helpers
  (`conventions.py`), and the L23 loaders (`robust/rt/data/l23.py`, with
  `a_ph` extraction) all exist and are reused, not rebuilt. L23 stores a_g
  separately from a_nap, so `a_cdom` extraction mirrors the `a_ph` pattern.
- **Entry state (2026-08-29, task 1, Fable):** the `inelastic-rt` → `main`
  merge has **not** happened yet and no fresh branch off `main` exists;
  task 1 was executed directly on `inelastic-rt` — a flagged deviation from
  the fresh-branch assumption above (see Q&A CQ1 rather than a silent call).
  Suite state observed at `inelastic-rt` HEAD **before** task 1 on this
  machine (JXP's Mac, darwin — not the tank server that pinned the hashes):
  428 passed, 3 failed, 1 skipped. The 3 failures are machine-anchored, not
  regressions: the two strict SHA-256 pins
  (`test_elastic_hash_regression_strict`, `test_gate_4_pre_change_pins`)
  reproduce a *different but deterministic* hash on this Mac
  (`02de5483…` — identical on pristine HEAD and on the task-1 tree, so the
  elastic bits are provably unchanged by task 1; both ULP-closeness tiers
  pass), plus one timing-flaky speed test that passes on re-run. Consistent
  with **431 passed, 1 skipped** on the machine where the pins hold. After
  task 1: **451 passed, 1 skipped**, same two strict-pin failures only
  (+22 tests).

## Prompts

1. Read this doc. Execute task 1 in the "M5" section below. If you have any
   questions, ask me in the Q&A section below. Use Fable if you can. Log your
   work.
2. Read this doc. Execute tasks 2–3. Use Fable if you can. Log your work.
3. Read this doc. Execute tasks 4–5. Check my answers in Q&A; if you have
   additional questions, ask in Q&A. Use Fable if you can. Log your work.
4. Read this doc. Execute tasks 6–7. Use Fable if you can. Log your work.
5. Read this doc. Execute task 8 — the notebook and record. Use Fable if you
   can. Log your work.
6. Read this doc. Execute task 9 — reviewing the pull request. Use Fable if
   you can. Log your work.

## M5

### Tasks

1. **Extend the types.** In `robust/rt/types.py`: `IOPs` gains optional
   `a_cdom` (default `None`; validated like `a_ph` — non-negative, ≤ a,
   shape-broadcast in `from_total_bb`, preserved by `select()`); new
   registered pytree `CDOMFl(scale=1.0)` (`scale` a differentiable leaf;
   frozen; validated finite and > 0); `Inelastic.cdom_fl` retyped
   `CDOMFl | None = None` — **`None` stays the default**, an instance is now
   accepted, and setting it is validated against the process being usable
   (clear error at `forward()` time if `iops.a_cdom is None`).

   **Gate.** `test_cdom_types.py`: pytree mechanics (flatten/unflatten,
   leaf/static split, jit/vmap/grad traversal of `scale`), validators, and
   bitwise indifference of the elastic and inelastic paths to a set-but-unused
   `a_cdom`. Both existing hash-regressions green; full suite passes.

2. **Loader wiring.** `robust/rt/data/l23.py`: extract `a_cdom` (= a_g) from
   L23 alongside `a_ph`, mirroring that pattern (marker-guarded on data-less
   machines). Pin the decomposition bookkeeping in tests: `a_cdom ≥ 0`,
   `a_ph + a_cdom ≤ a` on real scenes — the a_dg double-counting foot-gun of
   design §8.

   **Gate.** Loader tests green on this machine (data present) and skipping
   cleanly without `$OS_COLOR`; golden values for a few scenes.

3. **The Hawes kernel.** New `robust/rt/cdom_fl.py`: the Hawes et al. (1992)
   spectral fluorescence quantum-efficiency function η(λ′, λ) with its
   constants recorded (which published function/variant — HydroLight's
   default choice — and source, per design §7's matching requirement); the
   analytic kernel `K_cdom(IOPs, Ω, λ)` — source ∝ `a_cdom(λ′)`, `Ed(λ′)`
   weighting via the existing Ed module, excitation quadrature, quanta→energy
   factor, per-λ_em attenuation, two-flow transport with **L_u = E_u/π**, and
   A·rrs/(1−B·rrs), reusing the Chl-fl kernel's machinery wherever it
   transfers.

   **Gate.** Correctness pins (design §5.2): η reproduced against its
   published values; energy/quanta bookkeeping unit tests (e.g. emission
   red-shifted from excitation everywhere, quantum bookkeeping consistent
   with the tabulated efficiency); quadrature convergence under grid
   refinement (result stable to a stated rtol between the native grid and a
   2× refined grid).

4. **The 350 nm clamp + truncated-fraction diagnostic.** Impose the hard
   350 nm lower excitation limit (design §2); implement a committed diagnostic
   that quantifies, from the Hawes function itself, the fraction of emission
   truncated by the clamp per emission wavelength, and write the numbers into
   the implementation record as the documented caveat.

   **Gate.** Diagnostic test pins the truncated fraction (banded — it
   *characterizes*, it doesn't gate to zero); the kernel provably never reads
   IOPs or Ed below 350 nm.

5. **Composition + the extended bit-identity regression.** Wire
   `Rrs_total = (Rrs_ZTT + ΔRrs) × f_R + Rrs_fl + Rrs_cdom` into
   `hybrid.forward()` — the CDOM branch **unreachable** when `cdom_fl is None`
   (no-op by construction, not by arithmetic). Pin the new regression
   **before** wiring: `forward(..., inelastic=Inelastic(..., cdom_fl=None))`
   on the CI fixture, two-tier (strict hash on this machine, ULP-closeness
   fixture for CI), alongside the elastic pin.

   **Gate.** Both off-state regressions bit-identical/green; with
   `cdom_fl=CDOMFl()` set, output changes only additively at the kernel's
   value (spot-checked); full suite passes.

6. **δ_C head defined, untrained.** In `robust/rt/inelastic_corr.py`: the
   bounded (tanh-scaled) δ_C head per the δ_F pattern, **zero-initialized** so
   the untrained head is exactly the analytic backbone; a stubbed training
   entry point that raises with a message naming M6 and the missing truth.
   No weights file is committed (nothing to train on).

   **Gate.** Test: zero-init head ⇒ `(1 + δ_C) ≡ 1` bitwise; the stub raises
   informatively; composition with the head present equals composition
   without it.

7. **Plausibility, gradients, speed.** (i) The literature-plausibility band
   (design §5.3) on L23 IOPs: CDOM-fl contribution a few % of Rrs in the
   blue-green for CDOM-rich scenes, ≲ 1 % oligotrophic, monotone in a_g(440)
   — loose banded gates plus a reported table. (ii) `jax.grad` vs central
   differences for all inputs **including `scale`** and `a_cdom`. (iii) Speed:
   full-batch forward with CDOM-fl on ≤ **2×** the elastic hybrid (the
   existing budget, now including this term).

   **Gate.** All three as pytest tests (speed banded, per the established
   throughput-test pattern); this closes design §5 items 3–5.

8. **Notebook + record + docs note.** Executed notebook
   `notebooks/RT/rt_cdom_coding_1.ipynb` (the established `ocean14`
   kernelspec/nbconvert recipe; degrade without `$OS_COLOR`): what M5
   *decided* — the Hawes basis and its recorded variant, the clamp and its
   measured truncated fraction, why default-off is load-bearing, the
   plausibility table. Update `design/rt_inelastic_implementation.md` with an
   M5 section (or start a sibling record if cleaner — say which and why), and
   add a short changelog/docs note stating the term exists, is analytic-only,
   default-off, and **unvalidated until M6**.

   **Gate.** Notebook committed with outputs; record updated; the
   unvalidated-until-M6 language present verbatim.

9. **Pull request.** JXP will create the PR. Review it (multi-angle, per the
   inelastic M0 task-6 precedent), verify findings against live code, and fix
   what JXP triages in. Use Fable if you can. Log your work.

### Q&A

**CQ1 (task 1, 2026-08-29, Claude/Fable → JXP):** This doc assumes M5 work
happens on a fresh branch off `main`, created once `inelastic-rt` is merged —
but the merge hasn't happened (read-only `git log` shows no such branch), and
the checkout is `inelastic-rt`, so task 1 was executed directly on it.
Should M5 continue on `inelastic-rt` as-is, or should I pause until you merge
to `main` and create the fresh branch (carrying task 1's uncommitted changes
over)?

>A. Ok, I am on a branch named `cdom-rt` now. 

**CQ2 (task 3, 2026-08-29, Claude/Fable → JXP):** The Hawes FA7 constants in
`robust/rt/cdom_fl.py` need a primary-source check. The *functional form*
(η_Y Gaussian in wavenumber, Zhai et al. 2017 Eqs. 5–8) is
peer-reviewed-verified, but the FA7 numbers themselves came from Mobley's
Ocean Optics Web Book via an AI-mediated page fetch (the page is JS-rendered)
and could not be corroborated against a second source. If you have access to
Mobley's *Light and Water* §5.15 or Hawes (1992), Proc. SPIE 1750, please
confirm or correct: **A1 = 0.470** (dimensionless), **B1 = 8.077×10⁻⁴ nm⁻¹**,
**A2 = 0.407** (dimensionless), **B2 = −4.57×10⁻⁴ nm⁻¹** (r² = 0.987), and
the A0(λ_e) table (units ×10⁻⁵ nm⁻¹): 310→5.81, 330→6.34, 350→8.00, 370→9.89,
390→9.39, 410→10.48, 430→12.59, 450→13.48, 470→13.61, 490→9.24. The §5.2
correctness-pin test (`test_a0_table_reproduced_at_the_gaussian_peak`)
deliberately pins exactly these numbers, so a correction means editing the
constants in `cdom_fl.py` and the test re-pins itself trivially.

## Next

→ M6 (δ_C training + quantitative gate) is **deferred**: it opens only when
the HydroLight "X4 vs X4 + CDOM-fl" runs of
`design/rt_cdom_fluorescence_model.md` §7 exist. No prompt doc until then.

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-29 (M5 task 1 — types extended: `IOPs.a_cdom`, `CDOMFl`, `cdom_fl` retyped; 451 green) (model: Fable)

Executed task 1 — the pure interface/data-model change plus tests. No kernel,
no composition math, no head (tasks 3/5/6). Files touched:

- `robust/rt/types.py`: `IOPs` gains optional `a_cdom: Spectrum | None = None`,
  mirroring `a_ph` verbatim — docstring attribute entry (`Geometry.wind`
  pytree semantics: no leaves unset, treedef change when set), a mirrored
  `a_cdom=None` keyword on `from_total_bb` with the same
  batch-shape broadcast (the PR #14 uniform-batch-shape guarantee), and the
  same three `validate()` checks (shape match, `check_iop`, component-of-`a`
  bound `a_cdom ≤ a`). New registered pytree `CDOMFl(scale=1.0)`
  (design §3): frozen dataclass, `scale` its single differentiable leaf (the
  `s_C` amplitude on the Hawes reference kernel), `validate()` rejecting
  non-finite/non-positive scale with a message pointing at `cdom_fl=None`
  rather than `scale=0`; added to `__all__`. `Inelastic.cdom_fl` retyped from
  the reserved always-reject `Scalar | None` hook to `CDOMFl | None = None` —
  `None` stays the default (load-bearing: the X4 truth omits CDOM-fl);
  `validate()` now type-checks a set value (`isinstance(…, CDOMFl)`, so the
  pre-M5 bare-scalar calling convention fails loudly — deliberate choice over
  duck-typing, matching the module's explicit-boundary-check philosophy) and
  delegates to `CDOMFl.validate()`. Note: the task text's "preserved by
  `select()`" is a stray reference — no `select()` exists anywhere in the
  codebase (grepped); nothing to preserve beyond the pytree mechanics.
- `robust/rt/hybrid.py`: **one guard clause only**, in `rrs_forward` beside
  the `fluorescence`/`a_ph` twin: `cdom_fl is not None and iops.a_cdom is
  None` ⇒ `ValueError` naming `a_cdom` (the task-1 "usable at forward() time"
  requirement). No composition wiring — until task 5, `cdom_fl` set *with*
  `a_cdom` present passes the guard and is then ignored by the composition;
  that interim window closes when task 5 wires the term and pins the extended
  regression.
- `robust/tests/test_inelastic_types.py` (+22 tests, per the concurrency
  constraint kept here rather than a new `test_cdom_types.py` — the gate's
  filename is satisfied in substance, same module the a_ph/Inelastic
  contracts live in): CDOMFl pytree mechanics (defaults, flatten/unflatten,
  single-leaf, frozen/replace, jit/vmap/grad traversal of `scale`), CDOMFl
  validator accept/reject (zero/negative/nan), `Inelastic(cdom_fl=CDOMFl())`
  now accepted, nested-leaf accounting (set ⇒ `phi_C` + `scale` two leaves;
  unset ⇒ one), the old "cdom-set" reject case retargeted (bare scalar now
  rejected as a type error, plus a new nested-bad-scale case), the five a_ph
  test twins for `a_cdom` (default-None leaves, leaf/jit/vmap, from_total_bb
  passthrough + broadcast, validator rejects), **bitwise indifference** of
  both the elastic route (tiny_args, ztt) and the real inelastic route
  (Raman+Chl-fl on the 50-scene fixture, `cdom_fl=None`) to a set-but-unused
  `a_cdom`, and the new guard (`fluorescence=False, cdom_fl=CDOMFl()` without
  `a_cdom` raises with "a_cdom" at both entry points).

Suite: before task 1 (pristine `inelastic-rt` HEAD, this Mac) 428 passed /
3 failed / 1 skipped; after, **451 passed / 2 failed / 1 skipped**
(`conda run -n ocean14 python -m pytest robust/tests/ -q`). The failures are
the two **machine-anchored strict SHA-256 pins**, failing identically on
pristine HEAD — this Mac is not the tank server that pinned them, and it
reproduces its own deterministic hash `02de5483…`, *byte-identical between
pristine HEAD and the task-1 tree*, which is the strongest available evidence
the elastic bits are untouched; both ULP-closeness tiers pass. Per the pin's
own docstring this is a finding for JXP, not a hash to re-pin (recorded in
"Status entering M5"). The third pristine failure (speed gate) is timing
flakiness; it passes on the task-1 tree. ruff check/format clean on all three
files. `CDOMFl` is exported from `robust.rt.types.__all__` but deliberately
not re-exported from `robust/rt/__init__.py` (file outside task 1's allowed
edit set; task 5's wiring is the natural place). Branch finding: no fresh
branch off `main` exists — work proceeded on `inelastic-rt`; question posed
to JXP as Q&A CQ1.

### 2026-08-29 (M5 tasks 2–3 — loader `a_cdom` wiring + fixture regen; the Hawes kernel `cdom_fl.py`; 466 green) (model: Fable)

Executed tasks 2–3 on branch `cdom-rt` (verified before starting). No touch
to `types.py` (task 1, done) or `hybrid.py`'s composition (task 5); no δ_C
head (task 6); no truncated-fraction diagnostic (task 4 — but the hard clamp
itself is in, see below).

**Task 2 — loader wiring** (`robust/rt/data/l23.py`,
`design/py/gen_inelastic_fixture.py`, `robust/tests/test_l23_inelastic_data.py`):

- `INELASTIC_RAW_FIELDS` gains `"ag"` (the L23 netCDF name for CDOM/gelbstoff
  absorption; `ag` is distinct from detrital `ad`, confirmed by opening the
  files). `_read_inelastic_file` reads `ds1.ag` and the X=1/2/4
  consistency-assert loop now covers `ag` exactly like `aph`.
  `load_inelastic_batch` threads `a_cdom` through `parts` into the
  `IOPs(a_cdom=...)` constructor; `L23InelasticBatch`'s docstring and
  `validate()` mirror the `a_ph` presence contract for `a_cdom`.
  `inelastic_npz_reader` and `write_inelastic_fixture` both gain `"ag"` in
  their per-field loops, mirroring `aph`'s threading exactly.
- **The committed CI fixture's bytes changed** (expected and required):
  `robust/tests/files/l23_inelastic_fixture.npz` regenerated via
  `write_inelastic_fixture()` only (deliberately *not* the script's `main()`,
  which would also rewrite `ed_l23.npz` and the machine-anchored
  `elastic_reference_outputs.npz` — neither may change in this session).
  New size 285 kB, still under the 300 kB budget test; the script's own
  round-trip validation (real `load_inelastic_batch` through the real reader,
  now demanding `a_cdom`) passed before the atomic replace. The **elastic**
  fixture's bytes are untouched (its SHA-256 pin stays green).
- New loader tests (+4): the design-§8 a_dg foot-gun **pinned as real
  assertions** on the fixture batch — `a_cdom ≥ 0` and `a_ph + a_cdom ≤ a`
  everywhere (measured margin: max(a_ph + a_cdom − a) ≈ −5.3e-3) — plus the
  same pins at full-release scale (9960 samples, `needs_l23*`-guarded);
  golden absolute pins on fixture rows (`ag_0[0]@440 = 5.7960e-03`,
  `ag_30[7]@440 = 3.1390e-03`); a bit-faithful loader-vs-raw-netCDF golden at
  (scene 0, scene 7) × (0°, 60°); a `validate()`-requires-`a_cdom` twin; and
  `ag` rows added to the existing fixture-vs-netCDF bit-faithfulness sweep.
  Fixture-backed tests need no `$OS_COLOR`; the live-netCDF ones carry the
  established `needs_l23`/`needs_l23_inelastic` skip markers.

**Task 3 — the Hawes kernel** (`robust/rt/cdom_fl.py` new,
`robust/tests/test_cdom_fl.py` new, +11 tests):

- `eta_hawes(λ, λ_e)` implements Zhai, Hu, Lee et al. (2017), Opt. Express
  25(8), Eqs. (7)–(8) literally — the Gaussian argument in **reciprocal
  wavelength**: center `A1/λ_e + B1`, width `0.6·(A2/λ_e + B2)`, amplitude
  `A0(λ_e)` linearly interpolated (`jnp.interp`) between the ten tabulated
  nodes (a documented secondary uncertainty), gated by `g_Y` (310–490 nm).
- **Provenance, spelled out (also in the module docstring):** the functional
  form and the ≥350 nm excitation floor are **peer-reviewed-verified** (JXP
  extracted Eqs. 5–8 verbatim from the published PDF; Zhai et al. themselves
  clamp λ_e ≥ 350 nm citing UV ozone absorption + low solar irradiance —
  independent corroboration of our CFQ4 clamp). The **FA7 numeric constants
  are NOT independently verified**: sourced from Mobley's Ocean Optics Web
  Book (retrieved 2026-08-29, AI-mediated fetch of a JS-rendered page), FA7
  being HydroLight's own default (not Zhai's 9:1 FA7:HA6 mix). Flagged
  prominently on the constants, pinned as-is by the §5.2 test so re-pinning
  is trivial, and posed to JXP as **Q&A CQ2** (full numbers restated there).
  The "C. K. Carder" vs Kendall L. Carder citation discrepancy in Zhai's
  reference list is recorded in the docstring rather than silently resolved.
- `cdom_kernel(iops, geometry, wave)` mirrors `fluorescence_kernel`'s S&P98
  machinery term for term — source `b_bY = ½·a_cdom(λ_e)` (isotropic, no
  reference-yield division: `CDOMFl.scale` is applied by task 5's
  composition, never here), true `Ed(λ_e)` via the existing Ed module,
  trapezoid quadrature, `K(λ_e)=(a+b_b)/MU_D`, `κ_Y(λ)=(a+b_b)/MU_F`,
  `optimization_barrier`, **L_u = E_u/π**, `rrs_to_Rrs` — with the one
  **structural departure**: η_Y is non-separable in (λ, λ_e), so it
  multiplies the `(..., n_em, n_ex)` integrand *before* the excitation
  reduction instead of post-multiplying the reduced sum like Chl-fl's
  `emission_line`. Honest speed note: that costs one extra elementwise
  multiply on the big tensor by a *batch-free* (n_em, n_ex) matrix, but the
  contraction is 29 nodes vs Chl-fl's 65, so no speed regression is expected
  (task 7 measures it).
- Excitation grid `cdom_excitation_grid()`: **350–490 nm at 5 nm, 29 nodes**
  — the hard clamp *is* the grid (no clamping arithmetic), the 490 nm top is
  `g_Y`'s own cutoff, and 5 nm matches the canonical spacing so every node
  lands on a canonical grid point (asserted). A `step` argument exists solely
  for the convergence gate.
- Gate tests, honestly labeled: A0 reproduced at each tabulated λ_e's own
  emission peak (docstring says plainly it pins the table/interp *plumbing*,
  not the physics); η_Y ≥ 0 + `g_Y` gating; emission **peak** red-shifted for
  every admissible λ_e (analytic: peak wavenumber < 1/λ_e iff λ_e < ~656 nm)
  — stated plainly that the Gaussian-in-wavenumber form does *not* enforce a
  strict Stokes shift (blue tail at λ=λ_e is ~6 % of peak at 350 nm, ~22 % at
  490 nm; asserted subdominant, never asserted away); quadrature convergence
  5 nm vs 2.5 nm at rtol 1e-2 (measured max 5.6e-3); the clamp proved at the
  seams (spies on `interp_spectrum`/`ed.Ed` — every wavelength the kernel
  reads is ≥ 350 nm); a_cdom-required error; physicality (K ≥ 0, finite,
  broad, median peak in the blue-green — measured median K@440 ≈ 3.6e-5 sr⁻¹,
  ~0.5 % of median Rrs@440 on the fixture, inside the design-§5.3 ballpark
  task 7 will gate); jit/vmap agreement; finite + nonzero `a_cdom` gradient
  smoke (full FD gate incl. `scale` is task 7).

Suite (`conda run -n ocean14 python -m pytest robust/tests/ -q`, this Mac):
before **451 passed / 2 failed / 1 skipped**; after **466 passed / 2 failed /
1 skipped** (+15). The 2 failures are the same two machine-anchored strict
SHA-256 pins as task 1's entry, reproducing the identical local hash
`02de5483…` before *and* after — the elastic bits are provably untouched by
this session; both ULP-closeness tiers pass, and the occasionally-flaky speed
test passed in both runs. ruff check + format clean on all five touched .py
files. Not done here, by scope: no `hybrid.py` wiring (task 5), no δ_C
(task 6), no truncated-fraction diagnostic (task 4).

### 2026-08-29 (M5 tasks 4–5 — truncated-fraction diagnostic + record §8.1; pre-wiring pin, CDOM composition wired, no-op proved; 474 green) (model: Fable)

Executed tasks 4–5 on `cdom-rt` (verified; tree clean at start). Q&A checked:
CQ1 answered (this branch); CQ2 still open and deliberately untouched —
tasks 4–5 depend only on the kernel existing, not on the FA7 values, so no
re-ask and no invented answer. No δ_C head (task 6), no plausibility/FD/speed
gates (task 7).

**Task 4 — the truncated-fraction diagnostic** (`robust/rt/cdom_fl.py`,
`robust/tests/test_cdom_fl.py` +3, `design/rt_inelastic_implementation.md`):

- The clamp itself was already structural (task 3: the excitation grid
  *starts* at 350 nm); the new deliverable is
  `truncated_excitation_fraction(wave)` — for each emission λ,
  `∫₃₁₀³⁵⁰ η_Y dλ_e / ∫₃₁₀⁴⁹⁰ η_Y dλ_e`, from the Hawes FA7 function alone
  (no IOPs/Ed/scene; the one deliberate sub-350 nm evaluation of η_Y, which
  is its purpose). Trapezoid at 0.25 nm, converged (max 3.5e-6 relative vs a
  2× refinement — pinned as a test); 0/0 guarded: far outside the Hawes band
  both integrals underflow to exactly 0.0 and the fraction is *defined* as 0
  (no emission → nothing truncated), never a silent NaN — measured that the
  guard is provably inert on the canonical grid (min denominator > 0
  everywhere; min fraction 0.0697 at 605 nm), also pinned.
- **Measured numbers** (canonical grid): λ_em 350/400/450/500/550/600/650/
  700/750 nm → **0.846 / 0.566 / 0.297 / 0.142 / 0.083 / 0.070 / 0.078 /
  0.103 / 0.146**. The headline caveat: **57 % of the nominal
  310–490 nm-excited Hawes emission at 400 nm is excluded by the production
  clamp** (85 % at 350 nm, 30 % at 450 nm; minimum ~7 % near 605 nm, rising
  to ~15 % at 750 nm via the sub-350 Gaussians' red tails) — design §8's
  blue-band risk realized, recorded rather than asserted away, with the
  honesty note that the *realized* Rrs truncation is further suppressed by
  `a_cdom(λ_e)·Ed(λ_e)` weighting (UV Ed is weak — Zhai et al.'s own clamp
  rationale). Pinned banded (±0.03 abs) at those nine wavelengths — it
  characterizes, it doesn't gate to zero.
- The gate's other half ("kernel provably never reads IOPs/Ed below 350 nm")
  is **fully covered by task 3's spy test**
  (`test_kernel_never_reads_iops_or_ed_below_350`, seams on
  `interp_spectrum`/`ed.Ed`) plus `test_excitation_grid_is_the_hard_clamp` —
  stated in a comment at the task-4 test block instead of duplicating.
- Record updated: **`design/rt_inelastic_implementation.md` gains §8
  "M5 — CDOM fluorescence *(in progress …)*" with §8.1 only** — the table
  above, the caveat verbatim, the quadrature/guard notes, and a pointer to
  `design/rt_cdom_fluorescence_model.md` §2 for the clamp's rationale. The
  heading itself marks the section partial; tasks 1–3's retrospective and the
  rest of M5 are task 8's job.

**Task 5 — composition + the extended bit-identity regression**, in the
sequence-critical order:

- **Step 5a, the pin FIRST** (`design/py/gen_inelastic_fixture.py`,
  new committed `robust/tests/files/inelastic_default_reference_outputs.npz`,
  88 kB; `robust/tests/test_inelastic_types.py`): before touching
  `hybrid.py`, added `write_inelastic_default_reference()` (the
  `write_elastic_reference` template verbatim: compute on the fixture batch
  via the real reader, savez to temp, round-trip byte-verify, atomic
  replace; also appended to `main()`), ran it on the **unmodified** code, and
  pinned SHA-256 of the arrays as
  `PRE_CDOM_SHA256_RRS_ABOVE = 0dd365158e3037261ee061777fe51da8fa132d4f0972792ad068b9c73641291a`
  (`forward`) and
  `PRE_CDOM_SHA256_RRS_BELOW = 72d4a308e2222c802e18e1878d00f26853db831d9db82a8e529cfead883cc0b8`
  (`rrs_forward`) — the default `Inelastic()` (`cdom_fl=None` implicit),
  committed trained heads (`corrections=None`), `check_domain=False`, on the
  150-sample inelastic fixture. Two-tier test pair mirrors the elastic one:
  `test_inelastic_default_hash_regression_strict`
  (`@strict_bits_are_local`, also `@needs_weights` — absent weights would
  silently change the bytes) and
  `test_inelastic_default_regression_close_everywhere` (rtol 5e-7).
  **Both ran green before any `hybrid.py` edit** (tautological then; the
  harness-is-wired proof). **Machine-anchoring finding for JXP:** these pins
  are anchored to *this Mac* — a different machine from the tank server that
  anchored the M0 elastic pins — so on any one machine one strict set may
  fail while the other passes (documented on the constants; the closeness
  tiers carry the guard everywhere).
- **Step 5b, the wiring** (`robust/rt/hybrid.py`): (1) the **guard fix** —
  `_apply_inelastic`'s early return tested only `raman or fluorescence`, so
  a caller setting *only* `cdom_fl` (raman/fluorescence off) would have
  passed `forward()`'s a_cdom check and then silently received the untouched
  elastic `rrs` — a plausible-looking array with the requested physics
  missing, precisely the failure mode the module's loud-error philosophy
  exists to prevent; the condition now also treats `cdom_fl is not None` as
  an active process. (2) The additive term, mirroring the fluorescence
  block: `result += jnp.asarray(inelastic.cdom_fl.scale)[..., None] *
  _cdom_fl.cdom_kernel(iops, geometry, wave)`, with the comment that task 6
  will multiply by `(1 + δ_C)` once the head exists — until then this term
  IS the full CDOM contribution ((1+0)=1, CFQ3). Composition-law docstrings
  updated to `… + Rrs_fl + Rrs_cdom`. `forward()`'s task-1 a_cdom guard
  confirmed consistent, untouched; the heads-resolution block stays scoped
  to raman/fluorescence (task 6's concern, per instruction). Also completed
  the task-1 deferral: `CDOMFl` + the `cdom_fl` submodule re-exported from
  `robust/rt/__init__.py` (the wiring makes `CDOMFl` a genuine `forward()`
  argument type), with an export test twin.
- **Step 5c, the proofs** — (1) **no-op proof**: both new pin tests re-run
  after the wiring, **bit-identical, green** — the CDOM branch is unreachable
  when `cdom_fl=None`, by construction not by cancellation (and the elastic
  strict pin still reproduces the identical local hash `02de5483…` before and
  after, so the elastic bits are untouched too). (2) **Additive proof**
  (`robust/tests/test_inelastic.py`, the wiring-test home, +2):
  `test_forward_composes_cdom_fluorescence_additively` — with
  `CDOMFl(scale=2.0)` (deliberately ≠ 1 so a dropped amplitude can't pass),
  `forward(default+cdom) == forward(default) + 2·K_cdom` at rtol 5e-6/atol
  1e-10, **in Rrs space** — the composition law's own space (`K_cdom` ends in
  `rrs_to_Rrs`, so the term adds above the surface; the prompt's
  rrs_forward-difference phrasing would pick up Lee's non-linear conversion,
  ~1/A ≈ 1.9× off), matching the fluorescence twin's precedent; the same
  identity is asserted a second time from `rrs_forward` outputs explicitly
  converted up, pinning additivity at exactly the composed layer. (3) The
  guard-fix regression `test_cdom_fl_alone_composes`:
  `Inelastic(raman=False, fluorescence=False, cdom_fl=CDOMFl())` is *not*
  bitwise-elastic (the pre-fix silent no-op) and *is* elastic + K_cdom.
  (4) `test_gate_4_elastic_bit_identity` extended with the fully explicit
  `Inelastic(raman=False, fluorescence=False, cdom_fl=None)` third assertion.

Suite (`conda run -n ocean14 python -m pytest robust/tests/ -q`, this Mac):
before **466 passed / 2 failed / 1 skipped**; after **474 passed / 2 failed /
1 skipped** (+8: 3 diagnostic, 2 pins, 2 wiring, 1 export). The 2 failures
are the same two machine-anchored elastic strict pins, same local hash
`02de5483…` before and after; both closeness tiers green. The
occasionally-flaky speed gate passed in the before and final runs (it failed
once in an intermediate run mid-edit and passed on the clean re-run — timing
noise, and its config is `Inelastic()` default, where the CDOM branch is
provably unreachable). ruff check + format clean on all nine touched .py
files. Untouched, by scope: CQ2 (open, JXP's), `types.py`, the δ_C head,
`_resolve_corrections`. Noted in passing: `claude_prompts/RT/rt_docs_prompt_1.md`
carries uncommitted changes not from this session (JXP's, presumably) — left
alone.
