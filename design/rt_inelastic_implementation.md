# Inelastic RT Implementation Record

**Version:** 0.13
**Date:** 2026-08-24
**Authors:** JXP and Claude

**Status:** living document — updated as each milestone is implemented.
**Companions:** implements [`rt_inelastic_model.md`](rt_inelastic_model.md)
(design, *what/why*) following
[`rt_inelastic_model_coding_plan.md`](rt_inelastic_model_coding_plan.md)
(plan, *how/when*). This document records *what was actually built*: the
modules, their public API, the implementation decisions taken, the tests, and
the numerical results per milestone. The elastic predecessor is
[`rt_elastic_implementation.md`](rt_elastic_implementation.md), whose
conventions (§8 there) this effort inherits wholesale.
**Versioning convention** (same as the design docs): minor bump for
substantive changes (0.1 → 0.2), extra decimal for small edits (0.1.1); update
the Date on every bump.

> The chronological narrative lives in the "Logs" sections of the
> per-milestone prompts, `claude_prompts/RT/rt_inelastic_coding_prompt_*.md`.
> This document is the structured, current-state reference for the inelastic
> extension of `robust/rt/` as implemented.

---

## 1. Status at a glance

| M | Goal | Status | Package surface |
|---|------|--------|-----------------|
| **M0** | Environment (this machine) & API extension | ✅ done | `robust.rt.types` (extend), `robust.rt.hybrid` (extend), `robust/tests/test_inelastic_types.py` |
| **M1** | Ed module, excitation grid, X2/X4 data | ✅ done | `robust.rt.ed`, `robust.rt.conventions` (extend), `robust.rt.data.l23` (extend), `robust/rt/data/ed_l23.npz`, sibling CI fixture |
| **M2** | Analytic terms in JAX | ⬜ not started | `robust.rt.inelastic`, composition in `robust.rt.hybrid` |
| **M3** | Correction heads δ_R, δ_F | ⬜ not started | `robust.rt.inelastic_corr`, `robust/rt/files/{raman,fl}_corr_l23.npz`, `design/py/train_inelastic_corr.py` |
| **M4** | Validation (*prototype done*) | ⬜ not started | `robust.rt.validation` (extend), `design/py/run_validation.py` (extend), `design/validation/` |

Legend: ✅ done · 🟡 in progress · ⬜ not started.

**Branch.** All milestone work lands on **`inelastic-rt`** (JXP kept the
existing branch over the coding plan's `rt-inelastic-prototype` name —
prompt 1 Q&A Q2); each milestone is a reviewable commit/PR for JXP (Claude
runs no state-changing git — see `CLAUDE.md`). M0 is PR #14.

**Machine & environment.** Unlike the elastic effort (laptop), this effort
runs on **this machine (tank server)**, in the `ocean14` conda env, CPU-only
JAX. The stack was installed 2026-08-20 (M0 task 2) following the elastic
procedure — dry-run first, purely-additive verification — see §2.3, including
the one wrinkle (the `requirements.txt` `git+` lines for `bing`/`ocpy` vs
their local editable checkouts) and how it was resolved.

**The recurring gate.** From M0 onward, `forward(..., inelastic=None)` on the
elastic CI fixture must be **bit-identical** (hashed output array) to the
pre-change elastic hybrid, and the full elastic test suite (279 tests as of
the prototype merge) must pass unmodified. This is the design §1 guarantee:
the elastic path is a no-op *by construction* (the `None` branch takes the
existing code route), not by arithmetic that happens to cancel.

**Acceptance philosophy.** Unlike the elastic plan's relative-only gates, the
per-process delta gates here are **absolute** bars (≤ 5 %, fixed explicitly in
Q&A/Design DQ6), and total Rrs gates at ≤ 0.5 % held-out rRMS per zenith.
Diagnostics without truth data (φ_C-linearity, double-Gaussian emission) are
reported, never gated. The `bing` cross-check tests (M2) import the *fixed*
BING (branch `inelastic-fixes`) live and `skipif` when it is unavailable, so
GitHub CI stays green while local runs enforce the pin (CQ3a).

**Verification (current).** `pytest -q` from the repo root → **355 passed**
(`ocean14`, this machine, 2026-08-24, `$OS_COLOR` set): 279 elastic unmodified
+ 31 M0/hash-gate (two-tier since task 7, §2.8) + 45 M1 (`test_ed.py` 20,
Raman-grid additions in `test_conventions.py` 9, `test_l23_inelastic_data.py`
15, plus the review-debt regression test in `test_l23.py`). `ruff check` and
`ruff format --check` clean. With `CI=true` simulated: strict hash tier
skips, closeness tier green; with `$OS_COLOR` unset the raw-netCDF tests
skip and the fixture-fed loaders still run.

---

## 2. M0 — Environment & API extension

**Goal.** A green base on this machine: the JAX stack installed into
`ocean14` (tank server), the `robust/rt` API extended with the inelastic
types, and the elastic behavior provably untouched. Nothing scientific yet.

**Gate.** `robust/tests/test_inelastic_types.py`: pytree flatten/unflatten,
defaults, `jit`/`vmap` traversal. **Elastic hash-regression**:
`forward(..., inelastic=None)` on the elastic CI fixture is bit-identical
(hash the output array) to the pre-change result — pinned at M0, guarding
every later milestone. The full existing elastic suite passes unmodified;
environment checks green.

### 2.1 Task status

| # | Task | Status |
|---|------|--------|
| 1 | Implementation record (this document), seeded with M0-in-progress | ✅ done |
| 2 | Install the JAX stack into `ocean14` on this machine; verify purely additive; record versions | ✅ done |
| 3 | Extend the API (`types.py`: `IOPs.a_ph`, `Inelastic`, `Geometry.Ed`; `hybrid.forward(..., inelastic=None)`); pass the gate incl. the pinned elastic hash | ✅ done |
| 4 | `notebooks/RT/rt_inelastic_coding_1.ipynb` — the M0 explainer, executed | ✅ done |
| 5 | Update `rt_inelastic_coding_prompt_2.md` with the actual M0 outcome | ✅ done |
| 6 | Review PR #14 (added by JXP after M0 closed; Cursor pre-review verified) | ✅ done — see §2.7 |
| 7 | Fix the two failing PR tests (the predicted CI hash failures) | ✅ done — see §2.8 |

### 2.2 Modules extended

No new module — M0 touches `types.py`, `hybrid.py`, and the `robust.rt`
re-exports, all strictly backward compatible (every elastic call site and
test runs unmodified).

**`robust/rt/types.py`:**

- `IOPs` gains `a_ph: Spectrum | None = None` — phytoplankton absorption, the
  fluorescence source term's factor (`b_F = φ_C·a_ph`). Follows the
  `Geometry.wind` precedent exactly: unset contributes **no leaves** (the
  elastic three remain the only ones, so no existing `vmap` axis spec or
  `tree_map` changes), set adds a fourth leaf and changes the treedef (one
  `jit` recompile, the documented `PhaseParams` behavior).
  `from_total_bb(..., a_ph=None)` passes it through; `validate()` checks
  shape, non-negativity, and — the bookkeeping trap — `a_ph ≤ a` everywhere,
  since `a_ph` is a *component* of total absorption.
- New `Inelastic(phi_C=0.02, raman=True, fluorescence=True,
  emission_shape='single', cdom_fl=None)`, registered with
  `jax.tree_util.register_dataclass` like the elastic three. **Key decision —
  leaves vs static fields.** `phi_C` (and `cdom_fl`, once populated) are
  pytree *leaves*: `φ_C` is a differentiable input (design DQ4; `grad` of a
  scalar of an `Inelastic` returns an `Inelastic` with `∂/∂φ_C` — tested).
  The booleans and `emission_shape` are *static* metadata
  (`field(metadata=dict(static=True))`): they select code paths, so `jit`
  specializes on them (treedef changes per configuration — tested). New
  module constant `EMISSION_SHAPES = ('single', 'double')`. `validate()`
  rejects `φ_C ∉ (0, 1]` (with the message pointing at
  `fluorescence=False` rather than `phi_C=0`), unknown emission shapes, and
  any non-`None` `cdom_fl` (the v1-reserved hook cannot be silently
  "enabled").
- `Geometry` gains `Ed: (wave_Ed, Ed) | None = None` — the irradiance
  override seam (design §3). The pair's arrays are leaves when set;
  `validate()` checks the pair shape, 1-D equal lengths, strictly increasing
  `wave_Ed`, and finite non-negative `Ed`.

**`robust/rt/hybrid.py`:** `forward` and `rrs_forward` gain keyword-only
`inelastic=None`. The design's snippet writes it positionally after `wave`,
but the elastic signature already has `mode` in that slot — keyword-only is
the backward-compatible reading, and matches the elastic convention that
everything after `mode` is keyword-only. When `None` (the default), the guard
is an early `raise`-or-fall-through *before any computation*: the elastic
route is the pre-existing code, untouched — the no-op is by construction, not
by adding zeros. When an `Inelastic` is passed, `NotImplementedError` naming
M2 (the elastic M0 stub convention: a loud error, never a plausible-looking
array missing its physics).

**`robust/rt/__init__.py`:** re-exports `Inelastic`; docstring now states the
elastic-physics/inelastic-interface split and the bit-identical guarantee.

### 2.3 Environment

`ocean14` **on this machine (tank server)** as of 2026-08-20, after the M0
task-2 install:

| Package | Version | Note |
|---|---|---|
| Python | 3.14.6 | conda env `ocean14` (miniconda, `/home/xavier/miniconda3`) |
| jax | **0.11.1** | new — CPU backend (laptop had 0.11.0) |
| jaxlib | **0.11.1** | new — pulled by `jax` (CPU wheel) |
| flax | **0.12.9** | new — MLP heads (M3) (laptop had 0.12.8) |
| optax | **0.2.8** | new — training (M3) |
| jaxtyping | **0.3.11** | new — public-signature annotations |
| numpy | 2.4.6 | unchanged |
| scipy | 1.18.0 | unchanged |
| xarray | 2026.4.0 | unchanged |
| pandas | 3.0.3 | unchanged |
| matplotlib | 3.11.0 | unchanged |
| emcee | 3.1.6 | unchanged |
| pytest | 9.1.1 | unchanged |
| `bing` | 0.0.dev0 | unchanged — **editable** from `/mnt/tank/Oceanography/python/bing`, checkout on branch **`inelastic-fixes`** (verified — the M2 cross-check dependency, CQ3a) |
| `ocpy` | 0.1.dev0 | unchanged — **editable** from `/mnt/tank/Oceanography/python/ocpy` |

Transitive additions (all new, nothing upgraded or removed): `absl-py`,
`aiofiles`, `etils`, `humanize`, `markdown-it-py`, `mdurl`, `ml_dtypes`,
`msgpack`, `opt_einsum`, `orbax-checkpoint`, `prometheus-client`, `protobuf`,
`rich`, `simplejson`, `tensorstore`, `treescope`, `uvloop`, `wadler-lindig` —
the identical list the elastic laptop install recorded.

**The dry run, and a deviation from the letter of the task.** `pip install
--dry-run -r requirements.txt` was **not** purely additive — but not because
of the JAX stack. The `Would install` list included `bing-0.0.dev0` and
`ocpy-ocean-0.1.0`: `requirements.txt` declares both as `git+` GitHub
sources, while this machine has them as **editable installs from local
checkouts** — and the `bing` checkout sits on `inelastic-fixes`, the exact
physics M2 must cross-check against. Installing the full file would have
replaced that checkout's install with GitHub `main` (and `ocpy` with the
renamed `ocpy-ocean` distribution). Resolution: a second dry run restricted
to the JAX block (`pip install --dry-run jax flax optax jaxtyping`) **was**
purely additive — 23 packages, all new, no uninstalls/upgrades, `bing`/`ocpy`
untouched — so exactly that subset was installed. The coding plan's fallback
(dedicated env) targets dependency conflicts, which never materialized;
flagged in the prompt-1 Q&A for JXP rather than blocking the milestone.

**Verification (task-2 gate).** In `ocean14` on this machine:

- `import jax; jax.numpy.ones(3)` → `[1. 1. 1.]` on `CpuDevice(id=0)`;
  `jax.default_backend() == "cpu"`; every device is CPU. Note: this machine
  **has an NVIDIA GPU**, and jax prints "An NVIDIA GPU may be present ...
  Falling back to cpu" — the CPU-only wheel behaves exactly as CQ2 requires,
  but the message will appear in logs and is expected, not an error.
- `jax.config.update("jax_enable_x64", True)` yields float64 arrays — the
  precision the finite-difference gradient gates need is available.
- `jax.grad` smoke: `d/dx Σx² = 6.0` at `x = 3`; `jax.jit` compiles and
  computes.
- Regression check on the pre-existing env: `numpy`, `scipy`, `xarray`,
  `pandas`, `matplotlib`, `scikit-learn`, `emcee`, `bing`, `ocpy` (incl.
  `ocpy.hydrolight.loisel23`) all still import, from their pre-install
  locations.
- **The full elastic suite runs on this machine for the first time**:
  `pytest -q` from the repo root → **279 passed** in 53 s (`$OS_COLOR` set),
  matching the laptop count exactly.

**Risk retired.** The coding plan flagged that the JAX install might perturb
`ocean14` (fallback: a dedicated env). The subset dry run confirmed pure
additivity and the post-install checks confirm it held. **We stay in
`ocean14`**; no separate env is needed. The one real hazard found was the
`git+` lines vs the editable checkouts, documented above (JXP confirmed the
resolution — prompt 1 Q&A, Q1).

**Addendum (task 4, 2026-08-21) — notebook tooling.** This machine had no
jupyter kernelspecs at all (the elastic notebooks were executed on the
laptop). To execute the M0 notebook with the `ocean14` interpreter:
`ipykernel 7.3.0` was pip-installed into `ocean14` (dry-run verified purely
additive: `comm`, `debugpy`, `ipykernel`, `jupyter_client`, `jupyter_core`,
`nest-asyncio2`, `pyzmq` — all new, nothing touched), and a user kernelspec
**`ocean14`** was registered (`python -m ipykernel install --user --name
ocean14`). Execution itself is driven by `jupyter nbconvert --execute` from
the `os_313` env, which launches the `ocean14` kernel via that spec — so the
notebook's outputs are genuinely `ocean14`'s. The committed notebook's
kernelspec is therefore named `ocean14`, not the elastic notebooks'
`python3`.

### 2.4 Tests

`robust/tests/test_inelastic_types.py` — **30 tests**, all running in CI (the
committed fixture and synthetic inputs only; no `$OS_COLOR`).

- *Elastic hash-regression (the recurring gate)*: `forward(...,
  inelastic=None)` and `rrs_forward(..., inelastic=None)` on the 50-scene
  fixture hash (SHA-256 over the raw float32 bytes) to the **pre-change**
  values, which were computed on the unmodified elastic code immediately
  before `types.py`/`hybrid.py` were touched and verified deterministic
  across processes:
  - `Rrs`: `aaa0616119f179551e64969cd8407ed44e8eb0f8f5d9b27ba6ac7c97d826bbc7`
  - `rrs`: `d111464020aacb47bbc9dd9aa027dd11b2e15e019a735687b6c6c0fa504c2c38`
  (Both with `check_domain=False` — the fixture's 60° rows are outside the
  emulator domain, and the pin must match its own recipe.) Also: omitting
  `inelastic` and passing `None` are bitwise the same call; an `Inelastic`
  instance raises `NotImplementedError` matching "M2"; the elastic route
  still compiles under `jit` with `inelastic=None`.
- *`Inelastic` pytree*: design-§3 defaults field by field;
  flatten/unflatten round-trip with non-default statics; `phi_C` is the
  **only** default leaf; static fields change the treedef; frozen +
  `dataclasses.replace`; `jit` and `vmap` traversal (batched per-scene
  `phi_C`); `grad` returns an `Inelastic` with the labelled `∂/∂φ_C`;
  `validate()` accepts defaults and rejects each documented failure mode
  (φ_C ∈ {0, negative, >1, NaN}, unknown emission shape, `cdom_fl` set).
- *`IOPs.a_ph`*: `None` default leaves the elastic three leaves untouched;
  set, it is a fourth leaf traversed by `jit`/`vmap`; `from_total_bb`
  pass-through; `validate()` catches shape mismatch, negatives, and
  `a_ph > a`; and **the elastic path is bit-identical with and without
  `a_ph`** — the design's "ignores it" as arithmetic.
- *`Geometry.Ed`*: `None` default adds no leaves; a set override's two
  arrays are leaves and validate; each malformed override rejected (wrong
  arity, length mismatch, non-increasing grid, negative irradiance); elastic
  path bit-identical with and without the override.
- *Exports*: `rt.Inelastic` is `types.Inelastic` and appears in `__all__`.

One float32 lesson re-learned rather than new: a 405-term float32 sum carries
~1e-6 accumulation noise, so its check sits at `rel=1e-5` (the elastic
record's §2 tolerance gotcha, met on the first run of this suite too).

### 2.5 Results — M0 code gate ✅

```
$ pytest -q            # repo root, ocean14, tank server, $OS_COLOR set
309 passed in 53.79s   # 279 elastic (unmodified) + 30 new

$ ruff check robust/   && ruff format --check robust/
All checks passed!  /  23 files already formatted
```

The gate's assertions all hold: the new pytree flattens/unflattens with the
right leaf/static split and traverses `jit`/`vmap`/`grad`; the elastic
hash-regression is pinned and green; the full pre-existing elastic suite
passes unmodified.

**Branch state (for JXP).** The work sits uncommitted on branch
**`inelastic-rt`** — note the name differs from the coding plan's
`rt-inelastic-prototype` (CQ6); recorded as-is, no git run by Claude. Changed
files: `robust/rt/types.py`, `robust/rt/hybrid.py`, `robust/rt/__init__.py`,
new `robust/tests/test_inelastic_types.py`, this record, and the prompt-1
doc (Q&A + logs). One caveat worth JXP's eye: the pinned hashes are
*platform-anchored* — bit-identity across machines/XLA versions is not a JAX
guarantee, so if GitHub CI's hardware produces different low bits the two
hash tests would fail there even though nothing regressed. If that happens,
options are a platform-keyed pin or scoping the hash gate to this machine
(the M2 bing-xcheck precedent). Flagged in the prompt-1 Q&A (Q2).

### 2.6 Notebook

`notebooks/RT/rt_inelastic_coding_1.ipynb` — the M0 explainer (18 cells,
executed, ships with outputs and two figures; kernel `ocean14`, see the §2.3
addendum). Structure: where M0 sits in M0–M4 → the environment on this
machine (the three checked findings: the `git+`/editable hazard, the
GPU-present/CPU-wheel notice, the first 279-green elastic run here) → the
`Inelastic` pytree and the **leaf/static split** (live: `grad` returning a
labelled `∂/∂φ_C` that equals the toy kernel, since `Rrs_fl` is φ_C-linear) →
`a_ph`/`Ed` under the `wind` precedent (live bitwise elastic indifference) →
**why bit-identical rather than merely close** → the baseline in context →
the gate run inline. JAX itself is deliberately not re-explained; the
notebook links to elastic notebook 1 §3–§4. Everything runs from the
committed fixture — no `$OS_COLOR` dependency anywhere in it.

Two things in it worth more than their word count:

- **Figure 1 measures "merely close" instead of asserting it.** The candidate
  threat is real: routing the elastic output once through the
  `rrs ↔ Rrs` conversion pair — algebraically the identity (the maps are
  exact inverses, pinned by an elastic test), and exactly the restructure M2
  could plausibly make since the composition works below the surface —
  changes **39.9 % of the fixture's 12,150 elements**, by 1–2 ULP (median
  8.2e-8, max 2.2e-7 relative). A `rtol=1e-6` gate waves that through; the
  hash lights up. The code-route no-op changes 0 elements, exactly. That
  asymmetry is the quantitative case for the bit-identical gate.
- **A first draft of that figure was wrong in an instructive way.** The
  original demo divided and re-multiplied by a smooth spectrum `f ≈ 1+10⁻³` —
  and float32 round-tripped it *exactly* for all but 5 of 12,150 elements,
  contradicting the narrative around it. Whether an algebraic identity is an
  IEEE identity depends on the operation's magnitude structure, not on it
  "being float": divide-multiply by `f ≈ 1` usually round-trips; the
  non-linear conversion pair usually does not. The notebook keeps the honest
  version; recorded here so nobody reasons from the wrong intuition when M2
  restructures the composition.

Figure 2 puts the frozen baseline in context: median fixture `Rrs` per solar
zenith (single-hue sequence, legend + stated ordering — the three curves sit
within ~5 %, so per-curve direct labels would overprint), with the 550–700 nm
Raman gain region and the 685 nm fluorescence line marked as *where M2 will
touch it*, and the 400–750 nm official-support caveat in the caption text.

### 2.7 The review pass (task 6) — PR #14

Reviewed 2026-08-22 (multi-angle sweep, every finding adversarially verified
against the live code; Cursor's Bugbot pre-review checked finding by finding).
Nine confirmed findings, ranked. **Status 2026-08-23: findings 1–3 and 6–9
are fixed** (the M1 cleanup pass, §3.2.1); findings 4–5 (the `context/RT`
assessment scripts) remain open until that material is next touched:

1. **`ed.Ed()` truncates spectra on integer wavelength grids** — the values
   are cast to `wave.dtype`, so `Ed(30, np.arange(400, 705, 5))` returns all
   1.0s and `ratio()` goes inf/NaN, silently. *Reproduced.*
2. **`l23.select()` silently strips `a_ph`/`Ed`** (and `wind`) — it rebuilds
   `IOPs` field-by-field and geometry via `Geometry.nadir`. Latent until an
   inelastic batch is subset through the Splits masks, then physics vanishes
   with no error. Fix: `tree_map` the pytrees. *Reproduced; found by three
   independent angles.*
3. **`IOPs.from_total_bb` leaves `a_ph` unbroadcast** while broadcasting
   `bb_w` — breaks the constructor's own uniform-batch-shape contract.
   *Reproduced.*
4. **Cursor #1 confirmed**: `make_rt_inelastic_figures.py` resolves L23 via
   `$OS_COLOR_DATA` + hard-coded tank path; project standard is `$OS_COLOR`
   (the elastic sibling checks both).
5. **Cursor #2 confirmed**: `write_metrics` reuses the Raman error columns
   with different meanings on `ChlFl` rows (single/double Gaussian vs
   flat-Ed/true-Ed) — the committed CSV mixes quantities under one header.
6. `ed.py`'s zenith interpolation hard-codes the 30° stride instead of
   reusing `_interp_wave`'s searchsorted — silent wrong numbers if M5 adds a
   non-uniform anchor.
7. `tiny_args()` duplicated verbatim into `test_inelastic_types.py`
   (belongs in `conftest.py`).
8. `ed.load_table()`'s hand-rolled `_TABLE` cache vs the package's
   `@functools.cache` idiom.
9. `robust/rt/__init__.py`'s Status paragraph still calls
   `ztt`/`emulator`/`hybrid` stubs (staleness predates the PR; two-line fix).

Cleared explicitly: `setup.py` packaging (empirical build check — all three
`.npz` ship), `.gitignore` (no tracked file shadowed), CI dependency
coverage, split leak-freedom, and the `Md_plus` in-water cosine (the review
re-derived it and confirmed the code, not the docstring formula, is right).

**Out-of-scope finds worth JXP's eye** (elastic-era code, PR #13's scope, not
this diff): `ztt.bb_tilde` crashes on the per-sample scalar `B_p` that
`PhaseParams` documents and the emulator supports (verified crash);
`on_out_of_domain='ztt'` is a silent no-op for a domain-less emulator;
`emulator.out_of_domain` can pair `worst` with the wrong side's `excess`;
`write_fixture` keeps only the last zenith's wavelength grid unchecked; the
committed `.claude/settings.json` carries an allowlist rule ending in
`python -c ' *` (auto-approves arbitrary inline Python) plus several dead
machine-specific paths.

### 2.8 The CI hash failures (task 7) — the Q2 caveat materialized

PR #14's two failing tests were exactly the two elastic hash-regressions, and
exactly for the reason flagged in prompt 1 Q&A Q2: **the pins are
platform-anchored**. CI computed `e6488bf0…`/`3cf02bc9…` against the tank
pins `aaa06161…`/`d1114640…`. The decisive observation: **within a single CI
run, one matrix job reproduced the tank bits (328-equivalent green) and the
other did not** — and which Python version failed flipped between runs. So
the divergence is per-*runner* (GitHub's fleet is heterogeneous; XLA's CPU
codegen differs across microarchitectures), which rules out a platform-keyed
pin: there is no stable "CI platform" to key on.

**Fix — the gate now has two tiers** (per the option JXP pre-approved in Q2,
following the M2 bing-xcheck precedent; nothing was re-pinned):

- **Strict tier** (`test_elastic_hash_regression_strict`): the SHA-256 pins
  *plus* element-wise equality against the committed reference (below), so a
  failure names positions, not just digests. `skipif` when `$CI` is set;
  mandatory-green on dev machines — verified green here.
- **Closeness tier** (`test_elastic_regression_close_everywhere`): runs on
  every platform. New committed fixture
  `robust/tests/files/elastic_reference_outputs.npz` (88 kB) holds the
  pre-change `Rrs`/`rrs` arrays — **its bytes hash to the strict pins**, and
  the test asserts that first, so the reference cannot drift from the pin.
  Then `allclose` at `rtol 5e-7` (~4 ULP): CI's observed runner spread is
  1–2 ULP, while a genuine restructure of the elastic route (the notebook-1
  §4 conversion round trip) fails broadly. Generated by
  `write_elastic_reference()` in `design/py/gen_inelastic_fixture.py`
  (atomic write, verify-before-replace).

Verified: local `pytest` 328 passed (strict enforced); with `CI=true`
simulated, the strict test skips with an explanatory reason and the
closeness tier passes; ruff clean. CI itself goes green when JXP pushes.

---

## 3. M1 — Ed module, excitation grid, X2/X4 data

**Goal.** Packaged `Ed(θ_s, λ)` with the `Geometry.Ed` override, the Raman
excitation-grid helpers, X2/X4 loaders with truth channels reusing the
elastic splits verbatim, and the sibling CI fixture (same 50 scene indices as
the elastic fixture; the elastic fixture's bytes untouched — CQ4).

### 3.1 Task status

| # | Task | Status |
|---|------|--------|
| 1 | Ed module: `design/py/gen_inelastic_fixture.py` (part 1) + `robust/rt/data/ed_l23.npz` + `robust/rt/ed.py` + `test_ed.py` | ✅ done |
| 2 | Excitation-grid infrastructure in `conventions.py` | ✅ done |
| 3 | X2/X4 data + truth channels + sibling CI fixture | ✅ done |
| 4 | `notebooks/RT/rt_inelastic_coding_2.ipynb` | ✅ done |
| 5 | Update `rt_inelastic_coding_prompt_3.md` with the actual M1 outcome | ✅ done — **M1 complete** |

### 3.2 Modules added (task 1)

**`design/py/gen_inelastic_fixture.py` (part 1)** — extracts `Ed(0⁺)(λ)` from
the X=2 files. Asserts, per zenith, **before** collapsing: scene-independence
(< 10⁻³ relative scatter — measured **4.9e-5 / 4.7e-5 / 4.7e-5** at 0°/30°/60°,
float32 storage noise) and identity across X=1/2/4 (the sky must not depend on
the IOP scenario — verified, `rtol 1e-6`). Collapses by float64 mean, stores
float32; writes atomically with verify-before-replace (the elastic
`write_fixture` / PR #11 discipline), including an ordering check (lower sun →
less irradiance at every wavelength, so scrambled rows cannot ship). Part 2
(the sibling fixture) is reserved for task 3.

**`robust/rt/data/ed_l23.npz`** — 2 kB package data: `wave` (81), `zeniths`
(0/30/60), `Ed` (3×81). `Ed(440)` = 1.5435 / 1.3010 / 0.6467 W m⁻² nm⁻¹ at
0°/30°/60°. `setup.py`'s `package_data` gained `rt/data/*.npz` — it listed
only `rt/files` and `tests/files`, so a pip-installed `robust` would have
silently lacked the table.

**`robust/rt/ed.py`** — `Ed(theta_s, wave=None, *, override=None)` and
`ratio(theta_s, wave_num, wave_den, *, override=None)`; re-exported as
`robust.rt.ed`. Key decisions:

- **Zenith interpolation is linear between the three anchors and *clamps*
  outside 0–60°** — the `conventions.bb_w` precedent: no silent extrapolation,
  no boundary `raise` that could not run under `jit`. Differentiable in
  `theta_s` (piecewise-linear), batched, `jit`/`vmap`-safe.
- **Wavelength interpolation is a hand-rolled clamped linear gather**
  (`_interp_wave`), not `jnp.interp` — the latter is 1-D only, and the blended
  spectrum is batched. Differentiable in the spectrum values, which is the
  property task 2's excitation-grid interpolation needs; here it comes free.
- **The table loads lazily and is cached as NumPy** (the `conventions.WAVE`
  reasoning: a device array built at import would fix its dtype before a
  caller enables x64); `load_table()` refuses a file whose zenith rows are not
  (0, 30, 60), so the module and the packaged file version each other.
- **An override replaces the sky entirely and `theta_s` is then ignored** — an
  override *is* one particular sky, zenith dependence included. `ratio` exists
  so λ′ and λ are guaranteed to be evaluated from the *same* sky (packaged in
  the numerator, override in the denominator can never happen).
- **The DQ5 solar-model caveat is the module docstring's centerpiece**: v1
  inherits L23's solar spectrum deliberately (consistency with the truth data
  over absolute solar accuracy); the override is the seam for TSIS-era or PACE
  skies later.

### 3.2.1 The PR #14 review debt cleared (2026-08-23)

The robust/-side findings of §2.7 are fixed; the elastic hash pins stayed
green throughout (none of the touched code sits on the elastic route):

- **`ed.py` dtype rule (finding 1) — promote, never truncate.** All inputs
  are coerced to one common floating dtype via `jnp.result_type`; integer
  wavelength grids select float nodes (regression test:
  `test_integer_wavelength_grid_returns_float_irradiances`). The first fix
  attempt — casting everything to `wave.dtype` — *failed the module's own
  gradient gate in full-suite order*: a device conversion cached before
  `jax_enable_x64` is toggled stays float32, so `canonical_wave()` can
  return float32 under x64 mid-session, and truncating `theta_s` to it
  silently degraded the float64 FD comparison to float32 (rel err 4e-3 vs
  the 1e-6 gate). Promotion keeps a float64 `theta_s` in float64 regardless
  of `wave`'s dtype. Recorded because M1 task 2's excitation-grid
  interpolation will meet the identical trap.
- **`ed.py` zenith interpolation (finding 6)**: now the same clamped
  `searchsorted` rule as the wavelength axis, via a shared
  `_interp_weights` helper — no stride assumption; any strictly increasing
  anchor set works.
- **`ed.load_table` (finding 8)**: `@functools.cache`, the package idiom.
- **`types.from_total_bb` (finding 3)**: `a_ph` is broadcast like `bb_w`
  (regression test: `test_iops_from_total_bb_broadcasts_a_ph_like_bb_w`).
- **`l23.select()` (finding 2)**: `IOPs`/`PhaseParams` are subset leaf-wise
  with `tree_map` (every present and future field survives); `Geometry` is
  rebuilt per-sample-field-by-field **deliberately, not by tree_map** — its
  `Ed` override is one sky for the whole batch (two 1-D spectral arrays),
  so indexing it by the sample mask would corrupt it. `wind` is subset,
  `Ed` carried whole (regression test:
  `test_select_preserves_optional_fields`).
- **`tiny_args()` (finding 7)**: one copy, in `conftest.py`; both test
  modules import it.
- **`robust/rt/__init__.py` (finding 9)**: the Status paragraph now states
  the elastic prototype is complete and the inelastic types/`ed` have
  landed, physics at M2.

`pytest -q` → **331 passed** (328 + 3 regression tests); `CI=true`
simulation: strict tier skips, closeness tier green; ruff clean. Open from
§2.7: findings 4–5 (`context/RT` figure script + CSV), deferred to the next
touch of the assessment material.

### 3.2.2 Excitation-grid infrastructure (task 2)

**`robust/rt/conventions.py`** gains a "Raman excitation grid" section:

- `RAMAN_SHIFT = 3400.0` cm⁻¹ (Ge et al. 1993; equals
  `bing.rt.raman.WAVENUMBER_SHIFT_CENTER`, asserted by a test) and
  `RAMAN_WAVE_MIN_OFFICIAL = 400.0` nm.
- `raman_excitation(wave, shift=RAMAN_SHIFT)` — **the** excitation map λ′(λ)
  in wavenumber form (`1/λ′ = 1/λ + shift`) — and its exact inverse
  `raman_emission`. Pure, `jit`-able, differentiable. Pinned values:
  488 nm emission ← 418.553 nm excitation; 488 nm excitation → 585.076 nm
  emission (the corrected numbers from M0 task 5, cross-checked at rtol
  1e-12 against bing's pair).
- `interp_spectrum(wave_new, grid, spectra)` — **the package's one
  interpolation rule**, *promoted from `ed.py`* rather than written twice:
  clamped linear (constant extrapolation beyond the grid — the `bb_w`
  precedent and the documented sub-400 nm caveat), batched via
  weight-gathering (`jnp.interp` is 1-D only), differentiable in the
  spectrum values (the property the Raman term needs: gradients flow through
  excitation-grid IOPs back to the IOP inputs) and in `wave_new`, with the
  §3.2.1 promotion rule applied from birth (all inputs to one common
  floating dtype — integers select nodes, never truncate values). `ed.py`
  now aliases these helpers instead of owning copies.

**Key decision — support is documented and *derived*, not enforced by
error.** The official λ ≥ 400 nm bound exists because
`raman_excitation(400) = 352.11 nm`, just inside the L23 grid
(`WAVE_MIN = 350`); below 400 nm the maps still run and `interp_spectrum`
clamps — a caveat, not a gate (design §3). A test asserts the 352.11-inside-
the-grid *rationale* itself, so a future grid change that silently invalidates
the bound fails loudly.

**Tests** (`test_conventions.py`, +9): the pinned values and exact wavenumber
form (float64 via the fixture — the exact-form comparison at rtol 1e-12
*failed in float32 on its first run*, the elastic record's §2 dtype-tolerance
lesson met yet again, now noted in the test's own docstring); exact inverse
round-trip over the full grid at 1e-14; the bing cross-check
(`importorskip`); the support-bound rationale; `interp_spectrum` vs
`numpy.interp`, batched + clamped, integer-input promotion, `jit`, `grad`
w.r.t. the wavelengths, and **`grad` w.r.t. the spectrum values vs per-node
central differences** (atol 1e-9, plus the unit-total-weight identity
`Σ∂ = n_targets`).

**Results.** `pytest -q` → **340 passed** (331 + 9); elastic hash pins green;
`ruff check` + `format` clean.

### 3.2.3 X2/X4 data + sibling fixture (task 3)

**`robust/rt/data/l23.py`** gains the inelastic pipeline, mirroring the
elastic layout decision for decision:

- Constants: `INELASTIC_XS = (2, 4)`; `PHI_C_L23 = 0.02` — the quantum yield
  HydroLight used for X=4, i.e. the truth channel `Rrs_X4 − Rrs_X2` is
  fluorescence *at exactly this yield*, which is why `Inelastic.phi_C`
  defaults to it.
- `L23InelasticBatch`: the elastic container's sibling — same flat
  zenith-major sample axis, same host-side `scene` labels, not a pytree —
  with `iops.a_ph` set and three reference channels (`Rrs_x1/x2/x4`). Truth
  channels are *properties* (`truth_raman_factor = Rrs_X2/Rrs_X1`,
  `truth_fluorescence = Rrs_X4 − Rrs_X2`), asserted bitwise-equal to their
  definitions by a test so property and prose cannot drift. `validate()`
  additionally **requires `a_ph`** — an inelastic batch without the
  fluorescence source term is a loader bug, not a configuration.
- `load_inelastic_batch(zeniths, scenes=, validate=, reader=)` with the
  elastic reader seam. **One read spans all three scenarios**: the IOPs and
  `aph` are *bit-identical* across X=1/2/4 at every zenith (measured
  2026-08-24), so the default reader reads them once from X=1 and **asserts**
  equality against X=2/X=4 — a future release that varies them stops the
  loader instead of silently mixing scenarios.
- `select_inelastic` shares `_take_tree`/`_take_geometry` with the elastic
  `select` (factored out of the §3.2.1 fix), so the subset-leaf-wise /
  carry-Ed-whole lessons are applied once.
- `make_splits` needed **no change**: it reads only `scene`/`zenith`, which
  is what "reuse the elastic splits verbatim" means mechanically. Proved,
  not assumed: mask-for-mask equality tests at fixture scale (CI) and on the
  full 9960-sample release (data-gated).

**The sibling CI fixture** (`robust/tests/files/l23_inelastic_fixture.npz`,
**249 kB** — inside the corrected ≲300 kB bound; the plan's original 200 kB
was arithmetically impossible): the elastic fixture's 50 scenes ×
{a, aph, bb, Rrs1, Rrs2, Rrs4} × 3 zeniths, float32, written by
`write_inelastic_fixture()` (generator part 2, atomic verify-before-replace
through the *real* loader). Deliberate redundancy: the `a`/`bb`/`Rrs1`
copies exist so `inelastic_npz_reader(sibling, elastic)` can **prove** the
two fixtures describe the same water — any mismatch raises (tested with a
corrupted sibling). `bbnw`/`bnw` stay in the elastic fixture, whose bytes
are untouched — now enforced by a **SHA-256 pin of the elastic fixture
file** in the test module, turning CQ4's sentence into a failing test.

**Tests** (`test_l23_inelastic_data.py`, 15): fixture presence/size and the
CQ4 byte pin; the real loader running from the two committed files alone
(no `$OS_COLOR`); the corrupted-sibling rejection; truth-channel identity
(bitwise); golden absolute pins (`Rrs2(440, scene 0, 0°) = 9.2462e-3`,
`Rrs4(685) = 1.2273e-4`, `aph(440) = 3.711e-3`) as change-detectors the
netCDF cross-check cannot provide; physicality (Raman factor ≥ 1 everywhere
— measured min 1.0076 — and growing toward the red; fluorescence delta > 0
at 685 nm in every sample, with 685 the median spectral argmax); **split
equality with the elastic splits** (fixture scale + full release);
sample-ordering bitwise identity (`Rrs_x1` *is* the elastic `Rrs`);
subsetting incl. `a_ph`; the `a_ph`-required validation. Raw-netCDF halves
carry `needs_l23_inelastic` and skip cleanly when the data is absent
(verified with `$OS_COLOR` unset: 13 passed, 2 skipped).

**Results.** `pytest -q` → **355 passed**; elastic hash pins green; ruff
clean. Branch state for JXP: all M1 work (tasks 1–3 + the review-debt
cleanup) sits uncommitted on `inelastic-rt` on top of your "prompt 1"
commit.

### 3.2.4 Notebook (task 4)

`notebooks/RT/rt_inelastic_coding_2.ipynb` — the M1 explainer (11 cells,
executed, three figures; kernel `ocean14`). Organised around what M1
*decided*, linking to the assessment
(`context/RT/rt_inelastic_bing_summary.md` + its committed figures) rather
than re-deriving it, and running from **committed files only** (packaged
`ed_l23.npz` + the two CI fixtures — no `$OS_COLOR` anywhere in it).

- **Figure 1** — the packaged sky: the three `Ed(0⁺)` spectra with the
  scene-independence and scenario-identity assertions stated, and the DQ5
  solar-model caveat with the `Geometry.Ed` seam.
- **Figure 2** — the ratio the Raman term actually consumes:
  `Ed(λ′)/Ed(λ)` over the official band for all three zeniths against the
  flat-Ed line. Measured on the packaged data: **0.44 at 445 nm to 1.59 at
  720 nm — a ×3.6 swing**, nearly zenith-independent (the sky's shape
  barely changes with sun angle; its amplitude cancels in the ratio). This
  is the quantitative footing of the assessment's +60 %/−50 % flat-Ed
  error.
- **Figure 3** — the truth channels for three fixture scenes spanning the
  trophic range (picked by `a_ph(440)`): the Raman factor is *largest for
  the clearest water* (weakest elastic red signal — the oligotrophic curve
  sits on top), reaching 1.36 on the fixture and 2.5 on the full release;
  the 685 nm fluorescence delta grows with biomass (order 10⁻⁵–10⁻⁴ sr⁻¹
  at φ_C = 0.02).
- **Split reuse** demonstrated live: mask-for-mask equality printed from
  the real loaders, plus the bitwise `Rrs_x1 == elastic Rrs` identity.

One honesty catch made before committing outputs: the truth-channel panel
originally titled the Raman factor "up to more than 2×" — true of the full
release but not of the 50 fixture scenes actually plotted (max 1.36). The
committed version distinguishes the two explicitly; a figure must not
borrow numbers its own data doesn't show.

### 3.3 Tests (task 1)

`robust/tests/test_ed.py` — **19 tests**; only one needs data beyond the repo.
Also new in `conftest.py`: `L23_INELASTIC_FILES` (the six X=2/X=4 netCDFs),
`l23_inelastic_available()`, and the **`needs_l23_inelastic`** marker — the gap
flagged when prompt 2 was written: `needs_l23` guards only the elastic files,
so a machine with partial data must skip the inelastic raw-netCDF tests, not
fail them.

- *Packaged table* (CI-safe): shape/grid (the grid **is** `conventions.WAVE`),
  positivity; monotone fall with zenith (re-asserted from the generator so a
  scrambled file fails in CI too); and a **units sanity bound** — peak
  `Ed(0°)` must be O(1) W m⁻² nm⁻¹, since a mW/µm-type slip would pass every
  shape test.
- *Golden* (`needs_l23_inelastic`): packaged vs raw X=2 netCDF means at
  `rtol 1e-5`, all three zeniths.
- *Zenith interpolation*: exact at the three anchors; midpoints are exact
  anchor means; clamped at −5° and 75°; batched `theta_s` row-consistent.
- *Wavelength interpolation*: matches `numpy.interp` off-grid; clamps at the
  350/750 nm ends.
- *Override*: replaces the packaged sky, ignores `theta_s`, interpolates; the
  `Geometry.Ed` pair plumbs through verbatim.
- *Ratio*: matches a `numpy.interp` hand computation; and **is not flat** over
  the Raman excitation map (max/min > 1.5) — the spectral structure whose
  neglect caused the assessment's +60 %/−50 % flat-Ed error.
- *JAX*: `jit`/`vmap` agree with eager; `grad` w.r.t. `theta_s` matches
  central differences (float64 fixture, dtypes pinned); `grad` w.r.t. override
  values is finite and positive.

### 3.4 Results (task 1)

```
$ pytest -q            # repo root, ocean14, tank server, $OS_COLOR set
328 passed in 61.82s   # 279 elastic + 30 M0 (hash pins green) + 19 test_ed
$ ruff check / ruff format --check
clean
```

Remaining M1 sections (excitation grid, X2/X4 data, notebook) will be added
as their tasks land.

---

## 4. M2 — Analytic terms in JAX

**Goal.** `inelastic.py::raman_factor` and `::fluorescence_kernel` — the
fixed-BING physics ported to differentiable JAX — composed into
`hybrid.forward` per design §2; cross-checked live against fixed BING at
rtol ≤ 1e-6 (CQ3a) and characterized against the assessment's error table.

*Not started.*

---

## 5. M3 — Correction heads δ_R and δ_F

**Goal.** Two small bounded Flax heads
(`f_R = 1 + (f_phys − 1)(1 + δ_R)`; `Rrs_fl = φ_C · K_fl · (1 + δ_F)`),
trained on the X2/X1 and X4−X2 truth channels; committed weights; per-process
delta gates ≤ 5 % on held-out scenes at every zenith including 0°.

*Not started.*

---

## 6. M4 — Validation — *prototype complete*

**Goal.** The design §6 protocol: held-out total rRMS vs X4 ≤ 0.5 % per
zenith, per-process deltas ≤ 5 %, elastic hash-regression, gradient checks
incl. φ_C, speed ≤ 2× elastic; review pass; metrics + figures committed under
`design/validation/`.

*Not started.*

---

## 7. Module index (current)

Inelastic surface as of M0 task 3 (the elastic surface is otherwise
unchanged — see `rt_elastic_implementation.md` §9):

| Module | Inelastic surface | Since |
|---|---|---|
| `robust.rt.types` | `IOPs.a_ph` (optional), `Inelastic`, `EMISSION_SHAPES`, `Geometry.Ed` (optional) | M0 |
| `robust.rt.hybrid` | `forward(..., inelastic=None)`, `rrs_forward(..., inelastic=None)` — `None` = elastic route; instance raises until M2 | M0 |
| `robust.rt` | re-exports `Inelastic`; `ed` submodule | M0/M1 |
| `robust.rt.ed` | `Ed(θ_s, λ)` + `ratio(λ′/λ)` from packaged L23 spectra; `Geometry.Ed` override; `ZENITH_ANCHORS`, `load_table` | M1 |
| `robust.rt.conventions` | `RAMAN_SHIFT`, `RAMAN_WAVE_MIN_OFFICIAL`, `raman_excitation`/`raman_emission`, `interp_spectrum` | M1 |
| `robust/rt/data/ed_l23.npz` | the three packaged `Ed(0⁺)` spectra (2 kB) | M1 |
| `design/py/gen_inelastic_fixture.py` | part 1: Ed extraction w/ scene-independence + cross-scenario asserts | M1 |
| `robust/tests/test_inelastic_types.py` | the M0 gate, incl. the pinned elastic hash-regression | M0 |
| `robust/tests/test_ed.py` | the M1 task-1 gate | M1 |
| `robust.rt.data.l23` | `L23InelasticBatch`, `load_inelastic_batch`, `select_inelastic`, `inelastic_npz_reader`, `INELASTIC_XS`, `PHI_C_L23` | M1 |
| `robust/tests/files/l23_inelastic_fixture.npz` | the sibling CI fixture (CQ4, 249 kB) | M1 |
| `robust/tests/test_l23_inelastic_data.py` | the M1 task-3 gate, incl. the CQ4 byte pin and split-equality proofs | M1 |
| `robust/tests/conftest.py` | `L23_INELASTIC_FILES`, `needs_l23_inelastic` | M1 |
