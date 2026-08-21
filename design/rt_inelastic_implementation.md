# Inelastic RT Implementation Record

**Version:** 0.5
**Date:** 2026-08-21
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
| **M1** | Ed module, excitation grid, X2/X4 data | ⬜ not started | `robust.rt.ed`, `robust.rt.conventions` (extend), `robust.rt.data.l23` (extend), `robust/rt/data/ed_l23.npz`, sibling CI fixture |
| **M2** | Analytic terms in JAX | ⬜ not started | `robust.rt.inelastic`, composition in `robust.rt.hybrid` |
| **M3** | Correction heads δ_R, δ_F | ⬜ not started | `robust.rt.inelastic_corr`, `robust/rt/files/{raman,fl}_corr_l23.npz`, `design/py/train_inelastic_corr.py` |
| **M4** | Validation (*prototype done*) | ⬜ not started | `robust.rt.validation` (extend), `design/py/run_validation.py` (extend), `design/validation/` |

Legend: ✅ done · 🟡 in progress · ⬜ not started.

**Branch.** All milestone work lands on **`rt-inelastic-prototype`**; each
milestone is a reviewable commit/PR for JXP (Claude runs no state-changing
git — see `CLAUDE.md`).

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

**Verification (current).** `pytest -q` from the repo root → **309 passed**
(`ocean14`, this machine, 2026-08-20, `$OS_COLOR` set): the 279 elastic tests
unmodified plus the 30 new M0 tests, including the two pinned elastic
hash-regressions. `ruff check` and `ruff format --check` clean.

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

---

## 3. M1 — Ed module, excitation grid, X2/X4 data

**Goal.** Packaged `Ed(θ_s, λ)` with the `Geometry.Ed` override, the Raman
excitation-grid helpers, X2/X4 loaders with truth channels reusing the
elastic splits verbatim, and the sibling CI fixture (same 50 scene indices as
the elastic fixture; the elastic fixture's bytes untouched — CQ4).

*Not started. Sections (task status, modules, tests, results, notebook) to be
added when M1 opens — see `rt_inelastic_coding_prompt_2.md`.*

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
| `robust.rt` | re-exports `Inelastic` | M0 |
| `robust/tests/test_inelastic_types.py` | the M0 gate, incl. the pinned elastic hash-regression | M0 |
