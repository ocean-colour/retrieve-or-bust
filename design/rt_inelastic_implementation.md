# Inelastic RT Implementation Record

**Version:** 0.1
**Date:** 2026-08-20
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
| **M0** | Environment (this machine) & API extension | 🟡 in progress | `robust.rt.types` (extend), `robust.rt.hybrid` (extend), `robust/tests/test_inelastic_types.py` |
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
JAX. As of 2026-08-20 `ocean14` here has **no `jax`**; installing the stack
(already declared in `requirements.txt` by the elastic effort) is an M0 task,
repeating the elastic install procedure (dry-run first, purely-additive
verification, fallback to a dedicated env if not additive).

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

**Verification (current).** M0 in progress — nothing to verify yet. This
section will carry the canonical `pytest -q` / `ruff` counts as milestones
land.

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
| 2 | Install the JAX stack into `ocean14` on this machine; verify purely additive; record versions | ⬜ not started |
| 3 | Extend the API (`types.py`: `IOPs.a_ph`, `Inelastic`, `Geometry.Ed`; `hybrid.forward(..., inelastic=None)`); pass the gate incl. the pinned elastic hash | ⬜ not started |
| 4 | `notebooks/RT/rt_inelastic_coding_1.ipynb` — the M0 explainer, executed | ⬜ not started |
| 5 | Update `rt_inelastic_coding_prompt_2.md` with the actual M0 outcome | ⬜ not started |

### 2.2 Modules added / extended

*To be filled by task 3.* Planned surface (coding plan M0):

- `robust/rt/types.py` — `IOPs` gains optional `a_ph` (default `None`; the
  elastic path ignores it); new
  `Inelastic(phi_C=0.02, raman=True, fluorescence=True,
  emission_shape='single', cdom_fl=None)` registered pytree; `Geometry` gains
  optional `Ed` override (`(wave_Ed, Ed)`). All strictly backward compatible.
- `robust/rt/hybrid.py` — `forward(..., inelastic=None)`; `None` routes
  through the existing elastic code path untouched (no-op by construction).

### 2.3 Environment

*To be filled by task 2.* Will record: the `pip install --dry-run -r
requirements.txt` additivity check, exact versions of `jax`/`jaxlib`/`flax`/
`optax`/`jaxtyping` installed into `ocean14` on this machine, the
`jax.default_backend() == "cpu"` / x64 / `jax.grad` smoke verifications, and
the regression check that the pre-existing stack (`numpy`, `scipy`, `xarray`,
`pandas`, `matplotlib`, `emcee`, `bing`, `ocpy` incl.
`ocpy.hydrolight.loisel23`) still imports.

Known starting point (verified 2026-08-20): `ocean14` on this machine has no
`jax`; the elastic implementation record §2.3 describes the *laptop* install,
not this machine.

### 2.4 Tests

*To be filled by task 3.* Planned: `robust/tests/test_inelastic_types.py`
(pytree mechanics, defaults, `jit`/`vmap` traversal of the new types) and the
elastic hash-regression test pinning `forward(..., inelastic=None)` on the
elastic CI fixture.

### 2.5 Results

*To be filled — the M0 gate has not yet been run.*

### 2.6 Notebook

*To be filled by task 4.* Planned: `notebooks/RT/rt_inelastic_coding_1.ipynb`
— what M0 *decided* (the `Inelastic` pytree shape; why `inelastic=None` must
be bit-identical rather than merely close; the environment verification),
linking to elastic notebook 1 rather than re-explaining JAX. Follows the
elastic notebook conventions: degrades gracefully without `$OS_COLOR`,
`sys.path` bootstrap, the recorded figure style.

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

*To be populated as modules land. As of M0-in-progress, no inelastic module
exists yet; the elastic surface is unchanged — see
`rt_elastic_implementation.md` §9.*
