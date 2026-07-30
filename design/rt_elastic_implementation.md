# Elastic RT Implementation Record

**Version:** 0.2
**Date:** 2026-07-31
**Authors:** JXP and Claude

**Status:** living document — updated as each milestone is implemented.
**Companions:** implements [`rt_elastic_model.md`](rt_elastic_model.md) (design,
*what/why*) following
[`rt_elastic_model_coding_plan.md`](rt_elastic_model_coding_plan.md) (plan,
*how/when*). This document records *what was actually built*: the modules, their
public API, the implementation decisions taken, the tests, and the numerical
results per milestone.
**Versioning convention** (same as the design docs): minor bump for substantive
changes (0.1 → 0.2), extra decimal for small edits (0.1.1); update the Date on
every bump.

> The chronological narrative lives in the "Logs" sections of the per-milestone
> prompts, `claude_prompts/RT/rt_elastic_coding_prompt_*.md`. This document is
> the structured, current-state reference for `robust/rt/` as implemented.

---

## 1. Status at a glance

| M | Goal | Status | Package surface |
|---|------|--------|-----------------|
| **M0** | Environment & scaffold | 🟡 in progress | `robust.rt` (stubs), `robust/tests/` |
| **M1** | Data & conventions | ⬜ not started | `robust.rt.{conventions,types}`, `robust.rt.data.l23` |
| **M2** | ZTT analytic backbone (JAX) | ⬜ not started | `robust.rt.ztt` |
| **M3** | Residual emulator + hybrid | ⬜ not started | `robust.rt.{emulator,hybrid}` |
| **M4** | Validation (*prototype done*) | ⬜ not started | `robust.rt.validation`, `design/py/run_validation.py` |
| **M5** | Beyond week 1 | ⬜ future | — |

Legend: ✅ done · 🟡 in progress · ⬜ not started.

**Branch.** All milestone work lands on `rt-elastic-prototype`; each milestone is
a reviewable commit for JXP (Claude runs no state-changing git — see
`CLAUDE.md`).

**Acceptance philosophy.** Accuracy gates are *relative* ("beats standard
Gordon on the held-out splits"), never blind absolute targets; absolute rRMS and
latency are **reported** here, not thresholded. The gradient-correctness check
(`jax.grad` vs central finite differences) is a hard gate from M2 onward.

**Verification (current).** JAX 0.11.0 imports and runs on the CPU backend in
`ocean14` (§2.3). No test suite yet — M0 task 3 (scaffold + the `pytest` gate) is
outstanding; see §2.

---

## 2. M0 — Environment & scaffold

**Goal.** A green, importable base: JAX (CPU) in `ocean14`, the `robust/rt/`
stub package, and `robust/tests/` in BING layout. Nothing scientific.

**Gate.** `pytest -q` collects and passes; `robust/tests/test_env.py` asserts
`import jax` works on CPU and `from robust import rt` succeeds.

### 2.1 Task status

| # | Task | Status |
|---|------|--------|
| 1 | Implementation record (`design/rt_elastic_implementation.md`) | ✅ done |
| 2 | Dependencies: `jax`, `flax`, `optax`, `jaxtyping` (CPU) in `requirements.txt` + `ocean14` | ✅ done |
| 3 | Scaffold `robust/rt/` + `robust/tests/`; pass the `pytest` gate | ⬜ pending |

### 2.2 Modules added

*(none yet — task 3.)* Planned stubs, per the coding plan's package layout:

```
robust/rt/
  __init__.py         # will export forward(), public types
  types.py            # IOPs, PhaseParams, Geometry pytrees (jaxtyping)
  conventions.py      # A=0.52, B=1.7; wavelength grid; bb_w model; asserts
  data/l23.py         # L23 loader via ocpy.hydrolight.loisel23
  ztt.py              # Rrs_ZTT analytic backbone (JAX)
  emulator.py         # Flax MLP ΔRrs + Optax training
  hybrid.py           # forward(): Rrs_ZTT + ΔRrs; mode flag
  validation.py       # rRMS / speed / gradient protocol
robust/tests/
  conftest.py, files/, test_env.py
```

### 2.3 Environment

`ocean14` as of 2026-07-31, after the M0 task-2 install:

| Package | Version | Note |
|---|---|---|
| Python | 3.14.6 | conda env `ocean14` (miniforge) |
| jax | **0.11.0** | new — CPU backend |
| jaxlib | **0.11.0** | new — pulled by `jax` (CPU wheel) |
| flax | **0.12.8** | new — MLP (M3) |
| optax | **0.2.8** | new — training (M3) |
| jaxtyping | **0.3.11** | new — public-signature annotations |
| numpy | 2.4.6 | unchanged |
| scipy | 1.18.0 | unchanged |
| xarray | 2026.4.0 | unchanged |
| pytest | 9.1.1 | unchanged |
| `bing` | 0.0.dev0 | unchanged |
| `ocpy` | installed (no `__version__`) | unchanged |

Transitive additions (all new, nothing upgraded or removed): `absl-py`,
`aiofiles`, `etils`, `humanize`, `markdown-it-py`, `mdurl`, `ml_dtypes`,
`msgpack`, `opt_einsum`, `orbax-checkpoint`, `prometheus-client`, `protobuf`,
`rich`, `simplejson`, `tensorstore`, `treescope`, `uvloop`, `wadler-lindig`.

**Dependency declaration.** `requirements.txt` gains a CPU-JAX block (`jax`,
`flax`, `optax`, `jaxtyping`) and the same four are mirrored into `setup.py`'s
`install_requires`, matching this repo's existing practice of keeping the two
lists in sync. `jaxlib` is deliberately *not* pinned separately — `jax` requires
the matching version, so listing it would only invite version skew. Unpinned,
like every other entry in the file. A GPU build would be `jax[cuda12]`; on macOS
arm64 the plain wheel is CPU-only (Metal would be a separate `jax-metal`
plugin), so "CPU-only" needs no extra machinery here.

**Verification (task-2 gate).** In `ocean14`:

- `import jax; jax.numpy.ones(3)` → `[1. 1. 1.]` on `CpuDevice(id=0)`;
  `jax.default_backend() == "cpu"`, `jax.devices() == [CpuDevice(id=0)]`.
- `jax.config.update("jax_enable_x64", True)` yields float64 arrays — the
  precision the M2 finite-difference gradient gate needs is available.
- `jax.grad` smoke: `d/dx Σx² = 6.0` at `x = 3`.
- Regression check on the pre-existing env: `numpy`, `scipy`, `xarray`,
  `pandas`, `matplotlib`, `scikit-learn`, `emcee`, `bing`, `ocpy` (incl.
  `ocpy.hydrolight.loisel23`, the M1 loader) all still import.

**Risk retired.** The coding plan flagged that a JAX install might perturb
`ocean14` (fallback: a dedicated `rt-jax` env). A `pip install --dry-run` first
confirmed the install was purely additive — `numpy>=2.1` and `scipy>=1.15` were
already satisfied and nothing was scheduled for uninstall — and the
post-install import check above confirms it. **We stay in `ocean14`**; no
separate env is needed.

`robust` itself is already packaged (`setup.py` uses `find_packages()`), so
`robust.rt` and `robust.rt.data` become importable as soon as the `__init__.py`
stubs exist.

### 2.4 Tests

*(none yet — task 3.)* `test_env.py` will cover: `import jax` +
`jax.numpy.ones(3)` on the CPU backend (asserting the default device platform is
`cpu`), and `from robust import rt`.

### 2.5 Results

Task 2 verified by hand (§2.3): JAX 0.11.0 on the CPU backend, float64
available, `jax.grad` working, `ocean14` otherwise unchanged. The same checks
become the automated `test_env.py` gate in task 3; `pytest` has not been run yet.

---

## 3. M1 — Data & conventions

*(not started; see the coding plan §M1.)*

## 4. M2 — ZTT analytic backbone (JAX)

*(not started; see the coding plan §M2.)*

## 5. M3 — Residual emulator + hybrid

*(not started; see the coding plan §M3.)*

## 6. M4 — Validation

*(not started; see the coding plan §M4. Passing the M4 gate is the Week-1
prototype's definition of done.)*

## 7. M5 — Beyond week 1

*(future; detailed once M4 results are in.)*

---

## 8. Cross-cutting conventions (as implemented)

*Filled in as they land. Intended (from the design + coding plan):*

- **Conventions asserted once** at the package boundary: `A_RRS = 0.52`,
  `B_RRS = 1.7` (Lee 2002, matching BING); the canonical L23 wavelength grid
  (350–750 nm, 81 bands); the pure-water `bb_w(λ)` model.
- **Testing** — BING layout: `robust/tests/test_*.py`, one module per `rt`
  module, with `conftest.py` fixtures and a `files/` fixtures dir; CPU-
  deterministic (fixed seeds, float64 where the finite-difference check needs
  it). Recurring first-class gates: **gradient correctness** and **golden
  values** (pinned against the raw L23 netCDF and the ZTT paper).
- **Differentiability** — everything on the `forward` path is pure JAX, `jit`/
  `vmap`-friendly, and gradient-checked.
- **Lint/format** — `ruff`; Google-style docstrings; light `jaxtyping`
  annotations on public signatures.
- **Units** — m⁻¹ (`a`, `b`, `bb`), sr⁻¹ (`Rrs`, `rrs`), nm (λ), degrees
  (geometry angles), m s⁻¹ (wind); `B_p = bb_p / b_p` dimensionless.

---

## 9. Module index (current)

```
robust/
  __init__.py          ✅ package root (pre-existing)
  rt/                  ⬜ M0 task 3
  tests/               ⬜ M0 task 3
```

*Living document; updated at the close of each milestone.*
