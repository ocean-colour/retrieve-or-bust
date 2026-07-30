# Elastic RT Implementation Record

**Version:** 0.3
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
| **M0** | Environment & scaffold | ✅ done | `robust.rt` (stubs), `robust/tests/` |
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

**Verification (current).** `pytest -q` → **12 passed** in ~1.2 s (`ocean14`);
`ruff check robust/` → clean. The suite is green both with and without the L23
reference data on disk (missing data skips, never fails).

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
| 3 | Scaffold `robust/rt/` + `robust/tests/`; pass the `pytest` gate | ✅ done |

### 2.2 Modules added

The full layout from the coding plan, as documented stubs:

```
robust/rt/
  __init__.py         eager-imports every submodule; re-exports forward()
  conventions.py      docstring-only (M1): A/B, wavelength grid, bb_w, asserts
  types.py            docstring-only (M1): IOPs/PhaseParams/Geometry pytrees
  data/__init__.py    re-exports l23
  data/l23.py         docstring-only (M1): L23 elastic batch loader + splits
  ztt.py              Rrs_ZTT(iops, phase_params, geometry, wave) -> raises (M2)
  emulator.py         docstring-only (M3): Flax MLP + Optax training
  hybrid.py           MODES; forward(iops, phase_params, geometry, wave, mode)
                      -> raises (M3)
  validation.py       docstring-only (M4): rRMS / speed / gradient protocol
robust/tests/
  __init__.py, conftest.py, files/.gitkeep, test_env.py
```

**What "stub" means here.** Two of the modules carry their **final public
signatures** already — `hybrid.forward` (design §3) and `ztt.Rrs_ZTT` (coding
plan M2) — and raise `NotImplementedError` naming the milestone that fills them
in. The rest are docstring-only, each stating its role and its milestone. So the
scaffold pins the interface without faking any physics: a caller gets a loud
error, never a plausible-looking array. `MODES = ('ztt', 'emulator', 'hybrid')`
is declared in `hybrid.py` because the three-way comparison is a design
commitment (§4.5), not an implementation detail.

**Import cost.** `from robust import rt` currently pulls **no** heavy
dependency — not even `jax` — because the stubs have no module-level imports;
it measures ~0.00 s. Per the sibling-project convention, `emulator.py` will
import `flax`/`optax` *inside* its functions so the analytic-only path never pays
for the ML stack. This changes once M2 puts `import jax.numpy` at `ztt.py`'s
module scope, which is fine and expected — noted so the change is not mistaken
for a regression.

**Not added: a `pytest.ini`.** Bare `pytest -q` from the repo root already
collects exactly `robust/tests/` (there is no other `test_*.py` in the tree), so
config would be ceremony. Worth revisiting only if collection ever picks up
something it should not.

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
`flax`, `optax`, `jaxtyping`) and is the **sole** declaration — `setup.py`'s
`install_requires` deliberately does *not* list them (JXP's call, Q1 in the
prompt's Q&A), so `pip install -e .` for non-RT work never has to pull `jaxlib`.
A comment in `setup.py` records that, so the asymmetry does not get "helpfully"
undone. Install the RT stack with `pip install -r requirements.txt`.

`jaxlib` is deliberately *not* listed separately — `jax` requires the matching
version, so a second entry would only invite version skew. Unpinned, like every
other entry in the file; exact versions live in the table above. A GPU build
would be `jax[cuda12]`; on macOS arm64 the plain wheel is CPU-only (Metal would
be a separate `jax-metal` plugin), so "CPU-only" needs no extra machinery here.

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

`robust/tests/` follows the BING layout (`test_*.py`, `conftest.py`, `files/`).

**`conftest.py`** holds the two things every later module needs:

- `l23_available()` + the `needs_l23` skip marker. The L23 files (~17 MB each)
  live outside the repo, resolved from `$OS_COLOR`, so from M1 on the data
  dependency is declared once and **absence becomes a skip, not a failure** —
  `pytest -q` stays meaningful on a machine without the dataset. Ported from
  BING's `conftest.py`, which solves the same problem for the same files;
  narrowed here to the three *elastic* files (`Hydrolight100/130/160.nc`).
- The `jax_x64` fixture: enables float64 for one test and **restores the previous
  setting afterwards**, so a float64 test cannot silently change the dtype regime
  the rest of the suite runs under. (`jax.experimental.enable_x64`, the context
  manager older JAX docs suggest, was removed by JAX 0.11 — hence the explicit
  set/restore.)

**`test_env.py`** — 12 tests, deliberately more than the gate's letter, split
three ways:

- *JAX (6)*: `jax.numpy.ones(3)` returns the right values on `CpuDevice`;
  `default_backend() == "cpu"` **and every device is CPU** (CQ5 is CPU-only, so an
  accelerator sneaking in should fail here); float64 reachable via the fixture;
  the fixture restores float32 afterwards; `jax.grad` correct; `jax.jit`
  compiles and computes; `flax`/`optax` import.
- *`robust.rt` (4)*: `from robust import rt` works; every module of the planned
  layout is present and re-exported (incl. `rt.data.l23`) and `rt.forward` is
  callable; the two signature-carrying stubs raise `NotImplementedError`; `MODES`
  is declared.
- *Reference data (2)*: `l23_available()` returns a bool without raising
  whatever the environment (it gates every M1+ data test, so it must never be
  the thing that breaks); `ocpy.hydrolight.loisel23` imports.

The float64, `jit`, and grad tests exceed what M0 strictly needs. They are here
because M2's finite-difference gradient gate and the design's `jit`/`vmap`
requirement depend on them, and a broken XLA install should surface now rather
than three milestones later.

### 2.5 Results — M0 gate ✅

```
$ pytest -q          # repo root, ocean14
............                                                          [100%]
12 passed in 1.20s

$ ruff check robust/
All checks passed!
```

The gate's two required assertions both hold: `import jax` works on CPU, and
`from robust import rt` succeeds.

**Also checked, because a green suite on one machine proves less than it looks
like.** Re-ran with `OS_COLOR` pointed at a nonexistent path: `l23_available()`
returns `False` rather than raising, and all 12 tests still pass. That is the
property M1's data tests will lean on, verified rather than assumed.

**Lint.** `robust/` is clean under `ruff check` with two documented `noqa`s, both
deliberate: `RUF022` on `__all__` (grouped by role and ordered as the pipeline
runs, not alphabetically) and `BLE001` on the `except Exception` in
`l23_available` (any failure there genuinely means "no data" — same call BING's
conftest makes). Note there is **no ruff config in this repo**, so the rule set
above is whatever the installed ruff defaults to; see the open question in
`claude_prompts/RT/rt_elastic_coding_prompt_1.md` §Q&A (Q2) about pinning a
`ruff.toml` and whether to adopt `ruff format` (which would rewrite quote style
package-wide).

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
  rt/
    __init__.py        ✅ submodule imports + forward() re-export
    conventions.py     ⬜ M1  A/B, wavelength grid, bb_w, asserts
    types.py           ⬜ M1  IOPs / PhaseParams / Geometry pytrees
    data/
      __init__.py      ✅ re-exports l23
      l23.py           ⬜ M1  L23 elastic batches + seeded splits
    ztt.py             ⬜ M2  Rrs_ZTT — signature pinned, body pending
    emulator.py        ⬜ M3  Flax MLP ΔRrs + Optax training
    hybrid.py          ⬜ M3  forward() — signature + MODES pinned, body pending
    validation.py      ⬜ M4  rRMS / speed / gradient protocol
  tests/
    __init__.py        ✅
    conftest.py        ✅ needs_l23 marker, jax_x64 fixture
    files/             ✅ empty (.gitkeep); M1 caches an L23 batch here
    test_env.py        ✅ 12 tests — the M0 gate
```

*Living document; updated at the close of each milestone.*
