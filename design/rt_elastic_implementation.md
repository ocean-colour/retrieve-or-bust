# Elastic RT Implementation Record

**Version:** 0.6
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
| **M1** | Data & conventions | 🟡 in progress | `robust.rt.{conventions,types}`, `robust.rt.data.l23` |
| **M2** | ZTT analytic backbone (JAX) | ⬜ not started | `robust.rt.ztt` |
| **M3** | Residual emulator + hybrid | ⬜ not started | `robust.rt.{emulator,hybrid}` |
| **M4** | Validation (*prototype done*) | ⬜ not started | `robust.rt.validation`, `design/py/run_validation.py` |
| **M5** | Beyond week 1 | ⬜ future | — |

Legend: ✅ done · 🟡 in progress · ⬜ not started.

**Branch.** All milestone work lands on `rt-elastic-prototype`; each milestone is
a reviewable commit for JXP (Claude runs no state-changing git — see
`CLAUDE.md`).

**CI.** `.github/workflows/ci.yml` runs `pytest` on Python 3.12 and 3.14 plus
`ruff check` and `ruff format --check`, on every branch and pull request — §10 for
the design and the one packaging bug it surfaced.

**Acceptance philosophy.** Accuracy gates are *relative* ("beats standard
Gordon on the held-out splits"), never blind absolute targets; absolute rRMS and
latency are **reported** here, not thresholded. The gradient-correctness check
(`jax.grad` vs central finite differences) is a hard gate from M2 onward.

**Verification (current).** `pytest -q` → **39 passed** in ~2.3 s (`ocean14`);
with `$OS_COLOR` unset, 36 passed + 3 skipped. `ruff check robust/` and `ruff
format --check robust/` → clean. The suite is green both with and without the L23
reference data on disk (missing data skips, never fails). The M0 notebook
(`notebooks/RT/rt_elastic_coding_1.ipynb`) executes end to end with no errors.

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

### 2.6 Notebook

`notebooks/RT/rt_elastic_coding_1.ipynb` — the M0 explainer (27 cells, executed,
ships with outputs and two figures). Structure: where M0 sits in M0–M5 → the
environment → **why JAX** (a live `jax.grad` on the standard Gordon relation,
which is *not* the model, since M0 implements no physics) → **why float64**
(figure 1) → the scaffold (module tree, pinned signatures, the stubs raising) →
the gate (`pytest` run inline) → a preview of the L23 reference data (figure 2) →
what M1 does next.

Two things in it are worth more than their word count:

- **Figure 1 measures the finite-difference/dtype problem instead of asserting
  it.** Relative error of a central FD derivative against `jax.grad`, swept over
  33 step sizes in float32 and float64. At a 1e-6 tolerance, **float32 meets it
  at 0 of 33 step sizes; float64 at 21 of 33** (best error 1.2e-5 vs 8.5e-12).
  So the M2 gate in float32 would have to be loosened to ~1e-4 *or* tuned to one
  lucky `h` — it would test the step size, not the gradient. That is the
  quantitative case for the `jax_x64` fixture.
- **A trap the notebook documents because M2 will meet it.** The first draft of
  that cell passed *Python floats* to the model function. Python floats are
  float64 and the body is plain arithmetic, so both curves were silently computed
  in float64 and the "float32" one came out ~1e-7 — a wrong answer that looks
  like a good one. The dtype has to be pinned on the arrays. When M2 writes the
  real FD gate, a float64 perturbation around a float32 model will silently test
  something other than what was meant.

**Note on running it.** `robust` is *not* pip-installed in `ocean14`, so the
notebook puts the repo root on `sys.path` by walking up to the directory
containing `robust/__init__.py`; that also means `pytest` must be run **from the
repo root**. This note originally guessed the cause was Q1 (the JAX stack living
only in `requirements.txt`) — **that was wrong**. The real cause, found while
setting up CI (§10), was a bug in `setup.py`: `provides = ['retrieve-or-bust']`
is illegal distutils metadata because of the hyphen, so `pip install -e .` failed
outright. Fixed; the package is installable again, and the `sys.path` bootstrap
now merely makes the notebook work whether or not anyone has installed it.

Figure conventions, so later milestones' figures match: recessive grid and
frame, text in ink colours rather than series colours, legend plus direct labels
so identity is never carried by colour alone, a single hue light→dark for
sequential magnitude (`a`(440) in figure 2), and the two-series categorical pair
`#0072B2` / `#D55E00` — checked, not eyeballed, at OKLab ΔE 31 for normal vision
and 29–45 under simulated deuteranopia / protanopia / tritanopia (target ≥ 8).

---

## 3. M1 — Data & conventions

**Goal.** The shared conventions and the data layer every later milestone
consumes: `Rrs↔rrs`, the IOP/phase/geometry pytrees, and a one-call L23 loader
with `B_p` and the seeded splits.

### 3.1 Task status

| # | Task | Status |
|---|------|--------|
| 1 | `conventions.py` — A/B, `Rrs↔rrs`, wavelength grid, `bb_w(λ)`, validators | ✅ done |
| 2 | `types.py` — `IOPs` / `PhaseParams` / `Geometry` pytrees | ⬜ pending |
| 3 | `data/l23.py` — L23 batches, `B_p`, seeded splits | ⬜ pending |
| 4 | `notebooks/RT/rt_elastic_coding_2.ipynb` — the M1 explainer | ⬜ pending |

### 3.2 Modules added

**`robust/rt/conventions.py`** — the first implemented module in `robust/rt/`.

- `A_RRS = 0.52`, `B_RRS = 1.7` (Lee et al. 2002), plus `Rrs_to_rrs` /
  `rrs_to_Rrs` — pure, `jit`-able, differentiable, exact inverses of each other.
- `RRS_POLE = 1/B_RRS ≈ 0.588`, the `rrs` value where `rrs_to_Rrs` diverges and
  past which it returns *negative* Rrs. Named because it is the signature of a
  unit error, and `check_rrs` looks for it.
- The canonical grid: `WAVE_MIN/MAX/STEP`, `N_WAVE = 81`, the NumPy constant
  `WAVE`, and `canonical_wave(dtype=None)` returning a JAX array. `WAVE` is
  deliberately **NumPy, not `jnp`** — a device array built at import would fix
  its dtype before a caller can enable float64. The values are exact multiples
  of 5, so float32 holds them without error either way.
- `BB_W_L23` (81 values) and `bb_w(wave)`, which interpolates it.
- Boundary validators `check_wave`, `check_iop`, `check_rrs`.

**Key decision — `bb_w` is L23's own water, embedded as a table.**
`bb_w = bb − bbnw` taken from the L23 elastic file. For a model trained against
L23 this is not an approximation of the water model, it *is* the water model, and
any other choice would push a bias straight into `bb_p = bb − bb_w`. Two
supporting findings:

- **It is scene-independent**, verified before relying on it: the difference is
  constant to 1.6e-7 relative (float32 storage noise) across all 3320 scenes,
  all three solar zeniths, and both X=1 and X=4 (X=1 vs X=4 are bit-identical).
  So there is no scene to choose. `bing`'s `bbNWModel.init_bbw` and
  `ocpy.water.scattering.bbw_from_l23` compute the same quantity but each picks
  an arbitrary index (0 and 170 — the latter commented "Random choie") *without*
  checking that it does not matter. It does not, but now that is a test.
- **The physical alternative is a dead end today.** Both of those functions carry
  a TODO pointing at Zhang, Hu & He (2009) for a proper T/S-dependent
  calculation. `ocpy.water.scattering.betasw_ZHH2009` exists but raises
  `ValueError("THIS IS NOT SUCCESFULLY CONVERTED YET")` on the first line — an
  unfinished MATLAB port. Recorded so nobody re-discovers it; it matters at M5,
  when new HydroLight runs may not share L23's water column.

Embedding the 81 values (rather than reading the netCDF) keeps `conventions.py`
importable with no data — which is what lets CI run it — and a `needs_l23` test
re-derives the table from the file so it cannot drift.

**Validators raise `ValueError`, not `assert`.** `python -O` strips `assert`, and
a convention check that silently disappears is worse than none. They are
documented as *boundary* checks: they read concrete values, so they cannot run
inside `jit`, and are meant for where data enters the package — the loader, a
constructor — leaving `forward` clean.

**Note on import cost.** `conventions.py` imports `jax.numpy` at module scope, so
`from robust import rt` now pulls JAX (M0 recorded that it pulled nothing). This
is the expected transition, one milestone earlier than predicted.

### 3.3 Tests

**`robust/tests/test_conventions.py`** — 27 tests.

- *Round trip*: `Rrs→rrs→Rrs` at **float32 to 1e-6** and at **float64 to
  1e-12**, plus the reverse direction. Measured errors are 2.0e-7 and 2.6e-16,
  so the float32 gate has ~5× of headroom — asserted in both regimes precisely
  so nobody later tightens the float32 one into a test of the dtype.
- *Agreement with BING*: `A_RRS`/`B_RRS` equal `bing.rt.A_Rrs`/`B_Rrs`, under
  `importorskip` since CI installs a lean set. Fixing the constants only buys
  anything if the package sharing `rrs` agrees, so it is asserted, not commented.
- *Differentiability*: both conversions and `bb_w` survive `jit` and `grad`, with
  the sign of each derivative checked (`bb_w` falls toward the red).
- *The pole*: `rrs_to_Rrs` diverges just below `RRS_POLE` and goes negative past
  it.
- *Grid*: shape/ends/spacing, `canonical_wave()` follows `jax_enable_x64`, and a
  `needs_l23` golden check that `WAVE` **is** L23's `Lambda`, not a lookalike.
- *`bb_w`*: positive, monotone, fitted slope λ^-4.2 within a sanity band of
  Morel's -4.32; exact on the grid; interpolates at midpoints and *clamps*
  outside 350–750 nm rather than extrapolating; two `needs_l23` golden checks
  (matches `bb − bbnw`; is scene-independent).
- *Validators*: each rejects its failure mode (wrong length, shifted grid, wrong
  spacing; negative/NaN/inf IOPs; negative, non-finite, and beyond-the-pole
  `rrs`) and accepts valid input. The shifted-grid test also asserts the error
  message reports the offset, since a validator that says only "bad grid" costs
  more time than it saves.

### 3.4 Results

`pytest -q` → **39 passed** (12 M0 + 27 M1). With `$OS_COLOR` unset: **36 passed,
3 skipped** — exactly the three `needs_l23` golden tests, so CI stays green
without the reference data. `ruff check robust/` and `ruff format --check
robust/` both clean.

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
- **Lint/format** — `ruff`, configured in `ruff.toml` at the repo root:
  `select = E/F/I/W/UP/B`, line length 88, `target-version = py312`, and
  `quote-style = "double"` for the formatter. Matching the sibling PAB project,
  so the two repos lint the same way (JXP's answer to Q2). Both `ruff check` and
  `ruff format --check` are CI-gated, scoped to `robust/` — the pre-existing
  scripts under `reports/py/` and `context/RT/` predate the config and report 48
  findings, mostly `E702` (`import glob, os`) and `E501`; cleaning them is a
  separate decision. Two rules are ignored **because of `jaxtyping`**: `F722`
  (pyflakes parses a shape string like `" 81"` as a forward reference and reports
  a syntax error) and `UP037` (pyupgrade offers to strip quotes that are the
  shape specification). Docstrings are NumPy-style with `Parameters`/`Returns`,
  plus light `jaxtyping` annotations on public signatures.
- **Units** — m⁻¹ (`a`, `b`, `bb`), sr⁻¹ (`Rrs`, `rrs`), nm (λ), degrees
  (geometry angles), m s⁻¹ (wind); `B_p = bb_p / b_p` dimensionless.
- **Notebooks** — one explainer per milestone in `notebooks/RT/`
  (`rt_elastic_coding_<M>.ipynb`), committed **with outputs** (executed via
  `jupyter nbconvert --execute`) so it reads without a kernel, and written so any
  data-dependent section degrades to a message when `$OS_COLOR` is absent. Same
  split as the prose docs: the notebook *explains and demonstrates*, the
  implementation record *states current fact*, the prompt Logs carry the dated
  narrative.

---

## 9. Module index (current)

```
robust/
  __init__.py          ✅ package root (pre-existing)
  rt/
    __init__.py        ✅ submodule imports + forward() re-export
    conventions.py     ✅ M1  A/B, Rrs<->rrs, grid, bb_w, validators
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
    test_conventions.py ✅ 27 tests — M1 task 1

notebooks/RT/
  rt_elastic_coding_1.ipynb  ✅ M0 explainer (executed, 2 figures)

.github/workflows/
  ci.yml                     ✅ pytest (py3.12, py3.14) + ruff check/format
ruff.toml                    ✅ lint + format configuration
```

---

## 10. Continuous integration

`.github/workflows/ci.yml` — runs on **every branch** and on pull requests (all
milestone work happens on feature branches, so a main-only trigger would give no
signal while a milestone is being built), with a concurrency group so superseded
runs on the same ref cancel.

**Job `test`** — matrix `python-version: [3.12, 3.14]`, `fail-fast: false`. 3.12
is the floor declared in `setup.py` (and `jax` 0.11 requires ≥ 3.12); 3.14 is what
`ocean14` runs. Steps: install a **lean** dependency set
(`jax flax optax jaxtyping numpy scipy xarray pytest`), then `ocpy` with
`--no-deps`, then `pip install -e . --no-deps`, then `pytest -q -ra`.

Two deliberate departures from `pip install -r requirements.txt`:

- **The lean set.** `requirements.txt` is the developer's full environment. Via
  `ocpy` it pulls `cartopy`, `geopandas`, `healpy`, `netcdf4`, plus
  `bing`/`emcee`/`bokeh`/`seaborn` — none of which the test suite imports.
  Installing only what the tests use keeps CI fast and makes a red build mean
  "our code broke" rather than "a geospatial wheel failed to build".
- **`ocpy --no-deps`.** `robust` touches exactly one ocpy module,
  `ocpy.hydrolight.loisel23`, which needs only `numpy` + `xarray`. So the real
  integration is exercised without the heavy extras. If a later test reaches a
  part of ocpy that needs more, that is the moment to add the dependency.

`bing` is **not** installed in CI: nothing in the suite imports it yet. M1 may add
a test cross-checking `A_RRS`/`B_RRS` against `bing.rt` — when it does, either add
`bing` here or guard the test with `pytest.importorskip`.

**Job `lint`** — `ruff check robust/` **and** `ruff format --check robust/`, with
`ruff==0.16.0` pinned. Q2 is now resolved (JXP: yes to both), so `ruff.toml` pins
the rule selection and "clean" is a property of the repo rather than of whichever
ruff got installed; the version pin stays for determinism, since a new ruff can
still add rules inside the selected categories. Scoped to `robust/` because the
pre-existing scripts under `reports/py/` and `context/RT/` predate the config.

**Why the suite is green in CI without the reference data.** `$OS_COLOR` is unset
on the runner, so `ocpy` warns and falls back to `./`, `l23_available()` returns
`False`, and the `needs_l23` tests skip. Verified locally with `OS_COLOR` unset:
**12 passed, 1 warning**. `-ra` prints the skip reasons, so a skip that should
have been a pass shows up in the log instead of passing silently.

**One bug fixed to get here.** `setup.py` set `provides = ['retrieve-or-bust']`;
`provides` is legacy distutils metadata that must be a *module* name, so the
hyphen made `pip install -e .` fail with `ValueError: illegal provides
specification`. That is why nothing was ever pip-installed in `ocean14` and why
`pytest` only worked from the repo root. The key is removed (it is superseded by
`Provides-Dist` and does nothing useful); `pip install -e . --no-deps` now
succeeds. `bing`/`ocpy` carry the same line harmlessly because their names have no
hyphen.

**Badge** in `README.md`. It will read "no status" until `ci.yml` reaches the
default branch — the workflow file has to exist on `main` for the badge's default
view to resolve.

*Living document; updated at the close of each milestone.*
