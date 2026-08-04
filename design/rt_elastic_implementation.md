# Elastic RT Implementation Record

**Version:** 0.14
**Date:** 2026-08-04
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
| **M1** | Data & conventions | ✅ done | `robust.rt.{conventions,types}`, `robust.rt.data.l23` |
| **M2** | ZTT analytic backbone (JAX) | 🟡 in progress | `robust.rt.ztt`, `robust.rt.baselines` |
| **M3** | Residual emulator + hybrid | 🟡 code, tests, notebook done (tasks 1–3 of 5) | `robust.rt.{emulator,hybrid}` |
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

**Verification (current).** `pytest -q` → **225 passed** (`ocean14`); with
`$OS_COLOR` unset, **205 passed + 20 skipped** — which is what CI sees. The loader is
exercised without the dataset against a committed 50-scene fixture.
`ruff check robust/` and `ruff format --check robust/` → clean. The suite is green both with and without the L23
reference data on disk (missing data skips, never fails). All four notebooks in
`notebooks/RT/` execute end to end with no errors.

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
| 2 | `types.py` — `IOPs` / `PhaseParams` / `Geometry` pytrees | ✅ done |
| 3 | `data/l23.py` — L23 batches, `B_p`, seeded splits | ✅ done |
| 4 | `notebooks/RT/rt_elastic_coding_2.ipynb` — the M1 explainer | ✅ done |

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

**`robust/rt/types.py`** — the three arguments of `forward`, as registered JAX
pytrees with light `jaxtyping` annotations (`Spectrum = Float[Array, "*batch
wave"]`, `Scalar = Float[Array, "*batch"]`).

- `IOPs(a, bb_w, bb_p)` with derived `bb`, `u = bb/(a+bb)`, and `n_wave`
  properties (properties, so they are *not* pytree leaves); a `from_total_bb`
  constructor; and `validate()`.
- `PhaseParams(B_p)` with `validate()`.
- `Geometry(theta_s, theta_v, dphi, wind=None)`, **degrees**, with a `nadir()`
  constructor for the L23 case and `validate()`.
- All three re-exported from `robust.rt`, per the coding plan's "`__init__.py`
  exports `forward()`, public types".

**Key decision — `jax.tree_util.register_dataclass`, not
`flax.struct.dataclass`.** Both work; the argument is dependency direction. These
types sit on the analytic path — the M2 ZTT backbone needs them and needs nothing
from Flax — so having the core data model import a neural-network library to
describe a container would be backwards. JAX's own mechanism is more primitive and
stable, and stdlib `dataclasses` gives `dataclasses.replace` and a sane `repr` for
free. Flax arrives at M3 inside `emulator.py`, where it earns its place. Import
cost was explicitly *not* the argument: measured, `flax` adds only ~0.08 s once
`jax` is loaded, so the convention recorded at M0 ("keep flax out of the analytic
path") rests on structure, not on a speed claim that would not have survived
scrutiny.

Verified behaviour of the choice: field inference works with no explicit
`data_fields`; `grad` of a scalar of an `IOPs` returns an **`IOPs`** with
per-field derivatives (the shape the future inversion wants, and the reason these
are containers at all); `vmap`, `tree_map`, `jit`, and `dataclasses.replace` all
traverse them.

**Key decision — `bb_w` is broadcast to the batch shape** in `from_total_bb`.
Storing it as a bare `(81,)` spectrum would be more honest about the physics
(water is the same everywhere) but would force every caller to spell out
`in_axes=IOPs(a=0, bb_w=None, bb_p=0)` to use `vmap`. Broadcasting costs ~1 MB for
a full L23 batch and makes plain `vmap(f, in_axes=0)` work; a test asserts that
payoff.

**Key decision — validation is explicit, never `__post_init__`.** Under `jit` or
`vmap` the fields are tracers with no concrete value, so a constructor-time check
would either crash or pass vacuously. Each type therefore has `validate()`, called
where data enters. A test asserts `validate()` raises
`jax.errors.TracerArrayConversionError` under `jit` — pinning the contract, so a
future "improvement" that moves it into `__post_init__` fails loudly.

**`PhaseParams` is the API's extension point.** Week 1 carries only `B_p`; at M5
the ZTT backward-VSF descriptors join as *additional optional fields defaulting to
`None`*, which changes neither `forward`'s signature nor any call site. Two
consequences, both tested: an unset optional field contributes no leaves, but the
*treedef* does change once it is set, so `jit` recompiles once per variant —
correct and cheap, but visible in a profile.

`PhaseParams.validate()` checks only the definitional bound `B_p ∈ (0, 1]`, not
the ~[0.004, 0.03] expected of real particles. That tighter range is the loader's
business (task 3): a synthetic `B_p` sweep at M2/M3 must be able to probe outside
it without fighting a type-level invariant.

**`robust/rt/data/l23.py`** — the one-call L23 loader and the seeded splits.

- `load_batch(zeniths=ZENITHS, *, x=ELASTIC_X, scenes=None, validate=True)` →
  `L23Batch`, holding `IOPs`, `PhaseParams`, `Geometry`, the reference `Rrs`, the
  wavelength grid, and per-sample `scene` labels.
- `make_splits(batch, *, seed=SPLIT_SEED, ...)` → `Splits`: four boolean masks
  (`scene_train/test`, `zenith_train/test`) plus the held-out scene indices.
- `select(batch, mask)` → a subset batch, for M3/M4 to train and score on.
- Constants: `ELASTIC_X = 1`, `ZENITHS = (0, 30, 60)`, `N_SCENES = 3320`,
  `B_P_EXPECTED = (0.004, 0.03)`, `SPLIT_SEED = 23`, `TEST_FRACTION = 0.2`,
  `HELD_OUT_ZENITH = 60`.

**Key decision — one flat sample axis.** The batch stacks `n_zenith × n_scene =
9960` samples along a single leading axis, so every leaf shares the batch shape
and `jax.vmap(f, in_axes=0)` works with no per-field `in_axes`. `scene` and
`geometry.theta_s` label each sample, which is what makes both the per-zenith
metrics and the splits expressible as boolean masks.

**Key decision — the split is by *scene*, not by sample.** Each water body appears
three times (once per zenith). Splitting per sample would put the same IOPs in
both train and test at different sun angles, and every held-out number after that
would be optimistic. `make_splits` draws scenes and then expands to a sample mask;
a test asserts the train and test scene sets are disjoint. This is the property the
whole M4 acceptance gate rests on, so it is tested on a **synthetic** batch too and
therefore runs in CI, without the dataset.

**Key decision — `L23Batch` is deliberately *not* a registered pytree.** It is an
analysis container, and `scene` is host-side integer metadata that has no business
being traced or differentiated. The model inputs it holds are pytrees; the box
around them is not. A test pins that.

**Key decision — `bb_w`/`bb_p` come from the file, not from `conventions`.** The
loader uses `bb_w = bb − bbnw` and `bb_p = bbnw`, so a batch is exactly what L23
says with no convention drift. A test asserts the loader's `bb_w` agrees with the
embedded `BB_W_L23` table, closing the loop between the two modules — at `rtol =
1e-5`, and the reason is instructive: L23 stores float32, and the two paths differ
only in *where* the subtraction happens (the table was extracted with float32
arithmetic; the loader upcasts to float64 first). In the red tail, where `bb_w ≈
3e-4`, that shows as up to **3.4e-6 relative** disagreement on ~0.1% of elements.
Both are inside the reference data's own precision, so the tolerance follows
float32 rather than pretending to a precision the netCDF does not carry.

**`B_p` is reported, never clipped.** `load_batch` *warns* if `B_p` leaves
`B_P_EXPECTED`; it does not raise and does not clamp, because silently squashing it
would hide a change in the reference data.

### 3.2.1 Measured properties of the L23 elastic release

Answering the question prompt 2 raised — is the `B_p` range assert safe outside
440 nm? — and a few things found alongside. Each has a test.

| Property | Measurement |
|---|---|
| `B_p` range, all bands & zeniths | **[0.01026, 0.01800]** over 268,920 values |
| Inside design's ~[0.004, 0.03]? | **100%** — no UV or far-red failure |
| `bnw` minimum | 6.1e-3, so `B_p = bbnw/bnw` never divides by ~0 |
| `B_p` vs λ | **varies** within a scene (0.0134 at 350 nm → 0.0125 at 750 nm), so it is carried as a spectrum |
| IOPs across the three zenith files | **bit-identical** — the same 3320 water bodies, illuminated three ways |
| `Rrs(30°)/Rrs(0°)` | median **0.990** |
| `Rrs(60°)/Rrs(0°)` | median **0.949** |

Two consequences worth carrying forward.

**The range assert is safe at every wavelength**, so it needs no
wavelength-dependent escape hatch — the concern prompt 2 flagged does not
materialise. Tested per band, not just globally, since a narrow band-specific
excursion would otherwise hide inside the global min/max.

**L23 covers a narrow slice of phase-function space.** `B_p` spans a factor ~1.75
(0.0103–0.0180) where the design's nominal band spans ~7. So "explicit
phase-function dependence" is only weakly exercised before M5's HydroLight runs
vary it properly — an honest limit on what the week-1 prototype can demonstrate,
recorded now rather than discovered at M5. The `Rrs`-vs-zenith numbers are the
flip side: a ~5% effect at 60° *is* real signal, and standard Gordon has no
solar-zenith dependence at all, which is precisely what the M3/M4 comparison
exploits.

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

**`robust/tests/test_types.py`** — 32 tests. Because the point of these types is
that they are pytrees, most of the suite is JAX behaviour rather than attribute
access.

- *Pytree mechanics*: leaf counts (3 for `IOPs` — the properties are not leaves),
  frozen-ness, `dataclasses.replace`, `tree_map`, `jit`, `vmap`, and all three
  containers through one traced call in `forward`'s argument order.
- *`grad` returns the container*: `jax.grad` of a scalar of an `IOPs` is an
  `IOPs`, with `∂/∂a < 0` and `∂/∂bb_p > 0` — sensitivities stay labelled instead
  of arriving as an anonymous flat vector.
- *`from_total_bb`*: splits water off correctly, broadcasts so `vmap(in_axes=0)`
  needs no custom `in_axes`, and — the documented failure mode — produces a
  negative `bb_p` when handed *non-water* `bb`, which `validate()` then catches.
- *Extensibility*: a local M5-shaped `PhaseParams` variant with an extra optional
  field is pushed through `jit` and `grad`, so the design's extension promise is
  exercised rather than merely written down.
- *Validators*: shape mismatch, non-finite, wrong grid length; `B_p` outside
  `(0, 1]`; out-of-range angles per field; negative wind. Plus one test that pins
  a **blind spot** rather than a guarantee: 30° expressed in radians is 0.52,
  which sits inside `[0, 90]`, so the range check *cannot* detect that mix-up. It
  is recorded so nobody relies on protection that is not there.

**`robust/tests/test_l23.py`** — 36 tests, split deliberately in two.

*Logic tests run everywhere* (20), on a synthetic batch built in the test module
with no netCDF access: container shapes and `validate` failure modes; the
scene-split fraction, complementarity, determinism-per-seed, and **no-leakage**
property; the zenith hold-out; `select` correctness and its mask errors; the `B_p`
warning firing without clipping. Putting the split logic on a synthetic batch is
the point — the splits *are* the M4 gate, so they must be covered on machines
(CI included) that have no dataset.

*Data tests carry `needs_l23`* (16) and pin the measurements above: shapes and
grid; `a, bb ≥ 0` and `Rrs > 0`; `B_p` inside the design band **per wavelength**;
the observed `B_p` extremes; `B_p` varying with λ; the loader's `bb_w` against the
`conventions` table; IOPs identical across zeniths; `Rrs` falling with zenith at
the measured ratios; `scenes=` subsetting; and the splits sized on the real 3320.

Two independent golden checks, because they fail for different reasons. One
**re-reads the netCDF** and compares row by row, so a transposed axis, an
off-by-one in the scene labels, or a mis-ordered zenith concatenation shows up.
The other **pins the absolute numbers** (`Rrs(440, scene 0, 0°) = 8.5333e-03`,
etc.), which the first would not catch if the underlying release changed.

### 3.4 Notebook

`notebooks/RT/rt_elastic_coding_2.ipynb` — 23 cells, executed, three figures.
Organised around the four M1 *decisions* rather than the call signatures, and
deliberately not repeating M0's JAX material: why the fitting space is `rrs`; why
`bb_w` stays apart from `bb_p`; what `B_p` looks like in the reference data; and
what the two held-out splits protect against.

Two things the notebook produced that the code alone did not say:

- **The water/particle split matters more than expected.** For the median L23
  scene, pure water contributes **~72% of total backscatter at 400 nm**, ~50% at
  550 nm, and still **~29% at 750 nm** (across scenes, 37–87% at 400 nm). So
  `bb_w` is not a small correction to be absorbed into `bb` — over much of the
  spectrum it is the *larger* term, which is the quantitative case for keeping the
  split explicit.
- **The `rrs` fitting space is not cosmetic.** The true conversion sits **6% below
  a linear rescaling at `Rrs` = 0.02 and 14% below at 0.05**, so a relative error
  in `Rrs` space is not the same relative error in `rrs` space — which is why the
  design's protocol specifies rRMS in `rrs`.

Figures: (1) the `Rrs↔rrs` departure from linearity, and the pole on a log axis
showing it sits more than a decade beyond the ocean range; (2) `B_p` percentiles
per wavelength against the design band — the visual answer to "does the range hold
outside 440 nm?" *and* to "how much of the band does L23 actually cover?";
(3) the two splits, with the random hold-out drawn as interleaved spectra and the
zenith hold-out drawn as `Rrs(θ)/Rrs(0°)`, where the 1.0 line is exactly what
standard Gordon assumes.

### 3.5 Results

`pytest -q` → **117 passed** (12 M0 + 27 conventions + 32 types + 46 L23). With
`$OS_COLOR` unset: **100 passed, 17 skipped** — every skip is a full-release
`needs_l23` test; the loader, splits, and `select` all still run in CI against the
committed fixture. `ruff check robust/` and `ruff format --check robust/` clean.

## 4. M2 — ZTT analytic backbone (JAX)

**Goal.** A differentiable `Rrs_ZTT` with the backward VSF explicit, plus the
analytical benchmark it has to beat.

### 4.1 Task status

| # | Task | Status |
|---|------|--------|
| 1 | `baselines.py` — standard Gordon in JAX + shared `rrms`; reproduce the published ladder | ✅ done |
| 2 | `ztt.py` — transcribe the ZTT forward relation | ✅ done (µ∞ via TT2017; see below) |
| 3 | `test_ztt.py` — paper reference case, gradient gate, rRMS report | ✅ done |
| 4 | `notebooks/RT/rt_elastic_coding_3.ipynb` — the M2 explainer | ✅ done |

### 4.2 Modules added

**`robust/rt/baselines.py`** — the comparison models, standard Gordon first.
`G1_GORDON = 0.0949`, `G2_GORDON = 0.0794` (fixed, not fitted), `rrs_gordon` and
`Rrs_gordon`.

**Key decision — Gordon is not a fallback, it is a deliverable.** The coding plan
lists a Gordon-in-JAX backbone as the *de-risking* option if ZTT proves ambiguous.
But M3's gate is "hybrid beats standard Gordon" and M4 scores against
Gordon/PR05/O25, so it is required either way; building it first costs nothing and
makes the de-risk branch free. Placed in its own module (rather than inside `ztt.py`
or `validation.py`) so PR05 and O25 can join it at M4 — the coding plan's layout has
no home for comparison models, so this is an addition to it.

**Gordon takes `forward`'s signature and ignores two arguments.** `phase_params`
and `geometry` are accepted and discarded. That is not a shortcut but the model's
defining limitation, and keeping the signatures interchangeable lets M4 score every
model in one loop. Two tests assert the blindness rather than leaving it as a
docstring claim, including one showing that Gordon returns *bit-identical*
predictions at 0°/30°/60° while the reference `Rrs` does not.

**`robust/rt/validation.rrms` landed early.** The whole point of the metric is that
one definition is shared — the number in the M2 log, the M4 table, and the synthesis
figures must be the same quantity or the comparisons are meaningless. So
`rrms(truth, pred, axis=None)` = `100·sqrt(mean(((pred−truth)/truth)²))` lives in
`validation.py` from M2 rather than being written twice. It is pure JAX and
differentiable, so it can double as M3's training loss.

**`robust/rt/ztt.py`** — the Twardowski & Tonizzo (2018) model transcribed into
JAX, every function naming the equation it implements.

| Term | Equation | Status |
|---|---|---|
| scattering angle ψ, Snell refraction | §2 after (2), §2.1 | ✅ |
| `Ψ_KLu(ψ) = 1 + F(ψ)` | (4), (6) + Table A2 `fA` | ✅ |
| `β(ψ)/bb` | (10) | ✅ |
| `βw(ψ)/bbw` | §2.3, deferred to Zhang 2009 | ✅ derived analytically |
| `b̃b` | (11) | ✅ |
| `µd = M⁺_d × M*_d` | (13)–(17) + Table A2 `e`, `m*_d` | ✅ (µw is the **in-water** cosine — see §4.4) |
| `f_L(ψ, λ)` | (31) + Table A3 (91 values) | ✅ |
| `Pbb,ST(ψ)` | §4.2, from S&T (2009) | ✅ (their Table 2, `a3` typo corrected) |
| **`µ∞`** | **(8)** | 🟡 structure only; TT2017 stands in |
| assembly | (12) | ✅ |
| Raman | (18) | n/a — elastic scope |

**Equation (8)'s coefficients are missing from the paper — resolved via the
authors' own antecedent.** Equation (8) says "Coefficients m are provided in
Appendix A, Table A2"; Table A2 as printed covers Equations (3), (4), (16), (17)
and runs from `m*_d,8` straight into Table A3. A full-text search of all 30 pages
finds `m1`/`m16` only inside Equation (8) itself, and the MATLAB code cited at
ioccg.org is not there.

The stand-in is **Twardowski & Tonizzo (2017)**, *Optics Express* **25**(15),
18122 — reference [40], the study the 2018 text says Equation (8) "extended ... to
include near zero bb/a and increased resolution in η_bb". Its **Table 1** gives
`µ∞ = p0 + p1 log10(bb/a) + p2 log10²(bb/a)` at six discrete `η_bb`; the three
coefficients are interpolated in `log10 η_bb` to recover the 2-D surface, which
keeps it differentiable. Transcribed as `mu_infinity_tt2017` and used by `rrs_ZTT`
whenever Equation (8) coefficients are absent.

Table 2's alternative quartics were rejected: they reach `µ∞ = 1.35` at
`bb/a = 1e-4`, which is unphysical (µ∞ ≤ 1), and carry no `η_bb` dependence.
Table 1 stays in (0.63, 0.98] across L23. Note L23 reaches `bb/a ≈ 0.31` against
the fit's 1e-1 upper bound, so the brightest scenes extrapolate.

`mu_infinity` still implements Equation (8)'s exact structure and **requires** the
sixteen coefficients, so passing `mu_inf_coeffs=` restores the published 2018 model
the moment they arrive (JXP has emailed the authors). **Report present results as
"ZTT with the TT2017 µ∞", never as the 2018 model.**

**Two independent checks that the transcription is right.**

- **The paper's own worked example reproduces exactly.** §2.1 states "for θs' = 60°,
  θs will be 40.3° and ψ will be 139.7° for nadir viewing". The code gives
  **40.26°** and **139.74°**. That exercises Snell refraction, the scattering-angle
  formula, *and* the nadir convention (θv = 180° in the paper, 0° here) — the exact
  chain where a reversed convention would produce a plausible but wrong BRDF.
- **The water phase function matches its independent citation.** The paper defers
  `βw(ψ)` to Zhang et al. (2009), whose only available implementation
  (`ocpy.water.scattering.betasw_ZHH2009`) raises "THIS IS NOT SUCCESFULLY
  CONVERTED YET". Its shape is not in doubt, so it is derived here from
  `βw ∝ 1 + f cos²ψ`, `f = (1−δ)/(1+δ)`, normalized over the backward hemisphere:
  `βw(ψ)/bbw = (1 + f cos²ψ) / (2π(1 + f/3))`. At ψ = 180° that gives **0.2342
  sr⁻¹** against the **0.23 sr⁻¹** quoted in the synthesis (§3.5, citing Zhang 2009
  and this paper); the analytic normalization also matches numerical integration
  exactly. Only the shape is needed — Equation (10) multiplies by `bbw`, so the
  unknown `βw(90°)` cancels.

Other sanity values, all inside the paper's stated ranges: `f_L(180°, 440 nm)` =
1.057 (paper: natural range 1–1.12, Zaneveld's constant 1.05), `Ψ_KLu` = 1.024 at
ψ = 180° rising to 1.315 at 134°, `H(30°)` = 0.31 (Morel & Prieur assumed 0.4).

**`Pbb(ψ)` was an input, and is now supplied.** Sullivan & Twardowski (2009),
*Appl. Opt.* **48**(35), 6811, tabulate the measured average particulate backward
phase function (their Table 1, 90-170°) and fit it with a fourth-order polynomial
(Table 2). Both are transcribed as `P_bb_sullivan`, which `rrs_ZTT` now uses by
default — the paper's own best-performing choice (§4.2: "errors increase by only
~0.3%").

**Their published `a3` is a typo.** Table 2 prints `8.007E−02`; at ψ = 140° that
term alone would contribute ~1570 against a tabulated value of 0.137. Refitting
Table 1 independently here gives `a3 ≈ 7.79e-4` while reproducing the other four
published coefficients closely (5.65e-9 vs 5.885e-9; −3.41e-6 vs −3.526e-6;
−7.98e-2 vs −8.150e-2; 3.215 vs 3.266), so the intended value is **8.007E−04**.
With that correction the published fit matches Table 1 to 0.003 absolute,
consistent with its claimed "<0.5%". Table 1 stops at 170°, so nadir viewing
(ψ = 180°) extrapolates to 0.153 sr⁻¹ — an independent refit gives 0.156, so it is
well constrained.

**The remaining planning note.** The paper is explicit (§2.9) that four
parameters "must be provided": `bbp`, `apg`, `Pbb(ψ)`, `b̃bp`. Three already come
from M1's types (`IOPs`, and `B_p` *is* `b̃bp`); `Pbb(ψ)` is the fourth and is passed
by the caller. Its best-performing form is the constant `Pbb,ST(ψ)` of Sullivan &
Twardowski (2009) — tabulated in *their* paper, not this one. Worth noting for
planning: **adding `Pbb` to `PhaseParams` is exactly what the design's M5 "promote
phase_params to the ZTT backward-VSF parameterization" means**, so M5's scope is now
concrete.

### 4.3 Tests and gates (task 3)

`robust/tests/test_ztt.py` — 28 passing tests plus one strict `xfail`.

**(i) Paper reference cases.** The strongest is §2.1's worked example: "for
θs' = 60°, θs will be 40.3° and ψ will be 139.7° for nadir viewing" → the code
gives **40.26°** and **139.74°**. One assertion covers Snell refraction, the
scattering-angle formula, and the nadir convention together. Also pinned: the water
phase function against its independently quoted **0.23 sr⁻¹** (and its normalization
against numerical integration), `f_L` inside the paper's stated natural range
1–1.12, Table A3's 91 values, `Ψ_KLu` rising from 1.024 at ψ=180° to 1.32 at 134°,
`µd` inside the paper's quoted 0.79–0.94 band, and TT2017's µ∞ reproducing its own
six tabulated rows exactly.

**(ii) The gradient gate — passes.** `jax.grad` against central finite differences
for **`a`, `bb_p`, `B_p`, and `theta_s`**, under `jax_x64` with dtypes pinned on the
arrays, agreeing to `rel=1e-6`. Step sizes are scaled per variable (`a` is O(0.1),
`bb_p` O(1e-3)) since one global `h` cannot suit all four. `jax.grad` over the
container also returns a labelled `IOPs` with the right signs. This is the hard gate
from M2 onward and it is green.

**(iii) Accuracy is reported, not gated** — see §4.4.

### 4.4 Results — ZTT beats Gordon at every zenith, after one real bug

Standalone rRMS in `rrs` space (%), **ZTT with the TT2017 µ∞ and the Sullivan &
Twardowski (2009) `Pbb,ST(ψ)`**, against standard Gordon on the full 3320-scene
L23 set (bold = better):

| λ [nm] | ZTT 0° | Gordon 0° | ZTT 30° | Gordon 30° | ZTT 60° | Gordon 60° |
|---|---|---|---|---|---|---|
| 400 | 5.03 | **2.49** | 4.53 | **2.10** | 7.10 | **4.81** |
| 450 | 5.27 | **2.91** | 4.11 | **2.42** | 6.69 | **4.41** |
| 500 | **3.32** | 3.67 | 3.73 | **3.06** | 8.97 | **4.58** |
| 550 | **3.90** | 4.88 | 7.13 | **4.46** | 11.71 | **6.65** |
| 600 | **3.62** | 6.45 | **4.50** | 6.69 | **7.91** | 10.02 |
| 650 | **3.99** | 7.65 | **3.99** | 8.16 | **7.00** | 11.78 |
| 700 | **4.27** | 9.04 | **3.50** | 9.71 | **6.27** | 13.47 |

**Overall, ZTT wins at all three zeniths**: 4.30 vs 6.02 (0°), 4.70 vs 6.20 (30°),
8.09 vs 9.01 (60°). The pattern is the interesting part — Gordon is better in the
blue and degrades steadily toward the red (2.49% → 9.04% at nadir) while ZTT stays
flat at 3–5%, crossing over near 500–550 nm. An analytic model with an explicit VSF
holds up where a `u`-only polynomial cannot.

**The bug, and how it was found.** The first run of this table had ZTT at 21–25%
rRMS at 60° and predicting `rrs` *increasing* with solar zenith, where L23 has it
falling. The cause was a transcription error in `Md_plus`: the paper's `µw` is the
cosine of the **in-water** solar zenith — Equation (13) writes `µw = cos(θs)`,
unprimed, and §2 fixes unprimed angles as in-water — while `H` and `P3` in the same
expression take the *primed* above-water angle. Using the above-water cosine
throughout inverted the zenith trend and cost ~13 percentage points of rRMS at 60°.

**What is worth recording is how the wrong answer nearly survived.** Before finding
it, a diagnostic had already convinced me the cause was the missing `Pbb(ψ)`:
fitting one constant `P_bb` per zenith gave 0.148 / 0.134 / 0.092 at
ψ = 180° / 158° / 140° — monotonically falling away from backscatter, which is the
correct physical shape, with two of the three inside the literature's 0.12–0.16
sr⁻¹. It looked like confirmation. It was the fit absorbing the µd error into the
one free parameter available.

What actually settled it was an **independently quoted number**: §2.7 states µd runs
0.79–0.94 for sun angles 8°–62°. The above-water cosine gives 0.573 at 62°, far
outside; the in-water cosine gives 0.792 and 0.936, reproducing both endpoints. A
quoted constant discriminated where a physically plausible fit did not — the lesson
being that a free parameter fitted against the same data cannot diagnose a bug in
that fit. Both are now tests.

**Still imperfect, and pinned as such.** ZTT over-predicts the zenith effect: the
60°/0° ratio is **0.855** against L23's **0.949** — right sign, roughly three times
too strong in its departure from unity. A test pins the current value so it cannot
drift silently, and this is precisely the kind of structured residual M3's emulator
exists to absorb.

**Outstanding.** Equation (8)'s `m1..m16` (JXP has emailed the authors); until they
arrive, results are *ZTT with the TT2017 µ∞*, and `mu_inf_coeffs=` restores the
published 2018 model in one line.

### 4.5 Notebook

`notebooks/RT/rt_elastic_coding_3.ipynb` — 20 cells, executed, three figures.
Organised around the physics rather than the API: why an explicit VSF buys anything
over a `u`-only model; what was transcribed and what the papers failed to supply;
the bug and how it was caught; ZTT against Gordon; the gradient gate.

- **Figure 1** puts the water and particle backward phase functions on one axis.
  They cross: at 180° water returns **0.234** sr⁻¹ of its `bb` toward the sensor
  against particles' **0.153** — a factor 1.5 — which is the physical cause of the
  non-univocality notebook 2 showed empirically. The shaded band marks the only
  slice L23 reaches (ψ = 140–180°).
- **Figure 2** is the rRMS ladder per zenith, ZTT vs Gordon, three panels. It shows
  the crossover and the shape: Gordon better in the blue, degrading toward the red;
  ZTT flat at 3–5%.
- **Figure 3** is the gradient gate as a step-size sweep for all four inputs, with
  the gate line and the two steps quoted in the text.

Two things the notebook surfaced that the code had not:

- **No single finite-difference step clears the gate for all four inputs.** At
  `h = 1e-6` the three IOP-like variables sit at 1e-10 or better while `theta_s`
  misses at 1.3e-6; at `h = 1e-3` `theta_s` is superb at 3e-10 while the others fail
  by 4e-5 to 7e-3. `theta_s` is O(30) and the others O(1e-3)–O(0.1), so the same
  absolute step is a wildly different relative perturbation. This is why the test
  suite parameterises the step per variable.
- **A step larger than the variable can leave the physical domain.** For `bb_p`
  (O(3e-3)), steps ≳ 3e-3 drive it negative and the model returns NaN. A first pass
  at the figure let `argmin` select one of those NaNs as the "best" step, quoting
  5e-3; the sweep now masks non-finite results and reports how many steps were
  invalid. Worth remembering for M4's gradient protocol: an invalid step is not an
  accurate one.

### 4.6 Results — the Gordon benchmark

**The JAX implementation reproduces the synthesis figure script exactly.** Standard
Gordon rRMS per wavelength at Y = 0, versus `context/RT/fig_rrms_ladder.csv`:

| λ [nm] | 400 | 450 | 500 | 550 | 600 | 650 | 700 |
|---|---|---|---|---|---|---|---|
| published [%] | 2.4948 | 2.9092 | 3.6714 | 4.8786 | 6.4499 | 7.6535 | 9.0418 |
| this code [%] | 2.4948 | 2.9092 | 3.6714 | 4.8786 | 6.4499 | 7.6535 | 9.0418 |

Agreement is to better than 1e-5 percentage points at all seven wavelengths — an
independent NumPy implementation and this JAX one agree, so the metric, the
`Rrs→rrs` conversion, `u`, and the coefficients are all consistent.

**Per-zenith, and this is where the expectation was wrong.** The prompt predicted
Gordon's error would simply grow with solar zenith. It does not:

| λ [nm] | 0° | 30° | 60° |
|---|---|---|---|
| 400 | 2.49 | **2.10** | 4.81 |
| 550 | 4.88 | 4.46 | 6.65 |
| 700 | 9.04 | 9.71 | 13.47 |

60° is much the worst everywhere (roughly 1.5–2× the nadir error), but **30° is
*better* than nadir in the blue** — the fixed Gordon coefficients happen to suit
~30° better than 0° below ~550 nm. So the honest statement is "60° is the clear
loser", not "error grows with zenith".

That has a consequence for how M4's result should be read: the geometry hold-out is
*exactly* the angle where Gordon is weakest, so a hybrid win there is partly a win
against a baseline evaluated outside its best geometry. Worth stating plainly rather
than banking as a clean victory.

### 4.7 Tests — task 1 (baselines)

`robust/tests/test_baselines.py` — 19 tests. `rrms` known-answers (a uniform 10%
overprediction scores 10%), its relative-not-absolute property (equal relative
errors on 2e-2 and 2e-5 score identically — the reason the design specifies a
relative metric), axis reduction for the per-λ ladder, and `jit`/`grad` safety.
Gordon: the algebra spelled out independently, coefficient values, `Rrs`/`rrs`
consistency, batching, `jit`/`grad` with derivative signs, the `forward`-signature
call, and the geometry/phase-function blindness. Against data: three
fixture-backed tests that run in CI (including pinned per-zenith rRMS on the
50-scene subset), and three `needs_l23` tests — the ladder reproduction, the
60°-is-worst asymmetry at 400/550/700 nm, and the per-λ ladder shape. One test also
guards the reference CSV itself, since a silently missing or malformed reference
would make the gate vacuous.

## 5. M3 — Residual emulator + hybrid

**Goal.** The learned half: a small Flax MLP trained (Optax) on the residual
`rrs_L23 − rrs_ZTT` over the M1 train split only, and the public `forward()`
assembling `Rrs_ZTT + ΔRrs` behind the `mode` flag — the first trained,
end-to-end differentiable forward model.

### 5.1 Task status

| # | Task | Status |
|---|------|--------|
| 1 | `emulator.py` — Flax MLP residual emulator + Optax training | ✅ done |
| 2 | `hybrid.py` — `forward()`; gates in `test_hybrid.py` | ✅ done |
| 3 | `notebooks/RT/rt_elastic_coding_4.ipynb` — the M3 explainer | ✅ done |
| 4 | PR-review pass | ⬜ pending |
| 5 | Hand-off edit to `rt_elastic_coding_prompt_5.md` (M4) | ⬜ pending |

**Branch for JXP** — all on `rt-elastic-prototype`, awaiting his commit:
`robust/rt/emulator.py` and `robust/rt/hybrid.py` (both were stubs),
`robust/tests/test_emulator.py`, `robust/tests/test_hybrid.py`,
`design/py/train_emulator.py`, `robust/rt/files/emulator_l23.npz`,
`notebooks/RT/rt_elastic_coding_4.ipynb`, plus modifications to `setup.py` and
`robust/tests/test_env.py`.

### 5.2 Modules added

**`robust/rt/emulator.py`** — the residual emulator. Public surface: `FEATURES`
(seven, in order: `log10_u`, `eta_bb`, `B_p`, `wave_nm`, `cos_theta_s`,
`cos_theta_v`, `cos_dphi`), `EmulatorConfig` (`hidden=(16,16)`, `delta_max=0.5`,
`penalty=0.02`, `learning_rate=3e-3`, `steps=3000`, `seed=23`, `eval_every=100`),
`LINEAR_CONFIG` (`hidden=()` — the baseline), `Emulator` (a registered pytree:
`params`/`mean`/`std` are leaves, `config` is static, plus `domain` — §5.5),
`History`, `features()`, `fit()`, `fit_l23()`, `save()`/`load()`,
`DEFAULT_WEIGHTS`, `load_default()`.

Four decisions shape the module; each is a measurement or a proof, not a taste
(the full arguments live in the module docstring):

1. **The correction is relative.** The net emits a dimensionless `δ(λ)` and
   `Δrrs = δ · rrs_ZTT` — still additive as the design specifies, but the target
   is O(1) rather than spanning four decades (`rrs` runs 2.5e-2 in the blue to
   6e-6 in the red). Measured `|δ|` rms **6.44%**, against the residual's own
   measured **5.52%** sd: the emulator corrects the residual, it does not add a
   large correction that partly cancels.
2. **The feature set is provably complete, not a guess.** `rrs_ZTT` is
   *scale-invariant* — scaling `(a, bb_w, bb_p)` by k=10 moves it **8.8e-15**
   relative, and a test pins that — so the backbone sees its inputs only through
   ratios, and `(u, η_bb)` invert back to `(a : bb_w : bb_p)` exactly. Absolute
   magnitudes are therefore deliberately absent.
3. **λ and `cos θ_s` are first-class** — they carry the two structures M2
   measured in the residual (the zenith offset and the 550 nm hump).
   Standardisation statistics come from the **train split only** and live inside
   the `Emulator`, because a train/inference mismatch is silent.
4. **Bounded by construction.** `δ = delta_max·tanh(·)`; a soft size penalty in
   the same percent units as the loss; and a **zero-initialised output layer**,
   so an untrained hybrid *is* the backbone and every reported gain is a gain.
   `tanh` rather than `relu` because M3's gate is a finite-difference check and
   a relu kink breaks central differences.

One structural choice on top: the emulator is **pointwise in λ** — one shared
network per wavelength, mirroring the backbone's own locality — so it is defined
on any wavelength grid, not just the canonical 81 bands.

**`robust/rt/hybrid.py`** — the public interface: `forward()` (returns `Rrs`),
`rrs_forward()` (returns `rrs`, the scored quantity), `MODES`, and
`DomainWarning` (§5.5). What `mode="emulator"` does and does not mean is an M4
interface question — §5.7.

**`design/py/train_emulator.py`** — the real training run (~60 s), kept outside
the test suite for the same reason PAB's MCMC is; `--dry-run` and `--out` flags.

**`robust/rt/files/emulator_l23.npz`** — **6.5 kB of trained weights,
committed**, so `forward()` is a trained model out of the box and CI exercises
the real thing. `setup.py` gained `package_data` for `robust/rt/files/*.npz` and
`robust/tests/files/*.npz`, without which an installed copy would import fine
and then fail at the first `forward(mode='hybrid')`.

### 5.3 Tests and gates (task 2)

`robust/tests/test_emulator.py` — 29 tests. `robust/tests/test_hybrid.py` — 22
tests. `test_env.py`'s `test_unimplemented_stubs_raise` was replaced by
`test_no_stubs_remain` — `forward` was the last stub on its list, exactly as
`ztt.Rrs_ZTT` had been removed from it at M2.

**The M3 gate, strengthened.** The coding plan's gate ("hybrid beats standard
Gordon on the train split at all three solar zeniths") was already satisfied by
ZTT alone, so an emulator outputting exactly zero would have passed it. It was
strengthened to **beat `mode="ztt"`** at all three zeniths, with the reduction
reported as a number. That is Q5 in prompt 4 and is still **unanswered by JXP**;
the stronger test is what is implemented. Measured by the gate — a deterministic
400-step toy fit on the committed 50-scene fixture, train split, so it runs in
CI — rRMS in `rrs` space:

| zenith | ZTT | hybrid | Gordon |
|---|---|---|---|
| 0° | 4.154% | **0.527%** | 5.268% |
| 30° | 3.899% | **0.525%** | 5.533% |
| 60° | 7.409% | **0.542%** | 8.623% |

**Other gates in `test_hybrid.py`:**

- `mode="ztt"` reproduces `ztt.rrs_ZTT` **bitwise**, so the flag cannot silently
  change the physics.
- **Additivity is exact in `rrs` space and violated by 1.28e-4 sr⁻¹ in `Rrs`
  space** — the air-water interface is non-linear, which is why scoring happens
  in `rrs`.
- **The gradient gate passes through the full hybrid.** `jax.grad` vs central
  finite differences through the emulator, under float64 with per-variable steps
  (the M2 lesson): `a` 2.7e-9, `bb_p` 6.8e-11, `B_p` 1.7e-9, `theta_s` 5.4e-10
  (steps 1e-6, 1e-9, 1e-8, 1e-3; tolerance 1e-6).

### 5.4 Results — the hybrid on the full L23 batch

rRMS (%) in `rrs` space, full 9960-sample batch, scene split:

| model | train | held-out scenes | held-out scenes @60° |
|---|---|---|---|
| Gordon | 7.21 | 7.21 | 9.01 |
| ZTT backbone | 5.95 | 5.93 | 8.11 |
| hybrid, linear (8 params) | 2.57 | 2.54 | 2.48 |
| **hybrid, MLP(16,16) (417 params)** | **0.30** | **0.30** | **0.32** |

Held-out equals train to two decimals — 417 parameters against ~645k training
rows (7968 samples × 81 λ). MLP(32,32) reached only 0.27%, confirming the
design's "start small". **The linear baseline is the honest yardstick**: 2.57%
means the MLP's nonlinearity earns its place by ~8×, not the ~20× over the
backbone that a baseline-free presentation would suggest.

**Throughput** (jitted, CPU, 9960×81): the hybrid costs **≈ 4.8× the backbone** —
measured 17.1 ms against 3.5 ms (47 vs 228 M sample·λ/s) in notebook 4's run, and
14–15 ms against ~3 ms in others; wall-clock wanders ~20% between runs on this machine,
so the *ratio* is the reproducible number. Gordon is 0.3 ms. The learned half is
therefore the majority of the cost — it evaluates a 417-parameter network at each of
806,760 sample·λ points — but the hybrid does not collapse the analytic model's speed
advantage over calling an RT solver.

### 5.5 The open problem — geometry extrapolation is not reproducible

**This is M4's main risk.** The emulator interpolates superbly, but trained on
0°/30° only and asked for the unseen 60°, MLP(16,16) scores ~0.24% in sample
and, over the seeds {23, 1, 7, 101, 2024}: **4.7 / 8.4 / 7.8 / 5.4 / 12.2%** —
a 7.6× spread whose median, 7.75%, barely improves on ZTT's 8.09% there, and
whose worse half is beaten by Gordon's 9.01%. The cause: `cos θ_s` spans
[0.866, 1.0] in that training set while 60° needs 0.5, so every `tanh` unit is
outside its fitted range and the initialisation decides the answer. The
**linear** model gives up a lot in sample (2.40%) but is stable at **6.16%**,
beating both references — its inability to bend is what saves it.

**JXP's decision (Q6, answered): report and defer.** Extrapolating to 60° is a
stretch goal; trusted outputs at that angle can be used if we get them; and the
emulator will not be used at larger angles without warning the user. That
decision is implemented, not just noted: `Emulator.domain` carries the
per-feature train min/max with the weights, and `hybrid.DomainWarning` is raised
whenever a mode that uses the emulator is evaluated outside the trained range.
The check is a boundary check (concrete values), so it is skipped under
`jit`/`grad` — deliberately, and documented.

**A wrong inference caught before it shipped** — the record keeps M2's analogous
lesson, so here is M3's. A linear skip path (`δ_raw = W·x + MLP(x)`) appeared to
fix the extrapolation problem — 11.57% without it, 5.40% with — and was nearly
adopted on that comparison. But the two runs also differed in Flax parameter
*names*, which changes PRNG folding and hence the initialisation, so
architecture and seed had moved together. A seed sweep with the architecture
fixed showed the skip is no better (median 9.20% vs 7.75%, worst case 25%), and
it was removed rather than shipped as a knob justified by a fluke.

### 5.6 Two bugs found and fixed

1. **Training NaN-ed on the first chunk.** Two deliberate choices collided: the
   output layer is zero-initialised so `δ ≡ 0`, and the size penalty is an RMS
   whose derivative `δ/(N·√mean δ²)` is 0/0 exactly there. Fixed with
   `_RMS_EPS = 1e-24` inside the square root — it puts a 1e-12-percent floor
   under the term, unmeasurable, and makes the gradient at δ=0 a clean zero. A
   regression test asserts `jax.grad` of the objective is finite at init.
2. **The domain check's traced-input guard was partial.** It inspected only
   `iops.a` and `geometry.theta_s`, so `jax.grad` w.r.t. `bb_p` or `B_p` alone —
   which is precisely what an inversion differentiates — left those two
   concrete, the guard reported "not traced", and `out_of_domain`'s `np.asarray`
   died with `TracerArrayConversionError` on the default code path. Now every
   leaf is inspected via `jax.tree_util.tree_leaves`. Verified by reverting the
   fix: the new parametrised regression test fails on exactly `bb_p` and `B_p`
   and passes with it.

### 5.7 An interface question left open for M4

`mode="emulator"` returns the learned **correction term** `Δrrs`, not a
standalone learned model, because the emulator is parameterised as a relative
correction to the backbone. The design's "learned-only" comparison therefore
needs a differently trained network — one predicting `rrs` outright across four
decades — which belongs beside PR05 and O25 in M4's protocol rather than as a
flag on `forward`.

### 5.8 Notebook

`notebooks/RT/rt_elastic_coding_4.ipynb` — the M3 explainer, **executed** with
outputs (30 cells, 6 figures, ~5 min on the full batch; degrades to the committed
fixture without `$OS_COLOR`). House style and the CVD-validated categorical set from
notebooks 1–3, with the five seed replicates in one de-emphasised grey rather than five
hues — they are replicates, not five things to tell apart — and every series directly
labelled, since `#56B4E9` is below 3:1 contrast on white.

Eight sections. **§1 specifies the model itself**, stage by stage — the unfitted ZTT
backbone, the seven-feature map evaluated per wavelength, the train-split
standardisation, the 7→16→16→1 `tanh` MLP with its zero-initialised output, the
`δ = 0.5·tanh` bound, and the assembly `rrs = rrs_ZTT(1 + δ)` followed by the non-linear
interface — with the loss written out and the parameter inventory *counted from the
packaged weights* rather than asserted (417, and the three arrays that travel untrained:
`mean`, `std`, `domain`). It carries two figures: a dataflow schematic separating the
physics path from the learned half, and a term-by-term decomposition of one representative
60° water body showing the correction landing on the backbone's residual and cancelling
it. §1 also demonstrates `load_default()` and the `rrs`-space additivity of the three
modes (exactly 0 in `rrs`, 6.6e-4 sr⁻¹ in `Rrs`).

Then: why the correction is relative; the features as the backbone's complete state (the
scale-invariance check runs live, at 6.7e-16); the linear baseline *before* the MLP; fit
versus generalisation; **where it fails** — the five-seed extrapolation fan at an unseen
zenith, which is the notebook's real contribution; the gradient gate and throughput; and
what M3 leaves open.

**The notebook is also where four wrong numbers were caught**, because it recomputes
everything it claims rather than quoting this record:

- the hybrid/ZTT throughput ratio (§5.4) had been the *emulator*/ZTT ratio, 3.3×,
  misapplied to the total; measured end to end it is ~5×;
- the scale-invariance demonstration was silently running in float32 (2e-7, float32
  epsilon) because `jax_enable_x64` was set *after* the arrays were built — the dtype
  has to be pinned on the arrays, which is a gotcha this project had already recorded;
- the "four decades of dynamic range" argument had been measuring the *mean* spectrum
  (factor 419); the per-sample range is 4300;
- the domain check warned on in-range data, which exposed the flaw fixed in §5.2's
  `DOMAIN_TOL`.

A review pass over the executed notebook caught four more overstatements, all now
corrected in it: the finite differences are ~2e-7 rather than ~1e-9 for `B_p` and
`theta_s`; the title quoted the linear model's train number in a held-out sentence; the
`DOMAIN_TOL` headroom is 27× below and 48× above, not symmetric; and the per-λ figure
claimed the MLP hybrid sits "a decade" below the linear model with a "spectrally
uniform" improvement — in fact the gap narrows to ~3× where the linear model dips, and
the **blue retains the largest residual** (0.33–0.40% against 0.19% in the red). The
defensible claim, which is the one that matters, is that no wavelength region is
abandoned.

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
    types.py           ✅ M1  IOPs / PhaseParams / Geometry pytrees
    data/
      __init__.py      ✅ re-exports l23
      l23.py           ✅ M1  L23 elastic batches + seeded splits
    ztt.py             ✅ M2  ZTT; µ∞ from TT2017 pending Eq. (8) coeffs
    emulator.py        ✅ M3  relative-δ Flax MLP + Optax training, save/load
    hybrid.py          ✅ M3  forward()/rrs_forward(), MODES, DomainWarning
    validation.py      🟡 M2  rrms(); rest of the protocol at M4
    baselines.py       ✅ M2  standard Gordon (PR05/O25 at M4)
    files/
      emulator_l23.npz ✅ M3  trained MLP(16,16) weights (6.5 kB, committed)
  tests/
    __init__.py        ✅
    conftest.py        ✅ needs_l23 marker, jax_x64 fixture
    files/             ✅ empty (.gitkeep); M1 caches an L23 batch here
    test_env.py        ✅ 12 tests — the M0 gate
    test_conventions.py ✅ 27 tests — M1 task 1
    test_types.py      ✅ 32 tests — M1 task 2
    test_l23.py        ✅ 46 tests — M1 task 3 (+ cached-fixture layer)
    test_baselines.py  ✅ 19 tests — M2 task 1
    test_ztt.py        ✅ 28 tests + 1 xfail — M2 task 3
    test_emulator.py   ✅ 29 tests — M3 task 1
    test_hybrid.py     ✅ 22 tests — M3 task 2 (the strengthened gate)
    files/l23_small.npz ✅ 213 kB, 50 scenes x 3 zeniths (raw fields)

design/py/
  train_emulator.py          ✅ M3 real training run (~60 s; --dry-run, --out)

notebooks/RT/
  rt_elastic_coding_1.ipynb  ✅ M0 explainer (executed, 2 figures)
  rt_elastic_coding_2.ipynb  ✅ M1 explainer (executed, 3 figures)
  rt_elastic_coding_3.ipynb  ✅ M2 explainer (executed, 3 figures)
  rt_elastic_coding_4.ipynb  ✅ M3 explainer (executed, 4 figures)

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

**M3's packaging change.** `setup.py` gained `package_data` for
`robust/rt/files/*.npz` and `robust/tests/files/*.npz` — without it an installed
copy imports fine and then fails at the first `forward(mode='hybrid')`, because
the trained weights (`robust/rt/files/emulator_l23.npz`, 6.5 kB, committed)
would not ship. Committing the weights is also what lets CI exercise a *trained*
hybrid rather than a zero-initialised one; the M3 gate itself trains a
deterministic 400-step toy fit on the committed fixture, so it runs in CI too.

**Suite size at M3.** `pytest -q` → **225 passed** with `$OS_COLOR`; **205
passed, 20 skipped** without it, which is what CI sees. The suite was 171 at
M2's close.

**Badge** in `README.md`. It will read "no status" until `ci.yml` reaches the
default branch — the workflow file has to exist on `main` for the badge's default
view to resolve.

*Living document; updated at the close of each milestone.*
