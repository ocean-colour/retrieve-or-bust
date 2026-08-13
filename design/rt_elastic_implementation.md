# Elastic RT Implementation Record

**Version:** 0.30
**Date:** 2026-08-13
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
| **M2** | ZTT analytic backbone (JAX) | ✅ done | `robust.rt.ztt`, `robust.rt.baselines` |
| **M3** | Residual emulator + hybrid | ✅ done | `robust.rt.{emulator,hybrid}` |
| **M4** | Validation (*prototype done*) | ✅ done — **the Week-1 prototype is complete** | `robust.rt.validation`, `robust.rt.baselines`, `design/py/run_validation.py` |
| **M5** | Beyond week 1 (PB24: phase function + BRDF) | 🟡 in progress — tasks 0–15 done (11 **failed its gate**), 16 specified (§7) | `robust.rt.conventions` (grids), `robust.rt.data.pb24` |

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

**Verification (current).** `pytest -q` → **416 passed** (`ocean14`); with
`$OS_COLOR` unset, **372 passed + 44 skipped** — which is what CI sees. The loader is
exercised without the dataset against a committed 50-scene fixture.
`ruff check robust/` and `ruff format --check robust/` → clean. The suite is green both with and without the L23
reference data on disk (missing data skips, never fails). All five notebooks in
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
| 4 | PR-review pass (PR #11) | ✅ done |
| 5 | Hand-off edit to `rt_elastic_coding_prompt_5.md` (M4) | ✅ done — that doc's §M4 is the hand-off |

**Branch for JXP** — all on `rt-elastic-prototype`, since committed:
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

### 5.6b The PR review (task 4) — PR #11

`gh` is not authenticated here; the public REST API serves the review comments.
**PR #11** (`rt-elastic-prototype` → `RT`, the M3 diff at commit `6dcaf63`) drew two
Bugbot findings, both in `design/py/train_emulator.py`, both real and both fixed. The
history: #10 (M2) was reviewed at `4d6c628` and found no new issues — though that is
not M2's final commit, so M2's last few commits remain unreviewed — and #9's single
finding (a `bb_p` zenith comparison that skipped 30°) had already been fixed during
M1, its replacement docstring citing PR #9.

1. **High — the weights were written before the round-trip check.** A failed integrity
   check reported an error with the previously shipped `emulator_l23.npz` already
   overwritten: the one file that `load_default()` and `forward(mode="hybrid")` depend
   on, destroyed by the very check meant to protect it. The save path is now
   `write_weights()`: the candidate goes to a temporary file *beside* the destination,
   is loaded back and required to reproduce the correction, and only then moves into
   place with an atomic `os.replace`. A `finally` removes the candidate on every exit,
   so a failed check leaves nothing behind either.
2. **Medium — the shipped emulator was whatever the training loop left in scope.**
   Reordering or trimming the two-fit loop would have persisted the 8-parameter
   **linear** baseline as the default weights, and nothing would have crashed: `load`
   restores the stored `config`, so `forward(mode="hybrid")` would have quietly used a
   model worth 2.57% instead of 0.30%. The script now collects results into a dict and
   selects the shipped one **by name** (`SHIPPED`), and refuses to write if its
   architecture is not the package default.

**Both fixes were demonstrated against what was reported**, not merely asserted. With
`load` monkeypatched to return a different model, `write_weights` returns status 1, the
destination is byte-identical to before, and no temporary file survives; with `load`
intact the same call writes and returns 0. Handed the linear baseline, the script's
guard refuses it, and the new test
`test_packaged_weights_are_the_default_architecture` fails on a linear file while
passing on the real one (417 parameters, `hidden=(16,16)`).

**The class fix, not just the instance.** Finding 1's class is *validating an artifact
only after it has replaced a known-good one*, and the repo had one other instance:
`l23.write_fixture` wrote straight onto `robust/tests/files/l23_small.npz` — the
committed fixture the entire suite runs on without `$OS_COLOR` — with no check that
the result loads. It now snapshots to a temporary file, loads it back through
`npz_reader` and `load_batch` (the real consumers) and validates the batch, then
replaces atomically. `test_write_fixture_reproduces_the_committed_snapshot` pins that
regenerating gives the committed bytes back exactly, so fixture and loader cannot
silently drift; the guard itself was demonstrated by forcing the verification to fail
and confirming the destination survived. Finding 2's class — *shipping whatever a loop
left behind* — is guarded durably by the architecture test, which holds on the file
however the file came to be, including a hand-copied one.

A side benefit of re-running the training script end to end: the regenerated weights
are **bit-identical** to the committed ones, which is independent confirmation of the
determinism claim in §5.2 (full-batch, unshuffled, one seed).

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

**Goal.** The design §6 protocol and the acceptance gate that declares the Week-1
prototype done: every model scored on identical data, per λ / per solar zenith / per
`B_p` bin, on both held-out splits, plus throughput and the gradient gate.

### 6.1 Task status

| # | Task | Status |
|---|------|--------|
| 1 | O25 comparison model in `baselines.py` | ✅ done |
| 2 | `validation.py` protocol + `design/py/run_validation.py` | ✅ done |
| 3 | Acceptance gate in `test_validation.py`; artefacts committed | ✅ done |
| 4 | `notebooks/RT/rt_elastic_coding_5.ipynb` — the M4 explainer | ✅ done |
| 5 | PR-review pass (self-review; §6.9) | ✅ done |
| 6 | Prototype hand-off + edit to `rt_elastic_coding_prompt_6.md` | ✅ done |

### 6.2 The headline, and why it is not the one M3 implied

Full L23 batch, rRMS % in `rrs` space (`design/validation/metrics.md`):

| model | train | held-out scenes | held-out @60° |
|---|---|---|---|
| standard Gordon | 7.21 | 7.21 | 9.01 |
| ZTT backbone | 5.95 | 5.93 | 8.11 |
| **O25 form, refit on L23** (12 par) | **0.70** | **0.69** | **0.71** |
| hybrid, linear (8 par) | 2.57 | 2.54 | 2.48 |
| **hybrid, MLP** (417 par) | **0.30** | **0.30** | **0.32** |

**The hybrid's margin over the state of the art is 2.3×, not 24×.** Against Gordon it
is 24×; against O25 — twelve fitted numbers, four coefficients at each of three solar
zeniths — it is 2.3×, and O25's train and held-out figures are identical, so there is
nothing to write off as overfitting. Per solar zenith on held-out scenes the backbone
degrades (4.26 / 4.67 / 8.11) while O25 and the hybrid do not (0.68/0.69/0.71 and
0.30/0.30/0.32).

Per `B_p` bin nothing varies much (0.27–0.33 for the hybrid). The bins span a factor
1.72, against the design's ~7× nominal band, so **this cut cannot speak to
phase-function generalisation** — it says only that accuracy is flat across the narrow
slice L23 covers.

### 6.3 The acceptance gate as built

The coding plan's wording is *"hybrid beats standard Gordon on **both** held-out
splits, and passes the gradient-correctness gate"*. Two of JXP's decisions reshape it:

- **Gate on beating O25 on the scene split** (Q9). Gordon is the weakest thing in the
  table; a milestone gated on it would pass while losing to the actual benchmark. The
  Gordon and ZTT comparisons are kept as the plan's floor.
- **Report the zenith half, do not gate it** (Q6/Q7). The hybrid's unseen-60° error is
  seed-dependent: **4.74 / 8.37 / 7.75 / 5.40 / 12.24%** across five seeds against
  Gordon's 9.01%.

**An interaction worth recording, because it defeats the fix that was chosen for it.**
Q7 selected the out-of-domain *fallback* (option 3) partly because it would make the
zenith gate deterministic — the emulator, trained on 0°/30°, would degrade to the
backbone at 60°. But JXP's accompanying clarification set the sanctioned envelope at
**0–60°** ("it should be fine to do anything up to that angle"), so the fallback
deliberately does **not** fire at 60°: measured, it triggers on 0 of 9960 samples. The
fallback is still the right thing beyond 60°; it simply cannot rescue this half of the
gate, which is why the gate stops at the scene split. A test pins that inertness so the
reasoning cannot quietly rot.

**On the unseen 60°, the refit O25 wins outright: 4.63%, deterministic** — better than
the hybrid's best seed (4.74%), its median (7.75%), ZTT (8.09%) and Gordon (9.01%).

### 6.4 Modules and artefacts

- **`robust/rt/baselines.py`** gained O25: `Rrs_o25` (the primitive — O25 is defined in
  `Rrs`, the reverse of Gordon), `rrs_o25`, `o25_coefficients`, `fit_o25`,
  `O25_L23_REFIT`, `O25_RRS_CEILING`. The fit is a closed-form weighted `lstsq`, so it
  is deterministic by construction. **PR05 is deliberately absent** — its coefficients
  are a 4-D LUT the paper does not print and the repo does not hold, and L23 is
  nadir-only, so a refit could not populate the sensor-geometry axes (prompt 5, Q8).
- **`robust/rt/validation.py`** gained `rrms_per_wavelength`, `group_rrms`,
  `bp_bin_labels`, `throughput`, `gradient_report`, `score_models`, `markdown_table`,
  `FD_STEPS`, `GRADIENT_TOL`.
- **`robust/rt/hybrid.py`** gained `on_out_of_domain ∈ {"warn", "ztt"}` and
  `robust/rt/emulator.py` gained `SUPPORTED_THETA_S`, `out_of_domain_mask`, and a
  `theta_s_limits` argument — §6.5.
- **`design/py/run_validation.py`** regenerates everything into `design/validation/`:
  `metrics.md`, `metrics.csv`, `rrms_per_wavelength.csv`, and two figures
  (`rrms_per_wavelength.png`, `unseen_zenith.png`).
- **Tests**: `test_baselines.py` +22 (O25), new `test_validation.py` 20 (the protocol,
  the policy, the gate, and the review regressions). Suite **279 passed**;
  **256 passed + 23 skipped** without
  `$OS_COLOR`, which is what CI sees.

### 6.5 The supported envelope, and a policy that survives `jit`

`SUPPORTED_THETA_S = (0.0, 60.0)` is a **project decision**, not a property of a fit:
JXP's Q7 clarification was that anything up to 60° is fine and only beyond it warrants a
warning. The domain check therefore judges `cos_theta_s` against that envelope while the
other six features keep using the trained range, where "outside what I learned" does
mean unreliable. `out_of_domain(..., theta_s_limits=None)` still asks the other question
— whether *this fit* is extrapolating — which is what the research runs want.

The fallback is built on `out_of_domain_mask`, which is **traceable**, rather than on the
host-side warning check, which is not. A policy hung off the warning would lapse silently
under `jit` — the hot path — and a model that changes its answer when compiled is worse
than one with no policy. A test compares jitted against eager for the same function
(5e-7 apart, XLA fusing `rrs_ztt + 0.0` in float32) against what a lapsed policy looks
like (2e-1).

### 6.6 Speed and gradients

Jitted, 9960×81, CPU: Gordon 0.25 ms (0.08× ZTT), O25 0.55 ms (0.18×), ZTT 2.96 ms
(1.00×), hybrid 17.8 ms (**6.0×**). Repeated runs put the hybrid between **4.5× and
6.0×**, so neither column is reproducible to better than ~35% and the *ordering* is what
to rely on. (The table's reference row reads exactly 1.00 by construction, after a first
version that timed ZTT twice and reported it as 0.72× itself.)

Gradient gate, float64, per-variable steps, tolerance 1e-6 — every model, every variable
at or below **5e-9**.

Two properties of the report that are easy to get wrong, both now pinned:

1. **A model that ignores a variable is not "infinitely wrong".** O25 has no
   phase-function input, so `d/dB_p` is exactly zero on both sides; a ratio turned that
   into `inf`, converting a *documented blind spot* into a gate failure.
2. **O25 is not differentiable at its own table nodes.** Its coefficient lookup is
   piecewise linear in `θs`, so autodiff takes one one-sided slope while a central
   difference averages both — measured **69%** disagreement at 30°. L23's three angles
   *are* the nodes, so the protocol evaluates gradients at **45°**, where all four
   variables agree to ≤3e-9.

### 6.7 The fitting objective, and fairness to a rival

O25's coefficients are fitted with the **same relatively weighted objective everything
is scored with**, not the paper's unweighted least squares. It is worth 4×: weighted
reaches 0.70% rRMS, unweighted 2.5–2.7%, because an unweighted objective in `Rrs`
optimises the bright blue and abandons the dark red. Reproducing the paper's choice
would have made our own hybrid look four times better than a fair comparison allows, so
the fair fit is the default and the paper's sits behind `weighted=False`.

The consequence to state whenever these numbers are quoted: **O25's 0.69% is its best
case** — its coefficients were fitted on our training split with our metric as the
objective, and it is labelled *"O25 form, refit on L23"* rather than presented as the
published model. `fit_o25` requires an explicit `train=` mask for the same reason.

### 6.8 Notebook

`notebooks/RT/rt_elastic_coding_5.ipynb` — the M4 explainer, **executed** with outputs
(24 cells, 3 figures, ~8 min on the full batch; degrades to the committed fixture
without `$OS_COLOR`). Its subject is the *protocol and its verdict*, not another
accuracy claim: **§1 defines the two models being compared**, §2 the benchmark change
and what it does to the headline, §3 whether the comparison was fair, §4 the required
breakdowns, §5 the unseen zenith, §6 speed and gradients, §7 **what the prototype may
and may not say** — written as two explicit lists.

§1 exists because the notebook spent its length comparing against O25 without ever
saying what O25 *is*. It separates the model (the bivariate quadratic, every symbol
defined, and why the water/particle split is the whole idea rather than a detail), its
provenance and standing (Pitarch et al. 2025 from L11's form, calibrated on PB24, in
NASA HyperCP and EUMETSAT ThoMaS, operational in OLCI Collection 4 — hence *the*
benchmark), and what our version is not (coefficients refit on L23's train split, no
θv/Δφ axis, so its numbers are its best case). It then restates the hybrid — unfitted
backbone plus a bounded 417-parameter correction — and lays the two side by side in a
table: **12 fitted numbers with no phase-function input against 417 on top of unfitted
physics that has one**. A code cell makes both concrete on one water body, including how
little O25's coefficients move with zenith (`Gw0` 0.0587 → 0.0525 across 60°) and that
the hybrid's δ is only a few percent.

Three figures, each earning its place: the per-λ ladder for all five models (which
shows at a glance that the gap that matters is to O25, not to Gordon); **O25 fitted
both ways**, showing the unweighted objective doing better in the blue and far worse
in the red, i.e. the 3.8× that the paper's own choice would have handed us; and the
unseen-60° comparison as a dot-and-range plot in which only the MLP has a range.

The notebook also demonstrates the two gradient traps live, side by side: O25's
`theta_s` derivative disagrees by **7e-1** *on* a table node (30°) and by **2e-10**
between nodes (45°), and its `B_p` derivative reads exactly 0 because the model
genuinely ignores the phase function.

### 6.10 Definition of done — the prototype, and what it may claim

The coding plan's acceptance is *"hybrid beats standard Gordon on **both** held-out
splits, and passes the gradient-correctness gate"*. **By the gate as amended (§6.3), the
prototype passes** and the Week-1 milestone is complete. The amendments, both JXP's and
both recorded rather than quietly applied: gate on beating **O25** on the scene split
(Gordon is the weakest model in the table and ZTT alone already beats it), and **report**
the zenith split rather than gating it (the outcome is seed-dependent).

The reviewer-facing summary is [`prototype_summary.md`](prototype_summary.md), which
states the six things the prototype may **not** claim alongside the one it may. In short:
0.30% rRMS on held-out scenes, uniform across the spectrum and the three solar zeniths,
differentiable to ≤5e-9, at ~5× the backbone's cost — but a **2.3×** margin over the
modern benchmark rather than the 24× over a 1988 model, with geometry extrapolation
unresolved, phase-function generalisation untested, and O25's own number being its best
case on our training data.

Every number in that summary is checked programmatically against
`design/validation/metrics.csv` and against live `pytest` runs, not transcribed.

### 6.9 The review pass (task 5)

Two rounds. **PR #11** was merged on 2026-08-05 with its findings fixed during M3's
task 4 (§5.6b), leaving every commit after `6dcaf63` — all of M4 — unreviewed, so the
first round was a self-review of that diff, run adversarially against the model code,
the tests, the scripts and the artefacts. **PR #12** then opened on the finished M4 work
and Bugbot reviewed it; its single finding is §6.9's last entry, and it is about one of
the fixes the self-review had just made. Findings, all confirmed by reproduction before being fixed,
and each now pinned by a regression test proven to fail when the fix is reverted:

1. **High — a slightly off-nadir view passed the domain check while the emulator's
   output was meaningless.** `cos_theta_v` is constant in L23, so its trained span is
   zero, and the check scaled the excursion by the feature's own *value*: a sensor
   zenith of 5° looked like a 0.4% excursion, inside `DOMAIN_TOL`. But the
   standardisation divides the same excursion by `_STD_FLOOR`, so the network saw
   **−3.8e5**, every `tanh` saturated, and the correction collapsed from a spectrum
   spanning [−0.10, +0.27] to a **flat +0.046 at all 81 wavelengths** — with no
   warning and no fallback. The window was 0° < θ_v ≤ 8.1°. Both now divide by
   `_STD_FLOOR`, so the check measures the excursion in the units the network actually
   sees. Half a degree off nadir is now flagged; exact nadir (all of L23) stays clean.
2. **Medium — the two domain predicates disagreed on NaN.** `out_of_domain` (host) and
   `out_of_domain_mask` (traceable) implement one predicate twice, and `excess > tol`
   is False for NaN: the mask — the one the fallback policy acts on — answered "in
   domain" while the host answered "out". A single NaN in `a` is what an inversion
   overshoot produces, so that was the path that mattered. The mask now negates the
   in-range test, which sends NaN to `True`, and the host reports non-finite input as
   `excess = inf` rather than printing "nan% of the trained span".
3. **Medium — both committed CSVs were silently malformed.** `run_validation.py`
   joined fields with commas by hand, and the model names contain commas ("O25 form,
   refit on L23", "hybrid, MLP"). `metrics.csv` carried four fields under a
   three-field header, and `rrms_per_wavelength.csv`'s header expanded seven model
   names into ten columns — so a consumer would have mis-labelled every column with
   nothing raising. Found by trying to *consume* the artefact to cross-check §6.2's
   table. Now written through :mod:`csv`; two tests assert the committed files parse
   to their promised columns and that the two agree on the model list.
4. **Low — `gradient_report` silently replaced the caller's geometry with nadir**,
   discarding `theta_v`/`dphi`. Harmless on nadir-only L23 and exactly the thing that
   would certify a gradient at the wrong geometry once M5 goes off-nadir. A spy model
   pins it.
5. **Low — `gradient_report`'s `steps` dict.** A missing key raised `KeyError` from
   inside a closure; an extra key was worse, reporting **0.0** — "perfect agreement" —
   for a variable that is never perturbed. It now validates the key set.
   `throughput(repeats=0)` divided by zero and now raises.

A second reviewer, on the tests and scripts, found five more:

6. **High — a committed artefact was stale.** The `standard Gordon` column of
   `rrms_per_wavelength.csv` aggregated to **9.57%** where `metrics.csv` said
   **7.21%**, and the figure drawn from it overstated Gordon's blue-end error by ~2×.
   It was already corrected by this session's regeneration, but nothing would have
   *noticed*: so the check that found it is now a test —
   `test_the_per_wavelength_ladder_aggregates_to_the_scalar_table` RMS-es each per-λ
   column and requires the pooled scalar back. It needs no trust in the model code at
   all, being pure consistency between two files from one run.
7. **High (tests) — the gate's margin was seed luck, and its docstring overstated it.**
   With a 400-step toy fit the hybrid scored 0.454–0.575% across five seeds against
   O25's 0.578% — the worst seed passing by **0.6%** — while the docstring claimed
   "~0.53% against O25's ~0.9%". The fit is now 800 steps (0.376–0.419%, a 27%
   margin) and the assertion demands `hybrid < 0.9 × O25`, so a 3% shrink of the
   learned correction now fails where it used to pass.
8. **Medium (tests) — a test named for slicing passed when slicing was removed.**
   `test_score_models_slices_one_evaluation_per_model`'s fixture models had *uniform*
   relative error, so every subset scored identically and a `score_models` that
   ignored `masks` entirely passed. Its models now differ between the halves.
9. **Medium (tests) — the "0.0 is agreement" rule could hide an unexercised
   variable.** Finding 5's own fix — reporting 0.0 where a model genuinely ignores a
   variable — would equally have hidden a perturbation that was never applied: with
   the `B_p` offset dropped, every gradient test passed, including for ZTT, which does
   depend on it. The gate now also asserts each report entry is **non-zero** for the
   hybrid, which depends on all four.
10. **Medium (test infra) — the gate fixture's dtype depended on test order.** `jax_x64`
    is function-scoped while `gate_fit` is module-scoped, so whichever test asked first
    decided whether the emulator trained in float32 or float64 (0.5468% vs 0.5628%,
    half the old margin). The gradient test now trains its own short fit.
11. **Low — script guards.** `--quick` silently overwrote the committed artefacts with
    300-step numbers (now refused unless `--out` is explicit); a missing `$OS_COLOR`
    produced a raw h5py traceback pointing at a repo-relative path that never existed
    (now a clean message); the PNGs bypassed the temp-file-then-replace rule the
    docstring claimed for all outputs (now included, and the docstring notes that the
    *set* of five files is still not written transactionally); and
    `train_emulator.py`'s architecture guard fired only after ~2 minutes of fitting and
    was skipped by `--dry-run` — it now checks the config up front.

Also documented rather than changed: `bp_bin_labels`' "equal count" holds only for
mostly-distinct values (heavy ties can empty a bin; L23 gives exactly 2490 per bin),
and `Rrs_o25` with a column-shaped `theta_s` returns a cross product instead of
raising — outside the documented `Scalar` contract, but worth knowing.

**One reported finding was wrong, and checking mattered.** The review held that nothing
protects the emulator against train/test contamination, since a contaminated fit still
passes a score-comparison gate. True of the gate — but mutating `fit` to train on the
held-out scenes fails **six** tests, four of them the `test_emulator.py` checks that pin
the standardisation statistics and the domain to the *train* rows. The protection lives
upstream of the gate, which is the right place for it.

**PR #12 — Bugbot, on a fix the self-review had just made.** Finding 11 above moved
`train_emulator.py`'s architecture guard from *after* training (where it inspected the
trained emulator) to *before* it (where it inspects the module constant
`SHIPPED_CONFIG`), to fail in milliseconds and to cover `--dry-run`. Bugbot pointed out
what that traded away: the guard now validates a **proxy** for the artefact rather than
the artefact. A training loop that passed some other config while the constant still
read `EmulatorConfig()` would write weights whose architecture does not match
`load_default()`. Correct, and a regression I introduced while fixing something else.

Both properties are now kept: `check_architecture()` runs early on `SHIPPED_CONFIG` for
fast failure and `--dry-run` coverage, **and** inside `write_weights()` on the emulator
actually being serialised. Demonstrated: with the constant untouched and the loop
handing over a linear emulator, the early check passes and `write_weights` refuses,
leaving the destination byte-identical and no temporary file behind, while a
correct-architecture emulator still writes. The durable half remains
`test_packaged_weights_are_the_default_architecture`, which checks the committed file
however it was produced.

**The class, stated once**: a guard belongs next to the artefact it guards, and must
inspect the artefact rather than a stand-in for it. The repo's other guards were checked
against that rule and hold — `write_fixture` loads its snapshot back through the real
reader, `emulator.load` compares the file's own recorded feature list, the packaged-
weights test reads the committed file, and `run_validation.py`'s `--quick` refusal
inspects the actual arguments.

**A cross-check worth keeping.** §6.2's table is now verified against `metrics.csv`
programmatically rather than by eye, which is how finding 3 surfaced at all.

## 7. M5 — Beyond week 1

**Goal.** Close the two axes the Week-1 prototype could not speak to — the **particle
phase function** and the **full BRDF** — on the reference data that turned out to carry
both. M4's summary lists six things the prototype may not claim (§6.10); M5 exists to
convert items 4 and 5 from *untested* into *measured*.

**Status: in progress.** Tasks 0–2 (answers, answers, sequencing) and the two prerequisites
— **3** (a second wavelength grid, §7.6) and **4** (the PB24 loader, §7.7) — plus **5**
(the three splits, §7.8), **6** (the validation toolkit, §7.9), **7** (the surface
transfer, §7.10), **8** (O25's geometry table, §7.11) and **9** (the PB24 benchmark,
§7.12), **10** (the per-model envelope, §7.13) and **11** (the PB24 retrain, §7.14 — which
**failed its gate**) **12** (the cross-dataset check, §7.15) **13** (ZTT's internals, §7.16) **14** (the backward-VSF axis, §7.17) and **15** (the API freeze, §8.0) are done; task 16
is specified with a gate each in
[`rt_elastic_coding_prompt_6.md`](../claude_prompts/RT/rt_elastic_coding_prompt_6.md) §M5
and summarised in §7.5 below.

### 7.1 Task status

| # | Task | Gate | Status |
|---|------|------|--------|
| 0 | Fold in Q10/Q11; confirm the reference data | — | ✅ done |
| 1 | Fold in Q12/Q13/Q14 | — | ✅ done |
| 2 | Sequence the milestone; fill this section | — | ✅ done |
| 3 | `conventions` accepts a second wavelength grid | L23 path unchanged; a PB24-grid `IOPs` validates; `bb_w(753)` cannot clamp silently | ✅ done |
| 4 | `robust/rt/data/pb24.py` — the loader | golden values vs raw netCDF; fixture regenerates bit-identically; angle window asserts its count; zero-`rrs` gate on the **shell** load | ✅ done |
| 5 | `pb24.make_splits` — realisation / `B_p` band / geometry | disjoint, deterministic, **every test set non-empty**; the `B_p` split reports its chlorophyll confound | ✅ done |
| 6 | Extend `validation.py` — `gradient_report` axes, `rrms` masking, group↔header alignment | one regression test per limit, each demonstrated to fail first | ✅ done |
| 7 | Geometry-aware surface transfer in `conventions` | default path bit-identical to M4; fitted path ≥5× better at θv = 60°; gradient-checked | ✅ done |
| 8 | O25 gains a geometry-indexed coefficient table | 3-D refit beats θs-only off-nadir or is dropped; L23 path reproduces `O25_L23_REFIT` exactly; a missing zenith **raises** | ✅ done |
| 9 | PB24 benchmark — Gordon, ZTT, O25 refit | aggregation consistency **and header↔group alignment**; refit on train mask only; CSVs round-trip | ✅ done |
| 10 | Per-model sanctioned envelope | the L23 model stays 0–60° after the change, pinned by test; a view-angle envelope exists | ✅ done |
| 11 | Retrain the emulator with `theta_v`/`dphi` live | Q15's gate: beat O25 on the realisation *and* `B_p` splits | ❌ **FAILED — structural; no weights shipped (§7.14)** |
| 12 | Cross-dataset check — PB24 model on L23 | overlap computed not assumed; out-of-domain fraction reported; promotion rule encoded as a **conditional** | ✅ done — **does not transfer** |
| 13 | ZTT `mu_d` vs HydroLight; the µ∞ question | `mu_d` pinned; **µ∞ cannot be refit from PB24** — Q17 option 3 closed (§7.16) | ✅ done |
| 14 | Promote `PhaseParams` to the ZTT backward-VSF form | existing tests pass **untouched**; `None` path bit-identical; new fields provably perturbed | ✅ done |
| 15 | Freeze the `forward` API | signature-pinning test | ✅ done — §8.0 |
| 16 | Notebook 6, PR review, hand-off | the M0–M4 rhythm | ⬜ |

### 7.2 The reference data — PB24, as measured

`$OS_COLOR/SD/v5`, 10 001 files, 28 GB, inspected and independently audited 2026-08-08.
5000 realisations in two spectral resolutions — `SD_OLCI_no_R_NNNN.nc` (12 OLCI bands,
400–753 nm, 1.3 GB total) and `SD_hyp_no_R_NNNN.nc` (451 bands, 350–800 nm at 1 nm, 27 GB)
— each on `theta_s`(10) × `phi`(13) × `theta`(10) = **1300 geometries**, θs to 87.75° and
θv to 87.5°. Files carry IOP *components* (`aph ag aNAP`, `bph bNAP`, `bbph bbNAP`, plus
water), both **`rrs` and `Rrs`**, and `Q`, `mu_d`, `mu_u`, `mu_tot`, seven K's and `R`. A
`.mat` sidecar gives 12 optical water classes per realisation (unbalanced: 84–1042).
Provenance: Jaime Pitarch, CNR-ISMAR, 2024-02/03.

Four measured properties that shape the milestone:

1. **The particle phase function varies.** `bbph/bph` is flat in λ within a file
   (max/min ≤ 1.0008) but takes a unique value per realisation across **0.0010–0.0358
   (~30×)**; `bbNAP/bNAP` spans 0.0100–0.0200. Bulk `B_p` spans **6.2×** across
   realisations against L23's 1.7×, and is *not* spectrally flat (median 3% within a file,
   up to 17%) because the phyto/NAP mix shifts with λ. One family throughout
   (Fournier-Forand), so this exercises the design §4.2 parameterization, not
   generalisation across VSF families.
2. **The IOP space is richer than its three labels.** `C`, `N`, `Y` do not determine the
   IOPs: `S_g`, `S_NAP`, `aNAP*(440)` (230×), `aph*(440)` (4.2×) and `bph(440)/C` (~370×)
   vary independently; only `Y` is exact (`= ag(440)` to 0.1%). Normalised `aph` shapes
   come from a finite library and are reused verbatim across files, so a held-out-
   realisation split does not fully separate shapes.
3. **The Lee-2002 surface transfer is nadir-only.** See §7.3.
4. **Tails and a defect.** `C` to 938 mg m⁻³, `Y` to 74.5 m⁻¹, max `rrs` 0.397; and `rrs`
   is **exactly 0** at θv = 87.5°, θs ∈ {70, 80}, λ ≥ 721 nm (float32 underflow; `Rrs` is
   non-zero there). `rrms` divides by truth, so this must be filtered or guarded. The Q14
   training window excludes those geometries, so the guard is needed only on the
   extrapolation set.

### 7.3 What the data changed about the plan

**A correctness bug the prototype could not have seen.** `conventions.Rrs_to_rrs` /
`rrs_to_Rrs` hard-code Lee (2002) `A_RRS = 0.52`, `B_RRS = 1.7`. Against PB24 that holds
at nadir (1.8% median) and fails progressively off-nadir — median |deviation| by view
angle, over all θs, φ and λ:

| θv | 0° | 10° | 20° | 30° | 40° | 50° | 60° | 70° | 80° | 87.5° |
|---|---|---|---|---|---|---|---|---|---|---|
| median | 1.8% | 1.9% | 2.6% | 6.2% | 13.0% | 24.7% | **45.7%** | 83.6% | 162% | 275% |

`Rrs/rrs` runs from 0.530 at nadir to 0.160 at θv = 87.5°; the median over all 1300
geometries is 15–24% and the worst case 23×. Measured twice, by two independent
implementations.

**Who this actually affects — a correction.** The first draft of the M5 sequence claimed
the map was "on the path of every model in the package". It is not, and the code says so:
`rrs_gordon` (`baselines.py:57`) and `rrs_ZTT` are the primitives and never touch it —
`Rrs_gordon` (`baselines.py:126`) and `Rrs_ZTT` (`ztt.py:939`) are above-water *wrappers*
— and task 11 reads PB24's tabulated `rrs`, so the emulator's targets bypass it too (the
`emulator.py:978` conversion exists because **L23** ships `Rrs` only). In the `rrs` scoring
path exactly one model is contaminated: **`rrs_o25`** (`baselines.py:302`), because O25
alone is defined in `Rrs`, and `fit_o25` fits there while everything is scored in `rrs`.
That still orders task 7 before task 9 — O25 is the benchmark, so its score must be clean
before anything is measured against it — but on one model's account, not five. Anything
reported in `Rrs` at an off-nadir angle is affected regardless of model.

**The phase-function axis stopped being blocked.** M4 recorded it as untestable without
commissioned runs (§6.2's `B_p` paragraph). Property 1 above makes a **held-out-`B_p`
split** constructible from data on disk, which is the single largest change to the
milestone's scope — and it demotes "commission HydroLight runs" from the critical path to
a stretch item that answers only the across-families question.

### 7.4 Decisions taken (Q10–Q15)

| Q | Decision | Consequence in the code |
|---|---|---|
| **Q10** | PB24 is the reference data (downloaded by JXP) | no external fetch; task 4 reads `$OS_COLOR/SD/v5` |
| **Q11** | Retrain with the view angles as **live features** | `cos_theta_v`/`cos_dphi` are already in `FEATURES` and constant in L23 — which is exactly why the domain check flags every off-nadir view; PB24 makes them live and the envelope widens from data, with no API change |
| **Q12** | OLCI files; a documented geometry-subsampling knob | explicit argument, never a hidden sample, and it reports what it dropped; the reader stays factored for the hyperspectral set |
| **Q13** | Keep both datasets; L23 becomes an independent held-out **dataset** | ships `files/emulator_pb24.npz` beside the L23 weights; `load_default()` unchanged, so every M4 number stays reproducible; promotion rule written before the numbers exist (task 12) |
| **Q14** | Train 0–70°; hold out 80–87.75° | `SUPPORTED_THETA_S` → `(0, 70)` plus a view-angle counterpart; the fallback will fire on real data for the first time (at M4 it triggered on 0 of 9960 samples, §6.3) |
| **Q15** | **Option 2** — gate on realisation *and* `B_p` splits | task 11 beats O25-refit-on-PB24 on both; the geometry split is **reported**, on the same reasoning that took M4's zenith split out of its gate. Task 8 is what makes this meaningful — beating a θs-only O25 off-nadir would measure our fitter |
| **Q17** | *open* — the hybrid's form cannot survive PB24 | ZTT returns non-physical `rrs` on 22.3% of PB24 (§7.12); a bounded *relative* correction cannot repair a sign-flipped backbone. Recommendation: restrict the envelope in `bb/a` now, refit µ∞ at task 13 |
| **Q16** | **Option 1** — train on a subsample | all 5000 realisations, subsampled geometries; row count and subsample factor stated in the artefact; `fit()` keeps "reproducible from the seed alone", which the bit-identical-weights gate depends on. Two factors trained and compared, so the choice is evidenced |

### 7.5 The sequence, and what is blocked

**3 → 4 → 5** are prerequisites (make the machinery dataset-agnostic, then load, then
split); **6** unblocks 7, 11 and 14 at once; **7 → 8 → 9** builds an honest benchmark
before any model is trained; **10 → 11 → 12** is the model work; **13 and 14** need only
the loader; **15 is last**, because 7, 10, 11 and 14 can each still move the signature it
freezes; 16 is the milestone rhythm. Ordering is by what each step *unlocks*.

Not scheduled, stated plainly:

- **Generalisation across VSF families** — blocked on commissioned HydroLight runs with a
  non-Fournier-Forand family, which nobody has ordered. **The one headline gap M5 will not
  close**; §6.10's item 4 will narrow rather than disappear.
- **The hyperspectral λ-interpolation check** — unblocked, deferred by Q12.
- **PR05** — implementable for the first time (PB24 spans its 4-D `(θs, θv, Δφ, γb)` LUT),
  but it earns its place only if O25 stops being the benchmark.
- **Task 11's gate** — provisional pending Q15; its **training-set size** pending Q16. The
  work is not blocked; the pass/fail line and the sample size are.

### 7.6 Task 3 as built — a second wavelength grid

`robust/rt/conventions.py` gained a grid concept; `robust/rt/types.py` follows it.

- **`WaveGrid`** (`name`, `wave`, `description`, with `n_wave`/`span`) and **`GRIDS`**:
  `"canonical"` (alias `"l23"`) → L23's 81 bands, `"olci"` → PB24's 12. **`wave_grid()`**
  resolves `None` / name / object, raising `KeyError` on an unknown name — a typo must not
  fall back to the canonical grid. **`grid_wave()`** is `canonical_wave()`'s grid-aware
  counterpart.
- **`check_wave(..., grid=None)`** checks *per grid* rather than loosening. OLCI bands
  against L23 still raise; so does a 12-band grid that is not quite OLCI's. The L23 grid is
  named `"canonical"` so the M0–M4 messages, and the tests matching on them, are unchanged
  — that regression was caught by an existing test, which is what it was for.
- **`IOPs.validate(..., grid=None)`** compares the trailing axis with *that grid's* band
  count instead of `conventions.N_WAVE` — the line that made the check L23-only.
- **`bb_w(..., mode=)`** — `"clamp"` (default, unchanged), `"extrapolate"`, `"raise"` —
  plus **`check_bb_w_range`** and **`BB_W_RANGE`**. On L23's grid the question could not
  arise, since the table's support *is* the grid; PB24's 753 nm band sits 3 nm past it,
  where the clamp reads **1.6% high**, growing to **23% at 800 nm**.
- **`BB_W_TAIL_EXPONENT = -4.140855`**, fitted here from `BB_W_L23` over 650–750 nm; it
  reproduces that tail to 2.2e-4 relative and a test re-derives it. Deliberately not the
  whole-range fit (−4.215) or Morel's molecular value (−4.32): the constant is used only to
  continue the red tail past 750 nm, so it is fitted to the red tail.
- `IOPs.from_total_bb` gained `bb_w_mode=` (default `"clamp"`, so identical numbers) and a
  note that a dataset tabulating its own `bb_w` — PB24 does — should use those values
  instead.

**Tests: +16** (`test_conventions.py`, `test_types.py`), suite **295 passed**, ruff clean.
Two were wrong on the first run and both were worth the correction: an off-grid band count
I asserted as 9 when it is 6, and a continuity check whose 1e-4 tolerance mistook the
function's own slope (0.11% over 0.2 nm at −4.14) for a discontinuity — it now checks the
value *and* the log-slope across the seam.

### 7.7 Task 4 as built — the PB24 loader

`robust/rt/data/pb24.py`, following `l23.py`'s *shape* but not its field list.

- **`PB24Batch`** carries `rrs` **and** `Rrs` — the reason being §7.3: `L23Batch`
  holds `Rrs` alone and derives `rrs` through the nadir Lee map, which is wrong
  off-nadir by a median 45.7% at θv = 60°. PB24 tabulates `rrs`, so scoring never
  touches that map. Also `Q`, `mu_d`/`mu_u`/`mu_tot` by default, the seven K's on
  request (`extras=`), `realisation`, optional `water_class`, and a `LoadReport`.
- **`RAW_FIELDS` is complete rather than minimal** (31 fields). The fixture is
  gated bit-identically, so a field added later invalidates every cache in
  existence; storing the lot once is cheaper than a second migration.
- **Angle selection** is `angles="window"` (Q14's 0–70° envelope, the default),
  `"shell"` (its complement, the extrapolation set) or `"all"`. Window = 832 of
  1300 geometries; shell = 468.
- **Subsampling is explicit and reported.** `geometry_stride` never defaults to
  anything but 1, and `LoadReport` records what each stage dropped.
- **`bb_w` is the file's own `bbw`**, never `conventions.bb_w` — that table is
  L23's water column. It also means the 753 nm band needs no extrapolation, so
  task 3's `mode=` is not on this path.
- **Fixture:** `robust/tests/files/pb24_small.npz`, 471 kB, realisations
  (1, 993, 2500) with all 1300 geometries each. 993 is there because it carries
  the only defect in the OLCI set.

**Two findings from building it.**

1. **The stride aliases.** Flattening `(theta_s, theta, phi)` in C order puts the
   13 azimuths innermost, so `geometry_stride=13` keeps exactly one azimuth and
   returns a batch with no BRDF variation at all — while looking entirely normal.
   `LoadReport.coverage` now records kept-vs-available distinct values per angle
   axis, `aliased_axes` names any casualties, and the loader warns. This matters
   directly for Q16: the sanctioned subsample must be checked for coverage, not
   just for size.
2. **The zero-`rrs` values are narrower than feared.** In the OLCI band set they
   occur only at 753 nm with θs = 80°, θv = 87.5° — two values in realisation 993
   across the sampled files, none at all inside the Q14 window. Dropping whole
   spectra therefore costs 22 good bands to exclude 2 bad values, which the report
   states so the trade is visible when task 6 gives `rrms` a mask.

**Measured scale**, which Q16's answer needs: 20 realisations × 832 geometries =
16 640 samples load in 0.6 s and 8 MB, so the full window is **~2 min and ~2 GB**
resident before training touches it.

**Tests: +28** (`test_pb24.py`), suite **323 passed**, ruff clean. One of them was
a tautology on first writing (`assert x == approx(0) or True`) — the same
"test that cannot fail" defect M4's review found four of; it now asserts that
`mu_d` takes exactly 8 distinct values in the window, one per solar zenith, which
a broadcast bug would collapse to 1 and a mis-indexed gather would inflate.

### 7.8 Task 5 as built — the three splits

`pb24.make_splits` returns a :class:`Splits` of ``{kind: (train, test)}`` masks plus a
:class:`SplitReport` each. `PB24Batch` gained `labels` (`C`, `N`, `Y` per sample); those
were already in `RAW_FIELDS`, so **the committed fixture did not change** and its
bit-identical gate still holds.

| kind | holds out | answers |
|---|---|---|
| `realisation` | random 20% of realisations, whole | M4's scene split, transplanted |
| `bp_band` | interior quantile band 0.4–0.6 of per-realisation mean `B_p` | **phase-function interpolation** — why M5 exists |
| `geometry` | the 80/87.75° shell | successor to M4's unseen-60°, the half the prototype lost |

**Why an interior band.** Holding out the top or the bottom of `B_p` would test
extrapolation, which is already the geometry split's job; conflating them would leave a
bad number unattributable. Train is therefore both tails, and `detail` carries the band
edges, the train tails and `n_train_inside_band` (0 by construction) so the separation is
inspectable rather than asserted.

**The confound, and its yardstick.** The gate required the `B_p` split to report the
confound it induces. It does — `test/train` ratio of median `C`, `N`, `Y`, `a`, `bb_p`,
`B_p` — but building it surfaced that **the ratios are uninterpretable against 1.0**:
PB24's label distributions are heavy-tailed, so a *random* hold-out of the same size moves
median chlorophyll across **[0.53, 1.90]** over 12 seeds at 600 realisations. Hence
`confound_reference`, which measures that band on the batch in hand. Only `B_p_mean` is
tight under randomness ([0.98, 1.08]), so it is the single entry where a departure is
immediately meaningful.

Two limits of the metric, stated because they are easy to misread:

- For `bp_band` the `B_p_mean` ratio reads ~1.0, since an interior band and its two tails
  share a median. The intended separation lives in `detail`, not in a ratio of medians.
- For `geometry` every confound entry is exactly 1.0 — correct, and a useful control: that
  split divides angles, not water bodies, so it moves no IOP statistic at all.

**A correction to §7.2 and to the M5 hand-off.** Both recorded that `B_p` correlates with
chlorophyll at **−0.65**. That figure is task 0's and it is for the *phytoplankton
component ratio* `bbph/bph`. For the **bulk `B_p`** the split actually uses, measured over
600 realisations, it is **−0.49** — a real confound, weaker than advertised. A test pins
it. Relatedly, `B_p`'s per-realisation span is **12.4×** over 600 realisations
(0.0025–0.0315), against the 6.2× measured from ~50 files at task 0.

**Tests: +13**, suite **336 passed**, ruff clean. One was a tautology on first writing
(`assert splits.seed == 8`, which cannot fail); it now sweeps ten seeds and requires the
draw to land on more than one realisation — the honest version of "the seed reaches the
draw", given that with three fixture realisations two seeds can agree by chance.

### 7.9 Task 6 as built — the validation toolkit

Three limits, each a reasonable single-dataset choice, each blocking an M5 gate. The
common thread: **all three were invisible on L23 and all three are load-bearing on PB24.**

**1. `gradient_report` now perturbs any field.** Variables are resolved against the
dataclasses themselves (`_containers`) and every container is rebuilt with
`dataclasses.replace`, so a field this module has never heard of survives the round trip
and is provably perturbed. `FD_STEPS` is unchanged as the default, `FD_STEPS_EXTRA` adds
`theta_v`, `dphi`, `bb_w`, and `default_steps()` **raises rather than inventing** a step
for an unknown quantity — a guessed step turns a gradient gate into a statement about step
size.

The old guard demanded *exactly* M2's four variables. That contract is **deliberately
changed**, and its test rewritten rather than patched: subsets and extra *known* variables
are now legal, because the implementation can honour them. What still raises is a name the
inputs cannot offer, including `wind` — a real field that is `None` here and would
otherwise have become `None + 1e-3`. The failure the old guard existed to prevent (an
extra key reporting 0.0, "perfect agreement", for a variable never perturbed) is now
structural rather than enforced.

**2. `rrms(..., where=)`, masked twice.** `truth` is replaced by 1.0 where the mask is
False *before* the division, not merely zeroed after. Masking afterwards leaves `0/0` in
the graph: `jnp.where` hides the NaN in the forward pass while reverse-mode differentiation
propagates it through the discarded branch — so the loss would look healthy and the
gradient would be NaN. Since `rrms` doubles as M3's training loss, that distinction is the
whole point, and a test asserts the asymmetry (masked gradient finite, unmasked not). A
wholly-masked group returns `nan`, so an empty selection announces itself.

**3. `group_rrms(..., expected=)`.** Iterating `np.unique(labels)` can only produce
non-empty groups, so this function was *incapable* of returning a short dict — which is
precisely why zipping its `.values()` against hard-coded headers was safe for L23's three
fixed zeniths. PB24 has eight solar zeniths and eight view angles, any of which a split or
a subsample may omit, and a missing group would shift every column to its left without
raising. Missing labels now return `nan`, and `design/py/run_validation.py` derives its
headers from the labels and indexes by name.

**Demonstrated, not asserted.** Reverting `gradient_report`'s closure to its pre-task-6
form fails both new regression tests; restoring it passes them. And the refactored
`run_validation.py` reproduces **every deterministic row** of the committed `metrics.md`
bit-for-bit (Gordon, ZTT and O25 on the per-zenith and per-`B_p` tables), so the change is
behaviour-preserving on L23 while being correct on PB24.

**Tests: +8**, suite **346 passed**, ruff clean.

### 7.10 Task 7 as built — the geometry-aware surface transfer

`conventions` gained `SurfaceTransfer`: `A` and `B` of the Lee form tabulated on PB24's
10 × 10 × 13 `(theta_s, theta_v, dphi)` grid and interpolated trilinearly.
`Rrs_to_rrs` / `rrs_to_Rrs` take keyword-only `geometry=` and `transfer=`; **omitting both
is the M0–M4 path and is bit-identical**, pinned by a test, and every M0–M4 call site was
verified still to pass neither. The table ships as `robust/rt/files/surface_pb24.npz`
(19 kB), regenerated by `design/py/fit_surface.py`.

**Held-out performance** — fitted on 320 realisations, scored on 80 it never saw:

| θv | 0° | 30° | 50° | 60° | 70° | 87.5° | window |
|---|---|---|---|---|---|---|---|
| nadir constants | 1.84% | 4.32% | 17.81% | **33.15%** | 65.63% | 238.88% | 6.81% |
| fitted table | 1.56% | 2.04% | 3.34% | **4.57%** | 6.43% | 14.57% | 2.05% |
| gain | 1.2× | 2.1× | 5.3× | **7.2×** | 10.2× | 16.4× | 3.3× |

**Four things the data decided, none of them guessable from the design.**

1. **All three angles earn their place.** The task said "`A(θv)` at minimum, testing
   whether θs and Δφ terms earn their place". They do: at θv = 60° the per-geometry `A`
   still spans 0.28–0.46 across solar zenith and azimuth, and a θv-only table leaves a
   median 3.4% error against the full one, up to 70%. Azimuth is irrelevant at nadir (by
   symmetry) and worth 27% at θv = 60°.
2. **`Q` does not.** Lee's `B = 1.7` is really `r̄·Q` with `Q` assumed ~3.5; PB24 tabulates
   the real `Q`, which spans 0.9–6.0 in the window. Refitting as `1 − r̄·Q·rrs` scores a
   median **1.71%** against **1.74%** for simply fitting `B` per geometry — no gain. That
   is fortunate as well as tidy: `forward` has no `Q` to offer at prediction time, so a
   `Q`-dependent transfer would have been unusable where it is needed.
3. **A table beats a smooth function, and the gate decided it.** A 10-term smooth fit in
   the angle cosines reaches 4.4× at θv = 60° — below the ≥5× the gate requires — while the
   table gives 7.2×. The cost is that a piecewise-linear table has kinks at its nodes, so
   the gradient check runs *between* them (M4 gotcha 4), which the gate anticipated.
4. **The residual does not go to zero.** Fitting both coefficients at every one of the 1300
   geometries still leaves a median 1.8% in the window. The Lee *form* is the floor, not
   the coefficients — worth stating, because the obvious next move (more coefficients)
   will not help.

**Structural confirmations from the same probe**, both worth keeping:

- `rrs × Q = R` to **0.00%** at every geometry in the files checked, so PB24's `rrs` is
  genuinely direction-dependent and is paired with `Rrs` in the *same* direction. The
  alternative reading — that `rrs` is tabulated at the Snell-refracted angle — was tested
  and refuted: pairing `Rrs(θv)` with `rrs` interpolated to the refracted angle flattens
  the ratio only partly (0.53 → 0.49 at 60°, still 0.30 at 87.5°), so refraction explains
  some of the fall-off and the Fresnel transmittance the rest.
- The fit is one `lstsq` per grid cell, so it has no seed, no learning rate and no stopping
  rule, and `fit_surface_transfer` **raises** if any cell has no samples rather than
  leaving a hole for the interpolator to cross in silence.

**Tests: +12**, suite **358 passed**, ruff clean.

### 7.11 Task 8 as built — O25 over the full geometry

`O25Table` holds the four coefficients on the whole `(theta_s, theta_v, dphi)` grid;
`Rrs_o25` dispatches on the type, so `O25_L23_REFIT` and the 1-D path are untouched.
`fit_o25_table` is the 3-D counterpart of `fit_o25` — still one `lstsq` per cell, still no
seed or stopping rule — and refuses a cell with fewer than 40 samples. The trilinear
machinery is `conventions.interp_geometry`, shared with §7.10's `SurfaceTransfer`: one
implementation, tested once.

**`fit_o25`'s `zeniths` default is gone.** It was `(0.0, 30.0, 60.0)` — right for L23,
silently wrong anywhere else, and on PB24 it would have fitted 3 of the 8 in-window
zeniths and interpolated across the other 5 without a word. It now derives the list from
the training data, and naming a subset that omits an angle the data contains **raises**.
The old test pinned the old default and was rewritten; a companion test pins that holding
a zenith out of the mask yields a table with fewer rows, so the clamp at evaluation time
is visible rather than accidental.

**Held-out result** (120 realisations, fitted on 96, scored on 24), rRMS in `rrs` through
the §7.10 transfer:

| θv | 0° | 20° | 40° | 60° | 70° | all |
|---|---|---|---|---|---|---|
| θs-only refit | 4.69% | 5.76% | 9.51% | 15.44% | 18.95% | 11.03% |
| 3-D refit | 2.58% | 3.15% | 6.01% | 9.30% | 10.97% | **6.59%** |
| gain | 1.82× | 1.83× | 1.58× | 1.66× | 1.73× | **1.67×** |

Three consequences.

1. **The extra axes earn their place**, and the gain is roughly *uniform* in view angle
   rather than concentrated off-nadir. The reason is that the zenith-only fit's error comes
   from pooling all 104 view geometries into four numbers per solar zenith, which damages
   nadir as much as 70°.
2. **This is the measurement that justifies the task order.** Scored through the *nadir*
   transfer instead, the same comparison reads 20.02% vs 18.89% — a **1.06×** gain. Built
   in the other order, task 8 would have concluded the extra axes did not matter, because
   the interface error swamps both models. A test pins the contrast.
3. **The benchmark M5 is gated against is far weaker than M4's.** O25 refit on PB24 scores
   **6.59%** where its L23 refit scored **0.69%**. PB24 spans 12× in `B_p` and O25 has no
   phase-function input at all, so this is M5's own axis appearing in the rival's score. It
   must always be quoted as O25's number *on this data*; setting 6.59% beside 0.69% would
   be comparing two datasets, not two models.

**On the direction of the bias, restated.** §7.2 recorded that PB24 favours O25 because it
is O25's calibration set. That holds for the *published* O25. For the O25 we can fit, the
zenith-only table was a handicap we imposed, and this task removes it — so any margin the
hybrid shows at task 11 is now earned rather than manufactured.

**Tests: +10**, suite **368 passed**, ruff clean.

### 7.12 Task 9 as built — the analytic models on PB24

`design/py/run_pb24_validation.py` writes `design/validation_pb24/` (metrics.md, three
CSVs, three figures). 200 realisations x 1300 geometries; O25 refit on **each split's own
training mask**, with task 8's full-geometry table and scored through task 7's transfer.

| model | realisation (held out) | `B_p` band (held out) | geometry (held out) | non-physical % |
|---|---|---|---|---|
| standard Gordon | 20.41 | 20.81 | 45.46 | 0.00 |
| ZTT backbone | *not a usable number — see below* | | | **22.32** |
| **O25, refit on PB24** | **6.53** | **6.02** | **22.46** | 0.00 |

The realisation and `B_p` splits are restricted to Q14's window; the geometry split carries
the shell. Unrestricted on an `angles="all"` batch they would mix in-window and shell
samples on both sides, so one number would have answered two questions and neither cleanly.

**PB24 is much harder than L23 for every model.** Gordon goes 7.21% → 20.41%, O25's refit
0.69% → 6.53%. The benchmark M5's gate is measured against is therefore ~10x weaker than
M4's, which is not a weaker benchmark — it is a harder dataset, and setting the two numbers
side by side would be comparing datasets and calling it models.

**ZTT is outside its validity domain, and this is the milestone's most consequential
finding so far.**

> **Corrected at task 11.** This section originally attributed the failure to
> `mu_infinity_tt2017` being evaluated beyond the `bb/a` range it was fitted on. That
> mechanism is real but it is **not the main cause**: of the non-physical predictions,
> only **1%** have `µ∞ ≤ 0`, while **68%** have **`psi_KLu(ψ) < 0`**. ZTT's `psi_KLu`
> parameterization goes negative for scattering angles **75.5°–108.7°** — near 90° — which
> flips the `(a/bb)(1 − cos θv · ψ_KLu / µ∞)` term and hence the whole denominator. L23
> fixes the view at nadir, so ψ never left the backscatter region and this could not fire;
> PB24's full BRDF sweeps ψ down to 75°. The failure is therefore **geometric, not
> IOP-driven**, which the stratification confirms: 3% of predictions are >100% wrong at
> nadir against **41% at θv = 60°**, and 5% at Δφ = 0° against **34% at Δφ = 180°**, while
> restricting `bb/a` to L23's range barely helps at all. §7.13 below is written on the
> corrected mechanism.

For the record, the `bb/a` excursion is real even though it is not the main cause: L23
spans `bb/a` up to **0.59**, PB24 reaches **3.54**, and 7.4% of its values lie beyond
anything L23 probed, where the fitted µ∞ polynomial reaches −0.027. Measured
consequences of the failure as a whole:

- **22.3% of ZTT's predicted `rrs` are ≤ 0** — non-physical, not merely wrong.
- Its **rRMS is not a stable statistic**: 7061–18972% over 50/100/200/400 realisations,
  determined by a handful of sign flips. Its stable statistics are the **median relative
  error (~11%**, against 5.9% on L23) and the **non-physical share (14–22%)**.
- Hence the table's **non-physical %** column. A model returning negative reflectance is
  out of domain rather than inaccurate, and an rRMS cannot express the difference — the
  same few samples that dominate the RMS also conceal how many there are.

**What it does to task 11.** The hybrid is `rrs_ZTT · (1 + δ)` with `|δ| ≤ 0.5`. No bounded
*relative* correction can turn a negative backbone into a positive reflectance, so on ~a
fifth of PB24 the hybrid's functional form has nothing to correct toward. Raised as
**Q17**, with four options and a recommendation (restrict the envelope in `bb/a` now, refit
µ∞ from PB24's tabulated `mu_u`/`mu_tot` at task 13).

**Gates.** Six artefact tests: the CSVs parse with the columns they promise (model names
contain commas), O25 is labelled a refit in every artefact, the view-angle headers are
derived from the same labels as the values (task 6's alignment fix), the wavelength CSV
covers the OLCI grid, the non-physical column reports ZTT and not the others, and the
markdown and CSV of one run are checked against each other row by row — the stale-figure
lesson, enforced.

**Tests: +6**, suite **374 passed**, ruff clean.

### 7.13 Task 10 as built — the envelope belongs to the model

`SUPPORTED_THETA_S` was a single module constant consulted by **every** emulator's domain
check. Correct with one model; unsafe with two, because widening it for a PB24-trained net
would have widened the shipped L23 net's envelope with it — and a 65° query against a model
trained to 60° would have become "in domain", which is precisely the seed-dependent regime
M4 measured and warned about (§5.5).

So the envelope is now an **`Envelope`** — `theta_s`, `theta_v`, `dphi`, each a range or
`None` meaning "judge by the trained range" — carried as a **static field of `Emulator`**
and written into the weights file. `_effective_domain` generalises from the solar zenith to
all three angles; `fit()` takes `envelope=`; `SUPPORTED_THETA_S` survives as the default's
value, so `test_validation.py`'s pin keeps its intent.

Four details worth recording:

1. **Three call states have to stay distinguishable**, which is why a sentinel exists:
   unset means *this model's* envelope, `None` keeps its M4 meaning of "the trained range",
   and a `(lo, hi)` tuple keeps its M4 meaning of "the solar zenith specifically". Without
   the sentinel, "not passed" and "passed `None`" would be the same call and one of the two
   established meanings would have been lost silently.
2. **`cos` is not monotonic over azimuth.** Bounding `cos_dphi` by its endpoints would
   narrow an envelope that spans 0° or 180°, where the interior attains ±1. `_cos_bounds`
   walks the interior extrema; narrowing a domain check silently is the one direction of
   error it must not make.
3. **Old weight files keep their meaning.** The committed L23 `.npz` predates the envelope
   key, so `load` gives it the default — which is what it was evaluated under. Pinned by a
   test that asserts the key is *absent* from the shipped file.
4. **`None` is encoded as NaN** in the `.npz`, because npz has no null and any sentinel
   angle would be indistinguishable from a real one.

**A defect this task introduced, caught by its own gate.** `out_of_domain_mask`'s default
was left at the old constant while `out_of_domain`'s moved to the sentinel, so the traceable
predicate and the reported one disagreed — the exact PR #12 divergence, reintroduced. The
test written for that defect class failed on the first run and named the disagreement.

**Tests: +10**, suite **384 passed**, ruff clean.

### 7.14 Task 11 as built — the retrain, and why its gate fails

**No weights were shipped.** `design/py/train_emulator_pb24.py` refuses to write when the
gate fails, because a committed `.npz` is a claim that the model is usable and `load()`
carries no hint of provenance.

**The result** (300 realisations, 3 seeds, held out, scored on the **full** test set;
`fit_pb24` + per-axis geometry stride):

| stride | δ_max | split | hybrid | **oracle** | O25 | gate |
|---|---|---|---|---|---|---|
| (1,2,2) | 0.5 | realisation | 5484.65% | **5324.10%** | 5.43% | FAIL |
| (1,2,2) | 0.5 | `B_p` band | 4052.36% | **3954.03%** | 5.37% | FAIL |
| (1,2,2) | 1.0 | realisation | 892.21% | **34.89%** | 5.43% | FAIL |
| (1,2,2) | 1.0 | `B_p` band | 694.05% | **35.14%** | 5.37% | FAIL |
| (1,3,4) | 0.5 | realisation | 7872.24% | **7785.27%** | 5.43% | FAIL |

**The oracle column is the whole finding.** It is `rrs_ZTT · (1 + clip(truth/rrs_ZTT − 1,
−δ, δ))` — the correction chosen *with the truth in hand*. No emulator can beat it. At the
shipped `δ_max = 0.5` the trained hybrid is within **3%** of it, so the network is not the
limitation: **the functional form is**. And doubling the bound is not a fix either — the
oracle only reaches 34.89%, still 6× worse than O25, because a negative backbone requires
`1 + δ < 0` and **no bounded relative correction of any size can repair it**.

That closes off Q17 options 1 and 2 as sufficient answers. Either the backbone is fixed
(option 3 — refit µ∞ *and* `psi_KLu`, task 13) or the correction stops being multiplicative
(option 4). JXP kept option 4 live; this is the measurement that makes it necessary rather
than possible.

**Q16, answered by measurement rather than assumption.** Two geometry strides were trained
and compared: the denser (1,2,2) scores 5484% against (1,3,4)'s 7872%, so density helps
even in this regime. A **per-axis** stride had to be added to the loader first: a flat
stride over the flattened geometry list does not preserve the grid's product structure, so
`fit_o25_table` found 713 of 832 cells empty and refused — the gate's own rival could not be
fitted on the batch the hybrid was trained on.

**Three defects an adversarial audit found in the first version, all fixed and pinned:**

1. **A leak.** The emulator was fit once on the realisation split and scored on the `B_p`
   split as well — of which **75% of held-out realisations sit in that training set**. The
   `bp_band` number would have been training error compared against an honestly refit
   rival. `fit_pb24` takes `kind` for exactly this reason and is now called with it.
2. **A comparison that flattered us.** Both models were scored only where the backbone is
   physical. That excludes the samples only *our* form cannot represent, and since the
   cause is geometric it narrows the geometry range with them. Measured: **O25 scores
   9.46% on precisely the samples the restriction would drop**, so the exclusion was
   self-flattery. Everything is now scored on the full test set; the restricted view
   survives as a labelled diagnostic.
3. **A handicap on the rival.** O25 was fitted in `Rrs` and converted to `rrs`, paying the
   surface transfer's residual that the emulator never pays. Fitted directly in `rrs` it
   scores **5.43%** against **5.74%**. Both are reported.

A fourth, noted and not yet acted on: `make_splits(seed=23)` permutes whatever realisation
set it is given, so **"seed 23" does not name one partition** — the shipped surface transfer
was fitted on a 400-realisation split whose train side overlaps this gate's 800-realisation
test side. It sits only in O25's path, so it favours the rival and a FAIL is conservative,
but the hazard is repo-wide and belongs in task 16's report.

**Tests: +14**, suite **392 passed**, ruff clean.

### 7.15 Task 12 as built — the cross-dataset check

`design/py/cross_dataset.py`: train on PB24, score on L23 without refitting. Task 11's
failure sharpened the question. On PB24 the backbone is non-physical on ~20% of samples; on
L23 it is healthy (5.93%). So L23 discriminates between an emulator that learned the
residual physics and one that learned something local to its training set.

**Result** — 150 realisations, 3000 steps, seed 23, L23's held-out scenes, rRMS in `rrs`:

| model | all 81 bands | overlap (71 bands) | 350–395 nm |
|---|---|---|---|
| ZTT backbone | 5.93% | 5.99% | 5.54% |
| hybrid, **PB24**-trained | **27.01%** | 28.56% | 10.99% |
| hybrid, L23-trained (shipped) | 0.30% | 0.28% | 0.43% |

**It does not transfer.** The PB24-trained correction makes L23 four to five times worse
than applying no correction at all. Under Q13's promotion rule — fixed before any number
existed — `load_default()` therefore stays with the L23 model, and the test encodes that
as a conditional evaluated from both sides rather than as a pin on today's answer.

**A claim corrected before it was written down.** The obvious reading is "it learned to
compensate for a broken backbone". That asserts a structure, so it was measured: the
correlation between the PB24 model's correction and what L23's backbone actually needs is
**−0.028**, against **+0.999** for the L23-trained model on the same data. There is no
relationship — it applies a median **+21.6%** where the backbone needs **+2.4%**. The
supportable claim is the weaker one: **nothing transferable was learned**, and the
compensation story remains a hypothesis rather than a finding.

**Nor is it plain feature extrapolation.** Only `wave_nm` leaves the trained range (12.3%
of values, L23's ten bands below 400 nm); every IOP and geometry feature sits inside PB24's
domain, and restricting to the 71 overlapping bands leaves the number essentially unchanged
(28.56%). So "it was asked outside its box" does not explain it either.

**Both caveats travel with the number**: the model flags **100%** of L23 as out of domain
(the `wave_nm` breach is enough on its own, since `out_of_domain_mask` fires when any
feature at any wavelength breaches), and the grids share only 71 of 81 bands, with the rest
reported separately.

**Tests: +4** — the band overlap is computed and asserted non-empty; the transfer failure is
pinned together with the out-of-domain flag, so the number can never be quoted as if the
model were being used properly; and the promotion rule is a conditional that computes both
sides.

### 7.16 Task 13 as built — ZTT's internals, and the caveat that cannot be closed

Three questions were asked of PB24's tabulated AOPs. The answers explain the milestone's
central failure and close one of Q17's options.

**1. `mu_d` is ~4× worse than advertised, worst where the sun is low, and goes negative.**
Against PB24's tabulated `mu_d`, over all 5000 realisations: median **4.17%**, and the
paper's <1% claim (§2.7) is violated at *every* tabulated zenith, even at its best. Per
solar zenith: 1.6% at 40°, 5.4% at 50°, 9.2% at 60°, **14.1% at 70–80°**. (The 2.2% at
87.75° is a sign crossing of the error, not recovered skill.)

More seriously, **ZTT's `mu_d` reaches −1.020** — an average cosine of the downwelling
field cannot be negative — with 550 values ≤ 0 across the release. Every value below 0.3
occurs at **`bb/a` > 1.36**, far outside `Md_star`'s fitted range of 1e-4…1e-1; **36% of
PB24's (realisation, band) pairs exceed 0.1**, and the release reaches `bb/a` = 20.1.
Restricted to `bb/a` ≤ 0.1 the minimum is a physical 0.666. So this is the same story as
`F(ψ)` on a second axis: a fitted expression evaluated far outside its range. Pinned by a
test.

**2. The backbone's collapse is the published model's own stated validity domain.** This
supersedes §7.12's first attribution (`bb/a`) and completes task 11's correction:

- `F_psi` is a quartic in the in-water scattering angle whose docstring already records the
  paper's fitted range — **ψ ≳ 134°**, ">95% of the angles a polar orbiter sees".
- `Ψ_KLu = 1 + F(ψ)` **crosses zero at ψ = 110.4°** and is negative for everything below,
  which flips the sign of the `(a/bb)(1 − cos θv · Ψ_KLu / µ∞)` term and hence of the whole
  denominator.
- In PB24's **sanctioned Q14 window, 42% of geometries have ψ < 134°** (outside the fitted
  range) and **16% have ψ < 110.4°** (actually sign-flipped); the full grid reaches
  **ψ = 44.3°**, where `Ψ_KLu` is **−60.7**. The two percentages answer different
  questions and both matter: the first is how much of the window is extrapolation, the
  second is how much of it is structurally broken.
- **L23's minimum ψ is above 134°.** Nadir viewing pins the scattering angle near
  backscatter, so the Week-1 prototype could not have discovered this at any level of care.

So ZTT is not mis-transcribed here and not mis-used by accident: it is being evaluated
tens of degrees outside the range its authors fitted, on nearly half of the geometry M5
sanctioned.

**3. Q17's option 3 is closed — µ∞ cannot be *properly* refit from PB24.** µ∞ is the
*asymptotic* mean cosine, `µ∞ = a/K∞`, and `K∞` is by definition independent of the
incident field. PB24 tabulates `Kd`, `Ku`, `Ko`, `Kod`, `Kou`, `Knet`, `KLu` — and **every
one varies by a median ~1.4× across solar zenith** (39–47% of its mean over 0–80°), so none
is `K∞`. These are *surface* K's; the release contains no asymptotic quantity.

The proxies rank as that predicts: **`a/Kd` is the only defensible one** — 100% inside
(0, 1], because `Kd ≥ a` structurally, median 0.721, and closest to TT2017 (median
difference 13.7%) — while `a/Ko` leaves (0,1] on 14% of samples and `a/Kou` on 34%. But
`a/Kd` is a *downwelling surface* attenuation standing in for an asymptotic quantity, and
it carries the corresponding bias (0.721 against TT2017's 0.797). Adopting it would replace
a published parameterization with a proxy of our own whose error we could not characterise
— which is the thing §7.16's own gate was written to avoid. **So the standing Equation-(8)
caveat cannot be closed with PB24**, and a test pins the reason, written so it fails if any
K turns out to be θs-independent after all.

**A cross-check worth keeping**, because it separates "the model disagrees" from "the data
disagrees": Gershun's law, `a/Knet = mu_tot`, holds in PB24 to a median ratio of **0.9999**
(5–95%: 0.9964–1.0012). The tabulated K's, `a`, and the average cosines are mutually
consistent, so every disagreement above is ZTT's, not PB24's.

**Two out-of-domain axes, compounding.** The denominator term
`(a/bb)(1 − cos θv · Ψ_KLu / µ∞)` can change sign through *either* factor: the geometry (ψ
below 110.4°) or the IOPs (`bb/a` > 0.1 breaks both `Md_star` and the TT2017 µ∞, each of
which goes negative on this data). A third, milder extrapolation: `eta_bb` spans 3.9e-6 to
0.994 while TT2017's lowest knot is 0.0098, so the interpolation clamps below it.

**The consequence for the milestone.** Q17's answer made option 3 "task 13's output". It is
unavailable — and even if it were available it would address the wrong term, since task 11
measured µ∞ at 1% of the non-physical predictions against `psi_KLu`'s 68%. Refitting `F(ψ)`
is no better placed: it is *defined* as `K_Lu/K∞ − 1`, so it needs `K∞` as well. **Neither
of ZTT's two failing internals can be repaired from PB24.** What remains is Q17 option 4
(a correction form that does not multiply the backbone), a different backbone, or reference
data that tabulates an asymptotic K.

**Tests: +5**, suite **400 passed**, ruff clean.

### 7.17 Task 14 as built — the backward-VSF axis

`PhaseParams` gained two optional fields, and `robust/rt/ztt.py` gained
`P_bb_from_phase(phase_params, psi)` which `rrs_ZTT` calls when no explicit `P_bb` is
supplied:

```
Pbb(psi) = beta_tilde_pi * S_ST(psi)/S_ST(180) * (psi/180)**(-backward_slope)
```

- **`beta_tilde_pi`** — the `β̃(π)` design §4.2 names: `Pbb(180°) = βp(180°)/bb_p`, sr⁻¹.
  It *rescales* the shape, so passing Sullivan's own 0.153 reproduces the fixed shape.
- **`backward_slope`** — the second parameter: a dimensionless tilt across the backward
  hemisphere, pivoting at 180° so the two are independent. Zero leaves Sullivan's angular
  dependence untouched.
- Either field `None` takes its neutral value, so a `PhaseParams(B_p=...)` gives exactly
  `P_bb_sullivan(psi)`.

**The gate, clause by clause.** All 400 pre-existing tests passed **untouched** and **no
call site changed** — the M1 design decision (§3.2) to make the phase function a container
rather than a bare array, vindicated at the moment it was meant to be. `None` is
bit-identical, asserted with `assert_array_equal` rather than a tolerance because that is
an exact claim. And the fields are gradient-checked through task 6's extended
`gradient_report`, which is precisely why task 6 came first: the pre-M5 closure rebuilt
`PhaseParams(B_p=...)` and would have discarded these fields silently, certifying the model
at the wrong phase function while reporting a flawless 0.0.

**A silent-broadcast bug, found while writing the tests.** `rrs_ZTT` hands ψ in as
`(n_sample, 1)` so it spreads across wavelength; a per-sample parameter arrives as
`(n_sample,)`; and `(n, 1) * (n,)` broadcasts to **`(n, n)`** rather than raising. It fails
loudly only when `n_sample != n_wave`, so a test written with a square batch would have
passed while the model computed something else entirely. `P_bb_from_phase` now aligns
trailing axes explicitly, and a test exercises both a square and a non-square batch.

**Not calibrated, and the docstrings say so.** No dataset here constrains these fields:
PB24 prescribes its phase functions and does not tabulate `βp(ψ)` (§7.2). The power law is
the simplest smooth, differentiable, pivot-neutral one-parameter tilt — a choice, not a
measurement. The fields are an axis to sweep and an interface for the inversion to build
against; any result that varies them must report the values used and that they are inputs.

**Tests: +7**, suite **407 passed**, ruff clean.

### 7.18 What an adversarial review of the sequence found

The first draft of §7.5 was reviewed against the source by an independent agent instructed
to find what it got wrong. It found enough to justify a second draft, and the findings are
kept here because they are all of the same kind: **the M0–M4 machinery quietly assumes L23
is the only dataset**, and the plan had assumed it was general.

1. **The cross-dataset check could not have passed.** L23 spans 350–750 nm; PB24 OLCI
   spans 400–753. `wave_nm` is a live feature, the domain is the training min/max, and
   `out_of_domain_mask` flags a sample if **any** feature at **any** λ breaches
   (`emulator.py:704`). 350 nm sits 14% of the span below the boundary against
   `DOMAIN_TOL = 0.01`, so **every L23 sample** would be flagged and — under the default
   `on_out_of_domain="ztt"` — the "cross-dataset number" would have been the bare backbone
   scored on 100% of L23. Now task 12: score the overlap, report the flagged fraction,
   treat 350–395 nm as the extrapolation it is.
2. **The O25 benchmark would have been a straw man.** `fit_o25` groups by solar zenith
   only (`baselines.py:374`) and `o25_coefficients` interpolates a 1-D table in θs
   (`baselines.py:202`) — there is no view-angle axis. On PB24 that averages 104 view
   geometries into four coefficients per zenith, and the default
   `zeniths=(0.0, 30.0, 60.0)` (`baselines.py:313`) **succeeds silently** while using 3 of
   the 8 in-window zeniths. This also **reverses** §7.2's framing: the *published* O25 is
   favoured by its own calibration set, but the O25 we can currently fit is handicapped
   off-nadir, so "beat O25 on PB24" could have been won on the fitter's limitations. Now
   task 8, before the benchmark.
3. **Three gates could not be expressed.** `gradient_report` raises unless the perturbed
   set is exactly `{a, bb_p, B_p, theta_s}` (`validation.py:317`) — a guard added at M4 for
   good reason — so no gradient gate on `theta_v`, `dphi` or new `PhaseParams` fields was
   possible; and `scalar()` rebuilds `PhaseParams(B_p=...)` (`validation.py:337`), silently
   dropping any other field, which would certify a task-14 model at the wrong phase
   function with no symptom. Now task 6, upstream of 7, 11 and 14.
4. **The envelope is one package-wide constant.** `SUPPORTED_THETA_S` (`emulator.py:201`)
   is the default for *every* emulator's domain check, with no view-angle counterpart.
   Widening it for the PB24 model would have widened the shipped L23 model's envelope too.
   Now task 10, and it must land before the API freeze.
5. **The loader's field list was too small.** `L23Batch` carries `Rrs` alone
   (`l23.py:139`); tasks 7, 9 and 13 need `rrs`, `Q` and the µ's. Because the cache is
   gated bit-identically, adding fields later invalidates the fixture — so the field list
   is a task-4 decision, not an afterthought.
6. **Task 13's promise exceeded the code.** `ztt.py` has `mu_d`, `mu_infinity` and
   `mu_infinity_tt2017` — no `mu_u`, `mu_tot` or `Q` — and PB24's `mu_u`/`mu_tot` are
   near-surface AOPs while µ∞ is asymptotic. Only `mu_d` is directly comparable; §7.2's
   consequence 3 ("the Eq-(8) substitution becomes directly measurable") is **narrowed**
   to that, plus an honest statement about µ∞.
7. **Two gates would have passed vacuously.** The zero-`rrs` filter removes nothing inside
   the Q14 window, so asserting its count there proves nothing; and the geometry split's
   test set is empty on a default load, so "contains only θ ≥ 80°" is true of the empty
   set. Both moved onto the shell load, with non-emptiness asserted.
8. **A stated dependency was wrong**, and one doc claim was: task 7's gate needs the
   held-out split from task 5, not just the loader; and `out_of_domain` /
   `out_of_domain_mask` are `Emulator` **methods**, not module exports, contrary to the
   API list in the M5 hand-off (verified: `hasattr(emulator, "out_of_domain")` is `False`).

The scale problem the review raised — `fit()` is full-batch and PB24 is 83× L23 — became
**Q16**.

---

## 8. Cross-cutting conventions (as implemented)

### 8.0 The frozen `forward` API (M5 task 15)

`robust.rt.forward` is the shared engine for training-data generation and for the
separately designed inversion. Both need it to stop moving, so as of M5 task 15 its call
surface is **frozen and pinned by tests** (`robust/tests/test_env.py`, "M5 task 15"):

```python
forward(iops, phase_params, geometry, wave=None, mode="hybrid", *,
        emulator=None, check_domain=True, on_out_of_domain="warn") -> Rrs
```

Frozen means these four things are asserted, and a change to any of them fails the suite:

1. **The signature** — parameter names, order, kind (positional-or-keyword vs
   keyword-only), and defaults.
2. **The pytree field order** of `IOPs`, `PhaseParams`, `Geometry`. Order is part of the
   API, not an implementation detail: these are registered pytrees, so `tree_flatten`'s
   leaf order is observable and anything that zips leaves would silently misalign.
3. **That every optional container field defaults to `None`** — which is what makes
   appending one safe.
4. **The enumerations the signature refers to**, `MODES` and `OUT_OF_DOMAIN_POLICIES`.

**Permitted — but still a deliberate act.** The signature test asserts *exact* equality,
so even a permitted change fails it until `FROZEN_FORWARD` is edited in the same commit.
That is the point: the freeze does not decide what may change, it decides that nothing
changes by accident.

- Appending a **keyword-only** argument whose default preserves existing behaviour.
- Appending a field to `IOPs` / `PhaseParams` / `Geometry` that **defaults to `None`** and
  changes nothing when left unset. M1 designed for this (§3.2) and M5 task 14 did it —
  400 tests passed untouched and no call site changed (§7.17).
- Adding new *functions* alongside `forward`, or new optional parameters to them.
- Changing anything private (leading underscore), or any docstring.

**Forbidden**

- Renaming, reordering, or removing a parameter or a container field.
- Changing a parameter from positional-or-keyword to keyword-only, or the reverse.
- Changing what an existing default *does* — including indirectly. `emulator=None`
  resolves through `emulator.load_default()`, and `on_out_of_domain` consults the
  emulator's own `Envelope` (§7.13); repointing either changes `forward`'s numbers without
  touching its signature, which is why Q13's promotion rule is itself a test (§7.15).
- Changing the **return convention**: `forward` returns above-water `Rrs`, and the
  subsurface counterpart is the separately named `hybrid.rrs_forward`. Scoring happens in
  `rrs` (design §6), so the two must never be confused.

**What an audit of the freeze found, and what closed it.** A signature pin is worth less
than it looks, because the ways a downstream caller actually gets hurt mostly leave the
signature alone. Four gaps were found and closed:

1. **The numbers were unpinned.** `emulator=None` resolves through `load_default()`, and
   Q13's promotion rule explicitly *permits* that file to change (§7.15) — so training data
   generated today need not match a regeneration tomorrow, with every test green. Now the
   shipped weights are pinned **by SHA-256 digest** and `forward`'s output **by value** on
   the committed fixture. Changing the default model stays allowed; it becomes a deliberate
   edit with a diff.
2. **The `Rrs` convention could have slipped off-nadir.** Every hybrid test runs on L23,
   which is nadir-only, so wiring §7.10's geometry-aware transfer into `forward` *for
   off-nadir geometries only* would have passed the entire suite while changing every
   number a multi-angular caller sees. Now pinned at an off-nadir geometry.
3. **`rrs_forward` was not frozen**, though scoring and training both happen in `rrs`
   (design §6) — arguably making it the more load-bearing surface of the two. Now frozen
   against the same tuple.
4. **The `Emulator` pytree contract was untested.** `config` and `envelope` are static and
   the arrays are leaves, and the inversion is told to rely on that — but every test in the
   suite *closed over* an emulator rather than passing it through `jit`/`grad`, and
   closures are not flattened. Dropping the `static` metadata would have stayed green until
   an inversion tried to differentiate w.r.t. the weights. Now exercised as an argument
   across both transforms.

**What the freeze still does not cover, deliberately.** Dtype beyond the pinned float32
default, batch-rank conventions past two axes, and the `features()` trailing-axis rule
(`B_p` reads as a spectrum iff its last axis is `n_wave`, so a batch of exactly 81 samples
on the 81-band grid is ambiguous — documented in `emulator.py`, untested). These are noted
rather than fixed; the first two are cheap to add when a caller needs them, the third wants
an API change rather than a test.

### 8.1 Other cross-cutting conventions

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
    hybrid.py          ✅ M3  forward()/rrs_forward(), MODES, DomainWarning,
                             on_out_of_domain policy (M4)
    validation.py      ✅ M4  rrms(); per-λ/zenith/B_p, throughput, grads
    baselines.py       ✅ M4  standard Gordon + O25 refit (PR05: see §6.4)
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
  run_validation.py          ✅ M4 the §6 protocol; writes design/validation/
design/validation/
  metrics.md                 ✅ M4 every model x split, per zenith, per B_p bin
  metrics.csv                ✅ M4 the same table, machine-readable
  rrms_per_wavelength.csv    ✅ M4 the per-λ ladder for every model
  rrms_per_wavelength.png    ✅ M4 that ladder as a figure
  unseen_zenith.png          ✅ M4 the unseen-60° comparison, with the seed spread

notebooks/RT/
  rt_elastic_coding_1.ipynb  ✅ M0 explainer (executed, 2 figures)
  rt_elastic_coding_2.ipynb  ✅ M1 explainer (executed, 3 figures)
  rt_elastic_coding_3.ipynb  ✅ M2 explainer (executed, 3 figures)
  rt_elastic_coding_4.ipynb  ✅ M3 explainer (executed, 6 figures)
  rt_elastic_coding_5.ipynb  ✅ M4 explainer (executed, 3 figures)

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

**Suite size at M3.** `pytest -q` → **227 passed** with `$OS_COLOR`; **206
passed, 21 skipped** without it, which is what CI sees. The suite was 171 at
M2's close.

**Badge** in `README.md`. It will read "no status" until `ci.yml` reaches the
default branch — the workflow file has to exist on `main` for the badge's default
view to resolve.

*Living document; updated at the close of each milestone.*
