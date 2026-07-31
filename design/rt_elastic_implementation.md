# Elastic RT Implementation Record

**Version:** 0.9
**Date:** 2026-08-01
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

**Verification (current).** `pytest -q` → **116 passed** in ~4.5 s (`ocean14`);
with `$OS_COLOR` unset, **99 passed + 17 skipped** — the loader itself is exercised
without the dataset, against a committed 50-scene fixture. `ruff check robust/` and `ruff
format --check robust/` → clean. The suite is green both with and without the L23
reference data on disk (missing data skips, never fails). Both notebooks in
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

`pytest -q` → **116 passed** (12 M0 + 27 conventions + 32 types + 45 L23). With
`$OS_COLOR` unset: **99 passed, 17 skipped** — every skip is a full-release
`needs_l23` test; the loader, splits, and `select` all still run in CI against the
committed fixture. `ruff check robust/` and `ruff format --check robust/` clean.

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
    types.py           ✅ M1  IOPs / PhaseParams / Geometry pytrees
    data/
      __init__.py      ✅ re-exports l23
      l23.py           ✅ M1  L23 elastic batches + seeded splits
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
    test_types.py      ✅ 32 tests — M1 task 2
    test_l23.py        ✅ 45 tests — M1 task 3 (+ cached-fixture layer)
    files/l23_small.npz ✅ 213 kB, 50 scenes x 3 zeniths (raw fields)

notebooks/RT/
  rt_elastic_coding_1.ipynb  ✅ M0 explainer (executed, 2 figures)
  rt_elastic_coding_2.ipynb  ✅ M1 explainer (executed, 3 figures)

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
