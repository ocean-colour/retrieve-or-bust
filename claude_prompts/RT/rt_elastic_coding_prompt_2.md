# Elastic RT Coding — Prompt 2 (M1: Data & conventions)

## Goals

Implement **Milestone M1**: the data layer and shared conventions — the `Rrs↔rrs`
convention, the IOP/phase/geometry types, and a one-call **L23 loader** that returns JAX
arrays with the `B_p` phase parameter and the held-out splits. This is the foundation
every later milestone consumes.

M1 is where the package stops being a scaffold: three of M0's stub modules
(`conventions.py`, `types.py`, `data/l23.py`) get real bodies, and the first numbers
enter the repo.

## Claude

### Skills

Consider using `.claude/skills/` (`code-review`, `verify`) as helpful. If a figure is
involved, the `dataviz` skill applies (M0's notebook established the figure conventions —
see *Status entering M1*).

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements* (git by JXP; `ocean14`;
CPU-only JAX; reuse `ocpy`/`bing`; BING test conventions; pytest-gated; Fable; log).

Two additions M0 settled:

- **Ask in Q&A.** Judgment calls that are JXP's to make go in the *Q&A* section of this
  doc as a numbered question, stated so work can continue without an answer. Check for
  answers before the next task.
- **Each milestone ships a notebook** (task 4 below) — the pattern M0 set.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M1.
- **Design** — `design/rt_elastic_model.md` §3 (interface/data model), §4.1 (reference
  data), §4.2 (`B_p`), and the A=0.52, B=1.7 convention.
- **Implementation record** — `design/rt_elastic_implementation.md` (currently **v0.4**;
  §2 is M0 as built, §8 the cross-cutting conventions). Update at close: bump to v0.5,
  add a §3 for M1 with *Modules added / Tests / Results / Notebook* subsections, flip M1
  to ✅ in §1, and refresh the module index.
- **M0's notebook** — `notebooks/RT/rt_elastic_coding_1.ipynb`, for the house figure
  style and what has already been explained (do not re-explain it).
- **L23 loader** — `ocpy.hydrolight.loisel23`:
  - `load_ds(X, Y)` → `Hydrolight{X}{Y:02d}.nc`; use **X=1** (elastic) with
    **Y∈{0,30,60}** (solar zenith 0°/30°/60°).
  - **Resolve the directory through `ocpy`, never a hardcoded path**: `loisel23.l23_path`
    is `$OS_COLOR/Loisel2023`. On this laptop `$OS_COLOR` =
    `/Users/xavier/Projects/Oceanography/data/Color/`, so `l23_path` =
    `/Users/xavier/Projects/Oceanography/data/Color/Loisel2023`. (An older draft of this
    doc said `$OS_COLOR_DATA/…`, which is **unset**; `/Users/xavier/data/Color/Loisel2023`
    is the same directory reached through a symlink, not a second copy.)
  - Dataset anatomy, verified in M0: dims **`IOP_Scenario` = 3320** × **`Lambda` = 81**
    (350–750 nm, 5 nm), coordinate `Lambda`; variables `Rrs`, `Ed_0+`, `Lw`, `Lu_0+`,
    `a`, `anw`, `aph`, `ag`, `ad`, `b`, `bnw`, `bph`, `bd`, `bb`, `bbnw`, `bbph`, `bbd`.
    All three elastic files are on disk.

## Status entering M1

M0 is complete (commit `ccbc0cc`). What M1 inherits — build on it rather than rebuilding:

**The stubs to fill.** `conventions.py`, `types.py`, and `data/l23.py` are
docstring-only, and each docstring already lists its planned contents from the design.
Fill the bodies and keep those docstrings current — they are the module-level
documentation, not scaffolding to delete.

**Test machinery that already exists** in `robust/tests/conftest.py`:

- `needs_l23` — the skip marker for data-dependent tests, plus `l23_available()` and
  `L23_ELASTIC_FILES`. **Use it** rather than writing new skip logic; the suite must stay
  green on a machine with no `$OS_COLOR` mount (verified in M0).
- `jax_x64` — a fixture enabling float64 for one test and restoring the flag after.
- `robust/tests/files/` — empty, `.gitkeep`-ed, and intended for a small cached L23 batch
  so the data tests do not re-read the ~17 MB netCDFs. M1 is the milestone that uses it.

**Environment.** jax/jaxlib 0.11.0, flax 0.12.8, optax 0.2.8, jaxtyping 0.3.11, on
Python 3.14.6, CPU backend. `jax.experimental.enable_x64` **does not exist** in JAX 0.11
— use the `jax_x64` fixture.

**Two gotchas M0 paid for:**

1. **`robust` may not be pip-installed** in `ocean14`. `pip install -e .` was in fact
   *broken* until CI setup exposed why (`setup.py` declared an illegal `provides`
   value; now fixed), so nothing was ever installed and `pytest` has only ever been
   run **from the repo root**. Keep doing that, and keep the `sys.path` bootstrap in
   the notebook (M0's notebook shows the pattern) so it works either way.
2. **JAX defaults to float32.** A "round-trips to 1e-6" test has barely one digit of
   headroom at float32 (eps ≈ 1.2e-7). Either run it under `jax_x64` or state the
   tolerance as explicitly relative and justify it — do not let a tolerance silently
   become a test of the dtype. M0's notebook (§4) has the measurement behind this.

**Open, not blocking:** Q2 in prompt 1 (whether to pin a `ruff.toml`, and whether to
adopt `ruff format`). Until JXP rules, keep new code clean under `ruff check robust/`
and use a commented `noqa` where a default rule is wrong for the code.

## Prompts

1. Read this doc. Execute the 1st task in the "M1" section below. See my answers on ruff in the 1st prompt doc.  If you have any questions, ask me in the Q&A section below.  Use Fable if you can.
2. Read this doc. Execute the 2nd task in the "M1" section below. If you have any
   questions, ask me in the Q&A section below.  Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task in the "M1" section below. Check my answers in
   Q&A. If you have any additional questions, ask me in the Q&A section below.  Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can. Log your work.

## M1

### Tasks

1. **Conventions.** `robust/rt/conventions.py`: `A_RRS=0.52, B_RRS=1.7`;
   `Rrs_to_rrs`/`rrs_to_Rrs`; the canonical wavelength grid (L23 350–750, 81 bands);
   pure-water `bb_w(λ)`; load-time asserts.

   Reuse over reinvention: `bing.rt` exports `A_Rrs`/`B_Rrs` — a test asserting our
   constants equal BING's is worth more than a comment, since the two packages sharing
   `rrs` is the point of fixing them at all. For `bb_w(λ)`, check whether BING already
   has the pure-water model (`bbNWModel.init_bbw` reads `Hydrolight400.nc`) before
   writing one.

   **Test:** `Rrs→rrs→Rrs` round-trips to ~1e-6 (mind the float32 caveat above); asserts
   fire on bad input (wrong grid, negative IOPs).

2. **Types.** `robust/rt/types.py`: `IOPs(a, bb_w, bb_p)`, `PhaseParams(B_p, …)`,
   `Geometry(theta_s, theta_v, dphi, wind)` as JAX pytrees with `jaxtyping` shapes.

   Keep `bb_w` and `bb_p` separate (design §3 — the water/particle split is load-bearing,
   and free for us). Register the pytrees so `jit`/`vmap`/`grad` traverse them —
   `flax.struct.dataclass` and `jax.tree_util.register_dataclass` are both available;
   pick one and say why in the record. `PhaseParams` must be shaped so the ZTT
   backward-VSF parameters can join it at M5 **without changing the `forward`
   signature** (that signature is already pinned in `hybrid.py`).

3. **L23 loader + splits.** `robust/rt/data/l23.py`: load the elastic set via `ocpy`
   for Y∈{0,30,60}; assemble `(IOPs, Geometry, Rrs)` JAX batches; compute
   `B_p = bbnw / bnw`; expose the **seeded splits** (random 20% of scenes; and the
   solar-zenith hold-out: train 0°/30°, test 60°).

   Measured in M0, so the range assert has a real reference: at **440 nm**, `B_p` has
   median **0.0126** and 1st–99th percentiles **0.0105–0.0180** — comfortably inside the
   design's ~[0.004, 0.03]. That was one wavelength only. **Check the range across all
   81 bands and all three zeniths, and report (do not silently clip) if it fails in the
   UV or the far red** — an assert tuned to 440 nm that fires at 350 nm would be
   discovered at M3, when it is expensive.

   **Gate.** `test_conventions.py` + `test_l23.py`: shapes `(3320, 81)`; `a, bb ≥ 0`;
   `B_p` within ~[0.004, 0.03]; a **golden-value** row cross-checked against the raw
   netCDF. Data tests carry `needs_l23`, so `pytest -q` stays green without the dataset.
   Update the implementation record; note the branch for JXP.

4. **Notebook.** `notebooks/RT/rt_elastic_coding_2.ipynb` — the M1 explainer, following
   the conventions recorded in the implementation record §2.6 and §8:

   - **Executed**, committed with outputs (`jupyter nbconvert --to notebook --execute
     --inplace`), so it reads without a kernel.
   - Data-dependent cells **degrade to a message** when `$OS_COLOR` is absent; bootstrap
     `sys.path` to the repo root (M0's notebook shows both patterns).
   - Figures: recessive grid/frame, text in ink colours not series colours, legend plus
     direct labels, one hue light→dark for sequential magnitude, and the CVD-checked
     categorical pair `#0072B2`/`#D55E00`. **Render and look at** each figure before
     calling it done — M0 caught two label collisions that way.
   - Explain what M1 *decided*, not just what it calls: why `rrs` rather than `Rrs` is
     the fitting space, why `bb_w` stays separate from `bb_p`, what `B_p` is and how it
     varies across λ / zenith / scene, and what the two held-out splits protect against.
     Do not re-explain M0 (JAX, autodiff, the float64 argument) — link to notebook 1.
   - Worth a figure: the `B_p` distribution across λ (it justifies or corrects the range
     assert), and the reference `Rrs` spectra grouped by split so the hold-out is visible
     rather than asserted.

### Q&A

## Next

→ `rt_elastic_coding_prompt_3.md` (M2: ZTT-in-JAX backbone).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-07-31 (M1 task 1 — conventions.py, plus the ruff config JXP approved)

`pytest -q` → **39 passed** (12 M0 + 27 new). With `$OS_COLOR` unset: **36 passed,
3 skipped** — exactly the three `needs_l23` golden tests, so CI stays green
without the reference data. `ruff check robust/` and `ruff format --check robust/`
clean. Implementation record at v0.6 with a new §3.

**First, Q2 from prompt 1** (JXP: yes to both). Added `ruff.toml` — `select =
E/F/I/W/UP/B`, line length 88, `target-version = py312`, formatter
`quote-style = "double"` — matching PAB so the two repos lint alike. Then ran
`ruff format robust/` (6 files reformatted) *before* writing new code, so
`conventions.py` was authored in the final style rather than reformatted after.
Also added `ruff format --check` to the CI lint job and rewrote its now-stale
comment about there being no config.

Two rules are ignored, both **because of `jaxtyping`**: `F722` (pyflakes tries to
parse a shape string like `" 81"` as a forward reference and calls it a syntax
error) and `UP037` (pyupgrade offers to delete quotes that *are* the shape
specification). The coding plan asks for light jaxtyping on public signatures, so
the rules go rather than the annotations. Worth knowing: `ruff check .` across the
whole repo reports **48** findings, all in pre-existing non-package scripts
(`setup.py`, `reports/py/`, `context/RT/`) — mostly `E702` from `import glob, os`
and long lines. I left them alone and scoped CI to `robust/`; say the word if you
want that swept.

**`conventions.py`.** `A_RRS`/`B_RRS`, `Rrs_to_rrs`/`rrs_to_Rrs`, the canonical
grid, `bb_w(λ)`, and three boundary validators. Decisions worth recording:

- **`WAVE` is NumPy, not `jnp`.** A device array built at import would freeze its
  dtype before a caller can enable float64; `canonical_wave()` converts on demand
  so it follows `jax_enable_x64`. The values are exact multiples of 5, so float32
  holds them without error either way. A test asserts the x64 behaviour.
- **Validators raise `ValueError`, not `assert`.** `python -O` strips `assert`,
  and a convention check that silently vanishes is worse than none. They are
  documented as *boundary* checks — they read concrete values, so they cannot run
  under `jit`, and are for where data enters the package, leaving `forward` clean.
- **`RRS_POLE` is named and checked.** `rrs_to_Rrs` diverges at `rrs = 1/B ≈
  0.588` and goes *negative* past it. Ocean `rrs` is ~1e-3–5e-2, so the only way
  to get there is a unit error — exactly the kind that otherwise surfaces as an
  inexplicable negative Rrs at M3. `check_rrs` looks for it and says "check the
  units".

**On `bb_w`, the one number that came from data — I followed the "reuse" rule and
it led somewhere useful.** Both `bing.bbNWModel.init_bbw` and
`ocpy.water.scattering.bbw_from_l23` compute pure-water backscattering as
`bb − bbnw` from an L23 file, and **both carry a TODO** saying to replace it with
a proper calculation. Chasing that: `ocpy.water.scattering.betasw_ZHH2009` (Zhang,
Hu & He 2009, the T/S-dependent physical model both TODOs point at) **raises
`ValueError("THIS IS NOT SUCCESFULLY CONVERTED YET")` on its first line** — an
unfinished MATLAB port. So the physical path does not exist yet in either package,
which is *why* both fall back to the difference. Recorded in the module and the
record so nobody re-discovers it; it matters at M5, when new HydroLight runs may
not share L23's water column.

For our purposes the L23 difference is not a fallback but the *correct* choice:
the model is trained against L23, so any other `bb_w` would put a bias straight
into `bb_p = bb − bb_w`. But both existing implementations take that difference at
an **arbitrary scene index** (bing: 0; ocpy: 170, commented "Random choie")
without checking that the choice is immaterial. So I checked: `bb − bbnw` is
constant to **1.6e-7 relative** (float32 storage noise) across all 3320 scenes,
all three solar zeniths, and both X=1 and X=4 — X=1 vs X=4 are bit-identical.
That makes a single 81-value table legitimate, so I embedded it: no data
dependency at import, which is what lets CI exercise the module, and two
`needs_l23` tests re-derive it from the netCDF and re-assert the
scene-independence so neither claim can rot.

**Tests (27).** The round trip is asserted in **both** dtype regimes — float32 to
1e-6 (measured 2.0e-7, so ~5× headroom) and float64 to 1e-12 (measured 2.6e-16) —
specifically so the float32 tolerance cannot later be tightened into a test of the
dtype, which is the trap M0's notebook §4 measured. `A_RRS`/`B_RRS` are asserted
equal to `bing.rt`'s under `importorskip` (CI has no bing): fixing the constants
only buys something if the package we share `rrs` with agrees. Both conversions
and `bb_w` are checked under `jit` and `grad` with derivative *signs* verified,
since they sit on the `forward` path. `bb_w`'s slope is fitted at λ^-4.2 against a
sanity band around Morel's -4.32 — a band, not a gate, because L23's water is
close to but not identical with pure molecular scattering. And the shifted-grid
test asserts the error message reports the offset, because a validator that says
only "bad grid" costs more time than it saves.

One consequence to note: `conventions.py` imports `jax.numpy` at module scope, so
`from robust import rt` now pulls JAX. M0 recorded that it pulled nothing and
predicted this would change at M2; it changed at M1. Expected, not a regression.

New: `ruff.toml`, `robust/rt/conventions.py` (implemented), `robust/tests/
test_conventions.py`. Modified: `robust/rt/__init__.py` (its docstring claimed
every module was a stub), `.github/workflows/ci.yml` (format check + stale
comment), `design/rt_elastic_implementation.md` (v0.6, new §3, lint conventions,
module index), and the 6 files `ruff format` touched. Branch
`rt-elastic-prototype` for JXP to commit. Next: task 2, the `types.py` pytrees.
