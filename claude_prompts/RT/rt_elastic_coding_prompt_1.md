# Elastic RT Coding — Prompt 1 (M0: Environment & scaffold)

## Goals

Implement **Milestone M0** of the coding plan
(`design/rt_elastic_model_coding_plan.md`): stand up the environment and the package
scaffold for the differentiable elastic-RT forward model, and create the running
implementation record. Nothing scientific yet — the point is a green, importable base.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `critical-partner`, `code-review`,
`verify`) as helpful.

### Working agreements (hold for every M-prompt)

- **Git is handled by JXP** (per `CLAUDE.md`). Work on a branch (suggest
  `rt-elastic-prototype`); each milestone is a reviewable commit/PR. Do **not** run
  state-changing git commands; read-only inspection is fine.
- **Python only**, in the `ocean14` conda env; **CPU-only JAX** for now.
- **Reuse, don't reinvent.** Build on `ocpy` (the `loisel23` L23 loader) and the
  installed `bing` package; follow **BING conventions** — tests in `robust/tests/` as
  `test_*.py` with a `conftest.py` and a `files/` fixtures dir.
- **Every milestone is `pytest`-gated.** Accuracy gates are *relative* (no blind
  targets). Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` (M0 section + Ground rules
  + Package layout).
- **Design** — `design/rt_elastic_model.md` (the `forward(iops, phase_params, geometry)
  → Rrs` API and the `robust/rt/` module layout).
- **BING conventions** — `Oceanography/python/bing/bing/` (submodule layout) and
  `bing/bing/tests/` (`test_*.py`, `conftest.py`, `files/`).

## Prompts

1. Read this doc. Execute the 1st task in the "M0" section below.
2. Read this doc. Execute the 2nd task in the "M0" section below. If you have any questions, ask me in the Q&A section below.
3. Read this doc. Execute the 3rd task in the "M0" section below.  Check my answers in Q&A. If you have any additional questions, ask me in the Q&A section below.  Use Fable if you can.

4. Please add a Jupyter Notebook (or two) to explain what you have done so far.  Place it in the `notebooks/RT/` folder.  Name it `rt_elastic_coding_1.ipynb`.  Use Fable if you can.

5. The Notebook looks great.  Please modify the next prompt doc, `rt_elastic_coding_prompt_2.md`, to reflect the changes you have made.  Include the creation of a Notebook.  Log your work

6. Please add the necessary files to turn on CI on GitHub.  Log your work.

## M0

### Tasks

1. **Create the implementation record.** New file `design/rt_elastic_implementation.md`
   — the running log of what gets built each milestone (mirror PAB's
   `PAB_implementation.md`): a table of milestones with status, and a per-milestone
   section for modules added, tests, and results. Seed it with M0.

2. **Dependencies.** Add `jax`, `flax`, `optax`, `jaxtyping` (CPU) to `requirements.txt`
   and install into `ocean14`. Verify `import jax; jax.numpy.ones(3)` runs on CPU.
   Record versions in the implementation record.

3. **Scaffold the package.** Create `robust/rt/` with stub modules per the coding-plan
   layout (`__init__.py`, `types.py`, `conventions.py`, `data/l23.py`, `ztt.py`,
   `emulator.py`, `hybrid.py`, `validation.py`) and `robust/tests/` with `conftest.py`,
   an empty `files/`, and `test_env.py`.

   **Gate.** `pytest -q` collects and passes; `test_env.py` asserts `import jax` works on
   CPU and `from robust import rt` succeeds. Log the result and update the implementation
   record; note the branch for JXP to commit.

### Q&A

**Q1 (M0 task 2, Claude → JXP).** Task 2 said `requirements.txt`; I also mirrored
`jax`, `flax`, `optax`, `jaxtyping` into `setup.py`'s `install_requires`, because
this repo currently keeps that list in sync with `requirements.txt` (it duplicates
every pip-installable entry), and leaving the RT stack out would mean a fresh
`pip install -e .` silently lacks JAX. The cost is that anyone installing
`retrieve-or-bust` for non-RT work now pulls `jaxlib` (~100 MB). **Say the word
and I'll drop the `setup.py` half** and let `requirements.txt` be the sole
declaration. — *No answer needed to proceed; task 3 is unaffected.*
>A. Yes, drop the `setup.py` half and let `requirements.txt` be the sole declaration.

Done in task 3 — `install_requires` is back to its original list, with a comment
recording *why* the JAX stack is absent so it does not get "helpfully" re-added.

**Q2 (M0 task 3, Claude → JXP).** The coding plan names **ruff** as a convention
(CQ4), but this repo has **no ruff config** — no `ruff.toml`, no
`pyproject.toml`, and none in any parent directory. So `ruff check` runs against
whatever the installed ruff (0.16.0) defaults to, which is broad enough that
BING's own code would trip it (its `conftest.py` catches `Exception` the same way
mine does; its `__init__.py` import style trips `I001`). Two questions:

1. **Pin a `ruff.toml`?** PAB uses `E/F/I/W/UP/B`, line length 88, `py312`. That
   would make "ruff clean" a reproducible statement instead of a property of
   whichever ruff version is installed. It would also start linting your existing
   `reports/py/` and `context/RT/` scripts, which is why I did not just add it.
2. **Adopt `ruff format`?** I did **not** run it. `ruff format` would rewrite the
   package to double quotes (my code and BING use single; your
   `context/RT/make_rt_elastic_figures.py` uses double), so it is a one-time
   style churn best decided once, before M1 adds real code.

I left `robust/` clean under `ruff check` either way, with two documented
`noqa`s. — *No answer needed to proceed with M1.*

>A. Yes, generate a ruff.toml file.  And, yes, let's use ruff format to format the package.

## Next

→ `rt_elastic_coding_prompt_2.md` (M1: Data & conventions).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-07-31 (M0 task 1 — implementation record created)

Created `design/rt_elastic_implementation.md` (v0.1), the running record of what
actually gets built per milestone. Structure mirrors PAB's
`docs/design/PAB_implementation.md`: header block (version/date/authors, status,
companions, versioning convention), a §1 status-at-a-glance table (M0–M5 with
status + package surface), then one section per milestone for modules added,
tests, and results, closing with cross-cutting conventions and a module index.

Seeded with M0. Because M0 has three tasks and only task 1 is done, the M0
section carries a per-task status table (task 1 ✅, tasks 2–3 ⬜) rather than
claiming the milestone: the "Modules added", "Tests", and "Results" subsections
are explicitly empty with the *planned* stub layout quoted from the coding plan,
so the record never over-reports.

Recorded the **pre-install** `ocean14` snapshot as the environment baseline:
Python 3.14.6, numpy 2.4.6, scipy 1.18.0, xarray 2026.4.0, pytest 9.1.1, `bing`
0.0.dev0, `ocpy` installed (exposes no `__version__`), and `jax`/`flax`/`optax`/
`jaxtyping` **not installed** — so task 2 is a genuine install, and its
post-install versions have a table to land in. Also noted the coding-plan risk
that a JAX install could perturb `ocean14` (fallback: a dedicated `rt-jax` env).

Two things worth carrying forward. (1) `robust` already uses
`find_packages()` in `setup.py`, so `robust.rt`/`robust.rt.data` become
importable the moment the `__init__.py` stubs exist — no packaging change needed
in task 3. (2) PAB keeps the dated narrative in the prompt Logs and the
structured current state in the implementation record; I kept that split here
rather than duplicating prose across both.

No code, no dependency, and no git changes in this task. Branch is
`rt-elastic-prototype` (already checked out) for JXP to commit.

### 2026-07-31 (M0 task 2 — JAX stack installed into ocean14, CPU verified)

Added a CPU-JAX block to `requirements.txt` and installed into `ocean14`:
**jax 0.11.0, jaxlib 0.11.0, flax 0.12.8, optax 0.2.8, jaxtyping 0.3.11**
(Python 3.14.6). Versions and the full transitive list are recorded in
`design/rt_elastic_implementation.md` §2.3 (record bumped to v0.2).

**The `ocean14`-perturbation risk is retired, not assumed away.** The coding plan
listed it as an M0 risk with a dedicated `rt-jax` env as the fallback, so I ran
`pip install --dry-run` *before* touching the env: the plan was purely additive —
`numpy>=2.1` and `scipy>=1.15` were already satisfied by 2.4.6 / 1.18.0, and pip
scheduled nothing for uninstall. Post-install, numpy/scipy/xarray are byte-for-byte
the same versions and `ocpy` (incl. `ocpy.hydrolight.loisel23`, the M1 loader),
`bing`, `emcee`, `sklearn`, and `matplotlib` all still import. **We stay in
`ocean14`.**

Verification beyond the letter of the task. The task asked for
`import jax; jax.numpy.ones(3)` on CPU — got `[1. 1. 1.]` on `CpuDevice(id=0)`,
with `jax.default_backend() == "cpu"` and `jax.devices() == [CpuDevice(id=0)]`.
I also checked two things M0 doesn't need but later milestones do, so a surprise
surfaces now rather than at a gate: **float64 works** under
`jax.config.update("jax_enable_x64", True)` (the M2 finite-difference gradient
gate needs it — the default float32 would make an FD comparison noisy enough to
be useless), and **`jax.grad` runs** (`d/dx Σx² = 6.0` at 3). These become the
automated assertions in task 3's `test_env.py`.

Notes on the dependency spec. (1) I list `jax` but **not** `jaxlib` — `jax`
requires the exact matching `jaxlib`, so a separate unpinned entry only invites
version skew. (2) Nothing needed for "CPU-only": on macOS arm64 the plain PyPI
wheel *is* CPU (GPU would be `jax[cuda12]`; Metal is a separate `jax-metal`
plugin), and the same holds on Linux. Recorded in the implementation record so
nobody later adds a redundant CPU pin. (3) Unpinned, matching every other entry
in `requirements.txt`; the exact versions live in the implementation record,
which is where reproducibility information belongs in this repo.

One scope call, raised as **Q1** in the Q&A section above: I also mirrored the
four packages into `setup.py`'s `install_requires`, since this repo keeps that
list in sync with `requirements.txt` and a fresh `pip install -e .` would
otherwise silently lack JAX. Trivially reversible if JXP would rather keep
`jaxlib` off the install path for non-RT users.

Branch remains `rt-elastic-prototype`; `requirements.txt`, `setup.py`, and the
implementation record are modified for JXP to commit.

### 2026-07-31 (M0 task 3 — package scaffold + pytest gate green; M0 complete)

**Gate passed.** `pytest -q` → **12 passed in 1.20s**; `ruff check robust/` →
clean. Both required assertions hold: `import jax` works on CPU, and
`from robust import rt` succeeds. M0 is done; the implementation record is at
v0.3 with the modules, tests, and results filled in.

First, acted on **Q1**: reverted `setup.py`'s `install_requires` to its original
list and left a comment saying the JAX stack lives in `requirements.txt` only,
deliberately — otherwise the next person to notice the asymmetry re-adds it.

**What "stub" means here, and why it matters.** Two modules carry their *final*
public signatures already — `hybrid.forward(iops, phase_params, geometry, wave,
mode='hybrid')` from design §3 and `ztt.Rrs_ZTT(...)` from coding-plan M2 — and
raise `NotImplementedError` naming the milestone that fills them. The other five
are docstring-only, each stating its role and its milestone. The point is that
the scaffold pins the *interface* without faking any physics: a caller gets a
loud, milestone-naming error rather than a plausible-looking array. A test
asserts exactly that, so the stubs cannot quietly start returning zeros. I also
declared `MODES = ('ztt', 'emulator', 'hybrid')` in `hybrid.py`, since the
three-way comparison is a design commitment (§4.5), not an implementation detail.

**The test suite is deliberately larger than the gate's letter (12 tests, not
2).** The gate asks for two imports; I added the properties later milestones
actually depend on, so a broken install surfaces now instead of three milestones
in: float64 reachable (M2's finite-difference gradient gate is meaningless in
float32 — the differencing noise swamps it), `jax.jit` compiles and computes
(design §3 requires `jit`/`vmap`-friendliness, so a broken XLA should fail here),
`jax.grad` correct, `flax`/`optax` import, and — beyond `default_backend()` —
that *every* device is CPU, since CQ5 is CPU-only and an accelerator sneaking in
should be a failure, not a surprise at M4's throughput numbers.

**`conftest.py` carries two things, both load-bearing later.** (1) `l23_available()`
plus a `needs_l23` skip marker — the L23 files are ~17 MB each, live outside the
repo, and are resolved from `$OS_COLOR`, so from M1 on the data dependency gets
declared once and absence becomes a *skip, not a failure*. This is lifted from
BING's `conftest.py` (reuse, not reinvention), narrowed to the three elastic
files. I verified the degradation instead of assuming it: with `OS_COLOR` pointed
at a nonexistent path, `l23_available()` returns `False` rather than raising and
all 12 tests still pass. (2) A `jax_x64` fixture that enables float64 for one
test and **restores the prior setting**, with its own test asserting the restore
— a leaked global x64 flag would silently change the dtype regime of every test
that ran afterwards, which is the kind of bug that looks like a physics problem.

Three things learned worth writing down. (1) **`jax.experimental.enable_x64` no
longer exists in JAX 0.11** — the context manager the older docs recommend for
exactly this fixture is gone, hence the explicit set/restore. (2) `from robust
import rt` currently pulls **no** heavy dependency, not even `jax` (~0.00 s),
because the stubs have no module-level imports; that will change when M2 puts
`import jax.numpy` at `ztt.py`'s module scope, and I recorded it so the change
is not later mistaken for a regression. `emulator.py` is documented to import
`flax`/`optax` inside its functions so the analytic-only path never pays for the
ML stack. (3) **All three L23 elastic files are present locally** —
`Hydrolight100/130/160.nc` under `/Users/xavier/Projects/Oceanography/data/Color/Loisel2023`
(`$OS_COLOR` is set) — so M1 has its 0°/30°/60° data in hand and needs no
download.

Two judgment calls, both recorded rather than silently taken. I did **not** add a
`pytest.ini`: bare `pytest -q` from the repo root already collects exactly
`robust/tests/` (there is no other `test_*.py` in the tree), so config would be
ceremony. And on lint — there is **no ruff config anywhere** in this repo or its
parents, so `ruff check` runs against ruff 0.16's own defaults, which are strict
enough that BING's code would trip them too. Rather than contort the scaffold to
a rule set the sibling repo does not follow, I fixed what cost nothing (import
ordering) and left two documented `noqa`s where the rule is wrong for this code:
`RUF022` on `__all__` (grouped by role, ordered as the pipeline runs — the order
*is* the information) and `BLE001` on `l23_available`'s `except Exception` (any
failure there genuinely means "no data", the same call BING makes). I did not run
`ruff format`, which would rewrite the package to double quotes. Both raised as
**Q2** for JXP, since a pinned `ruff.toml` and a one-time format sweep are best
decided before M1 adds real code.

Branch remains `rt-elastic-prototype`. New: `robust/rt/` (9 files),
`robust/tests/` (4 files). Modified: `setup.py` (Q1 revert),
`design/rt_elastic_implementation.md` (v0.3). Ready for JXP to commit — **M0
complete**; next is `rt_elastic_coding_prompt_2.md` (M1: data & conventions).
*(Committed by JXP as `ccbc0cc "M0 done"`.)*

### 2026-07-31 (M0 explainer notebook — notebooks/RT/rt_elastic_coding_1.ipynb)

Added the M0 explainer notebook: 27 cells (16 markdown, 11 code), **executed**
via `jupyter nbconvert --execute` so it ships with outputs and two figures and
reads without a kernel. Flow: where M0 sits in M0–M5 → the environment → why JAX
→ why float64 → the scaffold → the gate → a preview of the L23 reference data →
what M1 does next. One notebook, not two: M0 is a single coherent story and a
second file would have been padding.

The pedagogical spine is *why each piece is shaped the way it is*, since the code
itself is nearly contentless at this milestone. Two cells do the real work.

**1. Why JAX.** A live `jax.grad` on the standard Gordon relation gives
`∂Rrs/∂a = -0.0131` and `∂Rrs/∂bb = +0.328` at a plausible coastal operating
point — right signs (absorption removes photons, backscatter returns them) and,
more interestingly, `Rrs` is **25× more sensitive to bb than to a** there. That
is exactly the conditioning information an inversion needs and a black-box
emulator cannot hand you. Labelled unambiguously as an *illustration*, not
`robust.rt.forward`, because M0 implements no physics and a reader skimming
figures should not come away thinking otherwise.

**2. Why float64 — measured, not asserted (figure 1).** Relative error of a
central finite-difference derivative against `jax.grad`, swept over 33 step sizes
in both dtypes. Result: at a 1e-6 tolerance **float32 meets it at 0 of 33 step
sizes and float64 at 21 of 33** (best error 1.2e-5 vs 8.5e-12, ~6 orders apart).
So the sharp statement is not "float32 is less accurate" but: an M2 gate in
float32 would have to be loosened to ~1e-4 *or* tuned to one lucky `h` — it would
be testing the step size rather than the gradient. In float64 it passes across
three decades of `h`. That is the quantitative justification for the `jax_x64`
fixture, replacing the hand-wave in the earlier log.

**A trap I hit while writing that cell, now documented in the notebook because
M2 will meet it.** My first version passed *Python floats* to the model function.
Python floats are float64 and the function body is plain arithmetic — so both
curves were silently computed in float64 and the "float32" one came out ~1e-7:
a wrong answer that looks like a good one, and one I would have shipped had I not
dry-run the cell as a script first. The dtype must be pinned on the arrays
(`jnp.asarray(x, dtype=...)`) and asserted. When M2 writes the real FD gate, a
float64 perturbation around a float32 model will silently test something else.

**Figures were checked, not eyeballed.** For the two-series pair I computed OKLab
ΔE under simulated deuteranopia/protanopia/tritanopia rather than trusting that
blue-vs-orange "looks fine": ΔE 31 for normal vision, 29–45 under CVD (target
≥ 8), contrast 3.9 and 5.2 on white. I also *rendered and looked at* both
figures, which caught two defects a numeric check never would: the direct labels
overprinted each other at the right edge, and the float32 curve vanished
underneath float64 wherever the two coincide — it looked like missing data.
Fixed by labelling at the left ends (where the curves are far apart), drawing
float32 as markers over a bare float64 line, and annotating *why* they merge
(truncation-dominated, so dtype stops mattering). Figure 2 uses a single hue
light→dark for `a`(440) — sequential magnitude, never a rainbow — truncated at
30 % so the palest spectrum still has contrast on white.

**Two findings from making it run.** (1) `robust` is **not** pip-installed in
`ocean14` — a consequence of Q1 keeping the JAX stack out of `setup.py`. My
earlier "12 passed" only worked because `pytest` ran from the repo root, which
puts the cwd on `sys.path`. The notebook therefore bootstraps by walking up to
the directory containing `robust/__init__.py`; the same asymmetry means **`pytest`
must be run from the repo root**, and I recorded that in the implementation
record before someone runs it from `robust/tests/` and files a bug. (2) The L23
preview confirms the data end to end: 3320 × 81, 350–750 nm, and
`B_p(440)` median **0.0126** — inside the design's expected ~[0.004, 0.03], so
§4.2's phase-function parameter is consistent with the file we will actually
train on.

Also added a "Notebooks" entry to the implementation record's cross-cutting
conventions (one executed explainer per milestone in `notebooks/RT/`,
data-dependent sections degrading gracefully) so M1–M4 have a pattern to follow
rather than reinventing one each time.

New: `notebooks/RT/rt_elastic_coding_1.ipynb`. Modified:
`design/rt_elastic_implementation.md` (v0.4 — added §2.6 Notebook, the figure and
`sys.path` conventions, module index). Suite still 12 passed. Branch
`rt-elastic-prototype`, on top of `ccbc0cc`, for JXP to commit.

### 2026-07-31 (Updated prompt 2 for M1 with what M0 actually established)

Rewrote `rt_elastic_coding_prompt_2.md` so M1 starts from what M0 built rather
than from the pre-M0 plan. The three M1 tasks are unchanged in substance — the
work was folding in the facts, conventions, and traps that only exist now.

**Added a fourth task: the notebook.** `notebooks/RT/rt_elastic_coding_2.ipynb`,
with the conventions M0 set spelled out rather than left to be rediscovered:
executed and committed with outputs, data cells degrading to a message without
`$OS_COLOR`, the `sys.path` bootstrap, the figure style (recessive frame, ink-
coloured text, legend + direct labels, one hue light→dark for magnitude, the
CVD-checked `#0072B2`/`#D55E00` pair), and *render-and-look-at-it* called out
explicitly because that is what caught M0's two label collisions. I also told it
what to explain — the *decisions* (why `rrs` is the fitting space, why `bb_w`
stays split from `bb_p`, what the hold-outs protect against) — and what **not** to
re-explain, since notebook 1 already covers JAX and the float64 argument.

**Added a "Status entering M1" section.** M1's real risk is rebuilding what
exists, so this names the inheritance: the three stubs to fill (docstrings
already list their planned contents — fill the bodies, keep the docs), the
`needs_l23` marker and `l23_available()` to use instead of new skip logic, the
`jax_x64` fixture, and `robust/tests/files/` sitting empty for exactly the cached
L23 batch M1 should put there.

**One factual correction.** The old draft pointed at `$OS_COLOR_DATA/Loisel2023/`
— that variable is **unset**. The variable actually set is `$OS_COLOR`, and
`ocpy` resolves `loisel23.l23_path` from it. I checked the hardcoded laptop path
the draft also quoted (`/Users/xavier/data/Color/Loisel2023`) before "fixing" it,
and it turns out to be the *same directory* — `Projects/Oceanography/data` is a
symlink to `~/data`, same device and inode — so the draft was not wrong about the
data, only about the variable. Recorded it that way instead of overstating the
error, and made the instruction "resolve through `ocpy`, never a hardcoded path".

**Carried the two gotchas forward** so M1 does not rediscover them: `pytest` must
run from the repo root (`robust` is not pip-installed, per Q1), and JAX's float32
default leaves barely one digit of headroom under M1's own "round-trips to 1e-6"
gate (float32 eps ≈ 1.2e-7) — so that test should either use `jax_x64` or justify
its tolerance, rather than let the tolerance quietly become a test of the dtype.

**Turned M0's one measurement into a real gate reference.** M1's plan asserts
`B_p ∈ ~[0.004, 0.03]`. M0 measured it at 440 nm only: median 0.0126, 1st–99th
percentile 0.0105–0.0180. The prompt now says so, and instructs M1 to check the
range across all 81 bands and all three zeniths and to *report rather than clip*
a failure — an assert tuned to 440 nm that fires at 350 nm would otherwise be
found at M3, when it is expensive.

**Verified every API I told M1 to use**, having just corrected a stale path in the
same document: `jax.tree_util.register_dataclass` ✓ and `flax.struct.dataclass` ✓
both exist in the installed versions (offered as the two pytree-registration
options); `bing.rt.A_Rrs` = **0.52** and `B_Rrs` = **1.7**, so the "assert our
constants equal BING's" test I recommend will actually pass; `bbNWModel.init_bbw`
✓ exists, so the suggestion to check BING before writing our own `bb_w(λ)` is
real; and `Lambda` spacing is exactly 5 nm across 350–750 (81 bands). Also
recorded the dataset anatomy — dims `IOP_Scenario`/`Lambda`, the full variable
list — so M1 does not have to open a file to learn the schema.

**Not touched: prompts 3–6.** They will each need the same notebook task and a
similar status hand-off, but their content depends on results that do not exist
yet (M2's ZTT transcription, M3's training outcome), so updating them now would
be guessing. Better done at the head of each milestone, as this one was.

Modified: `claude_prompts/RT/rt_elastic_coding_prompt_2.md`. No code changes;
suite untouched at 12 passed. Branch `rt-elastic-prototype` for JXP to commit.

### 2026-07-31 (CI on GitHub Actions — and the packaging bug it exposed)

Added `.github/workflows/ci.yml` (two jobs), a badge in `README.md` with a short
Development section, and §10 of the implementation record documenting the design.
**Fixed one pre-existing bug in `setup.py` along the way — see below; it is the
most consequential thing in this entry.**

**Job `test`** — matrix Python **3.12** (the floor `setup.py` declares, and `jax`
0.11 requires ≥ 3.12) and **3.14** (what `ocean14` runs), `fail-fast: false` so one
version's failure still reports the other. Installs a lean set, then `ocpy`
`--no-deps`, then `pip install -e . --no-deps`, then `pytest -q -ra`. **Job
`lint`** — `ruff check robust/` with `ruff==0.16.0` pinned. Triggers on **every
branch** plus PRs, with a concurrency group cancelling superseded runs: main-only
CI would give no signal at all right now, since every milestone is built on
`rt-elastic-prototype` before it merges.

**Why not `pip install -r requirements.txt`.** That file is the developer's whole
environment. Via `ocpy` it pulls `cartopy`, `geopandas`, `healpy`, `netcdf4`, plus
`bing`/`emcee`/`bokeh`/`seaborn` — and the suite imports *none* of them. I checked
ocpy's `install_requires` before deciding rather than assuming it was light.
Installing only what the tests use keeps CI fast and, more importantly, makes a
red build mean "our code broke" instead of "a geospatial wheel failed to build".
Same reasoning for `ocpy --no-deps`: `robust` touches exactly one ocpy module,
`ocpy.hydrolight.loisel23`, which needs only numpy + xarray, so the real
integration is exercised without the extras. `bing` is left out entirely (nothing
imports it yet) with a note that M1's constants cross-check should either add it
or use `importorskip`.

**The bug.** The CI step `pip install -e . --no-deps` failed on a dry run:
`ValueError: illegal provides specification: 'retrieve-or-bust'`. `setup.py` had
`provides = [name]`, and `provides` is legacy distutils metadata that must be a
*module* name — the hyphen makes it invalid. So **`pip install .` has never worked
in this repo**. Two things fall out of that. First, it explains a fact I had
already recorded but explained *wrongly*: I had written that `robust` isn't
pip-installed in `ocean14` "per Q1, because the JAX stack lives only in
requirements.txt". That was a guess, and it was wrong — the install was simply
broken. I corrected the claim in both the implementation record and prompt 2
rather than leaving a plausible-sounding wrong reason in the docs. Second, it
would have hit anyone setting up the project from a clean clone. Fix: removed the
key (superseded by `Provides-Dist`, and it does nothing useful); `pip install -e .
--no-deps` now resolves to "Would install retrieve-or-bust-0.0.dev0". `bing` and
`ocpy` carry the identical line harmlessly — I checked — because their names have
no hyphen; this repo inherited the template and the hyphen broke it.

**Verified rather than hoped.** (1) The YAML parses, and I asserted the parsed job
names, triggers, matrix, and step list rather than eyeballing indentation.
(2) `pytest -q -ra` with **`OS_COLOR` unset** — the actual CI condition, not the
bogus-path variant I tested at M0 — gives **12 passed, 1 warning**: `ocpy` warns
and falls back to `./`, `l23_available()` returns `False`, and the data tests skip.
So CI will be green without the ~17 MB netCDFs, which is exactly what M0's
conftest was built for. (3) The editable install now dry-runs clean. (4) Both
sibling repos are public (HTTP 200), so the `git+` install of ocpy will resolve.

I also confirmed mid-task that `ocpy/hydrolight/__init__.py` was **missing from
GitHub `main`** while present in the local checkout — which would have made
`pip install git+…/ocpy` omit the whole subpackage and break the CI step. JXP
pushed the fix; re-checked the GitHub tree and the file is there now.

Conservative choices worth flagging: `actions/checkout@v4` and
`actions/setup-python@v5` rather than the newest majors, to minimise the chance of
a first run failing on a bad action reference; and the ruff pin, which exists only
because Q2 is unresolved — a `ruff.toml` would be the better fix and would let the
pin relax. If the 3.14 job cannot find an interpreter on the runner, drop it to
3.13; `fail-fast: false` means the 3.12 job still reports either way. The badge
will read "no status" until `ci.yml` exists on the default branch.

New: `.github/workflows/ci.yml`. Modified: `setup.py` (the `provides` fix),
`README.md` (badge + Development), `design/rt_elastic_implementation.md` (v0.5 —
new §10, CI line in §1, corrected the pip-install note in §2.6),
`claude_prompts/RT/rt_elastic_coding_prompt_2.md` (corrected the same claim).
Suite 12 passed; `ruff check robust/` clean. Branch `rt-elastic-prototype` for JXP
to commit — CI starts reporting on the next push.
