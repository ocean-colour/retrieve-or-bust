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
