# Inelastic RT Docs — Prompt 1 (D1: scaffold & core pages; D2: full narrative, reports & figures)

## Goals

Build and publish the ReadTheDocs site for **`robust.rt` as one complete
forward model** — the elastic ZTT backbone, the residual emulator, `forward()`
and its composition law, Ed, the Raman and chlorophyll-a fluorescence terms,
the correction heads, the baselines, the data and validation protocol — at
`https://retrieve-or-bust.readthedocs.io/`.

Per the Q&A/Docs answers (DocQ1–DocQ9 in
`claude_prompts/RT/rt_inelastic_prompts.md`), the site:

- documents the **whole forward model**, with the inelastic material as the
  deepest chapters rather than the whole site; excludes the stub `rob/`
  package (`__init__.py` + `data/Dutkiewicz2015` only); and **states plainly
  that the inversion does not exist yet** — this is a forward model;
- is written for ocean-colour researchers who might use or check the model
  (plus Frouin and future sessions) but **opens with a basic introduction**,
  so a reader who has never seen this repo can follow page one (DocQ1);
- follows **PAB's shape** (flat `docs/`, `docs/conf.py`, MyST enabled) with
  **IOPtics' API pattern** (one `automodule` page, not BING's seven) and
  IOPtics' version single-sourcing — but a **different look than either**
  (DocQ2: "pick something cool"; the pick is made below and is not an open
  question);
- **installs the real JAX stack on RTD** — no `autodoc_mock_imports` — so the
  rendered signatures and `jaxtyping` shapes are the real ones (DocQ2).

The work is staged as **two milestones inside this one prompt doc** (DocQ9):

- **D1 — a real site on ReadTheDocs by the end of day one.** Scaffold,
  `conf.py`, `docs/requirements.txt`, `robust.__version__`, root
  `.readthedocs.yaml`, the strict CI docs job, Installation, Quickstart, Team,
  References, the single autodoc API page, the front page with its hero
  graphic, and a model-overview page. Building green under `sphinx-build -W`
  locally and *rehearsed in a clean venv the way RTD will build it*.
- **D2 — the prose.** The rest of "The model", the "Using it" pages including
  the verbatim-quoted limitations page, the Reports section, the figures
  script, the quickstart notebook and the development-record note, the seven
  missing docstrings, and a review pass.

**Sizing (DocQ9):** ~2,000–2,500 lines across ~15 pages, 2–3 days. No physics
risk here — only volume and mechanics. Nothing on the site may state a number
that was not measured; every figure and every metric traces to a report, the
implementation record, or a run performed in the task that writes it.

### The theme (decided, per DocQ2)

**`pydata-sphinx-theme`** (with `sphinx-design` and `sphinx-copybutton`).
Reasoning, since DocQ2 delegated the choice:

- It is neither of the exemplars' themes — BING and PAB use
  `sphinx_rtd_theme`, IOPtics uses `furo` — so the site reads as its own
  thing, which is what JXP asked for.
- It is **the** scientific-Python house theme (NumPy, SciPy, pandas, xarray,
  Matplotlib, scikit-image all ship it). For an ocean-colour researcher
  landing on a JAX forward model, that visual register says "library
  documentation" before a word is read.
- **Three-column layout**: persistent left nav *plus* a right-hand in-page
  table of contents. The ZTT and inelastic chapters are long, equation-dense
  pages with many subsections; furo and rtd-theme bury that structure, pydata
  exposes it.
- **Top-navbar sections** let the five parts of the site (Getting started /
  The model / Using it / Reports / Reference) be first-class tabs instead of
  one very long sidebar.
- Native **light/dark toggle**, MathJax legible in both, and `sphinx-design`
  card grids for a front page that can carry the hero figure and four entry
  points.
- Actively maintained under PyData/NumFOCUS, on a regular release cadence.

Pin `pydata-sphinx-theme>=0.16`. **Fallback, if and only if it fights the
strict `-W` build** (its version-switcher / announcement machinery is the
usual culprit): `sphinx-book-theme`, pydata's downstream sibling and the theme
JAX's own documentation uses. Do not switch silently — record the reason in
Q&A first.

## Claude

### Skills

Consider the skills in `.claude/skills/` as helpful — `critical-partner` when
deciding what a page should *not* claim, `code-review` for the D2 review task.

### Working agreements (hold for every D-prompt)

- **Git is handled by JXP** (per `CLAUDE.md`). The docs work belongs on a
  **fresh branch off `main`**, created by JXP once `inelastic-rt` merges
  (DocQ8). **Do not hardcode a branch name as fact**: the coding effort's
  plan said `rt-inelastic-prototype` and the actual checkout was
  `inelastic-rt` (prompt 1 Q&A Q2). Read the branch you are standing on, work
  there, and record its real name in the log. Never run state-changing git
  commands; read-only inspection is fine.
- **The docs describe the `main` state** (DocQ8, RQ7 precedent). If a page
  would have to describe something that only exists on an unmerged branch,
  ask in Q&A rather than documenting the future.
- **Environment: `ocean14`**, because the docs build is not lighter than the
  package — autodoc *imports* `robust.rt`, and the DocQ2 decision is to
  install the real JAX stack rather than mock it. Measured: `robust.rt`'s
  import-time dependencies are exactly **`jax`, `jaxtyping`, `numpy`**
  (`flax`, `optax` and `ocpy` are imported lazily inside functions), so a
  lighter env is *possible* — but a local build that mirrors RTD is worth more
  than a fast one. Add the Sphinx toolchain to `ocean14` with a
  `pip install --dry-run` **first** to confirm it is purely additive (house
  procedure since M0), and **never** `pip install -r requirements.txt`
  wholesale — its two `git+` lines clobber the editable `bing`/`ocpy`
  checkouts (prompt 1 Q&A Q1).
- **Reuse, don't reinvent.** The reports, the implementation records and the
  module docstrings already contain the physics, the numbers and the caveats.
  Pages should compress and link, not re-derive. Figures are *copied or
  regenerated* from `reports/`, never re-drawn by hand (DocQ7).
- **Every task is gated**, and from D1 task 1 onward the gate always includes
  `python -m sphinx -b html -W --keep-going docs docs/_build/html` returning
  clean. A page that is not in a toctree is a warning, and a warning is a
  failure.
- **Verify prose against output.** Every number, command and code snippet on
  the site must be run before it is written down — this repo's recurring
  documentation defect is numbers written before they were measured. Paste
  from a real run; if you cannot run it, do not print an output.
- Touching `robust/` is approved for this effort, but only for the two
  sanctioned edits: `__version__` (D1) and the missing docstrings (D2). No
  behavior changes; the test suite and the elastic hash-regression stay green.
  Use Opus if you can. Log your work.

## Context

Read before writing:

- **The answers that are the spec** — `claude_prompts/RT/rt_inelastic_prompts.md`,
  `## Q&A` → `### Docs`: DocQ1–DocQ9 with JXP's answers (≈ lines 78–278 as of
  2026-08-29; find the section by heading, the file is edited concurrently).
  Read them, not just the summary in this doc. Do **not** touch the `### CDOM`
  Q&A thread that follows.
- **What is being documented** — `robust/rt/`: `conventions.py`, `types.py`,
  `data/l23.py`, `ed.py`, `ztt.py`, `emulator.py`, `hybrid.py`,
  `inelastic.py`, `inelastic_corr.py`, `baselines.py`, `validation.py`
  (~5,900 lines, 11 modules + the data subpackage). `robust/rt/__init__.py`'s
  module docstring is already a compact site outline — start from it.
- **The two reports** (rendered in-site at D2) —
  `reports/report_rt_elastic_model.md` (313 lines) and
  `reports/report_rt_inelastic_model.md` (432 lines). §5 of each is the
  source of the limitations page; §2 of each is the source of the model
  chapters; §8 is reproducibility.
- **The implementation records** — `design/rt_elastic_implementation.md` and
  `design/rt_inelastic_implementation.md` (the measured numbers, the module
  index, the environment). Linked out to GitHub, not rendered (DocQ5).
- **The design docs** — `design/rt_elastic_model.md`,
  `design/rt_inelastic_model.md` and the two coding plans. Also linked, not
  rendered.
- **The three exemplar doc sets**, whose real configs were read for this doc:
  - `/Users/xavier/Oceanography/python/PAB/docs/conf.py` and
    `/Users/xavier/Oceanography/python/PAB/.readthedocs.yaml` — the shape to
    copy: flat `docs/`, `myst_nb` loaded *instead of* `myst_parser` (loading
    both conflicts), `source_suffix` mapping `.md`/`.ipynb` to `myst-nb`,
    `nb_execution_mode = "off"`, `suppress_warnings = ["myst.xref_missing"]`
    (essential: our Markdown carries repo-relative links that are not Sphinx
    targets, and the CI build is `-W`), RTD `fail_on_warning: false` with the
    strict gate in CI.
  - `/Users/xavier/Oceanography/python/IOPtics/docs/source/conf.py`,
    `/Users/xavier/Oceanography/python/IOPtics/.readthedocs.yaml`,
    `docs/requirements.txt`, `docs/figures/*.py` — the API pattern
    (`autosummary_generate`, `autodoc_default_options` with
    `member-order: bysource` and `undoc-members`, `autodoc_typehints =
    'description'`), the `release = ioptics.__version__` single-sourcing, the
    `ubuntu-24.04` / `python: "3.12"` RTD block, and the figure-script
    convention. **We deviate on one point**: IOPtics mocks the heavy imports;
    we do not (DocQ2).
  - `/Users/xavier/Oceanography/python/bing/docs/conf.py` — mostly a
    what-not-to-copy reference (`.rst`-only, hardcoded version, seven API
    pages), but two real lessons: `napoleon_use_ivar = True` is required or
    Attributes sections collide with autodoc's attribute docstrings (52
    duplicate-object warnings in BING), and its `intersphinx_mapping` set is a
    good starting list.
- **The repo mechanics** — `setup.py` (`python_requires='>=3.12'`, version
  `0.0.dev0`, `package_data` ships the `.npz` weights),
  `requirements.txt`, `.github/workflows/ci.yml` (two jobs today: `test`
  matrixed over py3.12/3.14, and `lint`/ruff), `ruff.toml`,
  `docs/member_policy.md`.

## Status entering D1

Verified fresh, 2026-08-29, on this checkout:

- **There is no docs scaffold at all.** `docs/` contains exactly one file,
  `docs/member_policy.md` (9 lines, team governance). No `conf.py`, no
  `index`, no `Makefile`, no `_static/`, no `.readthedocs.yaml` or
  `readthedocs.yaml` at the root, no docs job in `.github/workflows/ci.yml`.
  `.gitignore` already ignores `docs/_build/`. This is greenfield: nothing to
  extend, everything to create.
- **`robust/__init__.py` is one line** (a module docstring) — no
  `__version__`. `setup.py` hardcodes `version = '0.0.dev0'`.
- **The RTD project already exists** (DocQ3): slug `retrieve-or-bust`, URL
  `https://retrieve-or-bust.readthedocs.io/`. It has never had a successful
  build — there is no config for it to find yet.
- **Docstring coverage, re-measured by AST walk** (the DocQ4 method, rerun
  today): 138 public classes/functions across `robust/` excluding tests;
  **every module has a module docstring**; **9** public callables lack one —
  the **7** DocQ4 named, all *nested* helper functions:
  `emulator.py::_make_chunk::one_step` (905), `_make_chunk::chunk` (914),
  `_make_eval::evaluate` (927); `ztt.py::mu_infinity::cubic_in_L` (778),
  `rrs_ZTT::spectral` (883); `inelastic.py::emission_line::gaussian` (305);
  `validation.py::markdown_table::cell` (818) — **plus 2 that DocQ4's scan
  missed** because it globbed `robust/rt/*.py` and these live one level down:
  `data/l23.py::npz_reader::read` (362) and
  `inelastic_npz_reader::read` (826). All nine are closures, so **none of
  them appear in autodoc output**; filling them is a readability edit, not a
  rendering fix. Say so in the log rather than implying the API page needed
  it.
- **`robust.rt`'s import-time dependencies are `jax`, `jaxtyping`, `numpy`
  only.** `flax` and `optax` (emulator training) and `ocpy` (the L23 loader)
  are imported *inside* functions. Autodoc will therefore import the whole
  package without ocpy present — which is what makes the no-mocking decision
  cheap.
- **Figures on hand**: `reports/` holds 7 committed PNGs —
  `fig_architecture.png`, `fig_rrms_ladder.png`, `fig_unseen_zenith.png`
  (elastic, from `make_report_figures.py`) and
  `fig_inelastic_architecture.png`, `fig_inelastic_rrms_ladder.png`,
  `fig_inelastic_deltas.png`, `fig_inelastic_unseen_zenith.png` (inelastic,
  from `make_inelastic_report_figures.py`, which writes them at 200 dpi beside
  itself via `HERE / "fig_*.png"`). Both reports reference their figures by
  **bare relative filename** (`![...](fig_inelastic_architecture.png)`) —
  which is the path fixup D2 must handle.
- **Notebooks**: `notebooks/RT/` holds the ten milestone notebooks —
  `rt_elastic_coding_{1..5}.ipynb` and `rt_inelastic_coding_{1..5}.ipynb`.
  None are rendered (DocQ6); they get a one-line development-record note.
- **The suite is green** as the coding effort left it (416+ tests, ruff clean,
  elastic hash-regression two-tier). Re-run `pytest -q -ra` before the first
  edit to `robust/` so a docs-side change can never be blamed for a
  pre-existing failure.

## Prompts

1. Read this doc. Execute the 1st task in the "D1" section below. If you have
   any questions, ask me in the Q&A section below. Use Opus. Log your work.
2. Read this doc. Execute the 2nd task. Use Opus. Log your work.
3. Read this doc. Execute the 3rd task. Check my answers in Q&A; if you have
   additional questions, ask in Q&A. Use Opus. Log your work.
4. Read this doc. Execute the 4th task — Installation, Quickstart, Team,
   References. Use Opus. Log your work.
5. Read this doc. Execute the 5th task — the API page. Use Opus. Log your work.
6. Read this doc. Execute the 6th task — the front page and the model
   overview. Use Opus. Log your work.
7. Read this doc. Execute the 7th task — the D1 wrap-up. I will push the
   branch and confirm the site renders on ReadTheDocs before we start D2.
   Use Opus. Log your work.
8. Read this doc and the "Status entering D2" section. Execute the 1st task in
   the "D2" section — the quickstart notebook and the development record.
   Check my answers in Q&A. Use Opus. Log your work.
9. Read this doc. Execute D2's 2nd task — the elastic model chapters. Use
   Opus. Log your work.
10. Read this doc. Execute D2's 3rd task — the inelastic chapters. Use Opus.
    Log your work.
11. Read this doc. Execute D2's 4th task — Data and Validation. Use Opus. Log
    your work.
12. Read this doc. Execute D2's 5th task — Scope and limitations, plus the
    docstring fills. Use Opus. Log your work.
13. Read this doc. Execute D2's 6th task — the figures script and the Reports
    section. Use Opus. Log your work.
14. Read this doc. Execute D2's 7th task — the review pass and the wrap-up.
    Use Opus. Log your work.

**No "modify the next prompt doc" turn.** The coding effort needed one because
its five milestones lived in five files, each having to hand state to the
next. DocQ9 put both milestones in *this* file, so the equivalent hand-off is
D1's task 7, which fills the "Status entering D2" section **in this document**
with what D1 actually established. If D2's volume turns out to warrant a
`rt_docs_prompt_2.md` (say, a third milestone for a paper-facing report site
like PAB's `report_site/`), propose it in Q&A rather than creating it
unasked.

## D1

Goal: a real, published site — thin on prose, complete in mechanics.

### Tasks

1. **Scaffold and `conf.py`.** Create the flat `docs/` tree (PAB's shape):
   `docs/conf.py`, `docs/index.md`, `docs/Makefile`, `docs/requirements.txt`,
   `docs/_static/`, `docs/_templates/`, and the subdirectories the page set
   needs (`docs/model/`, `docs/using/`, `docs/reports/`, `docs/figures/`).
   `docs/member_policy.md` **stays exactly where it is** — it is an existing
   path that may be linked elsewhere; it becomes a page, it does not move.

   `conf.py` specifics, all of which were read off working configs:
   - `sys.path.insert(0, os.path.abspath(".."))` so autodoc imports the
     checkout;
   - `project = "retrieve-or-bust"`, author J. Xavier Prochaska and
     collaborators, `release` read from `robust.__version__` inside a
     `try/except` falling back to `"0.0.dev0"` (task 2 supplies it), `version`
     = the first two dotted fields;
   - extensions: `sphinx.ext.autodoc`, `autosummary`, `napoleon`, `viewcode`,
     `intersphinx`, `mathjax`, `myst_nb`, `sphinx_design`,
     `sphinx_copybutton`. **Load `myst_nb`, never `myst_parser` as well** —
     they conflict (PAB's comment says so from experience);
   - `source_suffix = {".rst": "restructuredtext", ".md": "myst-nb",
     ".ipynb": "myst-nb"}`; `nb_execution_mode = "off"` (notebooks are
     committed with outputs);
   - `myst_enable_extensions` including at least `dollarmath` (the reports and
     the model pages carry `$…$` math) and `colon_fence`;
   - `suppress_warnings = ["myst.xref_missing"]` — required, because the
     rendered reports link to repo paths that are not Sphinx targets and CI
     builds with `-W`;
   - `napoleon_use_ivar = True`, `napoleon_use_param`/`use_rtype` True, both
     Google and NumPy styles on;
   - `autodoc_default_options` = members, `member-order: "bysource"`,
     `undoc-members`, `show-inheritance`; `autodoc_typehints = "description"`;
     `autosummary_generate = True`; and **`autodoc_mock_imports = []`** with a
     comment saying the empty list is deliberate (DocQ2) and naming the reason
     it is safe: `robust.rt` imports only `jax`, `jaxtyping`, `numpy` at
     module level;
   - `intersphinx_mapping`: python, numpy, scipy, matplotlib, and jax
     (`https://docs.jax.dev/en/latest/`) — verify each `objects.inv` actually
     resolves during the build rather than assuming;
   - `html_theme = "pydata_sphinx_theme"`, `html_static_path = ["_static"]`,
     `html_title`, and `html_theme_options` with the GitHub `icon_links` entry
     pointing at `https://github.com/ocean-colour/retrieve-or-bust`, the
     navbar sections, and light/dark enabled. Keep the options minimal — every
     option is a chance to fail `-W`.

   `docs/requirements.txt` (BING/IOPtics style, *not* PAB's root-file style —
   DocQ2's one deviation): the Sphinx toolchain (`sphinx>=8`,
   `pydata-sphinx-theme>=0.16`, `myst-nb>=1.1`, `sphinx-design`,
   `sphinx-copybutton`) **plus the real import-time stack** (`jax`,
   `jaxtyping`, `numpy`, and `flax`/`optax` for completeness), with a comment
   explaining why the root `requirements.txt` is not reused (its `git+` lines
   pull `ocpy`'s heavy geospatial extras, which no docs page needs and which
   are a build-breaking risk on RTD).

   Add to `.gitignore`: the generated figure copies and report copies that D2
   introduces (`docs/_static/fig_*.png`, `docs/reports/report_rt_*.md`).
   `docs/_build/` is already ignored.

   `docs/index.md` for now: a title, a two-paragraph introduction (what the
   package is; that it is a **forward** model and the inversion does not exist
   yet), and a `toctree` with the pages this milestone will create. Stubs are
   fine at this task, but **no page may exist outside a toctree** at the end
   of the task.

   **Gate.** `pip install --dry-run -r docs/requirements.txt` in `ocean14` is
   purely additive (stop and ask in Q&A if not); then, after installing,
   `python -m sphinx -b html -W --keep-going docs docs/_build/html` exits 0
   with **zero** warnings. Paste the exact command and its tail into the log.

2. **`robust.__version__`, single-sourced, and the RTD config.** Two coupled
   mechanics (DocQ3, DocQ8):
   - Add `__version__ = "0.0.dev0"` to `robust/__init__.py` and make
     `setup.py` **read it** rather than repeating it — parse the literal out
     of `robust/__init__.py` with a regex (do not `import robust` at build
     time; that would drag `jax` into packaging). One source of truth, so the
     two can never drift. `conf.py`'s `release` picks it up.
   - Write the root **`.readthedocs.yaml`** (dotted, per DocQ3): `version: 2`;
     `build.os: ubuntu-24.04`, `build.tools.python: "3.12"` (the `setup.py`
     floor); `sphinx.configuration: docs/conf.py`, `builder: html`,
     `fail_on_warning: false` — with a comment saying the strict gate lives in
     CI so a stray environment warning never blocks publication; `python.install`
     = `{method: pip, path: .}` followed by `{requirements: docs/requirements.txt}`.
     Do **not** add `formats: [pdf, epub]` (BING does; nobody asked for it and
     each is another way for the build to fail).

   **Gate.** `python -c "import robust; print(robust.__version__)"` prints
   `0.0.dev0`; `python setup.py --version` prints the same string; `pip install
   -e . --no-deps` still succeeds. **Rehearse the RTD build**: create a throwaway
   `python3.12 -m venv` in the scratchpad, run exactly the two install steps the
   YAML declares, then `python -m sphinx -b html docs <tmp>` — this is the only
   way to learn *before pushing* that the site builds without `ocean14`'s
   incidental packages. Record the rehearsal's package list and any surprise in
   the log. Re-run `pytest -q -ra` after the `robust/__init__.py` edit.

3. **The CI docs job.** Add a third job to `.github/workflows/ci.yml`, beside
   `test` and `lint`, following that file's established style (it is heavily
   commented — match it; explain *why* each step exists):

   - `docs:` / `name: sphinx (-W)` / `runs-on: ubuntu-latest`;
   - `actions/checkout@v4`, `actions/setup-python@v5` with `python-version:
     '3.12'` and `cache: pip`;
   - `pip install -r docs/requirements.txt` then `pip install -e . --no-deps`
     (mirroring the `test` job's deliberate `--no-deps` reasoning);
   - `python -m sphinx -b html -W --keep-going docs docs/_build/html`.

   Note in a comment that the job needs **no `$OS_COLOR` and no L23 data** —
   the docs build only *copies* committed figures (D2's script defaults to
   copy mode; regeneration is opt-in). If that ever stops being true, the job
   breaks, and the comment tells the next person why.

   **Gate.** The workflow file parses (`python -c "import yaml,sys;
   yaml.safe_load(open('.github/workflows/ci.yml'))"`), the docs job's step
   sequence is reproduced locally end-to-end in the task-2 throwaway venv and
   passes, and `-W` is genuinely strict (prove it: introduce a deliberate bad
   cross-reference, watch the build fail, revert it).

4. **Installation, Quickstart, Team, References.**
   - `docs/installation.md` — the real dance, run and verified in this task,
     not recalled: Python ≥ 3.12; `pip install -r requirements.txt` **with the
     `git+` caveat stated** (it will replace editable `bing`/`ocpy` checkouts
     — the M0 lesson, worth a warning admonition); `pip install -e . --no-deps`;
     what `package_data` ships (`robust/rt/files/*.npz` — the trained emulator
     and correction heads, without which `mode='hybrid'` fails at the first
     call); CPU JAX vs `jax[cuda12]`; `$OS_COLOR` and what skips without it
     (`pytest -q -ra` and the `needs_l23` markers); and how to verify the
     install (`python -c "import robust.rt; ..."` plus `pytest -q -ra`, with
     the *actual* pass/skip counts from a run you performed).
   - `docs/quickstart.md` — hand-written prose with **executed** snippets:
     build `IOPs`/`PhaseParams`/`Geometry` for one L23 scene, call
     `forward(iops, phase_params, geometry, wave)`, switch on
     `inelastic=Inelastic()`, show the difference at 685 nm and in the
     550–700 nm Raman band, then take one `jax.grad` (w.r.t. `a` and w.r.t.
     `phi_C`) to make the point that gradients are the purpose. Every printed
     number pasted from a real run. Note the executed **notebook** version
     arrives at D2 and link forward to it.
   - `docs/team.md` is **not** created; instead `docs/member_policy.md` is
     added to the Reference toctree with an explicit title — a MyST toctree
     entry of the form `Team <member_policy>` (DocQ3 wanted it rendered as a
     "Team" page; retitling in the toctree is how to do that without moving or
     rewriting the file).
   - `docs/references.md` — the bibliography the site cites: Twardowski &
     Tonizzo (2018), Gordon et al. (1988), Loisel et al. (2023), Bartlett et
     al. (1998), Desiderio (2000), Gordon (1979), Maritorena et al. (2000),
     Behrenfeld et al. (2009), Sullivan & Twardowski, PR05/O25. Take the
     entries from the two reports' References sections rather than composing
     new ones, and check each against the report before pasting.

   **Gate.** `-W` build clean; every command in `installation.md` executed in
   this task (in `ocean14` and/or the throwaway venv) with its real output;
   every snippet in `quickstart.md` executed and its output pasted verbatim;
   the four pages reachable from the front-page toctree.

5. **The API page.** One `docs/api.rst` (IOPtics' pattern, DocQ4), with an
   `automodule` block per module in the order the package's own
   `__init__` docstring uses — `conventions`, `types`, `data.l23`, `ed`,
   `ztt`, `emulator`, `hybrid`, `inelastic`, `inelastic_corr`, `baselines`,
   `validation` — each with `:members:`, `:undoc-members:`,
   `:show-inheritance:`, and a one-line orientation sentence above it so the
   page is navigable rather than a wall. Include `robust` and `robust.rt`
   themselves so the package docstrings render.

   Watch for, and resolve rather than suppress: duplicate-object warnings from
   the dataclass Attributes sections (BING's `napoleon_use_ivar` lesson);
   `jaxtyping` annotations such as `Float[Array, "*batch wave"]` rendering as
   something legible; and the `__all__` lists (every module has one, grouped
   by role with `# noqa: RUF022`) — decide whether to honour them
   (`:members:` with no argument follows `__all__`) and say which you chose
   and why.

   **Gate.** `-W` build clean with **no autodoc import errors and no mocking**;
   spot-check the rendered HTML for four things: `forward()`'s full signature
   including the keyword-only `inelastic`/`corrections`, the four pytrees'
   attribute tables, a `jaxtyping` shape annotation, and a `viewcode` source
   link that resolves.

6. **The front page and the model overview.**
   - `docs/index.md` filled in properly: the hero figure
     (`reports/fig_inelastic_architecture.png`, reused as-is per DocQ7), a
     basic introduction (DocQ1: assume the reader knows ocean colour but not
     this package), an honest one-paragraph statement of **what exists and
     what does not** — a differentiable forward model, validated on L23; *no
     inversion yet* — the headline measured numbers with a link to where they
     were measured, and a `sphinx-design` card grid pointing at Getting
     started / The model / Using it / Reference.
   - Create `docs/figures/make_docs_figures.py` now, in **copy mode only**, so
     the hero has a supply chain rather than a committed duplicate (DocQ7): it
     copies named PNGs from `reports/` into `docs/_static/`, is pure
     `pathlib`/`shutil` (no matplotlib import at module level), is idempotent,
     and is invoked from `conf.py` at import time so **RTD and CI produce the
     copies themselves** and the copies stay gitignored. D2 extends it.
   - `docs/model/overview.md` — the model in one page: the composition law
     `Rrs = [Rrs_ZTT + ΔRrs_emulator] × f_R + φ_C·K_fl`, with the corrected
     form `f_R = 1 + (f_phys − 1)(1 + δ_R)` and `φ_C·K_fl(1 + δ_F)`; what each
     term is and which module owns it; the three `mode` values; the
     `inelastic=None` bit-identity guarantee; and a table mapping concept →
     module → API anchor. This page is the map that D2's chapters hang off,
     so write the toctree for the whole "The model" section here even though
     most targets are D2 stubs.

   **Gate.** `-W` build clean from a **clean tree** (delete `docs/_static/fig_*.png`
   and `docs/_build/` first, so the conf.py hook is proven to regenerate the
   hero); the front page renders the hero in both light and dark theme; every
   D2 stub page exists with a title and a one-line "arrives at D2" note so no
   toctree entry dangles.

7. **D1 wrap-up.** Fill this document's **"Status entering D2"** section with
   what D1 actually established: the exact Sphinx/theme/extension versions
   installed, the build command and its timing, anything the RTD rehearsal
   revealed, the final page inventory, any `conf.py` setting that had to
   change from the plan above and why, and the branch's real name. Summarize
   the state for JXP: which files are new, which two `robust/` lines changed,
   and that the site is ready to push.

   **Gate.** `-W` clean; `pytest -q -ra` green; `ruff check robust/` and
   `ruff format --check robust/` clean; the "Status entering D2" section is
   filled with measured facts, not intentions. **JXP then pushes the branch
   and confirms the site builds and renders at
   `https://retrieve-or-bust.readthedocs.io/`** — the first RTD build is the
   real gate on task 2's YAML, and it cannot be run from here. If it fails,
   the fix is task 1 of D2's turn, ahead of the notebook.

### Q&A

*(Empty — to be filled turn by turn.)*

## D2

Goal: the prose, the provenance and the figures — the site as a manual.

### Tasks

1. **The quickstart notebook and the development record.** Write
   `docs/quickstart_nb.ipynb` (new, short — DocQ6): load one L23 scene →
   `forward()` elastic → `forward(..., inelastic=Inelastic())` → one gradient
   → one plot of both spectra and their difference. Follow the house notebook
   conventions (degrade gracefully without `$OS_COLOR`, the `ocean14`
   kernelspec, executed and committed **with outputs**, figures in the
   recorded style). It is rendered by `myst-nb` with `nb_execution_mode =
   "off"`, so the committed outputs *are* the page — check the rendered HTML,
   not just the notebook.

   Then a short `docs/development_record.md` (one line plus a list, per
   DocQ6): the ten milestone notebooks `notebooks/RT/rt_elastic_coding_{1..5}.ipynb`
   and `rt_inelastic_coding_{1..5}.ipynb`, linked to GitHub on `main`, framed
   as a chronological build record and explicitly **not** a tutorial; plus
   links out to the design docs, the two coding plans and the two
   implementation records (DocQ5: linked, never rendered).

   This task is first in D2 deliberately: notebook rendering is the one piece
   of `conf.py` plumbing D1 configured but never exercised. Find out now.

   **Gate.** `-W` build clean with the notebook rendered; outputs visible in
   the HTML; no execution attempted at build time (confirm by building with
   the kernel unavailable); every GitHub link in the development record
   resolves (check the paths against the repo, and use `main`, not a branch
   name).

2. **"The model" — the elastic chapters.** `docs/model/conventions.md`,
   `ztt.md`, `emulator.md`, `forward.md`, `ed.md`. Source material: the module
   docstrings (already thorough), `design/rt_elastic_model.md` §§2–4, and
   `reports/report_rt_elastic_model.md` §2. Each page: what the piece computes,
   the equations that matter (MyST `dollarmath`), the conventions and gotchas a
   user will otherwise hit (`Rrs` ↔ `rrs` with A = 0.52, B = 1.7; the 81-point
   canonical grid and `check_wave`; `bb_w(λ)`; that `mode='emulator'` output
   is **not** additive with `mode='ztt'` in `Rrs` space; the domain guard and
   `on_out_of_domain`), and a cross-link to the API anchor rather than a
   restatement of the signature.

   **Gate.** `-W` clean; every equation checked against the module or the
   design doc it came from (cite which, in the page or the log); every
   `:func:`/`:class:` cross-reference resolves — with `-W`, a typo'd role is a
   build failure, which is the point.

3. **"The model" — the inelastic chapters.** `docs/model/inelastic.md`
   (Raman: the 3400 cm⁻¹ shift and the exact `585.08 nm` from 488 nm
   excitation — the number bing's own docstring gets wrong; the excitation
   grid; `raman_factor`), `docs/model/fluorescence.md` (the φ_C-linear
   kernel, `emission_line`, `emission_shape='single'` vs `'double'`),
   `docs/model/corrections.md` (the two learned heads, `HeadConfig`, the
   packaged `robust/rt/files/{raman,fl}_corr_l23.npz`, `corrections=None` vs
   `False`, and the tanh bounds), `docs/model/baselines.md` (Gordon, PR05/O25
   — what they are for and that the hybrid is scored against them).

   Carry the measured per-process numbers from
   `reports/report_rt_inelastic_model.md` §4 with attribution, and the
   corrections' honest framing: the heads are **interpolators over three
   zenith anchors** and carry no domain guard.

   **Gate.** `-W` clean; every quoted number traced to a report section or the
   implementation record, cited on the page; the Raman wavelength arithmetic
   re-derived in this task (not copied) and shown to match.

4. **"Using it" — Data and Validation.** `docs/using/data.md`: L23 via
   `ocpy.hydrolight.loisel23`, `$OS_COLOR`, the X=1/X=2/X=4 files and what the
   X levels mean, `L23Batch` / `L23InelasticBatch` / `Splits` / `load_batch` /
   `make_splits`, the committed test fixtures, and what a user without the
   ~17 MB netCDFs can still do. `docs/using/validation.md`: the protocol in
   `robust/rt/validation.py` (`rrms` and its rrs-space definition,
   `group_rrms`, `median_increment_error`, `peak_ratio_error`,
   `phi_c_linearity`, `speed_ratio`, the gradient reports), the design §6
   acceptance gate, and the measured results — **including that the total-rRMS
   gate is scored over the stated 400–700 nm domain** (prompt 5 Q&A Q1), with
   the full-grid number reported alongside rather than hidden.

   **Gate.** `-W` clean; the gate table's numbers match the report and the
   implementation record exactly (diff them, do not eyeball); every function
   named on the page exists in `validation.py` with that spelling.

5. **Scope and limitations, and the docstring fills.**
   `docs/using/limitations.md` is a prominent, unbowdlerized page (DocQ4): it
   quotes `reports/report_rt_inelastic_model.md` §5 **verbatim** — the "may
   claim" paragraph and all six "may not claim" items, including the **−74 %**
   unseen-zenith cliff, φ_C truth at 0.02 only, `emission_shape='double'` at
   −23.6 % and unvalidatable, λ ≥ 400 nm (13 % at 350 nm), the θ_s-anchor
   derivative kink, and the inherited elastic caveats — plus the elastic
   report's §5 for the backbone, and a plain statement that **there is no
   inversion**. Same bluntness as the reports; no softening verbs. Where a
   caveat is quoted, say so and link the source.

   Then fill the missing docstrings: the **7** in `robust/rt/*.py` that DocQ4
   named and the **2** in `robust/rt/data/l23.py` that its scan missed (see
   "Status entering D1" for the exact nine, with line numbers). All nine are
   closures and none is emitted by autodoc — this is a readability edit; say
   that in the log rather than claiming a rendering fix. Re-run the AST walk
   afterwards and report the new count.

   **Gate.** `-W` clean; the quoted passages diffed against the report file
   character-for-character (show the diff command in the log); AST walk
   reports zero missing docstrings across `robust/` excluding tests;
   `pytest -q -ra` green; ruff check + format clean.

6. **The figures script and the Reports section.** Extend
   `docs/figures/make_docs_figures.py` from D1's copy-only form to the full
   IOPtics-pattern tool (DocQ7):
   - **copy mode (default, and what RTD/CI run)**: copy all seven
     `reports/fig_*.png` into `docs/_static/`, and write
     `docs/reports/report_rt_elastic_model.md` and
     `report_rt_inelastic_model.md` as generated copies of the two reports
     with two mechanical rewrites — image references `](fig_*.png)` →
     `](../_static/fig_*.png)`, and repo-relative links (`design/…`,
     `robust/…`, `notebooks/…`, `reports/…`) → absolute
     `https://github.com/ocean-colour/retrieve-or-bust/blob/main/…` URLs.
     Prepend a short generated-file banner naming the source path so nobody
     edits the copy. Outputs stay **gitignored** (D1 task 1 added the
     patterns): they are derived from bytes already in the repo, so there is
     no second committed copy and no drift to guard.
   - **`--regenerate` mode (opt-in, dev machines only)**: re-run
     `reports/make_inelastic_report_figures.py` and
     `reports/make_report_figures.py` — reuse them, do not fork their plotting
     code — and only then copy. This mode needs `$OS_COLOR` and L23; it must
     fail loudly and early if either is missing, and must never run
     implicitly from `conf.py`.

   Add `docs/reports/index.md` introducing the two reports (what each measured,
   when, and that they are the site's evidence base) with the two generated
   pages in its toctree.

   **Gate.** From a clean tree (`docs/_static/fig_*.png` and
   `docs/reports/report_rt_*.md` deleted), a bare `-W` build regenerates
   everything and renders both reports with **every figure visible and every
   link resolving** — check the rendered HTML for broken images, since a
   missing image is a warning MyST may not raise the way a bad xref does. Run
   the script twice and confirm byte-identical output (idempotent). Confirm
   `--regenerate` is not reachable from the build.

7. **Review pass and wrap-up.** Run a review over the full branch diff (the
   `code-review` skill at high effort), and separately proofread the rendered
   site as a reader: click every toctree entry, read the front page cold, and
   check that a newcomer meets the "this is a forward model, there is no
   inversion" statement before any claim of accuracy. Fix or explicitly
   decline every finding with a reason in the log. JXP's per-milestone pattern
   is a PR plus `@cursor review` (M0's PR #14, M3's PR #18 — the Bugbot
   findings were real both times); when that PR appears, address those
   comments too, checking Q&A for pointers.

   Then close the effort: a final full `-W` build with timing, a page/line
   inventory against DocQ9's ~15-page, ~2,000–2,500-line estimate (report the
   actual, whatever it is), the "Next" section below updated with what remains,
   and a note to JXP on the branch state.

   **Gate.** `-W` clean; `pytest -q -ra` green; ruff clean; CI's three jobs
   green once pushed; the live RTD build succeeds and the published site
   matches the local one; every review finding resolved or declined in
   writing.

### Q&A

*(Empty — to be filled turn by turn.)*

## Next

After D2 the forward model is documented and published. Open follow-ons, none
of them in scope here:

- **The CDOM-fluorescence work** (`Q&A/CDOM` in
  `rt_inelastic_prompts.md`) will add a term to the model; the inelastic
  chapters and the limitations page are where it will land.
- **A paper-facing report site** (PAB's `report_site/` pattern) if the results
  ever need a separate community-facing target — not proposed, not needed for
  v1.
- **The inversion**, whenever it exists: the site is deliberately written so
  that adding it is a new section, not a rewrite of the claims.
- A purpose-drawn hero graphic, if `fig_inelastic_architecture.png` ever stops
  earning the front page (DocQ7 chose reuse for v1).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
