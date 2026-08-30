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

Questions from Claude (2026-08-29, model: Opus), raised during D1 tasks 1–2.
Neither blocked the two tasks — both gates were run and passed as written —
but the first contradicts a fact this document asserts, so it needs your call
before D1's task 7, whose gate demands a green suite.

**Q1 (The elastic hash-regression is red on this machine, and was red *before*
I touched anything).** "Status entering D1" says the suite is green as the
coding effort left it. It is not, on this checkout, in `ocean14`. I ran
`pytest -q -ra` **before** the first edit to `robust/` (exactly so a docs-side
change could never be blamed), and got:

```
2 failed, 451 passed, 1 skipped in 57.30s
FAILED robust/tests/test_inelastic_types.py::test_elastic_hash_regression_strict
FAILED robust/tests/test_inelastic_validation.py::test_gate_4_pre_change_pins
```

Re-running after the `robust/__init__.py` edit gave the identical result
(`2 failed, 451 passed, 1 skipped in 58.14s`, same two names), so the
`__version__` addition is provably not the cause.

Both failures are the same assertion — `sha256_of(Rrs) !=
PRE_CHANGE_SHA256_RRS_ABOVE` — and the evidence says **platform drift, not a
code regression**:

- the closeness tier, `test_elastic_regression_close_everywhere` (rtol 5e-7,
  ≈4 ULP), **passes**;
- measured deviation from the committed
  `robust/tests/files/elastic_reference_outputs.npz`: `Rrs` differs in
  2742/12150 elements (22.6 %), **max relative 3.33e-07, max 3.0 ULP**; `rrs`
  differs in 2862/12150 (23.6 %), max relative 1.64e-07, **max 2.0 ULP**.
  That is float32 rounding noise, not a restructured route;
- `ocean14` now carries **jax/jaxlib 0.11.0** and NumPy 2.4.6, which are
  almost certainly not the versions the pins were computed under.

`test_inelastic_types.py` anticipated exactly this, in the comment above the
pins: *"If a platform change — CPU, jax/XLA version — ever trips this without
any code change, that is a finding to take to JXP, not a hash to update
quietly."* So I am taking it to you rather than repinning. Note the strict
tier is `skipif(CI)`, so GitHub Actions is unaffected — this is a dev-machine
gate only.

My recommendation: **treat it as a JAX-version drift finding and repin
deliberately in a separate, non-docs commit**, recording the jax/jaxlib/NumPy
versions beside the new hashes, since the closeness tier is the guard that
actually detects a route change and it is green. The alternative — leaving it
red — costs D1 task 7 its "`pytest -q -ra` green" gate, which I would then
have to reinterpret as "green except these two, unchanged from the pre-edit
baseline". Which do you want? And should the repin happen here or on the
coding side, outside the docs effort?

**Q1 addendum, written after the fact — most of this is already answered.**
While I was finishing task 2 an uncommitted change to
`robust/tests/test_inelastic_types.py` appeared in the working tree (the CDOM
M5 effort, running concurrently), and it contains the real explanation, better
than my jax-version guess: *"Machine anchoring: pinned on JXP's Mac (darwin,
2026-08-29) — a **different machine from the tank server that anchored the
elastic pins** above, so on any one machine one strict set may fail while the
other passes; the closeness tiers carry the guard everywhere (the finding
recorded in the M5 prompt doc's task-1 log)."* So the elastic pins were
anchored on the tank server and this Mac reproduces them only to ~3 ULP, which
is expected behaviour and is already documented on the coding side. That
demotes Q1 from "is the elastic route broken?" (it is not) to a single
bookkeeping question: **should D1 task 7's "`pytest -q -ra` green" gate be read
as "green modulo the machine-anchored strict tiers"?** My recommendation: yes,
and no repin — the numbers above stand as the measured evidence, and I will
record the two expected failures explicitly at task 7 rather than claim a
green suite. Ignore the repin recommendation in the paragraph above; I wrote
it before seeing the M5 note.

>A. Yes and no repin.


**Q2 (The branch moved under me, mid-task).** The working agreement says to
read the branch I am standing on and record its real name. I started task 1 on
**`inelastic-rt`** with a clean tree. Partway through task 2 you committed the
work in progress and checked out **`cdom-rt`**; `git log` shows my scaffold
landing in `834466f "docs and cdom"` (and `f1bd765 "docs and more"` before
it), and the tree is clean again. Nothing was lost — every file I created is
committed byte-for-byte as I wrote it, and I re-verified both gates afterwards
on `cdom-rt`.

Two things I did not decide for you. (i) DocQ8 and this document's working
agreements both say the docs work belongs on a **fresh branch off `main` once
`inelastic-rt` merges**; it is now on `cdom-rt` instead, alongside the CDOM
work. That is your call, not mine, but it means the docs commits and the CDOM
commits will land together. (ii) `.readthedocs.yaml` is committed on a branch,
so the first RTD build (D1 task 7's real gate) will only happen once that
branch is pushed and RTD is pointed at it — RTD builds the project's default
version, usually `main`.

My recommendation: no action needed from me either way; just confirm which
branch you want the remaining D1 tasks (3–7) to be written on, and I will read
it fresh at the start of each turn rather than assuming.

>A. We are using cdom-rt for the rest.  I'll handle all git commands.

**Q3 (`-W` does *not* catch a broken `{func}`/`{class}` cross-reference, which
is what D2 task 2's gate says it does).** Raised at D1 task 3, where the
strictness proof turned it up. Task 3 asked me to prove `-W` is genuinely
strict by introducing a bad cross-reference, and offered "a `{func}` role
pointing at a nonexistent target" as the example. Measured, in a scratch copy
of `docs/`, all three with `-W --keep-going`:

```
{doc}`no_such_page_at_all`                    → WARNING [ref.doc]        EXIT=1
toctree entry model/this_page_does_not_exist  → WARNING [toc.not_readable] EXIT=1
{func}`robust.rt.no_such_function_whatsoever` → no warning              EXIT=0
  ... the same file with -n added             → WARNING [ref.func]      EXIT=1
```

So `-W` is strict about documents, toctrees and images, but a **typo'd Python
cross-reference renders as plain text and passes**, because Sphinx's `nitpicky`
is off by default and `conf.py` does not turn it on. Task 3's own gate is
satisfied by the first two (and I logged it that way), so this blocks nothing
today. It matters at **D2 task 2**, whose gate states as fact: *"every
`:func:`/`:class:` cross-reference resolves — with `-W`, a typo'd role is a
build failure, which is the point."* That is currently false, and the model
chapters are exactly the pages that will be dense with such roles.

Three options, in the order I'd rank them:

1. **Turn on `nitpicky = True` in `conf.py` at D1 task 5** (the API page task,
   which is where the API anchors first exist to point at), with a
   `nitpick_ignore` list for the unavoidable misses. This makes D2 task 2's
   gate mean what it says. The cost is real and should not be waved away:
   nitpick also flags every unresolvable *type annotation* in autodoc output,
   and with `autodoc_typehints = "description"` plus `jaxtyping`'s
   `Float[Array, "*batch wave"]` annotations, the first nitpicky build will
   almost certainly produce a wall of `py:class reference target not found`
   that has to be triaged into `nitpick_ignore` before it goes green.
2. **Add a narrower CI step** that greps the built HTML for roles that fell
   back to plain text, rather than flipping nitpick globally — catches the
   typos without fighting jaxtyping.
3. **Leave it, and reword D2 task 2's gate** to "every cross-reference
   spot-checked in the rendered HTML" — honest, but manual, and it will rot.

My recommendation: **(1), attempted at task 5, with a hard fallback to (3) if
the `nitpick_ignore` list turns into an unbounded jaxtyping-shaped
whack-a-mole** — the decision point being "does the ignore list stay short
enough to read". I'd rather find that out at task 5, when the API page is the
only autodoc surface, than at D2 task 2 with five chapters already written
against a gate that cannot enforce them. Either way D2 task 2's wording needs
a small edit, since as written it asserts a behaviour we have now measured to
be absent. Which do you want?

>A. Let's go with your recommendation.

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

### 2026-08-29 (D1 task 1 — scaffold and `conf.py`; strict build green, zero warnings)

**Branch.** Started on `inelastic-rt`, working tree clean (`git status`
verified before the first write). JXP committed the work and moved the
checkout to `cdom-rt` while task 2 was running — see Q&A Q2; nothing was lost.

**Created** (all new, none pre-existing): `docs/conf.py`, `docs/index.md`,
`docs/Makefile`, `docs/requirements.txt`, `docs/api.rst`,
`docs/installation.md`, `docs/quickstart.md`, `docs/references.md`,
`docs/model/overview.md`, and `.gitkeep` placeholders in `docs/_static/`,
`docs/_templates/`, `docs/using/`, `docs/reports/`, `docs/figures/` (git does
not track empty directories, and the task asked for the directories the page
set needs). `docs/member_policy.md` **was not moved or edited** — it is
reached through the Reference toctree as `Team <member_policy>`, and the
rendered navbar confirms the retitle took (`member_policy.html">Team`).

**`conf.py`.** Built to the spec: `sys.path.insert(0, "..")`; `project =
"retrieve-or-bust"`; `release` from `robust.__version__` in a `try/except`
falling back to `"0.0.dev0"`, `version` = first two dotted fields; the nine
extensions with `myst_nb` and *not* `myst_parser`; `source_suffix` mapping
`.rst`/`.md`/`.ipynb`; `nb_execution_mode = "off"`; `suppress_warnings =
["myst.xref_missing"]`; `napoleon_use_ivar/use_param/use_rtype = True` with
both docstring styles on; the four `autodoc_default_options` plus
`autodoc_typehints = "description"` and `autosummary_generate = True`;
`autodoc_mock_imports = []` with the comment naming *why* the empty list is
safe (`robust.rt` imports only `jax`, `jaxtyping`, `numpy` at module level);
the five intersphinx targets; `pydata_sphinx_theme` with a deliberately small
`html_theme_options`.

**Deviations from the plan, all recorded rather than papered over:**

1. **`linkify` had to go.** I initially enabled six
   `myst_enable_extensions`; `linkify` needs the separate `linkify-it-py`
   package and myst-nb *raises* at build time without it
   (`ModuleNotFoundError: Linkify enabled but not installed.`) — the first
   build died there. Rather than add a dependency nothing needs, I dropped it
   and left a comment in `conf.py` saying so. The enabled set is now
   `dollarmath`, `amsmath`, `colon_fence`, `deflist`. The spec required "at
   least `dollarmath` and `colon_fence`"; both are in.
2. **`myst_heading_anchors = 3`** added (not in the spec) so the long D2
   chapters can link to each other's subsections.
3. **The theme did *not* fight `-W`.** The fallback to `sphinx-book-theme`
   was not needed and was not taken. `pydata-sphinx-theme` 0.21.0 built clean
   on the first attempt with the minimal option set.
4. **Navbar shape, for task 6 to revisit.** pydata renders top-level toctree
   *documents* as navbar tabs, so right now the tabs read "Installation /
   Quickstart / The model in one page / API reference / References / Team"
   rather than the five section names. That is a consequence of every section
   currently being a flat list of stubs; it will resolve itself when tasks 4–6
   and D2 give the sections real children. Flagging it so nobody is surprised.

**`.gitignore`** gained `docs/_static/fig_*.png` and
`docs/reports/report_rt_*.md` (with a comment explaining they are byte-derived
from committed `reports/` files, so a second committed copy would only create
drift). `docs/_build/` was already ignored.

**Gate, run exactly as specified.**

Step 1 — the dry-run, before installing anything:

```
$ /Users/xavier/miniforge3/envs/ocean14/bin/python -m pip install --dry-run -r docs/requirements.txt
...
Would install SQLAlchemy-2.0.52 jupyter-cache-1.0.1 mdit-py-plugins-0.6.1
  myst-nb-1.4.0 myst-parser-5.1.0 pydata-sphinx-theme-0.21.0
  sphinx-copybutton-0.5.2 sphinx_design-0.7.0 tabulate-0.10.0
```

Purely additive: nine new packages, **no** "Would uninstall" and no upgrade
line anywhere in the output; everything else reported "Requirement already
satisfied". `sphinx>=8` was satisfied by the Sphinx 9.1.0 already in
`ocean14`, and `jax`/`jaxtyping`/`numpy`/`flax`/`optax` were all already
present, so the editable `bing`/`ocpy` checkouts were never at risk. Installed
for real after that.

Step 2 — the build:

```
$ python -m sphinx -b html -W --keep-going docs docs/_build/html
Running Sphinx v9.1.0
loading translations [en]... done
[autosummary] generating autosummary for: api.rst, index.md, installation.md,
  member_policy.md, model/overview.md, quickstart.md, references.md
loading intersphinx inventory 'numpy'      from https://numpy.org/doc/stable/objects.inv ...
loading intersphinx inventory 'scipy'      from https://docs.scipy.org/doc/scipy/objects.inv ...
loading intersphinx inventory 'python'     from https://docs.python.org/3/objects.inv ...
loading intersphinx inventory 'matplotlib' from https://matplotlib.org/stable/objects.inv ...
loading intersphinx inventory 'jax'        from https://docs.jax.dev/en/latest/objects.inv ...
...
build succeeded.

The HTML pages are in docs/_build/html.
EXIT=0
```

**Zero warnings** — the only two lines in the whole log matching
`warning|error` are field *names* inside myst's config dump
(`suppress_warnings=[]`, `execution_allow_errors=False`), not messages. Clean
build from an empty `_build/` takes **1.38 s wall** (`/usr/bin/time -p`: real
1.38, user 0.79, sys 0.15) for 7 pages.

**Intersphinx verified rather than assumed**, as the spec asked — each
`objects.inv` fetched and decoded independently:

```
python       OK | Project: Python            | Version: 3.14     | 19319 objects
numpy        OK | Project: NumPy             | Version: 2.5      |  8274 objects
scipy        OK | Project: SciPy             | Version: 1.18.0   | 11924 objects
matplotlib   OK | Project: Matplotlib        | Version: 3.11.1   | 11399 objects
jax          OK | Project: JAX               | Version: (blank)  |  4146 objects
```

All five resolve. (Aside worth keeping: numpy.org returns **403** to a bare
`urllib` User-Agent — it only served the inventory once I sent a Sphinx UA.
Sphinx's own fetch is fine, but a naive link-checking script would report a
false failure there.)

**Rendered-HTML spot checks** (because "the build exited 0" is not the same as
"the page is right"): `pydata-sphinx-theme` classes present in `index.html`;
the light/dark `theme-switch-button` present; the GitHub `icon_links` URL
present; `$R_{rs}(\lambda)$` rendered as MathJax (`R_{rs}` in the output); the
three toctree captions ("Getting started", "The model", "Reference") all
rendered; `sphinx-copybutton`'s CSS/JS copied into `_static/`; and
`member_policy.html` reached from the sidebar as **Team**.

**What I learned.** (a) The `-W` gate really is unforgiving in the useful way:
the one thing that broke was an extension I added on my own initiative, and it
broke *loudly* at build start rather than silently degrading. (b) `nitpicky`
is off by default, so a `{mod}`/`{func}` role pointing at a not-yet-documented
Python object does *not* warn — which means the API cross-references D2 relies
on will not be gate-enforced unless we turn nitpick on. I avoided the issue in
`index.md` by using plain code formatting for `robust.rt.forward()`, but this
is worth a decision at task 5. (c) Every page in the tree is in a toctree, so
there is not one orphan warning; the `.gitkeep` files are not source files and
Sphinx ignores them.

### 2026-08-29 (D1 task 2 — `robust.__version__` single-sourced, `.readthedocs.yaml`, RTD build rehearsed in a clean 3.12 venv)

**Files touched.** Two lines of `robust/` in spirit, nine in fact:
`robust/__init__.py` gains `__version__ = "0.0.dev0"` plus a comment block
saying it is the single source of truth and naming both consumers. `setup.py`
gains a `get_version()` helper that **regex-parses the literal** out of
`robust/__init__.py` — `re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
..., re.M)`, raising `RuntimeError` if it ever fails to match — and
`setup_keywords['version']` now calls it instead of repeating `'0.0.dev0'`.
Parsing rather than importing matters concretely and not just theoretically:
pip builds in an isolated environment where `jax` is absent, so `import
robust` at build time would have broken `pip install .` — and the rehearsal
below is where that would have surfaced. New root `.readthedocs.yaml`
(dotted), exactly the block DocQ3 specified: `version: 2`, `build.os:
ubuntu-24.04`, `build.tools.python: "3.12"`, `sphinx.configuration:
docs/conf.py` + `builder: html` + `fail_on_warning: false` with the comment
explaining that the strict gate lives in CI, and `python.install` = `{method:
pip, path: .}` then `{requirements: docs/requirements.txt}`. **No `formats:`
block**, per the spec. No deviations from the plan in this task.

**Gate, every sub-check.**

```
$ python -c "import robust; print(robust.__version__)"
0.0.dev0
$ python setup.py --version
0.0.dev0
$ python -c "import yaml; print(yaml.safe_load(open('.readthedocs.yaml')))"
{'version': 2, 'build': {'os': 'ubuntu-24.04', 'tools': {'python': '3.12'}},
 'sphinx': {'configuration': 'docs/conf.py', 'builder': 'html',
            'fail_on_warning': False},
 'python': {'install': [{'method': 'pip', 'path': '.'},
                        {'requirements': 'docs/requirements.txt'}]}}
$ pip install -e . --no-deps
Created wheel for retrieve-or-bust: filename=retrieve_or_bust-0.0.dev0-0.editable-py3-none-any.whl
Successfully installed retrieve-or-bust-0.0.dev0
```

The wheel *filename* carrying `0.0.dev0` is the proof that the regex ran
inside pip's isolated build environment.

**The RTD rehearsal — and the one place I had to substitute.** The gate says
"a throwaway `python3.12 -m venv`". There is **no `python3.12` on this
machine** (`ocean14` is Python **3.14.6**; the system interpreter is 3.9.6; no
`uv`, no Homebrew `python@3.12`, no python.org framework build). I therefore
created a throwaway conda prefix at `<scratchpad>/rtd312` with
`conda create -y -p ... python=3.12 --no-default-packages` (→ Python
**3.12.14**) and then made a genuine `python3.12 -m venv` from *that*
interpreter at `<scratchpad>/rtd-venv`. The venv started with exactly one
package (`pip 25.0.1`), so it is as clean as the gate intends; only the
provenance of the 3.12 binary differs from the letter of the instruction.

Then the two install steps the YAML declares, in order, and the build the way
RTD runs it (lenient, matching `fail_on_warning: false`):

```
$ <rtd-venv>/bin/python -m pip install .
Successfully built retrieve-or-bust
Successfully installed ... retrieve-or-bust-0.0.dev0 ...   (49 packages)

$ <rtd-venv>/bin/python -m pip install -r docs/requirements.txt
Successfully installed ... jax-0.11.1 jaxlib-0.11.1 jaxtyping-0.3.11
  flax-0.12.9 optax-0.2.8 sphinx-9.1.0 pydata-sphinx-theme-0.21.0
  myst-nb-1.4.0 myst-parser-5.1.0 sphinx-design-0.7.0
  sphinx-copybutton-0.5.2 ...                                (78 packages)

$ <rtd-venv>/bin/python -c "import robust; print(robust.__version__); import robust.rt"
0.0.dev0
robust.rt imported OK

$ <rtd-venv>/bin/python -m sphinx -b html docs <scratchpad>/rtd-build
build succeeded.       EXIT=0     (zero WARNING/ERROR lines in the log)

$ <rtd-venv>/bin/python -m sphinx -b html -W --keep-going docs <scratchpad>/rtd-build-strict
build succeeded.       EXIT=0
```

128 packages in the finished environment. **What the rehearsal actually
taught us, which is the point of doing it:**

1. **`import robust.rt` succeeds with no `ocpy` anywhere in the environment,
   no `$OS_COLOR`, and no L23 data.** That is the DocQ2 no-mocking decision
   validated on a real RTD-shaped environment rather than inferred from a
   grep. `ocpy` is not in the 128-package list and nothing asked for it.
2. **`pip install .` does not pull the JAX stack** — `setup.py`'s
   `install_requires` deliberately omits it — so jax arrives only via
   `docs/requirements.txt`. The two install steps are genuinely
   complementary; dropping either would break the build.
3. The **version single-sourcing survives the round trip**: the built site
   carries `VERSION: '0.0.dev0'` in `_static/documentation_options.js` and
   `# Version: 0.0` in `objects.inv`, i.e. `release` and `version` both
   derived from the one literal.
4. **No surprises in the resolved versions**: the venv got jax/jaxlib
   **0.11.1** where `ocean14` has 0.11.0, and NumPy 2.5.2 vs 2.4.6 — but the
   docs build touches neither numerically, and both builds are byte-clean.

**Post-edit test run** (the gate asks for it), compared against the baseline I
deliberately took *before* the first `robust/` edit:

```
before:  2 failed, 451 passed, 1 skipped in 57.30s
after:   2 failed, 451 passed, 1 skipped in 58.14s   (same two test names)
$ ruff check robust/        All checks passed!
$ ruff format --check robust/   32 files already formatted
```

Identical. The two failures are **pre-existing** — `test_inelastic_types.py::
test_elastic_hash_regression_strict` and `test_inelastic_validation.py::
test_gate_4_pre_change_pins`, both the same bitwise SHA-256 pin — and the
`__version__` addition provably did not cause them. This contradicts "Status
entering D1", which records the suite as green, so it is **Q&A Q1** with the
measured drift (max 3.0 ULP, closeness tier green) rather than something I
repinned quietly; the test file's own comment asks for exactly that. (Q1 then
picked up its own addendum: a concurrent CDOM/M5 edit to that same test file,
which landed in the working tree while I was writing this log, already records
the cause — the elastic pins were anchored on the **tank server**, not this
Mac. No repin needed; the open part is only how task 7 should word its
green-suite gate.)

**Stopping here**, per the turn's instruction: a genuine question for JXP
arose (Q1, plus the branch-move note as Q2), so tasks 3 and beyond are not
attempted. Task 2's own gate was nonetheless run in full and passed, since
none of its checks depend on the answer.

### 2026-08-29 (D1 task 3 — the CI `docs` job; `-W` proven strict, and a hole in that proof found)

**Branch.** `cdom-rt` (read fresh at the start of the turn, per the working
agreement). **Only file edited: `.github/workflows/ci.yml`.** `robust/` was not
touched — task 3 has no sanctioned `robust/` edit — and none of the concurrent
CDOM effort's uncommitted files were opened or written.

**Open questions re-checked, no new developments.** Q1 and Q2 both still stand
unanswered in this document (no `> A.` or equivalent has been added). Neither
blocks task 3: this task runs no pytest, so Q1's machine-anchored strict tiers
are not in its path, and Q2's own recommendation — read the branch you are
standing on — is what I did. Not re-litigating either here.

**What was added.** A third job, after `test` and `lint`:

```yaml
  docs:
    name: sphinx (-W)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.12', cache: pip}
      - name: Install the documentation toolchain
        run: pip install -r docs/requirements.txt     (after --upgrade pip)
      - name: Install the package
        run: pip install -e . --no-deps
      - name: Build the documentation strictly
        run: python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Every step carries a *why* comment, matching the file's established density.
The four things the comments record, because they are the things a future
reader will otherwise get wrong: (a) 3.12 is not a matrix here on purpose —
`.readthedocs.yaml` builds on ubuntu-24.04/3.12, so CI's job is to fail before
RTD does, on the same interpreter; a second version would be testing Sphinx,
not us; (b) `docs/requirements.txt` rather than the root file, the same
reasoning the `test` job already spells out about the two `git+` lines, plus
the fact that `autodoc_mock_imports = []` means autodoc genuinely imports
`robust.rt`, so the real stack must be present; (c) `--no-deps` mirrors the
`test` job — everything importable is already installed by the previous step,
and re-resolving could only pull a different jax; (d) **the job needs no
`$OS_COLOR` and no L23 data**, unlike `test`, because the docs build only
*copies* committed figures out of `reports/` — D2's `make_docs_figures.py`
defaults to copy mode and `--regenerate` is opt-in and never reachable from
`conf.py`. That comment names itself as the explanation for the day the job
breaks. (Worth noting for D2 task 6: right now there is nothing to regenerate
at all, since that script does not exist yet.)

**Gate, every sub-check, real output.**

1 — the workflow parses, and the job is shaped as specified:

```
$ python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"
PARSE_EXIT=0
jobs: ['test', 'lint', 'docs']
name: sphinx (-W) | runs-on: ubuntu-latest
  step: actions/checkout@v4
  step: actions/setup-python@v5 | {'python-version': '3.12', 'cache': 'pip'}
  step: Install the documentation toolchain | pip install -r docs/requirements.txt
  step: Install the package                 | pip install -e . --no-deps
  step: Build the documentation strictly    | python -m sphinx -b html -W --keep-going docs docs/_build/html
```

2 — **the venv was reused, not recreated.** The task-2 throwaway venv at
`<scratchpad>/rtd-venv` was checked before trusting it and was still healthy
(`Python 3.12.14`, `sphinx 9.1.0`, `robust 0.0.dev0`), so I reused it rather
than rebuilding — which also makes this a slightly *stronger* test than a fresh
venv, since it proves the job's steps are idempotent over an already-populated
environment. The three job steps run in order, verbatim:

```
$ <rtd-venv>/bin/python -m pip install --upgrade pip
Successfully installed pip-26.2.1            (was 25.0.1)
$ <rtd-venv>/bin/python -m pip install -r docs/requirements.txt
  ... every line "Requirement already satisfied"; nothing new resolved
$ <rtd-venv>/bin/python -m pip install -e . --no-deps
Successfully uninstalled retrieve-or-bust-0.0.dev0   (the task-2 non-editable install)
Successfully installed retrieve-or-bust-0.0.dev0     (now editable)
$ <rtd-venv>/bin/python -m sphinx -b html -W --keep-going docs docs/_build/html
build succeeded.        BUILD_EXIT=0
grep -cE "WARNING|ERROR" → 0
```

Zero warnings, exit 0. Note the editable install *replaced* task 2's plain
`pip install .` in that venv — expected, since the CI job installs `-e`, and
harmless in a throwaway.

3 — **`-W` proven genuinely strict, in a scratch copy** (`<scratchpad>/docs-break`,
a `cp -R` of `docs/`, so the repo tree was never dirtied — the copy was
confirmed to build clean first, `BASELINE_COPY_EXIT=0`). Two independent
deliberate breaks, each reverted by discarding and re-copying:

```
BREAK A — {doc} role at a nonexistent target
  index.md:51: WARNING: unknown document: 'no_such_page_at_all' [ref.doc]
  build finished with problems, 1 warning (with warnings treated as errors).
  EXIT=1

BREAK B — malformed toctree entry
  index.md:35: WARNING: toctree contains reference to nonexisting document
    'model/this_page_does_not_exist' [toc.not_readable]
  build finished with problems, 1 warning (with warnings treated as errors).
  EXIT=1
```

Both fail the build. Scratch copies deleted afterwards; the real tree rebuilt
clean (`FINAL_BUILD_EXIT=0`, zero WARNING/ERROR lines).

**The finding this task actually produced — and it contradicts D2's gate.** The
task text offered "a `{func}` role pointing at a nonexistent target" as the
canonical example of a bad cross-reference. I tried it, and **it does not fail
the build**:

```
BREAK C — {func}`robust.rt.no_such_function_whatsoever`
  build succeeded.        EXIT=0        ← no warning at all
same file, adding -n (nitpicky):
  index.md:51: WARNING: py:func reference target not found:
    robust.rt.no_such_function_whatsoever [ref.func]
  EXIT=1
```

This is task 1's log item (b) — "`nitpicky` is off by default, so a
`{mod}`/`{func}` role pointing at a not-yet-documented Python object does not
warn" — now demonstrated rather than inferred, and it matters more than it did
then, because **D2 task 2's gate asserts the opposite in writing**: "every
`:func:`/`:class:` cross-reference resolves — with `-W`, a typo'd role is a
build failure, which is the point." As configured today it is not. Raised as
**Q3**; it does not block task 3, whose gate asks only that `-W` be strict,
which breaks A and B establish.

**Deviations.** One, and it is mine to own: my first attempt at break B silently
did nothing — a heredoc read `os.environ['S']` while `S` was set but not
exported, so the Python raised `KeyError: 'S'` and the subsequent build
reported `BREAK_B_EXIT=0 / build succeeded`, which for about ten seconds looked
like evidence that toctree breaks pass `-W`. It was evidence that my edit never
landed. Re-run with `export S=...` plus an `assert new != t` guard on the
replacement, and it failed as it should. Recording it because a false green
from a no-op edit is exactly the failure mode this repo's "verify prose against
output" rule exists to catch, and I nearly wrote the wrong conclusion down.

**Assumption checked rather than trusted**, as the turn asked: `ocean14` did
already have everything (`pip install --dry-run -r docs/requirements.txt`
produced **no** "Would install" and no "Would uninstall" line — every
requirement already satisfied from tasks 1–2). Nothing new was installed into
`ocean14` in this task.

**Tree state — and JXP committed mid-task again.** `git status` came back
completely clean at the end, which was momentarily alarming; `git log` explains
it: `02e3583 "docs and cdom"` landed while the build was running and swept up
my `ci.yml` edit along with the CDOM work. Verified byte-for-byte that nothing
was lost — `git show HEAD:.github/workflows/ci.yml` and the working-tree file
both carry the `docs:` job at line 95, `name: sphinx (-W)` at 96, and the
strict build command at 146. Same pattern as Q2 in task 2; no action needed,
noting it so the sequence is on the record.

**Stopping at task 3**, per the turn's instruction — task 4 not attempted even
though the gate is green.

### 2026-08-30 (D1 task 4 — Installation, Quickstart, Team, References; and a gradient that corrected my prose)

**Branch.** `cdom-rt`, read fresh at the start of the turn, per Q2's answer.
**Files written: three, all under `docs/`** — `installation.md`,
`quickstart.md`, `references.md`. `robust/` was not touched; task 4 has no
sanctioned `robust/` edit. `docs/index.md` needed **no change**: task 1 already
put all four pages in the front-page toctree, `Team <member_policy>` retitle
included, and the rendered sidebar confirms it (`member_policy.html">Team`).
`docs/team.md` was not created, as specified. None of the concurrent CDOM
effort's files were opened for writing.

**Q&A re-checked.** Q1–Q3 all now carry answers. Q1's "Yes and no repin" is
what shapes this task's only real judgement call: the suite is described on the
Installation page as green *modulo* two machine-anchored strict-hash tiers,
named explicitly, with the measured ULP evidence — never as "fully green". Q2
confirms the branch. Q3 is a task-5 decision and does not touch this task.

**`docs/installation.md`** (~250 lines): requirements and the 3.12 floor;
`git clone`; the dependency install with a `warning` admonition built around
the `git+` caveat; `pip install -e . --no-deps`; what `package_data` ships and
what breaks without it; CPU vs CUDA JAX; `$OS_COLOR` and what skips without it;
verification; and a short section on building the docs themselves.

**`docs/quickstart.md`** (~330 lines): environment check → load one L23 scene
from the committed 50-scene fixture → build `IOPs`/`PhaseParams`/`Geometry`
→ `forward()` → `inelastic=Inelastic()` → the 685 nm and 550–700 nm differences
→ `jax.grad` w.r.t. `a`, `a_ph` and `phi_C`. Seven numbered sections, every
snippet executed, every output pasted from that run.

**`docs/references.md`** (~75 lines): 15 entries in five thematic groups, each
copied from one of the two reports' References sections with a **Source**
column saying which (E / I / both). Diffed entry by entry against the source
files before pasting; the only edits are the reports' HTML subscripts
(`b<sub>b</sub>` → `b_b`), stated on the page.

**Three things measured rather than assumed, each of which changed what I
wrote.**

1. **The `git+` caveat is worse than "it replaces your checkouts".** The
   `--dry-run` shows the GitHub `ocpy` installs under a *different
   distribution name* (`ocpy-ocean` 0.1.0) than the local editable one
   (`ocpy` 0.1.dev0), so pip's metadata will happily hold both while only one
   wins on `import ocpy`. That is on the page because it is the failure mode
   that would waste an afternoon.
2. **"Without the weights, `mode='hybrid'` fails" is only half true.** The
   *emulator* weights raise `FileNotFoundError` (verified by pointing
   `emulator.DEFAULT_WEIGHTS` at a nonexistent path in a subprocess, and the
   real message is pasted). The two *correction-head* weights do not: they fall
   back to analytic-only with a single `MissingCorrectionWarning`, by design,
   because the analytic backbone is a legitimate model. The stub page inherited
   the loose version; the written page states both behaviours.
3. **Desiderio has no bibliography entry to copy.** The task text lists
   Desiderio (2000) among the references, but the name appears **only** in a
   code comment above `B_RAMAN_488` in `robust/rt/inelastic.py`, with no year
   or journal, and is in *neither* report's References section. Per the
   instruction to copy rather than compose, I did not invent one; the page
   carries a short "A name with no entry" section recording the gap instead.
   Flagging it here in case JXP wants the citation run down for D2.

**The prose defect this task caught in the act.** I wrote, from physical
intuition, that the nonzero `∂Rrs(685)/∂a` at 440 nm "is Raman: blue absorption
suppresses the photons that get redistributed into the red." Then I ran the
gradient with the processes switched on one at a time, because the house rule
says to. It is **fluorescence**, not Raman — the Raman-only gradient w.r.t. `a`
is *exactly zero* at 440 nm:

```
inelastic=                        d/da@440      d/da@685     d/da_ph@685
None (elastic)                   +0.0000e+00   -1.4788e-04   +0.0000e+00
Inelastic(fluorescence=False)    +0.0000e+00   -1.8272e-04   +0.0000e+00
Inelastic(raman=False)           -1.1346e-06   -2.0024e-04   +2.3789e-04
Inelastic()                      -1.1346e-06   -2.3508e-04   +2.3789e-04
```

Fluorescence excitation is broadband; Raman excitation is not. The Raman-only
`∂Rrs(685)/∂a` is nonzero at exactly three of the 81 grid points —

```
raman-only d/da nonzero at wavelengths: [555. 560. 685.]
values: [-3.0453191e-05 -4.1577973e-06 -1.8272053e-04]
```

— the emission band itself plus the two points bracketing
1/(1/685 nm + 3400 cm⁻¹) = **555.6 nm**, the Stokes excitation wavelength.
Re-derived here, not copied. The corrected explanation and this table are now
*in* the quickstart, because the wrong version was more plausible than the
right one, which is exactly when a reader needs the evidence. This is the
"numbers written before they were measured" defect, caught one step before it
was written down.

**Gate, every sub-check.**

1 — the strict build, from a deleted `_build/`:

```
$ python -m sphinx -b html -W --keep-going docs docs/_build/html
build succeeded.
EXIT=0     grep -cE "WARNING|ERROR" → 0
/usr/bin/time -p: real 1.17  user 0.81  sys 0.12    (7 pages)
```

2 — **every command in `installation.md` executed in this task.** Two
exceptions, both labelled *on the page* rather than faked: `pip install
jax[cuda12]` (no NVIDIA device here) and the full `pip install -r
requirements.txt` (run as `--dry-run`, since the real one would have replaced
the editable `bing`/`ocpy` checkouts — and the dry-run's output *is* the
evidence for the warning). Everything else ran, including `git clone` into the
scratchpad (`Cloning into 'retrieve-or-bust'...`, HEAD `a6acd35`), which is a
network fetch and changes no repository state.

3 — **every snippet in `quickstart.md` executed**, from one script whose blocks
are the page's code blocks verbatim. Then re-run under `env -u OS_COLOR`: the
output is **byte-identical**, which is the page's "needs no `$OS_COLOR`" claim
verified rather than asserted. Two claims that were prose until I checked them
are now checked: the batched call really does return `(150, 81)`, and
`phi_C · ∂Rrs/∂phi_C` really does equal `Rrs_fl(685) − Rrs(685)` to every
printed digit (`2.964558e-05` both ways) — the φ_C-linearity guarantee, in one
line.

4 — **all four pages reachable from the front-page toctree**, checked in the
rendered HTML rather than the source: every internal `href` on
`index/installation/quickstart/references.html` resolves to a file that exists
(28 distinct targets, zero missing). This check matters *here* specifically
because `conf.py` sets `suppress_warnings = ["myst.xref_missing"]`, so a broken
Markdown link would not have failed `-W`. Also confirmed rendered: the
`warning` and two `note` admonitions, five tables on the references page, one
on the quickstart, and `member_policy.html` reached as **Team**.

**The one deviation, and it is about the test counts.** The gate wants the
*actual* pass/skip counts. The tree moved three times underneath me while I
measured — the concurrent CDOM effort committed and then added files during the
turn — and the counts went `474 → 480 → 482 passed`. Worse, the last run picked
up a third failure,
`test_cdom_validation.py::test_cdom_gate_5_speed_within_twice_elastic`, from an
**untracked** in-progress file (`?? robust/tests/test_cdom_validation.py`) that
is not part of the committed package at all. Pasting that into an installation
page would have misled every reader. So the page carries the last matched pair
taken from a **single invocation** before that file existed —
`2 failed, 480 passed, 1 skipped` with `$OS_COLOR`, `2 failed, 445 passed, 36
skipped` without — plus an explicit note that the counts are a snapshot from a
tree under concurrent development and that the durable facts are the *shape*:
the two named machine-anchored failures, one pre-existing skip, and a ~35-test
skip delta. Recording it here so the next person knows the numbers are honest
and why they will not match exactly.

**Numbers re-measured in this task rather than carried over from Q1** (the
house rule applies to my own earlier logs too):

```
Rrs: differ 2742/12150 (22.6%), max rel 3.326e-07, max ULP 3
rrs: differ 2862/12150 (23.6%), max rel 1.642e-07, max ULP 2
```

Identical to task 2's figures; the Installation page quotes them as the
evidence that the two strict-hash failures are float32 platform drift and not a
broken install.

**Tree state at the end.** `git status --short` shows my three `docs/` pages
plus this log edit; `robust/rt/inelastic_corr.py`, `robust/rt/validation.py`,
`robust/tests/test_cdom_fl.py` and the untracked
`robust/tests/test_cdom_validation.py` are the concurrent CDOM effort's, not
mine — none was opened for writing here.

**Stopping at task 4**, per the turn's instruction: the gate is green and no
new blocking question arose, so task 5 was not attempted. The Desiderio gap in
note 3 above is a loose end for D2, not a blocker, and is recorded here rather
than raised as a Q&A question.
