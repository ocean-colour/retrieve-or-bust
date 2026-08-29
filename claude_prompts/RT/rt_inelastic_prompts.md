# Inelastic Radiative Transfer (RT)

This prompt doc will guide our efforts to develop inelastic radiative transfer (RT) models for ocean color.

## Context

- Read the `context/context_summary.md` file.
- Examine the code and Notebooks in the `Oceanography/python/bing/` repository with emphasis on its treatment of inelastic RT processes: fluorescence and Raman scattering.


Read these papers (PDFs) in `context/RT/` for fluorescence:

- `gordon1979.pdf` : An early paper on Chlorophyll Fluorescence in Seawater
- `maritorena2000.pdf` : A paper on the determination of the fluorescence quantum yield by oceanic phytoplankton in their natural habitat
- `Behrenfeld2009.pdf` : A paper on the satellite-detected fluorescence reveals global physiology of ocean phytoplankton

Read these papers (PDFs) in `context/RT/` for Raman scattering:
- `bartlett1998.pdf` : A paper on Raman scattering by pure water and seawater
- `desiderio2000.pdf` : A paper on the application of the Raman scattering coefficient of water to calculations in marine optics

And this chapter:
- Ocean Optics Web Book: https://www.oceanopticsbook.info/view/scattering/level-2/raman-scattering

## Code

If you need to run Python use the `ocean14` conda environment. 

## Prompts

### Setup

1.  Let's start from the work done by JXP in the BING repo.  Please review that work. We will write a report named `context/rt_inelastic_bing_summary.md`.  Before doing so, please have a conversation with me in Q&A/Setup.  Use Fable if you can.  Log your work. 

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Ask another round of questions if needed.  Use Fable if you can.  Log your work.

3. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Now proceed to write the report.  Be sure to include figures.  Generate them with Python and save the script(s) to the `context/RT/` folder. Use Fable if you can.  Log your work. 

4. I decided to move the report into the `context/RT/` folder.  Please update it so the figures render properly.  Use Fable if you can.  Log your work.

### BING

1. You have identified a few errors in the BING implementation of inelastic RT.  We are going to fix them.  Please ask me a set of questions in Q&A/BING fixes before fixing them.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Then proceed to fix the errors.  Use Fable if you can.  Log your work.

### Design

1. Ok, let's generate a plan together for how to proceed.  We are going to write a design document for this elastic RT effort.  We will name it `design/rt_inelastic_model.md`.  Please ask me a set of questions in Q&A/Design before writing the document.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Ask another round of questions if needed.  Use Fable if you can.  Log your work.

3. Please proceed to write the design document.  Use Fable if you can.  Log your work.

4. The design document looks great!  We will now generate a separate coding plan with a staged implementation plan including milestones.  Name it `design/rt_inelastic_model_coding_plan.md`.  Please ask me a set of questions in Q&A/Coding before writing the document.  Use Fable if you can.  Log your work.

5. I have answered your questions in the Q&A section below.  Please review them and proceed to write the coding plan.  Use Fable if you can.  Log your work.

6. The coding plan looks great!  Please generate a series of prompt docs to execute the coding plan.  Name them `claude_prompts/RT/rt_inelastic_coding_prompt_1.md`, `claude_prompts/RT/rt_inelastic_coding_prompt_2.md`, etc.  Model them after the ones for the elastic RT effort. Use Fable if you can.  Log your work.

7. Re-read the design document and the coding plan and the Inelastic Report. We have thus far neglected CDOM fluorescence.  We will now add it to the model.  Please ask me a set of questions in Q&A/CDOM before writing the document.  Use Fable if you can.  Log your work.

8. Read my answers to the Q&A/CDOM section below. Proceed to write the prompt document and update the high-level design document and coding plan.  Use Fable if you can.  Log your work.

### Report

1. We have worked through all of the prompt docs.  Please write a report summarizing the work we have done.  Name it `report/rt_elastic_model.md`.  Please ask me a set of questions in Q&A/Report before writing the document.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them.  Then proceed to write the report.  Use Fable if you can.  Log your work.

### Docs

1. Generate a full set of documentation for the model.  Model it after the BING, PAB, and IOPtics docs.  These will appear on ReadTheDocs.  Please ask me a set of questions in Q&A/Docs before writing the documentation.  Use Opus.  Log your work.

2. See my answers to the Q&A/Docs section below. Proceed to generate the prompt docs.  Use Opus.  Log your work.

## Comments

## Q&A

### Docs

Questions from Claude (2026-08-29, model: Opus) before writing the ReadTheDocs
documentation. Context reviewed: the `robust/` package (11 modules in
`robust/rt/`, ~5,900 lines; `forward()` plus the `IOPs`/`PhaseParams`/
`Geometry`/`Inelastic` pytrees are the public surface), the design docs and
implementation records, both reports, the notebooks, and — in detail — the
three doc sets the prompt names as models. Two findings shape most of what
follows. First, **retrieve-or-bust has no docs scaffold at all**: `docs/`
exists but holds exactly one file, `docs/member_policy.md` (team governance,
not technical), with no `conf.py`, no `.readthedocs.yaml`, no Makefile and no
docs job in `.github/workflows/ci.yml`. This is greenfield. Second, **the
three exemplars do not agree** — all are Sphinx published on ReadTheDocs, but
they differ on layout, theme, and markup, so "model it after BING, PAB, and
IOPtics" needs a pick (DocQ2). One piece of good news measured rather than
assumed: of ~115 public classes/functions in `robust/rt/`, only **7 lack
docstrings** and every module has a module docstring — autodoc will work
essentially out of the box. Each question carries a recommendation.

**DocQ1 (Scope — what "the model" means here, and for whom).** The prompt
sits under the inelastic effort, but `forward()` is a *single* entry point
whose inelastic terms are meaningless without the elastic backbone they
multiply and add onto — a Raman-and-fluorescence-only site would have to
re-explain `Rrs_ZTT + ΔRrs` on page one anyway. My recommendation:
**document `robust.rt` as one complete forward model** — conventions, the ZTT
backbone, the residual emulator, the hybrid, Ed, the inelastic terms, the
correction heads, the baselines and the validation protocol — with the
inelastic material as the deepest chapters rather than the whole site. Two
consequences to confirm: (i) the second package `rob/` (only `__init__.py`
plus `data/Dutkiewicz2015`) is **excluded** for now; (ii) the site states
plainly that **the inversion does not exist yet** — this is a forward model
— the way IOPtics' front page says outright what has and has not been
measured. Audience: ocean-colour researchers who might use or check the
forward model, plus Frouin and future sessions — i.e. the report's audience,
one notch more practical. OK?

**DocQ1-answer:** . Yes, be complete and ignore Dutkiewicz2015.  And that is the correct audience, but do provide a basic introduction too.

**DocQ2 (Tooling — the three exemplars disagree; which do we follow?).**
What I found, concretely:

- **BING** — `docs/` flat, `docs/conf.py`, `sphinx_rtd_theme`, **`.rst`
  only** (no MyST), root `readthedocs.yaml`, a separate
  `docs/requirements.txt`, pdf+epub formats, version hardcoded in `conf.py`.
  ~5,850 lines: 15 hand-written narrative pages plus **seven** per-module
  `api/*.rst` files.
- **IOPtics** — `docs/source/` layout, **furo** theme (light/dark),
  `autosummary` + `autodoc_mock_imports`, `.readthedocs.yaml`, **one**
  `api/index.rst` built from `automodule` blocks, generated report pages
  under `source/reports/`, figure-generator scripts in `docs/figures/*.py`
  writing into `_static/`, and `release` read from `ioptics.__version__`.
  Its Markdown design docs sit in `docs/design/` and are deliberately **not**
  rendered (no MyST is loaded).
- **PAB** — `docs/` flat, `sphinx_rtd_theme` (alabaster fallback), and
  crucially **`myst-nb`**, so the Markdown design docs (`docs/design/*.md`)
  *and* nine executed notebooks (`docs/nb/*.ipynb`, `nb_execution_mode =
  "off"`) render into the site untouched; plus `sphinxcontrib-mermaid`, docs
  dependencies single-sourced in the root `requirements.txt`, and a *second*
  Sphinx target (`report_site/`) for the community-facing results.

My recommendation: **PAB's shape as the primary template** — flat `docs/`
with `docs/conf.py`, `sphinx_rtd_theme` (2 of 3 use it), and **MyST enabled**,
because every piece of source material we already own (two reports, four
design/plan docs, ten notebooks) is Markdown or `.ipynb`, and MyST is the
difference between reusing it and transcribing it into `.rst` — with
IOPtics' *API pattern* (a single `automodule` page, not BING's seven) and
IOPtics' version single-sourcing. Deviating from PAB on one point: docs deps
in a `docs/requirements.txt` (BING/IOPtics style) rather than the root
`requirements.txt`, so the RTD build does not drag in the JAX stack it does
not need — though autodoc *does* need `jax`/`flax`/`jaxtyping` importable
unless we mock them, which brings us to a real choice: **install the package
on RTD (honest signatures) or mock the JAX stack (fast, green builds)?** My
lean is **install** — `jax` is a plain CPU wheel and the type annotations are
`jaxtyping` shapes worth rendering. Agree with PAB-shape + MyST + rtd-theme,
or would you rather have IOPtics' furo look?

**DocQ2-answer:** . Yes, PAB's shape is the primary template.  And I would install the JAX stack on RTD.  Do use a different style than PAB and IOPtics.  Pick something cool.

**DocQ3 (Where it lives, `member_policy.md`, and the RTD project itself).**
I propose the Sphinx site at **`docs/`** (flat, per DocQ2) with a new root
**`.readthedocs.yaml`** (dotted — BING's undotted `readthedocs.yaml` also
works, but the dotted name is the current convention and is what PAB and
IOPtics use), `build.os: ubuntu-24.04`, `python: "3.12"` (the `setup.py`
floor), and `fail_on_warning: false` on RTD while CI carries the strict gate
(DocQ8). The existing `docs/member_policy.md` then sits inside the Sphinx
source tree and must be either rendered or excluded; my lean is to **render
it** as a short "Team" page (it is a real project policy, and MyST renders it
as-is), rather than move it or hide it in `exclude_patterns`. Two things only
you can answer: (i) has a ReadTheDocs project already been created for
retrieve-or-bust — and if so under what slug/URL — or do you need to create
it? (ii) the repo is public but the docs will be *the* public face of this
work well before any paper; any content you want kept off a public site?

**DocQ3-answer:** I have just created the project on ReadTheDocs.  The slug is `retrieve-or-bust` and the URL is `https://retrieve-or-bust.readthedocs.io/`.  I would render the `member_policy.md` page.  And I would not keep any content off the public site.

**DocQ4 (The page set — narrative vs API reference).** Proposed table of
contents, sized between PAB's ~1,000 narrative lines and BING's ~5,850:

- *Getting started* — Installation (the `requirements.txt` / `pip install -e
  . --no-deps` dance, the JAX stack, `$OS_COLOR` and the skipping tests) and
  a Quickstart that computes `Rrs` for one L23 scene, toggles
  `inelastic=Inelastic()`, and takes a gradient.
- *The model* — Conventions (`Rrs`↔`rrs`, the 81-point grid, `bb_w`); the
  ZTT backbone; the residual emulator; `forward()` and the composition law;
  Ed; the inelastic terms (Raman + Chl-a fluorescence); the correction heads;
  the baselines (Gordon, PR05/O25).
- *Using it* — Data (L23), Validation & accuracy (the gate numbers), and a
  prominent **Scope and limitations** page.
- *Reference* — one autodoc API page, References, Team.

Two calls inside that. (i) **One `automodule` API page (IOPtics) rather than
seven per-module pages (BING)** — `robust/rt` is 11 modules and the
docstrings are already thorough; and I would fill the 7 missing docstrings
(3 in `emulator.py`, 2 in `ztt.py`, 1 each in `inelastic.py` and
`validation.py`) as part of this effort, which is a small edit to `robust/`
— confirm you want code touched at all here. (ii) The **limitations page is
not buried**: it quotes the measured caveats verbatim from report §5 — the
−74 % unseen-zenith cliff, official support only for λ ≥ 400 nm, φ_C truth
at 0.02 only, `'double'` emission unvalidatable — at the reports' level of
bluntness. Right page set, right bluntness?

**DocQ4-answer:** Yes, that's fine.

**DocQ5 (Design docs, reports, and provenance — rendered, linked, or
omitted).** The exemplars split: PAB renders its design docs *and* its
implementation record and full-run report inside the site; IOPtics keeps
design Markdown out of `source/` (unrendered) but renders generated report
pages; BING renders neither. My recommendation, a middle path: **render the
two reports** (`reports/report_rt_elastic_model.md`,
`report_rt_inelastic_model.md`) as a "Reports" section — they are already
public-facing, figure-complete narrative and are the best evidence the model
works — and **link out to GitHub** for the design docs, coding plans and the
1,469-line implementation record, which are internal decision records full of
repo-relative links and would roughly double the site. The Q&A/prompt-doc
provenance (this file included) is **not** published: the docs should read as
a user-facing manual, with the reports carrying the "how we know" story. Do
you want the design docs in the site too (PAB-style), or is linking right?

**DocQ5-answer:** Yes, that sounds right with your recommended linking.

**DocQ6 (The ten milestone notebooks).** `notebooks/RT/` holds ten executed
notebooks (five elastic, five inelastic) with narrative, figures and outputs
committed — genuinely good material, and PAB proves the mechanics work
(`myst-nb`, execution off). But they are *chronological build records*: they
open on environment installs and `sys.path` bootstrapping, are organized by
milestone, and would be a poor first read. My recommendation: **do not render
them as tutorials**; instead write **one new short quickstart notebook**
(load an L23 scene → `forward()` elastic → `forward(..., inelastic=...)` →
one gradient → plot) that *is* rendered, and link the ten from a one-line
"development record" note. Alternative if you want them all up: render them
under a clearly-labelled "Development record" caption where their milestone
framing is a feature. Which?

**DocQ6-answer:** Yes, create a new quickstart notebook and link to the ten from a one-line "development record" note.

**DocQ7 (Figures).** We already own 7 committed PNGs in `reports/` (two
architecture schematics, two rRMS ladders, two unseen-zenith figures, the
inelastic deltas) with two regeneration scripts, plus 8 more in
`context/RT/`. I propose the IOPtics pattern: a small `docs/figures/*.py`
that *copies or regenerates* the needed figures into `docs/_static/` rather
than committing a second set of bytes, reusing
`reports/make_inelastic_report_figures.py` where possible. For the front
page, both IOPtics (`ioptics_graphic.png`) and PAB (`pab_summary_light/
dark.png`) have a hero graphic; `reports/fig_inelastic_architecture.png`
could serve as-is. Recommendation: **reuse it for v1**, and only commission a
purpose-drawn hero if you want one. New hero graphic, or reuse?

**DocQ7-answer:** Yes, follow the IOPtics pattern.

**DocQ8 (Version single-sourcing, the CI gate, and the branch).** Three
small mechanics. (i) `robust` has **no `__version__`** (setup.py says
`0.0.dev0`); IOPtics and PAB both single-source `release` into `conf.py` from
the package while BING hardcodes it. I recommend adding
`robust.__version__` and reading it in `conf.py` — a two-line change to
`robust/__init__.py`. (ii) **CI**: BING and PAB both run a docs-build job
(`sphinx-build -W --keep-going`); IOPtics does not. I recommend adding one to
`.github/workflows/ci.yml`, strict (`-W`), with RTD itself lenient — so a
broken cross-reference fails a PR, never the published site. (iii) Per RQ7's
precedent the docs describe the **`main`** state; I assume the docs work
happens on a fresh branch off `main` after `inelastic-rt` merges, and that
you run all git as usual. Confirm all three?

**DocQ8-answer:** I confirm all three.

**DocQ9 (Effort and staging — this one is bigger than a 1–2 day item).**
For calibration: BING's site is ~5,850 lines of narrative, PAB's ~1,000 plus
notebooks, IOPtics' ~800 plus generated report pages. A full set for
`robust.rt` at PAB-to-BING density is **~2,000–2,500 lines across ~15 pages,
plus scaffold, figures and the API page** — realistically 2–3 days, not one
pass in an afternoon, and unlike the CDOM item it has no physics risk, only
volume. My recommendation: stage it as two milestones inside a single prompt
doc (`claude_prompts/RT/rt_docs_prompt_1.md`, mirroring how the coding
prompts worked) — **D1: scaffold, `.readthedocs.yaml`, CI job, API page,
Installation + Quickstart + a model-overview page, site building green and
published**; then **D2: the full narrative set, the reports section,
limitations, figures**. That way something real is on ReadTheDocs at the end
of day one and the prose lands against a working build. OK, or would you
rather I write the whole thing in one pass and hand you a finished site?

**DocQ9-answer:** Let's follow your recommendation.

### CDOM

Questions from Claude (2026-08-29) before writing the CDOM-fluorescence design
document. Context: I re-read the design doc (`design/rt_inelastic_model.md` —
the reserved `Inelastic.cdom_fl` slot in §3, the §8 landing zone and wishlist
item 3), the coding plan, the Inelastic Report (`reports/report_rt_inelastic_model.md`
— open item 2 and priority 4 both point here), and the assessment's §4.3 (per
A8, the origin of this prompt). Two facts verified fresh: **BING implements no
CDOM fluorescence** (a grep over `bing/rt/` and `bing/models/` finds CDOM only
in the elastic a_g absorption models — so, unlike Raman/Chl-fl, there is no
fixed-BING reference to port or cross-check against), and **no CDOM-fluorescence
truth data exist in L23** (the data folder on this machine,
`$OS_COLOR/Loisel2023/`, holds exactly the X∈{1,2,4} × {0°,30°,60°} pairs —
no CDOM-fl scenario; expected, per A5/DQ2). Those two absences shape most of
what follows. Each question carries a recommendation.

**CFQ1 (Document target).** Prompt 7 says "before writing the document" without
naming it. Options: (a) amend `design/rt_inelastic_model.md` in place —
its §3 reserved the `cdom_fl` slot for exactly this; or (b) a new standalone
`design/rt_cdom_fluorescence_model.md` that cites the inelastic design as its
companion, the way the inelastic design cited the elastic one. I recommend
**(b)**: the inelastic design is now a *finished decision record* — its §6 gate
has been declared PASSED and the shipped report cites it as-is — and rewriting
it retroactively muddies that record. The new doc would be short (the Ed
module, composition law, correction-head machinery, and conventions are all
inherited by reference), and §8 of the old doc gets only a one-line pointer
when the new one lands. OK?

**CFQ1-answer:** Yes, (b) is the correct answer.  

**CFQ2 (Truth data and what "add it to the model" means now).** With no truth
in hand, the effort splits cleanly: (i) what can be built and verified *now* —
the analytic CDOM-fl term, the interface, correctness/plausibility tests; and
(ii) what must wait for HydroLight on/off pairs — correction-head training and
any quantitative rRMS gate. My recommendation: design the full two-part
architecture now, **implement the analytic term and interface now, and gate
head training behind data arrival**, with the design doc carrying the concrete
run spec (CFQ8) so the runs can be commissioned in parallel rather than
sitting on the critical path. The direct question: do you have CDOM-fl
HydroLight runs planned or underway on a known timeline that should change
this — i.e., should this design assume truth arrives during the effort?

**CFQ2-answer:** Yes, let's follow your recommendation.  I will eventually have HydroLight runs, but not yet and not before we release this to main.

**CFQ3 (Architecture — mirror the Chl-fl shape).** I recommend CDOM-fl take
exactly the fluorescence term's shape from design §2/§4.4: an **additive**
emission term with a physics backbone and a bounded correction head,
`Rrs_cdom = s_C · K_cdom(IOPs, Ω, λ) · (1 + δ_C)`, composed as
`Rrs_total = (Rrs_ZTT + ΔRrs) × f_R + Rrs_fl + Rrs_cdom`. The head δ_C is
*defined* now (zero-initialized, so untrained head ≡ analytic backbone — the
same decay-to-physics property the other heads have) but ships **untrained**
in v1: there is nothing to train it on, and inventing pseudo-truth would
defeat the point. So v1 CDOM-fl is analytic-only by construction, with the
training pipeline stubbed and waiting. Agree?

**CFQ3-answer:**  I agree.

**CFQ4 (Physics basis — Hawes, the a_cdom input, and the UV clamp).** CDOM
emission is broad and featureless (no 685 nm-style line), so the Chl-fl
single-Gaussian machinery does not transfer. Options: (a) the **Hawes et al.
(1992) spectral fluorescence quantum-efficiency functions η(λ′, λ)** — the
parameterization HydroLight itself implements and the wishlist runs would use,
so any future truth is Hawes-consistent by construction (the same
controlled-experiment property that made X2/X4 so clean); or (b) a simpler
Coble/Gaussian-style EEM. I recommend **(a)**, as a fixed reference kernel
scaled by a differentiable amplitude, with the source term proportional to
**a_CDOM(λ′)** rather than a_ph. Two consequences to confirm: (i) `IOPs`
grows an optional `a_cdom` field (the a_ph pattern exactly — a physical
requirement, not an API whim; L23 stores a_g separately from a_nap, so the
existing loaders can populate it); (ii) CDOM excitation extends into the UV
below the 350 nm L23/IOP grid edge — a sharper version of the Raman clamp —
so I propose a hard 350 nm lower limit on the excitation integral in v1, with
the truncated fraction quantified from the Hawes functions and documented as
a caveat. OK on (a), the `a_cdom` field, and the 350 nm truncation?

**CFQ4-answer:** Yes, ok on all of these.

**CFQ5 (The `cdom_fl` slot's concrete shape — and off-by-default).** The
reserved slot is `cdom_fl=None`. Designing it concretely, I recommend:
(i) **`None` (off) stays the default even inside `Inelastic()`** — this is
load-bearing, because the L23 X4 truth *omits* CDOM fluorescence, so the
shipped 0.34 % gate and every report claim remain valid only if the default
model is CDOM-fl-free; (ii) when set, a small pytree (e.g.
`CDOMFl(scale=1.0)`: a differentiable amplitude on the Hawes reference kernel,
with room for shape metadata once truth exists) rather than a bare scalar;
(iii) the bit-identity regression *extends*: `Inelastic(..., cdom_fl=None)`
must be bit-identical to the current inelastic output, alongside the existing
elastic pin. OK?

**CFQ5-answer:** Ok.

**CFQ6 (Acceptance gate without truth — what "done" means).** The DQ6-style
held-out rRMS gate is impossible here, and so is the M2-style rtol ≤ 1e-6
BING cross-check (no BING implementation exists). I propose the v1 gate be:
(1) off-state bit-identity (CFQ5-iii); (2) implementation-correctness pins —
the Hawes η function reproduced against its published values, energy/quanta
bookkeeping unit tests, quadrature convergence; (3) a **literature-plausibility
band**, reported and gated loosely — the CDOM-fl contribution on L23 IOPs
lands in the published range (a few % of Rrs in the blue-green for CDOM-rich
scenes, ≲1 % oligotrophic), monotone in a_g(440); (4) gradient checks
including the new amplitude; (5) the composed forward stays within the
existing 2× elastic speed budget. The quantitative ≤ 5 %-style per-process
gate is *written* in the design now but explicitly conditional on the truth
runs landing (it becomes the head-training milestone's gate). Are those the
right bars for a truth-less v1?

**CFQ6-answer:** yes, those are the right bars for a truth-less v1.

**CFQ7 (Relationship to the coding plan and prompt docs).** The scope —
one analytic term, interface extensions, tests, no training — is a ~1–2 day
effort, not another week-long M0–M4 arc. I recommend: the new design doc
(CFQ1) carries its own small milestone table (an **M5**: analytic term +
interface + CFQ6 gate; and a *deferred* **M6**: head training + quantitative
gate, unblocked by truth arrival), executed by a **single prompt doc**
(`claude_prompts/RT/rt_cdom_coding_prompt_1.md`) — no separate coding-plan
document. The existing coding plan's "Beyond v1" section gets a pointer. As
always you run git; I'd suggest a fresh branch off `main` once `inelastic-rt`
is merged (the report's header already assumes that merge). OK, or do you
want the full design → coding plan → prompt-doc ceremony?

**CFQ7-answer:** Yes, that's ok.

**CFQ8 (Refining wishlist item 3 into a commissioning spec).** Design §8's
wishlist item 3 is one line; since CDOM-fl is now actively planned, I
recommend the new design doc expand it into a concrete run request: **paired
runs "X4 vs X4 + CDOM-fl"** (an X5-style scenario, so the difference isolates
CDOM-fl on top of the realistic ocean, exactly as X4−X2 isolated Chl-fl); a
**CDOM-stratified scene subset** spanning the full a_g(440) range with the
CDOM-rich tail oversampled (the sparse-tail lesson from δ_F); all three
zeniths (or the denser grid if wishlist item 1 runs at the same time); full
350–750 nm output; and the **exact Hawes quantum-efficiency function/version
recorded** so the model kernel matches the truth's constants. One packaging
question: should these be requested *jointly* with the geometry runs the
report made priority 2, to amortize the HydroLight setup, or kept as an
independent smaller request?

**CFQ8-answer:** Yes, request jointly.

### Report

Questions from Claude (2026-08-27) before writing the end-of-effort report.
Context: all five coding prompt docs are complete, the design-§6 acceptance
gate PASSES (held-out total rRMS 0.343/0.341/0.340 % vs X4 at 0/30/60° on
400–700 nm; worst per-process delta 1.03 %; gradients to 5.9e-9 incl. φ_C;
1.59× elastic runtime), and the elastic precedent is
`reports/report_rt_elastic_model.md` (313 lines, 3 figures via
`reports/make_report_figures.py` from the committed validation artifacts).
Each question carries a recommendation.

**RQ1 (Filename and location).** The prompt says `report/rt_elastic_model.md`,
but the repo directory is `reports/` (plural) and this is the *inelastic*
effort — the elastic report already occupies
`reports/report_rt_elastic_model.md`. I propose
**`reports/report_rt_inelastic_model.md`**, mirroring the elastic naming.
OK?

**RQ1-answer:** Ok

**RQ2 (Scope — prototype-focused or the full arc).** "The work we have done"
spans four phases: (i) the BING assessment quantified against L23
(`context/RT/rt_inelastic_bing_summary.md`); (ii) the BING fixes (the 1/π
fluorescence normalization, the true-Ed Raman ratio — landed in the BING repo,
PR issued); (iii) the design + coding plan; (iv) the M0–M4 prototype in
`robust/rt`. The elastic report covered only its prototype and *linked* to its
context/design docs. My recommendation: the same shape — the report's
deliverable is the prototype (the combined elastic+inelastic `forward()`),
with the assessment and the BING fixes compressed into the
motivation/background section as the findings that set the correction targets
(the ×π fluorescence error and the flat-Ed Raman error are worth stating as
results there, since they affect published BING-based fits), linking out for
the detail. Or do you want a full-arc report treating all four phases at
comparable depth?

**RQ2-answer:**  Right, don't worry about the BING fixes.

**RQ3 (Template, audience, and the headline claim).** I plan to mirror the
elastic report exactly: same section skeleton (Executive summary → motivation
→ the model → data & validation protocol → results → what it may claim and
may not → open items → recommended next priorities → reproducibility →
document map → references), same header block (team report, "intended to
evolve into a public (non-refereed) write-up", authors J. Xavier Prochaska
and Claude (Fable 5), version 1.0), same audience (you + Robert Frouin +
future sessions). One framing choice: the natural headline is the *complete*
forward model — `forward(..., inelastic=Inelastic())` reproduces the
all-processes-on L23 ocean (X4) to 0.34 % held-out rRMS, i.e. the elastic-era
accuracy now holds for the realistic ocean rather than an elastic idealization
— with the elastic 0.30 % quoted as the inherited base. Agree with that
framing?

**RQ3-answer:** I agree

**RQ4 (Figures).** Elastic precedent: `reports/make_report_figures.py`
regenerates the report figures from committed artifacts only (no recompute of
the science). I propose a sibling script `reports/make_inelastic_report_figures.py`
producing ~4 figures: (a) an architecture schematic of the composition law
`Rrs = (Rrs_ZTT + ΔRrs) × f_R(1+δ_R) + φ_C·K_fl(1+δ_F)`; (b) the rRMS ladder
per wavelength (elastic-only ~16 % → analytic backbone ~3 % → corrected
0.34 %, ungated regions shaded) from `rrms_per_wavelength_inelastic.csv`;
(c) the nine per-process delta rows analytic → corrected against the ±5 %
band (the −38.6 → −0.14 % headline at 0°) from `metrics_inelastic.csv`;
(d) the zenith-holdout caveat figure (the −74 % cliff at an unseen 60°). For
the BING-fix motivation I'd *reuse* the committed assessment figures in
`context/RT/` by link rather than regenerate. OK, or different figure
choices?

**RQ4-answer:** Ok

**RQ5 (Candor about the BING errors and the caveats).** Two calibration
checks for a document Frouin will read and that may become public: (i) the
report states plainly that the pre-fix BING fluorescence term overestimated
Rrs_fl by ~×π (so published `include_Chl_fl` fits carry an effective-φ_C
~π× smaller than nominal) and that flat-Ed was the dominant Raman error —
final-state framing per your A6 preference (the finding and the fix, no
debugging narrative). (ii) The "what it may not claim" section quotes the
sharp measured caveats verbatim: the −74 % zenith-holdout cliff, φ_C truth
at only 0.02, the sub-400 nm excitation clamp, `'double'` emission
unvalidatable. I recommend both, at the elastic report's level of bluntness.
OK?

**RQ5-answer:** Ok

**RQ6 (Closing recommendations).** The elastic report closed with
"Recommended priorities (M5)". For the inelastic close I propose: (1) the
inversion step — the forward model is complete and differentiable incl.
∂Rrs/∂φ_C, so IOP+φ_C retrieval is the natural next effort; (2) the
HydroLight run wishlist (design §8) prioritized by which caveat each run
retires (geometry coverage first, varied-φ_C second); (3) CDOM fluorescence
via the reserved `cdom_fl` hook once truth exists. Should the report go
further and recommend *which* to do first (my lean: inversion — the caveats
bind at unseen geometries, not on the L23 domain where the inversion work
would start), or lay out the options without a pick?

**RQ6-answer:** Yes, include a recommendation on what to do first.

**RQ7 (Branch/PR state in the header).** The elastic report's header cited
its branch and milestones. The inelastic work sits on `inelastic-rt` (M0 =
PR #14, M3 = PR #18 merged; M4 pending your PR). Should the report header
cite the branch + PR trail as-is, or do you plan to merge before the report
lands so it can cite `main`?

**RQ7-answer:** I will develop on the inelastic-rt branch and then merge it before the report lands.
Therefore, refer to `main`.

### Coding

Questions from Claude (2026-08-20) before writing
`design/rt_inelastic_model_coding_plan.md`. I re-read the elastic coding plan
(`design/rt_elastic_model_coding_plan.md`) and its implementation record to
mirror what worked; each question carries a recommendation.

**CQ1 (Milestone → prompt-doc mapping and time budget).** The design doc's §7
has five milestones M0–M4. The elastic effort used one prompt doc per
milestone plus a wrap-up (6 docs). I propose the same: five prompt docs
(M0 scaffold/API → M1 Ed + data → M2 analytic terms → M3 correction heads →
M4 validation/report), each with tasks + a hard gate, and a coarse "M5 /
beyond" section in the coding plan only (no prompt doc until v1 ships).
Time budget ~1 week of prototype work, as for elastic. OK?

**CQ1-answer:**  ok

**CQ2 (Machine and environment).** I verified just now that `ocean14` on this
machine (the tank server) has **no JAX** — the elastic implementation record
(jax 0.11.0 into `ocean14`, 2026-07-31) evidently describes your laptop.
Where will the inelastic prototype run? My proposal either way: M0 re-runs
the elastic install procedure (`pip install --dry-run` first, then
`pip install -r requirements.txt` — the JAX stack is already declared there)
into `ocean14` on whichever machine executes the prompts, with the same
purely-additive verification gate the elastic M0 used. If both machines will
run it, M0 should verify on both. What's your setup?

**CQ2-answer:** Use this machine and install what is needed.

**CQ3 (BING cross-check mechanics and the branch dependency).** Design §5
pins the JAX analytic terms against fixed-BING at rtol ≤ 1e-6. Two ways to
implement: (a) tests import `bing` directly — simple, but couples
retrieve-or-bust CI to the BING checkout *and specifically to the
`inelastic-fixes` branch*, which is not yet merged; (b) a one-off generator
script evaluates fixed-BING on the fixture scenes and commits the reference
outputs; robust tests compare against the committed references and never
import bing (mirrors the elastic effort's committed-fixture philosophy; CI
stays hermetic). I recommend (b). Related: should the coding plan list
"BING `inelastic-fixes` merged to main" as an M1 prerequisite, or is pinning
the branch/commit hash in the generator script enough?

**CQ3-answer:** Use (a). I am issuing a PR for the BING work now.

**CQ4 (CI fixture policy).** The elastic effort committed a 50-scene fixture
so CI runs real numbers without the data mount. For X2/X4 I propose a
*sibling* fixture file (same 50 scene indices, adds Rrs_X2/Rrs_X4/aph/Ed)
rather than regenerating the elastic fixture in place — the elastic tests
keep their bytes, and the new file is small (~100 kB; the BING 40-scene
fixture is 69 kB). OK?

**CQ4-answer:** Ok.

**CQ5 (Training scripts and artifacts).** Mirror the elastic layout:
training entry points as scripts in `design/py/` (e.g.
`train_inelastic_corr.py`), validation regenerated by extending
`design/py/run_validation.py`, metrics/figures into `design/validation/`,
and the trained head weights committed as
`robust/rt/files/{raman,fl}_corr_l23.npz`. The M4 report lands as
`design/prototype_summary` update or a separate short doc? My lean: extend
the existing validation artifacts and write the narrative in the
implementation record (`design/rt_inelastic_implementation.md`, mirroring
the elastic one), leaving `report/` for the end-of-effort report prompt. OK?

**CQ5-answer:** Ok.

**CQ6 (Branch and review cadence).** Elastic ran on branch
`rt-elastic-prototype` with a PR review pass late in the week (PR #11). Same
here: you create `rt-inelastic-prototype`, you run all git; the coding plan
schedules a review pass (e.g. `/code-review`-style) as an M4 task before the
gate is declared. And CI: the new tests live in the same pytest tree, so the
existing workflow picks them up automatically — no CI changes planned unless
you want a separate job for the (slower) training smoke test. OK?

**CQ6-answer:** Ok.

### Design

Questions from Claude (2026-08-19) before writing `design/rt_inelastic_model.md`.
Context I'm designing against: the elastic design (`design/rt_elastic_model.md`)
pinned `forward(iops, phase_params, geometry, wave) → Rrs` as a JAX hybrid
`Rrs_ZTT + ΔRrs` (held-out rRMS 0.30 %); the inelastic report
(`context/RT/rt_inelastic_bing_summary.md`) quantified the truth signals
(Raman: 5–20 % of Rrs; fluorescence: ~35 % at 685 nm) and the now-fixed BING
physics gives, with true Ed, Raman increment errors of +1 %/−4 % at zenith
30°/60° (−39 % at 0°) and fluorescence amplitude ratios 1.00/0.95/0.86.
Each question carries my recommendation; push back where you disagree.

**DQ1 (Architecture).** Decision time on the §5 trade-space. My
recommendation: **(c) physics + bounded learned correction**, mirroring the
proven elastic architecture — the (fixed) BING-style analytic terms become
the inelastic backbone, and a small residual network (trained on X2−X1 and
X4−X2) absorbs exactly the errors we measured and cannot fix analytically
(the high-sun two-flow failure, the trophic/zenith fluorescence trends). The
numbers above say the physics alone [option (a)] misses the gate at zenith 0°,
and a pure emulator [option (b)] would bake in φ_C = 0.02 and forfeit the
physiology handle. Agree to lock (c)? And should Raman and fluorescence get
*separate* correction heads (my lean: yes — different inputs, different
failure modes) or one shared network?

**DQ1-answer:**  Yes, lock (c).  And I agree with you on the separate correction heads.

**DQ2 (Scope).** v1 covers Raman + Chl-a fluorescence, matching the available
truth. CDOM fluorescence: no truth exists in hand (absent from L23 and BING).
I recommend the design doc specify its *interface hooks* (a third inelastic
term slot) plus the HydroLight runs that would be needed to build it, but
declare implementation out of scope for v1. Agree, or do you want CDOM
fluorescence designed-in now (which puts new HydroLight runs on the critical
path)?

**DQ2-answer:** Yes, I agree.

**DQ3 (Interface).** Two unavoidable extensions to the pinned `forward()`
contract, and one design choice:
(i) fluorescence needs **a_ph(λ)** specifically — the `IOPs` pytree grows an
`a_ph` field (elastic path ignores it);
(ii) both processes need IOPs at **excitation wavelengths** — internal
(evaluate the model's IOP spectra on the shifted grid), no API change;
(iii) new inelastic parameters — I propose an optional `inelastic` pytree
(φ_C now; CDOM-fl slot later) with `inelastic=None` → **bit-identical
elastic-only output** (the elastic gate stays valid). Emission enters as
`Rrs = (Rrs_ZTT + ΔRrs) × f_Raman + Rrs_fl`, i.e. Raman multiplicative on the
elastic hybrid (the self-normalizing form BING validated), fluorescence
additive. OK?

**DQ3-answer:** Yes, ok.

**DQ4 (φ_C semantics).** The forward model exposes φ_C as a differentiable
input (default 0.02), so the future inversion can retrieve it or prior-load
it. But the learned correction is necessarily trained at the truth's fixed
φ_C = 0.02. My proposal: structure the fluorescence term so the correction
multiplies the *φ_C-linear* physics kernel — then training at 0.02
generalizes to nearby φ_C exactly to the extent the RT is φ_C-linear (it is,
to first order; the (1−B·rrs) nonlinearity is ~10⁻³). Also: single-Gaussian
emission (what L23/HydroLight used — validatable) with the 730 nm PS I
shoulder as a switchable extension that L23 cannot validate. Agree on both?

**DQ4-answer:** Yes, agree on both.

**DQ5 (Ed input).** The Raman/fluorescence terms need the solar spectral
shape Ed(λ)/Ed(λ′) — first-order important per the report. Under L23's fixed
atmosphere Ed is a deterministic function of solar zenith, so I propose:
ship the three L23 Ed(0⁺) spectra as package data, interpolate in zenith,
and accept an optional user-supplied Ed override in `geometry` for real-sky
use later. (No coupling to an atmosphere model in v1.) OK?

**DQ5-answer:** Yes, ok.  I do worry that the community uses a poor model of the Sun right now, but we will deal with that later.  Do note this in the design document.

**DQ6 (Acceptance gate).** Mirroring the elastic protocol (held-out-by-scene,
rrs-space rRMS), I propose gates on *both* levels:
- **total Rrs against the X4 runs** (the realistic all-processes-on case):
  held-out rRMS ≤ **0.5 %** at all three zeniths (elastic-only was 0.30 %;
  the inelastic deltas add budget);
- **per-process deltas**: Raman increment (X2/X1) median |error| ≤ 5 % at
  every zenith incl. 0°; fluorescence 685 nm peak (X4−X2) median |error|
  ≤ 5 %;
- **gradients**: finite-difference checks pass for all inputs including φ_C;
- **speed**: full-batch forward ≤ 2× the elastic hybrid's runtime.
Are these the right bars — in particular is 0.5 % total acceptable, or do
you want to hold the 0.30 % elastic standard end-to-end?

**DQ6-answer:** Yes, these are the right bars.  We'll see how it goes.

**DQ7 (Validation data & holdout).** Train/validate on L23 X1/X2/X4 at
zeniths 0/30/60°, held out **by scene** (same splits as the elastic effort so
results compose). Two known coverage gaps to accept or address: (i) only
three zeniths — the elastic effort already flagged geometry extrapolation as
unresolved; the inelastic corrections inherit it (document, don't solve, in
v1); (ii) Raman emission below 400 nm needs excitation IOPs below the 350 nm
L23 edge — I propose the v1 model officially supports λ ≥ 400 nm and
extrapolates below with a documented caveat. And: should the design doc
include a *wishlist of new HydroLight runs* (denser zeniths, varied φ_C,
CDOM fluorescence, varied atmospheres) as a §"beyond v1" so you can plan
compute?

**DQ7-answer:** I agree with your ideas for v1.  And, yes, do create a wishlist of new HydroLight runs.

**DQ8 (Prototype cadence and logistics).** Mirror the elastic effort:
a ~1-week prototype with milestones (M0 scaffold/API extension → M1 Ed +
excitation-grid infrastructure → M2 analytic terms in JAX, validated against
fixed-BING → M3 correction training on X-differences → M4 validation +
report), code in `robust/rt/` under the same conventions/CI with committed
test fixtures, followed by the coding plan + numbered prompt docs as before.
Anything you'd change about that cadence — e.g. a different time budget, or
folding this into the existing elastic modules rather than a new
`robust/rt/inelastic.py`?

**DQ8-answer:**  This looks good.  

---

Reaction from Claude (2026-08-20): all eight answers are affirmative and
mutually consistent — no second round needed. The two additions are
incorporated: the solar-model concern is recorded in the design doc (§4.2
caveat + §9 risk + wishlist item 5), and the HydroLight run wishlist is §8
(six prioritized run families). `design/rt_inelastic_model.md` is written.

### BING fixes

Questions from Claude (2026-08-18), before fixing the errors identified in
`context/RT/rt_inelastic_bing_summary.md`. One new result first: I tested the
π hypothesis for the fluorescence normalization directly — dividing the
two-flow irradiance reflectance R_F by π (isotropic emission: L_u = E_u/π)
before the rrs→Rrs conversion moves the median model/truth ratio at 685 nm
from **3.18 / 3.00 / 2.73** to **1.01 / 0.96 / 0.87** (zenith 0°/30°/60°).
The hypothesis is confirmed; the fix is a one-line normalization change, and
the residual ±13 % zenith trend is the separate two-flow geometry limitation.

**B1 (Scope of the fixes).** I propose fixing in BING:
(i) **Fluorescence normalization** — apply the 1/π factor in
`calc_Rrs_fluorescence` (with the derivation documented in the docstring);
(ii) **Raman flat-Ed default** — plumb a true Ed(λ′)/Ed(λ) ratio into the
production Raman path (`calc_raman_correction_factor` already accepts
`Ed_ratio`; it's the model wiring in `evaluate.calc_Rrs_from_models` that
never supplies it);
(iii) minor: pass per-λ_em Ed to the fluorescence term instead of the scalar
`Ed_em` (the function already broadcasts an array).
NOT proposed for BING: the fixed-μ two-flow geometry limits (the −25…−45 %
high-sun red Raman error and the residual trophic/zenith fluorescence trends)
— these are formulation-level and belong to the retrieve-or-bust redesign.
Agree with this split, or do you want any of the formulation items attempted
in BING too?

**B1-answer:** Agree with this split.

**B2 (Backward compatibility).** The π fix shrinks every fluorescence-enabled
Rrs_fl by ×3.14 — existing fits that used `include_Chl_fl` (e.g. biomass-paper
runs) would retrieve different posteriors, and any effective-φ_C
interpretation shifts by ~π. Options: (a) clean break — fix the code, update
the pinned regression tests, add a prominent note in the docs/changelog (my
recommendation); (b) keep old behavior behind a `rt_dict` legacy flag.
Related: the *additive* Raman path `raman.calc_Rrs_with_raman` (unused in
production) has the same normalization flaw — fix it consistently, or
deprecate/remove it?

**B2-answer:** (a)

**B3 (Ed source for the Raman ratio).** Where should Ed come from for (ii)?
My proposal: store `Ed_ratio` on the model objects at setup time alongside the
existing Raman precomputation (`init_raman()`), sourced the same way the
fluorescence path gets `Ed_ex` (user-supplied or `correct_atmosphere`), with
the current flat-Ed behavior as the documented fallback when no Ed is given.
Should the true-Ed path be the default when Ed is available, and should
`include_Raman` runs warn when falling back to flat Ed?

**B3-answer:** Yes, the true-Ed path should be the default when Ed is available.  And `include_Raman` runs should warn when falling back to flat Ed.

**B4 (Validation tests).** Should I add L23-anchored regression tests to BING
— e.g. a small committed fixture (a few X1/X2/X4 scenes) asserting the fixed
fluorescence term matches X4−X2 to ~±15 % at 685 nm and the true-Ed Raman
factor tracks X2/X1 in the green-red at zeniths 30–60°? Or keep BING's tests
self-referential (pinned to its own outputs) and leave L23 validation to
retrieve-or-bust?

**B4-answer:** Yes, add L23-anchored regression tests to BING.

**B5 (Workflow and logging).** The edits happen in the BING repo
(`~/Oceanography/python/bing`). You run all git commands — should I work on
the current checkout as-is and leave staging/branching to you, or do you want
to create a branch first? For logging: a new `bing/prompts/inelastic_fixes.md`
in BING style, appended entries to the existing `chl_fl.md`/`raman.md` logs,
or log only here? Finally: after the fixes land, should I re-run
`context/RT/make_rt_inelastic_figures.py` and update the report so it
documents the *fixed* state (with the pre-fix numbers kept as history)?

**B5-answer:** I have already branched.  I will handle all of the git commands.

### Setup

Questions from Claude (2026-08-16), after reviewing the BING inelastic RT work
(`bing/rt/raman.py`, `bing/rt/chl_fl.py`, `bing/rt/rrs.py`, the wiring in
`evaluate.py`/`inference.py`, the work logs in `bing/prompts/{raman,chl_fl,rt}.md`,
the `nb/Raman/` and `nb/ChlFl/` notebooks, and the tests). Please answer inline
below each question.

**Q1 (Purpose and audience).** Is `context/rt_inelastic_bing_summary.md` meant to
play the same role for the *inelastic* effort that `context/RT/rt_elastic_model.md`
played for the elastic one — i.e., the context document that feeds a
`design/rt_inelastic_model.md` and then a coding plan? And is the audience the
same (you + Robert Frouin + future Claude sessions)?

**A1:** Yes, that is correct on both counts.  

**Q2 (Scope: document vs. critique).** BING's implementation makes a stack of
specific approximations. For Raman: the Sathyendranath & Platt (1998) two-flow
formulation with fixed mean cosines (μ_d=0.9, μ_u=μ_R=0.5); a single 3400 cm⁻¹
wavenumber shift in the fitting path (the full Walrafen redistribution function
exists in `raman.py` but is not integrated over); Ed(λ′)/Ed(λ)=1 (flat solar
spectrum) as the default; and the Raman term applied as a *multiplicative
correction factor* on the Gordon elastic Rrs — which mixes two different RT
frameworks. For fluorescence: constant quantum yield φ_C (default 0.02, no NPQ
dependence in the fitting path), absorption efficiency g_C=1 (unweighted over
370–690 nm), and single- vs double-Gaussian emission. Should the report
(a) just document what BING does, or (b) critically assess each approximation
and quantify its error where feasible? I'd recommend (b) — it directly feeds the
design decisions for retrieve-or-bust.

**A2:** (b)

**Q3 (Relation to the new elastic model).** The retrieve-or-bust elastic effort
replaced BING's Gordon polynomial with a new forward model (PR05/O25 lineage per
`context/RT/rt_elastic_model.md`). Is the working assumption that the inelastic
terms will be *added onto that new elastic model* rather than onto Gordon? If so,
should the report explicitly discuss how BING's S&P98-based Raman correction and
Gordon-style fluorescence term would (or would not) transplant — e.g., the
correction-factor approach vs. an additive Rrs term?

**A3:** Yes, we wish to add to the new elastic model.  This report should comment on the feasibility of that approach.

**Q4 (Science results vs. methodology).** Beyond the formulation, BING contains
*results*: the Raman notebooks (`Raman_scaling`, `Raman_Lowbbp`, `raman_fitting`),
the biomass-paper `Check_Inelastic` analysis, and L23 MCMC fits with
`include_Raman` / `include_Chl_fl` on. Should the report summarize what was
*learned scientifically* (e.g., size of the Raman correction vs. wavelength and
trophic state, its effect on retrieved bb_p in low-bb_p waters, whether φ_C is
retrievable), or stick to the model formulation and implementation? If the
former, which findings do you consider the most important to capture?

**A4:**  Stick to the model formulation and implementation.

**Q5 (Validation truth for figures).** Prompt 4 says L23 is now at
`$OS_COLOR_DATA/Loisel2023/`. My understanding is the L23 database was generated
with HydroLight including inelastic processes (Raman + fluorescence + CDOM
fluorescence) — can you confirm which inelastic processes are in the L23 Rrs, and
whether an elastic-only variant exists in the database? This determines whether
L23 can serve as "truth" for isolating each inelastic contribution, or only for
the total. Are there other benchmarks you want used (e.g., HydroLight runs of
your own, PACE spectra)?

**A5:** Yes, L23 includes both elastic and inelastic Rrs.  The README.md file 
in the L23 folder explains this.  But here is the main point: X = 1: No inelastic processes included.  X = 2: Raman scattering by water molecules included.  X = 4: Raman scattering by water molecules and fluorescence of chlorophyll-a included.


**Q6 (Engineering story).** A large fraction of the logged BING work is
performance engineering: the per-λ_em κ_F bug fix (the double-Gaussian 730 nm
peak was ~3.7× too strong), the >100 GB chains-path memory blow-up and its
chunked fix (`FL_CHUNK_ELEMENTS`), and the 7× MCMC slowdown fix. Should the
report record this history (it's a real guide for re-implementation), or only
the final state of the algorithms?

**A6:** Stick with the final state of the algorithms.

**Q7 (Missing reference).** The Context section lists `mobley1994.pdf` in
`context/RT/`, but it is not present there (the other five papers are). Will you
add it, or should I rely on the Ocean Optics Web Book chapters for the
Mobley-level material?

**A7:** I am removing that paper from the context section.

**Q8 (CDOM fluorescence).** BING treats Raman and Chl fluorescence but not CDOM
fluorescence, which matters in the blue-green for CDOM-rich waters and is the
third inelastic process HydroLight models. Should the report flag it (with
literature pointers) as a gap to consider in the retrieve-or-bust design, or is
it explicitly out of scope?

**A8:** Yes, flag that as a gap to consider in the retrieve-or-bust design.

---

#### Setup — Round 2

Reaction from Claude (2026-08-16) to the round-1 answers, after verifying the
L23 data at `/mnt/tank/Oceanography/data/Color/Loisel2023/` (all nine
X∈{1,2,4} × zenith∈{0°,30°,60°} scenario pairs present) and reviewing the new
elastic model (`design/rt_elastic_model.md`, `design/prototype_summary.md`).

Three findings that sharpen the plan:

- **L23 is a controlled experiment for exactly our question.** Per the ESSD
  paper, L23's inelastic runs used HydroLight *defaults*: the Mobley (2012)
  Raman settings and quantum yield φ_C = 0.02 — the same values BING defaults
  to. So the scenario differences (X2−X1 = pure Raman, X4−X2 = pure Chl
  fluorescence, per zenith) isolate the *formulation* error of BING's two-flow /
  single-shift / flat-Ed approximations, with the physical constants held equal.
  This is the ideal quantification backbone for the (b)-style critique.
- **The transplant target is a JAX-differentiable hybrid** (`Rrs = Rrs_ZTT +
  ΔRrs`) trained on the *elastic-only* (X=1) release. BING's multiplicative
  Raman correction factor is Gordon-tied and would not transplant cleanly;
  additive terms (or a second residual head trained on the X-differences) fit
  the architecture naturally. The feasibility section will be organized around
  this.
- **The L23 data folder already contains inelastic-impact artifacts**
  (`inelastic_impact_summary.csv`, `loisel_inelastic.csv`,
  `Rrs_inelastic_comparison.png`, `plot_peakpct_vs_aph440.png`,
  `find_raman_vars.py`, ...) showing e.g. ~5–11 % Raman impact on Rrs in the
  blue-green and a ~8 % fluorescence peak contribution at 681 nm.

Three questions before I write the report (prompt 4):

**Q9 (Quantification protocol).** To honor A2 = (b), I propose the report's
figures quantify BING's approximations *against L23 truth*: feed L23's own
input IOPs through BING's `calc_raman_correction_factor` /
`calc_Rrs_fluorescence` and compare with the X2−X1 and X4−X2 Rrs differences,
per wavelength and zenith. This means running BING code (ocean14) over the L23
netCDFs from the figure scripts in `context/RT/`. OK, or would you rather the
report stay literature/analytic and defer numerics to the design phase? (I
recommend running it — the data and code are both in hand, and A5 suggests you
set up the X-scenarios for exactly this.)

**A9:** Yes, that is a good idea (run the BING code over the L23 netCDFs).

**Q10 (Existing artifacts).** Are the analysis files already in the Loisel2023
data folder (listed above) yours / from an earlier session, and should I treat
them as authoritative and build on them — or regenerate everything from scratch
with self-contained scripts in `context/RT/` (my default, keeping the data
folder untouched)?

**A10:** Regenerate whatever you need but resuse whatever you can from the existing artifacts.

**Q11 (Feasibility framing).** For the "can we add this to the new elastic
model" section (A3), I plan to assess three integration options: (a) physics
terms ported to JAX — an S&P98-style Raman term and the Gordon-style
fluorescence integral as *additive* rrs contributions; (b) a second residual
emulator trained directly on the X2−X1 / X4−X2 differences (no new physics);
(c) hybrid — analytic physics term plus a small learned correction, mirroring
the elastic architecture. Two confirmations: (i) is end-to-end differentiability
a hard requirement for the inelastic terms (I assume yes), and (ii) should the
report *recommend* one option or just lay out the trade-space for the design
doc to decide?

**A11:** Yes, end-to-end differentiability is a hard requirement for the inelastic terms. It might be too early to recommend one option or the other.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-08-16 (Reviewed JXP's inelastic RT work in BING; posed Setup Q&A round 1)

Executed Setup prompt 1 (model: Fable). Reviewed the BING repository's inelastic
RT treatment ahead of writing `context/rt_inelastic_bing_summary.md`. No report
written yet — per the prompt, I posed eight questions in Q&A/Setup first.

**What I reviewed:**

- `context/context_summary.md` (project framing: degeneracy, priors thesis,
  success criteria).
- `bing/rt/raman.py` (~1200 lines): Bartlett et al. (1998) Raman scattering
  coefficients (b_R(488) = 2.6×10⁻⁴ m⁻¹ HydroLight default, λ′⁻⁵·⁵ energy /
  λ′⁻⁵·³ photon exponents), Walrafen (1967) 4-Gaussian wavenumber redistribution,
  Raman phase function (ρ=0.17, analytic b_b/b = 0.5), and the Sathyendranath &
  Platt (1998) reflectance terms — first-order R^R (Eq. 11) plus second-order
  R^RE (Eq. 18) and R^ER (Eq. 23), with fixed mean cosines μ_d=0.9, μ_u=μ_R=0.5.
- `bing/rt/chl_fl.py` (~1200 lines): Gordon (1979)-style fluorescence with
  single/double Gaussian emission (685 nm FWHM 25 nm; 730 nm FWHM 50 nm;
  weights 0.75/0.25), b_F = φ_C·a_ph, isotropic emission (backscatter fraction
  0.5), constant / irradiance-dependent / depth-dependent quantum yield models
  (only constant φ_C is used in fitting), FLH/nFLH utilities.
- `bing/rt/rrs.py`: the production entry points. Raman enters `calc_Rrs` as a
  *multiplicative correction factor* ((R^E+R^Raman)/R^E from S&P98) on the
  Gordon elastic Rrs; fluorescence enters as an *additive* term via
  `calc_Rrs_fluorescence`, which integrates over excitation (370–690 nm) with
  per-λ_em upwelling attenuation κ_F and λ′/λ energy conversion, using
  Ed(λ′), Ed(λ_em) from the `correct_atmosphere` package.
- Fitting wiring: `evaluate.calc_Rrs_from_models` is the canonical forward
  model; `rt_dict` flags (`include_Raman`, `include_Chl_fl`, `phi_C`,
  `double_gaussian`, `variable_Gordon*`) from `bing/rt/defs.py`; model-side
  precomputation (`bbnw.init_raman()` → wave_ex via single 3400 cm⁻¹ shift +
  bb_R; `anw.eval_a_ex`, `i_Chl_ex`, `Ed_ex`, `Ed_em`).
- Work logs in `bing/prompts/{raman,chl_fl,rt}.md` — notably the chl_fl.md log:
  (1) per-λ_em κ_F correctness fix (old code froze κ_F at 685 nm, overestimating
  the double-Gaussian 730 nm peak ~3.7×); (2) chains-path memory blow-up
  (>100 GB for nsteps=40k) fixed by sample-chunking (`FL_CHUNK_ELEMENTS`,
  bit-identical, ~2.4 GiB); (3) subsequent 7× MCMC slowdown fixed by restoring
  vectorization within chunks (28.8 µs/call vs 252.5). Tests: 15 in
  `test_raman.py`, 36 in `test_chl_fl.py`, plus L23 fitting tests that run
  MCMC with both processes enabled.
- Notebook inventory: `nb/Raman/` (6 notebooks: model dev, Rrs, scaling,
  fitting, low-bbp) and `nb/ChlFl/` (2), plus `papers/biomass` usage
  (`Check_Inelastic.ipynb`, `Fit_PACE_v3_1_RT.ipynb`).

**Key approximations identified (feeding Q2):** S&P98 two-flow with fixed mean
cosines; single-shift (delta-function) Raman redistribution in the fitting path
(full Walrafen machinery exists but is unused there); Ed_ratio=1 default for
Raman; multiplicative Raman correction grafted onto the Gordon model; constant
φ_C; g_C=1 absorption efficiency; no CDOM fluorescence anywhere.

**Observations flagged in Q&A:** `mobley1994.pdf` is listed in Context but
missing from `context/RT/`; L23's inelastic content needs confirming before it
can serve as truth for the report's figures (prompt 4).

**Next:** await JXP's answers in Q&A/Setup (Setup prompt 2).

### 2026-08-16 (Reviewed round-1 answers; verified L23 + elastic design; posed Setup Q&A round 2)

Executed Setup prompt 2 (model: Fable). Reviewed JXP's answers A1–A8 and did
the follow-up verification they called for; posed three round-2 questions
(Q9–Q11) in Q&A/Setup.

**Decisions locked in by round 1:** the report is the context doc feeding
`design/rt_inelastic_model.md` (A1); critical assessment with quantified errors,
not just documentation (A2); inelastic terms target the *new* elastic model and
the report must assess feasibility of that (A3); formulation/implementation
only, no science results (A4); final-state algorithms only, no engineering
history (A6); `mobley1994.pdf` dropped from Context (A7); CDOM fluorescence
flagged as a design gap (A8).

**Verification work this round:**

- **L23 data** confirmed at `/mnt/tank/Oceanography/data/Color/Loisel2023/`
  (note: `$OS_COLOR_DATA` is not set in my shell; the path above is what
  resolves). All 18 netCDF files present — nine scenario pairs
  `HydrolightXYY[.nc|_profile.nc]`, X∈{1,2,4}, YY∈{00,30,60}, 81 wavelengths
  (350–750 nm @ 5 nm) × 3320 IOP combinations. README.md and
  Database_README.pdf read.
- **Key fact from the ESSD paper** (`/home/xavier/Loisel_et_al_ESSD_2023.pdf`,
  Sect. on RT simulations): the inelastic runs used HydroLight *default*
  settings — Mobley (2012) Raman coefficient/phase/redistribution and
  **φ_C = 0.02** — i.e., the same constants BING defaults to
  (`B_RAMAN_488_HYDROLIGHT = 2.6e-4`, `PHI_FL_DEFAULT = 0.02`). Scenario
  differencing (X2−X1, X4−X2) therefore isolates BING's *formulation* error
  (two-flow, single 3400 cm⁻¹ shift, Ed ratio = 1, correction-factor grafting)
  with physical constants controlled. This becomes the quantification backbone
  proposed in Q9.
- **New elastic model reviewed** (`design/rt_elastic_model.md`,
  `prototype_summary.md`, `rt_elastic_implementation.md` headers): JAX
  differentiable hybrid `Rrs = Rrs_ZTT + ΔRrs` (417-param MLP residual), trained
  and validated on the **elastic-only X=1** L23 release; held-out rRMS 0.30 %
  vs O25-refit 0.69 %. Implication for feasibility: BING's multiplicative Raman
  correction is Gordon-specific; additive rrs terms or a learned inelastic
  residual head fit this architecture better — framed as options (a)/(b)/(c)
  in Q11.
- **Noticed pre-existing inelastic analysis artifacts** in the L23 data folder
  (`inelastic_impact_summary.csv`: Raman ~5–11 % of Rrs in blue-green;
  `aph440_vs_fluorpeak_summary.csv`: ~8 % fluorescence peak at 681 nm;
  plus comparison PNGs and `find_raman_vars.py`). Provenance unknown to me —
  asked in Q10 whether to build on or regenerate.

**Next:** await answers to Q9–Q11, then proceed toward the report
(Setup prompts 3–4).

### 2026-08-16 (Wrote the report: BING inelastic RT quantified against L23; two headline findings)

Executed Setup prompt 3 (model: Fable). Reviewed A9–A11 (run BING over the L23
netCDFs: yes; reuse existing artifacts where possible, regenerate the rest;
differentiability is a hard requirement and the feasibility section lays out
the trade-space without recommending). No further questions needed — wrote the
deliverables:

- **`context/rt_inelastic_bing_summary.md`** — the report. Sections: size of
  each process in the L23 truth; physics + source literature; what BING
  implements (Raman as an S&P98 multiplicative correction on Gordon Rrs;
  fluorescence as an additive Gordon-style emission term); critical
  assessment quantified against L23; feasibility of adding inelastic terms to
  the new JAX hybrid elastic model (options (a) physics port / (b) residual
  emulator on X-differences / (c) physics+correction hybrid, with a
  decision-criteria table, per A11 no recommendation); recommendations carried
  to the design doc; references.
- **`context/RT/make_rt_inelastic_figures.py`** — self-contained figure
  script (ocean14; reads `$OS_COLOR_DATA/Loisel2023` with fallback to
  `/mnt/tank/Oceanography/data/Color/Loisel2023`). Produces
  `fig_l23_inelastic_impact.png`, `fig_raman_bing_vs_l23.png`,
  `fig_fluor_bing_vs_l23.png`, `fig_raman_redistribution.png`, and
  `rt_inelastic_metrics.csv` (all in `context/RT/`). Method: feed L23's own
  input IOPs through BING's production functions
  (`calc_raman_correction_factor`, `calc_Rrs_fluorescence`) and compare with
  the scenario differences (Raman truth = Rrs_X2/Rrs_X1; fluorescence truth
  = Rrs_X4 − Rrs_X2) per wavelength/zenith/scene, using L23's own Ed(0⁺) for
  the true-Ed variants. Verified the L23 input IOPs are identical across X
  scenarios and Ed(0⁺) is scene-independent (asserted in the script).

**Headline findings (all new this round):**

1. **Truth magnitudes:** Raman contributes a median 5–15 % of Rrs (520–750 nm)
   at zeniths 30–60° and up to ~20 % at 0°; Chl fluorescence a median ~35 % of
   Rrs at the 685 nm peak. Both dwarf the elastic model's 0.3 % rRMS.
2. **Raman — flat-Ed is the dominant error.** BING's default (Ed ratio = 1)
   gets the correction's spectral shape wrong (+58–66 % increment error at
   490 nm at 30–60°, −15–50 % in the red). Supplying the true Ed(λ′)/Ed(λ)
   removes most of the shape error at 30–60° (it even reproduces the truth's
   spectral fine structure, which enters via Ed). Remaining defect: at zenith
   0° the truth increment is systematically larger and all variants sit −25
   to −45 % low in the red — the fixed-μ two-flow can't capture the sun-angle
   dependence; μ_d = cos θ_w helps only marginally. The multiplicative-ratio
   grafting is self-normalizing (a virtue worth keeping in mind).
3. **Fluorescence — amplitude wrong by ×2.5–3.** With φ_C, emission shape,
   and solar spectrum all matched to L23, BING's term overestimates the
   685 nm truth by +214/196/169 % (zenith 0/30/60°), a tight systematic
   offset growing to ×5–9 in eutrophic scenes. Suspected root cause
   (hypothesis, flagged for design-phase verification): the two-flow
   *irradiance* reflectance R_F is passed through the rrs→Rrs conversion as
   if it were rrs (L_u/E_d), inflating by ≈π for isotropic emission — the
   Raman term escapes this because it only uses a ratio. The trophic growth
   is a second error (fixed-μ attenuation too weak in absorbing waters).
   The double Gaussian lands closer to L23 truth only by error cancellation
   (L23/HydroLight used a single Gaussian).
4. **Feasibility:** positive. BING's physics toolkit (coefficients,
   redistribution, line shapes) transplants to JAX trivially and
   differentiably; neither grafting convention survives. Inelastic terms add
   two interface ingredients: the solar-spectrum ratio (first-order
   important) and IOPs on the shifted excitation grid. L23's paired X-runs
   are ready-made training/validation truth. φ_C retrievability argues
   against a pure emulator for fluorescence (it bakes in φ_C = 0.02).

**Reuse per A10:** the pre-existing `inelastic_impact_summary.csv` numbers
(Raman ~8–11 %) are consistent with the regenerated truth figures; the
`aph440_vs_fluorpeak_summary.csv` FLH numbers measure baseline curvature on
X1 (no fluorescence present), so they were not comparable and were not used.

**Caveat for the record:** Raman comparison restricted to λ_em ≥ 400 nm
(excitation below the 350 nm L23 grid edge otherwise); fluorescence figures
use BING defaults μ_d = 0.9, μ_f = 0.5 and the L23 solar spectrum.

**Next:** JXP review of the report; then the Edits round (Robert Frouin
comments) and/or the Design prompts.

### 2026-08-16 (Moved the report into context/RT/ and fixed figure links)

Executed Setup prompt 4 (model: Fable). Moved
`context/rt_inelastic_bing_summary.md` → `context/RT/rt_inelastic_bing_summary.md`
(plain `mv`; no git commands per repo policy — the move will show as an
untracked-file rename to be staged by JXP). Updated all relative links for the
new location: the four figure embeds and the links to
`make_rt_inelastic_figures.py` and `rt_inelastic_metrics.csv` dropped their
`RT/` prefix (all six files are now siblings in `context/RT/`). Full repo
paths in prose (e.g. the reproduction command
`python context/RT/make_rt_inelastic_figures.py`) were left as-is since they
are location-independent. Verified no stale `](RT/...)` links remain.

### 2026-08-18 (Confirmed the π normalization hypothesis; posed BING-fixes Q&A round 1)

Executed BING prompt 1 (model: Fable). No code changed yet — per the prompt,
posed five questions (B1–B5) in Q&A/BING fixes covering scope, backward
compatibility, Ed plumbing, validation tests, and workflow.

**New result obtained while preparing the questions:** the report's §4.2
hypothesis for the fluorescence ×2.5–3 amplitude error is now *empirically
confirmed*. Test (scratchpad script, ocean14): recompute the L23 comparison
with the two-flow irradiance reflectance R_F divided by π (isotropic emission:
L_u = E_u/π) before BING's A·rrs/(1−B·rrs) conversion. The median model/truth
ratio at 685 nm moves from 3.18/3.00/2.73 to **1.01/0.96/0.87** at solar
zenith 0°/30°/60°. Conclusion: `calc_Rrs_fluorescence` treats an irradiance
reflectance as if it were rrs (L_u/E_d); the fix is a 1/π normalization, and
the remaining ±13 % zenith trend is the separate fixed-μ two-flow geometry
limitation flagged in the report (not proposed for fixing in BING).

**Proposed fix scope (pending answers):** (i) the 1/π fluorescence
normalization; (ii) supplying a true Ed(λ′)/Ed(λ) ratio to the production
Raman path (the report showed flat-Ed is the dominant Raman error:
+58–66 % increment error at 490 nm); (iii) per-λ_em Ed for the fluorescence
term. Formulation-level limits (high-sun red Raman, trophic fluorescence
growth) deliberately left to the retrieve-or-bust redesign.

**Next:** await answers B1–B5, then implement the fixes in the BING repo.

### 2026-08-18 (Implemented the BING inelastic fixes on branch `inelastic-fixes`; all tests pass; production validated against L23)

Executed BING prompt 2 (model: Fable). Reviewed B1–B5 answers (agreed split;
clean break; true-Ed default with fallback warning; L23-anchored tests; JXP
handles all git on the already-created `inelastic-fixes` branch) and
implemented the fixes in the BING repo. Full details in BING's
`prompts/inelastic_fixes.md`; summary:

**Code changes (BING repo, branch `inelastic-fixes`):**

- `bing/rt/rrs.py::calc_Rrs_fluorescence` — the 1/π fix: the two-flow
  irradiance reflectance R_F = E_u/E_d is now converted to rrs via
  L_u = E_u/π (isotropic emission) before the A·rrs/(1−B·rrs) step;
  derivation + L23 validation numbers in the docstring.
- `bing/rt/raman.py::calc_Rrs_with_raman` (additive path, unused in
  production) — same 1/π conversion for consistency.
- `bing/rt/rrs.py::calc_Rrs` — new `Ed_ratio` kwarg forwarded to the Raman
  correction factor.
- `bing/models/anw.py` — new `aNWModel.set_raman_Ed(wave_Ed, Ed)` storing
  `Ed_ratio_raman` (the Ed grid must extend ~50 nm blueward of the model
  grid); `init_Chl_fluorescence` now defaults `Ed_em` to the full Ed vector
  (per-λ_em normalization; legacy scalar still accepted).
- `bing/evaluate.py::calc_Rrs_from_models` — uses `Ed_ratio_raman` when set;
  emits a RuntimeWarning and falls back to flat Ed when not.
- `bing/fitting/l23.py` — wires `set_raman_Ed` from `correct_atmosphere`
  when `include_Raman` and passes the ratio to the synthetic-observation
  Rrs, keeping mock data and MCMC forward model consistent; vector Ed_em.
- Docs: `chlorophyll_fluorescence.rst` (formula rewritten with the E_u→L_u
  step + validation note), `radiative_transfer.rst` (set_raman_Ed note),
  `changelog.rst` (prominent breaking-change entry: pre-fix
  `include_Chl_fl` results overestimated Rrs_fl ~3×, effective φ_C ~π×
  smaller than nominal). New BING-side log `prompts/inelastic_fixes.md`.

**Tests:** new `bing/tests/test_l23_inelastic.py` with committed 69 kB
fixture (40 L23 scenes, zenith 30°; generator in `tests/files/`): pins the
fluorescence term to ±15 % of X4−X2 truth at 685 nm, the true-Ed Raman
factor to ±15 % median increment error (550–700 nm), and true-Ed strictly
better than flat-Ed at 490 nm. Updated the pinned reference in
`test_chl_fl.py`. Suite results: 53 (l23_inelastic+chl_fl+raman) + 10
(evaluate) + 26 passed/2 skipped (l23_fitting, matching the pre-change
tally) + 13 (inference+chisq_fit) = **102 passed, 2 skipped**.

**Production validation vs full L23 (3320 scenes/zenith):** fluorescence
median model/truth at 685 nm = **1.00/0.95/0.86** (zenith 0°/30°/60°; was
3.18/3.00/2.73); true-Ed Raman median increment error 550–700 nm =
**+1.2 %/−4.3 %** at 30°/60° (−39 % at 0° — the two-flow high-sun
limitation, deliberately left to the retrieve-or-bust redesign).

**Report updated:** added a dated update box at the top of §4 of
`context/RT/rt_inelastic_bing_summary.md` (fixes confirmed + post-fix
numbers; pre-fix figures retained as the motivating record) and a matching
note in the header of `context/RT/make_rt_inelastic_figures.py`.

**Two operational notes:** (1) `test_l23_fitting.py` blocks forever in
headless runs at `plt.show()` (Tk mainloop) in the Raman/Chl LM tests —
pre-existing; run with `MPLBACKEND=Agg` (documented in the BING log).
(2) `conda run` buffers all output until exit; use the env python directly
for long test runs.

**Next:** JXP reviews/commits the BING branch. Report and prompt doc are
current; the formulation-level items remain for the Design phase.

### 2026-08-19 (Fixed the test_l23_fitting.py GUI block in BING)

Per JXP's request, fixed the headless hang found during the BING-fixes
validation (model: Fable). Root cause: three tests in
`bing/tests/test_l23_fitting.py` called `plotting.show_fits(..., show=True)`
→ `plt.show()`, which blocks forever in a Tk mainloop when a DISPLAY is
reachable but nobody closes the window. Two-layer fix in the BING repo
(branch `inelastic-fixes`): (1) `bing/tests/conftest.py` forces the Agg
backend for the whole suite before pyplot is imported, making `plt.show()`
a no-op and hang-proofing all current and future plotting tests; (2) the
three tests now pass `show=False` (a unit test shouldn't request a GUI
window; figure construction is still exercised). Verified with no
MPLBACKEND override: the formerly hanging LM test + all of
`test_plotting.py` → 20 passed in 3.9 s; the two MCMC plot tests →
2 passed in 62 s. Logged on the BING side in `prompts/inelastic_fixes.md`.

### 2026-08-19 (Posed Design Q&A round 1 for design/rt_inelastic_model.md)

Executed Design prompt 1 (model: Fable). No document written yet — posed
eight design questions (DQ1–DQ8) in Q&A/Design, each with a recommendation,
after re-reading the elastic design (`design/rt_elastic_model.md`) to align
with its pinned contracts (the `forward(iops, phase_params, geometry, wave)`
API, JAX hybrid architecture, held-out-by-scene validation protocol, DQ-style
decision record).

The questions and my recommendations:

- **DQ1 architecture:** lock option (c) — analytic inelastic backbone
  (the fixed BING physics) + bounded learned corrections trained on the
  L23 X-differences; separate correction heads for Raman and fluorescence.
  Rationale: post-fix physics alone still fails at zenith 0° (−39 % Raman
  red-band increment error), while a pure emulator would bake in φ_C = 0.02.
- **DQ2 scope:** Raman + Chl fluorescence in v1; CDOM fluorescence gets
  interface hooks + a specified HydroLight-run wishlist but no v1
  implementation (no truth data exist).
- **DQ3 interface:** `IOPs` grows an `a_ph` field; new optional `inelastic`
  pytree (φ_C), `None` → bit-identical elastic output; composition
  `Rrs = (Rrs_ZTT + ΔRrs) × f_Raman + Rrs_fl` (multiplicative Raman —
  the self-normalizing form — additive fluorescence).
- **DQ4 φ_C semantics:** φ_C differentiable input, default 0.02; correction
  multiplies the φ_C-linear kernel so training at 0.02 generalizes;
  single-Gaussian emission (L23-validatable), PS I shoulder as untestable
  extension.
- **DQ5 Ed:** ship the three L23 Ed(0⁺) spectra as package data,
  zenith-interpolated, user-overridable; no atmosphere model in v1.
- **DQ6 acceptance gate:** held-out total-Rrs (X4) rRMS ≤ 0.5 % at all
  zeniths; per-process delta median |error| ≤ 5 % (incl. zenith 0°);
  gradient checks incl. dRrs/dφ_C; ≤ 2× elastic-hybrid runtime. Asked
  whether 0.5 % total is the right bar or the 0.30 % elastic standard
  should hold end-to-end.
- **DQ7 validation data:** same by-scene splits as the elastic effort;
  officially support λ ≥ 400 nm (Raman excitation below the 350 nm L23
  edge), document the 3-zenith geometry-extrapolation caveat.
- **DQ8 cadence:** ~1-week prototype, M0–M4 milestones, `robust/rt/`
  conventions/CI, then coding plan + numbered prompt docs as for elastic.

**Next:** await JXP's answers (Design prompt 2), then react / second round,
then write `design/rt_inelastic_model.md`.

### 2026-08-20 (Design answers reviewed; wrote design/rt_inelastic_model.md)

Executed Design prompt 2 (model: Fable). All eight DQ answers were
affirmative — architecture (c) locked with separate correction heads, CDOM
fluorescence hooks-only, interface/φ_C/Ed/gates/data/cadence as proposed —
plus two additions: record the solar-model quality concern, and create the
HydroLight run wishlist. No second Q&A round was needed, so I proceeded to
write **`design/rt_inelastic_model.md`** (the step Design prompt 3 assumes
complete). Structure mirrors the elastic design doc:

- §1 goals/non-goals (reproduce the all-processes-on X4 ocean; keep
  differentiability incl. ∂Rrs/∂φ_C; `inelastic=None` → bit-identical
  elastic output).
- §2 architecture: `Rrs_total = (Rrs_ZTT + ΔRrs) × f_R + Rrs_fl` —
  multiplicative Raman (the self-normalizing form), additive fluorescence,
  two bounded tanh-scaled correction heads that decay to the analytic
  physics.
- §3 interface: `IOPs.a_ph`, optional `Inelastic` pytree (φ_C default 0.02,
  `cdom_fl` and `emission_shape` slots), `Geometry.Ed` override; official
  support λ ≥ 400 nm.
- §4 components: L23 truth channels (X2/X1, X4−X2, X4) on the elastic
  by-scene splits; Ed module from the three L23 Ed(0⁺) spectra with the
  DQ5 solar-model caveat recorded; JAX ports of the *fixed* BING analytic
  terms with their measured residual errors quoted as the correction
  targets; δ_R rescales the Raman increment, δ_F multiplies the φ_C-linear
  kernel (trained at 0.02, generalizes by linearity); cross-check pins the
  JAX ports to fixed-BING at rtol ≤ 1e-6.
- §5 modules under `robust/rt/` (inelastic.py, inelastic_corr.py, Ed data,
  committed weights + extended CI fixture modeled on BING's).
- §6 the DQ6 gate verbatim (total ≤ 0.5 %; per-process deltas ≤ 5 % incl.
  zenith 0°; elastic hash-regression; gradients incl. φ_C; ≤ 2× runtime)
  plus ungated diagnostics (trophic deciles, φ_C-linearity check).
- §7 the M0–M4 one-week prototype table.
- §8 beyond-v1: CDOM-fl landing zone + the six-item HydroLight wishlist
  (denser zeniths; varied φ_C; CDOM-fl on/off pairs; off-nadir views;
  alternative solar spectra — the DQ5 concern made testable; sub-350 nm).
- §9 risks (geometry extrapolation, φ_C-linearity untested against varied-φ
  truth, trophic-tail sparsity, solar-spectrum quality, Raman–fluorescence
  coupling folded into δ_F).

**Next:** JXP review of the design doc, then Design prompt 3 (the coding
plan — note its filename in the prompt list still reads
`rt_elastic_model_coding_plan.md`; presumably it should be
`rt_inelastic_model_coding_plan.md`).

### 2026-08-20 (Design prompt 3: design document finalized)

Executed Design prompt 3 (model: Fable). The document
`design/rt_inelastic_model.md` was drafted under prompt 2 (when the Q&A
closed with no second round needed); this prompt served as its finalization
pass. Re-read the full document fresh against the eight locked DQ decisions —
all are represented (architecture (c) with separate bounded heads; CDOM-fl
hooks only; interface extensions with the `inelastic=None` bit-identical
guarantee; φ_C-linear kernel + single-Gaussian emission; L23 Ed package data
with the DQ5 solar-model caveat; the DQ6 gate verbatim; elastic by-scene
splits with λ ≥ 400 nm support; M0–M4 cadence; six-item HydroLight wishlist).
Two defects found and fixed: replaced the confusing "×0.5⁻¹" notation in §2
with the plain statement that the fluorescence amplitude ratio drifts to
~×2–3 at the eutrophic tail, and rewrote the Raman excitation-grid formula in
§3 in proper wavenumber form (λ′ such that 1/λ′ = 1/λ + 3400 cm⁻¹).

Note for the next step: the coding-plan filename inside the design doc (§7)
reads `design/rt_inelastic_model_coding_plan.md`, while Design prompt 4 in
this doc still says `design/rt_elastic_model_coding_plan.md` — flagged for
JXP to reconcile before invoking that prompt (recommendation: use the
`inelastic` name; the elastic coding plan already occupies the other).

**Next:** JXP review, then Design prompt 4 (coding-plan Q&A).

### 2026-08-20 (Posed Coding Q&A round 1 for the inelastic coding plan)

Executed Design prompt 4 (model: Fable). No coding plan written yet — posed
six questions (CQ1–CQ6) in the new Q&A/Coding section, each with a
recommendation, after re-reading the elastic coding plan and implementation
record to mirror its structure (ground rules → package layout → milestone
tables → testing strategy → requirements → risks → definition of done).

Topics: CQ1 milestone→prompt-doc mapping (five docs, M0–M4, ~1-week budget);
CQ2 machine/environment — **verified `ocean14` on this machine has no JAX**
(the elastic install record evidently describes the laptop), so M0 must
repeat the elastic install procedure on whichever machine runs the prompts;
CQ3 BING cross-check mechanics — recommend committed fixed-BING reference
fixtures over a live `bing` import (keeps CI hermetic; notes the
`inelastic-fixes` branch is unmerged and asks whether merging is an M1
prerequisite or a pinned-commit generator suffices); CQ4 sibling CI fixture
(same 50 scenes + X2/X4 channels) rather than regenerating the elastic one;
CQ5 training scripts in `design/py/`, artifacts in `design/validation/`,
narrative in a new `design/rt_inelastic_implementation.md`, weights committed
under `robust/rt/files/`; CQ6 branch `rt-inelastic-prototype` (JXP runs git),
M4 review pass, no CI changes needed.

**Next:** await answers CQ1–CQ6 (Design prompt 5), then write
`design/rt_inelastic_model_coding_plan.md`.

### 2026-08-20 (Coding answers reviewed; wrote design/rt_inelastic_model_coding_plan.md)

Executed Design prompt 5 (model: Fable). All six CQ answers accepted my
recommendations except CQ3, where JXP chose **(a)** — the M2 cross-check
tests import `bing` directly rather than using committed reference fixtures —
and noted the BING `inelastic-fixes` PR is being issued now. Wrote
**`design/rt_inelastic_model_coding_plan.md`**, mirroring the elastic coding
plan's format (ground rules → package layout → milestones-at-a-glance →
per-milestone tasks/deliverable/gate → beyond-v1 → testing strategy →
requirements → risks → definition of done). Key contents:

- **Ground rules** encode the CQ answers: five prompt docs (M0–M4, ~1 week);
  this machine with M0 installing the JAX stack into `ocean14` (verified
  absent here); live-`bing` cross-checks that `skipif` on CI but are
  mandatory locally; sibling 50-scene fixture; `design/py` +
  `design/validation` + `design/rt_inelastic_implementation.md` layout;
  branch `rt-inelastic-prototype`, JXP runs git, M4 review pass.
- **Milestones:** M0 env + API extension (gate: `inelastic=None`
  bit-identical hash + elastic suite green); M1 Ed module/excitation
  grid/X2-X4 loaders/sibling fixture (gate: packaged Ed vs raw netCDF,
  splits identical to elastic); M2 analytic terms (gate: rtol ≤ 1e-6 vs
  fixed-BING live import, error-table reproduction with banded tolerances,
  grads incl. ∂/∂φ_C); M3 heads (gate: the DQ6 per-process 5 % bars on
  held-out scenes incl. Raman at 0°, committed weights reproduce metrics);
  M4 validation (gate: full design §6, review pass, artifacts committed).
- **Risk register** includes a sentinel test so the M2 cross-check fails
  loudly ("BING checkout predates inelastic-fixes") instead of pinning
  against pre-fix physics if the PR lands late; the δ_R-at-0° lift (the
  −39 % backbone error) with an explicit escalation path; eutrophic-tail
  monitoring; and a speed fallback (precompute zenith-static quantities).

**Next:** JXP review, then Design prompt 6 (generate
`claude_prompts/RT/rt_inelastic_coding_prompt_1..5.md`).

### 2026-08-20 (Generated the five inelastic coding prompt docs)

Executed Design prompt 6 (model: Fable). Created
`claude_prompts/RT/rt_inelastic_coding_prompt_1.md` … `_5.md`, one per
coding-plan milestone, modeled on the elastic prompt docs' structure
(Goals → Claude/Skills/Working agreements → Context → Status entering Mx →
Prompts → Mx Tasks with per-task gates → Q&A → Next → Logging/Logs):

- **Prompt 1 (M0)** — JAX install into this machine's `ocean14` (dry-run
  additive check, the elastic procedure), implementation-record creation,
  API extension (`IOPs.a_ph`, `Inelastic`, `Geometry.Ed`), and the pinned
  **elastic hash-regression** test. Its "Status entering M0" is written now
  (elastic prototype merged; jax absent here, verified).
- **Prompt 2 (M1)** — Ed module + packaged L23 spectra (scene-independence
  assert), excitation-grid helpers (differentiable interp, λ ≥ 400 nm
  support), X2/X4 loaders + truth channels on the verbatim elastic splits,
  sibling 50-scene CI fixture (elastic fixture bytes untouched).
- **Prompt 3 (M2)** — analytic `raman_factor` and φ_C-linear
  `fluorescence_kernel` in JAX with the composition law; the live-`bing`
  cross-check (CQ3a) with the sentinel test that fails loudly if the BING
  checkout predates `inelastic-fixes`; error-table characterization with
  banded tolerances; gradient checks incl. ∂/∂φ_C.
- **Prompt 4 (M3)** — the two bounded heads (features per design §4.5, δ_F
  blind to φ_C), training script with relative weighting and committed
  weights, the held-out 5 % per-process gates incl. Raman at 0° (with the
  escalation path routed through Q&A), plus the reported zenith-holdout
  diagnostic.
- **Prompt 5 (M4)** — the design-§6 acceptance gate as
  `test_inelastic_validation.py`, `run_validation.py` artifacts into
  `design/validation/`, the CQ6 review pass as an explicit task before the
  gate is declared, and the implementation-record/notebook wrap-up.

Carried over from the elastic pattern: an executed notebook per milestone
(`notebooks/RT/rt_inelastic_coding_N.ipynb`), and each doc's final task
updates the *next* doc's "Status entering" section so the plan stays honest
as reality diverges. Working agreements in every doc: JXP runs git on branch
`rt-inelastic-prototype`, `ocean14`/CPU JAX on this machine, reuse over
reinvention, pytest gates, elastic hash-regression green from M0 onward.

**Next:** JXP review/commit of the docs, then begin execution with
`rt_inelastic_coding_prompt_1.md` prompt 1.

### 2026-08-27 (Report prompt 1: reviewed the completed effort; posed Q&A/Report round 1)

Executed Report prompt 1 (model: Fable). No report written yet — per the
prompt, posed seven questions (RQ1–RQ7) in the new Q&A/Report section, each
with a recommendation.

**State of the effort reviewed before asking:** all five coding prompt docs
executed to completion; `design/rt_inelastic_implementation.md` v0.27
(2026-08-27) records M0–M4 all ✅ and **the design-§6 acceptance gate
PASSES** — held-out total rRMS vs X4 of 0.343/0.341/0.340 % at 0/30/60° on
the 400–700 nm gate band (bar ≤ 0.5 %); worst per-process delta 1.03 %
(Raman 490 nm at 0°; bar ≤ 5 %; the −38.6 % analytic high-sun error closed
to −0.14 %); `inelastic=None` bit-identical (strict SHA-256 pins); gradients
for all six inputs incl. φ_C to 5.9e-9; speed 1.59× the elastic hybrid
(median; budget 2×). Final suite: 431 passed, 1 skipped; artifacts committed
under `design/validation/*inelastic*` and regenerable by
`python design/py/run_validation.py --inelastic`. Also re-read the elastic
report (`reports/report_rt_elastic_model.md`) as the structural template and
its figure pipeline (`reports/make_report_figures.py` from committed
artifacts).

**Two filename discrepancies flagged in RQ1:** the prompt names the
deliverable `report/rt_elastic_model.md`, but the repo directory is
`reports/` (plural) and the elastic report already occupies
`reports/report_rt_elastic_model.md` — recommended
`reports/report_rt_inelastic_model.md`.

**The seven questions:** RQ1 filename/location; RQ2 scope (prototype-focused
with the BING assessment/fixes compressed into motivation, vs full-arc);
RQ3 template/audience and the headline framing (the complete forward model
reproducing the realistic X4 ocean at 0.34 %, elastic 0.30 % as the inherited
base); RQ4 figures (sibling `make_inelastic_report_figures.py`, ~4 figures
from committed artifacts; reuse `context/RT/` figures by link); RQ5 candor
(state the ~×π BING fluorescence error and its consequence for published
fits, final-state framing per A6; quote the sharp caveats incl. the −74 %
zenith-holdout cliff verbatim); RQ6 closing recommendations (inversion next
vs options-only); RQ7 whether the header cites branch `inelastic-rt` + the
PR trail or waits for a merge to `main`.

**Next:** await JXP's answers (Report prompt 2), then write the report.

### 2026-08-27 (Report prompt 2: wrote reports/report_rt_inelastic_model.md + 4 figures)

Executed Report prompt 2 (model: Fable). Reviewed the RQ1–RQ7 answers — all
affirmative, with three directives: keep the BING fixes to passing mention
(RQ2), include an explicit first-priority recommendation (RQ6), and refer to
`main` since the `inelastic-rt` branch will be merged before the report lands
(RQ7). Wrote the deliverables:

- **`reports/report_rt_inelastic_model.md`** — the end-of-effort report,
  mirroring the elastic report's skeleton section for section (executive
  summary → motivation → the model → data & validation protocol → results →
  may-claim/may-not → open items → recommended priorities → reproducibility →
  document map → references). Headline per RQ3: the *complete* forward model
  reproduces the all-processes-on L23 ocean (X4) to 0.34 % held-out rRMS at
  all three zeniths on 400–700 nm — the elastic-era accuracy now holding
  against the realistic ocean (the elastic-only model scores 16–19 % on that
  truth, 48 % at the 685 nm peak). Per RQ5 the motivation states the two
  upstream BING findings plainly (the ~×π fluorescence normalization and its
  effective-φ_C consequence for published fits; flat-Ed as the dominant Raman
  shape error) in final-state framing, and §5 quotes the sharp caveats
  verbatim (the −74 % zenith-holdout cliff, φ_C truth at one point, the
  sub-400 nm clamp at 13 %, the unvalidatable 'double' emission, one-sided
  θ_s derivatives at the Ed anchors, the elastic inheritances). Per RQ6, §7
  opens with the pick: **do the inversion next** (the caveats bind at unseen
  geometries, not on the L23 domain; ∂Rrs/∂φ_C is the physiology handle the
  architecture preserved), with the geometry HydroLight runs commissioned in
  parallel and a domain guard for the heads until they exist.
- **`reports/make_inelastic_report_figures.py`** — sibling of the elastic
  figure script, committed-artifacts-only (reads
  `design/validation/{rrms_per_wavelength_inelastic,metrics_inelastic}.csv`;
  the zenith-holdout numbers transcribed from the implementation record §5.3
  with provenance noted, matching the elastic script's transcription
  precedent). Produces `fig_inelastic_architecture.png` (the composition-law
  dataflow), `fig_inelastic_rrms_ladder.png` (per-λ elastic-only → analytic →
  corrected vs X4, ungated regions shaded, the 0.5 % gate bar drawn only over
  its band), `fig_inelastic_deltas.png` (the nine per-process rows analytic →
  corrected against the ±5 % band), `fig_inelastic_unseen_zenith.png` (the
  −74 % cliff). House palette reused from the elastic script (same validated
  adjacent order; every series direct-labeled). `ruff check` clean (the
  sibling's status); all four figures rendered and visually inspected.

One small correction made en route: an early docstring draft cited a
`--exclude-zenith` flag that `train_inelastic_corr.py` does not have — the
script trains the zenith-holdout variant unconditionally; fixed before
committing outputs.

**Next:** JXP review of the report; merge of `inelastic-rt` (the report's
header already refers to `main` per RQ7). This closes the prompt docs for
the inelastic RT effort.

### 2026-08-29 (Design prompt 7: re-read the docs, confirmed the CDOM-fl double absence; posed Q&A/CDOM round 1)

Executed Design prompt 7 (model: Fable). No design document written or edited
yet — per the prompt, posed eight questions (CFQ1–CFQ8) in the new Q&A/CDOM
section, each with a recommendation, after re-reading the three governing
documents fresh:

- **`design/rt_inelastic_model.md`** — the relevant anchors: §3 reserved the
  `Inelastic.cdom_fl` slot (None in v1) explicitly for this moment; §4.4 is
  the additive-emission pattern CDOM-fl would mirror; §8 names the landing
  zone and wishlist item 3 (CDOM on/off pairs with the Hawes et al. quantum
  functions on a CDOM-stratified subset); §6 is the gate style that cannot
  transfer as-is (no truth to gate against).
- **`design/rt_inelastic_model_coding_plan.md`** — the M0–M4 arc is complete;
  its "Beyond v1" section already points at the wishlist; the M2-style
  live-BING cross-check pattern is *unavailable* for CDOM-fl (below).
- **`reports/report_rt_inelastic_model.md`** — open item 2 and recommended
  priority 4 both defer CDOM-fl to "when truth exists"; the headline 0.34 %
  X4 gate is scored against a truth that *omits* CDOM fluorescence, which
  makes off-by-default a correctness requirement, not a style choice (CFQ5).
- Also re-skimmed `context/RT/rt_inelastic_bing_summary.md` §4.3 and A8 —
  the origin of the "neglected CDOM fluorescence" framing.

**Two verification findings (both expected, both load-bearing):**

1. **BING implements no CDOM fluorescence.** Grep over `bing/rt/` and
   `bing/models/` (case-insensitive "cdom"): every hit is the elastic a_g
   absorption machinery in `bing/models/anw.py`; nothing in `bing/rt/`.
   Consequence: unlike Raman/Chl-fl there is no fixed-BING implementation to
   port or pin against at rtol ≤ 1e-6 — the analytic term must be built from
   the literature (Hawes et al. 1992, the HydroLight parameterization) and
   verified by correctness pins rather than a cross-check (CFQ4, CFQ6).
2. **No new CDOM-fluorescence truth data exist.** The L23 folder on this
   machine (`$OS_COLOR/Loisel2023/` →
   `/Users/xavier/Projects/Oceanography/data/Color/Loisel2023/`) holds
   exactly the known 18 files — `Hydrolight{1,2,4}{00,30,60}[.nc|_profile.nc]`
   — i.e. the X∈{1,2,4} scenario set, none of which toggles CDOM
   fluorescence (A5). Nothing new has landed since the survey. Stated
   plainly: this design pass proceeds with **zero truth data**, which shapes
   the architecture (analytic-only v1, head defined but untrained), the gate
   (plausibility + correctness, no rRMS), and the wishlist refinement.

**The eight questions and my recommendations:** CFQ1 document target — new
standalone `design/rt_cdom_fluorescence_model.md` rather than amending the
finished, gate-passed inelastic design in place; CFQ2 truth status — build
the analytic term + interface now, gate head training behind data arrival,
and ask whether HydroLight CDOM runs are already planned on a timeline;
CFQ3 architecture — mirror the Chl-fl shape (additive term, physics backbone
+ bounded head), head zero-initialized and shipped untrained; CFQ4 physics —
the Hawes et al. (1992) η(λ′,λ) quantum-efficiency functions (HydroLight's
own, so future truth is consistent by construction), source ∝ a_CDOM(λ′)
(so `IOPs` grows an optional `a_cdom` field, the a_ph pattern), 350 nm
excitation clamp documented; CFQ5 slot shape — `cdom_fl=None` stays the
default *inside* `Inelastic()` (the X4 truth omits CDOM-fl, so the shipped
gate is valid only for a CDOM-fl-free default), a small differentiable
`CDOMFl(scale=...)` pytree when set, bit-identity regression extended;
CFQ6 gate without truth — off-state bit-identity + Hawes correctness pins +
literature-plausibility band + gradients + the existing 2× speed budget,
with the quantitative ≤ 5 %-style gate written but conditional on truth;
CFQ7 process — a small M5 (+ deferred M6) milestone table inside the new
design doc and a single prompt doc, not a full design→coding-plan→five-doc
arc; CFQ8 wishlist — expand item 3 into a concrete commissioning spec
(X4 vs X4+CDOM-fl pairs, a_g(440)-stratified subset with the rich tail
oversampled, exact Hawes function/version recorded), and ask whether to
bundle it with the priority-2 geometry runs.

**Next:** awaiting JXP's answers to CFQ1–CFQ8 before writing the
CDOM-fluorescence design document.

### 2026-08-29 (Docs prompt 1: surveyed the BING/PAB/IOPtics doc sets; posed Q&A/Docs round 1)

Executed Docs prompt 1 (model: Opus). No documentation written — per the
prompt I posed nine questions (DocQ1–DocQ9) in Q&A/Docs first.

**What I reviewed:**

- **The thing to be documented.** `robust/rt/` — 11 modules, ~5,900 lines:
  `conventions`, `types` (the `IOPs`/`PhaseParams`/`Geometry`/`Inelastic`
  pytrees), `data.l23`, `ed`, `ztt`, `emulator`, `hybrid` (`forward()`, the
  public entry point), `inelastic`, `inelastic_corr`, `baselines`,
  `validation` — plus the 18-file test tree, `setup.py`, `requirements.txt`,
  the CI workflow, and both reports/design records as source material.
- **Docstring coverage, measured** (AST walk over `robust/rt/*.py`): of ~115
  public classes/functions, **7 lack docstrings** — 3 in `emulator.py`, 2 in
  `ztt.py`, 1 each in `inelastic.py` and `validation.py` — and every module
  carries a module docstring. Autodoc is therefore essentially free; the
  seven gaps are a small edit, offered in DocQ4.
- **The three exemplar doc sets**, in detail (conf.py, RTD yaml, toctrees,
  page inventories, line counts, CI jobs).

**Two findings that shape the questions:**

1. **retrieve-or-bust is greenfield.** `docs/` exists but holds exactly one
   file — `docs/member_policy.md`, team governance, not technical. No
   `conf.py`, no `.readthedocs.yaml`/`readthedocs.yaml`, no Makefile, no
   docs job in `.github/workflows/ci.yml`, no `robust.__version__`. Nothing
   to extend; everything to create.
2. **The three exemplars disagree**, so "model it after BING, PAB, and
   IOPtics" required a recommendation rather than a copy. All three are
   Sphinx on ReadTheDocs, but: **BING** = flat `docs/`, `sphinx_rtd_theme`,
   `.rst` only (no MyST), undotted `readthedocs.yaml`, pdf+epub, hardcoded
   version, ~5,850 lines over 15 narrative pages + *seven* per-module API
   pages; **IOPtics** = `docs/source/` layout, **furo**, autosummary +
   `autodoc_mock_imports`, *one* `automodule` API page, generated report
   pages, figure-generator scripts writing into `_static/`, version read
   from `ioptics.__version__`, and Markdown design docs deliberately left
   unrendered; **PAB** = flat `docs/`, rtd-theme, **`myst-nb`** rendering
   both Markdown design docs and nine executed notebooks in-site, mermaid,
   docs deps in the root `requirements.txt`, RTD lenient while CI builds
   `-W --keep-going`, plus a second Sphinx target (`report_site/`) for the
   community. BING and PAB gate docs in CI; IOPtics does not.

**The nine questions and my recommendations:** DocQ1 scope — document
`robust.rt` as *one* forward model (elastic backbone + inelastic terms), not
inelastic-only, excluding the stub `rob/` package and stating plainly that
the inversion does not exist yet; DocQ2 tooling — PAB's shape (flat `docs/`,
rtd-theme) **with MyST**, since every scrap of source material we own is
Markdown or `.ipynb`, plus IOPtics' single-`automodule` API page and version
single-sourcing, and a real sub-choice: install the package on RTD (my lean)
vs mock the JAX stack; DocQ3 location — `docs/` + a new dotted
`.readthedocs.yaml` (ubuntu-24.04 / py3.12, `fail_on_warning: false`), render
`member_policy.md` as a Team page, and two questions only JXP can answer (does
an RTD project already exist, and is anything to be kept off a public site);
DocQ4 page set — ~15 pages (installation, quickstart, the model chapter by
chapter, data, validation, a prominent limitations page, one API page,
references), with the report §5 caveats quoted verbatim rather than buried;
DocQ5 provenance — render the two *reports* in-site, link out to design docs
and the implementation record, publish no Q&A/prompt-doc material; DocQ6
notebooks — do **not** render the ten milestone notebooks as tutorials (they
open on environment installs and are organized chronologically); write one new
quickstart notebook instead; DocQ7 figures — reuse the 7 committed report
PNGs via an IOPtics-style `docs/figures/*.py` rather than a second copy, and
reuse `fig_inelastic_architecture.png` as the hero unless a new one is wanted;
DocQ8 mechanics — add `robust.__version__`, add a strict `-W` docs job to CI
while RTD stays lenient, and (per RQ7) document the `main` state from a fresh
branch after `inelastic-rt` merges; DocQ9 staging — this is a **2–3 day,
~2,000–2,500-line** effort, not a 1–2 day item, so I propose two milestones in
a single prompt doc (`rt_docs_prompt_1.md`): **D1** scaffold + CI + API +
installation/quickstart building green and published, then **D2** the full
narrative set, reports section, limitations and figures.

**Next:** awaiting JXP's answers to DocQ1–DocQ9 before writing any
documentation.

### 2026-08-29 (Docs prompt 2: wrote `rt_docs_prompt_1.md` — D1/D2 milestones, pydata theme picked)

Executed Docs prompt 2 (model: Opus). No documentation, no `docs/` scaffold,
no `robust/` edit and no CI change — per DocQ9 this turn produces the prompt
doc only, exactly as Design prompt 6 produced the five coding prompt docs
without doing M0's work.

**Reviewed first:** JXP's answers to DocQ1–DocQ9 in Q&A/Docs above (all nine
answered), the house template
(`claude_prompts/RT/rt_inelastic_coding_prompt_1.md` in full,
`rt_inelastic_coding_prompt_5.md` for how a wrap-up milestone's tasks and
gates read), the eleven `robust/rt/` modules and `robust/rt/__init__.py`'s
docstring (already a compact site outline), `setup.py` / `requirements.txt` /
`.github/workflows/ci.yml` / `ruff.toml`, `docs/member_policy.md`, both
reports and their two figure scripts, the ten `notebooks/RT/*.ipynb`, and —
in detail — the three exemplars' real configs (PAB `docs/conf.py` +
`.readthedocs.yaml`, IOPtics `docs/source/conf.py` + `.readthedocs.yaml` +
`docs/requirements.txt` + `docs/figures/`, BING `docs/conf.py` +
`readthedocs.yaml`), so the tasks carry working values rather than
hand-waving.

**Written:** `claude_prompts/RT/rt_docs_prompt_1.md` (687 lines), house
sections in house order — Goals (with the theme decision), Claude
(Skills / Working agreements), Context, Status entering D1, Prompts (14
numbered turns), D1 with 7 numbered tasks each ending in a concrete **Gate**,
D1 Q&A (empty), D2 with 7 tasks and gates, D2 Q&A (empty), Next, Logging,
Logs (empty). D1 = scaffold + `conf.py` + `docs/requirements.txt` +
`robust.__version__` + `.readthedocs.yaml` + CI docs job + Installation +
Quickstart + Team + References + the single autodoc API page + front page and
model overview, building green under `-W` and rehearsed in a clean venv. D2 =
the quickstart notebook and development record, the elastic and inelastic
model chapters, Data + Validation, the verbatim limitations page, the nine
docstring fills, the figures script and Reports section, and a review/PR turn.

**Theme picked: `pydata-sphinx-theme`** (+ `sphinx-design`,
`sphinx-copybutton`), pinned `>=0.16`. It is neither BING/PAB's
`sphinx_rtd_theme` nor IOPtics' `furo`; it is the theme NumPy, SciPy, pandas,
xarray and Matplotlib ship, so it reads as scientific-library documentation on
sight; its three-column layout gives the long equation-dense ZTT/inelastic
chapters a right-hand in-page TOC that furo and rtd-theme bury; the top navbar
lets the five site sections be tabs; light/dark is native and `sphinx-design`
gives the front page a card grid for the hero. Recorded fallback if it fights
`-W`: `sphinx-book-theme` (pydata's sibling, what JAX's own docs use) — to be
raised in Q&A, not switched silently.

**No "modify the next prompt doc" turn**, unlike the coding docs: those needed
one because five milestones lived in five files. Both milestones live in this
one file, so the hand-off is D1's task 7, which fills the doc's own "Status
entering D2" section. A `rt_docs_prompt_2.md` is only to be proposed in Q&A if
D2's volume warrants a third milestone.

**Findings from the fresh inspection worth recording:**

- **The docstring count is 9, not 7.** DocQ4's scan globbed `robust/rt/*.py`
  and missed `robust/rt/data/l23.py`, which has two more
  (`npz_reader::read`, `inelastic_npz_reader::read`). More usefully: **all
  nine are nested closures**, so none of them is ever emitted by autodoc —
  filling them is a readability edit, not a rendering fix, and the prompt doc
  says so to stop a future turn claiming otherwise.
- **`robust.rt` imports only `jax`, `jaxtyping` and `numpy` at module level**
  (`flax`, `optax` and `ocpy` are all function-level imports, verified by AST
  walk). That is what makes DocQ2's no-mocking decision cheap: autodoc can
  import the whole package on RTD without ocpy or its geospatial extras.
- **The RTD config must not reuse the root `requirements.txt`** — its two
  `git+` lines pull ocpy's heavy extras (cartopy/geopandas/healpy), a
  build-breaking risk on RTD and the same file that clobbers editable dev
  checkouts (prompt 1 Q&A Q1). Hence `docs/requirements.txt` carrying the
  Sphinx toolchain *plus* the real JAX stack.
- **PAB's `conf.py` carries two hard-won settings we need verbatim**: load
  `myst_nb` *instead of* `myst_parser` (loading both conflicts) and
  `suppress_warnings = ["myst.xref_missing"]`, without which the rendered
  Markdown's repo-relative links fail the strict `-W` CI build. BING's
  contribution is `napoleon_use_ivar = True` (52 duplicate-object warnings
  without it).
- **Both reports reference their figures by bare relative filename**
  (`![...](fig_inelastic_architecture.png)`), so the D2 figures script both
  copies the seven `reports/*.png` into `docs/_static/` **and** writes
  path-rewritten copies of the two reports into `docs/reports/` — invoked from
  `conf.py` at build time, outputs gitignored, so RTD and CI regenerate them
  and no second copy of anything is committed (DocQ7's intent, extended to the
  Markdown).
- The docs build needs **no `$OS_COLOR` and no L23 data** — copy mode is pure
  file copying; figure regeneration is an opt-in `--regenerate` flag that must
  never be reachable from `conf.py`.

**Next:** JXP to create the docs branch off `main` (name JXP's call — the
prompt doc deliberately does not hardcode one, per the
`rt-inelastic-prototype`-vs-`inelastic-rt` lesson) and start prompt 1 of
`claude_prompts/RT/rt_docs_prompt_1.md`.

### 2026-08-29 (Design prompt 8: wrote the CDOM design doc, prompt doc, and updated the two upstream docs)

Executed Design prompt 8 (model: Fable). Reviewed JXP's eight CFQ answers in
Q&A/CDOM — all affirmative and mutually consistent, no second round needed:
standalone companion doc (CFQ1), analytic term + interface now with head
training gated on truth arrival (CFQ2), the Chl-fl term's exact additive
shape with a zero-init untrained δ_C head (CFQ3), Hawes et al. (1992) basis
∝ a_CDOM(λ′) with a new `IOPs.a_cdom` field and a hard 350 nm excitation
clamp (CFQ4), `cdom_fl=None` default even inside `Inelastic()` with a
`CDOMFl(scale=1.0)` pytree when set and an extended bit-identity regression
(CFQ5), the five-item truth-less v1 gate (CFQ6), a single-prompt-doc M5 plus
deferred M6 (CFQ7), and the "X4 vs X4 + CDOM-fl" run spec requested jointly
with the geometry runs (CFQ8).

**Created (2 files):**

- `design/rt_cdom_fluorescence_model.md` (~250 lines, §§1–9: goals/non-goals,
  architecture, interface, components/layout, validation & acceptance gate,
  milestones M5/M6, HydroLight run request, risks, references). Deliberately
  short — the Ed module, composition law, two-flow transport, and head
  machinery are inherited from the companion inelastic design by reference,
  not re-derived. The Hawes citation is given as Mobley cites it (Proc. SPIE
  1750, 212–223) with an explicit note that the page range should be
  confirmed against the SPIE record at implementation time.
- `claude_prompts/RT/rt_cdom_coding_prompt_1.md` (~230 lines), the single M5
  prompt doc: Goals / Claude (skills note: `inelastic-rrs` is BING-only
  background — BING has no CDOM-fl; working agreements incl. both
  hash-regressions staying green) / Context / Status entering M5 (431 passed
  + 1 skipped at end of M4; `cdom_fl` currently a rejected reserved slot in
  `types.py`) / Prompts / M5 with 9 gated tasks (types, loader wiring, Hawes
  kernel + correctness pins, clamp + truncated-fraction diagnostic,
  composition + the new `cdom_fl=None` bit-identity pin, zero-init δ_C,
  plausibility/gradients/speed, notebook + record, PR review) / empty Q&A /
  Next (M6 deferred, blocked on truth) / Logging template / empty Logs.

**Edited (2 pointer insertions, additive only):**

- `design/rt_inelastic_model.md` §8, immediately after the CDOM-fluorescence
  landing-zone paragraph: "*Update (2026-08-29):* CDOM fluorescence is now
  designed in a companion document, `design/rt_cdom_fluorescence_model.md`
  (milestones M5/M6). The paragraph above is retained as the historical
  decision record." Original text untouched.
- `design/rt_inelastic_model_coding_plan.md`, end of "Beyond v1": "*Update
  (2026-08-29):* CDOM fluorescence now has its own design doc and M5/M6
  milestone plan in `design/rt_cdom_fluorescence_model.md`, executed by
  `claude_prompts/RT/rt_cdom_coding_prompt_1.md` …". Original text untouched.

No git commands were run; JXP handles branching (fresh branch off `main` once
`inelastic-rt` merges) and all commits. The concurrent Docs effort's files
and its Q&A/Docs section were not touched; this entry was appended after a
fresh re-read of the file's tail.

**Next:** the design doc and M5 prompt doc await JXP's review before
execution of M5 begins.
