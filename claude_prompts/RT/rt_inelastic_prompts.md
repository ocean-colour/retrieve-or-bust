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

### Report

1. We have worked through all of the prompt docs.  Please write a report summarizing the work we have done.  Name it `report/rt_elastic_model.md`.  Please ask me a set of questions in Q&A/Report before writing the document.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them.  Then proceed to write the report.  Use Fable if you can.  Log your work.

## Comments

## Q&A

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
