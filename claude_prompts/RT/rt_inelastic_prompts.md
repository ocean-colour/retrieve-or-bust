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


### Edits

1. I've pasted in comments from Robert Frouin on the RT.  Please review them and revise the report accordingly.  Use Fable if you can.  Log your work.

### Design

1. Ok, let's generate a plan together for how to proceed.  We are going to write a design document for this elastic RT effort.  We will name it `design/rt_elastic_model.md`.  Please ask me a set of questions in Q&A/Design before writing the document.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Ask another round of questions if needed.  Use Fable if you can.  Log your work.

3. The design document looks great!  Please generate a separate coding plan with a staged implementation plan including milestones.  Name it `design/rt_elastic_model_coding_plan.md`.  Please ask me a set of questions in Q&A/Coding before writing the document.  Use Fable if you can.  Log your work.

4. I have answered your questions in the Q&A section below.  Please review them and proceed to write the coding plan.  Use Fable if you can.  Log your work.

5. The coding plan looks great!  Please generate a series of prompt docs to execute the coding plan.  Name them `claude_prompts/RT/rt_elastic_coding_prompt_1.md`, `claude_prompts/RT/rt_elastic_coding_prompt_2.md`, etc.  Model them after the ones in the PAB repository (in that `claude_prompts` folder). Use Fable if you can.  Log your work.

### Report

1. We have worked through all of the prompt docs.  Please write a report summarizing the work we have done.  Name it `report/rt_elastic_model.md`.  Please ask me a set of questions in Q&A/Report before writing the document.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them.  Then proceed to write the report.  Use Fable if you can.  Log your work.

## Comments

## Q&A

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
