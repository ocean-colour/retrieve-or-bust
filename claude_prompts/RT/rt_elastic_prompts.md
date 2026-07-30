# Elastic Radiative Transfer (RT)

This prompt doc will guide our efforts to develop elastic radiative transfer (RT) models for ocean color.

## Context

- Read the `context/context_summary.md` file.
- Examine the code and Notebooks in the `Oceanography/python/bing/dev/Gordon` folder

The standard approach in the community is to adopt the so-called "Gordon approximation" for the RT.  This is a simple model that relates the water-leaving radiance to the inherent optical properties (IOPs) of the water.  The Gordon approximation is given by:

```
Rrs(λ) ≈ f/Q · bb(λ) / [a(λ) + bb(λ)]
       = Σ_i  G_i · u(λ)^i        with   u(λ) = bb(λ) / [a(λ) + bb(λ)]
```
where `Rrs(λ)` is the water-leaving radiance at wavelength `λ`, `f/Q` is a factor that depends on the illumination/viewing geometry and the volume scattering function (VSF), `bb(λ)` is the backscattering coefficient at wavelength `λ`, `a(λ)` is the absorption coefficient at wavelength `λ`, `u(λ)` is the single-scattering albedo at wavelength `λ`, and `G_i` are the geometric factors. The `G_i` are the geometric factors that depend on the illumination/viewing geometry and the volume scattering function (VSF).

JXP has found after doing a rather deep-dive that: (1) there is a wavelength dependence to the `G_i` and (ii) they are (not surprisingly) sensitive to `bb`.  

Read the papers (PDFs) in `context/RT/`. These are:

- `hansen1971.pdf` : An RT paper by James Hansen in the context of planetary atmospheres.
- `gordon1988b.pdf` : The original Gordon approximation paper.
- `park2005.pdf` : A 4th order polynomial fit to the Gordon approximation and also includes Gordon coefficients that are a function of `gamma_b = bbp/bb`.
- `tan2018.pdf` : A recent examination of the `park2005.pdf` approach
- `pitarch2025.pdf` : A recent examination of bidirectional reflectance distributions (BRDF)

## Code

If you need to run Python use the `ocean14` conda environment. 

## Prompts

### Setup

1.  Let's start from the work done by JXP in the BING repo.  Please review that work. We will write a report named `context/rt_bing_summary.md`.  Before doing so, please have a conversation with me in Q&A/Setup.  Use Fable if you can.  Log your work. 

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Ask another round of questions if needed.  Use Fable if you can.  Log your work.

3. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Ask another round of questions if needed.  Use Fable if you can.  Log your work.

4. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Now proceed to write the report.  Be sure to include figures.  Generate them with Python and save the script(s) to the `context/RT/` folder. Use Fable if you can.  Log your work.  Note that I have put the L23 data on this laptop now in `$OS_COLOR_DATA/Loisel2023/`.

5. I decided to move the report into the `context/RT/` folder.  Please update it so the figures render properly.  Use Fable if you can.  Log your work.

### Edits

1. I've pasted in comments from Robert Frouin on the RT.  Please review them and revise the report accordingly.  Use Fable if you can.  Log your work.

### Design

1. Ok, let's generate a plan together for how to proceed.  We are going to write a design document for this elastic RT effort.  We will name it `design/rt_elastic_model.md`.  Please ask me a set of questions in Q&A/Design before writing the document.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Ask another round of questions if needed.  Use Fable if you can.  Log your work.

3. The design document looks great!  Please generate a separate coding plan with a staged implementation plan including milestones.  Name it `design/rt_elastic_model_coding_plan.md`.  Please ask me a set of questions in Q&A/Coding before writing the document.  Use Fable if you can.  Log your work.

4. I have answered your questions in the Q&A section below.  Please review them and proceed to write the coding plan.  Use Fable if you can.  Log your work.

5. The coding plan looks great!  Please generate a series of prompt docs to execute the coding plan.  Name them `claude_prompts/RT/rt_elastic_coding_prompt_1.md`, `claude_prompts/RT/rt_elastic_coding_prompt_2.md`, etc.  Model them after the ones in the PAB repository (in that `claude_prompts` folder). Use Fable if you can.  Log your work.

## Comments

### Robert's

Most analytical models other than ZTT (Twardowski and Tonizzo, 2018) do not treat the particle phase function, or parameters describing its shape, as explicit and independently adjustable inputs. Instead, phase-function effects are incorporated implicitly into coefficients or LUTs derived from RT simulations using prescribed phase functions. Consequently, the models cannot explicitly represent independent variability in particle phase-function shape. Because the phase function is a primary determinant of the angular distribution of water-leaving radiance, this limitation can introduce geometry-dependent forward-model errors and associated biases in retrieved IOPs.  ZTT largely addresses this limitation by introducing the backward VSF and related phase-function parameters explicitly into the analytical forward model.

My recommendation would be to use a full radiative-transfer solver (most naturally HydroLight) as the reference forward model, with particle phase-function parameters explicitly varied. For computational retrieval, construct a fast differentiable emulator of that solver. You may use ZTT as the principal analytical benchmark and possibly as the physical backbone of a hybrid emulator. Retain O25 as a BRDF/retrieval comparison model, not as the final physical reference.

A hybrid architecture would be particularly attractive:

Rrs(model) = Rrs(ZTT) + deltaRrs(simulator)

where ZTT supplies the physically interpretable backbone and a small emulator learns the remaining multiple-scattering and phase-function effects. This preserves physical scaling and geometry while avoiding the unrestricted behavior of a wholly black-box model.

## Q&A

### Design

#### Round 1 (2026-07-31)

We have the synthesis + roadmap (`context/RT/rt_elastic_model.md`) and Robert's steer:
a full-RT (HydroLight) **reference** with the phase function varied → a fast
**differentiable emulator** for retrieval → a **hybrid** `Rrs = Rrs(ZTT) + ΔRrs(emulator)`,
with O25/PR05 as comparisons; data L23-first then HydroLight/PB24; elastic-only;
geometry + phase-function shape first-class. The design doc `design/rt_elastic_model.md`
should turn that into a buildable plan. Six questions to scope it:

**DQ1 — Design-doc horizon & code home.** Which should the doc be?
  (a) Near-term buildable plan (~6–12 mo): stand up the reference-data generator, ZTT,
      a first emulator, and the hybrid on L23/PB24 — at module/API detail.
  (b) Full multi-year architecture through operational PACE retrieval + priors —
      conceptual + milestones.
  (c) Both, layered: a concrete near-term build section inside a longer architecture
      map. [Claude's lean]
  Also: should the forward model live as a new `robust/rt/` subpackage exposing a
  single `forward(iops, phase_params, geometry) → Rrs` API (my proposal)?

DQ1: (a) near-term buildable plan.  I anticipate it only taking one week to get our first high quality prototype.

**DQ2 — RT reference solver & access.** Robert says HydroLight "most naturally." Do we
have a HydroLight license/machine to run batches, or should the design target an
**open** solver as the reference of record (OSOAA, a polarized Monte Carlo, PlanarRad)?
  (a) HydroLight is available — make it the reference.
  (b) Design to a **solver-agnostic interface** with an open MC/OSOAA as the
      reproducible default, HydroLight as a cross-check where available. [lean]
  (c) Other / you'll provide access details.

DQ2: (a); and we have in hand a significant set of outputs from Loisel+2023

**DQ3 — Autodiff stack.** For the differentiable emulator + hybrid (and eventual
gradient-based inversion): **JAX** or **PyTorch**? And do we implement ZTT analytically
in the same framework so `Rrs = Rrs(ZTT) + ΔRrs(emulator)` is differentiable
end-to-end? (Lean: **JAX** — natural for differentiable physics, analytic ZTT, and
`vmap` over geometry; happy to use PyTorch if the team is PyTorch-native.)

DQ3:  JAX; and yes, we should implement ZTT analytically in the same framework so `Rrs = Rrs(ZTT) + ΔRrs(emulator)` is differentiable end-to-end.

**DQ4 — Phase-function parameterization (the explicit axis).** How should the particle
phase function enter as an explicit, adjustable input?
  (a) Fournier–Forand via (backscatter ratio, Junge slope) — matches L23/PB24.
  (b) ZTT-style **backward-VSF parameters** (β(π)/bb + a backward-shape term) as the
      primary axis, with FF (a) as the training-data generator so it ties to L23/PB24.
      [lean]
  (c) Henyey–Greenstein / two-term-HG mixtures.
  (d) A low-dimensional basis learned from measured VSFs (Petzold + field VSFs).

DQ4:  I am not sure about this one.  Let's start with a simple approach for now.  Your pick

**DQ5 — Inversion coupling / scope of this doc.** Is `design/rt_elastic_model.md`:
  (a) **Forward-model only** (reference + emulator + hybrid + validation); inversion
      deferred to a companion design doc. 
  (b) Forward **+ inversion**, and if so — reuse BING's MCMC, or move to
      **gradient-based / amortized simulation-based inference (SBI)** that exploits the
      differentiable forward model and the project's learned priors?
  (Lean: forward model in depth **+ a clear inversion interface and recommended path**
  — differentiable → gradient/SBI + priors — with the full inversion design deferred.)

DQ5:  Forward-model only, but differentiable.

**DQ6 — Acceptance criteria / validation protocol.** What defines "done" for the
forward model? Candidates to bake in: forward rRMS vs the reference (per λ, geometry,
phase-function shape); **held-out generalization** across geometry & phase function;
and the decisive one — **component retrieval MAPE** (a_ph, a_dg, bb_p) vs L23/PB24
truth. Hard target numbers, or **protocol-only** (report spreads, no blind target —
consistent with the "unbiased view" stance from the biomass report)? (Lean:
protocol-only with fixed held-out splits.)

DQ6:  This is difficult to answer.  We want something that is fast and accurate and differentiable.

### Coding

#### Round 1 (2026-07-31)

Questions to shape `design/rt_elastic_model_coding_plan.md` — the staged implementation
plan with milestones. (Locked by the design doc: JAX/Flax/Optax; `robust/rt/` with a
`forward(iops, phase_params, geometry)→Rrs` API; ZTT analytic backbone + residual
emulator hybrid; L23 elastic (3 solar zeniths) as the in-hand reference; `B_p` phase
param; forward-model-only, differentiable; ~1-week first prototype.)

**CQ1 — Execution model (most important).** Should Claude **implement** the prototype
now — scaffolding `robust/rt/` and building the stages on a branch, with JXP running all
git (commits/PRs to review), per CLAUDE.md — or is this coding plan a **spec/handoff**
for you (or others) to implement? (Lean: Claude implements on a branch; each milestone a
reviewable commit/PR; JXP does the git.)

CQ1: Claude will implement the prototype now.  I will run git and review the commits.

**CQ2 — Milestone structure & cadence.** How to stage it?
  (a) **Milestone-gated** (M1 scaffold+data → M2 ZTT-in-JAX → M3 emulator+hybrid →
      M4 validation → M5 HydroLight/BRDF+phase-function), each with an explicit
      acceptance gate; fine-grained (task-level) for the Week-1 milestones, coarser
      after. [lean]
  (b) A flat ordered task list.
  (c) Time-boxed (day-by-day for week 1).

CQ2: (a)

**CQ3 — ZTT equations source (a real dependency).** ZTT (Twardowski & Tonizzo 2018) is
the analytic backbone, but the paper is **not** in `context/RT/`. To implement
`Rrs_ZTT` in JAX we need its equations. Should I:
  (a) Source the paper / equations myself (web) and transcribe them;
  (b) You will provide the ZTT equations / a PDF;
  (c) Start with an **analytic placeholder backbone** (Gordon/O25 in JAX) so the
      pipeline is end-to-end first, then swap in true ZTT once its equations are in
      hand. [lean: (c) in parallel with (a)]

CQ3: I have provided the paper in `context/RT/` and named it twardowski2018.pdf 

**CQ4 — Testing, tooling & conventions.** Gate each milestone with **pytest** (unit +
the gradient-correctness check)? Formatting/linting (ruff/black), type hints
(jaxtyping)? Should I follow any **existing retrieve-or-bust / BING conventions** (test
layout, CI), or set them up fresh here? (Lean: pytest gates + ruff + light jaxtyping;
match BING's layout if present.)

CQ4: Yes, gate each milestone with pytest.  I will follow the BING conventions.

**CQ5 — Environment & compute.** `ocean14` almost certainly lacks `jax/flax/optax`.
Add them to `ocean14`, or create a dedicated env (e.g. `rt-jax`)? And is the Week-1
prototype **CPU-only** (fine for L23-scale), or is a **GPU** available if we want it?
(Lean: CPU-only JAX added to a dedicated env to avoid disturbing `ocean14`.)

CQ5: CPU-only for now.  Feel free to update the requirements file as needed and add packages to `ocean14` 

**CQ6 — Validation split policy (the acceptance gate).** For "held-out generalization,"
hold out (a) a **seeded random 20% of scenes**, and (b) **one solar-zenith angle** (e.g.
train 0°/30°, test 60°) to probe geometry generalization — with the M4 gate = beats
standard Gordon on both splits and passes the gradient check. Agree, or a different
split?

CQ6: Agree

### Setup

Claude has reviewed JXP's Gordon work in the BING repo (`prompts/gordon.md`
logs, `dev/Gordon/calc_gordon.py`, `dev/Gordon/plot_gordon.py`, the four
`gordon_coefficients*.csv` files, and `bing/rt/rrs.py`). Before writing
`context/rt_bing_summary.md`, the following questions:

**Q1 — Report purpose.** What should `rt_bing_summary.md` primarily be?
  (a) Orienting synthesis — a technical summary of the BING Gordon findings +
      open threads to orient the retrieve-or-bust RT work (narrative + result
      tables, no new analysis).
  (b) Decision doc — same synthesis, ending in a concrete recommendation on
      which recipe RT should adopt as its baseline.
  (c) Synthesis + roadmap — synthesis plus a prioritized list of the next RT
      experiments (validation, retrieval impact, cross-dataset), framed as this
      project's starting plan.

A1: (c)

**Q2 — Baseline recipe.** Which recipe should the report treat as the current
canonical / recommended baseline?
  (a) Two-stage 4-param (G0, Gb fit to residuals vs bbp(700), G1/G2 held fixed)
      — what currently ships in `gordon_coefficients_with_G0_Gb.csv`;
      interpretable but ~2.8% rRMS at red.
  (b) Joint 4-param (G0 + G1·u + G2·u² + Gb·bbp fit simultaneously) — best
      forward-model rRMS everywhere (0.3% red, 0.76% at 550 nm), but G1/G2
      absorb bbp structure.
  (c) Report both, present the trade-off, defer the choice to the pending
      retrieval-impact study.

A2: The Park & Ruddick (2005) approach.

**Q3 — Open threads.** How much emphasis on the unclosed validation gaps
(held-out set, IOP-retrieval impact, cross-dataset stability of G0/Gb)?
  (a) Foreground them as the central to-do for retrieve-or-bust — the
      forward-model rRMS wins are an optimistic in-sample ceiling until
      validated.
  (b) Note briefly as caveats; keep the focus on documenting what was done.

A3: Ignore those for now.

**Q4 — Plumbing depth.** How deep should the report go on the package wiring?
  (a) Findings-focused — science/results; mention flags and CSVs only enough to
      reproduce, not the file-by-file plumbing.
  (b) Include a plumbing map — document how it's wired (`variable_Gordon_G0` /
      `variable_Gordon_bbp` flags, the four CSVs, and the
      `rrs.py`/`anw.py`/`l23.py`/`evaluate.py` code paths) so RT can build on
      the code directly.

A4:  The report should not worry about the BING plumbing.

**Q5 — Anything else.** Is there context I'm missing — e.g. the deeper JXP
"deep-dive" finding on wavelength dependence and bb-sensitivity of the G_i that
rt_prompts.md alludes to, or a direction you already want RT to take beyond the
Gordon approximation (full RT, learned forward model, etc.)?

A5: I have just put a set of PDFs in the `context/RT/` folder.  Please read them as they are relevant to the report

### Setup — Round 2 (2026-07-19)

Claude has now read the five `context/RT/` PDFs (Gordon 1988, Park & Ruddick 2005,
Tan 2018, Pitarch 2025, Hansen 1971) and re-reviewed the BING Gordon work now that
the BING repo is local (`Oceanography/python/bing`). Note: `hansen1971.pdf` is a
scanned image with no extractable text layer (pypdf returns only the download
watermark) — see Q9.

Reading A1–A5 against the papers, one substantive tension surfaced, plus three
scoping questions.

**Q6 — Report identity / name.** A2 makes Park & Ruddick (2005) the baseline, A4
drops the BING plumbing, and A5 adds the RT literature. The deliverable is
therefore no longer really a "BING summary." Which should it be?
  (a) Keep the name `context/rt_bing_summary.md`; the BING findings are the spine,
      literature is supporting context.
  (b) Rename to a broader synthesis (e.g. `context/rt_forward_model.md`): the
      Gordon → PR05 → Lee2011/Pitarch-O25 lineage + the BING G0/Gb finding +
      roadmap, with PR05 named as the baseline.
  (c) Two files: a short BING-findings note + a separate literature/roadmap doc.

Q6:  (b)

**Q7 — Baseline forward model (the substantive one).** You chose PR05 (A2). But the
very papers you added argue PR05 is the *weakest* of the modern schemes:
  - Tan (2018): PR05 reconstructs Rrs well, but its retrieved parameters (esp. Chl)
    do not represent true values, and PR05-reconstructed Rrs fed to QAA gives badly
    biased IOPs — a caution given our component-separation goal.
  - Pitarch (2025) / D'Alimonte (2025): rank Lee-2011 (L11) > Morel-2002 > PR05;
    PR05 blows up the LUT dimension per γb and provides *no mechanism to retrieve
    γb = bbp/bb from Rrs*.
  - Pitarch's O25 replaces PR05's 4th-order polynomial in ωb (γb-indexed
    coefficients) with a bivariate quadratic that splits ωb into a water part
    ωbw = bbw/(a+bb) and a particle part ωbp = bbp/(a+bb), with coefficients that
    depend on geometry ONLY. This "Rrs is not univocal in u" insight is exactly what
    your BING G0/Gb terms captured empirically — and because retrieve-or-bust
    already separates bb_w (known) from bb_p, the O25 split maps onto our model for
    free (no γb iteration needed).
  Which does RT adopt as its forward-model baseline?
  (a) PR05 as written (4th-order in ωb, γb-indexed coefficients).
  (b) The L11 / Pitarch-O25 bivariate (ωbw, ωbp) form — modern successor,
      structurally aligned with BING's G0/Gb.
  (c) Report PR05 as the baseline per A2, but explicitly recommend O25/L11 as the
      evolution to adopt (reconcile A2 with the newer papers). [Claude's lean]

Q7: (c) and also emphasize that we are likely to construct our own new approach

**Q8 — BRDF / geometry scope.** PR05 and O25 are fundamentally bidirectional
(θs, θv, Δφ); the BING Gordon fits were at a fixed geometry. For the RT roadmap:
  (a) Fixed nominal geometry first (e.g. nadir view); defer full BRDF and treat the
      geometry-indexed LUTs as future work.
  (b) Build variable geometry / BRDF in from the start, making the geometry-indexed
      coefficients first-class.

Q8: (b)

**Q9 — Hansen (1971).** The PDF is a scan with no text layer, so I could not read
its content (only the "planetary atmospheres / doubling method" role you noted).
Should I (a) source a text/OCR'd copy so I can fold in its specifics, or (b) treat
it only as background (the multiple-scattering / doubling-method lineage behind the
u-polynomial) and proceed?

Q9: (b)

### Setup — Round 3 (2026-07-19)

Q6–Q9 fix the report scope: a broad synthesis named `context/rt_forward_model.md`;
PR05 as the named baseline with O25/L11 recommended as the evolution and our own new
approach flagged; BRDF/variable geometry first-class; Hansen as background. Before
writing, three questions remain that materially shape the roadmap.

**Q10 — "Our own new approach" (from Q7).** You said retrieve-or-bust is likely to
construct its own forward model. Which flavor should the roadmap foreground?
  (a) Extend the analytic / physically-structured line — more terms or a better
      parameterization in the BING-G0/Gb ↔ O25-(ωbw, ωbp) family (interpretable,
      few parameters, keeps the inversion analytic).
  (b) Learned forward model — a neural emulator of RT mapping (IOPs, geometry) → Rrs,
      replacing the polynomial (max accuracy, differentiable for retrieval, but a
      black box).
  (c) Hybrid — an analytic backbone (O25-style water/particle split) plus a learned
      residual / coefficient network (keeps physics, learns what the polynomial
      misses). [Claude's lean]
  (d) Undecided — lay all three out as options in the roadmap and choose later.

Q10: (d)

**Q11 — Elastic-only scope.** The doc is titled *Elastic* RT, and Gordon/PR05/O25 are
all elastic (no Raman / fluorescence); BING already carries `raman.py` and
`chl_fl.py` separately. Should the report:
  (a) Stay strictly elastic — treat inelastic (Raman, CDOM/chl fluorescence) as out
      of scope, noted only as a boundary.
  (b) Cover elastic as the core but include an explicit inelastic-extension section
      in the roadmap.

Q11: (a).  Indeed, let's name it `context/rt_elastic_model.md` 

**Q12 — RT data / truth for a BRDF-capable forward model.** Q8(b) wants variable
geometry first-class, but the BING fits used L23 (Loisel 2023), which is effectively
a single nominal geometry — a multi-angular truth set is needed. Which source should
the roadmap build on?
  (a) Pitarch & Brando 2025 (PB24) — the open multi-angular synthetic set O25 was
      built on (5000 IOPs × 1300 geometries, Fournier–Forand); reuse it.
  (b) Generate our own RT runs (Hydrolight or an open RT code) over a chosen
      IOP × geometry grid — full control, more work.
  (c) Start on L23 at its fixed geometry to reproduce/extend the BING result, then
      add geometry via PB24 or new runs as a second phase. [Claude's lean]

Q12: (c)

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-07-16 (Review BING Gordon work; open Q&A/Setup before writing rt_bing_summary.md)

Executed the 1st Setup prompt: reviewed JXP's wavelength-dependent Gordon work
in the BING repo and posed setup questions (Q1–Q5 above) rather than writing the
report yet.

**Sources reviewed.**
- `bing/prompts/gordon.md` — the driving prompt doc + full Logs (2026-05-31 →
  2026-06-03), which is the authoritative record of the development arc.
- `bing/dev/Gordon/calc_gordon.py` (1726 lines) — all fit recipes.
- `bing/dev/Gordon/plot_gordon.py` — assessment figures (re-exported).
- `bing/dev/Gordon/gordon_coefficients*.csv` (4 files: 2-param, +G0, +Gb, +G0+Gb)
  and the deployed copies under `bing/bing/data/RT/`.
- `bing/bing/rt/rrs.py` — `wave_dependent_gordon*`, `calc_Rrs`,
  `calc_elastic_Rrs` (how the coefficients are consumed).

**What I learned (to be written up in rt_bing_summary.md after Q&A).**
- The physical crux: standard Gordon `rrs = G1·u + G2·u²` makes `rrs` a function
  of `u = bb/(a+bb)` alone, but the Hydrolight (L23) truth depends on `(a, bb)`
  separately. The wavelength-dependent coefficients + the extra G0 (constant
  offset) and Gb (bbp slope) terms capture that residual (a,bb) structure —
  matching the rt_prompts.md note that the G_i are λ-dependent and bb-sensitive.
- Two original bugs fixed: (1) `B=1.17` vs `1.7` Rrs↔rrs convention mismatch;
  (2) constant-σ (unweighted) fit letting G2 run away at red λ. Relative
  weighting + bounds + Tikhonov smoothness on G1(λ),G2(λ) made variable-Gordon
  beat standard everywhere.
- Recipe ladder, rRMS vs Hydrolight (3320 L23 scenes): standard → 2-param smooth
  → +G0 → +Gb → G0+Gb. G0 wins red (700 nm 9.05%→0.35%), Gb wins blue (400 nm
  2.54%→1.84%), joint 4-param wins/ties everywhere (550 nm 0.76%). All plumbed
  into the package (opt-in flags `variable_Gordon_G0`, `variable_Gordon_bbp`);
  80 tests pass.
- Key tension: the joint 4-param is best but lets G1/G2 absorb bbp structure;
  JXP then adopted a two-stage fit (Stage 1: G1,G2; Stage 2: G0,Gb vs bbp(700))
  for interpretability at the cost of red-λ rRMS (~2.8%). The two-stage recipe is
  what currently ships in `gordon_coefficients_with_G0_Gb.csv`.
- Recurring un-closed recommendations across every log entry: no held-out /
  out-of-sample validation, no retrieval-impact (IOP MAPE) study, no
  cross-dataset (PACE/MOBY) test of whether G0/Gb are structural or L23-specific.

Next: await answers to Q1–Q5, then write `context/rt_bing_summary.md`.

### 2026-07-19 (Read the five context/RT PDFs; BING repo now local; posed Round 2 Q&A before writing the report)

Executed the 2nd Setup prompt: reviewed answers A1–A5, read the five RT papers, and
posed a second round of questions (Q6–Q9 in Q&A above) driven by a tension between
A2 (PR05 baseline) and the newer papers JXP added.

**Sources read.**
- Extracted text from all five PDFs with `pypdf` (ocean14) — `pdftoppm`/poppler is
  not installed, so the Read tool's page-render path is unavailable; text extraction
  was sufficient for four of the five.
- `park2005.pdf` (PR05) — read in full. QSSA-derived Rrs model; 4th-order polynomial
  Rrs = Σ_{i=1..4} g_i(θo,θ,Δφ,γb)·ωb^i with ωb = bb/(a+bb) (their "backscattering
  albedo"); coefficients tabulated over Sun/sensor angles and γb = bbp/bb (particle
  fraction of backscatter, 0.2–1). Hydrolight 4.2, Fournier–Forand phase functions,
  case-1+case-2 IOPs. Model uncertainty ~2% (rms ~1%), dominated by residual
  phase-function variability. γb must be estimated (iteratively) before inversion;
  ~2% Rrs accuracy needs γb to ~0.05 (≈20–30% bbp at low γb, worse at high γb).
- `gordon1988b.pdf` — read (agent). Canonical R/Q = l1·u + l2·u², l1=0.0949,
  l2=0.0794; θo>20°, u≤~0.2; the l_i are treated wavelength-independent (λ-dependence
  parked in a, bb, and Q≈4–5). R→Rrs interface factors: (1−ρ)/m² ≈ 0.54, (1−rR),
  r=0.48. Max rms error of the scheme ~±20%; i>1 term matters at high radiance.
- `tan2018.pdf` — read in full. Evaluates PR05 in POLYMER vs Hydrolight (IOCCG L23)
  + AERONET-OC. PR05 reconstructs Rrs and band ratios well (RMS <15%), BUT retrieved
  Chl is biased (a fit parameter, not physical) and PR05-reconstructed Rrs → QAA
  gives badly biased IOPs. Recommendation: use reconstructed reflectance, not
  retrieved parameters. Two-param PR05 preferred over three-param (robustness).
- `pitarch2025.pdf` (O25) — read (agent). BRDF-normalization successor in the
  Gordon→Lee-QAA→L11 line. Bivariate quadratic Rrs = (Gw0+Gw1·ωbw)·ωbw +
  (Gp0+Gp1·ωbp)·ωbp with ωbw=bbw/(a+bb), ωbp=bbp/(a+bb); the four G-coeffs depend on
  geometry ONLY. Calibrated on PB24 (Fournier–Forand, 5000 IOPs × 1300 geometries).
  Ranks L11 > M02 > PR05. Physical crux: Rrs is *not* univocal in ωb — two branches
  split by ηb = bbw/bb, because at single scattering Rrs ∝ β(π)/bb (0.23 sr⁻¹ water
  vs 0.12–0.16 particles), so water vs particle backscatter must be separated. Code
  open-source (github.com/jaipipor/O25); operational in EUMETSAT OLCI Collection 4.
- `hansen1971.pdf` — **could not read**: scanned image, no text layer (pypdf returns
  only the "Unauthenticated | Downloaded …" watermark). Role per JXP: multiple-
  scattering RT / doubling method in planetary atmospheres. Flagged as Q9.
- Re-confirmed the BING Gordon artifacts now that the repo is local:
  `bing/dev/Gordon/{calc,plot}_gordon.py`, four `gordon_coefficients*.csv`, deployed
  copies under `bing/bing/data/RT/`, and `bing/bing/rt/rrs.py`.

**Key realization to carry into the report.** The central physical point is the same
across the BING G0/Gb finding and Pitarch's O25: **Rrs is not a function of
u = bb/(a+bb) alone** — the residual (a, bb) / water-vs-particle structure must be
represented. BING captured this with extra G0 (offset) + Gb (bbp slope) terms; O25
captures it structurally by splitting u into ωbw (water) and ωbp (particle) with
geometry-only coefficients. Because retrieve-or-bust separates bb_w (known) from
bb_p, the O25 split is essentially free for us — which is why Q7 asks whether to
keep PR05 (A2) as the literal baseline or evolve to the L11/O25 bivariate form.

Next: await answers to Q6–Q9, then write the report (name per Q6).

### 2026-07-19 (Reviewed Q6–Q9; held off on the report to pose Round 3 on the roadmap)

Executed the 3rd Setup prompt: reviewed the Round-2 answers (Q6=b broad synthesis
`rt_forward_model.md`; Q7=c PR05 baseline + O25/L11 recommendation + our-own-approach;
Q8=b BRDF first-class; Q9=b Hansen as background). Grounded the BING numbers directly
from the now-local repo — the four `gordon_coefficients*.csv` headers/recipes and the
per-wavelength rRMS ladder (standard → smooth 2-param → +G0 → +Gb → joint 4-param) and
two-stage shipped fit — so the synthesis tables will be exact.

At the user's request, paused before writing to pose Round 3 (Q10–Q12 above): the
direction of "our own new approach", elastic-only vs inelastic-extension scope, and
the RT truth dataset for a BRDF-capable model (L23 is single-geometry; a multi-angular
set like PB24 is needed for Q8=b). These three shape the roadmap section.

Next: await answers to Q10–Q12, then write `context/rt_forward_model.md`.

### 2026-07-19 (Wrote context/rt_elastic_model.md with three L23-derived figures)

Executed the 4th Setup prompt. Reviewed the Round-3 answers (Q10=d lay out all three
forward-model options; Q11=a strictly elastic, renamed the report to
`context/rt_elastic_model.md`; Q12=c L23 first, PB24 phase 2) and wrote the report.

**Figures — generated with Python, saved to `context/RT/`.** New script
`context/RT/make_rt_elastic_figures.py` (ocean14). It reads the L23 **elastic** set
directly (`$OS_COLOR_DATA/Loisel2023/Hydrolight100.nc`, i.e. `loisel23.load_ds(1,0)`,
X=1 = no inelastic — the same scenario the BING Gordon fits used, confirmed in
`calc_gordon.py`) and the BING coefficient CSVs. Convention `rrs = Rrs/(A+B·Rrs)`,
A=0.52, B=1.7. Three figures:
- `fig_rrs_vs_u.png` — the central point: at 440/550/665 nm, `rrs` vs `u` colored by
  `bbp` with the standard Gordon curve overlaid. Shows the `bbp` fan (blue) and the
  near-constant red-λ offset — `rrs` is not univocal in `u`.
- `fig_rrms_ladder.png` — per-λ rRMS recomputed from L23: standard → per-λ 2-param →
  `+G0` → `+Gb`. **Recomputation reproduces the BING logs** (standard 700 nm 9.04%
  vs logged 9.05%; `+G0` 550 nm 1.12% vs 1.11%; `+G0` 700 nm 0.35%), so the ladder is
  authentic, not transcribed. `+G0` wins red, `+Gb` wins blue — complementary.
- `fig_G_lambda.png` — `G0(λ)` (sign change ~510 nm) and `Gb(λ)` from the shipped
  BING CSVs.

**Report `context/rt_elastic_model.md`** (per Q6=b broad synthesis, Q11 rename;
A1=c synthesis+roadmap; A4 no plumbing; A3 validation gaps not foregrounded).
Sections: (1) why the forward model gates the retrieval; (2) the organizing fact —
`rrs ≠ f(u)` — with the L23 figure and the `β(π)/bb` = 0.23 (water) vs 0.12–0.16
(particle) physics from Pitarch; (3) the elastic lineage Gordon 1988 → PR05
(baseline) → Tan 2018 → L11/O25 (recommended evolution) → Hansen (background);
(4) the BING `G0`/`Gb` deep-dive with the two data figures, and the explicit
convergence BING-`G0/Gb` ↔ O25-`ωbw/ωbp`; (5) a one-table synthesis; (6) baseline =
PR05, recommend evolving to O25/L11, and our-own-approach kept open as three options
(analytic / learned / hybrid, per Q10=d); (7) a 5-step roadmap with BRDF first-class
(Q8=b) and L23→PB24 truth data (Q12=c), inelastic explicitly out of scope (Q11=a);
(8) references.

Learned/confirmed: L23 elastic is 3320 scenes × 81 λ (350–750 nm); `bbnw` is the
particle (non-water) backscatter, `bb_w = bb − bbnw`. The recomputed ladder is
independent corroboration of the BING result from the raw Hydrolight data.

Next: await review of `context/rt_elastic_model.md` (and any request to add the O25
bivariate curve to the ladder figure, per roadmap step 1).

### 2026-07-19 (Report moved to context/RT/; fixed figure paths)

Executed the 5th Setup prompt. JXP had already moved the report to
`context/RT/rt_elastic_model.md` (co-located with the figures and the figure
script). Updated the three image links from `](RT/fig_*.png)` to `](fig_*.png)` so
they resolve now that the report sits in `context/RT/`. Verified all three links
(`fig_rrs_vs_u.png`, `fig_rrms_ladder.png`, `fig_G_lambda.png`) point at the
co-located PNGs. The remaining `RT/`-prefixed strings in the report are prose repo
paths (Hansen PDF, BING CSVs, the figure script), not image links, and stay correct.
No content changes. The figure script's `HERE`-based output path already writes into
`context/RT/`, so re-running it stays consistent with the new location.

### 2026-07-31 (Revised context/RT/rt_elastic_model.md per Robert Frouin's RT comments)

Executed the 1st Edits prompt: reviewed R. Frouin's pasted comments and revised the
report. His two thrusts — (1) most analytical models bury the particle phase function
in coefficients/LUTs from a *prescribed* phase function and so cannot represent
independent phase-function-shape variability, causing geometry-dependent forward-model
error and IOP bias, with **ZTT (Twardowski & Tonizzo 2018)** the exception; (2) target
a **full-RT (HydroLight) reference** with phase-function parameters varied, a **fast
differentiable emulator** for retrieval, **ZTT** as analytical benchmark/backbone, the
**hybrid** `Rrs = Rrs(ZTT) + ΔRrs(emulator)`, and **O25 demoted to a comparison model**.

**Changes made.**
- Header: added an "Advisory input (R. Frouin)" note; broadened scope to
  `Rrs(λ; a, bb, phase function, geometry)`; added ZTT + HydroLight to the arc.
- §2: new paragraph making the **phase-function shape** the deeper axis behind
  `rrs ≠ f(u)` — most models hide it; only ZTT exposes it; it drives geometry-dependent
  bias. Framed as the single most important structural gap for our goal.
- §3.4: retitled O25 from "the recommended evolution" to "the modern semi-analytical
  **benchmark**"; noted its phase function is implicit (PB24 FF), so it is a
  BRDF/retrieval *comparison*, not the physical reference.
- §3.5 (new): **ZTT** — backward VSF / phase-function parameters explicit; analytical
  benchmark and candidate physical backbone. Hansen renumbered to §3.6.
- §5 table: added **ZTT** and **HydroLight (reference)** rows and a "Phase function"
  column (prescribed/implicit vs explicit/adjustable); rewrote the through-line to end
  on the phase-function axis.
- §6: restructured into three roles — **physical reference = HydroLight** (PF varied);
  **analytical benchmark/backbone = ZTT**; **comparison = PR05, O25/L11**; and the
  retrieval operator as our-own-approach with the **hybrid (c) now advisor-recommended**
  (`Rrs = Rrs(ZTT) + ΔRrs(emulator)`), keeping Q10=d's three options.
- §7 roadmap: now 6 steps — reproduce on L23 (+ overlay O25 & ZTT) → retrieval-impact
  MAPE → **build the HydroLight reference with the phase function varied** (PB24 in
  parallel) → **stand up ZTT** benchmark → **differentiable emulator + hybrid**, O25/PR05
  as comparisons → conventions. Geometry **and phase-function shape** are first-class.
- §8: added Twardowski & Tonizzo (2018) and Mobley/HydroLight references.

Note: reconciled with prior answers — Q7 already flagged "our own new approach" and
Q10=d left it open, so Robert's steer *fills in* the concrete architecture (hybrid,
ZTT backbone, HydroLight reference) and reframes O25 from "evolution to adopt" to
"comparison model" without contradicting JXP. Did not read the ZTT PDF (not in the
folder); represented it from Robert's description + the β(π)/bb physics already cited.

Next: await review; if wanted, add a ZTT/O25 overlay to the L23 ladder figure
(roadmap step 1) and/or source the ZTT paper to deepen §3.5.

### 2026-07-31 (Posed Round-1 Design Q&A before writing design/rt_elastic_model.md)

Executed the 1st Design prompt: posed six questions (DQ1–DQ6 in Q&A/Design) to scope
the design document, which will turn the synthesis roadmap + Robert's architecture
(HydroLight reference → differentiable emulator → ZTT-backbone hybrid; O25/PR05 as
comparisons; L23→HydroLight/PB24; elastic-only) into a buildable engineering plan.

The questions target the concrete build decisions the synthesis left open: DQ1
doc horizon (near-term buildable vs full multi-year vs layered — lean layered) and
code home (a `robust/rt/` subpackage with a single `forward(iops, phase_params,
geometry)→Rrs` API); DQ2 the RT reference solver & whether HydroLight is actually
available vs a solver-agnostic interface with an open MC/OSOAA default; DQ3 the autodiff
stack (JAX vs PyTorch; ZTT analytic in-framework for end-to-end differentiability —
lean JAX); DQ4 the phase-function parameterization (FF / ZTT backward-VSF / HG / learned
VSF basis — lean ZTT backward-VSF as primary, FF as the training-data generator); DQ5
inversion coupling / scope (forward-only vs forward+inversion; reuse BING MCMC vs
gradient/amortized SBI with priors — lean forward-in-depth + inversion interface,
full inversion deferred); DQ6 acceptance criteria (protocol-only with held-out
geometry/phase-function splits + component-retrieval MAPE, no blind target — consistent
with the biomass report's stance). Gave my lean on each.

Next: await DQ1–DQ6, then write `design/rt_elastic_model.md`.

### 2026-07-31 (Reviewed DQ answers; wrote design/rt_elastic_model.md)

Executed the 2nd Design prompt. The DQ answers were complete enough to write the design
doc without another Q&A round (DQ4 was explicitly delegated to me; DQ6 gave clear
acceptance themes), so I proceeded rather than asking again.

**DQ answers applied.** DQ1=(a) near-term buildable, ~1-week first prototype → the doc
centers a concrete Week-1 milestone. DQ2=(a) HydroLight available + L23 outputs in hand
→ prototype uses L23 as the in-hand reference; confirmed L23 elastic (X=1) exists at 3
solar-zenith angles (`Hydrolight100/130/160.nc` = 0/30/60°, nadir, fixed FF), so the
solar-zenith geometry axis is exercisable now. DQ3=JAX + analytic ZTT in-framework,
end-to-end differentiable. DQ4="your pick, keep it simple" → I chose the **particulate
backscattering ratio B_p = bb_p/b_p** (Fournier–Forand, one scalar; directly available
in L23 as bbnw/bnw), with the ZTT backward-VSF parameterization documented as the
planned generalization. DQ5=forward-model only but differentiable → inversion out of
scope, API kept inversion-ready. DQ6="fast, accurate, differentiable" → a protocol-only
validation section (no blind targets): accuracy (rRMS vs reference per λ/geometry/B_p +
held-out generalization, alongside Gordon/PR05/O25), speed (throughput/latency), and a
gradient-correctness gate (jax.grad vs finite differences).

**Doc `design/rt_elastic_model.md`** (new `design/` dir): goals/non-goals; the hybrid
architecture diagram; the `robust/rt/` subpackage layout + `forward(iops, phase_params,
geometry) → Rrs` API; components (reference data, phase-function param, ZTT-in-JAX,
residual emulator, hybrid); JAX/Flax/Optax stack + conventions (A=0.52, B=1.7); the §6
validation protocol; the §7 one-week prototype definition and sequence; the beyond-week-1
forward-model track; and a §9 "decisions I made on your behalf" list flagging the DQ4
pick, the prototype target, and the protocol-only acceptance for correction.

Next: await review of `design/rt_elastic_model.md`; on approval, scaffold `robust/rt/`
and build the Week-1 prototype.

### 2026-07-31 (Posed Round-1 Coding Q&A before writing the coding plan)

Executed the 3rd Design prompt: posed six questions (CQ1–CQ6 in Q&A/Coding) to scope
`design/rt_elastic_model_coding_plan.md` (the staged implementation plan + milestones).
The design doc already fixed architecture/stack, so these target the process/
implementation decisions still open: CQ1 execution model (Claude implements on a branch
with JXP running git, vs a spec/handoff — lean Claude implements); CQ2 milestone
structure (milestone-gated M1–M5 with acceptance gates, fine-grained for week 1);
CQ3 the **ZTT equations dependency** — the paper isn't in `context/RT/`, so source it /
JXP provides / start with a Gordon-O25 analytic placeholder backbone and swap ZTT in
(lean: placeholder now + source in parallel); CQ4 testing/tooling (pytest gates incl.
the gradient check, ruff, jaxtyping; match BING conventions); CQ5 environment (add
jax/flax/optax to a dedicated env vs ocean14; CPU vs GPU); CQ6 the validation split
policy / M4 gate (random-20% scenes + hold out one solar-zenith angle). Gave my lean on
each. CQ3 and CQ1 are the load-bearing ones.

Next: await CQ1–CQ6, then write `design/rt_elastic_model_coding_plan.md`.

### 2026-07-31 (Reviewed CQ answers; wrote design/rt_elastic_model_coding_plan.md)

Executed the 4th Design prompt. CQ answers: CQ1 Claude implements now on a branch, JXP
runs git & reviews; CQ2 milestone-gated; CQ3 the ZTT paper is now in the repo
(`context/RT/twardowski2018.pdf`, confirmed 30 pp, extractable); CQ4 pytest-gate each
milestone, follow BING conventions; CQ5 CPU-only, may update requirements + ocean14;
CQ6 agreed split (random 20% scenes + hold out 60° zenith).

Recon before writing: confirmed retrieve-or-bust layout (`robust/` package, one
`__init__.py`; requirements.txt already has pytest/xarray/ocpy/bing; no tests yet), and
BING's test convention (`bing/bing/tests/test_*.py` + `conftest.py` + `files/`) → mirror
as `robust/tests/`. Confirmed the ZTT PDF is Twardowski & Tonizzo 2018 (*Applied
Sciences*, VSF-explicit ocean-color model) and extractable for M2.

**Doc `design/rt_elastic_model_coding_plan.md`.** Ground rules from the CQ answers;
`robust/rt/` + `robust/tests/` layout mirroring BING; a milestone table and detailed
**M0–M5**: M0 env+scaffold (add jax/flax/optax CPU, update requirements) → M1 data+
conventions (L23 loader via ocpy, A/B, `B_p`, golden-value gate) → M2 ZTT-in-JAX
(transcribe twardowski2018; gates = reproduce a paper reference case + grad-vs-FD) →
M3 emulator+hybrid (Flax MLP ΔRrs, Optax, relative weighting; gate = beat standard
Gordon + grad check) → **M4 validation = prototype done** (CQ6 held-out splits; gate =
beat Gordon on both splits + grad gate; table+figures) → M5 beyond-week-1 (HydroLight
PF/BRDF, ZTT backward-VSF params, freeze API). Each milestone pytest-gated; the
gradient-correctness and golden-value tests are first-class; accuracy gates are
*relative* (no blind targets). Risks section keeps the Gordon/O25-in-JAX placeholder
backbone as ZTT de-risking.

Next: on approval, create the `rt-elastic-prototype` branch and start M0.

### 2026-07-31 (Generated the M0–M5 coding prompt docs, modeled on PAB)

Executed the 5th Design prompt: created six execution prompt docs
`claude_prompts/RT/rt_elastic_coding_prompt_{1..6}.md`, one per milestone (M0→1 … M5→6),
modeled after the PAB repo's `claude_prompts/coding_stage*.md` / `initial_coding.md`
style.

Studied PAB's pattern first (`PAB/claude_prompts/{start_up,initial_coding,coding_stage4}.md`):
each stage doc has Goals → Claude (Skills + Working agreements) → Context (files to read
+ existing seams) → Prompts (numbered, "Execute the Nth task in section X") → the
Stage/Tasks section with explicit gates → Q&A → pointer to next → Logging/Logs; PAB also
keeps a running `*_implementation.md` record.

Each RT prompt doc mirrors that: shared Working agreements (git by JXP on the
`rt-elastic-prototype` branch; ocean14; CPU-only JAX; reuse ocpy/bing; BING test layout;
pytest-gated; Fable; log), milestone-specific Context, numbered Prompts, a gated Tasks
section, an empty Q&A, and a Next pointer. Mapping: **1**=M0 env+scaffold (+ creates the
running record `design/rt_elastic_implementation.md`); **2**=M1 data+conventions (L23
loader, B_p, splits, golden-value gate); **3**=M2 ZTT-in-JAX (transcribe
`twardowski2018.pdf`; reference-case + grad-vs-FD gates; Gordon/O25 placeholder de-risk);
**4**=M3 emulator+hybrid (beat-Gordon + grad + throughput gates); **5**=M4 validation =
prototype done (CQ6 held-out splits; acceptance gate; a PR section like PAB's);
**6**=M5 beyond-week-1, deliberately a coarse placeholder to be detailed from M4 results
(HydroLight PF/BRDF, PB24, ZTT backward-VSF params, freeze API).

Next: on approval, execute `rt_elastic_coding_prompt_1.md` (M0).
