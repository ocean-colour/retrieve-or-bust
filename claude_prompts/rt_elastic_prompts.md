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

## Q&A

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
