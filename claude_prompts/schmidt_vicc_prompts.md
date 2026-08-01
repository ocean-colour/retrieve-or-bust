# Schmidt VICC Application Prompts

We will use this doc to prompt the generation of an application for the Schmidt Sciences VICC program.

## Goals

First, see the Goals in the other prompt files in this folder.

Second, this file will guide the generation of the application, starting with an EOI.

## Context

See the following:

- context_summary.md
- The Request for Expressions of Intent PDF named `VICC_Phase2_EOI.pdf` in the `proposals/Schmidt_Sciences` folder.
- The VICC program: `https://www.schmidtsciences.org/vicc/`

Read the papers in the `context/papers/Biomass` folder.

## Prompts

1. Perform the 1st Task under "EOI"
2. Perform the 2nd Task under "EOI"
3. Perform the 3rd Task under "EOI"
4. Perform the 4th Task under "EOI"
5. Perform the 5th Task under "EOI"
6. Perform the 6th Task under "EOI"
7. Perform the 7th Task under "EOI"

## EOI

1. Read all of the files in the Context section above.  We are going to discuss possible angles for a research proposal for the VICC program.  Put your questions in the Q&A section.  Log your work.  Use Fable if you can.

2. Read all of the literature in the `context/papers/Biomass` folder which discuss aspects of biomass on Earth.  Then search for the latest related papers.  Then write a summary of the literature in `reports/biomass_summary.md` with emphasis on the uncertainty in the primary carbon measurements.
Log your work. Use Fable if you can.

3. Please ask me questions about the biomass literature.  You will use my responses to improve your report.  Use Fable if you can.  Log your work.

4. I have answered your BQ questions.  Please use my responses to improve your report.  Be sure to include figures in the report.  Generate them with Python scripts and place those in one or moduels in the `reports/py` folder. Use Fable if you can.  Log your work.

5. Here are additional items/questions for you to address in the report:

 .  - How reliable is the 5-15 Gt C yr^-1 estimate of export from the literature?  Please generate a pie chart (or similar figure) describing the error budget.
    - Please add DOIs for all of the references in the report.
    - The current uncertainties in the report are unclear as to whether these are 1-sigma, 68% confidence, or other.  Please clarify each.  Ideally, provide 95% confidence intervals.

6. Thanks.  The title of the report says "from Optics".  Is there another approach for the ocean?  If so, include its measurements and uncertainties.  If not, make that clear.  Use Fable if you can.  Log your work.

7. One of the collaborators (Matt Mazloff) asks:  "does this assume the pure physical component uncertainty is negligible? Is the working hypothesis that determining NCP is the key to reducing ocean carbon inventory uncertainty? Do we know how the carbon content variability from physical components compares to that from biological components?". Please address these questions in the report.  Use Fable if you can.  Log your work.

### Q&A

#### Round 1 (2026-07-27)

I read the EOI RFP (`VICC_Phase2_EOI.pdf`), the VICC program page, and
`context_summary.md`. Below: the key EOI facts, the bridge from retrieve-or-bust to
carbon, four candidate angles, and questions to pick a direction.

**EOI facts (constraints that shape the pitch).**
- Mission: reduce global-carbon-cycle uncertainty **by an order of magnitude** to
  inform climate policy. Three themes — **T1** anthropogenic emissions & their fate /
  global-carbon-budget accuracy (incl. *ocean interior carbon cycle*, *trends in
  carbon sinks & variability*, *land–ocean–aquatic continuum*); **T2** policy-relevant
  timescales / attribution of natural vs anthropogenic / *near-real-time flux
  monitoring*; **T3** future land/ocean stocks & fluxes, decadal–centennial, surprises
  & abrupt change.
- Explicitly invites **AI/ML**: "novel inversion and data assimilation approaches,
  process-model emulation, up-/down-scaling."
- Format: **3 pages max, blind/anonymized** (no names/affiliations in the doc; a
  separate form collects team), Vancouver refs in a separate PDF. **Deadline Aug 28,
  2026.** Typical award **$10M / 5 yr** (start Oct 1 2027; indirect ≤10%); *smaller,
  shorter exploratory awards explicitly welcomed* for "highly exploratory,
  unconventional ideas… tightly scoped." 5–7 projects funded. International/
  multidisciplinary teams and ECR-lead PIs encouraged. Existing ocean-carbon projects
  to differentiate from / complement: **COCO2** (Southern-Ocean air–sea CO₂ on USVs)
  and **OBVI/InMOS** (models+obs integration).

**The bridge (retrieve-or-bust → carbon).** Ocean-color IOPs are carbon proxies:
`bbp` → particulate organic carbon (POC); phytoplankton absorption / `Cphyto` →
phytoplankton carbon & community composition; and together they drive net primary
production (NPP) and export — the **biological carbon pump**. The catch VICC cares
about: today's satellite POC/NPP products carry large, poorly-quantified uncertainty,
much of it inherited from the ill-posed IOP inversion. retrieve-or-bust's engine
(AI + priors + *honest uncertainty quantification*, on PACE-class hyperspectral) is
precisely a tool to shrink and *characterize* that uncertainty — a carbon-relevant
reframing of the project's existing thesis.

**Candidate angles.**
- **A — Particulate & phytoplankton carbon stocks.** Global POC + `Cphyto` (and
  composition) from hyperspectral color with AI/priors and calibrated
  per-pixel uncertainty. Anchors T1 (ocean interior carbon; sink-trend variability).
- **B — The biological carbon pump.** Reduce the *optical* uncertainty in
  satellite NPP & export by fixing the IOP bottleneck, propagating to carbon-flux
  uncertainty. T1 + T3.
- **C — Near-real-time, attribution-grade monitoring.** Fuse PACE + in-situ +
  time-series priors for policy-timescale ocean-carbon monitoring and natural-vs-
  forced attribution. T2.
- **D — Exploratory small award.** A tightly-scoped proof-of-concept: *can AI + priors
  break the IOP degeneracy enough to categorically cut ocean-carbon uncertainty?* —
  matches retrieve-or-bust's current lean identity and VICC's "exploratory" lane.

**Questions.**

**VQ1 — Carbon target.** Which carbon quantity is the headline deliverable?
  (a) POC + phytoplankton carbon **stocks** (+composition). (b) **NPP / export** (the
  biological pump). (c) A **chain** IOPs → POC/`Cphyto` → NPP/export, all with
  propagated uncertainty. (d) Something else (air–sea CO₂ context, PIC/calcification…).
  And: can we credibly promise a **categorical / order-of-magnitude** uncertainty
  reduction for the chosen quantity — do you already have a defensible target/number?

VQ1:

**VQ2 — Which VICC theme leads?** T1 (budget accuracy / ocean interior carbon / sink
trends), T2 (policy timescales / attribution / near-real-time), or T3 (future,
surprises)? (My lean: **T1 primary, T2 secondary**.)

VQ2:

**VQ3 — Scope & budget (the big fork).** A **flagship ~$10M/5-yr** program (retrieval
engine + global carbon products + validation network + team), or a **tightly-scoped
exploratory** smaller/shorter award (the AI-methods proof-of-concept, angle D)? This
sets everything downstream.

VQ3:

**VQ4 — Data posture.** (a) Satellite-retrieval methods core (AI inversion → carbon +
UQ). (b) + fuse **in-situ carbon** (BGC-Argo bbp/POC, HPLC, PIC) as priors *and*
validation. (c) + couple to **biogeochemical models / data assimilation**. (My lean:
**(b)** — retrieval core with in-situ priors & validation; DA as a stretch goal.)

VQ4:

**VQ5 — Differentiation & collaboration.** Position retrieve-or-bust as the **global,
hyperspectral, satellite** ocean-carbon complement to the in-situ-heavy COCO2 and the
model-obs InMOS, and propose explicit cross-project collaboration (shared data /
validation)? Or stake out a more independent lane?

VQ5:

**VQ6 — Team & PI.** EOI is blind, PI on only one EOI, international & ECR-lead
encouraged. Who is the intended **PI**? The current group (Bontemps/URI,
Dierssen/UConn, Housekeeper/WHOI, Frouin/SIO, Kavanaugh/OSU, Kudela/UCSC,
Prochaska/UCSC) is strong on ocean-color→biogeochemistry — do we add a
**carbon-flux / BGC modeler**, an **air–sea CO₂ / BGC-Argo** partner, and/or an
**international** institution to fit VICC's team expectations?

VQ6:

#### Round 2 — Biomass literature (2026-07-30)

Questions to sharpen `reports/biomass_summary.md`. (The report currently covers ocean
phytoplankton/particulate carbon — Cphyto, POC, NPP, export — with the uncertainty in
the primary carbon measurements as its spine, and ends with a VICC-pitch mapping.)

**BQ1 — Scope: ocean-only, or broaden to "biomass on Earth"?** The `Biomass` folder is
entirely ocean phytoplankton, but Task 2's wording ("biomass on Earth") and VICC's
land+ocean span could argue for adding a terrestrial section (above-ground biomass /
forest carbon from EO — the CLARiTy lane). Keep it **ocean-only** (my lean, matches the
papers and our expertise), or add a short **terrestrial-biomass** contrast so the
report speaks to VICC's whole carbon budget?

BQ1:  Ocean only, but note its values relative to the terrestrial biomass.

**BQ2 — Framing: VICC-pitch or neutral review?** Should the report keep its closing
"mapping to the VICC pitch" (retrieve-or-bust reduces the inversion + conversion
uncertainties), or read as a **neutral standalone review** with the VICC framing pulled
into a separate EOI doc? (Lean: keep a short pitch mapping, but clearly fenced.)

BQ2: Map to the VICC pitch.

**BQ3 — Which carbon quantity to foreground, and where's the scope boundary?** Lead
with **Cphyto** (where optics adds the most and the bbp→C slope problem is sharpest),
**POC** (the mature satellite product), **NPP**, or **export**? And do we treat the
**subsurface / first-optical-depth** gap (85% of Cphyto below 1/Kd) as *in scope* (via
BGC-Argo/lidar synergy) or as an explicit **boundary** we don't claim to solve?
(Lean: foreground Cphyto + POC; subsurface as an honest boundary.)

BQ3: All 4 quantities and sub-surface is in scope.

**BQ4 — The headline uncertainty number.** For emphasis, should the report commit to a
specific "number to beat"? Candidates from the literature: satellite **Cphyto MAPE
~32%** (Stoer & Fennel), the **~3.5× global-stock spread** from conversion-slope
choice (218–771 Tg), or the **POC 47%→28%** composition gain (Koestner). Which anchors
the story — and do you have a defensible **target** (e.g., halve Cphyto MAPE)?

BQ4: No, the report should give an unbiased view of the current state of the effort and their uncertainties.  Do not blindly accept error estimates from the literature itself.

**BQ5 — Missing must-cite literature.** Beyond the 14 papers, are there works you'd
want folded in? Candidates I'd suggest: **Kostadinov** (size-class carbon), **Bellacicco**
(NAP / non-algal backscatter), **Stramski 2008/2022** (POC algorithms), **Martínez-Vicente
2013** (bbp→C), **DeVries / Siegel / Nowicki** (export), **Cael** (POC/statistics). Also:
can you supply/confirm the 2024–26 refs I pulled from web search (entries 7, 14, 19–24)
so I can verify authors/volumes before the Vancouver list?

BQ5: Use all of the ones you have recommended.

**BQ6 — Do you agree with the uncertainty ranking?** The report argues the two
dominant, *addressable* uncertainties are (i) the ill-posed **IOP inversion** (satellite
bbp/aph error) and (ii) the **non-universal optics→carbon conversion** (taxonomy/NAP),
with the in-situ **validation/CRM gap** and **atmospheric correction** as the next tier.
Do you rank these the same, or would you elevate one (e.g., the "no certified reference
material for POC" problem, or NAP separation) as the real bottleneck?

BQ6:  I leave it to you to rank as you best decide.

## Logs

### 2026-07-27 (Read the VICC EOI materials; posed Round-1 Q&A on proposal angles)

Executed the 1st EOI task. Read `VICC_Phase2_EOI.pdf` (extracted with pypdf — 6 pp),
fetched the VICC program page, and re-read `context_summary.md`.

**What I learned / captured.** VICC (Schmidt Sciences Climate Center) funds carbon-
cycle projects aiming to cut global-carbon-cycle uncertainty by an *order of
magnitude* for policy. Three themes (T1 emissions & fate / budget accuracy incl.
ocean interior carbon & sink trends; T2 policy-timescales / attribution / near-real-
time; T3 future decadal–centennial / surprises). Explicitly invites AI/ML for novel
inversion, data assimilation, emulation, up/down-scaling. EOI = 3 pp, **blind/
anonymized**, Vancouver refs separate, **due Aug 28 2026**; typical **$10M/5 yr**
(start Oct 2027, indirect ≤10%), with an explicit lane for smaller/shorter
*exploratory* awards; 5–7 funded. Existing ocean-carbon projects to differentiate
from: COCO2 (Southern-Ocean air–sea CO₂) and OBVI/InMOS. Advisory board incl.
Michalak, Holland, Bala, Friedlingstein.

**The core move for the EOI** is to reframe retrieve-or-bust (an ocean-color IOP
inversion + AI project) as a *carbon-uncertainty-reduction* project: IOPs are carbon
proxies (bbp→POC; aph/Cphyto→phytoplankton carbon & composition; together→NPP/export
= biological pump), and today's satellite carbon products inherit large, poorly-
quantified uncertainty from the ill-posed IOP inversion — exactly what the project's
AI + priors + honest-UQ engine (on PACE-class hyperspectral) is built to shrink and
characterize.

**Posed Round-1 Q&A** (VQ1–VQ6) with four candidate angles (A stocks / B biological
pump / C near-real-time attribution / D exploratory small award) and my leans: carbon
target (stocks vs pump vs full chain + whether an order-of-magnitude claim is
defensible), which theme leads (lean T1+T2), the scope/budget fork (flagship $10M vs
exploratory small award — the decision that sets everything else), data posture
(retrieval core + in-situ priors/validation, DA as stretch), differentiation/
collaboration vs COCO2 & InMOS, and team/PI gaps (add a BGC modeler / air–sea CO₂ or
BGC-Argo / international partner).

Next: await VQ1–VQ6, then converge on an angle and draft the EOI.

### 2026-07-30 (Read the Biomass literature; wrote reports/biomass_summary.md)

Executed the 2nd EOI task (independent of the VQ answers, which are still open). Read
all 14 PDFs in `context/papers/Biomass` (pypdf text extraction), scanned the 2024–2026
literature via web search, and wrote `reports/biomass_summary.md` with the emphasis on
**uncertainty in the primary carbon measurements**.

**Method.** Extracted text from all 14 papers; summarized in three parallel subagent
clusters — (i) phytoplankton-carbon/bbp (Behrenfeld 2016; Graff 2012/2015; Bisson
2023; Fox 2022); (ii) NPP (Behrenfeld 2005; Kulk 2020; Ryan-Keogh 2025; Taylor 2018);
(iii) POC/backscatter/review (Brewin 2023 review; Bisson 2020/21; Li 2024; Wu 2023;
Stoer & Fennel 2024) — each told to extract quantified uncertainty. Ran web searches
for the latest PACE-era carbon/NPP/POC uncertainty work.

**Key findings captured (the uncertainty spine).**
- **Cphyto←bbp conversion slope is non-universal** — the biggest lever: published
  slopes 8,372–30,100 (median 15,124), taxonomy-driven 3,770–27,697; propagates to a
  ~3.5× global-stock range (218–771 Tg) and MAPE ~32% (Stoer & Fennel).
- **POC←bbp** high uncertainty from particle-assemblage variability; adding Chl cuts
  it ~47%→28% (Koestner 2024); no certified reference material for POC; GF/F loses
  ~3–6× of cells.
- **Chl is a weak carbon proxy** — C:Chl spans 31–408; >55% of Chl anomalies are
  photoacclimation, not biomass.
- **NPP** global spread 32–79 Pg C yr⁻¹; ±1σ P–I → ±~45%; six algorithms disagree on
  the sign of the trend; CMIP6 ΔNPP −0.76 ± 3.44 (SD>4× mean).
- **Export/BCP** still 5–15 Gt C yr⁻¹ (unchanged since the 1980s); 6.1 vs 10.6 across
  methods.
- **Upstream**: satellite bbp MPE 18–45% (biased low) → ±50% basin-scale Cphyto;
  first-optical-depth blindness (85% Cphyto below 1/Kd).
- **Latest**: PACE OCI L2 BGC v3.1 (2025) now ships carbon_phyto *with* a per-pixel
  uncertainty product — the field is moving to carbon-centric, composition-aware,
  uncertainty-quantified retrievals, exactly the retrieve-or-bust levers.

The report ends with a headline uncertainty table and the mapping to the VICC pitch:
the two largest, most quantifiable uncertainties (ill-posed inversion + non-universal
conversion) are what an AI + priors + honest-UQ hyperspectral engine can reduce; the
subsurface term is an honest scope boundary.

Caveat logged in the report: a few 2024–26 web-scan references need author/volume
verification before formal (Vancouver) use.

Next: await VQ1–VQ6 to converge on the EOI angle.

### 2026-07-30 (Posed Round-2 Q&A — biomass-literature questions to improve the report)

Executed the 3rd EOI task: posed six questions (BQ1–BQ6, Q&A Round 2) whose answers
will sharpen `reports/biomass_summary.md`. They target the genuine open choices from
the review: **scope** (ocean-only vs broaden to terrestrial "biomass on Earth" per
Task 2's wording + VICC's land+ocean span); **framing** (keep the VICC-pitch mapping
vs neutral standalone review); **which carbon quantity to foreground** and whether the
subsurface/first-optical-depth gap is in-scope or an honest boundary; **the headline
uncertainty number** to anchor the story (Cphyto MAPE ~32% / ~3.5× stock spread / POC
47%→28%) and whether to commit to a target; **missing must-cite literature** (Kostadinov,
Bellacicco, Stramski, Martínez-Vicente, DeVries/Siegel/Nowicki, Cael) plus asking JXP
to confirm the web-scan refs I couldn't fully verify; and a **framing check** on the
ranking of dominant uncertainties (inversion + conversion first; validation/CRM and
atmospheric correction next tier). Gave my lean on each.

Next: await BQ1–BQ6, then revise `reports/biomass_summary.md` accordingly.

### 2026-07-31 (Revised biomass_summary.md per BQ answers; added 4 Python-generated figures)

Executed the 4th EOI task. Applied JXP's BQ answers and rewrote
`reports/biomass_summary.md`, adding figures from a new module `reports/py/biomass_figs.py`
(PNGs in `reports/figs/`).

**BQ answers applied.**
- **BQ1 (ocean-only, note vs terrestrial):** new §1 orientation — ocean phytoplankton
  ~0.78–1.0 Gt C vs terrestrial vegetation ~450–650 Gt C (~0.2%), yet ~half of global
  NPP; small/fast pool. Fig. 2.
- **BQ2 (map to VICC pitch):** kept, as a fenced §4 mapping.
- **BQ3 (all four quantities + subsurface in scope):** Cphyto/POC/NPP/export all
  covered; subsurface promoted from "boundary" to an **in-scope** section (§2.6) with
  the BGC-Argo/lidar fusion path. Fig. 4.
- **BQ4 (unbiased; don't trust literature error bars):** new §3 "A caution on the
  uncertainty numbers themselves" — reported errors are in-sample, conversion-only, and
  lack ground truth (no CRM for POC), so they are optimistic lower bounds; the honest
  measure is cross-method disagreement. Fig. 3 recast as "factor of disagreement
  (max÷min)" rather than accepting any single ±. No committed target number.
- **BQ5 (add all suggested refs):** folded in Kostadinov (size-class carbon),
  Bellacicco (NAP — its own §2.3), Stramski 2008 & 2022 (POC algorithms),
  Martínez-Vicente 2013 (bbp→C 30,100 slope), DeVries/Siegel/Nowicki (export), Cael
  (POC/export statistics), plus Bar-On 2018 & Field 1998 for the terrestrial context.
- **BQ6 (I rank):** §4 gives my ranking — conversion non-universality > NAP separation
  > ill-posed inversion > validation/CRM gap > subsurface > NPP-model/atm-correction.

**Figures (reports/py/biomass_figs.py → reports/figs/).** All values compiled from the
cited literature (not new analysis), and built to show *disagreement across estimates*
per BQ4: (1) bbp→Cphyto conversion-slope fan + resulting 218–771 Tg stock spread;
(2) ocean vs terrestrial biomass & NPP (log scale); (3) factor-of-disagreement cascade
(bbp 2.5×, Cphyto 3.5×, NPP 2.5×, export 3×) with a footnote flagging single-study bars
as lower bounds; (4) first-optical-depth profile with BGC-Argo/lidar as the path to
depth. All four verified visually.

Report notes provenance (which refs were read in full vs from the web scan) and flags
that several 2024–26 refs need author/volume confirmation before formal Vancouver use.

Next: await feedback; VQ1–VQ6 (EOI angle) still open for the EOI draft itself.

### 2026-07-31 (Addressed the 3 items in EOI prompt 5: export reliability + pie, DOIs, uncertainty conventions)

Executed the 5th EOI prompt — three additions to `reports/biomass_summary.md`.

1. **Reliability of the 5–15 Gt C yr⁻¹ export estimate + error-budget pie.** New §2.5.1:
   the 5–15 range is a spread of best estimates (not a CI) that has not narrowed in ~40
   yr; the two most recent globals (model-ensemble 6.08 ± 1.17; hydrographic
   10.64 ± 0.80) have **non-overlapping 95% intervals** → method disagreement, not
   sampling error, dominates. Added **Fig. 5** (`fig5_export_budget` in
   `reports/py/biomass_figs.py`): a donut attributing the export uncertainty to
   e-ratio/ef parameterization (30%), surface NPP (22%), transfer efficiency (18%),
   ecosystem/zooplankton pathways (15%), depth/definition (8%), sampling (7%) — labeled
   explicitly as an **illustrative literature-informed attribution, not a formal variance
   decomposition** (grounded in Henson 2022, Nowicki 2022, Siegel 2016, Doney 2024,
   Comm. Earth Environ. 2024).
2. **DOIs for all references.** Added DOIs to every entry; **verified** the trickier ones
   by web search (Martínez-Vicente 10.1002/grl.50252; Kostadinov 10.5194/os-12-561-2016;
   Koestner 2022 10.3389/fmars.2022.941950 + 2023 10.3389/fmars.2023.1197953 — fixed the
   earlier mislabelled "Koestner 2024"; Bellacicco 2019 10.1029/2019GL084078; Nowicki
   10.1029/2021GB007083; Ryan-Keogh 10.1038/s43247-025-02051-4). Added 3 new refs (30
   hydrographic Nature 2023; 31 Henson 2022 Nat Geosci; 32 Comm. Earth Environ. 2024).
   Flagged the few I could not fully verify (15 Cael, 24 Siegel-EXPORTS, 29 PACE dataset
   DOI, 30 first author, the RSE-2024 article number) with explicit "*(confirm)*" notes.
3. **Uncertainty conventions clarified.** Added a conventions box up top defining
   ranges (min–max spread, not a CI), `x ± s` (1σ ≈ 68%), MAPE/median-%-error, R²/RMSE;
   and annotated the key ± figures inline with **95% CIs** (export 6.08→3.8–8.4;
   10.64→9.1–12.2; CMIP6 ΔNPP −0.76 ± 3.44 → −7.5 to +6.0). Also corrected the §2.2 POC
   text (dropped the unverifiable "Stramski 2022" claim; cite Koestner 2022/2023).

Figures regenerated (fig5 added); report verified. Next: await feedback / the EOI draft
(VQ1–VQ6 still open).

### 2026-07-31 (EOI prompt 6: added §5 on non-optical approaches to ocean carbon)

Executed the 6th EOI prompt: the report is titled "from Optics," and JXP asked whether
there is another approach and, if so, to include its measurements and uncertainties.
**Answer: yes** — the in-situ / geochemical methods. Added a new **§5** (and a pointer
in the scope note) making this explicit.

§5 frames optics as one of several approaches — uniquely global/synoptic but *indirect*
— and surveys the non-optical alternatives as the *calibration backbone* (and the only
direct route to export flux, gross production, and air–sea CO₂ flux). A table gives, per
quantity, the method + what it measures + typical uncertainty (with type): POC —
filtration+CHN (no CRM; GF/F loss); Cphyto — sorting/microscopy (biovolume→C spans ~an
order of magnitude); NPP/GPP — ¹⁴C / ¹³C / ¹⁷Δ (methods disagree ~2×; ¹⁷Δ-GPP ≈ ±20%);
NCP — O₂/Ar & NO₃/DIC drawdown (k ±20–30%; advection up to ~38%); export — ²³⁴Th /
traps / UVP (C:²³⁴Th ~2–3×; trap efficiency ~0.5–2×); air–sea CO₂ flux — pCO₂/SOCAT +
inversions (model-vs-product discrepancy ~0.4–0.6 Gt C yr⁻¹). Bottom line: no gold
standard; each geochemical method carries its own factor-~2 uncertainty, reinforcing
§3's point that the "truth" validating optics is itself uncertain — hence the fusion /
cross-calibration path (in-situ + BGC-Argo as priors & validation), which is exactly
retrieve-or-bust's thesis and complementary to air–sea-flux programs (COCO2).

Grounded the numbers + refs via web search; added **6 new references (33–39)** with
verified DOIs (Claustre 2020 BGC-Argo; Marra 2009 ¹⁴C; Juranek & Quay 2013 triple-O;
Reuer 2007 O₂/Ar; Buesseler 2006 C:²³⁴Th; Le Moigne 2013 ²³⁴Th database; Friedlingstein
GCB 2023). Report verified; provenance note updated.

Next: await feedback / the EOI draft (VQ1–VQ6 still open).

### 2026-08-01 (EOI prompt 7: added §6 answering Matt Mazloff's physics questions)

Executed the 7th EOI prompt. Matt Mazloff asked three questions — (MQ1) does the report
assume the pure physical-component uncertainty is negligible? (MQ2) is the working
hypothesis that determining NCP is the key to reducing ocean carbon inventory
uncertainty? (MQ3) do we know how carbon-content variability from physical components
compares with that from biological ones? Added a new **§6** to
`reports/biomass_summary.md` answering each, plus **Fig. 6** and **19 new references
(40–58)**.

**The answers (all three are partly concessions — this is the honest reading).**

- **MQ1: yes, the report did implicitly assume it, and the assumption fails.** Every
  uncertainty in §§1–5 lives in the biological pathway. Evidence that the physical terms
  are not small: in Marinov et al. 2008, holding *biology fixed* and varying only the
  circulation moves the soft-tissue carbon store over **1,278–2,350 Pg C** (1,072 Pg C)
  and equilibrium pCO₂ over **321–423 ppm** — against a total anthropogenic inventory of
  118 ± 19 Pg C. Ödalen 2018 agrees from the other side (drawdown potential set by the
  circulation-determined P\*; removing climate feedbacks changes it 4–7 ppm). And the
  single largest cut in ocean-sink uncertainty to date came from a *physical* predictor:
  Terhaar 2021 constrained the Southern Ocean Cant sink with **sea-surface salinity**,
  reducing uncertainty **46–54%**. Matsumoto & Gruber 2005 also show the ΔC\*
  steady-state bias (+7% global, +14% Indian) enters via air–sea disequilibrium and
  preformed properties, *not* remineralisation.
- **MQ2: no — not as stated.** Broke it out by quantity in a table. For the contemporary
  anthropogenic inventory NCP is nearly irrelevant (Hauck 2015: climate-driven natural
  CO₂ change offsets the trend only ~10%, negligible climate effect on Cant uptake south
  of 30°S). Where NCP *is* the key: the **non-steady-state natural-carbon residual** that
  every Cant estimate discards — 13 ± 10 Pg C industrial era, 5 ± 3 Pg C (1994–2007),
  7.9 ± 3.8 → 0.9 ± 2.9 Pg C dec⁻¹ (Müller 2023, sign-flipping under an alternative flux
  product) ≈ **0.4–0.8 Pg C yr⁻¹, 10–20% of the decadal sink** — which is *formally
  undetectable* (signal +0.1–0.5 vs ±0.6–0.8 Pg C yr⁻¹) and cannot be verified on the
  timescale it is assumed over (soft-tissue-pump trend needs 23 yr global / 27–85 yr
  regional to emerge vs 10–20 yr for physical signals). Restated the hypothesis in that
  form — it is more defensible *and* makes us complementary to the physical-constraint
  programmes (Terhaar-style constraints, COCO2) rather than a weaker competitor.
- **MQ3: known in part, and it inverts by question.** By *stock* biology wins (soft-tissue
  reservoir 10–20× the anthropogenic inventory) — but that whole range is generated by
  changing circulation alone, so the attribution is not well posed. By *rate of change*
  physics wins (2.6 ± 0.3 Pg C yr⁻¹ vs a 0.4–0.8 residual with 1σ of similar size). By
  *variability* the two are large, opposed and largely cancelling (Prend 2022 ×2 on
  entrainment-driven Southern Ocean outgassing — one of them Mazloff's own paper; Hauck
  2015 on SAM fertilisation nearly balancing circulation-driven outgassing), so the net
  flux is a small difference of large terms. By *detectability* physics wins decisively.
  Flagged explicitly that **no published formal variance decomposition of ocean DIC into
  physical and biological components exists** — an honest "we don't know", per BQ4.

**Fig. 6** (`fig6_physical_vs_biological` in `reports/py/biomass_figs.py`), 4 panels, all
values read from the cited papers: (a) stock ladder colour-coded by what sets each term;
(b) Marinov 2008 Table 2 transcribed — OCSsoft vs equilibrium pCO₂ across 8 circulation
states with biology fixed; (c) time-of-emergence, physical/chemical (10–20 yr) vs
biological (23–32+ yr); (d) the discarded non-steady-state term vs its own ±0.6–0.8
Pg C yr⁻¹ detection threshold. Verified visually; all figures regenerated.

**§4 amended** in consequence: the §4 ranking holds *within the optical retrieval problem*
but is not a ranking of ocean-carbon uncertainty overall; the project should be pitched as
reducing the biological limb while explicitly inheriting and propagating the physical one,
and should target the non-steady-state residual as the term physical constraints cannot
reach.

**Method note.** Sourced from full-text extraction of the papers rather than search
snippets; **all 18 new DOIs verified against DOI/crossref metadata** (first author,
journal, year) before use. Ref 56 (Prend GBC 2022) confirmed from the article's own text.

Next: await feedback / the EOI draft (VQ1–VQ6 still open).