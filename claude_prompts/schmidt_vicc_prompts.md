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

## Prompts

1. Perform the 1st Task under "EOI"

## EOI

1. Read all of the files in the Context section above.  We are going to discuss possible angles for a research proposal for the VICC program.  Put your questions in the Q&A section.  Log your work.  Use Fable if you can.

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