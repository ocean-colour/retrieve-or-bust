# Claude Science Cohort — Application Draft

*retrieve-or-bust: AI-driven retrieval of phytoplankton and inherent optical
properties from hyperspectral ocean color*

> **How to use this file.** Headings below are the **actual Google-Form fields**
> (Claude Science Cohort variant), in order, each annotated with its word cap.
> Paste each field's prose into the form. **Before pasting, strip the italic
> `*(note …)*` scaffolding lines and any `[TBD — user]` flags** — those are working
> notes, not answers. A submission checklist is at the bottom.
>
> **Program facts (from the form):** projects run **3 months, Sep 1 – Dec 1**;
> Cohort credit ceiling is **$30,000**; Modal compute up to $2,000 is optional.
> Anthropic treats submissions as **non-confidential** — nothing proprietary here.

---

## Page 1 — Program

### Which program are you applying to? *(required)*
**Claude Science Cohort**

### Your email *(required)*
`xavier@ucolick.org` *(form pre-filled under this account)*

---

## Contact information

### Name of primary contact *(required)*
J. Xavier Prochaska

### Name of organization or research institution *(required)*
University of California, Santa Cruz

### Position or title at organization *(required)*
Professor of Astronomy and Astrophysics; Associate of Ocean Sciences

### Website of org / research group, Google Scholar, or GitHub *(required)*
- IOPtics (project framework): https://ioptics.readthedocs.io/en/develop/
- ocean-colour GitHub org (BING, IOPtics, retrieve-or-bust): https://github.com/ocean-colour
- J. X. Prochaska (SciXplorer): https://scixplorer.org/search?p=1&q=prochaska%2C+j&sort=score+desc&sort=date+desc&d=general
- https://profxj.github.io/

### Where did you hear about this program?
In the release notes of the Anthropic Claude Science program.

---

## Project information

### Project title *(required)*
**[TBD — user to pick]** *(rec, impact-forward):*
"Retrieve-or-Bust: AI-driven retrieval of phytoplankton and inherent optical
properties from hyperspectral ocean color."
*Alternatives:* (a) "Breaking the ocean-color inversion with AI: hyperspectral
retrieval of marine phytoplankton and IOPs"; (b) "Retrieve-or-Bust: informing the
ill-posed ocean-color inversion with AI-learned priors."

### Scientific field(s) — select all that apply *(required)*
- [x] **Earth Science**
- [x] **Environmental Science**
- [x] **Biology / Life Sciences**
- [x] **Physics**

---

## Research team

### Email of the claude.ai account to receive credits *(required)*
`jxp@ucsc.edu` 

### Team description — expertise in science **and** AI/ML — **<300 words** *(required)*
*(note: ~215 words; Frouin's title/AI-ML line still [TBD — user]. Keep total < 300.)*

Our team pairs ocean-optics and radiative-transfer expertise with a
substantial track record applying machine learning to spectral inference.

THIS WILL BE UPDATED ONCE THE TEAM IS COMPLETE.

**J. Xavier Prochaska (PI)** — Professor of Astronomy & Astrophysics, UC Santa
Cruz. Co-author of **BING** (Bayesian INferences with Gordon coefficients;
Prochaska & Frouin 2025), the open-source Bayesian framework for ocean-color IOP
retrieval that quantified the inversion's fundamental degeneracy. Decades of
expertise in spectroscopy and statistical inference, and an **active
deep-learning practitioner since 2017** — the SPectral Image Typer (SPIT; Jankov
& Prochaska 2018), which classifies spectrograph frames at 98.7% accuracy, and
deep learning to detect damped Lyα systems in quasar spectra (Parks, Prochaska et
al. 2018) — direct precedents for the spectral pattern-recognition and inference
at the core of this project. An existing Anthropic customer (Team account since
2025), already running an agentic Claude workflow to build this project.

**Robert Frouin (Co-I)** is a Researcher at Scripps Institution of Oceanography, UC San Diego, and a leading authority on radiative transfer, satellite ocean color, atmospheric correction, and inverse problems involving light scattering. A co-author of BING, he has extensive experience developing Bayesian, neural-network, principal-component, and hybrid physics–machine-learning methods for retrieving ocean and atmospheric properties from satellite observations.

**Henry Houskeeper (Co-I)** — Assistant Scientist I, Woods Hole Oceanographic Institution. Co-developer of Kelpwatch.org, a global historical map of Earth's kelp forest ecosystems. Expertise in bio-optics of aquatic environments and aquatic radiometry, with over a decade of experience in ocean color algorithm development and validation.

Additional co-investigators — spanning ocean optics, remote sensing, and
astrophysics (where ill-posed inverse problems and Bayesian priors are native) —
are being recruited to contribute ideas, critiques, datasets, and validation
perspectives.

### Key team members using Claude Science (name, title, role) *(required)*
- **J. Xavier Prochaska** — PI, Professor (UCSC). Leads algorithm design; runs
  the Claude agentic experimentation loop; owns evaluation and the science.

- **Robert Frouin** -- Co-I, Researcher, Scripps Institution of Oceanography, UC San Diego. Leads the ocean-optics and radiative-transfer components and ensures that candidate retrieval methods remain consistent with the underlying physics. He will guide specification of the forward model, IOP and optically active constituent parameterizations, environmental and observational priors, and physically defensible constraints. He will also direct evaluation against synthetic and in-situ datasets, assess retrieval identifiability and uncertainty calibration, and help distinguish genuine information gains from correlations learned from a particular training dataset. Drawing on his experience in Bayesian inversion, neural-network and PCA-based atmospheric correction, satellite algorithm development, and NASA ocean-color missions, he will guide the transition from methodological experiments to a robust demonstration using PACE OCI observations.
- **Heidi Dierssen** - Co-I, **[title - TBD]** University of Connecticut.  Dr. Dierssen focuses on the development and application of ocean color remote sensing to understand ecological processes, biogeochemical cycling, and air–sea interactions across the global ocean. She currently serves as President-Elect of The Oceanography Society and was the Science and Applications Team Leader for NASA’s Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) mission, a flagship hyperspectral satellite mission designed to advance observations of ocean ecosystems. She also contributes her expertise to numerous international steering committees and working groups focused on aquatic hyperspectral remote sensing.
- **Heidi Dierssen** - Co-I, focuses on the development and application of ocean color remote sensing to understand ecological processes, biogeochemical cycling, and air–sea interactions across the global ocean. She currently serves as President-Elect of The Oceanography Society and was the Science and Applications Team Leader for NASA’s Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) mission, a flagship hyperspectral satellite mission designed to advance observations of ocean ecosystems. She also contributes her expertise to numerous international steering committees and working groups focused on aquatic hyperspectral remote sensing.
- **Henry Houskeepera** — Co-I, Assistant Scientist I (WHOI). Ocean-color expertise, in-situ data analysis and validation to inform ocean color inversion priors.
- **Additional collaborators** (ocean optics + astrophysics) being recruited.

### Links to Google Scholar / professional profiles
- Prochaska: https://scixplorer.org/search?p=1&q=prochaska%2C+j&sort=score+desc&sort=date+desc&d=general ; https://profxj.github.io/
- Frouin: https://scholar.google.com/citations?user=ctIbRzEAAAAJ&hl=en ;
https://rfrouin.scrippsprofiles.ucsd.edu/; 
- Houskeeper: https://scholar.google.com/citations?user=gFGL2hwAAAAJ&hl ; https://www.whoi.edu/profile/henry.houskeeper/
---

## Research proposal

*(Note: all Claude Science projects run 3 months, Sep 1 – Dec 1.)*

### Project description — **<500 words** *(required)*
*(note: ~430 words; covers question · methodology · outcomes · timeline.)*

**Scientific question.** Ocean color from space is our only global window on
marine phytoplankton — the base of the ocean food web and a central lever of the
ocean carbon cycle. Turning that color into biology requires retrieving the water's
**inherent optical properties (IOPs)**: absorption and backscattering and, most
valuably, their **components** — phytoplankton absorption `a_ph`, dissolved-plus-
detrital absorption `a_dg`, and particulate backscatter `bb_p`. But the inversion
from remote-sensing reflectance `Rrs(λ)` to IOPs is **fundamentally degenerate**:
`Rrs` constrains essentially the ratio `u = bb/(a+bb)`, so many distinct optical
states produce nearly identical spectra. The literature is unanimous that
multispectral sensors support only ~3 independent parameters, and that even the
new hyperspectral era (NASA PACE/OCI) does **not** break the degeneracy without
external information. **Our question**: can AI construct a retrieval algorithm that reliably
recovers more than four independent IOP parameters — full component separation
with calibrated uncertainties — by systematically injecting the external
information the physics demands?

**Methodology.** The degeneracy's only cure is likely information beyond `Rrs` itself.
We supply exactly that — **priors from in-situ observations, environmental
variables, and spatiotemporal context (e.g., time-series history and spatial
covariance)** — and use Claude to explore the vast space
of candidate methods (Bayesian, deep-learning, or hybrid) that couple those priors
to the Gordon reflectance forward model. Our point of departure is **BING**
(Bayesian inference on the Gordon model); we do not assume the final method
resembles it. All development and evaluation run on the **IOPtics** framework
against established truth sets: the Loisel et al. (2023) synthetic hyperspectral
database (L23) and in-situ archives (PANGAEA/Valente 2022; GLORIA/Lehmann 2023),
with legacy multispectral (MODIS/SeaWiFS) as a secondary target. Because
different in-situ archives carry different scatter and can bias an inversion, we
treat truth-set selection as a first-class variable: we QC L23 for non-physical
optically-active-constituent (OAC) combinations and require that conclusions hold
*across* independent in-situ datasets rather than being tuned to any one.

**Expected outcomes & deliverables.** 
(1) An open-source, AI-derived IOP-retrieval algorithm integrating 
environmental/in-situ priors that maximally reduces the degeneracy; 
(2) the stretch target —
**>4 independent parameters from hyperspectral `Rrs`** to resolve phytoplankton functional types— with an honest and quantified accounting
of where the degeneracy still bites; 
(3) a secondary analysis of *which* added information most reduces the degeneracy —
extended spectral coverage (e.g. UV/SWIR) versus external priors — quantifying the
marginal independent parameter each would buy;
(4) an operational-scale demonstration on NASA's PACE granules; 
(5) a methods paper.

**Timeline (3 months, Sep 1 – Dec 1).** M1: stand up infrastructure and reproduce
baselines (e.g. BING, GIOP) and QC/inter-compare the truth sets (L23 vs. in-situ
archives) on simulated and in-situ data. M2: Claude-driven exploration
of candidate methods and prior-integration schemes. M3: evaluation, uncertainty
calibration, benchmarking, and an operational-scale demo on PACE data.

### How specifically will Claude's capabilities be used? — **300 words max** *(required)*
*(note: ~225 words.)*

Claude is the **engine that designs the solution** — the novel algorithm we have
not been able to construct by hand — while the science (problem, physics, data,
and judgment of what counts as a real retrieval) stays firmly ours.

1. **Method design.** Claude (Fable-class) reasons over the radiative-transfer
   physics, the degeneracy structure, and the available priors to propose
   candidate retrieval architectures — Bayesian, deep-learning, or hybrid — and
   the ways to couple environmental/in-situ priors to the forward model.
2. **Implementation.** Claude writes and refactors the retrieval code, priors, and
   evaluation harness within the IOPtics/BING framework.
3. **Agentic experimentation.** A Claude tool-use loop runs candidate methods over
   L23 and in-situ corpora, reads the metrics (parameter count, component error,
   uncertainty calibration), diagnoses failures, and proposes the next iteration —
   the core research loop, run at high reasoning effort.
4. **Evaluation & vetting (LLM-as-judge).** Claude scores retrieved IOP spectra for
   physical plausibility and flags degeneracy artifacts, pruning the search.
5. **Physics-first exploration.** Claude reasons primarily from the
   radiative-transfer equations, known OAC relationships, and observed
   spatiotemporal variability — using the literature for *validation and to avoid
   known dead-ends*, and explicitly instructed to seek approaches the field has not
   tried rather than reproduce published algorithms.

This is an unusually deep, genuine use of Claude: a set of agents reason about the
physics, writes the code, runs the experiment, judges the result, and iterates —
compressing into weeks a methods search that has stalled the field for decades. A
standing instruction in the agents' system prompt directs them toward the *path
less trodden* — treating the ~100 published algorithms as a baseline to beat, not a
template to imitate.

### How will Claude Science significantly accelerate or enhance your research vs. existing methods? — **200 words max** *(required)*
*(note: ~170 words.)*

The field has been stuck not for lack of effort (there are ~100 published algorithms) but because the space of possible
retrievals — method family × prior structure × parameterization × validation
design — is enormous, and each candidate has historically taken researchers
many months to build and test. That is precisely why decades of
hand-designed semi-analytical algorithms (GIOP, GSM, QAA) have plateaued at ~3
parameters. Claude changes the economics: a single agent can propose a candidate
method, implement it, run it across the Rrs corpora, read the
diagnostics, and refine — in hours rather than months, and across many more
hypotheses than any human team could pursue. It also unifies tasks that would
otherwise need separate specialists: physical reasoning, code, statistical
inference, and result vetting. This lets us explore the informed-prior solution
space broadly and honestly — including approaches a human might dismiss
prematurely — and to converge on a method that generalizes, rather than one
over-fit to a single sensor or region.

---

## Impact assessment

### Potential scientific impact if successful — **200 words max** *(required)*
*(note: ~155 words.)*

Phytoplankton set the base of the marine food web and drive the ocean's biological
carbon pump, yet our global knowledge of their abundance and composition rests
entirely on ocean-color IOP retrievals whose component estimates remain uncertain
by up to an order of magnitude. A retrieval that reliably separates phytoplankton
absorption from dissolved/detrital material and particulate backscatter — with
trustworthy uncertainties — would sharpen every downstream product: phytoplankton
biomass and physiology, primary-production estimates, carbon-export models, and
water-quality and harmful-algal-bloom monitoring. Arriving at the dawn of NASA's
hyperspectral PACE mission, it would let the community extract far more of PACE's
information content than current algorithms. More broadly, demonstrating that
AI can break a famously ill-posed geophysical inversion by systematically
marshalling external information establishes a reusable template for the many
degenerate inverse problems across the Earth and physical sciences.

### Applications beyond pure discovery / societal benefit / paths to scale — **200 words max** *(required)*
*(note: ~175 words.)*

**Operational by design.** A retrieval only matters if it runs at scale: PACE and
its heritage sensors deliver **terabytes of ocean-color data per day**. Our target
is an algorithm efficient and robust enough to be applied operationally to that
full stream — turning raw global color into calibrated maps of phytoplankton and
IOPs. That underpins concrete societal benefits: monitoring fisheries habitat and
harmful algal blooms, tracking coastal water quality, and constraining the ocean
carbon sink for climate assessment.

**Open and reusable.** All code is open-source within the ocean-colour ecosystem
(IOPtics, BING), and evaluation uses public truth sets (L23, PANGAEA, GLORIA), so
results are independently verifiable and the tools reach the whole community, not
one group.

**The method generalizes.** "Break a degenerate inversion by injecting learned,
physically grounded priors" is a template that recurs across remote sensing,
geophysics, and laboratory spectroscopy — so a success here is a blueprint well
beyond ocean color.

---

## Resource requirements

### How much money in credits do you anticipate? + how it leads to impact *(required; max $30,000)*
*(note: **$25,000** requested — per PI. Justification below.)*

We request **$25,000** in API credits over the 3-month project, mapped to the
Claude workloads above (Opus-class for reasoning/design; Batch API and model
tiering where possible; prompt caching on the shared corpus context):

| Workload | Est. credits | Notes |
|---|---|---|
| Agentic experimentation loop (design → implement → evaluate → refine) | ~$16k | Dominant cost. Long tool-use traces over the L23 + in-situ corpora across many refinement rounds at high reasoning effort; shared context is prompt-cached. The primary lever on quality — more rounds and deeper reasoning directly improve the retrieval. |
| LLM-as-judge vetting of retrieved IOP spectra | ~$4k | High volume but batchable (50% off) and tierable to Sonnet/Haiku. |
| Literature & prior synthesis, parameterization search | ~$3k | |
| Contingency | ~$2k | |

**How it leads to impact:** the credits directly fund the iteration that turns
established truth sets into a working, informed-prior retrieval — every dollar
buys experiment rounds and search depth against L23/in-situ benchmarks, which is
exactly the depth a genuinely new inversion method requires to generalize across
sensors and regions.

### Does your project require compute? *(Modal, up to $2,000)*
**No.**

---

## Biosecurity assessment

### Does your research involve pathogen/virology, drug-resistance, toxicology, or synthetic biology? *(required)*
- [x] **None of the above** (earth/ocean remote sensing)

### Biosecurity safeguards
N/A — no boxes checked; benign earth/ocean science with no plausible dual-use.

---

## Additional information

### Anything else for the review committee? *(optional)*
*(note: ~120 words.)*

**Why Claude specifically.** This is not a single ML classification: it chains
physical reasoning over radiative transfer, statistical inference, code, and
result-vetting into one iterative research loop. Claude is uniquely suited because
a single agent can reason about the physics, implement the method, run it, judge
the output, and refine — unifying work that would otherwise require several
specialists and years.

**Program fit.** The work spans Earth, Environmental, Biology/Life-Sciences, and
Physics, and the payoff — quantifying marine phytoplankton and the ocean carbon
cycle — is squarely high-impact.

**Track record with Anthropic.** The team is an existing Anthropic customer (Team
account since 2025) and is already using Claude to build this project — ready to
put credits to work on day one.

### Terms of Service agreement *(required)*
- [ ] I agree *(check at submission)*

---

## Submission checklist

**Step 0 — before pasting:** strip every italic `*(…)*` note and every
`[TBD — user]` flag from the field prose.

**Word limits — all within cap** (approx counts):

| Field | Cap | Count |
|---|---|---|
| Team description | <300 | ~215 (Frouin AI/ML line still to add — keep < 300) |
| Project description | <500 | ~465 |
| How Claude is used | 300 | ~275 |
| Claude vs. existing methods | 200 | ~170 |
| Scientific impact | 200 | ~155 |
| Applications beyond / scale | 200 | ~175 |

**Form mechanics:**
- [x] Name / org / title / links / "where heard" — filled
- [x] Scientific fields: Earth + Environmental + Biology/Life + Physics
- [x] Compute: **No**
- [x] Biosecurity: **None of the above** (safeguards → blank)
- [ ] **Project title** — PI to pick from the options
- [ ] **Credits account email** — resolve eligibility (`jxp@ucsc.edu` Teams may be
      ineligible; use individual account or confirm with Anthropic)
- [ ] Terms of Service: check **I agree** at submission

**Outstanding PI input (blocks submission):**
- [ ] **Project title** (Q17)
- [ ] **Robert Frouin's** exact title, profile link, and any AI/ML experience (Q20)
- [ ] **Credits-account eligibility** (Q18)
- [ ] Confirm **$25k** request stands (Q10)

**Claims to spot-check before submitting:** degeneracy / `u = bb/(a+bb)` and the
~3-parameter multispectral limit (context_summary.md); L23 / PANGAEA / GLORIA
truth sets and IOPtics/BING framework (context_summary.md, IOPtics README);
Prochaska AI/ML precedents SPIT (Jankov & Prochaska 2018) and Parks, Prochaska et
al. 2018 (Holy-Grail application). PACE "terabytes/day" — order-of-magnitude
figure; soften to "vast data volumes" if a hard number is unwanted.
