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
*(note: ~290 words — within the <300 cap.)*

Our team pairs ocean-optics and radiative-transfer expertise with a substantial
track record applying machine learning to spectral inference.

**J. Xavier Prochaska (PI)** — Professor of Astronomy & Astrophysics, UC Santa
Cruz. Co-author of **BING** (Bayesian INferences with Gordon coefficients;
Prochaska & Frouin 2025), the open-source Bayesian framework that quantified the
IOP inversion's fundamental degeneracy. An active deep-learning practitioner since
2017 — the SPectral Image Typer (SPIT; Jankov & Prochaska 2018) and deep learning
to detect damped Lyα systems in quasar spectra (Parks, Prochaska et al. 2018),
direct precedents for the spectral inference at this project's core. An existing
Anthropic customer, already running an agentic Claude workflow to build the project.

**Robert Frouin (Co-I)** — Researcher, Scripps Institution of Oceanography, UC San
Diego; BING co-author and a leading authority on radiative transfer, satellite
ocean color, and atmospheric correction. Extensive experience with Bayesian,
neural-network, principal-component, and hybrid physics–machine-learning methods
for satellite retrieval.

**Heidi Dierssen (Co-I)** — University of Connecticut; NASA PACE science-team
leader and President-Elect of The Oceanography Society. Ocean-color remote sensing
of ecological and air–sea processes; develops algorithmic methods to map
phytoplankton community composition and benthic habitats.

**Henry Houskeeper (Co-I)** — Assistant Scientist, Woods Hole Oceanographic
Institution; co-developer of Kelpwatch.org. A decade of ocean-color algorithm
development, validation, and aquatic radiometry.

**Maria Kavanaugh (Co-I)** — Associate Professor, Oregon State University.
Ecological hyperspectral remote sensing; leads the NOAA-funded Plankton Pipelines
program operationalizing machine learning for plankton imaging, and the Northern
California Current MBON.

Additional collaborators across ocean optics and astrophysics — where ill-posed
inverse problems and Bayesian priors are native — are being recruited.

### Key team members using Claude Science (name, title, role) *(required)*
- **J. Xavier Prochaska** — PI, Professor (UCSC). Leads algorithm design; runs
  the Claude agentic experimentation loop; owns evaluation and the science.
- **Robert Frouin** — Co-I, Researcher (Scripps / UC San Diego). Leads the
  ocean-optics and radiative-transfer components: forward-model and prior
  specification, physically defensible constraints, and evaluation of retrieval
  identifiability and uncertainty, through to the PACE OCI demonstration.
- **Heidi Dierssen** — Co-I, University of Connecticut. Guides PACE-relevant
  validation and the ecological application of the retrievals.
- **Henry Houskeeper** — Co-I, Assistant Scientist (WHOI). In-situ data analysis
  and validation to inform ocean-color inversion priors.
- **Maria Kavanaugh** — Co-I, Associate Professor (Oregon State University).
  Explores information content and ecological representation in hyperspectral
  observations, integrating IOP retrievals with imaging.
- **Additional collaborators** (ocean optics + astrophysics) being recruited.

### Links to Google Scholar / professional profiles
- Prochaska: https://scixplorer.org/search?p=1&q=prochaska%2C+j&sort=score+desc&sort=date+desc&d=general ; https://profxj.github.io/
- Frouin: https://scholar.google.com/citations?user=ctIbRzEAAAAJ&hl=en ;
https://rfrouin.scrippsprofiles.ucsd.edu/; 
- Houskeeper: https://scholar.google.com/citations?user=gFGL2hwAAAAJ&hl ; https://www.whoi.edu/profile/henry.houskeeper/
- Kavanaugh: https://scholar.google.com/citations?user=4ZO4qAgAAAAJ&hl=en ; https://ceoas.oregonstate.edu/directory/maria-kavanaugh
- Dierssen: https://scholar.google.com/citations?user=-F_IYY8iRusC ; https://marinesciences.uconn.edu/person/heidi-dierssen/
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

### How specifically will Claude's capabilities be used? — **(1-2 sentences, 300 words max)** *(required)*
*(note: 2 sentences, ~205 words — satisfies the 1-2-sentence limit.)*

Claude is the engine that designs the solution — the novel algorithm we have not
been able to construct by hand, while the science (problem, physics, data, and
judgment of what counts as a real retrieval) stays firmly ours: within the
IOPtics/BING framework an Opus-class agent reasons over the radiative-transfer
physics, the degeneracy structure, and the available priors to propose candidate
retrieval architectures (Bayesian, deep-learning, or hybrid), writes and refactors
the retrieval code and evaluation harness, runs a tool-use loop that executes those
candidates over the L23 and in-situ corpora, reads the metrics (parameter count,
component error, uncertainty calibration), diagnoses failures, and proposes the
next iteration — with a separate LLM-as-judge pass scoring retrieved IOP spectra
for physical plausibility and pruning the search. Reasoning primarily from the
radiative-transfer equations, known optically-active-constituent relationships, and
observed spatiotemporal variability — using the literature only for validation and
dead-end avoidance, and explicitly instructed toward the *path less trodden* that
treats the ~100 published algorithms as a baseline to beat rather than a template
to imitate — this single agentic loop unifies physical reasoning, code, statistical
inference, and result-vetting that would otherwise require several specialists and
years, compressing into weeks a methods search that has stalled the field for
decades.

### How will Claude Science significantly accelerate or enhance your research vs. existing methods? — **(1-2 sentences, 200 words max)** *(required)*
*(note: 2 sentences, ~150 words — satisfies the 1-2-sentence limit.)*

The field has stalled not for lack of effort — there are ~100 published algorithms —
but because the space of possible retrievals (method family × prior structure ×
parameterization × validation design) is enormous and each hand-built candidate has
historically taken researchers months to build and test, which is why decades of
semi-analytical algorithms (GIOP, GSM, QAA) have plateaued at ~3 parameters. Claude
changes the economics: a single agent proposes a candidate method, implements it,
runs it across the Rrs corpora, reads the diagnostics, and refines in hours rather
than months and across far more hypotheses than any human team could pursue —
unifying physical reasoning, code, statistical inference, and result-vetting so we
can explore the informed-prior solution space broadly and honestly, converging on a
method that generalizes rather than one over-fit to a single sensor or region.

---

## Impact assessment

### Potential scientific impact if successful — **(1-2 sentences, 200 words max)** *(required)*
*(note: 2 sentences, ~140 words — satisfies the 1-2-sentence limit.)*

Phytoplankton set the base of the marine food web and drive the ocean's biological
carbon pump, yet our global knowledge of their abundance and composition rests
entirely on ocean-color IOP retrievals whose component estimates remain uncertain by
up to an order of magnitude — so a retrieval that reliably separates phytoplankton
absorption from dissolved/detrital material and particulate backscatter, with
trustworthy uncertainties, would sharpen every downstream product: phytoplankton
biomass and physiology, primary-production estimates, carbon-export models, and
water-quality and harmful-algal-bloom monitoring. Arriving at the dawn of NASA's
hyperspectral PACE mission it would let the community extract far more of PACE's
information content than current algorithms, and more broadly it would demonstrate
that AI can break a famously ill-posed geophysical inversion by systematically
marshalling external information — a reusable template for the many degenerate
inverse problems across the Earth and physical sciences.

### Applications beyond pure discovery / societal benefit / paths to scale — **(1-2 sentences, 200 words max)** *(required)*
*(note: 2 sentences, ~140 words — satisfies the 1-2-sentence limit.)*

A retrieval only matters if it runs at scale — PACE and its heritage sensors deliver
on the order of terabytes of ocean-color data per day — so our target is an
algorithm efficient and robust enough to be applied operationally to that full
stream, turning raw global color into calibrated maps of phytoplankton and IOPs and
underpinning concrete societal benefits: monitoring fisheries habitat and harmful
algal blooms, tracking coastal water quality, and constraining the ocean carbon sink
for climate assessment. All code is open-source within the ocean-colour ecosystem
(IOPtics, BING) and evaluation uses public truth sets (L23, PANGAEA, GLORIA) so
results are independently verifiable and the tools reach the whole community; and
because "break a degenerate inversion by injecting learned, physically grounded
priors" recurs across remote sensing, geophysics, and laboratory spectroscopy, a
success here is a blueprint well beyond ocean color.

---

## Resource requirements

### How much money in credits do you anticipate? + how it leads to impact *(required; max $30,000)*
*(note: **$25,000** requested — per PI. Justification below.)*

We request $25,000 in API credits over the 3-month project, allocated across the
Claude workloads described above (Opus-class for reasoning/design; Batch API and
model tiering where possible; prompt caching on the shared corpus context). The
dominant cost — roughly $16k — is the agentic experimentation loop (design,
implement, evaluate, refine): long tool-use traces over the L23 and in-situ corpora
across many refinement rounds at high reasoning effort, with shared context
prompt-cached, and it is the primary lever on quality because more rounds and deeper
reasoning directly improve the retrieval. About $4k covers LLM-as-judge vetting of
retrieved IOP spectra (high volume, but batchable at 50 percent off and tierable to
Sonnet/Haiku), about $3k covers literature and prior synthesis plus the
parameterization search, and about $2k is contingency. The credits directly fund the
iteration that turns established truth sets into a working, informed-prior retrieval:
every dollar buys experiment rounds and search depth against the L23 and in-situ
benchmarks, which is exactly the depth a genuinely new inversion method requires to
generalize across sensors and regions.

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

**Word + sentence limits — all within cap.** Four fields on the form carry a
**"1-2 sentences, N words max"** restriction (verified against `context/Application.pdf`);
those are now written as exactly 2 sentences. Team description / Key members /
Project description have a word cap only (no sentence limit).

| Field | Cap | Words | Sentences | Status |
|---|---|---|---|---|
| Team description | <300 words | ~271 | — | ✅ |
| Key team members | (none) | ~135 | — | ✅ |
| Project description | <500 words | ~465 | — | ✅ |
| How Claude is used | 1-2 sent, 300 words | 217 | 2 | ✅ |
| Claude vs. existing methods | 1-2 sent, 200 words | 155 | 2 | ✅ |
| Scientific impact | 1-2 sent, 200 words | 148 | 2 | ✅ |
| Applications beyond / scale | 1-2 sent, 200 words | 158 | 2 | ✅ |

**Form mechanics:**
- [x] Name / org / title / links / "where heard" — filled
- [x] Scientific fields: Earth + Environmental + Biology/Life + Physics
- [x] Compute: **No**
- [x] Biosecurity: **None of the above** (safeguards → blank)
- [ ] **Project title** — PI to pick from the options
- [ ] **Credits account email** — resolve eligibility (`jxp@ucsc.edu` Teams may be
      ineligible; use individual account or confirm with Anthropic)
- [ ] Terms of Service: check **I agree** at submission

**Outstanding PI input (blocks submission) — see `TODO.md` for the full list:**
- [ ] **Project title** — PI to pick (three options staged in the draft)
- [ ] **Credits-account eligibility** (`jxp@ucsc.edu` Teams account — PI handling)
- [x] Team description trimmed to <300 words (~271)
- [x] Team completed to 5 (Dierssen added; all titled Co-I)
- [x] Kavanaugh & Dierssen profile links added
- [x] ML expertise woven into the team bios
- [x] $25k confirmed

**Claims to spot-check before submitting:** degeneracy / `u = bb/(a+bb)` and the
~3-parameter multispectral limit (context_summary.md); L23 / PANGAEA / GLORIA
truth sets and IOPtics/BING framework (context_summary.md, IOPtics README);
Prochaska AI/ML precedents SPIT (Jankov & Prochaska 2018) and Parks, Prochaska et
al. 2018 (Holy-Grail application). PACE "terabytes/day" — order-of-magnitude
figure; soften to "vast data volumes" if a hard number is unwanted.
