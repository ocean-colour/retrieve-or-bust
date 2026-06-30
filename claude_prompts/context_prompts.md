# Context

We will use this doc to prompt the generation of context for Claude.

## Goals

This repository will be the humans last attempt to solve the IOP inversion.  With AI.  If we fail, we will have to give up.  Or wait for better AI.  

## Prompts

1. Read this PDFs in the `context` folder. Generate a summary of the context in the file `context_summary.md` in that same folder.  Pose a set of questions for me and put them in the Q&A section below. Log your work.

2. See my answers to the questions in the Q&A section below.  Update the context_summary.md file with my answers. Log your work.

3. Scour the great internet for like information and add it to the context_summary.md file. Do your best to include DOIs. Log your work.

## Q&A

Questions from Claude after reading the context PDFs (answers welcome inline):

1. **What is the AI thesis?** The literature is unanimous that the inversion is
   degenerate and that the only cure is *external information*. What is our bet
   for where AI supplies that information — learned priors over joint IOP
   spectral shapes, a learned forward/inverse operator, amortized Bayesian
   inference, an LLM-in-the-loop over physical models, or something else?
   >A. We will use priors from in-situ observations, environmental variables, and time-series information.  And then hope for the best!

2. **Relationship to BING.** BING (Prochaska & Frouin 2025) is already your
   Gordon+MCMC Bayesian retrieval. Is retrieve-or-bust meant to *extend* BING
   (e.g. AI-learned priors feeding the same MCMC), to *replace* its inference
   engine, or to be an independent approach we benchmark against BING?
   >A. We will let Claude generate the best approach it can.  Bayesian, deep learning, whatever.

3. **Success criterion.** "Solve the inversion" needs an operational
   definition. Is success: retrieving >3 independent parameters from
   multispectral data? Accurate `a_ph(440)` to within some factor? Calibrated
   uncertainties? Beating GIOP/GSM/QAA on a specific benchmark? What number, on
   what dataset, would make us declare success vs. "bust"?
   >A.  Good question.  Let us define success as retrieving >4 independent parameters from hyperspectral data.  We will revisit this.

4. **Data.** Which datasets do we train/evaluate on — Loisel et al. 2023 (L23)
   synthetic, the Defoin‑Platel COASTLooc-style synthetic sets, in-situ
   (PANGAEA/Valente, GLORIA/Lehmann), PACE observations? Do we have these
   locally (e.g. under `$OS_COLOR`), and what serves as ground truth?
   >A. We will use a range of datasets.  See what I am building already in the IOPtics repository.

5. **Sensor target.** Are we targeting hyperspectral (PACE/OCI) primarily, or
   must the method also work for legacy multispectral (MODIS/SeaWiFS)? This
   strongly shapes how much the degeneracy can be reduced.
   >A. We will target hyperspectral data primarily.  But we will also need to work for legacy multispectral (MODIS/SeaWiFS).

6. **Components vs. totals.** Is the goal total `a`/`bb`, or full component
   separation (`a_ph`, `a_dg`, `bb_p`)? The papers show components are far
   harder and more ambiguous than totals.
   >A. We will target component separation.

7. **Acceptable priors.** How strong a prior is "fair game"? Region/season-
   specific models and tight `S_dg` priors help but risk baking in the answer.
   Where is the line between a legitimate informative prior and circularity?
   >A. This is TBD, but we should leverage priors to the greatest extent possible.

8. **The Mobley book.** It's a 924-page reference, not a result. Do you want it
   treated purely as a physics reference, or are there specific
   chapters/derivations you want extracted into the repo for the forward model?
   >A. Just remember it is here to refer to.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-06-30 (Read context PDFs, wrote context_summary.md, posed Q&A)

Read the five PDFs in `context/` and synthesized them into
`context/context_summary.md`; posed eight questions in the Q&A section above.

Method: fanned out five parallel sub-agents, one per PDF, each returning a
structured summary (citation, problem, method, findings, relevance to IOP
inversion, notation). The 924-page Mobley book was handled by reading only its
table of contents plus the IOP/AOP/forward-model sections rather than all pages.

The PDFs and what they are:
- `bing.pdf` = **Prochaska & Frouin 2025**, Biogeosciences 22:4705 — your own
  BING (Bayesian INferences with Gordon coefficients) paper. Gordon-series
  forward model + MCMC (emcee) + AIC/BIC model selection.
- `sydor2004.pdf` = Sydor et al. 2004, Appl. Opt. 43:2156 — empirical
  non-uniqueness of IOP retrieval.
- `Defoin‐Platel2007.pdf` = Defoin‑Platel & Chami 2007, JGR 112:C03004 —
  quantifies ambiguity (~90% of spectra ambiguous) with a formal metric set.
- `werdell2018.pdf` = Werdell et al. 2018, Prog. Oceanogr. 160:186 — the
  authoritative review/taxonomy of semi-analytical IOP algorithms.
- `mobley-oceanicopticsbook.pdf` = Mobley, *Ocean Optics Web Book* — foundational
  physics reference (IOPs/AOPs, Gordon forward relation).

Key thing learned: the dominant, unanimous message across the literature is that
the IOP inversion is **physically degenerate / ill-posed**, not just hard.
`Rrs` constrains the ratio `u = bb/(a+bb)`, so many `(a, bb)` pairs give the same
spectrum; multispectral data support only ~3 independent parameters and even
hyperspectral (PACE) does not break the degeneracy without external information.
That framed the questions: they probe what the project's *AI thesis* is for
supplying that external information, how it relates to BING, and how we will
define success vs. "bust".

### 2026-06-30 (Folded Q&A answers into context_summary.md)

JXP answered all eight questions inline in the Q&A section. Added a new
**"Project decisions (from Q&A, 2026-06-30)"** section to
`context/context_summary.md` capturing them as the working scope.

Decisions recorded: (1) AI thesis = break the degeneracy with priors from
in-situ observations, environmental variables, and time-series; (2) method is
open — Claude picks the best approach (Bayesian, deep learning, or hybrid), with
BING as baseline/benchmark rather than a constraint; (3) provisional success bar
= reliably retrieving >4 independent parameters from hyperspectral data
(ambitious vs. the literature's ~3, to be revisited); (4) primary target is
hyperspectral (PACE/OCI) but must also handle legacy multispectral
(MODIS/SeaWiFS); (5) goal is full component separation (`a_ph`, `a_dg`, `bb_p`),
not just totals; (6) datasets follow what's being built in the IOPtics repo;
(7) priors policy is "leverage as aggressively as possible," with the
legitimacy/circularity line left TBD; (8) Mobley book is a reference only.

Takeaway: the project is deliberately aiming at the *hardest* version of the
problem (component separation, >4 params) and is betting entirely on
information-rich priors to make it tractable — consistent with the literature's
finding that external information is the only cure for the degeneracy. Next
prompt (3) is to scour the web for related work and add it with DOIs.

### 2026-06-30 (Web search for related literature; added to context_summary.md)

Scoured the web for related work and added a **"Related work from the literature
(web search, 2026-06-30)"** section to `context/context_summary.md`, organized
into: foundational forward model & SAAs; ill-posedness/uncertainty; priors /
ancillary / Bayesian / water-type / time-series; machine/deep learning; and
benchmark datasets & mission. ~30 references with DOIs.

Method: fanned out four parallel web-search sub-agents by theme (ML/DL;
ill-posedness & uncertainty; priors & ancillary constraints; foundational SAAs &
datasets). Each was instructed to report only DOIs seen on authoritative sources
and to flag anything unconfirmed rather than invent DOIs. I then independently
spot-checked the uncertain ones by resolving them through doi.org:
- Verified ✓ GSM = Maritorena, Siegel & Peterson 2002, Appl. Opt. 41(15):2705
  (DOI 10.1364/AO.41.002705) — one agent had marked it unverified.
- A guessed DOI for Pahlevan et al. 2024 (RSE 308:114215) returned HTTP 404, so
  I dropped it and marked the entry "DOI unverified."
- Left Krasnopolsky 2013 and Moore 2009 DOIs flagged "unverified."
The summary carries an explicit verification note distinguishing confirmed DOIs
from search-reported ones, and flags two future-dated (2026) online-first papers.

Most useful finds for our thesis: **Bisson et al. 2023** (Opt. Express) shows
seeding GIOP with ancillary bb observations as priors cuts seasonal absorption
biases >50% — a near-direct proof of concept for the in-situ-prior strategy; and
**Erickson et al. 2023** (Opt. Express) is a Bayesian GIOP that trades
deviation-from-prior against observational error — the closest published cousin
to "BING + learned priors." Also confirmed the dataset DOIs (L23, GLORIA,
PANGAEA V3) and the PACE/OCI instrument reference for the hyperspectral target.

Caveat worth remembering: a couple of agents initially produced plausible-but-
wrong DOIs (the 404 above), which is exactly the hallucination risk these tasks
carry — the verification note in the summary should be honored before citing.
