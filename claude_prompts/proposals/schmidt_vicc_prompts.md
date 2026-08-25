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

## EOI Guidelines

EOI submissions should provide a maximum 3-page proposal with the following information:

1. A project narrative, including: 
   - The primary scientific theme of your proposed project
   - How your project is critical to addressing the goals and mission of the VICC program
   - The scope of the proposed research
2. A clear research plan, including: 
   - Key project goals, scientific hypotheses, and methodologies
   - A brief description of data and computational tools that will be used
   - If applicable, how the proposed research could be used to inform environmental decision-making, management, investment, or policy-relevant science. 
3. A description of preliminary team composition, and potential external institutional partnerships, including the proposed team members’ scientific ability, relevant expertise, and scientific project management experience. Kindly note:
    - The EOI proposal document should not include any PI, team, and partner names and affiliations as the EOIs will go through a blind evaluation process. Team names and affiliations will be collected in a separate form. Proposals will be returned without evaluation if any identifying information is included in the EOI proposal document. See the “Formatting Questions” section in the FAQ document for further guidance on anonymization. 
    - An individual may only serve as the PI on one EOI. They may serve as a co-investigator / collaborator on multiple EOIs. 
    - Multidisciplinary and international teams of researchers are strongly encouraged
    - Early-to-mid-career researchers are welcome to submit proposals as the lead PI.
4. A preliminary budget, which includes a brief description of project expenses such as salary, equipment, travel, indirect costs, etc. Detailed budgets are not required for the EOI stage, but will be requested from teams invited to submit full RFPs in Step 2. Note that Schmidt Sciences caps indirect costs at 10%.
5. A brief statement indicating if, and how, generative AI was used in developing the proposal. This is for informational purposes only. Any details provided will be excluded from the formal merit evaluation process.

## Prompts

1. Perform the 1st Task under "EOI"
2. Perform the 2nd Task under "EOI"
3. Perform the 3rd Task under "EOI"
4. Perform the 4th Task under "EOI"
5. Perform the 5th Task under "EOI"
6. Perform the 6th Task under "EOI"
7. Perform the 7th Task under "EOI"

8. Perform the 1st Task under "First draft"
9. Perform the 2nd Task under "First draft"
10. Perform the 3rd Task under "First draft"
11. Perform the 4th Task under "First draft"
12. Perform the 5th Task under "First draft"
13. Perform the 6th Task under "First draft"
14. Perform the 7th Task under "First draft"
15. Perform the 8th Task under "First draft"
16. Perform the 9th Task under "First draft"
17. Perform the 10th Task under "First draft"

18. Perform the 1st Task under "Export"
19. Perform the 2nd Task under "Export"
20. Perform the 3rd Task under "Export"

21. Perform the 1st Task under "GO-SHIP"
22. Perform the 2nd Task under "GO-SHIP"

23. Perform the 1st Task under "Bits and pieces"
24. Perform the 2nd Task under "Bits and pieces"

25. Perform the 1st Task under "EOI continues"
26. Perform the 2nd Task under "EOI continues"
27. Perform the 3rd Task under "EOI continues"

28. Perform the 1st Task under "Final draft"
29. Perform the 2nd Task under "Final draft"
30. Perform the 3rd Task under "Final draft"
31. Perform the 4th Task under "Final draft"

## Final draft

1. Our draft of the EOI is on the Google Drive under the `Oceanography/Proposals/2026/Schmidt Sciences` folder.  It is named "VICC EOI".  Please read it and re-read the EOI guidelines which I have pasted in the `EOI Guidelines` section below.  You will see that we are currently far over the page limit.  Please discuss with me ways to shorten it being mindful of the unfinished sections.  Pose your questions in the Q&A section below.  Use Fable if you can.  Log your work.

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Then, ask another round before doing any Editing.  Use Fable if you can.  Log your work.

3. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Then, proceed to edit the EOI doc.  Please target 3.5 to 4 pages total and I will cut it down from there.  Try to use the writing style from the current Sections I an II which JXP wrote.  Use Fable if you can.  Log your work.

4. Ok, thanks for that.  Now, reduce all of the section headings by not using them as is standard.  Also, make the sections mirror the EOI guidelines. This will bring you back to ~3 pages.  So, re-introduce additional text to increase back to ~3.5 to 4 pages.  Use Fable if you can.  Log your work.

### Q&A

#### Round 8 — the length problem, measured, and a shortening plan (2026-08-25)

**The arithmetic first.** The doc today (team-revised since my Aug-19 edit: §III rewritten
around ECCO-Darwin + regional ROMS with NBLING→D-LITE, ML emulators, and Ensemble 4D-Var;
Hervé's hand on the float subsection; the TOC scaffold removed) measures, by section:
§I **811** words, §II **580**, §III **884**, §IV **648** — **2,923 words for I–IV alone**,
≈ **6–6.5 pages** at 11 pt/1.15 spacing, before Figure 1 (~⅓ page) and before the three
unfinished sections. The guidelines' five required elements mean V–VII are not optional:
team (item 3), budget (item 4), AI statement (item 5), plus item 2's "scientific
hypotheses" and the policy/decision-making relevance clause, which the current text
carries only implicitly. Reserving ~0.9 page for V–VII and ~⅓ page for the figure leaves
**≈1.8–2.0 pages ≈ 950–1,000 words for I–IV: a cut to about one-third of what stands.**
Trimming sentences will not get there; sections must merge or go.

**Proposed word budget** (≈1,500 body words total; every element of guidelines items 1–5
gets a home):

| Section | now | target | how |
|---|---|---|---|
| I. Problem + objectives + success metrics | 811 | ~380 | merge the three empirical-limits paragraphs (BGC-Argo limits / OC limits / "Are we stuck?") into one; keep our-lane + measurable-success paragraphs nearly intact — they ARE items 1–2 |
| II. (dissolved) | 580 | 0 | the Ltoa→Rrs pedagogy is tutorial for this panel; its two load-bearing paragraphs (vertical-information + unified-framework) compress into §I's close and §IV's lead |
| III. Modeling | 884 | ~400 | one paragraph of rationale, one of D-LITE+RT-embedded assimilation, one of two-scales+emulators; the D-LITE justification currently appears twice |
| IV. Observations | 648 | ~350 | Rrs record + Cphyto retrieval to ~2 sentences each; float layer + fieldwork merged into one "anchored in situ" paragraph (which also answers Heidi's realism comment and Hervé's fill-gaps-only framing) |
| V. Team + partnerships | stub | ~180 | roles only, incl. Hervé's suggested "BGC-Argo multi-instrument data integration and bio-optics" expertise line |
| VI. Timeline + budget | stub | ~110 | prose summary from the Work Packages sheet; indirect ≤10% |
| VII. Portfolio + openness + AI (+1 policy sentence for item 2c) | stub | ~130 | ~4 sentences differentiation; FAIR; factual AI statement |

The questions — all are genuinely yours or the team's to answer:

**FD1 — Is §III now editable?** The standing order (CQ5) is "Do not touch Section III,"
but no 3-page version exists that honors it: §III must drop from 884 to ≈400 words. Do I
(a) produce the compressed §III myself for the modeling group to review, or (b) hand
Dustin/Matt/Stephanie the 400-word budget and they self-edit? My lean: (a) — faster, and
they review the diff rather than face a blank budget.

FD1: Yes, III is now editable.

**FD2 — Dissolve §II as a standalone section?** This is the single biggest lever
(−580 words). The panel does not need Rrs = Lw/Ed explained; the degeneracy/AC/vertical
arguments survive in compressed form inside §I and §IV. Confirm, or is §II's pedagogy a
deliberate choice you want kept at reduced length (~150 words)?

FD2: I confirm.  It was a deliberate choice to have the pedagogy before but no longer

**FD3 — May I compress Robert's §IV text to ~2 sentences per block?** His Rrs-record and
Cphyto-retrieval paragraphs are ~150 words each and were adopted essentially verbatim
(CQ1). Cutting them to ~40 words each keeps the claims (full constellation,
sensor-specific observation operators, PACE→heritage transfer, per-pixel uncertainty)
and drops the mechanics. Go ahead, or route the compressed versions past Robert first?

FD3: Yes, please compress.

**FD4 — The figure.** Keep Figure 1 at reduced (half-column) size (~⅙–¼ page), or drop
it? My lean: keep — it is the only quantitative evidence of the problem, and item 1 asks
why the project is critical. (Note your margin comment "this will change" on the image.)

FD4: Keep it.  

**FD5 — The success metrics are marked in flux.** Your margin note on "from ~32%" says
"This seems too low; stand by," and Stephanie adds we need "an uncertainty measure on our
uncertainty measure." The shortened §I hard-wires these numbers in its most prominent
paragraph. Do you want (a) the current trio kept as placeholders and revised when you
settle them, or (b) the numbers stated more softly ("halve per-pixel uncertainty; reduce
the conversion-driven spread to <2×; bring trend detection within the existing record")
so later revisions don't ripple? My lean: (b) for the EOI — the claims survive, the
specific baselines move to the full proposal.

FD5:  I want that number to be properly updated.  We worked extensively on this and it should
be in your original context and report.  Please look back and suggest a rewording.

**FD6 — Blind-review violation now in §III: "the similarly data constrained UCSC ROMS
implementation."** "UCSC" names an institution; the RFP returns such proposals without
evaluation. I will replace it with "an established data-constrained regional ROMS
implementation for the California Current" during the shortening (and fix the "Froudin
+2019" typo → Frouin+2019, which as a third-person citation is fine). Any objection —
and shall I do a full anonymization sweep at the same time (e.g. "our 11-component
NEMURO" phrasing is borderline self-identifying)?

FD6: Yes, please do a full anonymization sweep.  We should not be naming institutions in the EOI.

**FD7 — I draft V–VII within the budget above?** Team roles (no names) built from the
RQ7 role set + Hervé's comment; the budget paragraph summarized from the "Work Packages"
sheet as it stands (noting your rows still carry placeholder-level costs); portfolio
differentiation per DQ7; the AI statement per DQ10(i); one sentence on MRV/policy
relevance to satisfy item 2c. Anything you want claimed or avoided in those sections?

FD7: Yes, please draft those.

**A final observation, not a question.** Several open margin comments are content
disputes the shortening will forcibly resolve (Stephanie's "can we assimilate cphyto/POC
if we don't measure them," Matt's NCP-from-NO3/O2 point, Hervé's subsection title). I
will treat the shortened text as the answer where the cut removes the disputed sentence,
and leave the comment threads for the team where it does not — but I will not adjudicate
science disagreements silently.

#### Round 9 — after the FD answers; last round before the edit (2026-08-25)

**All seven answered; the plan is locked**: §III is editable and I will compress it
myself for the modeling group to review (FD1); §II dissolves (FD2 — noted that the
pedagogy was deliberate and is now retired); Robert's blocks compress (FD3); the figure
stays at reduced size (FD4, refreshed when your "this will change" lands); full
anonymization sweep — "UCSC ROMS" out, and I will also degrade "our … NEMURO" to "an
established regional biogeochemical model," keeping community tool names (ECCO, Darwin,
ROMS, MITgcm, Tapenade) since tools are not identities (FD6); V–VII drafted (FD7).

**FD5, reworked per your instruction: high level only, no per-pixel framing.** I went
back to the report as you asked. Its §3 is unambiguous about why "~32%" was the wrong
anchor: that number is the bbp→C **conversion scatter alone**, assuming sensor drift,
biofouling and calibration error are negligible, validated against a "truth" that has
no certified reference material — the report's own conclusion is that single-study
error bars are **optimistic lower bounds** and the honest measure is the **spread
across independent methods: 2.5–3.5×** (global Cphyto stock 218–771 Tg C from the
conversion choice alone). So the metrics should be stated as global quantities:

> **Success is measurable.** We will (i) determine the **global phytoplankton carbon
> stock** — today uncertain to a factor of ~3.5 from the optics-to-carbon conversion
> alone (218–771 Tg C) — **to better than a factor of 1.8, with uncertainties validated
> against independent methods** rather than in-sample statistics; (ii) deliver the
> corresponding **global POC and biological-carbon records** with the same standard —
> methods that should agree currently differ by 2.5–3.5×, and our assimilative estimates
> must close that gap by at least half; and (iii) **shorten the time to detect a
> climate-driven trend in ocean biomass from >30 years to ~20** — within the existing
> multi-mission record.

**FE1 — Confirm that wording** (or mark up what to change). The factor-2 promise
survives in all three, every number traces to the report, and nothing is per-pixel.
One alternative if (ii) feels soft: replace it with the NPP/export propagation ("and
propagate these constraints into primary production and export, where independent
estimates currently span 2.5–3×"), which claims the consequence rather than a second
stock.

FE1: Let's use ~3.5x and then better than 50% error

**FE2 — What total does §VI state?** The "Work Packages" sheet currently sums to
**≈$2.6M over 5 years** (Prochaska 3×50K, storage 125K, Carroll 5×250K, Frouin 5×220K)
— well under the ~$10M flagship scale from DQ4(a), and several WPs have no rows yet
(inversion beyond yr 3, BGC-Argo integration, fieldwork, project management, the annual
meeting). Options: (a) state "approximately $8–10M over 5 years" as the intended scale
with the current sheet as partial detail (my lean — the EOI needs only a preliminary
budget, and underclaiming scale reads as an exploratory project); (b) state the sheet's
$2.6M as-is; (c) you update the sheet first and I summarize whatever it says. Which?

FE2: Reaccess the work package sheet on the Google Drive.  The total is now nearly $9M

**FE3 — Process for the edit itself.** As before: backup copy first, then in-place
update (same doc ID). Note the in-place update **orphans all open margin comments
again** — including the unresolved science threads (Stephanie's assimilation question,
Matt's NCP point). They will survive on the new backup. If the team would rather
resolve those threads in-doc first, say the word and I hold; otherwise I proceed on
your go-ahead in the next prompt.

FE3: I have already made a backup.  You can edit the active version without making another backup.

## EOI continues

1. Robert and Herve have provided comments on the EOI.  These are recorded as separate docs in `Oceanography/python/retrieve-or-bust/proposals/Schmidt_Sciences/comments`.  Please read them and prepare to edit the EOI doc accordingly (do not do so yet).  The PDFs that Herve refers to are located in `Oceanography/python/retrieve-or-bust/proposals/Schmidt_Sciences/context`.  The EOI doc is named "VICC EOI" and is on my Google Drive under the `Oceanography/Proposals/2026/Schmidt Sciences` folder.  Use Fable if you can.  Log your work.  Ask me a series of questions in the Q&A section below before editing the EOI doc.

2. I have answered your questions in the Q&A section below.  Please review them and react accordingly.  Then, let's resolve the write issue for the Google doc.  Use Fable if you can.  Log your work.

3. Ok, proceed to edit the EOI doc.  Use Fable if you can.  Log your work.

### Q&A

#### Round 7 — Robert's and Hervé's comments (2026-08-19)

I read all of it: Robert's six numbered comments (2026-08-18), Hervé's cover note
(2026-08-16) plus the two PDFs it refers to — his **confidential** concept note
(FRM-grade hyperspectral BGC-Argo sub-fleet for distributed ocean-colour validation:
hyperspectral Ed/Lu + bbp + FChla floats under Argo governance; REFINE prototype
flown; CNES already funding payload calibration; explicitly *not* to be circulated
within the consortium) and his NASEM BGC-Argo talk (innovation-layer-over-core
framing; 44-float UVP pilot; float transmissometers as ~1000 m export-flux proxies;
prototype float-Rrs validation vs Sentinel-3, N=519 spectra; "targeted sub-fleets"
as the scaling model; OSSEs to design them) — and the current **VICC EOI** Google
doc, including the six open margin comments (Mazloff ×4, Carroll, Dutkiewicz). The
doc as it stands runs through §III and stops at an empty "Biogeochemical Model
Emulation and Optimization" heading, with placeholders ("Our team will xxxx",
{NEED ONE SENTENCE}) — consistent with Robert's "quite incomplete."

One overall observation before the questions: **four independent comments converge
on the same fix.** Robert #1 (state the objective; do not imply we close the carbon
budget), Mazloff ("clearly after the bio component … should be explained and
justified in the first couple of paragraphs so as not to enrage reviewers"; "the
interior is needed. This is triggering."), and Dutkiewicz ("is this true of DIC? I
think we know that better than factor 3") are all the same demand: an early,
explicit our-lane paragraph that (i) claims the biological carbon stocks/fluxes,
(ii) scopes the factor-3 indictment to *optically-constrained biological*
quantities (DIC is indeed known far better), and (iii) says plainly what we do not
claim. We already have this move approved — DQ9's two "scoping precision"
sentences — so I intend to write that paragraph first and hang Robert's objective
sentence off it. The questions below are where your call is genuinely needed.

**CQ1 — Adopt Robert's supplied text blocks nearly verbatim?** Comments #1–#5 are
not really comments — they are drafted paragraphs (objectives sentence; §II
rewrite adding gliders + CALIOP/CALIGOLA lidar; §II closing "unified framework"
paragraph; a targeted-field-program paragraph; a multi-mission phytoplankton-carbon
retrieval paragraph with sensor-specific observation operators). They are good, and
they slot into exactly the holes the draft has. My lean: adopt all five with light
editing for voice and blind-review hygiene, and credit nothing (it is a proposal,
not a paper). Confirm, or do you want any of the five reworked rather than adopted?

CQ1: Yes, I confirm

**CQ2 — Robert's #6 changes the reprocessing scope: SeaWiFS through MODIS, MERIS,
VIIRS, OLCI, SGLI, and PACE OCI.** SQ1 settled "MODIS + PACE, SeaWiFS as
consistency check" precisely to bound compute. Robert's climate-quality Rrs record
is the full constellation — scientifically stronger (more overlap periods for the
discontinuity corrections that our detectability argument needs), but it is a
different-sized machine. Do we (a) adopt the full multi-mission list as the stated
WP, with MODIS+PACE as the priority spine, or (b) hold the SQ1 scope in the EOI and
leave the constellation ambition to the full proposal? My lean: (a) — his text, his
WP, and the EOI need not carry compute detail.

CQ2: (a)

**CQ3 — Blind-review handling of Robert's role statement.** His #6 paragraph is
first-person and names him and SIO; the RFP returns proposals containing
identifying information without evaluation. I will put the statement (as supplied)
in the separate team/names form and write an anonymized version for the body ("a
radiometry and atmospheric-correction group will lead …"). Same treatment for the
body's existing "[Snyder; Prochaska]" and "[Prochaska & Frouin 2025]" citations —
they can stay as third-person references in the separate reference PDF, per RQ3(a).
Confirm this is still the standing rule for this draft, i.e. the body stays
name-free even though the current text carries names in citation brackets?

CQ3: I confirm.  We must maintain anonymity in the body of the EOI.

**CQ4 — How much of Hervé's concept note can the EOI use, and does hardware enter
scope?** Two coupled halves. (i) *Confidentiality*: the note is not to circulate
within the consortium, but Schmidt has already seen it, and Hervé explicitly
invites us to draw on it. I propose using its *ideas* (a distributed FRM-grade
float validation layer; satellite-overpass-timed surfacing; per-profile uncertainty
budgets) without reproducing its text or citing it, and without circulating the PDF.
OK? (ii) *Scope*: a hyperspectral-radiometer float sub-fleet is the missing
radiometric-validation layer our TQ1 "PACE as calibration anchor" story needs, and
it directly attacks the float–satellite bbp offset [Bisson+2020] that the draft
already names. But DQ4(a) fixed a ~$10M computational/synthesis scope, and floats
are hardware. Options: (a) name the sub-fleet as a *leveraged, co-funded* capability
(CNES/REFINE/BGC-Argo governance; we fund analysis + OSSE-based fleet design, not
the floats); (b) budget a small pilot (a few FRM floats) inside VICC; (c) keep it
out of the EOI. My lean: (a) — it also matches Robert's #4 "targeted field
observations … rather than a new global survey" paragraph and Hervé's own
"co-funding between BGC-Argo and space agencies" slide. Which?

CQ4: (a)

**CQ5 — Section III governance: D-LITE vs B-GOSE, and who owns the modeling
text.** The draft's centerpiece is D-LITE (a lightened, adjointable Darwin inside
ECCO) with hyperspectral radiances assimilated through Darwin's own RT. Dustin's
margin comment suggests it "could also build off of Matt's B-GOSE set-up," and
Matt's comments push for more interior carbon and more-than-optics (Argo
T/S/O₂/NO₃/pH → pCO₂/DIC inferences). Questions: (a) is D-LITE the agreed
centerpiece, with B-GOSE named as the complementary interior/physical estimate —
or is that still being negotiated among Dustin/Matt/Stephanie? (b) Do you want me
to fold Matt's more-than-optics point into §II (BGC-Argo constrains the carbon
system, not just optics) and add the interior explicitly to the assimilated-data
list (GLODAP/SOCAT are already named)? I need (a) answered to write §III's ending;
I will do (b) unless told otherwise.

CQ5: Do not touch Section III

**CQ6 — Do the three factor-2 headline numbers and the n\* detectability argument
survive the rewrite?** The current draft's quantitative promise is only the
factor-3 indictment (Figure 1). The settled RQ4 trio (per-pixel Cphyto 32%→16%;
conversion-driven stock spread 3.5×→<1.8×; trend-detection time 31.5→~20 yr) and
the TQ1 PACE-era→heritage transfer framing are absent — yet Robert's #5 and #6 are
precisely the machinery that delivers the second and third numbers, and VICC asks
for named uncertainty reductions. My lean: reinstate the trio in the objectives
section (Robert's #1 slot) as the measurable success criteria. Confirm, or has the
team deliberately moved away from promised numbers?

CQ6:  I confirm.

**CQ7 — Delivery mechanics for the edit (when you give the word).** My Drive
access here is read-only — I can read "VICC EOI" (and its comments) but cannot
write to it. When we proceed I will produce the full revised text as a document
(markdown in `proposals/Schmidt_Sciences/`, formatted for direct paste, edits
marked if you want a diff view) including the table of contents / complete outline
Robert asks for, with bracketed placeholders where content is still team-owned
(budget, §III ending, figures). Also confirm the working title stays "Unravel:
untangling ocean color with data-constrained models" — noting only that the
subtitle synonym-echoes the verb (the Bits-and-pieces Round argued against that
pattern) and that a decision here affects the outline header only, nothing else.

CQ7: If you tell me how to gain access, I will provide it.  An option is to use `rclone`.  

#### Round 7 resolution — answers reviewed; the write issue is solved (2026-08-19)

**Decisions now locked:** **CQ1** adopt Robert's five text blocks nearly verbatim (light
voice/blind-review editing, no attribution); **CQ2(a)** the stated WP is the full
constellation (SeaWiFS→MODIS/MERIS/VIIRS/OLCI/SGLI→PACE OCI) with MODIS+PACE as the
priority spine; **CQ3** strict anonymity in the body — Robert's first-person statement
goes to the team form, an anonymized version to the body, name-bearing citations to the
separate reference PDF; **CQ4(a)** Hervé's ideas enter as a *leveraged, co-funded* float
validation layer (ideas only — no text reproduced, no citation of the confidential note,
the PDF stays out of circulation; VICC funds analysis and OSSE-based fleet design, not
floats); **CQ6** the RQ4 factor-2 trio + n\* detectability return as the objectives'
measurable success criteria.

**CQ5 "Do not touch Section III" — interpreted strictly.** §III passes through the edit
byte-for-byte, including its unfinished ending and the D-LITE/B-GOSE question; Matt's and
Dustin's §III margin comments are left for the team. I will still fold Matt's
more-than-optics point into **§II** only to the extent Robert's adopted §II text already
carries it (floats measuring "downwelling irradiance and other biogeochemical
quantities"); nothing beyond that without word.

**CQ7 — resolved, and no action is needed from you.** Your machine already has an rclone
remote `GDrive:` with full `drive` scope, and it reaches the folder. Verified end-to-end
today, on scratch files only (created under `GDrive:tmp_claude_wtest/`, then purged; the
EOI doc was only ever *read*):
  1. plain-file upload + delete round-trip in the Schmidt Sciences folder — OK;
  2. HTML upload with `--drive-import-formats html --drive-export-formats html`
     converts to a **native Google Doc** — OK;
  3. re-upload under the same name **updates the same document in place** (file ID
     `10k-hof…` unchanged across v1→v2, content verified) — OK;
  4. the current "VICC EOI" exports cleanly to HTML (120 KB, content verified).

**Editing pipeline for the next prompt:** draft in markdown under version control →
render to HTML → `rclone copyto … --drive-import-formats html` to update "VICC EOI" in
place. Two caveats to decide at edit time, neither blocking: (i) an in-place update
replaces the full content, so the six open margin comments lose their text anchors
(they remain in the comments pane; our edit addresses them anyway — but if the team
wants them resolved by hand first, say so); (ii) before the first in-place write I will
make a timestamped backup copy of the doc in the same folder (Drive version history is
also preserved). Alternative on request: write to a new "VICC EOI v2" doc and leave the
original untouched.

## Bits and pieces

1. We need a name for this project.  A catchy acronym.  See if you can use UNRAVEL.  Go for it.  Use Fable if you can.  Log your work.

2. Now consider the case where we use "Unravel" as is, i.e. without being an acronym, but then generate a project name that is:  "Unravel -- what we are doing blah blah".  Can you recommend something for the "what we are doing" part?  Use Fable if you can.  Log your work.

**Answer (2026-08-15).** One structural observation first: because *unravel* is a verb,
the strongest subtitles are the ones that complete it — "Unravel — X" then reads as an
imperative mission statement, *"Unravel X."* That is a trick an acronym could never do,
and I would exploit it rather than bolt on a gerund ("disentangling…", "untangling…"),
which just restates the verb we already have.

**Recommendation:**

> **Unravel — the ocean's living carbon**

Four words; reads as a sentence ("Unravel the ocean's living carbon"); names the
headline deliverable (Cphyto + POC = the *living* carbon) and leaves method and
uncertainty to the prose, where they belong. It also pairs naturally with the standing
tagline: *Unravel — the ocean's living carbon. Unravelling the ocean's carbon
uncertainty.*

Runners-up, in order, each buying a different emphasis:
  (a) **Unravel — weighing the ocean's living carbon from space.** "Weighing" is the
      concrete verb for a biomass census and lands instantly with a non-optics panel;
      "from space" declares the satellite lane. My pick if you want method in the name.
  (b) **Unravel — the ocean's carbon, with uncertainty you can trust.** Leads with
      VICC's own obsession; "honest uncertainty" is already our house phrase, and this
      is its name-sized form.
  (c) **Unravel — from ocean color to carbon accounting.** The value-chain form:
      declares input and output, and "accounting" speaks budget language to the panel.

Avoid: subtitles that synonym-echo the verb ("Unravel — disentangling…") and anything
with "expert system"/method jargon — the SQ3/Round-6 decisions already retired those.

**Answer (2026-08-15).** UNRAVEL works, and not just as a backronym — the verb is the
proposal: *unravelling* is disentangling threads, and disentangling is what we do at
every level (the aph/adg/bbp retrieval degeneracy; biological from physical carbon;
trend from variability in the 24-year record). Recommended expansion:

> **UNRAVEL — UNified Retrieval and Assimilation for Verifiable Estimates of ocean
> Living carbon**

Each word is load-bearing against the EOI as drafted: **UNified Retrieval** = one
physically consistent inversion across MODIS + PACE (the 27→43 yr cross-mission-
discontinuity argument, WP on harmonisation/AC); **Assimilation** = the ECCO-Darwin
coupling (iterated priors, adjoint stretch goal); **Verifiable Estimates** = calibrated
per-pixel uncertainty and the three factor-2 targets (and it resonates with the
carbon-MRV audience); **ocean Living carbon** = the Cphyto + POC headline. Tagline for
slides: *"Unravelling the ocean's carbon uncertainty."*

Two alternates if the lead reads too infrastructure-flavored:
  (a) **UNcertainty Reduction And Verification of ocean Ecosystem carbon over the
      muLti-mission record** — EXPORTS-style scattered letters; leads with VICC's own
      keyword (uncertainty) and encodes the long-record/detectability story.
  (b) **UNified Retrievals And Variability Estimation of the ocean's living-carbon
      Ledger** — "carbon ledger" speaks the budget/accounting language of the panel.

Collision check (web, 2026-08-15): no existing ocean/climate/carbon *research program*
named UNRAVEL. The one neighbor is **"Unravel Carbon,"** a corporate carbon-accounting
SaaS company — different styling and sphere, but worth knowing it exists since our
topic is carbon; the acronym form "UNRAVEL" plus the expansion keeps us clearly apart.
(An unrelated UNRAVEL consortium exists in dementia research.)


## GO-SHIP

1. Here is the main web page for the GO-SHIP program: https://www.go-ship.org/.  Please explore it and answer the following questions:
   - What is the GO-SHIP program?
   - To what extent has it measured back-scattering properties of the ocean?
   - To what extent does it have radiometry?
   - How about Cp measurements?
Write a report named `context/go_ship_summary.md` that summarizes your findings.  Use Fable if you can.  Log your work.

2. Read more about Bio-GO-SHIP and add to the report.  Note that I have mvoed the report to `reports/go_ship_summary.md`.  Use Fable if you can.  Log your work.

## Export

### Tasks

1. Over the past 10 years, NASA funded a major program called "EXPORTS". This is its main website: https://oceanexports.org/.  Please do a deep dive into what it was funded to do and what it actually accomplished.  If you need me to download any papers, let me know.  I will ask you to generate a report eventually, but first ask me questions in the Q&A section below.  Use Fable if you can.  Log your work.

2. See my answers to the Q&A questions below.  Ask me additional questions if needed.  Use Fable if you can.  Log your work.

3. Ok, read my answers to the FQ questions below.  Then proceed to write the report.  Use Fable if you can.  Log your work.

### Q&A

#### Round 1 — EXPORTS deep dive (2026-08-08)

I did the deep dive: the 2015 Science Plan, Siegel et al. 2016 (the *Front. Mar. Sci.* science
plan — our report's ref [24], now fully verified), the Oct 2016 Implementation Plan, the two
operational overviews (Siegel et al. 2021 *Elementa* for Station P; Johnson et al. 2024 *Prog.
Oceanogr.* for the N Atlantic), the Siegel/DeVries/Cetinić/Bisson 2023 *Ann. Rev. Mar. Sci.*
synthesis, Nowicki et al. 2022, the SeaBASS archive record, and the program's publication list.
Digest first, then questions.

**What it was funded to do.** A NASA OBB field campaign (NSF co-funded; conceived 2013 from a
ROSES-2012 scoping study, renamed from COOPEX), whose stated goal was "a predictive
understanding of the export and fate of global ocean NPP," under the explicit hypothesis that
export and fate can be predicted from surface-ecosystem characteristics **assessable from
satellite ocean color** — with "the next generation of ocean carbon cycle… satellite algorithms
to be used on the upcoming PACE mission" as funded rationale in every planning document. Three
science questions: SQ1 export pathways, SQ2 fate/transfer efficiency below the euphotic zone,
SQ3 use the knowledge to reduce uncertainty in contemporary and future export estimates (the
remote-sensing promise). Design: quantify all three export pathways (gravitational, migrant,
mixing) *simultaneously*, at two deployments bracketing the export-regime space — 18
science-team projects (13 NASA + 5 NSF), >100 scientists, ~30 institutions. Cost: the SDT's
proposed Goal Plan was $71.5M/7 yr; the only as-executed figure I can find is "$40M"
(single-sourced to WHOI's OTZ site — no NASA-official number exists publicly).

**What it accomplished.** (i) Both campaigns executed as designed and hit the intended
contrast: Station P (Aug–Sep 2018), the low end-member — Ez-ratio 0.10 ± 0.06, 100-m trap flux
1.38 ± 0.77 mmol C m⁻² d⁻¹, a ×3 trap-vs-²³⁴Th discrepancy, an episodic salp bloom raising
export efficiency 1.5–2.6×; PAP (May 2021), the high end-member — declining diatom bloom
sampled inside a retentive eddy, POC flux 11→14 mmol C m⁻² d⁻¹ (~10× the Pacific), BCP
efficiency rising 10%→30%, four storms resolved by >40 autonomous assets. (ii) The
EXPORTS-era global synthesis: export ~10.2 Pg C yr⁻¹ partitioned **gravitational ~70% /
mixing ~20% / migrant ~10%**, ~1,300 Pg C sequestered (Nowicki 2022 — already our ref [23];
Siegel 2023). (iii) An open data legacy: 11,031 SeaBASS granules, >400 parameters, 43 PIs.
(iv) ~105 publications, ~6,900 citations. **What it did not deliver:** the SQ3 promise. There
is no mechanistic Rrs→export satellite algorithm; the satellite-facing product is a
satellite-*driven* diagnostic inverse model, the 4–10 Pg C yr⁻¹ methodological spread stands
in Siegel 2023's own Table 1, and the program's own optics papers found bbp instrument
calibrations mutually inconsistent at 95% confidence (Erickson 2022) — i.e., the campaign ran
into precisely the retrieval/uncertainty wall our proposal targets. Siegel 2023's closing
section calls for a *future* operational satellite+in-situ+model system — which is, almost
verbatim, our WP structure.

---

**EQ1 — Purpose and framing of the eventual report.** Is the EXPORTS report (a) a
VICC-proposal input like `biomass_summary.md` — i.e., a fenced mapping at the end arguing
that EXPORTS supplied the process understanding and the data, and that the un-delivered SQ3
(satellite prediction with defensible uncertainty) is exactly our lane — or (b) a neutral
standalone review? My lean: (a), same structure as the biomass report.

EQ1: (a)

**EQ2 — How candid on promise-vs-delivery?** The honest evaluative claim is that EXPORTS
delivered pathway partitioning and a superb archive but not the satellite predictive
capability its name and funding rationale promised. Stated baldly, that reads as criticism of
colleagues (Siegel, Cetinić, Behrenfeld are the names on everything, and plausibly reviewers
or future partners). My lean: make the point entirely through the program's own words —
SQ3 as written in 2016, Siegel 2023's own "future research directions" concession, Erickson
2022's calibration result — with zero editorializing, framed as "the field's own synthesis
identifies the remaining gap." Agree, or soften further / sharpen?

EQ2: Agreed, no need to Editorialize.  But if you identify flaws or shortcoming that they did not report on, do discuss them.

**EQ3 — Scope and depth.** ~105 papers exist; paper-by-paper coverage is neither possible nor
useful. Proposed shape: §1 what was funded and why (incl. the PACE rationale); §2 the two
campaigns and their headline numbers; §3 the global synthesis and whether the numbers moved;
§4 the remote-sensing deliverable, honestly assessed; §5 data legacy; §6 lessons/mapping for
the VICC pitch. Length like `biomass_summary.md` or shorter? And should it live at
`reports/exports_summary.md`?

EQ3:  That sounds good.  The length should be about the same as `biomass_summary.md`.  And, yes, have it live in the `reports/exports_summary.md` file.

**EQ4 — The cost figure.** "$40M" is single-sourced (WHOI OTZ website); the $71.5M was the
SDT's *proposed* Goal Plan, never an appropriation; NASA published no figure. Do you know the
as-executed number, or should the report say "order $40M (reported), against a proposed
$71.5M Goal Plan" with the sourcing caveat? A wrong dollar figure in front of this audience
would be embarrassing, so I will not state one without a source.

EQ4: I do not have any numbers.  Report as you can.

**EQ5 — Papers to download.** Most key papers are open access or have open preprints I can
use (Siegel 2016/2021/2023, Nowicki 2022, Estapa 2021, Buesseler 2020, Steinberg 2023,
Stephens 2025; Johnson 2024 and Siegel 2025 via EarthArXiv; Kramer 2022 via Ifremer;
Cetinić 2024 via MTU). Four have no open copy I could find — please download if you want them
used at full depth, otherwise I will work from abstracts/citing papers:
  1. Clevenger et al. 2024, *Mar. Chem.* 258:104346 (the PAP ²³⁴Th/POC flux time series) — doi:10.1016/j.marchem.2023.104346
  2. Briggs et al. 2020, *Science* 367:791 (fragmentation = 49 ± 22% of mesopelagic flux loss; BGC-Argo, not EXPORTS data, but load-bearing context) — doi:10.1126/science.aay1790
  3. Traylor et al. 2025, *L&O* (autonomous observations across both export regimes) — doi:10.1002/lno.70002
  4. Johnson et al. 2024 *final* version, *Prog. Oceanogr.* 220:103170, if you want me citing the journal version rather than the preprint — doi:10.1016/j.pocean.2023.103170
Also: are the EarthArXiv/NTRS preprints acceptable as sources where finals are paywalled?

EQ5: I have put the first 3 papers in `Oceanography/python/retrieve-or-bust/context/papers/EXPORTS`.  You can use the preprint for the 4th.  That is acceptable as a source where finals are paywalled.

**EQ6 — Feedback into the existing documents.** Two concrete hooks. (i) `biomass_summary.md`
§2.5 carries "5–15 Gt C yr⁻¹, unchanged since the 1980s"; the EXPORTS-era synthesis gives a
defensible central ~10 Pg C yr⁻¹ *with pathway-resolved partitioning* (though the cross-method
spread genuinely stands). Update §2.5 to reflect this once the EXPORTS report exists? (ii) The
EOI names COCO2/InMOS/SUBSEA/CLARiTy for differentiation; EXPORTS is NASA, not Schmidt, but a
carbon panel will know it — should one sentence position us relative to EXPORTS ("EXPORTS
supplied the process understanding and field data; we build the satellite retrieval and UQ
system it called for"), or is that better left for the full proposal stage?

EQ6:  Just perform (i).  I will work on the EOI draft later.

#### Round 2 — after the EQ answers (2026-08-08)

All six answered; nearly everything is settled: **EQ1(a)** VICC-input framing with a fenced
mapping like the biomass report; **EQ2** critique through the program's own words, *plus* a
mandate to discuss flaws they did not report themselves; **EQ3** the six-section shape at
`reports/exports_summary.md`, biomass-report length; **EQ4** cost stated as "order $40M
(reported), against a proposed $71.5M Goal Plan" with the sourcing caveat; **EQ5** the three
PDFs are in `context/papers/EXPORTS` — verified present and text-extractable (and both
inferred DOIs, Clevenger `10.1016/j.marchem.2023.104346` and Traylor `10.1002/lno.70002`, are
now confirmed from the PDFs themselves) — with the Johnson 2024 EarthArXiv preprint as the
fourth source; **EQ6** only the `biomass_summary.md` §2.5 update.

**Decisions I am taking without asking** (say the word if any is wrong): the report inherits
the biomass report's conventions — Vancouver references with verified DOIs, the uncertainty-
conventions box, 95% CIs where derivable; the §2.5 update to `biomass_summary.md` happens in
the **same pass** as the report (it is a small surgical edit and the numbers come from the
same sources); preprints are cited alongside the journal DOI, per your EQ5 answer.

Two questions remain — both are calibration on the EQ2 mandate and the figures, not new
scope.

**FQ1 — Does the "unreported flaws" mandate extend to program-level design choices, or stay
scientific/technical?** The measurement-level shortcomings (the ×3 trap-vs-²³⁴Th gap, the bbp
calibration inconsistency, the broken steady-state assumption in the mesopelagic budget, the
salp-bloom stochasticity) were all *reported by EXPORTS itself* — candour is a strength of the
program. The flaws I found that they have **not** discussed in print are programmatic:
  (a) **The descope was never reconciled with the design logic.** The 2016 Goal Plan called
      for 4 deployments (two visits per basin, 388 sea days) precisely because the satellite-
      algorithm goal requires spanning the ecosystem-state space; the executed program was one
      snapshot per basin (one of them in the lowest-signal season at Station P), and no
      published assessment asks what the two-point sampling means for the SQ3/global-
      extrapolation promise.
  (b) **No Southern Ocean deployment**, despite it being a dominant export region and the
      original scoping trio including an ice/Southern-Ocean campaign concept (ICESOCC) that
      was never executed — so the global partitioning rests on extrapolation into the regime
      that matters most.
  (c) **The PACE timing mismatch is structural.** The field phase ended May 2021; PACE
      launched Feb 2024. Zero coincident field+PACE data exist, so the founding promise —
      algorithms *developed and validated* for PACE within the program — could not have been
      closed end-to-end regardless of scientific execution. (This also cuts favourably for
      us: the EXPORTS archive is exactly the pre-launch algorithm-development dataset our
      retrieval system needs, and we would be the ones closing the loop with actual PACE
      overpasses.)
  My lean: include all three, in a clearly-labelled program-level subsection of §4, stated
  factually with the descope numbers from the Implementation Plan — no judgement language.
  Confirm, or restrict to scientific/measurement critique only?

FQ1: Yes, please include all three.

**FQ2 — Figures.** The biomass report carries 6 Python-generated figures
(`reports/py/biomass_figs.py`); I propose the same treatment here, with a new module
`reports/py/exports_figs.py` and 2–3 figures: (i) the two-campaign contrast — POC flux,
Ez-ratio, and transfer efficiency at Station P vs PAP on a common axis, showing the designed
end-member bracketing was achieved; (ii) the global pathway partitioning (gravitational
~70% / mixing ~20% / migrant ~10% of ~10.2 Pg C yr⁻¹, with IQRs, and the 4–10 Pg C yr⁻¹
cross-method spread beside it — accomplishment and open problem in one image); (iii)
optionally, a promise-vs-delivery timeline (2013 scoping → 2015/16 plans → 2018/2021
deployments → 2024 PACE launch) that makes the FQ1(c) point visually. Include figures, and
if so, all three or drop (iii)?

FQ2: Yes, include all three.

## First draft

1. We will write a first draft of the EOI proposal.  Draw on these files for context:

    - Simons
    - The code in this repository
    - The `VICC_Phase2_EOI.pdf` file
    - The `VICC_Concept_v1.pdf` file in `proposals/Schmidt_Sciences`
    - The report in `reports/biomass_summary.md`
    - The "An expert ocean color system" section of the `Prochaska-Schmidt.pdf` file in `proposals/Schmidt_Sciences`
    - The relevant sections of the file `Simons_grant_proposal.pdf` in `proposals/Schmidt_Sciences`

Be sure to be wary of the EOI rules.  I will ask you to generate it as a Google doc named `EOI-proposal` on my Drive in the `Oceanography/Proposals/2026/Schmidt Sciences` folder.  But, first ask me questions in the Q&A section below.  Use Fable if you can.  Log your work.  Do not write the draft yet.

2. I have answered your BQ questions.  Please read my responses and ask additional questions if needed.  Use Fable if you can.  Log your work.  Do not write the draft yet.

3. I have answered your RQ questions.  Please read my responses and ask additional questions if needed.  Use Fable if you can.  Log your work.  Do not write the draft yet.

4. I have answered your RQ questions.  Please read my responses and ask additional questions if needed.  Use Fable if you can.  Log your work.  Do not write the draft yet.

5. I have answered your RQ questions.  Please read my responses and proceed to write the draft.  Use Fable if you can.  Log your work.  

6. The figure is an good first draft.  I fear, however, it will be too hard to fit the 2 panels onto one page.  So, let's make it 2 figures: (i) the problems nd (ii) our proposed "measurements". Also:
   - Be careful to keep text within the axes 
   - Avoid overlapping text
   - Generate PNGs instead of PDF

Use Fable if you can.  Log your work. 

7. The new figures are good.  Can you make another pair from the same code that are better size for Google Slides?  Add "slides" to their names.  Use Fable if you can.  Log your work.  

8. Better still.  Please make these changes:

   - Make the text "We do not know.." bold
   - Increase the font size for the axis label "factor by which.."

Use Fable if you can.  Log your work. 

9. Ok, now another figure.  Can you generate one that describes the methodology?  Something that describes the flow from Satellite observations (used to measure bbp) to carbon stock, etc.  Use Fable if you can.  Log your work.  Have this be a new figure and I only need one for Google Slides.

10. That figure is nice.  Now, make a new version with graphics replacing the main boxes (e.g. a satellite image, a diagram of the ocean, etc.).  If you want me to download any images, tell me.  Use Fable if you can.  Log your work.   Keep the original; add the word "graphics" to the name of this new version.  Perform the entire job without stopping me to ask any questions.

### Q&A

#### Round 3 — EOI first draft (2026-08-05)

I read all the context: `VICC_Phase2_EOI.pdf` (rules, re-read closely), `VICC_Concept_v1.pdf`
(+ the linked Google doc, which is the same text as `Prochaska-Schmidt.pdf`), the "An expert
ocean color system" section, the relevant Simons sections (III ocean-optics innovations, V
applied AI/ECCO, plus CMB and BOONUS for context), `reports/biomass_summary.md`, and the
**code** — which lives on the `rt-elastic-prototype` branch, not `main`.

**What the code actually gives us (the strongest card in the deck).** A *differentiable*
elastic RT forward model in JAX, `Rrs = Rrs_ZTT(θ) + ΔRrs_emulator(θ)`: the Twardowski &
Tonizzo analytic backbone with explicit phase-function dependence, plus a 417-parameter Flax
residual MLP. rRMS in `rrs` space on the L23 batch, held-out scenes: **Gordon 7.21% → ZTT
5.93% → hybrid 0.30%** (~24× over Gordon; the honest yardstick is the 8-parameter linear
hybrid at 2.54%, so the nonlinearity earns ~8×). Gradients verified against finite
differences; 225 tests passing; CI on 3.12/3.14. Caveats I would not hide: it is the
**forward** model (the inversion does not exist yet), M2 still has two `NotImplementedError`s,
M4 validation is not started, and **extrapolation to an unseen 60° solar zenith is not
reproducible** (seed spread 4.7–12.2%, median 7.75% vs ZTT's 8.09%) — decided as report-and-
defer, with a `DomainWarning` implemented.

**What the concept doc settles that Round 1 left open.** `VICC_Concept_v1.pdf` names
**ECCO-Darwin informed by ROMS tuned to observations** as the modeling core, with PACE/MODIS,
BGC-Argo, gliders, ships (AMT), lab work (a, bb, Kd, Cp→bbp), improved atmospheric
correction and inversion modeling. That is a *much* more ambitious data posture than my
Round-1 lean (VQ4b) — it is effectively VQ4(c), full coupling to an assimilative BGC model.
It also happens to answer Mazloff's critique in §6, which is why DQ1 below matters most.

**Still open from Round 1: VQ1–VQ6 are all unanswered.** I have taken a position on each
below rather than just re-asking, so you can confirm or overrule.

---

**DQ1 — The framing fork, and the most consequential question here.** §6 of the report
concluded that for the *contemporary anthropogenic inventory* the physical terms dominate and
NCP is nearly irrelevant; the defensible biological target is the **non-steady-state
natural-carbon residual** (~0.4–0.8 Pg C yr⁻¹, 10–20% of the decadal sink, formally
undetectable today). Meanwhile the concept doc leads with an ocean-colour **expert system**.
Three ways to open the EOI:

  (a) **Biomass/biology first** — "the first global ocean biomass census with fully
      characterized uncertainties." Truest to the concept and our expertise; but a
      carbon-cycle panel may ask why biomass stocks matter for the carbon *budget*, and §6
      says the honest answer is "not much, directly."
  (b) **The residual first** — "close the one carbon term physical constraints cannot
      reach." Strongest intellectual position post-§6, hardest to deliver in 5 yr, and it
      makes us sound like a physics project rather than an optics project.
  (c) **Coupled optics + ECCO-Darwin** — optics constrains the biological limb, the
      assimilative model carries the physical limb, and the deliverable is a *jointly*
      constrained, uncertainty-quantified ocean carbon state. This is the only option that
      answers Mazloff head-on, and ECCO-Darwin makes it newly available.

  **My lean: (c), opening on carbon and uncertainty, with the expert system as the *method*
  rather than the headline.** Reviewers here are carbon people (advisory board incl.
  Michalak, Friedlingstein); a tool-first pitch risks reading as infrastructure. Which?

DQ1:  Yes, definitely (c) and open on carbon and uncertainty, with the expert system as the *method* rather than the headline.  On biomass, would assessing the relative evolution over the past 20 years with satellite data be a worthwhile target too?

**DQ2 — What number do we promise?** VICC asks for an **order-of-magnitude** uncertainty
reduction. By the report's own standard (§3: no CRM for POC, cross-method disagreement
2.5–3.5×, literature error bars are optimistic lower bounds) promising 10× on any carbon
quantity would be indefensible — and we would be promising it to the people best equipped to
notice. COCO2 instead promised something specific and modest: **cut the observation-based
GCB ocean-sink uncertainty by 25%**. Options: (a) name one quantity and one defensible
number (my lean — e.g. a target reduction in satellite Cphyto/POC uncertainty, with the
*categorical* claim being the first end-to-end propagated uncertainty budget rather than a
10× headline); (b) claim order-of-magnitude on a deliberately narrow sub-quantity where it
is achievable; (c) reframe "categorical" as *characterizing* uncertainty honestly for the
first time, and argue that is the precondition for reducing it. Which — and do you have a
number you are willing to defend?

DQ2: (a) for sure.  Ideally we target 2 or 3 numbers to substantially improve.  I'd like to reduce by a factor of 2 the uncertainty.

**DQ3 — VQ1 redux: which carbon quantity is the headline deliverable?** BQ3 put all four
(Cphyto, POC, NPP, export) plus the subsurface in scope for the *report*. Three pages cannot
carry all five for a *proposal*. My lean: headline **Cphyto + POC** (where optics adds most
and the conversion problem is sharpest), with NPP/export as propagated consequences and the
subsurface via BGC-Argo/lidar fusion as the depth story. Confirm or reorder?

DQ3: Yes, headline Cphyto + POC, with NPP/export as propagated consequences and the subsurface via BGC-Argo/lidar fusion as the depth story.

**DQ4 — VQ3 redux: scope and budget.** The concept as written (expert system + ECCO-Darwin +
ROMS + international collaboration + lab/field optics) is unambiguously a **flagship
$10M/5-yr** program, not the exploratory lane. Note the RFP allows *exceeding* $10M if it
funds new instrumentation or significant novel data acquisition, justified. Do we (a) target
~$10M and keep it computational/synthesis (my lean), (b) exceed $10M by folding in the
Simons-style UV field/lab campaign (Hyper-a, C-HyR) and BGC-Argo/lidar deployments, or
(c) go tightly-scoped exploratory? Also: 5-year performance Oct 2027–Sep 2032, indirect ≤10%,
and budget for a 5-day annual science-team meeting (first Oct 2027, Europe) — I will include
those.

DQ4: (a)

**DQ5 — VQ6 redux: PI, and the blind-review problem.** Two coupled issues. (i) **Who is
PI?** An individual may be PI on only one EOI, so if you intend to be PI here that forecloses
others. (ii) The RFP requires a description of team composition *and* external institutional
partnerships (item 3) while forbidding **any** names or affiliations — "proposals will be
returned without evaluation if any identifying information is included." So the team section
must be written purely as roles/expertise/geography ("a hyperspectral radiative-transfer
group", "an ocean-state-estimation group at a US oceanographic institution"). Confirm that
approach, and tell me which *roles* to claim — in particular, are the ECCO/ROMS people
(Menemenlis, Forget, Edwards, and Mazloff on the SIO side) actually committed, or aspirational?
I should not imply partnerships that do not exist.

DQ5: JXP is the PI.  The ECCO/ROMS people are committed.

**DQ6 — VQ2 redux: which theme leads?** T1 (budget accuracy, ocean interior carbon, sink
trends), T2 (policy timescales, natural-vs-anthropogenic attribution, near-real-time), T3
(future, surprises, regime shifts). Under DQ1(c) I would lead **T1**, with **T2** strong
because the natural-vs-anthropogenic separation *is* §6.2's residual problem, and touch T3.
Under DQ1(a) T1 alone is the fit. Confirm?

DQ6: I confirm

**DQ7 — Differentiation, and a real overlap risk.** The Schmidt carbon portfolio already
contains: **COCO2** (Southern-Ocean air–sea CO₂, USVs, XAI, OSSEs), **InMOS** (integration of
models and observations across scales — OBVI), **SUBSEA** (subtropical underwater
biogeochemistry and *subsurface export*), **CALIPSO** (carbon loss in plants, soils, oceans),
and **CLARiTy** (the land/EO+AI analogue of what we would do for the ocean). Our
model–obs-integration and subsurface-export ambitions genuinely overlap InMOS and SUBSEA, and
CLARiTy is the precedent for "EO + AI reduces a flux uncertainty." Do we (a) name these and
propose explicit collaboration/complementarity — my lean, since the RFP *expects* cross-project
engagement — or (b) stay silent and risk a reviewer concluding it is already funded? If (a),
how much of our 3 pages is that worth (I would spend ~4 sentences)?

DQ7: (a) and yes, ~4 sentences is good

**DQ8 — How hard do we lean on the RT prototype, and do we show it?** The 7.21%→0.30% ladder
is genuine, quantitative, de-risking preliminary evidence, and a *differentiable* forward
model is precisely the prerequisite for variational inversion and for embedding the optics
inside an assimilative model — i.e. it is the technical bridge to DQ1(c). But it is the
forward half only. Options: (a) one figure showing the rRMS ladder + one paragraph, framed as
"the forward operator is built and differentiable; the inversion is the proposal" (my lean);
(b) a sentence, no figure, saving space; (c) lead with it. Also: with 3 pages *including*
figures and five required content elements, I recommend **exactly one figure** — do you agree,
and should it be the rRMS ladder or a concept/architecture schematic? (References go in a
separate PDF, so they cost us no page space.)

DQ8: Ignore the RT prototype for now.  I shouldn't have bothered to have you examine the code in this Repo

**DQ9 — Do we carry the §6 physics honesty into the EOI, or keep it internal?** Stating
plainly that the biological limb is *not* the dominant term in the anthropogenic inventory is
unusual candour for a proposal, and a hostile reviewer could quote it back as "they admit
their own quantity doesn't matter." But it is also our sharpest differentiator: it shows we
know exactly which term we are closing and why the physical-constraint programmes cannot close
it, and it pre-empts the obvious referee objection. My lean: **yes, two sentences, framed as
scoping precision rather than concession** — and it is the natural justification for the
ECCO-Darwin coupling. Agree?

DQ9: Yes, include 2 sentences, framed as scoping precision rather than concession

**DQ10 — Two mechanical confirmations.** (i) Item 5 requires a brief statement on
**generative-AI use**; this draft, the report and much of the prototype are Claude-assisted, so
I propose a plain, factual sentence or two (excluded from merit review per the RFP). Confirm
the wording posture — full disclosure, or minimal? (ii) When you do ask for the Google doc:
target folder confirmed to exist — `Schmidt Sciences` under `.../Proposals/2026/`
(id `1Xk9o3OcI9-f0KulOQFolsxFibjGCzS4a`), and I will name the doc `EOI-proposal`. Note a
Google doc cannot itself satisfy the "single .pdf, 11-pt, ≤3 pages" submission format, so
treat it as the drafting surface and plan a PDF export step.

DQ10: (i) yes, that is fine and (ii) I will download the final doc as a PDF for submission

#### Round 4 — after the DQ answers (2026-08-05)

All ten answered; the direction is now fixed: **DQ1(c)** coupled optics + ECCO-Darwin, opening
on carbon and uncertainty with the expert system as *method*; **factor-2 reductions on 2–3
named quantities**; **Cphyto + POC** headline with NPP/export propagated and the subsurface via
BGC-Argo/lidar; **~$10M/5 yr** computational-synthesis scope; **JXP as PI** with ECCO/ROMS
committed; **T1 lead, T2 strong, T3 touched**; portfolio differentiation in ~4 sentences; §6
candour in 2 sentences; RT prototype **dropped**; AI-use statement plain and factual.

**First, the answer to your DQ1 sub-question — and I think it is the best idea in this round.**

*"Would assessing the relative evolution over the past 20 years with satellite data be a
worthwhile target too?"* **Yes — but as a *detectability* deliverable, not as a trend number,
and framed that way it is strong enough to be one of the DQ2 headline numbers.**

Reported as a trend, 20 years is demonstrably too short, and the literature says so with
numbers we already have in the report: to detect a climate-driven trend at 90% power / 95%
confidence needs n\* = **31.5 yr for chlorophyll, 32.3 yr for integrated PP, 32.0 yr for export
flux** [55]; 50+ yr for chlorophyll and 25–40 yr for export production in the large-ensemble
comparison [54]; 23 yr globally and 27–85 yr regionally for the soft-tissue pump [53]. Gregg &
Rousseaux found −0.8 Pg C yr⁻¹ dec⁻¹ over 18 yr but cautioned the record is "still too short to
unequivocally distinguish natural variability from long-term trends" [58], and Behrenfeld's
1997–99 ENSO swing of ~1,930 Tg C yr⁻¹ [17] is far larger than any plausible secular signal. Six
satellite NPP algorithms do not even agree on the *sign* of the 1998–2023 trend [21]. So a
20-year trend claim would be exactly the kind of thing §3 of our report tells us not to believe.

But the detection-time formula is where this becomes a *deliverable*. Weatherhead et al. (1998),
as used by Henson [55] and Beaulieu et al. (2013):

    n* = [ 3.3 σ_N / |ω₀| · √((1+φ)/(1-φ)) ]^(2/3)

so **n\* ∝ σ_N^(2/3)**: halving the noise term shortens time-to-detection by 2^(2/3) ≈ **1.59×**
(a 37% cut). Applied to [55]'s numbers, 31.5 → **≈20 yr** for chlorophyll-class biomass and
32.0 → **≈20 yr** for export. *That is precisely the length of the record we already have.* And
there are three distinct levers, two of which are not merely "better error bars":

  1. **Chl → Cphyto** removes a genuine *variance* source, not just measurement error:
     >55% of interannual chlorophyll anomalies over >75% of the ocean are photoacclimation
     rather than biomass [17]. A carbon-based record has structurally lower σ_N than a
     chlorophyll record.
  2. **Cross-mission harmonisation.** Beaulieu et al. (2013) show a mid-record discontinuity
     inflates n\* through a 1/[1−3τ(1−τ)]^(1/3) factor — global chlorophyll goes from **27 yr to
     43 yr**. The SeaWiFS→MODIS→PACE seam *is* that discontinuity. One coherent physical
     retrieval across all three missions, rather than per-mission empirical algorithms, is worth
     roughly **16 years of record** on its own — and it is exactly what an "expert system"
     justifies.
  3. **Halving per-pixel retrieval uncertainty**, the DQ2 factor-2 target, contributing its
     own 1.59×.

**The honest caveat, which I would state in the draft.** σ_N in that formula is *total* residual
noise — real geophysical variability plus measurement error — so halving the *retrieval* error
does not by itself halve σ_N; ENSO-scale variability remains. Levers 1 and 2 are the ones that
attack σ_N structurally. I would therefore claim "brings biomass-trend detection within reach of
the existing 25-year record" rather than a hard "20.0 yr", and show the calculation.

**Proposed consequence for DQ2:** the three factor-2 numbers become **precision** (per-pixel
Cphyto uncertainty), **accuracy** (the conversion-driven global-stock spread), and
**detectability** (n\* for a biomass/export trend). That trio is coherent, each is quantitative,
and the third is the one a policy audience will care about most — it maps straight onto T2
("role of climate variability", "attribution") and T3 ("detection of regime shifts").

---

**RQ1 — Confirm the detectability framing and the trio above?** Specifically: (i) adopt
time-to-detection as the third headline number; (ii) claim the 20-year record becomes
*interpretable* rather than claiming a trend value; (iii) include the cross-mission-discontinuity
argument (the 27→43 yr result), which I think is the single most persuasive line available for why
a unified retrieval matters. Any objection to putting a version of the n\* calculation in the EOI?

RQ1: I confirm

**RQ2 — With the RT prototype dropped, what carries feasibility?** Your DQ8 removes the only
quantitative preliminary result, and a $10M EOI needs to show the team can do this. Candidates,
all citable without the repo: (a) **BING** (Prochaska & Frouin 2025, *Biogeosciences* 22:4705) —
peer-reviewed Bayesian Gordon+MCMC framework with honest UQ, and the published demonstration that
multispectral supports only ~3 independent parameters; (b) **Bisson et al. 2023** — seeding GIOP
with ancillary bb cuts seasonal absorption bias >50%, an *external* proof-of-concept for the
priors thesis; (c) the biomass report itself as evidence of command of the uncertainty landscape;
(d) ECCO-Darwin's own track record. **My lean: (a) + (b) as the technical case, (d) for the
modeling half.** Confirm — and see RQ3, because (a) is self-citation under blind review.

RQ2: I confirm

**RQ3 — The blind-review problem with the astronomy framing.** This one worries me. The
concept's central differentiator — astronomers bringing mature large-survey spectroscopic
methodology, with *Prospector* and *The Cannon* as analogues — points unambiguously to one
person, and the Simons text's "the PypeIt software package that I co-founded" is explicitly
identifying. The RFP is blunt: proposals containing identifying information are "returned without
evaluation." Options: (a) keep the methodological argument but strip all identifiers — "methods
developed for large astronomical spectroscopic surveys", no package names, no first person, and
cite BING in the third person in the separate reference PDF (my lean; this is standard blind
practice); (b) drop the astronomy framing entirely and argue the method on its merits; (c) keep
it fully and accept the risk. I strongly recommend (a) — but it does blunt your signature move,
so it is your call. Related: do you want the reference PDF to include your own papers at all?

RQ3: Sorry, drop the astronomy angle.  That's not part of this proposal.  We are all oceanographers.

**RQ4 — Baselines for the factor-2 claims.** "Factor of 2" needs a stated starting point or it
is unfalsifiable. My proposed baselines, all from the report: **(i) per-pixel Cphyto** — Stoer &
Fennel's MAPE ~32% [10] → target ~16%; **(ii) global-stock spread from conversion choice** —
3.5× (218–771 Tg C) [10] → target <1.8×; **(iii) n\*** — 31.5 yr [55] → ~20 yr. Note for POC the
honest baseline is Koestner's **28%** (bbp+Chl), *not* the 47% bbp-only figure — halving 47%
would be claiming credit for work already published [13,14]. Confirm these three and their
baselines, or substitute?

RQ4: I confirm these three

**RQ5 — What does "coupled to ECCO-Darwin" actually mean operationally?** This is the core of
the research plan and drives cost. (a) **One-way**: our optical products validate/force the
model. (b) **Two-way assimilation**: optical constraints enter through ECCO's adjoint — the most
powerful and the reason to involve the ECCO team at all, but a serious lift. (c) **Iterated
priors**: the model supplies environmental/time-series priors that break the retrieval
degeneracy, and the improved retrievals in turn constrain the model — which is the closest match
to the project's founding thesis that priors supply the missing information. **My lean: (c) as
the scientific core, with (b) as a named stretch goal.** Which?

RQ5: Yes, (c) as the scientific core, with (b) as a named stretch goal.

**RQ6 — Is atmospheric correction in scope?** The concept lists "improved atmospheric
corrections." It matters here more than usual: Werdell et al. 2018 put AC at up to ~50% of
retrieval error, and for a *multi-mission 20-year* record (RQ1) inter-mission radiometric
calibration plus AC is plausibly the dominant term in σ_N — i.e. RQ1's detectability claim may
depend on it. But it is a large programme in its own right. In scope as a work package,
explicitly out of scope with the risk acknowledged, or handled only as cross-mission
*harmonisation* (my lean — the narrower, sufficient version)?

RQ6: It is in scope and co-I Robert Frouin will lead that effort.  I expect we will wind up re-processing *all* of the PACE data and will need to budget the compute for this.

**RQ7 — Team roles to claim, and the international gap.** For item 3 I will write roles only, no
names. Proposed set: hyperspectral radiative-transfer & inversion; ocean state estimation and
adjoint data assimilation; regional ocean modeling; BGC-Argo / autonomous float optics;
radiometry and atmospheric correction; in-situ bio-optics and phytoplankton biogeochemistry;
statistical/AI methods for large spectroscopic surveys; project management. Two gaps: (i) the RFP
"strongly encourages" **international** teams and the concept promises an international
collaboration, but the group as named is entirely US-based — do we add a non-US partner, and in
which role? (ii) ECR leadership is encouraged; you are PI, which is fine, but should I foreground
ECR-held work-package leadership? Also confirm: is a carbon-flux/BGC-modelling role (beyond
ECCO-Darwin) needed, or does ECCO-Darwin cover it?

RQ7: That's fine. I can fill in the names.

#### Round 5 — after the RQ answers (2026-08-05)

RQ1, RQ2, RQ4, RQ5 confirmed as proposed. Two answers change things materially:

- **RQ3 — astronomy angle dropped entirely** ("we are all oceanographers"). Noted, and I will
  purge it: no *Prospector*/*The Cannon* analogy, no survey-spectroscopy heritage, no first
  person. This is a deliberate divergence from `VICC_Concept_v1.pdf` / `Prochaska-Schmidt.pdf`,
  whose central novelty argument *was* the astronomy transfer — so the EOI now needs a different
  stated source of novelty (SQ3).
- **RQ6 — atmospheric correction is in scope, Frouin leads, and we expect to reprocess *all* of
  the PACE data with compute budgeted.** This is the largest scope change since Round 3. It
  strengthens RQ1 (I had flagged that the detectability claim might depend on AC, and now it is
  owned), but it raises a consistency problem (SQ1) and a real budget question (SQ2).

**RQ7 half-answered.** "I can fill in the names" covers the separate team form, but the two
sub-questions — international partner, ECR work-package leadership — are still open (SQ5).

---

**SQ1 — Which missions get reprocessed? This is the one genuine inconsistency I see.** RQ6 says
"*all* of the PACE data," but RQ1's detectability claim is built on the **long** record: n\* ≈
20 yr only means something if we *have* ~20+ years, and PACE launched Feb 2024. The
27→43 yr discontinuity argument is likewise explicitly a SeaWiFS→MODIS→PACE seam argument. So
PACE alone gives 2–8 yr over the award and the n\* story collapses. Options: (a) reprocess
**MODIS (2002–) + PACE**, treating SeaWiFS (1997–2010) as a consistency check — ~24 yr baseline,
my lean; (b) all three missions, maximum baseline (~29 yr) and maximum cost; (c) PACE only, and
drop the detectability number back to a methods claim. Which? (a) or (b) is what makes RQ1
deliverable, and it roughly doubles or triples the reprocessing volume implied by RQ6 — so this
question and SQ2 must be answered together.

SQ1: Ok, good point.  Let's go with (a).

**SQ2 — Compute: how much, where, and does this break the $10M ceiling?** DQ4 fixed
"~$10M, computational/synthesis," but a full multi-mission hyperspectral reprocessing is a
different order of cost than a synthesis project, and the preliminary budget is a required EOI
element. I need three things from you: (i) **a volume/compute estimate** — I do *not* have a
verified figure for PACE OCI L1B daily volume (my rough sense is order-1 TB/day, which would make
a multi-year full reprocessing petabyte-scale and order-10⁵ core-hours; I would rather use your
number, or verify one, than guess in a budget); (ii) **where it runs** — NASA/OB.DAAC or ASDC
allocation, NSF ACCESS, campus cluster, or commercial cloud (each reads very differently to a
reviewer, and cloud egress on a petabyte-scale derived product is a real line item); (iii)
whether we now **exceed $10M**. Note the RFP explicitly permits exceeding $10M where justified,
and compute for a mission-scale reprocessing is a cleaner justification than most — but it
contradicts DQ4(a), so I want your call rather than my inference. Related: FAIR open-data is a
project requirement, so hosting/publishing the derived multi-mission product is also a cost —
include it as a named line?

SQ2: (i) Your estimate of 1Tb per day is good.  We will need to include MODIS too, however; (ii) AWS or like platform.  Note that we may use GPUs instead of CPUs; (iii) I don't think we will even come close to $10M.

**SQ3 — With the astronomy angle gone, what is the novelty claim, and what do we call the
thing?** Two parts. (i) **Novelty.** The remaining candidates, all oceanographic: physics-based
Bayesian inversion with *calibrated per-pixel* uncertainty (the BING lineage) rather than one of
the ~100 empirical algorithms; **priors** from in-situ, environmental and time-series information
as the way to break a degeneracy that is provably unbreakable from Rrs alone; a *single
physically consistent* retrieval applied across all missions instead of per-mission empiricism;
and end-to-end propagation from radiance to carbon. My lean is to lead on **"the degeneracy is
information-limited, so we supply the missing information, and we report honest uncertainty"** —
which is both true and the thing no existing product does. Agree? (ii) **Name.** "Expert system"
came from the dropped concept framing and is a dated AI term that a carbon panel may misread as
1980s rule-based AI. Alternatives: "a physics-based Bayesian retrieval and assimilation
framework", or just describe it functionally. Keep "expert system" or replace it?

SQ3:  This queestion is distracting.  The team has terrific expertise and the empahsis on ECCO and high-quality inversions is fine.

**SQ4 — Does "all of PACE" mean OCI only, or the polarimeters and the UV too?** Both bear
directly on the DQ3 headline (Cphyto + POC separation), so this is a science question, not a
detail. (i) **Polarimetry** — PACE carries SPEXone and HARP2 alongside OCI; multi-angle
polarised radiance constrains the particle **phase function**, which is exactly the `bb_p`/NAP
information the report identifies as the top-ranked addressable uncertainty (§4 items 1–2). In
scope, or OCI-only? (ii) **UV (350–400 nm)** — the Simons text made the UV window central to
separating CDOM/detritus from phytoplankton absorption. That argument is oceanographic and
survives RQ3's cut, and it directly serves the aph/adg degeneracy. Keep the UV lever in, or drop
it with the astronomy framing? My lean: keep UV (cheap, and it attacks our headline degeneracy),
and mention polarimetry as a named secondary constraint rather than a work package.

SQ4: (i) OCI is the baseline and polarimetry is a stretch goal; (ii) Definitely include UV.  This especially leans on atmospheric corrections.

**SQ5 — RQ7's two open halves.** (i) **International partner.** The RFP "strongly encourages"
international teams and the original concept promised an international collaboration, but the
group as it stands is entirely US-based. Do we add a non-US partner, and in which role? Natural
fits given the science: a European ocean-colour/radiometry group (OLCI/Copernicus heritage), a
BGC-Argo national programme, or a hyperspectral-algorithm group. This is a cheap, high-value box
to tick — but only if it is real. (ii) **ECR leadership.** Encouraged by the RFP; you are PI,
which is fine, but shall I state that specific work packages are ECR-led? (iii) Confirm Frouin
and the ECCO/ROMS people appear **only** in the separate names form, never in the EOI body.

SQ5:  (i) We will have non-US partners.  I will list them later. (ii) Yes, we will have ECR-led work packages; and (iii) Agreed.

**SQ6 — Does AC get its own headline number, making four?** DQ2 said 2–3 numbers and RQ4 fixed
three (per-pixel Cphyto 32%→16%; stock spread 3.5×→<1.8×; n\* 31.5→~20 yr). Now that AC is a led
work package, a fourth is available — e.g. a target on cross-mission radiometric consistency, or
on the AC contribution that Werdell et al. put at up to ~50% of retrieval error. My lean: **keep
three headline numbers and present AC as the *enabler* of the third** (it is the honest logical
role — reducing AC and inter-mission bias is what actually lowers σ_N in the n\* calculation).
Agree, or promote AC to a fourth number?

SQ6: AC probably can't get boiled down to a number.

#### Round 6 — after the SQ answers (2026-08-05)

All six answered. **I believe I now have enough to write the draft**, so this round is short: three
questions that genuinely block a required EOI element, plus a list of decisions I have taken myself
rather than asking (taking the SQ3 signal that framing questions are distracting).

**Settled:** SQ1(a) MODIS (2002–) + PACE reprocessed, SeaWiFS as consistency check → ~24-yr
baseline, which repairs the RQ1/RQ6 inconsistency; SQ2 ~1 TB/day PACE, MODIS included, AWS-like
platform, possibly GPU rather than CPU, and **compute will not approach $10M** (so DQ4(a) holds);
SQ3 lead on team expertise + ECCO + high-quality inversions; SQ4 **OCI baseline, polarimetry a
stretch goal, UV definitely in** and leaning on AC; SQ5 non-US partners exist (names later),
ECR-led work packages yes, names only in the separate form; SQ6 three headline numbers with AC as
enabler, not a fourth number.

**Decisions I am taking without asking** (say the word if any is wrong):
- Drop the phrase **"expert system"**; describe the system functionally (a physics-based Bayesian
  retrieval + assimilation framework). It came from the dropped astronomy framing and reads as
  1980s rule-based AI to a non-AI panel.
- Keep the three RQ4 numbers exactly as confirmed; AC appears as the mechanism behind the third.
- §6 candour: two sentences, scoping precision (DQ9).
- Portfolio differentiation: ~4 sentences naming COCO2 / InMOS / SUBSEA / CLARiTy (DQ7).
- Budget structure: personnel-dominated, with named lines for cloud compute/storage, the 5-day
  annual science-team meeting (first Oct 2027, Europe), open-data publication, and indirect ≤10%.
- No first person, no names, no institutions anywhere in the body; references in the separate PDF
  with self-citation in third person.

---

**TQ1 — A real tension I should flag before drafting: the UV lever and the long-record claim do
not cover the same data.** PACE/OCI reaches ~350 nm, but **MODIS and SeaWiFS have no band below
412 nm** — their shortest ocean-colour channel is 412. So the UV window you confirmed in SQ4(ii)
exists only for PACE, i.e. for ~8 of the ~24 years, while the detectability claim (RQ1) rests on
the *MODIS* record. Two of our three headline numbers are therefore likely to live on different
data than the third. My proposed wording, which I think is honest and still strong: the **precision
and accuracy** gains (Cphyto 32%→16%; stock spread 3.5×→<1.8×) are demonstrated in the
**PACE/hyperspectral+UV era**, and are then *transferred back* onto the multispectral record via
one physically consistent retrieval and cross-mission harmonisation — which is what delivers the
**detectability** number over 24 years. That framing makes PACE the calibration anchor for the
heritage record rather than pretending UV information exists before 2024. Does that match your
intent, or do you want the factor-2 claims stated as PACE-era only?

TQ1: Yes, that matches my intent.

**TQ2 — Team scale for the preliminary budget (required element 4).** You will supply names later,
but I need *counts* to write a credible budget paragraph. Proposed shape for ~$10M/5 yr with
indirect ≤10% and compute well under budget: roughly 5–7 institutions; ~4–6 postdocs, ~3–4
graduate students, 1–2 research software engineers, partial-FTE senior investigators across the
work packages (inversion, AC/radiometry, state estimation & assimilation, regional modeling, float
optics, in-situ bio-optics), plus a part-time project manager. Are those counts about right, and
how many institutions should I imply? (I will state numbers and roles only, never names.)

TQ2: Leave the budget to me.

**TQ3 — The one figure.** DQ8 removed the RT-prototype figure, so the single figure slot is open. My
lean is one two-panel figure that carries the whole argument: **left**, the current
factor-of-disagreement cascade from the report (satellite bbp 2.5×, Cphyto 3.5×, NPP 2.5×, export
3×) — i.e. why today's products cannot answer the carbon question; **right**, the n\* detectability
result — trend-detection time falling from 43 yr (discontinuous multi-mission record) and 31.5 yr
(continuous) to ≈20 yr, against the ~24-yr record length drawn as a vertical line. That shows the
problem and the deliverable in one image, and both panels are already generated from cited
literature in `reports/py/`. Approve, or would you rather the figure be an
architecture schematic (retrieval → priors ↔ ECCO-Darwin → carbon + uncertainty)?

TQ3:  Yes, that sounds great.  Generate it with Python and put the script in retrieve-or-bust/proposals/Schmidt_Sciences.

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

### 2026-08-05 (First draft prompt 1: read all context, posed Round-3 Q&A — no draft written)

Executed the 1st "First draft" task. The prompt explicitly says **do not write the draft yet**
and to ask questions first, so this pass is read-and-question only: nothing was drafted and no
Google doc was created.

**Read.** `VICC_Phase2_EOI.pdf` (re-read closely for the rules), `VICC_Concept_v1.pdf`, the
Google doc it links (`1jXBhHtzcCdrIFeYal6luBg…` — verified identical in content to
`Prochaska-Schmidt.pdf`), the "An expert ocean color system" section, the relevant Simons
sections (III ocean-optics innovations; V applied AI/ECCO; II CMB and IV BOONUS for context),
`reports/biomass_summary.md`, `context/context_summary.md`, and the repository code.

**Key discovery — the code is not on `main`.** `robust/` on `vicc-proposal`/`main` contains
only a 1-line `__init__.py`; the real package lives on the **`rt-elastic-prototype`** branch
(also partly on `RT`), along with `design/rt_elastic_implementation.md` (v0.14, 2026-08-04) and
`context/RT/fig_rrms_ladder.csv`. Read it via `git show` rather than checking out.

**What the prototype gives the EOI.** A differentiable elastic RT forward model in JAX,
`Rrs = Rrs_ZTT(θ) + ΔRrs_emulator(θ)` — ZTT analytic backbone with explicit phase-function
dependence plus a 417-parameter Flax residual MLP. Held-out rRMS in `rrs` space on the L23
batch: **Gordon 7.21% → ZTT 5.93% → linear hybrid 2.54% → MLP hybrid 0.30%**. Gradients
verified against central finite differences; 225 tests pass; CI on Python 3.12/3.14. Honest
caveats recorded for the draft: it is the **forward** model only (no inversion yet), M2 retains
two `NotImplementedError`s, M4 validation not started, and 60°-zenith extrapolation is
**not reproducible** across seeds (4.7–12.2%, median 7.75% vs ZTT 8.09%) — report-and-defer,
with `DomainWarning` implemented.

**What the concept doc changes.** `VICC_Concept_v1.pdf` names **ECCO-Darwin informed by ROMS
tuned to observations** as the modeling core. That is effectively VQ4**(c)** — full coupling to
an assimilative BGC model — not my Round-1 lean of (b). It also supplies the answer to
Mazloff's §6 critique: the assimilative model can carry the *physical* limb while optics
carries the *biological* one. The Simons doc supplies the plausible partners (Menemenlis and
Forget on ECCO; Edwards on ROMS), which is why DQ5 asks whether those are committed or
aspirational.

**EOI rules captured (they bind the draft).** ≤3 pages *including figures*, 11-pt single-spaced,
PDF; **fully blind** — any PI/team/institutional name returns the proposal unevaluated — yet
item 3 still requires a team-composition and partnership description, so that section must be
roles/expertise/geography only; Vancouver references in a **separate** PDF (so they cost no page
space); five required elements (narrative, research plan, team, preliminary budget, generative-AI
statement); $10M/5 yr typical with Oct 2027–Sep 2032 performance, indirect ≤10%, 5-day annual
science-team meeting budgeted (first Oct 2027, Europe); larger awards allowed if new
instrumentation/data acquisition is justified; PI on one EOI only; due **Aug 28 2026**.

**Posed DQ1–DQ10** (Q&A Round 3), with my lean on each rather than open-ended asks. The load-
bearing ones: **DQ1** the framing fork (biomass-first vs the §6 non-steady-state residual vs
coupled optics+ECCO-Darwin — I lean coupled, opening on carbon not on the tool); **DQ2** what
uncertainty-reduction number we are willing to defend, given that VICC asks for an order of
magnitude and §3 of our own report makes 10× indefensible (COCO2's precedent is a specific 25%);
**DQ5** PI identity plus the blind-review/team tension and whether the ECCO/ROMS partnerships are
real; **DQ7** genuine overlap risk with InMOS (model–obs integration), SUBSEA (subsurface
export) and CLARiTy (the land EO+AI precedent); **DQ9** whether to carry §6's candour about the
biological limb into the EOI. Also flagged that a Google doc cannot itself meet the "single PDF,
11-pt, ≤3 pp" format, so a PDF export step is needed, and confirmed the target Drive folder
exists (`Schmidt Sciences`, id `1Xk9o3OcI9-f0KulOQFolsxFibjGCzS4a`).

**Note.** VQ1–VQ6 from Round 1 remain unanswered; DQ1/DQ3/DQ4/DQ5/DQ6 fold them in so a single
pass of answers can unblock the draft.

Next: await DQ1–DQ10, then write the first draft (and, when asked, the `EOI-proposal` Google doc).

### 2026-08-05 (First draft prompt 2: read the DQ answers, posed Round-4 Q&A — still no draft)

Executed the 2nd "First draft" task: read JXP's answers to DQ1–DQ10 and posed **RQ1–RQ7**. The
prompt again says **do not write the draft yet**, so nothing was drafted and no Google doc created.

**Direction now fixed by the DQ answers.** DQ1(c) coupled optics + ECCO-Darwin, opening on carbon
and uncertainty with the expert system as *method* not headline; DQ2(a) with **factor-2 reductions
on 2–3 named quantities**; DQ3 **Cphyto + POC** headline, NPP/export propagated, subsurface via
BGC-Argo/lidar; DQ4(a) **~$10M/5 yr** computational-synthesis (no field/lab campaign); DQ5 **JXP
is PI**, ECCO/ROMS partners **committed**; DQ6 **T1 lead, T2 strong, T3 touched**; DQ7(a) name the
portfolio projects and propose complementarity, ~4 sentences; DQ8 **drop the RT prototype
entirely**; DQ9 include the §6 candour in 2 sentences as scoping precision; DQ10 plain factual
AI-use statement, JXP exports the PDF himself.

**Answered JXP's DQ1 sub-question (20-year biomass evolution) — the substantive result of this
pass.** Verified the Weatherhead et al. (1998) detection-time formula from the cached full text of
Beaulieu et al. (2013), Eq. 8: `n* = [3.3 σ_N/|ω₀| · √((1+φ)/(1-φ))]^(2/3)`, so **n\* ∝
σ_N^(2/3)** and halving the noise term shortens time-to-detection by 2^(2/3) ≈ **1.59×**. Against
Henson et al. 2016's n\* values (chlorophyll 31.5 yr, integrated PP 32.3 yr, export flux 32.0 yr)
that lands at **≈20 yr — the length of the existing record**. Three levers, two of them structural
rather than cosmetic: (1) Chl→Cphyto removes real variance, since >55% of interannual chlorophyll
anomalies over >75% of the ocean are photoacclimation not biomass (Behrenfeld 2016); (2)
cross-mission harmonisation removes the discontinuity penalty — Beaulieu's global chlorophyll n\*
goes **27 yr → 43 yr** with a mid-record break, so treating the SeaWiFS→MODIS→PACE seam with one
coherent physical retrieval is worth ~16 years of record, which is the strongest available argument
for the expert-system architecture; (3) the DQ2 factor-2 retrieval-uncertainty cut itself.

**Verdict given:** yes, worthwhile — but as a **detectability** deliverable, not a trend number,
and it should become the third DQ2 headline figure. Logged the honest caveat that σ_N is *total*
residual noise (geophysical variability + measurement error), so halving retrieval error does not
halve σ_N; ENSO-scale variability persists, and levers 1–2 are the ones that attack σ_N
structurally. Recommended claiming "brings biomass-trend detection within reach of the existing
record" with the calculation shown, rather than a hard 20.0 yr.

**Proposed the headline trio** — precision (per-pixel Cphyto), accuracy (conversion-driven
global-stock spread), detectability (n\*) — which maps cleanly onto T1/T2/T3.

**RQ1–RQ7 posed.** The two I consider blocking: **RQ2**, because dropping the RT prototype removes
the only quantitative feasibility evidence, so something must carry it (my lean: BING + Bisson 2023
for the technical case, ECCO-Darwin's record for the modeling half); and **RQ3**, a blind-review
hazard I had not fully appreciated until reading the concept and Simons text together — the
astronomy framing (*Prospector*/*The Cannon*, "the PypeIt package that I co-founded") identifies
one person unambiguously, and the RFP returns identifying proposals unevaluated. Recommended
keeping the methodological argument but stripping every identifier and self-citing in third person.
Also asked: **RQ4** baselines for the factor-2 claims (flagging that the honest POC baseline is
Koestner's 28%, not the 47% bbp-only number, or we would be claiming credit for published work);
**RQ5** what "coupled to ECCO-Darwin" means operationally (lean: model-supplied priors ↔ improved
retrievals, with adjoint assimilation as a named stretch goal); **RQ6** whether atmospheric
correction is in scope, since for a multi-mission record it may dominate σ_N and therefore RQ1's
claim (lean: the narrower cross-mission harmonisation only); **RQ7** the role list for the
anonymised team section, plus the international gap (the named group is entirely US-based while the
RFP strongly encourages international teams and the concept promises an international
collaboration) and whether to foreground ECR work-package leadership.

Next: await RQ1–RQ7, then write the first draft.

### 2026-08-05 (First draft prompt 3: read the RQ answers, posed Round-5 Q&A — still no draft)

Executed the 3rd "First draft" task: read JXP's RQ1–RQ7 answers and posed **SQ1–SQ6**. The prompt
again says **do not write the draft yet**; nothing drafted, no Google doc created.

**Confirmed as proposed:** RQ1 (detectability framing, the n\* calculation in the EOI, and the
27→43 yr cross-mission-discontinuity argument), RQ2 (BING + Bisson 2023 for the technical case,
ECCO-Darwin for the modeling half), RQ4 (the three factor-2 baselines: Cphyto MAPE 32%→16%;
conversion-driven stock spread 3.5×→<1.8×; n\* 31.5→~20 yr), RQ5 (iterated priors as the
scientific core, adjoint assimilation as a named stretch goal).

**Two answers changed the shape of the proposal.**

1. **RQ3 — drop the astronomy angle entirely** ("we are all oceanographers"). I will purge every
   trace: no *Prospector*/*The Cannon*, no survey-spectroscopy heritage, no first person. Worth
   recording that this is a deliberate divergence from `VICC_Concept_v1.pdf` and
   `Prochaska-Schmidt.pdf`, whose central novelty argument *was* the astronomy transfer — so the
   EOI now needs a different stated source of novelty, which is why SQ3 asks what replaces it and
   whether the phrase "expert system" (itself from the dropped framing, and a dated AI term a
   carbon panel could misread) survives.
2. **RQ6 — atmospheric correction is in scope, Frouin leads it, and we expect to reprocess *all*
   of the PACE data with compute budgeted.** Biggest scope change since Round 3. It strengthens
   RQ1 — I had flagged the detectability claim as possibly dependent on AC, and it is now owned
   rather than assumed — but it creates a consistency problem and a budget problem.

**The consistency problem (SQ1), which I think is the most important thing in this pass.** RQ6
says "all of the PACE data," but RQ1's n\* ≈ 20 yr claim requires a ~20+ year record and PACE
launched Feb 2024; the 27→43 yr discontinuity argument is explicitly about the SeaWiFS→MODIS→PACE
seam. PACE alone yields 2–8 yr over the award and the detectability number collapses. So the
reprocessing scope has to extend to MODIS (2002–) at minimum — my lean — with SeaWiFS as a
consistency check, which roughly doubles or triples the implied volume. SQ1 and SQ2 therefore have
to be answered together.

**The budget problem (SQ2).** DQ4 fixed "~$10M, computational/synthesis," but a multi-mission
hyperspectral reprocessing is a different order of cost, and the preliminary budget is a required
EOI element. Asked for a volume/compute estimate, where it runs (OB.DAAC/ASDC vs ACCESS vs campus
vs cloud — each reads differently to a reviewer, and petabyte-scale cloud egress is a real line),
and whether we now exceed $10M (the RFP permits it when justified, and mission-scale compute is a
cleaner justification than most — but it contradicts DQ4(a), so it is his call). **Deliberately did
not invent a number:** I have no verified figure for PACE OCI L1B daily volume and said so rather
than putting a guess into a budget.

**Also asked.** SQ4 — whether "all of PACE" includes the **polarimeters** (SPEXone/HARP2, which
constrain the particle phase function, i.e. precisely the bb_p/NAP terms the report ranks first
and second among addressable uncertainties) and the **UV 350–400 nm** lever (whose CDOM/detritus-
vs-phytoplankton argument is oceanographic and survives RQ3's cut); my lean is keep UV, mention
polarimetry as a secondary constraint rather than a work package. SQ5 — RQ7's two unanswered
halves (international partner, ECR-led work packages) plus confirmation that Frouin and the
ECCO/ROMS people appear only in the separate names form. SQ6 — whether AC becomes a fourth
headline number; my lean is keep three and present AC as the *enabler* of the third, since
lowering AC and inter-mission bias is literally what reduces σ_N in the n\* calculation.

Next: await SQ1–SQ6, then write the first draft.

### 2026-08-05 (First draft prompt 4: read the SQ answers, posed a short Round-6 — still no draft)

Executed the 4th "First draft" task: read JXP's SQ1–SQ6 answers and posed **TQ1–TQ3**. Prompt again
says **do not write the draft yet**; nothing drafted, no Google doc. Kept this round deliberately
short — SQ3 ("this question is distracting") was a signal to stop asking framing questions — and
recorded the decisions I am taking myself rather than re-asking.

**Settled by the SQ answers.** SQ1(a): reprocess **MODIS (2002–) + PACE**, SeaWiFS as a consistency
check → ~24-yr baseline. That repairs the RQ1/RQ6 inconsistency I raised last round, and it is what
makes the n\* detectability number deliverable. SQ2: ~1 TB/day for PACE is an acceptable working
figure, MODIS included, AWS-like platform, possibly **GPU** rather than CPU, and compute **will not
approach $10M** — so DQ4(a)'s ~$10M computational-synthesis scope stands and I do not need to argue
for exceeding the ceiling. SQ3: lead on team expertise plus ECCO and high-quality inversions. SQ4:
**OCI is baseline, polarimetry a stretch goal, UV definitely in**, and JXP notes UV leans especially
hard on AC. SQ5: non-US partners exist (names later), ECR-led work packages yes, names confined to
the separate form. SQ6: AC "probably can't get boiled down to a number" — so three headline numbers
stand with AC as the enabler of the third, as I'd leaned.

**The one substantive problem found this pass (TQ1).** Verified from the Simons text that PACE/OCI
reaches ~350 nm and polarises 385–770 nm; **MODIS and SeaWiFS have no band below 412 nm**. So the UV
lever confirmed in SQ4(ii) exists *only* for PACE — ~8 of the ~24 years — while the RQ1 detectability
claim rests on the MODIS record. Two of the three headline numbers would therefore live on different
data than the third. Proposed resolution: state the **precision and accuracy** gains (Cphyto
32%→16%, stock spread 3.5×→<1.8×) as demonstrated in the PACE hyperspectral+UV era, then
*transferred back* onto the multispectral record through one physically consistent retrieval plus
cross-mission harmonisation, which is what yields the **detectability** number across 24 years. That
makes PACE the calibration anchor for the heritage record instead of implying UV information exists
before 2024. Awaiting JXP's confirmation that this matches intent.

**Also asked:** TQ2, team *counts* (not names) needed to write the required preliminary-budget
element — proposed ~5–7 institutions, 4–6 postdocs, 3–4 students, 1–2 research software engineers,
partial-FTE senior investigators per work package, part-time project manager; TQ3, what the single
figure should be — my lean is a two-panel (left: the report's factor-of-disagreement cascade showing
why current products cannot answer the carbon question; right: n\* falling from 43/31.5 yr to ≈20 yr
against the ~24-yr record length), both panels already generated from cited literature in
`reports/py/`.

**Decisions taken without asking** (flagged for override): drop the phrase **"expert system"** (it
came from the dropped astronomy framing and reads as 1980s rule-based AI to a non-AI panel) in favour
of a functional description; keep the three RQ4 numbers verbatim; §6 candour in two sentences;
portfolio differentiation in ~4 sentences naming COCO2/InMOS/SUBSEA/CLARiTy; personnel-dominated
budget with named lines for cloud compute/storage, the 5-day annual meeting (first Oct 2027, Europe),
open-data publication and indirect ≤10%; and no first person, names or institutions anywhere in the
body, with third-person self-citation in the separate reference PDF.

Next: await TQ1–TQ3 — after which I have everything needed to write the first draft.

### 2026-08-05 (First draft prompt 5: wrote the first draft + figure + references; created the Google docs)

Executed the 5th "First draft" task. TQ1 confirmed the PACE-era→heritage-record transfer framing;
TQ2 "leave the budget to me"; TQ3 approved the two-panel figure, generated with Python in
`proposals/Schmidt_Sciences`. **Draft written.**

**Deliverables.**
- `proposals/Schmidt_Sciences/EOI-proposal.md` — the 3-page EOI draft.
- `proposals/Schmidt_Sciences/EOI-references.md` — 30 Vancouver references (separate PDF per the
  call, so they cost no page space).
- `proposals/Schmidt_Sciences/eoi_figure.py` → `eoi_fig1.png` / `.pdf` — the single figure.
- Google docs in `.../Proposals/2026/Schmidt Sciences`: **EOI-proposal**
  (`1LIB-KY-2bIge4jGvEn9gVzW6eskvIk4LLi9V2dATGys`) and **EOI-references**
  (`14mou0etqVDccN7gOKL_33MJgEWnlhg4SVqFId5D2oxg`). Uploaded as HTML so headings, bold and the
  target table convert to native Doc formatting.

**Structure**, mapping the RFP's five required elements: §1 problem + primary theme + the two-sentence
scope candour (DQ9); §2 hypotheses H1–H3 and the three factor-2 targets with the n\* derivation;
§3 research plan WP1–WP5; figure + caption; §4 data/tools + the ~4-sentence portfolio differentiation
(DQ7) + decision relevance; §5 anonymised team and partnerships; §6 preliminary budget (categories
only, figures left to JXP per TQ2); §7 generative-AI statement.

**Content decisions carried through.** DQ1(c) coupled optics + ECCO-Darwin, opening on carbon and
uncertainty; T1 lead with T2/T3; Cphyto + POC headline with NPP/export propagated and subsurface via
BGC-Argo/lidar; RQ5's iterated-priors coupling as the core with adjoint assimilation named as a
stretch goal; SQ1(a) MODIS+PACE reprocessed with SeaWiFS as check; SQ4 OCI baseline, UV in,
polarimetry a stretch goal; TQ1's PACE-as-calibration-anchor wording; AC as the enabler of T-C rather
than a fourth number (SQ6); "expert system" dropped; RT prototype absent (DQ8); astronomy absent
(RQ3).

**Verification performed.**
- **Blind-review audit** scripted over the body: no first person, no personal names, no institutions,
  no astronomy terms, no "expert system", no prototype references — all clean. Team section written as
  roles/expertise only.
- **Reference integrity**: every citation resolves to an entry, no orphans. **All DOIs verified
  against DOI/Crossref metadata.** Two fixes found in the process: the PACE OCI DOI in
  `context/context_summary.md` is **wrong** (`10.1109/TGRS.2024.3380416` 404s; the correct one is
  **`10.1109/TGRS.2024.3383812`**, Meister et al., IEEE TGRS 62:1–18), and the `Nature` 2023
  hydrographic export paper flagged "(first author to confirm)" in `reports/biomass_summary.md` is
  **Wang W-L, Fu W, Le Moigne FAC, et al.**, Nature 624:579–585 — that flag can now be cleared.
- **Length measured, not guessed**: laid the text out with real 11-pt font metrics against a
  6.5 × 9 in text block. Current draft ≈ **3.6 pages against a 3-page limit** (~1,650 words +
  a 2.2-in figure). JXP will cut ("I am good at cutting text"); the per-section line budget is
  §3 research plan 61 lines, §1 30, §2 27, §4 25, §5 14, §6 11, §7 8.

**Two things to check.** (i) The **target table** may not have converted cleanly into the Doc — the
read-back shows signs of literal markdown in the table cells and a spurious empty header row; the
source markdown has since been given a non-empty first header (`ID`), and the table is small enough
to retype if needed. (ii) The **figure is a placeholder box** in the Doc — the API cannot embed a
local PNG, so `eoi_fig1.png` must be inserted manually where the marked paragraph sits.

Next: JXP trims to 3 pages, inserts the figure, completes the budget, and exports to PDF.

### 2026-08-06 (First draft prompt 6: split the figure in two, reworked layout, PNG only)

Executed the 6th "First draft" task. The two-panel figure became **two separate figures**, both
regenerated from `proposals/Schmidt_Sciences/eoi_figure.py`, **PNG only** (the `.pdf` output was
dropped and the old `eoi_fig1.png`/`.pdf` deleted):

- **`eoi_fig1_problem.png`** — the problem. Factor of disagreement between independent estimates
  (satellite bbp 2.5×, Cphyto stock 3.5×, NPP 2.5×, export 3.0×), with each quantity's actual range
  set *inside* its bar in white and the factor as a bold label in a right-hand margin.
- **`eoi_fig2_targets.png`** — our proposed measurements. The three headline targets as
  baseline→target dumbbells on a common axis of "remaining uncertainty as a fraction of today's
  value", so the factor-of-two claim reads at a glance: T-A 32%→≤16%, T-B 3.5×→<1.8×,
  T-C 31.5 yr→≈20 yr. A shaded band and dashed line mark the factor-2 threshold.

**The three layout instructions were treated as hard constraints, not preferences.** Text inside the
axes: the right-hand label margin in Fig. 1 is computed from the *rendered* width of the label string
(`_text_w_data`) and the x-limit set from it, rather than padded by guesswork. No overlaps: two
collisions were found and fixed by inspecting the rendered PNGs — the Fig. 2 legend sat on top of the
T-A value labels (fixed by adding headroom above the top row and making the legend two-column), and
the earlier inline "record today / by 2032" annotations collided with each other and the title (those
reference lines were dropped from the figure; the record-length point now lives in the Fig. 2
caption, which is the better place for it).

**Consequential text change.** With Fig. 2 showing all three targets, the §2 **targets table became
redundant**, so it was replaced by one sentence carrying the same numbers and citations. That removes
a duplicated block, saves page space, and sidesteps the Docs table-conversion problem flagged last
pass. Figures were also **repositioned**: Fig. 1 now sits in §1 (the problem) and Fig. 2 in §2
(the targets), rather than both landing in §3.

**Page budget.** Splitting one figure into two costs space, as expected: measured with real 11-pt
metrics and the actual PNG aspect ratios, the draft is now **≈3.84 pages** (1,726 words; text 174
lines, figures 22 lines) against 3.57 before. Both figures were widened (7.8 in) so that scaled to a
6.5-in text column they are only **1.87 in and 1.72 in tall**, recovering ~0.1 page versus the first
attempt. Blind-review audit and reference integrity re-run: clean, no undefined citations.

**The Google docs are deliberately NOT updated.** The Drive tools available here can create, copy and
read files but cannot replace the contents of an existing doc, so refreshing `EOI-proposal` would mean
creating a *new* doc — which would discard any trimming JXP has already done in the live document.
The repo markdown and HTML are current; to bring the doc in line, two edits suffice: replace the
single figure placeholder with the two PNGs, and swap the §2 table for the new one-sentence targets
paragraph.

Next: JXP trims to 3 pages, inserts the two PNGs, completes the budget, exports to PDF.

### 2026-08-06 (First draft prompt 7: added a Google Slides-sized pair from the same code)

Executed the 7th "First draft" task. `eoi_figure.py` now emits **four PNGs from one code path** —
the same two figures in two sizes:

| file | size | use |
|---|---|---|
| `eoi_fig1_problem.png` | 7.8 × 2.3 in | proposal |
| `eoi_fig2_targets.png` | 7.8 × 2.2 in | proposal |
| `eoi_fig1_problem_slides.png` | 10.0 × 4.9 in | Google Slides |
| `eoi_fig2_targets_slides.png` | 10.0 × 4.9 in | Google Slides |

**How, rather than just rescaling.** Everything that differs between print and projection is
collected in a frozen `Style` dataclass (figure size, eleven font sizes, line widths, marker sizes,
axes spine width, and the two y-limits that have to clear the reference labels), with `PAPER` and
`SLIDES` instances and `main()` looping over both. The plotting functions take a style and contain no
branches, so the two variants **cannot drift apart in content — only in size**. `PAPER` values were
transcribed unchanged from the previous version, so the proposal figures are unaffected.

Slides sizing: Google Slides widescreen is 10 × 5.625 in, so 10 × 4.9 fills the width and leaves a
title band. Fonts go from 8–10 pt to 13–17.5 pt, line widths and markers roughly double, and the axes
spine thickens. This matters because the point of the variant is legibility when projected — a paper
figure at 8 pt scaled up on a screen is still an 8 pt figure's *proportions*, with hairline rules and
cramped labels.

**One collision found and fixed at the larger size, which also improved the paper version.** At
17 pt the bold "≈20 yr" target label for T-C is wide enough to reach back past x = 0.5, so the
full-height factor-of-2 line was drawn straight through it. Rather than shorten the reference line or
move the label away from its marker, the target labels now carry an opaque white backing box, so the
line passes *behind* the text. The same defect existed less visibly in the paper figure and is now
gone there too. Both styles re-inspected as rendered PNGs, not assumed: all text inside the axes, no
overlaps, in all four files.

Lint clean (`ruff check`, `ruff format --check`) and the whole set regenerates reproducibly from an
empty directory.

Next: unchanged — JXP trims to 3 pages, inserts the two proposal PNGs, completes the budget, exports
to PDF. The `_slides` pair is for talks and needs nothing further.

### 2026-08-06 (First draft prompt 8: bold figure titles, larger axis labels)

Executed the 8th "First draft" task — two requested changes to `eoi_figure.py`, applied through the
`Style` fields so they land in all four PNGs at once:

1. **"We do not know…" is now bold** (`fontweight="bold"` on the Fig. 1 title).
2. **The "factor by which…" axis label is larger**: `fs_axlabel` 8.5 → **10.0** pt for `PAPER` and
   14.0 → **16.5** pt for `SLIDES`.

**Two deliberate over-applications, both trivially reversible if unwanted.** (i) `fs_axlabel` is a
single shared field, so Fig. 2's x-label ("remaining uncertainty…") grew by the same amount — the two
figures appear together, and mismatched axis-label sizes across a pair reads as an error. (ii) For the
same reason **Fig. 2's title was also bolded**; one bold title beside one regular title looks like an
oversight rather than emphasis. Both are one-line changes to undo (drop `fontweight` in `fig2_targets`,
or split `fs_axlabel` into per-figure fields).

**Side effect, in the helpful direction.** Bold titles and bigger axis labels widen the saved canvas
(`bbox_inches="tight"`), so scaled into the 6.5-inch text column the proposal figures got *shorter*:
Fig. 1 1.87 → **1.79 in**, Fig. 2 1.72 → **1.62 in**. Net page estimate 3.84 → **3.82 pages**. The
slides pair is now 3.99 in and 3.75 in tall at full 10-inch slide width, still leaving 1.6–1.9 in for a
slide title.

All four PNGs re-inspected as rendered images: titles bold, labels larger, every annotation inside the
axes, no overlaps. Lint clean.

Next: unchanged — trim to 3 pages, insert the two proposal PNGs, complete the budget, export to PDF.

### 2026-08-06 (First draft prompt 9: methodology figure, slides only; permission allowlist widened)

Executed the 9th "First draft" task: a third figure describing the methodology — the flow from
satellite radiance through bbp to carbon — plus the requested permissions change.

**`eoi_fig3_methodology_slides.png`** (`fig3_methodology()` in `eoi_figure.py`). Slides only, per
request; no paper variant, and `main()` says so explicitly. Content, left to right: **Satellite
radiance** (PACE/OCI 350–890 nm; MODIS since 2002) → **Atmospheric correction** (cross-mission
harmonisation) → **Bayesian inversion** (differentiable RT + learned priors) → **IOPs with
posteriors** (bbp, aph, adg; per-pixel σ) → **Carbon stocks & fluxes** (Cphyto, POC → NPP, export).
Beneath it a box for the **external information that breaks the degeneracy** (in-situ bio-optics,
BGC-Argo profiles, ECCO-Darwin state estimate), with a "priors" arrow up into the inversion and a
return arrow from carbon back into the state estimate — the iterative coupling of RQ5/WP4. A ribbon
along the bottom carries the uncertainty message. So the figure encodes WP1–WP5 without naming work
packages.

**Two implementation notes worth keeping.**
1. *The first attempt's labels overflowed their boxes.* Cause: I sized boxes in a 0–100 coordinate
   space while `tight_layout` had shrunk the axes to less than the figure width, so my
   units→inches assumption was wrong. Fixed by `fig.add_axes([0, 0, 1, 1])` and saving **without**
   `bbox_inches="tight"`, which makes one x-unit exactly 0.10 in and pins the output at 10 × 5.3 in —
   the right geometry for a widescreen slide anyway.
2. *Added `_check_fits()`*, which measures every label's rendered extent against the canvas and
   against its own box and prints overflows at generation time. Figure 3 places everything by hand,
   so a font bump or a longer string would otherwise break the layout silently. It now reports
   `ok fig3 methodology: every checked label inside its bounds` on each run.

**Environment gotcha, recorded because it cost time:** writes to files from inside `Bash` heredocs
(`python - <<EOF … p.write_text(…) … EOF`) do **not** persist here, even with absolute paths and the
correct CWD — the script reports success and the file is unchanged. All file edits must go through the
Edit/Write tools. `conda run -n ocean14 …` does preserve the working directory, so that was not the
cause.

**Permissions (the "don't stop to ask me" request).** Scanned the 20 most recent transcripts across all
projects (3,822 tool calls). Most of the high-frequency read-only commands — `grep`, `head`, `tail`,
`ls`, `echo`, `sed`, `wc`, `find`, `sort`, and all read-only `git` subcommands — are auto-allowed by
Claude Code and never needed rules, and the interpreters (`python:*`, `python3:*`,
`conda run -n ocean14:*`, `pytest:*`) were **already** in `.claude/settings.json`. Added the genuinely
missing read-only entries: `Bash(ruff check:*)` (96 uses), `Bash(ruff format --check:*)` (63),
`Bash(git ls-tree:*)`, `Bash(git ls-files:*)`, `Bash(git rev-parse:*)`, `Bash(conda info:*)`,
`WebFetch(domain:doi.org)`, `WebFetch(domain:api.crossref.org)`, and the four read-only Google Drive
MCP tools (`get_file_metadata`, `list_recent_files`, `download_file_content`,
`get_file_permissions`). `permissions.deny` and `permissions.ask` were left untouched. Deliberately
**not** added: `Bash(ruff format:*)` (mutates files) and `Bash(curl:*)` (89 uses, but curl can POST/PUT
— worth adding only if JXP wants it). Also switched my own habit to the already-allowlisted
`conda run -n ocean14 python …` form instead of `source conda.sh && conda activate … && python …`,
which never matched the existing prefix rules and was itself a source of prompts.

Next: unchanged — trim the EOI to 3 pages, insert the two proposal PNGs, complete the budget, export to
PDF. Figure 3 is talk material and needs nothing further.

### 2026-08-06 (First draft prompt 10: graphics version of the methodology figure)

Executed the 10th "First draft" task. Added **`eoi_fig3_methodology_graphics_slides.png`** via a new
`fig3_methodology_graphics()`; the plain-box `eoi_fig3_methodology_slides.png` is **kept** as asked, and
both are emitted from `main()`. Slides only, 10 × 5.3 in.

**No images were downloaded, and none are needed.** Each of the five boxes is replaced by a graphic
*drawn* in matplotlib as its own small function on an inset axes: `g_satellite` (spacecraft with solar
panels over an ocean limb, star field), `g_atmosphere` (sun, graded sky, aerosol scatterers, sea band),
`g_inversion` (a family of candidate spectra collapsing onto a shaded credible band — the degeneracy and
its resolution in one glyph), `g_iops` (bbp power-law decline and the twin-peaked aph curve, each with a
posterior band), and `g_ocean_carbon` (graded water column, phytoplankton in the lit layer, sinking
aggregates with flux arrows). Chosen over stock photography deliberately: nothing to fetch, no licence to
clear, no binaries in the repo, and it regenerates with the rest. Each glyph is one function, so
photography can be dropped in later per station.

**Three defects caught by looking at the render, not by assuming.**
1. **The satellite arrows pointed the wrong way** — both drawn as emission from the spacecraft. A sensor
   *receives*; corrected to sunlight down on the left and water-leaving radiance up to the sensor on the
   right. Worth fixing precisely because a physics audience reads figures literally.
2. **Station labels were crowded** at one line each ("IOPs with posteriors" nearly touching "Carbon
   stocks & fluxes"), so all five were wrapped to two lines, with the detail text and the priors/feedback
   arrows moved down to match.
3. **The aph label sat inside the bbp uncertainty band**; moved into the trough between aph's two peaks,
   clear of it.

`_check_fits` reports `ok fig3 methodology (graphics)` on every run. Lint and format clean; the full set
of six PNGs regenerates from an empty directory.

**If photographic imagery is wanted later** (not required — the figure is complete as it stands), the
natural swaps are: a PACE/OCI true-colour granule or the NASA PACE spacecraft render for station 1
(both public domain from NASA), and a phytoplankton bloom image for station 5. Drop the file in
`proposals/Schmidt_Sciences/` and the corresponding `g_*` function becomes a two-line `imshow`.

Next: unchanged — trim the EOI to 3 pages, insert the two proposal PNGs, complete the budget, export to
PDF.

### 2026-08-08 (Export prompt 1: EXPORTS deep dive; posed EQ1–EQ6 — no report written)

Executed the 1st "Export" task: a deep dive into NASA's EXPORTS program (what it was funded to do,
what it accomplished). Per the prompt, **no report was written** — findings are digested in the
Export Q&A section with six questions (EQ1–EQ6), and the papers worth downloading are listed there
(EQ5).

**Method.** Three parallel research agents, each working from primary sources with web
search + full-text fetches: (i) program design and funding (the 2015 Science Plan PDF, Siegel et
al. 2016 *Front. Mar. Sci.*, the Oct 2016 Implementation Plan, ROSES selection documents, NASA/OCB
announcements); (ii) the two field campaigns and their results (full text of Siegel et al. 2021
*Elementa* and the Siegel et al. 2023 *Ann. Rev. Mar. Sci.* synthesis via NTRS, plus the campaign
papers); (iii) accomplishments, data legacy and the remote-sensing deliverable (SeaBASS archive
record, publications list, retrospectives). Claims were kept only when tied to a URL/DOI; extracted
full texts of Siegel 2021/2023 and the Science Plan are in the session scratchpad.

**What it was funded to do.** Goal: "a predictive understanding of the export and fate of global
ocean NPP"; central hypothesis that export/fate are predictable from surface-ecosystem properties
**assessable from satellite ocean color**, with next-generation algorithms for PACE as explicit
funded rationale in every planning document. SQ1 pathways / SQ2 subsurface fate / SQ3 reduce
uncertainty in export estimates via remote sensing. Two deployments bracketing the export-regime
space; all three export pathways quantified simultaneously; 18 science-team projects (13 NASA +
5 NSF), >100 scientists. Proposed Goal Plan $71.5M/7 yr; as-executed cost only single-sourced at
"$40M" (WHOI OTZ site) — no NASA-official figure exists, flagged to JXP in EQ4.

**What it accomplished.** Station P 2018 (low end-member: Ez-ratio 0.10 ± 0.06; 100-m flux
1.38 ± 0.77 mmol C m⁻² d⁻¹; ×3 trap-vs-²³⁴Th gap; salps raising export efficiency 1.5–2.6×) and
PAP 2021 (high end-member: eddy-following bloom demise; flux 11→14 mmol C m⁻² d⁻¹, ~10× the
Pacific; efficiency 10%→30%) — the designed contrast was achieved. Global synthesis: export
~10.2 Pg C yr⁻¹, gravitational ~70% / mixing ~20% / migrant ~10%, ~1,300 Pg C sequestered
(Nowicki 2022, Siegel 2023). Open archive: 11,031 SeaBASS granules, >400 parameters. ~105 papers,
~6,900 citations.

**The finding that matters for the EOI.** The SQ3 remote-sensing promise was not delivered: no
mechanistic Rrs→export algorithm exists; the satellite-facing product is a satellite-driven
diagnostic inverse model; the 4–10 Pg C yr⁻¹ cross-method spread stands in Siegel 2023's own
Table 1; and EXPORTS' own optics work found bbp instrument calibrations mutually inconsistent at
95% confidence (Erickson 2022). Siegel 2023 closes by calling for a future operational
satellite+in-situ+model system — nearly verbatim our proposed architecture. EQ2 asks how candidly
to state this (my lean: entirely through the program's own words); EQ6 asks whether it feeds back
into `biomass_summary.md` §2.5 and the EOI differentiation.

**Corrections to prior context.** Our report's ref [24] (Siegel 2016 science plan) is verified.
Briggs et al. 2020 (*Science*, fragmentation 49 ± 22%) is **not** an EXPORTS product — BGC-Argo
floats, N Atlantic + Southern Ocean — though it motivates the EXPORTS-NA fragmentation work; noted
so we do not miscredit it. The NA overview is Johnson et al. 2024 *Prog. Oceanogr.* 220:103170
(not *Elementa*), with an open EarthArXiv preprint.

**Unverified, flagged in the Q&A:** the as-executed program cost; the complete *Elementa*
special-collection contents (UC Press 403s automated access); a NA-specific pathway-partitioning
synthesis (none found — likely does not exist yet); the Clevenger 2024 DOI string (volume/pages
verified via ADS, DOI inferred).

Next: await EQ1–EQ6, then write the EXPORTS report.

### 2026-08-08 (Export prompt 2: read the EQ answers, posed a short Round-2 — no report yet)

Executed the 2nd "Export" task: read JXP's EQ1–EQ6 answers and posed **FQ1–FQ2**. The report is
not yet requested, so nothing was written to `reports/`.

**Settled by the EQ answers.** EQ1(a) VICC-input framing with a fenced mapping, biomass-report
structure; EQ2 no editorializing, critique through the program's own words — **but with a new
mandate to discuss flaws EXPORTS did not report on itself**; EQ3 the proposed six-section shape,
biomass-report length, at `reports/exports_summary.md`; EQ4 no insider cost figure — report
"order $40M (reported)" vs the proposed $71.5M Goal Plan, with the sourcing caveat; EQ5 papers
delivered; EQ6 update `biomass_summary.md` §2.5 only (JXP handles the EOI himself).

**Verification done this pass.** The three requested PDFs are in `context/papers/EXPORTS`
(`briggs2020.pdf`, `clevenger2024.pdf`, `traylor2025.pdf`) — all present, all text-extractable
via pypdf. The two DOI strings I had flagged as inferred are now **confirmed from the PDFs
themselves**: Clevenger 2024 `10.1016/j.marchem.2023.104346` (Mar. Chem. 258:104346 on the
masthead) and Traylor 2025 `10.1002/lno.70002` (L&O 70:S165–S178 — note it landed in a
supplement issue).

**Why a Round 2 at all.** EQ2's addendum ("if you identify flaws or shortcomings that they did
not report on, do discuss them") changes the report from summary to critique, and the flaws I
actually found that EXPORTS has *not* discussed in print are **programmatic, not scientific** —
the measurement-level warts (trap-vs-²³⁴Th ×3, bbp calibration inconsistency, broken mesopelagic
steady state, salp stochasticity) were all self-reported by the program. FQ1 therefore asks
whether the critique may extend to program-level design: (a) the descope from the 4-deployment /
two-visits-per-basin Goal Plan to one snapshot per basin, never reconciled in print with the
design's own ecosystem-state-space logic; (b) no Southern Ocean deployment (the unexecuted
ICESOCC concept), leaving the global partitioning extrapolated into a dominant export regime;
(c) the structural PACE timing mismatch (field phase ended May 2021, PACE launched Feb 2024 —
zero coincident data, so the founding "algorithms for PACE" promise could not be closed within
the program). My lean: include all three in a labelled program-level subsection, stated
factually; (c) doubles as our opening — the EXPORTS archive is the pre-launch training set and
we close the loop with actual PACE data. FQ2 asks about figures (proposed
`reports/py/exports_figs.py` with 2–3: campaign contrast, global pathway partitioning +
cross-method spread, and optionally the promise-vs-delivery timeline).

**Decisions taken without asking** (flagged for override in the Q&A): inherit the biomass
report's conventions (Vancouver + verified DOIs, uncertainty-conventions box, 95% CIs where
derivable); perform the §2.5 update in the same pass as the report; cite preprints alongside
journal DOIs where finals are paywalled, per EQ5.

Next: await FQ1–FQ2, then write `reports/exports_summary.md` and the §2.5 update.

### 2026-08-08 (Export prompt 3: wrote reports/exports_summary.md + 3 figures; updated biomass §2.5)

Executed the 3rd "Export" task. FQ1 (all three program-level critiques in) and FQ2 (all three
figures) both confirmed. **Report written.**

**Deliverables.**
- `reports/exports_summary.md` — ~5,200 words, six sections per EQ3: §1 what was funded and why
  (the PACE rationale quoted from the 2015/2016 plans; Goal Plan $71.5M vs the single-sourced
  "$40M" as-executed figure, caveated per EQ4); §2 the two campaigns with headline numbers and
  the method-level lesson; §3 the global synthesis and whether the numbers moved; §4 the
  remote-sensing deliverable, honestly assessed — §4.3 the gaps the program reported itself,
  §4.4 the three unreported program-level shortcomings (descope, no Southern Ocean, PACE timing);
  §5 data legacy, publication metrics, the untraceable synthesis phase, and the unplanned
  mCDR-verification legacy; §6 the fenced VICC mapping (five steps + cautions).
- `reports/py/exports_figs.py` → three PNGs in `reports/figs/`: `exports_fig1_campaign_contrast`
  (the ×10 site contrast achieved / same-site methods disagree ×2–3),
  `exports_fig2_pathways` (Nowicki partitioning 72/19/10% beside the still-open ~4–11 Gt C yr⁻¹
  cross-method ladder), `exports_fig3_timeline` (promise-vs-delivery; the 33-month field→PACE gap).
- `reports/biomass_summary.md` §2.5 updated per EQ6(i): a new EXPORTS-era bullet (10.2 Gt C yr⁻¹
  central, 72/19/10% partitioning, ~1,300 Pg C sequestration; envelope unchanged), a pointer in
  §2.5.1, and ref 30's "(first author to confirm)" flag cleared (Wang W-L, Fu W, Le Moigne FAC —
  established during the EOI-references verification pass).

**Sources actually read.** Full text: Siegel 2021 overview, Siegel 2023 review (NTRS), the 2015
Science Plan, and the three JXP-supplied PDFs (Clevenger 2024, Traylor 2025, Briggs 2020 — key
numbers confirmed against the PDFs themselves, e.g. Traylor's site contrast GPP ×5 / NCP ×13 /
fCorg 3.4 ± 0.7 vs 20.3 ± 2.3 mmol C m⁻² d⁻¹, ez-ratios 0.24 vs 0.69). Johnson 2024 via the
EarthArXiv preprint per EQ5.

**Verification performed.**
- **All 40 reference DOIs verified against Crossref metadata** (author/year/title, plus
  volume/pages for entries stating them) in two batch API passes; the two title-search stragglers
  resolved Saba 2021 (10.1002/lno.11709) and failed to resolve the Graff Elementa 11:00112 POC
  reconciliation paper, which is therefore **not cited**.
- **Citation integrity scripted**: every [n] in the body resolves, no orphaned references (40/40).
- **Figures inspected as rendered PNGs, not assumed.** Three defects found and fixed by looking:
  (i) fig2's estimate labels collided with their error bars (moved to y-tick labels); (ii) fig3's
  gap annotation ran through three event labels (given its own band; labels re-laid at three
  depths); (iii) a `"PACE" in txt` marker-nudge condition also matched the Science Plan label
  (which quotes "…PACE mission"), drawing a spurious diagonal across the plot — the classic
  substring-match bug, fixed with `startswith`.
- **Prose re-diffed against the figure values** (the recurring-defect check): 72/19/10%,
  7.3/1.9/1.0, the 4.0–10.64 ladder, 1.38 ± 0.77 / 3.4 ± 0.7 / 11→14 / 20.3 ± 2.3, and the
  33-month gap all match between report text, captions, and `exports_figs.py`.
- `reports/py/` is not ruff-clean by existing convention (biomass_figs.py fails identically), so
  exports_figs.py matches its sibling's style rather than diverging.

**Honest-critique balance, per EQ2/FQ1.** §4.3 carries only self-reported gaps (Erickson 2022's
bbp calibrations mutually inconsistent at 95%; Siegel 2023's own "operational system" concession;
the Eos sequel), each through the program's words. §4.4 carries the three unreported structural
observations, stated factually with the Implementation Plan's own numbers and an explicit "none
diminishes the science that was done." Briggs 2020 is explicitly marked as NOT an EXPORTS product.
One inference is labelled as such: the global inverse model is constrained by satellite
NPP/climatologies, not the 2018/2021 field data, so the field→global-number loop is indirect.

Next: the EXPORTS report is done; JXP reviews. The EOI draft remains with JXP (EQ6ii).

### 2026-08-09 (GO-SHIP prompt 1: explored the program, wrote context/go_ship_summary.md)

Prompt 21 (1st Task under "GO-SHIP"): explore https://www.go-ship.org/ and answer four
questions (what is GO-SHIP; extent of backscattering; radiometry; Cp), reporting to
`context/go_ship_summary.md`. Model: Fable.

**First finding, before any science: `www.go-ship.org` is dead.** The domain currently
serves a DreamHost parking page and subsidiary URLs (`HydroMan.html`, `DatReq.html`)
404; web.archive.org is not reachable from this environment. The report is therefore
built from substitutes and says so up front: `usgoship.ucsd.edu` (live), Sloyan et al.
2019 (*Front. Mar. Sci.* 6:445, the program paper), the GOOS network page, the SCOR
WG 154 report on plankton/bio-optics for GO-SHIP (Boss et al., misclab PDF — read as
extracted pages, not summarized blind), Clayton et al. 2022 (Bio-GO-SHIP,
*Front. Mar. Sci.* 9:767443), NASA OB.DAAC/SeaBASS holdings, and the
Gardner–Mishonov–Richardson transmissometer literature.

**Answers, compressed.** (i) GO-SHIP = the WOCE/CLIVAR-heritage network of ~55
sustained decadal-repeat full-depth hydrographic reference lines, ~12 nations, Level
1/2/3 measurement hierarchy, data at CCHDO; the only climate-quality full-water-column
element of GOOS. (ii) **Backscattering: historically absent** (never Level 1/2);
arriving now via two young channels — rosette FLBB (bbp 700 nm, single angle) on US
cruises since ~2021 explicitly to validate GO-BGC/BGC-Argo floats, and Bio-GO-SHIP
underway flow-through bbp/cp archived in SeaBASS from six occupations (A16N 2023 ×2,
I05 2023, I08S 2024, I09N, P02; PIs Graff/Mitchell/Poulton). No long, multi-spectral,
uncertainty-characterized bbp record exists. (iii) **Radiometry: effectively none** —
above-water PAR from ship met sensors only; no in-water Ed/Lu, no Rrs, nothing
radiometric in the SeaBASS Bio-GO-SHIP holdings; shipborne hyperspectral radiometry
lives outside GO-SHIP (AMT FRM radiometry; SO-PACE pySAS on ships of opportunity).
(iv) **Cp is the crown jewel**: C-Star-class 660 nm transmissometers on the CTD at
essentially every station of US-occupied lines since WOCE (~4 decades), consolidated
in the Gardner/Mishonov/Richardson Global Transmissometer Database V3 (order-10⁴
full-depth profiles, >70 cruises, cp in ~85% of casts), supporting decadal
repeat-transect particle comparisons, nepheloid-layer maps, and POC climatologies —
with honest caveats (single λ, calibration drift, regionally varying cp→POC).

**Report shape**: four Q&A sections + a brief §5 mapping to the proposal (cp archive
as full-depth POC-proxy constraint; GO-SHIP as the BGC-Argo bbp traceability chain;
the radiometry gap as a cheap enhancement opportunity; Bio-GO-SHIP as the
institutional on-ramp), 15 Vancouver references.

**Verification**: all six from-memory DOIs checked against Crossref; one was wrong —
the pySAS *Oceanography* paper is 10.5670/oceanog.2022.**210**, not .203 (the .203 I
first wrote resolves to an unrelated civic-activism paper) — fixed before delivery.
Two dead ends hit and reported rather than papered over: the JTECH-D-23-0161 bbp
paper surfaced by search turned out to be about SVP drifters (not Bio-GO-SHIP), and
the ESSD 2025 IOP compilation is AMT, not GO-SHIP — the latter kept only as the
contrast citation for what radiometry+IOPs on a repeat line looks like.

Next: JXP reviews `context/go_ship_summary.md`.

### 2026-08-09 (GO-SHIP prompt 2: Bio-GO-SHIP deep dive; expanded reports/go_ship_summary.md)

Prompt 22 (2nd Task under "GO-SHIP"): read more about Bio-GO-SHIP and add to the report,
now living at `reports/go_ship_summary.md` (move confirmed on disk — `context/` copy gone,
`reports/` copy present and intact). Model: Fable.

**Sources read this pass**: the live program site `biogoship.org` (overview, cruise
plans 404s aside), Clayton et al. 2022 (*Front. Mar. Sci.* 9:767443) re-read for the
measurement suite, sampling modes, roadmap, cost posture, and PACE framing; Larkin et
al. 2021 (*Sci. Data* 8:107) for the pilot-era omics record; Tanioka et al. 2022
(*Sci. Data* 9:688, GO-POPCORN) for the particle-stoichiometry dataset; Gerace et al.
2025 (*PNAS* 122:e2411835122) as the trend-detection exemplar from GO-SHIP data.

**What was added** — a new §5 "Bio-GO-SHIP in depth" (relevance section renumbered §6):
- **History in two phases.** Opportunistic pilot era 2011–2020: 971 surface metagenomes
  from 932 stations on eight campaigns (I09N 2016, P18 2016–17, I07N 2018, AMT-28 2018,
  C13.5 2020 + three pre-GO-SHIP cruises), sampled from the underway line every 4–6 h at
  ~26.5 km median spacing; plus GO-POPCORN (~2,600 paired surface POC/PON/POP, 70°S–73°N).
  Formal implementation 2022–: I05/P02/A13.5 designated the first fully sampled sections;
  completed occupations now include P02 2022, Arctic (*Healy*, Nov 2022), A16N 2023,
  I05 2023, I08S 2024, I09N — optics/pigments from six already in SeaBASS.
- **Measurement suite split by mode**: underway flow-through (hyperspectral AC-S a/c,
  bb3 multi-λ backscatter, FLBB, FRRF, IFCB, flow cytometry, discrete filtration for
  HPLC/POC/omics) vs CTD/discrete (omics, C:N:P, eDNA, UVP, bio-acoustics); claimed cost
  <~10% of GO-SHIP ops. Funders: NOPP pilot (NOAA+NASA), now NSF/NOAA/NASA/Minderoo.
- **The PACE framing is Clayton et al.'s own** — the underway IOP+pigment data as the
  algorithm testbed — and the report keeps the §3 caveat sharp: still no radiometry;
  Bio-GO-SHIP constrains water-side IOPs/constituents, not Rrs.
- **§6 relevance gains item 5**: the coincident AC-S+bbp+HPLC+POC/N/P data as a
  conversion-factor resource (Cphyto↔POC↔bbp/cp and its regional stoichiometric
  variability) — mapping onto the conversion-uncertainty terms the biomass report ranks
  as dominant.
- Header/source-note updated; §1's Bio-GO-SHIP paragraph now points to §5 (its wrong
  "formal pilot 2022–" phrasing fixed — the pilot era began ~2011 opportunistically).

**Verification**: the three new DOIs checked against Crossref before citing, and the
volume/article numbers (8:107, 9:688, 122:e2411835122) confirmed rather than trusted
from memory; scripted citation-integrity check passes — [1]–[19] all cited and all
defined, no orphans (one false positive from a year starting a line in ref 7).

Next: JXP reviews the expanded `reports/go_ship_summary.md`.

### 2026-08-15 (Bits-and-pieces prompt 1: project name — UNRAVEL)

Executed the 1st "Bits and pieces" task: a project name using UNRAVEL. Answer written under
the task in this file.

**Recommendation:** **UNRAVEL — UNified Retrieval and Assimilation for Verifiable Estimates
of ocean Living carbon**, with the slide tagline "Unravelling the ocean's carbon
uncertainty." Chosen so every word maps to a pillar of the EOI as drafted: UNified Retrieval
(one physically consistent inversion across MODIS + PACE — the 27→43 yr discontinuity
argument), Assimilation (ECCO-Darwin coupling), Verifiable Estimates (calibrated per-pixel
UQ + the factor-2 targets, with MRV resonance), ocean Living carbon (the Cphyto + POC
headline). The verb itself is the argument — unravelling = disentangling the retrieval
degeneracy, biology from physics, trend from variability — so the name survives even for
readers who never see the expansion.

Two alternates recorded with the answer: an uncertainty-led EXPORTS-style scattered-letter
expansion, and a "carbon ledger" variant that speaks budget language.

**Collision check performed** (web search): no existing ocean/climate/carbon research
program named UNRAVEL. Nearest neighbor is "Unravel Carbon," a corporate carbon-accounting
SaaS company — flagged in the answer since our topic is carbon, judged non-blocking
(different styling, different sphere). An UNRAVEL consortium exists in dementia research,
irrelevant.

Next: JXP picks the expansion (or the lead stands); the name can then propagate to the EOI
draft, figure titles, and eventually the repo/product naming.

### 2026-08-15 (Bits-and-pieces prompt 2: "Unravel" as a plain word — the subtitle)

Executed the 2nd "Bits and pieces" task: with "Unravel" used as-is (no acronym), recommend
the "what we are doing" half of "Unravel — …". Answer written under the task in this file.

**The structural point, which drives the recommendation:** *unravel* is a verb, so the best
subtitles are its grammatical object — "Unravel — X" then reads as the imperative sentence
"Unravel X," a move unavailable to any acronym. Gerund subtitles ("disentangling…") merely
restate the verb and were advised against.

**Recommendation:** **"Unravel — the ocean's living carbon."** Four words, reads as a
mission statement, names the Cphyto + POC headline ("living carbon"), leaves method and
uncertainty to prose. Three runners-up recorded with distinct emphases: (a) "weighing the
ocean's living carbon from space" (concrete census verb + satellite lane), (b) "the ocean's
carbon, with uncertainty you can trust" (VICC's keyword first), (c) "from ocean color to
carbon accounting" (value-chain form, budget language). Also flagged what to avoid:
verb-echo subtitles and retired jargon ("expert system").

No new research needed; the collision check from prompt 23 (Unravel Carbon, corporate SaaS)
carries over unchanged to the plain-word form — if anything the plain "Unravel" sits closer
to that company's name than the all-caps acronym did, worth one glance before it goes on a
public-facing page.

Next: JXP picks acronym vs plain-word form and a subtitle; then the name propagates to the
EOI draft and slides.
### 2026-08-19 (EOI continues prompt 1: read Robert's + Hervé's comments and the EOI doc; posed Round-7 CQ1–CQ7 — no edits made)

Prompt 25 (1st Task under "EOI continues"): read the collaborator comments, the PDFs Hervé
references, and the "VICC EOI" Google doc; prepare to edit but do NOT edit; ask questions in
the Q&A section. Model: Fable. **Nothing was edited** — not the EOI doc (my Drive access is
read-only anyway) and not any local proposal file; the only writes are the Q&A round and
this log entry.

**What was read.**
- `comments/robert_2026aug18.txt`: six numbered comments, five of which are drafted
  paragraphs, not notes — (1) objectives statement + the do-not-claim-budget-closure scope
  rule; (2) §II rewrite adding gliders and CALIOP/CALIGOLA lidar; (3) §II closing
  "unified framework" paragraph; (4) targeted-field-program paragraph ("rather than a new
  global survey"); (5) multi-mission Cphyto retrieval via common Bayesian inversion +
  sensor-specific observation operators (PACE degraded to heritage bands); (6) his own WP
  statement — a climate-quality Rrs record spanning SeaWiFS/MODIS/MERIS/VIIRS/OLCI/SGLI/PACE
  — plus budget (~$200K/yr + IDC: RF 1 mo, J. Tan 4 mo, S. Liu 3 mo, $5K/yr cloud).
- `comments/herve_2026aug16.txt` + the two context PDFs: his **confidential** concept note
  (Claustre/Vellucci/Mangin — FRM-grade hyperspectral Ed/Lu + bbp + FChla BGC-Argo sub-fleet
  for distributed ocean-colour validation; REFINE prototype deployed; CNES funding payload
  calibration; overpass-timed surfacing protocols; MRV/BBNJ co-benefits; expressly not for
  consortium circulation) and his 20-slide NASEM talk (core-vs-innovation-layer governance;
  44-float UVP pilot; float transmissometers as ~1000 m export proxies across 13 regions;
  N=519 float Rrs spectra validated against Sentinel-3A/B at |Δ|≈23%; targeted sub-fleets +
  OSSE-guided design as the scaling model).
- The **VICC EOI** Google doc (read with comments): now titled "Unravel: untangling ocean
  color with data-constrained models"; runs I → III and stops at an empty "Biogeochemical
  Model Emulation and Optimization" heading; §III centerpiece is **D-LITE** (a lightened,
  adjointable Darwin inside ECCO via MITgcm/Tapenade, hyperspectral radiance assimilation
  through Darwin's own RT, regional California Current ↔ global two-scale design). Six open
  margin comments: Mazloff ×4 (justify the bio-carbon focus in the opening paragraphs;
  "the interior is needed. This is triggering."; don't call BGC-Argo "too young"; Argo
  constrains pCO₂/DIC via T/S/O₂/NO₃/pH, "more than optics"), Carroll (consider building
  off Matt's B-GOSE), Dutkiewicz (DIC is known better than factor 3).

**The synthesis that shaped the questions:** four independent comments (Robert #1, Mazloff
×2, Dutkiewicz) converge on one fix — an early our-lane paragraph claiming the biological
carbon stocks/fluxes, scoping the factor-3 indictment to optically-constrained biological
quantities, and disclaiming budget closure. That is DQ9's already-approved "scoping
precision" move, so it needs no new permission; the CQs cover only what does.

**Round-7 questions posed (CQ1–CQ7):** adopt Robert's five text blocks nearly verbatim?
(CQ1); his full-constellation Rrs record vs the SQ1 MODIS+PACE compute bound (CQ2);
blind-review handling — his named first-person statement to the team form, anonymized in
the body, name-bearing citations to the separate reference PDF per RQ3(a) (CQ3); how much
of Hervé's confidential note is usable and whether float hardware enters scope — lean:
ideas-not-text, sub-fleet as leveraged/co-funded capability, no VICC-funded floats (CQ4);
§III governance — D-LITE vs B-GOSE, and folding Matt's more-than-optics + interior points
into §II (CQ5); do the RQ4 factor-2 trio and the n* detectability argument get reinstated
as the objectives' success criteria (CQ6); delivery mechanics for the eventual edit given
read-only Drive access — full revised text as a paste-ready local doc with TOC/outline —
plus a light flag that the working subtitle synonym-echoes the verb (CQ7).

Next: JXP answers CQ1–CQ7; then (prompt 26, presumably) I produce the revised EOI text.

### 2026-08-19 (EOI continues prompt 2: CQ answers reviewed; Google-doc write access solved via existing rclone remote)

Prompt 26 (2nd Task under "EOI continues"): review the CQ1–CQ7 answers and resolve the
write issue for the "VICC EOI" Google doc. Model: Fable. The EOI doc itself remains
untouched — all write testing was done on scratch files that were deleted afterwards.

**Answers digested and locked** (recorded in full in the Round-7 resolution note above):
adopt Robert's five text blocks (CQ1); full-constellation Rrs WP with MODIS+PACE spine
(CQ2a); strict body anonymity (CQ3); Hervé's sub-fleet as leveraged/co-funded, ideas-only
(CQ4a); **§III passes through byte-for-byte** per "Do not touch Section III" (CQ5) — the
D-LITE/B-GOSE margin thread stays with the team; the factor-2 trio + n* detectability
return as objectives success criteria (CQ6).

**The write issue dissolved on inspection (CQ7).** JXP offered to set up access via
rclone — it turned out the machine already has a `GDrive:` rclone remote with full
`drive` scope that reaches `Oceanography/Proposals/2026/Schmidt Sciences`. Rather than
request anything, I verified the pipeline end-to-end on scratch files:
- Upload + delete round-trip of a plain text file in the target folder: OK.
- First HTML-import attempt failed with "can't convert .html to a document with a
  different export filetype (.docx)" — rclone requires the import extension to match the
  session's export format. Fix: pass BOTH `--drive-import-formats html` and
  `--drive-export-formats html`.
- With matching flags: HTML upload → native Google Doc; same-name re-upload → **in-place
  update with the file ID preserved** (verified identical across two uploads, content
  round-trip confirmed via `rclone cat`); scratch folder purged.
- The real "VICC EOI" exports cleanly to HTML (120 KB) — saved to scratchpad as the
  base for the coming edit.

**Plan of record for the edit** (next prompt): markdown master under
`proposals/Schmidt_Sciences/` → HTML → `rclone copyto` in place, preceded by a
timestamped backup copy in the same Drive folder; documented caveat that a full-content
update orphans the six open margin comments (they remain listed, anchors lost), with the
"VICC EOI v2" side-doc offered as the non-destructive alternative.

Next: on JXP's word, produce the revised EOI (Robert's blocks + our-lane opening +
objectives with the factor-2 trio + anonymization pass + TOC/outline), §III untouched.

### 2026-08-19 (EOI continues prompt 3: edited the "VICC EOI" Google doc in place, per the locked Round-7 decisions)

Prompt 27 (3rd Task under "EOI continues"): proceed to edit the EOI doc. Model: Fable.
**The live doc was updated in place — same file ID, so the team's link is unchanged** —
after a full safety chain: fresh export → surgical edit → round-trip fidelity test on a
scratch doc → native backup copy → upload → verification.

**Safety chain, in order.**
1. Fresh HTML export of "VICC EOI" (doc had drifted trivially since the prompt-25 read;
   content re-checked against the margin comments before editing).
2. Surgical HTML edit preserving Google's own markup (classes, image, footnote):
   §III passed through untouched; comment anchors stripped everywhere (mechanical
   necessity of the import pipeline — the threads live on in the backup).
3. Round-trip test: uploaded the edited HTML to a scratch Google Doc, re-exported, and
   proved text fidelity (edited == imported, §III text char-identical to the original
   modulo my checker's footnote artifact), image survival (1/1), footnote survival.
   Scratch doc purged.
4. Native backup via Drive copy: "VICC EOI — backup 2026-08-19 (pre-Claude-edit)" in the
   same folder — preserves the six margin comment threads verbatim.
5. In-place upload (`rclone copyto … --drive-import-formats html`); one transient
   permission-classifier block on the first attempt, succeeded on retry as the error
   itself suggested.
6. Verification on the live doc via the Drive connector: file ID unchanged
   (1QISTR1NZ…), §III verbatim, all new content present, footnote intact as a linked
   bookmark. (A final rclone re-export for an image count was blocked by the
   permission classifier; image survival rests on the identical-bytes round-trip test
   in step 3, which showed 1/1.)

**What changed in the doc, per the locked decisions.**
- **Title block**: a bracketed one-line outline scaffold (I–VII), marked remove-before-
  submission — Robert's TOC/complete-outline ask without costing page space later.
- **§I**: two new paragraphs — the our-lane paragraph (bold-led "This proposal
  deliberately targets the biological component…", carrying DQ9's scoping-precision
  candour, Robert's #1 objective sentence, and the explicit not-the-budget disclaimer)
  and "Success is measurable." with the RQ4 factor-2 trio + n\* (32%→16%; 3.5×→<1.8×;
  >30 yr→~20 yr) framed per TQ1 (PACE-era gains transferred to the heritage record).
  Margin-comment fixes: "too young"/"sole purview" → "record remains far shorter than
  the multi-decadal baseline… that long baseline exists only in the satellite
  ocean-color record" (Mazloff); factor-3 scoped to "optically-constrained biological"
  quantities with the DIC concession folded in as sharpening the problem (Dutkiewicz);
  {NEED ONE SENTENCE} filled with the common-ground definition (forward-modeling each
  observation in its own measured quantity through observation operators and
  sensor-specific error models). Typos: toolss, "yet to any estimate", Wardell→Werdell.
- **§II**: "[mention Lidar somewhere?]" and "[true??]" resolved by adopting Robert's #2
  paragraph (gliders + CALIOP/CALIGOLA lidar); "Our team will xxxx" replaced by
  Robert's #3 unified-framework paragraph; "related"→"relate".
- **§III**: byte-for-byte untouched (CQ5), including the empty "Biogeochemical Model
  Emulation and Optimization" stub; D-LITE/B-GOSE margin thread left to the team (in
  the backup).
- **New §IV "A climate-quality observational foundation, anchored in situ"**, four
  subsections: anonymized Robert #6 (the SeaWiFS→…→PACE OCI Rrs record, per CQ2a/CQ3 —
  no names in the body; his named statement stays out of the doc for the team form);
  Robert #5 verbatim (multi-mission Cphyto via sensor-specific observation operators);
  the Hervé-ideas paragraph written in my own words per CQ4a (distributed
  fiducial-grade float validation layer, co-funded, overpass-coordinated, per-profile
  uncertainty budgets, funds analysis + OSSE design "— not the float hardware —"; his
  confidential note neither quoted nor cited); Robert #4 verbatim (targeted field
  observations, "rather than a new global survey").
- **§V–VII outline stubs** (team/partnerships, timeline/budget, portfolio/openness/AI),
  each with a bracketed note encoding the standing constraints (blind-review rule,
  indirect ≤10%, COCO2/InMOS/SUBSEA/CLARiTy differentiation, AI disclosure).

**Known side effect, as agreed in Round 7:** the in-place update orphaned the six open
margin comments on the live doc; they are preserved intact on the backup copy.

**Repo artifact:** the exact uploaded HTML archived at
`proposals/Schmidt_Sciences/eoi_edit_2026-08-19.html` (JXP handles git).

Next: JXP + team review the revised doc; §III's ending, the figure(s), and §V–VII
content remain team-owned. The [XX]/[Bisson+20XX] citation placeholders and the
reference PDF are natural next prompts.

### 2026-08-19 (Added Robert's work package to the "Work Packages" Google Sheet, in place)

Ad-hoc request (not a numbered prompt): add Robert's WP to the "Work Packages" sheet on
Drive — edit, do not replace; use Opus. Since a running session cannot switch its own
model (this one is Fable), the edit itself was delegated to an **Opus subagent** via the
Agent tool with a precise brief; Fable did the recon and the independent verification.

**What was added** — five rows (years 1–5), appended after the last data row, matching
the sheet's existing column conventions (numeric Year, numeric Cost with `#,##0` format,
slash-separated FTE string, row-10 cell styles copied):
  Frouin | Climate-quality multi-mission Rrs record (radiometry + atm. correction) |
  1–5 | Frouin + Tan + Liu | 0.08/0.33/0.25 | 220,000
FTE encodes Robert's stated effort (Frouin 1 mo/yr, Tan 4 mo/yr, Liu 3 mo/yr).

**One assumption JXP should check:** Robert quoted "about $200K/year **without IDC**"
(incl. the $5K/yr storage/cloud); the sheet column is "Cost (**with overhead**)", so the
rows carry 200K × 1.10 = **220,000**, using the RFP's ≤10% indirect cap. Adjust if the
actual rate differs.

**Edit-not-replace verified twice** (subagent + independent connector read): file ID
unchanged (18-o6lDbBZ…), all 10 pre-existing rows (header + 9 data) value-identical
after the Drive round-trip, no new file created. Benign round-trip notes from the
subagent: single tab; ~988 trailing formatting-only rows; two contentless Google-export
stub parts (empty drawing/person XML) dropped on save; no comments, charts, merges, or
validations existed, so nothing of substance was lost. Scratch xlsx files retained in
the session scratchpad.

### 2026-08-25 (Final draft prompt 1: measured the overrun, posed Round-8 FD1–FD7 shortening questions — no edits)

Prompt 28 (1st Task under "Final draft"): read the current "VICC EOI" doc + the pasted EOI
guidelines, assess the page-limit overrun, and pose shortening questions in the Q&A. Model:
Fable. **No edits were made to the doc.**

**State of the doc** (team-revised since my Aug-19 edit): §III rewritten by the modeling
group — now global ECCO-Darwin + a data-constrained regional ROMS (California Current),
NBLING→D-LITE lineage, ML emulators, Ensemble 4D-Var, dual-numbers adjoint for ROMS; §IV's
float subsection rewritten (clearly Hervé's hand: "A distributed BGC-Argo observational
layer", harmonized UVP/transmissometry/hyperspectral products, sub-fleet requirements
definition); targeted-fieldwork paragraph now cites ongoing funded projects; my TOC
scaffold removed; V–VII still stubs. 15 comment threads (5 resolved).

**The measurement** (txt export, per section): §I 811 / §II 580 / §III 884 / §IV 648 =
**2,923 words for I–IV ≈ 6–6.5 pages** at 11 pt, before the figure and before the three
required-but-unwritten sections (team, budget, AI statement + hypotheses/policy clauses of
item 2). Budgeting ~0.9 pg for V–VII and ~⅓ pg for Figure 1 leaves ~950–1,000 words for
I–IV — **a cut to ~⅓**. Sentence-trimming cannot get there; sections must merge or go.

**Proposed plan posed in Round 8** (word-budget table in the Q&A): §I 811→~380 (merge the
three empirical-limits paragraphs; keep our-lane + success-metrics nearly intact);
**dissolve §II** (−580, the single biggest lever — Ltoa→Rrs pedagogy is tutorial for this
panel; its two load-bearing paragraphs fold into §I/§IV); §III 884→~400 (D-LITE rationale
currently stated twice); §IV 648→~350 (Robert's two blocks to ~2 sentences each; float
layer + fieldwork merged); V ~180 / VI ~110 / VII ~130 drafted fresh.

**Questions FD1–FD7**: does the CQ5 "do not touch §III" order lift, and who compresses it
(my lean: I compress, modeling group reviews); confirm dissolving §II; may I compress
Robert's adopted text; keep the figure at reduced size (lean: yes); soften the in-flux
success metrics to claim-level wording (JXP's own margin note says the 32% baseline will
change; lean: soften for the EOI); fix the **"UCSC ROMS" blind-review violation** found in
§III (+ full anonymization sweep; "Froudin+2019" typo); and confirm I draft V–VII within
the stated budgets (roles per RQ7 + Hervé's suggested expertise line; budget prose from
the Work Packages sheet; DQ7 portfolio sentences; DQ10 AI statement; one MRV/policy
sentence for item 2c).

Also flagged, not asked: several open comment threads are science disputes
(Stephanie's assimilate-what-we-don't-measure, Matt's NCP-from-NO3/O2) that the cut will
resolve only where it deletes the disputed sentence — I will not adjudicate silently.

Next: JXP answers FD1–FD7; then the shortening pass produces the ≤3-page draft.

### 2026-08-25 (Final draft prompt 2: reviewed FD answers, reworked the success metrics at high level, posed Round-9 FE1–FE3 — no edits)

Prompt 29 (2nd Task under "Final draft"): review the FD1–FD7 answers, react, and ask one
more round before any editing. Model: Fable. **The EOI doc remains untouched.**

**Answers locked**: §III editable — I compress, modeling group reviews (FD1); §II
dissolves (FD2); Robert's blocks compress (FD3); figure stays, reduced (FD4); full
anonymization sweep (FD6); I draft V–VII (FD7).

**FD5 became the substantive work of this turn.** JXP's direction (reinforced by an
interjection mid-research): the ~32% baseline was wrong AND the framing must stay high
level — "no one is interested in per-pixel values." Re-read `reports/biomass_summary.md`
§2.1/§3/§4: the 32% MAPE is conversion-only scatter (assumes away drift/biofouling/
calibration; no CRM for the "truth"); the report's own standard is the cross-method
spread — 2.5–3.5×, with the conversion choice alone spanning the global Cphyto stock
218–771 Tg C. Proposed replacement metrics paragraph (verbatim in Round 9): (i) global
phytoplankton carbon stock from ~3.5× to <1.8× with uncertainties validated against
independent methods; (ii) global POC/biological-carbon records to the same standard
(close the 2.5–3.5× cross-method gap by half) — with an NPP/export-propagation variant
offered if (ii) reads soft; (iii) trend detection >30 yr → ~20 yr within the existing
record. Every number traces to the report; nothing per-pixel.

**Round-9 questions**: FE1 confirm the metrics wording (or choose the NPP/export
variant); FE2 what total §VI states — the Work Packages sheet sums to ≈$2.6M/5 yr
(several WPs rowless) vs the DQ4(a) ~$10M scale; lean (a): state ~$8–10M as intended
scale with the sheet as partial detail; FE3 process confirmation — backup + in-place
update will orphan the open comment threads again (incl. unresolved science disputes);
hold if the team wants them resolved in-doc first.

Next: JXP answers FE1–FE3; the following prompt executes the full shortening pass
(≈1,500-word, ≤3-page draft: §II dissolved, §III ~400 w, §IV ~350 w, V–VII drafted,
anonymization sweep, reduced figure).

### 2026-08-25 (Final draft prompt 3: executed the shortening — "VICC EOI" rewritten in place at v7, ~1,770 words ≈ 3.5–4 pages)

Prompt 30 (3rd Task under "Final draft"): review the FE answers, then edit the doc to
3.5–4 pages in JXP's §I/§II voice. Model: Fable. **The live doc was updated in place**
(same file ID; no backup per FE3 — JXP had already made one); version stamp advanced
v6 → v7.

**FE answers folded in**: FE1 — metric (i) now "global Cphyto stock, uncertain to a
factor of ~3.5 (218–771 Tg C), determined to better than 50%, validated against
independent methods"; (ii) global POC to the same standard (close ≥half of the 2.5–3.5×
cross-method gap); (iii) trend detection >30→~20 yr; nothing per-pixel anywhere. FE2 —
the Work Packages sheet re-read: now fully populated, **total $8,939,064/5 yr**
(~$1.76–1.88M/yr, 13 PI groups incl. Kudela, Mazloff/Verdy, Claustre/Andres/Vellucci,
Dutkiewicz/Jahn, Houskeeper, Dierssen, Takeshita, Koestner, Edwards/Mattern); §V states
"approximately $8.9M." FE3 — no new backup made.

**The rewrite** (2,923+stubs → 1,773 body words; figure kept, scaled to 70%):
- §I (~640 w): JXP's opening + our-lane + new metrics; the two empirical-limits
  paragraphs merged into one ("Why is this not already done? Because the measurements,
  on their own, cannot do it."); "Are we stuck? Probably —" kept and sharpened; new
  closing paragraph states the **central hypothesis** explicitly (missing information
  exists in vertical structure, model process knowledge, assimilation covariances) —
  satisfying guidelines item 2's "hypotheses."
- Old §II dissolved per FD2; its vertical-information argument survives inside §I's
  hypothesis paragraph and §III's BGC-Argo layer. **One deliberate casualty to flag:
  the lidar/CALIOP/CALIGOLA sentence (Robert's addition) did not survive** — at this
  length it was an aside about a future mission; restore if Robert objects.
- §II (was III, ~440 w): the modeling group's five paragraphs compressed to three,
  keeping every capability (ECCO + regional ROMS both Darwin-coupled, spectral RT,
  NBLING→D-LITE, Tapenade + dual-number adjoints, hyperspectral radiance assimilation,
  SOCAT/glider-pH/GLODAP, ML emulators + Ensemble 4D-Var, two-scale design); the
  twice-stated D-LITE rationale stated once.
- §III (was IV, ~430 w): four bolded run-in paragraphs (Rrs record; carbon across the
  record; Hervé's BGC-Argo layer compressed with his framing intact incl. the future
  sub-fleet requirements; targeted fieldwork per his fill-gaps-only framing + "not a
  new global survey").
- §IV Team (~140 w): anonymized roles matching the 13 sheet WPs; European float
  laboratory (international box); ECR-led WPs; community-infrastructure partnerships.
- §V Timeline/budget (~110 w): yr 1–2 / 2–4 / 4–5 phasing; ~$8.9M; indirect ≤10%.
- §VI (~130 w): DQ7 portfolio differentiation (COCO2/InMOS/SUBSEA/CLARiTy); the
  MRV/policy sentence (item 2c); FAIR; the factual Claude AI-use statement (item 5).
- **Anonymization sweep**: "UCSC" removed; "our … NEMURO" → "an established dual-number
  technique … 11-component regional ecosystem model"; typos fixed (Froudin→Frouin,
  Housekeeper→Houskeeper); footnote [1] dropped with its 20 words.

**Verification**: scripted checks (forbidden strings absent, required strings present,
1 image, word count) + full connector read of the live doc post-upload. Uploaded HTML
archived at `proposals/Schmidt_Sciences/eoi_edit_2026-08-25_v7.html`.

**Known side effect**: the in-place update again orphaned the open comment threads
(JXP's backup holds them). The [XX]-era citation placeholders are gone; remaining
bracketed citations ([Stoer+25], [Bisson+20XX], etc.) still need the reference PDF.

Next: JXP trims 3.5–4 → 3 pages and refreshes Figure 1 ("this will change"); the
separate team form and reference PDF remain.
