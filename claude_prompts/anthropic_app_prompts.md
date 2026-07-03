# Anthropic Application Prompts

We will use this doc to prompt the generation of an application for Anthropic.

## Goals

First, see the Goals in the other prompt files in this folder.

Second, this file will guide the generation of the application.

## Context

See the following:

- context_summary.md
- The Anthropic websites on their open calls
   - https://support.claude.com/en/articles/11199177-anthropic-s-ai-for-science-program
   - https://docs.google.com/forms/d/e/1FAIpQLSfwDGfVg2lHJ0cc0oF_ilEnjvr_r4_paYi7VLlr5cLNXASdvA/viewform

## Prompts

1. Perform the 1st Task under "Invite letter"
2. Perform the 2nd Task under "Invite letter"
3. Perform the 3rd Task under "Invite letter"
4. Perform the 4th Task under "Invite letter"
5. Perform the 5th Task under "Invite letter"

## Invite letter

We will generate an invite letter to potential co-investigators to join the project.  It needs to provide them context about what we will try to do and why.  The letter will be shared via Google docs.  You will need to generate a file named `invite_letter.txt` in the `context` folder that I will copy into the Google doc.

### Tasks

1. Read all of the context.  Have a conversation with me about the project and the invite letter.  Put your questions in the Q&A section. Log your work.

2. Read my responses in the Q&A.  If you have more questions, add them to the Q&A.  Write nothing else yet.  Log your work.

3. Read my responses in the Q&A.  If you have more questions, add them to the Q&A.  Write nothing else yet.  Log your work.

4. Ok, read my final responses.  Draft the invite letter.  Put it in the `context` folder as `invite_letter.txt`.  Log your work.

5. Make these edits:
   - Emphasize that AI has taken such a leap forward in the past few months that the time is ripe
   - Add that I (JXP) have a close colleague at Anthropic who might help out
Log your work.

6. I have downloaded version 1 of the invite letter.  It is in the `context` folder as `InviteLetter_v1.pdf`.  Please edit the `invite_letter.txt` accordingly and also to address the comments provided by Robert below:

"Letter is engaging and energetic, but there might be a few missing items. First, it would be good to specify what “success” mean (the target outcomes). Second, it would also be good to indicate more explicitly why the proposed approach may succeed where previous methods have difficulties. Third, expresssions about AI, such as “let modern AI search the solution space for us” and “handing to AI to discover” would make some invitees uneasy. It would be useful to elaborate a bit on the role of AI. Fourth,  BING is mentioned, but might not be known by some invitees. Adding a  short explanatory sentence would be appropriate. Fifth, Slack instruction might be too abrupt. Consider softening, e.g., "we will use Slack rather than email for follow-up". Fifth, it might be good to emphasize that the intent is to create a small, intellectually open group where ideas, critiques, datasets, and validation perspectives can shape the direction of the work."

Add a version number to the file and date.  Log your work


### Q&A

Questions from Claude before drafting `invite_letter.txt` (answers welcome inline).
Each has my recommended default in *(rec: …)*.

**Purpose & framing**

1. **Is the invite letter tied to the Anthropic AI for Science application, or
   separate?** i.e. are we recruiting co-investigators to be *on* the Anthropic
   proposal (sharing the $20k API credits, using Claude as the core method), or
   is "join the project" the broader science collaboration and Anthropic is just
   one funding thread? *(rec: the letter recruits scientific co-Is for the
   project; mention the Anthropic AI-for-Science angle as the near-term vehicle
   but don't make the letter about Anthropic.)*
   >A. The letter recruits scientific co-investigators to be on the project.  We will mention the Anthropic AI-for-Science angle as the near-term vehicle but don't make the letter entirely about Anthropic.

2. **How bold is the tone?** The repo framing is dramatic — "humans' last attempt
   to solve the IOP inversion; if we fail we give up or wait for better AI."
   Do you want that provocative "retrieve-or-bust" energy in the letter, or a
   measured tone for senior colleagues? *(rec: lead with the bold hook, then
   back it with the concrete technical bet — bold but credible.)*
   >A. We will lead with the bold hook, then back it with the concrete technical bet.  Bold but credible.

**Audience & the ask**

3. **Who is the audience?** Named individuals you'll each address, or one generic
   letter? And what mix — ocean-optics/IOP experts, ML/AI researchers, in-situ
   data holders, remote-sensing/PACE people? *(rec: one generic-but-personalizable
   letter with a slot for a personal opening line.)*
   >A. It will be a set of ocean optics experts, first and foremost.  But I am likely to include a few astrophysics experts as well.

4. **What exactly are we asking a co-investigator to contribute and get?** e.g.
   in-situ / time-series datasets, domain expertise, algorithm development,
   validation, co-authorship, being named on the Anthropic application. What is
   the value proposition *to them*? *(rec: ask for data + expertise + co-authorship;
   offer shared credit, shared compute, and a shot at a high-impact result.)*
   >A. We will ask for their input as things develop.  Ideas, mainly, and other feedback.

5. **Time/commitment expected of them, and over what horizon?** Does the 6-month
   Anthropic credit window set the clock? *(rec: light-touch — data + periodic
   input over ~6–12 months; you and the core team do the build.)*
   >A. They can contribute as they have time.  But, if they do not interact, they will eventually be dropped from the project.

**Team & credibility**

6. **Who is the "we"?** Is this you (JXP) solo, you + Rick Frouin (BING co-author),
   or a named core team, and at which institutions? Who signs the letter? *(rec:
   you as PI; name Frouin/BING as the technical pedigree; sign as yourself.)*
   >A. Right now it is myself and Robert Frouin (SCRIPPS)

7. **How much do we lean on BING and the 2025 Biogeosciences paper** as the
   credibility anchor / point of departure? *(rec: yes — cite it as the honest
   baseline that quantifies the degeneracy, and frame retrieve-or-bust as the
   next step beyond it.)*
   >A. It is the point of departure, but there is no expectation that it will relate closely to the final solution.

**Scientific content & scope**

8. **How much technical detail belongs in the letter?** Full degeneracy argument
   + the priors-from-in-situ/environmental/time-series thesis + ">4 params from
   hyperspectral" goal, or a high-level version with a link to
   `context_summary.md`? *(rec: a tight one-paragraph version of the thesis; keep
   the letter ~1 page; offer the summary as an attachment.)*
   >A. We will keep it tight and high-level.  But we will offer the summary as an attachment.

9. **Do we commit to specific targets in the letter** (PACE/OCI primary, MODIS/
   SeaWiFS legacy; full component separation a_ph/a_dg/bb_p; the ">4 independent
   parameters" success bar)? *(rec: name PACE and component separation as the
   goal; keep the ">4 params" bar internal — it's still provisional.)*
   >A. No specific targets.  Scientifically, it is about milking the most from hyperspectral data.

**Logistics**

10. **Practical bits:** target length (1 page?), deadline for them to respond /
    join, any co-authorship or data-sharing norms to state up front, and should
    it end with a concrete next step (a call? a shared doc?). *(rec: 1 page;
    soft 2–3 week response window; offer a short call as the next step.)*
   >A. 1 page.  Put in placeholders for the other details and I'll fill them in eventually.

---

**Follow-up questions from Claude (Task 2)** — a few things your answers opened up
that would sharpen the draft. Same *(rec: …)* convention.

11. **The astrophysics angle — why them, and do we say it?** My read: the IOP
    inversion is a classic *ill-posed inverse problem with a degeneracy*, and
    BING already borrows the astronomer's toolkit (Bayesian/MCMC). Astrophysicists
    are fluent in exactly that — priors, degeneracies, spectral inference. Should
    the letter explicitly draw that "this is your kind of problem, in a new
    ocean" bridge to hook the astro readers? *(rec: yes — one sentence naming the
    degeneracy as the shared language; it's the honest reason they're invited and
    it flatters without pandering.)*
    >A.  The ones I send it to will understand already.  No need to worry about them.

12. **How bluntly do we state "engage or be dropped"?** You noted inactive
    members are eventually dropped. In a *recruiting* letter that can read as
    cold. Options: (a) state it plainly as a fairness norm, (b) soften to
    "active contributors will be the ones carried through to authorship," or
    (c) leave it out of the letter and handle it privately. *(rec: (b) — frame it
    as an invitation to engage, with authorship following genuine contribution.)*
    >A. Frame it very politely.

13. **Is co-authorship the explicit carrot?** Since the ask is ideas/feedback
    (not data), authorship-for-engagement is likely the main incentive. State it
    outright, or keep it implied? *(rec: state it lightly — "we intend contributors
    to be co-authors on resulting work.")*
    >A.  There is no real carrot other than to be a part of it, i.e. for fun.

14. **Project name in the letter.** Use the provocative working name
    **"retrieve-or-bust"** as the hook, a more formal title, or no name yet?
    *(rec: use "retrieve-or-bust" once, in the bold opening, then describe the
    science plainly — it earns the drama.)*
    >A. Use "retrieve-or-bust" once, in the bold opening, then describe the science plainly — it earns the drama.

15. **How openly do we say the science itself is AI/Claude-driven?** The thesis is
    "solve it *with AI*," and Claude is doing the building. To a scientific
    audience, do we (a) foreground that AI is a full collaborator generating the
    approach, (b) mention AI as a powerful tool we'll lean on, or (c) keep it
    understated? *(rec: (a), briefly and candidly — it's the whole premise and the
    honesty is itself a selling point; the Anthropic vehicle makes it natural.)*
    >A. The science is not Claude driven.  Only the solution.

16. **Signature/affiliation lines.** Confirm the sign-off is
    **J. Xavier Prochaska (UC Santa Cruz)** and **Robert Frouin (Scripps
    Institution of Oceanography, UC San Diego)**, and whether the letter is signed
    by both or just you. *(rec: both names, sent under your name.)*
    >A. Both names, sent under your name.

---

**Final confirm-style questions from Claude (Task 3)** — the spec is essentially
complete; these just lock the two subtle points before I draft. *(rec: …)* as before.

17. **The "science vs. solution" line (your Q15 answer).** I'll frame it as:
    *the science is ours* — the problem, the physics, the data, the judging of
    what counts as success — and *the bet is that AI can generate the solution
    method* we haven't been able to construct by hand. Is that the distinction
    you mean? *(rec: yes — say AI is the engine we're pointing at the solution,
    not a co-author of the science.)*
    >A. Yes, that is right.

18. **"For fun," no carrot, and graceful drop-off — combined.** Given there's no
    incentive beyond being part of something ambitious, and inactive members
    simply drift off, I'll frame the whole ask as *an open invitation to come
    along for an audacious ride, engage when the spirit moves you, no
    obligations* — with **no promise of authorship or deliverables**. Good?
    *(rec: yes — this reads as generous and low-pressure, and matches "for fun.")*
    >A. That's good

19. **Placeholders I'll insert** (so nothing surprises you): `[DATE]`,
    `[SALUTATION / recipient name]`, `[RESPONSE / how to reply]`,
    `[NEXT STEP — e.g. a call or shared doc]`, `[YOUR CONTACT]`. Any to
    add or drop? *(rec: this set; you fill them in before sending.)*
    >A. No, that's good.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-07-01 (Read context for invite letter; posed Q&A questions)

Executed Task 1 under "Invite letter": read all context and opened a conversation
about the project and the letter, with 10 questions in the Q&A section above
(grouped: purpose/framing, audience/ask, team/credibility, scientific
content/scope, logistics), each with a recommended default.

Context consumed: the goals in `start_up.md` and `context_prompts.md`, the full
`context/context_summary.md` (degeneracy of the IOP inversion; the project's bet
that priors from in-situ + environmental + time-series data break it; open
choice of method; provisional ">4 independent parameters from hyperspectral"
success bar; PACE/OCI primary + MODIS/SeaWiFS legacy; component separation), and
the Anthropic **AI for Science** program page.

Anthropic program (learned via web fetch): **$20k in API credits over 6 months**,
standard Claude models, API-only; for researchers at institutions/nonprofits;
**environmental & earth science are named priority domains** (good fit); selection
on scientific merit, impact, technical feasibility (how much Claude helps), and
biosecurity screening; **rolling review on the first Monday of each month** (no
hard deadline); awardees bound by the standard Usage Policy, no stated publishing
requirement. The Google Form was not fetched (interactive/auth). This shaped Q1
(is the invite letter tied to the Anthropic proposal or the broader
collaboration?) and the 6-month-horizon question.

### 2026-07-01 (Read Q&A answers; added follow-up questions; no letter yet)

Executed Task 2: read JXP's inline answers to the 10 questions and added six
follow-up questions (11–16) to the Q&A. Wrote nothing else — the letter itself
is a later task.

What the answers changed about the plan:
- The letter recruits **scientific co-investigators** (not an Anthropic-centric
  pitch); Anthropic AI-for-Science is named as the near-term vehicle only.
- Tone: **bold hook, then the concrete technical bet** — bold but credible.
- Audience is **ocean-optics experts plus a few astrophysicists** — notable,
  and the reason for follow-up Q11 (draw the ill-posed-inverse-problem /
  Bayesian-degeneracy parallel that astronomers already speak?).
- The **ask is ideas & feedback, not data** — so authorship-for-engagement is
  the natural incentive (Q13) and the "engage or be dropped" norm needs a
  deliberate tone decision (Q12).
- Team = **J. X. Prochaska (UC Santa Cruz) + Robert Frouin (Scripps)**; BING is
  the point of departure but **not** expected to resemble the final solution —
  so the letter should cite it as origin, not commitment (informs Q7 already).
- Keep it **tight, high-level, one page**, summary as attachment, no numeric
  targets, framed as "milking the most from hyperspectral data," with
  placeholders for logistics.

The new follow-ups (11–16) target the remaining genuine drafting forks: the
astrophysics bridge, how bluntly to state the drop-if-inactive norm, whether to
name co-authorship as the carrot, whether to use the "retrieve-or-bust" name in
the hook, how openly to state that the science itself is AI/Claude-driven, and
the exact signature/affiliation lines.

Note: again attempted to run this review on **Fable 5** per JXP's standing
request; it returned "currently unavailable," so I used the default model.
Awaiting answers to 11–16 before drafting `invite_letter.txt`.

Note: attempted the web fetch on **Fable 5** per JXP's request, but the model
returned "Claude Fable 5 is currently unavailable"; fell back to the default
model. No `invite_letter.txt` written yet — drafting is a later task, gated on
the answers above.

### 2026-07-01 (Read answers to 11–16; added final confirms 17–19; no letter yet)

Executed Task 3: read JXP's answers to follow-ups 11–16 and added three final
confirm-style questions (17–19). Still wrote no letter.

What the 11–16 answers settled, and the two subtle points they surfaced:
- **Q11 (astro bridge):** the astrophysicists he'll send it to already grasp the
  inverse-problem/degeneracy framing — so DON'T over-explain it; keep it natural.
- **Q12 (drop-if-inactive):** frame **very politely** — natural attrition, not a
  threat.
- **Q13 (carrot):** there is **no carrot** beyond being part of something
  ambitious — "for fun." → the letter must NOT promise authorship or rewards.
- **Q14:** use "retrieve-or-bust" once, in the bold opening.
- **Q15 (AI framing):** CRITICAL nuance — *"The science is not Claude driven.
  Only the solution."* The problem, physics, data, and success-judging are
  human-led; the bet is only that AI can produce the *solution method*. The
  letter must draw this line cleanly or it will misrepresent the project.
- **Q16:** both names (Prochaska, UC Santa Cruz; Frouin, Scripps/UC San Diego),
  sent under Prochaska's name.

The Q13 "for fun" + Q12 "polite drop-off" answers reinforce each other: the ask
is a **low-pressure, no-obligation invitation to come along for an audacious
ride** — no deliverables, no authorship promise, engage when inspired. Q17–19
lock the "science vs. solution" wording, confirm the no-carrot/graceful-drop
framing, and list the placeholders I'll insert.

Note: attempted this gap-check on **Fable 5** again per JXP's standing request;
still "currently unavailable" (third attempt today), so used the default model.
The spec is now essentially complete — once 17–19 are confirmed (or "go with
recs"), the next task can draft the one-page `invite_letter.txt`.

### 2026-07-01 (Drafted invite_letter.txt)

Executed Task 4: read the final answers (17: yes; 18: good; 19: good — all recs
confirmed) and drafted the one-page letter at `context/invite_letter.txt`.

The draft, section by section, honors the full Q&A:
- Bold opening using "retrieve-or-bust" exactly once; the "solve it or learn it
  waits for better AI/machines" framing carries the drama without melodrama.
- A degeneracy paragraph that assumes an expert reader (doesn't over-explain, per
  Q11) and states the core bet: break the degeneracy with external information —
  in-situ, environmental, and time-series priors — and let AI search for the
  method.
- A dedicated paragraph drawing the CRITICAL line from Q15/Q17: the science is
  human-led; only the *solution method* is AI/Claude-driven. Anthropic AI-for-
  Science named in a single sentence as the near-term vehicle (Q1).
- BING (Prochaska & Frouin 2025) as point of departure, explicitly NOT expected
  to resemble the final solution (Q7); scientific aim stated as "milk the most
  from hyperspectral data," no numeric targets (Q9).
- The ask (Q4/Q13/Q18): ideas and feedback only, no data, no deliverables, no
  deadlines; honest that there's no carrot beyond the fun of it; drift-in/out
  framed very politely (Q12), no authorship promise.
- Attachment of the context summary offered (Q8).
- Signature: sent under J. X. Prochaska (UC Santa Cruz), naming Robert Frouin
  (Scripps/UC San Diego) as co-lead (Q16).
- Placeholders inserted per Q19: [DATE], [SALUTATION / recipient name],
  [RESPONSE / how to reply], [NEXT STEP …], [YOUR CONTACT].

Note: attempted to draft on **Fable 5** per JXP's standing request (fourth
attempt today); still "currently unavailable," so I wrote the draft on the
default model. Letter is ready for JXP to review, fill placeholders, and paste
into the Google Doc.

### 2026-07-01 (Task 5: two edits to invite_letter.txt — "why now" + Anthropic contact)

Executed Task 5's two requested edits to `context/invite_letter.txt`:
1. **"The time is ripe."** Added to the bet paragraph: AI has lurched forward so
   dramatically in the past few months that a problem which felt hopeless a year
   ago now looks worth attacking head-on — "I would rather try now than wait."
   This gives the letter an explicit why-now, reinforcing the bold hook.
2. **Anthropic colleague.** Extended the Anthropic sentence to note that JXP has
   a close colleague at Anthropic who may be able to help — kept to a single
   clause so the letter still isn't "about" Anthropic (consistent with Q1).

Both edits were surgical (kept the surrounding voice and the one-page length);
no other content changed. The "science vs. solution" line, no-carrot/for-fun
ask, BING-as-departure framing, and signature all remain as approved.

Note: per JXP's standing request I again checked **Fable 5** availability before
editing (fifth attempt today); still "currently unavailable," so the edits were
made on the default model.

### 2026-07-02 (Task 6: revised letter to v2 addressing Robert's review — drafted on Fable 5)

Executed Task 6. First read JXP's downloaded **`InviteLetter_v1.pdf`** and found
it contained several manual edits made in the Google Doc that were NOT in the
working `invite_letter.txt` (the two had diverged): the "or is simply
intractable" clause, the "basis functions are nearly infinite" sentence, "a big
bet," "hopeless to me a few months ago," removal of "this letter is about the
science, not the tool," the `ioptics.readthedocs.io/en/develop` link, the "FWIW …
additional context" line, the smiley, and a Slack sign-off. Reconciled the `.txt`
up to the v1 content and then applied Robert Frouin's six review comments,
producing **v2**.

How each of Robert's comments was addressed:
1. **What "success" means** — added a short paragraph stating success as an
   *outcome*, kept QUALITATIVE (no number, honoring JXP's Q9 "no specific
   targets"): more independent, physically meaningful IOP components from
   hyperspectral reflectance than current methods manage, with credible
   uncertainties, validated against in-situ truth.
2. **Why this may succeed where others struggled** — added: prior methods fixed
   spectral shapes by hand or lacked systematic external information; we supply
   exactly that (in-situ/environmental/time-series priors) and use AI to explore
   far more candidate methods than hand-design allows.
3. **Softened uneasy AI phrasing** — removed "let modern AI search the solution
   space for us" and "handing to AI to discover"; reframed AI as a "tireless
   collaborator" that proposes/codes/stress-tests under human scientific
   direction ("not asking a machine to divine the physics"). Kept the "science
   is ours" distinction.
4. **BING explained** — now "BING (Bayesian INferences with Gordon coefficients;
   Prochaska & Frouin 2025), our open-source framework that casts the inversion
   as Bayesian inference on the Gordon reflectance model."
5. **Slack softened** — dropped the abrupt "I won't be communicating by email any
   further"; now "for follow-up we'll use Slack rather than email, so the
   conversation stays in one lively place," with a [SLACK LINK] placeholder.
6. **Small, intellectually open group** — added explicitly: ideas, critiques,
   datasets, and validation perspectives all welcome and genuinely shape the
   direction (kept "no data required" but now welcomes data/validation).

Versioned the file with header `v2  2026-07-02`. Kept placeholders
([ADDITIONAL CONTEXT LINK], [SLACK LINK]), the ioptics link, and the signature.

One tension flagged for JXP: Robert asked to "specify success/target outcomes"
while JXP (Q9) wanted "no specific targets." Resolved by stating success as a
qualitative outcome, not a number — but JXP should confirm that framing.

Note: **Fable 5 became available today** — after five straight "unavailable"
results (2026-06-30 → 07-01), this v2 draft was produced on **Fable 5** per JXP's
standing "use Fable if allowed" request. I fixed one `&amp;` HTML artifact from
the model output back to `&` before saving.
