# Retrieve or Bust Websites

## Goals

This prompt doc will be used to update the websites for the Retrieve or Bust GitHub Repository.  This includes both the GitHub facing site -- `https://github.com/Sea-Meets-the-Stars/Retrieve-or-Bust` -- and the Read the Docs site (yet to be created)


## Prompts

1. Read this file.  Execute the 1st task under "GitHub Website"

## GitHub Website

### Tasks

1. Let's improve the GitHub website via our README.md file. Please:

    - Add the SeaMeetstheStars logo which you can find here: `https://github.com/Sea-Meets-the-Stars`
    - Add a long description of the project.  
    - Add the team names: 
      - Paula Bontemps (URI)
      - Heidi Dierssen (UConn)
      - Henry Housekeeper (WHOI)
      - Robert Frouin (SIO)
      - Mariah Kavanaugh  (OSU)
      - Raphe Kudela (UCSC)
      - J. Xavier Prochaska (UCSC) 

In addition, create a new graphic which summarizes the project.  Use your imagination, and feel free to ask questions first in the Q&A section.  I will also use this grahpic in talks and presentations.  If you can, use `matplotlib` to create the graphic and save the Python script as `rob_graphic.py` in the `docs/scripts` directory.  Log your work in the "Logs" section.  Use Fable if you can
    
2. I have answered the Q&A section below.  Please read them and ask a new set if needed.  Log your work in the "Logs" section.  Use Fable if you can

3. I have answered the next round of Q&A.  Please read them and proceed to create the graphic.  Note the new name for the script is `rob_graphic.py`. Log your work.  Use Fable if you can

4. Ok, the graphic for the README.md file looks very good.  But the one for the talk needs these edits:

    - There is too much wasted white space
    - Many of the fonts are too small
    - the ends of "environmental" is cut off
    - The "Claude" text overlaps "one solution" text
    - The "Retrieve or bust" title is too strange and the * isn't needed

Please improve that graphic.  Log your work.  Use Fable if you can

5. That's much better.  Now:

    - Get rid of the line under the title
    - Make all the names bold
    - Make sure the text doesn't touch the axes of the figures in each panel
    - "environmental" is still a little cut off; fix

### Q&A

Before building `docs/scripts/ioptics_graphic.py`, a few questions so the graphic
lands the right message in the right format. (README is already updated: logo, long
description, and the 7-person team.)

**GQ1 — Core message.** What single takeaway should the graphic land?
  (a) The workflow / pipeline — "observe reflectance → apply a method → retrieve
      IOPs (with uncertainties) → validate against truth."
  (b) The scientific thesis — "the IOP inversion is degenerate; priors + AI supply
      the missing information that breaks it."
  (c) The benchmarking angle — "which IOP algorithm performs best?", comparing
      methods on a common footing.
  (d) A blend (say which elements).

GQ1: (d) -- (b) with a bit of (a)

**GQ2 — Layout / form.** How should it be composed?
  (a) A horizontal 4-panel pipeline (numbered stages, arrows between them) — reads
      left-to-right, great for a talk slide.
  (b) A conceptual diagram — the degeneracy (many IOP states → one Rrs) with priors
      collapsing the fan to a solution.
  (c) One hero science panel (e.g. retrieved vs true IOP spectra with uncertainty).

GQ2:  Let's try (b)

**GQ3 — Format & style for talks.** Aspect ratio and theme?
  (a) 16:9 slide, light background, large fonts, minimal text.
  (b) 16:9 slide, dark background (for dark presentation decks).
  (c) Square / 4:3 for a poster or paper figure.
  Also: match the SeaMeetstheStars branding (logo + a project color palette), yes/no?

GQ3:  I want two versions: one for a talk and one for the README.md file.  For the talk, I want a light background with large fonts.  For the README.md file, I want a dark background with smaller fonts.

**GQ4 — What to name and feature.** Which elements must appear?
  - Project name shown as `retrieve-or-bust`, `IOPtics`, or both?
  - Feature BING / the Gordon model explicitly, or keep methods generic
    (BING, GIOP, QAA, a neural net…)?
  - Call out the three prior sources (in-situ, environmental, time-series)?
  - Name the sensor (PACE / OCI, hyperspectral)?

GQ4:  `Retrieve or bust`; do not feature BING; yes, call out the priors; hyperspectral but not PACE or OCI.

**GQ5 — Team affiliations.** The README lists the 7 names only. Do you want
institutional affiliations added (and if so, please provide them — I don't want to
guess for real people)? Also confirm the spelling "Mariah Kavanaugh" and "Raphe
Kudela".

GQ5:  Ok, I've added the affiliations.  And I confirm those spellings

### Q&A — Round 2 (2026-07-20)

Thanks — README affiliations are now in. One genuine design tension to resolve
before I build, plus two specifics. (Locked from Round 1: name "Retrieve or bust";
no BING; call out the three priors — in-situ, environmental, time-series;
hyperspectral but no PACE/OCI; two renders — light/large-font for talks, dark/
smaller-font for the README.)

**GQ6 — Concept vs. pipeline (the tension).** GQ2 = (b) is a *conceptual*
degeneracy diagram, but your later edits (tasks 3–5) describe a *numbered 4-stage
pipeline*: "(1) Observe" (Rrs curves), "(2) …", "(3) retrieved IOPs" (legend + error
regions), "(4) Evaluate" (an "Algorithms" panel; title "Which IOP algorithm performs
the best?"). Those are different figures. Which do you want?
  (a) Pure conceptual — many `(a, bb)` states → one `Rrs`; priors + AI collapse the
      fan to a tight solution. (GQ2 = b as written.)
  (b) 4-stage pipeline — Observe `Rrs` → priors + AI method → retrieve IOPs (with
      uncertainty) → evaluate/benchmark. (What tasks 3–5 assume.)
  (c) Hybrid — a left→right flow whose *centerpiece* is the degeneracy-collapse
      concept: (1) Observe `Rrs` → (2) the degeneracy + priors/AI collapsing the fan
      → (3) retrieved IOPs with uncertainty → (4) evaluate. This is my reading of
      GQ1 = (d) ("(b) with a bit of (a)") and it keeps tasks 3–5 applicable.
      [Claude's lean]

GQ6:  Let's try (c)

**GQ7 — The "priors + AI break it" visual.** In the collapse step, show the three
prior sources (in-situ, environmental, time-series) as labeled inputs feeding a
funnel that narrows the IOP fan to a posterior. Should **AI / Claude** appear as an
explicit element (e.g. the engine doing the collapsing), or stay implicit?

GQ7:  Make the AI explicit

**GQ8 — Palette.** I don't have the SeaMeetstheStars brand colors. Provide a hex
palette, or shall I derive one from the logo — ocean blues + a star/gold accent —
and apply it consistently across the light (talk) and dark (README) renders?

GQ8:  Derive the palette from the logo.  

## Logs

### 2026-07-20 (Rewrote README.md; posed Q&A for the summary graphic)

Executed the 1st GitHub Website task: overhauled `README.md` and opened the Q&A for
the project graphic (the graphic itself is built in task 2, after answers).

**README.md.** Replaced the two-line stub with a full landing page:
- SeaMeetstheStars logo, centered and linked to the org — served via
  `https://github.com/Sea-Meets-the-Stars.png` (verified: 302 → org avatar PNG, 200).
- Long description adapted from `context/invite_letter.txt` (third person, no
  hedging): Overview + success criterion, "Why this is hard" (degeneracy), "The bet"
  (priors from in-situ/environmental/time-series + AI accelerant, science stays with
  humans, engine = Claude / Anthropic AI-for-Science), "Point of departure: BING"
  (Prochaska & Frouin 2025, a starting line not a destination), Related work
  (IOPtics readthedocs, BING citation), Team (all 7), package layout, license.
- Team listed by name only — did **not** invent affiliations for real people; asked
  in GQ5 instead. Note the git remote is `ocean-colour/retrieve-or-bust` but the
  prompt's GitHub-facing org is `Sea-Meets-the-Stars`; used the latter for the logo
  as instructed (JXP owns git).

**Q&A (GQ1–GQ5).** Questions to shape `docs/scripts/ioptics_graphic.py`: core
message (workflow / thesis / benchmarking / blend), layout (4-panel pipeline vs
degeneracy diagram vs hero panel), format & style for talks (aspect ratio, light/dark,
branding), what to name/feature (retrieve-or-bust vs IOPtics; BING/Gordon vs generic
methods; the three prior sources; PACE/OCI), and team affiliations + name-spelling
confirmation.

Next: await GQ1–GQ5, then build the graphic (task 2).

### 2026-07-20 (Reviewed GQ1–GQ5; added README affiliations; posed Round-2 Q&A)

Executed the 2nd GitHub Website task. Reviewed the Round-1 answers: GQ1=(d) thesis
(b) with a bit of workflow (a); GQ2=(b) conceptual degeneracy diagram; GQ3 = two
renders (light/large for talks, dark/smaller for README); GQ4 = name "Retrieve or
bust", no BING, call out the three priors, hyperspectral but not PACE/OCI; GQ5 =
affiliations added, spellings confirmed.

- **README.md** — folded in the affiliations now supplied in the task list
  (URI, UConn, WHOI, SIO, OSU, UCSC, UCSC).
- **Held off on building the graphic** because GQ2=(b) (a conceptual degeneracy
  diagram) conflicts with the *numbered 4-stage pipeline* that tasks 3–5 clearly
  assume ("(1) Observe" Rrs, "(2)", "(3) retrieved IOPs", "(4) Evaluate",
  "Algorithms" panel, "Which IOP algorithm performs the best?"). Building the pure
  conceptual diagram would leave tasks 3–5 with no panels to refine. Posed Round-2
  Q&A (GQ6–GQ8): reconcile concept vs pipeline (lean = hybrid left→right flow with
  the degeneracy-collapse as centerpiece, which keeps tasks 3–5 applicable), the
  priors/AI collapse visual (are the three priors the inputs; is Claude explicit),
  and the color palette (derive ocean-blue + gold from the logo unless given hexes).

Next: await GQ6–GQ8, then build `docs/scripts/ioptics_graphic.py` (both renders).

### 2026-07-20 (Built the summary graphic — docs/scripts/rob_graphic.py, two renders)

Executed the 3rd GitHub Website task. Round-2 answers: GQ6=(c) hybrid left→right flow
with the degeneracy-collapse as centerpiece; GQ7 = AI explicit; GQ8 = derive the
palette from the logo. Script name is now `rob_graphic.py` (per the updated tasks).

**Palette.** The SeaMeetstheStars avatar is the default GitHub placeholder — a single
gold/tan (~#D9C878) on light gray. Sampled it (PIL) and took the gold as the *star*
accent, paired with ocean blues (`#0B2545`/`#2E7FB8`/`#3FA7A0`/`#7FD1E3`) — "sea meets
the stars," and the right register for an ocean-color project. Applied consistently
across both renders.

**`docs/scripts/rob_graphic.py`.** One left→right flow with the thesis at its heart:
(1) **Observe** — hyperspectral Rrs curves (synthesized from a toy `(a,bb)→Rrs`
Gordon model so shapes are realistic, blue-green peak decaying to red, not Gaussian);
(2) **Break the degeneracy** (centerpiece, wider card) — a fan of many `(a,bb)`
spectra that mimic one Rrs, three labeled prior chips (in-situ / environmental /
time-series) feeding a funnel, an explicit **AI · Claude** node at the throat, and a
single tight "one solution" bundle out; (3) **Retrieve** — `a(λ)`, `b_b(λ)` with
uncertainty bands + small legend; (4) **Validate** — retrieved-vs-in-situ 1:1 scatter.
Gold arrows connect the four cards; title "RETRIEVE *or* BUST ★"; footer with the
7-person team and a tagline. Name shown as "retrieve-or-bust"; no BING; priors called
out; "hyperspectral" but no PACE/OCI — all per GQ4.

**Two renders** (per GQ3), 16:9, written to `docs/figs/`:
- `rob_graphic_talk.png` — light bg, large fonts (talks/slides).
- `rob_graphic_readme.png` — dark bg, smaller fonts; **embedded as the hero image in
  `README.md`**.

Debugging notes: (1) opaque card patches (figure-level) initially painted over the
inset axes — fixed by raising each inset's zorder above the cards; (2) first pass had
cards (1)/(2) overlapping and panel-2 chip text clipped — fixed the card x-geometry
(explicit gaps for arrows) and rebuilt panel 2 (narrower fan left of the chips, wider
chips, PRIORS label repositioned). Both renders verified visually.

Next: await feedback / any refinements to `rob_graphic.py`.

### 2026-07-22 (Reworked the talk render of rob_graphic.py per task-4 feedback)

Executed the 4th GitHub Website task — five fixes to the **talk** graphic (the README
render was fine). Refactored the script so packing (card geometry, title/subtitle/
footer y's) and font sizes are **per-theme**, letting me repack/enlarge the talk
render without disturbing the README one.

1. **Wasted white space** — the talk theme now packs tighter: taller cards
   (`hc` 0.44→0.56, centerpiece 0.50→0.60), lower card band (`yc` 0.30→0.18), title
   up at 0.91, footer at 0.07. Cards now fill the 16:9 frame.
2. **Fonts too small** — bumped the talk sizes (title 40→48, stage 23→27, body
   16→23, small 14→19, footer 15→20); added dedicated `fs_cap` (17) for the italic
   panel captions and `fs_chip` (16) for the prior chips.
3. **"environmental" cut off** — widened the prior chips (0.275→0.340) and set them
   in `fs_chip`; the full word now fits.
4. **"Claude" overlapped "one solution"** — removed the inline "one solution" text;
   the idea now lives in a panel-2 bottom caption "priors + AI → one solution", so
   the AI node / Claude label and the tight-solution bundle no longer collide.
5. **Title too strange, star not needed** — replaced the tri-colour "RETRIEVE *or*
   BUST ★" with a clean single-weight "Retrieve or Bust" plus a short gold accent
   rule beneath; dropped the star.

Also shortened the subtitle to one line ("The ocean-color IOP inversion is
degenerate — priors + AI break it."), removed the redundant "PRIORS" label (the
three chips + the bottom caption already convey it), and left-cleared the
"many (a,b_b) mimic one R_rs" note. Fixed one README regression introduced by the
new title rule (it struck through the dark-theme subtitle) by lowering that
subtitle. Both renders regenerated to `docs/figs/` and verified visually; the README
embed points at `rob_graphic_readme.png` and updates in place.

Next: await feedback / any further refinements.
