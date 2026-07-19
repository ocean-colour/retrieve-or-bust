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
      - J. Xavier Prochaska 
      - Henry Houseke

In addition, create a new graphic which summarizes the project.  Use your imagination, and feel free to ask questions first in the Q&A section.  I will also use this grahpic in talks and presentations.  If you can, use `matplotlib` to create the graphic and save the Python script as `ioptics_graphic.py` in the `docs/scripts` directory.  Log your work in the "Logs" section.
    
2. I have answered the Q&A section below.  Please use the answers to create the graphic.  Log your work in the "Logs" section.

3. That is much better.  Here are a few edits to make to the graphic:

    - Use `BING` instead of `expb_pow`
    - Put the Rrs label on the (1) panel to the left of the axis
    - The (4) Evaluate panel is pretty..  uninspiring.  Do better
    - Make sure no fonts are smaller than 20pt.  The text at the bottom is too way small.
    - Modify "Which IOP algorithm should you trust?" to "Which IOP algorithm performs the best?"

4. Better!  Now:

    - Make the fonts in the Algorithms panel a bit smaller
    - Modify the Rrs curves in (1) Observe to be realistic.  They look like Gaussian curves, not real Rrs spectra.
    - Log your work


5. Almost there!  Now:

    - Make the "Legend" for panel (3) smaller.  
    - Think of a way to 'jazz up' panel (2).  It is boring
    - Make the error regions on the retrieved IOPs a bit easier to see (smaller alpha)
    - Log your work

### Q&A

**Claude (2026-07-10) — questions on the summary graphic:**

The graphic will be used in talks/presentations as well as the README, so a few
choices matter. My recommendations are marked ⭐.

1. **What should the graphic emphasize?**
   - (a) ⭐ **The pipeline flow** — Rrs(λ) spectrum → algorithm registry
     (Gordon/BING forward model) → retrieved IOPs (a_ph, a_dg, bb_p) ± uncertainty
     → uniform metrics/diagnostics (Taylor/Target, leaderboard). Shows what
     IOPtics *does*, end to end.
   - (b) **The comparison mission** — many algorithms (GIOP, QAA, GSM, expb_pow,
     ML…) run on common datasets against truth, ranked on a shared leaderboard.
   - (c) **Both** — pipeline on top feeding a multi-algorithm comparison below
     (richer but busier).
>A. It should attempt to do both, but be aware the audience is oceanographers and we are trying to grab their attention.

2. **Layout / aspect ratio?**
   - (a) ⭐ **Wide 16:9 slide** (good for talks *and* the top of the README).
   - (b) **Very wide 3:1 banner** (README header only).
   - (c) **Square / 4:3** (standalone figure).
>A. (a)

3. **Visual style?**
   - (a) ⭐ **Match the existing ocean palette** used in `docs/figures`
     (a_dg `#0b6fa4`, a_ph `#178a5a`, bb `#7b4fb0`) for consistency with the RTD
     figures.
   - (b) **Sea-meets-stars theme** — a deep ocean→night gradient (ocean blues
     fading to a starry dark) nodding to the Sea-Meets-the-Stars org, with the
     IOP component colors used for the data marks.
>A. Let's use (a)

4. **Title text on the graphic?** Default: **"IOPtics"** with subtitle
   *"Comparing IOP retrieval algorithms — for the community, by the community."*
   OK, or prefer different wording?
>A. This is ok


## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-07-11 (Summary graphic created from Q&A answers)

Executed prompt 2: built the IOPtics summary graphic using the Q&A answers
(1 = both pipeline + comparison for an oceanographer audience; 2a = wide 16:9;
3a = existing ocean palette; 4 = default title/subtitle).

**Deliverables**
- `docs/scripts/ioptics_graphic.py` — new matplotlib script (created the
  `docs/scripts/` directory, which did not exist). Data-free / schematic, so it
  runs anywhere without the `$OS_COLOR` data tree.
- `docs/source/_static/ioptics_graphic.png` — rendered 16:9 figure
  (12.8×7.2 in, 150 dpi) for the README, RTD, and talks.

**Design.** A four-stage left→right pipeline reads as a talk hero image:
1. *Observe* — three synthetic `Rrs(λ)` spectra (clear→productive water).
2. *Run algorithms* — a registry card listing `expb_pow, GIOP, QAA, GSM,
   ML/LUT…` (the "many algorithms" comparison input).
3. *Retrieve IOPs* — decomposed `a_dg` (blue), `a_ph` (green), `b_bp` (purple)
   with a ±σ uncertainty band, using the exact `docs/figures` OCEAN colors.
4. *Evaluate* — a shared leaderboard ranking the algorithms (the comparison
   payoff), plus the metric line (log-space accuracy · Rrs closure · ΔBIC).
   Teal flow arrows connect the stages; a bold value line ("Which IOP algorithm
   should you trust? …") anchors the audience; a footer names the truth datasets
   (L23 / PANGAEA / GLORIA) and the BING + ocpy engine.

Palette matches `docs/figures/make_model_components.py` (`a_dg #0b6fa4`,
`a_ph #178a5a`, `bb #7b4fb0`) for visual consistency with the RTD figures.

**Notes / learned**
- The prompt asked for `docs/scripts/`, though the repo's existing plotting
  scripts live in `docs/figures/`. Followed the instruction and created
  `docs/scripts/`; worth deciding later whether to consolidate the two.
- The illustrative leaderboard scores/ranking and algorithm list are schematic
  (for the graphic), not the output of an actual sweep.
- Also located the Sea-Meets-the-Stars logo for the README (prompt 1):
  canonical transparent PNG `https://profxj.github.io/images/SeaMeetsStars_logo_transparent_512.png`
  (fallback org avatar `https://avatars.githubusercontent.com/u/77852500?v=4`).

Regenerate with: `python docs/scripts/ioptics_graphic.py`

### 2026-07-11 (Graphic revision — prompt 3 edits)

Applied the five requested edits to `docs/scripts/ioptics_graphic.py` and
re-rendered `docs/source/_static/ioptics_graphic.png`:

1. **`BING` replaces `expb_pow`** in the algorithm registry (panel 2), the
   target diagram (panel 4), and the footer.
2. **`R_{rs}` label moved left of the axis** on panel 1 — now a rotated y-label
   (`set_ylabel`) instead of in-panel text.
3. **Panel 4 (Evaluate) redesigned.** The plain leaderboard bars were replaced
   with a **target diagram** (concentric skill rings + crosshair): each
   algorithm is a point scored on (bias, unbiased-RMSD); closest to the
   bull's-eye wins. BING is a gold-edged star at the centre; GIOP/GSM/QAA sit
   farther out. This is the recognizable ocean-color validation idiom and reads
   as far more dynamic than bars.
4. **Minimum font size raised to 20 pt.** The canvas was enlarged from
   12.8×7.2 → **16×9 in** (2400×1350 px @150 dpi) so every text element —
   including the previously tiny footer — is now ≥ 20 pt. Font sizes are
   centralized in `FS_*` constants at the top of the script.
5. **Value line reworded** to *"Which IOP algorithm performs the best? Run them
   all against known truth."* (was "…should you trust?…").

Layout tuning along the way: shortened the value line so it no longer runs off
the right edge, and nudged the card block up + the footer down so the two-line
stage captions no longer collide with the footer.

Regenerate with: `python docs/scripts/ioptics_graphic.py`

### 2026-07-11 (Graphic revision — prompt 4 edits)

Two edits to `docs/scripts/ioptics_graphic.py`, re-rendered
`docs/source/_static/ioptics_graphic.png`:

1. **Smaller Algorithms-panel fonts.** `FS_ALGO` 22 → 19 pt (this one panel is
   now intentionally just under the 20-pt floor set in prompt 3, per request;
   everything else stays ≥ 20 pt).
2. **Realistic `Rrs(λ)` spectra in panel 1.** The old curves were bare Gaussian
   humps. Replaced them with a small **Gordon forward model**
   (`water_rrs()`): total `a(λ) = a_w + a_ph + a_dg` and
   `b_b(λ) = b_bw + b_bp` are pushed through the Gordon quadratic
   `u = b_b/(a+b_b)`, `rrs = 0.0949u + 0.0794u²`, then the below→above-surface
   conversion `Rrs = 0.52 rrs / (1 − 1.7 rrs)`. Pure-water absorption `a_w(λ)`
   is the Pope & Fry / Smith & Baker table (interpolated onto the grid) and
   `b_bw(λ) ∝ λ^-4.32`. Three water types (clear / meso / eutrophic) are drawn
   by varying `Aph, Adg, Bp`; a small ~685 nm Gaussian adds chlorophyll
   fluorescence. The result shows the real features oceanographers expect — a
   blue peak in clear water, a green peak that shifts and rises with
   productivity, the ~675 nm chl-a absorption trough, and the fluorescence bump
   — instead of symmetric bumps.

**Learned / caveat.** Deriving the spectra from the forward model (rather than
faking shapes) makes them physically faithful and keeps the graphic self-
consistent with the pipeline it depicts, at essentially no extra code. The
absolute `Rrs` amplitudes are schematic (axes are unlabeled); the *shapes* are
the realistic part. The a_w table and forward-model constants live at the top of
the script if the curves ever need retuning.

Regenerate with: `python docs/scripts/ioptics_graphic.py`

### 2026-07-11 (Graphic revision — prompt 5 edits)

Three edits to `docs/scripts/ioptics_graphic.py`, re-rendered
`docs/source/_static/ioptics_graphic.png`:

1. **Smaller panel-3 legend.** `FS_LEGEND` 21 → 16 pt, with tighter
   `handlelength`/`handletextpad`/`labelspacing`, so the `a_dg`/`a_ph`/`b_bp`
   key no longer crowds the curves.
2. **Jazzed-up Algorithms panel (2).** The plain vertical list of chips was
   replaced with a small **convergence graphic**: five algorithm *pills*
   (BING, GIOP, QAA, GSM, ML/LUT) on the left, each linked by a curved,
   color-matched connector into a single navy **"Gordon forward model"** hub on
   the right. This visually states the core idea — many algorithms, one common
   engine — rather than just listing names, and reads as far livelier in a talk.
3. **Clearer IOP uncertainty bands.** Previously only `a_ph` had a ±σ envelope
   at `alpha=0.18`. Now every retrieved component (`a_dg`, `a_ph`, `b_bp`) gets
   a lighter band (`alpha=0.11`, ±14 %), so the uncertainty concept reads across
   all three curves without muddying them.

**Learned.** The convergence motif does double duty: it removes the "boring
list" and reinforces the pipeline story (registry → shared forward model → the
big teal arrow into *Retrieve*). Note the panel-2 hub/pills are drawn in axes
fraction on a near-square inset, so the circle is very slightly elliptical —
fine at this size; would need an equal-aspect inset to be a true circle.

Regenerate with: `python docs/scripts/ioptics_graphic.py`
