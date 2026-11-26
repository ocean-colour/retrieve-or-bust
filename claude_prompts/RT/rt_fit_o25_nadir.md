# Optimising the O25 form for nadir viewing — milestone M9

This prompt doc guides the step after M8 (`rt_fit_pb24.md`): take the **O25
form** (Pitarch et al. 2025), the best elastic model we have scored on PB24,
and optimise it for **nadir viewing**, the geometry of BRDF-normalised
satellite R<sub>rs</sub> (PACE). M8 closed the ZTT arm with a nadir-only
default, "ZTT (PB24_nadir)", at 4.36% median in `rrs` (4.91% in `Rrs`). The
O25 form, refit per geometry node on the same training rows, already reads
**1.16%** at nadir with nothing optimised for it. This milestone asks how
good a nadir O25-form model can be made, what it needs as input, and whether
it should become the backbone.

The primary dataset is **PB24**. L23 is a cross-check, never fitted (unless
Q11 says otherwise).

## Context

- `claude_prompts/RT/rt_fit_pb24.md`: M8's decisions (Q1–Q45) and logs.
  The O25 work is in tasks 5 and 6 (logs of 2026-11-05 and 2026-11-14), the
  design discussion (iv) and Q26 (the "arm B" proposal, deferred to this
  doc), and tasks 9–9c (the nadir machinery and what failed).
- `reports/report_rt_elastic_model.md` §10 (M8 summary) and
  `reports/report_pb24_baseline.md` (every model on PB24).
- `design/rt_elastic_implementation.md` §7c (M8 as built; §7c.5 O25
  reproduced, §7c.6 the baseline, §7c.11–§7c.13 the nadir variant).
- `design/ztt_pb24/`: `baseline_report.md` and its CSVs (O25 rows),
  `o25_reproduction.md`, `o25_published_score.md`, `o25_refit_lut.npz`,
  `nadir_metrics.md`.
- The papers, in `context/RT/`: `pitarch2025.pdf` (the O25 model) and
  `pitarch2025a.pdf` (the PB24 dataset).
- The published O25 code and lookup tables: `$O25_DIR` (default `../O25`
  beside the repo; GPL-3.0, read at run time, never committed: M8 Q17 (a)).

## What we already know

*Every number is from a committed file; "held out" is the realisation
split's 1,000 held-out realisations, suns 0–70°, all 451 bands.*

**The form.** With ω<sub>w</sub> = bb<sub>w</sub>/(a + bb) and
ω<sub>p</sub> = bb<sub>p</sub>/(a + bb),

    Rrs = (Gw0 + Gw1 ω_w) ω_w + (Gp0 + Gp1 ω_p) ω_p,

four coefficients per geometry. The published model tabulates them on PB24's
10 × 10 × 13 (θs, θv, Δφ) nodes, fitted by **unweighted** least squares on
the **OLCI** bands of all 5,000 realisations, for R<sub>rs</sub> ≤ 0.06 sr⁻¹.
It is linear in the G's, so every fit here is a closed-form `lstsq`. Around
the forward form the paper has an inversion (a<sub>nw</sub>(560) from a band
ratio, a spectral slope γ, a QAA-style step) that M8 did not touch.

**In the code** (`robust/rt/baselines.py`): `Rrs_o25`, `rrs_o25`,
`o25_coefficients` (θs-only table, linear interpolation, kinks at the
nodes), `O25_L23_REFIT` (three suns, fitted on L23), `O25Table`,
`fit_o25_table` (per-node fitter; `weighted=True` is ours, `weighted=False,
rrs_max=0.06` the paper's), `o25_published_table` (reads the GPL files).
`design/py/reproduce_o25.py` reproduces the published LUTs to 0.07% (the
residual is the bb<sub>w</sub> = b<sub>w</sub>/2 convention) and found the
azimuth mirrored (Δφ<sub>O25</sub> = 180° − φ<sub>PB24</sub>; irrelevant at
nadir).

**At nadir on PB24** (θv = 0; held out; `baseline_breakdowns.csv`, the
θv<sub>air</sub> = 0 rows, and `nadir_metrics.md`):

| model | space | median % | q90 % | rRMS % |
|---|---|---|---|---|
| **O25 form, node refit, relatively weighted** | Rrs | **1.16** | 3.77 | 2.39 |
| the same → rrs (Fresnel + downwelling transfer) | rrs | 1.68 | 4.01 | 2.72 |
| O25 form, node refit, the paper's objective | Rrs | 3.11 | 11.4 | 6.40 |
| O25 as published (in sample) | Rrs | 3.09 | 11.7 | 6.55 |
| ZTT (PB24_nadir), the package default | rrs / Rrs | 4.36 / 4.91 | 21.2 / 23.1 | – |
| standard Gordon | rrs | 6.52 | 17.5 | 10.2 |

- The fitting objective is worth a factor of 2.7 at nadir (3.11 → 1.16%):
  the paper minimises absolute error in R<sub>rs</sub>, we score relative
  error.
- No O25-form model produces a non-physical value anywhere on PB24.
- The nadir node refit is 40 numbers (4 G's × 10 suns) and none of them
  knows the wavelength, the phase function or the sky.

**Where the node refit's error sits** (the whole 0–70° window, all views;
nadir-only breakdowns do not exist yet and are task 1):

- By band: 2.5 / 2.1 / 1.9% at 350–400 / 401–450 / 451–500 nm, 0.8–0.9% at
  501–600 nm, 1.2–1.6% at 601–800 nm. A wavelength-independent form leaves
  a spectral residual, largest in the blue where water backscatter and the
  diffuse sky are largest.
- By B<sub>p</sub> (five equal-count bins): 2.43 / 1.82 / 1.24 / 1.22 /
  1.11%. The lowest-B<sub>p</sub> bin is twice the highest: the form has no
  phase-function input.
- By sun: 1.23% at 0° rising to 2.26% at 70°.
- By water class: 0.93–5.87% (class 2 worst).

**On L23** (nadir, three suns): the O25 form refit *on L23* reads 0.69% rRMS
held out (`O25_L23_REFIT`). How a PB24-fitted nadir model reads on L23 is
not measured.

**Carried forward from M8.**

- *The "arm B" proposal* (design (iv), Q26): the phase function entering
  where single scattering puts it, e.g. a factor Pbb(ψ; B<sub>p</sub>) /
  Pbb(ψ; B<sub>ref</sub>) on the particle term and β̃<sub>w</sub>(ψ)/β̃<sub>w</sub>(π)
  on the water term, with G's parametric in geometry. Its predictions were
  written then and never tested: (B1) the lowest-B<sub>p</sub> bin falls
  from 2.43% to ≤ 1.6% and the window median from 1.51% to ≤ 1.3%; (B3) on
  L23 ≤ 0.7% rRMS. If (B1) fails, the phase function is not a
  multiplicative single-scattering effect.
- *The nadir machinery*: `design/py/fit_ztt_pb24_nadir.py` loads every
  nadir quantity of the 5,000 realisations from the `pb24_v1` cache in
  memory (14.4 M training rows; seconds), with the splits, the breakdowns
  (θs, bb/a, η<sub>bb</sub>, B<sub>p</sub>, water class, band), the
  `Rrs` ↔ `rrs` Fresnel transfer and the L23 score. The 13 azimuth nodes at
  nadir are bit-identical, so φ = 0 is used.
- *The satellite target* (M8 Q34): `l2gen`'s f/Q correction normalises to
  θs = 0, θv = 0, so θs = 0 is a first-class row; the fit still covers
  suns 0–70°. PB24 is elastic; inelastic corrections are applied
  separately (M8 Q43).
- *Lessons that cost time in M8*: a prediction is written before each fit
  and its outcome stated plainly; the objective matters as much as the form
  (a least-squares optimum over bb/a to 17 is not the median optimum);
  a table interpolated between nodes has gradient kinks at the nodes; a
  term calibrated on PB24 may not transfer to L23, so L23 is scored at
  every rung, not only at the end.

## Code

- Python in the `ocean14` conda environment, from the retrieve-or-bust repo
  root (`robust` is not pip-installed). Data: `$OS_COLOR/SD/v5` and the
  cache `$OS_COLOR/SD/derived/pb24_v1`.
- **JXP runs all git.**
- Working agreements as in M8: pytest-gated tasks; suites green **with and
  without** the data (a small committed fixture; missing data skips, never
  fails); `ruff check` and `ruff format --check` clean on `robust` and
  `design`; fits deterministic (linear least squares, no optimiser), on the
  training side of the saved splits (`design/ztt_pb24/splits.json`), scored
  held out, with a `--check` that reruns byte for byte; constants shipped
  with provenance and a leak guard (the training digest); the `forward`
  freeze (record §8.0) in force, so any pinned number that moves is a
  deliberate re-baseline called out at the top of the log.
- Nothing from the GPL-3.0 O25 checkout is committed.
- This work is nadir-only and in memory: no multi-worker runs are needed.
  If one becomes necessary, the M8 OOM policy applies (≤ 6 workers, own
  `systemd-run` scope, not from Cursor's terminal).
- After each task: a subsection in `design/rt_elastic_implementation.md`
  (**§7d (M9)**) and an entry here under **Logs**.

## Prompts

### Plan

1. Read this doc and the items in Context. Read my answers in Q&A/Init.
   Ask further questions in Q&A/Plan if you have any, each with a
   recommendation. Then revise the Tasks below to match. Log your work.

### Tasks

*Draft of 2026-11-26 (Claude Fable 5.2), written before the Q&A/Init
answers; the Plan prompt revises it. Each task ends with a log entry and
JXP reviewing before the next. Every prediction is written into the
generated report before its outcome.*

1. **Task 1 — The nadir baseline for the O25 family.** A script
   `design/py/run_o25_nadir.py` writing `design/o25_nadir/` (CSV, markdown,
   figures), built on the nadir loader of `fit_ztt_pb24_nadir.py` (promoted
   into the package if Q15 agrees). Models, all at θv = 0, in `Rrs` and in
   `rrs`: O25 as published (the θv = 0 column of its LUT); the node refit
   under both objectives; ZTT (PB24_nadir); Gordon. Every breakdown the M8
   nadir report has (per sun including 80° and 87.75°, bb/a, η<sub>bb</sub>,
   B<sub>p</sub>, water class, band), the realisation and B<sub>p</sub>-band
   splits, and L23 for each model that can be evaluated there.
   **Gate:** the script reproduces the committed nadir numbers above
   (1.16 / 3.11 / 3.09%) from the cache; `--check` byte-identical; tests
   skip without the data and run on the cache miniature.

2. **Task 2 — What the four-coefficient form leaves behind at nadir.** No
   new model. Residual diagnostics of the node refit, each with a ceiling:
   the per-(sun, band) free refit (what a wavelength-dependent G could
   buy); the residual against B<sub>p</sub> at fixed (ω<sub>w</sub>,
   ω<sub>p</sub>) (what a phase-function input could buy); the adequacy of
   the quadratic (cubic and cross terms ω<sub>w</sub>ω<sub>p</sub>); the
   sensitivity to the bb<sub>w</sub> convention and to the sky (H);
   node-to-node smoothness of the G's in θs. **Gate:** every number in a
   committed CSV; nothing thresholded; this task measures.

3. **Task 3 — Design discussion.** No code. From tasks 1–2, a discussion in
   Q&A/Design of the candidate forms (Q5's ladder), each with a prediction,
   and of the acceptance gate with the numbers now in hand. Numbered
   questions; JXP chooses.

4. **Task 4 — Fit and ship "O25 (PB24_nadir)".** The chosen form, fitted by
   `design/py/fit_o25_nadir.py` (`--check`), constants with provenance in
   the package (location per Q12), smooth and differentiable in every input
   including θs (no interpolation kinks, Q3), a `DomainWarning` outside its
   calibrated domain. **Gate:** probes re-evaluated through the package
   without the data to 1e-12; constants == the committed JSON; the leak
   digest; float64 gradients against finite differences in every input;
   nothing existing moves a bit.

5. **Task 5 — The ladder and the acceptance gate.** Each rung added alone
   and removed alone; every breakdown of task 1; `Rrs` and `rrs`; the
   B<sub>p</sub>-band split refit on its own training side; L23 held out at
   every rung. The gate of Q10, pass or fail stated plainly, no tuning to
   pass. On a pass, the decision of Q12 (b) (whether it becomes a `forward`
   backbone) is put to JXP with the numbers; nothing changes a default
   without that answer.

6. **Task 6 — Close M9.** A review pass over the diff; record §7d
   completed; `reports/report_rt_elastic_model.md` gains §11 "Update (M9)"
   with figures reused from `design/o25_nadir/`; a hand-off naming the
   emulator retrain on the surviving backbone and anything task 5 left
   open.

## Q&A

### Init

*Questions from Claude (Fable 5.2), 2026-11-26, written with the doc. Answer
inline under each **JXP:** line.*

**Q1. What is "the O25 model" here: the forward form only, or the paper's
inversion too?** The forward form maps IOPs to R<sub>rs</sub> through four
G's. The paper's a<sub>nw</sub>(560), γ and QAA-style step invert it, and
are what a BRDF correction uses. Our use is a differentiable forward model
inside our own retrieval. Recommendation: **the forward form only**; the
paper's inversion pieces are out of scope and are not ported.

**JXP:**

**Q2. Which space is fitted: `Rrs` or `rrs`?** O25 is native in `Rrs`, the
satellite quantity, and fitting there avoids the surface transfer's own
error (1.5% at nadir with the shipped constants; the node refit goes from
1.16% in `Rrs` to 1.68% once pushed through it). ZTT and Gordon are native
in `rrs`. Recommendation: **fit in `Rrs`**, as the primary and gated
number; report `rrs` through the transfer beside it; and measure once what
a direct fit to PB24's `rrs` reads, so the comparison with ZTT does not
rest on the transfer.

**JXP:**

**Q3. The sun: nodes or a smooth function?** The published model is a table
on ten sun zeniths with interpolation, which has gradient kinks at the
nodes (measured in `o25_coefficients`: `jax.grad` and a central difference
disagree by 69% at a node). Options: (a) the node table, linear
interpolation; (b) each G as a low-order polynomial in cos θs<sub>w</sub>,
fitted to all suns jointly (still linear least squares), with the node
table kept as the ceiling. Recommendation: **(b)**, fitted over suns 0–70°
with 80° and 87.75° reported as extrapolation, and θs = 0 a first-class
row (the PACE geometry, M8 Q34). *Prediction:* a cubic in cos θs<sub>w</sub>
per G is within 0.05 pp of the node table.

**JXP:**

**Q4. Wavelength and the sky.** The four G's do not depend on λ, yet the
node refit's error is 2–2.5% below 500 nm and under 1% at 501–600 nm. Two
physical reasons a λ dependence exists: PB24's diffuse fraction H(θs, λ)
(the sky) and the water phase function's larger share in the blue. Options:
(a) keep the G's wavelength-independent, as published; (b) let the model
take the diffuse fraction H as an input (physical, and the seam
`Geometry.diffuse_fraction` already exists); (c) a per-band spectral factor
like ZTT's FL_AVE(λ) (451 numbers, calibrated to PB24's sky and IOP
models). Recommendation: task 2 measures the ceiling first; then **(b)** if
it captures most of it, since H is a quantity the application knows; (c)
only if you accept a model tied to PB24's spectral assumptions.

**JXP:**

**Q5. How far may the form move from the published one?** A ladder, each
rung one closed-form fit, all scored:
- (a) the published four G's, refit for nadir with our objective (exists:
  1.16%);
- (b) more terms in (ω<sub>w</sub>, ω<sub>p</sub>): cubic, and the cross
  term ω<sub>w</sub>ω<sub>p</sub>;
- (c) the phase function made explicit ("arm B"): factors
  Pbb(ψ; B<sub>p</sub>)/Pbb(ψ; B<sub>ref</sub>) and β̃<sub>w</sub>(ψ)/β̃<sub>w</sub>(π)
  on the particle and water terms, ψ = 180° − θs<sub>w</sub> at nadir;
- (d) G's that depend on B<sub>p</sub> directly (a calibration, not physics).

Recommendation: **(a) → (b) → (c)**, shipping the lowest rung within 0.1 pp
of the best in held-out median *and* on L23; (d) reported only. Is there a
rung you want excluded, or a hard requirement to stay with the published
four-coefficient form (as you kept TT's form in M8)?

**JXP:**

**Q6. Will the retrieval know B<sub>p</sub>?** Rung (c) needs the particle
backscattering ratio (or the two-component FF pair) as an input. From
satellite R<sub>rs</sub> alone it is not observed. If it must be assumed or
retrieved with a prior, a model that needs it may be worth less than its
PB24 score says. Recommendation: build the model so that the phase-function
input is **optional** (without it, B<sub>p</sub> = B<sub>ref</sub> and the
factor is 1), and score both ways, including with a deliberately wrong
B<sub>p</sub>, so the cost of not knowing it is a number. How do you intend
to treat B<sub>p</sub> in the PACE retrieval?

**JXP:**

**Q7. The objective and the domain.** Ours is relative least squares on
every row; the paper's is unweighted for R<sub>rs</sub> ≤ 0.06. M8 showed
the choice is worth 2.7× at nadir, and that a least-squares optimum over
PB24's whole bb/a range (to 17) can sacrifice the bulk. Recommendation:
relative weighting as the default; report the paper's objective beside it;
report the median and q90 per bb/a bin at every rung; and state a domain
for the shipped model only if a bin fails (as "ZTT (PB24_nadir)" has
bb/a ≤ 0.59). Do you want the fit restricted to a water range relevant to
PACE's open-ocean and coastal pixels, or all of PB24?

**JXP:**

**Q8. Bands.** PB24's hyperspectral set is 451 bands, 350–800 nm; the paper
fitted OLCI's 12. PACE OCI is hyperspectral from the UV to the NIR.
Recommendation: fit and score on **all 451 bands**, report the OLCI subset
once for comparison with the paper, and treat anything outside 350–800 nm
as outside the model.

**JXP:**

**Q9. Water backscatter.** The published calibration uses
bb<sub>w</sub> = b<sub>w</sub>/2; PB24's files carry bb<sub>w</sub> =
0.4992–0.4999 b<sub>w</sub> (a 0.07% effect on Gw0). Our package has its
own `conventions.bb_w` (M8 task 6 compared it with the files'), and the
application will use whichever pure-water model the retrieval adopts.
Recommendation: fit with the **files' own bb<sub>w</sub>**, and report the
error introduced by evaluating the fitted model with `conventions.bb_w`,
so the user knows the cost of a mismatched water model.

**JXP:**

**Q10. The acceptance gate, written before any fit.** Proposed, at nadir,
suns 0–70°, realisation split held out, all 451 bands, in `Rrs`:
- median below **1.16%** (the node refit of the published form, i.e. the
  optimised model must at least match a 40-number table while being smooth
  in θs) and below 1.0% at θs = 0;
- q90 below 4%;
- no non-physical value;
- L23 held-out rRMS at 0°/30°/60° **below 2.90%** (the best analytic
  backbone there; the O25 form fitted on L23 itself reads 0.69%, which a
  PB24-fitted model is not expected to reach).

Reported, never gated: `rrs`, suns 80° and 87.75°, the B<sub>p</sub>-band
split, every breakdown. Is this the right bar, and is beating
"ZTT (PB24_nadir)" (4.91% in `Rrs`) by this margin enough for it to replace
ZTT as the backbone, or is that a separate decision (Q12)?

**JXP:**

**Q11. L23: cross-check only, or part of the fit?** M8 kept L23 strictly
held out (Q44), and learned that PB24-calibrated terms can fail there.
For O25 the L23-fitted G's differ from PB24's (different sky, IOP models,
and R<sub>rs</sub> ↔ rrs handling). Recommendation: **cross-check only**,
scored at every rung; if the PB24-fitted model reads worse than 2.90% on
L23, the log says which term is responsible before anything ships.

**JXP:**

**Q12. Where it lives, and what it becomes.** (a) Location: the O25 code is
in `robust/rt/baselines.py` because it has been a rival. An optimised model
that we ship and may build on deserves its own module, e.g.
`robust/rt/o25.py`, with `baselines` keeping the published and refit
tables. (b) Role: on a pass, should it become a backbone that `forward`
can select (and the emulator can be retrained against), or stay a
stand-alone forward model beside ZTT? Recommendation: (a) a new module,
with the name "O25 (PB24_nadir)" in every table; (b) decided after task 5
with the numbers, no default changed before that.

**JXP:**

**Q13. Licence.** The published LUTs are GPL-3.0 and are read at run time
only (M8 Q17 (a)). Coefficients we fit ourselves on the public PB24 data,
with our own code, are ours to commit; the *form* is from the paper.
Recommendation: commit our fitted constants with provenance and a citation;
keep the no-vendoring rule for anything read from the O25 checkout. Confirm
you are comfortable with that.

**JXP:**

**Q14. Elastic only?** PB24 here is Raman-free. M8 Q43: inelastic
corrections are applied separately in the satellite application.
Recommendation: the same here, stated in the model's docstring and the
report.

**JXP:**

**Q15. Shared nadir code.** Task 1 needs the nadir loader, the breakdowns
and the scoring that `design/py/fit_ztt_pb24_nadir.py` holds privately.
Options: (a) import them from that script (as M8's scripts imported from
`run_pb24_baseline.py`); (b) promote the loader and the scoring into the
package (`robust.rt.data.pb24` gains a nadir batch; a small
`robust.rt.validation` addition), with tests, and have both scripts use
them. Recommendation: **(b)**, as the first step of task 1, with the M8
script's `--check` required to stay byte-identical after the move.

**JXP:**

**Q16. Who runs the tasks, and the milestone name.** You mentioned Opus 5.5
for the tasks. I have named this M9 and the record section §7d; the
outputs go to `design/o25_nadir/`. Recommendation: Opus 5.5 for tasks 1, 2,
4 and 5 (measured, gated work), Fable for task 3 (the design discussion)
and for the report in task 6. Change any of these names if you prefer.

**JXP:**

### Plan

### Design

## Logs

### 2026-11-26 — The doc created (Claude Fable 5.2; `rt_fit_pb24.md` task 10)

- **No code, no runs.** Read: `rt_fit_pb24.md` (task 10; tasks 5, 6, 9–9c
  and their logs; design (iv) and Q26; Q34, Q43–Q45),
  `robust/rt/baselines.py` (the O25 half), `design/ztt_pb24/
  baseline_report.md` and `baseline_breakdowns.csv` (the O25 rows, the
  θv<sub>air</sub> = 0 rows in particular), `nadir_metrics.md`.
- **Written:** this doc: the context, "What we already know" (the form,
  the code, the nadir numbers, where the error sits, what M8 carries
  forward), the working agreements, a Plan prompt, six draft tasks, and
  Q&A/Init with **Q1–Q16**, each with a recommendation.
- **The headline that motivates it:** at nadir, held out, the O25 form
  refit per node reads 1.16% median in `Rrs` (3.09% as published, 3.11%
  with the paper's objective), against 4.91% for ZTT (PB24_nadir) and
  6.52% (in `rrs`) for Gordon.
- **Not verified here:** nadir-only breakdowns of the O25 refit (by band,
  B<sub>p</sub>, bb/a) do not exist; the ones quoted are for the whole
  0–70° window and are labelled so. They are task 1. The baseline's nadir
  rows count each of the 13 identical azimuth nodes, which does not change
  a median.
- **Open:** Q1–Q16.
