# Elastic RT Coding — Prompt 5 (M4: Validation — Week-1 prototype complete)

## Goals

Implement **Milestone M4**: the **validation protocol** and the acceptance gate that
declares the Week-1 prototype **done**. Run the design §6 protocol on the held-out
splits, compare against Gordon / PR05 / O25, and ship a metrics table + figures.

M4 is the milestone that decides what the prototype may claim. Every module it needs
exists and is tested; what M4 adds is *judgement* — which numbers are the headline,
which are caveated, which comparison is fair. Two of its required inputs are not in the
repo at all (see **PR05 and O25**), and its acceptance gate has a half that M3 measured
as **seed-dependent** (see **The M4 gate**). Both are flagged below with numbers rather
than left to be discovered.

## Claude

### Skills

Consider `.claude/skills/` — `critical-partner` (M4 declares the prototype "done": the
pressure is to round a fragile result up), `grill-me`, plus the `dataviz` and
`code-review` conventions the earlier milestones used.

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements* (git by JXP; `ocean14`;
CPU-only JAX; reuse `ocpy`/`bing`; BING test conventions; pytest-gated; Fable; log).

The rhythm M0–M3 settled: **ask in Q&A** (numbered, phrased so work continues without
an answer; check for answers before each task), then after the code a **notebook**, a
**PR-review pass**, and a **hand-off edit to the next prompt doc**.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M4 (the acceptance gate)
  and the Testing strategy.
- **Design** — `design/rt_elastic_model.md` §6 (validation protocol: accuracy per
  λ / geometry / `B_p`; held-out generalisation; speed; the gradient gate).
- **Split policy (CQ6)** — random 20% of scenes; **hold out 60° solar zenith**
  (train 0°/30°). Both are built already: `L.make_splits(batch)`.
- **Implementation record** — `design/rt_elastic_implementation.md` (currently
  **v0.14**; §5 is all of M3 — §5.4 results, §5.5 the open problem, §5.6/§5.6b the bugs
  and the PR review, §5.8 the notebook; §8 cross-cutting conventions; §9 module index;
  §10 CI). §6 is an M4 stub. At close: bump the version, fill §6, flip M4 in §1, refresh
  the module index.
- **The four notebooks** — `notebooks/RT/rt_elastic_coding_{1,2,3,4}.ipynb`. Notebook 4
  §1 specifies the hybrid stage by stage and is the fastest way to understand what you
  are validating. **Do not re-explain any of them**; link.
- **The synthesis material** — `context/RT/rt_elastic_model.md` transcribes the PR05 and
  O25 equations (~lines 125–129 and 170–173); both PDFs are in `context/RT/`
  (`park2005.pdf`, `pitarch2025.pdf`).

## Status entering M4

M3 is complete: **227 tests pass** (206 + 21 skipped without `$OS_COLOR`), `ruff check`
and `ruff format --check` clean, CI green on Python 3.12 and 3.14. PR #11 is open and its
two Bugbot findings are fixed (record §5.6b).

### The API, by name

```python
from robust.rt import conventions as C   # A_RRS/B_RRS, Rrs_to_rrs, rrs_to_Rrs
                                         # canonical_wave(), bb_w(wave), check_*
from robust.rt import ztt as Z           # rrs_ZTT, Rrs_ZTT, P_bb_sullivan, mu_infinity*
from robust.rt import baselines as B     # rrs_gordon, Rrs_gordon, G1/G2_GORDON
from robust.rt import emulator as E      # fit, fit_l23, features, FEATURES, Emulator,
                                         # EmulatorConfig, LINEAR_CONFIG, DomainBreach,
                                         # DOMAIN_TOL, save, load, load_default
from robust.rt import hybrid as H        # forward, rrs_forward, MODES, DomainWarning
from robust.rt import validation as V    # rrms(truth, pred, axis=None)  -> percent
from robust.rt.data import l23 as L      # load_batch, make_splits, select, npz_reader
from robust.rt.types import IOPs, PhaseParams, Geometry
```

- `H.rrs_forward(iops, phase_params, geometry, wave, mode="hybrid", *, emulator=None,
  check_domain=True)`, and `H.forward(...)` for the same in `Rrs`.
  `mode ∈ {"ztt", "emulator", "hybrid"}`. `emulator=None` loads the **packaged weights**
  (`robust/rt/files/emulator_l23.npz`, 6.5 KB, committed), so `forward` is a *trained*
  model out of the box and needs no `$OS_COLOR`.
- **Additivity holds in `rrs`, not `Rrs`** (the interface is non-linear), so score in
  `rrs`. `mode="ztt"` reproduces `Z.rrs_ZTT` **bitwise**.
- `E.fit(...)` requires an explicit `train=` mask; `E.fit_l23(batch, splits)` trains on
  `scene_train` and records `scene_test` and `scene_test_60`.
- Regenerate weights with `python design/py/train_emulator.py` (~60 s; writes to a temp
  file and verifies a round-trip before replacing anything). Output is **bit-identical**
  run to run — measured, not assumed.

### What M3 measured — the numbers M4 has to contextualise

Full 9960-sample L23 batch, rRMS % in `rrs` space, **scene** split:

| model | train | held-out scenes | held-out @60° |
|---|---|---|---|
| standard Gordon | 7.21 | 7.21 | 9.01 |
| ZTT backbone | 5.95 | 5.93 | 8.11 |
| hybrid, linear (8 par) | 2.57 | 2.54 | 2.48 |
| **hybrid, MLP (417 par)** | **0.30** | **0.30** | **0.32** |

Held-out equals train to two decimals — 417 parameters against ~645k training rows.
**Keep the linear row in every table you produce**: it is what makes the MLP's gain
legible as ~8×, rather than the ~20× that quoting only ZTT implies.

Throughput, jitted, 9960×81, CPU: the hybrid is **≈4.8–5.0× the backbone** (≈16 ms vs
≈3.2 ms; Gordon 0.3 ms). Wall-clock wanders ~20% between runs, so report the *ratio*.

### **The M4 gate has a half that M3 measured as seed-dependent**

The plan's acceptance gate: *hybrid beats standard Gordon on **both** held-out splits
(random 20% scenes; unseen 60° zenith) **and** passes the gradient gate*.

- **Scene split: passes by 24×.** 0.30% against Gordon's 7.21%. Not in doubt.
- **Zenith split: depends on the seed.** Trained on 0°/30° only and scored at the unseen
  60°, MLP(16,16) gives **4.74 / 8.37 / 7.75 / 5.40 / 12.24%** over seeds
  {23, 1, 7, 101, 2024}, against Gordon's **9.01%** there. Four of five pass; one fails.
  The median (7.75%) passes, barely. The committed default (seed 23) passes comfortably
  at 4.74% — which is exactly the trap: *reporting that pass without the spread would be
  the over-claim this milestone exists to prevent.*
- For reference on the same split: **ZTT alone is 8.09%**, beating Gordon with no seed to
  worry about; the **linear** emulator is stable at **6.16%** and beats both.

JXP's standing decision (Q6, prompt 4): 60° extrapolation is a *stretch goal*, and the
emulator will not be used at larger angles without warning the user. That is already
implemented as `Emulator.domain` + `hybrid.DomainWarning`. What M4 must settle is what
the *gate* does with it — **Q7** below, with a recommendation. Do not quietly gate on the
seed that passes.

### PR05 and O25 are not in the repo — and this doc used to be wrong about that

The plan requires scoring "alongside Gordon, PR05, O25". Gordon is implemented; the other
two are **not**. An earlier version of this doc said the O25 form could be reused from
`context/RT/make_rt_elastic_figures.py`. **It cannot — that script contains the Gordon
ladder only**, verified by grep over the whole repo. What is actually available:

**O25 — Pitarch et al. (2025)**, *Remote Sens. Environ.* **329**, 114920. Form (repo
transcription `context/RT/rt_elastic_model.md` ~line 170; paper Eqs. 3–4):

```
Rrs = (Gw0 + Gw1 · ω_bw) · ω_bw  +  (Gp0 + Gp1 · ω_bp) · ω_bp
ω_bw = bb_w/(a+bb),   ω_bp = bb_p/(a+bb)
```

A bivariate quadratic in the **water/particle-split** pseudo-albedos — which maps exactly
onto `IOPs(a, bb_w, bb_p)`, the split this project already keeps explicit. The four `G`s
are **geometry-only** LUTs over (θs, θv, Δφ), wavelength- and IOP-agnostic; phase-function
dependence is implicit in its calibration set's Fournier-Forand functions (PB24). Models
**`Rrs`, not `rrs`**. Stated validity ceiling `Rrs ≤ 0.06 sr⁻¹`.
*Coefficients are **not** in the repo and **not tabulated in the paper** (plots only).*
Sources: `github.com/jaipipor/O25`, NASA HyperCP, EUMETSAT ThoMaS — or re-fit per
geometry, a 4-parameter unweighted least squares, which **the paper itself does on L23**
(its Fig. 3), finding `Gw0`/`Gw1` close to PB24's.
*Transcription trap:* the paper's §2.1 writes `ηb = ω_bw/ω_bp`, contradicting its own
Fig. 2 caption (`ηb = bb_w/bb`, the standard definition and the one consistent with
PR05's `γb = 1 − ηb`). Use the latter.

**PR05 — Park & Ruddick (2005)**, *Appl. Opt.* **44**(7), 1236–1249. Form (repo
transcription ~line 125; paper Eq. 6):

```
Rrs = Σ_{i=1..4} g_i(θs, θv, Δφ, γb) · ω_b^i,     γb = bb_p/bb
```

A 4th-order polynomial in `ω_b` whose coefficients form a **4-D lookup table**. Also in
`Rrs`. *Coefficients are neither in the repo nor printed in the paper* — only ranges
(`g1 ≈ 0.03–0.07`, `g2 ≈ 0–0.3`, `g3 ≈ −0.8–0.2`, `g4 ≈ 0.2–1.0`). The paper points at a
2005 MUMM URL we do not have; POLYMER embeds the LUT. **And L23 is nadir-only with three
solar zeniths, so a re-fit here can populate the θs and γb axes but not θv/Δφ** — a
re-fitted PR05 is a different object from the published one and must be labelled so.
See **Q8**.

Neither is importable from `bing` or `ocpy` — checked, including git history.

### What *is* reusable

`context/RT/make_rt_elastic_figures.py` (246 lines, numpy/scipy, **not** JAX): the L23
locator/loader, four Gordon-ladder fitters (fixed; per-λ `(G1,G2)`; `+G0`; `+Gb·bb_p`), an
`rrms` identical to `validation.rrms`, and three figures. Caveats: it reads **only the 0°
file** (no geometry axis) and its fits are **unweighted**, which the plan warns against.
`context/RT/fig_rrms_ladder.csv` holds its output (`lam,std,quad,const,bbp`) — e.g. at
700 nm 9.04 / 5.74 / 0.35 / 0.37. Gordon-ladder rows only; no PR05/O25 row exists
anywhere in `context/RT/`. BING's L23-fitted Gordon tables
(`bing/data/RT/gordon_coefficients*.csv`, per-λ with errors) are useful as **extra
comparison rows**, not as PR05/O25.

### Gotchas carried forward

1. **`pytest` from the repo root** (`robust` may not be pip-installed).
2. **Score in `rrs`, not `Rrs`** (design §6). PR05 and O25 are defined in `Rrs`, so they
   must be converted with `C.Rrs_to_rrs` before scoring — and say so in the table
   caption, since their coefficients were fitted in `Rrs` space.
3. **`splits.zenith_test` is not a held-out mask for a scene-split fit.** ~80% of its
   samples are training scenes there, so it reads as held-out while being mostly training
   error. `fit_l23` therefore records `scene_test_60`; for true zenith extrapolation,
   train on `splits.zenith_train` via `E.fit`. This bit once.
4. **Per-`B_p`-bin metrics will have little dynamic range.** L23 spans only 1.75× in
   `B_p` (0.0103–0.0180) against the design's ~7× band, so report the bins but do not
   read phase-function generalisation into them.
5. **The gradient gate needs a per-variable step**: `a` 1e-6, `bb_p` 1e-9, `B_p` 1e-8,
   `theta_s` 1e-3, under `jax_x64`, dtype pinned on the **arrays**. Achieved through the
   full hybrid: 2.7e-9, 6.8e-11, 1.6e-7, 2.1e-7. A step larger than the variable can
   leave the physical domain (NaN) — mask non-finite differences rather than letting them
   into a comparison.
6. **`check_domain` is a boundary check**, skipped whenever any input is traced (`jit`,
   or `grad` of a single input). Pass `check_domain=False` in hot loops, and expect a
   `DomainWarning` when scoring a full batch with an emulator trained on a subset — that
   is correct behaviour, not noise (`DOMAIN_TOL` is 1% of the trained span).
7. **`mode="emulator"` is the correction term `Δrrs`, not a standalone model.** The
   design's "learned-only" option needs a *differently trained* network predicting `rrs`
   across four decades — an M4 model to add beside PR05/O25 if it is wanted at all
   (record §5.7).
8. **float32 floor of ~5e-5** from ZTT's quartic (record §4); do not assert tighter
   agreement in float32.
9. **Validate before overwriting a committed artifact.** Both write paths in the repo
   (`train_emulator.py`, `l23.write_fixture`) write to a temp file, verify, then
   `os.replace`. Any new artifact writer in M4 — the metrics table, the figures — should
   follow it.

### Outstanding

- **Equation (8)'s `m1..m16`** — still not received; µ∞ comes from Twardowski & Tonizzo
  (2017). Report as *ZTT with the TT2017 µ∞*; `mu_inf_coeffs=` swaps them in.
- **M2's last commits were never reviewed** — Bugbot's clean pass on PR #10 was at
  `4d6c628`, not M2's final commit.
- **PR #11** is open; if Bugbot adds findings after JXP pushes, fold them into task 5.

> JXP says -- ignore all of these outstanding items

## Prompts

1. Read this doc. Execute the 1st task in the "M4" section below. If you have any
   questions, ask me in the Q&A section below. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task. Check my answers in Q&A. Use Fable if you can.
   Log your work.
3. Read this doc. Execute the 3rd task — the acceptance gate. Use Fable if you can. Log
   your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can. Log your
   work.
5. Thanks for the Notebook.  Can you add a section that:
   - Defines explicitly what O25 is
   - Reminds the reader what "hybrid" is
Use Fable if you can. Log your work.
6. Read this doc. Execute the 5th task — the PR review. Log your work.
7. Read this doc. Execute the 6th task — the prototype hand-off. Log your work.

## M4

### Tasks

1. **Comparison models.** Add **O25** to `robust/rt/baselines.py` (or a sibling module if
   it grows), matching the established convention: the same signature as `forward`,
   ignoring nothing it can legitimately see, and documenting what it is blind to. O25's
   form is fully specified above and its `(ω_bw, ω_bp)` split already lives in `IOPs`.
   Settle the coefficient question first (**Q8**): fetch the published `G` LUTs, or fit
   them per solar zenith on L23's **training** split and label the model *"O25 form,
   refit on L23"* everywhere it appears. A refit must use the train split only — fitting
   a comparison model on the test data would flatter the hybrid's rival, which is the one
   direction of bias nobody would think to check.

   Also, note that JXP has cleared all 3 Outstanding items.

   **PR05** is a judgement call rather than a coding task: see Q8. If it goes in, the same
   labelling rule applies.

   Tests: each model reproduces a hand-checked value; each is differentiable (the same
   harness scores them); and any refit is deterministic and demonstrably trained on the
   train split only.

2. **Validation module.** Fill out `robust/rt/validation.py` (currently `rrms` alone) with
   the design §6 protocol: rRMS **per λ**, **per solar zenith**, **per `B_p` bin**;
   metrics on **both** held-out splits; **throughput**; and the **gradient-correctness**
   check — every model on identical data, in `rrs` space. Then
   `design/py/run_validation.py` regenerates the metrics table and the figures.

   Report the **seed spread** for anything trained, not a single fit: M3 measured that one
   number hides a 2.6× range at an unseen geometry. A `seeds=` argument defaulting to
   several, and a table carrying a median and a range, is the honest shape.

3. **Acceptance gate.** `robust/tests/test_validation.py`. The plan's gate is the hybrid
   beating standard Gordon on **both** held-out splits plus the gradient gate; write it
   per the Q7 answer, and make the *reason* each assertion holds visible in its docstring.
   Keep the committed test cheap (the fixture plus a toy fit) and leave the real run to
   `run_validation.py`. Commit the metrics table and the figures. Update the record: fill
   §6, flip M4 in §1, bump the version.

4. **Notebook.** `notebooks/RT/rt_elastic_coding_5.ipynb` — the M4 explainer, following
   record §8 and the four existing notebooks: **executed** with outputs, degrading
   without `$OS_COLOR`, `sys.path` bootstrap, the house figure style, and the
   CVD-validated categorical set (blue `#0072B2`, vermillion `#D55E00`, sky `#56B4E9`,
   near-black `#1a1a1a`; validate any new colour — a green failed at ΔE 5.4).
   **Render and look at every figure**: doing so has caught a label collision, a clipped
   label, unreadable log ticks, a wrong colour encoding, and two different denominators
   on one axis — once each, in four notebooks.

   The story is the *protocol and its verdict*, not another accuracy boast: which model
   wins where, what the seed spread does to the zenith claim, and what the prototype may
   and may not say.

5. **PR Review.** Fetch the review comments on the open PR and address them. `gh` is not
   authenticated; the public REST API works:
   `curl -sS https://api.github.com/repos/ocean-colour/retrieve-or-bust/pulls/<n>/comments`
   and `.../reviews` — write the response to a file, since `conda run` does not forward
   stdin. Fix the *class* of each defect, not the single instance, and demonstrate the fix
   catches what was reported. If no PR is open, review the diff yourself and say so.

6. **Finally — the prototype hand-off.** The Week-1 prototype is complete. Prepare it
   (JXP issues and runs all git):
   - a short summary of what M0–M4 delivered, the validation numbers **with their
     caveats**, and the known gaps (nadir-only, fixed-FF L23; the 1.75× `B_p` slice; the
     unseen-zenith instability; the missing Equation (8) coefficients; PR05/O25
     provenance);
   - confirm `pytest` is green and the branch is clean;
   - then modify `rt_elastic_coding_prompt_6.md` (M5: beyond the prototype) given what M4
     established.

### Q&A

**Q7 (M3 → M4 hand-off, Claude → JXP). What should the zenith half of the M4 gate
assert?**

The plan wants the hybrid to beat standard Gordon on the unseen-60° split. M3 measured
that outcome as **seed-dependent**: 4.74 / 8.37 / 7.75 / 5.40 / 12.24% over five seeds
against Gordon's 9.01%, so four seeds pass and one fails. Meanwhile `mode="ztt"` beats
Gordon there deterministically (8.09%), and the linear emulator does better still
(6.16%), stably.

You have already said (Q6) that 60° extrapolation is a stretch goal and that we will not
use the emulator at larger angles without warning. Three ways to write the gate:

1. **Gate on the scene split; report the zenith split as a measured limitation**, with
   the full seed spread. Honest, and consistent with your Q6 answer — but it drops half
   of a plan-level criterion, so it is yours to approve.
2. **Gate the zenith split on the median over a fixed seed set** (say five). Keeps both
   halves, makes the claim reproducible, and forces the spread into the report. The
   median passes at 7.75% against 9.01% — with little margin.
3. **Add an out-of-domain policy to `forward` and gate on that.** Your Q6 phrasing was
   "we won't use the emulator at larger angles (**or** will warn the user)" — the first
   clause is a *fallback*, which is not yet implemented. An
   `on_out_of_domain ∈ {"warn", "ztt"}` option would make the hybrid degrade to the
   backbone exactly where M3 showed the emulator to be unreliable, and the gate would
   then pass deterministically at 8.09%, because that is the model actually being used.
   This is the option I would pick if you want both halves kept *and* a defensible claim;
   it is ~20 lines plus tests.

My recommendation: **3, with 1's reporting.** Unless you say otherwise I will implement
the fallback as an *option* (default unchanged, so nothing silently changes), gate the
scene split, and report the zenith split both ways — hybrid-with-fallback and
hybrid-without — with the seed spread. — *Not blocking task 1.*

>A. Yes, I like option 3.  Please implement it.  And, I meant to warn when we extrapolate *beyond* 60 deg.  It should be fine to do anything up to that angle.

**Q8 (M3 → M4 hand-off, Claude → JXP). PR05 and O25: fetch the published coefficients,
or re-fit their forms on L23?**

Neither model's coefficients are in the repo, and neither paper prints them (see above).

- **O25** is cheap either way. Its `G` LUTs are published as code at
  `github.com/jaipipor/O25` (also inside NASA HyperCP and EUMETSAT ThoMaS). Fetching
  needs network access and a licence check; re-fitting is a 4-parameter least squares per
  solar zenith, and the paper itself re-fits on L23. Re-fitting gives a *fair* comparison
  on our data but is no longer the published model, so it must be labelled.
- **PR05** is harder: a 4-D `(θs, θv, Δφ, γb)` LUT that exists only behind a 2005 MUMM
  URL or inside POLYMER. L23 is nadir-only, so we could never populate θv/Δφ ourselves.

My recommendation: **implement O25's form and re-fit it on the train split, labelled
"O25 form, refit on L23"**, and **drop PR05 from the gate**, saying in the report why
(coefficients unobtainable; a nadir-only refit of a BRDF model would be a different
object). If you would rather have the published numbers, say so — and tell me whether I
may fetch from GitHub, since I will not pull external code without your go-ahead.
— *Not blocking task 1: O25's form can be written before the coefficients are settled.*

>A. This is a good recommendation.  Please implement it.

**Q9 (M4 task 1, Claude → JXP). O25 changes what the prototype can claim, and it beats
the hybrid out of distribution.**

Task 1 is done and O25 is much stronger than Gordon. Measured on the full batch, rRMS %
in `rrs`:

| model | train | held-out scenes | held-out @60° |
|---|---|---|---|
| standard Gordon | 7.21 | 7.21 | 9.01 |
| ZTT backbone | 5.95 | 5.93 | 8.11 |
| **O25 form, refit on L23** (12 par) | **0.70** | **0.69** | **0.71** |
| hybrid, MLP (417 par) | 0.30 | 0.30 | 0.32 |

Two consequences:

1. **The hybrid's margin is ~2.3×, not 24×.** Against Gordon it is 24×; against the
   strongest benchmark available it is 2.3× — with 417 parameters against O25's 12. Still
   a win, and on held-out scenes, but a different claim from the one M3's table implies.
2. **At the unseen zenith, O25 wins.** Fitted on 0°/30° only and clamped to 60° (its
   lookup has no 60° row), O25 scores **4.63%** — *deterministically*, beating the
   hybrid's best seed (4.74%), its median (7.75%) and its worst (12.24%), as well as ZTT
   (8.09%) and Gordon (9.01%). So on the split M4's gate is most exposed on, the rival is
   better and more reproducible than we are.

This does not break anything — it sharpens what M4 may say. But it bears on Q7: gating
only on "beats Gordon" now looks weak, since Gordon is the *weakest* thing in the table.
My recommendation: **gate on beating O25 on the scene split** (which the hybrid does,
0.30 vs 0.69) and **report the zenith split as a loss** — plainly, with these numbers, as
M3 reported its own. A prototype that says "we beat the state of the art in distribution
by 2.3× and lose to it out of distribution" is more useful, and more defensible, than one
that beats a 1988 model by 24×.

Also worth your judgement: O25's numbers here are its *best case* — its coefficients were
fitted on our training split with our own metric as the objective. That is deliberate (it
is the fair way to run a rival), but it means the 0.69% is not a claim about the published
O25. — *Not blocking task 2.*

>A. I agree.  Gate on beating O25 on the scene split.

## Next

→ `rt_elastic_coding_prompt_6.md` (M5: beyond the prototype).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-07 (M4 task 6 — the prototype hand-off; **M4 and the Week-1 prototype are done**)

**Delivered** `design/prototype_summary.md` (the one-page reviewer-facing account), record
**v0.16** with M4 flipped to done and a new **§6.10 definition of done**, and a rewritten
`rt_elastic_coding_prompt_6.md` for M5 (66 → ~250 lines). Suite **279 passed** (256 + 23
skipped without `$OS_COLOR`), ruff clean, branch clean apart from these files.

**The prototype passes the gate as amended, and the summary says exactly that.** The
plan asked for "beats standard Gordon on both held-out splits + the gradient gate"; two
amendments were made deliberately and are recorded rather than quietly applied — gate on
beating **O25** on the scene split (Gordon is the weakest model in the table, and ZTT
alone already beats it), and **report** the zenith split rather than gate it (the outcome
is seed-dependent). By that gate it passes.

**The summary is built around what the prototype may *not* claim**, and that is the
deliberate shape: one paragraph for the result, six numbered items for its limits — the
2.3× margin over the modern benchmark rather than 24× over a 1988 one; O25's 0.69% being
its best case on our own training split; geometry extrapolation unresolved and losing to
O25 at an unseen angle; phase-function generalisation untested rather than passed; nadir
only; and µ∞ from TT2017 because the 2018 coefficients are unpublished. A reviewer who
reads only the table would over-claim; a reviewer who reads the list cannot.

**Every number in it is checked, not transcribed.** A script parses the summary's table
back out and compares it against `design/validation/metrics.csv` row by row, recomputes
the two ratios, and runs `pytest` twice to confirm the test counts — all 12 table entries
consistent, ratios 23.8× and 2.3×, counts 279 and 256. Given how many stale numbers this
milestone has produced, a summary written by hand and left unverified was not worth the
risk.

**The M5 hand-off leads with the finding that reframes the next milestone.** M5's agenda
is no longer "extend the model" but "attack the two reasons the margin is 2.3× and not
better": the emulator has never seen a varied phase function or an off-nadir view. The
doc carries the API by name (44 names, all verified by import), the results table with
what each number *forbids*, eight gotchas including the three notebook-discipline rules
this milestone learned the hard way, and the note that `on_out_of_domain="ztt"` will start
firing on every sample the moment multi-angular data arrives — correct behaviour, but
surprising if unexpected.

**Two questions raised, one of them blocking.** **Q10**: M5's entire sketch depends on
data the repo does not have, and the three routes differ by more than effort — commission
HydroLight runs (the only one that tests the phase-function axis), obtain PB24 (settles
geometry soonest, and is O25's own calibration set, so O25 will be at its strongest on
it), or neither, in which case M5 is API work and the two headline gaps stay open. I
recommend PB24 first with the HydroLight specification written in parallel, and I have
said I will not fetch external data or code without a go-ahead. **Q11**: what the hybrid
should do off-nadir before it has been trained there, now that the domain fix flags every
such view.

### 2026-08-07 (M4 task 5, second round — PR #12, on a fix I had just made)

**PR #12 is open** on the finished M4 work, at exactly our HEAD (`e552295`), and Bugbot
found **one Medium issue** — in `train_emulator.py`, on a change the self-review had made
two days earlier. Fixed, demonstrated, recorded in §6.9. Suite **279 passed**, ruff clean.

**The finding.** My earlier fix moved the architecture guard from *after* training, where
it inspected the trained emulator, to *before* it, where it inspects the module constant
`SHIPPED_CONFIG` — so that it fails in milliseconds instead of after two minutes and so
`--dry-run` exercises it. Bugbot pointed out what that traded away: **the guard now
validates a proxy for the artefact rather than the artefact**. If a training loop ever
passed a different config while the constant still read `EmulatorConfig()`, the script
would write weights whose architecture does not match `load_default()`. Correct, and a
regression I introduced while fixing something else — the second time this milestone that
one of my fixes became the next finding.

**Both properties are now kept**: `check_architecture()` runs early on the constant *and*
inside `write_weights()` on the emulator actually being serialised. Demonstrated with the
exact scenario reported — constant untouched, loop handing over a linear emulator: the
early check passes, `write_weights` refuses, the destination stays byte-identical, no
temporary file survives, and a correct-architecture emulator still writes.

**The class, and a sweep for it.** A guard belongs next to the artefact it guards and must
inspect the artefact, not a stand-in. I checked the repo's other guards against that rule
and they hold: `write_fixture` loads its snapshot back through the real reader,
`emulator.load` compares the file's own recorded feature list, the packaged-weights test
reads the committed file, and `run_validation.py`'s `--quick` refusal inspects the actual
arguments. No second instance to fix.

What I take from it: "fail faster" is a real improvement, but it is not free, and I traded
a correctness property for it without noticing. The tell was available at the time —
the check stopped naming the object it was protecting.

### 2026-08-07 (new 5th prompt — a section defining O25 and reminding what the hybrid is)

**Added §1 "The two models being compared, defined"** to
`notebooks/RT/rt_elastic_coding_5.ipynb`, renumbering the rest 2–7 and fixing the one
internal cross-reference. Now 24 cells. (The PR review I had already run is now prompt 6,
so it is banked and logged below.)

The notebook argued about O25 for six sections without ever saying what it *is*. §1 now
does, and separates three things that were tangled together:

- **The model**: the bivariate quadratic, with every symbol defined, and the reason the
  water/particle split is the whole idea rather than a detail — the two return light
  through different volume scattering functions, so they should not share a coefficient.
- **Its provenance and standing**: Pitarch et al. 2025, inheriting L11's form, calibrated
  on PB24, in NASA HyperCP and EUMETSAT ThoMaS and operational in OLCI Collection 4 —
  which is *why* it and not Gordon is the benchmark that decides whether our hybrid is
  worth anything.
- **What our version is not**: the published `G` tables exist only as plots in the paper,
  so ours are refit on L23's train split, with no θv/Δφ axis because L23 is nadir-only.
  Stated in the section rather than left to a footnote, along with the consequence that
  its numbers are its best case.

And a plain reminder of the hybrid — unfitted ZTT backbone plus a bounded 417-parameter
correction — with the contrast spelled out: **12 fitted numbers with no phase-function
input against 417 on top of unfitted physics that has one**. A comparison table lays the
two side by side on what is fitted, what each can see, and what each is blind to.

The code cell makes both concrete on a single water body: O25's actual refit coefficients
(showing how little they move with zenith — `Gw0` 0.0587 → 0.0525 across 60°), the two
branch albedos, and the hybrid term by term. That last line is the useful one: δ comes out
at a few percent, so the physics sets the answer and the learned half adjusts it.

**Three self-inflicted delays, and the practice that now prevents them.** This section
took four execution attempts, all my fault, and the failures were instructive because each
defeated the check I had just added:

1. **A split f-string.** I broke an expression across two adjacent literals
   (`f"...{sum(p.size for p in "  f"tree_leaves(...))}"`) — a syntax error. I also
   mis-read the harness's "exit code 0" as success when the notebook was in fact
   unexecuted. *Fix:* verify `execution_count`, not an exit status; and **AST-parse every
   code cell in the build step**.
2. **A Python list used as a JAX index.** `wave[bands]` with `bands` a list raises at
   runtime. The AST check passed it happily — syntax is not semantics. *Fix:* **run the
   new cell standalone before paying for the full eight-minute notebook**, which takes
   about five seconds and would have caught both of the first two.
3. **A number that contradicted its own prose.** I wrote that δ is "a few percent" and the
   cell printed **+8%, +22%, +10%**. The cause was choosing an *invented* IOP set to
   illustrate a data-driven model: it sat far from anything the emulator trained on. The
   cell now takes a real L23 scene from the committed fixture, prints whether that sample
   is in-domain (it is), and reports `|δ|` over the whole fixture as measured rms **5.6%**
   and max 27% — so the claim is a range someone can check rather than an impression.

The third is the one I keep repeating across milestones, and it is now clear that the
mechanism is always the same: prose written before the measurement, then not re-read
against it. The standalone-cell smoke test is cheap enough that it should be the default
for every notebook edit from here.

### 2026-08-07 (M4 task 5 — the review pass: no open PR, and five real findings)

**PR #11 was merged** on 08-05 and its two findings were already fixed in M3's task 4. No
PR is open — so, per the task, I reviewed the diff myself. The scope is
`git diff 6dcaf63..HEAD`: **every commit of M4 has never been seen by an automated
reviewer**, since #11 was reviewed at `6dcaf63` and everything since is mine.

Suite **278 passed** (255 + 23 skipped without `$OS_COLOR`), ruff clean, record §6.9
added, counts refreshed, M4 at tasks 1–5 of 6. Five findings, each reproduced before
being fixed and each now pinned by a regression test **proven to fail when the fix is
reverted**:

1. **High — a slightly off-nadir view passed the domain check while the emulator's
   output was meaningless.** `cos_theta_v` is constant in L23, so its trained span is
   zero, and the check scaled the excursion by the feature's own *value*: 5° of sensor
   zenith looked like a 0.4% excursion, inside `DOMAIN_TOL`. But the standardisation
   divides that same excursion by `_STD_FLOOR`, so the network saw **−3.8e5**, every
   `tanh` saturated, and the correction collapsed from a spectrum spanning [−0.10, +0.27]
   to a **flat +0.046 at all 81 wavelengths** — no warning, no fallback, for any θ_v in
   (0°, 8.1°]. Two parts of the same module disagreed about what "outside the domain"
   means. They now share the denominator, so the check measures the excursion in the
   units the network actually sees. This is the most useful thing the review found: the
   domain machinery exists precisely to prevent silent nonsense, and here it was
   permitting it.
2. **Medium — the two domain predicates disagreed on NaN.** `out_of_domain` and
   `out_of_domain_mask` implement one predicate twice, and `excess > tol` is False for
   NaN — so the *mask*, which the fallback policy acts on, said "in domain" while the
   host check said "out". A single NaN in `a` is what an inversion overshoot produces.
3. **Medium — both committed CSVs were silently malformed**, and I found this by trying
   to *consume* one: I wrote a script to cross-check §6.2's table against `metrics.csv`
   and it crashed on `float('train')`. The model names contain commas ("O25 form, refit
   on L23", "hybrid, MLP") and I had joined fields by hand, so `metrics.csv` carried four
   fields under a three-field header and the ladder's header expanded seven names into
   ten columns. Nothing raised; a consumer would have mis-labelled every column. Now
   written through the `csv` module, with tests asserting both files parse to their
   promised columns.
4. **Low — `gradient_report` silently replaced the caller's geometry with nadir**,
   discarding `theta_v`/`dphi`. Invisible on L23; would have certified gradients at the
   wrong geometry the moment M5 goes off-nadir. A spy model pins it.
5. **Low — `gradient_report`'s `steps` dict.** A missing key raised from inside a
   closure; an extra key was worse — it reported **0.0**, "perfect agreement", for a
   variable that is never perturbed. `throughput(repeats=0)` divided by zero.

**A note on how the first two were found.** I put two Fable reviewers on the diff with
different lenses (model code; tests and scripts) and told both to probe rather than read,
and to prove each claim before reporting it. The off-nadir finding came with the measured
standardised value, the collapsed δ range, and the exact 8.1° boundary — which is why I
could confirm it in one run rather than arguing about it. I reproduced every finding
myself before touching the code; two of the reported items I judged docstring-level rather
than bugs (`bp_bin_labels`' equal-count claim under heavy ties, and `Rrs_o25` returning a
cross product for a column-shaped `theta_s`) and recorded them as caveats instead.

**The second reviewer (tests and scripts) found five more, and they were about my
tests rather than my code.** Suite now **279 passed** (256 + 23 skipped); record §6.9
extended.

6. **A committed artefact was stale and nothing would have noticed.** The Gordon column
   of `rrms_per_wavelength.csv` aggregated to 9.57% where `metrics.csv` said 7.21%, and
   the figure drawn from it overstated Gordon's blue-end error by ~2×. My regeneration
   this session had already corrected it — but silently. So the *check* is now a test:
   RMS each per-λ column and require the pooled scalar back. It needs no trust in the
   model code at all, being pure consistency between two files from one run, and it is
   the kind of test I should have written when I first wrote two artefacts that must
   agree.
7. **The gate's margin was seed luck and its docstring overstated it.** At 400 steps the
   hybrid scored 0.454–0.575% across seeds against O25's 0.578% — the worst seed passing
   by **0.6%** — while my docstring claimed "~0.53% against O25's ~0.9%". Now 800 steps
   (0.376–0.419%, a 27% margin) and the assertion demands `hybrid < 0.9 × O25`, so a 3%
   shrink of the correction fails where it used to pass.
8. **A test named for slicing passed when slicing was removed.** My
   `test_score_models_slices_one_evaluation_per_model` used models with *uniform*
   relative error, so every subset scored identically. Fixed by making the error differ
   between the halves.
9. **My own fix from finding 5 opened a hole.** Reporting `0.0` where a model genuinely
   ignores a variable also hides a perturbation that was never applied — with the `B_p`
   offset dropped, every gradient test passed, including for ZTT, which does depend on
   it. The gate now also asserts each entry is non-zero for the hybrid.
10. **The gate fixture's dtype depended on test order.** `jax_x64` is function-scoped and
    `gate_fit` module-scoped, so whichever test asked first decided float32 vs float64 —
    0.5468% vs 0.5628%, half the old margin. The gradient test now trains its own fit.
11. **Four script guards**: `--quick` silently overwrote the committed artefacts with
    300-step numbers; a missing `$OS_COLOR` produced a raw h5py traceback naming a
    repo-relative file that never existed; the PNGs bypassed the temp-file rule the
    docstring claimed for *all* outputs; and `train_emulator.py`'s architecture guard
    fired only after two minutes of fitting and was skipped by `--dry-run`. All four
    fixed and demonstrated.

**One reported finding was wrong, and checking mattered.** The review held that nothing
protects the emulator against train/test contamination. True of the gate — but mutating
`fit` to train on the held-out scenes fails **six** tests, four of them the
`test_emulator.py` checks that pin the standardisation and the domain to the *train*
rows. The protection lives upstream of the gate, which is where it belongs. I verified
this by mutation before deciding not to act on the recommendation.

**What I take from this pass.** Findings 7–10 are all mine, all in *tests*, and all of
the same shape: an assertion that could not fail, or could fail for the wrong reason. The
code review found two real bugs; the test review found four places where I had written
something that looked like a check. That asymmetry is worth remembering — I have been
treating a green suite as evidence, and a green suite is only as good as the weakest
assertion in it.

One consequence to carry into M5: with the domain fix, **any** off-nadir geometry now
trips the check, so `on_out_of_domain="ztt"` will fall back to the backbone there. That is
correct — the emulator has never seen an off-nadir view — and it means the policy starts
earning its place as soon as M5's HydroLight runs vary the sensor angle.

### 2026-08-06 (M4 task 4 — the notebook, whose own output caught my throughput claim)

**Built** `notebooks/RT/rt_elastic_coding_5.ipynb` — 21 cells, 3 figures, executed on the
full batch (~8 min), degrading to the committed fixture without `$OS_COLOR`. Record §6.8
added, M4 flipped to "tasks 1–4 of 6". Suite still **269 passed**, ruff clean.

The subject is the protocol and its **verdict**, not another accuracy claim. §6 is two
explicit lists — *may say* and *may not say* — because that is the actual deliverable of a
validation milestone, and it is the section a reader will quote.

**Three figures, each doing work no table does.** The per-λ ladder for all five models,
which shows at a glance that the gap that matters is to O25 and not to Gordon. **O25
fitted both ways**, which is the fairness argument made visible: the unweighted objective
is *better in the blue* and far worse in the red, so the paper's own choice would have
handed us a 3.8× flattering comparison. And the unseen-60° dot-and-range plot where only
the MLP has a range — the figure the milestone turns on.

**The notebook demonstrates the two gradient traps side by side**, which a table cannot:
O25's `theta_s` derivative disagrees by **7e-1** *on* a table node (30°) and by **2e-10**
between them (45°), and its `B_p` derivative reads exactly 0 because the model genuinely
ignores the phase function. Both would look like gate failures to anyone who had not been
told, and L23's three angles are exactly the nodes.

**And once again the notebook's own output contradicted my prose.** I had written "the
hybrid costs ~6× the backbone"; the cell measured 4.53×. Checking four runs: 6.01, 4.95,
4.53, 4.45 — so **the ratio itself wanders ~35%**, and my standing line that "wall-clock
wanders but the ratio is the reproducible part" was an overstatement I had already copied
into the record. Both are now corrected to state the range and to say that the *ordering*
is what holds. That is the fifth notebook in a row where executing the prose against the
code caught something; the pattern is reliable enough that I now expect it rather than
being surprised.

### 2026-08-06 (M4 task 3 — the acceptance gate, and an interaction between two answers)

**Built** the gate in `robust/tests/test_validation.py` (now 20 tests), the two committed
figures, and the record's **§6** (bumped to **v0.15**, M4 flipped to "tasks 1–3 of 6",
module index refreshed). Suite **269 passed** (246 + 23 skipped without `$OS_COLOR`),
ruff clean. Artefacts in `design/validation/`: `metrics.md`, `metrics.csv`,
`rrms_per_wavelength.csv`, and two PNGs.

**The gate, written per your answers.** Q9 → gate on beating **O25** on the scene split;
the Gordon and ZTT comparisons stay as the plan's floor, labelled in their docstrings as
the weaker statement. Q6/Q7 → the zenith half is *reported*, not gated. Mutation-tested:
with the emulator's contribution zeroed — the "zero-output emulator" the original
Gordon-only gate would have waved through — the two gate assertions fail and the rest
pass.

**The interaction worth recording, because it defeats the fix chosen for it.** Q7
selected option 3 partly on my argument that the fallback would make the zenith gate
deterministic: an emulator trained on 0°/30° would degrade to the backbone at 60°. But
your accompanying clarification set the sanctioned envelope at **0–60°** — "it should be
fine to do anything up to that angle" — so the fallback deliberately does **not** fire
there. Measured: it triggers on **0 of 9960** samples, and the 0/30-trained hybrid scores
identically (12.24% for the worst seed) with the policy on or off. The fallback is still
right beyond 60°; it simply cannot rescue that half of the gate. So the gate stops at the
scene split, and a test pins the inertness so the reasoning cannot quietly rot. I would
have missed this if I had implemented option 3 and assumed it did what I proposed it for.

**Figures**, both rendered and inspected: the per-λ ladder for all five models on
held-out scenes (which shows at a glance that O25, not Gordon, is the line to beat), and
the unseen-60° comparison as a dot-and-range plot where only the MLP has a range. Two
rounds of fixes — the ladder's legend sat on top of two curves and moved below the axes;
and the linear emulator's row fell between both label branches and rendered as a bare dot
with no number, because its seed range collapses to a point. Every row now carries its
value, and a seed-independent model says so.

**§6 of the record states the headline as it actually is**: the hybrid beats the state of
the art by **2.3×** in distribution and **loses to it out of distribution** (O25 4.63%
deterministic at the unseen 60°, against the hybrid's 4.74–12.24% seed range). It also
records that O25's 0.69% is its *best case*, since its coefficients were fitted on our
training split with our own metric as the objective — the fair way to run a rival, and a
fact that has to travel with the number.

### 2026-08-06 (M4 task 2 — the validation protocol, and Q7's fallback)

**Built** the design §6 protocol in `robust/rt/validation.py`, the runner
`design/py/run_validation.py`, and the Q7 fallback in `robust/rt/hybrid.py`, with 16
tests in a new `robust/tests/test_validation.py`. Suite **265 passed** (242 + 23 skipped
without `$OS_COLOR`), ruff clean. Artefacts written to `design/validation/`
(`metrics.md`, `metrics.csv`, `rrms_per_wavelength.csv`).

**Q7 implemented as option 3, plus your correction.** `hybrid.rrs_forward` /`forward`
now take `on_out_of_domain ∈ {"warn", "ztt"}`; `"ztt"` zeroes the learned correction
outside the accepted range so the model degrades to the backbone exactly where M3
measured the emulator to be unreliable. Default unchanged, so no existing number moves.

Your clarification — *warn beyond 60°, anything up to it is fine* — turned out to change
the semantics, not just a threshold. The check was comparing against **the angles a
given emulator happened to see**; you are describing **the span the project sanctions**.
So `SUPPORTED_THETA_S = (0.0, 60.0)` is now a project constant, and the zenith feature is
judged against it while the other six keep using the trained range, where "outside what I
learned" genuinely does mean unreliable. For the shipped weights nothing changes (they
already saw 0–60°); what changes is that a 0/30-trained fit may now be asked for 60°
without complaint. `out_of_domain(..., theta_s_limits=None)` still asks the other
question — "is *this fit* extrapolating?" — which is the one the research runs want.

**The fallback is built on a traceable mask, not the warning check**, and that was a
deliberate design decision rather than a detail: the warning needs concrete values and is
skipped under `jit`, so a policy hung off it would lapse silently the moment anyone
compiled the model — the hot path. A test pins it by comparing jitted against eager for
the same function (5e-7 apart, XLA fusing `rrs_ztt + 0.0` in float32) against what a
lapsed policy looks like (2e-1, five orders of magnitude larger). Bitwise equality was
the wrong instrument here and my first check reported a false failure because of it.

**Two things the protocol found in code I had just written:**

1. **A model that ignores a variable was scored as infinitely wrong.** O25 has no
   phase-function input, so `d/dB_p` is exactly zero — and my ratio-based gradient report
   turned 0/0 into `inf`, i.e. it converted a *documented blind spot* into a gate
   failure. Now an exact zero on both sides reports 0.0, and a non-finite difference
   (a step that left the physical domain) still reports `inf`. Both are pinned.
2. **O25 is not differentiable at its own table nodes.** Its coefficient lookup is
   piecewise linear in `θs`, so `jax.grad` takes one one-sided slope at a node while the
   central difference averages both: measured **69%** disagreement at 30°. That is
   inherent to any LUT model, but it matters here because **L23's three angles *are* the
   nodes**, so a naive gradient check on L23 geometry lands on one every time. The
   protocol evaluates at 45° instead, where all four variables agree to ≤3e-9, and both
   `o25_coefficients` and `gradient_report` now say so.

**The measured protocol** (full batch, rRMS % in `rrs`, held-out scenes): Gordon 7.21,
ZTT 5.93, O25 0.69, hybrid-linear 2.54, hybrid-MLP 0.30. Per zenith the backbone
degrades (4.26 / 4.67 / 8.11) while O25 and the hybrid do not (0.68/0.69/0.71 and
0.30/0.30/0.32). Per `B_p` bin nothing varies much — as expected, and the table says
plainly that a 1.72× span cannot speak to phase-function generalisation. Throughput:
hybrid **6.0× ZTT**; O25 0.18×; Gordon 0.08×. Gradient gate: every model, every variable,
≤5e-9 against a 1e-6 tolerance.

**Two smaller fixes.** The throughput table reported ZTT as *0.72× itself*, because I
timed the reference twice and wall-clock wanders ~20%; the ratio now comes from the same
loop, so the reference reads exactly 1.00. And my own "write via a temp file" class fix
from the PR review had a latent defect: `mkstemp` creates files at **0600**, and
`os.replace` preserves that, so every artefact it produced would have been committed
unreadable to anyone else. All three writers now restore the permissions a plain `open()`
would have given under the process umask.

### 2026-08-06 (M4 task 1 — O25, and it is a much harder benchmark than Gordon)

**Built the O25 comparison model** in `robust/rt/baselines.py` (+274 lines) with 22 tests
in `robust/tests/test_baselines.py`. Suite **249 passed**, ruff clean. Q7/Q8 were
unanswered, so I proceeded on my Q8 recommendation: implement O25's form, refit it on the
train split, label it a refit, and leave PR05 out. Your clearing of the three Outstanding
items is noted — nothing below depends on them.

**API**: `Rrs_o25` (the primitive — O25 is defined in `Rrs`, the reverse of Gordon's
arrangement), `rrs_o25` for scoring, `o25_coefficients(theta_s)` for the lookup,
`fit_o25(...)` to refit, `O25_L23_REFIT` as the embedded table, `O25_RRS_CEILING`.

**The result, and it reframes the milestone.** Full batch, rRMS % in `rrs`:

| model | train | held-out scenes | held-out @60° |
|---|---|---|---|
| standard Gordon | 7.21 | 7.21 | 9.01 |
| ZTT backbone | 5.95 | 5.93 | 8.11 |
| **O25 form, refit on L23** (12 par) | **0.70** | **0.69** | **0.71** |
| hybrid, MLP (417 par) | 0.30 | 0.30 | 0.32 |

Twelve fitted numbers — four coefficients at each of three zeniths — get within **2.3×**
of our 417-parameter hybrid, and O25's train and held-out figures are identical, so there
is nothing to dismiss as overfitting. Per zenith on held-out scenes: O25 0.68 / 0.69 /
0.71 against ZTT's 4.26 / 4.67 / 8.11. **And at the unseen 60°** — O25 refit on 0°/30°
only, its lookup clamped to the 30° row — O25 scores **4.63% deterministically**, beating
the hybrid's *best* seed (4.74%) and its median (7.75%). Raised as **Q9**: the honest
headline for M4 is now "beats the state of the art in distribution by 2.3×, loses to it
out of distribution", not "beats a 1988 model by 24×".

**The fitting objective was the one real judgement call, and it went against us.** O25 is
linear in its four coefficients, so the fit is a closed-form `lstsq` — deterministic, no
seed, no learning rate. But the *weighting* is a choice, and it is worth 4×: the paper's
unweighted least squares in `Rrs` gives 2.5–2.7% rRMS, while weighting each residual by
`1/Rrs` — matching the relative metric we score everything with — gives 0.70%. An
unweighted objective optimises the bright blue and abandons the dark red, which is exactly
the failure mode our metric exists to expose. Reproducing the paper's choice would have
made our own hybrid look four times better than a fair comparison allows, so the fair fit
is the default and the paper's is behind `weighted=False`. This is the same BING lesson
that shaped the emulator's loss, arriving from the opposite direction: applied to a rival.

**PR05 is deliberately absent**, documented in the module docstring: its coefficients are
a 4-D `(θs, θv, Δφ, γb)` lookup the paper does not print and the repo does not hold, and
L23 is nadir-only so a refit could never populate the two sensor-geometry axes — it would
be a different object wearing the same name. Recorded as a gap rather than approximated.

**Provenance is the load-bearing test.** The fitted table is embedded as a constant, the
way `conventions.BB_W_L23` is, so O25 runs in CI without `$OS_COLOR`; and
`test_fit_o25_reproduces_the_embedded_table` refits on the real data and requires the
constants back (max difference 1.24e-7, the table's own 8-figure rounding), so they cannot
drift from the code that made them. `fit_o25` also *requires* an explicit `train=` mask —
fitting a rival on the test split would flatter the model it is compared against, which is
the one direction of bias nobody thinks to check.

**I mutation-tested the tests rather than trusting a green run**, since they were drafted
by a subagent: swapping `Gw1` with `Gp0` in the lookup fails 6 of them; making `fit_o25`
ignore its train mask fails 3, including the provenance gate; dropping the per-sample
coefficient broadcast fails 4. All 22 pass on the restored code. Also pinned as
*deliberate* rather than accidental: O25 ignores `wave` (its coefficients are
λ-independent by construction) and ignores `phase_params` entirely (the phase function is
baked into its calibration set) — both bitwise, both with the reason in the docstring.
