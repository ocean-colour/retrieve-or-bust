# Elastic RT Coding — Prompt 6 (M5: Beyond the Week-1 prototype)

## Goals

Extend the forward model past the L23-only prototype along the axes it could not
exercise: the **particle phase function** and the **full BRDF**. M4 measured exactly
where the prototype's claims stop, and those stopping points are M5's agenda — this doc
is no longer a sketch.

The single most important thing M4 established: **the prototype's margin over the modern
benchmark is 2.3×, not 24×, and at an unseen geometry the benchmark beats it.** M5's job
is to attack the two reasons for that — the emulator has never seen a varied phase
function or an off-nadir view — rather than to improve a number on L23.

**The data question is now settled** (Q10): JXP has PB24 on disk, and it turns out to
carry *both* missing axes, not just geometry — see *The reference data, confirmed* below.
M5 is therefore a data-and-training milestone, not a specification-writing one.

## Claude

### Skills

Consider `.claude/skills/` — `critical-partner` (M5 is where "we already beat the
baseline" becomes tempting), `grill-me`, plus the `dataviz` and `code-review`
conventions M0–M4 settled.

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements* (git by JXP; `ocean14`;
CPU-only JAX; reuse `ocpy`/`bing`; BING test conventions; pytest-gated; Fable; log).
Still **forward-model only** and **differentiable**; the inversion remains a separate
design.

The rhythm M0–M4 settled: **ask in Q&A** (numbered, phrased so work continues without an
answer; check for answers before each task), then after the code a **notebook**, a
**PR-review pass**, and a **hand-off edit to the next prompt doc**.

## Context

Read before coding:

- **The prototype summary** — [`design/prototype_summary.md`](../../design/prototype_summary.md).
  Start here: it is one page and it says what may and may not be claimed.
- **Implementation record** — `design/rt_elastic_implementation.md` (**v0.16**); §5 is
  M3, §6 is M4 — §6.2 results, §6.3 the gate as amended, §6.9 four review passes,
  §6.10 the definition of done. §7 is the M5 stub to fill.
- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M5.
- **Design** — `design/rt_elastic_model.md` §4.2 (phase-function parameterization), §7–§8.
- **The five notebooks** — `notebooks/RT/rt_elastic_coding_{1..5}.ipynb`. Notebook 4 §1
  specifies the hybrid stage by stage; notebook 5 §1 defines O25 and restates the
  hybrid. **Do not re-explain either**; link.

## Status entering M5

**The Week-1 prototype is complete.** 279 tests pass (256 + 23 skipped without
`$OS_COLOR`), ruff clean, CI green on py3.12 and py3.14, PR #12 open with its one
finding fixed.

### The reference data, confirmed

PB24 is on disk at **`$OS_COLOR/SD/v5`** — 28 GB, 10 001 files, inspected 2026-08-08.
Contents as measured, not as assumed:

| | |
|---|---|
| realisations | **5000**, each a separate file, in two spectral resolutions |
| `SD_OLCI_no_R_NNNN.nc` | 12 OLCI bands, 400–753 nm, ~270 kB each (**1.3 GB total**) |
| `SD_hyp_no_R_NNNN.nc` | 451 bands, 350–800 nm at 1 nm, ~5.8 MB each (**27 GB total**) |
| geometry, every file | `theta_s` (10: 0–60 by 10°, then 70, 80, 87.75) × `phi` (13: 0–180 by 15°) × `theta` (10: 0–60 by 10°, then 70, 80, 87.5) = **1300 geometries** |
| labelled parameters | three scalars per file: `C` chl, `N` NAP, `Y` a_cdom(440). **They do not determine the IOPs** — see consequence 5 |
| IOPs, per band | `aw aph ag aNAP`, `bw bph bNAP`, `bbw bbph bbNAP` — components, so `a`, `b_p`, `bb_p`, `B_p` are all derivable |
| AOPs | **`rrs` and `Rrs`** on the full 4-D grid, plus `Q`, `Kd Ku Ko Kod Kou Knet KLu`, **`mu_d`, `mu_u`, `mu_tot`**, `R` |
| sidecar | `classes_Rrs_OLCI_v5_20240214.mat` → `i_classes` (5000×1) uint8, **12 optical water classes** |
| provenance | attrs: Jaime Pitarch, CNR-ISMAR, created 2024-02/03 |

Six consequences, each measured — two of them found by an independent audit that refuted
my first reading of the data (see the log entry).

1. **PB24 varies the particle phase function** — this reverses Q10's own premise. The
   phytoplankton backscatter ratio `bbph/bph` is flat in λ within a file (within-file
   max/min ≤ 1.0008) but takes a **unique value per realisation spanning 0.0010–0.0358
   (~30×)**, apparently with a hard floor at 0.001; `bbNAP/bNAP` spans 0.0100–0.0200,
   apparently uniform on that interval. The bulk `B_p` spans **6.2× across realisations**
   (0.0033–0.0202 as per-file means), against L23's 1.7× and the design's ~7× band. A
   **held-out-`B_p` split is constructible from data already on disk** — M4's untested
   axis becomes measurable without commissioning anything. Two caveats that must travel
   with any result: these are one *family* (Fournier-Forand), so this tests our §4.2
   parameterization rather than generalisation across VSF families; and `bbph/bph` is
   correlated with chlorophyll (corr(log, log) = −0.65) without being a function of it,
   so a `B_p` split is **not** independent of the water type it holds out.
2. **The Lee-2002 surface transfer fails off-nadir, and this is a finding, not a
   footnote.** My first spot check — file 0001, 443 nm, nadir, θs=0 — agreed to 0.2%, and
   I wrote that our `conventions` map onto this dataset. Across all 1300 geometries the
   median deviation is **15–24%** and the worst is **23×**, and it is a clean function of
   view angle — median |dev| by θv, over all θs, φ and λ in file 0001:

   | θv | 0° | 10° | 20° | 30° | 40° | 50° | 60° | 70° | 80° | 87.5° |
   |---|---|---|---|---|---|---|---|---|---|---|
   | median \|dev\| | 1.8% | 1.9% | 2.6% | 6.2% | 13.0% | 24.7% | **45.7%** | 83.6% | 162% | 275% |

   `Rrs/rrs` falls from 0.530 at nadir to 0.160 at θv = 87.5°; restricted to θv, θs ≤ 60°
   the median is a tolerable 3.8–4.9% but the maximum is still 113–124%.
   `A = 0.52`, `B = 1.7` is a *nadir* constant. Since `forward` returns `Rrs`
   through exactly this map, **M5 must make the surface transfer geometry-aware** — and
   PB24 hands us `Rrs`, `rrs` and `Q` on the same grid to fit it with. Scoring is
   unaffected: we score in `rrs`, which is tabulated.
3. **`mu_d`, `mu_u`, `mu_tot`, `Q` and the K's are tabulated**, so ZTT's internals can be
   validated against HydroLight *component by component* instead of only end-to-end. The
   TT2017 µ∞ substitution (the standing Eq-(8) gap) becomes directly measurable.
4. **Scale needs a decision.** OLCI is 5000 × 1300 = **6.5 M spectra**; hyperspectral is
   the same count at 451 bands. Neither fits the "load the whole batch" habit M1–M4 was
   built on. See **Q12**.
5. **The IOP space is richer than `C`, `N`, `Y`** — those three scalars are labels, not
   generators. Independent per-realisation draws include `S_g` (0.0103–0.0253 nm⁻¹),
   `S_NAP` (0.0072–0.0277), `aNAP*(440)` (**230×**), `aph*(440)` (4.2×) and `bph(440)/C`
   (**~370×**). Only `Y` is exact: it *is* `ag(440)` to 0.1%. Good news for coverage; it
   also means a split on `C` is not a split on the IOPs. One wrinkle for split design:
   normalised `aph` shapes appear drawn from a **finite library of measured spectra** and
   are reused verbatim across files (two files matched to RMS 0.000), so held-out
   realisations can still share a shape with the training set.
6. **The tails are extreme, and `rrs` can be exactly zero.** Sampled ranges: `C`
   0.011–938 mg m⁻³, `N` 0.00085–1332 g m⁻³, `Y` 0.0021–74.5 m⁻¹, max `rrs` 0.397. And in
   2 of 15 files audited, `rrs` is **exactly 0** at θv = 87.5°, θs ∈ {70, 80}, λ ≥ 721 nm
   (float32 underflow — `Rrs` is non-zero there). Our metric divides by truth, so **the
   loader must filter or the `rrms` denominator must guard**; unguarded this is an `inf`
   in the first score we compute. Argues for the Q14 default of not training on grazing
   angles.

### The API, by name

```python
from robust.rt import conventions as C   # Rrs_to_rrs, rrs_to_Rrs, canonical_wave, bb_w
from robust.rt import ztt as Z           # rrs_ZTT, Rrs_ZTT, P_bb_sullivan, mu_infinity*
from robust.rt import baselines as B     # rrs_gordon, Rrs_gordon; rrs_o25, Rrs_o25,
                                         # fit_o25, o25_coefficients, O25_L23_REFIT
from robust.rt import emulator as E      # fit, fit_l23, features, FEATURES, Emulator,
                                         # EmulatorConfig, LINEAR_CONFIG, save, load,
                                         # load_default, SUPPORTED_THETA_S, DOMAIN_TOL,
                                         # out_of_domain, out_of_domain_mask
from robust.rt import hybrid as H        # forward, rrs_forward, MODES, DomainWarning,
                                         # OUT_OF_DOMAIN_POLICIES
from robust.rt import validation as V    # rrms, rrms_per_wavelength, group_rrms,
                                         # bp_bin_labels, throughput, gradient_report,
                                         # score_models, markdown_table, FD_STEPS
from robust.rt.data import l23 as L      # load_batch, make_splits, npz_reader, write_fixture
from robust.rt.types import IOPs, PhaseParams, Geometry
```

`PhaseParams` is **the extension point** and was designed for this milestone: adding
fields defaults-`None` changes neither `forward`'s signature nor any call site (record
§3.2). `design/py/{train_emulator,run_validation}.py` regenerate the weights and the
metrics; both verify before replacing anything.

### What M4 measured, and what each number forbids

| model | held-out scenes | unseen 60° (trained on 0/30) |
|---|---|---|
| standard Gordon | 7.21 | 9.01 |
| ZTT backbone | 5.93 | 8.09 |
| **O25 form, refit on L23** | **0.69** | **4.63** (deterministic) |
| hybrid, MLP | **0.30** | 4.74 / 8.37 / 7.75 / 5.40 / 12.24 (per seed) |

Three consequences M5 must carry:

1. **The benchmark is O25, not Gordon.** Any new result is measured against 0.69% on
   held-out scenes, and against O25 *refit on the same new data* — refitting a rival on
   the training split is the fair way to run it, and it costs one `lstsq`.
2. **At an unseen geometry we lose.** The hybrid's answer there depends on its random
   seed across a 2.6× range; O25 is deterministic and better. This is the prototype's
   main open risk, and PB24's 1300 geometries per realisation are what resolve it —
   note that O25 is at its strongest on its own calibration set, which is the honest
   direction for the comparison to be biased.
3. **The `B_p` and BRDF axes are untested, not passed.** L23 spans 1.7× in `B_p` against
   the design's ~7×, with a single fixed Fournier-Forand shape and a nadir view. M4's
   flat per-bin accuracy is *not* evidence of generalisation, and the notebook says so.
   PB24 turns both axes from untestable into measurable (5× in `B_p`, full BRDF) — so
   the M5 result to aim for is a **held-out-`B_p` split**, not a better L23 number.

### The one piece of M5 that is already built

`Emulator.domain` + `hybrid.on_out_of_domain` were added at M4 (record §6.5). The
sanctioned envelope is `SUPPORTED_THETA_S = (0, 60)` for the solar zenith, and **any**
off-nadir view is now flagged — so with `on_out_of_domain="ztt"` the hybrid degrades to
the backbone on **every sample of PB24**. That is a correct default, not a bug, and it is
also the M5 baseline to beat: *the backbone's own numbers on PB24 are what the hybrid
currently delivers there.*

Per **Q11**, the plan is now decided: **retrain with the view angles as live features and
let the domain widen from the data.** `features()` already carries `cos_theta_v` and
`cos_dphi`; they are constant in L23, which is exactly why the domain check flags any
off-nadir view. Training on PB24 makes them live, and the fallback then fires only
outside the *new* envelope — which **Q14** now sets at **0–70° in both zeniths**, holding
PB24's 80/87.75° shell out as the extrapolation test. So `SUPPORTED_THETA_S` becomes
`(0, 70)` and gains a view-angle counterpart, and for the first time the fallback will
fire on real data rather than on nothing (contrast M4, where it triggered on 0 of 9960
samples because the envelope reached exactly as far as the data did).

### Gotchas carried forward

1. **`pytest` from the repo root**; score in `rrs`, never `Rrs` (the interface is
   non-linear, so additivity holds only below the surface).
2. **A comparison model is refit on the train split only**, and labelled as a refit.
   The fitting *objective* matters: O25's unweighted fit scores 2.6% against 0.69%
   weighted, so using the paper's objective would flatter us ~4× (record §6.7).
3. **Report seed spreads for anything trained.** One number hid a 2.6× range at M4.
4. **The gradient gate needs per-variable steps** (`a` 1e-6, `bb_p` 1e-9, `B_p` 1e-8,
   `theta_s` 1e-3) under float64 with the dtype pinned on the arrays; mask non-finite
   differences. And **evaluate lookup-table models between their nodes** — O25's
   coefficients are piecewise linear in `theta_s`, so at a tabulated angle autodiff and
   central differences disagree by ~70% for a perfectly correct model.
5. **`gradient_report` returns 0.0 for a variable a model genuinely ignores** — that is
   agreement, not failure; but assert it is *non-zero* for models that should depend on
   it, or a perturbation that is never applied reads as perfect.
6. **Validate the artefact you are about to write, next to where you write it** — not a
   proxy for it (PR #12), and before it replaces a known-good file (PR #11).
7. **Notebook discipline**: AST-parse every cell and run a new cell standalone before an
   ~8-minute execution; check `execution_count` afterwards, not the exit code; and
   re-read every quantitative sentence against the output beside it. That last one has
   caught an error in five consecutive notebooks.
8. **`B_p` spectra, not scalars** — L23's varies with λ within a scene, and `features()`
   reads a trailing axis matching `n_wave` as a spectrum. PB24's bulk `B_p` also varies
   with λ (median 3% within a file, up to 17%) even though each *component* ratio is
   flat, because the phyto/NAP mix shifts across the spectrum. Do not collapse it to a
   scalar.
9. **PB24 is 5000 files, not one archive** — every load is 5000 `open_dataset` calls
   unless cached. Cache the assembled arrays to `.npz` once and validate the cache the way
   `l23.write_fixture` does (temp → verify → `os.replace`), or the loader becomes the
   bottleneck in every run.
10. **PB24 is O25's own calibration set.** Any comparison on it favours O25, and that must
    be said in the same sentence as the number. It is the honest direction for the bias to
    run, and it is not a reason to skip the comparison.
11. **`rrs` is exactly 0 at some grazing geometries** (θv = 87.5°, θs ∈ {70, 80},
    λ ≥ 721 nm — float32 underflow), so `rrms`, which divides by truth, returns `inf`
    unless the loader filters or the metric guards. Guard *and* report the count dropped;
    a silent filter is the same failure as a silent truncation.
12. **One geometry is not the dataset.** My nadir spot check of the surface transfer
    agreed to 0.2% and was wrong by up to 24× three angles away. On a 4-D grid, check the
    corners before believing the centre — consequence 2 exists because an independent
    audit did.

### Outstanding

- **PR #12** is open; if Bugbot adds findings after JXP pushes, fold them into the review
  task.
- **Equation (8)'s `m1..m16`** remain unpublished; µ∞ comes from Twardowski & Tonizzo
  (2017) and `mu_inf_coeffs=` swaps them in. Report as *ZTT with the TT2017 µ∞*. **New:**
  PB24 tabulates `mu_u`/`mu_d`/`mu_tot`, so the size of this substitution is now
  measurable rather than merely disclosed.
- **The published O25/PR05 coefficients** are still not in the repo; every O25 number is
  a refit. PR05 needs a 4-D `(θs, θv, Δφ, γb)` LUT — PB24 is the first dataset we hold
  that actually spans those axes, so implementing PR05 becomes possible if it earns its
  place.
- ~~**PB24 is not in the repo**~~ — resolved 2026-08-08; see *The reference data,
  confirmed*.

## Prompts

0. Read this doc. Execute the 0th task in the "M5" section below — the answers task. Ask
   your questions in the Q&A section first. Use Fable if you can. Log your work.
1. Read this doc. Execute the 1st task in the "M5" section below — the scoping task. Ask
   your questions in the Q&A section first. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task. Check my answers in Q&A. Use Fable if you can.
   Log your work.

## M5

### Tasks

0. **Answers.** ✅ done 2026-08-08. Q10 and Q11 are answered; PB24 is confirmed on disk
   and characterised (*The reference data, confirmed*, above), and the tasks below are
   rewritten around it. The finding that moved the most: **PB24 varies the particle
   backscatter ratio per realisation (~30× in `bbph/bph`, 6.2× in bulk `B_p`)**, so it
   covers the phase-function axis as well as the geometry axis, and Q10's option 1
   (commissioning HydroLight runs) is no longer on M5's critical path. Three new
   questions raised — **Q12** (which spectral resolution and how to subsample), **Q13**
   (does PB24 replace L23 as the training set, and do we reship the weights), **Q14**
   (how far the sanctioned envelope should now reach).

1. **More answers.** I have answered Q12, Q13, and Q14.  Please read them and revise the tasks below accordingly.  Use Fable if you can.

   ✅ done 2026-08-08. All three answers are folded into task 2's scope below, and each
   is now a *constraint* rather than an option: **OLCI files with a documented
   geometry-subsampling knob** (Q12); **both datasets kept, L23 as an independent held-out
   dataset, `load_default()` unchanged** (Q13); **train 0–70°, hold out 80–87.75°** (Q14).
   What that settles, beyond the three bullets: the loader has a fixed target shape rather
   than an open one, M5 gains a *cross-dataset* number it would not otherwise have had,
   and the grazing angles that carry the exactly-zero `rrs` samples (gotcha 11) fall
   outside training by construction — Q14's answer and that defect happen to cancel.

   One thing I checked rather than assumed while revising, because it changes the order of
   the work: **the nadir surface transfer is on the path of every model we have.**
   `grep` for `Rrs_to_rrs`/`rrs_to_Rrs` finds it in `baselines.py:126` (Gordon),
   `baselines.py:302` (`rrs_o25`), `ztt.py:939`, `hybrid.py:270` and — least obviously —
   `emulator.py:978`, where it converts L23's `Rrs` into the *training targets*. So on
   PB24, with the nadir constants in place, a 46%-at-60° interface error would contaminate
   the benchmark, the backbone and the emulator's own targets alike, and none of it would
   be the models' fault. That is why the surface-transfer bullet is marked "sequence
   early" — task 2 should put it before the benchmark, not after.

   Raised: **Q15** (what M5's acceptance gate should be), so task 2 can sequence against a
   target rather than invent one.

2. **Sequence this milestone and fill §7.** The data question is settled, so this task is
   scoping, not coding: its deliverable is the task list below rewritten as a sequence
   with a test gating each step, plus a filled §7 of the record. Sequence by what each
   step *unlocks*, not by size, and say plainly which steps are blocked and on what.

   The scope, as it now stands:

   - **A PB24 loader** — `robust/rt/data/pb24.py`, mirroring `l23.py`'s contract
     (`load_batch`, `make_splits`, a committed small fixture so CI runs real numbers with
     no data mount). Per **Q12** it reads the **OLCI files** (12 bands, 1.3 GB) and
     exposes a **documented geometry-subsampling knob** for cheap runs; keep the reader
     factored so the hyperspectral files open through the same path later. New problems
     `l23.py` never had: 5000 files rather than one archive, 6.5 M spectra, and a 4-D
     geometry grid to flatten. Derive `a`, `b_p`, `bb_p`, `B_p` from the components; carry
     `i_classes` from the `.mat` as a stratification variable (counts are very unbalanced:
     84 to 1042 of 5000); apply the **Q14** angle window (θs, θv ≤ 70°) as the default
     load, with the 80–87.75° shell loadable on request as the extrapolation set; and drop
     or guard the exactly-zero `rrs` samples (gotcha 11), reporting the count. Note the
     interaction: those zeros live at θv = 87.5°, so the Q14 window removes them from
     training anyway — the guard is for the extrapolation set, where they remain.
   - **Splits that test what M4 could not.** At least three: held-out **realisations**;
     held-out **`B_p` band** (the phase-function axis, now measurable); held-out
     **geometry** — and per **Q14** that last one is now defined by the data rather than
     chosen: train 0–70°, test the 80/87.75° shell. It is the direct successor to M4's
     unseen-60° split, which is the half we lost, so expect it to be the hardest number in
     the milestone and treat a bad result there as information. The `B_p` split is the one
     that converts M4's item 4 from "untested" to a number — but state its confound:
     `B_p` correlates with chlorophyll, so holding out a `B_p` band also shifts the water
     type, and `aph` shapes are reused across realisations, so a realisation split does
     not fully separate shapes either.
   - **A geometry-aware surface transfer.** `conventions.Rrs_to_rrs`/`rrs_to_Rrs` hard-code
     Lee's nadir `A = 0.52`, `B = 1.7`, which PB24 shows is wrong by a median 15–24% and up
     to 23× off-nadir, and by **45.7% at the median at θv = 60°** — inside the Q14 training
     window (consequence 2). `forward` returns `Rrs` through that map, so this is an
     API-visible correctness bug the moment we leave nadir — and PB24 gives `Rrs`, `rrs`
     and `Q` on one grid, so `A(θv, θs, Δφ)` is directly fittable. Keep the nadir constants
     as the default so every M4 number stays reproducible; gate the new path with a test
     that fails on the old one at 60°. **Sequence this before the benchmark**, not merely
     early: `grep` shows the map on the path of Gordon (`baselines.py:126`), O25
     (`baselines.py:302`), ZTT (`ztt.py:939`), the hybrid (`hybrid.py:270`) *and* the
     emulator's training targets (`emulator.py:978`) — so run out of order, it would
     charge every model with an interface error none of them committed. PB24 supplies
     `rrs` directly, which also lets the emulator's targets bypass the map entirely.
   - **Benchmark first, model second.** Before retraining anything, score ZTT, Gordon and
     **O25 refit on PB24** across the full BRDF. That fixes the target and tells us
     whether the analytic backbone alone already degrades off-nadir — and PB24 is O25's
     own calibration set, so this is the comparison at its least flattering to us. What
     M5's gate should then be is **Q15**.
   - **Retrain the emulator on PB24** with `theta_v` and `dphi` live (**Q11**'s answer),
     re-run the §6 protocol on all three splits, and report **seed spreads** (gotcha 3).
     Per **Q13** this **adds** a model rather than replacing one: ship the new weights
     alongside as `files/emulator_pb24.npz` and leave `load_default()` on the L23 model,
     so every M4 number stays reproducible from a clean checkout. Assert what retraining
     is supposed to achieve rather than assuming it: after training, an off-nadir view
     inside 0–70° must **stop** being flagged by the domain check, and 80°+ must still be.
   - **The cross-dataset check** (**Q13**'s second half) — score the PB24-trained emulator
     on **L23, without refitting**. This is the strongest generalisation evidence available
     to us and it costs one evaluation: a model trained on one HydroLight campaign, judged
     on another. It also fixes the rule for promotion — `load_default()` moves only if the
     PB24 model wins on **L23's own held-out split** too, which is a harder test than
     winning on its home data.
   - **Validate ZTT's internals against HydroLight** — PB24 tabulates `mu_d`, `mu_u`,
     `mu_tot` and `Q`, so the µ∞/µd substitution can be checked directly instead of only
     end-to-end. Cheap, and it either retires the standing Eq-(8) caveat or quantifies it.
   - **Promote `PhaseParams`** from `B_p` to the ZTT backward-VSF parameterization
     (design §4.2), as *additional* fields defaulting to `None` so no signature changes.
   - **Freeze the `forward` API** as the shared engine for training-data generation and
     the separately designed inversion.

   Still off the critical path, but not dead: **commissioned HydroLight runs** with a
   non-Fournier-Forand VSF family. PB24 varies the FF *parameter*, so it tests our §4.2
   parameterization; only a second family tests generalisation *across* families. Worth a
   specification if M5's held-out-`B_p` result is good, worth nothing if it is not.

   Sequence these by what each one *unlocks*, not by size, and gate each with a test as
   M0–M4 did. Say plainly which tasks are blocked on data that does not exist yet.

3. **(to be written by task 1)**

### Q&A

**Q10 ✅ answered — option 2, PB24. (M4 → M5 hand-off, Claude → JXP.) What reference data
will M5 actually have?**

Everything in M5's sketch depends on data the repo does not hold, and the three options
differ by more than effort:

1. **Commission HydroLight runs** varying the phase function and the view geometry. The
   only option that tests what the prototype cannot — but it is a real cost and a
   turnaround, and someone has to specify the grid. If this is the route, the useful
   thing I can do first is write that specification: which shapes, which angles, how many
   IOP realisations, and what the held-out split would be.
2. **Obtain PB24.** It exists, it is multi-angular, and it is O25's calibration set — so
   it settles the geometry axis quickly. It does *not* settle the phase-function axis
   (its shapes are prescribed Fournier-Forand), and running our model against a rival's
   calibration set needs saying out loud in any comparison.
3. **Neither yet** — in which case M5 is limited to the API work (promoting
   `PhaseParams`, freezing `forward`) plus whatever can be done with L23, and the two
   headline gaps stay open. Worth doing, but it should not be described as extending the
   model's validated range.

My recommendation: **2 then 1** — PB24 first because it closes the geometry gap soonest
and is the axis where we currently lose to O25, with the HydroLight specification written
in parallel so the phase-function runs can be commissioned early. Tell me which, and
whether I may fetch PB24 or the published O25 coefficients from their repositories; I
will not pull external data or code without your go-ahead.
— *Blocking: task 1's output depends on the answer, though the API work does not.*

>A. I have downloaded the PB24 data.  It is in `$OS_COLOR/SD/v5`. I am going to have you confirm.

**Confirmed, 2026-08-08** — 10 001 files, 28 GB, opened and characterised in *The
reference data, confirmed* above. Two things you should know because they are better than
what the question assumed:

- **The premise of option 2 was wrong in our favour.** I wrote that PB24 "does not settle
  the phase-function axis (its shapes are prescribed Fournier-Forand)". It is one FF
  *family*, but the FF *parameter* is drawn per realisation: `bbph/bph` spans
  **0.0010–0.0358 (~30×)**, a unique value in every file sampled, spectrally flat within
  a file. Bulk `B_p` spans **6.2×** against L23's 1.7×. So PB24 covers **both** headline
  gaps, and option 1 (commissioning runs) drops off M5's critical path — it would now
  only answer the narrower question of generalising across VSF *families*.
- **`rrs` is tabulated directly**, along with `Q`, `mu_d`, `mu_u`, `mu_tot` and the K's —
  so we score in the design's space without inverting the surface, and ZTT's internals
  become checkable against HydroLight component by component.
- **And one thing that is not good news:** the dataset shows our `Rrs ↔ rrs` conversion is
  a nadir approximation that breaks off-nadir (consequence 2). It costs M5 a task, and it
  would have silently corrupted every above-surface number at the new geometries.

No external fetch was needed and none was made.

**Q11 ✅ answered — retrain. (M4 → M5 hand-off, Claude → JXP.) What should the hybrid do
off-nadir before it has been trained there?**

M4's domain fix means **any** off-nadir view is now flagged, so
`on_out_of_domain="ztt"` degrades the hybrid to the analytic backbone for every sample
of any new multi-angular dataset — correct, but it means the hybrid contributes nothing
there until retrained. Three options once geometry data exists: keep the fallback and
report the backbone's numbers; retrain with the view angles as live features and let the
domain widen naturally; or treat off-nadir as a separate model. The second is the
obvious one, but it changes what "the emulator" means, so it is worth your view before I
build it. — *Not blocking task 1.*

>A. Let's try retraining

**Taken as decided.** `cos_theta_v` and `cos_dphi` are already in `FEATURES` — constant in
L23, which is precisely why the domain check flags every off-nadir view — so retraining on
PB24 makes them live and the domain widens from the data with no API change. Two
consequences worth stating before they surprise us: the fallback stops firing wherever
training data reached, so **the envelope becomes an empirical object rather than a
declared one** (hence **Q14**), and "the emulator" comes to mean a PB24-trained model,
which is **Q13**. I will keep `on_out_of_domain="ztt"` as the default policy throughout —
what changes is where it triggers, not what it does.

**Q12 ✅ answered — OLCI, with a subsampling knob. (Task 0 → task 1, Claude → JXP.) Which
spectral resolution, and how much of PB24 do we actually load?**

M1–M4 were built on "load the whole batch" — L23 is 9960 samples. PB24 at OLCI resolution
is 5000 × 1300 = **6.5 M spectra × 12 bands** (~310 MB as float32, still tractable);
hyperspectral is the same count at **451 bands** (~11.7 GB, not tractable whole). The
choice interacts with the benchmark, because O25 is band-agnostic but our `features()` is
not.

My plan unless you say otherwise: **build the loader on the OLCI files and train there**,
with a documented geometry-subsampling knob for cheap runs, and treat the hyperspectral
set as a *separate later check* on a subsample of realisations — it answers "does the
correction interpolate across λ" which OLCI's 12 bands cannot. The fixture committed for
CI would be a handful of OLCI realisations, as `l23.py` does.
— *Not blocking: I will build the loader OLCI-first and keep the reader factored so the
hyperspectral files open through the same path.*

>A. Yes, that's fine.  Build the loader on the OLCI files and train there, with a documented geometry-subsampling knob for cheap runs.

**Taken as decided.** The knob will be an explicit argument with a default that loads
everything inside the Q14 window, so a cheap run is a visible choice at the call site
rather than a hidden sample — and whatever it drops gets reported, per the no-silent-caps
rule that M4's review settled on. The hyperspectral files stay a later question ("does the
correction interpolate across λ", which 12 bands cannot answer); the reader will be
factored so they open through the same path when we want them.

**Q13 ✅ answered — keep both. (Task 0 → task 1, Claude → JXP.) Does PB24 replace L23, and
do we reship the weights?**

Once the emulator trains on PB24, "the emulator" means something different from what M4
validated, and `load_default()` currently returns the L23 model that every M4 number was
measured on.

My recommendation: **keep both, and use L23 as an independent held-out *dataset*.** Train
on PB24, then score the PB24-trained model on L23 without refitting — a cross-dataset
number is the strongest generalisation evidence available to us, and it is free. Ship the
new weights alongside as `emulator_pb24.npz`, and leave `load_default()` pointing at the
L23 model until the PB24 model beats it *on L23's own held-out split too*. That keeps
M4's committed metrics reproducible instead of silently restating them.
— *Not blocking: this is reversible, and I will not repoint `load_default()` without your
say-so.*

>A. Ok, keep both and use L23 as an independent held-out dataset.

**Taken as decided.** So M5 ships two emulators and one default: `files/emulator_l23.npz`
stays what `load_default()` returns, `files/emulator_pb24.npz` arrives beside it, and the
promotion rule is written down before we know the answer — the PB24 model takes the
default only if it also wins L23's own held-out split. Writing the rule first is the point:
after the numbers land, any rule we choose will look like it was chosen to fit them.

**Q14 ✅ answered — train 0–70°, hold out 80–87.75°. (Task 0 → task 1, Claude → JXP.) How
far should the sanctioned envelope reach?**

`SUPPORTED_THETA_S = (0, 60)` exists because you called 60° "a real stretch goal". PB24
tabulates θs at 70, 80 and **87.75°**, and θv to **87.5°** — near-horizon geometries where
the physics is extreme (in file 0001, `rrs`(443) spans **10.7×** across the 1300
geometries) and which no ocean-colour sensor observes.

My recommendation: **train on and sanction 0–70° in both zeniths, and hold 80° / 87.75°
out as a deliberate extrapolation test** rather than as training data — so the envelope
stays somewhere we would defend, and the grazing angles measure how the model fails rather
than teaching it to. Say if you would rather train on everything PB24 offers.
— *Not blocking: 0–70° is the default I will proceed with, and it is one constant.*

>A. Yes, train 0-70deg and hold out 80-87.75deg.

**Taken as decided.** `SUPPORTED_THETA_S` becomes `(0, 70)` and gains a view-angle
counterpart, and the 80/87.75° shell becomes the geometry *test* set rather than unseen
territory — which is a real gain over M4, where "unseen 60°" had no data behind it beyond
L23's three zeniths. Two side effects, both good: the exactly-zero `rrs` samples sit at
θv = 87.5°, so they leave the training path by construction (the guard is still needed for
the extrapolation set); and the fallback policy now has something to fall back *at* —
beyond 70° it fires on real data instead of on nothing, which is the case M4 could never
exercise.

**Q15 (task 1 → task 2, Claude → JXP). What is M5's acceptance gate?**

M4's gate was amended twice by you, and both amendments were right, so I would rather agree
this one before the numbers exist than after. The candidates, given three splits:

1. **Gate on the realisation split only** — the direct analogue of M4 (beat O25 refit on
   PB24), reporting the other two. Safest, and the weakest statement.
2. **Gate on the realisation split *and* the held-out-`B_p` split**, report the geometry
   split. The `B_p` split is *why* M5 exists — it converts the prototype's "phase-function
   generalisation untested" into a measured claim — so gating it is what makes the
   milestone mean something. The geometry split stays reported, on M4's own reasoning:
   extrapolation was seed-dependent there, and 80–87.75° is a harder extrapolation than
   60° was.
3. **Gate all three.** Honest, and I think unwise: it stakes the milestone on grazing-angle
   extrapolation that we have already lost once, and a failed gate would say nothing about
   whether the phase-function work succeeded.

My recommendation: **2**. Note the asymmetry it accepts — PB24 is O25's calibration set, so
we are proposing to gate on beating a rival *on its home data*, which is the hard direction
and the one worth claiming. — *Not blocking: task 2 will sequence against option 2 and mark
the gate provisional until you say otherwise.*

## Next

→ M5's own hand-off, when it closes.

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

### 2026-08-08 (Task 1 — Q12–Q14 folded in; the surface transfer moves ahead of the benchmark)

**What I did.** Read JXP's answers to Q12 (OLCI files, with a documented
geometry-subsampling knob), Q13 (keep both datasets; L23 as an independent held-out
dataset) and Q14 (train 0–70°, hold out 80–87.75°), and revised task 2's scope so each is a
constraint rather than an option. Marked the three questions answered with what I take them
to commit us to, and raised **Q15** — M5's acceptance gate — so task 2 sequences against a
target instead of inventing one. Doc only; no code touched.

**The one thing I checked rather than assumed.** Before deciding where the surface-transfer
work sits in the order, I grepped for the users of the nadir map: `baselines.py:126`
(Gordon), `baselines.py:302` (`rrs_o25`), `ztt.py:939`, `hybrid.py:270`, and
`emulator.py:978` — that last one converting L23's `Rrs` into the emulator's *training
targets*. `fit_o25` also fits in `Rrs` while we score in `rrs`, so the map sits between a
rival's fit and its score. Off-nadir on PB24 that means a 45.7%-at-60° interface error
would be charged to every model at once, including the benchmark, and none of it would be
theirs. So the bullet moved from "sequence early" to **"sequence before the benchmark"**,
with a reason that can be checked rather than a preference. PB24 tabulating `rrs` also lets
the emulator's targets bypass the map entirely — a second, independent reason to do it
first.

**Interactions between the three answers, which is where the value was.** Q14's window and
gotcha 11 cancel: the exactly-zero `rrs` samples live at θv = 87.5°, so training never sees
them and the guard is only needed for the extrapolation set. Q14 also gives the fallback
policy something to fall back *at* — beyond 70° it will fire on real data, the case M4
could never exercise (there, the sanctioned envelope reached exactly as far as the data, so
the fallback triggered on 0 of 9960 samples). And Q13 turns "which model do we ship" into a
rule I could write down *before* the numbers exist: the PB24 model takes `load_default()`
only if it also wins L23's own held-out split. Writing it now is the point — after the
numbers land, any rule will look chosen to fit them.

**What I learned.** Sequencing is a technical claim, not a preference, and it is checkable:
"this belongs before that" was worth one `grep`, and the `grep` moved the item and changed
the reason from aesthetics to arithmetic. The other thing, continuing from task 0: three
short answers from JXP were worth more than their length because of how they *interact* —
the useful work of an answers task is not transcription but working out what the
combination forecloses.

### 2026-08-08 (Task 0 — answers folded in; PB24 confirmed, and it covers both gaps)

**What I did.** Read JXP's answers to Q10 (PB24 is downloaded, in `$OS_COLOR/SD/v5`,
"I am going to have you confirm") and Q11 ("Let's try retraining"), confirmed the data by
opening it, and rewrote this doc's Goals, Status, Tasks, Gotchas, Outstanding and Q&A
around what the data actually contains. No repository code was touched; nothing was
fetched from outside.

**What the confirmation found.** 10 001 files, 28 GB: 5000 realisations in two spectral
resolutions (12 OLCI bands, 400–753 nm; 451 bands, 350–800 nm at 1 nm), each on a
10 × 13 × 10 grid of θs × φ × θv = 1300 geometries, plus a `.mat` sidecar giving 12
optical water classes per realisation. Files carry IOP *components* (`aph ag aNAP`,
`bph bNAP`, `bbph bbNAP`, and the water terms), both `rrs` and `Rrs`, and — unexpectedly —
`Q`, `mu_d`, `mu_u`, `mu_tot` and seven K's.

**The finding that changed the milestone.** Q10 asserted that PB24 "does not settle the
phase-function axis (its shapes are prescribed Fournier-Forand)". I wrote that from the
paper's description and it is wrong in the direction that matters: the FF *parameter* is
drawn per realisation. `bbph/bph` is flat in λ within a file but takes a unique value in
every file sampled, across **0.0010–0.0358, a factor ~30**; `bbNAP/bNAP` spans 2×; bulk
`B_p` spans **6.2×** against L23's 1.7× and the design's ~7× band. So the phase-function
axis M4 recorded as *untested* is measurable from data already on disk, a
**held-out-`B_p` split** is constructible, and commissioning HydroLight runs leaves M5's
critical path — it now answers only the narrower question of generalising across VSF
*families*, since PB24 is one family throughout.

**The audit, and what it refuted.** A Fable agent re-tested six of my claims on 50
pseudo-random files (different indices from mine), instructed to prefer refutation. It
confirmed the grid (1300 geometries, bit-identical across files), the phase-function
finding (with a hard floor at 0.001 and a 30× span rather than 25×), and the absence of
NaNs. It **refuted two claims**, both mine:

1. **The surface transfer.** I had checked Lee-2002 at *one* geometry — nadir, θs=0 —
   found 0.2% agreement, and wrote that our `conventions` map onto this dataset. Across
   all 1300 geometries the median deviation is **15–23%** and the worst **24×**; `Rrs/rrs`
   runs from ~0.525 at nadir to ~0.18 at θv = 87.5°, and even inside θ, θs ≤ 60° the
   maximum is 65–121%. `A = 0.52`, `B = 1.7` is a nadir constant. Since `forward` returns
   `Rrs` through it, this is an API-visible bug the moment we leave nadir; it is now an
   early M5 task, fittable from PB24's own `Rrs`/`rrs`/`Q`.
2. **`C`, `N`, `Y` are labels, not generators.** I described them as the generating
   parameters; in fact `S_g`, `S_NAP`, `aNAP*`(230×), `aph*`(4.2×) and `bph/C`(~370×) vary
   independently, and only `Y` is exact (it is `ag(440)`). Two consequences for splits:
   splitting on `C` does not split the IOPs, and `aph` shapes come from a **finite library
   reused verbatim across files**, so held-out realisations can share a shape with the
   training set.

I then re-verified the surface-transfer refutation myself on two files rather than take it
on trust, and it holds: the deviation is monotone in view angle (1.8% at nadir → 45.7% at
60° → 275% at 87.5°), so the table in consequence 2 is measured twice, by different code.

The audit also found `rrs` **exactly zero** at some grazing geometries (float32 underflow at
θv = 87.5°, θs ∈ {70, 80}, λ ≥ 721 nm) — our `rrms` divides by truth, so the first score
we compute would have been `inf` — and that bulk `B_p` is *not* spectrally flat (median 3%
within a file, up to 17%), contradicting a line I had already written into gotcha 8. All
four corrections are in the doc; the confirmations are marked as such.

**What I learned.** Three things, all about my own habits. First, a premise written into a
question can be wrong, and one nobody re-checks propagates into the plan — the "prescribed
Fournier-Forand" line came from the paper's prose and cost the milestone a commissioning
task it does not need. *Confirm the data before scoping around it*, which is exactly what
JXP asked for. Second — the recurring one — **one geometry is not the dataset**: my nadir
spot check agreed to 0.2% and was wrong by 24× three angles away, and I generalised from it
in writing before anyone measured the rest. On a 4-D grid, check the corners. Third, the
scale here breaks the "load the whole batch" pattern M1–M4 was built on (6.5 M spectra,
5000 files per load), and that is a loader-design constraint, not an afterthought — hence
Q12 and gotcha 9.

**Raised:** Q12 (OLCI vs hyperspectral, and the subsampling policy), Q13 (does PB24
replace L23 as the training set — I propose keeping L23 as an independent *dataset* test
and not repointing `load_default()`), Q14 (the sanctioned envelope, now that the data
reaches 87.75°). All three carry a default so task 1 proceeds without answers.

## Logs
