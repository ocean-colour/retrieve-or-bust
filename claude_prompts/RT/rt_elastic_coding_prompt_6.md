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
   so a `B_p` split is **not** independent of the water type it holds out. For the
   **bulk `B_p`** a split would actually use, the correlation is weaker — **−0.49** over
   600 realisations (task 5) — and the span is wider than first recorded: **12.4×**
   (0.0025–0.0315), against the 6.2× seen in ~50 files here.
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
   validated against HydroLight *component by component* instead of only end-to-end —
   **partly**. `ztt.py` exposes `mu_d`, `mu_infinity` and `mu_infinity_tt2017` and nothing
   else in that family, and PB24's `mu_u`/`mu_tot` are near-surface AOPs while µ∞ is the
   *asymptotic* average cosine — different quantities even for a perfect model. So `mu_d`
   is directly checkable; the TT2017 µ∞ substitution is **not**, without an asymptotic
   proxy whose assumptions we would have to state. Task 13 carries the narrowed version.
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
                                         # load_default, SUPPORTED_THETA_S, DOMAIN_TOL
                                         # NB: out_of_domain / out_of_domain_mask are
                                         # Emulator *methods*, not module exports
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
10. **PB24 is O25's own calibration set — but the bias runs the other way for the O25 we
    can fit.** The *published* O25 is favoured on its home data, and that must be said in
    the same sentence as any number. Our `fit_o25`, however, has **no view-angle axis**
    (`baselines.py:374`), so off-nadir it handicaps the benchmark rather than flattering
    it. Both halves must appear together, or the comparison misleads in whichever
    direction the reader assumes. Task 8 exists to remove the second half.
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
  (2017) and `mu_inf_coeffs=` swaps them in. Report as *ZTT with the TT2017 µ∞*. **New,
  and narrower than first claimed:** PB24 tabulates `mu_d`, which ZTT computes, so *that*
  becomes checkable; its `mu_u`/`mu_tot` are near-surface quantities and do not correspond
  to ZTT's asymptotic µ∞, so the Eq-(8) gap is **not** sized by simple comparison
  (consequence 3, task 13).
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
3. Read this doc. Execute the 3rd task — a second wavelength grid in `conventions`. Use
   Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the PB24 loader. Check my answers in Q&A. Use
   Fable if you can. Log your work.
5. Read this doc. Execute the 5th task — the three splits. Use Fable if you can. Log your
   work.
6. Read this doc. Execute the 6th task — extend the validation toolkit. Log your work.
7. Read this doc. Execute the 7th task — the geometry-aware surface transfer. Use Fable if
   you can. Log your work.
8. Read this doc. Execute the 8th task — give O25 a geometry-indexed table. Use Fable if
   you can. Log your work.
9. Read this doc. Execute the 9th task — the PB24 benchmark. Use Fable if you can. Log
   your work.
10. Read this doc. Execute the 10th task — make the envelope per-model. Log your work.
11. Read this doc. Execute the 11th task — retrain the emulator. Check Q15 and Q16 first.
    Use Fable if you can. Log your work.
12. Read this doc. Execute the 12th task — the cross-dataset check. Log your work.
13. Read this doc. Execute the 13th task — ZTT's internals against HydroLight. Log your
    work.
14. Read this doc. Execute the 14th task — promote `PhaseParams`. Log your work.
15. Read this doc. Execute the 15th task — freeze the `forward` API. Log your work.
16. Read this doc. Execute the 16th task — notebook, review, hand-off. Log your work.

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

2. **Sequence this milestone and fill §7.** ✅ done 2026-08-08. The sequence is **tasks
   3–12** below and the record's **§7** is filled (record now **v0.17**). The scope bullets
   that used to live here are gone deliberately — they would have become a second,
   divergent copy of the same list.

   **This sequence is the second draft.** The first was reviewed against the code by an
   adversarial agent and was wrong in ways worth recording: it assumed the L23 machinery
   would accept a second dataset (it will not — `check_wave` rejects any grid but L23's
   81 bands, and `gradient_report` *raises* unless the perturbed variables are exactly
   `{a, bb_p, B_p, theta_s}`); it treated the O25 benchmark as "one `lstsq`" when
   `fit_o25` has **no view-angle axis at all**, which would have made the milestone's gate
   a straw man; it scheduled a cross-dataset check that cannot pass by construction; and
   it repeated a claim of mine that was simply false — see the correction under task 7.
   Two prerequisite tasks were missing entirely. The findings are in §7.11 of the record.

   How it is ordered: **3 → 4 → 5** are prerequisites (make the machinery dataset-agnostic,
   then load, then split), **6** unblocks three later tasks at once, **7 → 8 → 9** builds an
   honest benchmark before any model is trained, and **10 → 11 → 12** is the model work.
   Tasks **13 and 14** need only the loader and can run in any slack; **15 goes last**
   because 7, 10, 11 and 14 can each still move the signature it freezes.

---

3. **Make `conventions` accept a second wavelength grid.** ✅ **done 2026-08-08.**
   *Unlocks: the loader — nothing about PB24 can be validated until this lands.*

   **What landed.** `WaveGrid` + a `GRIDS` registry (`canonical`/`l23` → L23's 81 bands,
   `olci` → PB24's 12), `wave_grid()` to resolve `None`/name/object, `grid_wave()` as
   `canonical_wave()`'s grid-aware counterpart, `check_wave(..., grid=)`,
   `IOPs.validate(..., grid=)` now checking the trailing axis against **that grid's** band
   count instead of `N_WAVE`, and `bb_w(..., mode=)` with `check_bb_w_range` beside it.
   **295 tests pass** (was 279), ruff clean.

   Three things worth knowing:
   - **The check kept its teeth.** OLCI bands against the L23 grid still raise, and so does
     a 12-band grid that is not quite OLCI's — the validator is now *per grid*, not
     relaxed. The L23 grid is named `"canonical"` precisely so the M0–M4 error messages,
     and the tests that match on them, are unchanged.
   - **`bb_w` past 750 nm is now a stated choice.** `"clamp"` stays the default so every
     M4 number is untouched; `"extrapolate"` continues the red tail via
     `BB_W_TAIL_EXPONENT = -4.140855`, **fitted here from the table** (it reproduces
     650–750 nm to 2.2e-4 relative and a test re-derives it); `"raise"` refuses, for
     boundary use. At PB24's 753 nm band the clamp reads **1.6% high**, and at 800 nm —
     where the hyperspectral files reach — **23% high**.
   - **PB24 tabulates its own `bbw`**, so its loader should use the file's values and never
     reach this table at all. The mode exists for callers that cannot.

   `check_wave` rejects anything that is not L23's canonical 81-band 350–750 nm grid
   (`conventions.py:228`), `IOPs.validate` hard-codes `N_WAVE` (`types.py:175`), and
   `bb_w` clamps outside 350–750 (`conventions.py:184`) — so PB24's 12 OLCI bands, one of
   which is **753 nm**, would either be rejected or silently given `bb_w(750)`. Add a grid
   parameter (or a named-grid registry) rather than loosening the check: the check has
   caught real bugs and should keep catching them per grid.

   **Gate:** the L23 path is unchanged — every existing `conventions`/`types` test passes
   untouched; a PB24-grid `IOPs` validates; `bb_w(753)` is either computed or refuses, and
   a test says which; the clamp can no longer fire silently.
   **Depends on:** nothing. **Blocked:** no.

4. **The PB24 loader** — `robust/rt/data/pb24.py`. ✅ **done 2026-08-09.**
   *Unlocks: everything below.*

   **What landed.** `PB24Batch` (carrying `rrs` **and** `Rrs` and `Q` and the µ's,
   unlike `L23Batch`), `LoadReport`, `load_batch`, `select`, `write_fixture`,
   `npz_reader`, `read_classes`, `data_dir`/`file_path`, and a committed
   **`robust/tests/files/pb24_small.npz`** (471 kB, 3 realisations, *all* 1300
   geometries each). **+28 tests, 323 pass**, ruff clean.

   Four things worth knowing:
   - **A stride aliases, and it nearly went unnoticed.** Flattening
     `(theta_s, theta, phi)` in C order puts the 13 azimuths innermost, so
     `geometry_stride=13` keeps **one azimuth** — deleting the BRDF axis this
     milestone exists to study, while still returning a plausible batch. The
     report now carries per-axis `coverage`, `aliased_axes` names the casualties,
     and the loader warns. Found by *running* the loader, not by reading it.
   - **The zero-`rrs` gate bites where it was told to.** In the OLCI set the zeros
     are only at 753 nm, θs = 80°, θv = 87.5° — realisation 993 has exactly two —
     so the fixture includes 993 and the assertions run on the **shell**. Inside
     the window the filter removes nothing, which a second test states explicitly
     so the choice is not folklore. The cost of dropping whole spectra is
     reported, not hidden: 2 spectra removed to exclude 2 values, so **22 good
     bands lost**.
   - **The sidecar is opt-in.** `water_classes="auto"` reads the `.mat`; the
     default attaches none. A loader that behaves differently depending on whether
     `$OS_COLOR` is mounted makes fixture-backed tests non-deterministic.
   - **Measured scale, for task 11 and Q16:** 20 realisations × 832 geometries =
     16 640 samples in 0.6 s and 8 MB, so the full window is **~2 min and ~2 GB**
     of resident arrays before training touches it.

   Follow `l23.py`'s *shape* (`load_batch`, `make_splits`, `npz_reader`, `write_fixture`,
   a committed fixture so CI runs real numbers with no data mount) but **not its field
   list**: `L23Batch` carries `Rrs` alone (`l23.py:139`) and derives `rrs` through the
   nadir map. PB24's batch must carry **`rrs` and `Rrs` and `Q`**, and **`mu_d`, `mu_u`,
   `mu_tot`** and the K's, because tasks 7, 9 and 13 consume them — and because task 4's
   own gate freezes the cache bit-identically, so a field added later invalidates it.
   Read the **OLCI** files per Q12; keep the reader factored for the hyperspectral set.
   Assemble atomically (temp → verify → `os.replace`), flatten the 4-D geometry grid,
   derive `a`, `b_p`, `bb_p`, `B_p` from the components, carry `i_classes`. Default load =
   the **Q14 window** (θs, θv ≤ 70°), with the 80–87.75° shell on request. Subsampling is
   an **explicit argument**, never a hidden sample, and reports what it dropped.

   **Gate:** golden values against the raw netCDF for two named realisations at named
   geometries (the M1 pattern); the committed fixture regenerates bit-identically; the
   angle window asserts the count it removed; the geometry grid is asserted identical
   across a sample of files; the module skips cleanly without `$OS_COLOR`. **The
   zero-`rrs` filter is gated on the *shell* load, not the default** — inside the Q14
   window it removes exactly zero samples, so asserting its count there would pass
   vacuously. Decide and document *what* it drops: the zeros are per-band (λ ≥ 721 nm at
   θv = 87.5°), so dropping whole spectra discards good bands.
   **Depends on:** 3. **Blocked:** no.

5. **The three splits** — `pb24.make_splits`. ✅ **done 2026-08-10.**
   *Unlocks: every comparison in M5.*

   **What landed.** `make_splits`, `Splits`, `SplitReport`, `confound_reference`,
   `SPLIT_KINDS`/`DEFAULT_SPLIT_KINDS`, `SPLIT_SEED`, `TEST_FRACTION`,
   `BP_BAND_QUANTILES`; `PB24Batch` gained `labels` (`C`, `N`, `Y` per sample — already
   in the fixture, so it did **not** need regenerating). **+13 tests, 336 pass**, ruff
   clean.

   Four decisions and one correction:
   - **`bp_band` holds out an *interior* band** (quantiles 0.4–0.6), so the train side is
     both tails and the test is *interpolation* across phase functions. Extrapolation is
     already the geometry split's job, and conflating them would leave a bad number
     unattributable. `detail` reports the band edges, the train tails, and
     `n_train_inside_band` (0 by construction).
   - **`geometry` refuses a window-only batch** rather than scoring an empty set — the
     interaction task 2 flagged. The error names the fix (`angles="all"`).
   - **The confound is measured, and so is its yardstick.** `confound_reference` reports
     what a *random* hold-out of the same size does, because ratios against 1.0 are
     uninterpretable here: over 12 seeds at 600 realisations a random split moves median
     chlorophyll across **[0.53, 1.90]**. Only `B_p_mean` is tight under randomness
     ([0.98, 1.08]), so it is the one entry where a departure means something immediately.
   - **A correction to this doc.** It said `B_p` correlates with chlorophyll at
     **−0.65**; that figure is task 0's, and it is for the *phytoplankton component ratio*
     `bbph/bph`. For the **bulk `B_p`** the split actually uses, measured over 600
     realisations, it is **−0.49**. Still a real confound, weaker than advertised, and now
     pinned by a test.
   - **`B_p` spans more than first recorded:** 12.4× across 600 realisations
     (0.0025–0.0315), against the 6.2× measured from ~50 files at task 0. Better news for
     the milestone than the earlier figure.

   `realisation` (random 20% by file — M4's analogue); `bp_band` (hold out a band of
   per-realisation mean `B_p` — **the phase-function axis, and the reason M5 exists**);
   `geometry` (train 0–70°, test the 80/87.75° shell, per Q14). The geometry split is the
   direct successor to M4's unseen-60°, which is the half we lost, so expect it to be the
   hardest number in the milestone and treat a bad result there as information.

   **Note the load interaction:** the geometry split's test set lives *outside* task 4's
   default window, so it requires the shell load. `l23.make_splits` raises on an empty
   split (`l23.py:531`); mirror that, or the geometry split silently scores nothing.

   **Gate:** disjoint, exhaustive and deterministic given a seed; **every split's test set
   asserted non-empty**; the geometry test set contains only θ ≥ 80°; the `bp_band` split
   **measures and reports its own confound** (`B_p` correlates with chlorophyll at
   corr(log, log) = **−0.49** for the bulk `B_p`, measured at task 5; the −0.65 first
   quoted here is the *component* ratio `bbph/bph`) and a test asserts that report exists
   and is non-trivial — the confound must be visible in the artefact, not just in this
   doc.
   **Depends on:** 4. **Blocked:** no.

6. **Extend the validation toolkit.** ✅ **done 2026-08-10.** *Unlocks: tasks 7, 11 and 14
   — a shared dependency the first draft of this sequence missed entirely.*

   **What landed.** `rrms(..., where=)`, `rrms_per_wavelength(..., where=)`,
   `group_rrms(..., expected=, where=)`, a generalised `gradient_report`,
   `FD_STEPS_EXTRA` and `default_steps()`; `design/py/run_validation.py` now derives its
   table headers from the labels. **+8 tests, 346 pass**, ruff clean.

   - **`gradient_report` perturbs any field, not four names.** Variables resolve against
     the dataclasses themselves and every container is rebuilt with
     `dataclasses.replace`, so a field this module has never heard of survives and is
     provably perturbed. Subsets and extra *known* variables are now legal; what still
     raises is a name the inputs cannot offer — including `wind`, a real field that is
     `None` here and would otherwise have become `None + 1e-3`. **The old guard's
     contract changed deliberately**, and its test was rewritten rather than patched: it
     demanded exactly M2's four variables, which is what made every M5 gate
     inexpressible.
   - **`rrms`'s mask is applied twice**, before and after the division. Masking only
     afterwards leaves `0/0` in the graph — `jnp.where` hides the NaN forward, and
     reverse-mode propagates it through the discarded branch, so the loss would look
     healthy and the gradient would be NaN. A test asserts exactly that asymmetry.
   - **`group_rrms(expected=)` fixes a hazard that could not fire on L23.** Iterating
     `np.unique` can only produce non-empty groups, so the function was *incapable* of
     returning a short dict — which is why zipping `.values()` against hard-coded headers
     was safe for three fixed zeniths and is not for PB24's eight. Missing groups now
     come back as `nan`, and `run_validation.py` zips labels.
   - **Both `gradient_report` regressions were demonstrated**, not asserted: reverting the
     closure to its pre-task-6 form fails them, restoring it passes. And the refactored
     `run_validation.py` reproduces every deterministic row of the committed
     `metrics.md` bit-for-bit.

   Three limits, all of them deliberate choices made when L23 was the only dataset:
   - `gradient_report` **raises** unless the perturbed set is exactly
     `{a, bb_p, B_p, theta_s}` (`validation.py:317`), so no gate can check a gradient
     w.r.t. `theta_v` or `dphi` (task 11) or w.r.t. new `PhaseParams` fields (task 14).
     The guard itself is right — it exists because an extra key used to report 0.0, i.e.
     "perfect agreement", for a variable never perturbed (gotcha 5). Extend it; do not
     relax it.
   - `scalar()` rebuilds `types.PhaseParams(B_p=...)` (`validation.py:337`), silently
     **dropping** any other field, so a task-14 model would be certified at the wrong
     phase function with no symptom.
   - `rrms` divides by truth with no mask (`validation.py:110`), and `group_rrms` **omits**
     empty groups (`validation.py:166`) while `run_validation.py:381` zips `.values()`
     against hard-coded headers — with 8 zeniths and 8 view angles on PB24, a possibly
     empty bin mislabels columns without crashing.

   **Gate:** a regression test per limit, each demonstrated to fail before the fix — the
   extended `gradient_report` still raises on an unknown variable; a perturbation of a new
   field changes the output (so it is provably applied); `group_rrms` returns labels
   alongside values and a test feeds it an empty bin.
   **Depends on:** nothing (pure `validation.py` work). **Blocked:** no.

7. **A geometry-aware surface transfer** — `conventions`. ✅ **done 2026-08-10.**
   *Unlocks: an honest `Rrs` at any view angle, and O25's score (task 8).*

   **What landed.** `SurfaceTransfer` (a trilinearly-interpolated `A`/`B` table on PB24's
   10 × 10 × 13 angle grid), `fit_surface_transfer`, `save_transfer`/`load_transfer`/
   `default_transfer`, `Rrs_to_rrs`/`rrs_to_Rrs` gaining keyword-only `geometry=` and
   `transfer=`, plus `design/py/fit_surface.py` and the committed
   `robust/rt/files/surface_pb24.npz` (19 kB). **+12 tests, 358 pass**, ruff clean.

   **Measured on held-out realisations** (fitted on 320, scored on 80 it never saw):

   | θv | 0° | 30° | 50° | 60° | 70° | 87.5° | window |
   |---|---|---|---|---|---|---|---|
   | nadir constants | 1.84% | 4.32% | 17.81% | **33.15%** | 65.63% | 238.88% | 6.81% |
   | fitted table | 1.56% | 2.04% | 3.34% | **4.57%** | 6.43% | 14.57% | 2.05% |
   | gain | 1.2× | 2.1× | 5.3× | **7.2×** | 10.2× | 16.4× | 3.3× |

   Four things the measurement decided, none of them guessable from the design:
   - **All three angles earn their place.** At θv = 60° the per-geometry `A` still spans
     0.28–0.46 across θs and Δφ, so an `A(θv)`-only table leaves a median 3.4% and up to
     70% against the full one.
   - **`Q` does not.** Lee's `B = 1.7` is really `r̄·Q` with `Q` assumed ~3.5, and PB24
     tabulates the real `Q` (0.9–6.0). Refitting with it in place scores **1.71% against
     1.74%** for simply fitting `B` per geometry — no gain, and fortunate, because
     `forward` has no `Q` to offer.
   - **A table, not a smooth function.** A 10-term smooth fit in the cosines reaches only
     4.4× at θv = 60°, below the gate; the table gives 7.2×. The cost is LUT kinks, so the
     gradient gate runs *between* nodes (gotcha 4).
   - **The residual does not go to zero:** even fitting both coefficients everywhere
     leaves a median 1.8%. The Lee *form* is the floor here, not the coefficients.
   - **The nadir path is bit-identical**, pinned by a test, and every M0–M4 call site still
     passes no geometry — verified, not assumed.

   `Rrs_to_rrs`/`rrs_to_Rrs` hard-code Lee's nadir `A = 0.52`, `B = 1.7`; PB24 measures a
   median 45.7% error at θv = 60°, **inside** the Q14 training window (consequence 2). Fit
   the geometry dependence from PB24's own `Rrs`, `rrs` and `Q`: `A(θv)` at minimum,
   testing whether θs and Δφ terms earn their place. Keep the nadir constants as the
   **default** so every M4 number stays reproducible, and decide explicitly whether the fit
   ships as an embedded table (the `O25_L23_REFIT` pattern) or as a function.

   **Correcting the claim that put this task here.** I wrote in task 1's log that the nadir
   map "would charge every model with an interface error none of them committed". That is
   wrong, and the code says so: `rrs_gordon` and `rrs_ZTT` are the primitives and never
   touch the map — `Rrs_gordon` (`baselines.py:126`) and `Rrs_ZTT` (`ztt.py:939`) are
   above-water *wrappers* — and task 11 reads PB24's tabulated `rrs` directly, so the
   emulator's targets bypass it too. **Exactly one scoring path is contaminated:**
   `rrs_o25` (`baselines.py:302`), because O25 alone is defined in `Rrs`. The ordering
   survives — O25 is the benchmark, so its score must be clean before task 9 — but it
   rests on one model, not five.

   **Gate:** the default path is bit-identical to today's, pinned by a test carrying M4's
   numbers; the fitted path cuts the θv = 60° transfer error by ≥5× on **held-out**
   realisations; gradient-checked through the task-6 toolkit, evaluated **between** table
   nodes if tabulated (gotcha 4); `jit`-safe under the `_is_traced` convention.
   **Depends on:** 4, 5 (the held-out gate needs a split), 6. **Blocked:** no.

8. **Give O25 a geometry-indexed coefficient table.** *Unlocks: a benchmark worth beating —
   without this, task 11's gate is a straw man.*

   `fit_o25` groups **by solar zenith only** (`baselines.py:374`) and `o25_coefficients` is
   a 1-D `jnp.interp` in θs (`baselines.py:202`). O25 as published indexes θs, θv **and**
   Δφ. On PB24 the current fitter would average 8 × 13 = 104 view geometries into four
   coefficients per zenith — and worse, its default `zeniths=(0.0, 30.0, 60.0)`
   (`baselines.py:313`) **succeeds silently** on PB24, fitting 3 of the 8 in-window
   zeniths and interpolating the rest. Extend the table to 3-D and make the zenith list
   dataset-derived rather than defaulted.

   **This reverses a claim in this doc.** I wrote that PB24 favours O25 because it is O25's
   own calibration set. For the *published* O25 that is true; for the O25 **we can
   currently fit**, the opposite holds off-nadir — a θs-only table cannot represent the
   BRDF, so our benchmark would lose for reasons that have nothing to do with our model,
   and we would claim the difference. Say this out loud wherever the comparison appears.

   **Gate:** the 3-D refit beats the θs-only refit off-nadir on held-out data by a stated
   margin (if it does not, the extra axes are not earning their place and we say so); the
   L23 path reproduces `O25_L23_REFIT` exactly, so M4's numbers are untouched; fitting
   with a zenith absent from the data **raises** instead of interpolating over it.
   **Depends on:** 5, 7. **Blocked:** no.

9. **Benchmark on PB24, before training anything** — `design/py/run_pb24_validation.py`.
   *Unlocks: the target every later number is measured against.*

   Score Gordon, ZTT and O25 (task 8's form, refit on the train mask only) across the full
   BRDF on all three splits, with the §6 cuts (per λ, per θs, **per θv**, per `B_p` bin).
   This also answers a question we have never asked: whether the analytic backbone degrades
   off-nadir on its own.

   **Gate:** an aggregation-consistency test (M4's stale-figure lesson — the per-cut
   numbers must aggregate to the table); **header↔group alignment asserted explicitly**,
   not just totals (task 6's third limit); O25 labelled a refit everywhere it appears;
   CSVs written with `csv.writer` and round-tripped by a test that parses them back.
   **Depends on:** 8. **Blocked:** no.

10. **Make the sanctioned envelope per-model.** *Unlocks: task 11 without corrupting the
    shipped L23 model.*

    `SUPPORTED_THETA_S = (0.0, 60.0)` is a single module constant used as the default for
    **every** emulator's domain check (`emulator.py:201, 571, 663`; `hybrid.py:77`), with
    no view-angle counterpart, and `test_validation.py:291` pins its value. Widening it to
    Q14's 0–70° for the PB24 model would silently widen the **shipped L23 model's**
    envelope too — a 65° query against a net trained to 60° would become "in domain",
    which is precisely the seed-dependent regime M4 measured and warned about. The envelope
    belongs with the weights (or `out_of_domain` needs per-axis limits); either way it is
    an API change, so it must land **before** task 15 freezes the signature.

    **Gate:** the L23 model's effective envelope is still 0–60° after the change — pinned
    by a test that fails if a PB24-trained model's envelope leaks into it; a view-angle
    envelope exists and is enforced; `test_validation.py:291`'s intent is preserved rather
    than deleted.
    **Depends on:** 4. **Blocked:** no.

11. **Retrain the emulator on PB24** — `fit_pb24`, `design/py/train_emulator_pb24.py`.
    *Unlocks: the milestone's claim.*

    `theta_v` and `dphi` **live** (Q11), reading `rrs` targets directly. Per Q13 this
    **adds** a model: ship `files/emulator_pb24.npz` beside the L23 weights and leave
    `load_default()` where it is. Sweep ≥5 seeds and report the spread (gotcha 3).

    **An unmade decision, on this task's critical path — see Q16.** `fit()` materialises
    features and trains **full-batch, unshuffled** (`emulator.py:831`), justified in its
    docstring by L23's ~0.6 M rows. PB24's Q14 window is 5000 × 8 × 8 × 13 ≈ 4.16 M samples
    × 12 λ ≈ **50 M rows** — 83× L23, plus a full-batch `rrs_ZTT` precompute and a
    standardised copy per eval mask. Either `fit()` grows mini-batching (which changes the
    "reproducible from the seed alone, no data-order dependence" property the
    bit-identical gate leans on) or training runs on a sanctioned subsample. Q12's knob
    permits the latter; nobody has said how much.

    **Per Q16, training runs on a subsample:** all 5000 realisations, subsampled
    geometries — realisations carry the phase-function variation M5 is about, while the
    832 in-window geometries are dense and smooth. `fit()` keeps its "reproducible from the
    seed alone" property. Train at **two** subsample factors and compare, so the choice is
    evidenced rather than assumed; mini-batching becomes a proposal only if the cheaper
    factor measurably costs accuracy.

    **Gate (Q15, decided — option 2):** beat O25-refit-on-PB24 (task 8's honest form) on
    the **realisation** split *and* the **held-out-`B_p`** split; **report** the geometry
    split with its seed spread. Plus: a θv = 40° view inside the window must **stop** being
    flagged and 80° must still be; the architecture guard runs on the emulator being
    serialised, not on a proxy (PR #12); regenerated weights are bit-identical between
    runs; the training-set size and subsample factor are **stated in the artefact**, not
    just chosen.
    **Depends on:** 9, 10. **Blocked:** no.

12. **The cross-dataset check: the PB24 model on L23.** *Unlocks: the strongest
    generalisation statement available to us — but not the one the first draft promised.*

    **The grids do not match, and this nearly shipped as a free number.** L23 spans
    350–750 nm in 81 bands (`conventions.py:107`); PB24 OLCI spans 400–753 in 12. `wave_nm`
    is a live feature and the domain is the training min/max, and `out_of_domain_mask`
    flags a sample if **any** feature at **any** λ breaches (`emulator.py:704`) — so 350 nm,
    which sits 14% of the span below the boundary against a `DOMAIN_TOL` of 0.01, flags
    **every L23 sample**. With `on_out_of_domain="ztt"` the "cross-dataset check" would
    have scored the bare backbone on 100% of L23 and called it a result.

    So: score on the **overlapping band range only**, report the count and the range, and
    treat 350–395 nm as what it is — genuine extrapolation, reported separately, never
    folded into the headline.

    **Gate:** the overlap is computed, not assumed, and asserted non-empty; the fraction of
    L23 flagged out-of-domain is reported beside the score; **the promotion rule is encoded
    as a conditional, not a constant** — a test that computes the rule's inputs and asserts
    `load_default()` returns the L23 model *unless* the PB24 model wins L23's own held-out
    split. A test that merely pins today's answer would pass trivially and get edited the
    day it mattered, which is the failure it is meant to prevent.
    **Depends on:** 11. **Blocked:** no.

13. **ZTT's internals against HydroLight** — *narrowed, because the first draft promised
    more than the code can deliver.* Runs in parallel with 7–12.

    `ztt.py` exposes `mu_d`, `mu_infinity` and `mu_infinity_tt2017` — **no `mu_u`, no
    `mu_tot`, no `Q`**. And PB24's `mu_u`/`mu_tot` are near-surface AOPs while ZTT's µ∞ is
    the *asymptotic* average cosine: different quantities, even for a perfect model. So the
    directly comparable pair is **`mu_d` against PB24's `mu_d`**, per band and per zenith.
    For µ∞, either derive an asymptotic proxy from the tabulated K's and say exactly what
    the proxy assumes, or report that the Eq-(8) caveat **cannot be sized this way** — and
    then say so in the record instead of leaving the promise standing.

    **Gate:** `mu_d` agreement reported per band and per zenith against a documented
    tolerance, with the measured discrepancy **pinned by a test** so a future change to
    `mu_d` announces itself; whatever is concluded about µ∞ is written into §7.3 and
    `prototype_summary.md`, including "not resolved" if that is the answer.
    **Depends on:** 4. **Blocked:** no.

14. **Promote `PhaseParams`** to the ZTT backward-VSF parameterization (design §4.2), as
    *additional* fields defaulting to `None`. *Unlocks: the design's phase-function
    parameterization, and the M5 → inversion hand-off.*

    **Gate:** every existing test passes **untouched** — that is the proof the extension
    changed no signature (record §3.2); `forward` with the new fields `None` is
    bit-identical to before; the new fields are gradient-checked **through task 6's
    extended `gradient_report`**, and a test proves the perturbation is actually applied
    (gotcha 5 — `scalar()` used to drop exactly these fields).
    **Depends on:** 4, 6; informed by 13. **Blocked:** no.

15. **Freeze the `forward` API.** *Unlocks: the inversion track and training-data
    generation, which both want a stable engine.*

    **Gate:** a signature-pinning test (the M0 pattern) plus a record §8 note stating what
    "frozen" permits and forbids.
    **Depends on:** 7, 10, 11, 14 — **last**, because each can still move the signature.
    **Blocked:** no.

16. **Notebook, review, hand-off** — `notebooks/RT/rt_elastic_coding_6.ipynb`, then a
    PR-review pass, then the edit to `rt_elastic_coding_prompt_7.md`. The rhythm M0–M4
    settled. **Depends on:** 15.

---

**Not scheduled, and why — stated plainly rather than left implied:**

- **VSF-family generalisation** — blocked on **commissioned HydroLight runs with a
  non-Fournier-Forand family**, which nobody has ordered. PB24 varies the FF *parameter*,
  so tasks 5 and 11 test our §4.2 parameterization; only a second family tests
  generalisation *across* families. Worth specifying if M5's held-out-`B_p` result is
  good, worth nothing if it is not. **This is the one headline gap M5 will not close.**
- **The hyperspectral λ-interpolation check** — unblocked, deferred by Q12. It answers
  "does the correction interpolate across λ", which 12 OLCI bands cannot.
- **PR05** — now *implementable* for the first time, since PB24 spans its 4-D
  `(θs, θv, Δφ, γb)` LUT, but it earns its place only if O25 stops being the benchmark.

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

**Q15 ✅ answered — option 2. (Task 1 → task 2, Claude → JXP.) What is M5's acceptance
gate?**

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

>A.  Ok, use 2.

**Taken as decided, and task 11's gate is no longer provisional:** beat O25-refit-on-PB24
on the **held-out-realisation** split **and** the **held-out-`B_p`** split; **report** the
geometry split with its seed spread. Recorded in `rt_elastic_implementation.md` §7.4 and in
the coding plan's §M5 gate. Note what task 8 now has to do for this gate to mean anything —
the O25 we can currently fit has no view-angle axis, so beating it off-nadir would measure
our fitter, not our model.

**Q16 ✅ answered — option 1, subsample. (Task 2 → task 11, Claude → JXP.) How much of PB24
does the emulator actually train on?**

This surfaced from a code review of the sequence, not from the data. `fit()` materialises
every feature and trains **full-batch and unshuffled** (`emulator.py:831`); its docstring
justifies that by L23's ~0.6 M rows. PB24 inside the Q14 window is 5000 × 8 θs × 8 θv ×
13 φ ≈ 4.16 M samples × 12 λ ≈ **50 M rows — 83× L23** — plus a full-batch `rrs_ZTT`
precompute and a standardised copy per evaluation mask. 3000 full-batch Adam steps over
that on CPU-only JAX is hours and many GB, against the ~60 s regime `train_emulator.py`
was built for. Two ways out:

1. **Train on a sanctioned subsample** — Q12's knob already permits it. Keeps `fit()`'s
   "reproducible from the seed alone, no data-order dependence" property, which the
   bit-identical-weights gate leans on. My instinct: keep **all 5000 realisations** and
   subsample *geometries*, since realisations carry the phase-function variation M5 is
   about, while the 832 in-window geometries are dense and smooth.
2. **Give `fit()` mini-batching.** Uses everything, but changes the reproducibility
   contract and is a real change to a module M3/M4 validated.

My recommendation: **1 for M5, with the training-set size stated in the artefact rather
than buried in a default**, and 2 only if the geometry subsample measurably costs accuracy.
— *Not blocking until task 11; tasks 3–10 are unaffected.*

>A.  Ok, let's go with 1 for now.

**Taken as decided.** Task 11 trains on a subsample: **all 5000 realisations, subsampled
geometries**, with the row count and the subsampling factor written into the artefact
rather than left in a default. `fit()` keeps its "reproducible from the seed alone"
property, which the bit-identical-weights gate depends on. If the geometry subsample
measurably costs accuracy, mini-batching becomes a proposal with evidence behind it rather
than a guess — and the way to find out is to train at two subsample factors and compare,
which costs one extra run.

## Next

→ M5's own hand-off, when it closes.

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

### 2026-08-10 (Task 7 — the surface transfer; four questions the data answered; record v0.22)

**What I did.** `SurfaceTransfer` in `conventions` — Lee's `A` and `B` tabulated on PB24's
10 × 10 × 13 angle grid, interpolated trilinearly — plus the fitter, the atomic
save/load, `design/py/fit_surface.py`, and the committed 19 kB table. `Rrs_to_rrs` and
`rrs_to_Rrs` take keyword-only `geometry=`/`transfer=`; omitting both is the M0–M4 path
and is bit-identical. **+12 tests, 358 pass**, ruff clean.

**I measured before designing, and it changed the design four times.**

1. **The pairing was worth checking first.** Before fitting anything I tested whether
   PB24's `rrs(θ)` is the same direction as `Rrs(θ)` or the Snell-refracted one — because
   if it were refracted, the whole task would be a coordinate fix rather than a fit.
   `rrs × Q = R` to **0.00%** at every geometry settles it: same direction. And pairing
   with the refracted angle flattens the ratio only partly (0.53 → 0.49 at 60°, still 0.30
   at 87.5°), so refraction is part of the story and Fresnel transmittance is the rest.
2. **`Q` looked like the answer and wasn't.** Fitted `A`,`B` per geometry still left 4.7%
   at nadir, which is poor for two free parameters at one geometry — and the reason is
   visible in Lee's own algebra: `B = 1.7` is `r̄·Q` with `Q` assumed ~3.5, while PB24's
   real `Q` spans 0.9–6.0. So I refit with `1 − r̄·Q·rrs`, expecting the residual to
   collapse. It scored **1.71% against 1.74%** — no gain. Fortunate as well as tidy:
   `forward` has no `Q` at prediction time, so a `Q`-dependent transfer would have been
   unusable exactly where it is needed.
3. **The gate chose table over function.** I wanted the smooth 10-term fit — compact, no
   LUT kinks, no gotcha-4 care needed. It reaches 4.4× at θv = 60°, and the gate says ≥5×.
   The table gives 7.2×. So the table ships and the gradient check runs between nodes.
4. **The residual has a floor.** Fitting both coefficients at all 1300 geometries still
   leaves a median 1.8%. That is the Lee *form*, not the coefficients, and it is worth
   writing down because the obvious next move — more coefficients — will not help.

**On the honesty of the number.** The fit is scored only on realisations it never saw: 320
train, 80 held out, using task 5's split. A 2600-parameter table fitted and reported on the
same water bodies would have produced a beautiful and meaningless table. The gate asked for
held-out and it was right to.

**A correction to my own framing, carried from task 2.** The doc still said this task
exists because the nadir map contaminates "every model". It doesn't — `rrs_gordon` and
`rrs_ZTT` never touch it, and task 11's targets read PB24's tabulated `rrs`. Exactly one
scoring path is affected, `rrs_o25`, which is why this lands before task 8 rather than
before everything.

**What I learned.** Three of the four decisions above were *available* from the data before
any code was written, and two of them contradicted what I would have built. The pattern
across this milestone is consistent: a twenty-line probe run before the design costs less
than the design that has to be undone. The one that stings slightly is `Q` — I had a
correct physical story for why it should help, and the measurement said it doesn't. A good
mechanism is not evidence of a good improvement.

### 2026-08-10 (Task 6 — the validation toolkit; a guard I deliberately changed; record v0.21)

**What I did.** The three limits task 2's review found in `validation.py`, each of which
blocks a later gate: `gradient_report`'s fixed four variables, `scalar()` silently
dropping `PhaseParams` fields, and `rrms` having no mask. Plus the real call site the
third one endangered — `design/py/run_validation.py` zipped `group_rrms(...).values()`
against hard-coded headers. **+8 tests, 346 pass** (316 + 30 skipped), ruff clean.

**The decision that needed care: I changed a contract on purpose.** The old guard demanded
that `steps` name *exactly* `{a, bb_p, B_p, theta_s}`, and an existing test pinned it —
including that a **subset** must raise. That guard was right for its time: the closure
genuinely could not honour a subset (KeyError from inside) or an extra key (reported 0.0,
"perfect agreement", for something never perturbed). But it is also precisely what makes
every M5 gate inexpressible, so "extend it, do not relax it" needed unpacking:

- subsets and extra *known* variables are now **legal**, because the implementation can
  honour them — task 7 wants to check only the new surface-transfer path, task 14 only the
  new fields;
- what still raises is a name the inputs **cannot offer**, including `wind` — a real field
  that is `None` here and would otherwise have become `None + 1e-3`;
- the failure the old guard prevented is now **structural**: every name is resolved to a
  real field and perturbed through `dataclasses.replace`, so "named but not perturbed" is
  no longer a reachable state rather than a forbidden one.

I rewrote that test rather than patching it, and said so in both the test's docstring and
the record. Quietly deleting an assertion that encodes an earlier decision is how a suite
stops meaning anything.

**The subtlety in the mask.** `rrms` divides by truth, so masking has to happen *before*
the division as well as after. Masking only afterwards leaves `0/0` in the graph:
`jnp.where` hides the NaN in the forward pass, and reverse-mode differentiation propagates
it through the discarded branch. Since `rrms` doubles as M3's training loss, that would
have produced a loss that looks healthy and a gradient that is NaN — the worst shape a bug
can take. The test asserts the asymmetry directly: masked gradient finite, unmasked not.

**A hazard that could not fire on L23.** `group_rrms` iterates `np.unique(labels)`, which
can only produce non-empty groups — so it was *incapable* of returning a short dict, and
zipping `.values()` against three fixed zenith headers was safe. On PB24, with eight
zeniths and eight view angles, a split can omit one and every column shifts left without
raising. `expected=` makes the key set fixed and `run_validation.py` now indexes by label.

**Demonstrated rather than asserted.** I reverted `gradient_report`'s closure to its
pre-task-6 form and watched both new regression tests fail, then restored it and watched
them pass. And I checked the refactored `run_validation.py` against the committed
`metrics.md`: every deterministic row — Gordon, ZTT, O25 on both the per-zenith and
per-`B_p` tables — reproduces bit-for-bit, so the change is behaviour-preserving on L23
while being correct on PB24.

**What I learned.** "Do not relax the guard" was the right instruction and the wrong
summary of the work. The guard conflated two things: *which variables the caller may ask
for* (too narrow, and the reason the task exists) and *whether an asked-for variable is
actually perturbed* (the real invariant). Separating them let the first widen while the
second got stronger — enforced by construction instead of by a set comparison. When an old
constraint blocks new work, the useful question is which invariant it was protecting, not
whether to keep or drop it.

### 2026-08-10 (Task 5 — the three splits, and a confound weaker than I claimed; record v0.20)

**What I did.** `pb24.make_splits` with the three splits M5 needs — `realisation`,
`bp_band`, `geometry` — each carrying a `SplitReport`, plus `confound_reference`.
`PB24Batch` gained `labels` (`C`, `N`, `Y`); they were already in `RAW_FIELDS`, so the
committed fixture did **not** change and its bit-identical gate still holds. **+13 tests,
336 pass** (306 + 30 skipped), ruff clean.

**The design decision.** `bp_band` holds out an **interior** quantile band, so the train
side is both tails and the split tests *interpolation* across phase functions.
Extrapolation is already the geometry split's job, and running both in one split would
leave a bad number unattributable to either. Choosing this before seeing any result is
the same discipline as writing the gate before the numbers exist.

**Where measurement changed what I would have written.** The gate said the `B_p` split
must report its confound, and I had the sentence ready: "corr(log, log) = −0.65, so
holding out a band also shifts the water type." Two things were wrong with it.

1. **−0.65 is the wrong quantity.** That is task 0's figure for the *phytoplankton
   component ratio* `bbph/bph`. The split uses the **bulk** `B_p`. Measured over 600
   realisations, that correlation is **−0.49**. Real, and weaker than advertised. Both the
   record and this doc said −0.65 in a sentence about the split; both now say what they
   mean, and a test pins the bulk figure.
2. **The ratios are uninterpretable against 1.0**, which I only saw by running the
   report. PB24's labels are heavy-tailed, so a *random* hold-out of the same size already
   moves median chlorophyll across **[0.53, 1.90]** over 12 seeds. A reported "C = 1.5"
   therefore means nothing on its own — and I would have shipped exactly that number as
   evidence of a confound. Hence `confound_reference`, which measures the random-split
   band on the batch in hand. Only `B_p_mean` is tight under randomness ([0.98, 1.08]),
   so it is the one entry where a departure is immediately meaningful.

**Two limits of the metric, written down because they read as results.** For `bp_band`
the `B_p_mean` ratio is ~1.0 — an interior band and its two tails share a median — so the
intended separation lives in `detail` (band edges, train tails, `n_train_inside_band`),
not in the ratio. For `geometry` every confound entry is exactly 1.0, which is correct and
is a useful control: that split divides angles, not water bodies.

**Also measured:** `B_p` spans **12.4×** across 600 realisations (0.0025–0.0315), against
the 6.2× recorded from ~50 files at task 0. The earlier figure was a small-sample floor,
and the milestone's headline axis is wider than the record claimed.

**One more tautology of mine.** `assert splits.seed == 8` — testing that a field I stored
was stored. It now sweeps ten seeds and requires the draw to land on more than one
realisation, which is the honest version given that with three fixture realisations two
seeds can agree by chance. That is the third in three tasks; the pattern is always the
same, a test written to *describe* behaviour rather than to *discriminate* between right
and wrong behaviour.

**What I learned.** A confound is not disclosed by quoting a correlation from a different
quantity in a different sample. It is disclosed by measuring, on the split that was
actually built, how far it moves the things it did not mean to move — and by measuring
what "far" means. The reference band is the part I would not have thought to build if I
had not first computed a ratio and been unable to say whether 1.5 was large.

### 2026-08-09 (Task 4 — the PB24 loader; a stride that deletes the BRDF; record v0.19)

**What I did.** `robust/rt/data/pb24.py`, following `l23.py`'s shape — `load_batch`,
`write_fixture`, `npz_reader`, `select`, a committed fixture holding the loader's *input*
so CI runs real numbers — with the field list deliberately wider: `rrs` **and** `Rrs` and
`Q` and the average cosines, because tasks 7, 9 and 13 consume them and the fixture is
gated bit-identically, so a field added later invalidates every cache in existence. Plus
`LoadReport`, which every load returns. **+28 tests, 323 pass** (294 + 29 skipped without
`$OS_COLOR`), ruff clean.

**The finding of the day, and I found it by running the thing rather than reading it.**
The geometry grid flattens `(theta_s, theta, phi)` in C order, so the 13 azimuths sit
innermost — which means **`geometry_stride=13` keeps exactly one azimuth**. My very first
smoke test used stride 13 because 13 looked like a nice number, and printed `dphi uniq
[0.0]`. A subsample that silently deletes the BRDF axis, in the milestone whose entire
purpose is the BRDF, while returning a batch that validates and looks completely normal.
`LoadReport` now carries per-axis `coverage`, `aliased_axes` names the casualties, and the
loader warns with the arithmetic. This lands directly on **Q16**: the sanctioned subsample
has to be checked for *coverage*, not just for size — "how many samples" was the wrong
question to have asked.

**Two gates that had to be made non-vacuous.** The zero-`rrs` filter removes nothing inside
the Q14 window — I said so in task 2 and it is true — so the assertions run on the *shell*,
and the fixture deliberately includes realisation **993**, which carries the only two
zero-`rrs` values in the OLCI set (both at 753 nm, θs = 80°, θv = 87.5°). A second test
states the inertness inside the window explicitly, so the choice of the shell is recorded
rather than folklore. And the cost of dropping whole spectra is now a number in the report:
2 spectra removed to exclude 2 bad values, **22 good bands lost**.

**Two things the code caught in me.** `write_fixture`'s own verification refused to write:
it loads the snapshot back through the real loader and counts samples, and my expected
count ignored that the default filter drops 993's two zeros — the atomic-write check
working exactly as designed, on its author. And my `mu_d` test contained
`assert ... == approx(0.0) or True`, a tautology, the same "test that cannot fail" defect
M4's review found four of. It now asserts `mu_d` takes exactly **8** distinct values in the
window, one per solar zenith: a broadcast bug collapses that to 1, a mis-indexed gather
inflates it.

**One design change while writing it.** `load_batch` originally read the water-class
sidecar automatically. That makes a fixture-backed test behave differently depending on
whether `$OS_COLOR` is mounted — non-determinism smuggled in as convenience — so classes
are opt-in via `water_classes="auto"`.

**Measured, for Q16 and task 11:** 20 realisations × 832 in-window geometries = 16 640
samples in 0.6 s and 8 MB, so the full window is **~2 min and ~2 GB** resident before
training touches it. That is the number the subsample decision should be made against.

**What I learned.** Run the thing on real data at the first opportunity, with a
deliberately awkward argument. Reading the loader would never have shown me the azimuth
collapse; one print of `dphi uniq` did. The pattern is the same one as yesterday's nadir
spot check — an artefact that looks right at the point you sample it — except this time the
sampling was cheap enough to do before writing any prose about it.

### 2026-08-08 (Task 3 — `conventions` learns a second wavelength grid; record v0.18)

**What I did.** The first code of M5, and the smallest task in it: give the package a grid
concept so PB24 can enter without loosening the check that has been catching grid bugs
since M1. `WaveGrid` + `GRIDS` registry + `wave_grid()`/`grid_wave()`; `check_wave` and
`IOPs.validate` take a `grid=`; `bb_w` takes a `mode=`; `check_bb_w_range` added beside it.
**+16 tests, 295 pass** (272 + 23 skipped without `$OS_COLOR`), ruff clean. Also folded in
JXP's Q15 (gate on the realisation **and** `B_p` splits) and Q16 (train on a subsample),
which turned task 11's provisional gate into a decided one.

**The design decision worth recording.** The obvious way to accept a second dataset is to
relax `check_wave` — accept any ascending grid and move on. That would have been wrong in a
specific way: the canonical-grid check has caught real bugs (M1's loader, M2's golden
values), and a check that accepts everything catches nothing. So the grid became a
*parameter* rather than a *loosening*: OLCI bands against the L23 grid still raise, and so
does a 12-band grid that is not quite OLCI's. Same for `bb_w` — instead of quietly widening
the table, the clamp became a named choice with a measured alternative.

**A number I measured rather than quoted.** The module docstring said `bb_w` "falls as
λ^-4.2 (fitted)", and Morel's molecular value is −4.32. Neither is right for extrapolating
*past* 750 nm: the whole-range fit is −4.215, but the red tail alone (650–750 nm) is
**−4.140855**, and that reproduces the tail to 2.2e-4 relative. Since continuing the tail is
the only thing the constant is used for, it is fitted to the tail — and a test re-derives it
from the table so it cannot drift. The clamp it replaces reads **1.6% high** at PB24's
753 nm band and **23% high** at 800 nm, where the hyperspectral files reach.

**Two of my own tests were wrong on the first run**, and both corrections were worth more
than the tests. I asserted 9 OLCI bands fall off the 5 nm grid; it is 6 — I had counted 490,
620 and 665 as off-grid when they are multiples of 5, and the same error was in the
docstring I wrote beside it. And my continuity check at the 750 nm seam failed at 1.1e-3
against a 1e-4 tolerance — not a discontinuity but **the function's own slope**: at
d ln bb_w/d ln λ = −4.14, 0.2 nm *is* 0.11%. The test now checks the value and the log-slope
across the seam, which is what "continuous" should have meant. A third failure was a real
regression: my new error messages said "the l23 grid" where M0–M4's said "the canonical
grid", and an existing test matching on that wording caught it. That is exactly the gate
"every existing test passes untouched" doing its job on the first task it applied to — so
the L23 grid is *named* `"canonical"`, and the messages are byte-identical.

**What I learned.** A prerequisite task is the cheapest place to discover that a package's
"general" API is not. Every one of the four changes here was a spot where M0–M4 made a
reasonable single-dataset choice that reads as universal — `N_WAVE` in a shape check, a
canonical grid in a validator, a table whose support silently defines a clamp. None was a
mistake at the time; all four would have surfaced as puzzling numbers three tasks later.

### 2026-08-08 (Task 2 — M5 sequenced as tasks 3–16; record §7 filled, v0.17)

**What I did.** Turned the scope bullets into a sequence of tasks, each with a gate that
can actually fail and an explicit dependency list, and filled §7 of the record
(§7.1 task status, §7.2 the data as measured, §7.3 what the data changed about the plan,
§7.4 the Q10–Q15 decisions, §7.5 the sequence and what is blocked). Bumped the record to
**v0.17** and extended the Prompts list so each task has a prompt. Also updated the coding
plan's §M5 stub, which still said "M5 milestones will be detailed once M4 results are in"
and still described commissioning HydroLight runs as the route — both now false.

**Then a Fable agent reviewed the sequence against the code and found it wrong in eight
places** — enough that what shipped is a second draft with **six more tasks** (16, not 10).
I verified every load-bearing finding myself before rewriting; all of them held. The full
list is record §7.11, and the pattern behind them is one thing: **the M0–M4 machinery
quietly assumes L23 is the only dataset**, and my sequence assumed it was general. The two
that would have cost the most: the cross-dataset check **could not have passed** (L23 starts
at 350 nm, an OLCI-trained emulator's domain starts at 400, and one out-of-domain λ flags
the whole sample — so the "free cross-dataset number" would have been the bare backbone
scored on all of L23 and reported as a result); and the O25 benchmark **would have been a
straw man** (`fit_o25` has no view-angle axis at all, so on PB24 it averages 104 view
geometries into four coefficients per zenith, and beating it off-nadir would have measured
our fitter's limitation rather than our model). Two prerequisite tasks were missing
outright — a second wavelength grid in `conventions`, and an extension to
`gradient_report`, which currently *raises* on any variable outside
`{a, bb_p, B_p, theta_s}` and so could not have gated a single one of the three gradient
checks I wrote.

**A correction I have to make in my own voice.** I wrote in task 1's log that the nadir
`Rrs ↔ rrs` map "is on the path of every model we have" and put task 5 before the benchmark
on that basis. It is not. `rrs_gordon` and `rrs_ZTT` are the primitives and never touch the
map; the lines I cited (`baselines.py:126`, `ztt.py:939`) are the above-water *wrappers*,
and the emulator's targets read PB24's tabulated `rrs`. **Exactly one scoring path is
contaminated: `rrs_o25`**, because O25 alone is defined in `Rrs`. The ordering survives —
O25 is the benchmark — but on one model's account, not five. I had the grep output in front
of me and read the function names without reading which direction they convert.

**Deliberate omission.** I deleted task 2's scope bullets rather than leaving them beside
the sequence they became. Two lists of the same work drift, and the drifting copy is
always the one someone reads — this milestone has already spent effort on a stale figure
and a stale CSV for exactly that reason.

**Gates, second time round.** The review also found **two of mine that would have passed
vacuously** — the zero-`rrs` filter removes nothing inside the Q14 window, so asserting its
count there proves nothing, and the geometry split's test set is empty on a default load,
so "contains only θ ≥ 80°" is true of the empty set. Both now sit on the shell load with
non-emptiness asserted. Task 12's promotion-rule test was worse than vacuous: pinning
"`load_default()` returns the L23 model" passes trivially today and gets *edited* the day
it matters, which is precisely the failure it was supposed to prevent. It is now a
conditional — compute the rule's inputs, assert the outcome follows.

**Raised Q16.** `fit()` trains full-batch and unshuffled on L23's 0.6 M rows; PB24's Q14
window is ≈50 M rows, 83× larger. Either `fit()` grows mini-batching (changing the
reproducibility contract the bit-identical-weights gate leans on) or we train on a
sanctioned subsample. Nobody had decided, and it sits on task 11's critical path.

**What I learned.** Two things, one of them uncomfortable. The uncomfortable one: I wrote
"sequencing is a technical claim, not a preference, and it is checkable" in this morning's
log, ran the check, and then **misread its output** — I saw `Rrs_to_rrs` in five files and
concluded five contaminated models without asking which direction each call converts. A
check you run but read carelessly is worse than no check, because it produces a confident
wrong claim instead of an admitted uncertainty. The other: a plan written against an API's
*documentation* generalises; a plan written against its *implementation* survives. Every
one of the eight findings was a place where M0–M4 made a reasonable L23-specific choice —
canonical grid, four FD variables, one envelope constant, θs-only O25 — that reads as
general until you open the file.

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
