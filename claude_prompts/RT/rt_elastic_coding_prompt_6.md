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
   main open risk and the first thing new geometry data should resolve.
3. **The `B_p` and BRDF axes are untested, not passed.** L23 spans 1.7× in `B_p` against
   the design's ~7×, with a single fixed Fournier-Forand shape and a nadir view. M4's
   flat per-bin accuracy is *not* evidence of generalisation, and the notebook says so.

### The one piece of M5 that is already built

`Emulator.domain` + `hybrid.on_out_of_domain` were added at M4 (record §6.5). The
sanctioned envelope is `SUPPORTED_THETA_S = (0, 60)` for the solar zenith, and **any**
off-nadir view is now flagged — so with `on_out_of_domain="ztt"` the hybrid already
degrades to the backbone on the geometry M5 is about to introduce. Expect that policy to
start firing the moment new data arrives; it is a correct default, not a bug, and the
interesting question is what it should do *instead* once the emulator has seen off-nadir
views.

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
   reads a trailing axis matching `n_wave` as a spectrum.

### Outstanding

- **PR #12** is open; if Bugbot adds findings after JXP pushes, fold them into the review
  task.
- **Equation (8)'s `m1..m16`** remain unpublished; µ∞ comes from Twardowski & Tonizzo
  (2017) and `mu_inf_coeffs=` swaps them in. Report as *ZTT with the TT2017 µ∞*.
- **The published O25/PR05 coefficients** are still not in the repo; every O25 number is
  a refit.

## Prompts

1. Read this doc. Execute the 1st task in the "M5" section below — the scoping task. Ask
   your questions in the Q&A section first. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task. Check my answers in Q&A. Use Fable if you can.
   Log your work.

## M5

### Tasks

1. **Detail this milestone, and decide its data.** M5 cannot start until the reference
   data question is settled, because every other task depends on it — so this task is
   scoping, not coding, and its deliverable is a task list in this doc plus a filled §7
   of the record.

   The scope the plan sketches, with what M4 now says about each:

   - **New reference runs.** HydroLight runs varying the **particle phase function** and
     the **sensor zenith/azimuth** — the two axes L23 fixes and the two the prototype
     therefore cannot speak to. Commissioning these is a project decision with a cost;
     see **Q10**.
   - **PB24** (Pitarch & Brando 2025: 5000 IOP realisations × 1300 geometries) as a
     multi-angular cross-comparison. It is O25's own calibration set, which makes it the
     natural geometry benchmark — and also means O25 will be at its strongest on it,
     which is the honest way round.
   - **Extend the emulator** on the richer reference and re-run the §6 protocol with
     **held-out phase-function shapes** and full geometry. The held-out-shape split is
     the one that would turn M4's untested axis into a measured one.
   - **Promote `PhaseParams`** from `B_p` to the ZTT backward-VSF parameterization
     (design §4.2), as *additional* fields defaulting to `None` so no signature changes.
   - **Freeze the `forward` API** as the shared engine for training-data generation and
     the separately designed inversion.

   Sequence these by what each one *unlocks*, not by size, and gate each with a test as
   M0–M4 did. Say plainly which tasks are blocked on data that does not exist yet.

2. **(to be written by task 1)**

### Q&A

**Q10 (M4 → M5 hand-off, Claude → JXP). What reference data will M5 actually have?**

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

**Q11 (M4 → M5 hand-off, Claude → JXP). What should the hybrid do off-nadir before it
has been trained there?**

M4's domain fix means **any** off-nadir view is now flagged, so
`on_out_of_domain="ztt"` degrades the hybrid to the analytic backbone for every sample
of any new multi-angular dataset — correct, but it means the hybrid contributes nothing
there until retrained. Three options once geometry data exists: keep the fallback and
report the backbone's numbers; retrain with the view angles as live features and let the
domain widen naturally; or treat off-nadir as a separate model. The second is the
obvious one, but it changes what "the emulator" means, so it is worth your view before I
build it. — *Not blocking task 1.*

## Next

→ M5's own hand-off, when it closes.

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
