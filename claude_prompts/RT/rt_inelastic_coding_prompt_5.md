# Inelastic RT Coding — Prompt 5 (M4: Validation — prototype complete)

## Goals

Implement **Milestone M4** of the coding plan
(`design/rt_inelastic_model_coding_plan.md`): the full design-§6 validation
protocol and acceptance gate, the review pass, the committed
metrics/figures, and the finished implementation record. The inelastic
prototype is **done** when the M4 gate passes.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `code-review` for the
review task, `critical-partner`) as helpful.

### Working agreements

As in `rt_inelastic_coding_prompt_1.md`: JXP runs git — branch
**`inelastic-rt`** (prompt 1 Q&A Q2; *not* the coding plan's
`rt-inelastic-prototype`); `ocean14` on this machine, CPU JAX; reuse over
reinvention; every task `pytest`-gated; the **elastic hash-regression stays
green** — two-tier (record §2.8): strict SHA-256 pins on dev machines, the
ULP-closeness regression everywhere. Never `pip install -r requirements.txt`
wholesale (the `git+` lines clobber the editable `bing`/`ocpy` checkouts).
Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — M4 section + Definition of done + the speed-risk entry.
- **Design** — `design/rt_inelastic_model.md` §6 (the gate, verbatim) and
  the reported-not-gated diagnostics.
- **The implementation record** — `design/rt_inelastic_implementation.md`
  §§4–5 (v0.23): everything M2–M3 built, its API, and every measured
  number. Chronology: prompts 3–4 Logs.
- **The M3 tests** — `robust/tests/test_inelastic_corr.py`: the held-out
  gates, the weights-integrity regression, and the corrected-path FD
  protocol M4's gradient gate should reference rather than re-derive.
- **Elastic validation** — `robust/rt/validation.py`,
  `design/py/run_validation.py`, `design/validation/` (the artifact
  pattern to extend, not fork).

## Status entering M4

*(Filled by M3's final task, 2026-08-26. Details: record v0.23 §5;
chronology: prompt 4 Logs. Q&A state: **all prior questions answered and
closed** — corrections default-on approved by JXP, the stray `rob/`
directory kept.)*

**M3 is complete — all six tasks (incl. the JXP-inserted PR-review task);
PR #18 merged.** Suite: **416 passed, 1 skipped** (the skip is deliberate —
the missing-weights fallback test retired itself when the weights landed);
ruff + format clean; elastic hash pins green. What M4 inherits:

- **The complete forward model, corrected by default.**
  `forward(..., inelastic=Inelastic())` is now the full design-§2 law —
  `(Rrs_ZTT + ΔRrs) × f_R + φ_C·K_fl(1 + δ_F)` with
  `f_R = 1 + (f_phys − 1)(1 + δ_R)` — via `corrections=None` resolving to
  the packaged heads (`robust/rt/files/{raman,fl}_corr_l23.npz`, 129
  params each, tanh bounds 1.0/0.5). `corrections=False` is the analytic
  backbone (the M2 pins use it, 13 call sites); `inelastic=None` is bitwise
  elastic. The a_ph requirement raises pre-emulator.
- **The per-process M4 gate lines are already met and test-pinned**
  (`test_inelastic_corr.py`, full release, held-out scenes, committed
  weights only): Raman increment median **−0.14/−0.10/−0.21 %** at
  0/30/60° (550–700 nm; gate ≤ 5 % — the analytic backbone's −38.6 % @ 0°
  closed), 490 nm ≤ +1.0 %; fluorescence 685 nm **+0.08/+0.07/+0.10 %**
  (analytic was −13.7 % @ 60°). The a_ph(440)-decile diagnostic is flat
  (±0.6 %; the analytic drifts −11 → +11 % across deciles). **M4's
  genuinely new gate numbers are (1) total held-out rRMS vs Rrs_X4 and
  (6) speed ≤ 2× elastic** — everything else is standing tests.
- **Gradient gates exist through both paths** (analytic:
  `test_inelastic.py`; corrected: `test_inelastic_corr.py`) for
  `a, bb_p, a_ph, φ_C, θ_s`, float64, per-variable steps, **θ_s at 35°** —
  packaged `Ed` is piecewise-linear with anchors at 0/30/60°, so the
  θ_s-derivative is one-sided at the anchors (kink; record §4.4).
- **The extrapolation finding M4's candor must carry** (record §5.3,
  notebook 4 §3): a δ_R trained without 60° produces **−74 %** increment
  error there — worse than no correction. The heads are interpolators in
  cos θ_s; the 3-zenith caveat is now a *measured* cliff, not a
  hypothetical. (φ_C-linearity likewise has truth only at 0.02;
  `emission_shape='double'` is unvalidatable; below 400 nm the excitation
  clamps — the model's stated domain is λ ≥ 400 nm, three zeniths,
  L23-like water.)
- **Speed is unmeasured** — the M4 risk. The fluorescence kernel's
  `(n_batch, n_em, 65)` contraction and the Raman factor's excitation
  interpolation are the new cost; the coding-plan fallback (precompute
  zenith-static quantities; fuse the quadrature) waits on a measurement,
  not a guess.
- **Weight regeneration is safe**: `design/py/train_inelastic_corr.py`
  writes candidate-beside-destination, verifies the reloaded δ, then
  `os.replace`s (the PR #18 Bugbot finding, fixed at M3 task 5 with the
  elastic `write_weights` pattern). ~60 s/head; deterministic from seeds.
- **Review pattern**: JXP issues a PR per milestone and triggers
  `@cursor review`; expect Bugbot findings to address alongside the
  task-3 self-review.
- **Notebook tooling**: kernelspec `ocean14`; execute via the `os_313`
  env's `jupyter nbconvert --execute`; build programmatically (`nbformat`
  from os_313) in the house style of notebooks 1–4; commit with outputs.

## Prompts

1. Read this doc. Execute the 1st task in the "M4" section below. If you have
   any questions, ask me in the Q&A section below. Use Fable if you can. Log
   your work.
2. Read this doc. Execute the 2nd task. Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task — the review pass. Use Fable if you
   can. Log your work.
4. Read this doc. Execute the 4th task — the notebook and the implementation
   record wrap-up. Check my answers in Q&A. Use Fable if you can. Log your
   work.

## M4

### Tasks

1. **Validation protocol.** Extend `robust/rt/validation.py` with the design
   §6 protocol: held-out (elastic splits) rrs-space rRMS vs `Rrs_X4`, per λ /
   per zenith; the per-process delta metrics; throughput vs the elastic
   hybrid (same batch, same machine); the gradient gate incl. φ_C; the
   elastic hash-regression. **Reuse over re-derivation**: the deltas,
   gradients, and hash lines are already standing tests
   (`test_inelastic_corr.py`, `test_inelastic.py`,
   `test_inelastic_types.py`) — the protocol should *compute and report*
   them through `validation.py`'s metric machinery (`rrms` is the metric,
   already differentiable), not fork new definitions; the genuinely new
   measurements are the **total rRMS vs X4** and the **speed ratio**.
   Diagnostics (reported, not gated): performance by a_ph(440) decile
   (measured flat ±0.6 % at M3 — confirm through the protocol); the
   φ_C-linearity check against the scaled-truth construction;
   `emission_shape='double'` behavior. If the ≤ 2× speed budget is
   threatened, apply the coding-plan fallback (precompute zenith-static
   quantities at trace time; fuse the excitation quadrature) — measure
   first, the fluorescence contraction is the suspect.

2. **Artifacts.** Extend `design/py/run_validation.py` so one command
   regenerates the inelastic metrics table + figures into
   `design/validation/` (alongside, not replacing, the elastic artifacts).
   Figures follow the recorded style conventions; every number quoted in
   the implementation record must be regenerable by this script.

   **Gate (acceptance = design §6).** `robust/tests/test_inelastic_validation.py`:
   (1) held-out total rRMS vs X4 ≤ **0.5 %** at each zenith; (2) Raman delta
   ≤ 5 % incl. 0°; (3) fluorescence delta ≤ 5 %; (4) `inelastic=None`
   bit-identical; (5) gradient checks for all inputs incl. φ_C; (6)
   full-batch forward ≤ **2×** elastic-hybrid runtime.

3. **Review pass (CQ6).** Run a code review over the full branch diff
   (e.g. the `code-review` skill at high effort); address the findings
   (fix or explicitly decline with a reason in the log) **before** the gate
   is declared. JXP's per-milestone pattern is a PR + `@cursor review`
   (M0's PR #14, M3's PR #18 — the Bugbot findings were real both times);
   address those comments too when the M4 PR appears, checking Q&A for
   pointers.

4. **Wrap-up.** Finish `design/rt_inelastic_implementation.md` (all
   milestones, environment, module index, results tables, the definition-
   of-done statement mirroring the elastic record §6.10). Notebook
   `notebooks/RT/rt_inelastic_coding_5.ipynb` — executed: the headline
   validation figures (total rRMS ladder incl. elastic-only baseline;
   per-process before/after; speed and gradient tables) with honest
   captions about what the prototype may and may not claim (mirror the
   elastic `prototype_summary.md` candor). The caveat list is now
   *measured*, not hypothetical — carry it verbatim from the record:
   the 3-zenith geometry cliff (−74 % at an unseen 60°, notebook 4 §3),
   φ_C-linearity with truth only at 0.02, `'double'` emission
   unvalidatable, λ ≥ 400 nm domain (excitation clamp), and the
   θ_s-anchor derivative kink. Note the branch state for JXP's PR.

### Q&A

**Q1 (Claude, 2026-08-26, task 1).** One gate-definition decision made that
deserves your veto: **the total-rRMS gate (design §6 line 1) is scored over
λ ≥ 400 nm** — the model's stated domain since M3 (the heads' training band;
below 400 nm the Raman excitation leaves the L23 grid and clamps, and the
heads never saw it). On that band the held-out corrected model measures
**0.352 / 0.339 / 0.349 %** at 0/30/60° — under the 0.5 % bar at every
zenith. On the *full* 350–750 nm grid it measures 2.61/2.27/2.28 %,
essentially all of it from the ten sub-400 nm bands (13 % rRMS at 350 nm,
falling to the gate level by 400). The design wrote "held-out rRMS ≤ 0.5 %"
before the domain restriction was measured; scoring a model outside its
stated domain seems wrong to me, but it is your gate. The full-grid number
will be *reported* in the metrics table either way (candor, not gating). If
you want the gate on the full grid instead, the prototype fails line 1 and
the fix is a training-band decision, not a validation one — say so before
task 2 pins the gate test.

>A. Right, do not gate on the rms outside the 400-700nm range.

## Next

The prototype is complete. Next steps live in
`claude_prompts/RT/rt_inelastic_prompts.md` (the Report prompts), and the
"Beyond v1" section of the coding plan (HydroLight wishlist) awaits JXP's
compute planning.

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-26 (M4 task 1 — the §6 protocol in validation.py; speed 6.3× → 1.6×; 423 green)

Model: Fable 5. Q&A checked first: **empty — nothing to answer** (prompt 4
closed everything). One new question asked above (Q1: the total-rRMS gate's
wavelength domain). Record §6 (v0.24) has the full task-1 write-up.

**The protocol** (`robust/rt/validation.py`, +7 public functions): the
per-process delta metrics under permanent names —
`median_increment_error` (Raman; `test_inelastic_corr.py`'s definition,
which now *delegates here*, so the M3 gate and the M4 table are one
quantity) and `peak_ratio_error` (fluorescence 685 nm; same reuse);
`quantile_bin_labels` (the generic form of `bp_bin_labels`, which now
delegates — deciles for the a_ph(440) diagnostic); `phi_c_linearity` (the
scaled-truth construction); `speed_ratio` (gate line 6 as a ratio, reusing
`throughput`); `INELASTIC_FD_STEPS` + `inelastic_gradient_report` (the
M2/M3 FD protocol for all six inputs incl. `B_p`, `a_ph`, `φ_C`; shares the
elastic report's classification rules via a factored `_grad_vs_fd`). Total
rRMS and per-λ/zenith cuts need nothing new — `rrms`/`group_rrms`/
`rrms_per_wavelength` already are the protocol. The elastic hash-regression
stays `test_inelastic_types.py`'s standing two-tier pin (reported through,
not re-derived). +7 tests in `test_validation.py` (hand-computed synthetic
references; the six-variable gradient gate through the real corrected
forward at θ_s = 35°, all ≤ 1e-6).

**Measured — the two genuinely new gate numbers.** (1) Held-out total rRMS
vs `Rrs_X4` (rrs space, all processes on, committed weights):
**0.352 / 0.339 / 0.349 %** at 0/30/60° over λ ≥ 400 nm — under the 0.5 %
gate; full-grid 2.61/2.27/2.28 % (the sub-400 clamp region — Q&A Q1).
Ladder: median 0.33 %, worst 0.84 % at 450 nm. The rRMS ladder: elastic-only
16.7 % → analytic inelastic 3.2 % → corrected 0.35 % (held-out, ≥ 400 nm).
(2) **Speed: first measurement 6.3×** the elastic hybrid (216 vs 33 ms,
full batch, jitted CPU) — the budget-threatened branch fired, fallback
applied (below) — **final median 1.60×** (52–55 vs 32–35 ms, 5 trials).

**The speed fallback (the coding-plan prescription, measured first).**
Profiling: `fluorescence_kernel` 167 ms of the 216 (the suspect confirmed);
heads ~20 ms each; `raman_factor` 1.4 ms. Three changes, all
gates/pins/xchecks green after each (423 passed, 1 skipped; ruff + format
clean; live-BING float64 pins untouched):
1. **Kernel quadrature fused** (`inelastic.py`): everything λ′-only
   (trapezoid weights, Ed(λ′), quanta→energy, source) folds into one
   `(..., n_ex)` numerator so the `(batch, n_em, 65)` tensor appears in a
   single divide-and-reduce — algebraically identical (float32 reorder
   ~7e-7). 167 → 17 ms.
2. **`optimization_barrier` on the reduced `r_f`** — the real discovery:
   XLA CPU consumer-fusion *re-ran the whole 52M-element reduction once per
   downstream use* of `r_f` (`rrs_to_Rrs` uses it twice, plus the emission
   line). The barrier pins it to materialize once — **bit-identical**
   output, differentiable (FD gates pass through it). 17 → 3.8 ms.
3. **Heads: flatten + fold** (`inelastic_corr.py::CorrectionHead.delta`):
   the (batch, wave) axes flatten to 2-D before the matmuls (XLA's threaded
   matmul path; bit-identical, 20 → 13 ms/head), and the feature
   standardization folds into Dense_0's kernel/bias
   (`(x−m)/s @ W = x @ (W/s) − (m/s)@W`; ULP-level, ~4e-7 on δ; a fresh
   head's δ stays exactly 0 since the output layer is zero-init).
   13 → 10 ms/head.
The elastic path was deliberately untouched (its strict hash pins are
bitwise; also keeping the denominator honest — optimizing the reference
would tighten the ratio for nothing).

**Diagnostics (reported, not gated), through the protocol functions:** the
a_ph(440)-decile fluorescence line is flat — max |err| **0.62 %** (decile 2
at +0.62; the eutrophic decile 10 at +0.00) — confirming M3.
`phi_c_linearity` at scales 0.5/1/2/5×: per-zenith errors
(+0.076/+0.072/+0.103 %) **identical across scales to <1e-4** — linear by
construction, as §4.4 promised. `emission_shape='double'`: −8.5 % at
685 nm, +9.8 % at 730 nm vs `'single'` (median, full batch); scored against
the single-shape truth it sits at −23.6 % at 685 — consistent with moving
25 % of the emission into an L23-invisible shoulder; unvalidatable, off
everywhere, reported only.

**Per-process deltas through the shared definitions** (held-out, unchanged
from M3, as required): Raman 550–700 nm −0.14/−0.10/−0.21 %, 490 nm
+1.03/+0.82/+0.58 %, fluorescence 685 nm +0.08/+0.07/+0.10 %.

**State for task 2:** `run_validation.py` extension + the committed
artifacts + `test_inelastic_validation.py` gate file. Gate lines (2)–(5)
are standing tests; (1) waits on Q1's answer for its wavelength domain;
(6) should assert the *median* of several `speed_ratio` trials (shared-
machine wander is ±5 %; 1.6× has real margin, single trials of the
pre-fallback code did not).

### 2026-08-26 (M4 task 2 — the §6 gate PASSES on all six lines; artifacts committed; 430 green)

Model: Fable 5. **Q&A: Q1 answered by JXP — "do not gate on the rms outside
the 400-700nm range" — no new questions.** The gate band is therefore
**400–700 nm**, defined once as `validation.INELASTIC_GATE_BAND` (one
definition, so the gate test and the committed table cannot score different
bands); the full 350–750 nm grid is reported alongside, never gated. Record
§6.6 (v0.25) has the full write-up.

**`robust/tests/test_inelastic_validation.py`** (7 tests) — the design-§6
acceptance spelled in one file, one numbered test per line, every metric
through the task-1 protocol functions. Where a line duplicates a standing
test (the deltas, the hash pins) that is deliberate and documented: the
standing tests earned the numbers, this file states the acceptance. Line 4
re-asserts `test_inelastic_types.py`'s strict SHA-256 tier under the gate's
name (same helper and pins imported — no second definition; machine-
anchored, CI keeps the closeness tier) plus release-level bitwise identity
of omitted/None/all-off. Line 6 asserts the **median of three
`speed_ratio` trials** (single ratios wander ±5 % here). Full-release lines
skip without `$OS_COLOR`; weight-dependent lines skip with a regenerate
message.

**`design/py/run_validation.py --inelastic`** — one command regenerates the
inelastic metrics table + figures into `design/validation/`, alongside the
elastic artifacts (`*_inelastic.*`, `inelastic_*.png`); committed weights
only, nothing trained, ~2 min; exits nonzero if any gate line fails. The
eager-loader lesson from task 1 is baked in (resolve `emulator`/`heads`
before anything jits — a memoised loader first touched inside a trace
caches tracers).

**The committed gate table** (`metrics_inelastic.md`): (1) total held-out
rRMS vs X4 over 400–700 nm **0.343/0.341/0.340 %** at 0/30/60° (full grid
2.61/2.27/2.28 %, reported); (2) Raman worst **1.03 %** (490 nm @ 0°);
(3) fluorescence worst **0.10 %**; (4) bit-identical **True**;
(5) gradients worst **5.9e-9** of six variables; (6) speed **1.67× median**
(55 vs 33 ms; trials 1.57–1.68). **All six lines PASS.** Figures: the
three-model ladder vs X4 (ungated regions shaded so the sub-400 cliff
stays visible — the corrected line crosses 0.5 % only at the 685 nm
fluorescence band and 450 nm, per-λ excursions the *total* gate absorbs)
and the nine-row analytic → corrected deltas plot (−38.6 → −0.14
headline).

**Results.** `pytest -q` → **430 passed, 1 skipped** (423 + 7 gate tests);
elastic hash pins green; ruff + format clean. **Gate lines (1)–(6) all
pass — the prototype's remaining work is task 3 (the review pass; findings
addressed before the gate is *declared*) and task 4 (notebook 5 + the
record wrap-up).**

### 2026-08-27 (M4 task 3 — the review pass: 8 angles, 13 findings fixed, 6 declined; gate re-verified; 431 green)

Model: Fable 5. **Q&A: no new answers found (Q1 remains the only entry,
already applied) and no new questions from this task.** No M4 PR exists yet,
so there were no Bugbot comments to fold in — when JXP opens the PR and
triggers `@cursor review`, any findings land as task-4-adjacent work.
Record §6.7 (v0.26) has the structured summary.

**The review**: the `code-review` skill at high effort over the full M4
branch diff (`0750970..HEAD`, 15 files) — eight finder angles (conventions,
simplification, reuse, altitude, efficiency, removed-behavior audit,
line-by-line, cross-file trace), deduped to ~15 distinct findings; the two
non-obvious ones verified empirically before acting. **Fixed (13):**

1. *Zenith-coverage regression in the M3 gates* (3 finders, independently —
   the review's best catch): my task-1 delegation to the shared metrics
   made `test_heldout_{raman,fluorescence}_gate` iterate whatever zenith
   groups exist; an empty group (split/loader regression) would pass
   vacuously where the pre-M4 hard-coded loops failed loudly. Both gates
   now assert `set(errors) == {0, 30, 60}` (module constant `ZENITHS`).
2. *Gate bars now share one definition*: `INELASTIC_GATE_TOTAL_RRMS/_DELTA/
   _SPEED` in `validation.py`; the gate test, the script's PASS/FAIL rows
   and both figures' annotated bars read them (each had its own literals —
   a tightened bar would have left a committed table contradicting pytest).
3. *`IC.corrected_fluorescence(delta, k_fl)` added* and used by
   `hybrid._apply_inelastic` (bit-identical expression), both gate files,
   the weights-regression test and the script — the Raman-rule symmetry the
   M3 design intended; four hand-spelled copies of `k_fl·(1+δ)` gone.
4. *Fold guard*: `test_delta_fold_matches_the_unfolded_network` pins the
   Dense_0 standardisation fold against the explicit unfolded path with
   noise params and non-trivial mean/std — the coupling to
   `emulator._network`'s layer naming now fails loudly instead of silently.
5. *`throughput` reuses jitted callables* (`jax.stages.Wrapped` check):
   the speed trial loops were recompiling both full forwards per trial —
   8 of the script's 10 speed-section compiles were pure waste.
6. *`speed_ratio(..., reverse=)`*: trials alternate measurement order;
   measured ordering bias ~6 % (within trial noise, but alternation is
   free and the median can now cancel it). Gate test + script alternate.
7. *`median_increment_error` slices to the band before dividing* — a truth
   factor of exactly 1 outside the scored band no longer emits
   divide-by-zero warnings.
8. *`needs_weights` moved to `conftest.py`* (was verbatim in three files).
9. *Duplicate six-variable gradient gate deleted* from `test_validation.py`
   — it lives once, as gate line 5 (`test_inelastic_validation.py`); the
   refusal-contract tests stay.
10. *Both x64 toggles in the script wrapped in try/finally* (elastic
    section too — same latent bug).
11. *`apply_house_style()`*: the 20-line rcParams block was pasted twice.
12. *Two redundant full-batch forwards removed* in the script (`omitted`
    and `single` derive from arrays already in hand;
    `forward ≡ rrs_to_Rrs(rrs_forward)`).
13. *Dead `wave` parameter / helper indirection removed* in
    `test_inelastic_corr.py` (the gates call the shared metrics directly);
    plus the nested-generator comprehension unrolled in the script.

**Declined (6, reasons):** per-process band literals (550–700/490/685/440
are design-§6-quoted physical definitions; centralizing churns M3-pinned
files against a hypothetical regrid); `speed_ratio` returning the ratio in
its tuple (ergonomics; all three callers want exactly that number);
package-level orchestration helpers for the script's gate rows (drift
channel closed by the shared constants; the rest is script plumbing);
generalizing the two FD-report closures into a spec-driven base (no third
consumer; the copies differ where the physics differs); jitting all ~12 of
the script's full-batch forwards (each distinct config pays a multi-second
compile for an eager total of a few seconds); `emission_shape`/decile
micro-optimizations beyond the two array reuses.

**Verification.** `pytest -q` → **431 passed, 1 skipped** (430 + 2
fold-guard − 1 deleted duplicate); elastic hash pins green; ruff + format
clean. Artifacts regenerated: **the §6 gate re-verified PASSED** — speed
median now **1.59×** (trials 1.51–1.60, tighter after compile-once +
alternation; was 1.67×), every other line unchanged (worst total 0.343 %,
Raman 1.03 %, fluorescence 0.10 %, gradients 5.9e-9, bit-identity True).
PNGs byte-identical. **Task 4 (notebook 5 + record wrap-up) is what stands
between here and "prototype done"; the branch is ready for JXP's M4 PR.**

### 2026-08-27 (M4 task 4 — notebook 5 + the record finished; **the inelastic prototype is done**)

Model: Fable 5. **Q&A: no new answers found (Q1 remains the only entry,
already applied at task 2) and no new questions — nothing pending.** Still
no M4 PR at execution time, so no Bugbot comments existed to address; when
JXP opens the PR, any `@cursor review` findings are post-wrap-up work.

**`notebooks/RT/rt_inelastic_coding_5.ipynb`** — built programmatically
(`nbformat` from `os_313`, notebooks 1–4 house style), executed via the
`os_313` nbconvert on the `ocean14` kernel, committed with outputs
(229 kB). Four sections (record §6.8): the total-rRMS ladder
(elastic-only → analytic → corrected, ungated regions shaded but visible);
the nine-row per-process before/after (−38.6 → −0.14 % at 0° as the top
line); speed + gradients **re-verified live in the notebook run** (this
run's median 1.71× printed beside the committed 1.59×, both under the 2×
gate — the ±5 % machine wander made visible rather than hidden; gradients
worst 5.9e-9; bit-identity re-checked True); and the claims/caveats
statement. The setup cell **asserts the recomputed gate-band totals equal
`design/validation/metrics_inelastic.csv`** before anything renders — the
elastic no-transcription rule, enforced by the notebook itself.

**The record is finished** (v0.27): §6.8 (notebook), §6.9 (definition of
done, mirroring the elastic §6.10 — the claim with its two recorded
amendments: branch `inelastic-rt` per prompt 1 Q2, gate band 400–700 nm
per prompt 5 Q1; and the five measured caveats verbatim: the −74 %
3-zenith cliff, φ_C truth only at 0.02, `'double'` unvalidatable at
−23.6 %, the sub-400 nm clamp at 13 %, the θ_s-anchor kink — plus the
elastic inheritances). Status table: M0–M4 all ✅; the §1 verification
block updated to the final count.

**Verification.** Final `pytest -q` → **431 passed, 1 skipped**; ruff +
format clean; elastic hash pins green; notebook figures inspected.
**The M4 gate is passed, reviewed, and recorded — the inelastic prototype
is complete.** Branch `inelastic-rt` holds M0–M4 ready for JXP's M4 PR
(`@cursor review` per the milestone pattern). Next steps:
`claude_prompts/RT/rt_inelastic_prompts.md` (the Report prompts) and the
coding plan's "Beyond v1" HydroLight wishlist, which awaits JXP's compute
planning.
