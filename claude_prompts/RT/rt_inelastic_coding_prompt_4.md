# Inelastic RT Coding — Prompt 4 (M3: Correction heads δ_R and δ_F)

## Goals

Implement **Milestone M3** of the coding plan
(`design/rt_inelastic_model_coding_plan.md`): the two bounded learned
corrections — δ_R on the Raman increment, δ_F on the φ_C-linear fluorescence
kernel — trained on the L23 X-difference truth channels, hitting the
per-process **5 % held-out gates at every zenith including 0°**. This is the
milestone that earns the design's accuracy: the analytic backbone alone fails
the Raman gate at high sun by −39 %.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `critical-partner`,
`code-review`) as helpful.

### Working agreements

As in `rt_inelastic_coding_prompt_1.md`: JXP runs git — branch
**`inelastic-rt`** (Q&A Q2 of prompt 1; *not* the coding plan's
`rt-inelastic-prototype`); `ocean14` on this machine, CPU JAX; reuse over
reinvention; every task `pytest`-gated; the **elastic hash-regression stays
green** — two-tier (record §2.8): strict SHA-256 pins on dev machines, the
ULP-closeness regression everywhere. Never `pip install -r requirements.txt`
wholesale (the `git+` lines clobber the editable `bing`/`ocpy` checkouts).
Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — M3 section + the δ_R-at-0° and eutrophic-tail risk
  entries (they carry the escalation paths).
- **Design** — `design/rt_inelastic_model.md` §4.5 (features, bounded
  forms, size budget), §2 (how the corrections compose), §4.4
  (φ_C-linearity: δ_F must not see φ_C).
- **The implementation record** — `design/rt_inelastic_implementation.md`
  §4 (v0.17+): what M2 actually built, its API, and the *measured* backbone
  errors M3 must close. Chronology: prompt 3's Logs.
- **The M2 tests** — `robust/tests/test_inelastic.py` (the characterization
  bands and the FD-gradient protocol to mirror) and
  `test_inelastic_bing_xcheck.py` (the sentinel + pins that must stay green).
- **Elastic emulator** — `robust/rt/emulator.py` and the training pipeline
  conventions (relative weighting, seeds, committed weights) plus
  `design/py/train_emulator.py`.

## Status entering M3

*(Filled by M2's final task, 2026-08-24. Details: record v0.17 §4;
chronology: prompt 3 Logs.)*

**M2 is complete — all five tasks, the code gate green** (390 passed,
`-W error` clean; xcheck green on this machine; elastic hash pins green;
ruff + format clean; notebook 3 executed and committed). What M3 can rely
on, and what it must not break:

- **The analytic terms are in, composed, and pinned to live BING.**
  - `robust.rt.inelastic.raman_factor(iops, geometry, wave) → f_phys` and
    `fluorescence_kernel(iops, geometry, wave, emission_shape='single') →
    K_fl`, both pure JAX, batched, `jit`/`vmap`-safe, differentiable in
    every input. Cross-checked against the fixed BING at rtol ≤ 1e-6 on all
    150 fixture samples in float64 (measured worst 4.1e-16 / 1.1e-13);
    float32 sits at ~3e-6 (trapezoid accumulation) — compare in x64.
  - **`K_fl` is defined at the truth's yield**: `PHI_C_REF = 0.02`,
    `K_fl = Rrs_fl(0.02)/0.02` (record §4.3) — so `φ_C·K_fl` is exactly
    BING at φ_C = 0.02 and φ_C-linear by construction elsewhere. δ_F
    multiplies this kernel and **must not see φ_C** (design §4.4); the
    identity `∂Rrs/∂φ_C = K_fl·(1+δ_F)` should survive M3.
  - Composition lives in one place, `hybrid._apply_inelastic`: up-convert
    once, `× f_R`, `+ φ_C·K_fl`, down-convert once. `inelastic=None` and
    all-off return the *same object* (hash pins by construction).
    Fluorescence without `IOPs.a_ph` is a clear `ValueError` at two entry
    points (`rrs_forward` pre-check + the kernel). `mode='emulator'`
    refuses composition. M3's `f_R = 1 + (f_phys − 1)(1 + δ_R)` and
    `× (1 + δ_F)` slot into these exact lines.
- **The targets, measured on the CI fixture and pinned as bands**
  (`test_inelastic.py`; M3's gates are ≤ 5 % where these characterize):
  Raman increment error (median) **+1.6 / −4.0 / −38.6 %** at 30/60/0°
  over 550–700 nm, **+29.7 / +30.4 / −3.0 %** at 490 nm (30/60/0°);
  fluorescence 685 nm model/truth **0.991 / 0.937 / 0.853** at 0/30/60°.
  *The recompute lesson stands*: these are 50-scene-fixture numbers
  (Raman factor max 1.36 there vs 2.51 on the full release) — training
  uses the full release (`$OS_COLOR`), so recompute any threshold on the
  set actually used. The M2 bands pin the **analytic** terms and must stay
  green untouched — the heads are a separate path behind weights.
- **The gradient gate protocol is established**
  (`test_gradient_matches_finite_differences_composed`): float64 under
  `jax_x64`, per-variable steps, through the full composed forward incl.
  `a_ph` and `φ_C`. **Caveat discovered there**: packaged `Ed` is
  piecewise-linear in θ_s with anchors at 0/30/60°, so the θ_s-derivative
  is *one-sided at the anchors* (kink; the gate evaluates at 35°). δ_R
  takes `cos θ_s` as a feature — training at exactly the anchor zeniths is
  fine (values, not θ-derivatives), but any θ_s-refinement work must
  remember the knots.
- **The xcheck sentinel guards the reference**: if
  `test_inelastic_bing_xcheck.py` ever fails its sentinel, the bing
  checkout rolled back to pre-`inelastic-fixes` — fix the checkout, do not
  touch the ports.
- **Splits**: elastic-split reuse is proved mask-for-mask
  (`robust.rt.data.l23.select_inelastic`, M1 task 3) — train/held-out
  scene selections come from there, as the elastic effort's did.
- **Housekeeping still open**: prompt 3 Q&A Q1 — the stray `rob/`
  directory (empty `__init__.py` + a Dutkiewicz-2015 optics table) was
  committed in `2c1c259`; JXP has not yet said keep-or-remove.
- **Notebook tooling**: kernelspec `ocean14`; execute via the `os_313`
  env's `jupyter nbconvert --execute`; commit with outputs. Notebooks are
  built programmatically (`nbformat` from `os_313`) in the house style.

## Prompts

1. Read this doc. Execute the 1st task in the "M3" section below. If you have
   any questions, ask me in the Q&A section below. Use Fable if you can. Log
   your work.
2. Read this doc. Execute the 2nd task. Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task. Check my answers in Q&A; if you have
   additional questions, ask in Q&A. Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can.
   Log your work.
5. Read this doc. Execute the 5th task. Use Fable if you can. Log your work.
6. Read this doc. Execute the 6th task — modifying the next prompt doc,
   `rt_inelastic_coding_prompt_5.md`. Use Fable if you can. Log your work.

## M3

### Tasks

1. **Heads.** `robust/rt/inelastic_corr.py`: two small Flax MLPs with
   bounded (`tanh`-scaled) outputs, per design §4.5 —
   `f_R = 1 + (f_phys − 1)(1 + δ_R)` with per-(scene, λ) features
   `a(λ), bb(λ), a(λ′), bb(λ′), cos θ_s, λ` (the λ′ values via the M1
   helpers **by name**: `conventions.raman_excitation` +
   `conventions.interp_spectrum` — the same calls `raman_factor` makes);
   `Rrs_fl = φ_C · K_fl · (1 + δ_F)` with features
   `a_ph(440), a(λ_em), bb(λ_em), a(490), cos θ_s, λ_em` and **no φ_C
   input** (K_fl's `PHI_C_REF` mechanics, record §4.3 — φ_C-linearity is
   the design promise δ_F must preserve). Start minimal (O(10²) params);
   grow only if the held-out gate demands it. Wire into
   `hybrid._apply_inelastic` — the two composition lines are built to take
   these exact forms (Status) — with weights loaded lazily from
   `robust/rt/files/`, absent-weights → analytic-only with a warning.

2. **Training.** `design/py/train_inelastic_corr.py`: Optax, fixed seeds,
   **train split only, all three zeniths**; splits via the proved
   `l23.select_inelastic` machinery on the **full release** (`$OS_COLOR` —
   not the 50-scene fixture; recompute any threshold on the set actually
   used, the standing M1 lesson); targets = the X2/X1 and X4−X2
   channels; **relatively weighted** losses (the BING/elastic lesson —
   unweighted lets the red run away); monitor the a_ph(440)-decile
   diagnostic for the fluorescence tail during training. Train both heads;
   commit `robust/rt/files/raman_corr_l23.npz` and `fl_corr_l23.npz`.
   Record architectures, sizes, and training curves in the implementation
   record. As a *reported* diagnostic (not gated), also train a
   zenith-holdout variant (train 0°/30°, test 60°) to measure geometry
   generalization, mirroring the elastic CQ6 split.

3. **Held-out gates.** `robust/tests/test_inelastic_corr.py` on the
   **held-out scenes**, loading the *committed* weights (no
   train-at-test-time): Raman increment median |error| ≤ **5 %** over
   550–700 nm at each zenith **including 0°** (the backbone starts at
   −38.6 % there); fluorescence 685 nm peak median |error| ≤ **5 %** per
   zenith (backbone: 0.85 at 60°); bounded-output property (corrections
   cannot exceed their tanh bound); a gradient check through the corrected
   path (mirror the M2 FD protocol — float64, per-variable steps, θ_s away
   from the 0/30/60° Ed anchors, where the θ-derivative is one-sided);
   **and the whole M2 gate untouched**: the analytic characterization
   bands, the bing xcheck, and the elastic hash-regression all stay green.
   If the 0° Raman gate cannot be reached, follow the coding plan's
   escalation path and ask JXP in Q&A **before** relaxing anything.
   Update the implementation record.

4. **Notebook.** `notebooks/RT/rt_inelastic_coding_4.ipynb` — executed
   (kernelspec `ocean14`, run via the `os_313` nbconvert; build
   programmatically with `nbformat` in the house style of notebooks 1–3).
   Show: before/after error spectra per zenith for both processes (the
   −39 % @ 0° closing); the a_ph(440)-decile diagnostic; the zenith-holdout
   result with an honest caption; head sizes vs the elastic emulator's 417
   params.

5. **Review.** I have issued a Pull Request.  Please examine the comments there
   and make any necessary changes.  Use Fable if you can.  Log your work.

6. **Finally.** Modify the next prompt doc,
   `rt_inelastic_coding_prompt_5.md`, given what M3 actually established.
   Log your work.

### Q&A

**Q1 (Claude, 2026-08-24, task 1).** One API decision made that deserves
your veto option: **the trained corrections become `forward`'s default
path** once task 2 commits weights. Concretely, `forward(...,
corrections=None)` (the default) resolves to the packaged heads; while the
weight files don't exist it degrades to analytic-only behind a single
`MissingCorrectionWarning`; `corrections=False` selects the analytic model
explicitly and silently. Rationale: design §2 defines `f_R` *with* δ_R —
the corrected model is the model, and a default that silently omitted the
trained physics would be the trap. Consequence, already applied: the M2
characterization tests pin the analytic terms via `corrections=False`
(13 call sites, deliberate), so they stay green bit-for-bit when weights
land. If you'd rather corrections be opt-in (default analytic even when
trained), say so before task 2 commits weights — it's a one-line default
flip plus doc updates now, a behavior change for downstream users later.

>A. Your move is fine.

*(Carried over, still open: prompt 3 Q&A Q1 — keep or remove the committed
stray `rob/` directory.)*

>A. Keep it

## Next

→ `rt_inelastic_coding_prompt_5.md` (M4: validation — prototype complete).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-24 (M3 task 1 — head machinery built, wired, gated; 408 green)

`robust/rt/inelastic_corr.py` + the `corrections` wiring are in; record §5.2
(v0.19). Model: Fable 5. Q&A: **one new question above (Q1: corrected model
as `forward`'s default once weights land — veto window open until task 2)**;
prompt 3's Q1 (the `rob/` directory) is still unanswered and carried over.
(Prompt arrived as doc 3's "execute the 5th prompt" with all five of doc 3's
tasks done — read as this doc's prompt 1, the same convention as every
prior hand-off.)

- **Module**: `HeadConfig` (per-kind `delta_max` from the measured errors:
  1.0 raman — closing −39 % needs +0.64 — and 0.5 fl), `CorrectionHead`
  pytree with `.delta(iops, geometry, wave)`, `CorrectionHeads` pair
  (`None` field = analytic by omission), design-§4.5 features with log10
  IOP columns (floored at 1e-10; excitation values via the M1 helpers by
  name; **no φ_C column for δ_F**), `save_head`/`load_head` in the
  emulator format with feature-mismatch refusal, cached `load_default`
  with the analytic fallback behind one `MissingCorrectionWarning`, and
  `corrected_raman_factor` written once for wiring and training to share.
  Machinery reused from the elastic emulator (`_network`/`_delta`
  duck-typed) — zero-init output layer, so **a fresh head is exactly the
  analytic model**; training starts from the physics.
- **Wiring**: `forward`/`rrs_forward` gain `corrections=None`
  (packaged-default / `False` = analytic-explicit / instance = as given);
  resolution only when an inelastic process is on — the elastic path never
  imports the ML stack or warns. `_apply_inelastic` applies
  `1 + (f_phys−1)(1+δ_R)` and `× (1+δ_F)`.
- **Deliberate M2-test update**: `test_inelastic.py`'s 13 forward calls now
  pass `corrections=False` (module docstring says why) — they pin the
  *analytic* terms and stay green bit-for-bit when trained weights land.
- **Tests** (`test_inelastic_corr.py`, 18): feature lists (incl. no-φ_C),
  δ ≡ 0 fresh, tanh bound under noise params, increment-form identities,
  zero-heads == analytic through `forward`, per-head band selectivity
  (fl head inert away from 685), **φ_C-linearity with a live randomized
  δ_F head**, grads w.r.t. head params (task 2's training path) and inputs,
  `jit`, save/load round trip, mismatch refusal, the one-warning fallback
  (skips itself once real weights exist), elastic-path-never-warns.
- Suite **408 passed** (390 + 18); elastic hash pins green; ruff + format
  clean.

Task 2 (training) starts with the objective's composition helper and the
parameter-gradient path already tested; it needs the full L23 release
(`$OS_COLOR`), `select_inelastic` splits, relatively-weighted losses, and
commits `robust/rt/files/{raman,fl}_corr_l23.npz` — at which point the
fallback-warning test steps aside automatically and the corrected model
becomes the default `forward` (Q1).

### 2026-08-24 (M3 task 2 — heads trained on the full release, weights committed; gates beaten ~25×)

`design/py/train_inelastic_corr.py` + committed weights; record §5.3
(v0.20). Model: Fable 5. Q&A: **no answers found — Q1 (corrections
default-on) drew no veto before task 2, so it now stands: with the weight
files committed, `forward(..., inelastic=...)` is the corrected model by
default**; prompt 3's Q1 (`rob/`) is still open. No new questions from this
task. (Prompt arrived as "execute the 1st prompt" with task 1 done and
committed in `3d3bd30` — read as task 2 per the standing convention.)

- **Training**: full release (9960 × 81, `$OS_COLOR`), elastic splits
  verbatim, full-batch Adam, fixed seeds, ~60 s/head. Relatively-weighted
  losses + the elastic size penalty: δ_R on the relative *increment* error
  (λ ≥ 400 nm), train fit 24.8 → 1.69 %, |δ|rms 30.6 %, max 0.905 (bound
  1.0 — ~10 % headroom at the extreme); δ_F on the emission window
  normalized per scene by its own 685 nm truth, train fit 5.6 → 0.77 %,
  |δ|rms 7.1 %, max 0.34 (bound 0.5). 129 params/head — the budget's low
  end sufficed.
- **Held-out scenes (the numbers task 3 must pin)**: Raman increment
  −38.6→−0.14 / +1.2→−0.10 / −4.2→−0.21 % (550–700, 0/30/60°); 490 nm
  −3.6→+1.0 / +30.9→+0.8 / +32.6→+0.6 %; fluorescence 685
  +0.3→+0.08 / −5.2→+0.07 / −13.7→+0.10 %. Worst gate metric 0.21 %
  (Raman), 0.10 % (fl) vs the 5 % bars. a_ph(440) deciles all within
  ±0.6 % — no eutrophic tail. *Full-release analytic medians differ from
  the fixture's (−5.2 vs −6.3 % @30° fl; +32.6 vs +30.4 % @490/60°) — the
  recompute lesson; pin task 3's held-out bands on these.*
- **Zenith-holdout diagnostic (reported, not shipped)**: at the unseen 60°,
  δ_R collapses to **−74 %** median increment error — far worse than the
  analytic backbone it corrects — and δ_F sits at −9.2 %. The elastic
  extrapolation caveat in sharper form: the heads are interpolators in
  cos θ_s; unseen geometries need coverage or a domain guard. Honest
  caption required in notebook 4.
- **Weights committed-ready**: `robust/rt/files/{raman,fl}_corr_l23.npz`
  (4.2/4.3 kB), reload-verified at write time (the refusal rule runs in the
  script, not at first use). Suite: **407 passed, 1 skipped** — the skip is
  the task-1 fallback-warning test retiring itself exactly as designed;
  the M2 analytic pins (all `corrections=False`) green untouched; elastic
  hash pins green; ruff + format clean.

Task 3 (held-out gates) has its numbers measured and logged above; it needs
`test_inelastic_corr.py` extended with the committed-weights gate tests
(≤ 5 % per process per zenith, bounded-output on the *loaded* heads, a
corrected-path FD gradient check with θ_s off the Ed anchors) — no
train-at-test-time, and the whole M2 gate stays green untouched.

### 2026-08-24 (M3 task 3 — held-out gates pinned; the M3 code gate is green; 416)

`test_inelastic_corr.py` extended with the acceptance gates; record §5.4
(v0.21). Model: Fable 5. Q&A: **no answers found — nothing new from this
task either**; still open: prompt 4 Q1 (default-on — now in effect, veto
still possible but is a behavior change) and prompt 3 Q1 (`rob/`).
(Prompt arrived as "execute the 2nd prompt" with task 2 done and committed
in `40becd8` — read as task 3 per the standing convention.)

- **Acceptance gates** (full release, held-out scenes, committed weights
  only — no train-at-test-time; `needs_l23_inelastic` so CI skips while
  this machine enforces): Raman median |increment error| ≤ 5 % over
  550–700 nm at every zenith incl. 0°, the 490 nm row at the same bar;
  fluorescence median |685 nm error| ≤ 5 % per zenith. All pass with
  ~25× margin (§5.3's table).
- **Bounds on the loaded heads** over the whole release (also the
  saturation canary for δ_R's 0.905/1.0 extreme).
- **Weights-integrity regression**, CI-runnable on the fixture (±2 %
  bands): catches a corrupt/stale/reverted weight file anywhere the repo
  runs, without `$OS_COLOR`.
- **Corrected-path FD gradient gate** (a, bb_p, a_ph, φ_C, θ_s): the M2
  protocol with `corrections=load_default()` — the inversion's actual
  differentiation path, through both tanh heads, pinned. θ_s at 35°, off
  the Ed anchors (the M2 kink caveat).
- Suite **416 passed, 1 skipped** (the task-1 fallback test, retired by
  design); CI simulation: 3 gates skip, regression + gradients run (the
  lone warning there is ocpy's own `OS_COLOR not set` import notice —
  external, pre-existing). M2 gate untouched; elastic hash pins green;
  ruff + format clean. **The M3 code gate (task 3's "Gate" line) is met.**

Task 4 (the notebook) is presentation: before/after error spectra per
zenith for both processes (the −39 % @ 0° closing), the a_ph(440)-decile
table, the zenith-holdout collapse (−74 % at unseen 60° — honest caption),
and head sizes (129 params each) vs the elastic emulator's 417. All
numbers are in §5.3–5.4 and the training log; lift, don't re-derive.

### 2026-08-25 (M3 task 4 — notebook 4 built and executed; Q&A answered and closed; 416 green)

`notebooks/RT/rt_inelastic_coding_4.ipynb` — executed with outputs
(321 kB), record §5.5 (v0.22). Model: Fable 5. **Q&A: both open items are
now ANSWERED and closed** — Q1 (corrections default-on): *"Your move is
fine"*, so the corrected-by-default `forward` stands; prompt 3's `rob/`
question: *"Keep it"*, so the directory stays. **No new questions.**
(Prompt arrived as "execute the 4th prompt" with tasks 1–3 done — for once
the number and the next task coincide.)

- Built programmatically (`nbformat`, os_313), house style, fixture-free:
  recomputes from the committed weights on the **full release**, held-out
  scenes only; the sole in-notebook training is §3's deliberately crippled
  zenith-holdout δ_R (the training script's own `fit_head` imported —
  reuse over reinvention).
- §1 Raman before/after per zenith: the −38.6 % @ 0° and +30 % @ 490 nm
  analytic structure flattened into the ±5 % band (plots sliced to
  λ ≥ 400 nm — the official band; the first execution's autoscale exposed
  the sub-400 clamp region where the heads never trained, +400 % medians —
  sliced, not hidden: it is outside the model's stated domain).
- §2 Fluorescence gate bars (−13.7 → +0.1 % at 60°) + the a_ph(440)-decile
  line — a finding worth keeping: the *analytic* 685 nm error runs from
  −11 % (clearest decile) to **+11 % (eutrophic tail)** — clean
  biomass-dependent structure the zenith medians averaged away; corrected
  is flat at ±0.6 %.
- §3 the honest panel: crippled δ_R at the unseen 60° = **−65.5 %** at
  1500 steps (−74 % at 3000), worse than no correction; caption states the
  interpolator-in-cos θ_s rule and the coverage-or-domain-guard condition.
- §4 economics (129 vs 417 params) and the M4 inheritance list.
- Suite untouched: **416 passed, 1 skipped**; ruff clean; hash pins green.

Task 5 closes M3: write `rt_inelastic_coding_prompt_5.md`'s Status from
record §5 (M4 = validation: total held-out rRMS ≤ 0.5 %/zenith vs Rrs_X4,
per-process deltas ≤ 5 % (met), elastic hash, gradient gate incl. φ_C
(met), speed ≤ 2× elastic, review pass CQ6, metrics + figures into
design/validation/) — and carry the closed Q&A state forward.

### 2026-08-26 (M3 task 5 — PR #18 review pass; 1 finding, fixed)

The JXP-inserted review task; record v0.22.1. Model: Fable 5. Q&A: **no new
answers needed — both prior questions stand answered and closed** (Q1
"Your move is fine"; `rob/` "Keep it"); this task raises **no new
questions**. (Doc edit noted: task 5 is now the PR review, the prompt-doc
hand-off renumbered to 6 — the Prompts list and the record's task table
follow.)

- **PR #18 ("Inelastic rt : M3") comments examined**: one inline finding
  (Cursor Bugbot, medium): `train_inelastic_corr.py` wrote the weight
  files **directly to the destination and verified afterwards** — a crash
  mid-write, a failed refusal check, or a broken serialisation would
  destroy the previous known-good committed weights. The same defect the
  elastic effort's PR #11 review caught in `train_emulator.py`; this script
  failed to inherit the fix. Valid on all counts.
- **Fixed with the elastic pattern**, as a `write_head(head, batch, out)`
  helper: candidate written *beside* the destination (`tempfile.mkstemp`),
  reloaded (the feature-mismatch refusal runs on the candidate), required
  to **reproduce the trained head's δ on the full batch**, permissions
  restored from the umask, then atomic `os.replace`; the temp file is
  removed on every exit path, and a failed round trip returns exit 1 with
  the destination untouched.
- **Exercised**: a 50-step run writing to a scratch directory took the new
  path end to end (both files written, round trip verified); the committed
  weights are byte-untouched (`git status` clean on `robust/`), no stray
  temp files, `test_inelastic_corr.py` 26 passed / 1 skipped, ruff +
  format clean. No retrain needed — the fix changes only the write path,
  not the weights.

Task 6 (the prompt-5 hand-off) closes M3.

### 2026-08-26 (M3 task 6 — prompt 5 rewritten from what M3 established; M3 COMPLETE)

`rt_inelastic_coding_prompt_5.md` updated; record v0.23 (M3 ✅, PR #18
merged). Model: Fable 5. Q&A: **no new answers and none needed — both prior
questions remain answered and closed**; this task raises **no new
questions**. (Prompt arrived as "execute the 6th prompt" — task 6 was next;
numbering aligned.)

What changed in prompt 5, and why:

- **Working agreements corrected** (the draft carried the stale
  `rt-inelastic-prototype` branch name; now `inelastic-rt`, the two-tier
  hash gate, and the no-wholesale-pip rule).
- **"Status entering M4" written** (the draft's empty block): M3 complete
  at 416 + 1-skipped with PR #18 merged; the corrected-by-default forward
  and its three switches (`corrections=None/False`, `inelastic=None`); the
  per-process gate lines *already met and pinned* (−0.14/−0.10/−0.21 %
  Raman, +0.08/+0.07/+0.10 % fluorescence — so M4's genuinely new numbers
  are the total rRMS vs X4 and the ≤ 2× speed ratio); both FD gradient
  gates and the θ_s-anchor kink; the measured −74 % extrapolation cliff
  the candor section must carry; speed unmeasured (the M4 risk, with the
  fluorescence contraction named as the suspect); the safe weight-write
  path; the PR + `@cursor review` pattern; notebook tooling.
- **Task text sharpened**: task 1 told to *report through* the standing
  tests and `validation.rrms` rather than fork definitions, and to measure
  before applying the speed fallback; task 3 carries the per-milestone
  PR/Bugbot pattern (real findings at M0 and M3); task 4's caveat list is
  now the *measured* one (geometry cliff, φ_C-at-0.02, 'double',
  λ ≥ 400 nm domain, θ_s kink), to be carried verbatim from the record.

**M3 is closed**: two 129-parameter bounded heads, trained on the elastic
splits of the full release, take every per-process error from the analytic
backbone's −39 %/−14 % failures to ≤ 0.21 % on held-out scenes — with the
φ_C handle intact, the extrapolation limit measured and documented, and
the whole M0–M2 gate stack green beneath them.
→ `rt_inelastic_coding_prompt_5.md` (M4: validation — prototype complete).
