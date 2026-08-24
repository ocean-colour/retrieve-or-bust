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
5. Read this doc. Execute the 5th task — modifying the next prompt doc,
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

5. **Finally.** Modify the next prompt doc,
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

*(Carried over, still open: prompt 3 Q&A Q1 — keep or remove the committed
stray `rob/` directory.)*

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
