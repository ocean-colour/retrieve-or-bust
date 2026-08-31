# Validation

{mod}`robust.rt.validation` is the protocol every accuracy, speed and gradient
claim on this site was produced by. Its module docstring states the stance in one
line: it is **a protocol, not a target**. Absolute rRMS and absolute latency are
*reported*; only the relative comparison — against a baseline, against the
elastic model, against a finite difference — is *gated*. One shared
implementation scores every model, so the number in a milestone log, the number
in a committed table and the number on this site are the same quantity or none of
the comparisons mean anything.

Three axes, from the design: **accurate**, **fast**, **differentiable**.

*Sources for this page: the {mod}`robust.rt.validation` module docstring and the
docstrings of every function named below;
[`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §6 (the
acceptance gate) and [`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md)
§§6–7 (the elastic protocol);
[`reports/report_rt_inelastic_model.md`](gh:reports/report_rt_inelastic_model.md)
§§3–4 and [`reports/report_rt_elastic_model.md`](gh:reports/report_rt_elastic_model.md)
§§3–4 (the results); and the generated artifacts
[`design/validation/metrics_inelastic.md`](gh:design/validation/metrics_inelastic.md)
and [`design/validation/metrics.md`](gh:design/validation/metrics.md), which are
what the reports quote. The gate table below was diffed cell by cell against the
report and the implementation record; the numbers labelled "re-measured here"
were reproduced in this environment when the page was written.*

## Accurate: rRMS, and the space it is computed in

{func}`~robust.rt.validation.rrms` is the metric, and it is one line:

$$\mathrm{rRMS} = 100 \times
\sqrt{\left\langle \left(\frac{\hat{y} - y}{y}\right)^{2} \right\rangle}\ \ [\%]$$

Two choices in it are load-bearing.

**It is relative, not absolute.** $R_{rs}$ spans more than a decade across the
spectrum — L23 runs from ~2.5×10⁻² in the blue to ~6×10⁻⁶ in the red — so an
absolute RMS would be almost entirely a statement about the blue, and a model
could look excellent while being useless past 600 nm. The relative form is also
BING's definition, which is what makes these numbers comparable with the
pre-existing rRMS ladder rather than a private scale.

**It is computed in $r_{rs}$, not $R_{rs}$.** The air–water interface is
non-linear ({doc}`../model/conventions`), so relative error in $R_{rs}$ is simply
a different number — a 6–14 % departure from a linear rescaling sits between the
two over the ocean range. Every table on this site is $r_{rs}$-space unless it
says otherwise, and `truth` must be non-zero because the metric divides by it.

`rrms` is pure JAX and differentiable, which is why it can double as the
emulator's training loss ({doc}`../model/emulator`) — the model is trained on the
metric it is scored with.

### The breakdowns

A single scalar was never enough: standard Gordon buys a respectable total by
fitting the bright blue and abandoning the dark red (2.5 % at 400 nm rising to
9.0 % at 700 nm). So the design asks for three cuts, and each is a named
function.

| function | the cut | why it exists |
| --- | --- | --- |
| {func}`~robust.rt.validation.rrms_per_wavelength` | rRMS per λ, shape `(n_wave,)` | shows whether a good total is being bought in one band at another's expense |
| {func}`~robust.rt.validation.group_rrms` | rRMS within each labelled group of samples | the per-zenith and per-`B_p`-bin rows; empty groups are omitted rather than reported as `NaN` |
| {func}`~robust.rt.validation.bp_bin_labels` / {func}`~robust.rt.validation.quantile_bin_labels` | equal-**count** bins over `B_p` or any per-sample quantity | L23's `B_p` spans a factor 1.75, so equal-*width* bins would put nearly every sample in the middle two |

`group_rrms` takes host-side NumPy labels — `batch.zenith`, or a bin-label array
— because a group label is metadata, not something to differentiate.
`quantile_bin_labels` is the generic form; `bp_bin_labels` delegates to it, so the
`B_p` cut and the trophic-state deciles are one implementation.

## The gated domain is 400–700 nm

This is the single most important qualifier on the headline accuracy number, and
it is stated here rather than in a footnote.

{data}`~robust.rt.validation.INELASTIC_GATE_BAND` is `(400.0, 700.0)` — **61 of
the 81 canonical grid points** (re-measured here). The total-rRMS acceptance line
is scored over that band, *not* over the full 350–750 nm grid the model computes
on, for a reason that is physics rather than presentation: below 400 nm the Raman
excitation wavelengths fall off the 350 nm edge of the L23 grid and the term runs
on clamped, extrapolated IOPs, and the correction heads never trained there.
400 nm is the model's stated domain, and has been since M3
({doc}`../model/inelastic`).

The band was **JXP's decision**, taken before the gate test was written, in
answer to a question that laid out both options: *"do not gate on the rms outside
the 400-700nm range"* (`claude_prompts/RT/rt_inelastic_coding_prompt_5.md`, Q&A
Q1). Scoring a model outside its stated domain is not a stricter test, it is a
different one.

Two safeguards make that honest rather than convenient:

1. **One definition.** The band is a module constant, so the acceptance test and
   `run_validation.py`'s committed table cannot score different bands.
2. **The full grid is always reported, never gated.** Both numbers appear
   side by side in the artifact, in the report, and in the table below. Over the
   full grid the corrected model reads 2.61/2.27/2.28 % against 0.34 % in band,
   and the twenty excluded points alone score 5.23/4.53/4.56 %. Split those
   twenty and the asymmetry is total (all re-measured here, held-out scenes):

   | domain | points | 0° | 30° | 60° |
   | --- | --- | --- | --- | --- |
   | 400–700 nm (gated) | 61 | 0.34 | 0.34 | 0.34 |
   | above 700 nm | 10 | 0.40 | 0.33 | 0.40 |
   | below 400 nm | 10 | 7.38 | 6.39 | 6.43 |

   So the red shoulder is *already* at the gate standard and the exclusion buys
   the model nothing there; the entire cost is the ten sub-400 nm bands, rising
   to 13 % at 350 nm. That is not hidden; it is *why* the band exists, and it is
   item 4 of the report's may-not-claim list ({doc}`limitations`).

## The inelastic increment metrics

A total can hide compensating errors, so the design gates each process
separately, on the exact truth the paired L23 scenarios give ({doc}`data`).

{func}`~robust.rt.validation.median_increment_error` — **Raman**.
`median((f_model − 1) / (f_truth − 1) − 1)` over a wavelength band, per group.
Scored on the *increment*, not the factor: the factor itself is `1 + small`, so a
ratio of factors would hide a large error in the small part. The demonstration is
the analytic backbone, whose factor looks fine while this metric reads −38.6 % at
0°. Fractional, not percent — the ≤ 5 % bar is `abs(value) <= 0.05`.

{func}`~robust.rt.validation.peak_ratio_error` — **fluorescence**.
`median(model[:, index] / truth[:, index]) − 1` per group, at the 685 nm peak. The
median of the *ratio* rather than the ratio of medians, so every scene counts once
and a handful of eutrophic outliers cannot carry the statistic.

Both are the definitions `test_inelastic_corr.py` delegates to, which is what
makes the M3 gate and the M4 table one quantity rather than two similar ones.

{func}`~robust.rt.validation.phi_c_linearity` — **the φ_C diagnostic, and an
honest one**. No varied-φ_C truth exists: HydroLight ran X = 4 at exactly one
yield. So the design asks for the next best thing — scale the *truth* linearly,
`truth(s) = s · (Rrs_X4 − Rrs_X2)`, evaluate the model at `φ_C = s · φ_ref`, and
report `peak_ratio_error` at each scale. A model exactly linear in φ_C reports the
*same* error at every scale, and drift across scales is nonlinearity leaking in.
**Reported, never gated**: the construction has real truth only at `s = 1`.
Identical rows are a check on the construction, not evidence about the ocean
({doc}`../model/fluorescence`).

## Fast

{func}`~robust.rt.validation.throughput` returns seconds per jitted call and
samples·λ per second, after a warm-up call that pays the XLA compile outside the
timed region. {func}`~robust.rt.validation.speed_ratio` divides one model's
timing by another's on **identical arguments**.

The ratio is the gated number, and the docstrings are emphatic about why:
wall-clock on a shared machine wanders ~20 % between runs, while the ratio of two
back-to-back timings reproduces. Two details in `speed_ratio` are there because a
review found them missing: an already-`jax.jit`-wrapped callable is reused rather
than re-wrapped (re-wrapping discards the compile cache, so a trial loop pays a
full compile per trial), and `reverse=` exists so a trial loop can **alternate
which model runs first** — whichever runs first sees a slightly different machine
state, and a fixed order would repeat the same bias in every trial where the
median could not cancel it.

## Differentiable

Three reports, one comparison: `jax.grad` against central finite differences,
per input variable.

| function | certifies | variables |
| --- | --- | --- |
| {func}`~robust.rt.validation.gradient_report` | the elastic model | `a`, `bb_p`, `B_p`, `theta_s` — {data}`~robust.rt.validation.FD_STEPS` |
| {func}`~robust.rt.validation.inelastic_gradient_report` | the composed corrected inelastic forward | those four **plus** `a_ph` and `phi_C` — {data}`~robust.rt.validation.INELASTIC_FD_STEPS` |
| {func}`~robust.rt.validation.cdom_gradient_report` | the forward with all three inelastic processes on | those six **plus** `a_cdom` and `scale` — {data}`~robust.rt.validation.CDOM_FD_STEPS` |

The tolerance is {data}`~robust.rt.validation.GRADIENT_TOL` = 1e-6, unchanged
since M2, and it is a **hard** gate: differentiability is the property the future
retrieval depends on.

Four rules the docstrings state and a caller has to respect:

- **Run under float64** (`jax.config.update("jax_enable_x64", True)`, arrays
  pre-cast). In float32 the differencing noise swamps the comparison.
- **The steps differ per variable by orders of magnitude** — `theta_s` is O(30)
  degrees and wants ~1e-3, `bb_p` is O(1e-3) and wants ~1e-9 — because no single
  step clears the tolerance for all of them. Too large a step drives `bb_p`
  negative, where the model returns `NaN`; that is a bad *step*, not a bad
  gradient, so it is reported as `inf` rather than silently compared.
- **A `steps` dict must name exactly the report's variables.** A missing key used
  to raise deep inside a closure; an extra one was worse — it reported `0.0`,
  i.e. perfect agreement, for a variable that was never perturbed. This is also
  why `cdom_gradient_report` is a separate function rather than an extension:
  growing the M4 dict would force `a_cdom`/`scale` steps on every existing caller
  of a gate that has nothing to do with CDOM.
- **Keep `theta_s` off the lookup-table nodes.** The packaged Ed table is
  piecewise-linear in θ_s ({doc}`../model/ed`), so at 0/30/60° the derivative is
  one-sided: autodiff takes one side, a central difference averages both, and they
  disagree at the 7th digit. The standing inelastic gate evaluates at **35°**; the
  elastic artifact evaluates O25 at **45°**, between its table nodes, for the same
  reason. L23 batches arrive *at* the anchors, so shift before calling.

A variable the model genuinely ignores reports `0.0` — both derivatives are
exactly zero, which is agreement, not infinite error. That is how O25 reads on
`B_p`.

## The design §6 acceptance gate

[`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §6 fixed six
lines **before implementation began**, and the prototype passes when all six
hold. Held out by scene, all processes on, φ_C = 0.02, committed weights.

% GATE-TABLE

| # | gate line | measured | bar | verdict |
| --- | --- | --- | --- | --- |
| 1 | total held-out rRMS vs X4, per zenith, **400–700 nm** | **0.343 / 0.341 / 0.340 %** (0/30/60°) | ≤ 0.5 % each | pass |
| 2 | Raman increment error, incl. 0° (550–700 nm and 490 nm) | **1.03 %** worst | ≤ 5 % | pass |
| 3 | fluorescence 685 nm error, per zenith | **0.10 %** worst | ≤ 5 % | pass |
| 4 | `inelastic=None` bit-identical to the elastic model | **True** | bitwise | pass |
| 5 | gradients, all six inputs incl. φ_C | **5.9×10⁻⁹** worst | ≤ 10⁻⁶ | pass |
| 6 | runtime vs the elastic hybrid, full batch | **1.59×** median | ≤ 2× | pass |

Every cell above was diffed against
[`reports/report_rt_inelastic_model.md`](gh:reports/report_rt_inelastic_model.md)
§3 and [`design/rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md)
§6.6 — the two places the gate is independently written down — and against the
generated [`design/validation/metrics_inelastic.md`](gh:design/validation/metrics_inelastic.md)
they both quote. All three agree.

Line 1 is the line the 400–700 nm band belongs to. Line 2's 0° is named in the
design *because* the analytic backbone fails it by −38.6 %: that line is what
δ_R had to earn ({doc}`../model/corrections`). Line 6 asserts the **median of
three trials**, because single ratios wander ±5 %.

Two things this gate is not. It is not a claim about geometries L23 does not
contain — three zeniths, nadir view. And its bars are per-line acceptance, not
uncertainty: the constants live in `robust.rt.validation` as
`INELASTIC_GATE_TOTAL_RRMS` = 0.5, `INELASTIC_GATE_DELTA` = 0.05 and
`INELASTIC_GATE_SPEED` = 2.0, defined once so a tightened bar cannot leave a
committed table contradicting the test. (Only
{data}`~robust.rt.validation.INELASTIC_GATE_TOTAL_RRMS` has an autodoc anchor;
the other two share its doc comment and so are not emitted — see {doc}`../api`.)

**The elastic §6 has no numeric bars at all.** The elastic design states the same
three axes as a protocol and deliberately sets no absolute target; its numeric
acceptance is §7's prototype definition — beat standard Gordon across λ and all
three zeniths, be competitive with the O25 refit on the same split, and pass the
gradient gate. All three hold: 0.30 % against Gordon's 7.21 %, 2.3× the O25
refit's 0.69 %, worst gradient disagreement 5.0×10⁻⁹
([`design/validation/metrics.md`](gh:design/validation/metrics.md); the ladder is
on {doc}`../model/baselines`).

## The measured results

Held-out scenes (1,992 samples), truth = `Rrs_X4`, $r_{rs}$ space, from
[`design/validation/metrics_inelastic.md`](gh:design/validation/metrics_inelastic.md).
**Over the gated band:**

% LADDER-BAND

| model, 400–700 nm | 0° | 30° | 60° |
| --- | --- | --- | --- |
| elastic-only (`inelastic=None`) | 18.71 | 15.80 | 16.34 |
| analytic inelastic (physics, no heads) | 4.29 | 1.94 | 2.18 |
| **corrected inelastic** (the default model) | **0.34** | **0.34** | **0.34** |

**And over the full grid the model computes on, reported alongside:**

% LADDER-FULL

| model, full 350–750 nm | 0° | 30° | 60° |
| --- | --- | --- | --- |
| elastic-only (`inelastic=None`) | 17.85 | 14.67 | 15.15 |
| analytic inelastic (physics, no heads) | 5.20 | 3.74 | 4.04 |
| corrected inelastic (the default model) | 2.61 | 2.27 | 2.28 |

Read the two tables together, because the pair says something neither says alone.
On its stated domain the corrected model is at the elastic accuracy standard
(0.34 % against the elastic hybrid's 0.30 %). Off that domain it is roughly eight
times worse (2.61 / 0.34 = 7.7 at 0°), and — per the decomposition above — the
whole of the difference is the *ten* grid points below 400 nm, 13.0 % at 350 nm
re-measured here, falling to the gate level by 400 nm; the ten points above
700 nm are already at 0.33–0.40 %. Note also that the *elastic-only* row is **lower** on the full grid than
in the band: the sub-400 nm region is where the inelastic terms matter least and
the excitation clamp hurts most, so the wider band flatters the model that
ignores them.

**Both ladders were reproduced independently in this environment** rather than
transcribed — the full release loaded, split with
{func}`~robust.rt.data.l23.make_splits`, scored through
{func}`~robust.rt.validation.group_rrms` — and every one of the eighteen cells
came back identical, as did the per-λ ladder's median 0.33 % and its worst
in-band 0.84 % at 450 nm.

### Per-process fidelity

Medians on held-out scenes, analytic backbone → corrected. These are the
quantities gate lines 2 and 3 are computed from:

% DELTAS

| metric [%] | 0° | 30° | 60° |
| --- | --- | --- | --- |
| Raman increment, 550–700 nm | −38.56 → **−0.14** | +1.23 → **−0.10** | −4.21 → **−0.21** |
| Raman increment, 490 nm | −3.60 → **+1.03** | +30.85 → **+0.82** | +32.55 → **+0.58** |
| fluorescence peak, 685 nm | +0.27 → **+0.08** | −5.20 → **+0.07** | −13.71 → **+0.10** |

The precision here is the generated artifact's; the report prints the *analytic*
column to one decimal (−38.6, +1.2, −4.2 · −3.6, +30.8, +32.5 · +0.3, −5.2,
−13.7) and the corrected column identically to the above. One of those roundings
is a truncation rather than a round-half-up — the artifact's +30.85 is printed
+30.8 — which is worth knowing only if you diff the two files mechanically, as
this page was. The physics these rows describe is
{doc}`../model/inelastic` and {doc}`../model/fluorescence`.

### Diagnostics — reported, not gated

- **Trophic state.** Deciles of $a_{ph}(440)$ over 0.0016–0.3527 m⁻¹ (held-out
  edges): the corrected fluorescence error is flat, max |err| **0.62 %**, where
  the analytic term drifts from −11 % to +11 %.
- **φ_C linearity.** At 0.5×/1×/2×/5× the reference yield the 685 nm error is
  +0.076 / +0.072 / +0.103 % per zenith, **identical across all four scales to
  < 10⁻⁴**. Linear by construction, as designed — and truth exists only at 1×.
- **`emission_shape='double'`.** −8.5 % at 685 nm and +9.8 % at 730 nm against
  `'single'`; scored against the single-shape truth, −23.6 % at every zenith.
  Unvalidatable and off everywhere in v1.
- **Speed.** 52.74 ms against the elastic hybrid's 33.52 ms, full 9,960 × 81
  batch, jitted CPU; trial ratios 1.60, 1.51, 1.57, 1.60, 1.59.
- **Gradients, all six.** `a` 2.5e-9, `bb_p` 3.6e-10, `B_p` 1.9e-9, `a_ph`
  5.9e-9, `phi_C` 1.4e-9, `theta_s` 3.0e-9 — at θ_s = 35°.

The **unseen-zenith holdout** is the diagnostic that matters most and is not a
gate line: retrained on 0°/30° only and scored at 60°, δ_R errs by −74 %, an
order of magnitude worse than the −4.2 % analytic backbone it corrects. It has
its own section on {doc}`limitations` and is discussed where the heads are
documented ({doc}`../model/corrections`).

## Running it yourself

The acceptance gate is seven tests in `robust/tests/test_inelastic_validation.py`,
one per design line, every metric routed through the protocol functions above:

```text
test_gate_1_total_rrms_vs_x4          test_gate_4_elastic_bit_identity
test_gate_2_raman_delta               test_gate_4_pre_change_pins
test_gate_3_fluorescence_delta        test_gate_5_gradients_all_inputs
                                      test_gate_6_speed_within_twice_elastic
```

Lines 1, 2, 3 and 6 need the full release, so they carry `needs_l23_inelastic`
and skip without `$OS_COLOR` ({doc}`data`). `robust/tests/test_validation.py`
(36 tests) checks the protocol functions themselves against hand-computed
references, including that a `steps` dict naming the wrong variables is refused.

:::{important}
**One of the seven fails on a machine other than the one that anchored it, and
this page will not pretend otherwise.** Re-run here:

```console
$ pytest robust/tests/test_inelastic_validation.py -q -ra
1 failed, 6 passed in 11.38s
FAILED robust/tests/test_inelastic_validation.py::test_gate_4_pre_change_pins
```

`test_gate_4_pre_change_pins` re-asserts a **strict SHA-256 pin** of the elastic
output bytes, and those pins were anchored on a different machine. Gate line 4's
substance — `inelastic=None` returning bit-identical arrays to the elastic model —
is `test_gate_4_elastic_bit_identity`, and it **passes**. So does the closeness
tier (`rtol` 5e-7, ≈4 ULP) that guards the same property portably. The measured
drift against the committed reference, re-measured for this page:

```text
Rrs: differ 2742/12150 (22.6%), max rel 3.326e-07, max ULP 3
rrs: differ 2862/12150 (23.6%), max rel 1.642e-07, max ULP 2
```

Three ULP of float32 rounding, not a changed route. The strict tier is
`skipif(CI)`, so it is a dev-machine gate only, and read the whole suite as
**green modulo the two machine-anchored strict-hash tiers** rather than as
unqualified green. {doc}`../installation` carries the same statement.
:::

To regenerate every committed number:

```console
$ python design/py/run_validation.py --inelastic
```

Committed weights only, nothing trained, ~2 minutes (mostly speed trials), exits
non-zero if any gate line fails. It needs `$OS_COLOR`. Its outputs land in
`design/validation/`: `metrics_inelastic.md`, two CSVs, and the ladder and delta
figures. Drop `--inelastic` for the elastic protocol and `metrics.md`.

For a table of your own, {func}`~robust.rt.validation.score_models` scores every
model on every split from *predictions* rather than callables — so every model is
evaluated once on the full batch and then sliced, which makes "identical data"
literally true instead of a claim about two code paths — and
{func}`~robust.rt.validation.markdown_table` formats the result.

At fixture scale, with no archive at all, the whole protocol runs. Measured under
`env -u OS_COLOR` on the 30 held-out fixture samples:

```text
rrms(corrected, 400-700 nm)   0.319 %        rrms(corrected, full grid)  2.547 %
group_rrms per zenith, band   0.326 / 0.313 / 0.318 %
median_increment_error 550-700 nm   -0.141 / -0.057 / -0.177 %
peak_ratio_error 685 nm             -0.264 / -0.263 / -0.240 %
phi_c_linearity                     identical to 4 decimals across 0.5x-5x
speed_ratio          2.28x median of 5 alternating trials, range 2.08-2.37
                                    (0.36 vs 0.16 ms, 30 samples)
inelastic_gradient_report worst     7.2e-08   (B_p; float64, theta_s = 35 deg)
```

Two of those deserve a word, because they are the two that do **not** simply
shrink. The speed ratio comes out **above the 2× gate** at this scale and wanders
by ±0.15 between trials: 30 samples cannot amortize the inelastic path's fixed
costs, and sub-millisecond timings are mostly dispatch overhead. That is a
statement about the batch, not about the model — the release-scale figure, on
9,960 samples, is 1.59×. The gradient residual is likewise ~20× the release
figure and is dominated by `B_p`; it is still two orders of magnitude inside the
10⁻⁶ tolerance, which is the only thing the gate asks.

:::{warning}
Those are the right *shape* and the wrong *value*: 50 scenes is not 3,320, ten
held-out scenes is not 664, and a 30-sample batch is far too small for the speed
ratio to mean what the 9,960-sample one does. **The citable numbers are the
report's**, above. Fixture-scale figures are for checking that a change did not
break something, not for quoting.
:::

## API

| What | Where |
| --- | --- |
| The metric | {func}`~robust.rt.validation.rrms`, {func}`~robust.rt.validation.rrms_per_wavelength`, {func}`~robust.rt.validation.group_rrms` |
| Binning | {func}`~robust.rt.validation.bp_bin_labels`, {func}`~robust.rt.validation.quantile_bin_labels` |
| Per-process metrics | {func}`~robust.rt.validation.median_increment_error`, {func}`~robust.rt.validation.peak_ratio_error`, {func}`~robust.rt.validation.phi_c_linearity` |
| Speed | {func}`~robust.rt.validation.throughput`, {func}`~robust.rt.validation.speed_ratio` |
| Gradients | {func}`~robust.rt.validation.gradient_report`, {func}`~robust.rt.validation.inelastic_gradient_report`, {func}`~robust.rt.validation.cdom_gradient_report`, {data}`~robust.rt.validation.FD_STEPS`, {data}`~robust.rt.validation.INELASTIC_FD_STEPS`, {data}`~robust.rt.validation.CDOM_FD_STEPS`, {data}`~robust.rt.validation.GRADIENT_TOL` |
| Gate constants | {data}`~robust.rt.validation.INELASTIC_GATE_BAND`, {data}`~robust.rt.validation.INELASTIC_GATE_TOTAL_RRMS` |
| Reporting | {func}`~robust.rt.validation.score_models`, {func}`~robust.rt.validation.markdown_table` |

Full signatures are on the {doc}`../api` page under *validation*. The data these
metrics are computed on is {doc}`data`; what the model may and may not claim on
the strength of them is {doc}`limitations`.
