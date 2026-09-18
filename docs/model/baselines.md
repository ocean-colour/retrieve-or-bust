# Baselines

{mod}`robust.rt.baselines` is **not part of the forward model**. It exists
because every accuracy claim this model makes is *relative* — "beats standard
Gordon on the held-out splits", "2.3× the modern benchmark" — and a claim like
that is only checkable if the thing being beaten is **in the repo,
differentiable, and computed on identical data** rather than quoted from a paper.

Two models live here: standard Gordon (1988), and the O25 bivariate form of
Pitarch et al. (2025) refit on L23. A third, PR05, is deliberately absent and
recorded as a gap rather than approximated.

*Sources for this page: the {mod}`robust.rt.baselines` module docstring and the
docstrings of {func}`~robust.rt.baselines.rrs_gordon`,
{func}`~robust.rt.baselines.Rrs_o25`,
{func}`~robust.rt.baselines.o25_coefficients` and
{func}`~robust.rt.baselines.fit_o25`;
[`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md) §§3,
4 and 5 (the ladder and its fairness caveats); and
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§4. Numbers labelled "measured" were measured in this environment when the page
was written, on the committed 50-scene fixture, and are **not** the report's
release-scale numbers — see the caveat below the table.*

## Standard Gordon (1988)

$$r_{rs} = g_1 u + g_2 u^2, \qquad u = \frac{b_b}{a + b_b}$$

with {data}`~robust.rt.baselines.G1_GORDON` = 0.0949 and `G2_GORDON` = 0.0794
(the second shares a doc comment with the first, so autodoc does not emit it
separately — see {doc}`../api`). **Fixed, not fitted**: these are the canonical
1988 values, and they are the same two numbers the synthesis figures used, which
is what makes the rRMS comparable with the pre-existing ladder.

{func}`~robust.rt.baselines.rrs_gordon` takes the **same arguments** as
{func}`robust.rt.hybrid.forward` and **ignores two of them**. That is not an
implementation shortcut; it is the model's defining limitation, made visible in
the signature:

```text
rrs_gordon(iops)  vs  rrs_gordon(iops, phase_params, geometry, wave)
   bitwise identical?  True     (measured)
```

Standard Gordon has **no phase-function input and no solar-zenith dependence**.
L23 says $R_{rs}$ falls by a median 5.1 % from 0° to 60°, so a zenith-blind model
*must* mis-fit at least one angle. That gap is precisely what the explicit-VSF
{doc}`ztt` backbone and the residual {doc}`emulator` are for. Keeping the
signatures interchangeable is what lets one loop score every model on one batch.

{func}`~robust.rt.baselines.Rrs_gordon` is the same thing through the air–water
interface. Scoring happens in $r_{rs}$ ({doc}`conventions`), so
`rrs_gordon` is the one the metrics call; the `Rrs` wrapper exists for
comparisons against observations and for plots.

## O25 form, refit on L23

$$R_{rs} = (G_{w0} + G_{w1}\omega_{bw})\,\omega_{bw}
         + (G_{p0} + G_{p1}\omega_{bp})\,\omega_{bp},
\qquad
\omega_{bw} = \frac{b_{bw}}{a + b_b},\;
\omega_{bp} = \frac{b_{bp}}{a + b_b}$$

Pitarch et al. (2025), *Remote Sens. Environ.* **329**, 114920, Eqs. 3–4. This is
a two-branch model — one branch per backscattering source — which is why it costs
this project nothing to evaluate: {attr}`IOPs.bb_w <robust.rt.types.IOPs.bb_w>`
and {attr}`IOPs.bb_p <robust.rt.types.IOPs.bb_p>` are already separate fields.

**This is the one comparison model defined in $R_{rs}$**, not $r_{rs}$, so the
pair is the reverse of Gordon's: {func}`~robust.rt.baselines.Rrs_o25` is the
primitive and {func}`~robust.rt.baselines.rrs_o25` converts for scoring. The
asymmetry that introduces is stated in the code and belongs on any table: O25's
coefficients were fitted against $R_{rs}$, so the interface conversion sits
*between* the fit and the score.

### Why it is a refit, and what that costs the comparison

:::{important}
**Every table must call this "O25 form, refit on L23".** The published $G$ lookup
tables are not printed in the paper — it shows them only as plots — and are not
in this repo; they live in the authors' code. Refitting is the alternative the
paper itself uses when it evaluates O25 on L23 (its Fig. 3). So the numbers here
are a **strong benchmark that has seen our training data**, not a statement about
the published model
([`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md)
§3 and §5, may-not-claim item 2).

Two consequences, both deliberate. The refit is **fitted on the `scene_train`
split only** — {func}`~robust.rt.baselines.fit_o25` *requires* a `train` mask,
because fitting a comparison model on the test split would flatter the model it
is being compared against, and that is the one direction of bias nobody thinks to
check. And it is fitted with **our relatively weighted objective**, which is
worth ~4× to O25 versus the paper's unweighted least squares (0.68–0.73 % rRMS
against 2.5–2.7 %). Fitting a rival with the wrong objective would have made our
own hybrid look four times better than a fair comparison allows, so the fair
choice is the default and the paper's is behind `weighted=False`.
:::

{data}`~robust.rt.baselines.O25_L23_REFIT` embeds the result, so O25 is usable and
testable without `$OS_COLOR` — the same reasoning that embeds `BB_W_L23` in
{doc}`conventions` — and `test_baselines.py` refits and checks the numbers
reproduce, so the table cannot drift from the code that made it:

```text
theta_s     Gw0         Gw1         Gp0         Gp1
   0.0   0.05866762  0.02490574  0.04018414  0.12313511
  30.0   0.05721442  0.02932806  0.04007408  0.15273771
  60.0   0.05249313  0.03749838  0.03936996  0.20619877
```

`fit_o25` is **closed form, not an optimisation** — the model is linear in its
coefficients, so it is one `lstsq` per zenith. No seed, no learning rate, no
stopping rule: deterministic by construction rather than by care.

### What the refit can and cannot index

{func}`~robust.rt.baselines.Rrs_o25` **uses** solar zenith, unlike Gordon, and
refuses to guess it:

```text
>>> Rrs_o25(iops, phase_params, None, wave)
ValueError: Rrs_o25 needs a geometry: its coefficients are indexed by solar
zenith. (rrs_gordon accepts geometry=None because Gordon has no zenith term.)
```

But it indexes the solar zenith **alone**. The published O25 indexes all three
geometry angles; L23 is nadir-only, so a refit here could never populate
`theta_v` or `dphi`. Phase-function shape is not adjustable either — O25's
coefficients were calibrated on a set with *prescribed* Fournier-Forand phase
functions, so the shape is baked into the fitted numbers. Wavelength enters only
through the IOPs, by construction; that is a feature of O25, not a
simplification made here.

{func}`~robust.rt.baselines.o25_coefficients` interpolates linearly in the angle
and is **flat outside** the tabulated range — `jnp.interp` clamps rather than
extrapolating a ramp, which is the conservative choice, since a linear
extrapolation of $G_{p1}$ past 60° grows without bound while a held value at
least stays inside the fitted family. Measured:

```text
o25_coefficients(45.0) = (0.05485377, 0.03341322, 0.03972202, 0.17946824)
o25_coefficients(75.0) = the 60 deg row, unchanged      (clamped: True)
```

:::{warning}
**O25 is non-differentiable at its own coefficient-table nodes**, and L23's three
solar zeniths *are* those nodes. A piecewise-linear lookup has a kink there:
`jax.grad` takes one one-sided slope while a central difference averages both,
and the two disagree by O(1) — measured 69 % at 30°. Check the gradient at an
intermediate angle instead; at 45° it agrees to 2.7 × 10⁻¹⁰. This is inherent to
a lookup-table model, not a defect in this port, but a finite-difference check on
L23 geometry lands on a node every single time.
:::

{data}`~robust.rt.baselines.O25_RRS_CEILING` = 0.06 is the paper's stated validity
ceiling for the quadratic. Its doc comment records that the full L23 release
reaches 0.0248; the committed fixture reaches 0.0164 (measured). Nothing here
extrapolates in brightness.

## PR05 is deliberately absent

The design documents pair "PR05/O25" as though both were rungs of the ladder.
The honest version is that **PR05 is not implemented**. Its coefficients are a 4-D
`(theta_s, theta_v, dphi, gamma_b)` lookup table that the paper does not print and
that is not in this repo — it exists behind a 2005 institutional URL or inside
POLYMER. And because L23 is nadir-only, a refit here could never populate the two
sensor-geometry axes, so the result would be a different object wearing the same
name. Both reports say so where the ladder is presented, and the module docstring
records it as a gap rather than approximating it.

## Their role in the validation story

The baselines are the rungs of the ladder both reports lead with. From
[`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md) §4,
rRMS (%) in $r_{rs}$ space over the full 9,960-sample batch against the
**elastic** truth:

| model | train | held-out scenes | held-out @ 60° |
| --- | --- | --- | --- |
| standard Gordon (1988) | 7.21 | 7.21 | 9.01 |
| ZTT backbone alone | 5.95 | 5.93 | 8.11 |
| **O25 form, refit on L23** (12 fitted numbers) | 0.70 | **0.69** | 0.71 |
| hybrid, linear emulator (8 parameters) | 2.57 | 2.54 | 2.48 |
| **hybrid, MLP emulator** (417 parameters) | **0.30** | **0.30** | **0.32** |

Three readings the report insists on, and this page repeats because they are what
the baselines are *for*. The unfitted ZTT backbone already beats Gordon overall —
explicit-VSF physics paying off before any learning. **The gap that matters is to
O25, not to Gordon**: the margin over the modern benchmark is **2.3×**, and
"24×" — the margin over a 1988 model — is the report's first may-not-claim item.
And at the **unseen 60° zenith the benchmark wins outright**: refit O25 scores
4.63 % deterministically while the MLP hybrid spans 4.74–12.24 % across five
seeds (median 7.75 %). That is reported rather than gated, and it is the elastic
report's §5 item 3.

For the inelastic model the baselines are scored the same way, and
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§4's ladder is against the all-processes-on truth: elastic-only 16–19 %,
analytic inelastic 2–4 % (the report's phrase; its own table reads
4.29 / 1.94 / 2.18 at 0/30/60°), corrected inelastic **0.34 %** at every zenith.

Speed, from the elastic report §4 (jitted, CPU, full batch): Gordon 0.28 ms,
O25 0.46 ms, ZTT backbone 3.76 ms, hybrid ~17 ms. The baselines are also the
evidence that the hybrid does not collapse the analytic advantage of *not*
calling an RT solver.

:::{warning}
**Do not compare that 17 ms with the 33.5 ms on the {doc}`../using/validation`
page.** They are the same quantity — the elastic hybrid over the full
9,960 × 81 batch, jitted, on CPU — measured by the two reports twelve days
apart, and they disagree by a factor of two. Neither is wrong; absolute
wall-clock here is a property of the machine, the jax build and what else was
running, and no run reconciling the two exists. What each report *does* control
for, by measuring both halves in the same session, is the **ratio**: 4.5–6× the
backbone (elastic report §4) and 1.59× elastic → inelastic (inelastic report
§4). Quote the ratios; treat the millisecond figures as order-of-magnitude.
:::

### Re-measured here, and the trap it exposes

Scored in this environment on the committed 50-scene fixture (150 samples), in
$r_{rs}$ space over 400–700 nm, using {func}`robust.rt.validation.score_models`:

```text
### vs X1 (elastic truth)              all      0     30     60
standard Gordon (1988)                5.83   4.48   4.65   7.76
O25 form, refit on L23                0.57   0.57   0.57   0.58
ZTT backbone alone (mode='ztt')       5.43   4.04   3.81   7.58
elastic hybrid (inelastic=None)       0.24   0.23   0.25   0.24
corrected inelastic (default)        25.40  27.82  23.30  24.89

### vs X4 (all processes on)           all      0     30     60
standard Gordon (1988)               14.80  17.97  13.98  11.80
O25 form, refit on L23               16.78  18.72  15.45  15.99
ZTT backbone alone (mode='ztt')      18.21  16.66  16.78  20.87
elastic hybrid (inelastic=None)      16.67  18.60  15.33  15.89
corrected inelastic (default)         0.32   0.32   0.31   0.32
```

:::{warning}
**These are not the reports' numbers, and must not be quoted as them.** The
fixture is 50 of L23's 3,320 water bodies with **no train/held-out split** — the
O25 coefficients were fitted on the full release's training split, so its row
here is neither a train nor a held-out score — and the band is 400–700 nm rather
than the elastic report's full grid. The ordering and the orders of magnitude
reproduce; the digits are a different quantity. The citable numbers are the
tables above.
:::

The table does earn its place, though, because it makes one trap unmissable:
**both baselines are elastic models, so the truth channel decides whether the
comparison means anything.** Against X4, Gordon (14.8 %) apparently "beats" O25
(16.8 %) — but both are simply missing the inelastic physics, and the ranking
there says nothing about how well either models elastic scattering. Symmetrically,
scoring the *corrected inelastic* model against elastic truth X1 gives 25 %,
which is not a failure of the model but a failure to pick the right reference.
Score an elastic model against X1 and an all-processes model against X4. The
protocol that formalizes this is {doc}`../using/validation`; the truth channels
themselves are {doc}`../using/data`.

## API

| What | Where |
| --- | --- |
| Gordon | {func}`~robust.rt.baselines.rrs_gordon`, {func}`~robust.rt.baselines.Rrs_gordon`, {data}`~robust.rt.baselines.G1_GORDON` |
| O25 | {func}`~robust.rt.baselines.Rrs_o25`, {func}`~robust.rt.baselines.rrs_o25` |
| Its coefficients | {data}`~robust.rt.baselines.O25_L23_REFIT`, {func}`~robust.rt.baselines.o25_coefficients`, {data}`~robust.rt.baselines.O25_RRS_CEILING` |
| Refitting it | {func}`~robust.rt.baselines.fit_o25` |
| Scoring them together | {func}`robust.rt.validation.score_models`, {func}`robust.rt.validation.rrms` |

Full signatures are on the {doc}`../api` page under *baselines*. The model these
are the yardstick for is {doc}`forward`; the metric is {doc}`../using/validation`.
