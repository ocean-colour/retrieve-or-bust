# The correction heads

The two analytic inelastic terms are *measurably* wrong in ways no constant in
them can fix: the fixed-mean-cosine two-flow Raman assembly errs by −38.6 % in
increment at overhead sun, and the fluorescence amplitude drifts from
model/truth 1.00 at 0° to 0.86 at 60°
([`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§4). {mod}`robust.rt.inelastic_corr` closes those gaps with two small bounded
learned heads — **129 parameters each** — that rescale the analytic terms rather
than replacing them.

This is the same architecture philosophy as the elastic {doc}`emulator`, applied
to a smaller target: analytic physics carries the spectral shape, and the network
pays only for the residual. The machinery is literally reused —
`emulator._network` and `emulator._delta` are duck-typed on
`config.hidden`/`config.delta_max` and called directly — rather than re-derived.

*Sources for this page: the {mod}`robust.rt.inelastic_corr` module docstring and
the docstrings of {class}`~robust.rt.inelastic_corr.HeadConfig`,
{class}`~robust.rt.inelastic_corr.CorrectionHead`,
{class}`~robust.rt.inelastic_corr.CorrectionHeads`,
{func}`~robust.rt.inelastic_corr.load_default`,
{func}`~robust.rt.inelastic_corr.corrected_raman_factor` and
{func}`~robust.rt.inelastic_corr.corrected_fluorescence`;
[`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §4.5;
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§§2, 4 and 5; and
[`design/rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md)
§§5.2–5.4 (the implementation record — the source for the training numbers).
Numbers labelled "measured" were measured in this environment when the page was
written, against the packaged weights on the committed 50-scene fixture.*

## Two heads, two composition forms

The forms are not interchangeable, and each is written **once** in the codebase
so a wiring, a gate test and a training objective can only ever score one
expression.

**δ_R corrects the Raman *increment*, never the baseline**
({func}`~robust.rt.inelastic_corr.corrected_raman_factor`):

$$f_R = 1 + (f_{\rm phys} - 1)\,(1 + \delta_R)$$

The structural property that buys is worth stating: $f_R \to 1$ wherever the
Raman increment vanishes, **no matter what the network outputs**. A head cannot
invent Raman where the physics says there is none, and it cannot push $f_R$
below 1 by more than its bound.

**δ_F rescales the φ_C-linear kernel**
({func}`~robust.rt.inelastic_corr.corrected_fluorescence`):

$$R_{rs}^{\rm fl} = \varphi_C\, K_{\rm fl}\,(1 + \delta_F)$$

with the multiplication by φ_C applied by the caller, afterwards — float
multiplication is not associative, so the order is part of the contract. **δ_F's
features exclude φ_C**, which is what keeps the design's φ_C-linearity promise:
$\partial R_{rs}/\partial\varphi_C = K_{\rm fl}(1 + \delta_F)$, a clean
multiplicative handle the head cannot bend. See {doc}`fluorescence`.

Both are δ ≡ 0 at initialization (below), so **an untrained head *is* the
analytic model** — training starts from the physics rather than from noise, and
a test pins that identity.

## `HeadConfig`

{class}`~robust.rt.inelastic_corr.HeadConfig` is frozen and hashable so it can be
a static field of a registered pytree — the same contract as
{class}`robust.rt.emulator.Emulator`'s config, which is what makes the duck-typing
legal. Read off the packaged weights:

```text
raman: kind='raman' hidden=(16,) delta_max=1.0 lr=0.003 steps=3000 seed=23
fl   : kind='fl'    hidden=(16,) delta_max=0.5 lr=0.003 steps=3000 seed=23
       129 parameters each
```

| Field | Meaning |
| --- | --- |
| {attr}`~robust.rt.inelastic_corr.HeadConfig.kind` | `'raman'`, `'fl'` or `'cdom'` — selects the feature builder *and* the composition form. Stored with the weights |
| {attr}`~robust.rt.inelastic_corr.HeadConfig.hidden` | hidden widths. `(16,)` is 129 parameters, the **low end** of the design's O(10²–10³) budget; `()` is the linear baseline |
| {attr}`~robust.rt.inelastic_corr.HeadConfig.delta_max` | the hard tanh bound on \|δ\| (below) |
| `learning_rate`, `steps`, `seed` | training knobs, stored so the weight-file format did not have to change when training landed |

`__post_init__` rejects an unknown kind, a non-positive width, bound or rate,
and a step count below 1 — with a reason, at construction time.

### The per-head bounds are sized against measured errors

`delta_max` differs by head because the errors do, and the implementation record
prices both (§5.2):

- **δ_R: 1.0.** Closing the −39 % increment gap at 0° needs δ_R to reach
  **+0.64** (1/0.61 − 1), so a bound of 1.0 leaves room.
- **δ_F: 0.5.** The 60° drift needs about **+0.18**, so the elastic default of
  0.5 has ample slack.

Note the asymmetry between the class default and the constructor default:
`HeadConfig('raman')` alone gives `delta_max=0.5`, while
{func}`~robust.rt.inelastic_corr.init_head` supplies the per-kind value
(`init_head('raman').config.delta_max` is 1.0, `init_head('fl')`'s is 0.5).
The packaged weights carry their own, so a loaded head is never guessing.

## The tanh bound

The bounding is one line, in `emulator._delta`, shared with the elastic
emulator:

$$\delta = \delta_{\max}\tanh(\text{raw network output})$$

so $|\delta| < \delta_{\max}$ **strictly**, for any input, any parameters, any
extrapolation. This is the mechanism that makes the design's promise — that
extrapolation degrades toward the analytic physics rather than toward network
free-wheeling — a construction rather than a hope. Combined with the increment
form above, the worst a runaway δ_R can do to a Raman factor is double or cancel
its increment; it cannot produce an arbitrary spectrum.

The output layer is **zero-initialised** (`kernel_init=zeros`,
`bias_init=zeros`), which is why a fresh head is exactly δ ≡ 0. Measured:

```text
init_head('raman').delta(...) over the fixture:  max |delta| = 0.0
```

Measured on the fixture with the **packaged trained** weights:

```text
delta_raman: min -0.3403  max +0.9032   max|.| 0.9032  of bound 1.0  ->  90.3 %
delta_fl   : min -0.1550  max +0.2936   max|.| 0.2936  of bound 0.5  ->  58.7 %
```

δ_R runs at 90 % of its bound at the extreme. The implementation record measured
the same thing on the training set (`max |δ_R| 0.905`, §5.3) and flagged it
explicitly as a saturation canary — *watch it if the loss band ever widens*. A
test asserts `|δ| < delta_max` over the whole release, which doubles as that
canary.

## The features, and the one that is deliberately absent

Standardized per head at train time; the four IOP-like columns enter as log10
because they span decades (the elastic emulator's `log10(u)` precedent), floored
at 10⁻¹⁰ so a caller's zero IOP yields a finite feature rather than −inf.

| Head | {data}`~robust.rt.inelastic_corr.RAMAN_FEATURES` / {data}`~robust.rt.inelastic_corr.FL_FEATURES` |
| --- | --- |
| δ_R | `log10_a_em`, `log10_bb_em`, `log10_a_ex`, `log10_bb_ex`, `cos_theta_s`, `wave` |
| δ_F | `log10_a_ph440`, `log10_a_em`, `log10_bb_em`, `log10_a_490`, `cos_theta_s`, `wave` |

Two details that are load-bearing rather than incidental. δ_R's excitation-grid
values come from {func}`~robust.rt.conventions.raman_excitation` and
{func}`~robust.rt.conventions.interp_spectrum` **by name** — the same calls
{func}`robust.rt.inelastic.raman_factor` makes — so the head sees exactly the
inputs the term it corrects computed with. And **δ_F has no φ_C column**, by
design (§4.4): a head that could see the yield could break the kernel's
linearity in it.

{func}`~robust.rt.inelastic_corr.load_head` stores the feature names in the
weight file and **refuses** a file whose list differs from the code's. That is a
refusal rather than a warning because the failure mode it guards is the nasty
one: weights against the wrong feature vector *run*, and return plausible
nonsense.

A third head, **δ_C** for CDOM fluorescence, is *defined* here on the δ_F
pattern ({data}`~robust.rt.inelastic_corr.CDOM_FEATURES`,
{func}`~robust.rt.inelastic_corr.corrected_cdom`) but ships **untrained and
unwired**: no CDOM-fluorescence truth exists, so
{func}`~robust.rt.inelastic_corr.train_cdom_corr` raises,
{func}`~robust.rt.inelastic_corr.load_default` looks for no CDOM weights
(`heads.cdom` is `None`, measured), and the shipped CDOM term is
`scale · K_cdom` — bit-for-bit what a zero-init head would produce. Its
`delta_max` is 0.5, an **arbitrary placeholder** rather than a measured-error
bound like the other two. The slot exists so wiring it later is an API no-op.
{mod}`robust.rt.cdom_fl` is off by default and unvalidated; the term, the head's
status and what unblocks it are {doc}`cdom_fluorescence`.

## The packaged weights

```text
robust/rt/files/raman_corr_l23.npz   4330 bytes   (~4.2 kB)
robust/rt/files/fl_corr_l23.npz      4366 bytes   (~4.3 kB)
```

Two files, not one, because the heads train, version and regenerate
independently. `setup.py`'s `package_data` ships them, so
{func}`robust.rt.hybrid.forward` works out of the box after
`pip install`. Each file carries everything needed to reproduce a prediction:
parameters, the per-feature standardisation, the feature names, and the config.

They were trained on the **L23 scenario differences** — X2/X1 for Raman,
X4 − X2 for fluorescence — on the elastic effort's training split, with
full-batch Adam and fixed seeds, ~60 s per head on CPU. Both losses are
**relatively weighted**, the same lesson this model's own metric is built on. From
the implementation record (§5.3):

| | training target | train fit | \|δ\|rms | max \|δ\| |
| --- | --- | --- | --- | --- |
| δ_R | relative increment error over λ ≥ 400 nm | 24.8 % → **1.69 %** | 30.6 % | 0.905 (bound 1.0) |
| δ_F | residual over 655–715 nm, normalized by each scene's own 685 nm truth | 5.6 % → **0.77 %** | 7.1 % | 0.34 (bound 0.5) |

δ_F's normalization choice is worth a sentence: normalizing by each scene's *own*
peak means trophic states weigh equally and the near-zero tails of the emission
band cannot blow up a pointwise relative error.

What the heads bought, held out, from
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§4:

| median error [%] | 0° | 30° | 60° |
| --- | --- | --- | --- |
| Raman increment 550–700 nm, analytic → corrected | −38.6 → **−0.14** | +1.2 → **−0.10** | −4.2 → **−0.21** |
| Raman increment 490 nm | −3.6 → **+1.03** | +30.8 → **+0.82** | +32.5 → **+0.58** |
| Fluorescence peak 685 nm | +0.3 → **+0.08** | −5.2 → **+0.07** | −13.7 → **+0.10** |

Total held-out rRMS goes from the analytic 2–4 % — 4.29 / 1.94 / 2.18 at
0/30/60° — to **0.34 % at every zenith**
over 400–700 nm (report §4) — the elastic-era accuracy standard, now holding
against the all-processes-on ocean. The ≤ 5 % per-process gates are beaten by
roughly 25× (implementation record §5.3).

## `corrections=None` versus `corrections=False`

These are **different models**, not a style choice, and the distinction is real
in the code — `hybrid._resolve_corrections` branches on both:

| Value | Behaviour |
| --- | --- |
| `None` *(default)* | the packaged trained heads, via {func}`~robust.rt.inelastic_corr.load_default`. **The corrected model is the model.** If a weight file is missing it degrades to analytic-only behind a single {exc}`~robust.rt.inelastic_corr.MissingCorrectionWarning` |
| `False` | analytic-only, **explicit and silent**. This is how the M2 characterization tests pin the analytic terms bit-for-bit, and how a study comparing backbones asks for one |
| a {class}`~robust.rt.inelastic_corr.CorrectionHeads` | used as given — training, ablations. A field left `None` leaves that term analytic *by omission of the correction arithmetic*, not by multiplying with a computed zero |

Measured through {func}`robust.rt.hybrid.forward` on one fixture scene, both
inelastic processes on:

```text
corrections=None  vs corrections=False        bitwise equal?  False
   max relative difference, full grid                        11.56 %
   max relative difference, 400-700 nm                        9.13 %
   at 685 nm:  corrected 1.225158e-04  analytic 1.136038e-04  +7.84 %

corrections=None  vs corrections=load_default()   bitwise equal?  True
corrections=False vs corrections=CorrectionHeads() (all None)     True
```

So `None` and `False` differ by up to ~12 % on this scene: the distinction is
load-bearing, and a page that treated them as synonyms would be wrong by the
size of the corrections themselves.

Three further behaviours worth knowing:

- **The fallback is a warning, not an error**, and deliberately so: the analytic
  backbone is a legitimate model — it *is* the M2 acceptance gate — but silence
  would hide missing physics from a caller who expected trained heads. The
  warning names the analytic terms' known errors and tells you how to silence it
  properly (`corrections=False`). It has its own category, so a pipeline that
  must not run analytic-only can promote it to an error with
  `warnings.simplefilter("error", MissingCorrectionWarning)`.
- **This differs from the elastic emulator on purpose.**
  {func}`robust.rt.emulator.load_default` *raises* on missing weights, because
  the hybrid without its emulator is a different model. The heads degrade,
  because without them the term is still the physics.
- **`corrections` is resolved only when an inelastic process is actually on.**
  Measured: `forward(...)` with `inelastic=None` raises no warning at all, and
  the elastic path never imports the ML stack on the heads' account.

## The honest framing: these are interpolators, and they carry no domain guard

:::{warning}
**The heads interpolate over three solar-zenith anchors. Nothing about them
extrapolates, and unlike the elastic emulator they have no domain guard.**

Retrained on 0°/30° only and scored at 60°, δ_R errs by **−74 %** — an order of
magnitude *worse* than the −4.2 % analytic backbone it is supposed to be
correcting. δ_F degrades more gently, to −9.2 %, still past the 5 % gate. Both
figures are
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§4 and §5 (may-not-claim item 1), and the implementation record §5.3 records the
same diagnostic with the instruction that these heads *must not be trusted at
unseen geometries*.

The reason is mechanical: `cos_theta_s` is one of six features, L23 supplies it
at exactly three values (1.0, 0.866, 0.5), and a tanh unit fitted on two of them
is unconstrained at the third.

**The shipped weights are trained on all three zeniths, so this failure is not
in the delivered model.** But there is nothing in the code that will tell you
when you have left the range. Verified by attribute:

```text
head('raman')  has 'domain'? False    has 'out_of_domain'? False
head('fl')     has 'domain'? False    has 'out_of_domain'? False
Emulator       has 'domain'? True     has 'out_of_domain'? True   (domain set)
```

So {func}`robust.rt.hybrid.forward`'s `check_domain` / `on_out_of_domain`
switches ({doc}`forward`) guard the **elastic emulator only**. An
out-of-range θ_s reaches the heads silently. The report's recommended priority 2
is that the heads gain the emulator's guard mechanism until denser-zenith
HydroLight runs exist (report §7).
:::

Two more limits belong with that one, both from report §5:

- **φ_C beyond 0.02** is linear by construction and untested against truth
  ({doc}`fluorescence` carries the detail).
- **θ_s derivatives at the sky anchors** are one-sided, because the packaged-$E_d$
  zenith interpolation is piecewise linear and 0°/30°/60° *are* its nodes.
  Differentiate off the anchors; the gradient gate is certified at 35°.

The full unbowdlerized list is {doc}`../using/limitations`.

## API

| What | Where |
| --- | --- |
| Configuration | {class}`~robust.rt.inelastic_corr.HeadConfig`, {attr}`~robust.rt.inelastic_corr.HeadConfig.delta_max` |
| One head | {class}`~robust.rt.inelastic_corr.CorrectionHead`, {meth}`~robust.rt.inelastic_corr.CorrectionHead.delta` |
| The pair `forward` carries | {class}`~robust.rt.inelastic_corr.CorrectionHeads` |
| Features | {data}`~robust.rt.inelastic_corr.RAMAN_FEATURES`, {data}`~robust.rt.inelastic_corr.FL_FEATURES`, {func}`~robust.rt.inelastic_corr.features_raman`, {func}`~robust.rt.inelastic_corr.features_fl` |
| The composition forms | {func}`~robust.rt.inelastic_corr.corrected_raman_factor`, {func}`~robust.rt.inelastic_corr.corrected_fluorescence` |
| Weights | {func}`~robust.rt.inelastic_corr.load_default`, {func}`~robust.rt.inelastic_corr.load_head`, {func}`~robust.rt.inelastic_corr.save_head`, {data}`~robust.rt.inelastic_corr.DEFAULT_RAMAN_WEIGHTS` |
| A fresh (analytic) head | {func}`~robust.rt.inelastic_corr.init_head` |
| The fallback warning | {exc}`~robust.rt.inelastic_corr.MissingCorrectionWarning` |
| The reserved CDOM head | {data}`~robust.rt.inelastic_corr.CDOM_FEATURES`, {func}`~robust.rt.inelastic_corr.corrected_cdom`, {func}`~robust.rt.inelastic_corr.train_cdom_corr` |

Full signatures are on the {doc}`../api` page under *inelastic_corr*. The
analytic terms these heads rescale are {doc}`inelastic` and
{doc}`fluorescence`; the elastic head they share machinery with is
{doc}`emulator`; the models they are scored against are {doc}`baselines`.
