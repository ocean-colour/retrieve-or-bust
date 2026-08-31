# The learned residual

{mod}`robust.rt.emulator` is the learned half of the hybrid: a small Flax MLP
that predicts what the analytic backbone misses — the multiple-scattering and
phase-function effects Equation (12) does not carry. Because it learns only a
*residual*, it can stay small and its extrapolation is bounded, which is the
entire argument for a hybrid over a wholly learned model
([`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md) §4.4).

*Sources for this page: the {mod}`robust.rt.emulator` module docstring (whose
four numbered decisions are the structure of this chapter) and the docstrings
of {class}`~robust.rt.emulator.EmulatorConfig`,
{class}`~robust.rt.emulator.Emulator` and each constant;
[`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md) §4.4;
[`reports/report_rt_elastic_model.md`](gh:reports/report_rt_elastic_model.md)
§2, §4 and §5. Numbers marked "measured" were re-measured in this environment,
against the packaged weights, when the page was written.*

## What it computes

The network emits a **dimensionless relative** correction $\delta(\lambda)$,
and the additive correction is that scaled by the backbone:

$$
\Delta r_{rs}(\lambda) \;=\; \delta(\lambda)\;r_{rs}^{\mathrm{ZTT}}(\lambda),
\qquad\text{so}\qquad
r_{rs} \;=\; r_{rs}^{\mathrm{ZTT}}\,\bigl(1 + \delta\bigr).
$$

{meth}`Emulator.relative_delta <robust.rt.emulator.Emulator.relative_delta>`
returns $\delta$; {meth}`Emulator.delta_rrs <robust.rt.emulator.Emulator.delta_rrs>`
returns $\Delta r_{rs}$ in sr⁻¹ and accepts a precomputed `rrs_ztt=` because
the hybrid already has it in hand and recomputing ZTT costs ~12× everything
else here.

**Why relative and not absolute.** $r_{rs}$ runs from ~2.5e-2 in the blue to
~6e-6 in the red, so a network emitting absolute sr⁻¹ would have to span four
decades with one set of weights, and the relative loss would weight its red-end
errors ~4000× more heavily than its blue-end ones. The relative residual is an
$O(1)$ quantity instead: measured over all 9 960 L23 samples at M2, mean
**+2.20 %**, sd **5.52 %**. That is also directly the number the design asks to
be kept small.

## The features are the complete state, not a guess

$r_{rs}^{\mathrm{ZTT}}$ is **scale-invariant**: multiplying
$(a, b_{bw}, b_{bp})$ by any $k$ leaves it unchanged to machine precision
(verified to 8.8e-15 at $k = 10$; a test pins it). So the backbone sees its
inputs only through ratios, and two numbers span them —
$u = b_b/(a+b_b)$ and $\eta_{bb} = b_{bw}/b_b$ invert back to
$(a : b_{bw} : b_{bp})$ exactly. With $B_p$, the geometry and $\lambda$, that is
the *whole* input state, so the emulator is not starved of anything the
backbone knew. Absolute magnitudes are deliberately absent: radiative transfer
in a homogeneous half-space has no absolute length scale either, so a feature
carrying one could only fit noise.

{data}`~robust.rt.emulator.FEATURES`, in order:

```text
('log10_u', 'eta_bb', 'B_p', 'wave_nm', 'cos_theta_s', 'cos_theta_v', 'cos_dphi')
```

$\lambda$ and $\theta_s$ are first-class because the residual's structure lives
there: M2 measured a monotone offset in solar zenith (≈ −2 %, +2 %, +8 % at
0°/30°/60°) and a spectral hump near 550 nm, and a polynomial in $\lambda$
alone explains **83.9 %** of the relative-residual variance at degree 1.

:::{warning}
`cos_theta_v` and `cos_dphi` are **constant in L23** (nadir view, zero
azimuth). Their weights are unidentified and they standardise to exactly zero.
They are carried so that the feature vector is the final one and off-nadir runs
need no interface change — but **nothing in any result on this site is evidence
about view geometry**. That is item 5 of the report's "may not claim" list.
:::

{func}`~robust.rt.emulator.features` builds the raw (un-standardised) vectors
and takes the same arguments as {func}`~robust.rt.hybrid.forward`, so the
emulator can be evaluated wherever the forward model can. Standardisation
statistics are computed on the **training split only** and stored *in* the
trained {class}`~robust.rt.emulator.Emulator`, because a mismatch between
fit-time and call-time statistics is silent — it produces plausible numbers
that are simply wrong.

## The network

Measured, by loading the packaged weights:

```text
config: EmulatorConfig(hidden=(16, 16), delta_max=0.5, penalty=0.02,
                       learning_rate=0.003, steps=3000, seed=23, eval_every=100)
parameter count: 417
weights file:    emulator_l23.npz, 6678 bytes
```

A 7 → 16 → 16 → 1 `tanh` MLP, **pointwise in $\lambda$**: one shared network
maps the features *at* a wavelength to $\delta$ at that wavelength, mirroring
the backbone (whose $r_{rs}^{\mathrm{ZTT}}(\lambda)$ depends only on the IOPs
at $\lambda$) and leaving the emulator defined on any wavelength grid.

Four structural choices are worth knowing before you trust or retrain it:

`tanh`, not `relu`
: The acceptance gate is a finite-difference gradient check, and a `relu` kink
  makes central differences straddling it wrong by $O(h)$ — a real hazard, not
  a hypothetical. The target is smooth too.

A bounded correction, by construction
: $\delta = \texttt{delta\_max}\cdot\tanh(\cdot)$, so
  $\lvert\delta\rvert < 0.5$ no matter how far out of distribution the inputs
  go. The hybrid can never be driven negative or wild by its learned half. The
  bound is ~9× the measured residual sd, so it never binds on L23; the soft
  `penalty` term keeps $\delta$ small *in* distribution, the hard cap keeps it
  sane outside.

The correction starts at exactly zero
: The output layer is zero-initialised, so an **untrained hybrid *is* the
  backbone** and every reported improvement is an improvement over ZTT rather
  than an artefact of initialisation.

Full-batch and unshuffled
: 3000 Adam steps over ~0.6 M rows of seven features, so a fit is reproducible
  from `seed` alone with no data-order dependence.

Measured on the first fixture scene of the {doc}`../quickstart` (0° sun), the
packaged emulator's correction is a few percent and negative over the whole
350–750 nm grid:

```text
delta: min -0.0705   max -0.0179   mean -0.0454
```

### The linear baseline is a first-class citizen

{data}`~robust.rt.emulator.LINEAR_CONFIG` is the same code with `hidden=()` —
an affine map in the standardised features, trained by the same loop on the
same features with the same loss. It exists because 83.9 % of the residual
variance is linear in $\lambda$ alone, and that is the number an MLP has to
beat to justify its nonlinearity. On the full L23 batch it takes the backbone's
5.95 % to **2.57 %** (train) / **2.54 %** (held-out scenes), against the default
MLP's **0.30 %** — so the nonlinearity earns ~8×, not the ~20× a baseline-free
comparison would suggest
([`reports/report_rt_elastic_model.md`](gh:reports/report_rt_elastic_model.md)
§4).

## The domain guard

The emulator carries its own training range —
{attr}`Emulator.domain <robust.rt.emulator.Emulator.domain>`, a
`(2, n_feature)` array of per-feature min/max over the training split. That is
what makes the extrapolation caveat operational rather than a remark in a log.
Read off the packaged weights:

| feature | trained range |
| --- | --- |
| `log10_u` | [−3.96572, −0.428425] |
| `eta_bb` | [0.0175722, 0.966027] |
| `B_p` | [0.0102644, 0.0180032] |
| `wave_nm` | [350, 750] |
| `cos_theta_s` | [0.5, 1] |
| `cos_theta_v` | [1, 1] |
| `cos_dphi` | [1, 1] |

{meth}`Emulator.out_of_domain <robust.rt.emulator.Emulator.out_of_domain>`
returns a dict of {class}`~robust.rt.emulator.DomainBreach` records — feature,
range, worst value, fraction outside, and `excess`, how far the worst value
lies beyond the range **as a fraction of the trained span**. It is a boundary
check and needs concrete values, so it cannot run under `jit`;
{meth}`Emulator.out_of_domain_mask <robust.rt.emulator.Emulator.out_of_domain_mask>`
is the traceable version a fallback policy can act on inside `jit`.

Two thresholds, both chosen from measurements rather than taste:

{data}`~robust.rt.emulator.DOMAIN_TOL` = 0.01
: The domain is the *training split's* min/max, so held-out data legitimately
  grazes it — on the full L23 batch the packaged emulator sees `eta_bb` reach
  3.7e-4 of a span beyond the boundary, for four values in a million. A
  genuinely unsupported input — a 75° sun against a `cos_theta_s` floor of 0.5
  — sits **48 %** of the span beyond, three orders of magnitude further out. A
  zero-tolerance check fires on the first case, trains the user to silence the
  warning, and then lets the second pass unnoticed. 1 % sits between them with
  ~27× headroom below and ~48× above.

`SUPPORTED_THETA_S` = (0°, 60°)
: A **project decision**, not a property of any fit: L23 provides 0°, 30° and
  60°, so 0–60° is the span the reference data covers and inside which
  interpolation is sanctioned. `cos_theta_s` is judged against this envelope,
  so a fit trained on 0°/30° only is *allowed* to be asked for 60° without
  complaint. Every other feature is judged against its own trained range. Pass
  `theta_s_limits=None` to judge the zenith by the trained range too — the
  right question when the subject is a particular fit's extrapolation rather
  than the package's supported envelope.

How the guard is wired into {func}`~robust.rt.hybrid.forward`, including
`on_out_of_domain='ztt'`, is in {doc}`forward`.

## The uncomfortable result: unseen geometry

Everything above concerns *interpolation*, where the emulator is excellent.
Extrapolation in geometry is a different story, and it is reported rather than
gated.

Trained on 0°/30° only and asked for the unseen 60°, the MLP hybrid scores
**4.74–12.24 % across five random seeds, median 7.75 %**, against ~0.24 % in
sample. The cause is plain: `cos_theta_s` spans [0.866, 1.0] in that training
set and 60° needs 0.5, so every `tanh` unit is evaluated outside its fitted
range, where nothing constrains it and the initialisation decides the answer.
The **linear** model gives up a great deal in sample and lands at **6.16 %**
there, stably — its inability to bend is what saves it — and the refit O25
benchmark wins outright and deterministically at **4.63 %**
([`reports/report_rt_elastic_model.md`](gh:reports/report_rt_elastic_model.md)
§4; the module docstring's decision 4).

:::{warning}
**Geometry generalization may not be claimed.** At an unseen solar zenith the
benchmark beats the hybrid and is more reproducible. The acceptance gate was
deliberately written on the scene split, and the headline 0.30 % is an
*interpolation* result. This is item 3 of the report's "may not claim" list.
:::

A cautionary note the module docstring keeps, because the lesson generalises: a
linear skip path appeared to fix this (11.57 % → 5.40 %) and was very nearly
adopted on that one comparison — but the two runs also differed in their Flax
parameter *names*, which changes PRNG folding and hence the initialisation, so
architecture and seed moved together. Sweeping seeds with the architecture
fixed showed the skip is no better (median 9.20 %, 25 % worst case). Both
numbers were real; the inference from them was not.

## `mode='emulator'` returns a term, not a model

Because the correction is parameterised *relative to the backbone*, "the
learned part on its own" is the correction term $\Delta r_{rs}$ — not a
standalone learned model that could replace the physics. A genuine learned-only
comparison would need a differently trained network predicting $r_{rs}$
outright across four decades of dynamic range; that would be a model to add
beside the baselines, not a flag on
{func}`~robust.rt.hybrid.forward`. Two consequences, both in {doc}`forward`:
the mode is incompatible with `inelastic=`, and its output is **not additive
with `mode='ztt'` in $R_{rs}$ space**.

## The packaged weights

{data}`~robust.rt.emulator.DEFAULT_WEIGHTS` points at
`robust/rt/files/emulator_l23.npz`: an MLP(16,16) fit on L23's elastic X=1
scenes, **`scene_train` split only**, by `design/py/train_emulator.py`. It is
committed (6.5 kB) so that {func}`~robust.rt.hybrid.forward` is a *trained*
model out of the box and CI can exercise the real thing.
{func}`~robust.rt.emulator.load_default` reads it once per process and
memoises. If the file is missing, that call raises `FileNotFoundError` naming
the command that regenerates it — unlike the two inelastic correction heads,
which degrade to analytic-only behind a warning.

{func}`~robust.rt.emulator.load` **refuses** a file whose stored feature list
differs from the current {data}`~robust.rt.emulator.FEATURES`. The weights
would still *run* — the shapes need not have changed — and would return
plausible nonsense, so this is a refusal rather than a warning.

Retraining is {func}`~robust.rt.emulator.fit` (arrays) or
{func}`~robust.rt.emulator.fit_l23` (a loaded batch plus splits), with
{func}`~robust.rt.emulator.save` writing the same `.npz` layout;
{class}`~robust.rt.emulator.History` carries the learning curve, including
`delta_rms` — the *magnitude of the correction* reported as a first-class
number rather than inferred from the loss.

## API

| What | Where |
| --- | --- |
| The trained object | {class}`~robust.rt.emulator.Emulator`, {meth}`~robust.rt.emulator.Emulator.relative_delta`, {meth}`~robust.rt.emulator.Emulator.delta_rrs` |
| Features | {data}`~robust.rt.emulator.FEATURES`, {func}`~robust.rt.emulator.features` |
| Configuration | {class}`~robust.rt.emulator.EmulatorConfig`, {data}`~robust.rt.emulator.LINEAR_CONFIG` |
| Domain guard | {attr}`~robust.rt.emulator.Emulator.domain`, {meth}`~robust.rt.emulator.Emulator.out_of_domain`, {meth}`~robust.rt.emulator.Emulator.out_of_domain_mask`, {class}`~robust.rt.emulator.DomainBreach`, {data}`~robust.rt.emulator.DOMAIN_TOL` |
| Weights | {data}`~robust.rt.emulator.DEFAULT_WEIGHTS`, {func}`~robust.rt.emulator.load_default`, {func}`~robust.rt.emulator.load`, {func}`~robust.rt.emulator.save` |
| Training | {func}`~robust.rt.emulator.fit`, {func}`~robust.rt.emulator.fit_l23`, {class}`~robust.rt.emulator.History` |

Full signatures are on the {doc}`../api` page under *emulator*.
