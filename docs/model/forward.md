# `forward()` and the composition law

{mod}`robust.rt.hybrid` owns the only signature the rest of the world should
depend on. {func}`~robust.rt.hybrid.forward` returns above-water $R_{rs}$;
{func}`~robust.rt.hybrid.rrs_forward` returns the subsurface $r_{rs}$ that
everything is scored in. Both are one JAX function, differentiable end to end,
batched over leading axes, and `jit`/`vmap`-safe.

This chapter is the entry point's *behaviour*: the `mode` flag, which space the
arithmetic happens in, the bit-identity guarantee, and the domain guard. The
signature itself is on the {doc}`../api` page —
{func}`robust.rt.hybrid.forward` — and is not restated here.

*Sources for this page: the {mod}`robust.rt.hybrid` module docstring and the
docstrings of {func}`~robust.rt.hybrid.rrs_forward`,
{func}`~robust.rt.hybrid.forward` and {data}`~robust.rt.hybrid.MODES` /
{data}`~robust.rt.hybrid.OUT_OF_DOMAIN_POLICIES`;
[`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md) §§3, 4.5 and 6.
Every code block's output was produced in this environment when the page was
written, on the committed 50-scene L23 fixture.*

## The two entry points

$$
\underbrace{r_{rs} \;=\; r_{rs}^{\mathrm{ZTT}} + \Delta r_{rs}}_{\texttt{rrs-forward}},
\qquad
\underbrace{R_{rs} \;=\; \frac{A\,r_{rs}}{1 - B\,r_{rs}}}_{\texttt{forward}} .
$$

{func}`~robust.rt.hybrid.forward` is literally
{func}`~robust.rt.conventions.rrs_to_Rrs` applied to
{func}`~robust.rt.hybrid.rrs_forward`; both take the same arguments. Use
`forward` for the above-water quantity an instrument would see, and
`rrs_forward` when the parts must sum and for **all scoring**
([`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md) §6).

The three positional arguments are the pytrees of {mod}`robust.rt.types` —
{class}`~robust.rt.types.IOPs`, {class}`~robust.rt.types.PhaseParams`,
{class}`~robust.rt.types.Geometry` — plus an optional `wave` that defaults to
the canonical grid. Everything after `mode` is keyword-only.

## The three modes

`mode` selects how much of the elastic half runs, so all three configurations
compare on identical splits rather than on separately prepared data
([`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md) §4.5). The
values are {data}`~robust.rt.hybrid.MODES`:

`'ztt'`
: The analytic backbone alone — **bitwise** equal to
  {func}`robust.rt.ztt.rrs_ZTT` (verified). Takes no emulator, loads no weights,
  cannot raise a domain warning, and needs neither `flax` nor `optax`.

`'emulator'`
: The learned correction $\Delta r_{rs}$ **alone**: a term, not a model. See
  {doc}`emulator`. Incompatible with `inelastic=`, and not additive with
  `'ztt'` in $R_{rs}$ space — both below.

`'hybrid'`
: $r_{rs}^{\mathrm{ZTT}} + \Delta r_{rs}$ — the default, and the model every
  accuracy number on this site refers to.

A `mode` outside that tuple raises rather than silently defaulting, because a
typo'd mode returning the hybrid would corrupt a comparison table with no
visible symptom:

```text
>>> forward(iops, phase_params, geometry, wave, "quadratic")
ValueError: forward: mode must be one of ('ztt', 'emulator', 'hybrid'); got 'quadratic'
```

## Which space the arithmetic happens in

**Additivity holds exactly in $r_{rs}$ space and fails in $R_{rs}$ space.** The
correction is defined and trained below the surface, so the three modes sum
there; the air–water interface $R_{rs} = A\,r_{rs}/(1 - B\,r_{rs})$ is
non-linear, so they do not sum above it. Measured on the first fixture scene at
440 nm:

```text
              ztt            emulator        sum            hybrid
rrs    1.699806e-02    -1.042945e-03    1.595511e-02    1.595511e-02   <- bitwise equal
Rrs    9.102007e-03    -5.413715e-04    8.560636e-03    8.527968e-03

Rrs:  sum - hybrid = +3.2668e-05   (0.38 % of the hybrid value)
```

Over the whole 81-point grid the discrepancy peaks at 6.22e-05 sr⁻¹ at 350 nm,
0.57 % of the hybrid value there. Small, but it is exactly the size of a
model-comparison margin, so:

:::{warning}
Never reconstruct the hybrid by adding `mode='ztt'` and `mode='emulator'`
outputs of {func}`~robust.rt.hybrid.forward`. Add them from
{func}`~robust.rt.hybrid.rrs_forward` — where the identity is bitwise — or call
`mode='hybrid'` and be done. This is also why the acceptance metric is rRMS in
$r_{rs}$ space: a relative error in $R_{rs}$ is a different number.
:::

## `inelastic=None` is bit-identical to the elastic model

`forward(..., inelastic=None)` — the default — is the elastic hybrid, byte for
byte. Not approximately, and not by arithmetic that happens to cancel: the
`None` branch returns the elastic result object **untouched**, rather than
composing terms that evaluate to zero and paying a round trip through
$r_{rs} \to R_{rs} \to r_{rs}$ at ULP precision. A test pins the SHA-256 of the
fixture output.

The same holds one level down. `corrections` is never even *resolved* when
`inelastic` is `None` or all its processes are off, so the elastic path never
imports the ML stack on the inelastic terms' account and never emits their
warnings.

The practical consequence: adding the inelastic terms to the package did not
disturb a single number the elastic report claims, and the elastic acceptance
gate stayed valid by construction.

When an {class}`~robust.rt.types.Inelastic` *is* passed, the composition law
is applied in $R_{rs}$ space on top of the finished elastic model:

$$
R_{rs} \;=\;
\bigl[R_{rs}^{\mathrm{ZTT}} + \Delta R_{rs}\bigr]\;f_R
\;+\; \varphi_C\,K_{\mathrm{fl}} .
$$

The terms, the learned corrections $\delta_R$/$\delta_F$, and
`corrections=None` versus `corrections=False` are the subject of
{doc}`inelastic`, {doc}`fluorescence` and {doc}`corrections`; the composition
law as a whole is on {doc}`overview`. Two guards belong here, though, because
{func}`~robust.rt.hybrid.rrs_forward` raises them before anything is computed:

```text
>>> forward(..., mode="emulator", inelastic=Inelastic(fluorescence=False))
ValueError: forward: mode='emulator' returns the learned correction term alone
            (a term, not a model — see the module docstring); the inelastic
            composition applies to a model output. Use mode='ztt' or
            mode='hybrid' with inelastic, or inelastic=None with mode='emulator'
```

and `Inelastic.fluorescence` requires `IOPs.a_ph` (the source term is
$b_F = \varphi_C\,a_{ph}$, and bulk absorption cannot stand in for the
phytoplankton component), just as `Inelastic.cdom_fl` requires `IOPs.a_cdom`.
Both fail fast, before the emulator loads.

## The domain guard, and `on_out_of_domain`

Any mode that involves the emulator checks its inputs against the trained
{attr}`Emulator.domain <robust.rt.emulator.Emulator.domain>` (see
{doc}`emulator`) and warns. Two separate switches control it, and the
difference between them matters.

`check_domain` (default `True`) — the **warning**
: Emits {class}`~robust.rt.hybrid.DomainWarning`, naming every offending
  feature, how much of the input is outside, the worst value, and how far
  beyond the trained span it lies. Measured with a 75° sun:

  ```text
  DomainWarning: the emulator is being evaluated outside its training range,
  where M3 measured its accuracy to be unreliable and occasionally worse than
  the analytic backbone: cos_theta_s 100.0% of values outside [0.5, 1], worst
  0.2588 — 48% of the trained span beyond it. Consider mode='ztt'.
  (cos_theta_s decreases with solar zenith, so a low sun breaches the lower
  bound.)
  ```

  It is its own warning category, so a pipeline that must not silently
  extrapolate can promote it to an error with
  `warnings.simplefilter("error", DomainWarning)`, and a study that means to
  extrapolate can silence it once.

`on_out_of_domain` — the **policy**
: {data}`~robust.rt.hybrid.OUT_OF_DOMAIN_POLICIES` is `("warn", "ztt")`.
  `"warn"` (default) evaluates the emulator anyway. `"ztt"` additionally
  **zeroes the learned correction** on samples outside the accepted range, so
  the model degrades to the analytic backbone exactly where the emulator was
  measured to be unreliable. Verified: at 75° with `on_out_of_domain="ztt"`,
  `rrs_forward(..., "hybrid")` is bitwise equal to `rrs_forward(..., "ztt")`.

  It is an option rather than the default because switching it on changes
  numbers, and a model whose output depends on a flag nobody set is its own
  kind of trap.

:::{important}
**The warning is skipped under `jit` and under `grad`; the policy is not.**
`check_domain` inspects concrete values, so it is skipped automatically
whenever any input is traced — measured: zero warnings from a jitted call and
zero from `jax.grad`, at the same 75° geometry that warns eagerly. That is
deliberate and documented rather than silent: `jit` is the hot path, and a
check that cannot run there should not pretend to.

`on_out_of_domain='ztt'`, by contrast, is built on a **traceable** mask, so it
holds under `jit` and `grad` exactly as it does outside them. A policy that
lapsed under compilation would be worse than none.

The tracer test looks at *every* leaf of the input pytrees, not just a
representative one: `jax.grad` with respect to a single input traces only that
one, and an earlier version that checked `iops.a` and `geometry.theta_s` alone
declared "not traced" while differentiating with respect to `bb_p` — which is
precisely the inversion's use case.
:::

## Batching and gradients

A full L23 batch is one call — leading axes broadcast, and the fixture's 150
samples return `(150, 81)`. Gradients are the *purpose* of the function rather
than a by-product: `jax.grad` of a scalar of {func}`~robust.rt.hybrid.forward`
with respect to an {class}`~robust.rt.types.IOPs` returns an
{class}`~robust.rt.types.IOPs` of per-field derivatives, which is the shape a
retrieval wants. Tests pin those against central finite differences; the
elastic report records agreement to ≤ 5×10⁻⁹
([`reports/report_rt_elastic_model.md`](gh:reports/report_rt_elastic_model.md)
§4). A worked gradient, including the $\varphi_C$-linearity identity, is in the
{doc}`../quickstart`.

:::{note}
The retrieval itself does not exist. `robust.rt` is a forward model —
retrieve-or-bust's first major contribution — and the inversion from a measured
spectrum back to IOPs is a separate component that has not been built. This
interface is *shaped* for it, which is not the same as containing it.
:::

## API

| What | Where |
| --- | --- |
| The entry points | {func}`~robust.rt.hybrid.forward`, {func}`~robust.rt.hybrid.rrs_forward` |
| Modes and policies | {data}`~robust.rt.hybrid.MODES`, {data}`~robust.rt.hybrid.OUT_OF_DOMAIN_POLICIES` |
| The warning | {class}`~robust.rt.hybrid.DomainWarning` |
| The arguments | {class}`~robust.rt.types.IOPs`, {class}`~robust.rt.types.PhaseParams`, {class}`~robust.rt.types.Geometry`, {class}`~robust.rt.types.Inelastic` |

Full signatures — including the keyword-only `inelastic` / `corrections` /
`emulator` / `check_domain` / `on_out_of_domain` — are on the {doc}`../api`
page under *hybrid*.
