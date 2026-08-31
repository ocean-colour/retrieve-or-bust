# Raman scattering

Water Raman scattering redistributes photons from blue excitation wavelengths
into longer emission wavelengths, and in the L23 truth it contributes a median
5–15 % of $R_{rs}$ over 520–750 nm at solar zeniths 30–60°, up to ~20 % at 0°
([`reports/report_rt_inelastic_model.md`](gh:reports/report_rt_inelastic_model.md)
§1). It is not a refinement; an elastic-only model aliases it into biased IOPs.

{mod}`robust.rt.inelastic` implements it as a **multiplicative factor**
$f_R(\lambda) \ge 1$ on the elastic reflectance, not as an additive term. That
choice is physical rather than cosmetic: the factor is assembled as a *ratio* of
two-flow reflectances, so normalization errors that would corrupt an absolute
term cancel in the quotient. The module docstring records the consequence
plainly — this is why BING's Raman was about right while its pre-fix additive
fluorescence was a factor of three too bright.

*Sources for this page: the {mod}`robust.rt.inelastic` module docstring and the
docstrings of {func}`~robust.rt.inelastic.raman_factor`,
{func}`~robust.rt.inelastic.raman_bb`,
{func}`~robust.rt.conventions.raman_excitation` and
{func}`~robust.rt.conventions.raman_emission`;
[`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §3 and §4.3;
[`reports/report_rt_inelastic_model.md`](gh:reports/report_rt_inelastic_model.md)
§§1, 2, 4 and 5; and
[`design/rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md)
§4.2. Every number labelled "measured" was measured in this environment when the
page was written, on the committed 50-scene fixture; the report's numbers are
labelled as the report's and are the ones to quote.*

## The wavenumber shift, and the arithmetic that follows from it

Raman scattering by liquid water is dominated by the O–H stretch band, and v1
uses its single centre shift rather than the full band:

$$\tilde\nu_{\rm em} = \tilde\nu_{\rm ex} - \Delta\tilde\nu,
\qquad \Delta\tilde\nu = 3400\ {\rm cm^{-1}}$$

{data}`~robust.rt.conventions.RAMAN_SHIFT` is that 3400 cm⁻¹, attributed in its
own doc comment to Ge, Gordon & Voss (1993). The physical band is ~3100–3700
cm⁻¹ wide (Walrafen 1967); the design's judgement is that at the 5 nm resolution
of the canonical grid, the single-shift error is subdominant and what remains is
δ_R's job ([`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md)
§4.3).

Written for the quantity the forward model actually needs — *given* an emission
wavelength λ, which excitation wavelength λ′ feeds it — the same relation is

$$\frac{1}{\lambda'} = \frac{1}{\lambda} + \Delta\tilde\nu$$

and the only thing that can go wrong is the unit bridge. Wavenumbers are per
centimetre, wavelengths are nanometres, and 1 cm = 10⁷ nm, so

$$3400\ {\rm cm^{-1}}
 = \frac{3400}{10^{7}}\ {\rm nm^{-1}}
 = 3.4\times10^{-4}\ {\rm nm^{-1}}$$

which is exactly the `shift * 1e-7` that appears in
{func}`~robust.rt.conventions.raman_excitation` and
{func}`~robust.rt.conventions.raman_emission`.

### The 488 nm case, worked

488 nm is the conventional anchor — it is where the Raman scattering
coefficient is quoted — so the emission wavelength it excites is worth doing by
hand:

```text
1 / 488 nm                        = 0.0020491803  nm^-1
minus 3.4e-4 nm^-1                = 0.0017091803  nm^-1
reciprocal                        = 585.075772    nm
```

so **488 nm excitation emits at 585.08 nm**. The same figure in the cm⁻¹ form,
as an independent path through the arithmetic: 10⁷/488 = 20491.8033 cm⁻¹, minus
3400 gives 17091.8033 cm⁻¹, and 10⁷/17091.8033 = 585.075772 nm. In exact
rational arithmetic the value is 3050000/5213 nm, and
{func}`~robust.rt.conventions.raman_emission` reproduces it to the last printed
digit:

```text
raman_emission(488.0)   = 585.075772108 nm   (float64)
exact rational          = 585.075772108 nm
difference              = 0.000e+00     nm
raman_emission(488.0)   = 585.075806     nm   (float32, the default dtype)
```

:::{important}
**This is one of the numbers to check rather than copy.** BING's own
`bing.rt.raman.excitation_to_emission_wavelength` *computes* 585.08 nm
correctly, but the worked example in its docstring reads

```text
>>> excitation_to_emission_wavelength(488)
583.0  # approximately
```

which is wrong by 2 nm. (Its second example, 400 nm → 463.0 nm, is right —
the true value is 462.96 nm.) `robust.rt`'s own docstrings and this page carry
585.08 nm, re-derived above rather than inherited, and the round trip
`raman_emission(raman_excitation(488))` returns 488.000000000 nm in float64.
:::

The inverse direction is the one the code runs, and it is a different number
from the same equation: an emission wavelength of 488 nm is fed by
`raman_excitation(488.0) = 418.553589 nm`.

## The excitation grid

{func}`~robust.rt.inelastic.raman_factor` does not take an excitation grid. It
builds one, by mapping the emission wavelengths it was asked for through
{func}`~robust.rt.conventions.raman_excitation`, then interpolating the supplied
IOP spectra onto it with {func}`~robust.rt.conventions.interp_spectrum` — the
package's single interpolation rule, differentiable in the spectrum values,
which is what lets gradients flow from the excitation-grid IOPs back to the IOP
inputs. There is no API surface: the grid is internal
([`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §3).

Measured, over the canonical 81-point 350–750 nm emission grid:

```text
lambda_em   350.000 .. 750.000 nm     (81 points, uniform 5 nm)
lambda_ex   312.779 .. 597.610 nm     (81 points, NOT uniform)
grid spacing of lambda_ex: 3.179 .. 3.987 nm
```

A **fixed shift in wavenumber is a varying step in wavelength**, so the
excitation grid is compressed in the blue and stretched in the red. Nothing
depends on it being uniform, but it is worth knowing before reading a plot
against λ′.

### Why the supported domain starts at 400 nm

The blue end of that excitation grid falls off the blue end of the L23 data:

```text
raman_excitation(400.0) = 352.113 nm     <- just inside WAVE_MIN = 350 nm
raman_excitation(395.0) = 348.246 nm     <- outside
excitation wavelengths below 350 nm: 10 of the 81 emission points,
  i.e. every emission wavelength <= 395 nm
```

That single fact is the whole reason
{data}`~robust.rt.conventions.RAMAN_WAVE_MIN_OFFICIAL` is 400 nm and the reported
gate band is 400–700 nm. Below 400 nm the maps and the interpolation still
*run* — no error, by design, because a `raise` there could not execute under
`jit` — but {func}`~robust.rt.conventions.interp_spectrum` clamps to the 350 nm
end value, so the term is evaluated on constant-extrapolated IOPs.

:::{warning}
**Below 400 nm this term runs on extrapolated inputs and the correction heads
never trained there.** The report measures the cost: **13 % error at 350 nm**,
against 0.34 % inside the band
([`reports/report_rt_inelastic_model.md`](gh:reports/report_rt_inelastic_model.md)
§4 and §5, may-not-claim item 4). The domain is documented, **not enforced by an
error**. The full 350–750 nm grid is always reported and never gated (report §3).
:::

## The scattering coefficient

{func}`~robust.rt.inelastic.raman_bb` is the backscattering coefficient of the
Raman source, in energy units:

$$b_{bR}(\lambda') = \tfrac{1}{2}\, b_R(488)\,
  \left(\frac{488}{\lambda'}\right)^{5.5}$$

| Constant | Value | Provenance |
| --- | --- | --- |
| {data}`~robust.rt.inelastic.B_RAMAN_488` | 2.6 × 10⁻⁴ m⁻¹ | the **HydroLight** value, i.e. what generated the L23 truth. Bartlett et al. (1998) measured 2.7 × 10⁻⁴; Desiderio 2.4 × 10⁻⁴. Matching the truth's generator wins (design §4.3) |
| {data}`~robust.rt.inelastic.RAMAN_EXPONENT` | 5.5 | excitation-wavelength dependence in **energy** units |
| {data}`~robust.rt.inelastic.RAMAN_BB_RATIO` | 0.5 | Rayleigh-like backward fraction. The ρ = 0.17 depolarization gives 0.489; BING's default rounds to ½, and term-for-term BING equality was the porting contract |

Measured: `raman_bb(488) = 1.300000e-04 m^-1`, which is ½ × 2.6 × 10⁻⁴ exactly,
and the λ′⁻⁵·⁵ slope is steep enough to matter — 7.83 × 10⁻⁴ m⁻¹ at the 352 nm
grid edge against 6.37 × 10⁻⁵ m⁻¹ at 555.60 nm, a factor of 12 across the
excitation band.

## `raman_factor`: inputs, output, and what it assembles

{func}`~robust.rt.inelastic.raman_factor` takes
`(iops, geometry, wave=None)` and returns $f_{\rm phys}(\lambda)$, shape
`(*batch, n_wave)`, dimensionless and ≥ 1. It is pure JAX: batched over leading
axes, `jit`/`vmap`-safe, and differentiable in every input.

What it needs, and what it deliberately does not:

- **{attr}`IOPs.a <robust.rt.types.IOPs.a>` and the backscattering split** — on
  the emission grid, then interpolated onto the excitation grid. `a_ph` is
  **not used**: water scatters regardless of what absorbs.
- **{attr}`Geometry.theta_s <robust.rt.types.Geometry.theta_s>`** — only to
  select the packaged sky. A {attr}`Geometry.Ed <robust.rt.types.Geometry.Ed>`
  override replaces the sky entirely.
- **{func}`robust.rt.ed.ratio`** — the true spectral ratio
  $E_d(\lambda')/E_d(\lambda)$. This is a first-order dependence, not a
  detail: measured over 400–750 nm it swings 0.445 → 1.579 at θ_s = 0° and
  0.424 → 1.619 at 60°, a factor of 3.6–3.8 across the band. Running the term
  on a flat solar spectrum instead is wrong by **+60 % in the blue to −50 % in
  the red** in increment
  ([`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §4.2) —
  one of the two BING defects the assessment found and fixed upstream. See
  {doc}`ed`.

The assembly is the Sathyendranath & Platt (1998) two-flow framework, term for
term the fixed BING's `calc_raman_correction_factor`:

| Term | S&P98 Eq. | What it is |
| --- | --- | --- |
| $R_E$ | 5 | elastic two-flow reflectance — the denominator |
| $R_R$ | 11 | first-order Raman: absorbed at λ′, re-emitted at λ |
| $R_{RE}$ | 18 | second order, Raman down then elastic up (~10 % of first order) |
| $R_{ER}$ | 23 | second order, elastic down then Raman up (~10 %) |

$$f_{\rm phys}(\lambda) =
  \frac{R_E + R_R + R_{RE} + R_{ER}}{R_E}$$

Raman–Raman terms (~1 %) are neglected, as in S&P98 §4.A. The mean cosines are
fixed: {data}`~robust.rt.inelastic.MU_D` = 0.9 for the downwelling stream
(clear sky, high sun), and 0.5 for both the upwelling and the isotropic
Raman-scattered streams (`MU_U`, `MU_R` in the module; they share a doc comment
with `MU_D`, so autodoc does not emit them separately — see {doc}`../api`).
{data}`~robust.rt.inelastic.S_E` = 1.0 is the isotropic elastic shape factor.

Measured on the fixture, all 150 samples × 81 wavelengths:

```text
f_phys:  min 1.020783   median 1.1228   max 1.2888
per zenith, median f_phys:  0 deg 1.1242 | 30 deg 1.1235 | 60 deg 1.1205
```

The function's own docstring quotes the range as 1.0076–2.5 on the full L23
release; the fixture is 50 of its 3,320 water bodies, so the narrower spread
above is a subset, not a disagreement.

## How it composes into `forward()`

The composition law is written in $R_{rs}$ space, and Raman is the
multiplicative factor in it
([`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §2):

$$R_{rs}^{\rm total}(\lambda) =
  \bigl(R_{rs}^{\rm ZTT} + \Delta R_{rs}\bigr) \times f_R(\lambda)
  \;+\; \varphi_C K_{\rm fl}(\lambda)$$

with the corrected factor

$$f_R = 1 + (f_{\rm phys} - 1)(1 + \delta_R)$$

The correction rescales the *increment*, never the baseline, so $f_R \to 1$
wherever Raman vanishes no matter what the network outputs — see
{doc}`corrections`. With no head, $f_R = f_{\rm phys}$ exactly.

Practically: {func}`robust.rt.hybrid.forward` grows one keyword. `inelastic=None` (the
default) is the elastic model, bit-identical by construction;
`inelastic=Inelastic()` turns both inelastic processes on at their L23-matched
defaults; `Inelastic(fluorescence=False)` is the Raman-only model, which needs
no `a_ph`. {doc}`forward` documents which space the arithmetic happens in and
why {func}`~robust.rt.conventions.rrs_to_Rrs` sits where it does.

## What the analytic backbone gets wrong

The two-flow assembly above is a *fixed-mean-cosine* approximation, and its
errors are structural rather than parametric — no constant in the table above
fixes them. Quoted from
[`reports/report_rt_inelastic_model.md`](gh:reports/report_rt_inelastic_model.md)
§4, per-process fidelity table (medians on held-out scenes, analytic →
corrected):

| Raman increment error [%] | 0° | 30° | 60° |
| --- | --- | --- | --- |
| 550–700 nm, analytic | **−38.6** | +1.2 | −4.2 |
| 550–700 nm, corrected | −0.14 | −0.10 | −0.21 |
| 490 nm, analytic | −3.6 | **+30.8** | **+32.5** |
| 490 nm, corrected | +1.03 | +0.82 | +0.58 |

The headline is the **−38.6 % at overhead sun**: the error a fixed-µ two-flow
cannot avoid, and the one that motivated a learned head at all. The
{mod}`robust.rt.inelastic` module docstring states the same bands as the
backbone's *known accuracy* and notes that M2's characterization test pins the
measured error band rather than zero — a test asserting 0 % there would be
asserting something the physics does not deliver.

Two consequences a user should carry away. The analytic term alone scores 2–4 %
total rRMS against the all-processes-on truth (report §4), which is already most
of the ~50× an elastic-only model loses. And the shipped default is the
*corrected* model, so these are not the numbers `forward()` produces unless you
ask for `corrections=False`.

## API

| What | Where |
| --- | --- |
| The wavenumber shift | {data}`~robust.rt.conventions.RAMAN_SHIFT` |
| Emission ↔ excitation | {func}`~robust.rt.conventions.raman_excitation`, {func}`~robust.rt.conventions.raman_emission` |
| The supported lower edge | {data}`~robust.rt.conventions.RAMAN_WAVE_MIN_OFFICIAL` |
| The scattering coefficient | {func}`~robust.rt.inelastic.raman_bb`, {data}`~robust.rt.inelastic.B_RAMAN_488`, {data}`~robust.rt.inelastic.RAMAN_EXPONENT`, {data}`~robust.rt.inelastic.RAMAN_BB_RATIO` |
| The factor | {func}`~robust.rt.inelastic.raman_factor` |
| Two-flow mean cosines | {data}`~robust.rt.inelastic.MU_D`, {data}`~robust.rt.inelastic.S_E` |
| The sky it consumes | {func}`robust.rt.ed.ratio` |
| The switch | {attr}`Inelastic.raman <robust.rt.types.Inelastic.raman>` |
| The learned rescaling | {func}`~robust.rt.inelastic_corr.corrected_raman_factor` |

Full signatures are on the {doc}`../api` page under *inelastic*. The additive
sibling term is {doc}`fluorescence`; the heads that correct both are
{doc}`corrections`; the protocol that scores them is {doc}`../using/validation`.
