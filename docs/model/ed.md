# Downwelling irradiance

{mod}`robust.rt.ed` supplies $E_d(0^+;\ \theta_s, \lambda)$ — the downwelling
irradiance just above the sea surface, in W m⁻² nm⁻¹ — from three packaged L23
sky spectra, with a caller-supplied override where a real sky plugs in.

**The elastic model never calls this module.** The inelastic terms are its only
consumers: Raman needs the true spectral ratio $E_d(\lambda')/E_d(\lambda)$,
and chlorophyll fluorescence needs $E_d(\lambda')$ inside its excitation
quadrature. It is documented among the elastic chapters because it is
infrastructure — a sky, an interpolation rule and a seam — rather than physics
of either kind.

*Sources for this page: the {mod}`robust.rt.ed` module docstring and the
docstrings of {func}`~robust.rt.ed.Ed`, {func}`~robust.rt.ed.ratio` and
{func}`~robust.rt.ed.load_table`;
[`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §4.2 and
[`design/rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md)
§3 (the measured ratio and the flat-Ed error).
Numbers marked "measured" were re-measured from the packaged file in this
environment when the page was written.*

## What is packaged

`robust/rt/data/ed_l23.npz` holds **one $E_d(0^+)$ spectrum per L23 solar
zenith**, on the canonical 81-point grid:

```text
zeniths:  (0.0, 30.0, 60.0) deg            <- ZENITH_ANCHORS
wave:     81 points, identical to conventions.WAVE
Ed:       (3, 81)
peak Ed(0 deg) = 1.8133 W m^-2 nm^-1 at 480 nm
Ed(60 deg) / Ed(0 deg) at 550 nm = 0.4394
```

It was extracted after asserting two things that make the single-spectrum-per-
zenith representation legitimate: **scene-independence** (below 1e-3 relative
scatter; measured ~5e-5) and **identity across the X=1/2/4 scenarios**. In other
words there is no scene to choose, and the sky does not depend on which
inelastic processes HydroLight was asked to include.

{func}`~robust.rt.ed.load_table` returns the file as
`(wave (81,), Ed (3, 81))` **NumPy** arrays, loaded lazily on first use and
cached — importing {mod}`robust.rt` must not cost a file read — and NumPy
rather than JAX for the same reason {data}`robust.rt.conventions.WAVE` is: a
device array built here would fix its dtype before a caller can enable float64.
It also raises `ValueError` if the file's zenith rows are not
{data}`~robust.rt.ed.ZENITH_ANCHORS`, so the file and the module version each
other.

## Interpolation, in two axes, both clamped

{func}`~robust.rt.ed.Ed` interpolates **linearly in $\theta_s$** between the
three anchors and clamps outside 0–60°, then interpolates in wavelength with
the package's single rule ({func}`~robust.rt.conventions.interp_spectrum`),
clamping at the 350/750 nm ends. No silent extrapolation, and no `raise` that
could not run under `jit` — the {func}`robust.rt.conventions.bb_w` precedent.

Verified:

```text
Ed(15 deg) == mean of Ed(0 deg) and Ed(30 deg)     True
Ed(75 deg) == Ed(60 deg)                           True
```

The zenith interpolation is `searchsorted`-based rather than stride-based, so
any strictly increasing anchor set works — a future non-uniform set needs no
code change. The result is differentiable in `theta_s` and in override values,
and safe under `jit`/`vmap`.

## The ratio the Raman term consumes

{func}`~robust.rt.ed.ratio` returns $E_d(\lambda_{\text{num}})/E_d(\lambda_{\text{den}})$
at one geometry. It exists as a helper rather than as two calls so that both
wavelength sets are guaranteed to be evaluated from the *same* sky: mixing the
packaged $E_d$ in the numerator with an override in the denominator can then
never happen.

The Raman term feeds it the excitation grid
$\lambda' = $ {func}`~robust.rt.conventions.raman_excitation`$(\lambda)$ in the
numerator and the emission wavelengths $\lambda$ in the denominator. Measured
from 400 nm — the inelastic model's official lower edge,
{data}`~robust.rt.conventions.RAMAN_WAVE_MIN_OFFICIAL` — to the top of the
grid:

| $\theta_s$ | min | at | max | at | swing |
| --- | --- | --- | --- | --- | --- |
| 0° | 0.445 | 445 nm | 1.579 | 720 nm | ×3.55 |
| 30° | 0.440 | 445 nm | 1.588 | 720 nm | ×3.61 |
| 60° | 0.424 | 445 nm | 1.619 | 720 nm | ×3.82 |

Two things fall out of that table. The ratio is **strongly spectral** —
a factor of ~3.6 across the band — and it is **nearly zenith-independent**,
because the sky's *shape* barely changes with sun angle while its amplitude
cancels in the ratio.

:::{important}
This is why the module exists at all. Assuming a flat $E_d$ — i.e. taking the
ratio as 1 — makes the Raman increment wrong by **+60 % in the blue to −50 % in
the red** ([`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md)),
and the table above is the quantitative footing of that assessment
([`design/rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md)
§3).
A test asserts that the ratio over the Raman excitation map is *not* flat
(max/min > 1.5), so a regression back to a flat sky fails loudly.
:::

## The `Geometry.Ed` override

{attr}`Geometry.Ed <robust.rt.types.Geometry.Ed>` is an optional
`(wave_Ed, Ed)` pair of 1-D arrays on their own grid, passed straight through
to {func}`~robust.rt.ed.Ed`'s `override=`. Its semantics are deliberately
absolute:

- it **replaces the packaged sky entirely**;
- `theta_s` is then **ignored** — an override *is* one particular sky, zenith
  dependence included;
- it is interpolated onto the requested wavelengths with the same clamped
  linear rule.

This is the seam through which real-sky irradiances enter later with no
interface change, and the elastic path ignores it entirely.

:::{warning}
**The solar model is inherited, not chosen.** The community's
solar-irradiance reference models are themselves imperfect. `robust.rt`
deliberately inherits whatever solar spectrum HydroLight/L23 used, because
consistency with the truth data matters more than absolute solar accuracy for a
forward model scored against that data. When the effort moves to real PACE
spectra, the $E_d$ source (TSIS-1-era references versus older standards) must
be revisited — and {attr}`Geometry.Ed <robust.rt.types.Geometry.Ed>` is where
that happens, with no interface change.
:::

## API

| What | Where |
| --- | --- |
| The sky | {func}`~robust.rt.ed.Ed`, {data}`~robust.rt.ed.ZENITH_ANCHORS` |
| The ratio Raman consumes | {func}`~robust.rt.ed.ratio` |
| The packaged table | {func}`~robust.rt.ed.load_table` |
| The override | {attr}`Geometry.Ed <robust.rt.types.Geometry.Ed>` |

Full signatures are on the {doc}`../api` page under *ed*. What the two
inelastic terms do with this sky is {doc}`inelastic` and {doc}`fluorescence`.
