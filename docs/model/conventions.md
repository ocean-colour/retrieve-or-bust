# Conventions

{mod}`robust.rt.conventions` is one file of choices that every other module
assumes and no other module may re-make: the air–water interface conversion,
the wavelength grid, pure-water backscattering, the Raman excitation maps, and
the three boundary validators. It holds no physics of its own — it holds the
things that must be *identical* across every run, model and figure, so that two
numbers computed a week apart are comparable.

Nothing here needs weights, data files, or the ML stack. Everything except the
validators is pure JAX: differentiable, `jit`-safe, batched.

*Sources for this page: the {mod}`robust.rt.conventions` module docstring and
the docstrings of each function named below;
[`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md) §3 (the interface
convention) and §4.1 (the grid);
[`reports/report_rt_elastic_model.md`](gh:reports/report_rt_elastic_model.md)
§3 (the reference data). Every printed number was re-measured in this
environment when the page was written.*

## `Rrs` ↔ `rrs`, with A = 0.52 and B = 1.7

Two reflectances appear throughout: $R_{rs}$, the **above-water**
remote-sensing reflectance an instrument sees, and $r_{rs}$, the **subsurface**
value the radiative transfer is actually written in. They are related by the
Lee et al. (2002) form:

$$
r_{rs} \;=\; \frac{R_{rs}}{A + B\,R_{rs}},
\qquad
R_{rs} \;=\; \frac{A\,r_{rs}}{1 - B\,r_{rs}},
\qquad
A = 0.52,\quad B = 1.7 .
$$

These are {func}`~robust.rt.conventions.Rrs_to_rrs` and
{func}`~robust.rt.conventions.rrs_to_Rrs`, exact inverses of each other, and
the two constants are {data}`~robust.rt.conventions.A_RRS` and `B_RRS`. They
are fixed rather than configurable for a specific reason: they must equal
`bing.rt.A_Rrs` / `bing.rt.B_Rrs`, because two packages disagreeing about what
`rrs` *means* is the failure this pins down. A test asserts the equality rather
than trusting the comment.

Measured here, so the shapes of the two curves are concrete:

```text
rrs_to_Rrs(0.01)  = 5.289929e-03
Rrs_to_rrs(0.005) = 9.460739e-03
Rrs_to_rrs(rrs_to_Rrs(x)) - x  ->  4.66e-10 worst case over x in [1e-3, 5e-2]  (float32)
```

### The conversion is non-linear, and that has consequences

$R_{rs} = A\,r_{rs}/(1 - B\,r_{rs})$ is not additive. Two terms that sum
exactly below the surface do not sum above it:

```text
rrs_to_Rrs(0.010 + 0.004) - [rrs_to_Rrs(0.010) + rrs_to_Rrs(0.004)]  =  +7.33e-05
```

That single fact propagates into two rules used everywhere on this site:

- **Score in $r_{rs}$.** The acceptance metric is rRMS in $r_{rs}$ space
  ([`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md) §6), because a
  relative error in $R_{rs}$ is simply a different number.
- **`mode='emulator'` is not additive with `mode='ztt'` in $R_{rs}$ space**,
  although it is *exactly* additive in $r_{rs}$ space. That is the subject of
  {doc}`forward`, where the arithmetic is shown.

### The pole, and what it catches

$R_{rs} \to \infty$ as $r_{rs} \to 1/B$, and goes *negative* beyond it. That
value is {data}`~robust.rt.conventions.RRS_POLE` = 0.5882…, roughly ten times
the brightest real ocean $r_{rs}$, so it is only reachable through a unit
error — which is precisely why {func}`~robust.rt.conventions.check_rrs` looks
for it and says so in the message.

## The canonical wavelength grid

L23's grid is the package's grid: **350–750 nm in 5 nm steps, 81 points**.

| name | value |
| --- | --- |
| `WAVE_MIN` | 350.0 nm |
| `WAVE_MAX` | 750.0 nm |
| `WAVE_STEP` | 5.0 nm |
| `N_WAVE` | 81 |

{data}`~robust.rt.conventions.WAVE` is the grid as **NumPy**, deliberately not
as a device array: a `jnp` array built at import time would fix its dtype
before a caller can enable float64. {func}`~robust.rt.conventions.canonical_wave`
is the JAX-side accessor and takes an optional `dtype`; with JAX's defaults it
returns `float32`, shape `(81,)`. The values are exact multiples of five, so
float32 represents them without error.

Every function that takes `wave` accepts *any* grid — the model is pointwise in
$\lambda$ and the emulator carries $\lambda$ as a feature, so hyperspectral or
satellite-band grids work. The canonical grid is the default and the one every
number on this site was measured on.

### `check_wave`, and what it is for

{func}`~robust.rt.conventions.check_wave` raises `ValueError` unless its
argument *is* the canonical grid, to 1e-3 nm. It is a **boundary** check: call
it where data enters (a loader, a public constructor), not in the hot path. It
inspects concrete values, so it cannot run inside `jit`, and it is not meant
to. The messages name the discrepancy rather than merely refusing:

```text
>>> check_wave(np.linspace(400, 700, 61))
ValueError: wave: expected the canonical grid of shape (81,), got (61,)

>>> check_wave(WAVE + 0.01)
ValueError: wave: not the canonical 350-750 nm grid; largest difference
            +0.01 nm at index 0 (got 350, expected 350)
```

The second message is worth looking at twice: the difference is 0.01 nm and the
printed values are identical at four significant figures. A grid that is
*almost* right is the case this validator exists for.

Its two siblings are {func}`~robust.rt.conventions.check_iop` (finite and
non-negative) and {func}`~robust.rt.conventions.check_rrs` (finite,
non-negative, and below the pole). All three raise `ValueError` rather than
using bare `assert`, because `python -O` strips `assert` and a silently skipped
convention check is worse than no check at all.

## `bb_w(λ)` — pure-water backscattering

{func}`~robust.rt.conventions.bb_w` returns pure-water backscattering in m⁻¹,
linearly interpolating the 81-point table {data}`~robust.rt.conventions.BB_W_L23`.
It is differentiable in `wave`, safe inside `jit`, and needs no data file.

**Its provenance is the point.** The table is `bb - bbnw` read from the L23
elastic file, so it is not an *approximation* of L23's water model — it **is**
that model. Since the forward model is trained and scored against L23, any
other $b_{bw}$ would put a bias straight into $b_{bp} = b_b - b_{bw}$. Before
being embedded it was verified constant to 1.6e-7 relative across all 3320
scenes, all three solar zeniths, and both the X=1 and X=4 scenarios, so there
is no scene to choose.

Measured on the packaged table:

```text
bb_w(350 nm) = 5.9173e-03 m^-1
bb_w(440 nm) = 2.1956e-03 m^-1
bb_w(750 nm) = 2.3573e-04 m^-1        ratio 350:750 = 25.1x
log-log fit  = lambda^-4.21
```

The fitted exponent is consistent with molecular scattering's $-4.32$ (Morel
1974). Outside 350–750 nm the interpolation **clamps to the end values** rather
than extrapolating — the L23 reference says nothing beyond its own range, and a
constant extrapolation is a documented caveat where a wild one would be a
silent error. Verified: `bb_w(300) == bb_w(350)` and `bb_w(800) == bb_w(750)`.

:::{note}
Why the split is carried at all: {class}`~robust.rt.types.IOPs` keeps `bb_w`
and `bb_p` as separate fields instead of a single `bb`. The ZTT backbone
*requires* it — $\eta_{bb} = b_{bw}/(b_{bp}+b_{bw})$ appears in two of its
terms — and physically water returns ~0.23 sr⁻¹ of its backscatter toward the
sensor at 180° while particles return ~0.12–0.16 sr⁻¹, so the composition of
$b_b$ matters independently of its total
([`reports/report_rt_elastic_model.md`](gh:reports/report_rt_elastic_model.md)
§1). See {doc}`ztt`.
:::

## The Raman excitation maps, and one interpolation rule

Two small maps live here rather than in the inelastic modules, because they are
grid conventions and the inelastic chapters consume them:

$$
\frac{1}{\lambda'} = \frac{1}{\lambda} + \Delta\tilde\nu ,
\qquad
\Delta\tilde\nu = 3400\ \mathrm{cm}^{-1},
$$

with {func}`~robust.rt.conventions.raman_excitation` mapping emission
$\lambda \to \lambda'$ and {func}`~robust.rt.conventions.raman_emission` its
exact inverse. {data}`~robust.rt.conventions.RAMAN_SHIFT` is the single-shift
centre; measured here,

```text
raman_excitation(400 nm) = 352.11 nm      raman_excitation(685 nm) = 555.60 nm
raman_excitation(488 nm) = 418.55 nm      raman_emission(488 nm)   = 585.08 nm
```

{data}`~robust.rt.conventions.RAMAN_WAVE_MIN_OFFICIAL` = 400 nm is the lower
edge of the inelastic model's *supported* band, and the first line above is
why: the excitation for a 400 nm emission is 352.11 nm, barely inside the grid.
Below 400 nm the maps still run — no error, by design — but the excitation
wavelengths leave the grid and the interpolation clamps at 350 nm. The physics
of these maps is {doc}`inelastic`; only the arithmetic is here.

{func}`~robust.rt.conventions.interp_spectrum` is the package's **one**
interpolation rule, shared by the Raman excitation grid and by
{mod}`robust.rt.ed`'s sky so that neither carries a private stride assumption:
linear between grid points, clamped to the end values outside, batched,
`jit`-safe, and differentiable **in the spectrum values** as well as in the
wavelengths. That last property is not incidental — it is what lets gradients
flow from a Raman emission wavelength back through the excitation-grid IOPs to
the IOP inputs.

## API

| What | Where |
| --- | --- |
| Interface conversion | {func}`~robust.rt.conventions.Rrs_to_rrs`, {func}`~robust.rt.conventions.rrs_to_Rrs`, {data}`~robust.rt.conventions.A_RRS`, {data}`~robust.rt.conventions.RRS_POLE` |
| Wavelength grid | {data}`~robust.rt.conventions.WAVE`, {func}`~robust.rt.conventions.canonical_wave` |
| Pure water | {data}`~robust.rt.conventions.BB_W_L23`, {func}`~robust.rt.conventions.bb_w` |
| Raman grid | {data}`~robust.rt.conventions.RAMAN_SHIFT`, {func}`~robust.rt.conventions.raman_excitation`, {func}`~robust.rt.conventions.raman_emission`, {func}`~robust.rt.conventions.interp_spectrum` |
| Validators | {func}`~robust.rt.conventions.check_wave`, {func}`~robust.rt.conventions.check_iop`, {func}`~robust.rt.conventions.check_rrs` |

Full signatures and every remaining constant are on the {doc}`../api` page
under *conventions*.
