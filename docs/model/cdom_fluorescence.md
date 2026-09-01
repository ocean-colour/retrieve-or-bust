# CDOM fluorescence

Coloured dissolved organic matter absorbs in the ultraviolet and blue and
re-emits, broadly and without a line, across the blue-green. It is the **third
inelastic process** in `robust.rt`, and the newest: {mod}`robust.rt.cdom_fl`
ships the analytic emission term $K_{\rm cdom}$, {class}`robust.rt.types.CDOMFl`
carries its amplitude, and {func}`robust.rt.hybrid.forward` composes it
additively — but only when you ask, because the term is **off by default** and
stays off in everything this site reports.

:::{warning}
**This term is shipped and gate-passed, and it is not validated.** No truth
channel for CDOM fluorescence exists anywhere in hand: the Loisel et al. (2023)
X4 release omits the process by design, and BING never implemented it, so there
is nothing to score the term against and nothing to train its correction head
on. What has been checked is *implementation correctness* — the published
function reproduced at its own tabulated values, quadrature convergence, the
clamp proved at the seams, eight-variable gradients, bit-identity when the term
is off — plus a **loose literature-plausibility band**, which the design calls
"reported and gated loosely" and which is explicitly *not* a validation.

Until the HydroLight "X4 vs X4 + CDOM-fl" runs of the design's §7 exist and
milestone **M6** trains and scores the term, the honest description is
*Hawes-consistent and plausible*, never *accurate*. **Unvalidated until M6** is
the phrase the implementation record, the `robust.rt` package docstring and this
site all use, and it means what it says: no error bar of any kind attaches to
$K_{\rm cdom}$ anywhere. The same status is stated on
{doc}`../using/limitations`, {doc}`overview` and the {doc}`../api` page.
:::

*Sources for this page: the {mod}`robust.rt.cdom_fl` module docstring and the
docstrings of {func}`~robust.rt.cdom_fl.eta_hawes`,
{func}`~robust.rt.cdom_fl.cdom_kernel`,
{func}`~robust.rt.cdom_fl.cdom_excitation_grid` and
{func}`~robust.rt.cdom_fl.truncated_excitation_fraction`;
[`design/rt_cdom_fluorescence_model.md`](gh:design/rt_cdom_fluorescence_model.md)
§§1–5 and §8; and
[`design/rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md)
§8, the measured M5 record. Numbers labelled "measured" were measured in this
environment when the page was written — on the committed 50-scene inelastic
fixture (150 samples: 50 scenes × three zeniths) where the page says fixture,
and on the full 9,960-sample L23 inelastic release where it says release.
Neither report covers this term: both predate it.*

## The Hawes kernel, and where its two halves come from

The redistribution function is the **Hawes et al. (1992)** spectral fluorescence
quantum efficiency — HydroLight's own default CDOM-fluorescence choice, picked so
that the truth runs of the design's §7, when they exist, share constants with
this kernel by construction. The functional form is taken from Zhai, Hu, Lee et
al. (2017), *Opt. Express* 25(8), Eqs. (7)–(8):

$$\eta_Y(\lambda, \lambda_e) \;=\; g_Y(\lambda_e)\,A_0(\lambda_e)\,
\exp\!\left\{-\left[\frac{1/\lambda - A_1/\lambda_e - B_1}
{0.6\,(A_2/\lambda_e + B_2)}\right]^{2}\right\},
\qquad
g_Y(\lambda_e) = \begin{cases}1 & 310 \le \lambda_e \le 490\ \text{nm}\\
0 & \text{otherwise}\end{cases}$$

Read it literally: **the Gaussian argument is in reciprocal wavelength**. For
each excitation wavelength the emission is a Gaussian in *wavenumber*, centred at
$A_1/\lambda_e + B_1$ with width $0.6\,(A_2/\lambda_e + B_2)$, and both the centre
and the width move with $\lambda_e$. $\eta_Y$ is therefore **not separable** into
an excitation factor times an emission shape, which is the one structural
difference from the chlorophyll kernel and has a real consequence in the code
(below). Its units are nm⁻¹ — a per-nm emission density — so $\eta_Y$ plays the
role $\varphi_C h_C(\lambda)$ plays for {doc}`fluorescence`, jointly rather than
as two factors.

The constants are the Hawes Station FA7 fulvic-acid fit (Gulf of Mexico, West
Florida Shelf), {data}`~robust.rt.cdom_fl.HAWES_A0` tabulated at
{data}`~robust.rt.cdom_fl.HAWES_A0_WAVE` and linearly interpolated between its
ten nodes:

| | value |
| --- | --- |
| {data}`~robust.rt.cdom_fl.HAWES_A1` / {data}`~robust.rt.cdom_fl.HAWES_B1` (centre) | 0.470 (dimensionless) / 8.077 × 10⁻⁴ nm⁻¹ |
| {data}`~robust.rt.cdom_fl.HAWES_A2` / {data}`~robust.rt.cdom_fl.HAWES_B2` (width) | 0.407 (dimensionless) / −4.57 × 10⁻⁴ nm⁻¹ |
| {data}`~robust.rt.cdom_fl.HAWES_A0` at 310…490 nm, ×10⁻⁵ nm⁻¹ | 5.81, 6.34, 8.00, 9.89, 9.39, 10.48, 12.59, 13.48, 13.61, 9.24 |
| {data}`~robust.rt.cdom_fl.GY_EX_MIN` / {data}`~robust.rt.cdom_fl.GY_EX_MAX` | 310 / 490 nm |

:::{note}
**The two halves have different provenance, and the module says so.** The
*functional form* — Zhai et al. Eqs. (5)–(8) — was extracted verbatim from the
published paper, and Zhai et al. impose the same 350 nm excitation floor this
model does, for the same stated reason. The *FA7 numeric constants* were sourced
from Mobley's Ocean Optics Web Book and **accepted as-sourced by JXP without
independent primary-source verification** (the CDOM coding doc's Q&A CQ2,
2026-08-30). That is a provenance statement, not a peer-review claim: no
cross-check against Hawes (1992), Proc. SPIE 1750, or *Light and Water* §5.15 was
performed. Every number below inherits it, and re-pins with it if the constants
are ever corrected. The module docstring also records that Zhai et al.'s
reference list prints the middle author's initials as "C. K. Carder" where the
ocean-optics literature has Kendall L. Carder — noted rather than silently fixed.
:::

Measured here, one row per excitation wavelength: the emission peak sits exactly
where $1/(A_1/\lambda_e + B_1)$ says it should, and the peak height reproduces
$A_0(\lambda_e)$ — the test
`test_a0_table_reproduced_at_the_gaussian_peak` pins the second identity.

```text
lam_e   peak emission   analytic 1/(A1/lam_e+B1)   eta at peak   A0(lam_e)
350 nm      465.0 nm            465.0 nm           8.000e-05    8.000e-05
390 nm      496.8 nm            496.8 nm           9.390e-05    9.390e-05
430 nm      526.1 nm            526.1 nm           1.259e-04    1.259e-04
470 nm      553.2 nm            553.2 nm           1.361e-04    1.361e-04
490 nm      566.0 nm            566.0 nm           9.240e-05    9.240e-05
```

The emission peak is red of the excitation for every admissible $\lambda_e$, but
the Gaussian-in-wavenumber form does **not** enforce a strict Stokes shift: the
blue tail is genuinely non-zero — measured, $\eta_Y(\lambda_e, \lambda_e)$ is
6.2 % of the peak at $\lambda_e$ = 350 nm rising to 22.5 % at 490 nm.
`test_emission_peak_is_red_shifted` pins the peak's red shift and bounds the tail
as subdominant (below 30 % of the peak); nothing asserts it away.

## What the kernel assembles

{func}`~robust.rt.cdom_fl.cdom_kernel` takes `(iops, geometry, wave=None,
wave_ex=None)` and returns $K_{\rm cdom}$ in sr⁻¹ **at unit amplitude**, shape
`(*batch, n_wave)` and ≥ 0. It is the {doc}`fluorescence` kernel's
Sathyendranath & Platt (1998) machinery term for term, with $\eta_Y$ in place of
the separable emission line:

- **The source**, $b_{bY}(\lambda_e) = \tfrac{1}{2}a_{\rm cdom}(\lambda_e)$ —
  isotropic emission, half backward. This is where
  {attr}`IOPs.a_cdom <robust.rt.types.IOPs.a_cdom>` earns its place: the source
  is physically the light CDOM absorbed, and bulk $a$ cannot stand in for it.
- **No reference-yield division.** Chl-fl divides by
  {data}`~robust.rt.inelastic.PHI_C_REF` because the yield is a physical handle;
  here the handle is {attr}`CDOMFl.scale <robust.rt.types.CDOMFl.scale>`, applied
  by the composition rather than inside the kernel, so this function returns the
  raw kernel at $s_C = 1$.
- **The excitation integral**, a trapezoid over
  {func}`~robust.rt.cdom_fl.cdom_excitation_grid`, carrying the
  $\lambda_e/\lambda$ quanta-to-energy factor and the true $E_d(\lambda_e)$
  normalised by $E_d(\lambda)$ — both from one sky ({doc}`ed`, override
  honoured), and $\eta_Y$ *inside* the sum.
- **Attenuation** $K(\lambda_e) = (a + b_b)/\mu_D$ downwelling at excitation and
  $\kappa_Y(\lambda) = (a + b_b)/\mu_F$ upwelling at emission, per emission
  wavelength, with the same {data}`~robust.rt.inelastic.MU_F` = 0.5 as
  chlorophyll fluorescence.
- **$L_u = E_u/\pi$** and {func}`~robust.rt.conventions.rrs_to_Rrs` (A = 0.52,
  B = 1.7) — the same two calls in the same order as Chl-fl, so the term arrives
  above the surface.

**The structural departure, and its cost.** Chl-fl's emission line depends on
$\lambda$ only, so it post-multiplies a reduced `(..., n_ex)` sum. $\eta_Y$
depends on both axes, so it multiplies the `(..., n_em, n_ex)` integrand *before*
the reduction — one extra elementwise multiply by a batch-free `(n_em, n_ex)`
matrix. In exchange the contraction is 29 excitation nodes against Chl-fl's 65.
As with Chl-fl, one `jax.lax.optimization_barrier` pins the reduced result so
XLA's consumer fusion cannot re-run the reduction downstream.

Measured on the fixture, the shape is what the physics advertises — broad,
featureless, no 685 nm-style line — with the batch-mean kernel peaking at 500 nm:

```text
mean K_cdom(lambda) over the 150-sample fixture, as a fraction of its own peak
400 nm 0.18   450 nm 0.56   500 nm 1.00   550 nm 0.62
600 nm 0.13   650 nm 0.04   700 nm 0.01
```

### The excitation quadrature is the clamp

```text
n = 29    350.0 .. 490.0 nm    step 5.0 nm
every node lands on a canonical grid point:  True
```

{data}`~robust.rt.cdom_fl.CDOM_EX_MIN` is 350 nm and
{data}`~robust.rt.cdom_fl.CDOM_EX_MAX` is 490 nm — $g_Y$'s own red cutoff. The
blue edge is the **hard 350 nm clamp**: $g_Y$ nominally admits excitation down to
310 nm, but the L23/IOP/$E_d$ grids start at 350 nm, so the grid simply *starts*
there and the production kernel **provably never reads IOPs or $E_d$ below
350 nm** — no clamping arithmetic, a structural guarantee, pinned at the seams by
a spy test (`test_kernel_never_reads_iops_or_ed_below_350`) that records every
wavelength the kernel interpolates at. The 5 nm step is the canonical grid
spacing, so all 29 nodes land on canonical grid points and the excitation IOPs
interpolate losslessly — the same convention as
{func}`~robust.rt.inelastic.fl_excitation_grid`.

## The clamp's cost, quantified

:::{warning}
**The 350 nm clamp throws away most of the Hawes function's nominal emission in
the violet-blue.** {func}`~robust.rt.cdom_fl.truncated_excitation_fraction` is the
committed diagnostic that says how much:

$$\text{fraction}(\lambda) \;=\;
\frac{\int_{310}^{350}\eta_Y(\lambda,\lambda_e)\,{\rm d}\lambda_e}
     {\int_{310}^{490}\eta_Y(\lambda,\lambda_e)\,{\rm d}\lambda_e}$$

Re-measured for this page on the canonical grid (trapezoid at 0.25 nm; agrees
with a 2× refinement to 3.5 × 10⁻⁶ relative), and identical to the pinned table
in
[`design/rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md)
§8.5 at every wavelength:

| λ_em (nm) | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 750 |
|---|---|---|---|---|---|---|---|---|---|
| truncated fraction | 0.846 | 0.566 | 0.297 | 0.142 | 0.083 | 0.070 | 0.078 | 0.103 | 0.146 |

**57 % of the nominal 310–490 nm-excited emission at 400 nm is excluded by the
clamp** — 85 % at 350 nm, 30 % at 450 nm, a minimum of 7.0 % at 605 nm, rising
again to 15 % at 750 nm through the sub-350 nm excitation Gaussians' red tails.
This is the design's own §8 risk ("if the truncated fraction turns out large in
the blue emission bands, the caveat hardens into a wishlist dependency") coming
true in the violet-blue.

Two things it is not. It is a property of $\eta_Y$ **alone** — no IOPs, no
$E_d$, no scene — so the *realised* truncation in an $R_{rs}$ spectrum is further
weighted by $a_{\rm cdom}(\lambda_e)E_d(\lambda_e)$, and sub-350 nm surface $E_d$
is strongly suppressed by ozone and the solar spectrum, which is Zhai et al.'s
own reason for the identical clamp. So the table is the conservative,
scene-free flavour of the caveat rather than a measured $R_{rs}$ error. And it
inherits the FA7 constants' provenance above. Lifting the clamp needs
excitation-side IOPs below 350 nm, which no data in hand provides.
:::

## `a_cdom` is required, and says so

```text
>>> cdom_kernel(IOPs(a=..., bb_w=..., bb_p=...), geometry, wave)
ValueError: cdom_kernel: IOPs.a_cdom is None, but the CDOM-fluorescence source
term is b_Y = 0.5 * a_cdom -- bulk absorption cannot stand in for the CDOM
component. Provide a_cdom (e.g. IOPs.from_total_bb(..., a_cdom=...); the L23
loaders extract it from `ag`) or leave the process off with
Inelastic(cdom_fl=None)
```

{func}`robust.rt.hybrid.forward` carries the same guard as a fast pre-check, so
the error arrives before the emulator loads — exactly the arrangement
`Inelastic.fluorescence` has with `IOPs.a_ph` ({doc}`forward`).
{attr}`IOPs.a_cdom <robust.rt.types.IOPs.a_cdom>` is optional, validated like
`a_ph` (shape-matched, non-negative, and bounded by the total $a$), and the L23
loaders populate it from the release's `ag` field, so a batch from
{func}`~robust.rt.data.l23.load_inelastic_batch` is ready to use
({doc}`../using/data`).

## How it composes into `forward()`

Additively, in $R_{rs}$ space, after Raman has multiplied and beside the
chlorophyll term:

$$R_{rs}^{\rm total}(\lambda) =
  \bigl(R_{rs}^{\rm ZTT} + \Delta R_{rs}\bigr) f_R(\lambda)
  \;+\; \varphi_C K_{\rm fl}(\lambda)\bigl(1 + \delta_F\bigr)
  \;+\; s_C\,K_{\rm cdom}(\lambda)$$

$s_C$ is {attr}`CDOMFl.scale <robust.rt.types.CDOMFl.scale>` — a differentiable
pytree leaf, scalar or per-scene, default 1.0, aligned onto the wavelength axis
at composition time. {class}`~robust.rt.types.CDOMFl` is a pytree rather than a
bare float so that M6 can grow shape metadata without an API break; today `scale`
is its only field. It is the $\varphi_C$-analogue handle a future retrieval would
recover, and `validate()` refuses a non-positive value with a message pointing at
`Inelastic(cdom_fl=None)` rather than `scale=0` — disabling a process and setting
its amplitude to zero are different statements.

Note what the law does **not** contain: there is no $(1 + \delta_C)$ factor. The
head exists but is not wired; see below.

### Off by default, and bit-identical when off

`None` is the default **inside `Inelastic()` itself**, and that is load-bearing
rather than tidy: the X4 truth omits CDOM fluorescence, so the inelastic report's
0.34 % rRMS gate — and every accuracy number on this site — is valid only while
the default model is provably CDOM-fl-free. The guarantee is the same kind the
elastic path has: the term is skipped **by construction**, not by multiplying in
a computed zero.

Measured here on the committed fixture, hashing the `forward()` output bytes:

```text
Inelastic()               sha256 0dd365158e3037261ee061777fe51da8fa132d4f0972792ad068b9c73641291a
Inelastic(cdom_fl=None)   sha256 0dd365158e3037261ee061777fe51da8fa132d4f0972792ad068b9c73641291a
bit-identical: True
```

That hash is not a fresh one: it is `PRE_CDOM_SHA256_RRS_ABOVE`, the pin taken in
`robust/tests/test_inelastic_types.py` on the **unmodified** code *before*
`hybrid.py` was ever touched, and reproduced above after the wiring landed. The
pin is a two-tier pair — a strict SHA-256 tier that runs on its anchor machine and
a rtol 5 × 10⁻⁷ closeness tier that runs everywhere — for the reasons
{doc}`../using/limitations` gives.

With the term on, the composition is additive to float precision. Measured with
`CDOMFl(scale=2.0)` — deliberately ≠ 1, so a dropped amplitude cannot pass:

```text
max | forward(default + cdom) / (forward(default) + 2 * K_cdom) - 1 |  =  2.980e-07
```

One more thing worth knowing, because it was a real bug found by wiring the term:
`cdom_fl` **counts as an active process on its own**. Before the fix,
`Inelastic(raman=False, fluorescence=False, cdom_fl=CDOMFl())` returned the
untouched elastic result — a plausible-looking spectrum with the requested
physics missing. Measured today, that configuration differs from the elastic
model by up to 9.3 % and equals `elastic + K_cdom` to 2.4 × 10⁻⁷ relative.

## δ_C — defined, zero-initialised, not wired

The learned-correction machinery of {doc}`corrections` has a third slot.
{data}`~robust.rt.inelastic_corr.CDOM_FEATURES` is
`FL_FEATURES` with the chlorophyll handle swapped for the CDOM one —
`log10_a_cdom440`, `log10_a_em`, `log10_bb_em`, `log10_a_490`, `cos_theta_s`,
`wave`, with **no `scale` column**, the same rule that keeps $\varphi_C$ out of
$\delta_F$ — and {func}`~robust.rt.inelastic_corr.corrected_cdom` is the
`corrected_fluorescence` twin, $K_{\rm cdom}(1 + \delta_C)$.

**None of it runs.** Stated plainly, because it is load-bearing:

- {func}`~robust.rt.inelastic_corr.train_cdom_corr` raises
  `NotImplementedError` naming M6 and the missing HydroLight truth. There is
  nothing to train on, so no weights file exists.
- {func}`~robust.rt.inelastic_corr.load_default` looks for no CDOM weights.
  Measured: `heads.cdom` is `None` on a default
  {class}`~robust.rt.inelastic_corr.CorrectionHeads`.
- `hybrid._apply_inelastic` never consults the slot. The shipped term is
  $s_C K_{\rm cdom}$, which is bit-for-bit what a zero-initialised head would
  produce — measured: a fresh `init_head('cdom')` gives $\delta_C \equiv 0$
  exactly, and `corrected_cdom(0, K) == K` byte for byte.
- Its `delta_max` bound is 0.5, and unlike the Raman and fluorescence heads'
  bounds that number is an **arbitrary placeholder**, not a measured-error
  envelope. It is documented as such on
  {class}`~robust.rt.inelastic_corr.HeadConfig`.

The slot exists so that wiring it at M6 is an API no-op, not because anything
learned is in it today.

## What has been checked, and what that is worth

The M5 acceptance gate is truth-less by necessity
([`design/rt_cdom_fluorescence_model.md`](gh:design/rt_cdom_fluorescence_model.md)
§5). All five items pass; here is what each one actually establishes.

| Gate item | Result | What it shows |
| --- | --- | --- |
| 1. Off-state bit-identity | passes, two-tier | the default model is unchanged — an *elastic-and-inelastic* claim, not a CDOM one |
| 2. Implementation-correctness pins | pass | $A_0$ reproduced at each tabulated node's own peak; $\eta_Y \ge 0$ and $g_Y$-gated; red-shifted peaks; excitation quadrature converged (5 nm vs 2.5 nm, max 5.6 × 10⁻³) |
| 3. Literature plausibility | passes, **loosely** | the term's size is in the published ballpark and orders sensibly with CDOM — *not* an accuracy statement |
| 4. Gradients | pass | ≤ 2.2 × 10⁻⁸ over all eight variables against a 10⁻⁶ tolerance ($a_{\rm cdom}$ 1.4 × 10⁻⁸, `scale` 9.8 × 10⁻¹⁰), record §8.8 |
| 5. Speed | passes a **rescoped** bar | 2.31× the elastic hybrid median against a 2.6× budget that is machine-anchored, not portable — record §8.8 |

The plausibility line is the only one that touches the physics, so it is worth
seeing. $K_{\rm cdom}$ at unit scale, as a fraction of the elastic hybrid
$R_{rs}$, averaged over 440–500 nm and stratified into deciles of
$a_{\rm cdom}(440)$. Re-derived for this page on the full 9,960-sample release
and reproduced exactly: both rows are the values pinned in
`test_cdom_gate_3_plausibility_band`'s docstring, and the second rounds to
record §8.8's percentages.

| decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_cdom(440) median [m⁻¹] | 0.0016 | 0.0027 | 0.0037 | 0.0048 | 0.0059 | 0.0074 | 0.0095 | 0.0125 | 0.0183 | 0.0381 |
| mean fraction | 0.00297 | 0.00474 | 0.00632 | 0.00781 | 0.00950 | 0.01145 | 0.01401 | 0.01737 | 0.02304 | 0.04162 |

So 0.30 % of $R_{rs}$ in the blue-green for the least coloured decile and 4.16 %
for the most, strictly increasing across all ten — which is what the gate asserts,
being stronger than a top-versus-bottom comparison — and stable in geometry
(top-decile means 4.20 / 4.16 / 4.12 % at 0° / 30° / 60°, measured). That lands
inside the literature's "a few percent in the blue-green for CDOM-rich water,
≲ 1 % oligotrophic", which is exactly as much as it claims: the gate bands are
0.3–15 % at the top and ≤ 1.2 % at the bottom, deliberately loose because there
is no truth to tighten them against.

The speed bar is **machine-anchored in the same sense as the strict hash pins**:
2.6× characterises the measured behaviour of the Mac this milestone ran on, not a
portable physical requirement, and most of the overage is baseline drift on that
machine rather than CDOM — the shipped Raman + Chl-fl model alone measured ~1.9×
there against the 1.59× its own acceptance recorded on the reference machine,
and the CDOM marginal is ~0.3–0.4× elastic. The shipped model's own 2× gate,
{data}`~robust.rt.validation.INELASTIC_GATE_SPEED`, is untouched.

Gradients go through {func}`~robust.rt.validation.cdom_gradient_report` with its
own {data}`~robust.rt.validation.CDOM_FD_STEPS` — a separate report rather than an
extension of the inelastic one, for the reason {doc}`../using/validation` gives.

## What would change this page

M6, and nothing short of it: paired HydroLight "X4 vs X4 + CDOM-fl" runs on the
L23 ensemble, with the CDOM-rich tail oversampled, at all three zeniths, over
350–750 nm, and with the exact Hawes function and constants recorded on the truth
side so they match this kernel's
([`design/rt_cdom_fluorescence_model.md`](gh:design/rt_cdom_fluorescence_model.md)
§7). With that in hand, $\delta_C$ trains on the difference channel and the term
is gated at the same bar as the other two — median absolute error of the CDOM-fl
delta ≤ 5 % on held-out scenes at every zenith, with the total-$R_{rs}$ rRMS gate
re-verified with the term on. Those runs do not exist, there is no prompt document
for M6, and until they land nothing on this site will report an accuracy for this
term.

The milestone that built the term is written up in
[`design/rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md)
§8 and shown live in
[`notebooks/RT/rt_cdom_coding_1.ipynb`](gh:notebooks/RT/rt_cdom_coding_1.ipynb).

## API

| What | Where |
| --- | --- |
| The kernel | {func}`~robust.rt.cdom_fl.cdom_kernel` |
| The Hawes function | {func}`~robust.rt.cdom_fl.eta_hawes` |
| The FA7 constants | {data}`~robust.rt.cdom_fl.HAWES_A0_WAVE`, {data}`~robust.rt.cdom_fl.HAWES_A0`, {data}`~robust.rt.cdom_fl.HAWES_A1`, {data}`~robust.rt.cdom_fl.HAWES_A2` |
| The excitation quadrature and the clamp | {func}`~robust.rt.cdom_fl.cdom_excitation_grid`, {data}`~robust.rt.cdom_fl.CDOM_EX_MIN`, {data}`~robust.rt.cdom_fl.GY_EX_MIN` |
| The clamp's cost | {func}`~robust.rt.cdom_fl.truncated_excitation_fraction` |
| The amplitude, as a leaf | {class}`robust.rt.types.CDOMFl`, {attr}`CDOMFl.scale <robust.rt.types.CDOMFl.scale>` |
| The switch | {attr}`Inelastic.cdom_fl <robust.rt.types.Inelastic.cdom_fl>` |
| The source term's IOP | {attr}`IOPs.a_cdom <robust.rt.types.IOPs.a_cdom>` |
| The reserved head | {data}`~robust.rt.inelastic_corr.CDOM_FEATURES`, {func}`~robust.rt.inelastic_corr.corrected_cdom`, {func}`~robust.rt.inelastic_corr.train_cdom_corr` |
| The gradient report | {func}`~robust.rt.validation.cdom_gradient_report`, {data}`~robust.rt.validation.CDOM_FD_STEPS` |

Full signatures are on the {doc}`../api` page under *cdom_fl*. The sibling
emission term is {doc}`fluorescence`; the multiplicative one is {doc}`inelastic`;
the heads are {doc}`corrections`; the scope boundary is
{doc}`../using/limitations`.
