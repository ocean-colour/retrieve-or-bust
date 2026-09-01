# Chlorophyll-a fluorescence

Chlorophyll-a fluorescence is an **emission source**, so unlike Raman it enters
the model additively. In the L23 truth it carries a median ~35 % of $R_{rs}$ at
its 685 nm peak
([`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§1) — which is why an elastic-only model, 0.30 % accurate against elastic truth,
is 48 % wrong at that one wavelength against the real ocean (report §4).

{mod}`robust.rt.inelastic` implements it as
{func}`~robust.rt.inelastic.fluorescence_kernel`, which returns
$K_{\rm fl}(\lambda)$ **with the quantum yield factored out**, so the composed
term is $R_{rs}^{\rm fl} = \varphi_C K_{\rm fl}$ and therefore
$\partial R_{rs}/\partial\varphi_C = K_{\rm fl}$. That is the whole design
intent: φ_C is a physiology handle a future retrieval can recover, not a
constant baked into an emulator at the truth's value.

*Sources for this page: the {mod}`robust.rt.inelastic` module docstring and the
docstrings of {func}`~robust.rt.inelastic.fluorescence_kernel`,
{func}`~robust.rt.inelastic.emission_line` and
{func}`~robust.rt.inelastic.fl_excitation_grid`;
[`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §4.4;
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§§1, 2, 4 and 5; and
[`design/rt_inelastic_implementation.md`](gh:design/rt_inelastic_implementation.md)
§4.3. Numbers labelled "measured" were measured in this environment when the
page was written, on the committed 50-scene fixture; the report's numbers are
labelled as the report's.*

## The φ_C-linear kernel

The kernel is evaluated at an internal reference yield and divided by it:

$$K_{\rm fl}(\lambda) \;=\;
  \frac{R_{rs}^{\rm fl}(\varphi_C = 0.02;\ \lambda)}{0.02},
\qquad
R_{rs}^{\rm fl}(\lambda) = \varphi_C\, K_{\rm fl}(\lambda)$$

{data}`~robust.rt.inelastic.PHI_C_REF` is that 0.02 — HydroLight's default and
therefore the L23 truth's, and `PHI_C_REF == PHI_C_L23` is pinned by a test.
Two things follow. At the truth's yield the composed term equals the fixed BING
implementation *exactly*, which is where the cross-check and all training
happen. Away from it the term is φ_C-linear **by construction**, and the only
neglected nonlinearity is the $(1 - B\,r_{rs})$ surface-transfer denominator,
which is O(10⁻³) at fluorescence amplitudes (design §4.4).

Measured, on one fixture scene at 685 nm:

```text
phi_C=0.01: Rrs(685) = 1.076931e-04   increment/phi_C = 1.482279e-03
phi_C=0.02: Rrs(685) = 1.225158e-04   increment/phi_C = 1.482279e-03
phi_C=0.04: Rrs(685) = 1.521614e-04   increment/phi_C = 1.482279e-03
phi_C=0.10: Rrs(685) = 2.410982e-04   increment/phi_C = 1.482279e-03

dRrs(685)/dphi_C            = 1.482279e-03 sr^-1     (jax.grad)
phi_C * dRrs/dphi_C         = 2.964558e-05
Rrs(phi_C=0.02) - Rrs(0)    = 2.964558e-05           <- identical
```

The increment per unit yield is constant to every printed digit over a factor of
ten in φ_C, and the gradient identity closes exactly. This is the same check the
{doc}`../quickstart` runs; it is repeated here because it is the property the
page is about.

:::{warning}
**Linearity by construction is not linearity verified against truth.** L23
provides fluorescence truth at exactly one quantum yield, φ_C = 0.02. The report
measures that the corrected model's error is identical at 0.5×/1×/2×/5× the
reference yield to < 10⁻⁴ (§4) — but that measures *our* model's linearity, not
the ocean's. Whether the real ocean's fluorescence is φ_C-linear at the ±few-%
level is untested; varied-φ_C HydroLight runs are the design's wishlist item 2
([`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§5, may-not-claim item 2).
:::

## What the kernel assembles

{func}`~robust.rt.inelastic.fluorescence_kernel` takes
`(iops, geometry, wave=None, emission_shape='single')` and returns
$K_{\rm fl}$ in sr⁻¹ per unit φ_C, shape `(*batch, n_wave)`, ≥ 0 and peaked at
685 nm. Term for term, following Gordon (1979) fluorescence-as-inelastic-
scattering in the same Sathyendranath & Platt (1998) two-flow frame as
{doc}`inelastic`:

- **The source**, $b_{bF}(\lambda') = \tfrac{1}{2}\varphi_C a_{\rm ph}(\lambda')$
  — isotropic emission, so half goes backward. **This is where
  {attr}`IOPs.a_ph <robust.rt.types.IOPs.a_ph>` earns its place**: the source is
  physically the light phytoplankton pigments absorbed, and bulk $a$ cannot
  stand in for it.
- **The excitation integral**, a trapezoid over
  {func}`~robust.rt.inelastic.fl_excitation_grid`, carrying the λ′/λ
  quanta-to-energy factor and the true $E_d(\lambda')$ normalized by
  $E_d(\lambda)$ — both from one sky ({doc}`ed`, override honoured, so numerator
  and denominator can never come from different skies).
- **Per-emission-wavelength attenuation**,
  $\kappa_F(\lambda) = (a + b_b)/\mu_F$ with
  {data}`~robust.rt.inelastic.MU_F` = 0.5. Freezing this at 685 nm instead is
  the ~4× error at a 730 nm shoulder that BING's history warns about, which is
  why it is evaluated per λ.
- **$L_u = E_u/\pi$** — the emission is isotropic, so the upwelling radiance is
  the irradiance over π. Its absence made pre-fix BING **~3× too bright** in
  every fluorescence-enabled $R_{rs}$, one of the two headline defects the
  assessment found and fixed upstream (report §1). A sentinel in the
  cross-check guards it.
- **The emission line** $h_C(\lambda)$ and the standard surface transfer
  {func}`~robust.rt.conventions.rrs_to_Rrs` (A = 0.52, B = 1.7).

Unlike Raman, **nothing divides out here**. Every normalization is load-bearing,
which is exactly why fluorescence carried the π lesson and the self-normalizing
Raman ratio never felt it (module docstring).

### The excitation quadrature

{func}`~robust.rt.inelastic.fl_excitation_grid` is a **fixed** grid, not a subset
of the caller's `wave`:

```text
n = 65    370.0 .. 690.0 nm    step 5.0 nm
every node lands on a canonical grid point:  True
```

{data}`~robust.rt.inelastic.FL_EX_MIN` is 370 nm; the upper bound and the step
(690 nm, 5 nm) share a doc comment with it, so autodoc does not emit them
separately. Three things the fixed grid buys, from the function's own docstring:
the quadrature is identical whatever emission grid is requested (a satellite
band set, a slice); shapes stay static under `jit`; and the BING cross-check can
be fed the very same nodes. The 5 nm step is what puts all 65 nodes exactly on
canonical grid points, so on that grid
{func}`~robust.rt.conventions.interp_spectrum` is lossless.

The `(..., n_em, n_ex)` contraction this implies is the design's budgeted cost
(§4.6) and was also the model's one speed pathology: the first end-to-end
measurement was 6.3× the elastic hybrid, traced to the quadrature and an XLA
fusion that re-ran a 52-million-element reduction once per consumer. Fusing the
integral and pinning one materialization with `jax.lax.optimization_barrier`
brought it to **1.59×** with bit-identical outputs (report §4). The comments in
the source say so at the line that does it.

### `a_ph` is required, and says so

```text
>>> fluorescence_kernel(IOPs(a=..., bb_w=..., bb_p=...), geometry, wave)
ValueError: fluorescence_kernel: IOPs.a_ph is None, but the fluorescence
source term is b_F = phi_C * a_ph — bulk absorption cannot stand in for the
phytoplankton component. Provide a_ph (e.g. IOPs.from_total_bb(..., a_ph=...))
or turn the process off with Inelastic(fluorescence=False)
```

A physical requirement, not an API whim. {func}`robust.rt.hybrid.rrs_forward`
also carries a fast pre-check, so the error arrives before the emulator loads.
Callers with only bulk $a$ get elastic + Raman: `Inelastic(fluorescence=False)`.

## `emission_line`, and the two shapes

{func}`~robust.rt.inelastic.emission_line` returns $h_C(\lambda)$ in nm⁻¹,
**unit-normalized in λ**, so it redistributes the emitted energy without
changing its total. Measured on a 0.05 nm grid over 400–1000 nm:

```text
'single': integral = 1.000000    peak 685.00 nm   h = 0.037636 nm^-1
'double': integral = 1.000000    peak 685.20 nm   h = 0.028726 nm^-1
```

| | `'single'` | `'double'` |
| --- | --- | --- |
| Primary line | 685 nm, σ = 10.6 nm, weight 1 | same line, weight 0.75 |
| Secondary | — | {data}`~robust.rt.inelastic.LAMBDA_FL_SECONDARY` = 730 nm, σ = 21.2 nm, weight 0.25 |
| FWHM | 24.96 nm (measured, from σ = 10.6) | 24.96 / 49.92 nm |
| Physics | PS II emission, Gordon (1979) | adds the PS I shoulder — physically better |
| Status | **the validated default** | implemented, **off everywhere in v1** |

The centres and widths are {data}`~robust.rt.inelastic.LAMBDA_FL` /
`SIGMA_FL` and {data}`~robust.rt.inelastic.LAMBDA_FL_SECONDARY` /
`SIGMA_FL_SECONDARY`, with `FL_WEIGHT_PRIMARY` = 0.75 (the σ and weight
constants share doc comments with their centres, so only the centres reach the
API page). {data}`robust.rt.types.EMISSION_SHAPES` is `('single', 'double')`, and
an unknown shape raises rather than silently picking a line:

```text
ValueError: emission_line: shape must be 'single' or 'double'; got 'triple'
```

`emission_shape` is **static metadata** on {class}`robust.rt.types.Inelastic`,
not a differentiable leaf — pass it through
{attr}`Inelastic.emission_shape <robust.rt.types.Inelastic.emission_shape>` as
{func}`robust.rt.hybrid.forward` does.

### Why `'double'` is off, and must be reported as unvalidated

:::{warning}
**`emission_shape='double'` cannot be validated with the data in hand, and the
reports say so twice.** L23's HydroLight runs used a single-Gaussian 685 nm
emission line, so there is no truth channel in which a PS I shoulder exists.
Scored against that truth anyway, `'double'` sits at **−23.6 % at 685 nm** —
consistent with moving 25 % of the emission into a shoulder L23 cannot see, not
with the shape being wrong
([`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§5, may-not-claim item 3; the same judgement is
[`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §4.4's, which
calls it *physically better, unvalidatable against L23*). It is off by default
and off everywhere in v1 training and validation.

Re-measured here on the fixture kernel at 685 nm: `'double'` is
1.090167 × 10⁻³ against `'single'`'s 1.428532 × 10⁻³, i.e. **−23.7 %** — the
report's figure reproduced to 0.1 percentage points on 50 scenes. Validating the
shoulder needs dedicated HydroLight runs or field spectra
([`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §8).
:::

Also worth knowing: the two correction heads were trained with `'single'`, so
switching to `'double'` changes the analytic term underneath a δ_F that never saw
it.

## How it composes into `forward()`

Additively, in $R_{rs}$ space, after Raman has multiplied
([`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §2):

$$R_{rs}^{\rm total}(\lambda) =
  \bigl(R_{rs}^{\rm ZTT} + \Delta R_{rs}\bigr) f_R(\lambda)
  \;+\; \varphi_C\, K_{\rm fl}(\lambda)\,\bigl(1 + \delta_F\bigr)$$

{attr}`Inelastic.phi_C <robust.rt.types.Inelastic.phi_C>` is a **differentiable
leaf** — scalar or per-scene — aligned onto the wavelength axis at composition
time. The order of the multiplication is part of the contract, because float
multiplication is not associative: `phi_C * corrected_fluorescence(δ_F, K_fl)`,
which is what `hybrid._apply_inelastic`,
{func}`~robust.rt.inelastic_corr.corrected_fluorescence`, the gate tests and the
validation script all evaluate — one expression, spelled once.

δ_F's features deliberately **exclude** φ_C, which is what preserves the
linearity above: $\partial R_{rs}/\partial\varphi_C = K_{\rm fl}(1 + \delta_F)$
and not something the network can bend. See {doc}`corrections`.

## What the analytic backbone gets wrong

Quoted from
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§4, per-process fidelity (median error at the 685 nm peak on held-out scenes,
analytic → corrected):

| Fluorescence peak error [%] | 0° | 30° | 60° |
| --- | --- | --- | --- |
| analytic | +0.3 | −5.2 | **−13.7** |
| corrected | +0.08 | +0.07 | +0.10 |

The failure is a **drift, not an offset**: model/truth at 685 nm reads
1.00 / 0.95 / 0.86 across the three zeniths (design §4.3's assembly of the
assessment; the module docstring quotes the same three numbers as the kernel's
known accuracy). The report also records a **trophic** drift the total hides —
across deciles of $a_{\rm ph}(440)$ spanning 0.0016–0.35 m⁻¹, the analytic
term's error runs from −11 % to +11 % while the corrected term stays flat at
≤ 0.62 % (report §4).

## API

| What | Where |
| --- | --- |
| The kernel | {func}`~robust.rt.inelastic.fluorescence_kernel` |
| The excitation quadrature | {func}`~robust.rt.inelastic.fl_excitation_grid`, {data}`~robust.rt.inelastic.FL_EX_MIN` |
| The emission line | {func}`~robust.rt.inelastic.emission_line`, {data}`~robust.rt.inelastic.LAMBDA_FL`, {data}`~robust.rt.inelastic.LAMBDA_FL_SECONDARY` |
| The reference yield | {data}`~robust.rt.inelastic.PHI_C_REF` |
| The upwelling mean cosine | {data}`~robust.rt.inelastic.MU_F` |
| The shapes | {data}`robust.rt.types.EMISSION_SHAPES`, {attr}`Inelastic.emission_shape <robust.rt.types.Inelastic.emission_shape>` |
| The yield, as a leaf | {attr}`Inelastic.phi_C <robust.rt.types.Inelastic.phi_C>` |
| The source term's IOP | {attr}`IOPs.a_ph <robust.rt.types.IOPs.a_ph>` |
| The learned rescaling | {func}`~robust.rt.inelastic_corr.corrected_fluorescence` |

Full signatures are on the {doc}`../api` page under *inelastic*. The
multiplicative sibling term is {doc}`inelastic`; the heads that correct both are
{doc}`corrections`; the protocol that scores them is {doc}`../using/validation`.
