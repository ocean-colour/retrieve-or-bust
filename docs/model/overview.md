# The model in one page

Everything `robust.rt` computes is one function — {func}`robust.rt.hybrid.forward`
— and one composition law. This page is that law, term by term, with the module
that owns each term and the API entry that documents it. The chapters listed at
the bottom take the terms one at a time; this page is the map they hang off.

## The composition law

$$
R_{rs}(\lambda) \;=\;
\underbrace{\bigl[\,R_{rs}^{\text{ZTT}}(\lambda) \;+\; \Delta R_{rs}^{\text{emulator}}(\lambda)\,\bigr]}_{\text{elastic}}
\;\times\; f_R(\lambda)
\;+\;
\underbrace{\varphi_C \, K_{\mathrm{fl}}(\lambda)}_{\text{fluorescence}}
$$

with the two learned corrections entering as relative rescalings of the analytic
inelastic terms — never of the elastic model:

$$
f_R = 1 + \bigl(f_{\text{phys}} - 1\bigr)\,\bigl(1 + \delta_R\bigr),
\qquad
\varphi_C K_{\mathrm{fl}} \;\rightarrow\; \varphi_C K_{\mathrm{fl}}\,\bigl(1 + \delta_F\bigr).
$$

Both correction forms are chosen so that an **untrained or absent head is the
analytic physics exactly**: $\delta_R = 0$ leaves $f_R = f_{\text{phys}}$, and
$\delta_F = 0$ leaves the kernel alone. The heads are not free to invent a
signal; they can only rescale one the physics already produced.

Read the law left to right and it is the pipeline:

`elastic backbone` → `learned elastic residual` → `× Raman` → `+ fluorescence`.

### The terms

**$R_{rs}^{\text{ZTT}}$ — the analytic backbone.** The Twardowski & Tonizzo
(2018) model transcribed into JAX, carrying the backward volume scattering
function *explicitly* rather than burying it in coefficients fitted under one
prescribed phase function. This is the physically interpretable half of the
model, and the only part that runs without the ML stack or any weights file.
Owned by {mod}`robust.rt.ztt`.

**$\Delta R_{rs}^{\text{emulator}}$ — the learned elastic residual.** A small
Flax MLP (417 parameters) predicting the *relative* residual
$\Delta r_{rs} = \delta \cdot r_{rs}^{\text{ZTT}}$ — the multiple-scattering and
phase-function effects the backbone misses. Because it is parameterised
relatively, `mode='emulator'` returns a **term, not a model**. Owned by
{mod}`robust.rt.emulator`.

**$f_R$ — the Raman factor.** Multiplicative and self-normalising: the
Sathyendranath & Platt (1998) two-flow reflectances with the Bartlett et al.
(1998) Raman scattering coefficient, excitation wavelengths set by the single
3400 cm⁻¹ wavenumber shift, and the true solar ratio
$E_d(\lambda')/E_d(\lambda)$ from a packaged sky. Owned by
{mod}`robust.rt.inelastic`; the sky it needs comes from {mod}`robust.rt.ed`.

**$\varphi_C K_{\mathrm{fl}}$ — chlorophyll-a fluorescence.** An additive
Gordon (1979)-style emission integral written *per unit quantum yield*, so
$K_{\mathrm{fl}}$ is independent of $\varphi_C$ and the model is
**$\varphi_C$-linear by construction**. That is a deliberate inversion
affordance: $\varphi_C$ stays a differentiable leaf rather than being baked into
a network at the training value. Owned by {mod}`robust.rt.inelastic`; it
requires `IOPs.a_ph`, because the source term is
$b_{bF} = \tfrac{1}{2}\varphi_C\,a_{ph}(\lambda')$ and bulk absorption cannot
stand in for the phytoplankton component.

**$\delta_R, \delta_F$ — the correction heads.** Two bounded tanh MLPs of 129
parameters each, trained on the L23 scenario *differences* rather than on
$R_{rs}$ itself. $\delta_F$ is deliberately blind to $\varphi_C$ so that it
cannot break the kernel's linearity. Owned by {mod}`robust.rt.inelastic_corr`.

### Which space the arithmetic happens in

The composition law above is written in $R_{rs}$ space, which is where the
inelastic terms are defined and where the model is composed. **Additivity of the
elastic parts, however, holds only in $r_{rs}$ space**:
`rrs_forward(hybrid) == rrs_forward(ztt) + rrs_forward(emulator)` exactly, while
the same identity in $R_{rs}$ fails, because the air–water interface conversion
$R_{rs} = A\,r_{rs} / (1 - B\,r_{rs})$ is non-linear. Use
{func}`robust.rt.hybrid.rrs_forward` when the parts must sum and for all
scoring; use {func}`robust.rt.hybrid.forward` for the above-water quantity an
instrument would see.

## The three modes, and the two guarantees

{data}`robust.rt.hybrid.MODES` is `('ztt', 'emulator', 'hybrid')`: the analytic
backbone alone, the learned correction $\Delta r_{rs}$ alone, and their sum,
which is the default and the model every accuracy number on this site refers to.
Having all three behind one signature is what lets them be scored on identical
splits rather than on separately prepared data.

Two properties are worth knowing before you use any of them, and both have their
full treatment — with the measured evidence — on {doc}`forward`:

- **`'ztt'` and `'emulator'` sum in $r_{rs}$ space and do *not* sum in $R_{rs}$
  space**, because the air–water interface is non-linear. Reconstructing the
  hybrid by adding two `forward()` outputs is wrong by about the size of a
  model-comparison margin.
- **`inelastic=None` — the default — is the elastic hybrid bit for bit**, not
  approximately: the `None` branch returns the elastic result untouched rather
  than composing terms that evaluate to zero, and a test pins the SHA-256. So
  adding the inelastic terms disturbed no number the elastic report claims.

Any mode that involves the emulator also checks its inputs against the trained
domain and warns (`DomainWarning`) outside it; `on_out_of_domain='ztt'`
additionally zeroes the learned correction there, degrading to the analytic
backbone exactly where the emulator was measured to be unreliable. The warning
needs concrete values, so it is skipped under `jit` — deliberately, and
documented on the function rather than silently.

`corrections` is never even resolved when `inelastic` is `None` or all processes
are off, so the elastic path owes nothing to the ML stack.

:::{note}
A third inelastic term, **CDOM fluorescence** ({mod}`robust.rt.cdom_fl`), exists
in the package. It is **off by default** (`Inelastic(cdom_fl=None)`),
analytic-only (its $\delta_C$ head is defined but untrained), and **unvalidated**
— the L23 X4 truth omits CDOM fluorescence, so there is nothing yet to score it
against. It is therefore not part of the law above. See the package docstring on
the {doc}`../api` page.
:::

## Concept → module → API

| Concept | Module | API |
| --- | --- | --- |
| Wavelength grid, `Rrs` ↔ `rrs`, `bb_w(λ)` | {mod}`robust.rt.conventions` | {func}`~robust.rt.conventions.rrs_to_Rrs`, {func}`~robust.rt.conventions.canonical_wave`, {func}`~robust.rt.conventions.bb_w` |
| The inputs, as JAX pytrees | {mod}`robust.rt.types` | {class}`~robust.rt.types.IOPs`, {class}`~robust.rt.types.PhaseParams`, {class}`~robust.rt.types.Geometry` |
| $R_{rs}^{\text{ZTT}}$ — analytic backbone | {mod}`robust.rt.ztt` | {func}`~robust.rt.ztt.rrs_ZTT`, {func}`~robust.rt.ztt.Rrs_ZTT` |
| $\Delta R_{rs}$ — learned elastic residual | {mod}`robust.rt.emulator` | {class}`~robust.rt.emulator.Emulator`, {func}`~robust.rt.emulator.load_default` |
| The composition, `mode`, the domain guard | {mod}`robust.rt.hybrid` | {func}`~robust.rt.hybrid.forward`, {func}`~robust.rt.hybrid.rrs_forward`, {data}`~robust.rt.hybrid.MODES` |
| $E_d(\theta_s, \lambda)$ — the sky | {mod}`robust.rt.ed` | {func}`~robust.rt.ed.Ed`, {func}`~robust.rt.ed.ratio` |
| $f_{\text{phys}}$ — analytic Raman | {mod}`robust.rt.inelastic` | {func}`~robust.rt.inelastic.raman_factor`, {func}`~robust.rt.inelastic.raman_bb` |
| $K_{\mathrm{fl}}$ — fluorescence kernel | {mod}`robust.rt.inelastic` | {func}`~robust.rt.inelastic.fluorescence_kernel`, {func}`~robust.rt.inelastic.emission_line` |
| $\varphi_C$ and the process switches | {mod}`robust.rt.types` | {class}`~robust.rt.types.Inelastic` |
| $\delta_R$, $\delta_F$ — the learned heads | {mod}`robust.rt.inelastic_corr` | {func}`~robust.rt.inelastic_corr.corrected_raman_factor`, {func}`~robust.rt.inelastic_corr.corrected_fluorescence`, {class}`~robust.rt.inelastic_corr.CorrectionHeads` |
| What the model must beat | {mod}`robust.rt.baselines` | {func}`~robust.rt.baselines.rrs_gordon`, {func}`~robust.rt.baselines.rrs_o25` |
| The reference data (L23) | {mod}`robust.rt.data.l23` | {func}`~robust.rt.data.l23.load_batch`, {func}`~robust.rt.data.l23.make_splits` |
| The scoring protocol | {mod}`robust.rt.validation` | {func}`~robust.rt.validation.rrms`, {func}`~robust.rt.validation.score_models` |

## The chapters

```{toctree}
:maxdepth: 1

conventions
ztt
emulator
forward
ed
inelastic
fluorescence
corrections
baselines
```
