# The ZTT backbone

{mod}`robust.rt.ztt` is the analytic half of the forward model: the
Zaneveld–Twardowski–Tonizzo relation of Twardowski & Tonizzo (2018) transcribed
into JAX, function by function, each naming the paper equation it implements.

It is the backbone rather than one analytic model among several for one reason:
**the backward volume scattering function enters explicitly** instead of being
absorbed into coefficients fitted under a single prescribed phase function.
That gives the hybrid an interpretable handle on phase-function *shape* that
the Gordon family structurally lacks, and it carries a real BRDF (solar zenith,
view zenith, relative azimuth). It is also the only part of the model that runs
with no weights file and no ML stack.

*Sources for this page: the {mod}`robust.rt.ztt` module docstring and each
function's own docstring, which name the paper equations;
[`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md) §4.3;
[`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md)
§2 and §4. Every printed number was re-measured in this environment when the
page was written. The paper itself is* Appl. Sci. **2018**, 8, 2684,
doi:10.3390/app8122684.

## What it computes

{func}`~robust.rt.ztt.rrs_ZTT` is the paper's Equation (12), subsurface
$r_{rs}$:

$$
r_{rs} \;=\;
\frac{1}{\mu_d}\;
\frac{\beta(\psi)/b_b}
     {\dfrac{a}{b_b}\left(1 - \cos\theta_v\,\dfrac{\Psi_{K_{Lu}}}{\mu_\infty}\right)
      \;+\; f_L\left(1 - \tilde b_b^{\,-1}\right)
      \;+\; \tilde b_b^{\,-1}} .
$$

{func}`~robust.rt.ztt.Rrs_ZTT` is that put through the interface,
$R_{rs} = A\,r_{rs}/(1 - B\,r_{rs})$ — nothing more (verified bitwise against
{func}`~robust.rt.conventions.rrs_to_Rrs` composed with
{func}`~robust.rt.ztt.rrs_ZTT`).

**Elastic by scope.** The paper's Equation (18) is Equation (12) *plus* an
additive $r_{rs,\mathrm{Raman}}$ term. `robust.rt` composes the inelastic terms
on top of the finished elastic model instead (see {doc}`forward`), so here that
term is simply **absent, not approximated**.

### The terms, and which equation each is

| Symbol | Meaning | Paper eq. | Function |
| --- | --- | --- | --- |
| $\beta(\psi)/b_b$ | backward VSF over total backscattering | (10) | {func}`~robust.rt.ztt.backward_phase_over_bb` |
| $\mu_d$ | average cosine of the downwelling field | (14) = (15) × (17) | {func}`~robust.rt.ztt.mu_d` |
| $\Psi_{K_{Lu}}$ | $K_{Lu}/K_\infty = 1 + F(\psi)$ | (6), with (4) | {func}`~robust.rt.ztt.psi_KLu` |
| $\mu_\infty$ | asymptotic average cosine | (8) — see below | {func}`~robust.rt.ztt.mu_infinity` |
| $\tilde b_b$ | total backscattering ratio $b_b/b$ | (11) | {func}`~robust.rt.ztt.bb_tilde` |
| $f_L$ | upwelling-radiance shape factor | (31) | {func}`~robust.rt.ztt.f_L` |

The paper's own summary of the assembly is its §2.9, and the transcription
follows it line for line.

## Angles: the convention trap

**The paper's zenith angles are in-water and measured from straight down, so
nadir viewing is $\theta_v = 180°$** and $-\cos\theta_v = +1$ in the
denominator above. {class}`~robust.rt.types.Geometry` uses the opposite and
more usual convention — `theta_v = 0` for nadir — and its `theta_s` is the
*above-water* solar zenith as L23 reports it, which is the paper's **primed**
$\theta_s'$. The paper's unprimed $\theta_s$ is the *in-water* angle, and
inverse Snell relates the two through a refractive index of 1.34
({data}`~robust.rt.ztt.REFRACTIVE_INDEX`):

$$
\theta_s = \arcsin\!\left(\frac{\sin\theta_s'}{1.34}\right) .
$$

Note the direction: it is the sine that is *divided* by 1.34, so the in-water
angle is always the smaller one. Writing it the other way up is the mistake this
paragraph exists to prevent — $\arcsin(1.34\sin 60°)$ has no solution at all.
{func}`~robust.rt.ztt.in_water_zenith` is the implementation.

Both conversions happen in exactly one place,
{func}`~robust.rt.ztt.geometry_to_paper_angles`, because getting either
backwards produces a plausible-looking BRDF that is wrong. It returns the
in-water solar zenith, the above-water solar zenith, the paper-convention view
zenith, and the in-water scattering angle $\psi$. Measured for the L23 case
(nadir view, 60° sun):

```text
geometry_to_paper_angles(Geometry.nadir(60.0)):
  theta_s (in water) = 40.2623 deg
  theta_s' (in air)  = 60.0000 deg
  theta_v (paper)    = 180.0 deg
  psi                = 139.7377 deg
```

That is the paper's own worked example — $\theta_s' = 60°$ giving 40.3° in
water and $\psi = 139.7°$ — reproduced to 0.04°, and it is one of the two
anchors the report cites for the transcription
([`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md)
§2). The other is $\mu_d$ reproducing the paper's quoted 0.79–0.94 range.

## The phase function, explicitly

Equation (10) is where the explicitness lives:

$$
\frac{\beta(\psi)}{b_b} \;=\;
\frac{P_{bb}(\psi)\,b_{bp} \;+\; \beta_w(\psi)}{b_{bp} + b_{bw}} .
$$

The particulate backward phase function appears as itself. Two inputs feed it.

**Water, $\beta_w(\psi)$.** The paper defers this to Zhang et al. (2009), whose
only implementation to hand is an unfinished port, so
{func}`~robust.rt.ztt.beta_w_over_bb_w` uses the closed molecular form instead.
Only the *shape* is needed, since Equation (10) multiplies by $b_{bw}$ and the
unknown $\beta_w(90°)$ cancels:

$$
\frac{\beta_w(\psi)}{b_{bw}} =
\frac{1 + f\cos^2\psi}{2\pi\,(1 + f/3)},
\qquad
f = \frac{1-\delta}{1+\delta},
\qquad
\delta = 0.039 ,
$$

with $\delta$ the depolarization ratio
({data}`~robust.rt.ztt.WATER_DEPOLARIZATION`). Checkable rather than assumed:
this gives **0.23417 sr⁻¹ at $\psi = 180°$** (measured), against the ~0.23 sr⁻¹
quoted for pure water in the synthesis document.

**Particles, $P_{bb}(\psi)$.** A **required model input** — the paper's §2.9 is
explicit that $b_{bp}$, $a_{pg}$, $P_{bb}(\psi)$ and $\tilde b_{bp}$ must be
supplied. The default is {func}`~robust.rt.ztt.P_bb_sullivan`, the measured
average of Sullivan & Twardowski (2009) from several million field VSFs, which
is the choice the ZTT paper reports its best performance with. Measured:

```text
P_bb_sullivan(90 deg)  = 0.2323 sr^-1
P_bb_sullivan(140 deg) = 0.1352 sr^-1     (the minimum region)
P_bb_sullivan(180 deg) = 0.1529 sr^-1     (nadir; a short extrapolation)
```

:::{warning}
"Constant shape" in the source paper means **constant across water types, not
across angle**. $P_{bb}$ still varies with $\psi$. Passing a fixed scalar
`P_bb=` to {func}`~robust.rt.ztt.rrs_ZTT` — legitimate for a sensitivity test —
*inverts* the modelled solar-zenith trend, and a regression test pins that.
:::

Two smaller notes on the same coefficients. Sullivan & Twardowski's published
polynomial carries a typographic error in its $a_3$ term (printed
`8.007E−02`, which at $\psi = 140°$ would contribute ≈1570 against a tabulated
value of 0.137); {data}`~robust.rt.ztt.P_BB_ST_COEFFS` uses the intended
`8.007E−04`, an independent refit of their Table 1 having confirmed it. And
their measurements stop at 170°, so nadir viewing extrapolates by 10° — an
independent refit gives 0.156 and fitting a constant against L23 at that
geometry gives 0.148, so 0.153 is well constrained.

The other phase-function input is the scalar
{attr}`PhaseParams.B_p <robust.rt.types.PhaseParams.B_p>`, the paper's
$\tilde b_{bp} = b_{bp}/b_p$, which enters Equation (11):

$$
\tilde b_b \;=\; \frac{b_{bp} + b_{bw}}{b_{bp}/\tilde b_{bp} + b_w},
\qquad b_w = 2\,b_{bw} ,
$$

the last identity because pure water scatters symmetrically about 90°, so
exactly half its scattering is backward. This is the design's chosen
phase-function parameterisation ([`design/rt_elastic_model.md`](gh:design/rt_elastic_model.md)
§4.2), and {class}`~robust.rt.types.PhaseParams` is deliberately a container so
that the fuller ZTT backward-VSF descriptors can join it later without changing
{func}`~robust.rt.hybrid.forward`'s signature.

## $\mu_d$, and the missing coefficients of $\mu_\infty$

$\mu_d$ factorises into an atmosphere/geometry part and an IOP part
(Equation (14)):

$$
\mu_d \;\approx\; M^+_d(\theta_s', V)\;\times\;M^*_d(b_b/a,\ \eta_{bb}),
\qquad
\eta_{bb} = \frac{b_{bw}}{b_{bp}+b_{bw}} .
$$

{func}`~robust.rt.ztt.Md_plus` is Equation (15), the Morel & Prieur (1977)
cardioidal-skylight form with a variable diffuse fraction
{func}`~robust.rt.ztt.diffuse_fraction` (Equation (16), a fit to Gregg & Carder
1990 with a visibility dependence, default $V = 15$ km — HydroLight's).
{func}`~robust.rt.ztt.Md_star` is Equation (17), a cubic in $\log_{10}(b_b/a)$
whose coefficients are linear in $\log_{10}\eta_{bb}$.

:::{note}
One transcription subtlety that a reader re-deriving this will hit: Equation
(13) defines $\mu_w = \cos\theta_s$ with $\theta_s$ **unprimed**, i.e. the
*in-water* angle, while $H$ and $P_3$ take the primed above-water angle. Using
the above-water cosine gives $\mu_d = 0.573$ at $\theta_s' = 62°$ against the
0.79 the paper quotes; the in-water cosine reproduces 0.792. Tests pin both
endpoints.
:::

**One published coefficient set is missing, and it is the honest caveat of the
whole backbone.** Equation (8) fits $\mu_\infty(b_b/a, \eta_{bb})$ with sixteen
coefficients $m_1 \ldots m_{16}$ and states that they are in Appendix A,
Table A2. They are not there: as printed, Table A2 lists coefficients for
Equations (3), (4), (16) and (17) and then runs straight into Table A3, and the
MATLAB code the paper points to is not at the given address.
{func}`~robust.rt.ztt.mu_infinity` therefore implements Equation (8)'s
*structure* and **requires the sixteen coefficients from the caller**, rather
than inventing them.

With none supplied, {func}`~robust.rt.ztt.rrs_ZTT` falls back to
{func}`~robust.rt.ztt.mu_infinity_tt2017` — the same two authors' 2017
antecedent (*Opt. Express* **25**, 18122), which is the parameterisation the
2018 text says Equation (8) "extended". Its Table 1
({data}`~robust.rt.ztt.MU_INF_TT2017_TABLE1`) gives
$\mu_\infty = p_0 + p_1 L + p_2 L^2$ with $L = \log_{10}(b_b/a)$ at six
discrete $\eta_{bb}$; the three coefficients are interpolated linearly in
$\log_{10}\eta_{bb}$, which recovers the two-dimensional surface Equation (8)
would have provided and keeps it differentiable.

:::{warning}
Results computed this way must be reported as **"ZTT with the TT2017
$\mu_\infty$"**, never as the published 2018 model. This is item 6 of the
elastic report's "may not claim" list
([`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md)
§5). Passing `mu_inf_coeffs=` restores the 2018 model in one line the moment
the sixteen numbers arrive.
:::

The 2017 fits cover $b_b/a$ over 1e-4…1e-1 and $\eta_{bb}$ over 0.0098…0.98
(the module constants `MU_INF_TT2017_BB_OVER_A_RANGE` and
`MU_INF_TT2017_ETA_RANGE`). L23 reaches $b_b/a \approx 0.31$, so the brightest
scenes extrapolate.

## $f_L$, and one conditioning hazard

{func}`~robust.rt.ztt.f_L` is Equation (31),

$$
f_L(\psi, \lambda) = f_{L,\mathrm{ave}}(\lambda)\,
\bigl[0.07762 \sin\psi + 1.0405\bigr],
$$

a tabulated spectral shape ({data}`~robust.rt.ztt.FL_AVE`, the paper's
Table A3, 350–800 nm) scaled by scattering angle. Measured at the L23 nadir /
60° geometry: $f_L(139.74°, 550\ \mathrm{nm}) = 1.0787$.

:::{warning}
**$F(\psi)$ is badly conditioned by construction.**
{func}`~robust.rt.ztt.F_psi` is a quartic in $\psi$ *in degrees*; at
$\psi = 180°$ its five terms are −398, +1412, −1866, +1089, −236 and they
cancel to **0.0239**, a factor of ~78 000. float32 loses about five significant
figures there, and that one term accounts for essentially all of the model's
float32/float64 disagreement (~5e-5 relative on $r_{rs}$). Negligible against
a 3–5 % model error and irrelevant to the gradient gate, which runs in float64
— but do not rewrite that evaluation without re-checking, and do not tighten a
float32 tolerance that depends on it. A test pins the current agreement.
:::

## How well it does on its own

From [`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md)
§4, rRMS in $r_{rs}$ space on the full 9 960-sample L23 batch:

| model | train | held-out scenes | held-out @ 60° |
| --- | --- | --- | --- |
| standard Gordon (1988) | 7.21 % | 7.21 % | 9.01 % |
| **ZTT backbone alone** | **5.95 %** | **5.93 %** | **8.11 %** |
| O25 form, refit on L23 | 0.70 % | 0.69 % | 0.71 % |
| hybrid, MLP emulator | 0.30 % | 0.30 % | 0.32 % |

The unfitted backbone already beats Gordon overall — flat at 4–6 % where Gordon
degrades toward the red (12 % at 750 nm) — which is the explicit VSF paying off
before any learning happens. What it does *not* do is reach the modern
benchmark; that is what {doc}`emulator` is for.

## API

| What | Where |
| --- | --- |
| The model | {func}`~robust.rt.ztt.rrs_ZTT`, {func}`~robust.rt.ztt.Rrs_ZTT` |
| Geometry | {func}`~robust.rt.ztt.geometry_to_paper_angles`, {func}`~robust.rt.ztt.in_water_zenith`, {func}`~robust.rt.ztt.above_water_zenith`, {func}`~robust.rt.ztt.scattering_angle` |
| Phase function | {func}`~robust.rt.ztt.backward_phase_over_bb`, {func}`~robust.rt.ztt.P_bb_sullivan`, {func}`~robust.rt.ztt.beta_w_over_bb_w`, {func}`~robust.rt.ztt.bb_tilde` |
| Average cosines | {func}`~robust.rt.ztt.mu_d`, {func}`~robust.rt.ztt.Md_plus`, {func}`~robust.rt.ztt.Md_star`, {func}`~robust.rt.ztt.mu_infinity`, {func}`~robust.rt.ztt.mu_infinity_tt2017` |
| Shape factors | {func}`~robust.rt.ztt.psi_KLu`, {func}`~robust.rt.ztt.F_psi`, {func}`~robust.rt.ztt.f_L` |

Full signatures, every coefficient table, and the `[source]` links are on the
{doc}`../api` page under *ztt*.
