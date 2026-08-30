# retrieve-or-bust

```{figure} /_static/fig_inelastic_architecture.png
:alt: The composed forward-model architecture: IOPs, phase function and geometry enter the ZTT analytic backbone and a learned residual emulator; the elastic result is multiplied by a corrected Raman factor and added to a phi_C-linear chlorophyll fluorescence kernel, giving Rrs.
:align: center
:width: 100%

The whole model on one diagram: an analytic backbone, a small learned elastic
residual, and two inelastic emission terms each with a bounded learned
correction. Reproduced from `reports/report_rt_inelastic_model.md` §2.
```

## What this is

**retrieve-or-bust** is a differentiable radiative-transfer forward model for
ocean colour, written in [JAX](https://docs.jax.dev/en/latest/). Its public
entry point, {func}`robust.rt.forward() <robust.rt.hybrid.forward>`, maps
inherent optical properties —
absorption, backscattering, and an *explicit* phase-function descriptor — plus
a viewing and illumination geometry to a remote-sensing reflectance spectrum
$R_{rs}(\lambda)$.

If you work in ocean colour, the pitch is short. Almost every analytical
forward model buries the particle phase function in coefficients derived from
radiative-transfer runs with one *prescribed* phase function, so none can
represent independent variability in its shape — a first-order driver of the
angular distribution of water-leaving radiance. The backbone here is
Twardowski & Tonizzo (2018), which carries the backward volume scattering
function explicitly; a 417-parameter neural residual corrects the
multiple-scattering effects the analytic form misses; and on top sit the two
inelastic emission terms that real spectra contain and elastic models cannot
express — **Raman scattering by water** and **chlorophyll-a fluorescence** —
each as analytic physics with a small bounded learned correction. Because the
whole chain is JAX, every output is exactly differentiable with respect to
every input, including the fluorescence quantum yield $\varphi_C$. That is the
point of the package: gradients are what an inversion will need.

## What exists, and what does not

**This is a forward model, and only a forward model. The inversion — the
retrieval of IOPs from a measured spectrum, the thing the project is named
after — does not exist yet.** What is documented here is the map from IOPs to
$R_{rs}$: its accuracy against the Loisel et al. (2023) synthetic archive, its
speed, its gradients, and — at least as prominently — the places where it is
not yet trustworthy. Held out from training, the complete model reaches
**0.34 % rRMS** against the realistic L23 truth (all processes on) at every
solar zenith over 400–700 nm, where the elastic model alone scores 16–19 %
against the same truth and 48 % at the 685 nm fluorescence peak; the elastic
half on its own — backbone plus residual — is **0.30 % rRMS** against
elastic-only truth, **2.3×** better than the O25 benchmark of Pitarch et al.
(2025) refit on identical data; gradients agree with finite
differences to $\le 5.9\times10^{-9}$, at **1.59×** the elastic model's
runtime; and `inelastic=None` leaves the elastic model **bit-identical**,
SHA-256 pinned. Those numbers come with limits that belong in the same breath:
the correction heads are *interpolators* in solar zenith and err by **−74 %**
at a zenith held out of their training, $\varphi_C$ generalisation is linear by
construction with truth at only one value ($\varphi_C = 0.02$), and
wavelengths below 400 nm are outside the supported domain.

Every number above was measured, and the measurements live in two reports in
the repository — `reports/report_rt_elastic_model.md` (executive summary and
§5, limitations) and `reports/report_rt_inelastic_model.md` (executive summary,
§4, and §5). They are the site's evidence base; the Reports section will render
them in full, and the scope-and-limitations page will quote their limits
verbatim rather than paraphrase them. Nothing on this site states a number that
was not measured.

## Where to go

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: installation
:link-type: doc

Install the package and the JAX stack, then run `forward()` on one L23 scene,
switch the inelastic terms on, and take a gradient.
:::

:::{grid-item-card} The model
:link: model/overview
:link-type: doc

The composition law in one page — every term, the module that owns it, and the
chapters that take them one at a time.
:::

:::{grid-item-card} Using it
:link: using/data
:link-type: doc

The L23 reference data, the validation protocol behind every number on this
site, and a blunt statement of what the model may not be used to claim.
:::

:::{grid-item-card} Reference
:link: api
:link-type: doc

The full API, generated from the docstrings in the checkout with the real JAX
signatures — no mocked imports.
:::

::::

```{toctree}
:caption: Getting started
:maxdepth: 2
:hidden:

installation
quickstart
```

```{toctree}
:caption: The model
:maxdepth: 2
:hidden:

model/overview
```

```{toctree}
:caption: Using it
:maxdepth: 2
:hidden:

using/data
using/validation
using/limitations
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

api
references
Team <member_policy>
```
