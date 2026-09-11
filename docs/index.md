# retrieve-or-bust

```{figure} /figs/rob_graphic_readme.png
:alt: Retrieve or Bust: observe hyperspectral Rrs, break the IOP-inversion degeneracy with priors and AI, retrieve IOPs with uncertainty, validate against in-situ truth.
:align: center
:width: 100%

The project in four steps, as the [README](gh:README.md) tells it. This site
documents the physics the middle two will be built on — the forward model — and
not the retrieval, which does not exist yet.
```

## Overview

**retrieve-or-bust** is a genuine, no-hedging attempt to finally crack the
ocean-colour **inherent optical property (IOP) inversion**: recovering the
absorption $a(\lambda)$, backscattering $b_b(\lambda)$, and their constituent
components from remote-sensing reflectance $R_{rs}(\lambda)$. Success is defined
as an outcome, not a number — **retrieving more independent, physically
meaningful IOP components from hyperspectral reflectance than current methods
manage, with credible uncertainties, validated against in-situ truth.**

**Why this is hard.** The inversion is fundamentally *degenerate*. Reflectance
constrains a ratio, $u = b_b/(a + b_b)$, not the pieces, so many distinct
optical states masquerade as one another — and the quantities being retrieved
live in a nearly infinite space of candidate spectral shapes. No amount of
clever fitting removes that ambiguity; the only thing that can is **external
information**.

**The bet.** Supply exactly what has been missing: **priors** from in-situ
observations, environmental context and the history a location carries in its
time series, and a far wider exploration of candidate methods — Bayesian,
deep-learning, or hybrid — than hand-design has ever allowed, with **modern AI
as the accelerant** (in the near term, Claude, under scientific direction). The
point of departure is BING (Prochaska & Frouin 2025, *Biogeosciences* 22, 4705),
which is a starting line and not a destination. The
[README](gh:README.md) states this in full;
[`proposals/Claude_Science/anthropic_application.md`](gh:proposals/Claude_Science/anthropic_application.md)
is the complete statement of scope and
[`context/context_summary.md`](gh:context/context_summary.md) the physics and
literature background.

## This site: the first major contribution

**What is documented here is `robust.rt`, a differentiable radiative-transfer
forward model** for ocean colour, written in
[JAX](https://docs.jax.dev/en/latest/) — retrieve-or-bust's first major
contribution, and the physics the retrieval will be built on. Its public entry
point,
{func}`robust.rt.forward() <robust.rt.hybrid.forward>`, maps
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
point of this component: gradients are what the retrieval will need.

```{figure} /_static/fig_inelastic_architecture.png
:alt: The composed forward-model architecture: IOPs, phase function and geometry enter the ZTT analytic backbone and a learned residual emulator; the elastic result is multiplied by a corrected Raman factor and added to a phi_C-linear chlorophyll fluorescence kernel, giving Rrs.
:align: center
:width: 100%

The whole model on one diagram: an analytic backbone, a small learned elastic
residual, and two inelastic emission terms each with a bounded learned
correction. Reproduced from `reports/report_rt_inelastic_model.md` §2.
```

## What exists, and what does not

**`robust.rt` is a forward model, and only a forward model. The retrieval
itself — the inversion from a measured spectrum back to IOPs, the thing the
project is named after — is a separate component of retrieve-or-bust and does
not exist yet.** What is documented here is the map from IOPs to
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
§4, and §5). They are the site's evidence base, and the
{doc}`Reports <reports/index>` section renders both of them in full and
verbatim; the {doc}`scope-and-limitations <using/limitations>` page quotes their
limits word for word rather than paraphrasing them. Nothing on this site states
a number that was not measured.

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

:::{grid-item-card} Reports
:link: reports/index
:link-type: doc

The two team reports behind every number on this site, reproduced in full: the
elastic model (2026-08-15) and the inelastic completion (2026-08-27).
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
quickstart_nb
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
:caption: Reports
:maxdepth: 2
:hidden:

reports/index
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

api
references
development_record
Team <member_policy>
```
