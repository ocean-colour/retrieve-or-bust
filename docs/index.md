# retrieve-or-bust

**retrieve-or-bust** is a differentiable radiative-transfer forward model for
ocean colour, written in [JAX](https://docs.jax.dev/en/latest/). Its public
entry point, `robust.rt.forward()`, maps inherent optical properties
— absorption, backscattering, a phase-function parameterisation — plus a
viewing and illumination geometry to a remote-sensing reflectance spectrum
$R_{rs}(\lambda)$. The elastic backbone is the Twardowski & Tonizzo (2018)
analytic model, carrying an *explicit* phase-function dependence; a small
learned residual corrects the multiple-scattering and phase-function effects
the backbone misses. On top of that sit the two inelastic emission terms that
ocean colour cannot honestly ignore — Raman scattering by water and
chlorophyll-a fluorescence — each with a learned correction head. Because the
whole chain is JAX, every output is differentiable with respect to every
input.

**This is a forward model, and only a forward model.** The inversion — the
retrieval of IOPs from a measured spectrum, the thing the project is named
after — **does not exist yet**. What is documented here is the map from IOPs
to $R_{rs}$, its accuracy against the Loisel et al. (2023) synthetic archive,
its speed, its gradients, and — at least as prominently — the places where it
is not yet trustworthy. Nothing on this site states a number that was not
measured; every figure and metric traces back to one of the two reports in
`reports/`, and the scope-and-limitations page states plainly where the model
should not be trusted.

```{toctree}
:caption: Getting started
:maxdepth: 2

installation
quickstart
```

```{toctree}
:caption: The model
:maxdepth: 2

model/overview
```

```{toctree}
:caption: Reference
:maxdepth: 2

api
references
Team <member_policy>
```
