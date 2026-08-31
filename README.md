# retrieve-or-bust

[![CI](https://github.com/ocean-colour/retrieve-or-bust/actions/workflows/ci.yml/badge.svg)](https://github.com/ocean-colour/retrieve-or-bust/actions/workflows/ci.yml)

Our last best effort at IOP Retreivals

## What this is

**retrieve-or-bust** is an AI-driven effort to retrieve phytoplankton and
inherent optical properties (IOPs) — absorption, backscattering, and their
components — from hyperspectral ocean color. The inversion from remote-sensing
reflectance `Rrs(λ)` to IOPs is fundamentally degenerate: `Rrs` constrains
essentially the ratio `u = bb/(a+bb)`, so many distinct optical states produce
nearly identical spectra. The project's bet is that AI can break that degeneracy
by systematically injecting the external information the physics demands —
in-situ, environmental and spatiotemporal priors — and by searching the space of
candidate retrieval methods (Bayesian, deep-learning, or hybrid) far more
broadly than a hand-built approach can.

The fuller statement of scope is
[`proposals/Claude_Science/anthropic_application.md`](proposals/Claude_Science/anthropic_application.md);
the physics and literature background is
[`context/context_summary.md`](context/context_summary.md).

**The project is being built in components, and only the first exists today:**
the differentiable radiative-transfer forward model in `robust/rt/` (see below)
— the physics the retrieval will be built on. **The retrieval itself — the
inversion — does not exist yet.**

## Development

```bash
pip install -r requirements.txt     # full environment, incl. the CPU JAX stack
pip install -e . --no-deps          # the package itself
pytest -q                           # from the repo root
```

The Loisel+2023 reference data lives outside the repo; `ocpy` finds it via
`$OS_COLOR`. Tests that need it **skip** when it is absent, so `pytest -q` stays
meaningful either way (`-ra` prints the skip reasons).

## The forward model — the first component

The radiative-transfer forward model lives in `robust/rt/`: a differentiable
(JAX) map from IOPs, phase function and geometry to `Rrs(λ)` — an analytic
backbone plus a learned residual, with Raman scattering and chlorophyll-a
fluorescence on top. It is documented under [`docs/`](docs/) (published as the
ReadTheDocs project `retrieve-or-bust`).

Its working documents, elastic half then inelastic half:

|  | Elastic | Inelastic |
|---|---|---|
| Design | [`rt_elastic_model.md`](design/rt_elastic_model.md) | [`rt_inelastic_model.md`](design/rt_inelastic_model.md) |
| Milestones | [`rt_elastic_model_coding_plan.md`](design/rt_elastic_model_coding_plan.md) | [`rt_inelastic_model_coding_plan.md`](design/rt_inelastic_model_coding_plan.md) |
| What is built | [`rt_elastic_implementation.md`](design/rt_elastic_implementation.md) | [`rt_inelastic_implementation.md`](design/rt_inelastic_implementation.md) |
| Report | [`report_rt_elastic_model.md`](reports/report_rt_elastic_model.md) | [`report_rt_inelastic_model.md`](reports/report_rt_inelastic_model.md) |

plus the build notebooks in [`notebooks/RT/`](notebooks/RT/).
