# Scope and limitations

This page is the one place on the site where the model's limits are stated
without softening. It exists because everything else here describes what
`robust.rt` does; a reader deciding whether to use it needs the other half, and
needs it in the authors' own words rather than a paraphrase that has had its
edges rounded off.

So the two reports' §5 sections — *What the prototype may claim — and what it
may not* — are **quoted verbatim** below, character for character. Where a
passage is quoted, the box says so and links the file it came from. Nothing in
those boxes was rewritten, reordered, or shortened.

*Sources for this page:
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§5 (lines 290–324 of that file) and
[`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md)
§5 (lines 201–224). Both quotations were diffed against the report files
character for character when this page was written, and both reports are
rendered in full in the {doc}`Reports <../reports/index>` section if you would
rather read the passage in place. The scope statement in the next section comes
from `README.md`,
[`proposals/Claude_Science/anthropic_application.md`](gh:proposals/Claude_Science/anthropic_application.md)
and [`context/context_summary.md`](gh:context/context_summary.md).*

## The retrieval does not exist

**retrieve-or-bust** is an AI-driven effort to retrieve phytoplankton and
inherent optical properties from hyperspectral ocean colour. This site documents
`robust.rt`, the project's **first component**: a differentiable *forward* model
that maps IOPs to remote-sensing reflectance.

The retrieval — the inversion, $R_{rs} \rightarrow$ IOPs — is a **separate
component and it has not been built**. There is no inversion code in this
package, no retrieval result on this site, and no accuracy figure anywhere here
that describes one. Every number quoted on this site is a forward-model number:
how well `robust.rt` reproduces a reference ocean it was given the IOPs of.

That is a scope boundary, not a defect in this model. A forward model is what
the inversion will be built on, and building it first is the plan. But it means
one specific misreading has to be closed off: **nothing on this site may be read
as evidence that IOPs can be retrieved from $R_{rs}$.** The literature holds
that inversion to be fundamentally degenerate — $R_{rs}$ constrains
$u = b_b/(a + b_b)$ far better than it constrains $a$ and $b_b$ separately — and
none of the work documented here addresses that degeneracy. Beating it is the
project's actual subject, and it is future work.

## What the forward model may claim

The all-processes-on model, quoted from the inelastic report:

:::{important}
Quoted verbatim from
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§5, *What the prototype may claim — and what it may not*.

**It may claim:** on held-out water bodies from the reference ensemble, a
complete differentiable forward model — elastic scattering, Raman, and
chlorophyll-a fluorescence with a retrievable quantum yield — that reproduces
the all-processes-on ocean to 0.34% rRMS on its 400–700 nm domain at all three
solar zeniths, with per-process errors ≤ 1%, machine-precision gradients
including ∂*R*<sub>rs</sub>/∂φ<sub>C</sub>, at 1.59× the elastic hybrid's
cost, while leaving elastic-only behavior bit-identical.
:::

## What it may not claim

Six items. They are the reasons the sentence above is hedged the way it is, and
they are quoted with the rest of the section so that the claim and its
qualifications cannot be separated.

:::{warning}
Continues the same verbatim quotation from
[`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§5.

**It may not claim:**

1. **Geometry generalization.** The heads interpolate over three zenith
   anchors; trained without 60° the Raman head errs by **−74%** there — worse
   than no head at all. The elastic report's unseen-zenith warning applies in
   sharper form, and unlike the elastic emulator the heads carry **no domain
   guard** yet.
2. **φ<sub>C</sub> beyond 0.02.** The φ<sub>C</sub>-linearity of the corrected
   model is exact *by construction*; L23 provides truth at exactly one yield.
   Whether the real ocean's fluorescence is φ<sub>C</sub>-linear at the ±few-%
   level is untested (varied-φ<sub>C</sub> HydroLight runs would test it —
   §7).
3. **The 730 nm PS I shoulder.** `emission_shape='double'` is implemented but
   sits at −23.6% against the single-Gaussian truth at 685 nm — consistent
   with moving 25% of the emission into a shoulder L23 cannot see. Off
   everywhere; unvalidatable with data in hand.
4. **Wavelengths below 400 nm.** Raman excitation falls off the L23 grid edge
   (352 nm at λ = 400); the terms run on clamped IOPs and the heads never
   trained there. Measured: 13% at 350 nm. The domain is documented, not
   enforced by error.
5. **θ<sub>s</sub> derivatives at the sky anchors.** The packaged-*E*<sub>d</sub>
   zenith interpolation is piecewise-linear; at exactly 0°/30°/60° the
   derivative is one-sided. Differentiate off the anchors (the gradient gate
   does).
6. **Anything the elastic report already declined to claim.** Nadir-only
   viewing, L23-like water, the narrow *B*<sub>p</sub> slice, and the TT2017
   µ<sub>∞</sub> stand-in are inherited unchanged.
:::

Three of those deserve a pointer to where they are visible in this
documentation, since a reader may otherwise meet them by surprise:

- **Item 1, the −74 % unseen-zenith result**, is the sharpest number in the
  effort. The correction heads are interpolators over exactly three solar-zenith
  anchors (0°, 30°, 60°) and carry **no domain guard** — unlike the elastic
  emulator, which does ({doc}`../model/emulator`). Ask a head for a zenith it
  was not trained at and it will answer, confidently, and it can be worse than
  having no head at all. See {doc}`../model/corrections`.
- **Item 3, `emission_shape='double'`**, is implemented and reachable
  ({doc}`../model/fluorescence`). It is off by default, it was off for all
  training and all validation, and there is no data in hand that can score it.
  The −23.6 % figure is what it scores against the single-Gaussian truth, which
  is a statement about L23, not a measurement of the shoulder.
- **Item 4, λ < 400 nm**, is a *documented* domain, not an *enforced* one. The
  model runs below 400 nm and returns numbers; the Raman excitation has left the
  L23 grid by then and is clamped. 13 % at 350 nm is the measured cost. The
  total-rRMS gate is scored over 400–700 nm for this reason, with the full-grid
  number reported alongside it — see {doc}`validation`.

## The elastic backbone's limits, inherited

Item 6 above inherits the elastic report's list unchanged, so it is reproduced
here in full rather than left as a cross-reference. This is the earlier
milestone's §5, about the elastic model that the inelastic terms are composed
onto.

:::{warning}
Quoted verbatim from
[`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md)
§5, *What the prototype may claim — and what it may not*.

**It may claim:** on held-out water bodies from the reference ensemble, a
differentiable elastic forward model with an explicit phase-function input that
reaches 0.30% rRMS, uniform across the spectrum and the three solar zeniths, with
machine-precision gradients, at ~5× the cost of its analytic backbone.

**It may not claim:**

1. **A 24× advance.** That is the margin over a 1988 model. Over the modern
   benchmark (O25) it is **2.3×**.
2. **Victory over O25 as published.** O25's 0.69% comes from coefficients refit
   on *our* training split with *our* objective; the published paper prints only
   plots. It is labeled "O25 form, refit on L23" everywhere for this reason.
3. **Geometry generalization.** At the unseen 60° the answer depends on the seed
   (4.7–12.2%) and the benchmark wins deterministically. The acceptance gate was
   deliberately written on the scene split.
4. **Phase-function generalization.** L23 spans a factor 1.7 in *B*<sub>p</sub>
   against the design's ~7× nominal band, with one fixed Fournier–Forand shape.
   Flat per-bin accuracy on that slice is not evidence of generalization.
5. **Any off-nadir capability.** L23 fixes the sensor at nadir and the azimuth at
   zero; the BRDF axes are untested, and the domain check correctly flags any
   off-nadir view.
6. **The published 2018 ZTT model.** Its Equation (8) µ<sub>∞</sub> coefficients
   are not in the paper; the TT2017 surface stands in. Results are "ZTT with the
   TT2017 µ<sub>∞</sub>".
:::

## Also in the API, and not validated

{mod}`robust.rt.cdom_fl` — CDOM fluorescence — is exported by `robust.rt`, is
documented on the {doc}`../api` page, and is **not validated**. It is off by
default, analytic-only, and the truth data that would score it does not exist:
L23 has no CDOM-fluorescence channel. Its learned correction head, δ_C, ships
untrained and unwired ({doc}`../model/corrections`). Nothing in either report's
§5 covers it, because it postdates both.

Treat it as present and unmeasured. It is under active development, and this
page will say something different when there is a measurement to report.

## Two tests that skip off their anchor machine

For completeness, since a reader who clones the repository will meet them:
`test_elastic_hash_regression_strict` and `test_gate_4_pre_change_pins` assert
byte-exact SHA-256 pins on the elastic reference outputs. Those pins were
anchored on the tank server, not on the Mac this documentation was written on,
and here the outputs reproduce to 3 ULP (max relative 3.3×10⁻⁷) rather than
bit-for-bit — float32 arithmetic is not bit-reproducible across CPUs and
JAX/XLA builds. Since there is a second pin set (the default-inelastic one)
anchored on a *different* machine again, no single machine can reproduce both.

Each strict tier therefore runs only where it is anchored and **skips with a
reason** everywhere else — declared by `ROBUST_HASH_ANCHOR=tank|mac`, or by the
hostname table in `robust/tests/test_inelastic_types.py` when that variable is
unset; CI skips both regardless. So the suite is **green**, with those two
skips visible and explained in the `-ra` summary, and the guard against an
actual route change is the closeness tier
(`test_elastic_regression_close_everywhere` and its inelastic sibling), which
runs and passes on every machine. What this site still does not claim is
bit-identity on a machine that has not demonstrated it.
