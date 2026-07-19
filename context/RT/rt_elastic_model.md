# Elastic Radiative Transfer for Ocean-Color IOP Retrieval

*A synthesis of the elastic forward model — Gordon → Park & Ruddick → Lee/Pitarch —
alongside the BING wavelength-dependent Gordon deep-dive, with a starting roadmap
for retrieve-or-bust.*

Scope: **elastic** radiative transfer only — the map from inherent optical
properties (IOPs) and Sun–sensor geometry to remote-sensing reflectance,
`Rrs(λ; a, bb, geometry)`, with no inelastic processes (Raman scattering, CDOM or
chlorophyll fluorescence). Inelastic terms are a later layer and are noted here
only as a boundary. This document is a synthesis plus roadmap; it does not
document the BING package wiring.

---

## 1. Why the forward model is the whole game

retrieve-or-bust inverts `Rrs(λ)` for IOPs, and ultimately for the *components*
`a_ph`, `a_dg`, `bb_p`. Every inversion — Bayesian, learned, or hybrid — is only
as good as the forward operator it inverts. Two facts set the stakes:

1. **The inverse problem is ill-posed.** `Rrs` couples `a` and `bb` mainly through
   their ratio, so many `(a, bb)` pairs give nearly identical spectra
   (`context_summary.md`). External information (priors, ancillary data) is what
   breaks the degeneracy — *not* a better fit.
2. **Forward-model error is not neutral.** A biased forward operator injects
   *structured* error into the retrieval that priors cannot remove, because the
   inversion will faithfully reproduce the operator's bias. So the elastic RT map
   must be both accurate and — for a learned/Bayesian engine — differentiable and
   fast.

The community's workhorse is the **Gordon approximation**: a low-order polynomial
in the single quantity

```
u(λ) = bb(λ) / [a(λ) + bb(λ)]          rrs ≈ Σ_i  G_i · u(λ)^i
```

Everything below is the story of what that single-variable form gets right, where
it breaks, and how each successive scheme repairs it.

---

## 2. The one physical fact that organizes everything

**`rrs` is not a univocal function of `u`.** The Gordon form makes `rrs` depend on
`u` alone, but the true (Hydrolight) reflectance depends on `a` and `bb`
*separately* — equivalently, on how the backscatter is split between molecular
(`bb_w`) and particulate (`bb_p`) sources, because those have different volume
scattering functions. At fixed `u`, `rrs` still moves with `bb_p`.

![rrs is not univocal in u](fig_rrs_vs_u.png)

*L23 elastic (3320 Hydrolight scenes). At each wavelength the standard Gordon curve
(red) is a single line in `u`, but the simulations scatter around it and the
scatter is organized by `bb_p` (color). In the blue (440 nm) the spread at fixed
`u` is a genuine `bb_p` fan; in the red (665 nm) the standard curve carries a
near-constant positive bias — a wavelength-dependent offset. These are the two
residual structures that every enrichment below is built to capture.*

The physics behind the two branches (Pitarch 2025): at single scattering, `rrs` is
proportional to the backscatter VSF `β(π)/bb`, which is **0.23 sr⁻¹ for pure water**
vs **0.12–0.16 sr⁻¹ for particles** (Zhang 2009; Twardowski & Tonizzo 2018). So for
a given `u`, clearer (higher molecular-fraction) water yields a different `rrs` than
turbid water — the relationship genuinely has (at least) two dimensions, not one.

---

## 3. The elastic lineage

### 3.1 Gordon et al. (1988) — the origin, and its limits

The canonical result expands the irradiance reflectance / Q as a quadratic in `u`:

```
R/Q = l1·u + l2·u²        l1 = 0.0949,  l2 = 0.0794
```

derived from Monte-Carlo RT (Gordon, Brown & Jacobs 1975; Gordon 1986). Key
caveats that the modern literature spends its effort on:

- **The `l_i` are treated wavelength-independent.** All λ-dependence is parked in
  `a`, `bb`, and the `Q` factor (≈ 4–5, "somewhat wavelength dependent").
- **Validity: `θ₀ > 20°` and `u ≲ 0.2`.** Below 20° the backward VSF governs `R/Q`;
  the `i>1` term becomes important at high radiance / high backscatter.
- **Geometry lives in `Q` and the surface factors**, not in the `l_i`. Converting
  subsurface `R/Q` to above-water `Rrs` needs the interface terms
  `(1−ρ)/m² ≈ 0.54`, `(1−rR)`, `r = 0.48`. The whole scheme's stated max error is
  **~±20%**.

Everything after 1988 is an attempt to put back the structure that the constant,
single-variable form throws away: **wavelength, geometry, and the phase-function /
water-vs-particle split.**

### 3.2 Park & Ruddick (2005) — the named baseline (PR05)

PR05 is the project's chosen baseline. It generalizes Gordon to a **fourth-order**
polynomial whose coefficients are tabulated over geometry *and* a phase-function
parameter:

```
Rrs(θo, θv, Δφ) = Σ_{i=1..4} g_i(θo, θv, Δφ, γb) · ωb^i
ωb = bb/(a+bb)   ("backscattering albedo")
γb = bbp/bb   (particle fraction of backscatter, ~0.2–1)
```

- Built from **Hydrolight 4.2**, **Fournier–Forand** phase functions, case-1 + case-2
  IOPs, 412–780 nm; coefficients on a grid of 7 solar × 10 sensor zenith × 13
  relative-azimuth angles × 8 `γb` values.
- **Model uncertainty ~2%** (rms ~1%), dominated by residual phase-function
  variability after `γb` is fixed.
- **`γb` is the price of admission.** It is not observed; PR05 estimates it
  iteratively (their §5C). A `γb` error of 0.05 (needed for ~2% `Rrs`) corresponds
  to a **20–30% `bbp` error at low `γb`**, worse at high `γb` — a real weakness for
  a component-retrieval target.

PR05 is a strong, physically-motivated baseline and correctly identifies the
phase-function (`γb`) axis as the missing degree of freedom. Its two liabilities
for us are the LUT dimensionality (a full 4-D angle×`γb` grid) and the lack of an
`Rrs → γb` inversion path.

### 3.3 Tan et al. (2018) — what PR05 does and does not deliver

Tan evaluated PR05 (as used in POLYMER) against Hydrolight (IOCCG/L23 IOPs) and
9824 AERONET-OC spectra. The verdict is nuanced and directly relevant:

- **`Rrs` reconstruction is good** — RMS < 15%, and **band ratios excellent**
  (bias < 5%). Two parameters suffice; the third barely helps and hurts convergence.
- **But the retrieved parameters are not physical.** Fitted Chl is badly biased,
  and — the load-bearing result — **PR05-reconstructed `Rrs` fed to QAA produces
  significantly biased IOPs** (`a_ph`, `a_dg`, `bb_p`).
- Their recommendation: *use the reconstructed reflectance, not the retrieved
  model parameters.*

Implication for retrieve-or-bust: a scheme can reproduce `Rrs` beautifully and
still be a poor *inversion* engine for components. Forward-model rRMS is necessary
but not sufficient; the retrieval-impact test is the one that matters (and is the
one BING's own logs repeatedly flag as unclosed).

### 3.4 Lee (2011) / Pitarch et al. (2025, "O25") — the recommended evolution

The current state of the art in the elastic Gordon→QAA lineage replaces PR05's
`γb`-indexed 4th-order polynomial with a **bivariate quadratic** that splits the
backscatter albedo into water and particle parts:

```
Rrs = (Gw0 + Gw1·ωbw)·ωbw + (Gp0 + Gp1·ωbp)·ωbp
ωbw = bbw/(a+bb)     ωbp = bbp/(a+bb)     (ωb = ωbw + ωbp)
```

The design choice that matters: **the four coefficients depend on geometry ONLY** —
they are IOP- and wavelength-agnostic by construction. This directly encodes the
"two-branch" physics of §2 (water and particle backscatter contribute through
different VSFs). O25 is calibrated on **PB24** (Pitarch & Brando 2025): a synthetic,
multi-angular, hyperspectral set — 5000 IOP realizations × 1300 geometries, with
Fournier–Forand phase functions chosen over the older Petzold average.

- In independent inter-comparison (D'Alimonte 2025; Pitarch 2025) the ranking is
  **L11 > Morel-2002 > PR05**. O25 refines L11's empirical steps and validates as a
  BRDF corrector (normalizing `Rrs` to nadir/zenith) *and* as a semi-analytical IOP
  retriever.
- Practical: open-source (`github.com/jaipipor/O25`), integrated in NASA HyperCP
  and EUMETSAT ThoMaS, operational in **OLCI Collection 4**.

Because retrieve-or-bust already models `bb_w` (a known constant of pure water) and
`bb_p` separately, **the O25 water/particle split is essentially free for us** — no
`γb` iteration is required. This is the single strongest reason to treat O25/L11 as
the evolution beyond PR05.

### 3.5 Hansen (1971) — background

Hansen's planetary-atmosphere work is the multiple-scattering / **doubling-method**
lineage that underpins how reflectance relates to single-scattering albedo in a
scattering medium — the conceptual ancestor of the `u`-polynomial. (The PDF in
`context/RT/` is a scanned image with no extractable text layer; it is cited here
as background, not mined for specifics.)

---

## 4. The BING deep-dive: `rrs ≠ f(u)`, quantified

JXP's BING work fit the Gordon coefficients directly to the **L23 elastic** dataset
(3320 Hydrolight scenes, 350–750 nm) and found exactly the residual structure of
§2. Two enrichment terms, both wavelength-dependent, capture it:

- **`G0(λ)`** — a constant offset: `rrs = G0 + G1·u + G2·u²`.
- **`Gb(λ)`** — a slope on particulate backscatter: `rrs = G1·u + G2·u² + Gb·bbp`.

The recipe ladder — recomputed here from L23 rather than quoted — shows what each
term buys:

![rRMS ladder](fig_rrms_ladder.png)

*Per-wavelength rRMS vs Hydrolight (`rrs`-space), recomputed from L23 elastic with
unregularized per-λ fits. Standard Gordon degrades monotonically to ~9% at 700 nm.
Adding `G0(λ)` (red) collapses the red-λ error by ~10× (700 nm: 9.0% → 0.35%),
because at red `u` is narrow and almost everything is set by water absorption, so a
constant offset is the right correction. Adding `Gb·bbp` (green) wins in the
blue/green instead (400 nm: 2.5% → 1.8%), where the residual is a `bb_p` fan. The
two are complementary; a joint 4-parameter fit `G0 + G1·u + G2·u² + Gb·bbp` wins or
ties everywhere (550 nm reaches 0.76% — from the BING logs).*

The fitted enrichment terms have clear wavelength structure:

![G0 and Gb vs wavelength](fig_G_lambda.png)

*`G0(λ)` changes sign near 510 nm — the one wavelength where neither a pure offset
nor a pure `bb_p` slope is clean, and where trophic state (the water-vs-particle
mix) matters most. In the joint 4-parameter fit, `Gb` absorbs that trophic-state
component and `G0` stops crossing zero — direct evidence the two terms are the
*right* enrichment, not just extra degrees of freedom.*

**The convergence worth underlining:** BING's `G0`/`Gb` terms and O25's `ωbw`/`ωbp`
split are two routes to the same destination — representing that `rrs` depends on
`(a, bb)` / the water-vs-particle mix, not on `u` alone. BING discovered it
empirically at fixed geometry; O25 built it in structurally with geometry-only
coefficients. They should be read as the same physics.

(Two implementation notes from the BING logs that a re-implementation must respect:
the `Rrs↔rrs` convention is Lee-2002 `A=0.52, B=1.7`; and the per-λ quadratic must
be relatively weighted, or `G2` runs away at red wavelengths.)

---

## 5. Synthesis: one picture

| Scheme | Form | Extra structure beyond `u` | Geometry | Calibration | For component IOPs |
|---|---|---|---|---|---|
| Gordon 1988 | `l1·u + l2·u²` | none | in `Q`, fixed `l_i` | MC RT | crude; ±20% |
| **PR05 (baseline)** | 4th-order in `ωb` | `γb = bbp/bb` (phase fn) | full LUT (θo,θv,Δφ) | Hydrolight + FF | needs `γb` iter; Tan: biased |
| BING G0/Gb | `+G0(λ)`, `+Gb·bbp` | offset + `bb_p` slope | fixed | L23 elastic | best fwd rRMS; retrieval untested |
| **L11 / O25** | bivariate `(ωbw, ωbp)` | water/particle split | geometry-only coeffs | PB24 (FF, multi-angle) | ranks above PR05; split is free for us |

The through-line: **each advance re-introduces a dimension the constant
single-variable Gordon form discarded** — first wavelength (BING), then the
phase-function / water-vs-particle axis (PR05's `γb`, BING's `Gb`, O25's split),
and geometry (PR05 LUT, O25's geometry-only coefficients).

---

## 6. Baseline decision and recommendation

- **Baseline (as chosen):** PR05. It is a defensible, physically-grounded reference
  and the right point of departure — it names the `γb` axis explicitly.
- **Recommended evolution:** the **L11 / O25 bivariate `(ωbw, ωbp)`** form. It ranks
  above PR05 in independent tests, needs no `γb` inversion, its coefficients are
  clean functions of geometry only, and — decisively — its water/particle split
  maps onto retrieve-or-bust's component model for free and is the same physics
  BING found via `G0`/`Gb`.
- **Our own approach — an open choice (three options, to be decided).** The RT
  forward model for retrieve-or-bust is not committed. Three viable directions,
  carried forward as options rather than a decision:

  - **(a) Analytic / physically-structured.** Extend the BING-`G0/Gb` ↔ O25-split
    family (more terms, better parameterization). *Pro:* interpretable, few
    parameters, keeps the inversion analytic and differentiable in closed form.
    *Con:* a polynomial ceiling — the `~2%` blue residual and the 510 nm behavior
    suggest diminishing returns.
  - **(b) Learned forward model.** A neural emulator of RT, `(IOPs, geometry) → Rrs`.
    *Pro:* highest accuracy, naturally BRDF-aware, differentiable for gradient-based
    or amortized inversion. *Con:* a black box; data-hungry; extrapolation risk
    outside the training manifold.
  - **(c) Hybrid.** An analytic backbone (O25-style water/particle split) plus a
    learned residual or coefficient network. *Pro:* keeps the physics and
    interpretability while learning what the polynomial misses; the residual is
    small and smooth, so the network is light. *Con:* two moving parts to validate.

  The choice interacts with the data plan (§7): (b)/(c) want the multi-angular PB24
  set sooner; (a) can mature on L23 first.

---

## 7. Starting roadmap for retrieve-or-bust RT

Ordered, with **variable geometry (BRDF) treated as first-class** and the truth-data
plan **L23-first, PB24-second**.

1. **Reproduce the elastic baseline on L23 (fixed geometry).** Re-fit the Gordon
   ladder (`standard → +G0 → +Gb → joint`) and the O25 bivariate `(ωbw, ωbp)` form
   on L23 elastic; confirm the rRMS ladder above and add the O25 form to the same
   plot. Deliverable: a single elastic forward operator with a documented rRMS
   surface over (λ, water type). *(The figure script here is the seed.)*
2. **Close the retrieval-impact gap (Tan's warning).** For each candidate operator,
   run the component inversion (`a_ph`, `a_dg`, `bb_p`) and report per-IOP MAPE, not
   just forward `Rrs` rRMS. This is the test PR05 fails in Tan (2018) and the one
   BING never ran. It decides whether forward accuracy actually buys retrieval
   accuracy.
3. **Go multi-angular with PB24 (phase 2).** L23 is a single nominal geometry;
   introduce PB24 (5000 IOPs × 1300 geometries, Fournier–Forand) to fit and test
   **geometry-dependent** coefficients. Target O25's geometry-only-coefficient
   design so BRDF is native, not bolted on. Cross-check against O25's published
   coefficients and `github.com/jaipipor/O25`.
4. **Decide the forward-model architecture (§6 a/b/c).** With (1)–(3) in hand,
   choose analytic / learned / hybrid on the evidence: does a learned residual beat
   the analytic ceiling by enough to justify the black-box cost, measured in
   *retrieval* MAPE (step 2), across geometry?
5. **Nail down conventions once.** `Rrs↔rrs` (`A=0.52, B=1.7`), the `bb_w`/`bb_p`
   split, wavelength grid (PACE/OCI 340–895 vs L23 350–750), and the geometry
   parameterization — one config, asserted at load, so results are comparable
   across steps.

Out of scope here (elastic-only): Raman scattering and CDOM/chlorophyll
fluorescence. They matter for real `Rrs` (Tan shows the 665–685 nm and NIR effects)
and will be a separate inelastic layer added on top of whichever elastic operator
wins.

---

## 8. References

- Gordon, H. R., et al. (1988). A semianalytic radiance model of ocean color.
  *JGR* 93(D9), 10909–10924.
- Park, Y.-J., & Ruddick, K. (2005). Model of remote-sensing reflectance including
  bidirectional effects for case 1 and case 2 waters. *Appl. Opt.* 44(7), 1236–1249.
- Tan, J., Frouin, R., Ramon, D., & Steinmetz, F. (2018). Adequacy of
  semi-analytical water reflectance models in ocean-color remote sensing.
  *Proc. SPIE* 10778, 107780A.
- Pitarch, J., et al. (2025). Analytical modeling and correction of the ocean colour
  bidirectional reflectance across water types (O25). *Remote Sens. Environ.* 329,
  114920. — builds on Lee et al. (2011, "L11") and PB24 (Pitarch & Brando 2025,
  *ESSD* 17, 435–460).
- Hansen, J. E. (1971). Multiple scattering of polarized light in planetary
  atmospheres (doubling method). *J. Atmos. Sci.* — background.
- Loisel, H., et al. (2023). A synthetic optical database (L23). *ESSD* 15,
  3711–3731.
- BING Gordon deep-dive: `bing/prompts/gordon.md` logs; coefficient tables in
  `bing/bing/data/RT/gordon_coefficients*.csv`.

*Figures generated by `context/RT/make_rt_elastic_figures.py` (ocean14) from the L23
elastic set (`Hydrolight100.nc`) and the BING coefficient CSVs.*
