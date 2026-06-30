# Context Summary — IOP Inversion

Synthesis of the five reference PDFs in this `context/` folder. The project goal
is to make a last, AI-assisted attempt at the **IOP inversion problem**:
retrieving inherent optical properties — absorption `a(λ)`, backscattering
`bb(λ)`, and their constituent components — from remote-sensing reflectance
`Rrs(λ)`.

## The big picture

Across these references one theme dominates: **the IOP inversion is
fundamentally ill-posed (non-unique), not merely algorithmically hard.** The
forward radiative-transfer relationship couples absorption and backscatter
through their *ratio*, so many distinct `(a, bb)` combinations produce nearly
identical `Rrs(λ)` spectra. No amount of algorithmic cleverness removes this
degeneracy; only external information (priors, ancillary data, spectral
constraints) can. Three of the five papers (Sydor 2004, Defoin‑Platel & Chami
2007, Prochaska & Frouin 2025) are devoted to demonstrating and quantifying this
ambiguity; Werdell et al. 2018 reviews the algorithmic landscape that lives with
it; Mobley provides the foundational physics.

## The forward model (the physics everyone shares)

The water-leaving signal relates to IOPs through Gordon's quasi-single-scattering
approximation:

```
Rrs(λ) ≈ f/Q · bb(λ) / [a(λ) + bb(λ)]
       = Σ_i  G_i · u(λ)^i        with   u(λ) = bb(λ) / [a(λ) + bb(λ)]
```

- `a(λ)` total absorption = `a_w` (water) + `a_ph` (phytoplankton) + `a_dg`
  (detritus + CDOM, often modeled together).
- `bb(λ)` total backscatter = `bb_w` (water) + `bb_p` (particles).
- `f/Q` (≈ 0.0949·π-type factors) depend on illumination/viewing geometry and
  the volume scattering function (VSF).

Because `Rrs` depends on `u = bb/(a+bb)`, it is essentially a function of the
**ratio** `bb/a`. Scaling `a` and `bb` together, or trading high-absorption /
low-backscatter against low-absorption / high-backscatter, leaves `Rrs` nearly
unchanged. This is the root of the degeneracy.

Component spectral shapes typically assumed:
- `a_ph(λ)`: exponential slope or Gaussian basis vectors (per pigment/type).
- `a_dg(λ) = a_dg(440)·exp[-S_dg(λ-440)]`, `S_dg ≈ 0.01–0.02 nm⁻¹`.
- `bb_p(λ) = bb_p(λ₀)·(λ₀/λ)^n`, power-law exponent `n ≈ 0–3`.

## The references

### 1. Mobley — *Ocean Optics Web Book* (foundational reference)
The standard text for the underlying physics. Defines IOPs (`a`, `b`, `bb`,
`c = a+b`), AOPs (irradiance reflectance `R = Eu/Ed`, remote-sensing reflectance
`Rrs = Lw/Ed`), and the Gordon forward relation `Rrs ∝ bb/(a+bb)`. Most relevant
chapters: **Ch. 4** (AOPs, `Rrs`, `Kd`), **Ch. 8** (optical constituents:
phytoplankton, CDOM, minerals), **Ch. 10** (radiative-transfer solutions,
Gordon normalization). Use as the definitional/physics backbone, not a result.

### 2. Sydor et al. 2004 — *Uniqueness in remote sensing of IOPs* (Appl. Opt. 43:2156)
Empirical demonstration of non-uniqueness using field data (South Africa,
Oregon, Pearl River, Gulf of Mexico). Shows that very different `a(λ)`/`bb(λ)`
combinations fit measured `Rrs(λ)` to `r² ≈ 0.99` while recovering non-physical
(e.g. non-monotonic) `b(λ)` shapes. Concludes `Rrs` alone cannot reliably
determine the magnitude and spectral dependence of `a` (especially `a_ph`), and
that prior knowledge of spectral shapes — or in-situ data, or info from
`λ > 676 nm` where absorption dominates — is required.

### 3. Defoin‑Platel & Chami 2007 — *How ambiguous is the inverse problem in coastal waters?* (JGR 112:C03004)
The rigorous **quantification** of ambiguity. Built 10,000 synthetic
`(Rrs, IOP)` pairs (412–665 nm) via radiative-transfer modeling and defined a
formalism: spectral distance `δ_iop`, condition number `CV` (>>10 ⇒ ill-posed),
spectral neighborhood `N(Rrs)` (ε ≈ 5%), ambiguity rate `R_iop`, mean ambiguity
distance `Δ_iop`, and minimum inverse-problem error (MIPE). Findings: **~90–92%
of spectra are ambiguous**; total absorption error up to ~166% in absorbing
waters, backscatter ~153% in turbid waters; IOP components strongly covary
(`a_chl`–`a_cdom`, `bb_sed`–`a_tot`), locking solutions together. Proposes
mitigations: ancillary/in-situ data, hyperspectral and directional measurements,
and "divide-and-conquer" region/season-specific models.

### 4. Werdell et al. 2018 — *An overview of approaches and challenges for IOP retrieval* (Prog. Oceanogr. 160:186)
The authoritative **review** (large author list; PACE-era). Taxonomy of
semi-analytical algorithms (SAAs), in roughly decreasing flexibility:
1. **Non-linear spectral optimization** (e.g. Levenberg–Marquardt) — flexible,
   costly, prone to local minima (GIOP-style).
2. **Look-up tables (LUTs)** — fast, bounded solution space.
3. **Spectral decomposition / linear inversion** — fixed spectral shapes,
   solve for magnitudes; robust to miscalibration (QAA, Lee et al. 2002).
4. **Bulk inversion** — only total `a`, `bb` at bands; minimal assumptions.

Key challenges echoed throughout: unknowns exceed independent observations;
strong `a_dg`↔`bb_p` degeneracy; dominant measurement uncertainties (`a` ~5–10%,
`bb`/VSF ~2–20%); atmospheric correction contributes up to ~50% error; fixed
spectral-shape assumptions are poorly validated globally; shallow/coastal waters
unsolved. Best practices: run multiple algorithms, check inter-sensor
consistency, pair with in-situ validation, publish uncertainty budgets, go
hyperspectral and region-specific, share open-source code.

### 5. Prochaska & Frouin 2025 — *On the challenges of retrieving phytoplankton properties from remote sensing* (Biogeosciences 22:4705) — the BING paper
The project's own Bayesian framework (this is the user's paper, describing the
**BING** package = *Bayesian INferences with Gordon coefficients*). Uses the
Gordon series forward model with **MCMC (emcee)** to produce full posteriors and
covariances, plus AIC/BIC model selection across five parametrizations of
increasing complexity (`k=2` … `k=free`). Findings: from multispectral data
(e.g. MODIS 8 bands) **no more than ~3 independent parameters** can be reliably
constrained; even PACE-simulated hyperspectral data cannot overcome the
degeneracy without strong shape priors; `k≥4` models improve the `Rrs` fit only
marginally without reducing parameter uncertainty; `a_ph(440)` from MODIS shows
~1 order-of-magnitude scatter/bias; operational GIOP/GSM underestimate `a_ph`
due to fixed `S_dg`. The Bayesian approach's value is **honest uncertainty
quantification**, not defeating the degeneracy.

## Shared notation

| Symbol | Meaning |
|--------|---------|
| `Rrs(λ)` | remote-sensing reflectance, `Lw/Ed` (sr⁻¹) |
| `a(λ)` | total absorption (m⁻¹); = `a_w + a_ph + a_dg` |
| `bb(λ)` | total backscatter (m⁻¹); = `bb_w + bb_p` |
| `a_ph`, `a_dg`, `a_cdom`, `a_w` | phytoplankton / detritus+CDOM / CDOM / water absorption |
| `bb_p`, `bb_w` | particle / water backscatter |
| `u = bb/(a+bb)` | the quantity `Rrs` actually constrains |
| `f/Q` | geometry/VSF reflectance factor |
| `S_dg` | CDOM+detritus exponential spectral slope (~0.01–0.02 nm⁻¹) |
| `n` | `bb_p` power-law exponent (~0–3) |
| `Kd` | diffuse attenuation coefficient |
| GIOP, GSM, QAA | operational semi-analytical algorithms |
| BING | this project's Bayesian (Gordon + MCMC) retrieval package |
| AIC, BIC | model-selection information criteria |

## Implications for this project

1. The enemy is **non-uniqueness/degeneracy**, quantified and unavoidable from
   `Rrs` alone — the AI angle must add *information*, not just a better fit.
2. Multispectral data support only ~3 free parameters; hyperspectral (PACE)
   helps but does not by itself break the degeneracy.
3. The leverage points the literature keeps pointing to: **priors on spectral
   shapes** (especially `S_dg`), **ancillary/in-situ data**, **region/season-
   specific models**, and **honest uncertainty quantification** (BING's stance).
4. Open question for an AI approach: can learned priors over the *joint*
   `(a_ph, a_dg, bb_p)` shape space supply exactly the external constraint the
   degeneracy requires — and can we tell when they're hallucinating vs. informed?

## Project decisions (from Q&A, 2026-06-30)

JXP's answers to the questions above fix the following scope and direction. These
are the working decisions for retrieve-or-bust; some are explicitly provisional.

- **AI thesis — where the external information comes from.** The degeneracy is
  broken with **priors derived from (a) in-situ observations, (b) environmental
  variables, and (c) time-series information**. This is the project's core bet:
  supply the missing information the inversion physically requires via these
  priors. ("And then hope for the best!")

- **Method — open.** No fixed commitment to BING's Bayesian/MCMC engine or to
  any one paradigm. **Claude is to generate the best approach it can** —
  Bayesian inference, deep learning, or a hybrid — and we evaluate on merit.
  BING (Prochaska & Frouin 2025) is the incumbent baseline to learn from and
  benchmark against, not a constraint.

- **Success criterion (provisional).** Success = **reliably retrieving > 4
  independent parameters from hyperspectral data**. This is a deliberately
  ambitious bar (the literature says multispectral supports only ~3, and even
  hyperspectral struggles past that without strong priors). **To be revisited**
  as the work matures.

- **Sensor targets.** **Hyperspectral (PACE/OCI) is the primary target**, but
  the method **must also work for legacy multispectral (MODIS, SeaWiFS)**.

- **Retrieval goal — component separation.** We target full **component
  separation** (`a_ph`, `a_dg`, `bb_p`), not just total `a`/`bb` — the harder,
  more ambiguous problem, which is exactly why strong priors are central.

- **Data.** A **range of datasets** will be used; the concrete set follows what
  JXP is already assembling in the **IOPtics repository** (e.g. L23 synthetic,
  PANGAEA/Valente and GLORIA/Lehmann in-situ, and PACE-era observations). Ground
  truth and loaders to be inherited/adapted from there.

- **Priors policy.** The boundary between a legitimate informative prior and
  baking in the answer is **TBD**; the working stance is to **leverage priors as
  aggressively as possible** and revisit the legitimacy/circularity question
  later (ties directly to the in-situ / environmental / time-series priors
  above).

- **Mobley book.** Treated purely as a **physics reference** to consult as
  needed; nothing to extract into the repo for now.

## Related work from the literature (web search, 2026-06-30)

Additional peer-reviewed work beyond the five `context/` PDFs, gathered to widen
the background. DOIs are given where found on authoritative sources.
**Verification note:** ✓ = DOI independently resolved and confirmed during this
pass; entries marked *(DOI unverified)* are as reported by web search and were
**not** independently confirmed — check before citing. Future-dated 2026 entries
are early-access/online-first.

### Foundational forward model & semi-analytical algorithms (SAAs)

- **Gordon, H. R., et al.** (1988). "A semianalytic radiance model of ocean
  color." *J. Geophys. Res.* 93(D9), 10909–10924.
  DOI: 10.1029/JD093iD09p10909. — The radiative-transfer basis for the
  `Rrs ∝ bb/(a+bb)` forward model underlying BING and most SAAs.
- **Garver, S. A., & Siegel, D. A.** (1997). "Inherent optical property
  inversion of ocean color spectra... 1. Time series from the Sargasso Sea."
  *J. Geophys. Res.* 102(C8), 18607–18625. DOI: 10.1029/96JC03243. — Precursor
  to GSM; early nonlinear inversion showing spectral-shape assumptions are
  needed for uniqueness.
- **Lee, Z. P., Carder, K. L., & Arnone, R. A.** (2002). "Deriving inherent
  optical properties from water color: a multiband quasi-analytical algorithm
  for optically deep waters." *Appl. Opt.* 41(27), 5755–5772.
  DOI: 10.1364/AO.41.005755. — **QAA**, the canonical stepwise SAA.
- **Maritorena, S., Siegel, D. A., & Peterson, A. R.** (2002). "Optimization of
  a semianalytical ocean color model for global-scale applications."
  *Appl. Opt.* 41(15), 2705–2714. DOI: 10.1364/AO.41.002705 ✓. — **GSM**, global
  simultaneous retrieval of Chl, a_cdm, bb_p.
- **Werdell, P. J., et al.** (2013). "Generalized ocean color inversion model
  for retrieving marine inherent optical properties." *Appl. Opt.* 52(10),
  2019–2037. DOI: 10.1364/AO.52.002019. — **GIOP**, the modular SAA framework
  (NASA operational); the natural baseline for component retrieval.
- **Loisel, H., & Stramski, D.** (2000). "Estimation of the inherent optical
  properties of natural waters from the irradiance attenuation coefficient and
  reflectance in the presence of Raman scattering." *Appl. Opt.* 39(18),
  3001–3011. DOI: 10.1364/AO.39.003001.
- **Loisel, H., et al.** (2018). "An inverse model for estimating the optical
  absorption and backscattering coefficients of seawater from remote-sensing
  reflectance over a broad range of oceanic and coastal marine environments."
  *J. Geophys. Res. Oceans* 123(4), 2141–2171. DOI: 10.1002/2017JC013632. —
  **LS2** inverse model; retrieves IOPs with minimal spectral-shape assumptions.

### Ill-posedness, uncertainty & error propagation

- **Wang, P., Boss, E. S., & Roesler, C.** (2005). "Uncertainties of inherent
  optical properties obtained from semianalytical inversions of ocean color."
  *Appl. Opt.* 44(19), 4074–4085. DOI: 10.1364/AO.44.004074. — Quantifies
  solution non-uniqueness via ensembles within the Rrs error envelope.
- **Lee, Z. P., et al.** (2010). "Uncertainties of optical parameters and their
  propagations in an analytical ocean color inversion algorithm." *Appl. Opt.*
  49(3), 369–381. DOI: 10.1364/AO.49.000369. — Analytical error propagation for
  QAA-derived IOPs.
- **Moore, T. S., Campbell, J. W., & Dowell, M. D.** (2009). "A class-based
  approach to characterizing and mapping the uncertainty of the MODIS ocean
  chlorophyll product." *Remote Sens. Environ.* 113(11), 2424–2430.
  *(DOI unverified)* — Optical-water-type-dependent uncertainty; ties to the
  water-type idea below.

### Priors, ancillary data, Bayesian inversion, water-type & time-series constraints

*(Most directly relevant to this project's "priors break the degeneracy" thesis.)*

- **Bisson, K. M., et al.** (2023). "Informing ocean color inversion products by
  seeding with ancillary observations." *Opt. Express* 31(24), 40557–40572.
  DOI: 10.1364/OE.503496. — **Seeding GIOP with ancillary bb measurements as
  priors cuts seasonal absorption biases >50%** — almost a direct proof of
  concept for our in-situ-prior strategy.
- **Erickson, Z. K., McKinna, L., Werdell, P. J., & Cetinić, I.** (2023).
  "Bayesian approach to a generalized inherent optical property model."
  *Opt. Express* 31(14), 22790–22801. DOI: 10.1364/OE.486581. — Bayesian GIOP;
  trades off deviation-from-prior against observational error to retrieve more
  shape parameters. The closest published cousin to BING + learned priors.
- **Mukherjee, S., Mabit, R., & Bélanger, S.** (2026). "A semi-analytical
  Bayesian estimate retrieval algorithm (SABER) for the inversion of
  remote-sensing reflectance in optically deep and shallow waters."
  *Limnol. Oceanogr. Methods* 24(2), 70–86. DOI: 10.1002/lom3.70004
  *(future-dated/online-first)*. — Bayesian inversion with prior distributions
  across deep + shallow waters.
- **Singh, S., et al.** (2023). "Quantifying Uncertainties in OC-SMART Ocean
  Color Retrievals: A Bayesian Inversion Algorithm." *Algorithms* 16(6), 301.
  DOI: 10.3390/a16060301.
- **Bi, S., & Hieronymi, M.** (2024). "Holistic optical water type
  classification for ocean, coastal, and inland waters." *Limnol. Oceanogr.*
  69(7), 1547–1561. DOI: 10.1002/lno.12606. — Water-type framework to select/
  constrain retrievals per optical regime (a "divide-and-conquer" prior).
- **Werdell, P. J., & Bailey, S. W.** (2005). "An improved in-situ bio-optical
  data set for ocean color algorithm development and satellite data product
  validation." *Remote Sens. Environ.* 98(1), 122–140.
  DOI: 10.1016/j.rse.2005.07.001. — **NOMAD**, a primary in-situ source for
  building empirical priors.
- **Gregg, W. W., & Conkright, M. E.** (2001). "Global seasonal climatologies of
  ocean chlorophyll: Blending in situ and satellite data..." *J. Geophys. Res.
  Oceans* 106(C2), 2499–2515. DOI: 10.1029/2000JC000543. — Climatological
  blending — a template for environmental/seasonal priors.
- **Sauzède, R., et al.** (2016). "A neural network-based method for merging
  ocean color and Argo data to extend surface bio-optical properties to depth:
  Retrieval of the particulate backscattering coefficient." *J. Geophys. Res.
  Oceans* 121, 1762–1777. DOI: 10.1002/2015JC011408. — NN fusing ocean color +
  in-situ (Argo) — an example of the ancillary/time-series fusion we plan.

### Machine / deep learning approaches

- **Pahlevan, N., et al.** (2020). "Seamless retrievals of chlorophyll-a from
  Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A
  machine-learning approach." *Remote Sens. Environ.* 240, 111604.
  DOI: 10.1016/j.rse.2019.111604. — Foundational **Mixture Density Network
  (MDN)** ocean-color paper; MDNs give predictive distributions (uncertainty).
- **Smith, B., et al.** (2021). "A Chlorophyll-a Algorithm for Landsat-8 Based
  on Mixture Density Networks." *Front. Remote Sens.* 1, 623678.
  DOI: 10.3389/frsen.2020.623678.
- **Balasubramanian, S. V., et al.** (2025). "Mixture density networks for
  re-constructing historical ocean-color products over inland and coastal
  waters." *Front. Remote Sens.* 6, 1488565. DOI: 10.3389/frsen.2025.1488565.
- **Lou, J., et al.** (2025). "Variational Autoencoder Framework for
  Hyperspectral Retrievals (Hyper-VAE) of Phytoplankton Absorption and
  Chlorophyll a... for NASA's EMIT and PACE Missions." arXiv:2504.13476.
  DOI: 10.48550/arXiv.2504.13476. — Generative/latent-variable approach for
  hyperspectral a_ph — relevant to learned priors over IOP shapes.
- **Pahlevan, N., et al.** (2024). "Hyperspectral retrievals of phytoplankton
  absorption and chlorophyll-a... using a hybrid mixture-of-experts +
  Variational Autoencoder framework." *Remote Sens. Environ.* 308, 114215.
  *(DOI unverified — a guessed DOI 404'd; confirm before citing.)*
- **Fasnacht, Z., et al.** (2022). "Using Machine Learning for Timely Estimates
  of Ocean Color Information From Hyperspectral Satellite Measurements in the
  Presence of Clouds, Aerosols, and Sunglint." *Front. Remote Sens.* 3, 846174.
  DOI: 10.3389/frsen.2022.846174.
- **Krasnopolsky, V., et al.** (2013). "Deriving ocean color products using
  neural networks." *Remote Sens. Environ.* 134, 235–244. *(DOI unverified)* —
  Early NN ocean-color retrieval.

### Benchmark datasets & mission

- **Loisel, H., et al.** (2023). "A synthetic optical database generated by
  radiative transfer simulations in support of studies in ocean optics and
  optical remote sensing of the global ocean." *Earth Syst. Sci. Data* 15,
  3711–3731. DOI: 10.5194/essd-15-3711-2023. — **L23**, the synthetic
  hyperspectral IOP/AOP database used as ground truth.
- **Lehmann, M. K., et al.** (2023). "GLORIA – A globally representative
  hyperspectral in situ dataset for optical sensing of water quality."
  *Sci. Data* 10, 100. DOI: 10.1038/s41597-023-01973-y. — In-situ hyperspectral
  Rrs (7,572 spectra).
- **Valente, A., et al.** (2022). "A compilation of global bio-optical in situ
  data for ocean-colour satellite applications – version three."
  *Earth Syst. Sci. Data* 14, 5737–5770. Dataset DOI: 10.1594/PANGAEA.941318. —
  Global in-situ Rrs/Chl/IOP compilation (the PANGAEA V3 set).
- **Meister, G., et al.** (2024). "The Ocean Color Instrument (OCI) on the PACE
  Mission: System Design and Prelaunch Radiometric Performance." *IEEE Trans.
  Geosci. Remote Sens.* 62, 4402524. DOI: 10.1109/TGRS.2024.3380416. — The
  primary hyperspectral target sensor (340–895 nm), launched Feb 2024.
