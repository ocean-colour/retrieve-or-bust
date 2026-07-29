# Ocean Biomass & Carbon from Optics — Literature Summary

*Focus: the uncertainty in the **primary carbon measurements** (particulate organic
carbon, phytoplankton carbon, net primary production, and export), as derived from
ocean optics. Prepared to inform the retrieve-or-bust / VICC effort.*

Scope: the `context/papers/Biomass` collection (Behrenfeld 2005, 2016; Graff 2012,
2015; Bisson 2020/21, 2023; Fox 2022; Kulk 2020; Taylor & Landry 2018; Wu 2022/23;
Li 2024; Stoer & Fennel 2024; Brewin et al. 2023 review) plus a 2024–2026 literature
scan. Citations are numbered (Vancouver style); see **References**.

---

## 1. Why this matters, and what the "primary carbon measurements" are

The ocean carbon quantities that ocean color aims to constrain form a chain, and
**each link adds uncertainty**:

```
   Rrs(λ)  →  IOPs (bbp, aph)  →  POC / phytoplankton carbon (Cphyto)  →  NPP  →  export (biological pump)
             [the inversion]      [the conversion]                       [the model]   [attenuation]
```

The headline for a carbon-cycle program: **the optical-carbon products carry large,
often poorly-quantified uncertainties, and much of that uncertainty is inherited
from the two empirical steps in the middle** — the ill-posed IOP inversion and the
non-universal optics→carbon conversion. The global ocean biological carbon pump
(export) is still quoted at **5–15 Gt C yr⁻¹ — the same range as in the 1980s** [1].
That stagnation is the gap a categorical uncertainty-reduction program targets.

Global magnitudes for orientation (Brewin review, Table 1) [1]: POC stock **2.3–4.0
Gt C** (0.58–1.3 Gt C in the mixed layer); phytoplankton carbon **0.78–1.0 Gt C**;
primary production **~50 Gt C yr⁻¹**; export **5–15 Gt C yr⁻¹**; DOC ~662, DIC
~38,000 Gt C.

---

## 2. Uncertainty by carbon quantity

### 2.1 Phytoplankton carbon (Cphyto) from backscatter — the conversion-slope problem

Cphyto is retrieved from the particulate backscattering coefficient `bbp` via a
near-linear scaling, but **the slope is not universal — it is the single largest
uncertainty source.**

- Foundational field regression (Graff 2015) [2]:
  **Cphyto = 12,128·bbp(470) + 0.59** (R² = 0.69, RMSE = 4.6 µg C L⁻¹) — i.e. ~31%
  of variance unexplained even in the calibration set. bbp explains ~20% more
  variance than Chl or POC do.
- Published fixed slopes disagree ~2–4×: **13,000** (Behrenfeld 2005, bbp 440 nm) [3],
  **12,128** (Graff 2015, 470 nm) [2], **30,100** (Martínez-Vicente 2013, cell-volume
  based). Compiled across studies the slope spans **8,372–30,100 (median 15,124)** [4].
- Taxonomy drives the spread. Fox 2022 [5] finds the true scalar ranges
  **3,770–27,697** with community — group values **~8,372** (cyanobacteria/
  haptophytes), **~13,832** (green algae), **~22,641** (diatoms/dinoflagellates). A
  single fixed slope is **25–30% too high** in picoplankton-dominated gyres and
  **~100% too low** in diatom blooms. A composition-adaptive scalar (using the
  absorption ratio anw(690)/anw(580)) improved Cphyto vs flow-cytometry from
  R² = 0.46 to **R² = 0.79**.
- Propagated to the **global stock**, the conversion choice alone spans **~3.5×**:
  applying the min / median / max slopes gives **218 / 390 / 771 Tg C** [4]. Stoer &
  Fennel's BGC-Argo synthesis (99,341 profiles, 903 floats) reports **314 Tg C_phy
  (range 213–414)** with a stated **MAPE ≈ 32%**, and states plainly that
  *"variability in the relationship between C_phy and bbp is the main source of
  uncertainty"* [4]. Prior global Cphyto estimates span **250–2,400 Tg C** [4].
- **Non-algal particles (NAP).** `bbp` integrates *all* particles; detritus, bacteria,
  minerals and coccoliths (PIC) contribute to bbp but not to Cphyto. Separating living
  from non-living backscatter is called the central retrieval challenge [1]; per-profile
  "background-bbp" subtraction is imperfect and the NAP:Cphyto ratio varies in space
  and time [4]. Background-bbp offsets used across studies themselves range
  **0.00027–0.00067 m⁻¹** [3,5].
- **Chlorophyll is a weak carbon proxy.** Field Cphyto:Chl spans **31–408 (median
  ~100)** — an order of magnitude — so a single C:Chl gives up to **3× errors** [2];
  C:Chl variability exceeds **1.5 orders of magnitude** [3,6]. Photoacclimation, not
  biomass, drives **>55% of interannual chlorophyll anomalies over >75% of the ocean**
  [6] — chlorophyll and carbon are physiologically decoupled.

### 2.2 Particulate organic carbon (POC) from optics

- The NASA global POC product is an empirical **blue-to-green Rrs band ratio**
  (Stramski 2008); backscatter-based and Kd-based algorithms are alternatives [1].
- A **single** POC–bbp relationship "is subject to high uncertainties because of the
  variable nature of particulate assemblages" [7,8]. Adding chlorophyll as a second
  predictor (Koestner et al. 2024, multivariable) cut POC uncertainty from
  **~47% (bbp-only) to ~28%** — direct evidence that particle *composition* is the
  missing information [8].
- **The in-situ reference is itself biased and un-standardized.** Glass-fibre (GF/F,
  nominal 0.7 µm) filtration misses submicron POC and rare large particles, so
  filter-POC ≠ the total POC that optics "sees"; **no certified reference material
  exists for POC**, so accuracy cannot be cross-checked across labs [1]. Satellite POC
  and BGC-Argo POC are significantly inconsistent at high latitudes in winter [1].
- Standard global POC is generated "indiscriminately with respect to optical water
  type… implies unknown and potentially large uncertainties," worst in coastal/
  high-NAP waters [1].

### 2.3 Net primary production (NPP) — model spread and the biomass lever

- **Global NPP estimates span ~32–79 Pg C yr⁻¹** across algorithms [9]; depth/
  wavelength-resolved Chl models cluster at **48.7–52.5 Gt C yr⁻¹** (interannual
  variability only ±2.7%) [10], while the carbon-based CbPM gives ~67 vs the Chl-based
  VGPM's ~60, differing **regionally by −21% to +49%** [3].
- **The biomass/carbon term is the dominant lever.** In the CbPM, NPP = Cphyto·μ, so
  all of §2.1's conversion uncertainty enters directly; "satellite-derived biomass
  estimates can at times be the largest contributor to uncertainties in NPP" [9].
- The physiology parameters are equally large: varying the photosynthesis–irradiance
  parameters by ±1 SD swings global NPP by **−46% to +45% (25.7–76.2 Gt C yr⁻¹)**
  [10]; in absorption-based models the quantum-yield term ϕ_m dominates the total
  sensitivity (**S_T = 0.79 > aph 0.41 > PAR 0.25**) [9].
- **Algorithms disagree even on the sign of the trend.** Across six satellite NPP
  algorithms the 1998–2023 global trend ranges from slightly positive (VGPMs,
  SD > mean → unreliable) to **−0.27 to −1.45% yr⁻¹** (CbPM/AbPM/CAFE) [11].
  Absorption-based models (Lee-AbPM, Silsbe-CAFE) have the lowest RMSE vs in-situ ¹⁴C.
  CMIP6 projected ΔNPP to 2100 is **−0.76 ± 3.44 Pg C yr⁻¹** — the SD is >4× the mean,
  i.e. no consensus on sign, and this projection uncertainty has **grown >50% since the
  previous IPCC cycle** [11]. Notably, CbPM *trends* are driven by Chl, not bbp — the
  carbon proxy adds little temporal information [11].
- The underlying carbon-to-chlorophyll ratio (from in-situ autotrophic-carbon census)
  varies **~170→20 (factor ~8–9)** across trophic gradients: AC = 52.9·Chl^0.64 [12].

### 2.4 Export / the biological carbon pump

- Export at 100 m: model ensemble mean **6.08 ± 1.17 Pg C yr⁻¹** (export ratio
  0.154 ± 0.026) vs a hydrographic-data estimate of **10.64 ± 0.80 Pg C yr⁻¹** [13,14]
  — a ~1.7× disagreement between approaches, on top of the review's **5–15 Gt C yr⁻¹**
  spread that has not narrowed in ~40 years [1]. Uncertainty in carbon export
  dominates above ~900 m; transfer efficiency dominates below [14].

### 2.5 Cross-cutting sources (upstream and geometric)

- **Satellite bbp retrieval error** (the input to every Cphyto/POC product): median
  percentage error vs BGC-Argo is **~18% (CALIOP lidar), 24% (MODIS-GIOP), 31% (VIIRS),
  45% (OLCI)**, all biased low; MODIS carries a seasonal bias traced to Rrs [15]. This
  propagates to **±50% basin-scale disagreement in satellite phytoplankton carbon**
  between MODIS and CALIOP [15]. Viewing-angle/phase-function effects alone move Rrs by
  up to **65%** [15]; seeding inversions with an external bbp shifts retrieved
  absorption by **>50%** in some regions/seasons [16].
- **First-optical-depth blindness.** Passive ocean color sees ~90% of its signal from
  one optical depth (1/Kd), yet **~85% of global Cphyto and ~88% of Chl lie below it**,
  and the Chl-max is offset >10 m from the Cphyto-max over ~84% of the ocean [4].
- **Validation scarcity.** Direct Cphyto has "no standard method" and "high
  uncertainties" [1]; the sorting-flow-cytometry method (Graff 2012) depends on
  sheath-DOC blanks (3–39% of signal) while GF/F loses **3.1–6.5× (sometimes >50%)** of
  cells [17]. Cphyto:POC ranges **12–97% (mean ~44%)** [2], so POC is a poor stand-in
  for phytoplankton carbon.

---

## 3. Latest literature (2024–2026 scan)

- **PACE OCI Level-2 Regional Ocean Biogeochemical Properties v3.1** (2025) now ships
  `carbon_phyto` **with a per-pixel uncertainty product** (`carbon_phyto_unc`) and POC
  — a step toward routine, uncertainty-quantified satellite carbon [18].
- **Koestner et al. 2024** — multivariable POC(bbp, Chl) algorithm; composition
  awareness cuts uncertainty ~47% → ~28% [8].
- **Stoer & Fennel 2024 (PNAS)** — carbon-centric global phytoplankton from 903
  BGC-Argo floats; MAPE ~32%; quantifies the conversion-slope and depth-mismatch
  problems [4].
- **Ryan-Keogh, Tagliabue & Thomalla 2025 (Comms. Earth Environ.)** — six-algorithm RS
  NPP trends; CMIP6 sign disagreement; growing projection uncertainty [11].
- **Marine-NPP uncertainty quantification via probability-prediction models**,
  *Biogeosciences* 22:5463 (2025) — explicit per-estimate NPP UQ [19].
- **"Global declines in NPP in the ocean-color era,"** *Nat. Commun.* (2025) [20];
  and **CbPM sensitivity to satellite products** (Kd(490) a key error source),
  *Remote Sens. Environ.* (2024) [21].
- **Phytoplankton-carbon proxies in oligotrophic waters**, *Biogeosciences* 23:2641
  (2026) — proxy variability from community/physiology/NAP [22].
- **BGC-Argo + satellite bbp** water-column reconstruction, *Ocean Sci.* 21:1677 (2025)
  [23]; multi-model global NPP data product, *ESSD* 15:4829 (2023) [24]; biological-
  pump constraints, Doney et al., *GBC* (2024) [25].

The consistent 2024–26 message: the community is (a) moving from Chl to **carbon-
centric, backscatter/absorption-based** retrievals, (b) adding **composition
information** (multivariable, hyperspectral, polarimetry) to break the single-slope
assumption, and (c) beginning to attach **formal per-pixel uncertainties** — exactly
the three levers a retrieve-or-bust-style AI + priors + UQ engine is built to pull.

---

## 4. Synthesis — the dominant uncertainties in the primary carbon measurements

| Measurement | Dominant uncertainty | Magnitude (quantified) |
|---|---|---|
| **Cphyto** (from bbp) | non-universal bbp→C slope (taxonomy/composition) | slopes 8,372–30,100 (~3.5× stock: 218–771 Tg); MAPE ~32% [2,4,5] |
| **Cphyto / POC** | NAP vs living-particle separation in bbp | background-bbp 0.00027–0.00067 m⁻¹; POC MAPE 47%→28% with composition [5,8] |
| **POC** | in-situ reference bias; no CRM; water-type | GF/F loses ~3–6× cells; sat vs Argo inconsistent at high lat [1,17] |
| **Chl as biomass** | photoacclimation / C:Chl variability | C:Chl 31–408 (order of mag); >55% of Chl anomalies are physiology [2,6] |
| **NPP** | biomass term + P–I / ϕ_m physiology; model form | global 32–79 Pg C yr⁻¹; ±1σ P–I → ±~45%; trend sign disputed [9,10,11] |
| **Export (BCP)** | flux magnitude + transfer efficiency | 5–15 Gt C yr⁻¹ (unchanged since 1980s); 6.1 vs 10.6 across methods [1,13,14] |
| **Upstream: satellite bbp** | inversion + geometry + atmospheric correction | MPE 18–45%; ±50% basin-scale Cphyto; Rrs varies 65% with geometry [15,16] |
| **Vertical mismatch** | first-optical-depth blindness | ~85% Cphyto / ~88% Chl below 1/Kd [4] |

**Bottom line for retrieve-or-bust / VICC.** Two of the three largest, most
quantifiable uncertainties in ocean carbon products — the **ill-posed IOP inversion**
(satellite bbp/aph error, §2.5) and the **non-universal optics→carbon conversion**
(§2.1–2.2) — are precisely what an AI + priors + honest-uncertainty engine on PACE-
class hyperspectral is designed to address: composition-aware conversion (breaking the
single-slope assumption), better-conditioned inversions, and calibrated per-pixel
error. The third (subsurface/first-optical-depth) requires BGC-Argo/lidar synergy and
sets an honest scope boundary. A defensible VICC pitch reduces and *characterizes* the
first two, not the physically inaccessible subsurface.

---

## References

1. Brewin RJW, Sathyendranath S, Kulk G, et al. Ocean carbon from space: current status and priorities for the next decade. *Earth-Sci Rev.* 2023;240:104386.
2. Graff JR, Westberry TK, Milligan AJ, et al. Analytical phytoplankton carbon measurements spanning diverse ecosystems. *Deep-Sea Res I.* 2015;102:16–25.
3. Behrenfeld MJ, Boss E, Siegel DA, Shea DM. Carbon-based ocean productivity and phytoplankton physiology from space. *Global Biogeochem Cycles.* 2005;19:GB1006.
4. Stoer AC, Fennel K. Carbon-centric dynamics of Earth's marine phytoplankton. *PNAS.* 2024;121(45):e2405354121.
5. Fox J, Kramer SJ, Graff JR, et al. An absorption-based approach to improved estimates of phytoplankton biomass and net primary production. *Limnol Oceanogr Lett.* 2022;7:419–426.
6. Behrenfeld MJ, O'Malley RT, Boss ES, et al. Revaluating ocean warming impacts on global phytoplankton. *Nat Clim Change.* 2016;6:323–330.
7. Stramski D, Reynolds RA, Kaczmarek S, et al. Improved multivariable algorithms for estimating oceanic particulate organic carbon from optical backscattering and chlorophyll-a. *Front Mar Sci.* 2022/2023.
8. Koestner D, Stramski D, Reynolds RA. A multivariable empirical algorithm for estimating particulate organic carbon from optical backscattering and chlorophyll-a. *Front Mar Sci.* 2024.
9. Wu J, Goes JI, Gomes H do R, Lee Z, et al. Estimates of diurnal and daily net primary productivity using GOCI data. *Remote Sens Environ.* 2022;280:113183.
10. Kulk G, Platt T, Dingle J, et al. Primary production, an index of climate change in the ocean: satellite-based estimates over two decades. *Remote Sens.* 2020;12:826.
11. Ryan-Keogh TJ, Tagliabue A, Thomalla SJ. Global decline in net primary production underestimated by climate models. *Commun Earth Environ.* 2025;6:75.
12. Taylor AG, Landry MR. Phytoplankton biomass and size structure across trophic gradients in the southern California Current and adjacent ecosystems. *Mar Ecol Prog Ser.* 2018;592:1–17.
13. Doney SC, et al. Observational and numerical modeling constraints on the global ocean biological carbon pump. *Global Biogeochem Cycles.* 2024;38:e2024GB008156.
14. Nowicki M, DeVries T, Siegel DA. (biological-pump export magnitude & transfer-efficiency uncertainty). *Global Biogeochem Cycles* / *Nature* hydrographic estimate 2023.
15. Bisson KM, Boss E, Werdell PJ, Ibrahim A, Behrenfeld MJ. Particulate backscattering in the global ocean: a comparison of independent assessments. *Geophys Res Lett.* 2021;48:e2020GL090909.
16. Bisson KM, Werdell PJ, Chase AP, et al. Informing ocean color inversion products by seeding with ancillary observations. *Opt Express.* 2023;31(24):40557–40572.
17. Graff JR, Milligan AJ, Behrenfeld MJ. The measurement of phytoplankton biomass using flow-cytometric sorting and elemental analysis of carbon. *Limnol Oceanogr Methods.* 2012;10:910–920.
18. NASA OB.DAAC. PACE OCI Level-2 Regional Ocean Biogeochemical Properties, v3.1 (incl. carbon_phyto, carbon_phyto_unc, POC). 2025.
19. (Authors). Refining marine net primary production estimates: advanced uncertainty quantification through probability prediction models. *Biogeosciences.* 2025;22:5463.
20. (Authors). Global declines in net primary production in the ocean color era. *Nat Commun.* 2025.
21. (Authors). Sensitivity of a carbon-based primary production model on satellite ocean color products. *Remote Sens Environ.* 2024.
22. (Authors). Potential of optical and ecological proxies to quantify phytoplankton carbon in oligotrophic waters. *Biogeosciences.* 2026;23:2641.
23. (Authors). Combining BGC-Argo floats and satellite observations for water-column estimation of the particulate backscattering coefficient. *Ocean Sci.* 2025;21:1677–1694.
24. (Authors). A new global oceanic multi-model net primary productivity data product. *Earth Syst Sci Data.* 2023;15:4829.
25. Bisson KM, Boss E, Werdell PJ, et al. (2020 preprint / GRL 2021 companion). Global comparison of satellite/lidar/Argo backscatter. See ref. 15.
26. Li Z, Sun D, Wang S, et al. Ocean-scale patterns of environment and climate changes driving global marine phytoplankton biomass dynamics. *Sci Adv.* 2024;10:eadm7556.

*Note on references:* entries 7, 14, and 19–24 are from the 2024–2026 web scan; author
lists / volume details should be verified against the source before use in a formal
(Vancouver) reference list. Entries 1–6, 9–12, 15–17, 26 are from the papers read in
full in `context/papers/Biomass`.
