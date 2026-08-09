# GO-SHIP: program summary and optical-measurement inventory

*Prepared 2026-08-09 as context for the Schmidt Sciences VICC proposal. Task: explore
https://www.go-ship.org/ and assess (i) what GO-SHIP is, (ii) the extent of its
backscattering measurements, (iii) the extent of its radiometry, and (iv) its cp
(beam attenuation) measurements.*

> **A note on sources.** As of 2026-08-09 the official site `www.go-ship.org` is not
> serving program content — the domain resolves to a DreamHost parking page, and
> subsidiary pages (e.g. `HydroMan.html`, `DatReq.html`) return 404. This summary is
> therefore built from the US program site (`usgoship.ucsd.edu`), the GO-SHIP program
> description paper (Sloyan et al. 2019 [1]), the GOOS network page [2], the SCOR
> Working Group 154 report on plankton/bio-optical measurements for GO-SHIP [3], the
> Bio-GO-SHIP program paper (Clayton et al. 2022 [4]), NASA OB.DAAC/SeaBASS holdings
> [5,6], and the Gardner–Mishonov–Richardson transmissometer literature [7–10].

---

## 1. What is GO-SHIP?

The **Global Ocean Ship-based Hydrographic Investigations Program** is the
internationally coordinated network of **sustained, decadal-repeat, full-depth,
coast-to-coast hydrographic sections** — the successor to the WOCE Hydrographic
Programme / JGOFS survey (1986–1996) and the CLIVAR Repeat Hydrography program
(2000s), formalized in the late 2000s under GOOS/GCOS and the IOCCP. Key facts:

- **Network**: ~55 sustained reference lines spanning all major ocean basins, each
  nominally reoccupied about once per decade (some more frequently). Roughly 12
  contributing nations (US, UK, Japan, Canada, Germany, Spain, Australia, Norway,
  France, South Africa, Ireland, Sweden); the US performs about one-third of the
  survey [1,2]. Order-100 cruises completed or planned since the current decadal
  survey structure began.
- **What it is for**: the only observing-system element that delivers
  **climate-quality, full-water-column** (surface to bottom, including below the
  2000 m Argo horizon) measurements of physical and biogeochemical state —
  quantifying decadal change in heat, freshwater, carbon, oxygen, nutrients, and
  transient tracers, and serving as the reference/calibration standard for
  autonomous platforms (Argo, BGC-Argo) and models [1].
- **Measurement hierarchy**: **Level 1** (core, mandatory on every cruise):
  full-depth CTD (T, S, P), discrete + CTD oxygen, nutrients, the inorganic carbon
  system (DIC, total alkalinity, pH — at least two of three), CFCs/SF₆, and
  underway + lowered ADCP velocity. **Level 2**: highly desirable (e.g. discrete
  ¹⁴C, N₂O, transmissometer on many lists); **Level 3**: ancillary/experimental —
  which is where essentially all bio-optics and biology have historically lived
  [1,3].
- **Data**: publicly archived, principally at CCHDO (`cchdo.ucsd.edu`) for
  CTD/bottle data; bio-optical extensions now flow to NASA SeaBASS/OB.DAAC [5,6].

The decade-scale synthesis of the first GO-SHIP survey (heat, carbon, ventilation
change) is Talley et al. 2016 [11].

**Bio-GO-SHIP** [4] is the ongoing effort (formal pilot 2022–) to add systematic
biological/bio-optical observations to GO-SHIP: HPLC pigments, POC, flow cytometry,
omics, imaging, and underway inherent optical properties. It is the main vehicle by
which optics beyond the transmissometer is entering the program.

---

## 2. To what extent has GO-SHIP measured backscattering?

**Historically: essentially not at all. Recently: rapidly growing, but young, sparse,
and single-wavelength on the core platform.**

- Backscattering has never been a Level 1 or Level 2 GO-SHIP measurement. The SCOR
  WG 154 report [3] lists "optical scattering sensors" among *recommended*
  rosette-mounted bio-optical additions — i.e., as of that assessment they were an
  aspiration, not a program standard.
- **Rosette bbp is now appearing via the BGC-Argo/GO-BGC connection.** US GO-SHIP
  cruises are the primary deployment platform for GO-BGC/BGC-Argo floats (which
  carry bbp(700) sensors), and recent US cruises mount a WET Labs ECO **FLBB**
  (chlorophyll fluorescence + bbp at 700 nm, single angle) on the rosette at every
  station, explicitly to validate the float sensors — e.g. the GO-BGC North
  Atlantic cruise plans specify "full depth Rosette/CTD/LADCP/FLBB/Transmissometer
  casts at every station" [12]. This yields full-depth, single-wavelength,
  single-angle bbp profiles on the lines occupied since roughly 2021 — valuable,
  but not a multi-decade nor multi-spectral record, and factory-calibration
  consistency across ECO-class sensors is a known weak point (cf. the EXPORTS
  finding that concurrent bbp calibrations disagreed at 95% confidence; see
  `reports/exports_summary.md`).
- **Underway (surface) multi-spectral bbp via Bio-GO-SHIP.** The SeaBASS
  BIO-GO-SHIP experiment [6] archives flow-through data including **bbp and cp**
  (with HPLC pigments, imaging, absorption) from six recent occupations: A16N 2023
  (two legs), I05 2023, I08S 2024, I09N, and P02 (PIs Graff, Mitchell, Poulton).
  This is near-surface only (ship intake, ~5–7 m) but hyperspectral/multi-spectral
  and directly satellite-relevant.

**Bottom line**: GO-SHIP contributes bbp mainly *indirectly* — as the deployment and
validation backbone of the BGC-Argo bbp array — plus a young (post-2021) rosette
FLBB record and a handful of Bio-GO-SHIP underway transects. There is no long,
quality-controlled, uncertainty-characterized global bbp climatology from GO-SHIP
itself.

---

## 3. To what extent does GO-SHIP have radiometry?

**Effectively none, in the ocean-color sense.**

- The only radiometric quantity with any standing in the program is **above-water
  downwelling PAR** from ship meteorological sensors, which WG 154 recommends as a
  cheap ship-mounted standard [3].
- There is **no in-water radiometry** (profiled Ed/Lu), **no above-water
  remote-sensing reflectance**, and no ocean-color radiometry protocol in GO-SHIP's
  measurement levels. The SeaBASS Bio-GO-SHIP holdings [6] contain no radiometric
  parameters. Clayton et al. [4] mention multispectral downwelling irradiance only
  generically among possible future additions.
- For contrast: the programs that do supply shipborne hyperspectral radiometry for
  satellite validation are outside GO-SHIP — e.g. AMT (fiducial-reference
  above-water radiometry plus underway ACS IOPs [13]) and the PACE validation
  ships-of-opportunity effort SO-PACE (pySAS hyperspectral Rrs, 350–750 nm, on
  research vessels since 2024 [5,14]). GO-SHIP lines would be a natural host for
  exactly this kind of low-cost autonomous package, and (as far as this survey
  found) are not yet one.

---

## 4. Cp (beam attenuation) measurements

**This is GO-SHIP's one deep, long optical record — arguably the most underexploited
global particle/optics dataset in existence.**

- Beam transmissometers (SeaTech, then WET Labs/Sea-Bird **C-Star, 660 nm**) have
  been interfaced to the CTD on US repeat-hydrography sections **since WOCE
  (late 1980s)**, continuing through CLIVAR Repeat Hydrography into GO-SHIP — i.e.,
  full-depth cp(660) profiles at essentially every station on the US-occupied
  lines for nearly four decades. Formally the transmissometer sits at Level 2/3
  (not mandatory internationally), so coverage is strongest on US cruises [1,3].
- The record has been consolidated by Gardner, Mishonov & Richardson into the
  **Global Transmissometer Database** (V3, 2020): a quality-controlled ODV
  collection built from WOCE/SAVE/JGOFS/CLIVAR/GO-SHIP cruises — order 10⁴
  full-depth profiles from >70 cruises, with beam-cp present in ~85% of casts
  [7]. From it they have produced global assessments of benthic nepheloid layers
  [8], **decadal repeat-transect comparisons of particulate matter** in the
  Atlantic, Pacific, and Indian basins [9], and (with satellite data) global POC
  climatologies [10].
- **cp → carbon**: cp(660) is a well-established proxy for POC (tight local
  regressions; [10], Cetinić et al. 2012 [15]) and its spectral slope carries
  particle-size information. The GO-SHIP cp archive is thus a full-depth,
  decade-spanning POC-proxy record that connects the satellite-visible surface
  layer to the deep ocean.
- **Caveats**: single wavelength; calibration/air-cal drift and inter-cruise
  offsets require the kind of careful QC Gardner et al. performed (much of the raw
  CCHDO transmissometer data is flagged unprocessed/uncalibrated); the cp→POC
  conversion varies regionally and is a genuine uncertainty term; and the program
  itself publishes voltage/attenuation, not carbon.

---

## 5. Relevance to our proposal (brief)

1. **The cp archive is a ready-made, four-decade, full-depth particle record** on
   repeat lines with coincident Level 1 carbon-system data — a natural constraint
   and validation set for the depth-resolved side of our POC/Cphyto retrieval
   story, and a bridge between satellite bbp/cp proxies and the interior carbon
   inventory that ECCO-Darwin carries.
2. **GO-SHIP is the calibration backbone of BGC-Argo bbp** — the float array we
   name as our subsurface extension. Its rosette FLBB + bottle (POC, HPLC) casts
   are precisely the traceability chain our uncertainty budget needs.
3. **The radiometry gap is an opportunity, not a resource**: GO-SHIP cannot supply
   Rrs validation data today; conversely, low-cost autonomous radiometry (pySAS
   class) on GO-SHIP lines is an obvious, cheap enhancement should we want to
   propose one — the platform, berths, and data system already exist.
4. **Bio-GO-SHIP is the institutional on-ramp** for adding optics to GO-SHIP; its
   SeaBASS pipeline (bbp, cp, absorption, HPLC, POC on recent A16N/I05/I08S/I09N/
   P02 occupations) is the template a satellite-era in-situ optics contribution
   would follow.

---

## References

1. Sloyan BM, Wanninkhof R, Kramp M, et al. The Global Ocean Ship-Based
   Hydrographic Investigations Program (GO-SHIP): a platform for integrated
   multidisciplinary ocean science. *Front Mar Sci.* 2019;6:445.
   doi:10.3389/fmars.2019.00445
2. GOOS Observations Coordination Group, GO-SHIP network page.
   https://goosocean.org/who-we-are/observations-coordination-group/global-ocean-observing-networks/global-ocean-ship-based-hydrographic-investigations-programme-go-ship/
3. Boss E, Waite AM, Uitz J, et al. (SCOR Working Group 154). Recommendations for
   plankton measurements on the GO-SHIP program with relevance to other sea-going
   expeditions. SCOR WG 154 report.
   https://misclab.umeoce.maine.edu/documents/GO-SHIP_report_draft_final.pdf
4. Clayton S, Alexander H, Graff JR, et al. Bio-GO-SHIP: the time is right to
   establish global repeat sections of ocean biology. *Front Mar Sci.*
   2022;9:767443. doi:10.3389/fmars.2021.767443
5. NASA Earthdata / OB.DAAC catalog entries: GO-SHIP
   (https://www.earthdata.nasa.gov/data/catalog/ob-daac-go-ship-0), Bio-GO-SHIP
   (https://www.earthdata.nasa.gov/data/catalog/ob-daac-bio-go-ship-0), SO-PACE
   (https://www.earthdata.nasa.gov/data/catalog/ob-daac-pvst-sopace-0).
6. SeaBASS BIO-GO-SHIP experiment page.
   https://seabass.gsfc.nasa.gov/experiment/BIO-GO-SHIP
7. Gardner WD, Mishonov AV, Richardson MJ. Global Transmissometer Database V3.
   2020. https://odv.awi.de/data/ocean/global-transmissometer-database/
8. Gardner WD, Richardson MJ, Mishonov AV. Global assessment of benthic nepheloid
   layers and linkage with upper ocean dynamics. *Earth Planet Sci Lett.*
   2018;482:126–134. doi:10.1016/j.epsl.2017.11.008
9. Gardner WD, Mishonov AV, Richardson MJ. Decadal comparisons of particulate
   matter in repeat transects in the Atlantic, Pacific, and Indian Ocean basins.
   *Geophys Res Lett.* 2018;45:277–286. doi:10.1002/2017GL076571
10. Gardner WD, Mishonov AV, Richardson MJ. Global POC concentrations from in-situ
    and satellite data. *Deep-Sea Res II.* 2006;53:718–740.
    doi:10.1016/j.dsr2.2006.01.029
11. Talley LD, Feely RA, Sloyan BM, et al. Changes in ocean heat, carbon content,
    and ventilation: a review of the first decade of GO-SHIP global repeat
    hydrography. *Annu Rev Mar Sci.* 2016;8:185–215.
    doi:10.1146/annurev-marine-052915-100829
12. GO-BGC North Atlantic cruise plans (rosette/CTD/LADCP/FLBB/transmissometer at
    every station). https://www.go-bgc.org/expedition/north-atlantic-cruise-plans
13. Jordan TM, Dall'Olmo G, Tilstone G, et al. A compilation of surface inherent
    optical properties and phytoplankton pigment concentrations from the Atlantic
    Meridional Transect. *Earth Syst Sci Data.* 2025;17:493–516.
    doi:10.5194/essd-17-493-2025
14. Haëntjens N, Boss E, et al. pySAS: autonomous solar tracking system for
    surface water radiometric measurements. *Oceanography.* 2022.
    doi:10.5670/oceanog.2022.210
15. Cetinić I, Perry MJ, Briggs NT, et al. Particulate organic carbon and inherent
    optical properties during 2008 North Atlantic Bloom Experiment. *J Geophys
    Res.* 2012;117:C06028. doi:10.1029/2011JC007771
