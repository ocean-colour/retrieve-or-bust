# Dutkiewicz prompts


## Goals

We will use this doc to prompt the exploration of using the Dutkiewicz ocean model outputs as priors for the IOP inversion problem.

## Context

See the following:

- context_summary.md
- The PDFs in the `Oceanography/python/retrieve-or-bust/context/papers` 
   - dutkiewicz2015.pdf : The main model paper
   - dutkiewicz2018.pdf : An application of the model, with emphasis on Chl-a 
   - Loisel_et_al_ESSD_2023.pdf : The Loisel+2023 manuscript

## Prompts

1. Figure 10 of dutkiewicz2018.pdf shows the joint PDF between the concentration of Chl-a and CDOM.  Please see if you can extract those data into a CSV file.  If you use Python, save your script in `Oceanography/python/retrieve-or-bust/robust`. Log your work. Use Fable if you can.  If you are successful, regenerate the figure using the CSV file and a Python script.  Save that script in `Oceanography/python/retrieve-or-bust/robust` or add to the other. 

2. That's great!  Now, please add to your figure the Chl-a and CDOM concentrations used by Loisel+2023.  You can find their data in `$OS_COLOR/Loisel2023`.  And I have put their manuscript in the `context/papers` folder as `Loisel_et_al_ESSD_2023.pdf`.  Please add the data to your figure and update the Python script accordingly.  Log your work.  Use Fable if you can.

3. Raphe Kudela has kindly shared with me a set of CDOM and Chl-a concentration measurements that he has collected.  I put these in `$OS_COLOR/Raphe`.  Please inspect those 2 CSV files.  Then ask me questions in the Q&A section about them before we use them.  Log your work.  Use Fable if you can.

4. I have answered your questions in the Q&A section.  Please proceed to add the data from Raphe Kudela to your figure and update the Python script accordingly.  Show his points in a different color from the Dutkiewicz points and Loisel+2023 points.  Log your work.  Use Fable if you can.

## Q&A

Questions from Claude (2026-08-14) about the two files in `$OS_COLOR/Raphe`,
after inspection (see the log entry of the same date for details):

**Q1 (provenance of CDOM_TChla.csv).** The 789 rows span 2010–2017 with a
2010 cluster in the Chukchi/Beaufort seas (lat 66–72.6 N), later years across
the Pacific (California coast, western Pacific, tropics), and a few early-2013
points near Lake Tahoe. Which cruises/programs are these, and are all samples
surface/near-surface?

**A1.** I don't know. But we won't worry about it for now.

**Q2 (units & methods).** Can you confirm TChla is total chlorophyll-a in
mg m^-3 (HPLC or fluorometric?) and aCDOM(440) is the CDOM absorption
coefficient at 440 nm in m^-1?

**A2.**  I will ask, but let's assume so for now.

**Q3 (repeated values).** Only 242 unique (TChla, aCDOM) pairs occur among
the 789 rows — the same lab pair repeats at up to 10 nearby positions/times.
Are these discrete bottle samples matched to multiple station/underway
records (e.g. satellite match-up replicates)? For building a prior, should we
deduplicate to the unique samples or keep all rows (which implicitly weights
by station occupancy)?

**A3.** Let me ask.

**Q4 (column conventions).** Is SDY the sequential (decimal) day of year in
UTC? And just to confirm the header order: StnLon then StnLat (the values are
consistent with that, but the reversed convention is common enough to check).

**A4.** Let me ask.

**Q5 (GLORIA file).** `GLORIA_meta_and_lab.csv` looks like the published
community GLORIA dataset (Lehmann et al. 2023, Sci. Data) — 7,572 rows from
47 organizations, dominated by lakes and rivers; the UCSC contribution is 34
Pinto Lake rows that have no aCDOM440. 3,182 rows have both Chla and
aCDOM440. Should we use all of those, or restrict to particular
Water_body_type codes (what do codes 1–5 mean in this copy — lake, estuary,
river, coastal, ...)? An open-ocean prior comparison may only want the
coastal/estuarine end members.

**A5.** Let's use them all.

**Q6 (dynamic range).** GLORIA reaches Chla ~ 13,300 mg m^-3 and
aCDOM(440) ~ 25 m^-1 — far beyond the Fig. 10a panel; with our unit
conversion only ~1,300 of the 3,182 joint rows fall inside the Dutkiewicz
axes. When we add these data to the figure, extend the axes to show
everything, or clip to the original panel and report the clipped fraction?

**A6.** Yes, that's correct

**Q7 (unit conversion).** To place the measurements on Fig. 10a's
concentration axes I propose: shift aCDOM(440) to 450 nm with the Darwin
spectral slope (x exp(-0.021 x 10) = 0.81), then divide by
ccdom(450) = 0.18 m^2 (mmol C)^-1, exactly as done for Loisel+2023; Chl-a
used as-is (for GLORIA, the `Chla` column, ignoring `Chla_plus_phaeo`).
Sound right?

**A7.**  Good choice.  Let's do that.

**Q8 (QC).** GLORIA has 1 row with Chla <= 0 and 6 with aCDOM440 <= 0,
which I would drop. Any other exclusions you want (method flags,
Special_event_flag, distance-to-river, minimum sample depth, ...)?

**A8.** That sounds fine.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-08-14 (Added the Kudela in situ data to the Fig. 10a comparison)

Executed prompt 4 with Claude Fable 5, following the Q&A answers (assume
mg m^-3 / m^-1; use all GLORIA rows with both Chla and aCDOM440; the
agreed 440->450 nm Darwin conversion; drop only non-positive values).

**Script changes.** `robust/dutkiewicz2018_fig10.py` gained a `raphe`
subcommand that writes `robust/data/raphe_chl_cdom.csv` (source, Chl,
aCDOM440, CDOM) combining both files: `source=field` for CDOM_TChla.csv
(all 789 rows kept — Q3 on deduplication is still open, so no dedup) and
`source=GLORIA` for the 3,182 GLORIA rows with both variables. CDOM
(mmol C m^-3) = aCDOM440 x exp(-0.021 x 10) / 0.18. The QC cut dropped 0
rows — the non-positive values found in prompt 3 all lie in rows missing
the other variable. `plot` now overlays the field data as green filled
triangles and GLORIA as violet open squares (Loisel+2023 stays blue),
extends the axes to hold the full dynamic range (CDOM to ~10^2, Chl to
~2x10^4), and draws the original Fig. 10a panel as a dashed rectangle.

**Result.**

- Kudela field (Pacific/Arctic): 655/789 (83%) inside the original panel.
  The points straddle the upper flank of the Darwin ridge and its
  high-CDOM vertical branch, at systematically higher CDOM than the
  Loisel+2023 cloud for the same Chl — i.e. the in situ data agree with
  Darwin's Chl–CDOM relationship better than with L23's sampling.
- GLORIA (inland/coastal): only 1,310/3,182 (41%) inside the panel. The
  rest extend the phase space continuously up to Chl ~ 1.3x10^4 mg m^-3
  and CDOM(equiv) ~ 100 mmol C m^-3, roughly along the extrapolation of
  the Darwin ridge — optically complex waters the Darwin prior simply
  does not cover.
- Takeaway for the prior: a Darwin-based joint prior looks well-supported
  by open-ocean field data, but would need to be widened/extended (or
  blended with GLORIA-like statistics) for coastal and inland
  applications.

### 2026-08-14 (Inspected the Kudela CDOM/Chl-a files; questions posted in Q&A)

Executed prompt 3 with Claude Fable 5. Inspected the two CSVs in
`$OS_COLOR/Raphe` (no analysis or plotting yet, per the prompt); posted
eight questions in the Q&A section above.

**`CDOM_TChla.csv`** — 789 rows, 6 columns (TChla, aCDOM(440), Year, SDY,
StnLon, StnLat), no missing values, file has a UTF-8 BOM.

- Years 2010–2017 (no 2011/2012). Geography by year: 2010 is an Arctic
  cluster (Chukchi/Beaufort, 66–72.6 N); 2013–2017 span the Pacific rim —
  California coast, western Pacific (~127–150 E), tropics down to 10 N —
  plus a few early-2013 points near Lake Tahoe (-120.07, 39.11).
- TChla 0.056–67.5 (median 0.85), aCDOM(440) 0.001–2.146 (median 0.056);
  assuming mg m^-3 and m^-1 respectively (to be confirmed, Q2).
- Notable structure: only 242 unique (TChla, aCDOM) pairs among 789 rows;
  the same lab pair repeats at up to 10 slightly different positions/times.
  Looks like discrete samples matched to multiple station/underway records
  (Q3 asks how to weight these).

**`GLORIA_meta_and_lab.csv`** — 7,572 rows x 64 columns; recognizably the
published GLORIA community dataset of inland/coastal waters (Lehmann et
al. 2023): metadata + lab measurements (Chla, Chla_plus_phaeo, TSS,
aCDOM440, Turbidity, Secchi, methods, water-body descriptors), 47
contributing organizations, dates 1990–2022, dominated by the USA
(n=3,339), water-body types mostly lakes (4,510) and code-3 (2,426).

- Chla present in 5,132 rows (median 8.0, max 13,297 mg m^-3 —
  hypereutrophic lakes); aCDOM440 in 4,393 rows (median 0.55, max
  25.1 m^-1); both together in 3,182 rows.
- With the Darwin unit conversion, only ~1,300 of those 3,182 fall inside
  the Fig. 10a panel — these data live at far higher Chl and CDOM than the
  open-ocean Darwin/L23 distributions (Q5/Q6 ask about subsetting and axis
  extension).
- QC oddities: 1 row with Chla <= 0, 6 with aCDOM440 <= 0 (Q8).
- The UCSC (Kudela lab) contribution is 34 Pinto Lake rows, none of which
  have aCDOM440 — so the value of this file for the joint Chl/CDOM prior
  comes from the full multi-organization dataset, not the UCSC subset.

**Takeaway.** The two files complement the story so far: CDOM_TChla.csv is
open-ocean/coastal Pacific field truth that can be compared directly against
the Darwin joint PDF, while GLORIA extends the phase space to optically
complex inland/coastal waters mostly outside the Darwin panel. Awaiting
answers in the Q&A section before adding either to the figure.

### 2026-08-10 (Overlaid Loisel+2023 Chl-a/CDOM on the digitized Fig. 10a)

Executed prompt 2 with Claude Fable 5.

**Data.** Used the Loisel et al. (2023) synthetic database netCDFs in
`$OS_COLOR/Loisel2023` (`Hydrolight100.nc`; the 3320 IOP combinations are
identical across the nine RT scenario files). Note: the manuscript was not
yet in `context/papers/` — I found it at `~/Loisel_et_al_ESSD_2023.pdf` and
copied it to `context/papers/Loisel_et_al_ESSD_2023.pdf`.

**Unit conversion.** Fig. 10a's axes are concentrations (Chl in mg m^-3,
CDOM in mmol C m^-3) while the Loisel database provides absorption
coefficients, so:

- Chl = aph(440) / 0.05582 m^2 (mg Chl)^-1 — the Bricaud et al. (1998)
  mean chlorophyll-specific absorption, which is the same conversion
  Loisel et al. use themselves (their Sect. 6.2 / Fig. 7a).
- CDOM = ag(450) / 0.18 m^2 (mmol C)^-1 — inverting the Darwin model's
  CDOM-specific absorption ccdom(450) from Dutkiewicz et al. (2015)
  Table 1, i.e. the CDOM tracer concentration Darwin would need to
  reproduce each Loisel ag(450).

**Script changes.** `robust/dutkiewicz2018_fig10.py` gained a `loisel`
subcommand that writes `robust/data/loisel2023_chl_cdom.csv` (index,
aph440, ag440, ag450, Chl, CDOM for all 3320 samples), and `plot` now
overlays those points (blue open circles) on the digitized joint PDF
whenever that CSV exists. The figure
(`robust/data/dutkiewicz2018_fig10a_regenerated.png`) was regenerated.

**Result / what I learned.**

- Loisel+2023 spans Chl = 0.015–8.4 mg m^-3 and CDOM(equivalent) =
  0.0016–2.2 mmol C m^-3, sitting almost entirely inside the Dutkiewicz
  panel range.
- The Loisel cloud crosses the Darwin ridge at mid concentrations
  (Chl ~ 0.1–1, CDOM ~ 0.03–0.3) but a large fraction of the points lie
  above/left of the Darwin relationship — lower CDOM per unit Chl. This is
  by construction: Loisel et al. deliberately sampled a broad range of
  ag/anw ratios for each aph to cover diverse water types, whereas the
  Darwin model ties CDOM production to DOM (and hence to productivity),
  yielding a much tighter joint distribution. The Darwin joint PDF is
  therefore a genuinely informative prior relative to the flat-ish
  sampling of the L23 database.
- Caveat: both axis conversions carry the uncertainty of the assumed
  specific-absorption coefficients (aph*(440) varies ~2x with community
  composition; ccdom is a single Darwin model constant). Comparisons in
  absorption space (ag vs aph) would avoid this, but Fig. 10a's axes are
  concentrations, so the conversion is unavoidable here.

### 2026-08-10 (Digitized Fig. 10a of Dutkiewicz et al. 2018 — joint PDF of Chl-a vs CDOM)

Executed prompt 1 with Claude Fable 5. Figure 10a (2-D histogram of model
"actual" Chl-a vs CDOM) is stored in the paper PDF as an embedded raster
image (2047x921 px at 626 ppi), which I extracted losslessly with
`pdfimages` rather than re-rasterizing the page.

**Method.** Wrote `robust/dutkiewicz2018_fig10.py` with two subcommands:

- `extract` — pulls the embedded image from page 10 of
  `context/papers/dutkiewicz2018.pdf`, calibrates pixel-to-data coordinates
  from the axis ticks (both axes: 243 px/decade; CDOM = 0.01 at column 421,
  Chl = 1.0 at row 317), and builds a color lookup table by sampling the
  figure's own colorbar (log-scaled, 187 px/decade, N = 1 at row 804, top
  ~2x10^4). Each panel pixel is mapped to a sample count by nearest-color
  match against the LUT; white = empty bin. Pixels whose color doesn't match
  the LUT (axis ticks, the "(a)" label, anti-aliased text edges) are
  rejected, with a dilation step around dark rejects so tick/text halos
  don't leak in as spurious dark-red values (they did on the first pass).
  The per-pixel counts are then rebinned onto a regular 40-bins-per-decade
  log10 grid and written to CSV.
- `plot` — regenerates the figure from the CSV alone, using the exact
  colormap sampled from the original colorbar (saved as a companion LUT CSV).

**Outputs** (all under `robust/data/`):

- `dutkiewicz2018_fig10a_chl_cdom.csv` — 6,535 non-empty bins, long format:
  log10_CDOM, log10_Chl, CDOM (mmol C m^-3), Chl (mg m^-3), count, fraction
  (normalized to sum to 1, i.e. a discrete joint PMF usable as a prior).
- `dutkiewicz2018_fig10_colorbar_lut.csv` — the sampled colorbar.
- `dutkiewicz2018_fig10a_regenerated.png` — the fidelity check.

**Result.** The regenerated figure reproduces Fig. 10a well: the dense
low-Chl/low-CDOM ridge, the bend near CDOM ~ 0.1, the vertical
high-CDOM branch, and the single-instance speckle fringe are all present.
The distribution mode lands at CDOM = 0.043 mmol C m^-3,
Chl = 0.077 mg m^-3, on the darkest part of the ridge as in the original.

**What I learned / caveats.**

- The colorbar's darkest color is (53, 0, 0), not pure black, which is what
  makes text/tick rejection by color distance possible at all.
- The colorbar is labeled in counts (1 to ~2x10^4) even though the caption
  says "log of the fraction"; the CSV reports both raw count and normalized
  fraction, and only ratios matter for use as a prior.
- Counts are recovered from a log color scale spanning ~4.3 decades in
  8-bit color (~187 px, ~43 px/decade of value), so individual bin values
  carry perhaps ~5-10% relative error — fine for a prior, not for
  quantitative reanalysis.
- The Chl axis extends slightly beyond its labels (to ~20 mg m^-3 top,
  down to 0.01 at the bottom frame); the CSV covers the full panel:
  CDOM in [10^-3, 10^0.3], Chl in [10^-2, 10^1.3].
- Panel (b) (Chl vs detritus) was not extracted; the same script needs only
  a second set of panel/axis constants if we want it.
