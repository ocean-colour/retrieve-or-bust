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

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

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
