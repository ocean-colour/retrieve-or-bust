""" Digitize Figure 10a of Dutkiewicz et al. (2018, Biogeosciences 15, 613-630).

Figure 10a is a two-dimensional histogram of Darwin model output:
"actual" Chl-a (mg m^-3) plotted against model CDOM (mmol C m^-3), with the
colour giving the (log-scaled) number of model grid-cell/time samples that
fall in each bin of the log-log phase space.  This joint PDF is a candidate
prior for the IOP inversion problem.

The figure is stored in the paper PDF as an embedded raster image
(2047 x 921 px, 626 ppi).  This script:

1. ``extract`` -- pulls the embedded image out of the PDF (via ``pdfimages``),
   maps every pixel of panel (a) back to (CDOM, Chl) coordinates using the
   axis tick positions, maps every pixel colour back to a sample count using
   the figure's own colorbar as a lookup table, rebins the result onto a
   regular grid in log10 space, and writes it to a CSV file.
2. ``loisel`` -- derives equivalent (Chl, CDOM) pairs for the 3320 IOP
   combinations of the Loisel et al. (2023, ESSD 15, 3711-3731) synthetic
   database and writes them to a companion CSV.  The database provides
   absorption coefficients, not concentrations, so we convert with:

   * Chl-a (mg m^-3) = aph(440) / 0.05582 -- the mean chlorophyll-specific
     phytoplankton absorption of Bricaud et al. (1998), which is the exact
     conversion Loisel et al. use themselves (their Sect. 6.2 / Fig. 7a).
   * CDOM (mmol C m^-3) = ag(450) / 0.18 -- inverting the Darwin model's
     CDOM-specific absorption ccdom(450 nm) = 0.18 m^2 (mmol C)^-1
     (Dutkiewicz et al. 2015, Table 1), i.e. the CDOM concentration the
     Darwin model would need to produce the ag(450) of each Loisel sample.

3. ``raphe`` -- converts the in situ Chl-a/CDOM measurements shared by
   Raphe Kudela (``$OS_COLOR/Raphe``) to the same axes and writes them to a
   companion CSV.  Two sources: ``CDOM_TChla.csv`` (789 Pacific/Arctic field
   samples, 2010-2017) and ``GLORIA_meta_and_lab.csv`` (the GLORIA community
   dataset of optically complex inland/coastal waters, Lehmann et al. 2023,
   Sci. Data 10, 100; rows with both Chla and aCDOM440).  Both report
   aCDOM(440) in m^-1, which is shifted to 450 nm with the Darwin spectral
   slope, exp(-0.021 * 10), before dividing by ccdom(450).  QC: rows with
   non-positive Chl or aCDOM are dropped (see the Q&A section of
   claude_prompts/dutkiewicz_priors.md).

4. ``plot`` -- regenerates the figure from the CSVs alone, as a check that
   the extraction is faithful, overlaying the Loisel et al. (2023) and
   Kudela points on the digitized Dutkiewicz joint PDF when their CSVs are
   present.  If the Kudela data are shown, the axes are extended to hold
   the full GLORIA dynamic range and the original Fig. 10a panel is drawn
   as a dashed rectangle.

Pixel calibration (measured from the embedded image; see claude_prompts/
dutkiewicz_priors.md logs for the derivation):

* Panel (a) interior: columns 178-980, rows 2-802.
* X ticks: CDOM = 0.01 at column 421, 1.0 at column 907 -> 243 px/decade.
* Y ticks: Chl = 10 at row 75, 0.1 at row 561 -> 243 px/decade.
* Colorbar (columns 1874-1955): N = 1e4 at row 56, N = 1 at row 804
  -> 187 px/decade, log-scaled.

Usage (from the repo root, in the ``ocean14`` environment)::

    python robust/dutkiewicz2018_fig10.py extract
    python robust/dutkiewicz2018_fig10.py loisel
    python robust/dutkiewicz2018_fig10.py raphe
    python robust/dutkiewicz2018_fig10.py plot
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = REPO_ROOT / 'context' / 'papers' / 'dutkiewicz2018.pdf'
DATA_DIR = Path(__file__).resolve().parent / 'data'
CSV_PATH = DATA_DIR / 'dutkiewicz2018_fig10a_chl_cdom.csv'
LUT_PATH = DATA_DIR / 'dutkiewicz2018_fig10_colorbar_lut.csv'
FIG_PATH = DATA_DIR / 'dutkiewicz2018_fig10a_regenerated.png'
LOISEL_CSV = DATA_DIR / 'loisel2023_chl_cdom.csv'
RAPHE_CSV = DATA_DIR / 'raphe_chl_cdom.csv'

# External data roots
OS_COLOR = (Path(os.environ.get('OS_COLOR', '~/Oceanography/data/Color'))
            .expanduser())
# Loisel et al. (2023) synthetic database (netCDF files from Dryad,
# doi:10.6076/D1630T); the IOPs are identical across the nine RT scenarios.
LOISEL_NC = OS_COLOR / 'Loisel2023' / 'Hydrolight100.nc'
# In situ measurements shared by Raphe Kudela (UCSC)
RAPHE_DIR = OS_COLOR / 'Raphe'

# Conversions to the Dutkiewicz Fig. 10a axes (see module docstring)
APH_STAR_440 = 0.05582   # m^2 (mg Chl)^-1, Bricaud et al. (1998)
CCDOM_450 = 0.18         # m^2 (mmol C)^-1, Dutkiewicz et al. (2015) Table 1
SCDOM = 0.021            # nm^-1 CDOM spectral slope, Dutkiewicz et al. (2015)
AG440_TO_450 = float(np.exp(-SCDOM * (450 - 440)))   # ~0.81

# ---------------------------------------------------------------------------
# Pixel calibration of the embedded raster (2047 x 921 px)
# ---------------------------------------------------------------------------
IMG_SHAPE = (921, 2047)          # (rows, cols) sanity check after extraction

# Panel (a) interior (frame lines excluded)
PANEL_COLS = (178, 981)          # python slice bounds
PANEL_ROWS = (2, 803)

# Axis calibration: log10(value) = REF_LOG + (pixel - REF_PX) / PX_PER_DECADE
X_REF_PX, X_REF_LOG, X_PX_PER_DECADE = 421.0, -2.0, 243.0   # CDOM = 0.01
Y_REF_PX, Y_REF_LOG, Y_PX_PER_DECADE = 317.0, 0.0, -243.0   # Chl = 1.0 (rows increase downward)

# Colorbar: interior rows 2-802 at (e.g.) column 1914;
# log10(N) = (CB_REF_PX - row) / CB_PX_PER_DECADE with N = 1 at row 804
CB_COL = 1914
CB_ROWS = (2, 803)
CB_REF_PX, CB_PX_PER_DECADE = 804.0, 187.0

# Colour handling
WHITE_THRESH = 250    # all RGB channels >= this -> empty bin (count 0)
MATCH_THRESH = 40.0   # max RGB distance to the colorbar LUT for a valid pixel

# Output binning: 40 bins per decade in log10 space
BINS_PER_DECADE = 40
X_LOG_RANGE = (-3.0, 0.3)        # log10 CDOM span of the panel
Y_LOG_RANGE = (-2.0, 1.3)        # log10 Chl span of the panel


def extract_embedded_image(pdf_path):
    """Extract the Figure 10 raster from page 10 of the paper PDF.

    Uses the poppler ``pdfimages`` utility.  Page 10 holds two embedded
    images; Figure 10 is the 2047 x 921 px one.

    Returns
    -------
    np.ndarray of uint8, shape (921, 2047, 3)
    """
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ['pdfimages', '-f', '10', '-l', '10', '-png',
             str(pdf_path), str(Path(tmpdir) / 'img')],
            check=True)
        for png in sorted(Path(tmpdir).glob('img-*.png')):
            img = np.asarray(Image.open(png).convert('RGB'))
            if img.shape[:2] == IMG_SHAPE:
                return img
    raise RuntimeError(
        f'No embedded image with shape {IMG_SHAPE} found on page 10 of '
        f'{pdf_path}; the PDF may differ from the expected copernicus copy.')


def build_colorbar_lut(img):
    """Sample the figure's colorbar into a (RGB colour -> count) lookup table.

    Returns
    -------
    lut_rgb : (n, 3) float array of colours, top of the colorbar first
    lut_counts : (n,) float array of the corresponding sample counts N
    """
    rows = np.arange(CB_ROWS[0], CB_ROWS[1])
    lut_rgb = img[rows, CB_COL, :].astype(float)
    lut_counts = 10.0 ** ((CB_REF_PX - rows) / CB_PX_PER_DECADE)
    return lut_rgb, lut_counts


def digitize_panel(img, lut_rgb, lut_counts):
    """Map every pixel of panel (a) to (log10 CDOM, log10 Chl, count).

    White pixels are empty bins (count 0).  Pixels whose colour is not close
    to any colorbar colour (axis ticks, the "(a)" label, anti-aliased text)
    are flagged invalid and excluded from the rebinning.

    Returns
    -------
    logx, logy : 2D arrays of pixel-centre coordinates in log10 space
    counts : 2D float array of per-pixel sample counts (NaN where invalid)
    """
    panel = img[PANEL_ROWS[0]:PANEL_ROWS[1],
                PANEL_COLS[0]:PANEL_COLS[1], :].astype(float)
    nrow, ncol = panel.shape[:2]

    cols = np.arange(PANEL_COLS[0], PANEL_COLS[1])
    rows = np.arange(PANEL_ROWS[0], PANEL_ROWS[1])
    logx = X_REF_LOG + (cols - X_REF_PX) / X_PX_PER_DECADE
    logy = Y_REF_LOG + (rows - Y_REF_PX) / Y_PX_PER_DECADE
    logx, logy = np.meshgrid(logx, logy)

    counts = np.full((nrow, ncol), np.nan)

    white = np.all(panel >= WHITE_THRESH, axis=2)
    counts[white] = 0.0

    # Nearest-colour match of the remaining pixels against the colorbar LUT
    todo = ~white
    pix = panel[todo]                                   # (m, 3)
    d2 = ((pix[:, None, :] - lut_rgb[None, :, :]) ** 2).sum(axis=2)
    best = d2.argmin(axis=1)
    dist = np.sqrt(d2[np.arange(len(best)), best])
    vals = lut_counts[best]
    vals[dist > MATCH_THRESH] = np.nan                  # text / tick pixels
    counts[todo] = vals

    # Anti-aliased halos around text and tick marks blend towards dark
    # colormap colours and survive the distance cut.  Grow the rejection
    # around *dark* rejected pixels only (text/tick cores); light rejects
    # are data/background anti-aliasing at the speckle fringe and must not
    # erode their neighbours.
    from scipy.ndimage import binary_dilation
    dark_reject = np.isnan(counts) & (panel.sum(axis=2) < 450)
    counts[binary_dilation(dark_reject, iterations=4)] = np.nan

    # The "(a)" panel label sits over empty background at top left; mask its
    # bounding box outright (absolute image coords, rows 20-145 / cols 195-350).
    counts[max(0, 20 - PANEL_ROWS[0]):145 - PANEL_ROWS[0],
           195 - PANEL_COLS[0]:350 - PANEL_COLS[0]] = np.nan

    n_bad = int(np.isnan(counts).sum())
    print(f'digitize: {counts.size} pixels, {int(white.sum())} empty (white), '
          f'{n_bad} rejected as non-colormap ({100 * n_bad / counts.size:.2f}%)')
    return logx, logy, counts


def rebin(logx, logy, counts):
    """Average the per-pixel counts onto a regular grid in log10 space.

    Empty (zero-count) pixels are averaged in; invalid (NaN) pixels are
    excluded.  Returns bin edges and the 2D gridded mean count.
    """
    nx = int(round((X_LOG_RANGE[1] - X_LOG_RANGE[0]) * BINS_PER_DECADE))
    ny = int(round((Y_LOG_RANGE[1] - Y_LOG_RANGE[0]) * BINS_PER_DECADE))
    xedges = np.linspace(X_LOG_RANGE[0], X_LOG_RANGE[1], nx + 1)
    yedges = np.linspace(Y_LOG_RANGE[0], Y_LOG_RANGE[1], ny + 1)

    good = np.isfinite(counts)
    sample = (logx[good], logy[good])
    total, _, _ = np.histogram2d(*sample, bins=[xedges, yedges],
                                 weights=counts[good])
    npix, _, _ = np.histogram2d(*sample, bins=[xedges, yedges])
    with np.errstate(invalid='ignore'):
        mean = np.where(npix > 0, total / np.maximum(npix, 1), np.nan)
    return xedges, yedges, mean


def write_csv(xedges, yedges, mean):
    """Write the gridded joint PDF to CSV (non-empty bins only, long format)."""
    DATA_DIR.mkdir(exist_ok=True)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    ix, iy = np.nonzero(np.nan_to_num(mean) > 0)
    frac = mean[ix, iy] / np.nansum(mean)

    header = (
        '# Joint PDF of model "actual" Chl-a vs CDOM, digitized from Fig. 10a of\n'
        '# Dutkiewicz et al. (2018), Biogeosciences 15, 613-630,\n'
        '# doi:10.5194/bg-15-613-2018, by robust/dutkiewicz2018_fig10.py.\n'
        '# Bin centres on a regular log10 grid '
        f'({BINS_PER_DECADE} bins per decade).\n'
        '# count    : mean per-native-bin sample count from the colorbar '
        '(relative density)\n'
        '# fraction : count normalized to sum to 1 over all listed bins\n'
        '# CDOM in mmol C m-3; Chl in mg m-3.\n'
        'log10_CDOM,log10_Chl,CDOM,Chl,count,fraction\n')
    with open(CSV_PATH, 'w') as f:
        f.write(header)
        for i, j, fr in zip(ix, iy, frac):
            f.write(f'{xc[i]:.4f},{yc[j]:.4f},{10**xc[i]:.6g},'
                    f'{10**yc[j]:.6g},{mean[i, j]:.6g},{fr:.6e}\n')
    print(f'wrote {len(ix)} bins to {CSV_PATH}')


def write_lut_csv(lut_rgb, lut_counts):
    """Save the sampled colorbar so the plot step can reuse the exact colormap."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(LUT_PATH, 'w') as f:
        f.write('# Colorbar of Dutkiewicz et al. (2018) Fig. 10, sampled '
                'top-to-bottom.\ncount,R,G,B\n')
        for n, (r, g, b) in zip(lut_counts, lut_rgb):
            f.write(f'{n:.6g},{r:.0f},{g:.0f},{b:.0f}\n')
    print(f'wrote colorbar LUT to {LUT_PATH}')


def extract():
    """Run the full extraction: PDF -> digitized panel -> CSV."""
    img = extract_embedded_image(PDF_PATH)
    lut_rgb, lut_counts = build_colorbar_lut(img)
    logx, logy, counts = digitize_panel(img, lut_rgb, lut_counts)
    xedges, yedges, mean = rebin(logx, logy, counts)
    write_csv(xedges, yedges, mean)
    write_lut_csv(lut_rgb, lut_counts)


def extract_loisel():
    """Derive (Chl, CDOM) pairs for the Loisel et al. (2023) database.

    Reads aph(440) and ag(450) for the 3320 IOP combinations and converts
    them to the concentration units of Dutkiewicz et al. (2018) Fig. 10a;
    writes the result to LOISEL_CSV.
    """
    import xarray as xr

    ds = xr.open_dataset(LOISEL_NC)
    aph440 = ds['aph'].sel(Lambda=440).values.astype(float)
    ag440 = ds['ag'].sel(Lambda=440).values.astype(float)
    ag450 = ds['ag'].sel(Lambda=450).values.astype(float)
    chl = aph440 / APH_STAR_440
    cdom = ag450 / CCDOM_450

    DATA_DIR.mkdir(exist_ok=True)
    header = (
        '# Chl-a and CDOM for the 3320 IOP combinations of the Loisel et al.\n'
        '# (2023, ESSD 15, 3711-3731) synthetic database, expressed in the\n'
        '# units of Dutkiewicz et al. (2018) Fig. 10a:\n'
        f'#   Chl (mg m-3)       = aph(440) / {APH_STAR_440}  '
        '(Bricaud et al. 1998)\n'
        f'#   CDOM (mmol C m-3)  = ag(450) / {CCDOM_450}  '
        '(Dutkiewicz et al. 2015 ccdom)\n'
        '# Written by robust/dutkiewicz2018_fig10.py.\n'
        'index,aph440,ag440,ag450,Chl,CDOM\n')
    with open(LOISEL_CSV, 'w') as f:
        f.write(header)
        for i in range(len(chl)):
            f.write(f'{i},{aph440[i]:.6g},{ag440[i]:.6g},{ag450[i]:.6g},'
                    f'{chl[i]:.6g},{cdom[i]:.6g}\n')
    print(f'wrote {len(chl)} samples to {LOISEL_CSV}')
    print(f'  Chl  range {chl.min():.4g} - {chl.max():.4g} mg m-3')
    print(f'  CDOM range {cdom.min():.4g} - {cdom.max():.4g} mmol C m-3')


def extract_raphe():
    """Convert the Kudela in situ measurements to Fig. 10a axes.

    Combines the two files in RAPHE_DIR into RAPHE_CSV with a `source`
    column ('field' = CDOM_TChla.csv, 'GLORIA' = GLORIA_meta_and_lab.csv).
    aCDOM(440) is shifted to 450 nm with the Darwin spectral slope before
    dividing by ccdom(450); rows with non-positive values are dropped.
    """
    import pandas as pd

    field = pd.read_csv(RAPHE_DIR / 'CDOM_TChla.csv', encoding='utf-8-sig')
    field = field.rename(columns={'TChla': 'Chl', 'aCDOM(440)': 'aCDOM440'})
    field['source'] = 'field'

    gloria = pd.read_csv(RAPHE_DIR / 'GLORIA_meta_and_lab.csv',
                         usecols=['Chla', 'aCDOM440'], low_memory=False)
    gloria = gloria.rename(columns={'Chla': 'Chl'}).dropna()
    gloria['source'] = 'GLORIA'

    df = pd.concat([field[['source', 'Chl', 'aCDOM440']],
                    gloria[['source', 'Chl', 'aCDOM440']]], ignore_index=True)
    n0 = len(df)
    df = df[(df.Chl > 0) & (df.aCDOM440 > 0)]
    df['CDOM'] = df.aCDOM440 * AG440_TO_450 / CCDOM_450

    DATA_DIR.mkdir(exist_ok=True)
    header = (
        '# In situ Chl-a and CDOM shared by Raphe Kudela (UCSC), expressed in\n'
        '# the units of Dutkiewicz et al. (2018) Fig. 10a.\n'
        '# source: field  = CDOM_TChla.csv (789 Pacific/Arctic samples, '
        '2010-2017;\n'
        '#                  distinct near-surface samples, Chl-a by HPLC --\n'
        '#                  R. Kudela via Q&A in claude_prompts/'
        'dutkiewicz_priors.md)\n'
        '#         GLORIA = GLORIA_meta_and_lab.csv (Lehmann et al. 2023), '
        'rows with\n'
        '#                  both Chla and aCDOM440\n'
        '# Chl in mg m-3 (as reported);\n'
        f'# CDOM (mmol C m-3) = aCDOM440 * exp(-{SCDOM}*10) / {CCDOM_450}\n'
        '#   (shift to 450 nm with the Darwin spectral slope, then invert '
        'ccdom(450))\n'
        '# QC: rows with non-positive Chl or aCDOM440 dropped.\n'
        '# Written by robust/dutkiewicz2018_fig10.py.\n'
        'source,Chl,aCDOM440,CDOM\n')
    with open(RAPHE_CSV, 'w') as f:
        f.write(header)
        for row in df.itertuples(index=False):
            f.write(f'{row.source},{row.Chl:.6g},{row.aCDOM440:.6g},'
                    f'{row.CDOM:.6g}\n')
    print(f'wrote {len(df)} rows to {RAPHE_CSV} '
          f'({n0 - len(df)} dropped by QC)')
    for src, sub in df.groupby('source'):
        print(f'  {src:7s} n={len(sub):4d}  Chl {sub.Chl.min():.4g} - '
              f'{sub.Chl.max():.4g} mg m-3,  CDOM {sub.CDOM.min():.4g} - '
              f'{sub.CDOM.max():.4g} mmol C m-3')


def plot():
    """Regenerate Fig. 10a from the CSV as a fidelity check."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    import pandas as pd

    df = pd.read_csv(CSV_PATH, comment='#')
    lut = pd.read_csv(LUT_PATH, comment='#')

    # Rebuild the regular grid from the long-format table
    xc = np.round(df['log10_CDOM'], 4).unique()
    dx = 1.0 / BINS_PER_DECADE
    xgrid = np.arange(X_LOG_RANGE[0] + dx / 2, X_LOG_RANGE[1], dx)
    ygrid = np.arange(Y_LOG_RANGE[0] + dx / 2, Y_LOG_RANGE[1], dx)
    grid = np.full((len(ygrid), len(xgrid)), np.nan)
    ii = np.searchsorted(xgrid - dx / 2, df['log10_CDOM']) - 1
    jj = np.searchsorted(ygrid - dx / 2, df['log10_Chl']) - 1
    grid[jj, ii] = df['count']

    # Exact colormap of the original figure, bottom (low N) first
    colors = lut[['R', 'G', 'B']].values[::-1] / 255.0
    cmap = LinearSegmentedColormap.from_list('dutkiewicz_hot', colors)
    vmin, vmax = lut['count'].min(), lut['count'].max()

    fig, ax = plt.subplots(figsize=(6, 5.4))
    xe = np.append(xgrid - dx / 2, xgrid[-1] + dx / 2)
    ye = np.append(ygrid - dx / 2, ygrid[-1] + dx / 2)
    pc = ax.pcolormesh(10.0 ** xe, 10.0 ** ye, grid, cmap=cmap,
                       norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10.0 ** X_LOG_RANGE[0], 10.0 ** X_LOG_RANGE[1])
    ax.set_ylim(10.0 ** Y_LOG_RANGE[0], 10.0 ** Y_LOG_RANGE[1])
    ax.set_xlabel(r'CDOM (mmol C m$^{-3}$)')
    ax.set_ylabel(r'Model actual Chl (mg m$^{-3}$)')
    ax.set_title('Dutkiewicz et al. (2018) Fig. 10a, regenerated from CSV')
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label('Samples per bin')

    # Overlay the Loisel et al. (2023) synthetic database, if available
    any_overlay = False
    if LOISEL_CSV.exists():
        loisel = pd.read_csv(LOISEL_CSV, comment='#')
        ax.scatter(loisel['CDOM'], loisel['Chl'], s=4, marker='o',
                   facecolors='none', edgecolors='royalblue', linewidths=0.5,
                   alpha=0.6, label=f'Loisel et al. (2023), n={len(loisel)}')
        any_overlay = True
    else:
        print(f'{LOISEL_CSV} not found; run the "loisel" command to '
              'create it -- plotting without the overlay')

    # Overlay the Kudela in situ data, if available; these extend far beyond
    # the Fig. 10a panel, so widen the axes and mark the original panel.
    if RAPHE_CSV.exists():
        raphe = pd.read_csv(RAPHE_CSV, comment='#')
        field = raphe[raphe.source == 'field']
        gloria = raphe[raphe.source == 'GLORIA']
        ax.scatter(gloria['CDOM'], gloria['Chl'], s=5, marker='s',
                   facecolors='none', edgecolors='mediumorchid',
                   linewidths=0.5, alpha=0.5,
                   label=f'Kudela: GLORIA (inland/coastal), n={len(gloria)}')
        ax.scatter(field['CDOM'], field['Chl'], s=10, marker='^',
                   color='forestgreen', alpha=0.7,
                   label=f'Kudela: field (Pacific/Arctic), n={len(field)}')

        lo = min(10.0 ** X_LOG_RANGE[0], raphe.CDOM.min() / 1.3)
        hi = max(10.0 ** X_LOG_RANGE[1], raphe.CDOM.max() * 1.3)
        ax.set_xlim(lo, hi)
        lo = min(10.0 ** Y_LOG_RANGE[0], raphe.Chl.min() / 1.3)
        hi = max(10.0 ** Y_LOG_RANGE[1], raphe.Chl.max() * 1.3)
        ax.set_ylim(lo, hi)
        ax.add_patch(plt.Rectangle(
            (10.0 ** X_LOG_RANGE[0], 10.0 ** Y_LOG_RANGE[0]),
            10.0 ** X_LOG_RANGE[1] - 10.0 ** X_LOG_RANGE[0],
            10.0 ** Y_LOG_RANGE[1] - 10.0 ** Y_LOG_RANGE[0],
            fill=False, edgecolor='0.4', linestyle='--', linewidth=1))

        inpanel = ((raphe.CDOM >= 10.0 ** X_LOG_RANGE[0])
                   & (raphe.CDOM <= 10.0 ** X_LOG_RANGE[1])
                   & (raphe.Chl >= 10.0 ** Y_LOG_RANGE[0])
                   & (raphe.Chl <= 10.0 ** Y_LOG_RANGE[1]))
        for src, sub in raphe.groupby('source'):
            print(f'{src}: {inpanel[sub.index].sum()}/{len(sub)} points '
                  'inside the original Fig. 10a panel (dashed)')
        any_overlay = True
    else:
        print(f'{RAPHE_CSV} not found; run the "raphe" command to '
              'create it -- plotting without the overlay')

    # Add 1:1 line
    ax.plot([10.0 ** X_LOG_RANGE[0], 10.0 ** X_LOG_RANGE[1]],
            [10.0 ** Y_LOG_RANGE[0], 10.0 ** Y_LOG_RANGE[1]],
            color='0.4', linestyle='--', linewidth=1)

    if any_overlay:
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    print(f'saved {FIG_PATH}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command',
                        choices=['extract', 'loisel', 'raphe', 'plot'],
                        help='extract: PDF -> CSV;  loisel: netCDF -> CSV;  '
                             'raphe: in situ CSVs -> CSV;  '
                             'plot: CSVs -> figure')
    args = parser.parse_args()
    if args.command == 'extract':
        extract()
    elif args.command == 'loisel':
        extract_loisel()
    elif args.command == 'raphe':
        extract_raphe()
    else:
        plot()


if __name__ == '__main__':
    main()
