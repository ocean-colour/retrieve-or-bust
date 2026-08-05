"""
eoi_figure.py — the single figure for the VICC EOI proposal.

Two panels carrying the whole argument of the EOI:

  (a) **The problem.** Independent estimates of the same ocean-carbon quantity
      disagree by factors of 2.5-3.5. This is the honest measure of the current
      uncertainty -- larger than any single study's stated error bar, because those
      are in-sample and conversion-only (see reports/biomass_summary.md, section 3).

  (b) **The deliverable.** Years of record needed to detect a climate-driven trend,
      n*, from the Weatherhead et al. (1998) formulation used by Henson et al.
      (2016) and Beaulieu et al. (2013):

          n* = [ 3.3 sigma_N / |omega_0| * sqrt((1+phi)/(1-phi)) ]^(2/3)

      so n* scales as sigma_N^(2/3): halving the noise term shortens time-to-
      detection by 2^(2/3) ~ 1.59x. An uncalibrated multi-mission seam *inflates*
      n* (Beaulieu's global chlorophyll goes 27 -> 43 yr with a mid-record
      discontinuity). Removing the seam and halving sigma_N together move detection
      inside the length of the record we already have.

All values are read from the cited literature; nothing here is a new analysis.
The 2^(2/3) target values are computed from the published n* via the scaling above,
which is stated in the figure caption rather than presented as a measurement.

Run in the `ocean14` conda environment:
    python proposals/Schmidt_Sciences/eoi_figure.py

Writes: proposals/Schmidt_Sciences/eoi_fig1.png (and .pdf for the submission)
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

OCEAN = "#1D6FA5"
TEAL = "#2E8B8B"
GOLD = "#C79A3A"
RUST = "#B4551F"
GREEN = "#3E7D3E"
INK = "#1a2b33"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": "#555",
        "axes.linewidth": 0.9,
    }
)

#: n* scales as sigma_N^(2/3) (Weatherhead et al. 1998), so halving sigma_N gives:
SIGMA_HALVING_GAIN = 2.0 ** (2.0 / 3.0)  # ~1.587

#: MODIS Aqua record: mid-2002 onward. Length now, and at the end of the award.
RECORD_NOW_YR = 2026 - 2002
RECORD_2032_YR = 2032 - 2002


def _panel_a(ax):
    """Factor of disagreement across independent estimates (max / min)."""
    # (label, lo, hi, unit shown, colour)  -- all from reports/biomass_summary.md
    items = [
        ("satellite $b_{bp}$\n(across sensors)", 18, 45, "18–45 %", OCEAN),
        ("$C_{phyto}$ global stock\n(conversion choice)", 218, 771, "218–771 Tg C", TEAL),
        ("global NPP\n(across algorithms)", 32, 79, "32–79 Gt C yr$^{-1}$", GOLD),
        ("export flux\n(across methods)", 5, 15, "5–15 Gt C yr$^{-1}$", RUST),
    ]
    y = np.arange(len(items))[::-1]
    for yi, (lab, lo, hi, unit, c) in zip(y, items):
        f = hi / lo
        ax.barh(yi, f, color=c, height=0.62, zorder=2)
        ax.text(f + 0.06, yi, f"{f:.1f}×  ({unit})", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items], fontsize=8)
    ax.set_xlim(1, 4.6)
    ax.axvline(1, color="0.6", lw=1)
    ax.set_xlabel("factor of disagreement between independent estimates (max ÷ min)")
    ax.set_title(
        "(a) The problem: we do not know the ocean's carbon\n"
        "stocks or fluxes to better than a factor of 2.5–3.5",
        fontsize=9.5,
    )


def _panel_b(ax):
    """Years of record needed to detect a trend, and where we move it to."""
    # (label, n*, colour, is_target)
    rows = [
        ("multi-mission record with an\nuncalibrated inter-mission seam", 43.0, RUST, False),
        ("chlorophyll, single continuous\nmission", 31.5, GOLD, False),
        ("export flux at 100 m", 32.0, GOLD, False),
        ("this project: seam removed and\n$\\sigma_N$ halved  (biomass)", 31.5 / SIGMA_HALVING_GAIN, GREEN, True),
        ("this project: seam removed and\n$\\sigma_N$ halved  (export)", 32.0 / SIGMA_HALVING_GAIN, GREEN, True),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, c, is_t) in zip(y, rows):
        ax.barh(yi, v, color=c, height=0.6, zorder=3,
                hatch="//" if is_t else None, edgecolor="white" if is_t else "none")
        ax.text(v + 0.9, yi, f"{v:.1f} yr", va="center", fontsize=8,
                fontweight="bold" if is_t else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)

    # The record we already have, and will have. These are identified in the legend
    # rather than by inline text: at x = 24 and 30 the two annotations would collide
    # with each other, with the bar value labels, and with the title.
    ax.axvline(RECORD_NOW_YR, color=OCEAN, lw=1.7, ls="--", zorder=1)
    ax.axvline(RECORD_2032_YR, color=OCEAN, lw=1.1, ls=":", zorder=1)

    ax.set_xlim(0, 58)
    ax.set_xlabel("years of record required to detect a trend  ($n^*$)", fontsize=8.5)
    ax.set_title(
        "(b) The deliverable: trend detection moves inside\n"
        "the record we already have",
        fontsize=9.5,
    )
    ax.legend(
        handles=[
            Patch(facecolor=GOLD, label="published $n^*$"),
            Patch(facecolor=RUST, label="inflated by the mission seam"),
            Patch(facecolor=GREEN, hatch="//", edgecolor="white", label="this project's target"),
            Line2D([], [], color=OCEAN, lw=1.7, ls="--",
                   label=f"record available today ({RECORD_NOW_YR} yr)"),
            Line2D([], [], color=OCEAN, lw=1.1, ls=":",
                   label=f"record by award end ({RECORD_2032_YR} yr)"),
        ],
        fontsize=7.0, frameon=False, loc="lower right",
        bbox_to_anchor=(1.005, -0.04), labelspacing=0.35,
    )


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 3.5))
    _panel_a(axA)
    _panel_b(axB)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"eoi_fig1.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
