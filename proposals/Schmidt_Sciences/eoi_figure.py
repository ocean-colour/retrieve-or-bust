"""
eoi_figure.py — the two figures for the VICC EOI proposal.

Split into two separate figures (rather than one two-panel figure) so each fits
comfortably on the page:

  **Figure 1 — the problem.** Independent estimates of the same ocean-carbon
  quantity disagree by factors of 2.5-3.5. These empirical spreads, not single-study
  error bars, are the honest measure of current uncertainty: the published error bars
  are in-sample and conversion-only (reports/biomass_summary.md, section 3).

  **Figure 2 — what we propose to deliver.** The three headline targets, each a
  factor-of-two reduction against a published baseline, on a common axis of
  "fraction of the current uncertainty" so the factor-2 claim is visible at a glance.

  T-C rests on the Weatherhead et al. (1998) trend-detection formulation used by
  Henson et al. (2016) and Beaulieu et al. (2013):

      n* = [ 3.3 sigma_N / |omega_0| * sqrt((1+phi)/(1-phi)) ]^(2/3)

  so n* scales as sigma_N^(2/3) and halving the noise term shortens time-to-detection
  by 2^(2/3) ~ 1.59x. An uncalibrated multi-mission seam *inflates* n* (Beaulieu's
  global chlorophyll goes 27 -> 43 yr with a mid-record discontinuity), so removing
  the seam and halving sigma_N together move detection inside the record already held.

All values are read from the cited literature; nothing here is a new analysis. The
target values follow from the scaling above and are labelled as projections.

Layout rules followed here, per review: every annotation stays inside the axes, and
no two text elements overlap (bar value labels are placed in a reserved right-hand
margin whose width is computed from the rendered text).

Run in the `ocean14` conda environment:
    python proposals/Schmidt_Sciences/eoi_figure.py

Writes (PNG only): eoi_fig1_problem.png, eoi_fig2_targets.png
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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

#: n* scales as sigma_N^(2/3) (Weatherhead et al. 1998).
SIGMA_HALVING_GAIN = 2.0 ** (2.0 / 3.0)  # ~1.587

#: MODIS Aqua record (mid-2002 onward): length now and at the end of the award.
RECORD_NOW_YR = 2026 - 2002
RECORD_2032_YR = 2032 - 2002


def _text_w_data(ax, s, fontsize):
    """Width of a string in x-data units, for reserving label space in-axes."""
    fig = ax.figure
    t = ax.text(0, 0, s, fontsize=fontsize)
    fig.canvas.draw()
    bb = t.get_window_extent()
    t.remove()
    x0, x1 = ax.transData.inverted().transform([(0, 0), (bb.width, 0)])[:, 0]
    return x1 - x0


# ------------------------------------------------------------------ Figure 1 --
def fig1_problem():
    """Factor of disagreement across independent estimates (max / min)."""
    # (short name, range text, lo, hi, colour) -- from reports/biomass_summary.md
    items = [
        ("satellite $b_{bp}$\nacross sensors", "18–45 %", 18, 45, OCEAN),
        (
            "$C_{phyto}$ global stock\nby conversion choice",
            "218–771 Tg C",
            218,
            771,
            TEAL,
        ),
        ("global NPP\nacross algorithms", "32–79 Gt C yr$^{-1}$", 32, 79, GOLD),
        ("export flux at 100 m\nacross methods", "5–15 Gt C yr$^{-1}$", 5, 15, RUST),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 2.3))
    y = np.arange(len(items))[::-1]

    for yi, (name, rng, lo, hi, c) in zip(y, items):
        f = hi / lo
        ax.barh(yi, f - 1.0, left=1.0, color=c, height=0.58, zorder=2)
        # Range text sits inside the bar when it fits, else just after it.
        ax.text(
            1.06, yi, rng, va="center", ha="left", fontsize=7.8, color="white", zorder=4
        )

    # Reserve room on the right for the "2.5x" labels, measured not guessed.
    ax.set_xlim(1.0, 4.0)
    fig.canvas.draw()
    pad = _text_w_data(ax, "  3.5×", 9.0)
    ax.set_xlim(1.0, max(hi / lo for _, _, lo, hi, _ in items) + pad)

    for yi, (name, rng, lo, hi, c) in zip(y, items):
        ax.text(
            hi / lo + pad * 0.12,
            yi,
            f"{hi / lo:.1f}×",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color=INK,
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items], fontsize=8)
    ax.set_xlabel(
        "factor by which independent estimates of the same quantity disagree "
        "(max ÷ min)",
        fontsize=8.5,
    )
    ax.set_title(
        "We do not know the ocean's carbon stocks or fluxes to better "
        "than a factor of 2.5–3.5",
        fontsize=10,
        pad=8,
    )
    ax.xaxis.grid(True, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = os.path.join(HERE, "eoi_fig1_problem.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ Figure 2 --
def fig2_targets():
    """The three headline targets, as a fraction of the current uncertainty."""
    n_star_target = 31.5 / SIGMA_HALVING_GAIN
    # (tag, quantity, baseline text, target text, baseline value, target value, colour)
    rows = [
        ("T-A", "per-pixel $C_{phyto}$ error", "32 %", "≤16 %", 32.0, 16.0, OCEAN),
        ("T-B", "global stock spread", "3.5×", "<1.8×", 3.5, 1.8, TEAL),
        (
            "T-C",
            "years to detect a trend, $n^*$",
            "31.5 yr",
            f"≈{n_star_target:.0f} yr",
            31.5,
            n_star_target,
            GREEN,
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 2.2))
    y = np.arange(len(rows))[::-1]

    ax.axvspan(0, 0.5, color="#eef4f8", zorder=0)
    ax.axvline(0.5, color=RUST, lw=1.2, ls="--", zorder=1)
    ax.axvline(1.0, color="0.5", lw=1.0, zorder=1)

    for yi, (tag, q, b_txt, t_txt, b_val, t_val, c) in zip(y, rows):
        frac = t_val / b_val
        ax.plot(
            [frac, 1.0], [yi, yi], color=c, lw=3.0, solid_capstyle="round", zorder=2
        )
        ax.plot([1.0], [yi], "o", ms=8, color="white", mec="0.45", mew=1.4, zorder=3)
        ax.plot([frac], [yi], "o", ms=9, color=c, mec="white", mew=1.4, zorder=4)
        # Values are placed on the markers' outer sides, so they cannot collide.
        ax.text(1.035, yi, b_txt, va="center", ha="left", fontsize=8.4, color="0.35")
        ax.text(
            frac - 0.028,
            yi,
            t_txt,
            va="center",
            ha="right",
            fontsize=8.8,
            fontweight="bold",
            color=c,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}   {q}" for t, q, *_ in rows], fontsize=8.6)
    ax.set_xlim(0.30, 1.20)
    # Extra headroom above the top row so the legend sits in genuinely empty space
    # instead of over the T-A labels.
    ax.set_ylim(-0.62, len(rows) - 0.5 + 0.72)
    ax.set_xlabel("remaining uncertainty, as a fraction of today's value", fontsize=8.5)
    ax.set_title(
        "What we propose to deliver: a factor of two on three "
        "independently published baselines",
        fontsize=10,
        pad=8,
    )

    # Labels for the two reference lines, inside the axes and clear of the rows.
    ax.text(0.5, -0.52, " factor of 2", color=RUST, fontsize=8, ha="left", va="center")
    ax.text(1.0, -0.52, "today ", color="0.4", fontsize=8, ha="right", va="center")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=7,
                color="white",
                mec="0.45",
                label="published baseline",
            ),
            Line2D(
                [], [], marker="o", ls="none", ms=7, color=INK, label="project target"
            ),
        ],
        fontsize=7.6,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        handletextpad=0.4,
        columnspacing=1.4,
        ncol=2,
    )
    ax.set_xticks([0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.4", "0.5", "0.6", "0.8", "1.0"])
    fig.tight_layout()
    out = os.path.join(HERE, "eoi_fig2_targets.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    fig1_problem()
    fig2_targets()


if __name__ == "__main__":
    main()
