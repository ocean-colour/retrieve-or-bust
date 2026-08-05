"""
eoi_figure.py — the two figures for the VICC EOI proposal, in two sizes.

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

**Two output styles from the same code**, selected by :class:`Style`:

- ``PAPER`` — wide and short, so that scaled into the proposal's 6.5-inch text column
  each figure is under 2 inches tall. Page space is the binding constraint there.
- ``SLIDES`` — sized for a Google Slides widescreen page (10 x 5.625 in), leaving
  room above for a slide title, with every font, line and marker enlarged. Paper
  font sizes (8-10 pt) are illegible when projected, which is the whole reason this
  variant exists; it is not merely the paper figure rescaled.

Layout rules held in both styles: every annotation stays inside the axes, and no two
text elements overlap. Where that depends on text size it is *measured*, not guessed —
Figure 1's right-hand label margin comes from the rendered width of the label string.

Run in the `ocean14` conda environment:
    python proposals/Schmidt_Sciences/eoi_figure.py

Writes (PNG only):
    eoi_fig1_problem.png         eoi_fig2_targets.png          (proposal)
    eoi_fig1_problem_slides.png  eoi_fig2_targets_slides.png   (Google Slides)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
        "axes.edgecolor": "#555",
    }
)

#: n* scales as sigma_N^(2/3) (Weatherhead et al. 1998).
SIGMA_HALVING_GAIN = 2.0 ** (2.0 / 3.0)  # ~1.587

#: MODIS Aqua record (mid-2002 onward): length now and at the end of the award.
RECORD_NOW_YR = 2026 - 2002
RECORD_2032_YR = 2032 - 2002


@dataclass(frozen=True)
class Style:
    """Everything that differs between the print and projected versions.

    Kept as data rather than as branches inside the plotting code so the two
    variants cannot drift apart in content -- only in size.
    """

    suffix: str
    axes_lw: float
    fig1_size: tuple[float, float]
    fig2_size: tuple[float, float]
    fs_range: float  # the range text set inside each bar (Fig. 1)
    fs_value: float  # the bold "2.5x" labels (Fig. 1)
    fs_tick: float
    fs_axlabel: float
    fs_title: float
    title_pad: float
    fs_baseline: float  # "32 %" at the open marker (Fig. 2)
    fs_target: float  # the bold target value (Fig. 2)
    fs_ref: float  # "factor of 2" / "today"
    fs_legend: float
    lw_dumbbell: float
    ms_baseline: float
    ms_target: float
    ms_legend: float
    mew: float
    y_bottom: float  # lower y-limit, must clear the reference labels
    y_ref_label: float


PAPER = Style(
    suffix="",
    axes_lw=0.9,
    fig1_size=(7.8, 2.3),
    fig2_size=(7.8, 2.2),
    fs_range=7.8,
    fs_value=9.5,
    fs_tick=8.0,
    fs_axlabel=10.0,
    fs_title=10.0,
    title_pad=8.0,
    fs_baseline=8.4,
    fs_target=8.8,
    fs_ref=8.0,
    fs_legend=7.6,
    lw_dumbbell=3.0,
    ms_baseline=8.0,
    ms_target=9.0,
    ms_legend=7.0,
    mew=1.4,
    y_bottom=-0.62,
    y_ref_label=-0.52,
)

#: Google Slides widescreen is 10 x 5.625 in; 4.9 in of height leaves a title band.
SLIDES = Style(
    suffix="_slides",
    axes_lw=1.3,
    fig1_size=(10.0, 4.9),
    fig2_size=(10.0, 4.9),
    fs_range=14.0,
    fs_value=17.0,
    fs_tick=14.5,
    fs_axlabel=16.5,
    fs_title=17.5,
    title_pad=14.0,
    fs_baseline=15.0,
    fs_target=16.0,
    fs_ref=13.5,
    fs_legend=13.0,
    lw_dumbbell=6.0,
    ms_baseline=14.0,
    ms_target=16.0,
    ms_legend=12.0,
    mew=2.2,
    y_bottom=-0.78,
    y_ref_label=-0.56,
)


def _text_w_data(ax, s, fontsize):
    """Width of a string in x-data units, for reserving label space in-axes."""
    fig = ax.figure
    t = ax.text(0, 0, s, fontsize=fontsize)
    fig.canvas.draw()
    bb = t.get_window_extent()
    t.remove()
    x0, x1 = ax.transData.inverted().transform([(0, 0), (bb.width, 0)])[:, 0]
    return x1 - x0


def _save(fig, name, st):
    out = os.path.join(HERE, f"{name}{st.suffix}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ Figure 1 --
def fig1_problem(st=PAPER):
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
    fig, ax = plt.subplots(figsize=st.fig1_size)
    ax.spines[:].set_linewidth(st.axes_lw)
    y = np.arange(len(items))[::-1]

    for yi, (_name, rng, lo, hi, c) in zip(y, items):
        ax.barh(yi, hi / lo - 1.0, left=1.0, color=c, height=0.58, zorder=2)
        # Range text sits inside the bar, where there is always room for it.
        ax.text(
            1.06,
            yi,
            rng,
            va="center",
            ha="left",
            fontsize=st.fs_range,
            color="white",
            zorder=4,
        )

    # Reserve room on the right for the "2.5x" labels, measured not guessed.
    ax.set_xlim(1.0, 4.0)
    fig.canvas.draw()
    pad = _text_w_data(ax, "  3.5×", st.fs_value)
    ax.set_xlim(1.0, max(hi / lo for _, _, lo, hi, _ in items) + pad)

    for yi, (_name, _rng, lo, hi, _c) in zip(y, items):
        ax.text(
            hi / lo + pad * 0.12,
            yi,
            f"{hi / lo:.1f}×",
            va="center",
            ha="left",
            fontsize=st.fs_value,
            fontweight="bold",
            color=INK,
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items], fontsize=st.fs_tick)
    ax.tick_params(labelsize=st.fs_tick)
    ax.set_xlabel(
        "factor by which independent estimates of the same quantity disagree "
        "(max ÷ min)",
        fontsize=st.fs_axlabel,
    )
    ax.set_title(
        "We do not know the ocean's carbon stocks or fluxes to better "
        "than a factor of 2.5–3.5",
        fontsize=st.fs_title,
        fontweight="bold",
        pad=st.title_pad,
    )
    ax.xaxis.grid(True, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "eoi_fig1_problem", st)


# ------------------------------------------------------------------ Figure 2 --
def fig2_targets(st=PAPER):
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
    fig, ax = plt.subplots(figsize=st.fig2_size)
    ax.spines[:].set_linewidth(st.axes_lw)
    y = np.arange(len(rows))[::-1]

    ax.axvspan(0, 0.5, color="#eef4f8", zorder=0)
    ax.axvline(0.5, color=RUST, lw=st.axes_lw * 1.35, ls="--", zorder=1)
    ax.axvline(1.0, color="0.5", lw=st.axes_lw * 1.1, zorder=1)

    for yi, (_tag, _q, b_txt, t_txt, b_val, t_val, c) in zip(y, rows):
        frac = t_val / b_val
        ax.plot(
            [frac, 1.0],
            [yi, yi],
            color=c,
            lw=st.lw_dumbbell,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [1.0],
            [yi],
            "o",
            ms=st.ms_baseline,
            color="white",
            mec="0.45",
            mew=st.mew,
            zorder=3,
        )
        ax.plot(
            [frac],
            [yi],
            "o",
            ms=st.ms_target,
            color=c,
            mec="white",
            mew=st.mew,
            zorder=4,
        )
        # Values sit on the markers' outer sides, so they cannot collide.
        ax.text(
            1.035,
            yi,
            b_txt,
            va="center",
            ha="left",
            fontsize=st.fs_baseline,
            color="0.35",
        )
        # A target label wide enough to reach back past x = 0.5 would be crossed by
        # the factor-of-2 line (this happens to T-C, whose target is to its right).
        # An opaque backing box lets the line pass behind the text instead of through
        # it, which keeps the reference line full-height and the label legible.
        ax.text(
            frac - 0.028,
            yi,
            t_txt,
            va="center",
            ha="right",
            fontsize=st.fs_target,
            fontweight="bold",
            color=c,
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.6, "alpha": 0.92},
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}   {q}" for t, q, *_ in rows], fontsize=st.fs_tick)
    ax.tick_params(labelsize=st.fs_tick)
    ax.set_xlim(0.30, 1.20)
    # Headroom above the top row so the legend sits in genuinely empty space rather
    # than over the T-A labels; y_bottom likewise clears the reference labels.
    ax.set_ylim(st.y_bottom, len(rows) - 0.5 + 0.72)
    ax.set_xlabel(
        "remaining uncertainty, as a fraction of today's value", fontsize=st.fs_axlabel
    )
    ax.set_title(
        "What we propose to deliver: a factor of two on three "
        "independently published baselines",
        fontsize=st.fs_title,
        # Bolded to match Figure 1: the two appear together, and one bold title
        # beside one regular title reads as an oversight rather than emphasis.
        fontweight="bold",
        pad=st.title_pad,
    )

    ax.text(
        0.5,
        st.y_ref_label,
        " factor of 2",
        color=RUST,
        fontsize=st.fs_ref,
        ha="left",
        va="center",
    )
    ax.text(
        1.0,
        st.y_ref_label,
        "today ",
        color="0.4",
        fontsize=st.fs_ref,
        ha="right",
        va="center",
    )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=st.ms_legend,
                color="white",
                mec="0.45",
                label="published baseline",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=st.ms_legend,
                color=INK,
                label="project target",
            ),
        ],
        fontsize=st.fs_legend,
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
    _save(fig, "eoi_fig2_targets", st)


def main():
    for st in (PAPER, SLIDES):
        fig1_problem(st)
        fig2_targets(st)


if __name__ == "__main__":
    main()
