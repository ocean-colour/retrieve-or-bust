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

**Figure 3 — the methodology**, radiance -> IOPs -> carbon, in two variants, both
Google Slides only (a 3-page EOI has no room for either):

- ``fig3_methodology`` — plain coloured boxes.
- ``fig3_methodology_graphics`` — the same flow with each box replaced by a *drawn*
  graphic (spacecraft over an ocean limb, aerosol-laden atmosphere, a posterior
  collapsing out of a family of candidate spectra, bbp/aph spectra with uncertainty
  bands, and a water column with sinking particles). Nothing is downloaded: no network
  fetch, no licensing question, no binary assets, and it regenerates with everything
  else. Swap in photography later if wanted -- each glyph is one function.

Writes (PNG only):
    eoi_fig1_problem.png         eoi_fig2_targets.png          (proposal)
    eoi_fig1_problem_slides.png  eoi_fig2_targets_slides.png   (Google Slides)
    eoi_fig3_methodology_slides.png
    eoi_fig3_methodology_graphics_slides.png                   (Google Slides)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

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


#: Figure 3's canvas, in inches. A Google Slides widescreen page is 10 x 5.625 in.
FIG3_W, FIG3_H = 10.0, 5.3


def _check_fits(ax, artists, label):
    """Report any text that escapes the canvas or its own box.

    A layout assertion rather than decoration: Figure 3 positions everything by hand,
    so a font change or a longer string can silently push a label out of its box. This
    turns that into a printed failure at generation time instead of something to be
    spotted by eye.  ``artists`` is a list of ``(Text, (x, y, w, h) or None)``.
    """
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for txt, box in artists:
        tb = txt.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([(tb.x0, tb.y0), (tb.x1, tb.y1)])
        lo_x, lo_y, hi_x, hi_y = (0.0, 0.0, 100.0, 100.0)
        if box is not None:
            lo_x, lo_y, w, h = box
            hi_x, hi_y = lo_x + w, lo_y + h
        tol = 0.4
        if x0 < lo_x - tol or x1 > hi_x + tol or y0 < lo_y - tol or y1 > hi_y + tol:
            bad.append((txt.get_text().split("\n")[0][:32], round(x0, 1), round(x1, 1)))
    if bad:
        print(f"  !! {label}: {len(bad)} text overflow(s)")
        for b in bad:
            print("      ", b)
    else:
        print(f"  ok {label}: every checked label inside its bounds")


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


# ------------------------------------------------------------------ Figure 3 --
def fig3_methodology(st=SLIDES):
    """The methodology: radiance -> IOPs -> carbon, and what breaks the degeneracy.

    Slides only (there is no page budget for it in a 3-page EOI). Laid out on a
    fixed 0-100 canvas with the axis switched off, so every element's position is
    stated explicitly and nothing can be pushed outside the frame by autoscaling.
    """
    # The axes fills the whole figure, so one x-unit is exactly FIG3_W/100 inches
    # and one y-unit FIG3_H/100. That mapping has to be exact: sizing boxes against
    # an axes that tight_layout had shrunk is what made the first attempt's labels
    # spill out of their boxes.
    fig = plt.figure(figsize=(FIG3_W, FIG3_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    checks = []

    ax.text(
        50,
        93.0,
        "Methodology: from satellite radiance to ocean carbon,\n"
        "carrying the uncertainty all the way through",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        color=INK,
        linespacing=1.35,
    )

    # -- the main chain: short bold label in the box, detail beneath ------------
    # (centre x, box colour, box label, detail beneath)
    chain = [
        (10.4, OCEAN, "Satellite\nradiance", "PACE/OCI 350–890 nm\nMODIS since 2002"),
        (30.2, GOLD, "Atmospheric\ncorrection", "cross-mission\nharmonisation"),
        (50.0, TEAL, "Bayesian\ninversion", "differentiable RT\n+ learned priors"),
        (
            69.8,
            "#4E6E8E",
            "IOPs with\nposteriors",
            "$b_{bp}$, $a_{ph}$, $a_{dg}$\nper-pixel σ",
        ),
        (89.6, GREEN, "Carbon stocks\n& fluxes", "$C_{phyto}$, POC\n→ NPP, export"),
    ]
    bw, by, bh = 16.8, 61.0, 17.0  # box width, bottom, height

    for cx, col, label, detail in chain:
        ax.add_patch(
            FancyBboxPatch(
                (cx - bw / 2, by),
                bw,
                bh,
                boxstyle="round,pad=0.4,rounding_size=1.8",
                facecolor=col,
                edgecolor="none",
                zorder=3,
            )
        )
        checks.append(
            (
                ax.text(
                    cx,
                    by + bh / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=12.5,
                    fontweight="bold",
                    color="white",
                    linespacing=1.25,
                    zorder=4,
                ),
                (cx - bw / 2, by, bw, bh),
            )
        )
        checks.append(
            (
                ax.text(
                    cx,
                    by - 2.2,
                    detail,
                    ha="center",
                    va="top",
                    fontsize=10.0,
                    color="0.32",
                    linespacing=1.3,
                    zorder=4,
                ),
                None,
            )
        )

    for i in range(len(chain) - 1):
        ax.annotate(
            "",
            xy=(chain[i + 1][0] - bw / 2 - 0.4, by + bh / 2),
            xytext=(chain[i][0] + bw / 2 + 0.4, by + bh / 2),
            arrowprops={"arrowstyle": "-|>", "lw": 2.0, "color": "0.45"},
        )

    # -- what supplies the missing information ---------------------------------
    # Sits low enough that the "priors" arrow below the chain's detail text has room
    # to read as a flow rather than as a tick mark.
    px, pw, py, ph = 15.0, 70.0, 28.0, 13.5
    ax.add_patch(
        FancyBboxPatch(
            (px, py),
            pw,
            ph,
            boxstyle="round,pad=0.4,rounding_size=1.8",
            facecolor="#eef2f6",
            edgecolor="0.6",
            lw=1.2,
            zorder=3,
        )
    )
    checks.append(
        (
            ax.text(
                px + pw / 2,
                py + ph - 3.8,
                "External information that breaks the degeneracy",
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color=INK,
                zorder=4,
            ),
            (px, py, pw, ph),
        )
    )
    checks.append(
        (
            ax.text(
                px + pw / 2,
                py + 4.2,
                "in-situ bio-optics  ·  BGC-Argo profiles  ·  "
                "ECCO-Darwin state estimate",
                ha="center",
                va="center",
                fontsize=10.0,
                color="0.3",
                zorder=4,
            ),
            (px, py, pw, ph),
        )
    )

    # Priors up into the inversion; retrievals back down into the state estimate.
    ax.annotate(
        "",
        xy=(46.0, by - 10.5),
        xytext=(46.0, py + ph),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": TEAL},
    )
    checks.append(
        (
            ax.text(
                44.8,
                (py + ph + by - 10.5) / 2,
                "priors",
                ha="right",
                va="center",
                fontsize=10.5,
                color=TEAL,
                fontweight="bold",
            ),
            None,
        )
    )
    ax.annotate(
        "",
        xy=(px + pw - 3.0, py + ph),
        xytext=(89.6, by - 10.5),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 2.0,
            "color": GREEN,
            "connectionstyle": "angle3,angleA=-80,angleB=10",
        },
    )
    checks.append(
        (
            ax.text(
                50,
                py - 3.6,
                "iterative coupling: model-informed priors in, "
                "uncertainty-quantified biological fields out",
                ha="center",
                va="top",
                fontsize=10.0,
                color="0.35",
            ),
            None,
        )
    )

    # -- the uncertainty ribbon ------------------------------------------------
    rx, rw, ry, rh = 4.0, 92.0, 7.0, 11.0
    ax.add_patch(
        FancyBboxPatch(
            (rx, ry),
            rw,
            rh,
            boxstyle="round,pad=0.4,rounding_size=1.8",
            facecolor="#fbf1ea",
            edgecolor=RUST,
            lw=1.3,
            zorder=2,
        )
    )
    checks.append(
        (
            ax.text(
                rx + rw / 2,
                ry + rh / 2,
                "Calibrated uncertainty propagated at every step — every carbon "
                "number ships with an interval",
                ha="center",
                va="center",
                fontsize=11.0,
                color=RUST,
                fontweight="bold",
                zorder=4,
            ),
            (rx, ry, rw, rh),
        )
    )

    _check_fits(ax, checks, "fig3 methodology")
    # No bbox_inches="tight": keep the exact 10 x 5.3 in slide geometry.
    out = os.path.join(HERE, f"eoi_fig3_methodology{st.suffix}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------- Figure 3 (graphics variant) --
# The five glyphs below replace the plain colour boxes of `fig3_methodology`. They are
# *drawn*, not downloaded: no network fetch, no licensing question, no binary assets in
# the repo, and they regenerate with the rest of the figures. Each takes an inset axes
# spanning 0-1 in both directions and paints inside it.


def _glyph_frame(gax, colour):
    """Tint an inset axes and outline it, so each station still reads as a step."""
    gax.set_xlim(0, 1)
    gax.set_ylim(0, 1)
    gax.set_xticks([])
    gax.set_yticks([])
    for s in gax.spines.values():
        s.set_edgecolor(colour)
        s.set_linewidth(1.8)


def g_satellite(gax):
    """A spacecraft over a limb of ocean, sending radiance down/up."""
    _glyph_frame(gax, OCEAN)
    gax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#0d1b2a", zorder=0))
    rng = np.random.default_rng(7)
    gax.scatter(
        rng.uniform(0, 1, 26),
        rng.uniform(0.45, 1, 26),
        s=1.6,
        color="white",
        alpha=0.75,
        zorder=1,
    )
    # Earth limb
    gax.add_patch(
        Circle(
            (0.5, -1.05), 1.35, facecolor=OCEAN, edgecolor="#7fb2d4", lw=1.2, zorder=2
        )
    )
    gax.add_patch(
        Circle(
            (0.5, -1.05),
            1.42,
            facecolor="none",
            edgecolor="#9fd0ea",
            lw=1.0,
            alpha=0.55,
            zorder=2,
        )
    )
    # Spacecraft: body + two panels + antenna
    gax.add_patch(
        Rectangle(
            (0.44, 0.63),
            0.12,
            0.14,
            facecolor="#e8eef2",
            edgecolor="0.35",
            lw=0.8,
            zorder=4,
        )
    )
    for x0 in (0.20, 0.58):
        gax.add_patch(
            Rectangle(
                (x0, 0.655),
                0.22,
                0.095,
                facecolor="#3f6fa5",
                edgecolor="#dfe8ef",
                lw=0.7,
                zorder=4,
            )
        )
    gax.plot([0.5, 0.5], [0.77, 0.86], color="#e8eef2", lw=1.1, zorder=4)
    gax.add_patch(Circle((0.5, 0.875), 0.022, facecolor="#e8eef2", zorder=4))
    # Sunlight in on the left; water-leaving radiance up to the sensor on the right.
    # The sensor receives, it does not emit -- worth getting right in a science figure.
    gax.annotate(
        "",
        xy=(0.22, 0.30),
        xytext=(0.33, 0.60),
        arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "#ffd88a"},
        zorder=5,
    )
    gax.annotate(
        "",
        xy=(0.60, 0.60),
        xytext=(0.70, 0.30),
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#9fe0ff"},
        zorder=5,
    )


def g_atmosphere(gax):
    """Sun, aerosol-laden atmosphere, and the signal being pulled back out."""
    _glyph_frame(gax, GOLD)
    grad = np.linspace(0, 1, 128).reshape(-1, 1)
    gax.imshow(
        grad,
        extent=(0, 1, 0.28, 1),
        aspect="auto",
        origin="lower",
        cmap=LinearSegmentedColormap.from_list("sky", ["#cfe4f2", "#5b8fc4"]),
        zorder=0,
    )
    gax.add_patch(Rectangle((0, 0), 1, 0.28, facecolor="#14618f", zorder=1))
    gax.add_patch(
        Circle(
            (0.17, 0.84),
            0.085,
            facecolor="#ffd35c",
            edgecolor="#f0b429",
            lw=0.8,
            zorder=2,
        )
    )
    rng = np.random.default_rng(3)
    gax.scatter(
        rng.uniform(0.05, 0.95, 60),
        rng.uniform(0.33, 0.92, 60),
        s=rng.uniform(2, 9, 60),
        color="white",
        alpha=0.5,
        zorder=2,
    )
    # scattered path in, corrected signal out
    gax.annotate(
        "",
        xy=(0.62, 0.30),
        xytext=(0.30, 0.90),
        arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": "#ffe9b0"},
        zorder=4,
    )
    gax.annotate(
        "",
        xy=(0.90, 0.93),
        xytext=(0.68, 0.30),
        arrowprops={"arrowstyle": "-|>", "lw": 1.9, "color": "white"},
        zorder=4,
    )


def g_inversion(gax):
    """A posterior: many candidate spectra collapsing onto a credible band."""
    _glyph_frame(gax, TEAL)
    gax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f2f8f8", zorder=0))
    x = np.linspace(0.08, 0.94, 120)
    base = 0.30 + 0.42 * np.exp(-(((x - 0.36) / 0.20) ** 2))
    rng = np.random.default_rng(11)
    for _ in range(14):  # candidate solutions -- the degeneracy
        wob = base * rng.uniform(0.78, 1.22) + rng.normal(0, 0.012, x.size)
        gax.plot(x, wob, color=TEAL, lw=0.7, alpha=0.28, zorder=1)
    gax.fill_between(x, base * 0.9, base * 1.1, color=TEAL, alpha=0.30, zorder=2)
    gax.plot(x, base, color="#12595a", lw=2.1, zorder=3)
    gax.annotate(
        "",
        xy=(0.94, 0.10),
        xytext=(0.08, 0.10),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "0.45"},
        zorder=3,
    )
    gax.text(0.5, 0.035, "λ", ha="center", va="bottom", fontsize=8, color="0.35")


def g_iops(gax):
    """bbp and aph spectra, each with a posterior uncertainty band."""
    _glyph_frame(gax, "#4E6E8E")
    gax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f4f6f9", zorder=0))
    x = np.linspace(0.10, 0.94, 140)
    bbp = 0.78 - 0.42 * (x - 0.10)  # smooth power-law-ish decline
    aph = (
        0.20
        + 0.30 * np.exp(-(((x - 0.20) / 0.11) ** 2))
        + 0.20 * np.exp(-(((x - 0.80) / 0.075) ** 2))
    )  # blue + red peaks
    for curve, col in ((bbp, "#2f6f9f"), (aph, GREEN)):
        gax.fill_between(x, curve * 0.86, curve * 1.14, color=col, alpha=0.25, zorder=1)
        gax.plot(x, curve, color=col, lw=2.0, zorder=2)
    # Each label sits on its own curve: bbp at its high blue end, aph above its red peak.
    gax.text(
        0.13,
        0.84,
        "$b_{bp}$",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#2f6f9f",
        fontweight="bold",
    )
    # In the trough between aph's two peaks, well clear of the bbp band above it.
    gax.text(
        0.52,
        0.25,
        "$a_{ph}$",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=GREEN,
        fontweight="bold",
    )
    gax.annotate(
        "",
        xy=(0.94, 0.08),
        xytext=(0.10, 0.08),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "0.45"},
        zorder=3,
    )


def g_ocean_carbon(gax):
    """The water column: phytoplankton at the surface, particles sinking."""
    _glyph_frame(gax, GREEN)
    grad = np.linspace(0, 1, 128).reshape(-1, 1)
    gax.imshow(
        grad,
        extent=(0, 1, 0, 1),
        aspect="auto",
        origin="upper",
        cmap=LinearSegmentedColormap.from_list("sea", ["#bfe3ef", "#08304a"]),
        zorder=0,
    )
    gax.plot([0, 1], [0.86, 0.86], color="white", lw=1.4, alpha=0.85, zorder=2)
    rng = np.random.default_rng(5)
    # phytoplankton in the lit layer
    gax.scatter(
        rng.uniform(0.05, 0.95, 30),
        rng.uniform(0.60, 0.84, 30),
        s=rng.uniform(9, 26, 30),
        color="#7fd18b",
        alpha=0.95,
        edgecolor="#2f6d3f",
        linewidth=0.4,
        zorder=3,
    )
    # sinking aggregates
    ys = rng.uniform(0.10, 0.58, 16)
    gax.scatter(
        rng.uniform(0.05, 0.95, 16),
        ys,
        s=rng.uniform(6, 18, 16),
        color="#d9c39a",
        alpha=0.9,
        edgecolor="#8a7448",
        linewidth=0.3,
        zorder=3,
    )
    for xa in (0.24, 0.52, 0.79):
        gax.annotate(
            "",
            xy=(xa, 0.07),
            xytext=(xa, 0.55),
            arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "white", "alpha": 0.8},
            zorder=4,
        )


def fig3_methodology_graphics(st=SLIDES):
    """Figure 3 with drawn graphics in place of the plain boxes. Slides only."""
    fig = plt.figure(figsize=(FIG3_W, FIG3_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    checks = []

    ax.text(
        50,
        93.0,
        "Methodology: from satellite radiance to ocean carbon,\n"
        "carrying the uncertainty all the way through",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        color=INK,
        linespacing=1.35,
    )

    # (centre x, glyph fn, accent colour, label, detail)
    chain = [
        (
            10.4,
            g_satellite,
            OCEAN,
            "Satellite\nradiance",
            "PACE/OCI 350–890 nm\nMODIS since 2002",
        ),
        (
            30.2,
            g_atmosphere,
            GOLD,
            "Atmospheric\ncorrection",
            "cross-mission\nharmonisation",
        ),
        (
            50.0,
            g_inversion,
            TEAL,
            "Bayesian\ninversion",
            "differentiable RT\n+ learned priors",
        ),
        (
            69.8,
            g_iops,
            "#4E6E8E",
            "IOPs with\nposteriors",
            "$b_{bp}$, $a_{ph}$, $a_{dg}$\nper-pixel σ",
        ),
        (
            89.6,
            g_ocean_carbon,
            GREEN,
            "Carbon stocks\n& fluxes",
            "$C_{phyto}$, POC\n→ NPP, export",
        ),
    ]
    gw, gy, gh = 17.0, 62.0, 18.0  # glyph panel: width, bottom, height

    for cx, gfn, col, label, detail in chain:
        gax = fig.add_axes([(cx - gw / 2) / 100.0, gy / 100.0, gw / 100.0, gh / 100.0])
        gfn(gax)
        checks.append(
            (
                ax.text(
                    cx,
                    gy - 1.6,
                    label,
                    ha="center",
                    va="top",
                    fontsize=11.0,
                    fontweight="bold",
                    color=col,
                    zorder=4,
                ),
                None,
            )
        )
        checks.append(
            (
                ax.text(
                    cx,
                    gy - 9.6,
                    detail,
                    ha="center",
                    va="top",
                    fontsize=9.5,
                    color="0.32",
                    linespacing=1.3,
                    zorder=4,
                ),
                None,
            )
        )

    for i in range(len(chain) - 1):
        ax.annotate(
            "",
            xy=(chain[i + 1][0] - gw / 2 - 0.5, gy + gh / 2),
            xytext=(chain[i][0] + gw / 2 + 0.5, gy + gh / 2),
            arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "0.4"},
        )

    px, pw, py, ph = 15.0, 70.0, 26.0, 13.0
    ax.add_patch(
        FancyBboxPatch(
            (px, py),
            pw,
            ph,
            boxstyle="round,pad=0.4,rounding_size=1.8",
            facecolor="#eef2f6",
            edgecolor="0.6",
            lw=1.2,
            zorder=3,
        )
    )
    checks.append(
        (
            ax.text(
                px + pw / 2,
                py + ph - 3.6,
                "External information that breaks the degeneracy",
                ha="center",
                va="center",
                fontsize=11.0,
                fontweight="bold",
                color=INK,
                zorder=4,
            ),
            (px, py, pw, ph),
        )
    )
    checks.append(
        (
            ax.text(
                px + pw / 2,
                py + 4.0,
                "in-situ bio-optics  ·  BGC-Argo profiles  ·  "
                "ECCO-Darwin state estimate",
                ha="center",
                va="center",
                fontsize=9.5,
                color="0.3",
                zorder=4,
            ),
            (px, py, pw, ph),
        )
    )

    ax.annotate(
        "",
        xy=(46.0, gy - 16.2),
        xytext=(46.0, py + ph),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": TEAL},
    )
    checks.append(
        (
            ax.text(
                44.8,
                (py + ph + gy - 16.2) / 2,
                "priors",
                ha="right",
                va="center",
                fontsize=10.0,
                color=TEAL,
                fontweight="bold",
            ),
            None,
        )
    )
    ax.annotate(
        "",
        xy=(px + pw - 3.0, py + ph),
        xytext=(89.6, gy - 16.2),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 2.0,
            "color": GREEN,
            "connectionstyle": "angle3,angleA=-80,angleB=10",
        },
    )
    checks.append(
        (
            ax.text(
                50,
                py - 3.4,
                "iterative coupling: model-informed priors in, "
                "uncertainty-quantified biological fields out",
                ha="center",
                va="top",
                fontsize=9.5,
                color="0.35",
            ),
            None,
        )
    )

    rx, rw, ry, rh = 4.0, 92.0, 6.0, 10.5
    ax.add_patch(
        FancyBboxPatch(
            (rx, ry),
            rw,
            rh,
            boxstyle="round,pad=0.4,rounding_size=1.8",
            facecolor="#fbf1ea",
            edgecolor=RUST,
            lw=1.3,
            zorder=2,
        )
    )
    checks.append(
        (
            ax.text(
                rx + rw / 2,
                ry + rh / 2,
                "Calibrated uncertainty propagated at every step — every carbon "
                "number ships with an interval",
                ha="center",
                va="center",
                fontsize=10.5,
                color=RUST,
                fontweight="bold",
                zorder=4,
            ),
            (rx, ry, rw, rh),
        )
    )

    _check_fits(ax, checks, "fig3 methodology (graphics)")
    out = os.path.join(HERE, f"eoi_fig3_methodology_graphics{st.suffix}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    for st in (PAPER, SLIDES):
        fig1_problem(st)
        fig2_targets(st)
    # Figure 3 is for talks only, per request: no paper variant. Both the plain-box
    # original and the graphics version are kept.
    fig3_methodology(SLIDES)
    fig3_methodology_graphics(SLIDES)


if __name__ == "__main__":
    main()
