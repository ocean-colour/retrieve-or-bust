"""
exports_figs.py — figures for reports/exports_summary.md.

All values are COMPILED FROM THE PUBLISHED LITERATURE cited in the report; these
are not new analyses. As in biomass_figs.py, the figures emphasize *disagreement
across independent estimates and methods* as the honest measure of uncertainty.

Figures (written to reports/figs/):
  exports_fig1_campaign_contrast.png — Station P vs PAP: the designed end-member
      contrast was achieved (×10 in flux), but independent methods at the SAME
      site still disagree by ×2–3.
  exports_fig2_pathways.png          — the EXPORTS-era accomplishment (pathway-
      resolved partitioning of ~10.2 Pg C/yr) beside the open problem (the
      cross-method spread of global export estimates still spans ~4–11).
  exports_fig3_timeline.png          — promise vs delivery: planning documents,
      the descoped field program, and the structural PACE timing mismatch.

Run in the `ocean14` conda environment:
    python reports/py/exports_figs.py

Author: Claude (Fable) for J. Xavier Prochaska, 2026-08-08.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figs")
os.makedirs(FIGS, exist_ok=True)

OCEAN = "#1D6FA5"
TEAL = "#2E8B8B"
GOLD = "#C79A3A"
RUST = "#B4551F"
GREEN = "#3E7D3E"
INK = "#1a2b33"

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#555", "axes.linewidth": 0.9})


# ---------------------------------------------------------------- Fig 1 -----
def fig1_campaign_contrast():
    """Station P vs PAP: sinking-POC flux and export efficiency, by method.

    Values (all as published):
      Station P (Aug–Sep 2018): trap POC flux at 100 m 1.38 ± 0.77 mmol C/m2/d
        and Ez-ratio 0.10 ± 0.06 (Estapa 2021); autonomous export potential
        fCorg 3.4 ± 0.7 mmol C/m2/d and ez-ratio 0.24 (Traylor 2025).
      PAP (May 2021): 234Th-derived POC flux at 110 m rising 11 ± 1 -> 14 ± 2
        mmol C/m2/d and BCP efficiency ~0.10 -> ~0.30 over the cruise
        (Clevenger 2024); autonomous fCorg 20.3 ± 2.3 mmol C/m2/d and
        ez-ratio 0.69 (Traylor 2025).
    """
    fig, (axf, axe) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # x positions: two site groups, two methods each
    xs = {"P_ship": 0.0, "P_auto": 0.55, "PAP_ship": 1.85, "PAP_auto": 2.40}

    # ---- left panel: sinking POC flux (log scale) ----
    axf.errorbar(xs["P_ship"], 1.38, yerr=0.77, fmt="o", ms=9, color=OCEAN,
                 capsize=4, lw=1.8, label="ship (traps / $^{234}$Th)")
    axf.errorbar(xs["P_auto"], 3.4, yerr=0.7, fmt="s", ms=9, color=RUST,
                 capsize=4, lw=1.8, label="autonomous (O$_2$/NCP budget)")
    # PAP 234Th flux rose 11 -> 14 over the cruise: plot midpoint with the
    # cruise evolution as an asymmetric range plus the epoch SDs
    axf.errorbar(xs["PAP_ship"], 12.5, yerr=[[12.5 - (11 - 1)], [(14 + 2) - 12.5]],
                 fmt="o", ms=9, color=OCEAN, capsize=4, lw=1.8)
    axf.errorbar(xs["PAP_auto"], 20.3, yerr=2.3, fmt="s", ms=9, color=RUST,
                 capsize=4, lw=1.8)
    axf.set_yscale("log")
    axf.set_ylim(0.4, 40)
    axf.set_xlim(-0.5, 2.9)
    axf.set_xticks([0.275, 2.125])
    axf.set_xticklabels(["Station P\n(NE Pacific, Aug–Sep 2018)",
                         "PAP\n(N Atlantic, May 2021)"])
    axf.set_ylabel("sinking POC flux near 100 m\n(mmol C m$^{-2}$ d$^{-1}$)")
    axf.set_title("The designed ×10 site contrast was achieved…", fontsize=12)
    axf.legend(fontsize=9, frameon=False, loc="upper left")
    # ×10 bracket between site means (ship values)
    axf.annotate("", xy=(1.30, 12.5), xytext=(1.30, 1.38),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    axf.text(1.36, 4.2, "~×10", fontsize=11, color=INK, fontweight="bold")

    # ---- right panel: export efficiency (fraction of NPP) ----
    axe.errorbar(xs["P_ship"], 0.10, yerr=0.06, fmt="o", ms=9, color=OCEAN,
                 capsize=4, lw=1.8)
    axe.plot(xs["P_auto"], 0.24, "s", ms=9, color=RUST)
    # PAP efficiency rose ~0.10 -> ~0.30 over the cruise (plot as range bar)
    axe.plot([xs["PAP_ship"]] * 2, [0.10, 0.30], color=OCEAN, lw=2.6,
             solid_capstyle="round")
    axe.plot(xs["PAP_ship"], 0.30, "o", ms=9, color=OCEAN)
    axe.plot(xs["PAP_ship"], 0.10, "o", ms=6, color=OCEAN, mfc="white")
    axe.plot(xs["PAP_auto"], 0.69, "s", ms=9, color=RUST)
    axe.annotate("rose over\nthe cruise", xy=(xs["PAP_ship"], 0.21),
                 xytext=(1.12, 0.42), fontsize=8.5, color=OCEAN,
                 arrowprops=dict(arrowstyle="->", color=OCEAN, lw=1.0))
    axe.set_ylim(0, 0.85)
    axe.set_xlim(-0.5, 2.9)
    axe.set_xticks([0.275, 2.125])
    axe.set_xticklabels(["Station P\n(NE Pacific, Aug–Sep 2018)",
                         "PAP\n(N Atlantic, May 2021)"])
    axe.set_ylabel("export efficiency\n(fraction of NPP exported)")
    axe.set_title("…but methods at the same site disagree ×2–3", fontsize=12)
    # per-site method-spread brackets
    for x0, x1, lo, hi, lab in [(xs["P_ship"], xs["P_auto"], 0.10, 0.24, "×2.4"),
                                (xs["PAP_ship"], xs["PAP_auto"], 0.30, 0.69, "×2.3")]:
        xm = 0.5 * (x0 + x1)
        axe.annotate("", xy=(xm, hi), xytext=(xm, lo),
                     arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.2))
        axe.text(xm + 0.06, 0.5 * (lo + hi), lab, fontsize=9.5, color="0.35")

    fig.suptitle("EXPORTS field campaigns: end-member contrast delivered; "
                 "method-level uncertainty remains", fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGS, "exports_fig1_campaign_contrast.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- Fig 2 -----
def fig2_pathways():
    """Left: Nowicki 2022 pathway partitioning (the accomplishment).
    Right: independent global export estimates (the open problem).

    Left values (Pg C/yr, 25–75% quartiles): gravitational 7.3 (6.8–7.7),
    mixing 1.9 (1.7–2.2), migrant 1.0 (0.7–1.3); total 10.2 (9.7–10.7).
    Right values as compiled in Siegel et al. 2023 Table 1 plus the report's
    §2.5 entries: Henson 2011 4.0 ± 2.2 (std); ensemble mean 6.08 ± 1.17 (std);
    Laws 2011 6.9; DeVries & Weber 2017 9.1 ± 0.2 (std); Laws 2000 9.9;
    Nowicki 2022 10.2 (9.7–10.7 qrt); hydrographic 10.64 ± 0.80 (std).
    Historical envelope 5–15 shaded.
    """
    fig, (axp, axs) = plt.subplots(1, 2, figsize=(12.2, 4.9),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # ---- left: pathway bars with quartile whiskers ----
    paths = [("Gravitational\n(sinking particles)", 7.3, 6.8, 7.7, OCEAN),
             ("Mixing\n(subduction + DOC)", 1.9, 1.7, 2.2, TEAL),
             ("Migrant\n(zooplankton DVM)", 1.0, 0.7, 1.3, GOLD)]
    for i, (lab, v, lo, hi, c) in enumerate(paths):
        axp.bar(i, v, color=c, width=0.6)
        axp.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none",
                     ecolor=INK, capsize=5, lw=1.6)
        axp.text(i, hi + 0.35, f"{v:.1f}", ha="center", fontsize=11,
                 fontweight="bold")
        axp.text(i, v / 2, f"{100 * v / 10.2:.0f}%", ha="center", va="center",
                 fontsize=11, color="white", fontweight="bold")
    axp.set_xticks(range(3))
    axp.set_xticklabels([p[0] for p in paths], fontsize=9)
    axp.set_ylabel("export at the euphotic-zone base  (Pg C yr$^{-1}$)")
    axp.set_ylim(0, 9.4)
    axp.set_title("The accomplishment: pathway-resolved\npartitioning of "
                  "~10.2 Pg C yr$^{-1}$ export", fontsize=12)
    axp.text(2.35, 8.9, "whiskers:\n25–75% quartiles", fontsize=8, color="0.4",
             ha="right", va="top")

    # ---- right: cross-method ladder of global export estimates ----
    ests = [("Henson 2011 (e-ratio, 100 m)", 4.0, 2.2, 2.2, "std"),
            ("Model ensemble (Doney 2024)", 6.08, 1.17, 1.17, "std"),
            ("Laws 2011 (e-ratio)", 6.9, np.nan, np.nan, ""),
            ("DeVries & Weber 2017 (inverse)", 9.1, 0.2, 0.2, "std"),
            ("Laws 2000 (f-ratio)", 9.9, np.nan, np.nan, ""),
            ("Nowicki 2022 (inverse, EXPORTS-era)", 10.2, 0.5, 0.5, "qrt"),
            ("Hydrographic (Wang 2023)", 10.64, 0.80, 0.80, "std")]
    ypos = np.arange(len(ests))[::-1]
    axs.axvspan(5, 15, color="0.92", zorder=0)
    axs.text(14.7, len(ests) - 0.75, "historical 5–15\nenvelope", fontsize=8.5,
             color="0.45", ha="right", va="top")
    for (lab, v, lo, hi, kind), y in zip(ests, ypos):
        c = RUST if "EXPORTS-era" in lab else OCEAN
        if np.isnan(lo):
            axs.plot(v, y, "o", ms=8, color=c)
        else:
            axs.errorbar(v, y, xerr=[[lo], [hi]], fmt="o", ms=8, color=c,
                         capsize=4, lw=1.7)
    axs.set_yticks(ypos)
    axs.set_yticklabels([e[0] for e in ests], fontsize=9)
    axs.set_xlim(0, 15.6)
    axs.set_ylim(-0.7, len(ests) - 0.3)
    axs.set_xlabel("global carbon export  (Pg C yr$^{-1}$)")
    axs.set_title("The open problem: independent estimates\nstill span "
                  "~4–11 Pg C yr$^{-1}$", fontsize=12)

    fig.tight_layout()
    out = os.path.join(FIGS, "exports_fig2_pathways.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- Fig 3 -----
def fig3_timeline():
    """Promise vs delivery: the EXPORTS timeline against the PACE launch."""
    fig, ax = plt.subplots(figsize=(12.2, 4.4))

    t0, t1 = 2012.0, 2026.6
    ax.set_xlim(t0, t1)
    ax.set_ylim(-2.6, 2.9)
    ax.axhline(0, color=INK, lw=1.6, zorder=1)
    # Layout note: labels below the line sit at three depths chosen so that no
    # label's text block shares both an x-range and a y-range with another; the
    # 33-month gap annotation gets its own band beneath all event labels.

    # shaded spans: field phase and PACE era
    ax.axvspan(2018.6, 2021.42, color=OCEAN, alpha=0.16, zorder=0)
    ax.text(2020.0, 2.62, "field phase", ha="center", fontsize=9.5,
            color=OCEAN, fontweight="bold")
    ax.axvspan(2024.1, t1, color=GOLD, alpha=0.20, zorder=0)
    ax.text(2025.35, 2.62, "PACE era", ha="center", fontsize=9.5,
            color="#8a6a20", fontweight="bold")

    # events: (year, level, text, color) — above the line = plans/promises,
    # below = execution/delivery
    events = [
        (2012.5, 1.0, "ROSES-2012 scoping\nstudy (COOPEX)", "0.35"),
        (2013.4, 2.0, "renamed EXPORTS", "0.35"),
        (2015.4, 1.2, "Science Plan:\n“algorithms …for the\nupcoming PACE mission”", RUST),
        (2016.8, 2.3, "Implementation Plan:\n4 deployments, 388 sea\ndays, \\$71.5M Goal Plan", RUST),
        (2018.65, -1.2, "NE Pacific deployment\n(Station P; 1 of 2\nplanned visits)", OCEAN),
        (2020.3, -2.6, "N Atlantic deployment\npostponed (COVID)", "0.35"),
        (2021.35, -1.2, "N Atlantic deployment\n(PAP) — field phase ends", OCEAN),
        (2022.6, -2.6, "Nowicki et al. global\npathway partitioning", TEAL),
        (2023.1, 1.15, "Siegel et al. review calls for\na future operational system", RUST),
        (2024.55, -1.2, "PACE launches\n(Feb 2024)", "#8a6a20"),
    ]
    for yr, lev, txt, c in events:
        # PACE marker sits at the true launch date even though its label is
        # nudged right, clear of the N Atlantic label
        mark = 2024.1 if txt.startswith("PACE launches") else yr
        ax.plot([mark, yr], [0, lev * 0.42], color=c, lw=1.1, zorder=2)
        ax.plot(mark, 0, "o", ms=6, color=c, zorder=3)
        va = "bottom" if lev > 0 else "top"
        ax.text(yr, lev * 0.42 + (0.06 if lev > 0 else -0.06), txt,
                ha="center", va=va, fontsize=8.3, color=c, linespacing=1.15)

    # the structural gap: last cruise -> PACE launch, in its own band beneath
    # all event labels
    ax.annotate("", xy=(2024.1, -1.95), xytext=(2021.42, -1.95),
                arrowprops=dict(arrowstyle="<->", color=RUST, lw=1.6))
    ax.text(2021.9, -2.08, "33 months: no coincident\nEXPORTS field + PACE data",
            ha="center", va="top", fontsize=9, color=RUST, fontweight="bold")

    ax.set_yticks([])
    ax.set_xticks(np.arange(2012, 2027, 2))
    ax.set_xticklabels([str(y) for y in np.arange(2012, 2027, 2)])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title("Promise vs delivery: the founding “algorithms for PACE” goal "
                 "could not be closed within the program", fontsize=12.5)

    fig.tight_layout()
    out = os.path.join(FIGS, "exports_fig3_timeline.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig1_campaign_contrast()
    fig2_pathways()
    fig3_timeline()
