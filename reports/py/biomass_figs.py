"""
biomass_figs.py — figures for reports/biomass_summary.md.

All values are COMPILED FROM THE PUBLISHED LITERATURE cited in the report; these are
not new analyses. The figures are deliberately built to show the *disagreement across
independent estimates* (the honest, empirical measure of uncertainty) rather than to
reproduce any single paper's internally-derived error bar — per the report's stance
that literature error estimates are often in-sample / conversion-only and likely
underestimate the true uncertainty.

Figures (written to reports/figs/):
  fig1_conversion_slopes.png  — the bbp -> Cphyto conversion-slope spread + stock range
  fig2_ocean_vs_land.png      — ocean phytoplankton carbon vs terrestrial biomass & NPP
  fig3_uncertainty_cascade.png— factor-of-disagreement along the carbon chain
  fig4_subsurface.png         — first-optical-depth blindness & the path to depth

Run in the `ocean14` conda environment:
    python reports/py/biomass_figs.py

Author: Claude (Fable) for J. Xavier Prochaska, 2026-07-31.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
def fig1_conversion_slopes():
    """bbp(470) -> Cphyto: published linear relations diverge ~3.5x."""
    bbp = np.linspace(0.0005, 0.004, 100)  # m^-1, the Graff-2015 field range
    # (label, slope, intercept, color, style)  -- slopes in mg C m^-3 per m^-1
    rels = [
        ("Martínez-Vicente 2013 (30,100)", 30100, -22.9, RUST, "-"),
        ("Fox 2022 diatoms/dino (22,641)", 22641, 0.0, GOLD, "--"),
        ("Behrenfeld 2005 (13,000)", 13000, 4.55, TEAL, "-"),
        ("Graff 2015 (12,128)", 12128, 0.59, OCEAN, "-"),
        ("Fox 2022 cyano/hapto (8,372)", 8372, 0.0, GREEN, "--"),
    ]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12.2, 5.0),
                                  gridspec_kw={"width_ratios": [1.55, 1]})

    # left: the fan of conversion lines
    lo = 8372 * bbp
    hi = 30100 * bbp - 22.9
    ax.fill_between(bbp * 1e3, np.clip(lo, 0, None), np.clip(hi, 0, None),
                    color="0.85", zorder=0, label="published spread")
    for lab, m, b, c, ls in rels:
        ax.plot(bbp * 1e3, np.clip(m * bbp + b, 0, None), ls, color=c, lw=2.3, label=lab)
    ax.set_xlabel(r"$b_{bp}(470)$  ($10^{-3}$ m$^{-1}$)")
    ax.set_ylabel(r"phytoplankton carbon  $C_{phyto}$  (mg C m$^{-3}$)")
    ax.set_title("One backscatter value → a ~3.5× range of carbon\n"
                 "(published $b_{bp}\\rightarrow C_{phyto}$ conversions)", fontsize=12)
    ax.legend(fontsize=8.2, frameon=False, loc="upper left")
    ax.set_xlim(0.5, 4.0); ax.set_ylim(0, 120)
    # annotate divergence at high bbp
    ax.annotate("", xy=(3.9, 8372*0.0039), xytext=(3.9, 30100*0.0039-22.9),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(3.72, 60, "~3.5×", rotation=90, va="center", ha="center",
            fontsize=11, color=INK, fontweight="bold")

    # right: resulting global stock estimates
    labels = ["min slope\n(8,372)", "median\n(15,124)", "max slope\n(30,100)",
              "Stoer &\nFennel '24", "prior\nrange"]
    vals = [218, 390, 771, 314, np.nan]
    colors = [GREEN, OCEAN, RUST, "#444", "0.7"]
    xpos = np.arange(len(labels))
    for i, (v, c) in enumerate(zip(vals, colors)):
        if not np.isnan(v):
            axb.bar(i, v, color=c, width=0.62)
            axb.text(i, v + 18, f"{v:.0f}", ha="center", fontsize=9.5)
    # prior range 250-2400 as a whisker
    axb.plot([4, 4], [250, 2400], color="0.4", lw=2.4)
    axb.plot([3.85, 4.15], [250, 250], color="0.4", lw=2.4)
    axb.plot([3.85, 4.15], [2400, 2400], color="0.4", lw=2.4)
    axb.text(4, 2470, "250–2400", ha="center", fontsize=9, color="0.3")
    axb.set_xticks(xpos); axb.set_xticklabels(labels, fontsize=8.4)
    axb.set_ylabel("global $C_{phyto}$ stock  (Tg C)")
    axb.set_title("Same data, different slope →\nglobal stock spans 218–771 Tg",
                  fontsize=12)
    axb.set_ylim(0, 2600)
    axb.axhspan(0, 0, color="none")

    fig.tight_layout()
    out = os.path.join(FIGS, "fig1_conversion_slopes.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


# ---------------------------------------------------------------- Fig 2 -----
def fig2_ocean_vs_land():
    """Ocean phytoplankton carbon is tiny vs land biomass, yet turns over fast."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # left: standing stock (log scale)
    names = ["Terrestrial\nvegetation", "Marine POC\n(total)",
             "Ocean\nphytoplankton"]
    lo = np.array([450.0, 2.3, 0.78])
    hi = np.array([650.0, 4.0, 1.0])
    mid = np.sqrt(lo * hi)
    cols = [GREEN, TEAL, OCEAN]
    y = np.arange(len(names))[::-1]
    for yi, l, h, m, c in zip(y, lo, hi, mid, cols):
        axL.plot([l, h], [yi, yi], color=c, lw=8, solid_capstyle="round", alpha=0.85)
        axL.text(h * 1.25, yi, f"{l:g}–{h:g} Gt C", va="center", fontsize=9.5)
    axL.set_yticks(y); axL.set_yticklabels(names, fontsize=10)
    axL.set_xscale("log"); axL.set_xlim(0.3, 3000)
    axL.set_xlabel("standing carbon stock (Gt C, log scale)")
    axL.set_title("Ocean phytoplankton hold ~0.2% of\nterrestrial biomass carbon",
                  fontsize=12)
    axL.grid(axis="x", which="both", alpha=0.25)

    # right: annual NPP roughly equal
    n2 = ["Terrestrial\nNPP", "Ocean\nNPP"]
    npp = [56.0, 50.0]
    turnover = ["~decades", "~2–6 days"]
    yy = np.arange(len(n2))[::-1]
    for yi, v, c, t in zip(yy, npp, [GREEN, OCEAN], turnover):
        axR.barh(yi, v, color=c, height=0.5)
        axR.text(v + 1.5, yi, f"{v:g} Gt C yr$^{{-1}}$\n(turnover {t})",
                 va="center", fontsize=9.2)
    axR.set_yticks(yy); axR.set_yticklabels(n2, fontsize=10)
    axR.set_xlim(0, 95); axR.set_xlabel("net primary production (Gt C yr$^{-1}$)")
    axR.set_title("…yet fix roughly HALF of global NPP\n(fast turnover)", fontsize=12)

    fig.suptitle("Why the ocean matters despite a tiny standing stock",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig2_ocean_vs_land.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)


# ---------------------------------------------------------------- Fig 3 -----
def fig3_uncertainty_cascade():
    """Factor-of-disagreement (max/min across independent estimates) per quantity."""
    # (label, lo, hi, unit string, color)
    items = [
        ("satellite $b_{bp}$ retrieval\n(MPE across sensors)", 18, 45, "18–45 %", OCEAN),
        ("$C_{phyto}$ global stock\n(conversion-slope choice)", 218, 771, "218–771 Tg C", TEAL),
        ("global NPP\n(across algorithms)", 32, 79, "32–79 Gt C yr$^{-1}$", GOLD),
        ("export / biological pump\n(across methods)", 5, 15, "5–15 Gt C yr$^{-1}$", RUST),
    ]
    factors = [hi / lo for _, lo, hi, _, _ in items]
    y = np.arange(len(items))[::-1]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    for yi, f, (lab, lo, hi, unit, c) in zip(y, factors, items):
        ax.barh(yi, f, color=c, height=0.6)
        ax.text(f + 0.05, yi, f"  {f:.1f}×   ({unit})", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels([it[0] for it in items], fontsize=10)
    ax.set_xlabel("factor of disagreement across independent estimates  (max ÷ min)")
    ax.set_xlim(1, 4.4)
    ax.axvline(1, color="0.6", lw=1)
    ax.set_title("Uncertainty as disagreement: the spread of published estimates\n"
                 "for each primary ocean-carbon quantity", fontsize=12)
    ax.text(0.99, -0.20,
            "Spreads are empirical (independent estimates disagree). Single-study "
            "error bars (e.g. Cphyto MAPE ~32%) are in-sample / conversion-only "
            "and are likely lower bounds.",
            transform=ax.transAxes, fontsize=7.8, color="0.35", ha="right", va="top")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig3_uncertainty_cascade.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", out)


# ---------------------------------------------------------------- Fig 4 -----
def fig4_subsurface():
    """First optical depth sees only the surface; most Cphyto is below it."""
    z = np.linspace(0, 200, 400)
    # a subsurface-max Cphyto profile (arbitrary but realistic shape)
    surf = 0.55 * np.exp(-((z - 5) / 40) ** 2)
    dcm = 1.0 * np.exp(-((z - 90) / 30) ** 2)
    cphy = 0.35 + surf + dcm
    z_od = 20.0  # ~first optical depth (1/Kd), schematic

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.plot(cphy, z, color=OCEAN, lw=2.6, label=r"$C_{phyto}(z)$")
    ax.fill_betweenx(z, 0, cphy, where=(z <= z_od), color=GOLD, alpha=0.55,
                     label=f"seen by passive ocean color\n(first optical depth ≈ {z_od:.0f} m)")
    ax.fill_betweenx(z, 0, cphy, where=(z >= z_od), color=OCEAN, alpha=0.14,
                     label="below (≈85% of column $C_{phyto}$)")
    ax.axhline(z_od, color=GOLD, lw=1.4, ls="--")
    ax.set_ylim(200, 0); ax.set_xlim(0, cphy.max() * 1.15)
    ax.set_xlabel(r"$C_{phyto}$  (relative)")
    ax.set_ylabel("depth (m)")
    ax.set_title("Passive ocean color sees only the first optical depth;\n"
                 "~85% of phytoplankton carbon lies below it", fontsize=11.5)
    # the in-scope path to depth
    ax.annotate("BGC-Argo profiles", xy=(0.42, 150), xytext=(0.75, 150),
                fontsize=9, color=GREEN, va="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    ax.annotate("lidar (~3 optical depths)", xy=(0.6, 55), xytext=(0.78, 40),
                fontsize=9, color=RUST, va="center",
                arrowprops=dict(arrowstyle="->", color=RUST, lw=1.4))
    ax.legend(fontsize=8.4, frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig4_subsurface.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


def main():
    fig1_conversion_slopes()
    fig2_ocean_vs_land()
    fig3_uncertainty_cascade()
    fig4_subsurface()


if __name__ == "__main__":
    main()
