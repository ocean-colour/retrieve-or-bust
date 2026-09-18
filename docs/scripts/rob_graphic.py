"""
rob_graphic.py — the "Retrieve or Bust" project summary graphic.

A single left-to-right conceptual flow whose centerpiece is the IOP-inversion
*degeneracy* and how the project breaks it (priors + AI):

    (1) Observe  ->  (2) Break the degeneracy  ->  (3) Retrieve  ->  (4) Validate
       Rrs             priors + AI collapse         IOPs + errs        vs truth
                        the fan of solutions

Two renders are produced:
  docs/figs/rob_graphic_talk.png    light background, large fonts   (talks/slides)
  docs/figs/rob_graphic_readme.png  dark  background, smaller fonts (README)

Palette is derived from the SeaMeetstheStars logo (a gold/star accent) paired with
ocean blues — "sea meets the stars." All curves are synthesized from a small toy
bio-optical forward model (Gordon u-polynomial), so the Rrs and IOP spectra have
realistic shapes rather than generic bumps.

Run in the `ocean14` conda environment:
    python docs/scripts/rob_graphic.py

Author: Claude (Fable) for J. Xavier Prochaska, 2026-07-20.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ palette --
GOLD = "#D9C878"      # star accent, from the logo
GOLD_HI = "#F0E4A8"
OCEAN = "#2E7FB8"     # mid ocean blue
DEEP = "#0B2545"      # deep navy
TEAL = "#3FA7A0"
CYAN = "#7FD1E3"

THEMES = {
    # talk: light, large fonts, tightly packed to fill the 16:9 frame
    "talk": dict(
        out="rob_graphic_talk.png",
        bg="#F6F8FB", ink="#0B2545", sub="#3A5573",
        card="#FFFFFF", card_edge=GOLD, card_shadow="#00000018",
        line_a=DEEP, line_bb=OCEAN, faint="#9BB4C9", tight=OCEAN,
        fs_title=48, fs_tag=22, fs_stage=27, fs_body=23, fs_small=19,
        fs_cap=17, fs_chip=16, fs_foot=20,
        title_y=0.910, sub_y=0.840, foot_y1=0.070, foot_y2=0.030, stage_dy=0.030,
        yc=0.180, hc=0.560, yc2=0.140, hc2=0.600, ymid=0.470,
        dpi=150,
    ),
    # readme: dark, smaller fonts (embedded in README)
    "readme": dict(
        out="rob_graphic_readme.png",
        bg="#0A1826", ink="#EAF2F8", sub="#9FC0D8",
        card="#11273D", card_edge=GOLD, card_shadow="#00000000",
        line_a=GOLD, line_bb=CYAN, faint="#43607A", tight=CYAN,
        fs_title=30, fs_tag=15, fs_stage=17, fs_body=12, fs_small=11,
        fs_cap=11, fs_chip=11, fs_foot=12,
        title_y=0.930, sub_y=0.840, foot_y1=0.075, foot_y2=0.038, stage_dy=0.028,
        yc=0.300, hc=0.440, yc2=0.265, hc2=0.500, ymid=0.500,
        dpi=140,
    ),
}

TEAM = ("Bontemps · Dierssen · Housekeeper · Frouin · "
        "Kavanaugh · Kudela · Prochaska")

# ----------------------------------------------------- toy bio-optical model -
def _aw(wl):
    """Rough pure-water absorption (m^-1), interpolated (Pope & Fry-like)."""
    x = np.array([400, 440, 480, 510, 540, 580, 620, 660, 690, 710])
    y = np.array([0.0066, 0.0064, 0.015, 0.033, 0.051, 0.089, 0.28, 0.41, 0.55, 0.80])
    return np.interp(wl, x, y)


def toy_rrs(wl, chl, cdom, bbp0):
    """A minimal (a, bb) -> Rrs forward model giving realistic Rrs shapes."""
    aph = chl * (0.045 * np.exp(-((wl - 440) / 42) ** 2)
                 + 0.028 * np.exp(-((wl - 675) / 18) ** 2))
    ag = cdom * np.exp(-0.017 * (wl - 440))
    a = _aw(wl) + aph + ag
    bb = 0.0019 * (500.0 / wl) ** 4.3 + bbp0 * (550.0 / wl) ** 1.0
    u = bb / (a + bb)
    return 0.0949 * u + 0.0794 * u ** 2, a, bb


# ---------------------------------------------------------------- card utils -
def add_card(fig, xy, wh, th, lw=2.2, z=1):
    x, y = xy
    w, h = wh
    if th["card_shadow"] != "#00000000":
        fig.patches.append(FancyBboxPatch(
            (x + 0.004, y - 0.008), w, h, transform=fig.transFigure,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            fc=th["card_shadow"], ec="none", zorder=z))
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, transform=fig.transFigure,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        fc=th["card"], ec=th["card_edge"], lw=lw, zorder=z + 0.1))


def inset(fig, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.patch.set_alpha(0.0)
    ax.set_zorder(5)          # draw content ABOVE the (opaque) card patches
    return ax


def stage_label(fig, x, y, num, title, th):
    fig.text(x, y, num, color=GOLD, fontsize=th["fs_stage"], fontweight="bold",
             ha="left", va="center", family="DejaVu Sans")
    fig.text(x + 0.028, y, title, color=th["ink"], fontsize=th["fs_stage"],
             fontweight="bold", ha="left", va="center")


def arrow(fig, x0, x1, y, th):
    fig.patches.append(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=26, lw=3.2,
        color=GOLD, zorder=0.5))


# --------------------------------------------------------------- the figure --
def build(theme_name):
    th = THEMES[theme_name]
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(16, 9), dpi=th["dpi"])
    fig.patch.set_facecolor(th["bg"])

    wl = np.linspace(400, 700, 300)

    # ---- title band ---- (clean single-weight title)
    ty = th["title_y"]
    fig.text(0.045, ty, "Retrieve or Bust", color=th["ink"], fontsize=th["fs_title"],
             fontweight="bold", ha="left", va="center")
    fig.text(0.045, th["sub_y"],
             "The ocean-color IOP inversion is degenerate — priors + AI break it.",
             color=th["sub"], fontsize=th["fs_tag"], ha="left", va="center")

    yc = th["yc"]; hc = th["hc"]
    yc2 = th["yc2"]; hc2 = th["hc2"]
    ymid = th["ymid"]   # arrow height / vertical mid of cards

    # card x-geometry (stage 2 is the wide centerpiece); gaps left for arrows
    c1 = (0.040, 0.155)   # ends 0.195
    c2 = (0.230, 0.310)   # ends 0.540  (centerpiece)
    c3 = (0.585, 0.170)   # ends 0.755
    c4 = (0.800, 0.150)   # ends 0.950

    for (x0, w), tall in ((c1, False), (c2, True), (c3, False), (c4, False)):
        if tall:
            add_card(fig, (x0, yc2), (w, hc2), th)
        else:
            add_card(fig, (x0, yc), (w, hc), th)

    # arrows between cards
    arrow(fig, c1[0] + c1[1] + 0.004, c2[0] - 0.004, ymid, th)
    arrow(fig, c2[0] + c2[1] + 0.004, c3[0] - 0.004, ymid, th)
    arrow(fig, c3[0] + c3[1] + 0.004, c4[0] - 0.004, ymid, th)

    # stage labels (above each card)
    stage_label(fig, c1[0] + 0.012, yc + hc + 0.028, "1", "Observe", th)
    stage_label(fig, c2[0] + 0.012, yc2 + hc2 + 0.028, "2", "Break the degeneracy", th)
    stage_label(fig, c3[0] + 0.012, yc + hc + 0.028, "3", "Retrieve", th)
    stage_label(fig, c4[0] + 0.012, yc + hc + 0.028, "4", "Validate", th)

    # ---------------------------------------------------- (1) Observe: Rrs ---
    ax1 = inset(fig, c1[0] + 0.022, yc + 0.075, c1[1] - 0.042, hc - 0.145)
    waters = [(0.05, 0.02, 0.0016), (0.5, 0.05, 0.0022),
              (3.0, 0.15, 0.0038), (12.0, 0.4, 0.0065)]
    cmap = [CYAN, OCEAN, TEAL, GOLD] if theme_name == "readme" else [OCEAN, TEAL, "#2E8B57", GOLD]
    for (chl, cdom, bbp0), col in zip(waters, cmap):
        rrs, _, _ = toy_rrs(wl, chl, cdom, bbp0)
        ax1.plot(wl, rrs, color=col, lw=2.4)
    ax1.set_xlim(394, 706)                     # small margin so curves clear the axis
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.12)
    ax1.annotate("", xy=(0.0, 1.02), xytext=(0.0, 0.0), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-", color=th["ink"], lw=1.4))
    ax1.annotate("", xy=(1.02, 0.0), xytext=(0.0, 0.0), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-", color=th["ink"], lw=1.4))
    fig.text(c1[0] + 0.010, yc + hc - 0.055, r"$R_{rs}$", color=th["ink"],
             fontsize=th["fs_body"], rotation=90, va="center", ha="center")
    fig.text(c1[0] + c1[1] / 2, yc + 0.052, "wavelength (nm)", color=th["sub"],
             fontsize=th["fs_small"], ha="center")
    fig.text(c1[0] + c1[1] / 2, yc + 0.018, r"hyperspectral $R_{rs}$",
             color=th["ink"], fontsize=th["fs_cap"], ha="center", style="italic")

    # ------------------------------ (2) Break the degeneracy: fan -> collapse
    ax2 = inset(fig, c2[0] + 0.016, yc2 + 0.050, c2[1] - 0.032, hc2 - 0.105)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    # left: a fan of many candidate IOP spectra (all consistent with one Rrs)
    xf = np.linspace(0.02, 0.135, 40)
    for k in range(15):
        spread = (k / 14 - 0.5) * 2                       # -1 .. 1
        curve = 0.5 + spread * (0.05 + 0.45 * (xf - 0.02) / 0.115)
        ax2.plot(xf, np.clip(curve, 0.06, 0.94), color=th["faint"], lw=1.0, alpha=0.75)
    ax2.text(0.0, 0.99, "many (a, b$_b$)\nmimic one $R_{rs}$", color=th["sub"],
             fontsize=th["fs_cap"], ha="left", va="top", linespacing=1.0)

    # three prior inputs (labeled chips), fed into the funnel
    cw, ch, cx0 = 0.380, 0.150, 0.160
    priors = ["in-situ", "environmental", "time-series"]
    py = [0.76, 0.50, 0.24]
    for name, yy in zip(priors, py):
        ax2.add_patch(FancyBboxPatch((cx0, yy - ch / 2), cw, ch,
                      boxstyle="round,pad=0.004,rounding_size=0.03",
                      fc=OCEAN if theme_name == "talk" else "#0E3A5C",
                      ec=CYAN, lw=1.3, zorder=3))
        ax2.text(cx0 + cw / 2, yy, name,
                 color="white" if theme_name == "talk" else CYAN,
                 fontsize=th["fs_chip"], ha="center", va="center", zorder=4)
        ax2.annotate("", xy=(0.605, 0.5 + (yy - 0.5) * 0.24), xytext=(0.555, yy),
                     arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=1.8), zorder=2)

    # funnel (wedge) collapsing the fan, throat at the AI node
    fun = plt.Polygon([[0.605, 0.06], [0.605, 0.94], [0.78, 0.60], [0.78, 0.40]],
                      closed=True, fc=GOLD, ec="none", alpha=0.16, zorder=1)
    ax2.add_patch(fun)
    ax2.plot([0.605, 0.78], [0.94, 0.60], color=GOLD, lw=2.0)
    ax2.plot([0.605, 0.78], [0.06, 0.40], color=GOLD, lw=2.0)

    # AI node at the funnel throat (explicit, per GQ7)
    ax2.add_patch(Circle((0.795, 0.50), 0.075, fc=GOLD, ec=th["ink"], lw=1.6, zorder=5))
    ax2.text(0.795, 0.50, "AI", color=DEEP, fontsize=th["fs_body"],
             fontweight="bold", ha="center", va="center", zorder=6)
    ax2.text(0.795, 0.405, "Claude", color=th["ink"], fontsize=th["fs_cap"],
             ha="center", va="top", zorder=6)

    # collapsed, tight solution emerging on the far right
    xt = np.linspace(0.90, 1.0, 24)
    for dz in (-0.018, 0.0, 0.018):
        ax2.plot(xt, 0.55 + dz - 0.02 * (xt - 0.90) / 0.10,
                 color=th["tight"], lw=2.6, alpha=0.95)
    # panel-2 bottom caption carries the "one solution" idea (keeps right side clear)
    fig.text(c2[0] + c2[1] / 2, yc2 + 0.020,
             "priors + AI  →  one solution", color=th["ink"],
             fontsize=th["fs_cap"], ha="center", style="italic")

    # ---------------------------------------- (3) Retrieve: IOPs + errors ----
    ax3 = inset(fig, c3[0] + 0.024, yc + 0.075, c3[1] - 0.044, hc - 0.145)
    _, a, bb = toy_rrs(wl, 1.5, 0.08, 0.003)
    a_n = a / a.max() * 0.88          # peaks below 1 so the legend clears the curves
    bb_n = bb / bb.max() * 0.88
    ax3.plot(wl, a_n, color=th["line_a"], lw=2.6, label=r"$a(\lambda)$")
    ax3.fill_between(wl, a_n * 0.9, a_n * 1.1, color=th["line_a"], alpha=0.12)
    ax3.plot(wl, bb_n, color=th["line_bb"], lw=2.6, label=r"$b_b(\lambda)$")
    ax3.fill_between(wl, bb_n * 0.85, bb_n * 1.15, color=th["line_bb"], alpha=0.12)
    ax3.set_xlim(394, 706); ax3.set_ylim(0, 1.10)
    ax3.annotate("", xy=(0.0, 1.02), xytext=(0.0, 0.0), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-", color=th["ink"], lw=1.4))
    ax3.annotate("", xy=(1.02, 0.0), xytext=(0.0, 0.0), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-", color=th["ink"], lw=1.4))
    leg = ax3.legend(loc="upper center", fontsize=th["fs_small"], frameon=False,
                     labelcolor=th["ink"], handlelength=1.2, ncol=2,
                     columnspacing=1.2, bbox_to_anchor=(0.5, 1.02))
    fig.text(c3[0] + c3[1] / 2, yc + 0.052, "wavelength (nm)", color=th["sub"],
             fontsize=th["fs_small"], ha="center")
    fig.text(c3[0] + c3[1] / 2, yc + 0.018, "IOPs with uncertainty",
             color=th["ink"], fontsize=th["fs_cap"], ha="center", style="italic")

    # -------------------------------------------- (4) Validate: 1:1 scatter --
    ax4 = inset(fig, c4[0] + 0.024, yc + 0.075, c4[1] - 0.044, hc - 0.145)
    rng = np.random.default_rng(11)
    tv = rng.uniform(0.12, 0.92, 40)
    rv = tv + rng.normal(0, 0.05, tv.size)
    ax4.plot([0.04, 0.98], [0.04, 0.98], color=GOLD, lw=2.0, ls="--")
    ax4.scatter(tv, rv, s=32, color=th["line_bb"], edgecolor=th["ink"],
                linewidth=0.5, alpha=0.9, zorder=3)
    ax4.set_xlim(-0.06, 1.06); ax4.set_ylim(-0.06, 1.06)   # keep points off the axes
    ax4.annotate("", xy=(0.0, 1.02), xytext=(0.0, 0.0), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-", color=th["ink"], lw=1.4))
    ax4.annotate("", xy=(1.02, 0.0), xytext=(0.0, 0.0), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-", color=th["ink"], lw=1.4))
    fig.text(c4[0] + c4[1] / 2, yc + 0.052, "in-situ truth", color=th["sub"],
             fontsize=th["fs_small"], ha="center")
    fig.text(c4[0] + c4[1] / 2, yc + 0.018, "validate vs truth",
             color=th["ink"], fontsize=th["fs_cap"], ha="center", style="italic")
    fig.text(c4[0] + 0.012, yc + hc / 2 + 0.02, "retrieved", color=th["sub"],
             fontsize=th["fs_small"], rotation=90, va="center", ha="center")

    # ---- footer ----
    fig.text(0.045, th["foot_y1"], TEAM, color=th["sub"], fontsize=th["fs_foot"],
             ha="left", va="center", fontweight="bold")
    fig.text(0.045, th["foot_y2"],
             "retrieve-or-bust  ·  Sea Meets the Stars  ·  hyperspectral · AI-accelerated",
             color=GOLD, fontsize=th["fs_foot"], ha="left", va="center", fontweight="bold")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, th["out"])
    fig.savefig(out, dpi=th["dpi"], facecolor=th["bg"])
    plt.close(fig)
    print("wrote", out)


def main():
    for name in ("talk", "readme"):
        build(name)


if __name__ == "__main__":
    main()
